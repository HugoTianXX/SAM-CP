import json
from transformers import TrainingArguments, Trainer, BertForMaskedLM, BertTokenizerFast, DebertaForMaskedLM
from torch import nn
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def map_word_to_token_idxs(text, tokenizer):
    split_text = text.split() 
    tokenized_text = tokenizer.tokenize(text) 
    token_pos = 0
    word_to_token_map = {}

    for word_pos, word in enumerate(split_text):
        word_to_token_map[word_pos] = []

        for token in tokenized_text[token_pos:]:
            if token.startswith('##'):
                token = token[2:]
            word_to_token_map[word_pos].append(token_pos)
            token_pos += 1
            if word == token:
                break

    return word_to_token_map


def pad_sequences(sequences, pad_token_id, max_len=256):
    padded_sequences = []

    if isinstance(sequences[0][0], int):
        for seq in sequences:
            pad_length = max_len - len(seq)
            if pad_length > 0:
                padded_seq = seq + [pad_token_id] * pad_length
            else:
                padded_seq = seq[:max_len]
            padded_sequences.append(padded_seq)
    else:
        for seq_pair in sequences:
            padded_pair = []
            for sub_seq in seq_pair:
                sub_pad_length = max_len - len(sub_seq)
                if sub_pad_length > 0:
                    processed_seq = sub_seq + [pad_token_id] * sub_pad_length
                else:
                    processed_seq = sub_seq[:max_len]
                padded_pair.append(processed_seq)
            padded_sequences.append(padded_pair)

    return padded_sequences

def preprocess_data(data, tokenizer):  
    inputs = []
    labels = []
    attention_masks = []
    for item in tqdm(data):  
        text = item['text']
        target_words = item['replace']

        encoding = tokenizer.encode_plus(text, return_offsets_mapping=True)
        word_to_token_map = map_word_to_token_idxs(text, tokenizer)

        input_ids = encoding['input_ids']
        labels_ids = [-100] * len(input_ids)
        offset = 0
        for idx, syns in target_words.items():
            idx = int(idx)
            token_idxs = word_to_token_map.get(idx, None)

            if token_idxs is not None and token_idxs != []:
                start_idx = token_idxs[0] # + offset

                input_ids_tmp = input_ids[:start_idx] + \
                                [tokenizer.additional_special_tokens_ids[0]] + \
                                [input_ids[start_idx]] + \
                                [tokenizer.additional_special_tokens_ids[1]] + \
                                input_ids[start_idx+1:]

                labels_ids_tmp = labels_ids[:start_idx] + [-100] + [input_ids[start_idx+1]] + [-100] + labels_ids[start_idx+1:]
                syns = syns[:17]
                for syn in syns:
                    labels_ids_tmp[start_idx + 1] = tokenizer.encode(syn, add_special_tokens=False)[0]
                # labels_ids_tmp[start_idx + 1] = [labels_ids_tmp[start_idx + 1]] + [tokenizer.encode(syn, add_special_tokens=False)[0] for syn in syns]
                # labels_ids_tmp[start_idx + 1] = torch.tensor([labels_ids_tmp[start_idx + 1]] + [tokenizer.encode(syn, add_special_tokens=False)[0] for syn in syns])
                    inputs.append(input_ids_tmp)
                    labels.append(labels_ids_tmp)
                    attention_masks.append([1]*len(input_ids_tmp))


                # offset += 2
        # inputs.append(input_ids)
        # labels.append(labels_ids)
    inputs = pad_sequences(inputs, tokenizer.pad_token_id)
    labels = pad_sequences(labels, -100)
    attention_masks = pad_sequences(attention_masks, 0)
    return inputs, labels, attention_masks

inputs, labels, attention_masks = preprocess_data(data, tokenizer)

# Prepare for training
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))
import torch.nn as nn

def compute_loss(outputs, labels):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = 0
    for output, label in zip(outputs.view(-1, outputs.size(-1)), labels.view(-1)):
        if isinstance(label, list):
            for li in label:
                loss += loss_fn(output, li)
        else:
            loss += loss_fn(output, label)
    return loss / len(outputs)

class MyTrainer(Trainer):

    def compute_loss_with_synonyms(outputs, labels):
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  
        loss = 0 

        for output, label in zip(outputs.view(-1, outputs.size(-1)), labels.view(-1)):
            if isinstance(label, list):  
                for tokens in label:
                    for token in tokens:
                        loss += loss_fn(output.unsqueeze(0), torch.tensor([token]).to(output.device))
            else:
                loss += loss_fn(output.unsqueeze(0), torch.tensor([label]).to(output.device))

        return loss / len(outputs)
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  

        start_indices = (labels == tokenizer.additional_special_tokens_ids[0]).nonzero(as_tuple=True)[1]
        end_indices = (labels == tokenizer.additional_special_tokens_ids[1]).nonzero(as_tuple=True)[1]

        loss = 0
        for start, end in zip(start_indices, end_indices):
            # Only consider labels within [SYN] and [EOSYN]
            relevant_logits = logits[:, start:end]
            relevant_labels = labels[:, start:end]

            loss += loss_fct(relevant_logits.view(-1, relevant_logits.size(-1)), relevant_labels.view(-1))

        return loss
    
args = TrainingArguments(
    "bert-sim-train",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=30000
)


class FinetuneDataset(Dataset):
    def __init__(self, inputs, labels, attention_masks):
        self.inputs = inputs
        self.labels = labels
        self.attention_masks= attention_masks

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.inputs[idx]), 
                'attention_masks': torch.tensor(self.attention_masks[idx]), 
                'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.inputs)

json_file = 'data/Finetune_data_with_gpt_ranking.json'
# Load JSON data
data = json.load(open(json_file, 'r'))
# Replace 'json_file' with your variable

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Add special tokens
special_tokens_dict = {'additional_special_tokens': ['[SYN]', '[EOSYN]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')

inputs_train, inputs_eval, labels_train, labels_eval, attention_masks_train, attention_masks_eval = train_test_split(inputs, labels, attention_masks, test_size=0.1)
train_dataset = FinetuneDataset(inputs_train, labels_train, attention_masks_train)
eval_dataset = FinetuneDataset(inputs_eval, labels_eval, attention_masks_eval)
trainer = Trainer(model, args, train_dataset = train_dataset, eval_dataset=eval_dataset)

# Train model
trainer.train()