import json
from transformers import TrainingArguments, Trainer, DebertaTokenizerFast
from .DebertaForMLM4Contra_add_sim import DebertaForMaskedLM
from torch import nn
from tqdm import tqdm
import torch
import random
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
    sim_label = []
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
                    if random.random() < 0.5:

                        inputs.append([input_ids, input_ids_tmp])
                        labels.append([labels_ids, labels_ids_tmp])
                        attention_masks.append([[1]*len(input_ids), [1]*len(input_ids_tmp)])
                    else:
                        input_ids_tmp_neg = input_ids_tmp
                        input_ids_tmp_neg[start_idx + 1] = tokenizer.encode(syn, add_special_tokens=False)[0]
                        labels_ids_tmp_neg = labels_ids_tmp
                        labels_ids_tmp_neg[start_idx + 1] = -100
                        inputs.append([input_ids_tmp_neg, input_ids_tmp])
                        labels.append([labels_ids_tmp_neg, labels_ids_tmp])
                        attention_masks.append([[1]*len(input_ids_tmp_neg), [1]*len(input_ids_tmp)])


                # offset += 2
        # inputs.append(input_ids)
        # labels.append(labels_ids)
    inputs = pad_sequences(inputs, tokenizer.pad_token_id)
    labels = pad_sequences(labels, -100)
    attention_masks = pad_sequences(attention_masks, 0)
    return inputs, labels, attention_masks

inputs, labels, attention_masks = preprocess_data(data, tokenizer)



# Prepare for training
model = DebertaForMaskedLM.from_pretrained('deberta-base-fix')
model.resize_token_embeddings(len(tokenizer))
import torch.nn as nn


args = TrainingArguments(
    "deberta-sim-train-loss-cos-sim",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
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
                'attention_mask': torch.tensor(self.attention_masks[idx]),
                'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.inputs)

# Load JSON data
data = 'data/Finetune_data_with_gpt_ranking.json'

# Initialize tokenizer
tokenizer = DebertaTokenizerFast.from_pretrained('deberta-base-fix')

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
trainer.save_model("deberta-sim-train-loss-cos-sim-last")
tokenizer.save_pretrained("deberta-sim-train-loss-cos-sim-last")