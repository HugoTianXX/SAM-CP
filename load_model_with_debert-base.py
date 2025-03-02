from transformers import DebertaModel, DebertaTokenizer

model = DebertaModel.from_pretrained('microsoft/deberta-base')
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

model.save_pretrained('./deberta-base')
tokenizer.save_pretrained('./deberta-base')