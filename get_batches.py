import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BertTokenizer
from clean_dataset import train_data

class wmtDataset(Dataset):
    def __init__(self, xData, yData, batch_size, max_seq_len):
        self.xData = xData
        self.yData = yData
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __len__(self):
        return len(self.xData)

    def __getitem__(self, idx):

        # tokenize the input data
        english_tokens = self.tokenizer.encode_plus(
            self.xData[idx], 
            add_special_tokens = True,
            max_length = self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # tokenize the input data
        german_tokens = self.tokenizer.encode_plus(
            self.yData[idx], 
            add_special_tokens = True,
            max_length = self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_token_ids = english_tokens['input_ids'].squeeze(0)
        output_token_ids = german_tokens['input_ids'].squeeze(0)

        return input_token_ids, output_token_ids

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

        

