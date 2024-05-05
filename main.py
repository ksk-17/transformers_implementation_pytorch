import torch
import torch.nn as nn
from transformer import Transformer
import torch.optim as optim
from clean_dataset import train_data
from torch.utils.data import DataLoader
from get_batches import wmtDataset
from tqdm import tqdm

batch_size = 64
max_seq_length = 100
train_epochs = 18

num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
d_model = 768  # embedding size of tokenizer

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    print("started training")
    english_lines, german_lines = train_data()
    train_dataset = wmtDataset(english_lines, german_lines, batch_size, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    src_vocab_size = tgt_vocab_size = train_dataset.get_vocab_size()

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch_id in range(train_epochs):
        print(f'started epoch: {epoch_id + 1}')
        for batch_id, (src_data, tgt_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move data to GPU
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)

            optimizer.zero_grad()
            output = transformer(src_data, tgt_data[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch_id + 1} batch_id: {batch_id + 1}, Loss: {loss.item()}")

train_model()

    






