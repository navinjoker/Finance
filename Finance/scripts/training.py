
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import LSTMForecast
import numpy as np
import pandas as pd

def create_sequences(data, window):
    sequences = []
    targets = []
    for i in range(len(data) - window):
        sequences.append(data[i:i+window])
        targets.append(data[i+window])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def train_model(model, train_loader, optimizer, criterion, epochs=50):
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Load data
log_returns = pd.read_csv('data/log_returns.csv', index_col=0).values
X, y = create_sequences(log_returns, window=30)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMForecast(input_size=log_returns.shape[1], hidden_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

train_model(model, train_loader, optimizer, criterion)
