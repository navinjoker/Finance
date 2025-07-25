import torch
import pandas as pd
import numpy as np
from models.transformer import TimeSeriesTransformer

def create_sequences(data, window):
    sequences = []
    for i in range(len(data) - window):
        seq = data[i:i+window]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.float32)

# Load log returns
df = pd.read_csv('data/log_returns.csv', index_col=0)
log_returns = df.values
tickers = df.columns.tolist()

# Parameters
window = 30
input_size = log_returns.shape[1]
model_path = 'models/transformer_trained.pt'

# Prepare input
X = create_sequences(log_returns, window)

# Transformer expects [seq_len, batch_size, feature_dim]
X = X.permute(1, 0, 2)  # [N, W, F] -> [W, N, F]

# Load model
model = TimeSeriesTransformer(input_size=input_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Predict
predictions = []
with torch.no_grad():
    for i in range(X.shape[1]):
        x = X[:, i:i+1, :]  # [W, 1, F]
        pred = model(x).squeeze().numpy()
        predictions.append(pred)

# Save predictions
pred_df = pd.DataFrame(predictions, columns=tickers, index=df.index[window:])
pred_df.to_csv('data/transformer_predictions.csv')
print("✅ Transformer predictions saved to data/transformer_predictions.csv")
