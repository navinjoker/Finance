
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1])
        return out
