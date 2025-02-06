import torch
import torch.nn as nn
import transformers

class DescriptionWriter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.batch_norm = nn.BatchNorm1d(input_size)

        self.embed = nn.Embedding(output_size+1, input_size)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)
        x = self.embed(x)        
        
        x, _ = self.lstm(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.linear(x)

        return x

