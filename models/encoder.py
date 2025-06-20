import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Transformer-style Positional Encoding for time-sequence awareness
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model//2)

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class ResidualGRULayer(nn.Module):
    """
    GRU Layer with residual connection and optional LayerNorm
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)  # (B, T, H)
        out = self.dropout(out)
        out = self.norm(out + x)  # residual + norm
        return out


class MeaningEncoder(nn.Module):
    """
    시계열 의미 흐름 인코더 (GRU + Positional Encoding + Residual)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, use_positional=True, output_mode='sequence'):
        super().__init__()
        self.use_pos = use_positional
        self.output_mode = output_mode

        if self.use_pos:
            self.pos_encoder = PositionalEncoding(input_dim)

        self.layers = nn.ModuleList([
            ResidualGRULayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (B, T, D)
        return: (B, T, H) or (B, H)
        """
        if self.use_pos:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        if self.output_mode == 'sequence':
            return x  # (B, T, H)
        elif self.output_mode == 'last':
            return x[:, -1, :]  # (B, H)
        else:
            raise ValueError("output_mode must be 'sequence' or 'last'")
