# models/embedding.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeaningEmbedding(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, embedding_dim: int, dropout=0.1):
        super(MeaningEmbedding, self).__init__()

        # 각각의 흐름을 독립적으로 임베딩
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.delta_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 잔차 결합 구조
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, state_tensor, delta_tensor):
        """
        state_tensor: (B, N)
        delta_tensor: (B, N)
        """
        s_feat = self.state_embed(state_tensor)        # (B, H)
        d_feat = self.delta_embed(delta_tensor)        # (B, H)

        # 다양한 방식의 결합
        x = torch.cat([
            s_feat,
            d_feat,
            s_feat - d_feat
        ], dim=-1)  # (B, 3H)

        embedding = self.fusion(x)  # (B, D)
        return embedding
