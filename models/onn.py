# models/onn.py

import torch
import torch.nn as nn
from models.embedding import MeaningEmbedding
from models.encoder import MeaningEncoder
from models.interaction import TemporalCausalAttention
from models.predictor import MeaningPredictor  # 예측 모듈


class ONN(nn.Module):
    """
    Ontology Neural Network (ONN) 모델.
    의미 기반 시계열 학습 및 객체 간 관계 추론 모델.
    """
    def __init__(self, state_dim, hidden_dim, embedding_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(ONN, self).__init__()

        # 1. 의미 임베딩 계층 (state + delta)
        self.embedding = MeaningEmbedding(state_dim, hidden_dim, embedding_dim, dropout)

        # 2. 의미 흐름 인코더 (GRU 또는 Transformer)
        self.encoder = MeaningEncoder(embedding_dim, hidden_dim, num_layers, use_positional=True)

        # 3. 상호작용 통합 (GAT, Temporal Causal Attention)
        self.interaction = TemporalCausalAttention(d_model=hidden_dim, d_relation=32, d_hidden=128, dropout=dropout)

        # 4. 예측기 (목적 추론, 상태 예측)
        self.predictor = MeaningPredictor(hidden_dim)

        # 추가적인 확장을 위해 dropout 및 LayerNorm을 적용할 수 있음
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Hyperparameters (추후 튜닝 가능)
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, state_tensor, delta_tensor, relation_tensor):
        """
        ONN 모델의 전방향 연산.
        
        state_tensor: (B, T, D) - 상태 텐서 (S_i(t))
        delta_tensor: (B, T, D) - 변화율 텐서 (dS_i(t)/dt)
        relation_tensor: (B, N, N, d_relation) - 객체 간 관계 텐서
        
        return: 예측된 상태 (B, T, D) 또는 분류된 목적 (B, 1)
        """

        # 1. 의미 임베딩 (S_i(t) + dS_i(t)) -> E_t
        embedding = self.embedding(state_tensor, delta_tensor)

        # 2. 의미 흐름 인코딩 (GRU 또는 Transformer) -> H_t
        encoded = self.encoder(embedding)

        # 3. 상호작용 통합 (관계 기반 텐서 통합) -> Z_t
        interacted = self.interaction(encoded, relation_tensor, delta_tensor)

        # 4. 예측기 -> 목적 예측 또는 상태 예측
        prediction = self.predictor(interacted)

        # LayerNorm + Dropout 추가
        prediction = self.layer_norm(prediction)
        prediction = self.dropout(prediction)

        return prediction
