# models/predictor.py

import torch
import torch.nn as nn


class MeaningPredictor(nn.Module):
    """
    ONN의 예측기.
    상태 예측과 목적 추론을 모두 처리.
    """

    def __init__(self, hidden_dim, output_dim=128, num_classes=10):
        """
        hidden_dim: 입력 차원 (인코딩된 의미 흐름 차원)
        output_dim: 출력 차원 (예측할 상태의 차원)
        num_classes: 분류할 클래스의 수 (목적 예측)
        """
        super(MeaningPredictor, self).__init__()

        # 예측을 위한 MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

        # 목적 예측 (분류용)
        self.classifier = nn.Linear(output_dim, num_classes)

        # 상태 예측 (회귀용)
        self.regressor = nn.Linear(output_dim, 1)

        # 드롭아웃 및 정규화
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        x: (B, T, D) - 상호작용 후 텐서
        return: 예측된 상태 또는 분류 결과
        """
        # 예측을 위한 feature extraction
        x = self.fc(x)  # (B, T, output_dim)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 예측할 목적이나 상태에 따라 출력 분기
        # 예시 1: 목적 예측 (classification)
        pred_class = self.classifier(x)  # (B, T, num_classes)

        # 예시 2: 상태 예측 (regression)
        pred_state = self.regressor(x)  # (B, T, 1)

        return pred_class, pred_state
