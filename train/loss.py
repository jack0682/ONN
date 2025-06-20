# train/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeaningLoss(nn.Module):
    """
    ONN 모델에서 사용되는 의미 기반 Loss를 정의한 클래스.
    """

    def __init__(self, lambda_sem=1.0, lambda_flow=1.0, lambda_rel=1.0, lambda_pred=1.0):
        """
        Args:
        - lambda_sem (float): 의미 예측 Loss의 가중치
        - lambda_flow (float): 의미 흐름 Loss의 가중치
        - lambda_rel (float): 상호작용 관계 Loss의 가중치
        - lambda_pred (float): 미래 상태 예측 Loss의 가중치
        """
        super(MeaningLoss, self).__init__()
        
        self.lambda_sem = lambda_sem
        self.lambda_flow = lambda_flow
        self.lambda_rel = lambda_rel
        self.lambda_pred = lambda_pred

    def forward(self, predictions, targets, relations, delta_states, future_states):
        """
        :param predictions: 모델 예측 값 (목적 예측 및 상태 예측)
        :param targets: 실제 목표 값 (ground truth)
        :param relations: 객체 간 관계 텐서
        :param delta_states: 의미 흐름 변화율 (dS/dt)
        :param future_states: 미래 상태의 ground truth (for future prediction loss)
        
        :return: 총 Loss 값
        """
        
        # 1. 의미 예측 Loss
        sem_loss = self.compute_semantic_loss(predictions, targets)
        
        # 2. 의미 흐름 Loss
        flow_loss = self.compute_flow_loss(predictions, delta_states)
        
        # 3. 상호작용 관계 Loss
        rel_loss = self.compute_relation_loss(predictions, relations)
        
        # 4. 미래 상태 예측 Loss
        pred_loss = self.compute_future_prediction_loss(predictions, future_states)

        # 전체 Loss는 각 항목의 가중 합
        total_loss = (self.lambda_sem * sem_loss + 
                      self.lambda_flow * flow_loss + 
                      self.lambda_rel * rel_loss + 
                      self.lambda_pred * pred_loss)
        
        return total_loss

    def compute_semantic_loss(self, predictions, targets):
        """
        의미 예측 Loss (Cross-Entropy for classification or MSE for regression)
        """
        if predictions.shape[-1] == 1:  # 회귀인 경우
            return F.mse_loss(predictions, targets)
        else:  # 분류인 경우
            return F.cross_entropy(predictions, targets)

    def compute_flow_loss(self, predictions, delta_states):
        """
        의미 흐름 Loss (MSE for temporal change)
        """
        # 예측된 변화율과 실제 변화율의 차이
        flow_loss = torch.mean((predictions - delta_states) ** 2)
        return flow_loss

    def compute_relation_loss(self, predictions, relations):
        """
        관계 Loss (MSE between predicted relations and true relations)
        """
        # 관계 텐서 예측 값과 실제 관계 텐서 값 사이의 차이
        relation_loss = torch.mean((predictions - relations) ** 2)
        return relation_loss

    def compute_future_prediction_loss(self, predictions, future_states):
        """
        미래 상태 예측 Loss (MSE for future prediction)
        """
        # 예측된 미래 상태와 실제 미래 상태의 차이
        future_loss = torch.mean((predictions - future_states) ** 2)
        return future_loss
