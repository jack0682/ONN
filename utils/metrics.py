# utils/metrics.py

import numpy as np

def meaning_accuracy(preds, targets, tol=0.1):
    """
    의미 텐서 정확도: 유클리드 거리가 일정 임계값 이하인 비율
    """
    error = np.linalg.norm(preds - targets, axis=-1)  # (batch, ...)
    return float(np.mean(error < tol))


def flow_consistency(preds, targets):
    """
    시계열 흐름 방향 정합성 (Angular Deviation 기반)
    """
    norm_preds = preds / (np.linalg.norm(preds, axis=-1, keepdims=True) + 1e-8)
    norm_targets = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
    dot_product = np.sum(norm_preds * norm_targets, axis=-1)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return float(np.mean(angle_rad < 0.2))  # ≈ 11.5도


def relation_alignment_score(preds, targets):
    """
    관계 정렬 점수 (MSE 기반 역오차 정규화)
    """
    mse = np.mean((preds - targets) ** 2)
    denom = np.mean(targets ** 2) + 1e-6
    return float(1.0 - mse / denom)


def temporal_prediction_score(preds, targets):
    """
    시간 기반 미래 예측 정합도 (비율 오차)
    """
    mae = np.mean(np.abs(preds - targets))
    mean_target = np.mean(np.abs(targets)) + 1e-6
    return float(1.0 - mae / mean_target)
