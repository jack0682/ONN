"""Pair Feature Extraction for Relation Embeddings.

This module extracts pair features φ_ij from node pairs for computing
observed relation embeddings x_obs.

Features include:
- Geometric: position difference, distance, angle
- Bound tensor: difference/similarity of bound tensors
- Form tensor: difference/similarity of form tensors

Reference:
    - spec/20_impl_plan.ir.yml: IMPL_019
    - User roadmap: "phi_ij from detections"

Author: Claude (via IMPL_019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import numpy as np

if TYPE_CHECKING:
    from onn.core.tensors import SemanticNode

logger = logging.getLogger(__name__)


# ==============================================================================
# PAIR FEATURE EXTRACTION
# ==============================================================================

def compute_pair_features(
    bound_i: torch.Tensor,
    bound_j: torch.Tensor,
    form_i: Optional[torch.Tensor] = None,
    form_j: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pair features φ_ij from node tensors.
    
    Features:
    - Position difference (from bound tensors)
    - Euclidean distance
    - Bound tensor difference
    - Bound tensor element-wise product (interaction)
    - Form tensor features (if provided)
    
    Args:
        bound_i: Bound tensor of node i, shape (16,)
        bound_j: Bound tensor of node j, shape (16,)
        form_i: Form tensor of node i, shape (32,) optional
        form_j: Form tensor of node j, shape (32,) optional
        
    Returns:
        Pair feature vector φ_ij
    """
    features = []
    
    # 1. Position difference (first 3 dims of bound = position)
    pos_i = bound_i[:3]
    pos_j = bound_j[:3]
    pos_diff = pos_j - pos_i
    features.append(pos_diff)
    
    # 2. Euclidean distance
    distance = torch.norm(pos_diff)
    features.append(distance.unsqueeze(0))
    
    # 3. Bound tensor difference
    bound_diff = bound_j - bound_i
    features.append(bound_diff)
    
    # 4. Element-wise product (captures interaction)
    bound_prod = bound_i * bound_j
    features.append(bound_prod)
    
    # 5. Form tensor features (if provided)
    if form_i is not None and form_j is not None:
        form_diff = form_j - form_i
        features.append(form_diff)
        
        # Cosine similarity of form tensors
        form_sim = torch.dot(form_i, form_j) / (
            torch.norm(form_i) * torch.norm(form_j) + 1e-8
        )
        features.append(form_sim.unsqueeze(0))
    
    return torch.cat(features)


def compute_pair_features_batch(
    bounds: torch.Tensor,
    edge_index: torch.Tensor,
    forms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pair features for all edges in batch.
    
    Args:
        bounds: Node bound tensors, shape (n, 16)
        edge_index: Edge indices, shape (2, m)
        forms: Node form tensors, shape (n, 32) optional
        
    Returns:
        Pair features, shape (m, feature_dim)
    """
    m = edge_index.shape[1]
    features_list = []
    
    for e in range(m):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        
        form_i = forms[i] if forms is not None else None
        form_j = forms[j] if forms is not None else None
        
        phi_ij = compute_pair_features(
            bounds[i], bounds[j], form_i, form_j
        )
        features_list.append(phi_ij)
    
    return torch.stack(features_list)


def standardize_features(
    phi: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize pair features to zero mean, unit variance.
    
    Args:
        phi: Pair features, shape (m, d) or (d,)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)
        
    Returns:
        Tuple of (normalized_phi, mean, std)
    """
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    
    if mean is None:
        mean = phi.mean(dim=0)
    if std is None:
        std = phi.std(dim=0) + 1e-8
    
    phi_norm = (phi - mean) / std
    
    return phi_norm, mean, std
