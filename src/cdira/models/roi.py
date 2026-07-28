from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ROIResult:
    pooled: Tensor
    saliency: Tensor
    indices: Tensor


def topk_roi_pool(feature_map: Tensor, k: int) -> ROIResult:
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B,C,H,W]")
    _, _, height, width = feature_map.shape
    if not 1 <= k <= height * width:
        raise ValueError(f"Top-K must be between 1 and H*W={height * width}")
    saliency = torch.linalg.vector_norm(feature_map, ord=2, dim=1)
    indices = saliency.flatten(1).topk(k, dim=1).indices
    spatial = feature_map.flatten(2).transpose(1, 2)
    gathered = spatial.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, spatial.shape[-1])
    )
    return ROIResult(gathered.mean(dim=1), saliency, indices)
