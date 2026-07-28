from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from cdira.models.cdira import TrainOutput


@dataclass(frozen=True)
class LossWeights:
    global_ce: float
    fused_ce: float
    routing_bce: float
    routing_regularizer: float
    domain_ce: float


@dataclass(frozen=True)
class LossBreakdown:
    global_ce: Tensor
    fused_ce: Tensor
    routing_bce: Tensor
    routing_regularizer: Tensor
    domain_ce: Tensor
    total: Tensor
    routing_targets: Tensor
    positive_weight: Tensor


def routing_targets(
    global_logits: Tensor, targets: Tensor, confidence_threshold: float
) -> Tensor:
    with torch.no_grad():
        probabilities = global_logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        return ((prediction != targets) | (confidence < confidence_threshold)).float()


def compute_cdira_loss(
    output: TrainOutput,
    targets: Tensor,
    domains: Tensor,
    weights: LossWeights,
    confidence_threshold: float = 0.9,
) -> LossBreakdown:
    route = routing_targets(output.global_logits, targets, confidence_threshold)
    positive = route.sum()
    negative = route.numel() - positive
    positive_weight = (
        negative / positive if positive > 0 and negative > 0 else route.new_tensor(1.0)
    )
    global_ce = F.cross_entropy(output.global_logits, targets)
    fused_ce = F.cross_entropy(output.fused_logits, targets)
    routing_bce = F.binary_cross_entropy_with_logits(
        output.routing_logits, route, pos_weight=positive_weight
    )
    routing_regularizer = output.routing_logits.sigmoid().mean()
    domain_ce = F.cross_entropy(output.domain_logits, domains)
    total = (
        weights.global_ce * global_ce
        + weights.fused_ce * fused_ce
        + weights.routing_bce * routing_bce
        + weights.routing_regularizer * routing_regularizer
        + weights.domain_ce * domain_ce
    )
    values = (global_ce, fused_ce, routing_bce, routing_regularizer, domain_ce, total)
    if not all(torch.isfinite(value).item() for value in values):
        raise FloatingPointError("C-DIRA loss contains a non-finite value")
    return LossBreakdown(
        global_ce,
        fused_ce,
        routing_bce,
        routing_regularizer,
        domain_ce,
        total,
        route,
        positive_weight,
    )
