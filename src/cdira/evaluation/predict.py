from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from cdira.models.cdira import InferenceOutput, RoutingPolicy


@dataclass(frozen=True)
class PredictionTable:
    paths: list[str]
    targets: Tensor
    predictions: Tensor
    logits: Tensor
    global_logits: Tensor
    confidence: Tensor
    routing_probability: Tensor
    roi_mask: Tensor
    domains: Tensor


def routing_policy_name(policy: RoutingPolicy) -> str:
    return policy.value


def _table_from_batches(rows: list[dict[str, Any]]) -> PredictionTable:
    if not rows:
        raise ValueError("Prediction loader yielded no rows")
    return PredictionTable(
        paths=[path for row in rows for path in row["paths"]],
        targets=torch.cat([row["targets"] for row in rows]),
        predictions=torch.cat([row["logits"].argmax(dim=1) for row in rows]),
        logits=torch.cat([row["logits"] for row in rows]),
        global_logits=torch.cat([row["global_logits"] for row in rows]),
        confidence=torch.cat([row["confidence"] for row in rows]),
        routing_probability=torch.cat([row["routing_probability"] for row in rows]),
        roi_mask=torch.cat([row["roi_mask"] for row in rows]),
        domains=torch.cat([row["domains"] for row in rows]),
    )


def collect_predictions(
    model: Any,
    loader: Iterable[dict[str, Any]],
    policy: RoutingPolicy,
    threshold: float,
    device: torch.device,
) -> PredictionTable:
    if hasattr(model, "eval"):
        model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            output: InferenceOutput = model.predict(images, policy, threshold)
            rows.append(
                {
                    "paths": [str(path) for path in batch["relative_path"]],
                    "targets": batch["target"].detach().cpu(),
                    "logits": output.logits.detach().cpu(),
                    "global_logits": output.global_logits.detach().cpu(),
                    "confidence": output.global_confidence.detach().cpu(),
                    "routing_probability": output.routing_probability.detach().cpu(),
                    "roi_mask": output.roi_mask.detach().cpu(),
                    "domains": batch["domain"].detach().cpu(),
                }
            )
    return _table_from_batches(rows)


def routing_sweep(
    model: Any,
    loader: Iterable[dict[str, Any]],
    policies: Sequence[RoutingPolicy],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for threshold in thresholds:
            table = collect_predictions(
                model, loader, policy, threshold, torch.device("cpu")
            )
            rows.append(
                {
                    "policy": policy.value,
                    "threshold": threshold,
                    "accuracy": float(
                        (table.predictions == table.targets).float().mean()
                    ),
                    "roi_usage": float(table.roi_mask.float().mean()),
                }
            )
    return pd.DataFrame(rows)
