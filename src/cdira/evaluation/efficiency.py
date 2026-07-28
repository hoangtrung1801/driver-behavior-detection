from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from cdira.models.cdira import RoutingPolicy


def expected_conditional_flops(
    global_flops: float, roi_extra_flops: float, roi_usage: float
) -> float:
    if not 0.0 <= roi_usage <= 1.0:
        raise ValueError("roi_usage must be between 0 and 1")
    return global_flops + roi_extra_flops * roi_usage


@dataclass(frozen=True)
class EfficiencyReport:
    parameters: int
    global_flops: float
    roi_extra_flops: float
    expected_flops: float
    latency_ms: dict[str, float]


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_efficiency(
    model: Any, sample: torch.Tensor, roi_usage: float, device: torch.device
) -> EfficiencyReport:
    model = model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        for _ in range(100):
            model.predict(sample.to(device), policy=RoutingPolicy.NONE, threshold=0.9)
        _synchronize(device)
        timings: list[float] = []
        for _ in range(1000):
            started = time.perf_counter()
            model.predict(sample.to(device), policy=RoutingPolicy.NONE, threshold=0.9)
            _synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    return EfficiencyReport(
        parameters,
        0.0,
        0.0,
        0.0,
        {
            "mean": statistics.mean(timings),
            "median": statistics.median(timings),
            "p90": sorted(timings)[int(len(timings) * 0.90) - 1],
            "p95": sorted(timings)[int(len(timings) * 0.95) - 1],
        },
    )
