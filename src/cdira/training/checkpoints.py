from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn


@dataclass
class TrainingState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Any
    epoch: int
    best_validation_loss: float
    patience_count: int
    fingerprints: dict[str, str]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path, state: TrainingState, fingerprints: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state.model,
        "optimizer": state.optimizer,
        "scaler": state.scaler,
        "epoch": state.epoch,
        "best_validation_loss": state.best_validation_loss,
        "patience_count": state.patience_count,
        "fingerprints": fingerprints,
        "rng": _rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    expected: dict[str, str],
    mode: Literal["resume", "weights"],
) -> TrainingState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actual = payload["fingerprints"]
    if mode == "resume" and actual != expected:
        raise ValueError(
            f"Checkpoint fingerprints differ: expected={expected} actual={actual}"
        )
    if mode == "resume":
        _restore_rng(payload["rng"])
        return TrainingState(
            payload["model"],
            payload["optimizer"],
            payload["scaler"],
            payload["epoch"],
            payload["best_validation_loss"],
            payload["patience_count"],
            actual,
        )
    return TrainingState(
        payload["model"],
        payload["optimizer"],
        payload["scaler"],
        0,
        float("inf"),
        0,
        actual,
    )
