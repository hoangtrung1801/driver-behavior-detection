from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cdira.models.cdira import CDIRA
from cdira.training.losses import LossBreakdown, LossWeights, compute_cdira_loss


@dataclass(frozen=True)
class FitResult:
    best_validation_loss: float
    epochs_completed: int
    history: list[dict[str, float]]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_epochs: int,
        patience: int,
        learning_rate: float = 1e-5,
        loss_weights: LossWeights | None = None,
        confidence_threshold: float = 0.9,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_weights = loss_weights or LossWeights(0.5, 1.0, 0.5, 0.01, 0.5)
        self.confidence_threshold = confidence_threshold

    def _loss(self, batch: dict[str, Any]) -> LossBreakdown | torch.Tensor:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)
        if isinstance(self.model, CDIRA):
            domains = batch["domain"].to(self.device)
            if (domains < 0).any():
                raise ValueError("C-DIRA training requires fixed pseudo-domain labels")
            output = self.model.forward_train(images)
            return compute_cdira_loss(
                output, targets, domains, self.loss_weights, self.confidence_threshold
            )
        return torch.nn.functional.cross_entropy(self.model(images), targets)

    def _run_epoch(self, loader: Any, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        phase = "train" if training else "validation"
        try:
            total_batches: int | None = len(loader)
        except TypeError:
            total_batches = None
        for batch_index, batch in enumerate(loader, start=1):
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(batch)
            value = loss.total if isinstance(loss, LossBreakdown) else loss
            if training:
                value.backward()  # type: ignore[no-untyped-call]
                self.optimizer.step()
            batch_size = len(batch["target"])
            total += float(value.detach().cpu()) * batch_size
            count += batch_size
            if batch_index % 50 == 0 or (
                total_batches is not None and batch_index == total_batches
            ):
                denominator = str(total_batches) if total_batches is not None else "?"
                print(
                    f"{phase} progress: batch {batch_index}/{denominator} "
                    f"loss={float(value.detach().cpu()):.4f}",
                    flush=True,
                )
        return total / max(count, 1)

    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        on_epoch_end: Callable[[list[dict[str, float]]], None] | None = None,
    ) -> FitResult:
        best = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(self.max_epochs):
            started = time.perf_counter()
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch + 1}/{self.max_epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if on_epoch_end is not None:
                on_epoch_end(history)
            if validation_loss < best:
                best = validation_loss
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        return FitResult(best, len(history), history)
