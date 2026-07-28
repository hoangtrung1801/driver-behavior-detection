from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from cdira.config import ExperimentConfig
from cdira.data.dataset import StateFarmDataset, build_transform
from cdira.evaluation.metrics import classification_metrics
from cdira.evaluation.predict import collect_predictions
from cdira.models.cdira import CDIRA, MobileNetBaseline, RoutingPolicy
from cdira.pipeline import _baseline_predictions
from cdira.runtime import select_device
from cdira.training.engine import Trainer
from cdira.training.losses import LossWeights


@dataclass(frozen=True)
class LocoFold:
    held_out_domain: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def run_loco_fold(
    fold: LocoFold,
    model_kind: str,
    config: ExperimentConfig,
) -> dict[str, float | int | str]:
    if model_kind not in {"baseline", "cdira"}:
        raise ValueError("model_kind must be baseline or cdira")
    fold_root = config.paths.artifact_root / "loco" / f"domain-{fold.held_out_domain}" / model_kind
    fold_root.mkdir(parents=True, exist_ok=True)
    manifest_root = fold_root / "manifests"
    manifest_root.mkdir(exist_ok=True)
    frames = {"train": fold.train, "validation": fold.validation, "test": fold.test}
    paths = {}
    for name, frame in frames.items():
        path = manifest_root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    transform_train = build_transform(True, config.data.image_size, config.data.horizontal_flip, config.data.brightness)
    transform_eval = build_transform(False, config.data.image_size, False, config.data.brightness)
    domain_map = {
        str(row.relative_path): int(row.domain_id)
        for row in pd.concat([fold.train, fold.validation, fold.test]).itertuples()
    }
    loaders = {
        "train": torch.utils.data.DataLoader(
            StateFarmDataset(paths["train"], config.paths.data_root, transform_train, domain_map),
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        ),
        "validation": torch.utils.data.DataLoader(
            StateFarmDataset(paths["validation"], config.paths.data_root, transform_eval, domain_map),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            StateFarmDataset(paths["test"], config.paths.data_root, transform_eval, domain_map),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
    }
    device = select_device(config.training.device)
    if model_kind == "baseline":
        baseline_model = MobileNetBaseline(config.model.num_classes, pretrained=True)
        trainer = Trainer(baseline_model, device, config.training.max_epochs, config.training.patience, config.training.learning_rate)
        trainer.fit(loaders["train"], loaders["validation"])
        table = _baseline_predictions(baseline_model, loaders["test"], device)
    else:
        num_domains = int(pd.concat([fold.train, fold.validation, fold.test]).domain_id.max()) + 1
        cdira_model = CDIRA(
            config.model.num_classes,
            num_domains,
            top_k=config.model.top_k,
            global_hidden=config.model.global_hidden,
            roi_hidden=config.model.roi_hidden,
            fused_hidden=config.model.fused_hidden,
            routing_hidden=config.model.routing_hidden,
            domain_hidden=config.model.domain_hidden,
            grl_strength=config.model.grl_strength,
            pretrained=True,
        )
        trainer = Trainer(
            cdira_model,
            device,
            config.training.max_epochs,
            config.training.patience,
            config.training.learning_rate,
            LossWeights(**config.training.loss_weights.model_dump()),
            config.training.confidence_threshold,
        )
        trainer.fit(loaders["train"], loaders["validation"])
        table = collect_predictions(cdira_model, loaders["test"], RoutingPolicy.HEAD, config.routing.threshold, device)
    metrics = classification_metrics(table, config.model.num_classes)
    result = {
        "domain_id": fold.held_out_domain,
        "sample_count": len(fold.test),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro"]["f1"],
        "model": model_kind,
    }
    (fold_root / "complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_loco_folds(domain_manifest: pd.DataFrame, seed: int) -> list[LocoFold]:
    if "domain_id" not in domain_manifest or "class_id" not in domain_manifest:
        raise ValueError("LOCO manifest requires domain_id and class_id columns")
    folds: list[LocoFold] = []
    for held_out in sorted(int(value) for value in domain_manifest.domain_id.unique()):
        test = domain_manifest.loc[domain_manifest.domain_id == held_out].copy()
        remaining = domain_manifest.loc[domain_manifest.domain_id != held_out].copy()
        try:
            train, validation = train_test_split(
                remaining,
                test_size=0.2,
                stratify=remaining.class_id,
                random_state=seed,
            )
        except ValueError:
            train, validation = train_test_split(
                remaining, test_size=0.2, random_state=seed
            )
        folds.append(LocoFold(held_out, train, validation, test))
    return folds


def aggregate_loco(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate empty LOCO results")
    accuracies = [float(result["accuracy"]) for result in results]
    f1_scores = [float(result["macro_f1"]) for result in results]
    counts = np.asarray([int(result["sample_count"]) for result in results])
    order = np.argsort(counts, kind="stable")
    groups = np.array_split(order, 3)
    grouped = {}
    for name, indices in zip(("small", "middle", "large"), groups, strict=True):
        grouped[name] = {
            "mean_accuracy": float(np.mean([accuracies[index] for index in indices]))
            if len(indices)
            else None,
            "mean_macro_f1": float(np.mean([f1_scores[index] for index in indices]))
            if len(indices)
            else None,
        }
    return {
        "fold_count": len(results),
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_macro_f1": float(np.mean(f1_scores)),
        "groups": grouped,
    }
