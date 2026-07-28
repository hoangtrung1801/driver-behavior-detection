from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LocoFold:
    held_out_domain: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


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
