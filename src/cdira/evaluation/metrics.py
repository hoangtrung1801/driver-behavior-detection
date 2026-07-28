from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from cdira.evaluation.predict import PredictionTable


def classification_metrics(table: PredictionTable, num_classes: int) -> dict[str, Any]:
    targets = table.targets.numpy()
    predictions = table.predictions.numpy()
    labels = list(range(num_classes))
    macro = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "weighted": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
        },
        "per_class": [
            {
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index in range(num_classes)
        ],
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
        "roi_usage": float(table.roi_mask.float().mean()),
        "per_class_roi_usage": {
            str(class_id): float(
                table.roi_mask[table.targets == class_id].float().mean()
            )
            if bool((table.targets == class_id).any())
            else 0.0
            for class_id in labels
        },
    }
