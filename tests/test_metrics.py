import torch

from cdira.evaluation.metrics import classification_metrics
from cdira.evaluation.predict import PredictionTable


def test_metrics_report_macro_weighted_and_per_class() -> None:
    table = PredictionTable(
        paths=["a", "b", "c", "d"],
        targets=torch.tensor([0, 0, 1, 2]),
        predictions=torch.tensor([0, 1, 1, 2]),
        logits=torch.zeros(4, 3),
        global_logits=torch.zeros(4, 3),
        confidence=torch.ones(4),
        routing_probability=torch.zeros(4),
        roi_mask=torch.zeros(4, dtype=torch.bool),
        domains=torch.zeros(4, dtype=torch.long),
    )
    result = classification_metrics(table, num_classes=3)
    assert set(result) >= {
        "accuracy",
        "macro",
        "weighted",
        "per_class",
        "confusion_matrix",
    }
    assert result["accuracy"] == 0.75
