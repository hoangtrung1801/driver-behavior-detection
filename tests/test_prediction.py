import torch

from cdira.evaluation.predict import PredictionTable, routing_sweep
from cdira.models.cdira import InferenceOutput, RoutingPolicy


class StubModel:
    def predict(
        self, images: torch.Tensor, policy: RoutingPolicy, threshold: float
    ) -> InferenceOutput:
        probabilities = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        confidence = probabilities.max(dim=1).values
        route_probability = torch.tensor([0.1, 0.9])
        mask = (
            route_probability >= threshold
            if policy is RoutingPolicy.HEAD
            else confidence < threshold
        )
        return InferenceOutput(
            torch.log(probabilities),
            torch.log(probabilities),
            route_probability,
            confidence,
            mask,
            torch.zeros(2, 7, 7),
            torch.where(
                mask[:, None],
                torch.zeros(2, 1, dtype=torch.long),
                torch.full((2, 1), -1),
            ),
        )


def test_prediction_table_contains_one_row_per_input() -> None:
    table = PredictionTable(
        paths=["a", "b"],
        targets=torch.tensor([0, 1]),
        predictions=torch.tensor([0, 1]),
        logits=torch.zeros(2, 2),
        global_logits=torch.zeros(2, 2),
        confidence=torch.ones(2),
        routing_probability=torch.zeros(2),
        roi_mask=torch.zeros(2, dtype=torch.bool),
        domains=torch.zeros(2, dtype=torch.long),
    )
    assert len(table.paths) == len(table.targets) == 2


def test_head_threshold_sweep_has_nonincreasing_roi_usage() -> None:
    loader = [
        {
            "image": torch.zeros(2, 3, 8, 8),
            "target": torch.tensor([0, 1]),
            "relative_path": ["a", "b"],
            "domain": torch.zeros(2, dtype=torch.long),
        }
    ]
    frame = routing_sweep(StubModel(), loader, [RoutingPolicy.HEAD], [0.1, 0.5, 0.9])
    assert frame["roi_usage"].tolist() == sorted(frame["roi_usage"], reverse=True)
