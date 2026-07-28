import torch

from cdira.models.cdira import TrainOutput
from cdira.training.losses import LossWeights, compute_cdira_loss, routing_targets


def test_routing_targets_use_detached_global_difficulty() -> None:
    logits = torch.tensor([[8.0, 0.0], [0.2, 0.1], [0.0, 8.0]], requires_grad=True)
    targets = torch.tensor([0, 0, 0])
    result = routing_targets(logits, targets, confidence_threshold=0.9)
    assert result.tolist() == [0.0, 1.0, 1.0]
    assert not result.requires_grad


def test_total_loss_uses_paper_weights() -> None:
    output = TrainOutput(
        global_logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True),
        fused_logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True),
        routing_logits=torch.tensor([0.2, -0.2], requires_grad=True),
        domain_logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
        saliency=torch.ones(2, 2, 2),
        topk_indices=torch.zeros(2, 1, dtype=torch.long),
    )
    weights = LossWeights(0.5, 1.0, 0.5, 0.01, 0.5)
    breakdown = compute_cdira_loss(
        output, torch.tensor([0, 1]), torch.tensor([0, 1]), weights
    )
    expected = (
        0.5 * breakdown.global_ce
        + breakdown.fused_ce
        + 0.5 * breakdown.routing_bce
        + 0.01 * breakdown.routing_regularizer
        + 0.5 * breakdown.domain_ce
    )
    torch.testing.assert_close(breakdown.total, expected)
