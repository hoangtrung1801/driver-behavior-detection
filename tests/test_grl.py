import pytest
import torch

from cdira.models.grl import gradient_reverse


def test_gradient_reversal_negates_and_scales_gradient() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(x, 0.25).sum().backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(-0.25)
