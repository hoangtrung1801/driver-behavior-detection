from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None  # type: ignore[attr-defined]


def gradient_reverse(x: Tensor, strength: float) -> Tensor:
    return cast(Tensor, _GradientReverse.apply(x, strength))  # type: ignore[no-untyped-call]
