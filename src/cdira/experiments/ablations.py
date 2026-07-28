from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model_mode: str
    domain_loss_weight: float
    routing_policy: str


def required_ablations() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("full", "full", 0.5, "head"),
        AblationSpec("no_roi", "global_only", 0.0, "none"),
        AblationSpec("no_adversarial", "full", 0.0, "head"),
        AblationSpec("no_dynamic_routing", "full", 0.5, "always"),
    )
