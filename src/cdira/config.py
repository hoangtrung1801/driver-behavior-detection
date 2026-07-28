from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictModel):
    data_root: Path
    manifest_root: Path
    domain_root: Path
    artifact_root: Path
    cache_root: Path


class DataConfig(StrictModel):
    image_size: int = Field(ge=32)
    split: tuple[float, float, float]
    horizontal_flip: bool
    brightness: float = Field(ge=0.0)
    num_workers: int = Field(ge=0)


class DomainConfig(StrictModel):
    candidates: list[int]
    sample_size: int = Field(gt=0)
    n_init: int = Field(gt=0)


class ModelConfig(StrictModel):
    num_classes: int = Field(gt=1)
    top_k: int = Field(gt=0)
    global_hidden: int = Field(gt=0)
    roi_hidden: int = Field(gt=0)
    fused_hidden: int = Field(gt=0)
    routing_hidden: int = Field(gt=0)
    domain_hidden: int = Field(gt=0)
    grl_strength: float = Field(gt=0)


class LossWeights(StrictModel):
    global_ce: float = Field(ge=0)
    fused_ce: float = Field(ge=0)
    routing_bce: float = Field(ge=0)
    routing_regularizer: float = Field(ge=0)
    domain_ce: float = Field(ge=0)


class TrainingConfig(StrictModel):
    device: Literal["mps", "cuda", "cpu"]
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    mixed_precision: bool
    confidence_threshold: float = Field(gt=0, lt=1)
    loss_weights: LossWeights


class RoutingConfig(StrictModel):
    primary_policy: Literal["head", "confidence"]
    comparison_policy: Literal["head", "confidence"]
    threshold: float = Field(gt=0, lt=1)


class EvaluationConfig(StrictModel):
    thresholds: list[float]
    corruption_levels: Literal["paper", "standard", "smoke"]


class ExperimentConfig(StrictModel):
    seed: int
    paths: PathsConfig
    data: DataConfig
    domains: DomainConfig
    model: ModelConfig
    training: TrainingConfig
    routing: RoutingConfig
    evaluation: EvaluationConfig
    profile: Literal["paper", "standard", "smoke"]


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override {dotted_key}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value: {override}")
        key, value = override.split("=", 1)
        _set_nested(raw, key, yaml.safe_load(value))
    return ExperimentConfig.model_validate(raw)
