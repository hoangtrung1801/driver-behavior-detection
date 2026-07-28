from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cdira.artifacts import RunArtifacts
from cdira.config import ExperimentConfig
from cdira.data.dataset import StateFarmDataset, build_transform
from cdira.data.download import download_competition
from cdira.data.manifests import SplitBundle, build_split_manifests
from cdira.domains.clustering import PseudoDomains, fit_pseudo_domains
from cdira.domains.features import (
    FeatureCache,
    extract_feature_cache,
    load_feature_cache,
)
from cdira.evaluation.metrics import classification_metrics
from cdira.evaluation.predict import collect_predictions
from cdira.models.cdira import CDIRA, MobileNetBaseline, RoutingPolicy
from cdira.reporting.report import build_reproduction_report
from cdira.runtime import select_device
from cdira.training.engine import Trainer
from cdira.training.losses import LossWeights


@dataclass(frozen=True)
class PreparedData:
    bundle: SplitBundle
    domains: PseudoDomains
    caches: dict[str, FeatureCache]


def domain_mapping(domains: PseudoDomains) -> dict[str, int]:
    return dict(domains.labels_by_path)


def _loader(
    manifest: Path,
    root: Path,
    transform: Any,
    domains: dict[str, int] | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[Any]:
    dataset = StateFarmDataset(manifest, root, transform, domains)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def prepare_domains(config: ExperimentConfig) -> PreparedData:
    if not config.paths.data_root.exists():
        download_competition(config.paths.data_root)
    bundle = build_split_manifests(
        config.paths.data_root, config.paths.manifest_root, config.seed
    )
    eval_transform = build_transform(False, config.data.image_size, False, config.data.brightness)
    caches: dict[str, FeatureCache] = {}
    for name, manifest in (
        ("train", bundle.train_path),
        ("validation", bundle.validation_path),
        ("test", bundle.test_path),
    ):
        destination = config.paths.domain_root / f"{name}.npz"
        if destination.exists() and destination.with_suffix(".json").exists():
            caches[name] = load_feature_cache(destination)
        else:
            loader = _loader(
                manifest,
                config.paths.data_root,
                eval_transform,
                None,
                config.training.batch_size,
                config.data.num_workers,
                False,
            )
            caches[name] = extract_feature_cache(
                loader, select_device(config.training.device), destination
            )
    domains = fit_pseudo_domains(
        caches["train"],
        caches["validation"],
        caches["test"],
        config.domains,
        config.seed,
    )
    domains.save(config.paths.domain_root / "pseudo_domains.npz")
    pd.DataFrame(
        [{"relative_path": path, "domain_id": domain_id} for path, domain_id in domains.labels_by_path.items()]
    ).to_csv(config.paths.domain_root / "domain_manifest.csv", index=False)
    return PreparedData(bundle, domains, caches)


def run_core_pipeline(config: ExperimentConfig, run: RunArtifacts) -> Path:
    prepared = prepare_domains(config)
    domain_ids = domain_mapping(prepared.domains)
    train_transform = build_transform(
        True,
        config.data.image_size,
        config.data.horizontal_flip,
        config.data.brightness,
    )
    eval_transform = build_transform(False, config.data.image_size, False, config.data.brightness)
    train_loader = _loader(
        prepared.bundle.train_path,
        config.paths.data_root,
        train_transform,
        domain_ids,
        config.training.batch_size,
        config.data.num_workers,
        True,
    )
    validation_loader = _loader(
        prepared.bundle.validation_path,
        config.paths.data_root,
        eval_transform,
        domain_ids,
        config.training.batch_size,
        config.data.num_workers,
        False,
    )
    test_loader = _loader(
        prepared.bundle.test_path,
        config.paths.data_root,
        eval_transform,
        domain_ids,
        config.training.batch_size,
        config.data.num_workers,
        False,
    )
    device = select_device(config.training.device)
    weights = LossWeights(**config.training.loss_weights.model_dump())
    model = CDIRA(
        num_classes=config.model.num_classes,
        num_domains=prepared.domains.k,
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
        model,
        device,
        config.training.max_epochs,
        config.training.patience,
        config.training.learning_rate,
        weights,
        config.training.confidence_threshold,
    )
    fit = trainer.fit(train_loader, validation_loader)
    model_path = run.root / "checkpoints" / "cdira.pt"
    torch.save(model.state_dict(), model_path)
    table = collect_predictions(
        model,
        test_loader,
        RoutingPolicy.HEAD,
        config.routing.threshold,
        device,
    )
    metrics = classification_metrics(table, config.model.num_classes)
    metrics.update({"epochs_completed": fit.epochs_completed, "selected_domains": prepared.domains.k})
    (run.root / "metrics" / "full.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    baseline = MobileNetBaseline(config.model.num_classes, pretrained=True)
    baseline_trainer = Trainer(
        baseline,
        device,
        config.training.max_epochs,
        config.training.patience,
        config.training.learning_rate,
    )
    baseline_trainer.fit(train_loader, validation_loader)
    baseline_table = _baseline_predictions(baseline, test_loader, device)
    (run.root / "metrics" / "baseline.json").write_text(
        json.dumps(classification_metrics(baseline_table, config.model.num_classes), indent=2),
        encoding="utf-8",
    )
    return build_reproduction_report(run.root)


def _baseline_predictions(
    model: MobileNetBaseline, loader: DataLoader[Any], device: torch.device
) -> Any:
    from cdira.evaluation.predict import PredictionTable

    model.eval().to(device)
    paths: list[str] = []
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    domains: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device)).cpu()
            paths.extend(str(path) for path in batch["relative_path"])
            targets.append(batch["target"])
            logits.append(output)
            domains.append(batch["domain"])
    joined_logits = torch.cat(logits)
    joined_targets = torch.cat(targets)
    return PredictionTable(
        paths,
        joined_targets,
        joined_logits.argmax(dim=1),
        joined_logits,
        joined_logits,
        joined_logits.softmax(dim=1).max(dim=1).values,
        torch.zeros(len(paths)),
        torch.zeros(len(paths), dtype=torch.bool),
        torch.cat(domains),
    )
