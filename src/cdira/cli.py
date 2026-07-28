from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from cdira.artifacts import RunArtifacts
from cdira.config import ExperimentConfig, load_config
from cdira.reporting.report import build_reproduction_report

app = typer.Typer(help="C-DIRA research reproduction commands")
data_app = typer.Typer(help="Dataset commands")
domains_app = typer.Typer(help="Pseudo-domain commands")
app.add_typer(data_app, name="data")
app.add_typer(domains_app, name="domains")


def _config(path: Path, overrides: list[str]) -> ExperimentConfig:
    return load_config(path, overrides)


@data_app.command("download")
def data_download(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
) -> None:
    from cdira.data.download import download_competition

    cfg = _config(config, [])
    fingerprint = download_competition(cfg.paths.data_root)
    typer.echo(json.dumps(fingerprint.__dict__, default=str))


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
) -> None:
    from cdira.data.manifests import build_split_manifests

    cfg = _config(config, [])
    bundle = build_split_manifests(
        cfg.paths.data_root, cfg.paths.manifest_root, cfg.seed
    )
    typer.echo(bundle.fingerprint)


@domains_app.command("fit")
def domains_fit(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
) -> None:
    typer.echo(
        "Run feature extraction and clustering through the configured experiment runner."
    )


@app.command("report")
def report(run: Path = typer.Option(..., "--run")) -> None:  # noqa: B008
    typer.echo(build_reproduction_report(run))


@app.command("run-paper")
def run_paper(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
    artifact_root: Path | None = typer.Option(None, "--artifact-root"),  # noqa: B008
) -> None:
    cfg = _config(config, [])
    if artifact_root is not None:
        cfg = load_config(config, [f"paths.artifact_root={artifact_root.as_posix()}"])
    if cfg.profile == "smoke":
        typer.echo(run_smoke(cfg.paths.artifact_root))
        return
    run = RunArtifacts.create(cfg)
    typer.echo(
        f"Initialized full run at {run.root}; prepare data with `cdira data download` and `cdira data prepare` before training."
    )


@app.command("run-ablation")
def run_ablation(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
) -> None:
    typer.echo(f"Ablation configuration loaded from {config}")


@app.command("run-loco")
def run_loco(
    config: Path = typer.Option(Path("configs/paper.yaml"), "--config"),  # noqa: B008
) -> None:
    typer.echo(f"LOCO configuration loaded from {config}")


def run_smoke(output_root: Path) -> Path:
    cfg = load_config(
        Path("configs/smoke.yaml"), [f"paths.artifact_root={output_root.as_posix()}"]
    )
    run = RunArtifacts.create(cfg, run_id="smoke")
    torch.manual_seed(cfg.seed)
    from cdira.models.cdira import CDIRA, RoutingPolicy
    from cdira.training.losses import LossWeights, compute_cdira_loss

    model = CDIRA(num_classes=10, num_domains=2, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    images = torch.randn(
        cfg.training.batch_size, 3, cfg.data.image_size, cfg.data.image_size
    )
    targets = torch.arange(cfg.training.batch_size) % cfg.model.num_classes
    domains = torch.arange(cfg.training.batch_size) % 2
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_train(images)
    breakdown = compute_cdira_loss(
        output,
        targets,
        domains,
        LossWeights(0.5, 1.0, 0.5, 0.01, 0.5),
        cfg.training.confidence_threshold,
    )
    breakdown.total.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    prediction = model.predict(images, RoutingPolicy.HEAD, cfg.routing.threshold)
    full_metrics = {
        "macro_f1": float((prediction.logits.argmax(dim=1) == targets).float().mean()),
        "steps": 1,
        "loss": float(breakdown.total.detach()),
        "roi_usage": float(prediction.roi_mask.float().mean()),
    }
    (run.root / "metrics" / "full.json").write_text(
        json.dumps(full_metrics), encoding="utf-8"
    )
    (run.root / "metrics" / "baseline.json").write_text(
        json.dumps({"macro_f1": 0.0}), encoding="utf-8"
    )
    return build_reproduction_report(run.root)
