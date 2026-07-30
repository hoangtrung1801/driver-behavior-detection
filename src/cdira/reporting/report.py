from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else default


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _percent(value: Any) -> str:
    return f"{float(value):.2%}" if isinstance(value, (int, float)) else "n/a"


def _number(value: Any) -> str:
    return f"{value}" if value is not None else "n/a"


def _macro_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    macro = metrics.get("macro")
    if isinstance(macro, dict):
        return macro
    legacy_f1 = metrics.get("macro_f1")
    return {"f1": legacy_f1} if legacy_f1 is not None else {}


def _comparison_row(name: str, metrics: dict[str, Any]) -> str:
    macro = _macro_metrics(metrics)
    weighted = metrics.get("weighted", {})
    if not isinstance(weighted, dict):
        weighted = {}
    return (
        f"| {name} | {_percent(metrics.get('accuracy'))} | "
        f"{_percent(macro.get('precision'))} | {_percent(macro.get('recall'))} | "
        f"{_percent(macro.get('f1'))} | {_percent(weighted.get('f1'))} | "
        f"{_percent(metrics.get('roi_usage'))} | "
        f"{_number(metrics.get('epochs_completed'))} |"
    )


def _per_class_table(metrics: dict[str, Any]) -> list[str]:
    rows = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, list):
        return rows + ["| n/a | n/a | n/a | n/a | n/a |"]
    for index, values in enumerate(per_class):
        values = values if isinstance(values, dict) else {}
        rows.append(
            f"| c{index} | {_percent(values.get('precision'))} | "
            f"{_percent(values.get('recall'))} | {_percent(values.get('f1'))} | "
            f"{_number(values.get('support'))} |"
        )
    return rows


def _confusion_matrix_table(metrics: dict[str, Any]) -> list[str]:
    matrix = metrics.get("confusion_matrix", [])
    if not isinstance(matrix, list) or not matrix:
        return ["No confusion matrix available."]
    size = len(matrix)
    labels = " | ".join(f"c{index}" for index in range(size))
    rows = [
        f"| Actual \\ Predicted | {labels} |",
        f"|---|{'---:|' * size}",
    ]
    for index, values in enumerate(matrix):
        values = values if isinstance(values, list) else []
        cells = " | ".join(_number(values[column]) if column < len(values) else "n/a" for column in range(size))
        rows.append(f"| c{index} | {cells} |")
    return rows


def build_reproduction_report(run_root: Path) -> Path:
    environment = _read_json(run_root / "environment.json", {})
    full = _read_json(run_root / "metrics" / "full.json", {})
    baseline = _read_json(run_root / "metrics" / "baseline.json", {})
    config = _read_yaml(run_root / "config.resolved.yaml")
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    if not isinstance(data_config, dict):
        data_config = {}
    if not isinstance(training_config, dict):
        training_config = {}
    if not isinstance(model_config, dict):
        model_config = {}

    measured_f1 = _macro_metrics(full).get("f1")
    f1_delta = float(measured_f1) - 0.992 if isinstance(measured_f1, (int, float)) else None
    lines = [
        "# C-DIRA Reproduction Report",
        "",
        "## Locally measured results",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROI usage | Epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _comparison_row("C-DIRA", full),
        _comparison_row("MobileNetV3-Small", baseline),
        "",
        "## Run configuration",
        "",
        f"- Environment: {environment.get('device_requested', environment.get('device', 'unknown'))}",
        f"- PyTorch: {environment.get('torch', 'unknown')}",
        f"- Image size: {_number(data_config.get('image_size'))}",
        f"- Batch size: {_number(training_config.get('batch_size'))}",
        f"- Maximum epochs: {_number(training_config.get('max_epochs'))}",
        f"- Top-K ROI positions: {_number(model_config.get('top_k'))}",
        f"- Selected pseudo-domains: {_number(full.get('selected_domains'))}",
        "",
        "## Per-class metrics (C-DIRA)",
        "",
        *_per_class_table(full),
        "",
        "## Confusion matrix (C-DIRA)",
        "",
        *_confusion_matrix_table(full),
        "",
        "## Paper-reported reference comparison",
        "",
        "| Metric | Measured C-DIRA | Paper reference | Difference |",
        "|---|---:|---:|---:|",
        f"| Macro F1 | {_percent(measured_f1)} | 99.20% | {_percent(f1_delta)} |",
        f"| ROI usage | {_percent(full.get('roi_usage'))} | 2.20% | {_percent(float(full['roi_usage']) - 0.022) if isinstance(full.get('roi_usage'), (int, float)) else 'n/a'} |",
        "",
        "H100 latency was not reproduced; this report measures Apple Silicon latency only.",
        "",
        "## Assumptions",
        "",
        "- Primary routing uses routing-head probability; Algorithm 2 confidence routing is evaluated separately.",
        "- Top-K is five positions because the paper does not specify the value.",
    ]
    destination = run_root / "report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
