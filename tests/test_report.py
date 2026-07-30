import json
from pathlib import Path

from cdira.reporting.report import build_reproduction_report


def test_report_distinguishes_measured_and_paper_values(tmp_path: Path) -> None:
    for name in ["logs", "checkpoints", "predictions", "metrics", "plots"]:
        (tmp_path / name).mkdir()
    (tmp_path / "environment.json").write_text(
        json.dumps({"device": "M4 Pro"}), encoding="utf-8"
    )
    (tmp_path / "fingerprints.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics" / "full.json").write_text(
        json.dumps({"macro_f1": 0.99}), encoding="utf-8"
    )
    report = build_reproduction_report(tmp_path)
    text = report.read_text()
    assert "Locally measured" in text
    assert "Paper-reported reference" in text
    assert "M4 Pro" in text
    assert "H100 latency was not reproduced" in text


def test_report_renders_multi_metric_comparison_and_details(tmp_path: Path) -> None:
    for name in ["logs", "checkpoints", "predictions", "metrics", "plots"]:
        (tmp_path / name).mkdir()
    (tmp_path / "environment.json").write_text(
        json.dumps({"device_requested": "mps", "torch": "2.7.1"}), encoding="utf-8"
    )
    (tmp_path / "config.resolved.yaml").write_text(
        "data:\n  image_size: 224\n"
        "training:\n  batch_size: 32\n  max_epochs: 10\n"
        "model:\n  top_k: 5\n",
        encoding="utf-8",
    )
    metrics = {
        "accuracy": 0.9,
        "macro": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
        "weighted": {"f1": 0.85},
        "per_class": [
            {"precision": 1.0, "recall": 0.5, "f1": 0.67, "support": 10},
        ],
        "confusion_matrix": [[5]],
        "roi_usage": 0.1,
        "epochs_completed": 3,
        "selected_domains": 4,
    }
    (tmp_path / "metrics" / "full.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (tmp_path / "metrics" / "baseline.json").write_text(
        json.dumps(
            {
                "accuracy": 0.8,
                "macro": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
                "weighted": {"f1": 0.75},
                "per_class": [],
                "roi_usage": 0.0,
            }
        ),
        encoding="utf-8",
    )

    report = build_reproduction_report(tmp_path)
    text = report.read_text()

    assert "| C-DIRA | 90.00% | 80.00% | 70.00% | 75.00% | 85.00% |" in text
    assert "| MobileNetV3-Small | 80.00% | 70.00% | 60.00% | 65.00% | 75.00% |" in text
    assert "## Per-class metrics (C-DIRA)" in text
    assert "| c0 | 100.00% | 50.00% | 67.00% | 10 |" in text
    assert "## Confusion matrix (C-DIRA)" in text
    assert "| Actual \\ Predicted | c0 |" in text
    assert "Selected pseudo-domains: 4" in text
