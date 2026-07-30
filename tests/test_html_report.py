import json
from pathlib import Path

import pytest
import torch

from cdira.reporting.html_report import (
    build_html_reproduction_report,
    load_report_data,
)


def test_load_report_data_reads_nested_metrics(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "environment.json").write_text(
        json.dumps({"device_requested": "mps", "torch": "2.7.1"}),
        encoding="utf-8",
    )
    (tmp_path / "config.resolved.yaml").write_text(
        "profile: standard\ntraining:\n  max_epochs: 10\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics" / "full.json").write_text(
        json.dumps({"accuracy": 0.9, "macro": {"f1": 0.8}}),
        encoding="utf-8",
    )
    (tmp_path / "metrics" / "baseline.json").write_text(
        json.dumps({"accuracy": 0.85, "macro": {"f1": 0.75}}),
        encoding="utf-8",
    )

    data = load_report_data(tmp_path)

    assert data.profile == "standard"
    assert data.full["macro"]["f1"] == 0.8
    assert data.baseline["accuracy"] == 0.85


def test_load_report_data_names_invalid_json_path(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    path = tmp_path / "metrics" / "full.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="full.json"):
        load_report_data(tmp_path)


def test_load_report_data_excludes_batchnorm_buffers_from_parameters(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir()
    torch.save(
        {
            "layer.weight": torch.ones(2, 3),
            "batchnorm.running_mean": torch.ones(3),
            "batchnorm.running_var": torch.ones(3),
            "batchnorm.num_batches_tracked": torch.zeros(1),
        },
        tmp_path / "checkpoints" / "cdira.pt",
    )

    data = load_report_data(tmp_path)

    assert data.parameter_count == 6


@pytest.fixture
def populated_run(tmp_path: Path) -> Path:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "environment.json").write_text(
        json.dumps(
            {
                "device_requested": "mps",
                "torch": "2.7.1",
                "python": "3.12.9",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.resolved.yaml").write_text(
        "profile: standard\n"
        "data:\n  image_size: 224\n"
        "model:\n  top_k: 5\n"
        "training:\n"
        "  batch_size: 32\n"
        "  max_epochs: 10\n"
        "  loss_weights:\n"
        "    global_ce: 0.5\n"
        "    fused_ce: 1.0\n"
        "    routing_bce: 0.5\n"
        "    routing_regularizer: 0.01\n"
        "    domain_ce: 0.5\n",
        encoding="utf-8",
    )
    class_metrics = [
        {"precision": 0.98, "recall": 0.97, "f1": 0.975, "support": 20}
        for _ in range(10)
    ]
    confusion = [
        [20 if row == column else 0 for column in range(10)]
        for row in range(10)
    ]
    (tmp_path / "metrics" / "full.json").write_text(
        json.dumps(
            {
                "accuracy": 0.9764,
                "macro": {
                    "precision": 0.9765,
                    "recall": 0.9750,
                    "f1": 0.9755,
                },
                "weighted": {"f1": 0.9763},
                "per_class": class_metrics,
                "confusion_matrix": confusion,
                "roi_usage": 0.0259,
                "epochs_completed": 9,
                "selected_domains": 25,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "metrics" / "baseline.json").write_text(
        json.dumps(
            {
                "accuracy": 0.9813,
                "macro": {
                    "precision": 0.9815,
                    "recall": 0.9804,
                    "f1": 0.9808,
                },
                "weighted": {"f1": 0.9813},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_build_html_report_contains_solution_and_real_metrics(
    populated_run: Path,
) -> None:
    destination = build_html_reproduction_report(populated_run)
    html = destination.read_text(encoding="utf-8")

    assert "<!doctype html>" in html.lower()
    assert "<style>" in html
    assert "https://" not in html
    assert "C-DIRA architecture" in html
    assert "End-to-end pipeline" in html
    assert "Training objective" in html
    assert "Video frame aggregation" in html
    assert "97.55%" in html
    assert "MobileNetV3-Small" in html
    assert "Confusion matrix" in html
    assert "c7" in html
    assert "Reaching behind" in html
