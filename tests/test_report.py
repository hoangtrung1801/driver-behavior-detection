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
