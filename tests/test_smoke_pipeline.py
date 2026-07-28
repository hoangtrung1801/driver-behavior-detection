from pathlib import Path

from cdira.cli import run_smoke


def test_smoke_pipeline_creates_report(tmp_path: Path) -> None:
    report = run_smoke(tmp_path)
    assert report.exists()
    assert report.name == "report.md"
    assert '"steps": 1' in (report.parent / "metrics" / "full.json").read_text()
