from pathlib import Path

from cdira.artifacts import RunArtifacts
from cdira.config import load_config


def test_run_artifacts_have_required_layout(tmp_path: Path) -> None:
    cfg = load_config(
        Path("configs/smoke.yaml"), [f"paths.artifact_root={tmp_path.as_posix()}"]
    )
    run = RunArtifacts.create(cfg, run_id="test-run")
    assert {p.name for p in run.root.iterdir()} >= {
        "logs",
        "checkpoints",
        "predictions",
        "metrics",
        "plots",
    }
    assert (run.root / "config.resolved.yaml").exists()
    assert (run.root / "environment.json").exists()
    assert (run.root / "fingerprints.json").exists()
