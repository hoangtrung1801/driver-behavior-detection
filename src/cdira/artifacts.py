from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from cdira.config import ExperimentConfig


@dataclass(frozen=True)
class RunArtifacts:
    root: Path

    @classmethod
    def create(
        cls, config: ExperimentConfig, run_id: str | None = None
    ) -> RunArtifacts:
        identifier = run_id or "run"
        root = config.paths.artifact_root / identifier
        root.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "checkpoints", "predictions", "metrics", "plots"):
            (root / name).mkdir()
        (root / "config.resolved.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        (root / "environment.json").write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "mps_available": bool(torch.backends.mps.is_available()),
                    "device_requested": config.training.device,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "fingerprints.json").write_text("{}", encoding="utf-8")
        return cls(root)
