from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build_reproduction_report(run_root: Path) -> Path:
    environment = _read_json(run_root / "environment.json", {})
    full = _read_json(run_root / "metrics" / "full.json", {})
    baseline = _read_json(run_root / "metrics" / "baseline.json", {})
    lines = [
        "# C-DIRA Reproduction Report",
        "",
        "## Locally measured",
        "",
        f"- Environment: {environment.get('device_requested', environment.get('device', 'unknown'))}",
        f"- C-DIRA macro F1: {full.get('macro_f1', 'not available')}",
        f"- MobileNetV3-Small macro F1: {baseline.get('macro_f1', 'not available')}",
        "",
        "## Paper-reported reference",
        "",
        "- C-DIRA macro F1: 0.992",
        "- C-DIRA ROI usage: 0.022",
        "- C-DIRA parameters: 2.165M",
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
