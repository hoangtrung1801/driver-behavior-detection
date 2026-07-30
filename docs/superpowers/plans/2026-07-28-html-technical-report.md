# C-DIRA HTML Technical Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a detailed, responsive, offline HTML report for a completed C-DIRA run using the run's real configuration and metrics.

**Architecture:** Add a focused HTML-report module that loads and validates run artifacts, formats model and experiment data, and renders one self-contained semantic HTML document with embedded CSS and SVG-free CSS visualizations. Expose the generator through a Typer command while leaving the Markdown report unchanged.

**Tech Stack:** Python 3.12, standard-library HTML escaping, JSON, PyYAML, Typer, pytest, Ruff, mypy, semantic HTML5, embedded CSS.

## Global Constraints

- The output must work offline with no CDN, external fonts, images, scripts, or stylesheets.
- Preserve the existing Markdown report behavior.
- Use actual values from the selected run artifacts.
- Support desktop, mobile, and print layouts.
- Escape all values read from artifacts before inserting them into HTML.
- Do not claim real-time performance without a measured latency benchmark.
- Distinguish static-frame video aggregation from temporal action recognition.
- Keep the existing dirty working tree intact and avoid unrelated edits.

---

### Task 1: Artifact loading and metric formatting

**Files:**
- Create: `src/cdira/reporting/html_report.py`
- Create: `tests/test_html_report.py`

**Interfaces:**
- Consumes: `Path` pointing to a run directory containing `environment.json`, `config.resolved.yaml`, and metric JSON files.
- Produces: `load_report_data(run_root: Path) -> ReportData` and `_percent(value: object) -> str`.

- [x] **Step 1: Write failing artifact-loading tests**

```python
import json
from pathlib import Path

import pytest

from cdira.reporting.html_report import load_report_data


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
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_html_report.py -q
```

Expected: collection fails because `cdira.reporting.html_report` does not exist.

- [x] **Step 3: Implement typed data loading**

Create:

```python
@dataclass(frozen=True)
class ReportData:
    run_root: Path
    environment: dict[str, Any]
    config: dict[str, Any]
    full: dict[str, Any]
    baseline: dict[str, Any]
    profile: str
    checkpoint_bytes: int | None


def load_report_data(run_root: Path) -> ReportData:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_root}")
    environment = _read_json(run_root / "environment.json", required=False)
    full = _read_json(run_root / "metrics" / "full.json", required=False)
    baseline = _read_json(run_root / "metrics" / "baseline.json", required=False)
    config = _read_yaml(run_root / "config.resolved.yaml")
    checkpoint = run_root / "checkpoints" / "cdira.pt"
    return ReportData(
        run_root,
        environment,
        config,
        full,
        baseline,
        str(config.get("profile", "unknown")),
        checkpoint.stat().st_size if checkpoint.exists() else None,
    )
```

Wrap JSON and YAML parser errors in `ValueError` containing the source path.

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_html_report.py -q
```

Expected: artifact-loading tests pass.

- [ ] **Step 5: Commit artifact-loading unit**

Commit is pending because the environment denies writes to `.git/index.lock`.

```bash
git add src/cdira/reporting/html_report.py tests/test_html_report.py
git commit -m "feat: load HTML report artifacts"
```

---

### Task 2: Self-contained technical report rendering

**Files:**
- Modify: `src/cdira/reporting/html_report.py`
- Modify: `tests/test_html_report.py`

**Interfaces:**
- Consumes: `ReportData`.
- Produces: `render_html_report(data: ReportData) -> str` and `build_html_reproduction_report(run_root: Path) -> Path`.

- [x] **Step 1: Write failing document-content test**

```python
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
```

The `populated_run` fixture must write representative nested full/baseline
metrics, a ten-class confusion matrix, environment metadata, and resolved
configuration.

- [x] **Step 2: Run document test and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_html_report.py::test_build_html_report_contains_solution_and_real_metrics -q
```

Expected: failure because the renderer is not implemented.

- [x] **Step 3: Implement safe formatting and reusable visual components**

Add helpers with exact responsibilities:

```python
def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: object) -> str:
    return f"{float(value):.2%}" if isinstance(value, (int, float)) else "Not available"


def _metric(metrics: dict[str, Any], group: str, name: str) -> object:
    nested = metrics.get(group, {})
    return nested.get(name) if isinstance(nested, dict) else None
```

Add render functions for:

- executive metric cards;
- C-DIRA versus baseline table;
- class glossary;
- end-to-end pipeline;
- C-DIRA branch diagram;
- weighted loss expression;
- per-class F1 bars;
- confusion-matrix heatmap;
- paper comparison;
- efficiency and deployment facts;
- limitations;
- source-module map and commands.

All widths and heatmap intensities must be derived from validated numeric values
and bounded between zero and one.

- [x] **Step 4: Implement the complete semantic HTML shell**

`render_html_report` must emit:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C-DIRA Reproduction · Technical Report</title>
  <style>...</style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to report</a>
  <header>...</header>
  <div class="report-layout">
    <nav aria-label="Report sections">...</nav>
    <main id="main">...</main>
  </div>
  <footer>...</footer>
</body>
</html>
```

The embedded stylesheet must define design tokens, responsive breakpoints,
high-contrast focus states, data tables, architecture nodes, bar charts,
heatmap cells, code blocks, and `@media print` rules.

- [x] **Step 5: Implement file generation**

```python
def build_html_reproduction_report(run_root: Path) -> Path:
    data = load_report_data(run_root)
    destination = run_root / "report.html"
    destination.write_text(render_html_report(data), encoding="utf-8")
    return destination
```

- [x] **Step 6: Run focused HTML tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_html_report.py -q
```

Expected: all HTML-report tests pass.

- [ ] **Step 7: Commit rendering unit**

Commit is pending because the environment denies writes to `.git/index.lock`.

```bash
git add src/cdira/reporting/html_report.py tests/test_html_report.py
git commit -m "feat: render detailed C-DIRA HTML report"
```

---

### Task 3: CLI integration, real artifact generation, and documentation

**Files:**
- Modify: `src/cdira/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Generate: `artifacts/standard/run/report.html`

**Interfaces:**
- Consumes: `build_html_reproduction_report(run_root: Path) -> Path`.
- Produces: `cdira report-html --run <run-root>`.

- [x] **Step 1: Write failing CLI test**

```python
from typer.testing import CliRunner

from cdira.cli import app


def test_report_html_command_writes_destination(
    populated_run: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        ["report-html", "--run", str(populated_run)],
    )

    assert result.exit_code == 0
    assert (populated_run / "report.html").exists()
    assert "report.html" in result.stdout
```

- [x] **Step 2: Run CLI test and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli.py -q
```

Expected: failure because the `report-html` command is not registered.

- [x] **Step 3: Add the CLI command**

Add:

```python
@app.command("report-html")
def report_html(run: Path = typer.Option(..., "--run")) -> None:  # noqa: B008
    from cdira.reporting.html_report import build_html_reproduction_report

    typer.echo(build_html_reproduction_report(run))
```

- [x] **Step 4: Document generation and viewing**

Add to `README.md`:

```bash
uv run cdira report-html --run artifacts/standard/run
open artifacts/standard/run/report.html
```

Explain that the document is self-contained, offline, printable, and generated
from the run artifacts.

- [x] **Step 5: Generate the actual standard-run report**

Run:

```bash
PYTHONPATH=src .venv/bin/cdira report-html --run artifacts/standard/run
```

Expected: `artifacts/standard/run/report.html`.

- [x] **Step 6: Run full automated verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests app.py
.venv/bin/mypy src/cdira
git diff --check
```

Expected: all tests pass, Ruff is clean, mypy reports no issues, and the diff
contains no whitespace errors.

- [ ] **Step 7: Inspect the real HTML in a browser**

Browser inspection is pending because no in-app browser is available in this
session. Structural validation passed with 11 sections, 12 valid local links,
zero missing anchors, and zero remote references.

Start a local server:

```bash
.venv/bin/python -m http.server 8765
```

Open `http://localhost:8765/artifacts/standard/run/report.html`. Inspect at
desktop width and a mobile viewport near 390 pixels. Verify navigation,
architecture diagrams, tables, bars, heatmap, code blocks, focus states, and
print preview remain readable without horizontal clipping.

- [ ] **Step 8: Commit integration**

Commit is pending because the environment denies writes to `.git/index.lock`.

```bash
git add src/cdira/cli.py tests/test_cli.py README.md artifacts/standard/run/report.html
git commit -m "feat: add HTML reproduction report"
```
