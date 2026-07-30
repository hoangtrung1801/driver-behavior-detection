from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class ReportData:
    run_root: Path
    environment: dict[str, Any]
    config: dict[str, Any]
    full: dict[str, Any]
    baseline: dict[str, Any]
    profile: str
    checkpoint_bytes: int | None
    parameter_count: int | None


def _read_json(path: Path, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required report artifact not found: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON report artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in report artifact: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read YAML report artifact {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in report artifact: {path}")
    return value


def load_report_data(run_root: Path) -> ReportData:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_root}")
    environment = _read_json(run_root / "environment.json", required=False)
    full = _read_json(run_root / "metrics" / "full.json", required=False)
    baseline = _read_json(run_root / "metrics" / "baseline.json", required=False)
    config = _read_yaml(run_root / "config.resolved.yaml")
    checkpoint = run_root / "checkpoints" / "cdira.pt"
    parameter_count: int | None = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            parameter_count = sum(
                int(value.numel())
                for name, value in state.items()
                if isinstance(value, torch.Tensor)
                and not str(name).endswith(
                    ("running_mean", "running_var", "num_batches_tracked")
                )
            )
    return ReportData(
        run_root=run_root,
        environment=environment,
        config=config,
        full=full,
        baseline=baseline,
        profile=str(config.get("profile", "unknown")),
        checkpoint_bytes=checkpoint.stat().st_size if checkpoint.exists() else None,
        parameter_count=parameter_count,
    )


CLASS_NAMES = {
    "c0": "Safe driving",
    "c1": "Texting — right hand",
    "c2": "Talking on phone — right",
    "c3": "Texting — left hand",
    "c4": "Talking on phone — left",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger",
}


STYLES = """
:root {
  color-scheme: dark;
  --bg: #071018;
  --surface: #0d1924;
  --surface-2: #122331;
  --surface-3: #172c3a;
  --line: #274253;
  --text: #edf7fb;
  --muted: #9eb4c1;
  --cyan: #40d9ff;
  --cyan-soft: #9bebff;
  --amber: #f5bb57;
  --green: #55d69e;
  --red: #ff7d7d;
  --radius: 18px;
  --shadow: 0 18px 70px rgba(0, 0, 0, .28);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--text);
  background:
    radial-gradient(circle at 90% 0%, rgba(64, 217, 255, .12), transparent 32rem),
    radial-gradient(circle at 10% 30%, rgba(85, 214, 158, .07), transparent 30rem),
    var(--bg);
  line-height: 1.65;
}
a { color: var(--cyan-soft); }
a:focus-visible { outline: 3px solid var(--amber); outline-offset: 4px; }
.skip-link {
  position: fixed;
  left: 1rem;
  top: -5rem;
  z-index: 100;
  padding: .75rem 1rem;
  background: var(--amber);
  color: #101820;
  border-radius: 8px;
}
.skip-link:focus { top: 1rem; }
.hero {
  min-height: 72vh;
  display: grid;
  align-content: end;
  padding: clamp(4rem, 10vw, 9rem) clamp(1.25rem, 7vw, 8rem);
  border-bottom: 1px solid var(--line);
}
.eyebrow, .kicker {
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: .16em;
  font-weight: 700;
  font-size: .75rem;
}
h1, h2, h3 { line-height: 1.12; letter-spacing: -.025em; }
h1 { font-size: clamp(3rem, 8vw, 7rem); max-width: 12ch; margin: .5rem 0 1.5rem; }
h2 { font-size: clamp(2rem, 4vw, 3.4rem); margin: 0 0 1.25rem; }
h3 { font-size: 1.25rem; margin: 0 0 .65rem; }
.hero-summary { max-width: 68ch; color: var(--muted); font-size: 1.12rem; }
.hero-meta { display: flex; gap: .65rem; flex-wrap: wrap; margin-top: 2rem; }
.pill {
  display: inline-flex;
  align-items: center;
  padding: .4rem .72rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(13, 25, 36, .65);
  color: var(--muted);
  font-size: .82rem;
}
.report-layout {
  width: min(1500px, calc(100% - 2rem));
  margin: 0 auto;
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: clamp(1.5rem, 4vw, 5rem);
}
.toc {
  position: sticky;
  top: 1.5rem;
  align-self: start;
  padding: 2rem 0;
  max-height: 95vh;
  overflow: auto;
}
.toc strong { display: block; margin-bottom: .75rem; }
.toc a {
  display: block;
  color: var(--muted);
  text-decoration: none;
  padding: .36rem .55rem;
  border-left: 1px solid var(--line);
  font-size: .86rem;
}
.toc a:hover { color: var(--text); border-color: var(--cyan); }
main { min-width: 0; }
.section {
  padding: clamp(4.5rem, 9vw, 8rem) 0;
  border-bottom: 1px solid var(--line);
}
.lede { max-width: 72ch; color: var(--muted); font-size: 1.08rem; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}
.metric-card, .panel, .class-card, .flow-node, .arch-node {
  border: 1px solid var(--line);
  background: linear-gradient(145deg, rgba(18, 35, 49, .96), rgba(13, 25, 36, .96));
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.metric-card { padding: 1.4rem; min-height: 150px; }
.metric-card .label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: clamp(1.8rem, 4vw, 3rem); line-height: 1; margin: .7rem 0; font-weight: 760; }
.metric-card .note { color: var(--muted); font-size: .82rem; }
.cyan { color: var(--cyan); }
.amber { color: var(--amber); }
.green { color: var(--green); }
.red { color: var(--red); }
.panel { padding: clamp(1.25rem, 3vw, 2rem); margin: 1.5rem 0; overflow: hidden; }
.callout {
  border-left: 3px solid var(--amber);
  background: rgba(245, 187, 87, .08);
  padding: 1rem 1.2rem;
  border-radius: 0 12px 12px 0;
  color: #f8deb0;
}
.good-callout { border-color: var(--green); background: rgba(85, 214, 158, .08); color: #c6f7e2; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; margin: 1.4rem 0; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
caption { text-align: left; padding: 1rem; color: var(--muted); }
th, td { padding: .8rem .9rem; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead { background: var(--surface-3); }
tbody tr:hover { background: rgba(64, 217, 255, .035); }
.class-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .75rem; }
.class-card { padding: 1rem 1.1rem; display: flex; gap: 1rem; align-items: center; }
.class-code { color: var(--cyan); font-family: ui-monospace, monospace; font-weight: 800; font-size: 1.1rem; }
.flow {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 2rem 1.5rem;
  margin: 2rem 0;
}
.flow-node { position: relative; padding: 1.2rem; min-height: 150px; }
.flow-node:not(:last-child)::after {
  content: "→";
  position: absolute;
  right: -1.45rem;
  top: 42%;
  color: var(--cyan);
  font-size: 1.4rem;
}
.step { color: var(--cyan); font-family: ui-monospace, monospace; font-size: .76rem; }
.flow-node p, .arch-node p { color: var(--muted); margin: .5rem 0 0; font-size: .9rem; }
.architecture {
  display: grid;
  grid-template-columns: 1fr 1.2fr 1.2fr;
  gap: 1rem;
  align-items: stretch;
  margin: 2rem 0;
}
.arch-column { display: grid; gap: 1rem; }
.arch-node { padding: 1.25rem; }
.arch-node.training { border-color: rgba(245, 187, 87, .55); }
.arch-node.inference { border-color: rgba(64, 217, 255, .55); }
.connector {
  display: grid;
  place-items: center;
  color: var(--cyan);
  font-family: ui-monospace, monospace;
  font-size: .82rem;
}
.equation {
  padding: 1.5rem;
  background: #050b11;
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--cyan-soft);
  font-size: clamp(.82rem, 2vw, 1rem);
}
.bar-list { display: grid; gap: .75rem; }
.bar-row { display: grid; grid-template-columns: 190px 1fr 74px; gap: .8rem; align-items: center; }
.bar-track { height: 12px; background: #061019; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #1d8cac, var(--cyan)); border-radius: inherit; }
.heatmap th, .heatmap td { min-width: 48px; text-align: center; padding: .62rem; }
.heatmap td { color: white; text-shadow: 0 1px 2px #000; }
.two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.fact-list { list-style: none; padding: 0; margin: 0; }
.fact-list li { padding: .8rem 0; border-bottom: 1px solid var(--line); }
.fact-list strong { color: var(--text); }
.fact-list span { color: var(--muted); float: right; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre {
  padding: 1.25rem;
  background: #050b11;
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow-x: auto;
  color: var(--cyan-soft);
}
footer { padding: 3rem 1.25rem; text-align: center; color: var(--muted); }
@media (max-width: 1050px) {
  .metric-grid { grid-template-columns: repeat(2, 1fr); }
  .flow { grid-template-columns: repeat(2, 1fr); }
  .flow-node:nth-child(2n)::after { display: none; }
  .architecture { grid-template-columns: 1fr; }
  .connector { min-height: 40px; }
}
@media (max-width: 760px) {
  .hero { min-height: auto; padding-top: 6rem; }
  .report-layout { display: block; width: min(100% - 2rem, 900px); }
  .toc { position: static; max-height: none; display: flex; gap: .4rem; overflow-x: auto; }
  .toc strong { display: none; }
  .toc a { border: 1px solid var(--line); border-radius: 999px; white-space: nowrap; }
  .metric-grid, .class-grid, .flow, .two-column { grid-template-columns: 1fr; }
  .flow-node::after { display: none; }
  .bar-row { grid-template-columns: 115px 1fr 58px; font-size: .82rem; }
  .section { padding: 4rem 0; }
}
@media print {
  :root { color-scheme: light; --bg: white; --surface: white; --surface-2: #f4f7f9; --surface-3: #e9eff3; --line: #cad5dc; --text: #101820; --muted: #52636d; --cyan: #007d9d; --cyan-soft: #00617a; }
  body { background: white; font-size: 10pt; }
  .hero { min-height: auto; padding: 2rem 0; }
  .report-layout { display: block; width: 100%; }
  .toc, .skip-link { display: none; }
  .section { padding: 2rem 0; break-inside: auto; }
  .panel, .metric-card, .class-card, .flow-node, .arch-node { box-shadow: none; break-inside: avoid; }
  .table-wrap { overflow: visible; break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: object) -> str:
    return (
        f"{float(value):.2%}"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else "Not available"
    )


def _number(value: object, suffix: str = "") -> str:
    if value is None:
        return "Not available"
    return f"{_escape(value)}{suffix}"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(metrics: dict[str, Any], group: str, name: str) -> object:
    return _mapping(metrics.get(group)).get(name)


def _config_group(data: ReportData, name: str) -> dict[str, Any]:
    return _mapping(data.config.get(name))


def _metric_card(label: str, value: str, note: str, tone: str = "cyan") -> str:
    return (
        '<article class="metric-card">'
        f'<div class="label">{_escape(label)}</div>'
        f'<div class="value {tone}">{_escape(value)}</div>'
        f'<div class="note">{_escape(note)}</div>'
        "</article>"
    )


def _comparison_table(data: ReportData) -> str:
    def row(name: str, metrics: dict[str, Any]) -> str:
        return (
            f"<tr><th scope=\"row\">{_escape(name)}</th>"
            f"<td>{_percent(metrics.get('accuracy'))}</td>"
            f"<td>{_percent(_metric(metrics, 'macro', 'precision'))}</td>"
            f"<td>{_percent(_metric(metrics, 'macro', 'recall'))}</td>"
            f"<td>{_percent(_metric(metrics, 'macro', 'f1'))}</td>"
            f"<td>{_percent(_metric(metrics, 'weighted', 'f1'))}</td>"
            f"<td>{_percent(metrics.get('roi_usage'))}</td></tr>"
        )

    return (
        '<div class="table-wrap"><table>'
        "<caption>Measured on the local image-level test split.</caption>"
        "<thead><tr><th>Model</th><th>Accuracy</th><th>Macro precision</th>"
        "<th>Macro recall</th><th>Macro F1</th><th>Weighted F1</th>"
        "<th>ROI usage</th></tr></thead><tbody>"
        + row("C-DIRA", data.full)
        + row("MobileNetV3-Small", data.baseline)
        + "</tbody></table></div>"
    )


def _class_glossary() -> str:
    return '<div class="class-grid">' + "".join(
        '<article class="class-card">'
        f'<span class="class-code">{code}</span>'
        f"<span>{_escape(label)}</span></article>"
        for code, label in CLASS_NAMES.items()
    ) + "</div>"


def _pipeline() -> str:
    steps = [
        ("01", "Raw competition data", "22,424 labeled JPEGs plus subject metadata."),
        ("02", "Validation & manifests", "Fingerprint files and create reproducible split CSVs."),
        ("03", "Feature cache", "Frozen ImageNet MobileNetV3 features for every split."),
        ("04", "Pseudo-domains", "K-means candidates selected by silhouette validation."),
        ("05", "C-DIRA training", "Global, ROI, routing, and adversarial domain objectives."),
        ("06", "Baseline training", "MobileNetV3-Small control under the same split."),
        ("07", "Evaluation", "Multi-metric results, class detail, confusion, and ROI use."),
        ("08", "Deployment app", "Image inference and sampled-frame video aggregation."),
    ]
    return '<div class="flow">' + "".join(
        '<article class="flow-node">'
        f'<div class="step">STEP {step}</div><h3>{_escape(title)}</h3>'
        f"<p>{_escape(description)}</p></article>"
        for step, title, description in steps
    ) + "</div>"


def _architecture(data: ReportData) -> str:
    top_k = _config_group(data, "model").get("top_k", 5)
    return f"""
<div class="architecture" role="img" aria-label="C-DIRA training and inference architecture">
  <div class="arch-column">
    <article class="arch-node inference">
      <div class="kicker">Shared encoder</div>
      <h3>MobileNetV3-Small</h3>
      <p>Produces a spatial feature map and a 576-dimensional global vector.</p>
    </article>
    <div class="connector">feature map + global vector ↓</div>
  </div>
  <div class="arch-column">
    <article class="arch-node inference">
      <div class="kicker">Fast path</div>
      <h3>Global classifier</h3>
      <p>Predicts ten driver-behavior classes from the pooled global feature.</p>
    </article>
    <article class="arch-node inference">
      <div class="kicker">Conditional path</div>
      <h3>Routing head</h3>
      <p>Estimates whether ROI refinement should replace the global prediction.</p>
    </article>
    <article class="arch-node training">
      <div class="kicker amber">Training only</div>
      <h3>Domain classifier + GRL</h3>
      <p>Predicts pseudo-domain while gradient reversal makes the backbone domain-invariant.</p>
    </article>
  </div>
  <div class="arch-column">
    <article class="arch-node inference">
      <div class="kicker">ROI branch</div>
      <h3>Top-{_escape(top_k)} salient positions</h3>
      <p>Feature norms rank spatial locations; selected vectors are pooled and refined.</p>
    </article>
    <article class="arch-node inference">
      <div class="kicker">Final classifier</div>
      <h3>Global + ROI fusion</h3>
      <p>Concatenates global and refined ROI features to produce fused logits.</p>
    </article>
  </div>
</div>
"""


def _loss_section(data: ReportData) -> str:
    weights = _mapping(_config_group(data, "training").get("loss_weights"))
    rows = [
        ("Global CE", "global_ce", "Entire-image behavior classification"),
        ("Fused CE", "fused_ce", "Global plus ROI behavior classification"),
        ("Routing BCE", "routing_bce", "Predict when global output needs refinement"),
        ("Routing regularizer", "routing_regularizer", "Discourage unnecessary ROI computation"),
        ("Domain CE", "domain_ce", "Pseudo-domain prediction through gradient reversal"),
    ]
    table_rows = "".join(
        f"<tr><th scope=\"row\">{_escape(name)}</th>"
        f"<td>{_number(weights.get(key))}</td><td>{_escape(purpose)}</td></tr>"
        for name, key, purpose in rows
    )
    return f"""
<div class="equation" aria-label="Weighted C-DIRA loss">
L = λg·CE(global) + λf·CE(fused) + λr·BCE(route)
  + λu·mean(route probability) + λd·CE(domain)
</div>
<div class="table-wrap"><table>
  <caption>Loss weights from the resolved run configuration.</caption>
  <thead><tr><th>Term</th><th>Weight</th><th>Purpose</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table></div>
"""


def _per_class_results(data: ReportData) -> str:
    values = data.full.get("per_class")
    if not isinstance(values, list) or not values:
        return '<div class="callout">Per-class metrics were not available.</div>'
    rows: list[str] = []
    bars: list[str] = []
    for index, raw in enumerate(values):
        metrics = _mapping(raw)
        code = f"c{index}"
        f1 = metrics.get("f1")
        numeric_f1 = float(f1) if isinstance(f1, (int, float)) else 0.0
        width = min(max(numeric_f1, 0.0), 1.0) * 100
        rows.append(
            f"<tr><th scope=\"row\">{code} · {_escape(CLASS_NAMES.get(code, code))}</th>"
            f"<td>{_percent(metrics.get('precision'))}</td>"
            f"<td>{_percent(metrics.get('recall'))}</td>"
            f"<td>{_percent(f1)}</td><td>{_number(metrics.get('support'))}</td></tr>"
        )
        bars.append(
            '<div class="bar-row">'
            f"<span>{code} · {_escape(CLASS_NAMES.get(code, code))}</span>"
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f"<strong>{_percent(f1)}</strong></div>"
        )
    return (
        '<div class="panel"><h3>Macro view by class</h3><div class="bar-list">'
        + "".join(bars)
        + '</div></div><div class="table-wrap"><table>'
        "<caption>Precision, recall, F1, and support for each behavior.</caption>"
        "<thead><tr><th>Behavior</th><th>Precision</th><th>Recall</th>"
        "<th>F1</th><th>Support</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _confusion_matrix(data: ReportData) -> str:
    matrix = data.full.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return '<div class="callout">Confusion matrix was not available.</div>'
    numeric = [
        [int(value) if isinstance(value, (int, float)) else 0 for value in row]
        for row in matrix
        if isinstance(row, list)
    ]
    if not numeric:
        return '<div class="callout">Confusion matrix had an unexpected shape.</div>'
    maximum = max(max(row, default=0) for row in numeric) or 1
    labels = [f"c{index}" for index in range(len(numeric))]
    header = "".join(f"<th scope=\"col\">{label}</th>" for label in labels)
    rows = []
    for row_index, row in enumerate(numeric):
        cells = []
        for column in range(len(labels)):
            value = row[column] if column < len(row) else 0
            alpha = 0.08 + 0.78 * min(max(value / maximum, 0.0), 1.0)
            cells.append(
                f'<td style="background:rgba(64,217,255,{alpha:.3f})">{value}</td>'
            )
        rows.append(
            f'<tr><th scope="row">{labels[row_index]}</th>{"".join(cells)}</tr>'
        )
    return (
        '<div class="table-wrap"><table class="heatmap">'
        "<caption>Rows are actual classes; columns are predicted classes.</caption>"
        f"<thead><tr><th>Actual \\ Predicted</th>{header}</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _hardest_classes(data: ReportData) -> str:
    per_class = data.full.get("per_class")
    if not isinstance(per_class, list):
        return "Per-class interpretation is unavailable."
    ranked: list[tuple[float, str]] = []
    for index, raw in enumerate(per_class):
        f1 = _mapping(raw).get("f1")
        if isinstance(f1, (int, float)):
            ranked.append((float(f1), f"c{index}"))
    if not ranked:
        return "Per-class interpretation is unavailable."
    ranked.sort()
    summaries = [
        f"{code} ({CLASS_NAMES.get(code, code)}) at {score:.2%}"
        for score, code in ranked[:2]
    ]
    return "The lowest measured class F1 scores are " + " and ".join(summaries) + "."


def _paper_comparison(data: ReportData) -> str:
    measured_f1 = _metric(data.full, "macro", "f1")
    roi_usage = data.full.get("roi_usage")
    f1_delta = (
        float(measured_f1) - 0.992
        if isinstance(measured_f1, (int, float))
        else None
    )
    roi_delta = (
        float(roi_usage) - 0.022 if isinstance(roi_usage, (int, float)) else None
    )
    return f"""
<div class="table-wrap"><table>
  <caption>Paper values are references, not locally measured results.</caption>
  <thead><tr><th>Metric</th><th>Local C-DIRA</th><th>Paper</th><th>Difference</th></tr></thead>
  <tbody>
    <tr><th scope="row">Macro F1</th><td>{_percent(measured_f1)}</td><td>99.20%</td><td>{_percent(f1_delta)}</td></tr>
    <tr><th scope="row">ROI usage</th><td>{_percent(roi_usage)}</td><td>2.20%</td><td>{_percent(roi_delta)}</td></tr>
  </tbody>
</table></div>
"""


def _executive_conclusion(data: ReportData) -> str:
    cdira = _metric(data.full, "macro", "f1")
    baseline = _metric(data.baseline, "macro", "f1")
    if isinstance(cdira, (int, float)) and isinstance(baseline, (int, float)):
        delta = float(cdira) - float(baseline)
        if delta >= 0:
            return (
                f"C-DIRA exceeded the local baseline by {delta:.2%} macro F1. "
                "The result supports the added routing and adaptation complexity for this run."
            )
        return (
            f"C-DIRA trailed the local baseline by {abs(delta):.2%} macro F1. "
            "This standard-profile run does not demonstrate an accuracy gain from the "
            "additional routing, ROI, and domain-adaptation machinery."
        )
    return "The run does not contain enough metrics for a model comparison."


def _source_map() -> str:
    modules = [
        ("Data validation and manifests", "src/cdira/data/download.py · manifests.py"),
        ("Pseudo-domain features", "src/cdira/domains/features.py · clustering.py"),
        ("C-DIRA model", "src/cdira/models/cdira.py · roi.py · grl.py"),
        ("Loss and training loop", "src/cdira/training/losses.py · engine.py"),
        ("Evaluation and metrics", "src/cdira/evaluation/"),
        ("Image and video demo", "src/cdira/streamlit_app.py"),
        ("Experiment orchestration", "src/cdira/pipeline.py · cli.py"),
    ]
    return '<ul class="fact-list">' + "".join(
        f"<li><strong>{_escape(name)}</strong><span>{_escape(path)}</span></li>"
        for name, path in modules
    ) + "</ul>"


def render_html_report(data: ReportData) -> str:
    environment = data.environment
    training = _config_group(data, "training")
    model = _config_group(data, "model")
    data_config = _config_group(data, "data")
    cdira_f1 = _metric(data.full, "macro", "f1")
    baseline_f1 = _metric(data.baseline, "macro", "f1")
    checkpoint_mb = (
        f"{data.checkpoint_bytes / (1024 * 1024):.1f} MB"
        if data.checkpoint_bytes is not None
        else "Not available"
    )
    parameter_text = (
        f"{data.parameter_count:,}"
        if data.parameter_count is not None
        else "Not available"
    )
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    nav = [
        ("summary", "Executive summary"),
        ("task", "Driver behavior task"),
        ("pipeline", "End-to-end pipeline"),
        ("architecture", "C-DIRA architecture"),
        ("training", "Training objective"),
        ("domains", "Pseudo-domains"),
        ("inference", "Inference"),
        ("results", "Results"),
        ("efficiency", "Efficiency"),
        ("limitations", "Limitations"),
        ("engineering", "Engineering map"),
    ]
    nav_html = "".join(f'<a href="#{target}">{label}</a>' for target, label in nav)
    class_count = data_config.get("num_classes", model.get("num_classes", 10))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Detailed C-DIRA driver-behavior reproduction report">
  <title>C-DIRA Reproduction · Technical Report</title>
  <style>{STYLES}</style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to report</a>
  <header class="hero">
    <div class="eyebrow">Research reproduction · {_escape(data.profile)} profile</div>
    <h1>C-DIRA driver behavior detection</h1>
    <p class="hero-summary">A technical and engineering report covering the complete
    static-image reproduction, conditional ROI architecture, pseudo-domain adaptation,
    evaluation results, and sampled-frame video deployment.</p>
    <div class="hero-meta">
      <span class="pill">Device · {_escape(environment.get("device_requested", environment.get("device", "unknown")))}</span>
      <span class="pill">PyTorch · {_escape(environment.get("torch", "unknown"))}</span>
      <span class="pill">Generated · {_escape(generated)}</span>
      <span class="pill">Run · {_escape(data.run_root)}</span>
    </div>
  </header>
  <div class="report-layout">
    <nav class="toc" aria-label="Report sections"><strong>Contents</strong>{nav_html}</nav>
    <main id="main">
      <section class="section" id="summary">
        <div class="kicker">01 · Decision view</div>
        <h2>Executive summary</h2>
        <p class="lede">{_escape(_executive_conclusion(data))}</p>
        <div class="metric-grid">
          {_metric_card("C-DIRA macro F1", _percent(cdira_f1), "Local standard-profile run")}
          {_metric_card("Baseline macro F1", _percent(baseline_f1), "MobileNetV3-Small control", "green")}
          {_metric_card("ROI usage", _percent(data.full.get("roi_usage")), "Conditional refinement frequency", "amber")}
          {_metric_card("Pseudo-domains", _number(data.full.get("selected_domains")), "Silhouette-selected clusters")}
        </div>
        <div class="callout">This report separates locally measured values from
        paper-reported references. The local run used the standard profile with at
        most {_number(training.get("max_epochs"))} epochs; it is not the full 50-epoch paper profile.</div>
      </section>

      <section class="section" id="task">
        <div class="kicker">02 · Problem framing</div>
        <h2>Driver-behavior task</h2>
        <p class="lede">The system assigns one of {_escape(class_count)} State Farm
        behavior labels to a cabin-camera frame. It is a supervised static-image
        classifier: posture, hands, objects, and cabin context drive the prediction.</p>
        {_class_glossary()}
        <div class="callout">A single frame cannot prove temporal intent. Video support
        in the demo aggregates independent frame predictions; it is not a recurrent,
        transformer, or optical-flow action-recognition model.</div>
      </section>

      <section class="section" id="pipeline">
        <div class="kicker">03 · Data to deployment</div>
        <h2>End-to-end pipeline</h2>
        <p class="lede">Every stage leaves inspectable artifacts, allowing the run to
        be reproduced and audited independently from the interactive demo.</p>
        {_pipeline()}
      </section>

      <section class="section" id="architecture">
        <div class="kicker">04 · Model system</div>
        <h2>C-DIRA architecture</h2>
        <p class="lede">C-DIRA shares one compact MobileNetV3-Small backbone, then
        combines a fast global classifier with a selectively invoked ROI branch.
        A domain-adversarial head shapes training but is not needed for final behavior output.</p>
        {_architecture(data)}
        <div class="two-column">
          <article class="panel"><h3>Routing-head policy</h3><p>The primary implementation
          invokes ROI refinement when the learned routing probability crosses
          {_percent(_mapping(data.config.get("routing")).get("threshold"))}.</p></article>
          <article class="panel"><h3>Confidence policy</h3><p>The comparison policy invokes
          ROI refinement when the global classifier confidence is below its threshold.
          Both are retained because the paper's prose and algorithm differ.</p></article>
        </div>
      </section>

      <section class="section" id="training">
        <div class="kicker">05 · Optimization</div>
        <h2>Training objective</h2>
        <p class="lede">The backbone must classify behavior, support a stronger fused
        decision, learn when refinement is useful, limit expensive routing, and remove
        pseudo-domain information from shared features.</p>
        {_loss_section(data)}
      </section>

      <section class="section" id="domains">
        <div class="kicker">06 · Generalization mechanism</div>
        <h2>Pseudo-domain adaptation</h2>
        <div class="two-column">
          <article class="panel"><h3>Discover domains</h3><ol>
            <li>Resize and normalize every split.</li>
            <li>Extract frozen ImageNet MobileNet features.</li>
            <li>Evaluate K-means candidates from the configuration.</li>
            <li>Select K using validation silhouette score.</li>
            <li>Persist one domain label per image path.</li>
          </ol></article>
          <article class="panel"><h3>Remove domain shortcuts</h3><p>A gradient-reversal
          layer passes features unchanged forward but reverses the domain gradient.
          The domain classifier gets better at recognizing clusters while the backbone
          is pushed to erase cluster-specific cues.</p>
          <p class="cyan"><strong>{_number(data.full.get("selected_domains"))}</strong>
          pseudo-domains were selected in this run.</p></article>
        </div>
      </section>

      <section class="section" id="inference">
        <div class="kicker">07 · Runtime behavior</div>
        <h2>Image and video inference</h2>
        <div class="two-column">
          <article class="panel"><h3>Single image</h3><p>Resize to
          {_number(data_config.get("image_size"), " × " + str(data_config.get("image_size", "?")))},
          normalize with ImageNet statistics, compute global logits and routing
          probability, optionally run ROI fusion, then return ten probabilities,
          confidence, route decision, and saliency.</p></article>
          <article class="panel"><h3>Video frame aggregation</h3><p>The Streamlit app
          samples up to 32 evenly spaced frames, batches C-DIRA inference, and averages
          frame probabilities. It reports the dominant behavior, aggregate confidence,
          per-frame timeline, routing probability, and ROI usage.</p></article>
        </div>
        <div class="callout">For a production driver-monitoring camera, add continuous
        capture, latency measurement, temporal smoothing, alert hysteresis, privacy
        controls, and evaluation on real cabin video before safety use.</div>
      </section>

      <section class="section" id="results">
        <div class="kicker">08 · Measured outcome</div>
        <h2>Experimental results</h2>
        {_comparison_table(data)}
        <p class="lede">{_escape(_hardest_classes(data))}</p>
        <h3>Per-class performance</h3>
        {_per_class_results(data)}
        <h3>Confusion matrix</h3>
        {_confusion_matrix(data)}
        <h3>Paper-reported reference comparison</h3>
        {_paper_comparison(data)}
      </section>

      <section class="section" id="efficiency">
        <div class="kicker">09 · Deployment economics</div>
        <h2>Efficiency and deployment</h2>
        <div class="metric-grid">
          {_metric_card("Parameters", parameter_text, "Counted from checkpoint tensors")}
          {_metric_card("Checkpoint", checkpoint_mb, "Local cdira.pt size")}
          {_metric_card("ROI activation", _percent(data.full.get("roi_usage")), "Extra branch frequency", "amber")}
          {_metric_card("Latency", "Unmeasured", "No reproducible MPS benchmark", "red")}
        </div>
        <div class="good-callout callout">Repeated prediction reuses a cached model.
        The MobileNet backbone and global head always run; ROI pooling and fusion run
        only for routed samples. Low ROI usage therefore limits incremental work, but
        it does not remove the backbone cost.</div>
        <p class="lede">Training is heavier than inference: feature extraction,
        clustering, C-DIRA optimization, and baseline optimization run sequentially.
        The current trainer configuration includes a mixed-precision flag, but the
        implemented engine does not wrap computation in autocast.</p>
      </section>

      <section class="section" id="limitations">
        <div class="kicker">10 · Research quality</div>
        <h2>Reproduction assumptions and limitations</h2>
        <div class="panel"><ul>
          <li>The standard profile uses at most 10 epochs; the paper profile specifies 50.</li>
          <li>The primary split is image-level and can include the same subject across splits.</li>
          <li>The paper's routing description is contradictory; this implementation reports both policies.</li>
          <li>Top-K is set to {_number(model.get("top_k", 5))} because the paper does not specify it.</li>
          <li>The local MobileNet baseline outperformed C-DIRA in macro F1.</li>
          <li>Video output averages static-frame probabilities and does not model motion.</li>
          <li>H100 latency and Apple Silicon real-time FPS were not reproduced.</li>
          <li>The State Farm competition distribution is not a substitute for production cabin-camera validation.</li>
        </ul></div>
      </section>

      <section class="section" id="engineering">
        <div class="kicker">11 · Implementation map</div>
        <h2>Engineering map and reproduction</h2>
        <div class="panel">{_source_map()}</div>
        <h3>Core commands</h3>
        <pre><code>uv sync --all-extras
uv run cdira data prepare --config configs/standard.yaml
uv run cdira run-paper --config configs/standard.yaml --artifact-root artifacts/standard
uv run cdira report-html --run artifacts/standard/run
uv run streamlit run app.py</code></pre>
        <div class="two-column">
          <article class="panel"><h3>Resolved environment</h3>
            <ul class="fact-list">
              <li><strong>Python</strong><span>{_escape(environment.get("python", "unknown"))}</span></li>
              <li><strong>PyTorch</strong><span>{_escape(environment.get("torch", "unknown"))}</span></li>
              <li><strong>Device</strong><span>{_escape(environment.get("device_requested", "unknown"))}</span></li>
              <li><strong>Batch size</strong><span>{_number(training.get("batch_size"))}</span></li>
            </ul>
          </article>
          <article class="panel"><h3>Run identity</h3>
            <ul class="fact-list">
              <li><strong>Profile</strong><span>{_escape(data.profile)}</span></li>
              <li><strong>Epochs completed</strong><span>{_number(data.full.get("epochs_completed"))}</span></li>
              <li><strong>Image size</strong><span>{_number(data_config.get("image_size"))}</span></li>
              <li><strong>Classes</strong><span>{_number(model.get("num_classes", 10))}</span></li>
            </ul>
          </article>
        </div>
      </section>
    </main>
  </div>
  <footer>C-DIRA reproduction · self-contained offline report · generated {_escape(generated)}</footer>
</body>
</html>
"""


def build_html_reproduction_report(run_root: Path) -> Path:
    data = load_report_data(run_root)
    destination = run_root / "report.html"
    destination.write_text(render_html_report(data), encoding="utf-8")
    return destination
