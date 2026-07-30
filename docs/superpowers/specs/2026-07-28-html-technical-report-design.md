# C-DIRA Self-Contained HTML Report Design

## Goal

Generate a detailed, portable HTML report for a completed C-DIRA reproduction
run. The report must serve both research reviewers and software engineers,
combining concise conclusions with enough architectural and implementation
detail to audit the solution.

## Deliverable

The generator will create:

```text
artifacts/standard/run/report.html
```

The HTML file will embed its CSS and visualizations, require no external CDN,
and remain usable offline. It will support desktop, tablet, mobile, and print
layouts.

## Generation interface

Add a reusable function:

```python
build_html_reproduction_report(run_root: Path) -> Path
```

Expose it through:

```bash
cdira report-html --run artifacts/standard/run
```

The generator will read:

- `environment.json`
- `config.resolved.yaml`
- `metrics/full.json`
- `metrics/baseline.json`
- `data/manifests/manifest_metadata.json`, when available
- `data/manifests/subject_overlap.json`, when available

Missing optional artifacts will render as clearly marked unavailable values.
Malformed required metric files will raise a descriptive error rather than
silently producing misleading output.

## Information architecture

### 1. Cover and executive summary

- Project title and reproduction profile
- Device, framework, checkpoint, and dataset summary
- C-DIRA and baseline macro F1
- ROI usage, selected pseudo-domains, and completed epochs
- A concise conclusion explaining whether the local run reproduced the paper
  result and whether C-DIRA improved over the local baseline

### 2. Driver-behavior task

- State Farm distracted-driver dataset context
- Human-readable definitions for classes `c0` through `c9`
- Static-image classification scope
- Distinction between image classification and true temporal action recognition

### 3. End-to-end solution pipeline

A horizontal architecture diagram will show:

```text
Raw images
  → validation and manifests
  → image-level train/validation/test split
  → MobileNet feature cache
  → pseudo-domain clustering
  → C-DIRA training
  → evaluation
  → image/video inference app
```

Each stage will include its input, output, and implementation module.

### 4. C-DIRA architecture

The report will explain and visualize:

- MobileNetV3-Small feature backbone
- Global classification branch
- Routing head
- Top-K saliency-based ROI pooling
- ROI refinement and fused classifier
- Domain classifier with gradient reversal
- Conditional inference path

The diagram will distinguish training-only components from inference
components and identify tensor-level data flow where useful.

### 5. Training objective

Document the implemented weighted objective:

- Global cross-entropy
- Fused ROI cross-entropy
- Routing binary cross-entropy
- Routing-usage regularizer
- Domain classification cross-entropy through gradient reversal

The report will present the combined loss in readable mathematical notation
and list the exact weights from the resolved configuration.

### 6. Pseudo-domain pipeline

- Extract frozen ImageNet MobileNet features
- Search candidate cluster counts
- Select the cluster count using validation silhouette score
- Assign fixed pseudo-domain labels
- Use gradient reversal to encourage domain-invariant features

The local selected-domain count will be displayed from the measured run.

### 7. Inference pipelines

#### Image inference

Show preprocessing, global prediction, routing decision, optional ROI
refinement, final probabilities, and saliency output.

#### Video inference

Explain that the current solution:

- samples up to 32 evenly spaced frames;
- predicts each frame independently;
- averages frame probabilities into a video-level prediction;
- displays frame predictions, confidence, routing probability, and ROI usage.

The report will explicitly state that this is frame aggregation, not a temporal
neural network.

### 8. Experimental results

Include:

- C-DIRA versus MobileNet comparison table
- Accuracy, macro precision, macro recall, macro F1, weighted F1, ROI usage,
  and completed epochs
- Per-class precision, recall, F1, and support
- Accessible horizontal per-class F1 chart
- Confusion-matrix heatmap with readable counts
- Measured-versus-paper comparison
- Plain-language interpretation of the hardest classes and largest confusion
  pairs

All measured values will come from the run artifacts.

### 9. Efficiency and deployment

- Actual parameter count from the checkpoint architecture
- Conditional ROI usage
- Checkpoint file size
- Apple Silicon MPS deployment path
- Difference between model-load time and repeated inference
- Lack of a completed reproducible latency benchmark
- Training-cost caveats
- Current lack of live-camera streaming

The report will not claim real-time performance without measured latency.

### 10. Reproduction quality and limitations

- Standard profile uses at most 10 epochs rather than the paper profile's 50
- Image-level split and subject-overlap implications
- Routing-description ambiguity in the paper
- Assumed Top-K value of five
- Baseline outperforming C-DIRA in the measured standard run
- Frame aggregation limitations for video
- No H100 latency reproduction

### 11. Engineering map and reproduction commands

- Key source modules and their responsibilities
- Data preparation, training, report, and Streamlit commands
- Artifact directory structure
- Configuration values required to reproduce the displayed run

## Visual system

Use a restrained technical-editorial aesthetic:

- Deep navy background with slate surfaces
- Cyan for measured C-DIRA values
- Amber for paper references and caveats
- Green for successful states
- Red only for errors or negative deltas
- System font stack for offline portability
- Clear typographic hierarchy and generous spacing
- Sticky desktop contents rail and compact mobile navigation
- Semantic HTML landmarks, table headers, captions, and high-contrast text
- CSS-only charts and inline HTML/SVG diagrams
- Dedicated `@media print` styles

The report will avoid decorative animation, external fonts, and remote assets.

## Implementation boundaries

`src/cdira/reporting/html_report.py` will own HTML generation and formatting
helpers. Existing Markdown generation in `report.py` will remain intact.
`cli.py` will expose `report-html`. Tests will use temporary run artifacts and
assert the generated document contains real nested metrics, architecture
sections, class labels, confusion data, inline styles, and no remote
dependencies.

## Error handling

- Missing run directory: raise `FileNotFoundError`
- Missing required metrics: render an explicit unavailable state
- Invalid JSON/YAML: raise a descriptive `ValueError` containing the path
- Unexpected metric shape: omit the affected visualization and render an
  explanatory fallback
- HTML-escape all values loaded from artifacts

## Verification

- Unit tests for percentage, delta, table, and heatmap generation
- Integration test using synthetic artifact files
- Generation against `artifacts/standard/run`
- HTML structure check for required sections and offline-only assets
- Existing pytest suite
- Ruff
- mypy
- Browser inspection at desktop and mobile viewport widths

## Acceptance criteria

- `artifacts/standard/run/report.html` opens directly in a browser
- It includes architecture, pipeline, training, inference, results,
  efficiency, limitations, and reproduction detail
- It displays the real standard-run metrics
- It remains readable without network access
- It is usable by both research and engineering audiences
- It does not make unsupported real-time or reproduction claims
