# C-DIRA Colab Walkthrough Notebook Design

## Goal

Provide a committed Google Colab notebook that runs the full C-DIRA pipeline
one stage at a time and visualizes each part — preprocessing, data
distribution, pseudo-domain discovery, model architecture, live training, and
evaluation — so a reader can see what happens at every step rather than only
the final report.

This is a sibling to the one-run design
(`2026-07-28-colab-one-run-design.md`), not a replacement. The one-run script
is a thin end-to-end orchestrator that prints report links; this notebook is a
balanced, cell-by-cell walkthrough that renders a figure or table at each
stage.

## Scope

The notebook will:

1. verify repository root, a supported Python version, and a usable CUDA GPU;
2. install the editable project and request `kaggle.json` when valid
   credentials are not already present;
3. download and validate the State Farm Distracted Driver Detection dataset;
4. visualize preprocessing: class distribution, split sizes, a sample grid, and
   an augmentation before/after comparison;
5. fit pseudo-domains and visualize silhouette-vs-k, a 2-D PCA feature scatter
   colored by cluster, and per-domain sample grids;
6. instantiate the C-DIRA model and visualize its architecture schematic and a
   per-module parameter-count table;
7. train C-DIRA (with a live-updating loss curve) and the MobileNetV3 baseline;
8. evaluate: confusion matrix, per-class F1, routing usage, C-DIRA-vs-baseline
   comparison, and ROI saliency overlays; and
9. generate the existing Markdown and self-contained HTML reports and print
   artifact paths and Colab download links.

It will not embed credentials, change the model architecture, add temporal
video modeling, mount Google Drive, introduce t-SNE or other heavy new
dependencies, or bypass Kaggle competition rules. Full real GPU training is
required; there is no fast/subsampled mode.

## Architecture

The repository convention is that logic lives in tested `src/cdira/` modules
and notebooks/CLIs are thin wrappers. This design follows that convention.

**New files**

- `docs/colab/cdira_walkthrough.ipynb` — the notebook. Each stage is a markdown
  cell (a 2–4 sentence intro plus how to read the figure) followed by thin code
  cells. The notebook orchestrates the pipeline stage by stage — rather than
  calling the monolithic `run_core_pipeline` — so each stage's in-memory
  objects are available to visualize. The stage *logic* is unchanged; only its
  orchestration is unrolled into cells.
- `src/cdira/reporting/figures.py` — pure figure/table builders. Every function
  takes already-computed data and returns a `matplotlib.figure.Figure` or a
  `pandas.DataFrame`; none call `plt.show()`. Returning figures is what makes
  them unit-testable without a display.
- `src/cdira/colab.py` — environment and setup helpers: repository-root check,
  Python-version check, CUDA assertion, and Kaggle credential upload/install.
  `google.colab` is imported lazily inside the functions that need it so the
  module imports cleanly off Colab (and under tests).
- `configs/colab.yaml` — a reviewable derivative of `configs/standard.yaml`
  selecting `training.device: cuda`, `training.mixed_precision: true`,
  `data.num_workers: 2`, artifacts rooted at `artifacts/colab`, and otherwise
  the standard ten-epoch experiment settings.
- Tests: `tests/test_figures.py`, `tests/test_colab.py`, and additions to
  `tests/test_training_engine.py`.

**One library change**

- `Trainer.fit` gains an optional keyword `on_epoch_end`. After each epoch's
  record is appended to `history`, if the callback is set it is invoked with the
  running history (and the epoch index and `max_epochs`). It is purely
  observational: it must not affect optimization, metrics, early stopping, or
  checkpoints. The default `None` preserves today's behavior and the public API,
  and `run_core_pipeline` continues to call `fit` with no callback.

**Layering:** notebook → `cdira.colab` / `cdira.reporting.figures` / existing
`cdira.pipeline`, `cdira.training`, and `cdira.evaluation`. No new heavy runtime
dependencies; matplotlib, pandas, and scikit-learn are already present. The
domain feature scatter uses `sklearn.decomposition.PCA` (already available via
scikit-learn) to project to two dimensions — deliberately not t-SNE, to avoid
runtime cost and a new dependency.

## Notebook Stage Flow

Each stage is a markdown intro followed by thin code cells. Stages are numbered
and note elapsed time, matching the one-run design's presentation.

**0. Setup & environment.** `cdira.colab` verifies repository root, Python
version, and a usable CUDA GPU (hard stop if absent); installs the editable
project; uploads and installs `kaggle.json` at `/root/.kaggle/kaggle.json` with
mode `0600`, never printing its contents. Loads `configs/colab.yaml`.

**1. Data acquisition.** Downloads and extracts the State Farm dataset via the
existing `cdira` data path and validates the `c0`–`c9` layout. An existing valid
dataset is reused. Kaggle 401/403 responses are translated into a concise
message covering invalid/missing credentials, the need to accept the
competition rules, and rerunning after fixing access; no retry loop reissues a
forbidden request.

**2. Preprocessing & data visualization.**
- Class-distribution bar chart (c0–c9 counts) and a train/validation/test split
  size table.
- A sample grid of random raw images per class with labels.
- An augmentation before/after comparison: the same image through
  `build_transform(train=True, …)` versus the eval transform, so resize, flip,
  and brightness are visible.

**3. Pseudo-domains.** Runs `prepare_domains`, then visualizes:
- A silhouette-score-versus-k bar chart indicating the selected `k`.
- A 2-D PCA scatter of train features colored by cluster id.
- A per-domain representative sample grid and a domain-size table.

**4. Model architecture.** Instantiates `CDIRA`, then visualizes:
- A labeled matplotlib schematic of the branches: backbone → global head; ROI
  top-k pool → refinement → fused head; routing head; and GRL → domain
  classifier.
- A per-module parameter-count table (backbone versus each head) with the total.

**5. Training (live).** Builds the loaders and trains C-DIRA with an
`on_epoch_end` callback that redraws a train/validation loss curve in the output
cell each epoch; the existing batch and epoch text still streams. Then trains
the MobileNetV3 baseline. Checkpoints are written under a fresh timestamped run
below `artifacts/colab`.

**6. Evaluation & interpretability.**
- Confusion-matrix heatmap and per-class F1 bar chart.
- Routing usage (fraction of test images routed to the ROI/fused path) and a
  C-DIRA-versus-baseline macro-F1 comparison.
- ROI saliency overlays for a few test images, reusing
  `evaluation/visualize.py`.

**7. Report & artifacts.** Generates the existing Markdown and self-contained
HTML reports and prints the run directory, checkpoint path, report paths, and
Colab file-download links when `google.colab.files` is available.

## Rerun Behavior

The notebook is restart-friendly in the same sense as the one-run design:
dependency installation is safe to repeat; existing credentials are reused
without printing their contents; a valid existing dataset is validated and
reused; existing manifests and feature caches are reused by the current
pipeline where supported; and training writes to a fresh timestamped run below
`artifacts/colab` rather than overwriting checkpoints. It is restart-friendly
rather than a training-resume implementation: an interrupted training stage
starts a new run.

## Error Handling

The notebook stops before expensive work when any of the following hold, and it
never prints credential values:

- the repository root or a required file cannot be found (stop with the path);
- Python is outside the supported range;
- CUDA is unavailable (stop with a clear "requires a GPU runtime" message);
- the uploaded credential file is missing or malformed — not exactly one file
  named `kaggle.json`;
- Kaggle denies access (the concise credentials/accept-rules/rerun message, no
  retry loop); or
- the extracted dataset fails the existing `c0`–`c9` layout validator.

Figure builders validate their inputs — for example an empty prediction table
or mismatched array lengths — and raise clear errors rather than rendering blank
or misleading plots.

## Testing

Unit tests run without a real GPU, dataset download, or display, using the
matplotlib `Agg` backend and fakes for `google.colab`, CUDA detection, and
filesystem side effects.

- `tests/test_figures.py`: each builder returns a `Figure` with the expected
  number of axes, bars, or heatmap shape given tiny synthetic inputs (a fake
  `PseudoDomains`, a small `PredictionTable`, a short training history, and a
  couple of PIL images), and carries the labels that convey meaning — for
  example the predicted class in the ROI overlay title.
- `tests/test_colab.py`: repository-root validation; credential installation
  writes mode `0600` and does not log contents; reuse of existing valid
  credentials; the CUDA-absent hard stop; and the actionable Kaggle-access
  error, with `google.colab` faked.
- `tests/test_training_engine.py`: `on_epoch_end` fires once per epoch with a
  monotonically growing history; absence of the callback leaves behavior
  identical; and the early-stop path is still respected.

The notebook itself is not unit-tested because its cells are thin; the modules
it calls are tested. After implementation, the full pytest suite, Ruff, mypy,
and `git diff --check` will be run. A real Kaggle download and full GPU training
are not part of automated local verification, matching the one-run design's
stance.
