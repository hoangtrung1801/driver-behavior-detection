# C-DIRA Research Reproduction

This repository implements the static-image C-DIRA driver behavior recognition
paper on Apple Silicon using PyTorch and MPS.

```bash
uv python install 3.12
uv sync --all-extras
uv run cdira data download --config configs/paper.yaml
uv run cdira data prepare --config configs/paper.yaml
uv run cdira run-paper --config configs/paper.yaml
```

## Interactive demo

Install the optional Streamlit dependency and launch the image tester:

```bash
uv sync --extra app
uv run streamlit run app.py
```

Upload a driver image or video in the browser to see the predicted class,
confidence, class probabilities, routing decision, and ROI saliency overlay.
Videos are sampled at up to 32 evenly spaced frames and aggregated into one
prediction. The default checkpoint is
`artifacts/standard/run/checkpoints/cdira.pt`; it can be changed in the sidebar.

## Colab walkthrough

`docs/colab/cdira_walkthrough.ipynb` runs the whole pipeline one stage at a
time on a Colab GPU runtime and visualizes preprocessing, pseudo-domains, the
model architecture, live training curves, and evaluation. It uses
`configs/colab.yaml` and requires a Kaggle token for an account that has
accepted the competition rules.

## Detailed HTML report

Generate the self-contained technical report from a completed run:

```bash
uv run cdira report-html --run artifacts/standard/run
open artifacts/standard/run/report.html
```

The document works offline and includes the architecture, data and training
pipeline, image and video inference, measured multi-metric results, per-class
analysis, confusion matrix, efficiency discussion, limitations, and
reproduction commands. It also includes responsive and print layouts.

Configure Kaggle credentials outside this repository and accept the State Farm
competition rules before downloading. The paper profile runs the full data,
ablation, corruption, efficiency, and LOCO suites; `configs/smoke.yaml` is
offline and intended for validation of the pipeline.

The primary split intentionally follows the paper's image-level stratified
protocol. When the competition subject CSV is available, a subject-overlap
audit is written. The paper's routing description is contradictory, so both
routing-head and global-confidence policies are reported.
