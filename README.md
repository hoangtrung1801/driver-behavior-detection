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

Configure Kaggle credentials outside this repository and accept the State Farm
competition rules before downloading. The paper profile runs the full data,
ablation, corruption, efficiency, and LOCO suites; `configs/smoke.yaml` is
offline and intended for validation of the pipeline.

The primary split intentionally follows the paper's image-level stratified
protocol. When the competition subject CSV is available, a subject-overlap
audit is written. The paper's routing description is contradictory, so both
routing-head and global-confidence policies are reported.
