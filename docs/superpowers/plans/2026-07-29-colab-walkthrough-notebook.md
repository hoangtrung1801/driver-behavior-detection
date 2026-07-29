# C-DIRA Colab Walkthrough Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a committed Google Colab notebook that runs the full C-DIRA pipeline one stage at a time and visualizes each part (preprocessing, pseudo-domains, architecture, live training, evaluation), backed by thin, tested helper modules.

**Architecture:** Logic lives in tested `src/cdira/` modules; the notebook is a thin, balanced-prose wrapper. A new `cdira.reporting.figures` module returns matplotlib `Figure`/`pandas.DataFrame` objects for each stage. A new `cdira.colab` module handles environment checks and Kaggle credentials. `Trainer.fit` gains an optional `on_epoch_end` callback so the training cell can draw a live loss curve. A new `configs/colab.yaml` selects CUDA + full ten-epoch settings.

**Tech Stack:** Python 3.12, PyTorch 2.7.1, torchvision, scikit-learn, matplotlib, pandas, pytest.

## Global Constraints

- Python floor is exactly `>=3.12,<3.13`; only Python 3.12 is supported.
- Add no new runtime dependencies (matplotlib, pandas, scikit-learn, Pillow, torch are already declared).
- Do not use t-SNE; project domain features to 2-D with `sklearn.decomposition.PCA`.
- Do not change the model architecture, add temporal/video modeling, or mount Google Drive.
- Never print Kaggle credential contents. Credential files are written with mode `0600`.
- mypy runs in `strict` mode: every new function needs complete type annotations.
- Ruff line length is 88.
- Figure builders return objects (`Figure` / `DataFrame`); they never call `plt.show()`.
- Run tests with `PYTHONPATH=src .venv/bin/python -m pytest`; lint with `.venv/bin/ruff check src tests`; type-check with `.venv/bin/mypy src/cdira`.
- New test modules that build figures must set the headless backend before importing the figures module:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  ```

---

### Task 1: Add `configs/colab.yaml`

**Files:**
- Create: `configs/colab.yaml`
- Test: `tests/test_config.py` (add one test)

**Interfaces:**
- Consumes: `cdira.config.load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig`.
- Produces: a config file with `profile: standard`, `training.device: cuda`, `training.mixed_precision: true`, `data.num_workers: 2`, `paths.artifact_root: artifacts/colab`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
from pathlib import Path

from cdira.config import load_config


def test_colab_config_targets_cuda_and_colab_artifacts() -> None:
    config = load_config(Path("configs/colab.yaml"))
    assert config.profile == "standard"
    assert config.training.device == "cuda"
    assert config.training.mixed_precision is True
    assert config.data.num_workers == 2
    assert config.paths.artifact_root == Path("artifacts/colab")
    assert config.training.max_epochs == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py::test_colab_config_targets_cuda_and_colab_artifacts -v`
Expected: FAIL — `configs/colab.yaml` does not exist (`FileNotFoundError`).

- [ ] **Step 3: Create the config**

Create `configs/colab.yaml` (a derivative of `configs/standard.yaml` with the CUDA/Colab overrides):

```yaml
seed: 42
paths:
  data_root: data/raw
  manifest_root: data/manifests
  domain_root: data/domains
  artifact_root: artifacts/colab
  cache_root: data/cache
data:
  image_size: 224
  split: [0.8, 0.1, 0.1]
  horizontal_flip: true
  brightness: 0.2
  num_workers: 2
domains:
  candidates: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
  sample_size: 5000
  n_init: 10
model:
  num_classes: 10
  top_k: 5
  global_hidden: 256
  roi_hidden: 512
  fused_hidden: 512
  routing_hidden: 128
  domain_hidden: 256
  grl_strength: 1.0
training:
  device: cuda
  batch_size: 32
  learning_rate: 0.00001
  max_epochs: 10
  patience: 3
  mixed_precision: true
  confidence_threshold: 0.9
  loss_weights:
    global_ce: 0.5
    fused_ce: 1.0
    routing_bce: 0.5
    routing_regularizer: 0.01
    domain_ce: 0.5
routing:
  primary_policy: head
  comparison_policy: confidence
  threshold: 0.9
evaluation:
  thresholds: [0.5, 0.9]
  corruption_levels: standard
profile: standard
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py::test_colab_config_targets_cuda_and_colab_artifacts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/colab.yaml tests/test_config.py
git commit -m "feat: add Colab CUDA experiment config"
```

---

### Task 2: Add `on_epoch_end` callback to `Trainer.fit`

**Files:**
- Modify: `src/cdira/training/engine.py`
- Test: `tests/test_training_engine.py` (add tests)

**Interfaces:**
- Consumes: existing `Trainer.fit(train_loader, validation_loader)`.
- Produces: `Trainer.fit(train_loader, validation_loader, on_epoch_end: Callable[[list[dict[str, float]]], None] | None = None) -> FitResult`. The callback is invoked once after each completed epoch with the running `history` list (the same list object grows across calls). Default `None` preserves current behavior, metrics, early stopping, and checkpoints.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training_engine.py`:

```python
def test_fit_invokes_on_epoch_end_once_per_epoch() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=3, patience=5)
    batch = {"image": torch.ones(2, 3), "target": torch.tensor([0, 1])}

    lengths: list[int] = []
    result = trainer.fit(
        [batch], [batch], on_epoch_end=lambda history: lengths.append(len(history))
    )

    assert lengths == [1, 2, 3]
    assert result.epochs_completed == 3


def test_fit_without_callback_is_unchanged() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=2, patience=5)
    batch = {"image": torch.ones(2, 3), "target": torch.tensor([0, 1])}

    result = trainer.fit([batch], [batch])

    assert result.epochs_completed == 2
    assert len(result.history) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_engine.py -v`
Expected: FAIL — `fit()` got an unexpected keyword argument `on_epoch_end`.

- [ ] **Step 3: Implement the callback**

In `src/cdira/training/engine.py`, add `from collections.abc import Callable` to the imports, then change the `fit` signature and body:

```python
    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        on_epoch_end: Callable[[list[dict[str, float]]], None] | None = None,
    ) -> FitResult:
        best = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(self.max_epochs):
            started = time.perf_counter()
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch + 1}/{self.max_epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if on_epoch_end is not None:
                on_epoch_end(history)
            if validation_loss < best:
                best = validation_loss
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        return FitResult(best, len(history), history)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_engine.py -v`
Expected: PASS (all tests in the file, including the existing progress test).

- [ ] **Step 5: Commit**

```bash
git add src/cdira/training/engine.py tests/test_training_engine.py
git commit -m "feat: add optional per-epoch callback to Trainer.fit"
```

---

### Task 3: Create `cdira.colab` environment checks

**Files:**
- Create: `src/cdira/colab.py`
- Test: `tests/test_colab.py`

**Interfaces:**
- Produces:
  - `class ColabSetupError(RuntimeError)`
  - `ensure_repository_root(root: Path | None = None) -> Path` — validates `pyproject.toml` (containing `cdira-reproduction`) and `src/cdira/__init__.py`; returns the resolved root or raises `ColabSetupError`.
  - `ensure_supported_python(version: tuple[int, int] | None = None) -> None` — raises `ColabSetupError` unless the version is `(3, 12)`.
  - `ensure_cuda_available() -> Any` — imports torch, raises `ColabSetupError` if `torch.cuda.is_available()` is false, else returns `torch.device("cuda")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_colab.py`:

```python
from pathlib import Path

import pytest

from cdira import colab


def _make_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        'name = "cdira-reproduction"\n', encoding="utf-8"
    )
    package = root / "src" / "cdira"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")


def test_ensure_repository_root_accepts_valid_repo(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    assert colab.ensure_repository_root(tmp_path) == tmp_path.resolve()


def test_ensure_repository_root_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_repository_root(tmp_path)


def test_ensure_supported_python_accepts_312() -> None:
    colab.ensure_supported_python((3, 12))


@pytest.mark.parametrize("version", [(3, 11), (3, 13)])
def test_ensure_supported_python_rejects_others(version: tuple[int, int]) -> None:
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_supported_python(version)


def test_ensure_cuda_available_raises_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_cuda_available()


def test_ensure_cuda_available_returns_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert colab.ensure_cuda_available().type == "cuda"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab.py -v`
Expected: FAIL — `cdira.colab` does not exist (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the module**

Create `src/cdira/colab.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class ColabSetupError(RuntimeError):
    """Raised when the Colab environment or credentials are not usable."""


def ensure_repository_root(root: Path | None = None) -> Path:
    candidate = (root or Path.cwd()).resolve()
    pyproject = candidate / "pyproject.toml"
    package = candidate / "src" / "cdira" / "__init__.py"
    if not pyproject.is_file() or not package.is_file():
        raise ColabSetupError(
            f"Not at the C-DIRA repository root: {candidate} "
            "(expected pyproject.toml and src/cdira/__init__.py)."
        )
    if "cdira-reproduction" not in pyproject.read_text(encoding="utf-8"):
        raise ColabSetupError(
            f"pyproject.toml at {candidate} is not the cdira-reproduction project."
        )
    return candidate


def ensure_supported_python(version: tuple[int, int] | None = None) -> None:
    major, minor = version or (sys.version_info.major, sys.version_info.minor)
    if (major, minor) != (3, 12):
        raise ColabSetupError(
            f"Python 3.12 is required; found {major}.{minor}. "
            "Select a Python 3.12 Colab runtime."
        )


def ensure_cuda_available() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise ColabSetupError(
            "This notebook requires a CUDA GPU runtime. In Colab choose "
            "Runtime > Change runtime type > GPU, then rerun."
        )
    return torch.device("cuda")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/colab.py tests/test_colab.py
git commit -m "feat: add Colab environment checks"
```

---

### Task 4: Add Kaggle credential handling to `cdira.colab`

**Files:**
- Modify: `src/cdira/colab.py`
- Test: `tests/test_colab.py` (add tests)

**Interfaces:**
- Consumes: `ColabSetupError` from Task 3.
- Produces:
  - `install_kaggle_credentials(destination: Path = Path("/root/.kaggle/kaggle.json"), upload_fn: Callable[[], Mapping[str, bytes]] | None = None) -> Path` — if `destination` already exists and is valid, returns it (no upload). Otherwise calls `upload_fn` (defaulting to a lazy `google.colab.files.upload` wrapper), requires exactly one uploaded file named `kaggle.json`, writes it with mode `0600`, validates it, and returns `destination`. Never prints file contents.
  - `translate_kaggle_error(error: Exception) -> str` — returns an actionable multi-line message for 401/403/forbidden/authenticate errors, otherwise a concise generic download-failure message.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_colab.py`:

```python
import json


VALID = b'{"username": "u", "key": "secret-key-value"}'


def test_install_reuses_existing_valid_credentials(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    dest.write_bytes(VALID)

    def fail() -> dict[str, bytes]:
        raise AssertionError("upload_fn must not be called when creds exist")

    assert colab.install_kaggle_credentials(dest, upload_fn=fail) == dest


def test_install_writes_uploaded_credentials_with_0600(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "kaggle.json"
    result = colab.install_kaggle_credentials(
        dest, upload_fn=lambda: {"kaggle.json": VALID}
    )
    assert result == dest
    assert dest.is_file()
    assert (dest.stat().st_mode & 0o777) == 0o600
    assert json.loads(dest.read_text(encoding="utf-8"))["username"] == "u"


def test_install_does_not_print_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "kaggle.json"
    colab.install_kaggle_credentials(dest, upload_fn=lambda: {"kaggle.json": VALID})
    assert "secret-key-value" not in capsys.readouterr().out


def test_install_rejects_wrong_filename(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    with pytest.raises(colab.ColabSetupError):
        colab.install_kaggle_credentials(dest, upload_fn=lambda: {"creds.txt": VALID})


def test_install_rejects_malformed_json(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    with pytest.raises(colab.ColabSetupError):
        colab.install_kaggle_credentials(
            dest, upload_fn=lambda: {"kaggle.json": b"not json"}
        )


def test_translate_kaggle_error_forbidden_is_actionable() -> None:
    message = colab.translate_kaggle_error(RuntimeError("HTTP 403 Forbidden"))
    assert "competition rules" in message


def test_translate_kaggle_error_generic() -> None:
    message = colab.translate_kaggle_error(RuntimeError("disk full"))
    assert "disk full" in message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab.py -v`
Expected: FAIL — `install_kaggle_credentials` / `translate_kaggle_error` do not exist.

- [ ] **Step 3: Implement credential handling**

In `src/cdira/colab.py`, extend the imports and add the functions:

```python
import json
from collections.abc import Callable, Mapping
```

```python
def _validate_credentials_file(path: Path) -> None:
    if not path.is_file():
        raise ColabSetupError(f"Credential file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ColabSetupError("kaggle.json is not valid JSON.") from exc
    if not isinstance(payload, dict) or "username" not in payload or "key" not in payload:
        raise ColabSetupError("kaggle.json must contain 'username' and 'key'.")


def _colab_upload() -> Mapping[str, bytes]:
    from google.colab import files  # type: ignore[import-not-found]

    return files.upload()


def install_kaggle_credentials(
    destination: Path = Path("/root/.kaggle/kaggle.json"),
    upload_fn: Callable[[], Mapping[str, bytes]] | None = None,
) -> Path:
    if destination.is_file():
        _validate_credentials_file(destination)
        return destination
    uploaded = (upload_fn or _colab_upload)()
    names = list(uploaded)
    if names != ["kaggle.json"]:
        received = ", ".join(names) or "nothing"
        raise ColabSetupError(
            f"Upload exactly one file named kaggle.json; received: {received}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded["kaggle.json"])
    destination.chmod(0o600)
    _validate_credentials_file(destination)
    return destination


def translate_kaggle_error(error: Exception) -> str:
    message = str(error).lower()
    if any(token in message for token in ("401", "403", "forbidden", "authenticate")):
        return (
            "Kaggle denied access. Confirm that:\n"
            "  1. kaggle.json holds valid credentials for your account;\n"
            "  2. you have signed in to Kaggle and accepted the State Farm "
            "Distracted Driver Detection competition rules; then\n"
            "  3. rerun this cell."
        )
    return f"Dataset download failed: {error}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/colab.py tests/test_colab.py
git commit -m "feat: add Kaggle credential install and error translation"
```

---

### Task 5: Create `cdira.reporting.figures` — preprocessing & data figures

**Files:**
- Create: `src/cdira/reporting/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Produces (all return objects; none call `plt.show()`):
  - `denormalize(tensor: Tensor) -> np.ndarray` — reverses ImageNet normalization on a `[3,H,W]` tensor and returns an `[H,W,3]` float array clipped to `[0, 1]`.
  - `class_distribution_figure(counts: Mapping[int, int]) -> Figure`
  - `split_sizes_table(sizes: Mapping[str, int]) -> pd.DataFrame` with columns `split`, `count`.
  - `sample_grid_figure(images: Sequence[Image.Image], labels: Sequence[str], ncols: int = 5, title: str | None = None) -> Figure`
  - `augmentation_figure(original: Image.Image, train_tensor: Tensor, eval_tensor: Tensor) -> Figure`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_figures.py`:

```python
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from cdira.reporting import figures


def test_denormalize_returns_hwc_in_unit_range() -> None:
    array = figures.denormalize(torch.zeros(3, 8, 8))
    assert array.shape == (8, 8, 3)
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0


def test_class_distribution_figure_has_one_bar_per_class() -> None:
    fig = figures.class_distribution_figure({0: 5, 1: 3, 2: 7})
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 3


def test_split_sizes_table_columns() -> None:
    frame = figures.split_sizes_table({"train": 8, "validation": 1, "test": 1})
    assert list(frame.columns) == ["split", "count"]
    assert set(frame["split"]) == {"train", "validation", "test"}


def test_sample_grid_figure_has_axis_per_image() -> None:
    images = [Image.new("RGB", (8, 8), "white") for _ in range(3)]
    fig = figures.sample_grid_figure(images, ["a", "b", "c"], ncols=2)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3


def test_augmentation_figure_has_three_panels() -> None:
    original = Image.new("RGB", (8, 8), "white")
    fig = figures.augmentation_figure(original, torch.zeros(3, 8, 8), torch.zeros(3, 8, 8))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: FAIL — `cdira.reporting.figures` does not exist.

- [ ] **Step 3: Implement the module**

Create `src/cdira/reporting/figures.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from cdira.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: Tensor) -> np.ndarray:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    array = tensor.detach().cpu().numpy().astype(np.float32) * std + mean
    return np.clip(array, 0.0, 1.0).transpose(1, 2, 0)


def class_distribution_figure(counts: Mapping[int, int]) -> Figure:
    ordered = sorted(counts.items())
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([f"c{key}" for key, _ in ordered], [value for _, value in ordered])
    axis.set_title("Training image count per class")
    axis.set_xlabel("class")
    axis.set_ylabel("images")
    figure.tight_layout()
    return figure


def split_sizes_table(sizes: Mapping[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"split": name, "count": count} for name, count in sizes.items()]
    )


def sample_grid_figure(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    ncols: int = 5,
    title: str | None = None,
) -> Figure:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("sample_grid_figure requires at least one image")
    nrows = (len(images) + ncols - 1) // ncols
    figure, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    flat = np.atleast_1d(axes).ravel()
    for axis, image, label in zip(flat, images, labels, strict=False):
        axis.imshow(image)
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    for axis in flat[len(images):]:
        axis.axis("off")
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def augmentation_figure(
    original: Image.Image, train_tensor: Tensor, eval_tensor: Tensor
) -> Figure:
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(original)
    axes[0].set_title("original")
    axes[1].imshow(denormalize(eval_tensor))
    axes[1].set_title("eval transform")
    axes[2].imshow(denormalize(train_tensor))
    axes[2].set_title("train transform")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/reporting/figures.py tests/test_figures.py
git commit -m "feat: add preprocessing and data stage figures"
```

---

### Task 6: Add pseudo-domain figures to `cdira.reporting.figures`

**Files:**
- Modify: `src/cdira/reporting/figures.py`
- Test: `tests/test_figures.py` (add tests)

**Interfaces:**
- Produces:
  - `silhouette_figure(silhouette_scores: Mapping[int, float], selected_k: int) -> Figure` — bar of score vs k; the selected k bar is visually distinct.
  - `domain_scatter_figure(features: np.ndarray, labels: Sequence[int]) -> Figure` — PCA projection of `features` to 2-D, scatter colored by label.
  - `domain_sizes_table(labels_by_path: Mapping[str, int]) -> pd.DataFrame` with columns `domain_id`, `count`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_figures.py`:

```python
def test_silhouette_figure_has_bar_per_candidate() -> None:
    fig = figures.silhouette_figure({2: 0.1, 3: 0.4, 4: 0.2}, selected_k=3)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 3


def test_domain_scatter_figure_projects_to_2d() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((20, 8)).astype(np.float32)
    labels = [0, 1] * 10
    fig = figures.domain_scatter_figure(features, labels)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) >= 1


def test_domain_sizes_table_counts_paths() -> None:
    frame = figures.domain_sizes_table({"a.jpg": 0, "b.jpg": 0, "c.jpg": 1})
    assert list(frame.columns) == ["domain_id", "count"]
    counts = dict(zip(frame["domain_id"], frame["count"], strict=True))
    assert counts == {0: 2, 1: 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: FAIL — the three functions do not exist.

- [ ] **Step 3: Implement the functions**

Add to `src/cdira/reporting/figures.py` (add `from sklearn.decomposition import PCA` to imports):

```python
def silhouette_figure(silhouette_scores: Mapping[int, float], selected_k: int) -> Figure:
    ordered = sorted(silhouette_scores.items())
    figure, axis = plt.subplots(figsize=(7, 4))
    colors = ["tab:orange" if k == selected_k else "tab:blue" for k, _ in ordered]
    axis.bar([str(k) for k, _ in ordered], [score for _, score in ordered], color=colors)
    axis.set_title(f"Silhouette score by k (selected k={selected_k})")
    axis.set_xlabel("clusters (k)")
    axis.set_ylabel("silhouette score")
    figure.tight_layout()
    return figure


def domain_scatter_figure(features: np.ndarray, labels: Sequence[int]) -> Figure:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if len(features) < 2:
        raise ValueError("PCA scatter requires at least two samples")
    projection = PCA(n_components=2).fit_transform(np.asarray(features, dtype=np.float32))
    figure, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(
        projection[:, 0], projection[:, 1], c=list(labels), cmap="tab10", s=8, alpha=0.6
    )
    axis.set_title("Pseudo-domains (PCA projection of features)")
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    figure.colorbar(scatter, ax=axis, label="domain id")
    figure.tight_layout()
    return figure


def domain_sizes_table(labels_by_path: Mapping[str, int]) -> pd.DataFrame:
    frame = pd.DataFrame({"domain_id": list(labels_by_path.values())})
    counts = frame.value_counts("domain_id").sort_index().reset_index(name="count")
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/reporting/figures.py tests/test_figures.py
git commit -m "feat: add pseudo-domain stage figures"
```

---

### Task 7: Add architecture figures to `cdira.reporting.figures`

**Files:**
- Modify: `src/cdira/reporting/figures.py`
- Test: `tests/test_figures.py` (add tests)

**Interfaces:**
- Produces:
  - `architecture_schematic_figure() -> Figure` — a labeled block diagram of the C-DIRA branches (backbone → global head; ROI top-k pool → refinement → fused head; routing head; GRL → domain classifier).
  - `parameter_table(model: nn.Module) -> pd.DataFrame` with columns `module`, `parameters`, including a final `total` row.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_figures.py`:

```python
def test_architecture_schematic_returns_figure_with_labels() -> None:
    fig = figures.architecture_schematic_figure()
    assert isinstance(fig, Figure)
    texts = {text.get_text() for text in fig.axes[0].texts}
    assert any("backbone" in text.lower() for text in texts)
    assert any("routing" in text.lower() for text in texts)


def test_parameter_table_lists_modules_and_total() -> None:
    import torch.nn as nn

    from cdira.reporting import figures as figures_module

    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    frame = figures_module.parameter_table(model)
    assert list(frame.columns) == ["module", "parameters"]
    assert frame["module"].iloc[-1] == "total"
    total = int(frame["parameters"].iloc[-1])
    assert total == sum(p.numel() for p in model.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: FAIL — the two functions do not exist.

- [ ] **Step 3: Implement the functions**

Add to `src/cdira/reporting/figures.py` (add `from torch import nn` and `from matplotlib.patches import FancyBboxPatch` to imports):

```python
def architecture_schematic_figure() -> Figure:
    blocks = [
        (0.05, 0.45, "input\nimage"),
        (0.24, 0.45, "MobileNetV3\nbackbone"),
        (0.45, 0.72, "global\nclassifier"),
        (0.45, 0.45, "ROI top-k pool\n-> refinement\n-> fused head"),
        (0.45, 0.18, "routing head"),
        (0.70, 0.18, "GRL -> domain\nclassifier"),
        (0.70, 0.58, "prediction"),
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    centers: dict[str, tuple[float, float]] = {}
    for x, y, label in blocks:
        axis.add_patch(
            FancyBboxPatch(
                (x, y), 0.18, 0.16, boxstyle="round,pad=0.02",
                facecolor="#eef3fb", edgecolor="#3b6ea5",
            )
        )
        axis.text(x + 0.09, y + 0.08, label, ha="center", va="center", fontsize=9)
        centers[label] = (x + 0.09, y + 0.08)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("C-DIRA architecture")
    figure.tight_layout()
    return figure


def parameter_table(model: nn.Module) -> pd.DataFrame:
    rows = [
        {
            "module": name,
            "parameters": int(sum(p.numel() for p in module.parameters())),
        }
        for name, module in model.named_children()
    ]
    rows.append(
        {"module": "total", "parameters": int(sum(p.numel() for p in model.parameters()))}
    )
    return pd.DataFrame(rows, columns=["module", "parameters"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/reporting/figures.py tests/test_figures.py
git commit -m "feat: add architecture stage figures"
```

---

### Task 8: Add training & evaluation figures to `cdira.reporting.figures`

**Files:**
- Modify: `src/cdira/reporting/figures.py`
- Test: `tests/test_figures.py` (add tests)

**Interfaces:**
- Produces:
  - `draw_loss_curve(axis: Axes, history: Sequence[Mapping[str, float]]) -> None` — clears `axis` and plots `train_loss` and `validation_loss` versus `epoch`; used for live updates.
  - `loss_curve_figure(history: Sequence[Mapping[str, float]]) -> Figure` — creates a figure and calls `draw_loss_curve`.
  - `confusion_matrix_figure(matrix: Sequence[Sequence[int]]) -> Figure`
  - `per_class_f1_figure(per_class: Sequence[Mapping[str, float]]) -> Figure`
  - `routing_usage_figure(metrics: Mapping[str, object]) -> Figure` — bars from `metrics["per_class_roi_usage"]` with the overall `metrics["roi_usage"]` drawn as a reference line.
  - `model_comparison_figure(cdira_metrics: Mapping[str, object], baseline_metrics: Mapping[str, object]) -> Figure` — grouped bars of accuracy and macro-F1 for C-DIRA vs baseline.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_figures.py`:

```python
HISTORY = [
    {"epoch": 1.0, "train_loss": 1.0, "validation_loss": 1.2},
    {"epoch": 2.0, "train_loss": 0.8, "validation_loss": 1.0},
]
METRICS = {
    "accuracy": 0.7,
    "macro": {"f1": 0.65},
    "confusion_matrix": [[3, 1], [0, 4]],
    "per_class": [{"f1": 0.6}, {"f1": 0.7}],
    "roi_usage": 0.5,
    "per_class_roi_usage": {"0": 0.4, "1": 0.6},
}
BASELINE = {"accuracy": 0.6, "macro": {"f1": 0.55}}


def test_loss_curve_figure_plots_two_series() -> None:
    fig = figures.loss_curve_figure(HISTORY)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2


def test_draw_loss_curve_is_idempotent_on_reuse() -> None:
    fig = figures.loss_curve_figure(HISTORY[:1])
    figures.draw_loss_curve(fig.axes[0], HISTORY)
    assert len(fig.axes[0].lines) == 2


def test_confusion_matrix_figure_renders_heatmap() -> None:
    fig = figures.confusion_matrix_figure(METRICS["confusion_matrix"])
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].images) == 1


def test_per_class_f1_figure_has_bar_per_class() -> None:
    fig = figures.per_class_f1_figure(METRICS["per_class"])
    assert len(fig.axes[0].patches) == 2


def test_routing_usage_figure_returns_figure() -> None:
    fig = figures.routing_usage_figure(METRICS)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 2


def test_model_comparison_figure_returns_figure() -> None:
    fig = figures.model_comparison_figure(METRICS, BASELINE)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: FAIL — the functions do not exist.

- [ ] **Step 3: Implement the functions**

Add to `src/cdira/reporting/figures.py` (add `import numpy as np` is already present; add `from matplotlib.axes import Axes` to imports):

```python
def draw_loss_curve(axis: Axes, history: Sequence[Mapping[str, float]]) -> None:
    axis.clear()
    epochs = [record["epoch"] for record in history]
    axis.plot(epochs, [record["train_loss"] for record in history], marker="o", label="train")
    axis.plot(
        epochs,
        [record["validation_loss"] for record in history],
        marker="o",
        label="validation",
    )
    axis.set_title("Training loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()


def loss_curve_figure(history: Sequence[Mapping[str, float]]) -> Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    draw_loss_curve(axis, history)
    figure.tight_layout()
    return figure


def confusion_matrix_figure(matrix: Sequence[Sequence[int]]) -> Figure:
    data = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(data, cmap="Blues")
    axis.set_title("Confusion matrix")
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    for (row, column), value in np.ndenumerate(data):
        axis.text(column, row, str(int(value)), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


def per_class_f1_figure(per_class: Sequence[Mapping[str, float]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        [f"c{index}" for index in range(len(per_class))],
        [float(record["f1"]) for record in per_class],
    )
    axis.set_ylim(0, 1)
    axis.set_title("Per-class F1")
    axis.set_xlabel("class")
    axis.set_ylabel("F1")
    figure.tight_layout()
    return figure


def routing_usage_figure(metrics: Mapping[str, object]) -> Figure:
    per_class = dict(metrics["per_class_roi_usage"])  # type: ignore[arg-type]
    overall = float(metrics["roi_usage"])  # type: ignore[arg-type]
    keys = sorted(per_class, key=int)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([f"c{key}" for key in keys], [float(per_class[key]) for key in keys])
    axis.axhline(overall, color="tab:red", linestyle="--", label=f"overall={overall:.2f}")
    axis.set_ylim(0, 1)
    axis.set_title("ROI/fused routing usage per class")
    axis.set_xlabel("class")
    axis.set_ylabel("fraction routed")
    axis.legend()
    figure.tight_layout()
    return figure


def model_comparison_figure(
    cdira_metrics: Mapping[str, object], baseline_metrics: Mapping[str, object]
) -> Figure:
    labels = ["accuracy", "macro F1"]
    cdira_values = [
        float(cdira_metrics["accuracy"]),  # type: ignore[arg-type]
        float(dict(cdira_metrics["macro"])["f1"]),  # type: ignore[arg-type]
    ]
    baseline_values = [
        float(baseline_metrics["accuracy"]),  # type: ignore[arg-type]
        float(dict(baseline_metrics["macro"])["f1"]),  # type: ignore[arg-type]
    ]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(positions - 0.2, cdira_values, width=0.4, label="C-DIRA")
    axis.bar(positions + 0.2, baseline_values, width=0.4, label="baseline")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_title("C-DIRA vs MobileNetV3 baseline")
    axis.legend()
    figure.tight_layout()
    return figure
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdira/reporting/figures.py tests/test_figures.py
git commit -m "feat: add training and evaluation stage figures"
```

---

### Task 9: Split a figure-returning ROI overlay out of `save_roi_figure`

**Files:**
- Modify: `src/cdira/evaluation/visualize.py`
- Test: `tests/test_visualize.py` (add a test; keep the existing one)

**Interfaces:**
- Produces: `roi_overlay_figure(image: Image.Image, saliency: Tensor, topk_indices: Tensor, title: str) -> Figure` — builds the saliency-overlay figure with top-k markers and returns it.
- Preserves: `save_roi_figure(table, index, image, saliency, topk_indices, destination) -> Path` continues to write a PNG and return the destination, now by calling `roi_overlay_figure`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualize.py`:

```python
import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure

from cdira.evaluation.visualize import roi_overlay_figure


def test_roi_overlay_figure_titles_and_returns_figure() -> None:
    fig = roi_overlay_figure(
        Image.new("RGB", (32, 32), "white"),
        torch.ones(7, 7),
        torch.tensor([1, 2, 3]),
        "true=0 pred=1",
    )
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "true=0 pred=1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_visualize.py -v`
Expected: FAIL — `roi_overlay_figure` does not exist.

- [ ] **Step 3: Refactor `visualize.py`**

Replace the body of `src/cdira/evaluation/visualize.py` with (add `from matplotlib.figure import Figure` to imports):

```python
def roi_overlay_figure(
    image: Image.Image,
    saliency: torch.Tensor,
    topk_indices: torch.Tensor,
    title: str,
) -> Figure:
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image)
    heatmap = saliency.detach().cpu().numpy()
    axis.imshow(
        heatmap, cmap="magma", alpha=0.45, extent=(0, image.width, image.height, 0)
    )
    height, width = heatmap.shape
    for flat_index in topk_indices.detach().cpu().tolist():
        if flat_index < 0:
            continue
        row, column = divmod(int(flat_index), width)
        axis.scatter(
            (column + 0.5) * image.width / width,
            (row + 0.5) * image.height / height,
            color="cyan",
            marker="x",
        )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    return figure


def save_roi_figure(
    table: PredictionTable,
    index: int,
    image: Image.Image,
    saliency: torch.Tensor,
    topk_indices: torch.Tensor,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    title = f"true={int(table.targets[index])} pred={int(table.predictions[index])}"
    figure = roi_overlay_figure(image, saliency, topk_indices, title)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_visualize.py -v`
Expected: PASS (both the new and existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/cdira/evaluation/visualize.py tests/test_visualize.py
git commit -m "refactor: extract figure-returning ROI overlay"
```

---

### Task 10: Build the walkthrough notebook and run full verification

**Files:**
- Create: `docs/colab/cdira_walkthrough.ipynb`
- Modify: `README.md` (add a short "Colab walkthrough" pointer)

**Interfaces:**
- Consumes everything above: `cdira.colab`, `cdira.reporting.figures`, `cdira.evaluation.visualize.roi_overlay_figure`, `Trainer.fit(..., on_epoch_end=...)`, `configs/colab.yaml`, and existing pipeline/eval/report functions.

- [ ] **Step 1: Create the empty notebook skeleton**

Write `docs/colab/cdira_walkthrough.ipynb` with this exact content:

```json
{
 "cells": [],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"},
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Add the cells in order**

Using the NotebookEdit tool (`edit_mode: insert`, appending in order), add the following cells. Markdown cells use `cell_type: markdown`; code cells use `cell_type: code`.

**Cell 1 — markdown:**
```
# C-DIRA Walkthrough

This notebook runs the full C-DIRA pipeline one stage at a time and visualizes
each part: preprocessing, pseudo-domains, model architecture, live training,
and evaluation.

**Requirements:** a CUDA GPU runtime (Runtime > Change runtime type > GPU) and a
Kaggle `kaggle.json` API token for an account that has accepted the *State Farm
Distracted Driver Detection* competition rules. Run the cells top to bottom from
the repository root.
```

**Cell 2 — code (bootstrap; stdlib only, before importing cdira):**
```python
import subprocess
import sys
from pathlib import Path

assert Path("pyproject.toml").exists() and Path("src/cdira").is_dir(), (
    "Run this notebook from the C-DIRA repository root."
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
```

**Cell 3 — markdown:**
```
## Stage 0 — Environment

Verify the repository, Python 3.12, and a usable CUDA GPU, then load the Colab
experiment config.
```

**Cell 4 — code:**
```python
import time

import matplotlib.pyplot as plt

from cdira import colab
from cdira.config import load_config

ROOT = colab.ensure_repository_root()
colab.ensure_supported_python()
DEVICE = colab.ensure_cuda_available()
CONFIG = load_config(Path("configs/colab.yaml"))
print("Repository:", ROOT, "| Device:", DEVICE)
```

**Cell 5 — code (Kaggle credentials):**
```python
colab.install_kaggle_credentials()
```

**Cell 6 — markdown:**
```
## Stage 1 — Dataset

Reuse a valid existing dataset, or download and extract the competition data.
Access errors are translated into an actionable message.
```

**Cell 7 — code:**
```python
from cdira.data.download import (
    DatasetLayoutError,
    download_competition,
    validate_dataset,
)

started = time.perf_counter()
try:
    fingerprint = validate_dataset(CONFIG.paths.data_root)
    print("Reusing dataset:", fingerprint.image_count, "images")
except DatasetLayoutError:
    try:
        fingerprint = download_competition(CONFIG.paths.data_root)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(colab.translate_kaggle_error(error)) from error
    print("Downloaded:", fingerprint.image_count, "images")
print(f"elapsed={time.perf_counter() - started:.1f}s")
```

**Cell 8 — markdown:**
```
## Stage 2 — Preprocessing & data

Build the stratified splits, then show class balance, split sizes, sample
images, and the augmentation applied during training.
```

**Cell 9 — code:**
```python
import pandas as pd

from cdira.data.manifests import build_split_manifests
from cdira.reporting import figures

bundle = build_split_manifests(
    CONFIG.paths.data_root, CONFIG.paths.manifest_root, CONFIG.seed
)
frames = {
    "train": pd.read_csv(bundle.train_path),
    "validation": pd.read_csv(bundle.validation_path),
    "test": pd.read_csv(bundle.test_path),
}
sizes = {name: len(frame) for name, frame in frames.items()}
counts = frames["train"]["class_id"].value_counts().sort_index().to_dict()
display(figures.split_sizes_table(sizes))
figures.class_distribution_figure(counts)
plt.show()
```

**Cell 10 — code:**
```python
from PIL import Image

from cdira.data.dataset import build_transform

samples, labels = [], []
for class_id, group in frames["train"].groupby("class_id"):
    row = group.iloc[0]
    samples.append(Image.open(CONFIG.paths.data_root / row.relative_path).convert("RGB"))
    labels.append(f"c{class_id}")
figures.sample_grid_figure(samples, labels, ncols=5, title="One sample per class")
plt.show()

row = frames["train"].iloc[0]
original = Image.open(CONFIG.paths.data_root / row.relative_path).convert("RGB")
train_tensor = build_transform(
    True, CONFIG.data.image_size, CONFIG.data.horizontal_flip, CONFIG.data.brightness
)(original)
eval_tensor = build_transform(False, CONFIG.data.image_size, False, CONFIG.data.brightness)(
    original
)
figures.augmentation_figure(original, train_tensor, eval_tensor)
plt.show()
```

**Cell 11 — markdown:**
```
## Stage 3 — Pseudo-domains

Cluster backbone features into pseudo-domains. Show the silhouette-based choice
of k, a 2-D PCA projection colored by domain, domain sizes, and a sample per
domain.
```

**Cell 12 — code:**
```python
from cdira.pipeline import prepare_domains

prepared = prepare_domains(CONFIG)
domains = prepared.domains
print("Selected k =", domains.k)

figures.silhouette_figure(domains.silhouette_scores, domains.k)
plt.show()

train_cache = prepared.caches["train"]
figures.domain_scatter_figure(
    train_cache.features,
    [domains.labels_by_path[path] for path in train_cache.paths],
)
plt.show()

display(figures.domain_sizes_table(domains.labels_by_path))

first_by_domain: dict[int, str] = {}
for path, domain_id in domains.labels_by_path.items():
    first_by_domain.setdefault(domain_id, path)
domain_images = [
    Image.open(CONFIG.paths.data_root / path).convert("RGB")
    for path in first_by_domain.values()
]
figures.sample_grid_figure(
    domain_images,
    [f"domain {domain_id}" for domain_id in first_by_domain],
    ncols=5,
    title="Representative image per domain",
)
plt.show()
```

**Cell 13 — markdown:**
```
## Stage 4 — Model architecture

Instantiate C-DIRA and inspect its branch structure and parameter budget.
```

**Cell 14 — code:**
```python
from cdira.models.cdira import CDIRA

model = CDIRA(
    num_classes=CONFIG.model.num_classes,
    num_domains=domains.k,
    top_k=CONFIG.model.top_k,
    global_hidden=CONFIG.model.global_hidden,
    roi_hidden=CONFIG.model.roi_hidden,
    fused_hidden=CONFIG.model.fused_hidden,
    routing_hidden=CONFIG.model.routing_hidden,
    domain_hidden=CONFIG.model.domain_hidden,
    grl_strength=CONFIG.model.grl_strength,
    pretrained=True,
)
figures.architecture_schematic_figure()
plt.show()
display(figures.parameter_table(model))
```

**Cell 15 — markdown:**
```
## Stage 5 — Training

Train C-DIRA with a live loss curve, then train the MobileNetV3 baseline.
Checkpoints are written to a fresh timestamped run under `artifacts/colab`.
```

**Cell 16 — code:**
```python
import torch
from IPython.display import clear_output
from torch.utils.data import DataLoader

from cdira.artifacts import RunArtifacts
from cdira.data.dataset import StateFarmDataset
from cdira.models.cdira import MobileNetBaseline
from cdira.pipeline import domain_mapping
from cdira.runtime import select_device
from cdira.training.engine import Trainer
from cdira.training.losses import LossWeights

domain_ids = domain_mapping(domains)


def make_loader(manifest: Path, training: bool) -> DataLoader:
    transform = build_transform(
        training,
        CONFIG.data.image_size,
        CONFIG.data.horizontal_flip if training else False,
        CONFIG.data.brightness,
    )
    dataset = StateFarmDataset(manifest, CONFIG.paths.data_root, transform, domain_ids)
    return DataLoader(
        dataset,
        batch_size=CONFIG.training.batch_size,
        shuffle=training,
        num_workers=CONFIG.data.num_workers,
    )


train_loader = make_loader(bundle.train_path, True)
validation_loader = make_loader(bundle.validation_path, False)
test_loader = make_loader(bundle.test_path, False)

run = RunArtifacts.create(CONFIG, run_id=time.strftime("colab-%Y%m%d-%H%M%S"))
device = select_device(CONFIG.training.device)
weights = LossWeights(**CONFIG.training.loss_weights.model_dump())
trainer = Trainer(
    model,
    device,
    CONFIG.training.max_epochs,
    CONFIG.training.patience,
    CONFIG.training.learning_rate,
    weights,
    CONFIG.training.confidence_threshold,
)

curve_figure, curve_axis = plt.subplots(figsize=(6, 4))


def update_curve(history: list[dict[str, float]]) -> None:
    figures.draw_loss_curve(curve_axis, history)
    clear_output(wait=True)
    display(curve_figure)


trainer.fit(train_loader, validation_loader, on_epoch_end=update_curve)
torch.save(model.state_dict(), run.root / "checkpoints" / "cdira.pt")

baseline = MobileNetBaseline(CONFIG.model.num_classes, pretrained=True)
Trainer(
    baseline,
    device,
    CONFIG.training.max_epochs,
    CONFIG.training.patience,
    CONFIG.training.learning_rate,
).fit(train_loader, validation_loader)
```

**Cell 17 — markdown:**
```
## Stage 6 — Evaluation & interpretability

Confusion matrix, per-class F1, routing usage, the C-DIRA-vs-baseline
comparison, and ROI saliency overlays on a few test images.
```

**Cell 18 — code:**
```python
import json

from cdira.evaluation.metrics import classification_metrics
from cdira.evaluation.predict import collect_predictions
from cdira.models.cdira import RoutingPolicy
from cdira.pipeline import _baseline_predictions

table = collect_predictions(
    model, test_loader, RoutingPolicy.HEAD, CONFIG.routing.threshold, device
)
metrics = classification_metrics(table, CONFIG.model.num_classes)
baseline_table = _baseline_predictions(baseline, test_loader, device)
baseline_metrics = classification_metrics(baseline_table, CONFIG.model.num_classes)
(run.root / "metrics" / "full.json").write_text(json.dumps(metrics, indent=2))
(run.root / "metrics" / "baseline.json").write_text(json.dumps(baseline_metrics, indent=2))

figures.confusion_matrix_figure(metrics["confusion_matrix"])
plt.show()
figures.per_class_f1_figure(metrics["per_class"])
plt.show()
figures.routing_usage_figure(metrics)
plt.show()
figures.model_comparison_figure(metrics, baseline_metrics)
plt.show()
```

**Cell 19 — code:**
```python
from cdira.evaluation.visualize import roi_overlay_figure

batch = next(iter(test_loader))
images = batch["image"][:4].to(device)
output = model.predict(images, RoutingPolicy.HEAD, CONFIG.routing.threshold)
for index in range(images.shape[0]):
    raw = Image.open(CONFIG.paths.data_root / batch["relative_path"][index]).convert("RGB")
    title = f"true={int(batch['target'][index])} pred={int(output.logits[index].argmax())}"
    roi_overlay_figure(raw, output.saliency[index], output.topk_indices[index], title)
    plt.show()
```

**Cell 20 — markdown:**
```
## Stage 7 — Reports & artifacts

Generate the Markdown and self-contained HTML reports and print the artifact
paths. In Colab the HTML report is offered as a download.
```

**Cell 21 — code:**
```python
from cdira.reporting.html_report import build_html_reproduction_report
from cdira.reporting.report import build_reproduction_report

markdown_path = build_reproduction_report(run.root)
html_path = build_html_reproduction_report(run.root)
print("Run directory:", run.root)
print("Checkpoint:", run.root / "checkpoints" / "cdira.pt")
print("Markdown report:", markdown_path)
print("HTML report:", html_path)
try:
    from google.colab import files  # type: ignore[import-not-found]

    files.download(str(html_path))
except Exception:  # noqa: BLE001
    pass
```

- [ ] **Step 3: Validate the notebook is well-formed JSON**

Run: `python -c "import json; json.load(open('docs/colab/cdira_walkthrough.ipynb')); print('ok')"`
Expected: prints `ok` (no exception).

- [ ] **Step 4: Verify every referenced helper symbol exists**

Run:
```bash
PYTHONPATH=src .venv/bin/python -c "
from cdira import colab
from cdira.reporting import figures
from cdira.evaluation.visualize import roi_overlay_figure
from cdira.pipeline import domain_mapping, _baseline_predictions
for name in ['ensure_repository_root','ensure_supported_python','ensure_cuda_available','install_kaggle_credentials','translate_kaggle_error']:
    assert hasattr(colab, name), name
for name in ['class_distribution_figure','split_sizes_table','sample_grid_figure','augmentation_figure','silhouette_figure','domain_scatter_figure','domain_sizes_table','architecture_schematic_figure','parameter_table','draw_loss_curve','loss_curve_figure','confusion_matrix_figure','per_class_f1_figure','routing_usage_figure','model_comparison_figure']:
    assert hasattr(figures, name), name
print('symbols ok')
"
```
Expected: prints `symbols ok`.

- [ ] **Step 5: Add a README pointer**

Add this section to `README.md` after the "Interactive demo" section:

```markdown
## Colab walkthrough

`docs/colab/cdira_walkthrough.ipynb` runs the whole pipeline one stage at a
time on a Colab GPU runtime and visualizes preprocessing, pseudo-domains, the
model architecture, live training curves, and evaluation. It uses
`configs/colab.yaml` and requires a Kaggle token for an account that has
accepted the competition rules.
```

- [ ] **Step 6: Run full verification**

Run:
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src/cdira
git diff --check
```
Expected: all tests pass, Ruff is clean, mypy reports no issues, and `git diff --check` reports no whitespace errors.

- [ ] **Step 7: Commit**

```bash
git add docs/colab/cdira_walkthrough.ipynb README.md
git commit -m "feat: add C-DIRA Colab walkthrough notebook"
```

---

## Self-Review

**Spec coverage:**
- Repo-root / Python / CUDA checks → Task 3 (notebook Stage 0, Cell 4).
- Editable install + Kaggle credential upload (0600, no logging) → Tasks 3–4, notebook Cells 2 and 5.
- Dataset download + `c0`–`c9` validation + reuse + actionable Kaggle error → Task 4 (`translate_kaggle_error`), notebook Cell 7.
- Preprocessing visuals (class distribution, split sizes, sample grid, augmentation before/after) → Task 5, notebook Cells 9–10.
- Pseudo-domains (silhouette-vs-k, PCA scatter, per-domain grid, sizes) → Task 6, notebook Cell 12.
- Architecture (schematic + parameter table) → Task 7, notebook Cell 14.
- Live training + baseline → Task 2 (`on_epoch_end`) + Task 8 (`draw_loss_curve`), notebook Cell 16.
- Evaluation (confusion, per-class F1, routing usage, comparison, ROI overlays) → Tasks 8–9, notebook Cells 18–19.
- Reports + artifact links → notebook Cell 21 (existing `build_reproduction_report` / `build_html_reproduction_report`).
- `configs/colab.yaml` (CUDA, mixed precision, num_workers 2, artifacts/colab) → Task 1.
- No new dependencies; PCA not t-SNE; figures return objects → enforced in Global Constraints and Tasks 5–8.

**Placeholder scan:** none — every code and test step contains concrete content.

**Type consistency:** `on_epoch_end: Callable[[list[dict[str, float]]], None]` is consistent between Task 2 and its use in notebook Cell 16; `draw_loss_curve(axis, history)` / `loss_curve_figure(history)` names match Task 8 and Cell 16; `roi_overlay_figure(image, saliency, topk_indices, title)` matches Task 9 and Cell 19; figure builders consistently return `Figure` / `DataFrame`.
