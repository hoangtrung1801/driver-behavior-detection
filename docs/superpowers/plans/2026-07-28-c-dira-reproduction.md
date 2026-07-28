# C-DIRA Research Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch implementation of C-DIRA that downloads the State Farm dataset, trains on Apple Silicon, runs the paper's core evaluations, and produces an auditable reproduction report.

**Architecture:** A configuration-driven `src/cdira` package owns dataset preparation, fixed pseudo-domain labels, the complete joint C-DIRA model, training, conditional inference, evaluation, and reporting. The full architecture is implemented before the main run; small modules keep tensor, persistence, and experiment boundaries independently testable.

**Tech Stack:** Python 3.12 managed by `uv`, PyTorch 2.7.1, torchvision 0.22.1, MPS, scikit-learn, pandas, NumPy, Pillow, PyYAML, Pydantic, Typer, torchmetrics, fvcore, matplotlib, seaborn, pytest, Ruff, and mypy.

## Global Constraints

- Target hardware is an Apple M4 Pro with 24 GB unified memory; MPS is the default accelerator.
- The static-image reproduction is in scope; webcam, video, mobile packaging, and edge deployment are out of scope.
- Inputs are resized directly to 224×224 and normalized with ImageNet statistics.
- The primary split is class-stratified 80/10/10 with seed 42.
- The paper profile uses AdamW at `1e-5`, batch size 32, at most 50 epochs, and validation-loss patience 5.
- Pseudo-domain candidates are every integer from 2 through 40, selected on 5,000 training features with seed 42 and `n_init=10`.
- C-DIRA uses Top-K `5`, routing hidden width `128`, ROI width `512`, domain hidden width `256`, and GRL strength `1.0`.
- The primary inference policy routes when `routing_probability >= threshold`; the Algorithm 2 comparison routes when `global_confidence < threshold`.
- Raw data, credentials, derived caches, checkpoints, and run artifacts must not enter version control.
- Every persisted derivative must carry source/configuration fingerprints and be written atomically.
- The implementation must not silently fall back from MPS to CPU.
- Exact resolved dependencies are recorded in `uv.lock`.

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Runtime, development dependencies, and entry point |
| `configs/*.yaml` | Validated experiment settings and named profiles |
| `src/cdira/config.py` | Pydantic configuration schema and YAML resolution |
| `src/cdira/runtime.py` | Seeds, device selection, synchronization, atomic writes, fingerprints |
| `src/cdira/data/download.py` | Kaggle authentication, archive download, validation, promotion |
| `src/cdira/data/manifests.py` | Dataset inventory, split CSVs, subject-overlap audit |
| `src/cdira/data/dataset.py` | Dataset and transform construction |
| `src/cdira/domains/features.py` | Frozen pretrained feature extraction and cache |
| `src/cdira/domains/clustering.py` | Silhouette search, K-means fit, label persistence |
| `src/cdira/models/grl.py` | Gradient reversal autograd operation |
| `src/cdira/models/roi.py` | Saliency, Top-K selection, ROI pooling |
| `src/cdira/models/cdira.py` | Complete model, training forward, conditional inference |
| `src/cdira/models/baseline.py` | MobileNetV3-Small comparison model |
| `src/cdira/training/losses.py` | Routing targets and five-term objective |
| `src/cdira/training/checkpoints.py` | Exact resume and weights-only loading |
| `src/cdira/training/engine.py` | Epoch loops, early stopping, metrics, non-finite checks |
| `src/cdira/evaluation/predict.py` | Batched predictions and routing traces |
| `src/cdira/evaluation/metrics.py` | Classification, per-class, routing, confusion metrics |
| `src/cdira/evaluation/corruptions.py` | Deterministic robustness transformations |
| `src/cdira/evaluation/efficiency.py` | Parameters, FLOPs, conditional compute, MPS latency |
| `src/cdira/evaluation/visualize.py` | Saliency/Top-K and result plots |
| `src/cdira/experiments/ablations.py` | Full model and three ablation definitions |
| `src/cdira/experiments/loco.py` | Resumable leave-one-cluster-out jobs and aggregation |
| `src/cdira/reporting/report.py` | JSON/CSV/Markdown reproduction report |
| `src/cdira/cli.py` | User-facing commands and paper-run orchestration |
| `tests/` | Unit, integration, MPS, failure, and opt-in dataset tests |

---

### Task 1: Reproducible Project Foundation

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `configs/base.yaml`
- Create: `configs/paper.yaml`
- Create: `configs/standard.yaml`
- Create: `configs/smoke.yaml`
- Create: `src/cdira/__init__.py`
- Create: `src/cdira/config.py`
- Create: `src/cdira/runtime.py`
- Create: `tests/test_config.py`
- Create: `tests/test_runtime.py`

**Interfaces:**
- Produces: `load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig`
- Produces: `select_device(requested: Literal["mps", "cuda", "cpu"]) -> torch.device`
- Produces: `seed_everything(seed: int) -> None`
- Produces: `fingerprint_json(value: Mapping[str, object]) -> str`
- Produces: `atomic_path(destination: Path) -> ContextManager[Path]`
- Produces: `run_cpu_boundary(name: str, reason: str, tensors: Sequence[Tensor], operation: Callable) -> object`

- [ ] **Step 1: Write failing configuration and runtime tests**

```python
# tests/test_config.py
from pathlib import Path
from cdira.config import load_config

def test_paper_config_resolves_fixed_reproduction_values() -> None:
    cfg = load_config(Path("configs/paper.yaml"))
    assert cfg.seed == 42
    assert cfg.data.image_size == 224
    assert cfg.model.top_k == 5
    assert cfg.model.routing_hidden == 128
    assert cfg.training.batch_size == 32
    assert cfg.training.max_epochs == 50
    assert cfg.routing.primary_policy == "head"

def test_override_is_validated() -> None:
    cfg = load_config(Path("configs/paper.yaml"), ["training.batch_size=8"])
    assert cfg.training.batch_size == 8
```

```python
# tests/test_runtime.py
from cdira.runtime import fingerprint_json

def test_fingerprint_is_order_independent() -> None:
    assert fingerprint_json({"a": 1, "b": 2}) == fingerprint_json({"b": 2, "a": 1})
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `uv run pytest tests/test_config.py tests/test_runtime.py -q`

Expected: FAIL because `cdira.config` and `cdira.runtime` do not exist.

- [ ] **Step 3: Add package metadata, dependencies, configuration, and runtime helpers**

Use Python `==3.12.*`; add `torch==2.7.1`, `torchvision==0.22.1`, `numpy`, `pandas`, `scikit-learn`, `Pillow`, `PyYAML`, `pydantic`, `typer`, `torchmetrics`, `fvcore`, `matplotlib`, `seaborn`, `kaggle`, and `rich`. Add pytest, pytest-cov, Ruff, and mypy as development dependencies. Register `cdira = "cdira.cli:app"` and configure a `src` layout.

Implement nested Pydantic models with these required sections: `paths`, `data`, `domains`, `model`, `training`, `routing`, `evaluation`, and `profile`. Resolve YAML by deep-merging `base.yaml`, the selected profile, then dot-path overrides. Reject unknown keys.

```python
# src/cdira/runtime.py
def fingerprint_json(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def select_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)

@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
```

`run_cpu_boundary` synchronizes MPS, times the transfer and CPU operation, and
logs the operation name, source device, destination device, duration, and
reason supplied by the caller. It is the only permitted compatibility path for
an operation that cannot run on MPS.

Set `base.yaml` to seed 42, image size 224, split `[0.8, 0.1, 0.1]`, Top-K 5, the specified head widths/loss weights, MPS, and paths below the workspace. `paper.yaml` sets batch 32/50 epochs/all evaluations; `standard.yaml` uses the full data split with 10 epochs, thresholds `[0.5,0.9]`, one level per corruption, and three LOCO folds; `smoke.yaml` sets synthetic data/one epoch/two batches/one LOCO fold.

Add `.gitignore` entries for `.venv/`, `data/`, `artifacts/`, `*.pt`, `*.npz`, `.kaggle/`, and macOS/Python caches.

- [ ] **Step 4: Lock dependencies and verify foundation**

Run:

```bash
uv python install 3.12
uv sync --all-groups
uv run pytest tests/test_config.py tests/test_runtime.py -q
uv run ruff check src tests
```

Expected: tests PASS, Ruff exits 0, and `uv.lock` is created.

- [ ] **Step 5: Commit project foundation**

Run:

```bash
git init
git add .gitignore pyproject.toml uv.lock configs src/cdira/__init__.py src/cdira/config.py src/cdira/runtime.py tests/test_config.py tests/test_runtime.py docs
git commit -m "build: initialize C-DIRA reproduction"
```

Expected: one root commit; data and artifact directories remain untracked/ignored.

---

### Task 2: Dataset Download, Validation, Splits, and Loading

**Files:**
- Create: `src/cdira/data/__init__.py`
- Create: `src/cdira/data/download.py`
- Create: `src/cdira/data/manifests.py`
- Create: `src/cdira/data/dataset.py`
- Create: `configs/diagnostics/no_horizontal_flip.yaml`
- Create: `configs/diagnostics/subject_disjoint.yaml`
- Create: `tests/data_factory.py`
- Create: `tests/test_download.py`
- Create: `tests/test_manifests.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `atomic_path`, `fingerprint_json`
- Produces: `DatasetFingerprint(root: Path, sha256: str, image_count: int)`
- Produces: `download_competition(destination: Path) -> DatasetFingerprint`
- Produces: `build_split_manifests(dataset_root: Path, output_dir: Path, seed: int) -> SplitBundle`
- Produces: `build_subject_disjoint_manifests(dataset_root: Path, output_dir: Path, seed: int) -> SplitBundle`
- Produces: `validate_split_bundle(bundle: SplitBundle, dataset_root: Path) -> None`
- Produces: `StateFarmDataset(manifest: Path, dataset_root: Path, transform: Callable)`
- Produces: `build_transform(training: bool, image_size: int, horizontal_flip: bool) -> Callable`

- [ ] **Step 1: Write failing dataset tests using a generated ten-class fixture**

```python
def test_split_is_stratified_deterministic_and_disjoint(tmp_path: Path) -> None:
    root = make_state_farm_fixture(tmp_path, images_per_class=20, subjects=5)
    first = build_split_manifests(root, tmp_path / "a", seed=42)
    second = build_split_manifests(root, tmp_path / "b", seed=42)
    assert first.fingerprint == second.fingerprint
    assert set(first.train.relative_path).isdisjoint(first.validation.relative_path)
    assert set(first.train.relative_path).isdisjoint(first.test.relative_path)
    assert first.train.groupby("class_id").size().tolist() == [16] * 10

def test_transform_has_expected_shape_and_normalization(tmp_path: Path) -> None:
    image = Image.new("RGB", (640, 480), "white")
    tensor = build_transform(False, 224, False)(image)
    assert tensor.shape == (3, 224, 224)

def test_subject_disjoint_diagnostic_has_no_driver_overlap(tmp_path: Path) -> None:
    root = make_state_farm_fixture(tmp_path, images_per_class=20, subjects=10)
    bundle = build_subject_disjoint_manifests(root, tmp_path / "subjects", seed=42)
    assert set(bundle.train.subject).isdisjoint(bundle.validation.subject)
    assert set(bundle.train.subject).isdisjoint(bundle.test.subject)
```

Mock `KaggleApi` in `test_download.py` and assert missing authentication, missing class folders, and partial archives raise messages containing the recovery action.

- [ ] **Step 2: Verify dataset tests fail**

Run: `uv run pytest tests/test_download.py tests/test_manifests.py tests/test_dataset.py -q`

Expected: FAIL because the data modules do not exist.

- [ ] **Step 3: Implement atomic download and structural validation**

Authenticate with `KaggleApi.authenticate()`, call
`competition_download_files("state-farm-distracted-driver-detection", path=staging)`,
extract with `zipfile`, require `train/c0` through `train/c9`, inventory JPEGs,
hash sorted `(relative_path, size)` pairs, and atomically promote the validated
directory. Never log credential values.

```python
def validate_dataset(root: Path) -> DatasetFingerprint:
    records = []
    for class_id in range(10):
        folder = root / "train" / f"c{class_id}"
        if not folder.is_dir():
            raise DatasetLayoutError(f"Missing {folder}; rerun `cdira data download`")
        records.extend((path.relative_to(root).as_posix(), path.stat().st_size)
                       for path in sorted(folder.glob("*.jpg")))
    if not records:
        raise DatasetLayoutError("No training JPEGs found")
    return DatasetFingerprint(root, fingerprint_json({"files": records}), len(records))
```

- [ ] **Step 4: Implement manifests, overlap audit, dataset, and transforms**

Read optional `driver_imgs_list.csv`; stratify twice with
`train_test_split(..., stratify=class_id, random_state=42)` to obtain exact
80/10/10 proportions. Persist columns `relative_path,class_id,class_name,subject,split,dataset_sha256`.
Validate unique paths and matching fingerprints. Emit `subject_overlap.json`.
For the diagnostic subject-disjoint split, assign whole seeded subject groups
to approximately 80/10/10 partitions, reject any overlap, and record the
resulting class imbalance. Configure that path in
`configs/diagnostics/subject_disjoint.yaml`; configure the approved image-level
split with horizontal flipping disabled in
`configs/diagnostics/no_horizontal_flip.yaml`.

`StateFarmDataset.__getitem__` returns a dictionary with `image`, `target`,
`domain=-1`, `relative_path`, and `subject`. Training uses `Resize((224,224))`,
`RandomHorizontalFlip(0.5)`, `ColorJitter(brightness=0.2)`, tensor conversion,
and ImageNet normalization; evaluation omits random transforms.

- [ ] **Step 5: Run dataset tests**

Run: `uv run pytest tests/test_download.py tests/test_manifests.py tests/test_dataset.py -q`

Expected: PASS, including malformed archive and split-overlap failures.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/data tests/data_factory.py tests/test_download.py tests/test_manifests.py tests/test_dataset.py
git commit -m "feat: add reproducible State Farm data pipeline"
```

---

### Task 3: Pseudo-Domain Feature Cache and Clustering

**Files:**
- Create: `src/cdira/domains/__init__.py`
- Create: `src/cdira/domains/features.py`
- Create: `src/cdira/domains/clustering.py`
- Create: `tests/test_domain_features.py`
- Create: `tests/test_clustering.py`
- Modify: `src/cdira/data/dataset.py`

**Interfaces:**
- Consumes: split manifests and `StateFarmDataset`
- Produces: `FeatureCache(paths: list[str], features: NDArray[np.float32], metadata: dict[str, object])`
- Produces: `extract_feature_cache(loader: DataLoader, device: torch.device, destination: Path) -> FeatureCache`
- Produces: `PseudoDomains(k: int, centroids: NDArray[np.float32], labels_by_path: dict[str, int], silhouette_scores: dict[int, float], fit_paths: list[str])`
- Produces: `fit_pseudo_domains(train: FeatureCache, validation: FeatureCache, test: FeatureCache, config: DomainConfig) -> PseudoDomains`
- Updates: `StateFarmDataset(..., domains: Mapping[str, int] | None)` returns persisted domain IDs

- [ ] **Step 1: Write failing cache and clustering tests**

```python
def test_cluster_selection_uses_train_only_and_lower_k_wins_tie() -> None:
    train = separable_feature_cache(cluster_count=3, samples=600)
    validation = separable_feature_cache(cluster_count=3, samples=60)
    test = separable_feature_cache(cluster_count=3, samples=60)
    result = fit_pseudo_domains(
        train, validation, test,
        DomainConfig(candidates=[2, 3, 4], sample_size=500, seed=42, n_init=10),
    )
    assert result.k == 3
    assert set(validation.paths).issubset(result.labels_by_path)
    assert result.fit_paths == train.paths

def test_feature_cache_rejects_transform_fingerprint_mismatch(tmp_path: Path) -> None:
    with pytest.raises(FingerprintMismatch):
        load_feature_cache(tmp_path / "features.npz", expected_transform_sha256="wrong")
```

- [ ] **Step 2: Confirm tests fail**

Run: `uv run pytest tests/test_domain_features.py tests/test_clustering.py -q`

Expected: FAIL because domain modules do not exist.

- [ ] **Step 3: Implement frozen pretrained feature extraction**

Create `mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features`,
set evaluation mode, disable gradients, adaptive-average-pool to `[B,576]`,
move features to CPU float32, and save one compressed NPZ plus JSON metadata.
Metadata includes ordered paths, weights enum, dataset/split/transform
fingerprints, feature dimension, versions, and seed. CPU transfer is an
explicit logged boundary. Use the non-random evaluation transform for train,
validation, and test feature caches so pseudo-domain labels cannot depend on
augmentation draws.

- [ ] **Step 4: Implement deterministic silhouette search and fixed labels**

```python
sample_size = min(config.sample_size, len(train.features))
indices = np.random.default_rng(config.seed).choice(
    len(train.features), sample_size, replace=False
)
scores: dict[int, float] = {}
for k in config.candidates:
    candidate = KMeans(n_clusters=k, random_state=config.seed, n_init=config.n_init)
    sample_labels = candidate.fit_predict(train.features[indices])
    scores[k] = float(silhouette_score(train.features[indices], sample_labels))
best_k = min(k for k, score in scores.items() if score == max(scores.values()))
final = KMeans(n_clusters=best_k, random_state=config.seed, n_init=config.n_init)
train_labels = final.fit_predict(train.features)
```

Use `final.predict` for validation and test, persist centroids and
`relative_path,domain_id,split`, and reject caches whose metadata fingerprints
differ. Add domain mapping support to `StateFarmDataset`.

- [ ] **Step 5: Run domain tests**

Run: `uv run pytest tests/test_domain_features.py tests/test_clustering.py tests/test_dataset.py -q`

Expected: PASS and no clustering fit call receives validation/test features.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/domains src/cdira/data/dataset.py tests/test_domain_features.py tests/test_clustering.py
git commit -m "feat: add pseudo-domain labeling pipeline"
```

---

### Task 4: Complete C-DIRA and Baseline Models

**Files:**
- Create: `src/cdira/models/__init__.py`
- Create: `src/cdira/models/grl.py`
- Create: `src/cdira/models/roi.py`
- Create: `src/cdira/models/cdira.py`
- Create: `src/cdira/models/baseline.py`
- Create: `tests/test_grl.py`
- Create: `tests/test_roi.py`
- Create: `tests/test_cdira_model.py`

**Interfaces:**
- Produces: `gradient_reverse(x: Tensor, strength: float) -> Tensor`
- Produces: `topk_roi_pool(feature_map: Tensor, k: int) -> ROIResult`
- Produces: `CDIRA.forward_train(images: Tensor) -> TrainOutput`
- Produces: `CDIRA.predict(images: Tensor, policy: RoutingPolicy, threshold: float) -> InferenceOutput`
- Produces: `MobileNetBaseline.forward(images: Tensor) -> Tensor`
- `TrainOutput` fields: `global_logits`, `fused_logits`, `routing_logits`, `domain_logits`, `saliency`, `topk_indices`
- `InferenceOutput` fields: `logits`, `global_logits`, `routing_probability`, `global_confidence`, `roi_mask`, `saliency`, `topk_indices`

- [ ] **Step 1: Write failing mathematical and shape tests**

```python
def test_gradient_reversal_negates_and_scales_gradient() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(x, 0.25).sum().backward()
    assert x.grad.item() == pytest.approx(-0.25)

def test_topk_roi_pool_matches_hand_calculation() -> None:
    fmap = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    result = topk_roi_pool(fmap, k=2)
    assert result.indices.tolist() == [[1, 2]]
    assert result.pooled.tolist() == [[3.5]]

def test_full_training_forward_shapes() -> None:
    model = CDIRA(num_classes=10, num_domains=30, pretrained=False)
    output = model.forward_train(torch.randn(2, 3, 224, 224))
    assert output.global_logits.shape == (2, 10)
    assert output.fused_logits.shape == (2, 10)
    assert output.routing_logits.shape == (2,)
    assert output.domain_logits.shape == (2, 30)
    assert output.saliency.shape == (2, 7, 7)
```

Add two spies asserting `predict` calls ROI pooling for selected indices only
and does not call it for an all-easy batch.

- [ ] **Step 2: Verify model tests fail**

Run: `uv run pytest tests/test_grl.py tests/test_roi.py tests/test_cdira_model.py -q`

Expected: FAIL because model modules do not exist.

- [ ] **Step 3: Implement GRL and Top-K pooling**

```python
class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None

def topk_roi_pool(feature_map: Tensor, k: int) -> ROIResult:
    saliency = torch.linalg.vector_norm(feature_map, ord=2, dim=1)
    flat_saliency = saliency.flatten(1)
    indices = flat_saliency.topk(k, dim=1).indices
    spatial = feature_map.flatten(2).transpose(1, 2)
    gathered = spatial.gather(1, indices.unsqueeze(-1).expand(-1, -1, spatial.shape[-1]))
    return ROIResult(gathered.mean(dim=1), saliency, indices)
```

Validate `1 <= k <= H*W`.

- [ ] **Step 4: Implement the complete model and true conditional inference**

Build the exact heads from the specification. `forward_train` always evaluates
all branches. `predict` first computes the shared map, global logits, routing
logits, probabilities, and confidence; then selects:

```python
if policy == RoutingPolicy.HEAD:
    roi_mask = routing_probability >= threshold
elif policy == RoutingPolicy.CONFIDENCE:
    roi_mask = global_confidence < threshold
else:
    roi_mask = torch.zeros_like(global_confidence, dtype=torch.bool)
```

Start final logits as a clone of global logits. If `roi_mask.any()`, index the
feature map and global features, compute ROI/fused logits for only those rows,
and scatter them into final logits. Return empty/NaN-marked saliency and `-1`
indices for un-routed rows so downstream code cannot confuse absent ROI data
with zero saliency. Implement the baseline with the same feature module and
`Linear(576,10)`.

- [ ] **Step 5: Run model tests**

Run: `uv run pytest tests/test_grl.py tests/test_roi.py tests/test_cdira_model.py -q`

Expected: PASS, including all-easy and mixed-batch short-circuit tests.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/models tests/test_grl.py tests/test_roi.py tests/test_cdira_model.py
git commit -m "feat: implement complete C-DIRA architecture"
```

---

### Task 5: Joint Loss, Training, Checkpoints, and Exact Resume

**Files:**
- Create: `src/cdira/training/__init__.py`
- Create: `src/cdira/training/losses.py`
- Create: `src/cdira/training/checkpoints.py`
- Create: `src/cdira/training/engine.py`
- Create: `tests/test_losses.py`
- Create: `tests/test_checkpoints.py`
- Create: `tests/test_training_engine.py`

**Interfaces:**
- Consumes: `TrainOutput`, class targets, fixed domain targets, `TrainingConfig`
- Produces: `routing_targets(global_logits: Tensor, targets: Tensor, confidence_threshold: float) -> Tensor`
- Produces: `compute_cdira_loss(output: TrainOutput, targets: Tensor, domains: Tensor, weights: LossWeights) -> LossBreakdown`
- Produces: `save_checkpoint(path: Path, state: TrainingState, fingerprints: Fingerprints) -> None`
- Produces: `load_checkpoint(path: Path, expected: Fingerprints, mode: Literal["resume", "weights"]) -> TrainingState`
- Produces: `Trainer.fit(train_loader, validation_loader) -> FitResult`

- [ ] **Step 1: Write failing loss and resume tests**

```python
def test_routing_targets_use_detached_global_difficulty() -> None:
    logits = torch.tensor([[8.0, 0.0], [0.2, 0.1], [0.0, 8.0]], requires_grad=True)
    targets = torch.tensor([0, 0, 0])
    result = routing_targets(logits, targets, confidence_threshold=0.9)
    assert result.tolist() == [0.0, 1.0, 1.0]
    assert not result.requires_grad

def test_total_loss_uses_paper_weights() -> None:
    breakdown = compute_cdira_loss(fixed_train_output(), targets(), domains(), paper_weights())
    expected = (
        0.5 * breakdown.global_ce
        + breakdown.fused_ce
        + 0.5 * breakdown.routing_bce
        + 0.01 * breakdown.routing_regularizer
        + 0.5 * breakdown.domain_ce
    )
    torch.testing.assert_close(breakdown.total, expected)
```

Create a four-step deterministic toy training test: save after two steps,
resume for two, and compare parameters/optimizer/RNG state with uninterrupted
four-step training.

- [ ] **Step 2: Verify training tests fail**

Run: `uv run pytest tests/test_losses.py tests/test_checkpoints.py tests/test_training_engine.py -q`

Expected: FAIL because training modules do not exist.

- [ ] **Step 3: Implement routing targets and exact five-term loss**

```python
with torch.no_grad():
    probabilities = global_logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    route_target = ((prediction != targets) | (confidence < threshold)).float()

positive = route_target.sum()
negative = route_target.numel() - positive
positive_weight = negative / positive if positive > 0 and negative > 0 else positive.new_tensor(1.0)
routing_bce = F.binary_cross_entropy_with_logits(
    output.routing_logits, route_target, pos_weight=positive_weight
)
routing_regularizer = output.routing_logits.sigmoid().mean()
```

Calculate both cross-entropies and domain cross-entropy, apply exactly
`0.5,1.0,0.5,0.01,0.5`, and return tensors plus routing counts/probability.
Raise immediately on non-finite terms.

- [ ] **Step 4: Implement MPS-aware trainer and exact resume**

Seed Python/NumPy/PyTorch; create AdamW at `1e-5`; use `torch.autocast("mps",
dtype=torch.float16)` only after a one-batch capability probe, otherwise log
that the entire run is float32. Do not set global fallback environment flags.
Move only batches/models to MPS; keep clustering and reporting on CPU.

Track validation total loss, best checkpoint, patience five, per-term losses,
accuracy, macro F1, routing class balance, learning rate, and epoch duration.
Persist model, optimizer, scaler, epoch, early-stop state, RNG states, resolved
config, fingerprints, and versions. In `resume` mode require exact fingerprint
matches; in `weights` mode load model tensors only.

- [ ] **Step 5: Run training tests**

Run: `uv run pytest tests/test_losses.py tests/test_checkpoints.py tests/test_training_engine.py -q`

Expected: PASS, including parameter equality after exact resume.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/training tests/test_losses.py tests/test_checkpoints.py tests/test_training_engine.py
git commit -m "feat: add joint C-DIRA training and exact resume"
```

---

### Task 6: Prediction, Classification Metrics, and Routing Sweeps

**Files:**
- Create: `src/cdira/evaluation/__init__.py`
- Create: `src/cdira/evaluation/predict.py`
- Create: `src/cdira/evaluation/metrics.py`
- Create: `tests/test_prediction.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_routing_sweep.py`

**Interfaces:**
- Consumes: trained `CDIRA`, loader, routing policy, threshold
- Produces: `PredictionTable` with paths, targets, logits, global logits, confidence, routing probability, ROI mask, class, domain
- Produces: `classification_metrics(table: PredictionTable) -> dict[str, object]`
- Produces: `routing_sweep(model, loader, policies, thresholds) -> pandas.DataFrame`

- [ ] **Step 1: Write failing metric and sweep tests**

```python
def test_metrics_report_macro_weighted_and_per_class() -> None:
    table = prediction_table(targets=[0, 0, 1, 2], predictions=[0, 1, 1, 2])
    result = classification_metrics(table)
    assert set(result) >= {"accuracy", "macro", "weighted", "per_class", "confusion_matrix"}
    assert result["accuracy"] == pytest.approx(0.75)

def test_head_threshold_sweep_has_nonincreasing_roi_usage() -> None:
    frame = routing_sweep(fake_model(), fake_loader(), ["head"], [0.1, 0.5, 0.9])
    assert frame["roi_usage"].tolist() == sorted(frame["roi_usage"], reverse=True)
```

- [ ] **Step 2: Confirm evaluation tests fail**

Run: `uv run pytest tests/test_prediction.py tests/test_metrics.py tests/test_routing_sweep.py -q`

Expected: FAIL because prediction and metric modules do not exist.

- [ ] **Step 3: Implement auditable prediction tables and metrics**

Collect tensors on CPU per batch and save one row per sample. Assert each
manifest path appears exactly once. Compute accuracy, macro and weighted
precision/recall/F1 with explicit zero-division behavior, per-class metrics,
10×10 confusion matrix, total/per-class ROI usage, and routed/global-only
accuracy. Save logits/probabilities in NPZ and tabular values in Parquet or CSV.

- [ ] **Step 4: Implement both-policy threshold sweep**

Evaluate thresholds `[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]` for
`RoutingPolicy.HEAD` and `RoutingPolicy.CONFIDENCE` from one fixed checkpoint.
Write a row per policy/threshold with accuracy, macro F1, ROI count/rate, and
per-class ROI rates. Verify the head policy is nonincreasing and record, rather
than fail on, confidence-policy direction.

- [ ] **Step 5: Run evaluation tests**

Run: `uv run pytest tests/test_prediction.py tests/test_metrics.py tests/test_routing_sweep.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/evaluation tests/test_prediction.py tests/test_metrics.py tests/test_routing_sweep.py
git commit -m "feat: add classification and routing evaluation"
```

---

### Task 7: Robustness, Saliency, and Efficiency Evaluation

**Files:**
- Create: `src/cdira/evaluation/corruptions.py`
- Create: `src/cdira/evaluation/robustness.py`
- Create: `src/cdira/evaluation/efficiency.py`
- Create: `src/cdira/evaluation/visualize.py`
- Create: `tests/test_corruptions.py`
- Create: `tests/test_efficiency.py`
- Create: `tests/test_visualize.py`

**Interfaces:**
- Produces: `apply_corruption(image: Image, kind: CorruptionKind, severity: float) -> Image`
- Produces: `evaluate_robustness(models: Mapping[str, Module], loader_factory, config) -> DataFrame`
- Produces: `measure_efficiency(model: CDIRA, sample: Tensor, roi_usage: float, device: torch.device) -> EfficiencyReport`
- Produces: `save_roi_figure(record: PredictionRecord, image: Image, destination: Path) -> Path`

- [ ] **Step 1: Write failing deterministic corruption and compute tests**

```python
@pytest.mark.parametrize(
    ("kind", "severity"),
    [("blur", 2), ("jpeg", 25), ("low_light", 0.25), ("occlusion", 0.3)],
)
def test_corruptions_are_deterministic(kind: str, severity: float) -> None:
    image = patterned_image()
    first = np.asarray(apply_corruption(image, kind, severity))
    second = np.asarray(apply_corruption(image, kind, severity))
    np.testing.assert_array_equal(first, second)

def test_expected_flops_interpolates_observed_roi_usage() -> None:
    assert expected_conditional_flops(global_flops=60, roi_extra_flops=5, roi_usage=0.2) == 61
```

- [ ] **Step 2: Confirm diagnostic tests fail**

Run: `uv run pytest tests/test_corruptions.py tests/test_efficiency.py tests/test_visualize.py -q`

Expected: FAIL because the diagnostic modules do not exist.

- [ ] **Step 3: Implement fixed corruption suite**

Apply corruptions before tensor normalization:

```python
BLUR_RADII = [0, 1, 2, 3, 4]
JPEG_QUALITIES = [100, 75, 50, 25, 10]
LOW_LIGHT_FACTORS = [1.0, 0.75, 0.50, 0.25, 0.10]
OCCLUSION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]
```

Use Pillow Gaussian blur, in-memory JPEG encode/decode, pixel multiplication,
and a centered black square whose side is `sqrt(area_fraction)` times the
shorter image dimension. Evaluate baseline and C-DIRA accuracy/macro F1 for
every level and persist rows with the clean dataset fingerprint.

- [ ] **Step 4: Implement parameter/FLOP/latency measurement and ROI plots**

Use `sum(p.numel())` for total/trainable parameters and fvcore analysis on CPU
for global and forced-ROI paths. Calculate conditional FLOPs as
`global + roi_usage * roi_extra`. For MPS batch-one latency, run 100 warm-ups,
call `torch.mps.synchronize()` around each of 1,000 iterations, and return mean,
median, p90, and p95 milliseconds. Save environment and profiler warnings.

Render original image, heatmap, Top-K positions, true/predicted labels,
confidence, routing probability, and route decision. Select correct and
incorrect samples per class deterministically by path.

- [ ] **Step 5: Run diagnostic tests**

Run: `uv run pytest tests/test_corruptions.py tests/test_efficiency.py tests/test_visualize.py -q`

Expected: PASS. MPS timing tests are skipped when MPS is unavailable.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/evaluation tests/test_corruptions.py tests/test_efficiency.py tests/test_visualize.py
git commit -m "feat: add robustness saliency and efficiency evaluation"
```

---

### Task 8: Ablation and LOCO Experiment Orchestration

**Files:**
- Create: `src/cdira/experiments/__init__.py`
- Create: `src/cdira/experiments/ablations.py`
- Create: `src/cdira/experiments/loco.py`
- Create: `configs/ablations/no_roi.yaml`
- Create: `configs/ablations/no_adversarial.yaml`
- Create: `configs/ablations/no_dynamic_routing.yaml`
- Create: `configs/loco.yaml`
- Create: `tests/test_ablations.py`
- Create: `tests/test_loco.py`

**Interfaces:**
- Produces: `AblationSpec(name: str, model_mode: str, domain_loss_weight: float, routing_policy: str)`
- Produces: `required_ablations() -> tuple[AblationSpec, ...]`
- Produces: `build_loco_folds(domain_manifest: DataFrame, seed: int) -> list[LocoFold]`
- Produces: `run_loco_fold(fold: LocoFold, model_kind: Literal["baseline", "cdira"], config: ExperimentConfig) -> FoldResult`
- Produces: `aggregate_loco(results: Sequence[FoldResult]) -> dict[str, object]`

- [ ] **Step 1: Write failing ablation and LOCO tests**

```python
def test_required_ablations_match_paper() -> None:
    specs = {spec.name: spec for spec in required_ablations()}
    assert set(specs) == {"full", "no_roi", "no_adversarial", "no_dynamic_routing"}
    assert specs["no_adversarial"].domain_loss_weight == 0.0
    assert specs["no_dynamic_routing"].routing_policy == "always"

def test_loco_fold_has_no_held_out_domain_in_train_or_validation() -> None:
    folds = build_loco_folds(domain_frame(domains=range(6)), seed=42)
    for fold in folds:
        assert fold.held_out_domain not in set(fold.train.domain_id)
        assert fold.held_out_domain not in set(fold.validation.domain_id)
        assert set(fold.test.domain_id) == {fold.held_out_domain}
```

- [ ] **Step 2: Verify experiment tests fail**

Run: `uv run pytest tests/test_ablations.py tests/test_loco.py -q`

Expected: FAIL because experiment modules do not exist.

- [ ] **Step 3: Implement exact ablations**

`no_roi` trains/evaluates the global branch only and excludes fused loss;
`no_adversarial` removes domain execution and sets domain contribution to zero;
`no_dynamic_routing` trains the full model and forces every inference sample
through the fused branch. Persist the resolved semantic difference in each run
artifact, not only the YAML filename.

- [ ] **Step 4: Implement independent resumable LOCO jobs**

For each domain ID, hold every image in that cluster as test data. Stratify the
remaining images by class into train/validation where possible; if a class has
fewer than two remaining images, use seeded non-stratified sampling and record
the exception. Train baseline and C-DIRA from identical pretrained weights.
Use `<artifacts>/loco/domain-<id>/<model>/complete.json` as an atomic completion
marker containing fingerprints.

Rank clusters by sample count and divide them into deterministic small, middle,
and large thirds with `numpy.array_split`. Aggregate per-fold accuracy/macro F1
and unweighted group means; never weight away small-domain behavior.

- [ ] **Step 5: Run experiment tests**

Run: `uv run pytest tests/test_ablations.py tests/test_loco.py -q`

Expected: PASS, including resuming only incomplete folds.

- [ ] **Step 6: Commit**

```bash
git add src/cdira/experiments configs/ablations configs/loco.yaml tests/test_ablations.py tests/test_loco.py
git commit -m "feat: add ablation and LOCO experiment runners"
```

---

### Task 9: CLI, Artifacts, Reporting, and End-to-End Smoke Run

**Files:**
- Create: `src/cdira/reporting/__init__.py`
- Create: `src/cdira/reporting/report.py`
- Create: `src/cdira/artifacts.py`
- Create: `src/cdira/cli.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_report.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_smoke_pipeline.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all prior task interfaces
- Produces: `RunArtifacts.create(config: ExperimentConfig) -> RunArtifacts`
- Produces: `build_reproduction_report(run_root: Path) -> Path`
- Produces CLI commands: `data download`, `data prepare`, `domains fit`, `train`, `evaluate`, `run-ablation`, `run-loco`, `report`, `run-paper`

- [ ] **Step 1: Write failing CLI, artifact, and report tests**

```python
def test_run_artifacts_have_required_layout(tmp_path: Path) -> None:
    run = RunArtifacts.create(fake_config(tmp_path))
    assert {p.name for p in run.root.iterdir()} >= {
        "logs", "checkpoints", "predictions", "metrics", "plots"
    }
    assert (run.root / "config.resolved.yaml").exists()
    assert (run.root / "environment.json").exists()
    assert (run.root / "fingerprints.json").exists()

def test_report_distinguishes_measured_and_paper_values(tmp_path: Path) -> None:
    seed_complete_artifacts(tmp_path)
    report = build_reproduction_report(tmp_path)
    text = report.read_text()
    assert "Locally measured" in text
    assert "Paper-reported reference" in text
    assert "M4 Pro" in text
    assert "H100 latency was not reproduced" in text
```

Use Typer's `CliRunner` to assert malformed config, missing credentials, stale
fingerprints, and incomplete upstream stages exit nonzero with recovery
commands.

- [ ] **Step 2: Verify orchestration tests fail**

Run: `uv run pytest tests/test_artifacts.py tests/test_report.py tests/test_cli.py tests/test_smoke_pipeline.py -q`

Expected: FAIL because artifact, report, and CLI modules do not exist.

- [ ] **Step 3: Implement immutable run artifacts and report**

Create `artifacts/<UTC timestamp>-<config hash>/` atomically with resolved YAML,
environment JSON, fingerprints JSON, and required subdirectories. Refuse to
reuse an existing run ID except through explicit resume.

The report reads only machine-readable artifacts and emits:

- dataset/split/subject-overlap summary;
- selected K and silhouette curve;
- locally measured baseline/full/ablation metrics;
- paper-reported reference metrics marked by source;
- routing curves for both policies and comparison with 0.022 ROI usage;
- parameter/FLOP/M4 latency tables;
- corruption curves and ROI figures;
- LOCO fold/group tables;
- all declared assumptions and deviations;
- pass/divergence statement for the 0.010 macro-F1 target.

- [ ] **Step 4: Implement CLI and full paper orchestration**

Use Typer sub-apps. Each command accepts `--config` and repeated `--set
key=value`. `run-paper` executes and fingerprints these stages:

```text
download -> prepare -> domains -> baseline train/evaluate
-> full train/evaluate -> routing -> ablations
-> robustness -> efficiency -> visualization -> LOCO -> report
```

Skip a completed stage only when its completion marker matches every upstream
fingerprint. Print the next recovery command on failure. `data download` is the
only command that contacts Kaggle.

- [ ] **Step 5: Add synthetic end-to-end smoke test and README**

The smoke test creates ten tiny classes, fits two-domain synthetic features,
trains for two batches, saves/resumes, runs both routing policies, one
corruption, one LOCO fold, and generates `report.md`. Assert every required
artifact exists and contains finite metrics.

Document:

```bash
uv python install 3.12
uv sync --all-groups
uv run cdira data download --config configs/paper.yaml
uv run cdira run-paper --config configs/paper.yaml
uv run cdira report --run artifacts/<run-id>
```

Also document Kaggle competition acceptance, external credential setup, MPS
limitations, estimated long duration of 30 LOCO folds, resuming, smoke/standard
profiles, the driver-overlap audit, horizontal-flip diagnostic, and both
routing interpretations.

- [ ] **Step 6: Run complete verification**

Run:

```bash
uv run pytest -q
uv run pytest tests/test_smoke_pipeline.py -q
uv run ruff check src tests
uv run mypy src/cdira
uv run cdira --help
uv run cdira run-paper --config configs/smoke.yaml
```

Expected: all tests PASS, static checks exit 0, CLI help lists every command,
and the smoke run creates a complete report without Kaggle access.

- [ ] **Step 7: Commit**

```bash
git add src/cdira/reporting src/cdira/artifacts.py src/cdira/cli.py tests README.md
git commit -m "feat: complete reproducible C-DIRA experiment pipeline"
```

---

## Full-Data Execution Checkpoints

These are execution checkpoints after implementation, not additional coding
tasks:

1. Run `uv run cdira data download --config configs/paper.yaml`; verify ten
   classes and record the dataset fingerprint.
2. Run `uv run cdira data prepare --config configs/paper.yaml`; inspect class
   counts and the subject-overlap audit.
3. Run `uv run cdira domains fit --config configs/paper.yaml`; inspect the
   silhouette curve and record whether K=30 is selected.
4. Run the MobileNet baseline; require finite validation/test metrics before
   spending compute on C-DIRA.
5. Run full C-DIRA; inspect loss terms, routing target balance, and non-finite
   guards after the first epoch.
6. Run threshold, ablation, corruption, efficiency, and visualization stages.
7. Run LOCO folds as resumable jobs; verify each completion fingerprint.
8. Generate the final report and compare macro F1 with 0.992, ROI use with
   0.022, parameters with 2.165M, and locally measured M4 latency only with
   other local measurements.
