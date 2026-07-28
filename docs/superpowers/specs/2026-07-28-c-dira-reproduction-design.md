# C-DIRA Research Reproduction Design

## Purpose

This project will reproduce the core experiments from *C-DIRA: Computationally
Efficient Dynamic ROI Routing and Domain-Invariant Adversarial Learning for
Lightweight Driver Behavior Recognition* (arXiv:2512.08647v2) as an integrated
PyTorch research package.

The reproduction targets an Apple M4 Pro MacBook Pro with 24 GB of unified
memory. Training and inference will use the PyTorch MPS backend where supported,
with explicit CPU execution for unsupported operations. The work reproduces the
paper's static-image system and does not include webcam inference, temporal
video modeling, mobile packaging, or an edge-device application.

## Reproduction Goals

The project must:

1. Download and validate the State Farm Distracted Driver Detection dataset
   through the Kaggle API.
2. Create persistent, deterministic, stratified 80/10/10 train, validation, and
   test splits.
3. Generate pseudo-domain labels by clustering pretrained MobileNetV3-Small
   features.
4. Implement the complete C-DIRA model and all five terms of its training
   objective.
5. Train the full model as one integrated system.
6. Evaluate classification, routing, efficiency, ablations, robustness,
   saliency, and leave-one-cluster-out domain generalization.
7. Record every assumption required because the paper omits or contradicts an
   implementation detail.
8. Produce a machine-readable and human-readable reproduction report.

The core comparison model is torchvision's MobileNetV3-Small. Reimplementing
USH, SwiftFormer-S, MambaOut-Femt, and MobileViTv3-XS is outside the core scope
because the C-DIRA paper does not provide canonical implementations or enough
configuration detail to ensure fair reproduction. Their published values will
still appear as reference rows in the final report and will be clearly marked
as paper-reported rather than locally measured.

## Success Criteria

A successful `paper` run must:

- Complete dataset preparation, pseudo-domain labeling, training, and every
  evaluation stage without manual editing of generated files.
- Produce the complete artifact set described in this design.
- Achieve a full-model test macro F1 within 0.010 of the paper's reported 0.992.
- Measure and report the primary routing policy's ROI usage relative to the
  paper's reported 0.022 without treating an exact match as a pass condition.
- Complete the three required component ablations and both routing-policy
  evaluations.
- Complete all pseudo-domain LOCO folds, with each fold resumable independently.
- Report parameter count, theoretical compute, and M4 Pro latency without
  presenting Apple Silicon latency as a reproduction of the paper's H100
  result.

If the F1 target is not met, the run is still a valid research result when all
configuration, predictions, metrics, and diagnostics are preserved. The report
must describe the divergence instead of hiding or tuning around it.

## Project Shape

The codebase will remain a single integrated package while using focused
modules with explicit interfaces:

```text
driver-behavior-detection/
├── configs/                 # paper, standard, smoke, ablation, and LOCO YAML
├── src/cdira/
│   ├── data/                # download, validation, manifests, transforms
│   ├── domains/             # feature cache, K-means, LOCO manifests
│   ├── models/              # C-DIRA, GRL, ROI pooling, MobileNet baseline
│   ├── training/            # losses, trainer, checkpoints, device handling
│   ├── evaluation/          # metrics, routing, corruptions, efficiency, plots
│   ├── reporting/           # consolidated reproduction report
│   └── cli.py               # command-line entry points
├── tests/                   # unit, integration, MPS, and smoke tests
├── data/                    # ignored raw/derived local data
├── artifacts/               # ignored experiment outputs
└── pyproject.toml
```

"Monolithic reproduction" means the complete architecture and joint objective
are implemented together before the main training run. It does not mean placing
unrelated responsibilities in one source file.

## Dataset Acquisition and Validation

The downloader will use the Kaggle competition slug
`state-farm-distracted-driver-detection`. It will require the user to configure
Kaggle credentials outside the repository and accept the competition rules in
Kaggle. Credentials, archives, extracted images, checkpoints, and experiment
artifacts must be ignored by version control.

The data command will:

1. Check that Kaggle authentication is available.
2. Download the competition archive into a configurable cache directory.
3. Extract into a staging directory.
4. Validate the presence of `train/c0` through `train/c9`.
5. Validate that every discovered training image has exactly one class.
6. Record file counts, file sizes, and a content fingerprint.
7. Atomically promote validated data to the configured dataset directory.

The competition data includes a driver-to-image CSV even though the paper says
driver IDs are unavailable. To remain faithful to the paper, the primary split
will be image-level and stratified only by class. The preparation report will
also calculate subject overlap across splits when the subject CSV is present,
making the resulting leakage visible. Subject-disjoint splitting is a
diagnostic configuration, not the primary reproduction protocol.

## Persistent Splits

The default split seed is 42. Images are divided into 80% training, 10%
validation, and 10% test sets with class-stratified sampling. The split command
will persist one CSV per split with:

- relative image path;
- behavior class ID and name;
- optional subject ID;
- split name;
- source dataset fingerprint.

The validator will reject duplicate paths, cross-split overlap, missing files,
unexpected labels, or a manifest generated from a different dataset
fingerprint.

## Image Processing

All images are resized directly to 224 by 224 pixels and normalized with
ImageNet mean and standard deviation.

The paper training transform is:

- random horizontal flip with probability 0.5;
- brightness jitter with factor 0.2;
- conversion to tensor;
- ImageNet normalization.

Validation and test transforms contain no random augmentation. Horizontal
flipping is scientifically questionable because classes c1/c2 and c3/c4 encode
right-hand versus left-hand phone use. It remains enabled in the primary
paper-compatible profile because the paper explicitly reports random flipping.
A named `no_horizontal_flip` diagnostic configuration will quantify its effect.

## Pseudo-Domain Labeling

Pseudo-domain labeling is a preprocessing stage and its output remains fixed
during a training run.

1. Load an ImageNet-pretrained torchvision MobileNetV3-Small.
2. Use its convolutional `features` module and global average pooling to produce
   one 576-dimensional vector per image.
3. Cache vectors with image paths, preprocessing fingerprint, weight
   identifier, and dataset fingerprint.
4. Select exactly 5,000 training vectors with seed 42.
5. Fit K-means candidates for every integer cluster count from 2 through 40
   using seed 42 and `n_init=10`.
6. Select the candidate with the highest mean silhouette score. A lower cluster
   count wins an exact tie.
7. Refit K-means on all training vectors at the selected cluster count.
8. Assign validation and test samples using the fitted training centroids.
9. Persist centroids, labels, scores, sample counts, and fingerprints.

The paper reports 30 clusters. The implementation will report whether the
deterministic procedure also selects 30 but will not force that outcome. All
clustering operations execute on CPU through scikit-learn.

## C-DIRA Model

### Backbone

The shared feature extractor is the `features` module from ImageNet-pretrained
torchvision MobileNetV3-Small. A 224 by 224 input produces a feature map with
576 channels and a nominal spatial size of 7 by 7. The backbone is trainable
during C-DIRA training.

### Global path

Adaptive global average pooling produces `g` with shape `[B, 576]`. The global
classifier is:

```text
Linear(576, 256) -> ReLU -> Linear(256, 10)
```

It produces the global classification logits used by the global
cross-entropy loss and routing pseudo-label generation.

### Saliency and ROI path

The saliency score at each spatial position is the L2 norm across feature
channels. The five positions with the highest score are gathered and averaged,
producing an ROI vector with shape `[B, 576]`.

The ROI refinement network is:

```text
Linear(576, 512) -> ReLU
```

The refined ROI vector is concatenated with the unmodified global vector,
giving a 1,088-dimensional fused representation. The fused classifier is:

```text
Linear(1088, 512) -> ReLU -> Linear(512, 10)
```

Top-K is implemented with tensor gathering and averaging, not image cropping or
a second backbone pass.

### Routing head

The routing head is:

```text
Linear(576, 128) -> ReLU -> Linear(128, 1)
```

Its sigmoid output is the estimated probability that ROI processing is useful.
Routing logits remain available so binary cross entropy can use a numerically
stable logits formulation.

### Domain classifier and gradient reversal

The domain branch receives the global feature through a gradient reversal
operation with fixed strength 1.0:

```text
GRL(1.0) -> Linear(576, 256) -> ReLU -> Linear(256, selected_cluster_count)
```

The GRL is the identity in the forward pass and multiplies gradients flowing
to the backbone by `-1.0`. The domain classifier itself receives normal
cross-entropy gradients.

## Training Objective

Every training sample executes the global, ROI/fused, routing, and domain
branches. Dynamic short-circuiting is an inference optimization only.

The routing target is calculated from detached global predictions:

```text
requires_roi = global_prediction_is_wrong OR global_confidence < 0.9
```

Positive routing examples receive a per-batch weight equal to
`negative_count / positive_count` when both classes occur. The positive weight
is 1.0 when a batch contains only one routing class.

The total loss is:

```text
0.50 * global_cross_entropy
+ 1.00 * fused_cross_entropy
+ 0.50 * weighted_routing_bce_with_logits
+ 0.01 * mean_routing_probability
+ 0.50 * pseudo_domain_cross_entropy
```

The trainer logs every unweighted loss, every weighted contribution, the total
loss, routing target balance, routing probability, and global confidence.

## Inference Routing Policies

The paper contains an internal contradiction:

- Figure 3 routes using the routing-head probability.
- Algorithm 2 routes using global confidence and never uses the trained routing
  head.
- Figure 4 says ROI usage decreases as a threshold increases, which is
  consistent with thresholding routing probability but inconsistent with
  routing when global confidence falls below an increasing threshold.

The primary policy therefore routes a sample when:

```text
routing_probability >= routing_threshold
```

The default routing threshold is 0.9.

The required comparison policy follows Algorithm 2 and routes when:

```text
global_confidence < confidence_threshold
```

Its default confidence threshold is also 0.9. Both policies must use true
conditional execution: an all-easy batch must not calculate saliency, Top-K
features, or fused logits. Mixed batches calculate the ROI path only for
selected indices and scatter fused predictions back into the batch output.

## Training Profiles and Device Policy

The paper-compatible training settings are:

- AdamW optimizer;
- learning rate `1e-5`;
- maximum 50 epochs;
- early stopping patience 5 based on validation total loss;
- batch size 32 on the target M4 Pro;
- four data-loader workers;
- MPS autocast using float16 when runtime support is confirmed;
- seed 42 for Python, NumPy, PyTorch, splits, sampling, and clustering.

No scheduler is used because the paper does not specify one.

Three profiles control run length, not model semantics:

- `smoke`: synthetic or tiny real-data subsets, two batches, one epoch, and one
  LOCO fold.
- `standard`: full train/validation/test split with a reduced epoch count and a
  selected subset of robustness and LOCO evaluations.
- `paper`: full data, up to 50 epochs, all thresholds, all corruptions, all
  ablations, and all LOCO folds.

Device selection order is explicit: requested MPS, requested CUDA, then CPU.
An unsupported MPS operation may execute on CPU only through a named
compatibility boundary that logs the operation, source device, destination
device, duration, and reason. Silent global MPS fallback is not permitted.

## Checkpointing and Resume

Each checkpoint contains:

- model state;
- optimizer and mixed-precision state;
- current epoch and best validation value;
- early-stopping state;
- Python, NumPy, and PyTorch RNG states;
- resolved configuration;
- dataset, split, transform, feature-cache, and cluster fingerprints;
- package and platform versions.

Resume rejects incompatible fingerprints or model dimensions. A user may
explicitly start a new run from model weights, but that operation is distinct
from exact resume and does not restore optimizer or RNG state.

## Evaluation

### Classification

The evaluator reports accuracy and macro and weighted precision, recall, and
F1. Macro F1 is the primary F1 because the paper's routing-threshold discussion
names macro F1. It also saves per-class metrics, predictions, probabilities,
and confusion matrices.

### Ablations

Required ablations are:

1. no ROI feature: global classifier output only;
2. no adversarial learning: domain loss weight zero and no GRL branch;
3. no dynamic routing: fused classifier for every sample.

The full model and every ablation receive separate configurations, checkpoints,
and reports.

### Routing

For both inference policies, thresholds 0.1 through 0.9 in increments of 0.1
are evaluated from a fixed checkpoint. The report includes macro F1, accuracy,
ROI usage, per-class ROI usage, global-only latency, ROI-path latency, and
observed conditional latency.

### Efficiency

Efficiency reporting includes:

- trainable and total parameter counts;
- backbone, head, global-path, and ROI-path MACs/FLOPs;
- expected conditional FLOPs using observed ROI usage;
- peak process memory when measurable;
- batch-one MPS latency after 100 warm-up iterations over 1,000 timed
  iterations;
- median, p90, p95, and mean latency.

MPS is synchronized before and after each timed region. The report labels the
hardware and software environment and does not compare the absolute latency
directly to the paper's H100 value.

### ROI Visualization

For correctly and incorrectly classified examples from each class, the
evaluator saves the original image, normalized saliency heatmap, selected
Top-K positions, global prediction, routed prediction, confidence, and routing
probability. Visualization is derived from the same tensors used for inference.

### Robustness

Robustness evaluation applies deterministic corruptions to the unaugmented test
set:

- Gaussian blur radii: 0, 1, 2, 3, 4;
- JPEG quality: 100, 75, 50, 25, 10;
- low-light multipliers: 1.0, 0.75, 0.50, 0.25, 0.10;
- centered black-square occlusion covering area fractions: 0.0, 0.1, 0.2,
  0.3, 0.4.

The paper does not publish exact corruption definitions, so these values are
reproduction assumptions. Every corruption is applied before ImageNet
normalization. Accuracy and macro F1 are saved for the MobileNetV3-Small
baseline and full C-DIRA.

### Leave-One-Cluster-Out Evaluation

For each selected pseudo-domain cluster:

1. Exclude that cluster from training and validation data.
2. Use the excluded cluster as the unseen-domain test fold.
3. Build a deterministic validation split from the remaining clusters.
4. Train MobileNetV3-Small and C-DIRA from the same pretrained initialization.
5. Save accuracy, macro F1, fold size, class distribution, and training
   metadata.

Each fold is an independent resumable job. The aggregate report groups clusters
into small, middle, and large thirds by sample count using deterministic
rank-order boundaries and reports per-fold and group-level results.

## Error Handling

Commands fail before expensive work when they detect:

- missing Kaggle credentials or unaccepted competition access;
- partial archives or unexpected dataset structure;
- dataset and manifest fingerprint mismatches;
- train/validation/test overlap;
- feature-cache or cluster metadata created from different weights or
  transforms;
- checkpoints incompatible with the current number of classes or domains;
- non-finite losses or model parameters;
- an unavailable requested device;
- incomplete experiment artifacts required by a downstream stage.

Failures include the failing path or configuration key and a concrete recovery
command. Atomic writes are used for manifests, metadata, checkpoints, and
reports so interruption cannot make a partial file appear complete.

## Experiment Artifacts

Every run receives an immutable run ID and stores:

```text
artifacts/<run-id>/
├── config.resolved.yaml
├── environment.json
├── fingerprints.json
├── logs/
├── checkpoints/
├── predictions/
├── metrics/
├── plots/
└── report.md
```

Machine-readable metrics use JSON and CSV. The Markdown report links all plots,
states which values were locally measured or copied from the paper, lists every
reproduction assumption, and records deviations from the success criteria.

## Testing Strategy

Unit tests cover:

- split determinism, stratification, and overlap rejection;
- cache and manifest fingerprint validation;
- K-means fitting on training vectors only;
- global, ROI, fused, routing, and domain tensor shapes;
- Top-K pooling against hand-calculated feature maps;
- gradient reversal direction and magnitude;
- routing target generation and single-class batch weighting;
- each loss and the exact weighted loss sum;
- both routing policies and actual ROI branch short-circuiting;
- deterministic corruption severity;
- classification, routing, efficiency, and LOCO aggregation metrics.

Integration tests cover:

- a synthetic ten-class dataset through preparation, training, checkpointing,
  evaluation, and reporting;
- interrupted training followed by an equivalent resumed run;
- a smoke run on CPU;
- the same smoke run on MPS when MPS is available;
- command failures for missing credentials, stale fingerprints, and malformed
  checkpoints without requiring a real Kaggle download.

Tests requiring the competition dataset are marked separately and never run as
part of the default unit-test suite.

## Reproduction Risks and Declared Assumptions

The following paper details are absent or contradictory and are fixed by this
design:

| Detail | Reproduction choice |
|---|---|
| Top-K count | 5 positions |
| Routing hidden width | 128 |
| Domain hidden width | 256 |
| GRL strength | fixed 1.0 |
| Candidate cluster counts | every integer from 2 through 40 |
| K-means initialization | seed 42, `n_init=10` |
| Split seed | 42 |
| Brightness strength | 0.2 |
| Batch size | 32 on M4 Pro |
| Primary routing decision | `routing_probability >= threshold` |
| Algorithm 2 comparison | `global_confidence < threshold` |
| Metric averaging | macro primary, weighted also reported |
| Corruption definitions | fixed values in the robustness section |
| Latency procedure | batch one, 100 warm-up, 1,000 timed iterations |

The report will treat these as assumptions rather than claims about the
authors' unpublished implementation.
