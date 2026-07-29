from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.decomposition import PCA
from torch import Tensor, nn

from cdira.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: Tensor) -> np.ndarray:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    array = tensor.detach().cpu().numpy().astype(np.float32) * std + mean
    result = np.clip(array, 0.0, 1.0).transpose(1, 2, 0)
    return cast(np.ndarray, result)


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
    for axis in flat[len(images) :]:
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


def silhouette_figure(
    silhouette_scores: Mapping[int, float], selected_k: int
) -> Figure:
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
    projection = PCA(n_components=2).fit_transform(
        np.asarray(features, dtype=np.float32)
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(
        projection[:, 0],
        projection[:, 1],
        c=list(labels),
        cmap="tab10",
        s=8,
        alpha=0.6,
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
    for x, y, label in blocks:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                0.18,
                0.16,
                boxstyle="round,pad=0.02",
                facecolor="#eef3fb",
                edgecolor="#3b6ea5",
            )
        )
        axis.text(x + 0.09, y + 0.08, label, ha="center", va="center", fontsize=9)
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
        {
            "module": "total",
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    return pd.DataFrame(rows, columns=["module", "parameters"])


def draw_loss_curve(
    axis: Axes, history: Sequence[Mapping[str, float]]
) -> None:
    axis.clear()
    epochs = [record["epoch"] for record in history]
    axis.plot(
        epochs,
        [record["train_loss"] for record in history],
        marker="o",
        label="train",
    )
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
    per_class = cast(Mapping[str, float], metrics["per_class_roi_usage"])
    overall = float(cast(float, metrics["roi_usage"]))
    keys = sorted(per_class, key=int)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([f"c{key}" for key in keys], [float(per_class[key]) for key in keys])
    axis.axhline(
        overall, color="tab:red", linestyle="--", label=f"overall={overall:.2f}"
    )
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
        float(cast(float, cdira_metrics["accuracy"])),
        float(cast(Mapping[str, float], cdira_metrics["macro"])["f1"]),
    ]
    baseline_values = [
        float(cast(float, baseline_metrics["accuracy"])),
        float(cast(Mapping[str, float], baseline_metrics["macro"])["f1"]),
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
