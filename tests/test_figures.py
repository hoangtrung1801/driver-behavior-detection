import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from cdira.reporting import figures

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
    fig = figures.augmentation_figure(
        original, torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


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


def test_architecture_schematic_returns_figure_with_labels() -> None:
    fig = figures.architecture_schematic_figure()
    assert isinstance(fig, Figure)
    texts = {text.get_text() for text in fig.axes[0].texts}
    assert any("backbone" in text.lower() for text in texts)
    assert any("routing" in text.lower() for text in texts)


def test_parameter_table_lists_modules_and_total() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    frame = figures.parameter_table(model)
    assert list(frame.columns) == ["module", "parameters"]
    assert frame["module"].iloc[-1] == "total"
    total = int(frame["parameters"].iloc[-1])
    assert total == sum(p.numel() for p in model.parameters())


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
