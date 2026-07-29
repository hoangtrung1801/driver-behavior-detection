from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from PIL import Image

from cdira.evaluation.predict import PredictionTable


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
