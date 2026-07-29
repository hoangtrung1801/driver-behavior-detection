from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
from matplotlib.figure import Figure
from PIL import Image

from cdira.evaluation.predict import PredictionTable
from cdira.evaluation.visualize import roi_overlay_figure, save_roi_figure


def test_roi_figure_is_written(tmp_path: Path) -> None:
    table = PredictionTable(
        paths=["sample.jpg"],
        targets=torch.tensor([0]),
        predictions=torch.tensor([1]),
        logits=torch.zeros(1, 2),
        global_logits=torch.zeros(1, 2),
        confidence=torch.tensor([0.8]),
        routing_probability=torch.tensor([0.9]),
        roi_mask=torch.tensor([True]),
        domains=torch.tensor([0]),
    )
    path = save_roi_figure(
        table,
        0,
        Image.new("RGB", (32, 32), "white"),
        torch.ones(7, 7),
        torch.tensor([1, 2, 3]),
        tmp_path / "roi.png",
    )
    assert path.exists()


def test_roi_overlay_figure_titles_and_returns_figure() -> None:
    fig = roi_overlay_figure(
        Image.new("RGB", (32, 32), "white"),
        torch.ones(7, 7),
        torch.tensor([1, 2, 3]),
        "true=0 pred=1",
    )
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "true=0 pred=1"
