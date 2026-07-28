from pathlib import Path

import torch
from PIL import Image

from cdira.evaluation.predict import PredictionTable
from cdira.evaluation.visualize import save_roi_figure


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
