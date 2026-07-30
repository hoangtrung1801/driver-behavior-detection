import numpy as np
import torch
from PIL import Image

from cdira.streamlit_app import (
    CLASS_NAMES,
    aggregate_frame_predictions,
    make_saliency_overlay,
)


def test_class_names_cover_all_driver_behavior_classes() -> None:
    assert list(CLASS_NAMES) == [f"c{index}" for index in range(10)]


def test_make_saliency_overlay_preserves_image_size() -> None:
    image = Image.new("RGB", (64, 48), color="white")
    saliency = torch.ones(3, 4)

    overlay = make_saliency_overlay(image, saliency)

    assert overlay.size == image.size


def test_aggregate_frame_predictions_returns_video_vote_and_confidence() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    result = aggregate_frame_predictions(probabilities)

    assert result["class_index"] == 0
    assert result["confidence"] == np.float32(0.53333336)
    assert result["frame_predictions"] == [0, 0, 1]
