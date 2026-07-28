import numpy as np
import pytest
from PIL import Image

from cdira.evaluation.corruptions import apply_corruption


@pytest.mark.parametrize(
    ("kind", "severity"),
    [("blur", 2), ("jpeg", 25), ("low_light", 0.25), ("occlusion", 0.3)],
)
def test_corruptions_are_deterministic(kind: str, severity: float) -> None:
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    first = np.asarray(apply_corruption(image, kind, severity))
    second = np.asarray(apply_corruption(image, kind, severity))
    np.testing.assert_array_equal(first, second)
