from pathlib import Path

import pytest

from cdira.data.download import DatasetLayoutError, validate_dataset
from tests.data_factory import make_state_farm_fixture


def test_validate_dataset_rejects_missing_class(tmp_path: Path) -> None:
    root = make_state_farm_fixture(tmp_path, images_per_class=1, subjects=2)
    (root / "train" / "c9" / "img_9_0.jpg").unlink()
    (root / "train" / "c9").rmdir()
    with pytest.raises(DatasetLayoutError, match="Missing"):
        validate_dataset(root)
