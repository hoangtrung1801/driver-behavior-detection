from pathlib import Path

from PIL import Image

from cdira.data.dataset import StateFarmDataset, build_transform
from cdira.data.manifests import build_split_manifests
from tests.data_factory import make_state_farm_fixture


def test_transform_has_expected_shape_and_normalization(tmp_path: Path) -> None:
    image = Image.new("RGB", (640, 480), "white")
    tensor = build_transform(False, 224, False)(image)
    assert tensor.shape == (3, 224, 224)


def test_dataset_returns_image_target_and_metadata(tmp_path: Path) -> None:
    root = make_state_farm_fixture(
        tmp_path / "dataset", images_per_class=10, subjects=2
    )
    bundle = build_split_manifests(root, tmp_path / "manifests", seed=42)
    dataset = StateFarmDataset(
        bundle.train_path, root, build_transform(False, 224, False)
    )
    item = dataset[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["target"].dtype.is_floating_point is False
    assert item["domain"].item() == -1
