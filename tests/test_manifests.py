from pathlib import Path

from cdira.data.manifests import build_split_manifests, validate_split_bundle
from tests.data_factory import make_state_farm_fixture


def test_split_is_stratified_deterministic_and_disjoint(tmp_path: Path) -> None:
    root = make_state_farm_fixture(
        tmp_path / "dataset", images_per_class=20, subjects=5
    )
    first = build_split_manifests(root, tmp_path / "a", seed=42)
    second = build_split_manifests(root, tmp_path / "b", seed=42)
    assert first.fingerprint == second.fingerprint
    assert set(first.train.relative_path).isdisjoint(first.validation.relative_path)
    assert set(first.train.relative_path).isdisjoint(first.test.relative_path)
    assert first.train.groupby("class_id").size().tolist() == [16] * 10
    validate_split_bundle(first, root)


def test_subject_disjoint_diagnostic_has_no_driver_overlap(tmp_path: Path) -> None:
    root = make_state_farm_fixture(
        tmp_path / "dataset", images_per_class=20, subjects=10
    )
    from cdira.data.manifests import build_subject_disjoint_manifests

    bundle = build_subject_disjoint_manifests(root, tmp_path / "subjects", seed=42)
    assert set(bundle.train.subject).isdisjoint(bundle.validation.subject)
    assert set(bundle.train.subject).isdisjoint(bundle.test.subject)
