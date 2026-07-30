from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cdira.data.download import validate_dataset


@dataclass(frozen=True)
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    fingerprint: str
    train_path: Path
    validation_path: Path
    test_path: Path


def _inventory(root: Path) -> pd.DataFrame:
    subject_map: dict[tuple[int, str], str] = {}
    subject_csv = root / "driver_imgs_list.csv"
    if subject_csv.exists():
        raw = pd.read_csv(subject_csv)
        subject_map = {
            (int(row.classname[1:]), row.img): str(row.subject)
            for row in raw.itertuples(index=False)
        }
    rows = []
    for class_id in range(10):
        for path in sorted((root / "train" / f"c{class_id}").glob("*.jpg")):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "class_id": class_id,
                    "class_name": f"c{class_id}",
                    "subject": subject_map.get((class_id, path.name), "unknown"),
                }
            )
    return pd.DataFrame(rows)


def _bundle(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: Path, dataset_sha: str) -> SplitBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("train", "validation", "test")
    paths: list[Path] = []
    normalized: list[pd.DataFrame] = []
    for name, frame in zip(names, frames, strict=True):
        result = frame.copy().sort_values("relative_path").reset_index(drop=True)
        result["split"] = name
        result["dataset_sha256"] = dataset_sha
        path = output_dir / f"{name}.csv"
        result.to_csv(path, index=False)
        paths.append(path)
        normalized.append(result)
    payload = pd.concat(normalized, ignore_index=True).to_csv(index=False).encode()
    fingerprint = hashlib.sha256(payload).hexdigest()
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps({"dataset_sha256": dataset_sha, "split_sha256": fingerprint}, indent=2),
        encoding="utf-8",
    )
    return SplitBundle(normalized[0], normalized[1], normalized[2], fingerprint, *paths)


def build_split_manifests(dataset_root: Path, output_dir: Path, seed: int) -> SplitBundle:
    dataset_fingerprint = validate_dataset(dataset_root)
    frame = _inventory(dataset_root)
    train, remainder = train_test_split(
        frame, test_size=0.2, stratify=frame["class_id"], random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, stratify=remainder["class_id"], random_state=seed
    )
    bundle = _bundle((train, validation, test), output_dir, dataset_fingerprint.sha256)
    overlap = {
        "train_validation": sorted(set(bundle.train.subject) & set(bundle.validation.subject)),
        "train_test": sorted(set(bundle.train.subject) & set(bundle.test.subject)),
        "validation_test": sorted(set(bundle.validation.subject) & set(bundle.test.subject)),
    }
    (output_dir / "subject_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    return bundle


def build_subject_disjoint_manifests(dataset_root: Path, output_dir: Path, seed: int) -> SplitBundle:
    dataset_fingerprint = validate_dataset(dataset_root)
    frame = _inventory(dataset_root)
    subjects = np.array(sorted(set(frame["subject"])), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    train_end = int(len(subjects) * 0.8)
    validation_end = train_end + max(1, int(len(subjects) * 0.1))
    assignments = {
        subject: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, subject in enumerate(subjects)
    }
    frame = frame.assign(split=frame["subject"].map(assignments))
    return _bundle(
        tuple(frame.loc[frame.split == name].drop(columns="split") for name in ("train", "validation", "test")),
        output_dir,
        dataset_fingerprint.sha256,
    )


def validate_split_bundle(bundle: SplitBundle, dataset_root: Path) -> None:
    paths = [set(bundle.train.relative_path), set(bundle.validation.relative_path), set(bundle.test.relative_path)]
    if paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2]:
        raise ValueError("Split manifests contain overlapping image paths")
    dataset_sha = validate_dataset(dataset_root).sha256
    for frame in (bundle.train, bundle.validation, bundle.test):
        if set(frame.dataset_sha256) != {dataset_sha}:
            raise ValueError("Manifest dataset fingerprint does not match dataset")
