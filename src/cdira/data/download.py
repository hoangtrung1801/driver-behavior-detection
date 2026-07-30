from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


class DatasetLayoutError(ValueError):
    """Raised when the extracted competition data is incomplete or malformed."""


@dataclass(frozen=True)
class DatasetFingerprint:
    root: Path
    sha256: str
    image_count: int


def validate_dataset(root: Path) -> DatasetFingerprint:
    records: list[tuple[str, int]] = []
    for class_id in range(10):
        folder = root / "train" / f"c{class_id}"
        if not folder.is_dir():
            raise DatasetLayoutError(f"Missing {folder}; rerun `cdira data download`")
        images = sorted(folder.glob("*.jpg"))
        records.extend(
            (path.relative_to(root).as_posix(), path.stat().st_size) for path in images
        )
    if not records:
        raise DatasetLayoutError("No training JPEGs found")
    payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return DatasetFingerprint(root=root, sha256=hashlib.sha256(payload).hexdigest(), image_count=len(records))


def download_competition(destination: Path) -> DatasetFingerprint:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install the kaggle package before downloading data") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cdira-kaggle-") as temporary:
        staging = Path(temporary)
        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            raise RuntimeError(
                "Kaggle authentication failed; configure credentials and accept the competition rules"
            ) from exc
        api.competition_download_files(
            "state-farm-distracted-driver-detection", path=staging.as_posix()
        )
        archives = sorted(staging.glob("*.zip"))
        if not archives:
            raise DatasetLayoutError("Kaggle returned no archive")
        extracted = staging / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(extracted)
        candidate = extracted / "imgs"
        if not (candidate / "train").is_dir():
            candidate = extracted
        fingerprint = validate_dataset(candidate)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(candidate, destination)
        return DatasetFingerprint(destination, fingerprint.sha256, fingerprint.image_count)
