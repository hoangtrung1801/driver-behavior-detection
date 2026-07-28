from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class FeatureCache:
    paths: list[str]
    features: np.ndarray
    metadata: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, paths=np.asarray(self.paths), features=self.features)
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )


def load_feature_cache(
    path: Path, expected_transform_sha256: str | None = None
) -> FeatureCache:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Feature cache is incomplete: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        expected_transform_sha256 is not None
        and metadata.get("transform_sha256") != expected_transform_sha256
    ):
        raise ValueError(
            "Feature cache transform fingerprint does not match expected fingerprint"
        )
    with np.load(path, allow_pickle=False) as payload:
        paths = [str(value) for value in payload["paths"].tolist()]
        features = np.asarray(payload["features"], dtype=np.float32)
    if len(paths) != len(features):
        raise ValueError("Feature cache path and feature lengths differ")
    return FeatureCache(paths, features, metadata)


def extract_feature_cache(
    loader: Any, device: torch.device, destination: Path
) -> FeatureCache:
    backbone = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ).features
    backbone.eval().to(device)
    vectors: list[np.ndarray] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in loader:
            features = backbone(batch["image"].to(device))
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
            vectors.append(pooled.detach().to("cpu").numpy().astype(np.float32))
            paths.extend(str(value) for value in batch["relative_path"])
    cache = FeatureCache(
        paths=paths,
        features=np.concatenate(vectors, axis=0),
        metadata={
            "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            "feature_dim": 576,
        },
    )
    cache.save(destination)
    return cache
