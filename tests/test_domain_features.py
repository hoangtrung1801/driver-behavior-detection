from pathlib import Path

import numpy as np
import pytest

from cdira.domains.features import FeatureCache, load_feature_cache


def test_feature_cache_rejects_transform_fingerprint_mismatch(tmp_path: Path) -> None:
    cache = FeatureCache(
        paths=["one.jpg"],
        features=np.ones((1, 4), dtype=np.float32),
        metadata={"transform_sha256": "expected"},
    )
    path = tmp_path / "features.npz"
    cache.save(path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_feature_cache(path, expected_transform_sha256="wrong")
