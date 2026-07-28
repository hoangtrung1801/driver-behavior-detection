import numpy as np

from cdira.config import DomainConfig
from cdira.domains.clustering import fit_pseudo_domains
from cdira.domains.features import FeatureCache


def separable_feature_cache(
    cluster_count: int, samples: int, prefix: str
) -> FeatureCache:
    points = []
    paths = []
    for index in range(samples):
        cluster = index % cluster_count
        point = np.zeros(4, dtype=np.float32)
        point[cluster % 4] = 10.0
        point += np.random.default_rng(index).normal(0, 0.01, size=4)
        points.append(point)
        paths.append(f"{prefix}-{index}.jpg")
    return FeatureCache(paths, np.asarray(points, dtype=np.float32), {"split": prefix})


def test_cluster_selection_uses_train_only_and_selects_separable_k() -> None:
    train = separable_feature_cache(cluster_count=3, samples=600, prefix="train")
    validation = separable_feature_cache(
        cluster_count=3, samples=60, prefix="validation"
    )
    test = separable_feature_cache(cluster_count=3, samples=60, prefix="test")
    result = fit_pseudo_domains(
        train,
        validation,
        test,
        DomainConfig(candidates=[2, 3, 4], sample_size=500, n_init=10),
        seed=42,
    )
    assert result.k == 3
    assert result.fit_paths == train.paths
    assert set(validation.paths).issubset(result.labels_by_path)
