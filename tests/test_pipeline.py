import numpy as np

from cdira.domains.clustering import PseudoDomains
from cdira.pipeline import domain_mapping


def test_domain_mapping_is_path_indexed() -> None:
    domains = PseudoDomains(
        k=2,
        centroids=np.zeros((2, 3), dtype=np.float32),
        labels_by_path={"a.jpg": 1, "b.jpg": 0},
        silhouette_scores={2: 0.5},
        fit_paths=["a.jpg"],
    )
    assert domain_mapping(domains) == {"a.jpg": 1, "b.jpg": 0}
