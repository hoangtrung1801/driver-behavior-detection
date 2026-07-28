from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max((os.cpu_count() or 1) - 1, 1))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cdira.config import DomainConfig
from cdira.domains.features import FeatureCache


@dataclass(frozen=True)
class PseudoDomains:
    k: int
    centroids: np.ndarray
    labels_by_path: dict[str, int]
    silhouette_scores: dict[int, float]
    fit_paths: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, centroids=self.centroids)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "k": self.k,
                    "labels_by_path": self.labels_by_path,
                    "silhouette_scores": self.silhouette_scores,
                    "fit_paths": self.fit_paths,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def fit_pseudo_domains(
    train: FeatureCache,
    validation: FeatureCache,
    test: FeatureCache,
    config: DomainConfig,
    seed: int,
) -> PseudoDomains:
    if not train.paths:
        raise ValueError("Cannot cluster an empty training feature cache")
    sample_count = min(config.sample_size, len(train.features))
    indices = np.random.default_rng(seed).choice(
        len(train.features), sample_count, replace=False
    )
    scores: dict[int, float] = {}
    for k in config.candidates:
        if k >= sample_count:
            raise ValueError(
                f"Cluster count {k} must be less than sample count {sample_count}"
            )
        candidate = KMeans(n_clusters=k, random_state=seed, n_init=config.n_init)
        labels = candidate.fit_predict(train.features[indices])
        scores[k] = float(silhouette_score(train.features[indices], labels, n_jobs=1))
    best_score = max(scores.values())
    best_k = min(k for k, score in scores.items() if score == best_score)
    final = KMeans(n_clusters=best_k, random_state=seed, n_init=config.n_init)
    train_labels = final.fit_predict(train.features)
    validation_labels = final.predict(validation.features)
    test_labels = final.predict(test.features)
    labels_by_path = {
        **dict(zip(train.paths, map(int, train_labels), strict=True)),
        **dict(zip(validation.paths, map(int, validation_labels), strict=True)),
        **dict(zip(test.paths, map(int, test_labels), strict=True)),
    }
    return PseudoDomains(
        k=best_k,
        centroids=np.asarray(final.cluster_centers_, dtype=np.float32),
        labels_by_path=labels_by_path,
        silhouette_scores=scores,
        fit_paths=list(train.paths),
    )
