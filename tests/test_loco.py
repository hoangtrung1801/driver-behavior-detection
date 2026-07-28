import pandas as pd
import pytest

from cdira.experiments.loco import aggregate_loco, build_loco_folds


def domain_frame(domains: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relative_path": [f"{i}.jpg" for i in range(60)],
            "domain_id": [i % len(domains) for i in range(60)],
            "class_id": [i % 10 for i in range(60)],
        }
    )


def test_loco_fold_has_no_held_out_domain_in_train_or_validation() -> None:
    folds = build_loco_folds(domain_frame(range(6)), seed=42)
    for fold in folds:
        assert fold.held_out_domain not in set(fold.train.domain_id)
        assert fold.held_out_domain not in set(fold.validation.domain_id)
        assert set(fold.test.domain_id) == {fold.held_out_domain}


def test_loco_aggregation_reports_group_means() -> None:
    result = aggregate_loco(
        [
            {"domain_id": 0, "sample_count": 10, "accuracy": 0.8, "macro_f1": 0.7},
            {"domain_id": 1, "sample_count": 20, "accuracy": 0.9, "macro_f1": 0.8},
        ]
    )
    assert result["fold_count"] == 2
    assert result["mean_accuracy"] == pytest.approx(0.85)
