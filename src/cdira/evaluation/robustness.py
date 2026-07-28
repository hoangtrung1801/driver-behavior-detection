from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cdira.evaluation.corruptions import apply_corruption


def evaluate_robustness(
    models: Mapping[str, Any], loader_factory: Any, config: Any
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    levels: dict[str, list[float]] = {
        "blur": [0, 1, 2, 3, 4],
        "jpeg": [100, 75, 50, 25, 10],
        "low_light": [1.0, 0.75, 0.5, 0.25, 0.1],
        "occlusion": [0.0, 0.1, 0.2, 0.3, 0.4],
    }
    for corruption, severities in levels.items():
        for severity in severities:
            loader = loader_factory(
                lambda image, kind=corruption, level=severity: apply_corruption(
                    image, kind, level
                )
            )
            for name, model in models.items():
                result = model.evaluate(loader)
                rows.append(
                    {
                        "model": name,
                        "corruption": corruption,
                        "severity": severity,
                        **result,
                    }
                )
    return pd.DataFrame(rows)
