import torch

from cdira.models.roi import topk_roi_pool


def test_topk_roi_pool_matches_hand_calculation() -> None:
    fmap = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    result = topk_roi_pool(fmap, k=2)
    assert result.indices.tolist() == [[1, 2]]
    assert result.pooled.tolist() == [[3.5]]
