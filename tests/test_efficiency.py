from cdira.evaluation.efficiency import expected_conditional_flops


def test_expected_flops_interpolates_observed_roi_usage() -> None:
    assert (
        expected_conditional_flops(global_flops=60, roi_extra_flops=5, roi_usage=0.2)
        == 61
    )
