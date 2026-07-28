from cdira.experiments.ablations import required_ablations


def test_required_ablations_match_paper() -> None:
    specs = {spec.name: spec for spec in required_ablations()}
    assert set(specs) == {"full", "no_roi", "no_adversarial", "no_dynamic_routing"}
    assert specs["no_adversarial"].domain_loss_weight == 0.0
    assert specs["no_dynamic_routing"].routing_policy == "always"
