from cdira.evaluation.predict import routing_policy_name
from cdira.models.cdira import RoutingPolicy


def test_routing_policy_name_is_stable() -> None:
    assert routing_policy_name(RoutingPolicy.HEAD) == "head"
