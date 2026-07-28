from cdira.runtime import fingerprint_json


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint_json({"a": 1, "b": 2}) == fingerprint_json({"b": 2, "a": 1})
