from pathlib import Path

from cdira.config import load_config


def test_paper_config_resolves_fixed_reproduction_values() -> None:
    cfg = load_config(Path("configs/paper.yaml"))
    assert cfg.seed == 42
    assert cfg.data.image_size == 224
    assert cfg.model.top_k == 5
    assert cfg.model.routing_hidden == 128
    assert cfg.training.batch_size == 32
    assert cfg.training.max_epochs == 50
    assert cfg.routing.primary_policy == "head"


def test_override_is_validated() -> None:
    cfg = load_config(Path("configs/paper.yaml"), ["training.batch_size=8"])
    assert cfg.training.batch_size == 8
