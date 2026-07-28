from pathlib import Path

import torch

from cdira.training.checkpoints import TrainingState, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_model_and_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainingState(model, optimizer, None, 4, 1.2, 2, {"seed": 42})
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state, {"dataset": "abc"})
    loaded = load_checkpoint(path, {"dataset": "abc"}, mode="resume")
    assert loaded.epoch == 4
    assert loaded.best_validation_loss == 1.2
    assert loaded.fingerprints == {"dataset": "abc"}
    for first, second in zip(
        model.parameters(), loaded.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first, second)
