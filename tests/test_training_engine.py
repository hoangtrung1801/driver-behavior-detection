import torch

from cdira.training.engine import Trainer


def test_trainer_type_is_constructible_with_minimal_configuration() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=1, patience=1)
    assert trainer.max_epochs == 1


def test_trainer_reports_batch_and_epoch_progress(capsys) -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=1, patience=1)
    batch = {
        "image": torch.ones(2, 3),
        "target": torch.tensor([0, 1]),
    }

    trainer.fit([batch, batch], [batch])

    output = capsys.readouterr().out
    assert "train progress: batch 2/2" in output
    assert "validation progress: batch 1/1" in output
    assert "epoch 1/1" in output
    assert "train_loss=" in output
    assert "val_loss=" in output


def test_fit_invokes_on_epoch_end_once_per_epoch() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=3, patience=5)
    batch = {"image": torch.ones(2, 3), "target": torch.tensor([0, 1])}

    lengths: list[int] = []
    result = trainer.fit(
        [batch], [batch], on_epoch_end=lambda history: lengths.append(len(history))
    )

    assert lengths == [1, 2, 3]
    assert result.epochs_completed == 3


def test_fit_without_callback_is_unchanged() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=2, patience=5)
    batch = {"image": torch.ones(2, 3), "target": torch.tensor([0, 1])}

    result = trainer.fit([batch], [batch])

    assert result.epochs_completed == 2
    assert len(result.history) == 2
