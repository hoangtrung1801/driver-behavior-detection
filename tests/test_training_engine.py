import torch

from cdira.training.engine import Trainer


def test_trainer_type_is_constructible_with_minimal_configuration() -> None:
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model=model, device=torch.device("cpu"), max_epochs=1, patience=1)
    assert trainer.max_epochs == 1
