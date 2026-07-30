from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Mapping

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(training: bool, image_size: int, horizontal_flip: bool, brightness: float = 0.2) -> transforms.Compose:
    steps: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
    if training and horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if training and brightness > 0:
        steps.append(transforms.ColorJitter(brightness=brightness))
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


class StateFarmDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: Path,
        dataset_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        domains: Mapping[str, int] | None = None,
    ) -> None:
        self.frame = pd.read_csv(manifest)
        self.dataset_root = dataset_root
        self.transform = transform
        self.domains = domains or {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        relative_path = str(row.relative_path)
        with Image.open(self.dataset_root / relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "target": torch.tensor(int(row.class_id), dtype=torch.long),
            "domain": torch.tensor(self.domains.get(relative_path, -1), dtype=torch.long),
            "relative_path": relative_path,
            "subject": str(row.subject),
        }
