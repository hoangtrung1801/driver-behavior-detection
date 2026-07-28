from __future__ import annotations

import hashlib
import json
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


def fingerprint_json(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def select_device(requested: str) -> Any:
    import torch

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but torch.backends.mps.is_available() is false"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return torch.device(requested)


@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=".partial-", dir=destination.parent)[1])
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_cpu_boundary(
    name: str,
    reason: str,
    tensors: Sequence[Any],
    operation: Callable[..., Any],
) -> Any:
    import logging
    import time

    import torch

    logger = logging.getLogger("cdira.runtime")
    started = time.perf_counter()
    cpu_tensors = [tensor.detach().to("cpu") if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
    result = operation(*cpu_tensors)
    logger.info(
        "CPU boundary name=%s reason=%s source=%s destination=cpu duration_ms=%.3f",
        name,
        reason,
        {str(getattr(tensor, "device", "unknown")) for tensor in tensors},
        (time.perf_counter() - started) * 1000,
    )
    return result
