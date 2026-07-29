from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast


class ColabSetupError(RuntimeError):
    """Raised when the Colab environment or credentials are not usable."""


def ensure_repository_root(root: Path | None = None) -> Path:
    candidate = (root or Path.cwd()).resolve()
    pyproject = candidate / "pyproject.toml"
    package = candidate / "src" / "cdira" / "__init__.py"
    if not pyproject.is_file() or not package.is_file():
        raise ColabSetupError(
            f"Not at the C-DIRA repository root: {candidate} "
            "(expected pyproject.toml and src/cdira/__init__.py)."
        )
    if "cdira-reproduction" not in pyproject.read_text(encoding="utf-8"):
        raise ColabSetupError(
            f"pyproject.toml at {candidate} is not the cdira-reproduction project."
        )
    return candidate


def ensure_supported_python(version: tuple[int, int] | None = None) -> None:
    major, minor = version or (sys.version_info.major, sys.version_info.minor)
    if (major, minor) != (3, 12):
        raise ColabSetupError(
            f"Python 3.12 is required; found {major}.{minor}. "
            "Select a Python 3.12 Colab runtime."
        )


def ensure_cuda_available() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise ColabSetupError(
            "This notebook requires a CUDA GPU runtime. In Colab choose "
            "Runtime > Change runtime type > GPU, then rerun."
        )
    return torch.device("cuda")


def _validate_credentials_file(path: Path) -> None:
    if not path.is_file():
        raise ColabSetupError(f"Credential file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ColabSetupError("kaggle.json is not valid JSON.") from exc
    if (
        not isinstance(payload, dict)
        or "username" not in payload
        or "key" not in payload
    ):
        raise ColabSetupError("kaggle.json must contain 'username' and 'key'.")


def _colab_upload() -> Mapping[str, bytes]:
    from google.colab import files

    return cast(Mapping[str, bytes], files.upload())


def install_kaggle_credentials(
    destination: Path = Path("/root/.kaggle/kaggle.json"),
    upload_fn: Callable[[], Mapping[str, bytes]] | None = None,
) -> Path:
    if destination.is_file():
        _validate_credentials_file(destination)
        return destination
    uploaded = (upload_fn or _colab_upload)()
    names = list(uploaded)
    if names != ["kaggle.json"]:
        received = ", ".join(names) or "nothing"
        raise ColabSetupError(
            f"Upload exactly one file named kaggle.json; received: {received}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded["kaggle.json"])
    destination.chmod(0o600)
    _validate_credentials_file(destination)
    return destination


def translate_kaggle_error(error: Exception) -> str:
    message = str(error).lower()
    if any(
        token in message for token in ("401", "403", "forbidden", "authenticate")
    ):
        return (
            "Kaggle denied access. Confirm that:\n"
            "  1. kaggle.json holds valid credentials for your account;\n"
            "  2. you have signed in to Kaggle and accepted the State Farm "
            "Distracted Driver Detection competition rules; then\n"
            "  3. rerun this cell."
        )
    return f"Dataset download failed: {error}"
