import json
from pathlib import Path

import pytest

from cdira import colab

VALID = b'{"username": "u", "key": "secret-key-value"}'


def _make_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        'name = "cdira-reproduction"\n', encoding="utf-8"
    )
    package = root / "src" / "cdira"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")


def test_ensure_repository_root_accepts_valid_repo(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    assert colab.ensure_repository_root(tmp_path) == tmp_path.resolve()


def test_ensure_repository_root_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_repository_root(tmp_path)


def test_ensure_supported_python_accepts_312() -> None:
    colab.ensure_supported_python((3, 12))


@pytest.mark.parametrize("version", [(3, 11), (3, 13)])
def test_ensure_supported_python_rejects_others(version: tuple[int, int]) -> None:
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_supported_python(version)


def test_ensure_cuda_available_raises_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(colab.ColabSetupError):
        colab.ensure_cuda_available()


def test_ensure_cuda_available_returns_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert colab.ensure_cuda_available().type == "cuda"


def test_install_reuses_existing_valid_credentials(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    dest.write_bytes(VALID)

    def fail() -> dict[str, bytes]:
        raise AssertionError("upload_fn must not be called when creds exist")

    assert colab.install_kaggle_credentials(dest, upload_fn=fail) == dest


def test_install_writes_uploaded_credentials_with_0600(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "kaggle.json"
    result = colab.install_kaggle_credentials(
        dest, upload_fn=lambda: {"kaggle.json": VALID}
    )
    assert result == dest
    assert dest.is_file()
    assert (dest.stat().st_mode & 0o777) == 0o600
    assert json.loads(dest.read_text(encoding="utf-8"))["username"] == "u"


def test_install_does_not_print_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "kaggle.json"
    colab.install_kaggle_credentials(dest, upload_fn=lambda: {"kaggle.json": VALID})
    assert "secret-key-value" not in capsys.readouterr().out


def test_install_rejects_wrong_filename(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    with pytest.raises(colab.ColabSetupError):
        colab.install_kaggle_credentials(dest, upload_fn=lambda: {"creds.txt": VALID})


def test_install_rejects_malformed_json(tmp_path: Path) -> None:
    dest = tmp_path / "kaggle.json"
    with pytest.raises(colab.ColabSetupError):
        colab.install_kaggle_credentials(
            dest, upload_fn=lambda: {"kaggle.json": b"not json"}
        )


def test_translate_kaggle_error_forbidden_is_actionable() -> None:
    message = colab.translate_kaggle_error(RuntimeError("HTTP 403 Forbidden"))
    assert "competition rules" in message


def test_translate_kaggle_error_generic() -> None:
    message = colab.translate_kaggle_error(RuntimeError("disk full"))
    assert "disk full" in message
