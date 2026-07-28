from typer.testing import CliRunner

from cdira.cli import app


def test_cli_help_lists_pipeline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-paper" in result.stdout
    assert "data" in result.stdout


def test_run_paper_smoke_executes_offline(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-paper",
            "--config",
            "configs/smoke.yaml",
            "--artifact-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "smoke" / "report.md").exists()
