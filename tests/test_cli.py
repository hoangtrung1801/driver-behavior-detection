import json

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


def test_report_html_command_writes_destination(tmp_path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "config.resolved.yaml").write_text(
        "profile: standard\n", encoding="utf-8"
    )
    (tmp_path / "metrics" / "full.json").write_text(
        json.dumps({"macro": {"f1": 0.9}}), encoding="utf-8"
    )
    (tmp_path / "metrics" / "baseline.json").write_text(
        json.dumps({"macro": {"f1": 0.8}}), encoding="utf-8"
    )

    result = CliRunner().invoke(app, ["report-html", "--run", str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()
    assert "report.html" in result.stdout
