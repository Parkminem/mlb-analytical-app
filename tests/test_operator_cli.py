from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import analysis.operator.cli as cli_module
from analysis.operator.bundles import DEFAULT_BUNDLE_ROOT, ManagedBundleCopy
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    ConfirmedBundle,
    ProjectedBundle,
    ReviewBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from analysis.operator.service import DEFAULT_LEDGER_PATH, DEFAULT_REPORT_DIR
from tests.test_operator_schemas import (
    _raw,
    confirmed_bundle_mapping,
    projected_bundle_mapping,
    review_bundle_mapping,
)


@dataclass(frozen=True)
class _Envelope:
    record_id: str


@dataclass(frozen=True)
class _Capture:
    capture_id: str


@dataclass(frozen=True)
class _Forecast:
    forecast_run_id: str


@dataclass(frozen=True)
class _LinkedRecord:
    envelope: _Envelope


@dataclass(frozen=True)
class _PregameResult:
    stage: str
    game_pk: int
    lineup_state: str
    capture: _Capture
    forecast: _Forecast
    link: _LinkedRecord
    managed_bundle: ManagedBundleCopy
    capture_replayed: bool
    forecast_replayed: bool
    link_replayed: bool


@dataclass(frozen=True)
class _ReviewResult:
    stage: str
    game_pk: int
    capture: _Capture
    forecast: _Forecast
    link: _LinkedRecord
    closing_observations: tuple[object, ...]
    official_result: _LinkedRecord
    reviews: tuple[object, ...]
    managed_bundle: ManagedBundleCopy
    closing_replay_count: int
    official_result_replayed: bool
    review_replay_count: int
    classification_counts: MappingProxyType
    calibration_json_path: Path
    calibration_markdown_path: Path
    calibration_generation_dir: Path
    calibration_current_generation_path: Path
    calibration_report_replayed: bool
    calibration_sample_size: int
    confirmed_game_count: int
    confirmed_decision_count: int
    confirmed_decision_counts: MappingProxyType


runner = CliRunner()


def _bundle(command: str) -> ProjectedBundle | ConfirmedBundle | ReviewBundle:
    if command == "projected":
        return parse_projected_bundle_json(_raw(projected_bundle_mapping()))
    if command == "confirmed":
        return parse_confirmed_bundle_json(_raw(confirmed_bundle_mapping()))
    return parse_review_bundle_json(_raw(review_bundle_mapping()))


def _pregame_result(command: str, *, replayed: bool) -> _PregameResult:
    return _PregameResult(
        stage=command,
        game_pk=744_001,
        lineup_state=command,
        capture=_Capture(f"capture-{command}"),
        forecast=_Forecast(f"forecast-{command}"),
        link=_LinkedRecord(_Envelope(f"link-{command}")),
        managed_bundle=ManagedBundleCopy(
            path=Path(f"/managed/{command}.json"),
            bundle_hash=f"{command}-bundle-sha256",
            replayed=replayed,
        ),
        capture_replayed=replayed,
        forecast_replayed=replayed,
        link_replayed=replayed,
    )


def _review_result(*, replayed: bool = True) -> _ReviewResult:
    return _ReviewResult(
        stage="review",
        game_pk=744_001,
        capture=_Capture("capture-confirmed"),
        forecast=_Forecast("forecast-confirmed"),
        link=_LinkedRecord(_Envelope("link-confirmed")),
        closing_observations=(object(), object()),
        official_result=_LinkedRecord(_Envelope("result-final-1")),
        reviews=(object(), object(), object()),
        managed_bundle=ManagedBundleCopy(
            path=Path("/managed/review.json"),
            bundle_hash="review-bundle-sha256",
            replayed=replayed,
        ),
        closing_replay_count=2,
        official_result_replayed=replayed,
        review_replay_count=3,
        classification_counts=MappingProxyType(
            {"A": 1, "B": 0, "C": 1, "D": 0, "unclassified": 1}
        ),
        calibration_json_path=Path("/reports/calibration-latest.json"),
        calibration_markdown_path=Path("/reports/calibration-latest.md"),
        calibration_generation_dir=Path("/reports/generations/generation-7"),
        calibration_current_generation_path=Path("/reports/current"),
        calibration_report_replayed=replayed,
        calibration_sample_size=17,
        confirmed_game_count=5,
        confirmed_decision_count=9,
        confirmed_decision_counts=MappingProxyType(
            {"full_total": 3, "ml": 4, "plus_1_5": 2}
        ),
    )


@pytest.mark.parametrize("command", ("projected", "confirmed", "review"))
def test_commands_use_default_paths(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    loaded_bundle = _bundle(command)
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: loaded_bundle)

    def fake_run(bundle, **kwargs):
        calls.append((bundle, kwargs))
        return _review_result() if command == "review" else _pregame_result(
            command,
            replayed=False,
        )

    monkeypatch.setattr(cli_module, f"run_{command}", fake_run)

    result = runner.invoke(
        cli_module.app,
        [command, "--bundle", "input.json"],
    )

    assert result.exit_code == 0, result.output
    expected = {
        "ledger_path": DEFAULT_LEDGER_PATH,
        "bundle_root": DEFAULT_BUNDLE_ROOT,
    }
    if command == "review":
        expected["report_dir"] = DEFAULT_REPORT_DIR
    assert calls == [(loaded_bundle, expected)]


@pytest.mark.parametrize("command", ("projected", "confirmed", "review"))
def test_commands_accept_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    loaded_bundle = _bundle(command)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: loaded_bundle)

    def fake_run(bundle, **kwargs):
        calls.append(kwargs)
        return _review_result() if command == "review" else _pregame_result(
            command,
            replayed=False,
        )

    monkeypatch.setattr(cli_module, f"run_{command}", fake_run)
    arguments = [
        command,
        "--bundle",
        "input.json",
        "--ledger",
        "custom-ledger.jsonl",
        "--bundle-root",
        "custom-bundles",
    ]
    if command == "review":
        arguments.extend(("--report-dir", "custom-reports"))

    result = runner.invoke(cli_module.app, arguments)

    assert result.exit_code == 0, result.output
    expected = {
        "ledger_path": Path("custom-ledger.jsonl"),
        "bundle_root": Path("custom-bundles"),
    }
    if command == "review":
        expected["report_dir"] = Path("custom-reports")
    assert calls == [expected]


@pytest.mark.parametrize(
    ("command", "replayed"),
    (("projected", False), ("confirmed", True)),
)
def test_pregame_summary_contains_stable_ids_paths_hashes_and_replay_state(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    replayed: bool,
) -> None:
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: _bundle(command))
    monkeypatch.setattr(
        cli_module,
        f"run_{command}",
        lambda bundle, **kwargs: _pregame_result(command, replayed=replayed),
    )

    result = runner.invoke(
        cli_module.app,
        [command, "--bundle", "input.json"],
    )

    expected_replay = str(replayed).lower()
    assert result.exit_code == 0, result.output
    for expected in (
        f"stage: {command}",
        "game_pk: 744001",
        f"lineup_state: {command}",
        f"capture_id: capture-{command}",
        f"forecast_run_id: forecast-{command}",
        f"capture_forecast_link_record_id: link-{command}",
        f"managed_bundle_path: /managed/{command}.json",
        f"managed_bundle_hash: {command}-bundle-sha256",
        f"managed_bundle_replayed: {expected_replay}",
        f"capture_replayed: {expected_replay}",
        f"forecast_replayed: {expected_replay}",
        f"link_replayed: {expected_replay}",
    ):
        assert expected in result.output


def test_review_summary_contains_all_frozen_report_and_migration_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: _bundle("review"))
    monkeypatch.setattr(
        cli_module,
        "run_review",
        lambda bundle, **kwargs: _review_result(),
    )

    result = runner.invoke(
        cli_module.app,
        ["review", "--bundle", "input.json"],
    )

    assert result.exit_code == 0, result.output
    for expected in (
        "stage: review",
        "game_pk: 744001",
        "capture_id: capture-confirmed",
        "forecast_run_id: forecast-confirmed",
        "capture_forecast_link_record_id: link-confirmed",
        "official_result_record_id: result-final-1",
        "closing_observation_count: 2",
        "closing_replay_count: 2",
        "official_result_replayed: true",
        "review_count: 3",
        "review_replay_count: 3",
        "classification.A: 1",
        "classification.B: 0",
        "classification.C: 1",
        "classification.D: 0",
        "classification.unclassified: 1",
        "managed_bundle_path: /managed/review.json",
        "managed_bundle_replayed: true",
        "calibration_sample_size: 17",
        "confirmed_game_count: 5",
        "confirmed_decision_count: 9",
        "confirmed_decision_count.full_total: 3",
        "confirmed_decision_count.ml: 4",
        "confirmed_decision_count.plus_1_5: 2",
        "calibration_json_path: /reports/calibration-latest.json",
        "calibration_markdown_path: /reports/calibration-latest.md",
        "calibration_generation_dir: /reports/generations/generation-7",
        "calibration_current_generation_path: /reports/current",
        "calibration_report_replayed: true",
    ):
        assert expected in result.output


@pytest.mark.parametrize(
    ("command", "loaded_bundle"),
    (
        ("projected", _bundle("confirmed")),
        ("confirmed", _bundle("projected")),
        ("review", _bundle("confirmed")),
    ),
)
def test_wrong_concrete_bundle_type_exits_four(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    loaded_bundle: object,
) -> None:
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: loaded_bundle)

    result = runner.invoke(
        cli_module.app,
        [command, "--bundle", "input.json"],
    )

    assert result.exit_code == 4
    assert "bundle stage/type" in result.output
    assert "Traceback" not in result.output


def test_wrong_stage_on_concrete_bundle_exits_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_stage = _bundle("projected").model_copy(update={"stage": "confirmed"})
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: wrong_stage)

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", "input.json"],
    )

    assert result.exit_code == 4
    assert "bundle stage/type" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("error", "exit_code"),
    (
        (BundleValidationError("invalid schema"), 3),
        (OperatorPreconditionError("wrong stage"), 4),
        (OperatorLedgerError("ledger conflict"), 5),
        (OperatorIOError("bundle write failed"), 6),
    ),
)
def test_operator_errors_have_stable_exit_codes_without_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
) -> None:
    def fail(path):
        raise error

    monkeypatch.setattr(cli_module, "load_bundle", fail)

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", "input.json"],
    )

    assert result.exit_code == exit_code
    assert str(error) in result.output
    assert "Traceback" not in result.output


def test_unexpected_exception_exits_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "load_bundle", lambda path: _bundle("projected"))

    def fail(bundle, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(cli_module, "run_projected", fail)

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", "input.json"],
    )

    assert result.exit_code == 1
    assert "internal error: injected failure" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "arguments",
    (
        ["projected"],
        ["projected", "--bundle", "input.json", "--unknown-option"],
        [],
    ),
)
def test_native_usage_errors_exit_two(arguments: list[str]) -> None:
    result = runner.invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_module_entry_point_help_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "analysis.operator", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "projected" in output
    assert "confirmed" in output
    assert "review" in output
