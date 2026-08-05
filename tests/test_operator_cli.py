from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import analysis.operator.cli as cli_module
import analysis.operator.service as service_module
from analysis.forecasting import LineupState
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import capture_payload
from analysis.forecasting.validation import (
    PregameRelationshipError,
    PregameTimingError,
)
from analysis.ledger import LedgerStore
from analysis.operator.bundles import ManagedBundleCopy, ReplayOnlyPregameBundle
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    ConfirmedBundle,
    HistoricalConfirmedBundle,
    HistoricalProjectedBundle,
    ProjectedBundle,
    ReviewBundle,
    parse_confirmed_bundle_json,
    parse_historical_confirmed_bundle_json,
    parse_historical_projected_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from tests.test_forecasting_engine import _complete_game
from tests.test_operator_schemas import (
    _complete_legacy_decision_mapping,
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
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _bundle(command: str) -> ProjectedBundle | ConfirmedBundle | ReviewBundle:
    if command == "projected":
        return parse_projected_bundle_json(_raw(projected_bundle_mapping()))
    if command == "confirmed":
        return parse_confirmed_bundle_json(_raw(confirmed_bundle_mapping()))
    return parse_review_bundle_json(_raw(review_bundle_mapping()))


def _historical_pregame_bundle(
    command: str,
) -> HistoricalProjectedBundle | HistoricalConfirmedBundle:
    raw = _historical_pregame_raw(command)
    if command == "projected":
        return parse_historical_projected_bundle_json(raw)
    return parse_historical_confirmed_bundle_json(raw)


def _historical_pregame_raw(command: str) -> str:
    mapping = (
        projected_bundle_mapping()
        if command == "projected"
        else confirmed_bundle_mapping()
    )
    mapping["payload"]["legacy_decision"] = (
        _complete_legacy_decision_mapping()
    )
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    return _raw(mapping)


def _write_hash_named_managed_bundle(
    root: Path,
    raw: bytes,
    *,
    game_pk: int,
    stage: str,
) -> Path:
    bundle_hash = hashlib.sha256(raw).hexdigest()
    path = root / str(game_pk) / f"{stage}-v1-{bundle_hash[:16]}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return path


def test_pregame_loader_classifies_managed_pre_change_path_as_historical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "managed"
    managed_path = bundle_root / "900001" / "projected-v1-deadbeefdeadbeef.json"
    managed_path.parent.mkdir(parents=True)
    managed_path.write_text("historical", encoding="utf-8")
    expected = _historical_pregame_bundle("projected")
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        cli_module,
        "load_bundle",
        lambda path: (_ for _ in ()).throw(
            BundleValidationError("pre-change managed artifact")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_historical_managed_bundle",
        lambda path, *, root: calls.append((path, root))
        or ReplayOnlyPregameBundle(
            bundle=expected,
            managed_bundle=ManagedBundleCopy(
                path=managed_path,
                bundle_hash="historical-bundle-sha256",
                replayed=True,
            ),
        ),
        raising=False,
    )

    loaded = cli_module._load_pregame_bundle(
        managed_path,
        expected_type=ProjectedBundle,
        expected_stage="projected",
        bundle_root=bundle_root,
    )

    assert loaded.bundle is None
    assert loaded.historical_path == managed_path.resolve()
    assert calls == [(managed_path, bundle_root)]


@pytest.mark.parametrize("command", ("projected", "confirmed"))
def test_historical_managed_path_dispatches_only_to_replay_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    bundle_root = tmp_path / "managed"
    managed_path = bundle_root / "900001" / f"{command}-v1-deadbeefdeadbeef.json"
    managed_path.parent.mkdir(parents=True)
    managed_path.write_text("historical", encoding="utf-8")
    pregame_result = _pregame_result(command, replayed=True)
    calls: list[tuple[Path, Path, Path]] = []

    monkeypatch.setattr(
        cli_module,
        "load_bundle",
        lambda path: (_ for _ in ()).throw(
            BundleValidationError("pre-change managed artifact")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_historical_managed_bundle",
        lambda path, *, root: ReplayOnlyPregameBundle(
            bundle=_historical_pregame_bundle(command),
            managed_bundle=ManagedBundleCopy(
                path=managed_path,
                bundle_hash="historical-bundle-sha256",
                replayed=True,
            ),
        ),
    )

    def fake_historical(
        path: Path,
        *,
        ledger_path: Path,
        bundle_root: Path,
        registry_root: Path | None = None,
    ) -> _PregameResult:
        assert registry_root is None
        calls.append((path, ledger_path, bundle_root))
        return pregame_result

    def forbidden_normal(*args: object, **kwargs: object) -> _PregameResult:
        del args, kwargs
        raise AssertionError("historical input must not reach normal runner")

    monkeypatch.setattr(
        cli_module,
        f"replay_historical_{command}",
        fake_historical,
        raising=False,
    )
    monkeypatch.setattr(cli_module, f"run_{command}", forbidden_normal)

    ledger_path = tmp_path / "ledger.jsonl"
    result = runner.invoke(
        cli_module.app,
        [
            command,
            "--bundle",
            str(managed_path),
            "--ledger",
            str(ledger_path),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(managed_path.resolve(), ledger_path, bundle_root)]


@pytest.mark.parametrize(
    ("bundle", "historical_path"),
    (
        (None, None),
        (_bundle("projected"), Path("historical.json")),
    ),
)
def test_pregame_load_requires_exactly_one_dispatch_target(
    bundle: ProjectedBundle | None,
    historical_path: Path | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one dispatch target"):
        cli_module._PregameBundleLoad(
            bundle=bundle,
            historical_path=historical_path,
        )


def test_historical_pregame_outside_managed_root_is_not_classified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "managed"
    source = tmp_path / "outside" / "projected.json"
    source.parent.mkdir()
    source.write_text(_historical_pregame_raw("projected"), encoding="utf-8")

    def forbidden_historical_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("outside paths must not enter historical loading")

    monkeypatch.setattr(
        cli_module,
        "load_historical_managed_bundle",
        forbidden_historical_loader,
    )

    result = runner.invoke(
        cli_module.app,
        [
            "projected",
            "--bundle",
            str(source),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 3
    assert "Traceback" not in result.output


def test_strict_wrong_stage_under_managed_root_never_enters_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "managed"
    raw = _raw(confirmed_bundle_mapping()).encode("utf-8")
    source = _write_hash_named_managed_bundle(
        bundle_root,
        raw,
        game_pk=744_001,
        stage="projected",
    )

    def forbidden_historical_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("strict wrong-stage input must not enter replay")

    monkeypatch.setattr(
        cli_module,
        "load_historical_managed_bundle",
        forbidden_historical_loader,
    )

    result = runner.invoke(
        cli_module.app,
        [
            "projected",
            "--bundle",
            str(source),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 4
    assert "bundle stage/type mismatch" in result.output
    assert "Traceback" not in result.output


def test_malformed_managed_pregame_never_dispatches_a_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "managed"
    source = _write_hash_named_managed_bundle(
        bundle_root,
        b'{"stage":',
        game_pk=744_001,
        stage="projected",
    )

    def forbidden_service(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("malformed input must not dispatch a service")

    monkeypatch.setattr(cli_module, "run_projected", forbidden_service)
    monkeypatch.setattr(
        cli_module,
        "replay_historical_projected",
        forbidden_service,
    )

    result = runner.invoke(
        cli_module.app,
        [
            "projected",
            "--bundle",
            str(source),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 3
    assert "Traceback" not in result.output


def test_review_never_classifies_historical_pregame_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "managed"
    historical = _historical_pregame_bundle("projected")
    raw = _historical_pregame_raw("projected").encode("utf-8")
    source = _write_hash_named_managed_bundle(
        bundle_root,
        raw,
        game_pk=historical.game_pk,
        stage="projected",
    )

    def forbidden_historical_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("review must use strict loading only")

    def forbidden_review_runner(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("historical pregame input must not run review")

    monkeypatch.setattr(
        cli_module,
        "load_historical_managed_bundle",
        forbidden_historical_loader,
    )
    monkeypatch.setattr(cli_module, "run_review", forbidden_review_runner)

    result = runner.invoke(
        cli_module.app,
        [
            "review",
            "--bundle",
            str(source),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 3
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ("projected", "confirmed"))
def test_strict_current_path_dispatches_only_to_normal_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    bundle_root = tmp_path / "managed"
    current_path = bundle_root / "900001" / f"{command}-v1-deadbeefdeadbeef.json"
    current_path.parent.mkdir(parents=True)
    current_path.write_text("strict", encoding="utf-8")
    strict_bundle = _bundle(command)
    pregame_result = _pregame_result(command, replayed=False)
    calls: list[tuple[object, Path, Path]] = []

    monkeypatch.setattr(cli_module, "load_bundle", lambda path: strict_bundle)

    def forbidden_historical(*args: object, **kwargs: object) -> _PregameResult:
        del args, kwargs
        raise AssertionError("strict input must not reach historical replay")

    def fake_normal(
        bundle: object,
        *,
        ledger_path: Path,
        bundle_root: Path,
    ) -> _PregameResult:
        calls.append((bundle, ledger_path, bundle_root))
        return pregame_result

    monkeypatch.setattr(
        cli_module,
        f"replay_historical_{command}",
        forbidden_historical,
        raising=False,
    )
    monkeypatch.setattr(cli_module, f"run_{command}", fake_normal)

    ledger_path = tmp_path / "ledger.jsonl"
    result = runner.invoke(
        cli_module.app,
        [
            command,
            "--bundle",
            str(current_path),
            "--ledger",
            str(ledger_path),
            "--bundle-root",
            str(bundle_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(strict_bundle, ledger_path, bundle_root)]


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
        "ledger_path": (
            REPOSITORY_ROOT / "data" / "ledger" / "shadow.jsonl"
        ),
        "bundle_root": (
            REPOSITORY_ROOT / "data" / "shadow" / "bundles"
        ),
    }
    if command == "review":
        expected["report_dir"] = (
            REPOSITORY_ROOT / "reports" / "private" / "shadow"
        )
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
        raise RuntimeError(
            "api_token=top-secret https://private.example/token /private/path"
        )

    monkeypatch.setattr(cli_module, "run_projected", fail)

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", "input.json"],
    )

    assert result.exit_code == 1
    assert "internal error: RuntimeError" in result.output
    assert "top-secret" not in result.output
    assert "private.example" not in result.output
    assert "/private/path" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "attacker_key",
    (
        "https://attacker.example/private?api_token=url-secret",
        "api_token=field-name-secret",
    ),
)
def test_malicious_unknown_bundle_key_and_value_are_never_printed(
    tmp_path: Path,
    attacker_key: str,
) -> None:
    secret = "api-token-secret-value"
    mapping = projected_bundle_mapping()
    mapping["payload"][attacker_key] = secret
    bundle_path = tmp_path / "projected-secret.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert "payload.<field>" in result.output
    assert "extra_forbidden" in result.output
    assert attacker_key not in result.output
    assert "attacker.example" not in result.output
    assert "field-name-secret" not in result.output
    assert secret not in result.output
    assert "input_value" not in result.output
    assert "errors.pydantic.dev" not in result.output
    assert "Traceback" not in result.output


def test_custom_validator_diagnostic_never_prints_secret_key(
    tmp_path: Path,
) -> None:
    secret_key = "component-api-token-secret"
    mapping = review_bundle_mapping()
    mapping["payload"]["official_result"]["component_values"] = {
        secret_key: 1.0,
    }
    bundle_path = tmp_path / "review-secret-component.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        ["review", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert "payload.official_result" in result.output
    assert "invalid_value" in result.output
    assert secret_key not in result.output
    assert "Traceback" not in result.output


def test_validation_diagnostic_never_prints_secret_value(
    tmp_path: Path,
) -> None:
    secret_value = "component-value-api-token-secret"
    mapping = review_bundle_mapping()
    mapping["payload"]["official_result"]["component_values"] = {
        "home_pitcher_chain_runs_allowed": secret_value,
    }
    bundle_path = tmp_path / "review-secret-component-value.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        ["review", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert "payload.official_result.component_values" in result.output
    assert "invalid_number" in result.output
    assert secret_value not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "dynamic_key",
    (
        "https://attacker.example/?api_token=map-key-secret",
        "game_pk",
    ),
)
def test_dynamic_mapping_key_is_rendered_as_safe_placeholder(
    tmp_path: Path,
    dynamic_key: str,
) -> None:
    secret_value = "map-value-secret"
    mapping = review_bundle_mapping()
    mapping["payload"]["official_result"]["component_values"] = {
        dynamic_key: secret_value,
    }
    bundle_path = tmp_path / "review-secret-map-key.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        ["review", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert (
        "payload.official_result.component_values.<field>: invalid_number"
        in result.output
    )
    assert f"component_values.{dynamic_key}" not in result.output
    assert "attacker.example" not in result.output
    assert "map-key-secret" not in result.output
    assert secret_value not in result.output
    assert "Traceback" not in result.output


def test_malformed_nested_datetime_has_safe_actionable_diagnostic(
    tmp_path: Path,
) -> None:
    secret_datetime = "malformed-datetime-api-token-secret"
    mapping = review_bundle_mapping()
    mapping["payload"]["closing_observations"][0]["observed_at"] = (
        secret_datetime
    )
    bundle_path = tmp_path / "review-malformed-datetime.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        ["review", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert (
        "payload.closing_observations[0].observed_at: invalid_datetime"
        in result.output
    )
    assert secret_datetime not in result.output
    assert "Traceback" not in result.output


def test_forecast_config_version_secret_is_not_printed(
    tmp_path: Path,
) -> None:
    secret_version = "forecast-api-token-secret"
    mapping = confirmed_bundle_mapping()
    mapping["payload"]["forecast_config_version"] = secret_version
    bundle_path = tmp_path / "confirmed-secret-config.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        [
            "confirmed",
            "--bundle",
            str(bundle_path),
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--bundle-root",
            str(tmp_path / "bundles"),
        ],
    )

    assert result.exit_code == 3
    assert "unknown forecast config" in result.output
    assert secret_version not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("error_type", "exit_code", "safe_message"),
    (
        (
            PregameRelationshipError,
            3,
            "invalid pregame relationship",
        ),
        (PregameTimingError, 4, "invalid pregame timing"),
    ),
)
def test_cli_pregame_cross_field_error_omits_bundle_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[ValueError],
    exit_code: int,
    safe_message: str,
) -> None:
    secret = "https://attacker.example/?api_token=pregame-cross-field-secret"
    game = _complete_game()
    game = replace(
        game,
        away_lineup=replace(game.away_lineup, state=LineupState.PROJECTED),
        observations=(
            replace(game.observations[0], source_record=secret),
        ),
    )
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(game)
    bundle_path = tmp_path / "projected-cross-field-secret.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    def fail_validation(game_input):
        del game_input
        raise error_type(f"invalid provenance observation: {secret}")

    monkeypatch.setattr(
        service_module,
        "validate_pregame_input",
        fail_validation,
    )

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == exit_code
    assert safe_message in result.output
    assert secret not in result.output
    assert "attacker.example" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("command", "mapping_factory", "missing_path"),
    (
        (
            "projected",
            projected_bundle_mapping,
            "payload.forecast_config_version",
        ),
        (
            "confirmed",
            confirmed_bundle_mapping,
            "payload.forecast_config_version",
        ),
        ("review", review_bundle_mapping, "payload.capture_id"),
    ),
)
def test_malformed_bundle_reports_its_own_stage_schema_location(
    tmp_path: Path,
    command: str,
    mapping_factory,
    missing_path: str,
) -> None:
    mapping = mapping_factory()
    section, field = missing_path.split(".")
    del mapping[section][field]
    bundle_path = tmp_path / f"{command}-malformed.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        [command, "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert missing_path in result.output
    assert "literal_error" not in result.output
    assert "input_value" not in result.output
    assert "errors.pydantic.dev" not in result.output
    assert "Traceback" not in result.output


def test_valid_wrong_stage_bundle_is_a_cli_precondition(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "confirmed.json"
    bundle_path.write_text(
        _raw(confirmed_bundle_mapping()),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 4
    assert "bundle stage/type mismatch" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ("projected", "confirmed", "review"))
@pytest.mark.parametrize(
    ("stage_kind", "stage_value"),
    (
        ("list", ["projected"]),
        ("object", {"name": "projected"}),
        ("unknown", "unknown-stage-api-token-secret"),
        ("missing", None),
    ),
)
def test_invalid_stage_envelope_has_common_safe_diagnostic(
    tmp_path: Path,
    command: str,
    stage_kind: str,
    stage_value: object,
) -> None:
    mapping = projected_bundle_mapping()
    if stage_kind == "missing":
        del mapping["stage"]
    else:
        mapping["stage"] = stage_value
    bundle_path = tmp_path / f"{command}-{stage_kind}.json"
    bundle_path.write_text(_raw(mapping), encoding="utf-8")

    result = runner.invoke(
        cli_module.app,
        [command, "--bundle", str(bundle_path)],
    )

    assert result.exit_code == 3
    assert "stage envelope" in result.output
    assert "unknown-stage-api-token-secret" not in result.output
    assert "literal_error" not in result.output
    assert "Traceback" not in result.output


def test_cli_prints_authoritative_partial_progress_and_reruns_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "projected.json"
    mapping = projected_bundle_mapping()
    game = _complete_game()
    game = replace(
        game,
        away_lineup=replace(game.away_lineup, state=LineupState.PROJECTED),
    )
    mapping["payload"]["game_input"] = capture_payload(game)
    bundle_path.write_text(_raw(mapping), encoding="utf-8")
    ledger_path = tmp_path / "ledger" / "shadow.jsonl"
    bundle_root = tmp_path / "managed"
    original_persist = service_module.persist_managed_bundle
    interrupted = False
    monkeypatch.setattr(
        service_module,
        "load_forecast_config",
        lambda version, *, registry_root=None: SimulationConfig(
            version="shadow-forecast-2026-07-14.1",
            simulations=12,
        ),
    )

    def interrupt_once(bundle, *, root):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OperatorIOError("managed bundle persistence interrupted")
        return original_persist(bundle, root=root)

    monkeypatch.setattr(
        service_module,
        "persist_managed_bundle",
        interrupt_once,
    )
    arguments = [
        "projected",
        "--bundle",
        str(bundle_path),
        "--ledger",
        str(ledger_path),
        "--bundle-root",
        str(bundle_root),
    ]

    failed = runner.invoke(cli_module.app, arguments)

    assert failed.exit_code == 6, failed.output
    records = tuple(LedgerStore(ledger_path).iter_records())
    capture = next(
        record
        for record in records
        if record.envelope.record_kind == "capture"
    )
    forecast = next(
        record
        for record in records
        if record.envelope.record_kind == "forecast"
    )
    link = next(
        record
        for record in records
        if record.envelope.record_kind == "capture_forecast_link"
    )
    assert f"partial.capture_id: {capture.capture_id}" in failed.output
    assert f"partial.forecast_run_id: {forecast.forecast_run_id}" in failed.output
    assert (
        "partial.capture_forecast_link_record_id: "
        f"{link.envelope.record_id}"
    ) in failed.output
    assert "partial.managed_bundle" not in failed.output
    assert "partial.rerun_safe: true" in failed.output
    assert "Traceback" not in failed.output

    completed = runner.invoke(cli_module.app, arguments)

    assert completed.exit_code == 0, completed.output
    assert "capture_replayed: true" in completed.output
    assert "forecast_replayed: true" in completed.output
    assert "link_replayed: true" in completed.output
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 3


def test_confirmed_cli_exact_replay_skips_forecast_and_preserves_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "confirmed.json"
    mapping = confirmed_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(_complete_game())
    bundle_path.write_text(
        _raw(mapping),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger" / "shadow.jsonl"
    bundle_root = tmp_path / "managed"
    monkeypatch.setattr(
        service_module,
        "load_forecast_config",
        lambda version, *, registry_root=None: SimulationConfig(
            version=version,
            simulations=12,
        ),
    )
    arguments = [
        "confirmed",
        "--bundle",
        str(bundle_path),
        "--ledger",
        str(ledger_path),
        "--bundle-root",
        str(bundle_root),
    ]

    first = runner.invoke(cli_module.app, arguments)
    assert first.exit_code == 0, first.output
    records = tuple(LedgerStore(ledger_path).iter_records())
    capture = next(
        record
        for record in records
        if record.envelope.record_kind == "capture"
    )
    forecast = next(
        record
        for record in records
        if record.envelope.record_kind == "forecast"
    )
    link = next(
        record
        for record in records
        if record.envelope.record_kind == "capture_forecast_link"
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("exact CLI replay must not run forecast")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    replay = runner.invoke(cli_module.app, arguments)

    assert replay.exit_code == 0, replay.output
    assert f"capture_id: {capture.capture_id}" in replay.output
    assert f"forecast_run_id: {forecast.forecast_run_id}" in replay.output
    assert (
        f"capture_forecast_link_record_id: {link.envelope.record_id}"
        in replay.output
    )
    assert "capture_replayed: true" in replay.output
    assert "forecast_replayed: true" in replay.output
    assert "link_replayed: true" in replay.output


@pytest.mark.parametrize("command", ("projected", "confirmed", "review"))
def test_missing_required_bundle_exits_two(command: str) -> None:
    result = runner.invoke(cli_module.app, [command])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_unknown_option_exits_two() -> None:
    result = runner.invoke(
        cli_module.app,
        ["projected", "--bundle", "input.json", "--unknown-option"],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_no_args_exits_two() -> None:
    result = runner.invoke(cli_module.app, [])

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_module_entry_point_help_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "analysis.operator", "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "projected" in output
    assert "confirmed" in output
    assert "review" in output


def test_module_entry_point_no_args_exits_two_without_traceback() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "analysis.operator"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 2, output
    assert "Traceback" not in output
