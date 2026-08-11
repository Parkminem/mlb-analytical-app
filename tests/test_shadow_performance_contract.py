from dataclasses import replace
import json
from pathlib import Path

import pytest

import analysis.operator.service as service_module
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.ledger import ForecastCommand, LedgerCheckpoint, LedgerStore
from analysis.ledger.forecast_codec import expand_component_distributions
from analysis.ledger.pregame import PregameLedgerSession
from analysis.operator.reporting import (
    build_report_mapping,
    render_report_markdown,
)
from analysis.operator.schemas import parse_confirmed_bundle_json
from analysis.operator.service import run_confirmed
from analysis.review.calibration import build_calibration_report
from analysis.review.watchlist import (
    build_review_watchlist,
    load_review_watchlist_config,
)
from analysis.shadow import (
    canonical_legacy_decision_payload,
    deserialize_forecast_context,
    deserialize_shadow_comparison,
)
from tests.test_calibration import calibration_row
from tests.test_ledger_pregame import capture_command, forecast_command
from tests.test_operator_pregame import (
    _confirmed_game,
    _ordered_legacy_decision,
    _run_paths,
)
from tests.test_operator_schemas import (
    _raw,
    confirmed_bundle_mapping,
)


@pytest.fixture(scope="module")
def representative_confirmed_run(
    tmp_path_factory: pytest.TempPathFactory,
):
    root = tmp_path_factory.mktemp("shadow-performance-contract")
    ledger_path, bundle_root = _run_paths(root)
    mapping = confirmed_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(_confirmed_game())
    mapping["payload"]["legacy_decision"] = (
        canonical_legacy_decision_payload(_ordered_legacy_decision())
    )
    bundle = parse_confirmed_bundle_json(_raw(mapping))
    result = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    return bundle, ledger_path, bundle_root, result


def _historical_outputs(compact_outputs: object) -> dict[str, object]:
    outputs = json.loads(canonical_json(compact_outputs))
    outputs["component_distributions"] = expand_component_distributions(
        outputs["component_distributions"]
    )
    return outputs


def _forecast_command_bytes(record, outputs: dict[str, object]) -> bytes:
    command = ForecastCommand(
        forecast_run_id=record.forecast_run_id,
        model_version=record.model_version,
        simulation_config_hash=record.simulation_config_hash,
        normalized_input_hash=record.normalized_input_hash,
        outputs=outputs,
    )
    return canonical_json(command).encode("utf-8") + b"\n"


def test_warm_checkpoint_performs_zero_full_record_reads(
    representative_confirmed_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, ledger_path, _, _ = representative_confirmed_run

    def fail_full_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("warm checkpoint must not reconstruct full records")

    monkeypatch.setattr(LedgerStore, "_read_records", fail_full_read)

    checkpoint = LedgerStore(ledger_path).pregame_checkpoint()

    assert checkpoint.record_count == 3


def test_new_confirmed_run_after_warm_checkpoint_performs_zero_full_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    registry_root = tmp_path / "forecasting"
    registry_root.mkdir()
    config = SimulationConfig(simulations=12)
    (registry_root / f"{config.version}.json").write_text(
        canonical_json(config.to_mapping()),
        encoding="utf-8",
    )
    mapping = confirmed_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(_confirmed_game())
    mapping["payload"]["forecast_config_version"] = config.version
    bundle = parse_confirmed_bundle_json(_raw(mapping))
    LedgerStore(ledger_path).pregame_checkpoint()

    def fail_full_read(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "a warm confirmed run must not reconstruct full ledger records"
        )

    monkeypatch.setattr(LedgerStore, "_read_records", fail_full_read)

    result = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )

    assert result.capture_replayed is False
    assert result.forecast_replayed is False
    assert result.link_replayed is False


def test_one_new_confirmed_append_uses_one_exclusive_pregame_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    real_enter = PregameLedgerSession.__enter__
    exclusive_sessions = 0

    def enter_spy(session: PregameLedgerSession) -> PregameLedgerSession:
        nonlocal exclusive_sessions
        if session._exclusive:
            exclusive_sessions += 1
        return real_enter(session)

    monkeypatch.setattr(PregameLedgerSession, "__enter__", enter_spy)

    records = store.append_shadow_run_atomic(
        "capture:performance-contract",
        capture_command(),
        forecast_command(),
        None,
    )

    assert exclusive_sessions == 1
    assert records.link.capture_id == records.capture.capture_id
    assert records.link.forecast_run_id == records.forecast.forecast_run_id


def test_same_store_reuses_trusted_checkpoint_between_pregame_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    store.lookup_shadow_pregame(
        "capture:missing",
        "forecast:missing",
        900001,
    )

    def fail_checkpoint_reload(*args: object, **kwargs: object):
        raise AssertionError(
            "an unchanged store must reuse its trusted in-memory checkpoint"
        )

    monkeypatch.setattr(
        LedgerCheckpoint,
        "load",
        classmethod(fail_checkpoint_reload),
    )

    records = store.append_shadow_run_atomic(
        "capture:checkpoint-reuse",
        capture_command(),
        forecast_command(),
        None,
    )

    assert records.link.capture_id == records.capture.capture_id


def test_exact_confirmed_replay_performs_zero_forecasts(
    representative_confirmed_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, ledger_path, bundle_root, first = representative_confirmed_run
    calls = 0

    def count_forecast(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        raise AssertionError("exact replay must not run forecast")

    monkeypatch.setattr(service_module, "run_forecast", count_forecast)

    replay = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert calls == 0
    assert replay.capture == first.capture
    assert replay.forecast == first.forecast
    assert replay.link == first.link


def test_representative_compact_forecast_is_at_least_sixty_percent_smaller(
    representative_confirmed_run,
) -> None:
    _, _, _, result = representative_confirmed_run
    compact_outputs = json.loads(canonical_json(result.forecast.outputs))
    historical_outputs = _historical_outputs(compact_outputs)

    compact_bytes = _forecast_command_bytes(
        result.forecast,
        compact_outputs,
    )
    historical_bytes = _forecast_command_bytes(
        result.forecast,
        historical_outputs,
    )

    assert len(compact_bytes) <= len(historical_bytes) * 0.40


def test_decoded_forecast_and_generated_report_mapping_are_identical(
    representative_confirmed_run,
) -> None:
    bundle, ledger_path, _, result = representative_confirmed_run
    compact_outputs = json.loads(canonical_json(result.forecast.outputs))
    historical_outputs = _historical_outputs(compact_outputs)
    compact_context = deserialize_forecast_context(
        compact_outputs,
        away_team=bundle.payload.game_input.identity.away_team,
        home_team=bundle.payload.game_input.identity.home_team,
    )
    historical_context = deserialize_forecast_context(
        historical_outputs,
        away_team=bundle.payload.game_input.identity.away_team,
        home_team=bundle.payload.game_input.identity.home_team,
    )
    comparison = deserialize_shadow_comparison(
        compact_outputs["shadow_comparison"]
    )
    selected = comparison.market_comparisons[0]
    market_input = compact_context.market_inputs[selected.market_key]
    row = calibration_row(
        game_pk=bundle.game_pk,
        capture_id=result.capture.capture_id,
        forecast_run_id=result.forecast.forecast_run_id,
        market_key=selected.market_key,
        model_version=result.forecast.model_version,
        market_type=market_input.market_type.value,
        settlement_format=market_input.settlement_format.value,
        line=market_input.line,
        lineup_state="confirmed",
        legacy_priority_score=selected.legacy_score,
    )
    rows = (row,)
    report = build_calibration_report(rows)
    watchlist = build_review_watchlist(rows, load_review_watchlist_config())
    compact_mapping = build_report_mapping(
        ledger_path=ledger_path,
        ledger_records=(result.forecast,),
        rows=rows,
        report=report,
        watchlist=watchlist,
    )
    historical_mapping = build_report_mapping(
        ledger_path=ledger_path,
        ledger_records=(
            replace(result.forecast, outputs=historical_outputs),
        ),
        rows=rows,
        report=report,
        watchlist=watchlist,
    )

    assert compact_context == historical_context
    assert canonical_json(compact_mapping) == canonical_json(
        historical_mapping
    )
    assert render_report_markdown(compact_mapping) == render_report_markdown(
        historical_mapping
    )
