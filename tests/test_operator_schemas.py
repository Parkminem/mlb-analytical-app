from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.forecasting.inputs import NormalizedGameInput
from analysis.ledger import MarketObservationCommand, OfficialResultCommand
from analysis.operator.config import load_forecast_config
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from analysis.shadow import LegacyDecisionSnapshot
from tests.test_forecasting_inputs import complete_forecast_game


UTC = timezone.utc
GAME_PK = 900001


def _legacy_decision_mapping() -> dict[str, object]:
    snapshot = LegacyDecisionSnapshot(
        model_version="legacy-fixed-v1",
        candidates=(),
        final_priority=(),
        review_bindings=(),
        notes=("legacy remains authoritative",),
    )
    return json.loads(canonical_json(snapshot))


def projected_bundle_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "projected",
        "game_pk": GAME_PK,
        "created_at": "2026-07-14T01:35:00Z",
        "payload": {
            "game_input": capture_payload(complete_forecast_game()),
            "legacy_decision": _legacy_decision_mapping(),
            "forecast_config_version": "shadow-forecast-2026-07-14.1",
        },
    }


def confirmed_bundle_mapping() -> dict[str, object]:
    mapping = projected_bundle_mapping()
    mapping["stage"] = "confirmed"
    return mapping


def review_bundle_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "review",
        "game_pk": GAME_PK,
        "created_at": "2026-07-15T04:00:00Z",
        "payload": {
            "capture_id": "capture:confirmed:900001",
            "forecast_run_id": "forecast:900001",
            "capture_forecast_link_record_id": "record:link:900001",
            "official_result": {
                "game_pk": GAME_PK,
                "source_name": "MLB",
                "source_record": "game/900001",
                "source_revision": "final-1",
                "finalized_at": "2026-07-15T03:30:00Z",
                "away_f5_runs": 2,
                "home_f5_runs": 1,
                "away_full_runs": 4,
                "home_full_runs": 3,
                "component_values": {},
                "documented_events": [],
                "supersedes_record_id": None,
            },
            "closing_observations": [
                {
                    "game_pk": GAME_PK,
                    "source_name": "ExampleBook",
                    "source_record": "quote/900001/ml",
                    "source_revision": "close-1",
                    "observation_kind": "close",
                    "market_key": "game:900001:ml:SEA",
                    "market_type": "ml",
                    "settlement_format": "two_way",
                    "selected_side": "SEA",
                    "line": None,
                    "selected_decimal_price": 1.91,
                    "opposing_decimal_price": 2.01,
                    "observed_at": "2026-07-14T22:55:00Z",
                    "available_at": "2026-07-14T22:55:00Z",
                    "supersedes_record_id": None,
                }
            ],
        },
    }


def _raw(mapping: dict[str, object]) -> str:
    return json.dumps(mapping, ensure_ascii=False)


@pytest.mark.parametrize(
    ("error_type", "exit_code"),
    (
        (BundleValidationError, 3),
        (OperatorPreconditionError, 4),
        (OperatorLedgerError, 5),
        (OperatorIOError, 6),
    ),
)
def test_operator_exception_categories_have_stable_exit_codes(
    error_type: type[Exception],
    exit_code: int,
) -> None:
    error = error_type("useful message")

    assert error.exit_code == exit_code
    assert str(error) == "useful message"


def test_current_registry_file_exactly_matches_default_config() -> None:
    registry_path = (
        Path(__file__).parents[1]
        / "config"
        / "forecasting"
        / "shadow-forecast-2026-07-14.1.json"
    )

    assert json.loads(registry_path.read_text(encoding="utf-8")) == (
        SimulationConfig().to_mapping()
    )
    assert load_forecast_config("shadow-forecast-2026-07-14.1") == (
        SimulationConfig()
    )


def test_unknown_registry_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BundleValidationError, match="unknown forecast config"):
        load_forecast_config("missing", registry_root=tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_registry_requires_exact_complete_config_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    mapping = SimulationConfig().to_mapping()
    if mutation == "missing":
        mapping.pop("league_woba")
    else:
        mapping["unexpected"] = 1
    path = tmp_path / f"{mapping['version']}.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(BundleValidationError, match=f"{mutation} config field"):
        load_forecast_config(str(mapping["version"]), registry_root=tmp_path)


def test_registry_rejects_filename_content_version_mismatch(
    tmp_path: Path,
) -> None:
    mapping = SimulationConfig().to_mapping()
    path = tmp_path / "historical-complete.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="filename/content version"):
        load_forecast_config("historical-complete", registry_root=tmp_path)


def test_registry_loads_second_complete_historical_config_without_defaults(
    tmp_path: Path,
) -> None:
    historical_version = "shadow-forecast-2026-06-01.1"
    mapping = SimulationConfig().to_mapping()
    mapping.update(
        version=historical_version,
        simulations=12345,
        league_woba=0.315,
    )
    assert set(mapping) == {field.name for field in fields(SimulationConfig)}
    path = tmp_path / f"{historical_version}.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    loaded = load_forecast_config(historical_version, registry_root=tmp_path)

    assert loaded == SimulationConfig.from_mapping(mapping)
    assert loaded.simulations == 12345
    assert loaded.league_woba == 0.315


@pytest.mark.parametrize(
    ("mapping_factory", "parser", "stage"),
    (
        (projected_bundle_mapping, parse_projected_bundle_json, "projected"),
        (confirmed_bundle_mapping, parse_confirmed_bundle_json, "confirmed"),
    ),
)
def test_pregame_bundles_round_trip_frozen_domain_values(
    mapping_factory,
    parser,
    stage: str,
) -> None:
    bundle = parser(_raw(mapping_factory()))

    assert bundle.stage == stage
    assert isinstance(bundle.payload.game_input, NormalizedGameInput)
    assert isinstance(bundle.payload.legacy_decision, LegacyDecisionSnapshot)
    with pytest.raises(FrozenInstanceError):
        bundle.payload.game_input.captured_at = datetime.now(UTC)


def test_parsing_does_not_run_pregame_service_preflight() -> None:
    mapping = projected_bundle_mapping()
    game_input = mapping["payload"]["game_input"]
    game_input["away_lineup"]["team"] = "OAK"
    game_input["captured_at"] = game_input["identity"]["first_pitch"]

    bundle = parse_projected_bundle_json(_raw(mapping))

    assert bundle.payload.game_input.away_lineup.team == "OAK"
    assert (
        bundle.payload.game_input.captured_at
        == bundle.payload.game_input.identity.first_pitch
    )


@pytest.mark.parametrize("location", ["envelope", "payload"])
def test_bundle_rejects_unknown_fields(location: str) -> None:
    mapping = projected_bundle_mapping()
    target = mapping if location == "envelope" else mapping["payload"]
    target["unexpected"] = True

    with pytest.raises(BundleValidationError, match="unexpected"):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize("schema_version", [0, 2, "1"])
def test_bundle_rejects_unsupported_or_wrong_type_schema_versions(
    schema_version: object,
) -> None:
    mapping = projected_bundle_mapping()
    mapping["schema_version"] = schema_version

    with pytest.raises(BundleValidationError, match="schema_version"):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize(
    ("mapping_factory", "parser"),
    (
        (projected_bundle_mapping, parse_projected_bundle_json),
        (confirmed_bundle_mapping, parse_confirmed_bundle_json),
        (review_bundle_mapping, parse_review_bundle_json),
    ),
)
@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_all_bundles_require_schema_version_json_integer_token(
    mapping_factory,
    parser,
    schema_version: object,
) -> None:
    mapping = mapping_factory()
    mapping["schema_version"] = schema_version

    with pytest.raises(BundleValidationError, match="schema_version.*integer"):
        parser(_raw(mapping))


def test_bundle_rejects_duplicate_json_keys() -> None:
    raw = _raw(projected_bundle_mapping())
    raw = raw.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )

    with pytest.raises(BundleValidationError, match="duplicate JSON object key"):
        parse_projected_bundle_json(raw)


def test_bundle_translates_invalid_utf8_surrogate_in_python_string() -> None:
    with pytest.raises(
        BundleValidationError,
        match="valid UTF-8",
    ) as captured:
        parse_projected_bundle_json("\ud800")

    assert captured.value.exit_code == 3
    assert isinstance(captured.value.__cause__, UnicodeEncodeError)


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
def test_bundle_rejects_non_finite_json_numbers(constant: float) -> None:
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"]["away_lineup"][
        "projected_runs_per_9"
    ]["value"] = constant

    with pytest.raises(BundleValidationError, match="invalid JSON number"):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda mapping: mapping.update(created_at="2026-07-14T01:35:00"),
        lambda mapping: mapping["payload"]["game_input"].update(
            captured_at="2026-07-14T01:30:00"
        ),
    ),
)
def test_bundle_rejects_naive_timestamps(mutate) -> None:
    mapping = projected_bundle_mapping()
    mutate(mapping)

    with pytest.raises(BundleValidationError, match="timezone|offset|aware"):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize(
    ("field_name", "mutate"),
    (
        (
            "game_pk",
            lambda game: game["identity"].update(game_pk="900001"),
        ),
        (
            "value",
            lambda game: game["away_lineup"]["projected_runs_per_9"].update(
                value="4.5"
            ),
        ),
        (
            "value",
            lambda game: game["away_lineup"]["recent_form_included"].update(
                value=1
            ),
        ),
        (
            "state",
            lambda game: game["away_lineup"].update(state=1),
        ),
        (
            "first_pitch",
            lambda game: game["identity"].update(first_pitch=123),
        ),
    ),
)
def test_bundle_rejects_wrong_nested_scalar_types(
    field_name: str,
    mutate,
) -> None:
    mapping = projected_bundle_mapping()
    mutate(mapping["payload"]["game_input"])

    with pytest.raises(BundleValidationError, match=field_name):
        parse_projected_bundle_json(_raw(mapping))


def test_pregame_envelope_game_pk_must_match_game_identity() -> None:
    mapping = projected_bundle_mapping()
    mapping["game_pk"] = GAME_PK + 1

    with pytest.raises(BundleValidationError, match="game_pk.*game_input"):
        parse_projected_bundle_json(_raw(mapping))


def test_domain_construction_failures_map_to_bundle_validation_error() -> None:
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"]["identity"]["away_team"] = ""

    with pytest.raises(BundleValidationError, match="away_team") as captured:
        parse_projected_bundle_json(_raw(mapping))

    assert captured.value.__cause__ is not None


def test_review_bundle_round_trips_domain_values_and_optional_closes() -> None:
    mapping = review_bundle_mapping()
    bundle = parse_review_bundle_json(_raw(mapping))

    assert isinstance(bundle.payload.official_result, OfficialResultCommand)
    assert isinstance(
        bundle.payload.closing_observations[0],
        MarketObservationCommand,
    )
    assert bundle.payload.capture_id == mapping["payload"]["capture_id"]

    mapping["payload"].pop("closing_observations")
    without_closes = parse_review_bundle_json(_raw(mapping))
    assert without_closes.payload.closing_observations == ()


@pytest.mark.parametrize(
    "field_name",
    (
        "capture_id",
        "forecast_run_id",
        "capture_forecast_link_record_id",
    ),
)
def test_review_bundle_requires_exact_nonempty_frozen_ids(field_name: str) -> None:
    mapping = review_bundle_mapping()
    mapping["payload"][field_name] = "   "

    with pytest.raises(BundleValidationError, match=field_name):
        parse_review_bundle_json(_raw(mapping))


def test_review_envelope_game_pk_must_match_official_result() -> None:
    mapping = review_bundle_mapping()
    mapping["game_pk"] = GAME_PK + 1

    with pytest.raises(BundleValidationError, match="game_pk.*official_result"):
        parse_review_bundle_json(_raw(mapping))


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("observation_kind", "entry"),
        ("selected_decimal_price", None),
        ("selected_decimal_price", 1.0),
        ("opposing_decimal_price", None),
        ("opposing_decimal_price", 1.0),
        ("available_at", None),
        ("available_at", "2026-07-14T22:55:00"),
    ),
)
def test_review_bundle_requires_strict_closing_terms(
    field_name: str,
    value: object,
) -> None:
    mapping = review_bundle_mapping()
    mapping["payload"]["closing_observations"][0][field_name] = value

    with pytest.raises(BundleValidationError, match=field_name):
        parse_review_bundle_json(_raw(mapping))


def test_review_closing_observation_must_match_bundle_game() -> None:
    mapping = review_bundle_mapping()
    mapping["payload"]["closing_observations"][0]["game_pk"] = GAME_PK + 1

    with pytest.raises(BundleValidationError, match="closing.*game_pk"):
        parse_review_bundle_json(_raw(mapping))


def test_review_domain_errors_map_to_bundle_validation_error() -> None:
    mapping = review_bundle_mapping()
    mapping["payload"]["official_result"]["home_full_runs"] = 4

    with pytest.raises(BundleValidationError, match="winner") as captured:
        parse_review_bundle_json(_raw(mapping))

    assert captured.value.__cause__ is not None
