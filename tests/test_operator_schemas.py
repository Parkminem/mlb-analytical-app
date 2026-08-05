from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.forecasting.inputs import NormalizedGameInput
from analysis.ledger import MarketObservationCommand, OfficialResultCommand
from analysis.operator.config import load_forecast_config
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPartialProgress,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    CURRENT_PREGAME_AUTHORITY,
    CURRENT_PREGAME_SCHEMA_VERSION,
    ConfirmedBundle,
    HistoricalRelaxedConfirmedBundleV1,
    HistoricalRelaxedProjectedBundleV1,
    HistoricalStrictConfirmedBundleV1,
    HistoricalStrictProjectedBundleV1,
    ProjectedBundle,
    parse_confirmed_bundle_json,
    parse_historical_confirmed_bundle_json,
    parse_historical_projected_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from analysis.shadow import LegacyDecisionSnapshot
from tests.test_forecasting_inputs import complete_forecast_game


UTC = timezone.utc
GAME_PK = 900001


def test_partial_progress_is_not_rerun_safe_without_authoritative_state() -> None:
    progress = OperatorPartialProgress(stage="projected", game_pk=GAME_PK)
    field_names = {field.name for field in fields(OperatorPartialProgress)}

    assert progress.rerun_safe is False
    assert "managed_bundle_path" not in field_names
    assert "managed_bundle_replayed" not in field_names


def _legacy_decision_mapping() -> dict[str, object]:
    snapshot = LegacyDecisionSnapshot(
        model_version="2026-08-03",
        candidates=(),
        final_priority=(),
        review_bindings=(),
        notes=("legacy remains authoritative",),
    )
    return json.loads(canonical_json(snapshot))


def _complete_legacy_decision_mapping() -> dict[str, object]:
    return {
        "model_version": "2026-08-03",
        "candidates": [
            {
                "market": "ml",
                "score": 78,
                "decision_rating": "play",
                "evidence_paths": [
                    "starter_suppression",
                    "bullpen_protection",
                ],
                "same_game_pair_compatible": False,
            }
        ],
        "final_priority": ["ml"],
        "review_bindings": [
            {
                "market_key": "game:900001:ml:SEA",
                "legacy_candidate_index": 0,
                "legacy_market": "ml",
                "principal_paths": [
                    {
                        "evidence_path": "starter_suppression",
                        "component_ids": [
                            "home_pitcher_chain_runs_allowed"
                        ],
                        "contradiction_rule": "starter_suppression",
                    },
                    {
                        "evidence_path": "bullpen_protection",
                        "component_ids": [
                            "home_bullpen_runs_allowed",
                            "home_protected_bullpen_close_entry",
                        ],
                        "contradiction_rule": "bullpen_protection",
                    }
                ],
            }
        ],
        "notes": [],
    }


def projected_bundle_mapping() -> dict[str, object]:
    return {
        "schema_version": 2,
        "authority": "current",
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


def _two_candidate_legacy_decision_mapping() -> dict[str, object]:
    legacy = _complete_legacy_decision_mapping()
    second_candidate = deepcopy(legacy["candidates"][0])
    second_candidate.update(
        market="full_total",
        score=74,
        same_game_pair_compatible=True,
    )
    legacy["candidates"].append(second_candidate)
    legacy["final_priority"].append("full_total")
    return legacy


def historical_bundle_mapping(
    mapping_factory,
    candidate_contract: str,
) -> dict[str, object]:
    mapping = mapping_factory()
    mapping["schema_version"] = 1
    mapping.pop("authority")
    legacy = mapping["payload"]["legacy_decision"]
    if candidate_contract != "empty":
        legacy = _two_candidate_legacy_decision_mapping()
        mapping["payload"]["legacy_decision"] = legacy
    if candidate_contract == "relaxed":
        for candidate in legacy["candidates"]:
            candidate.pop("same_game_pair_compatible")
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
    secret_version = "missing-api-token-secret"

    with pytest.raises(
        BundleValidationError,
        match="unknown forecast config",
    ) as raised:
        load_forecast_config(secret_version, registry_root=tmp_path)

    assert secret_version not in str(raised.value)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_registry_requires_exact_complete_config_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    mapping = SimulationConfig().to_mapping()
    secret_unknown_field = "https://attacker.example/?api_token=config-secret"
    if mutation == "missing":
        mapping.pop("league_woba")
    else:
        mapping[secret_unknown_field] = 1
    path = tmp_path / f"{mapping['version']}.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(
        BundleValidationError,
        match=f"{mutation} config field",
    ) as raised:
        load_forecast_config(str(mapping["version"]), registry_root=tmp_path)

    message = str(raised.value)
    if mutation == "missing":
        assert "league_woba" in message
    else:
        assert secret_unknown_field not in message
        assert "attacker.example" not in message


def test_registry_rejects_filename_content_version_mismatch(
    tmp_path: Path,
) -> None:
    mapping = SimulationConfig().to_mapping()
    filename_secret = "filename-api-token-secret"
    content_secret = "content-api-token-secret"
    mapping["version"] = content_secret
    path = tmp_path / f"{filename_secret}.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(
        BundleValidationError,
        match="filename/content version",
    ) as raised:
        load_forecast_config(filename_secret, registry_root=tmp_path)

    assert filename_secret not in str(raised.value)
    assert content_secret not in str(raised.value)


def test_registry_parse_error_does_not_echo_secret_content(
    tmp_path: Path,
) -> None:
    version = "historical-invalid"
    secret_key = "api_token_secret_duplicate_key"
    (tmp_path / f"{version}.json").write_text(
        '{"' + secret_key + '": 1, "' + secret_key + '": 2}',
        encoding="utf-8",
    )

    with pytest.raises(BundleValidationError) as raised:
        load_forecast_config(version, registry_root=tmp_path)

    assert secret_key not in str(raised.value)


def test_registry_simulation_validation_does_not_echo_secret_value(
    tmp_path: Path,
) -> None:
    version = "historical-invalid-simulation"
    secret_value = "simulation-api-token-secret"
    mapping = SimulationConfig().to_mapping()
    mapping.update(version=version, simulations=secret_value)
    (tmp_path / f"{version}.json").write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )

    with pytest.raises(BundleValidationError) as raised:
        load_forecast_config(version, registry_root=tmp_path)

    assert secret_value not in str(raised.value)


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


@pytest.mark.parametrize(
    ("parser", "mapping_factory"),
    (
        (parse_projected_bundle_json, projected_bundle_mapping),
        (parse_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_current_pregame_requires_v2_authority(parser, mapping_factory) -> None:
    mapping = mapping_factory()
    mapping["schema_version"] = CURRENT_PREGAME_SCHEMA_VERSION
    mapping["authority"] = CURRENT_PREGAME_AUTHORITY

    parsed = parser(json.dumps(mapping))

    assert parsed.schema_version == 2
    assert parsed.authority == "current"
    assert CURRENT_PREGAME_SCHEMA_VERSION == 2
    assert CURRENT_PREGAME_AUTHORITY == "current"


@pytest.mark.parametrize(
    ("mutation", "field_name"),
    (
        (lambda mapping: mapping.pop("authority"), "authority"),
        (
            lambda mapping: mapping.update(authority="historical"),
            "authority",
        ),
        (
            lambda mapping: mapping.update(authority="current-extra"),
            "authority",
        ),
        (lambda mapping: mapping.update(authority=1), "authority"),
        (lambda mapping: mapping.update(authority=None), "authority"),
        (lambda mapping: mapping.update(schema_version=1), "schema_version"),
        (lambda mapping: mapping.update(schema_version="2"), "schema_version"),
    ),
)
@pytest.mark.parametrize(
    ("parser", "mapping_factory"),
    (
        (parse_projected_bundle_json, projected_bundle_mapping),
        (parse_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_current_pregame_rejects_non_v2_authority_envelopes(
    parser,
    mapping_factory,
    mutation,
    field_name: str,
) -> None:
    mapping = mapping_factory()
    mutation(mapping)

    with pytest.raises(BundleValidationError, match=field_name):
        parser(_raw(mapping))


@pytest.mark.parametrize(
    ("parser", "mapping_factory"),
    (
        (parse_projected_bundle_json, projected_bundle_mapping),
        (parse_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_current_v2_authority_admits_empty_candidates(
    parser,
    mapping_factory,
) -> None:
    parsed = parser(_raw(mapping_factory()))

    assert parsed.authority == "current"
    assert parsed.payload.legacy_decision.candidates == ()


@pytest.mark.parametrize("candidate_index", (0, 1))
@pytest.mark.parametrize(
    ("parser", "mapping_factory"),
    (
        (parse_projected_bundle_json, projected_bundle_mapping),
        (parse_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_current_v2_requires_compatibility_on_every_candidate(
    parser,
    mapping_factory,
    candidate_index: int,
) -> None:
    mapping = mapping_factory()
    legacy = _two_candidate_legacy_decision_mapping()
    mapping["payload"]["legacy_decision"] = legacy
    legacy["candidates"][candidate_index].pop("same_game_pair_compatible")

    with pytest.raises(
        BundleValidationError,
        match="same_game_pair_compatible",
    ):
        parser(_raw(mapping))


@pytest.mark.parametrize("candidate_index", (0, 1))
@pytest.mark.parametrize(
    ("current_type", "mapping_factory"),
    (
        (ProjectedBundle, projected_bundle_mapping),
        (ConfirmedBundle, confirmed_bundle_mapping),
    ),
)
def test_current_v2_direct_validation_requires_compatibility_on_every_candidate(
    current_type,
    mapping_factory,
    candidate_index: int,
) -> None:
    mapping = mapping_factory()
    legacy = _two_candidate_legacy_decision_mapping()
    mapping["payload"]["legacy_decision"] = legacy
    legacy["candidates"][candidate_index].pop("same_game_pair_compatible")

    for validate in (
        lambda: current_type.model_validate(mapping),
        lambda: TypeAdapter(current_type).validate_python(mapping),
        lambda: current_type.model_validate_json(_raw(mapping)),
    ):
        with pytest.raises(ValidationError, match="same_game_pair_compatible"):
            validate()


@pytest.mark.parametrize("compatibility", (0, 1, "false", None))
def test_current_v2_requires_real_json_boolean_compatibility(
    compatibility: object,
) -> None:
    mapping = projected_bundle_mapping()
    legacy = _complete_legacy_decision_mapping()
    mapping["payload"]["legacy_decision"] = legacy
    legacy["candidates"][0]["same_game_pair_compatible"] = compatibility

    with pytest.raises(
        BundleValidationError,
        match="same_game_pair_compatible.*invalid_boolean",
    ):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize("compatibility", (0, 1, "false", "true"))
@pytest.mark.parametrize(
    ("current_type", "mapping_factory"),
    (
        (ProjectedBundle, projected_bundle_mapping),
        (ConfirmedBundle, confirmed_bundle_mapping),
    ),
)
def test_current_compatibility_is_intrinsically_strict_boolean(
    current_type,
    mapping_factory,
    compatibility: object,
) -> None:
    mapping = mapping_factory()
    mapping["payload"]["legacy_decision"] = _complete_legacy_decision_mapping()
    mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ] = compatibility
    adapter = TypeAdapter(current_type)

    for validate in (
        lambda: current_type.model_validate(mapping),
        lambda: adapter.validate_python(mapping),
        lambda: current_type.model_validate_json(_raw(mapping)),
        lambda: adapter.validate_json(_raw(mapping)),
    ):
        with pytest.raises(ValidationError, match="same_game_pair_compatible"):
            validate()


@pytest.mark.parametrize(
    "dump_path",
    (
        "model_dump",
        "model_dump_json",
        "adapter_dump_python",
        "adapter_dump_json",
    ),
)
@pytest.mark.parametrize(
    ("current_type", "mapping_factory"),
    (
        (ProjectedBundle, projected_bundle_mapping),
        (ConfirmedBundle, confirmed_bundle_mapping),
    ),
)
def test_current_bundle_public_dump_paths_are_warning_free(
    current_type,
    mapping_factory,
    dump_path: str,
) -> None:
    mapping = mapping_factory()
    mapping["payload"]["legacy_decision"] = _complete_legacy_decision_mapping()
    bundle = current_type.model_validate(mapping)
    adapter = TypeAdapter(current_type)

    if dump_path == "model_dump":
        dumped = bundle.model_dump(warnings="error")
    elif dump_path == "model_dump_json":
        dumped = json.loads(bundle.model_dump_json(warnings="error"))
    elif dump_path == "adapter_dump_python":
        dumped = adapter.dump_python(bundle, warnings="error")
    else:
        dumped = json.loads(adapter.dump_json(bundle, warnings="error"))

    assert dumped["schema_version"] == 2
    assert dumped["authority"] == "current"
    assert dumped["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ] is False


@pytest.mark.parametrize(
    (
        "historical_parser",
        "mapping_factory",
        "relaxed_type",
        "strict_type",
    ),
    (
        (
            parse_historical_projected_bundle_json,
            projected_bundle_mapping,
            HistoricalRelaxedProjectedBundleV1,
            HistoricalStrictProjectedBundleV1,
        ),
        (
            parse_historical_confirmed_bundle_json,
            confirmed_bundle_mapping,
            HistoricalRelaxedConfirmedBundleV1,
            HistoricalStrictConfirmedBundleV1,
        ),
    ),
)
@pytest.mark.parametrize("candidate_contract", ("relaxed", "strict", "empty"))
def test_historical_shape_is_classified_and_preserved(
    historical_parser,
    mapping_factory,
    relaxed_type,
    strict_type,
    candidate_contract: str,
) -> None:
    mapping = historical_bundle_mapping(mapping_factory, candidate_contract)

    parsed = historical_parser(_raw(mapping))
    dumped = parsed.model_dump()
    dumped_json = json.loads(parsed.model_dump_json())
    nested_payload = parsed.payload.model_dump()

    expected_type = strict_type if candidate_contract == "strict" else relaxed_type
    assert type(parsed) is expected_type
    assert parsed.schema_version == 1
    assert "authority" not in type(parsed).model_fields
    assert "authority" not in dumped
    expected_presence = candidate_contract == "strict"
    for payload in (dumped["payload"], dumped_json["payload"], nested_payload):
        candidates = payload["legacy_decision"]["candidates"]
        assert all(
            ("same_game_pair_compatible" in candidate) is expected_presence
            for candidate in candidates
        )
    if candidate_contract == "strict":
        assert [
            candidate.same_game_pair_compatible
            for candidate in parsed.payload.legacy_decision.candidates
        ] == [False, True]
    else:
        assert all(
            candidate.same_game_pair_compatible is False
            for candidate in parsed.payload.legacy_decision.candidates
        )


@pytest.mark.parametrize(
    ("historical_parser", "mapping_factory"),
    (
        (parse_historical_projected_bundle_json, projected_bundle_mapping),
        (parse_historical_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_historical_shape_rejects_mixed_candidate_contracts(
    historical_parser,
    mapping_factory,
) -> None:
    mapping = historical_bundle_mapping(mapping_factory, "strict")
    mapping["payload"]["legacy_decision"]["candidates"][0].pop(
        "same_game_pair_compatible"
    )

    with pytest.raises(
        BundleValidationError,
        match="mixed compatibility contract",
    ):
        historical_parser(_raw(mapping))


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


@pytest.mark.parametrize(
    ("container_path", "field_name"),
    (
        ((), "model_version"),
        ((), "candidates"),
        ((), "final_priority"),
        ((), "review_bindings"),
        ((), "notes"),
        (("candidates", 0), "market"),
        (("candidates", 0), "score"),
        (("candidates", 0), "decision_rating"),
        (("candidates", 0), "evidence_paths"),
        (("candidates", 0), "same_game_pair_compatible"),
        (("review_bindings", 0), "market_key"),
        (("review_bindings", 0), "legacy_candidate_index"),
        (("review_bindings", 0), "legacy_market"),
        (("review_bindings", 0), "principal_paths"),
        (
            ("review_bindings", 0, "principal_paths", 0),
            "evidence_path",
        ),
        (
            ("review_bindings", 0, "principal_paths", 0),
            "component_ids",
        ),
        (
            ("review_bindings", 0, "principal_paths", 0),
            "contradiction_rule",
        ),
    ),
)
def test_legacy_snapshot_requires_every_nested_field_explicitly(
    container_path: tuple[str | int, ...],
    field_name: str,
) -> None:
    mapping = projected_bundle_mapping()
    legacy_decision = _complete_legacy_decision_mapping()
    mapping["payload"]["legacy_decision"] = legacy_decision
    target: object = legacy_decision
    for path_part in container_path:
        target = target[path_part]
    assert isinstance(target, dict)
    target.pop(field_name)

    with pytest.raises(
        BundleValidationError,
        match=rf"legacy_decision.*{field_name}",
    ):
        parse_projected_bundle_json(_raw(mapping))


def test_unbound_candidate_cannot_default_omitted_evidence_paths() -> None:
    mapping = projected_bundle_mapping()
    legacy_decision = _complete_legacy_decision_mapping()
    legacy_decision["review_bindings"] = []
    legacy_decision["candidates"][0].pop("evidence_paths")
    mapping["payload"]["legacy_decision"] = legacy_decision

    with pytest.raises(
        BundleValidationError,
        match="legacy_decision.*evidence_paths",
    ):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize("location", ["envelope", "payload"])
def test_bundle_rejects_unknown_fields(location: str) -> None:
    mapping = projected_bundle_mapping()
    target = mapping if location == "envelope" else mapping["payload"]
    target["unexpected"] = True

    expected_path = "<field>" if location == "envelope" else "payload.<field>"
    with pytest.raises(
        BundleValidationError,
        match=rf"{expected_path}: extra_forbidden",
    ):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize("schema_version", [0, 1, 3, "2"])
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
        '"schema_version": 2',
        '"schema_version": 2, "schema_version": 2',
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
    ("mutate", "diagnostic"),
    (
        (
            lambda mapping: mapping.update(created_at="2026-07-14T01:35:00"),
            "created_at: invalid_timezone",
        ),
        (
            lambda mapping: mapping["payload"]["game_input"].update(
                captured_at="2026-07-14T01:30:00"
            ),
            "payload.game_input: invalid_value",
        ),
    ),
)
def test_bundle_rejects_naive_timestamps(mutate, diagnostic: str) -> None:
    mapping = projected_bundle_mapping()
    mutate(mapping)

    with pytest.raises(BundleValidationError, match=diagnostic):
        parse_projected_bundle_json(_raw(mapping))


@pytest.mark.parametrize(
    ("field_name", "value", "category"),
    (
        ("game_pk", "900001", "invalid_integer"),
        ("created_at", 123, "invalid_datetime"),
        ("created_at", "not-a-datetime", "invalid_datetime"),
    ),
)
def test_bundle_common_scalar_failures_use_fixed_safe_categories(
    field_name: str,
    value: object,
    category: str,
) -> None:
    mapping = projected_bundle_mapping()
    mapping[field_name] = value

    with pytest.raises(
        BundleValidationError,
        match=rf"{field_name}: {category}",
    ):
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

    with pytest.raises(BundleValidationError, match="bundle: invalid_value"):
        parse_projected_bundle_json(_raw(mapping))


def test_domain_construction_failures_map_to_bundle_validation_error() -> None:
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"]["identity"]["away_team"] = ""

    with pytest.raises(
        BundleValidationError,
        match="payload.game_input.identity: invalid_value",
    ) as captured:
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
    ("container_path", "field_name"),
    (
        (("official_result",), "supersedes_record_id"),
        (("closing_observations", 0), "available_at"),
        (("closing_observations", 0), "supersedes_record_id"),
    ),
)
def test_review_commands_require_audit_fields_even_when_nullable(
    container_path: tuple[str | int, ...],
    field_name: str,
) -> None:
    mapping = review_bundle_mapping()
    target: object = mapping["payload"]
    for path_part in container_path:
        target = target[path_part]
    assert isinstance(target, dict)
    target.pop(field_name)

    with pytest.raises(BundleValidationError, match=field_name):
        parse_review_bundle_json(_raw(mapping))


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

    with pytest.raises(BundleValidationError, match="bundle: invalid_value"):
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

    with pytest.raises(
        BundleValidationError,
        match="payload.*invalid_value",
    ):
        parse_review_bundle_json(_raw(mapping))


def test_review_closing_observation_must_match_bundle_game() -> None:
    mapping = review_bundle_mapping()
    mapping["payload"]["closing_observations"][0]["game_pk"] = GAME_PK + 1

    with pytest.raises(BundleValidationError, match="bundle: invalid_value"):
        parse_review_bundle_json(_raw(mapping))


def test_review_domain_errors_map_to_bundle_validation_error() -> None:
    mapping = review_bundle_mapping()
    mapping["payload"]["official_result"]["home_full_runs"] = 4

    with pytest.raises(
        BundleValidationError,
        match="payload.official_result: invalid_value",
    ) as captured:
        parse_review_bundle_json(_raw(mapping))

    assert captured.value.__cause__ is not None
