from dataclasses import FrozenInstanceError, fields, replace
from datetime import date, datetime, timezone
import math

import pytest

from analysis.forecasting import (
    BooleanFeature,
    BullpenInput,
    BullpenRole,
    BullpenSegmentInput,
    FeatureProvenance,
    GameIdentity,
    IntegerFeature,
    LineupInput,
    LineupSlot,
    LineupState,
    MarketInput,
    MarketType,
    NormalizedGameInput,
    NumericFeature,
    PitcherChainInput,
    PitcherInput,
    PitcherRole,
    PitchUsage,
    PitchVelocityResponse,
    RelieverAppearanceInput,
    SettlementFormat,
    SourceObservation,
    StarterProjectionComponents,
    WorkloadHistory,
)


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 14, 1, 30, tzinfo=UTC)
AVAILABLE_AT = datetime(2026, 7, 14, 1, 0, tzinfo=UTC)


def provenance() -> FeatureProvenance:
    return FeatureProvenance(
        observation_ids=("obs:1",),
        latest_available_at=AVAILABLE_AT,
        lookback_end=AVAILABLE_AT,
    )


def numeric(value: float = 1.0) -> NumericFeature:
    return NumericFeature(value=value, provenance=provenance())


def boolean(value: bool = False) -> BooleanFeature:
    return BooleanFeature(value=value, provenance=provenance())


def integer(value: int = 90) -> IntegerFeature:
    return IntegerFeature(value=value, provenance=provenance())


def workload(*outs: int) -> WorkloadHistory:
    values = outs or (15, 18, 12)
    return WorkloadHistory(recorded_outs=values, provenance=provenance())


def projection_components() -> StarterProjectionComponents:
    return StarterProjectionComponents(
        strikeout_minus_walk_rate=numeric(0.18),
        walk_rate=numeric(0.07),
        home_runs_per_contact=numeric(0.04),
        pitches_per_inning=numeric(16.2),
        recent_pitch_count=numeric(88.0),
        recent_workload_outs=numeric(16.0),
    )


def pitcher(
    pitcher_id: str = "pitcher:starter",
    role: PitcherRole = PitcherRole.STARTER,
) -> PitcherInput:
    return PitcherInput(
        pitcher_id=pitcher_id,
        pitcher_name=f"Pitcher {pitcher_id}",
        role=role,
        throws="R",
        projected_ra9=numeric(4.1),
        projection_components=projection_components(),
        log_standard_error=numeric(0.12),
        eligible_workload=workload(),
        league_prior_workload=workload(12, 15, 18),
        pitch_limit=integer(95),
        pitch_mix=(
            PitchUsage(
                pitcher_id=pitcher_id,
                opposing_side="R",
                pitch_type="FF",
                usage=0.55,
                average_velocity_mph=95.1,
                provenance=provenance(),
            ),
        ),
        command_risk=boolean(),
        home_run_risk=boolean(),
        injury_return=boolean(),
        recent_short_exit=boolean(),
    )


def conventional_chain(pitcher_id: str = "pitcher:starter") -> PitcherChainInput:
    chain_pitcher = pitcher(pitcher_id)
    return PitcherChainInput(
        pitchers=(chain_pitcher,),
        official_pitcher_id=pitcher_id,
        rotowire_pitcher_id=pitcher_id,
        matchup_pitcher_id=pitcher_id,
        starter_confirmed=boolean(True),
        starter_source_conflict=boolean(),
        handoff_verified=boolean(),
        timing_uncertain=boolean(),
        eligible_combined_workload=None,
        league_prior_combined_workload=None,
        combined_pitch_limit=None,
    )


def opener_bulk_chain(**overrides: object) -> PitcherChainInput:
    values: dict[str, object] = {
        "pitchers": (
            pitcher("pitcher:opener", PitcherRole.OPENER),
            pitcher("pitcher:bulk", PitcherRole.BULK),
        ),
        "official_pitcher_id": "pitcher:opener",
        "rotowire_pitcher_id": "pitcher:bulk",
        "matchup_pitcher_id": "pitcher:bulk",
        "starter_confirmed": boolean(True),
        "starter_source_conflict": boolean(),
        "handoff_verified": boolean(True),
        "timing_uncertain": boolean(),
        "eligible_combined_workload": workload(15, 18, 21),
        "league_prior_combined_workload": workload(12, 15, 18),
        "combined_pitch_limit": integer(105),
    }
    values.update(overrides)
    return PitcherChainInput(**values)


def chain_with_roles(*roles: PitcherRole) -> PitcherChainInput:
    role_pitchers = tuple(
        pitcher(f"pitcher:role-{index}", role)
        for index, role in enumerate(roles, start=1)
    )
    return opener_bulk_chain(
        pitchers=role_pitchers,
        official_pitcher_id=role_pitchers[0].pitcher_id,
        rotowire_pitcher_id=role_pitchers[-1].pitcher_id,
        matchup_pitcher_id=role_pitchers[-1].pitcher_id,
    )


def lineup_slot(order: int, *, batter_id: str | None = None) -> LineupSlot:
    resolved_id = batter_id or f"batter:{order}"
    return LineupSlot(
        order=order,
        batter_id=resolved_id,
        batter_name=f"Batter {order}",
        bats="S" if order == 1 else "R",
        expected_side="L" if order == 1 else "R",
        responses=(),
    )


def lineup(team: str = "SEA", slots: tuple[LineupSlot, ...] | None = None) -> LineupInput:
    return LineupInput(
        team=team,
        state=LineupState.CONFIRMED,
        state_provenance=provenance(),
        slots=slots or tuple(lineup_slot(order) for order in range(1, 10)),
        projected_runs_per_9=numeric(4.5),
        projected_traffic_per_9=numeric(11.5),
        projected_extra_base_hits_per_9=numeric(2.4),
        walk_rate=numeric(0.08),
        strikeout_rate=numeric(0.22),
        isolated_power=numeric(0.16),
        barrel_rate=numeric(0.08),
        recent_form_index=numeric(1.0),
        lower_order_continuity=numeric(1.0),
        log_standard_error=numeric(0.08),
        recent_form_included=boolean(True),
    )


def market_input(
    *,
    selected_decimal_price: float | None = 1.91,
    opposing_decimal_price: float | None = 1.91,
) -> MarketInput:
    return MarketInput(
        market_key="game:900001:ml:SEA",
        market_type=MarketType.ML,
        settlement_format=SettlementFormat.TWO_WAY,
        selected_side="SEA",
        line=None,
        selected_decimal_price=selected_decimal_price,
        opposing_decimal_price=opposing_decimal_price,
        provenance=provenance(),
    )


def bullpen(team: str) -> BullpenInput:
    return BullpenInput(
        team=team,
        segments=(
            BullpenSegmentInput(
                segment_id=f"{team}:high",
                role=BullpenRole.HIGH_LEVERAGE,
                projected_ra9=numeric(3.8),
                log_standard_error=numeric(0.15),
                appearances=(),
                available=boolean(True),
                limited_performance_or_use_risk=boolean(),
            ),
        ),
    )


def source_observation(**overrides: object) -> SourceObservation:
    values: dict[str, object] = {
        "observation_id": "obs:1",
        "source_name": "MLB",
        "source_record": "game/900001",
        "observed_at": AVAILABLE_AT,
        "available_at": AVAILABLE_AT,
        "fetched_at": datetime(2026, 7, 14, 1, 1, tzinfo=UTC),
        "effective_date": date(2026, 7, 14),
        "content_hash": "a" * 64,
    }
    values.update(overrides)
    return SourceObservation(**values)


def normalized_game(**overrides: object) -> NormalizedGameInput:
    values: dict[str, object] = {
        "identity": GameIdentity(
            game_pk=900001,
            korean_date=date(2026, 7, 15),
            official_game_date=date(2026, 7, 14),
            first_pitch=datetime(2026, 7, 14, 23, 10, tzinfo=UTC),
            away_team="SEA",
            home_team="MIN",
        ),
        "captured_at": CAPTURED_AT,
        "away_lineup": lineup("SEA"),
        "home_lineup": lineup("MIN"),
        "away_pitching": conventional_chain("pitcher:away"),
        "home_pitching": conventional_chain("pitcher:home"),
        "away_bullpen": bullpen("SEA"),
        "home_bullpen": bullpen("MIN"),
        "markets": (market_input(),),
        "observations": (source_observation(),),
    }
    values.update(overrides)
    return NormalizedGameInput(**values)


def test_string_enums_have_stable_values() -> None:
    assert [state.value for state in LineupState] == ["projected", "confirmed"]
    assert [role.value for role in PitcherRole] == ["starter", "opener", "bulk", "prim"]
    assert [role.value for role in BullpenRole] == [
        "bridge",
        "middle",
        "high_leverage",
        "closer",
    ]
    assert [market.value for market in MarketType] == [
        "f5_side",
        "f5_total",
        "ml",
        "plus_1_5",
        "full_total",
    ]
    assert [format_.value for format_ in SettlementFormat] == ["two_way", "three_way"]


@pytest.mark.parametrize("field_name", ["observed_at", "available_at", "fetched_at"])
def test_source_observation_rejects_naive_datetime(field_name: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        source_observation(**{field_name: datetime(2026, 7, 14, 1, 0)})


def test_source_observation_allows_unknown_available_at() -> None:
    observation = source_observation(available_at=None)

    assert observation.available_at is None


@pytest.mark.parametrize(
    "field_name", ["observation_id", "source_name", "source_record", "content_hash"]
)
def test_source_observation_rejects_empty_canonical_fields(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        source_observation(**{field_name: "   "})


@pytest.mark.parametrize("field_name", ["latest_available_at", "lookback_end"])
def test_feature_provenance_rejects_naive_datetime(field_name: str) -> None:
    values = {
        "observation_ids": ("obs:1",),
        "latest_available_at": AVAILABLE_AT,
        "lookback_end": AVAILABLE_AT,
    }
    values[field_name] = datetime(2026, 7, 14, 1, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        FeatureProvenance(**values)


@pytest.mark.parametrize("observation_ids", [(), ("",), ("obs:1", "obs:1")])
def test_feature_provenance_requires_nonempty_unique_observation_ids(
    observation_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="observation_ids"):
        FeatureProvenance(
            observation_ids=observation_ids,
            latest_available_at=AVAILABLE_AT,
            lookback_end=AVAILABLE_AT,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_numeric_feature_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumericFeature(value=value, provenance=provenance())


def test_integer_and_boolean_features_enforce_their_scalar_types() -> None:
    with pytest.raises(ValueError, match="integer"):
        IntegerFeature(value=True, provenance=provenance())
    with pytest.raises(ValueError, match="boolean"):
        BooleanFeature(value=1, provenance=provenance())


@pytest.mark.parametrize("usage", [-0.01, 1.01, math.nan])
def test_pitch_usage_must_be_a_finite_proportion(usage: float) -> None:
    with pytest.raises(ValueError, match="usage"):
        PitchUsage(
            pitcher_id="pitcher:1",
            opposing_side="R",
            pitch_type="FF",
            usage=usage,
            average_velocity_mph=95.0,
            provenance=provenance(),
        )


@pytest.mark.parametrize("opposing_side", ["S", "X", ""])
def test_pitch_usage_opposing_side_must_be_right_or_left(opposing_side: str) -> None:
    with pytest.raises(ValueError, match="opposing_side"):
        PitchUsage(
            pitcher_id="pitcher:1",
            opposing_side=opposing_side,
            pitch_type="FF",
            usage=0.5,
            average_velocity_mph=95.0,
            provenance=provenance(),
        )


def test_side_validation_rejects_non_string_with_value_error() -> None:
    with pytest.raises(ValueError, match="opposing_side"):
        PitchUsage(
            pitcher_id="pitcher:1",
            opposing_side=[],
            pitch_type="FF",
            usage=0.5,
            average_velocity_mph=95.0,
            provenance=provenance(),
        )


@pytest.mark.parametrize("band_width_mph", [1.5, 3.0])
def test_pitch_velocity_response_accepts_exact_and_fallback_bands(
    band_width_mph: float,
) -> None:
    response = PitchVelocityResponse(
        batter_id="batter:1",
        pitcher_id="pitcher:1",
        expected_side="L",
        pitch_type="FF",
        velocity_center_mph=95.0,
        band_width_mph=band_width_mph,
        plate_appearances=25,
        adjusted_woba=0.345,
        sample_strength="eligible",
        slot_label="플러스",
        provenance=provenance(),
    )

    assert response.band_width_mph == band_width_mph


@pytest.mark.parametrize("band_width_mph", [0.0, 2.0, 3.1, math.nan])
def test_pitch_velocity_response_rejects_other_bands(band_width_mph: float) -> None:
    with pytest.raises(ValueError, match="band_width_mph"):
        PitchVelocityResponse(
            batter_id="batter:1",
            pitcher_id="pitcher:1",
            expected_side="L",
            pitch_type="FF",
            velocity_center_mph=95.0,
            band_width_mph=band_width_mph,
            plate_appearances=25,
            adjusted_woba=0.345,
            sample_strength="eligible",
            slot_label="중립",
            provenance=provenance(),
        )


@pytest.mark.parametrize("slot_label", ["플러스", "중립", "마이너스", "표본 부족"])
def test_pitch_velocity_response_accepts_model_slot_labels(slot_label: str) -> None:
    response = PitchVelocityResponse(
        batter_id="batter:1",
        pitcher_id="pitcher:1",
        expected_side="R",
        pitch_type="SL",
        velocity_center_mph=87.0,
        band_width_mph=1.5,
        plate_appearances=0,
        adjusted_woba=0.320,
        sample_strength="limited",
        slot_label=slot_label,
        provenance=provenance(),
    )

    assert response.slot_label == slot_label


def test_pitch_velocity_response_rejects_non_string_slot_label_with_value_error() -> None:
    with pytest.raises(ValueError, match="slot_label"):
        PitchVelocityResponse(
            batter_id="batter:1",
            pitcher_id="pitcher:1",
            expected_side="R",
            pitch_type="SL",
            velocity_center_mph=87.0,
            band_width_mph=1.5,
            plate_appearances=10,
            adjusted_woba=0.320,
            sample_strength="eligible",
            slot_label=[],
            provenance=provenance(),
        )


def test_pitch_velocity_response_rejects_switch_expected_side_and_negative_sample() -> None:
    values = {
        "batter_id": "batter:1",
        "pitcher_id": "pitcher:1",
        "expected_side": "R",
        "pitch_type": "SL",
        "velocity_center_mph": 87.0,
        "band_width_mph": 1.5,
        "plate_appearances": 10,
        "adjusted_woba": 0.320,
        "sample_strength": "eligible",
        "slot_label": "중립",
        "provenance": provenance(),
    }
    with pytest.raises(ValueError, match="expected_side"):
        PitchVelocityResponse(**(values | {"expected_side": "S"}))
    with pytest.raises(ValueError, match="plate_appearances"):
        PitchVelocityResponse(**(values | {"plate_appearances": -1}))


def test_lineup_slot_allows_switch_bats_but_not_switch_expected_side() -> None:
    assert lineup_slot(1).bats == "S"
    with pytest.raises(ValueError, match="expected_side"):
        replace(lineup_slot(1), expected_side="S")


def test_lineup_requires_exactly_nine_slots() -> None:
    slots = tuple(lineup_slot(order) for order in range(1, 9))

    with pytest.raises(ValueError, match="exactly nine"):
        lineup(slots=slots)


def test_lineup_requires_unique_order_values_one_through_nine() -> None:
    slots = tuple(lineup_slot(order) for order in range(1, 9)) + (lineup_slot(8),)

    with pytest.raises(ValueError, match="1 through 9"):
        lineup(slots=slots)


@pytest.mark.parametrize("recorded_outs", [(-1,), (28,), (True,)])
def test_workload_outs_must_be_integers_from_zero_through_27(
    recorded_outs: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="recorded_outs"):
        WorkloadHistory(recorded_outs=recorded_outs, provenance=provenance())


@pytest.mark.parametrize(
    ("days_ago", "pitches"),
    [(-1, 10), (0, -1), (True, 10), (0, True)],
)
def test_reliever_appearance_requires_nonnegative_integer_usage(
    days_ago: object,
    pitches: object,
) -> None:
    with pytest.raises(ValueError, match="days_ago|pitches"):
        RelieverAppearanceInput(
            pitcher_id="reliever:1",
            days_ago=days_ago,
            pitches=pitches,
            provenance=provenance(),
        )


@pytest.mark.parametrize(
    ("field_name", "unknown_id"),
    [
        ("official_pitcher_id", "pitcher:unknown-official"),
        ("rotowire_pitcher_id", "pitcher:unknown-rotowire"),
        ("matchup_pitcher_id", "pitcher:unknown-matchup"),
    ],
)
def test_pitcher_chain_source_ids_must_reference_chain_members(
    field_name: str,
    unknown_id: str,
) -> None:
    chain = conventional_chain()

    with pytest.raises(ValueError, match=field_name):
        replace(chain, **{field_name: unknown_id})


def test_pitcher_chain_rejects_duplicate_pitcher_ids() -> None:
    duplicate = pitcher("pitcher:duplicate", PitcherRole.OPENER)

    with pytest.raises(ValueError, match="unique pitcher_id"):
        opener_bulk_chain(pitchers=(duplicate, duplicate))


def test_multi_pitcher_chain_rejects_reversed_bulk_opener_roles() -> None:
    with pytest.raises(ValueError, match="first pitcher role must be opener"):
        chain_with_roles(PitcherRole.BULK, PitcherRole.OPENER)


def test_multi_pitcher_chain_rejects_opener_then_starter() -> None:
    with pytest.raises(ValueError, match="later pitcher roles must be bulk or prim"):
        chain_with_roles(PitcherRole.OPENER, PitcherRole.STARTER)


def test_multi_pitcher_chain_rejects_multi_starter_roles() -> None:
    with pytest.raises(ValueError, match="first pitcher role must be opener"):
        chain_with_roles(PitcherRole.STARTER, PitcherRole.STARTER)


@pytest.mark.parametrize("duplicate_role", [PitcherRole.BULK, PitcherRole.PRIM])
def test_multi_pitcher_chain_rejects_duplicate_later_roles(
    duplicate_role: PitcherRole,
) -> None:
    with pytest.raises(ValueError, match="later pitcher roles must be distinct"):
        chain_with_roles(PitcherRole.OPENER, duplicate_role, duplicate_role)


@pytest.mark.parametrize(
    "roles",
    [
        (PitcherRole.OPENER, PitcherRole.BULK),
        (PitcherRole.OPENER, PitcherRole.PRIM),
        (PitcherRole.OPENER, PitcherRole.BULK, PitcherRole.PRIM),
    ],
)
def test_multi_pitcher_chain_accepts_legal_role_sequences(
    roles: tuple[PitcherRole, ...],
) -> None:
    chain = chain_with_roles(*roles)

    assert tuple(pitcher_input.role for pitcher_input in chain.pitchers) == roles


@pytest.mark.parametrize(
    "missing_field",
    ["eligible_combined_workload", "league_prior_combined_workload"],
)
def test_opener_bulk_chain_requires_both_combined_workload_histories(
    missing_field: str,
) -> None:
    with pytest.raises(ValueError, match=missing_field):
        opener_bulk_chain(**{missing_field: None})


def test_opener_bulk_chain_uses_actual_pitcher_roles_and_combined_workload() -> None:
    chain = opener_bulk_chain()

    assert [pitcher_input.role for pitcher_input in chain.pitchers] == [
        PitcherRole.OPENER,
        PitcherRole.BULK,
    ]
    assert chain.matchup_pitcher_id == "pitcher:bulk"


@pytest.mark.parametrize(
    "combined_field",
    [
        "eligible_combined_workload",
        "league_prior_combined_workload",
        "combined_pitch_limit",
    ],
)
def test_single_conventional_starter_rejects_combined_fields(
    combined_field: str,
) -> None:
    value = integer(100) if combined_field == "combined_pitch_limit" else workload()

    with pytest.raises(ValueError, match="single conventional starter"):
        replace(conventional_chain(), **{combined_field: value})


def test_pitcher_chain_has_only_role_based_structure_and_no_strength_penalty() -> None:
    assert [field.name for field in fields(PitcherChainInput)] == [
        "pitchers",
        "official_pitcher_id",
        "rotowire_pitcher_id",
        "matchup_pitcher_id",
        "starter_confirmed",
        "starter_source_conflict",
        "handoff_verified",
        "timing_uncertain",
        "eligible_combined_workload",
        "league_prior_combined_workload",
        "combined_pitch_limit",
    ]


@pytest.mark.parametrize("price_name", ["selected_decimal_price", "opposing_decimal_price"])
@pytest.mark.parametrize("price", [0.0, -1.0, math.nan, math.inf])
def test_market_prices_must_be_positive_and_finite_when_present(
    price_name: str,
    price: float,
) -> None:
    values = {
        "selected_decimal_price": 1.91,
        "opposing_decimal_price": 1.91,
    }
    values[price_name] = price

    with pytest.raises(ValueError, match=price_name):
        market_input(**values)


def test_market_input_accepts_both_prices_as_none() -> None:
    market = market_input(selected_decimal_price=None, opposing_decimal_price=None)

    assert market.selected_decimal_price is None
    assert market.opposing_decimal_price is None


def test_game_identity_preserves_korean_date_distinct_from_official_date() -> None:
    identity = GameIdentity(
        game_pk=900001,
        korean_date=date(2026, 7, 15),
        official_game_date=date(2026, 7, 14),
        first_pitch=datetime(2026, 7, 14, 23, 10, tzinfo=UTC),
        away_team="SEA",
        home_team="MIN",
    )

    assert identity.korean_date == date(2026, 7, 15)
    assert identity.official_game_date == date(2026, 7, 14)


def test_game_identity_rejects_naive_first_pitch() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        GameIdentity(
            game_pk=900001,
            korean_date=date(2026, 7, 15),
            official_game_date=date(2026, 7, 14),
            first_pitch=datetime(2026, 7, 14, 23, 10),
            away_team="SEA",
            home_team="MIN",
        )


def test_normalized_game_rejects_naive_capture_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        normalized_game(captured_at=datetime(2026, 7, 14, 1, 30))


def test_schema_is_frozen() -> None:
    observation = source_observation()

    with pytest.raises(FrozenInstanceError):
        observation.source_name = "changed"
