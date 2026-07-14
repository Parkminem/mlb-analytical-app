from dataclasses import dataclass, FrozenInstanceError, fields, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import unittest

import analysis.forecasting.identity as identity_module
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
from analysis.forecasting.engine import resolve_forecast_input
from analysis.forecasting.identity import (
    build_forecast_run_id,
    canonical_json,
    capture_payload,
    normalized_input_hash,
    semantic_forecast_payload,
)
from analysis.forecasting.validation import (
    iter_feature_provenance,
    validate_pregame_input,
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


def complete_chain(team: str) -> PitcherChainInput:
    opener_id = f"pitcher:{team}:opener"
    bulk_id = f"pitcher:{team}:bulk"
    return opener_bulk_chain(
        pitchers=(
            pitcher(opener_id, PitcherRole.OPENER),
            pitcher(bulk_id, PitcherRole.BULK),
        ),
        official_pitcher_id=opener_id,
        rotowire_pitcher_id=bulk_id,
        matchup_pitcher_id=bulk_id,
    )


def lineup_against(team: str, pitcher_id: str) -> LineupInput:
    slots = []
    for order in range(1, 10):
        slot = lineup_slot(order, batter_id=f"batter:{team}:{order}")
        response = PitchVelocityResponse(
            batter_id=slot.batter_id,
            pitcher_id=pitcher_id,
            expected_side=slot.expected_side,
            pitch_type="FF",
            velocity_center_mph=95.1,
            band_width_mph=1.5,
            plate_appearances=30,
            adjusted_woba=0.340,
            sample_strength="eligible",
            slot_label="플러스",
            provenance=provenance(),
        )
        slots.append(replace(slot, responses=(response,)))
    return lineup(team, tuple(slots))


def bullpen_with_appearance(team: str) -> BullpenInput:
    base = bullpen(team)
    segment = replace(
        base.segments[0],
        appearances=(
            RelieverAppearanceInput(
                pitcher_id=f"reliever:{team}:1",
                days_ago=1,
                pitches=18,
                provenance=provenance(),
            ),
        ),
    )
    return replace(base, segments=(segment,))


def complete_forecast_game(**overrides: object) -> NormalizedGameInput:
    away_pitching = complete_chain("SEA")
    home_pitching = complete_chain("MIN")
    total_market = replace(
        market_input(),
        market_key="game:900001:full_total:over:8.0",
        market_type=MarketType.FULL_TOTAL,
        selected_side="over",
        line=8.0,
    )
    values: dict[str, object] = {
        "away_pitching": away_pitching,
        "home_pitching": home_pitching,
        "away_lineup": lineup_against("SEA", home_pitching.matchup_pitcher_id),
        "home_lineup": lineup_against("MIN", away_pitching.matchup_pitcher_id),
        "away_bullpen": bullpen_with_appearance("SEA"),
        "home_bullpen": bullpen_with_appearance("MIN"),
        "markets": (total_market,),
    }
    values.update(overrides)
    return normalized_game(**values)


def replace_all_provenance(
    value: object,
    replacement: FeatureProvenance,
) -> object:
    if isinstance(value, FeatureProvenance):
        return replacement
    if isinstance(value, tuple):
        return tuple(replace_all_provenance(item, replacement) for item in value)
    if is_dataclass(value):
        changes = {
            field.name: replace_all_provenance(getattr(value, field.name), replacement)
            for field in fields(value)
        }
        return replace(value, **changes)
    return value


def replace_lineup_slot(
    lineup_input: LineupInput,
    index: int,
    slot: LineupSlot,
) -> LineupInput:
    slots = list(lineup_input.slots)
    slots[index] = slot
    return replace(lineup_input, slots=tuple(slots))


def replace_chain_pitcher(
    chain: PitcherChainInput,
    index: int,
    pitcher_input: PitcherInput,
) -> PitcherChainInput:
    pitchers = list(chain.pitchers)
    pitchers[index] = pitcher_input
    return replace(chain, pitchers=tuple(pitchers))


class ForecastInputSchemaTests(unittest.TestCase):
    def test_string_enums_have_stable_values(self) -> None:
        self.assertEqual(
            [state.value for state in LineupState],
            ["projected", "confirmed"],
        )
        self.assertEqual(
            [role.value for role in PitcherRole],
            ["starter", "opener", "bulk", "prim"],
        )
        self.assertEqual(
            [role.value for role in BullpenRole],
            ["bridge", "middle", "high_leverage", "closer"],
        )
        self.assertEqual(
            [market.value for market in MarketType],
            ["f5_side", "f5_total", "ml", "plus_1_5", "full_total"],
        )
        self.assertEqual(
            [format_.value for format_ in SettlementFormat],
            ["two_way", "three_way"],
        )

    def test_source_observation_rejects_naive_datetime(self) -> None:
        for field_name in ("observed_at", "available_at", "fetched_at"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "timezone-aware"):
                    source_observation(
                        **{field_name: datetime(2026, 7, 14, 1, 0)}
                    )

    def test_source_observation_allows_unknown_available_at(self) -> None:
        observation = source_observation(available_at=None)

        self.assertIsNone(observation.available_at)

    def test_source_observation_rejects_empty_canonical_fields(self) -> None:
        for field_name in (
            "observation_id",
            "source_name",
            "source_record",
            "content_hash",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    source_observation(**{field_name: "   "})

    def test_feature_provenance_rejects_naive_datetime(self) -> None:
        for field_name in ("latest_available_at", "lookback_end"):
            with self.subTest(field_name=field_name):
                values = {
                    "observation_ids": ("obs:1",),
                    "latest_available_at": AVAILABLE_AT,
                    "lookback_end": AVAILABLE_AT,
                }
                values[field_name] = datetime(2026, 7, 14, 1, 0)

                with self.assertRaisesRegex(ValueError, "timezone-aware"):
                    FeatureProvenance(**values)

    def test_feature_provenance_requires_nonempty_unique_observation_ids(
        self,
    ) -> None:
        for observation_ids in ((), ("",), ("obs:1", "obs:1")):
            with self.subTest(observation_ids=observation_ids):
                with self.assertRaisesRegex(ValueError, "observation_ids"):
                    FeatureProvenance(
                        observation_ids=observation_ids,
                        latest_available_at=AVAILABLE_AT,
                        lookback_end=AVAILABLE_AT,
                    )

    def test_numeric_feature_rejects_nonfinite_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    NumericFeature(value=value, provenance=provenance())

    def test_integer_and_boolean_features_enforce_their_scalar_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            IntegerFeature(value=True, provenance=provenance())
        with self.assertRaisesRegex(ValueError, "boolean"):
            BooleanFeature(value=1, provenance=provenance())

    def test_pitch_usage_must_be_a_finite_proportion(self) -> None:
        for usage in (-0.01, 1.01, math.nan):
            with self.subTest(usage=usage):
                with self.assertRaisesRegex(ValueError, "usage"):
                    PitchUsage(
                        pitcher_id="pitcher:1",
                        opposing_side="R",
                        pitch_type="FF",
                        usage=usage,
                        average_velocity_mph=95.0,
                        provenance=provenance(),
                    )

    def test_pitch_usage_opposing_side_must_be_right_or_left(self) -> None:
        for opposing_side in ("S", "X", ""):
            with self.subTest(opposing_side=opposing_side):
                with self.assertRaisesRegex(ValueError, "opposing_side"):
                    PitchUsage(
                        pitcher_id="pitcher:1",
                        opposing_side=opposing_side,
                        pitch_type="FF",
                        usage=0.5,
                        average_velocity_mph=95.0,
                        provenance=provenance(),
                    )

    def test_side_validation_rejects_non_string_with_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "opposing_side"):
            PitchUsage(
                pitcher_id="pitcher:1",
                opposing_side=[],
                pitch_type="FF",
                usage=0.5,
                average_velocity_mph=95.0,
                provenance=provenance(),
            )

    def test_pitch_velocity_response_accepts_exact_and_fallback_bands(self) -> None:
        for band_width_mph in (1.5, 3.0):
            with self.subTest(band_width_mph=band_width_mph):
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

                self.assertEqual(response.band_width_mph, band_width_mph)

    def test_pitch_velocity_response_rejects_other_bands(self) -> None:
        for band_width_mph in (0.0, 2.0, 3.1, math.nan):
            with self.subTest(band_width_mph=band_width_mph):
                with self.assertRaisesRegex(ValueError, "band_width_mph"):
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

    def test_pitch_velocity_response_accepts_model_slot_labels(self) -> None:
        for slot_label in ("플러스", "중립", "마이너스", "표본 부족"):
            with self.subTest(slot_label=slot_label):
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

                self.assertEqual(response.slot_label, slot_label)

    def test_pitch_velocity_response_rejects_non_string_slot_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "slot_label"):
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

    def test_pitch_velocity_response_rejects_invalid_side_and_sample(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "expected_side"):
            PitchVelocityResponse(**(values | {"expected_side": "S"}))
        with self.assertRaisesRegex(ValueError, "plate_appearances"):
            PitchVelocityResponse(**(values | {"plate_appearances": -1}))

    def test_lineup_slot_allows_switch_bats_but_not_switch_expected_side(
        self,
    ) -> None:
        self.assertEqual(lineup_slot(1).bats, "S")
        with self.assertRaisesRegex(ValueError, "expected_side"):
            replace(lineup_slot(1), expected_side="S")

    def test_lineup_slot_rejects_fixed_batter_side_mismatch(self) -> None:
        for bats, expected_side in (("R", "L"), ("L", "R")):
            with self.subTest(bats=bats, expected_side=expected_side):
                with self.assertRaisesRegex(ValueError, "expected_side"):
                    replace(
                        lineup_slot(2),
                        bats=bats,
                        expected_side=expected_side,
                    )

    def test_lineup_slot_accepts_switch_batter_expected_sides(self) -> None:
        for expected_side in ("R", "L"):
            with self.subTest(expected_side=expected_side):
                slot = replace(
                    lineup_slot(1),
                    bats="S",
                    expected_side=expected_side,
                )
                self.assertEqual(slot.expected_side, expected_side)

    def test_lineup_requires_exactly_nine_slots(self) -> None:
        slots = tuple(lineup_slot(order) for order in range(1, 9))

        with self.assertRaisesRegex(ValueError, "exactly nine"):
            lineup(slots=slots)

    def test_lineup_requires_unique_order_values_one_through_nine(self) -> None:
        slots = tuple(lineup_slot(order) for order in range(1, 9)) + (
            lineup_slot(8),
        )

        with self.assertRaisesRegex(ValueError, "1 through 9"):
            lineup(slots=slots)

    def test_lineup_rejects_duplicate_batter_ids(self) -> None:
        slots = tuple(lineup_slot(order) for order in range(1, 9)) + (
            lineup_slot(9, batter_id="batter:1"),
        )

        with self.assertRaisesRegex(ValueError, "unique batter_id"):
            lineup(slots=slots)

    def test_workload_outs_must_be_integers_from_zero_through_27(self) -> None:
        for recorded_outs in ((-1,), (28,), (True,)):
            with self.subTest(recorded_outs=recorded_outs):
                with self.assertRaisesRegex(ValueError, "recorded_outs"):
                    WorkloadHistory(
                        recorded_outs=recorded_outs,
                        provenance=provenance(),
                    )

    def test_reliever_appearance_requires_nonnegative_integer_usage(self) -> None:
        for days_ago, pitches in ((-1, 10), (0, -1), (True, 10), (0, True)):
            with self.subTest(days_ago=days_ago, pitches=pitches):
                with self.assertRaisesRegex(ValueError, "days_ago|pitches"):
                    RelieverAppearanceInput(
                        pitcher_id="reliever:1",
                        days_ago=days_ago,
                        pitches=pitches,
                        provenance=provenance(),
                    )

    def test_pitcher_chain_source_ids_must_reference_chain_members(self) -> None:
        chain = conventional_chain()
        references = (
            ("official_pitcher_id", "pitcher:unknown-official"),
            ("rotowire_pitcher_id", "pitcher:unknown-rotowire"),
            ("matchup_pitcher_id", "pitcher:unknown-matchup"),
        )
        for field_name, unknown_id in references:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    replace(chain, **{field_name: unknown_id})

    def test_pitcher_chain_rejects_duplicate_pitcher_ids(self) -> None:
        duplicate = pitcher("pitcher:duplicate", PitcherRole.OPENER)

        with self.assertRaisesRegex(ValueError, "unique pitcher_id"):
            opener_bulk_chain(pitchers=(duplicate, duplicate))

    def test_multi_pitcher_chain_rejects_reversed_bulk_opener_roles(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "first pitcher role must be opener",
        ):
            chain_with_roles(PitcherRole.BULK, PitcherRole.OPENER)

    def test_multi_pitcher_chain_rejects_opener_then_starter(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "later pitcher roles must be bulk or prim",
        ):
            chain_with_roles(PitcherRole.OPENER, PitcherRole.STARTER)

    def test_multi_pitcher_chain_rejects_multi_starter_roles(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "first pitcher role must be opener",
        ):
            chain_with_roles(PitcherRole.STARTER, PitcherRole.STARTER)

    def test_multi_pitcher_chain_rejects_duplicate_later_roles(self) -> None:
        for duplicate_role in (PitcherRole.BULK, PitcherRole.PRIM):
            with self.subTest(duplicate_role=duplicate_role):
                with self.assertRaisesRegex(
                    ValueError,
                    "later pitcher roles must be distinct",
                ):
                    chain_with_roles(
                        PitcherRole.OPENER,
                        duplicate_role,
                        duplicate_role,
                    )

    def test_multi_pitcher_chain_accepts_legal_role_sequences(self) -> None:
        role_sequences = (
            (PitcherRole.OPENER, PitcherRole.BULK),
            (PitcherRole.OPENER, PitcherRole.PRIM),
            (PitcherRole.OPENER, PitcherRole.BULK, PitcherRole.PRIM),
        )
        for roles in role_sequences:
            with self.subTest(roles=roles):
                chain = chain_with_roles(*roles)
                self.assertEqual(
                    tuple(pitcher_input.role for pitcher_input in chain.pitchers),
                    roles,
                )

    def test_opener_chain_requires_both_combined_workload_histories(self) -> None:
        for missing_field in (
            "eligible_combined_workload",
            "league_prior_combined_workload",
        ):
            with self.subTest(missing_field=missing_field):
                with self.assertRaisesRegex(ValueError, missing_field):
                    opener_bulk_chain(**{missing_field: None})

    def test_opener_bulk_chain_uses_roles_and_combined_workload(self) -> None:
        chain = opener_bulk_chain()

        self.assertEqual(
            [pitcher_input.role for pitcher_input in chain.pitchers],
            [PitcherRole.OPENER, PitcherRole.BULK],
        )
        self.assertEqual(chain.matchup_pitcher_id, "pitcher:bulk")

    def test_single_conventional_starter_rejects_combined_fields(self) -> None:
        for combined_field in (
            "eligible_combined_workload",
            "league_prior_combined_workload",
            "combined_pitch_limit",
        ):
            with self.subTest(combined_field=combined_field):
                value = (
                    integer(100)
                    if combined_field == "combined_pitch_limit"
                    else workload()
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "single conventional starter",
                ):
                    replace(
                        conventional_chain(),
                        **{combined_field: value},
                    )

    def test_pitcher_chain_has_only_role_based_structure(self) -> None:
        self.assertEqual(
            [field.name for field in fields(PitcherChainInput)],
            [
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
            ],
        )

    def test_multi_pitcher_chain_rejects_opener_as_matchup_pitcher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "matchup_pitcher_id.*bulk or prim",
        ):
            replace(
                opener_bulk_chain(),
                matchup_pitcher_id="pitcher:opener",
            )

    def test_multi_pitcher_chain_accepts_bulk_and_prim_matchup_pitchers(
        self,
    ) -> None:
        for role in (PitcherRole.BULK, PitcherRole.PRIM):
            with self.subTest(role=role):
                chain = chain_with_roles(PitcherRole.OPENER, role)
                matchup_pitcher = next(
                    pitcher_input
                    for pitcher_input in chain.pitchers
                    if pitcher_input.pitcher_id == chain.matchup_pitcher_id
                )
                self.assertEqual(matchup_pitcher.role, role)

    def test_market_prices_must_be_positive_and_finite_when_present(self) -> None:
        for price_name in (
            "selected_decimal_price",
            "opposing_decimal_price",
        ):
            for price in (0.0, -1.0, math.nan, math.inf):
                with self.subTest(price_name=price_name, price=price):
                    values = {
                        "selected_decimal_price": 1.91,
                        "opposing_decimal_price": 1.91,
                    }
                    values[price_name] = price

                    with self.assertRaisesRegex(ValueError, price_name):
                        market_input(**values)

    def test_market_input_accepts_both_prices_as_none(self) -> None:
        market = market_input(
            selected_decimal_price=None,
            opposing_decimal_price=None,
        )

        self.assertIsNone(market.selected_decimal_price)
        self.assertIsNone(market.opposing_decimal_price)

    def test_game_identity_preserves_korean_date_distinct_from_official(
        self,
    ) -> None:
        identity = GameIdentity(
            game_pk=900001,
            korean_date=date(2026, 7, 15),
            official_game_date=date(2026, 7, 14),
            first_pitch=datetime(2026, 7, 14, 23, 10, tzinfo=UTC),
            away_team="SEA",
            home_team="MIN",
        )

        self.assertEqual(identity.korean_date, date(2026, 7, 15))
        self.assertEqual(identity.official_game_date, date(2026, 7, 14))

    def test_game_identity_rejects_naive_first_pitch(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            GameIdentity(
                game_pk=900001,
                korean_date=date(2026, 7, 15),
                official_game_date=date(2026, 7, 14),
                first_pitch=datetime(2026, 7, 14, 23, 10),
                away_team="SEA",
                home_team="MIN",
            )

    def test_normalized_game_rejects_naive_capture_timestamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            normalized_game(captured_at=datetime(2026, 7, 14, 1, 30))

    def test_schema_is_frozen(self) -> None:
        observation = source_observation()

        with self.assertRaises(FrozenInstanceError):
            observation.source_name = "changed"


class CanonicalNumber(Enum):
    TWO = 2


@dataclass(frozen=True)
class CanonicalFixture:
    when: datetime
    day: date
    state: LineupState
    values: tuple[object, ...]


@dataclass
class MutableFixture:
    value: int


class ForecastIdentityTests(unittest.TestCase):
    def test_canonical_json_normalizes_supported_values(self) -> None:
        fixture = CanonicalFixture(
            when=datetime(
                2026,
                7,
                14,
                10,
                0,
                tzinfo=timezone(timedelta(hours=9)),
            ),
            day=date(2026, 7, 14),
            state=LineupState.CONFIRMED,
            values=(1, CanonicalNumber.TWO),
        )
        equivalent_mapping = {
            "values": [1, 2],
            "state": "confirmed",
            "day": "2026-07-14",
            "when": datetime(2026, 7, 14, 1, 0, tzinfo=UTC),
        }

        self.assertEqual(
            canonical_json(fixture),
            canonical_json(equivalent_mapping),
        )
        self.assertEqual(
            canonical_json(fixture),
            '{"day":"2026-07-14","state":"confirmed",'
            '"values":[1,2],"when":"2026-07-14T01:00:00Z"}',
        )
        self.assertEqual(
            canonical_json({"b": 2, "a": {"d": 4, "c": 3}}),
            canonical_json({"a": {"c": 3, "d": 4}, "b": 2}),
        )

    def test_canonical_json_rejects_unsupported_and_nonfinite_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    canonical_json({"value": value})

        for value in (object(), {1, 2}, MutableFixture(1)):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TypeError):
                    canonical_json(value)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            canonical_json(datetime(2026, 7, 14, 1, 0))

    def test_canonical_json_normalizes_integral_floats_as_integers(self) -> None:
        for integer_value, float_value in (
            (1, 1.0),
            (0, -0.0),
            (-2, -2.0),
        ):
            with self.subTest(float_value=float_value):
                self.assertEqual(
                    canonical_json({"value": integer_value}),
                    canonical_json({"value": float_value}),
                )

        self.assertEqual(canonical_json({"value": 1.25}), '{"value":1.25}')
        self.assertNotEqual(
            canonical_json({"value": True}),
            canonical_json({"value": 1}),
        )

    def test_integral_float_config_does_not_change_forecast_id(self) -> None:
        game = complete_forecast_game()
        integer_config = {
            "simulations": 20_000,
            "weights": {"starter": 1},
        }
        float_config = {
            "simulations": 20_000.0,
            "weights": {"starter": 1.0},
        }

        self.assertEqual(
            build_forecast_run_id(game, "shadow-1", integer_config),
            build_forecast_run_id(game, "shadow-1", float_config),
        )

    def test_normalized_input_hash_hashes_the_semantic_payload(self) -> None:
        game = complete_forecast_game()
        expected = hashlib.sha256(
            canonical_json(semantic_forecast_payload(game)).encode("utf-8")
        ).hexdigest()

        self.assertEqual(normalized_input_hash(game), expected)
        self.assertEqual(len(expected), 64)

    def test_forecast_id_ignores_audit_timing_and_generated_ids(self) -> None:
        baseline = complete_forecast_game()
        replacement_provenance = FeatureProvenance(
            observation_ids=("obs:renamed",),
            latest_available_at=datetime(2026, 7, 14, 1, 20, tzinfo=UTC),
            lookback_end=datetime(2026, 7, 14, 0, 45, tzinfo=UTC),
        )
        changed = replace_all_provenance(baseline, replacement_provenance)
        self.assertIsInstance(changed, NormalizedGameInput)
        changed_observation = replace(
            baseline.observations[0],
            observation_id="obs:renamed",
            observed_at=datetime(2026, 7, 14, 1, 21, tzinfo=UTC),
            available_at=datetime(2026, 7, 14, 1, 20, tzinfo=UTC),
            fetched_at=datetime(2026, 7, 14, 1, 50, tzinfo=UTC),
        )
        changed = replace(
            changed,
            captured_at=datetime(2026, 7, 14, 1, 45, tzinfo=UTC),
            observations=(changed_observation,),
        )
        config = {"simulations": 20_000, "seed_mode": "forecast"}

        self.assertEqual(
            normalized_input_hash(baseline),
            normalized_input_hash(changed),
        )
        self.assertEqual(
            build_forecast_run_id(baseline, "shadow-1", config),
            build_forecast_run_id(changed, "shadow-1", config),
        )

    def test_source_order_and_config_mapping_order_do_not_change_id(self) -> None:
        game = complete_forecast_game()
        second_observation = source_observation(
            observation_id="obs:2",
            source_name="RotoWire",
            source_record="lineups/900001",
            content_hash="b" * 64,
        )
        first = replace(
            game,
            observations=(game.observations[0], second_observation),
        )
        second = replace(
            game,
            observations=(second_observation, game.observations[0]),
        )
        first_config = {
            "simulations": 20_000,
            "distribution": {"away": 1, "home": 2},
        }
        second_config = {
            "distribution": {"home": 2, "away": 1},
            "simulations": 20_000,
        }

        self.assertEqual(
            semantic_forecast_payload(first),
            semantic_forecast_payload(second),
        )
        self.assertEqual(
            build_forecast_run_id(first, "shadow-1", first_config),
            build_forecast_run_id(second, "shadow-1", second_config),
        )

    def test_market_tuple_order_does_not_change_semantic_identity(self) -> None:
        game = complete_forecast_game()
        second_market = market_input()
        first = replace(
            game,
            markets=(game.markets[0], second_market),
        )
        reordered = replace(
            game,
            markets=(second_market, game.markets[0]),
        )
        first_capture_keys = [
            market["market_key"] for market in capture_payload(first)["markets"]
        ]
        reordered_capture_keys = [
            market["market_key"]
            for market in capture_payload(reordered)["markets"]
        ]

        self.assertEqual(
            first_capture_keys,
            [game.markets[0].market_key, second_market.market_key],
        )
        self.assertEqual(
            reordered_capture_keys,
            [second_market.market_key, game.markets[0].market_key],
        )
        with self.subTest(result="semantic payload"):
            self.assertEqual(
                semantic_forecast_payload(first),
                semantic_forecast_payload(reordered),
            )
        with self.subTest(result="forecast id"):
            self.assertEqual(
                build_forecast_run_id(first, "shadow-1", {"simulations": 20_000}),
                build_forecast_run_id(
                    reordered,
                    "shadow-1",
                    {"simulations": 20_000},
                ),
            )

    def test_semantic_source_aliases_do_not_change_identity(self) -> None:
        game = complete_forecast_game()
        alias = replace(
            game.observations[0],
            observation_id="obs:generated-alias",
        )
        with_alias = replace(
            game,
            observations=(alias, game.observations[0]),
        )
        captured_observations = capture_payload(with_alias)["observations"]

        self.assertEqual(
            [observation["observation_id"] for observation in captured_observations],
            ["obs:generated-alias", "obs:1"],
        )
        with self.subTest(result="semantic payload"):
            self.assertEqual(
                semantic_forecast_payload(game),
                semantic_forecast_payload(with_alias),
            )
        with self.subTest(result="forecast id"):
            self.assertEqual(
                build_forecast_run_id(
                    game,
                    "shadow-1",
                    {"simulations": 20_000},
                ),
                build_forecast_run_id(
                    with_alias,
                    "shadow-1",
                    {"simulations": 20_000},
                ),
            )

    def test_capture_payload_rebuilds_exact_resolved_semantic_identity(self):
        semantic_from_capture = getattr(
            identity_module,
            "semantic_forecast_payload_from_capture_payload",
            None,
        )
        run_id_from_capture = getattr(
            identity_module,
            "build_forecast_run_id_from_capture_payload",
            None,
        )
        self.assertTrue(callable(semantic_from_capture))
        self.assertTrue(callable(run_id_from_capture))
        game = normalized_game()
        alias_observation = replace(
            game.observations[0],
            observation_id="obs:semantic-alias",
        )
        total_market = replace(
            game.markets[0],
            market_key="game:900001:full_total:under:8.5",
            market_type=MarketType.FULL_TOTAL,
            selected_side="under",
            line=8.5,
        )
        captured_game = replace(
            game,
            markets=(total_market, game.markets[0]),
            observations=(alias_observation, game.observations[0]),
        )
        resolved_game = resolve_forecast_input(captured_game)
        config = {"simulations": 20_000, "seed_mode": "forecast"}

        rebuilt = semantic_from_capture(capture_payload(captured_game))

        self.assertEqual(
            canonical_json(rebuilt),
            canonical_json(semantic_forecast_payload(resolved_game)),
        )
        self.assertNotIn("captured_at", rebuilt)
        self.assertEqual(
            run_id_from_capture(
                capture_payload(captured_game),
                "shadow-1",
                config,
            ),
            build_forecast_run_id(resolved_game, "shadow-1", config),
        )

    def test_capture_projection_ignores_unused_observation_exactly(self):
        semantic_from_capture = (
            identity_module.semantic_forecast_payload_from_capture_payload
        )
        baseline = complete_forecast_game()
        unused = source_observation(
            observation_id="obs:unused",
            source_name="manual",
            source_record="unused/audit-only",
            content_hash="b" * 64,
        )
        with_unused = replace(
            baseline,
            observations=baseline.observations + (unused,),
        )

        baseline_semantic = semantic_forecast_payload(baseline)
        unused_semantic = semantic_forecast_payload(with_unused)
        projected = semantic_from_capture(capture_payload(with_unused))

        self.assertEqual(
            canonical_json(baseline_semantic),
            canonical_json(unused_semantic),
        )
        self.assertEqual(
            canonical_json(projected),
            canonical_json(unused_semantic),
        )

    def test_capture_payload_preserves_unused_observation_for_audit(self):
        baseline = complete_forecast_game()
        unused = source_observation(
            observation_id="obs:unused",
            source_name="manual",
            source_record="unused/audit-only",
            content_hash="b" * 64,
        )

        captured = capture_payload(
            replace(
                baseline,
                observations=baseline.observations + (unused,),
            )
        )

        self.assertEqual(
            tuple(
                observation["observation_id"]
                for observation in captured["observations"]
            ),
            ("obs:1", "obs:unused"),
        )

    def test_semantic_identity_rejects_missing_or_duplicate_referenced_ids(self):
        semantic_from_capture = (
            identity_module.semantic_forecast_payload_from_capture_payload
        )
        baseline = complete_forecast_game()
        duplicate = replace(
            baseline.observations[0],
            source_record="conflicting/source-record",
            content_hash="d" * 64,
        )
        cases = (
            (
                "missing",
                replace(baseline, observations=()),
                "missing observation_id",
            ),
            (
                "duplicate",
                replace(
                    baseline,
                    observations=(baseline.observations[0], duplicate),
                ),
                "duplicate observation_id",
            ),
        )
        for label, game, expected in cases:
            for path, operation in (
                ("object", lambda: semantic_forecast_payload(game)),
                (
                    "capture",
                    lambda: semantic_from_capture(capture_payload(game)),
                ),
            ):
                with self.subTest(label=label, path=path):
                    with self.assertRaisesRegex(ValueError, expected):
                        operation()

    def test_capture_semantic_rebuild_rejects_malformed_payload_schema(self):
        semantic_from_capture = getattr(
            identity_module,
            "semantic_forecast_payload_from_capture_payload",
            None,
        )
        self.assertTrue(callable(semantic_from_capture))
        baseline = capture_payload(normalized_game())

        malformed_payloads = []
        missing_identity = json.loads(canonical_json(baseline))
        missing_identity.pop("identity")
        malformed_payloads.append(missing_identity)
        additional_field = json.loads(canonical_json(baseline))
        additional_field["unexpected"] = True
        malformed_payloads.append(additional_field)
        missing_market_field = json.loads(canonical_json(baseline))
        missing_market_field["markets"][0].pop("market_type")
        malformed_payloads.append(missing_market_field)
        duplicate_market = json.loads(canonical_json(baseline))
        duplicate_market["markets"].append(duplicate_market["markets"][0])
        malformed_payloads.append(duplicate_market)
        unknown_team_alias = json.loads(canonical_json(baseline))
        unknown_team_alias["markets"][0]["selected_side"] = "NYY"
        malformed_payloads.append(unknown_team_alias)

        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    semantic_from_capture(payload)

    def test_behavior_changing_inputs_change_forecast_id(self) -> None:
        game = complete_forecast_game()
        config = {"simulations": 20_000, "seed_mode": "forecast"}
        baseline_id = build_forecast_run_id(game, "shadow-1", config)

        changed_line = replace(
            game,
            markets=(replace(game.markets[0], line=8.5),),
        )
        changed_price = replace(
            game,
            markets=(replace(game.markets[0], selected_decimal_price=1.87),),
        )
        old_slot = game.away_lineup.slots[3]
        replacement_batter_id = "batter:SEA:replacement"
        replacement_slot = replace(
            old_slot,
            batter_id=replacement_batter_id,
            batter_name="Replacement Batter",
            responses=tuple(
                replace(response, batter_id=replacement_batter_id)
                for response in old_slot.responses
            ),
        )
        changed_lineup = replace(
            game,
            away_lineup=replace_lineup_slot(
                game.away_lineup,
                3,
                replacement_slot,
            ),
        )
        changed_source = replace(
            game,
            observations=(
                replace(game.observations[0], content_hash="c" * 64),
            ),
        )
        cases = (
            ("line", changed_line, "shadow-1", config),
            ("price", changed_price, "shadow-1", config),
            ("lineup batter", changed_lineup, "shadow-1", config),
            ("source content hash", changed_source, "shadow-1", config),
            ("model version", game, "shadow-2", config),
            (
                "simulation config",
                game,
                "shadow-1",
                {"simulations": 40_000, "seed_mode": "forecast"},
            ),
        )

        for label, changed_game, model_version, changed_config in cases:
            with self.subTest(label=label):
                self.assertNotEqual(
                    baseline_id,
                    build_forecast_run_id(
                        changed_game,
                        model_version,
                        changed_config,
                    ),
                )

    def test_forecast_id_validates_version_and_config(self) -> None:
        game = complete_forecast_game()
        for model_version in ("", "   ", None):
            with self.subTest(model_version=model_version):
                with self.assertRaisesRegex(ValueError, "model_version"):
                    build_forecast_run_id(game, model_version, {})

        with self.assertRaises(TypeError):
            build_forecast_run_id(game, "shadow-1", {"invalid": object()})
        with self.assertRaisesRegex(ValueError, "finite"):
            build_forecast_run_id(game, "shadow-1", {"rate": math.nan})

    def test_capture_payload_retains_audit_fields_semantic_payload_excludes(self) -> None:
        game = complete_forecast_game()
        capture = capture_payload(game)
        semantic = semantic_forecast_payload(game)

        self.assertEqual(capture["captured_at"], "2026-07-14T01:30:00Z")
        capture_observation = capture["observations"][0]
        for field_name in (
            "observation_id",
            "observed_at",
            "available_at",
            "fetched_at",
        ):
            self.assertIn(field_name, capture_observation)
        capture_provenance = capture["away_lineup"]["state_provenance"]
        for field_name in (
            "observation_ids",
            "latest_available_at",
            "lookback_end",
        ):
            self.assertIn(field_name, capture_provenance)

        def nested_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                keys = set(value)
                for nested in value.values():
                    keys.update(nested_keys(nested))
                return keys
            if isinstance(value, list):
                keys: set[str] = set()
                for nested in value:
                    keys.update(nested_keys(nested))
                return keys
            return set()

        semantic_keys = nested_keys(semantic)
        for excluded in (
            "captured_at",
            "observation_id",
            "observation_ids",
            "observed_at",
            "available_at",
            "fetched_at",
            "latest_available_at",
            "lookback_end",
        ):
            self.assertNotIn(excluded, semantic_keys)

        self.assertEqual(
            set(semantic["observations"][0]),
            {"source_name", "source_record", "effective_date", "content_hash"},
        )
        semantic_market = semantic["markets"][0]
        for retained in (
            "market_key",
            "market_type",
            "settlement_format",
            "selected_side",
            "line",
            "selected_decimal_price",
            "opposing_decimal_price",
        ):
            self.assertIn(retained, semantic_market)

    def test_complete_valid_game_is_out_of_sample_eligible(self) -> None:
        result = validate_pregame_input(complete_forecast_game())

        self.assertTrue(result.out_of_sample_eligible)
        self.assertEqual(result.flags, ())

    def test_iter_feature_provenance_yields_every_reference(self) -> None:
        provenances = tuple(iter_feature_provenance(complete_forecast_game()))

        self.assertEqual(len(provenances), 131)
        self.assertTrue(
            all(isinstance(item, FeatureProvenance) for item in provenances)
        )

    def test_capture_at_or_after_first_pitch_is_rejected(self) -> None:
        game = complete_forecast_game()
        for captured_at in (
            game.identity.first_pitch,
            game.identity.first_pitch + timedelta(seconds=1),
        ):
            with self.subTest(captured_at=captured_at):
                with self.assertRaisesRegex(ValueError, "captured_at.*first_pitch"):
                    validate_pregame_input(replace(game, captured_at=captured_at))

    def test_feature_latest_available_at_after_capture_is_rejected(self) -> None:
        game = complete_forecast_game()
        late = replace(
            game.away_lineup.state_provenance,
            latest_available_at=game.captured_at + timedelta(seconds=1),
        )
        changed = replace(
            game,
            away_lineup=replace(game.away_lineup, state_provenance=late),
        )

        with self.assertRaisesRegex(
            ValueError,
            "latest_available_at.*after captured_at",
        ):
            validate_pregame_input(changed)

    def test_feature_lookback_end_after_capture_is_rejected(self) -> None:
        game = complete_forecast_game()
        late = replace(
            game.away_lineup.state_provenance,
            lookback_end=game.captured_at + timedelta(seconds=1),
        )
        changed = replace(
            game,
            away_lineup=replace(game.away_lineup, state_provenance=late),
        )

        with self.assertRaisesRegex(ValueError, "lookback_end.*after captured_at"):
            validate_pregame_input(changed)

    def test_referenced_observation_available_after_capture_is_rejected(self) -> None:
        game = complete_forecast_game()
        changed = replace(
            game,
            observations=(
                replace(
                    game.observations[0],
                    available_at=game.captured_at + timedelta(seconds=1),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "observation.*available_at.*after captured_at",
        ):
            validate_pregame_input(changed)

    def test_missing_and_duplicate_observation_ids_are_rejected(self) -> None:
        game = complete_forecast_game()
        duplicate = replace(
            game.observations[0],
            source_record="duplicate-record",
        )
        cases = (
            ("missing", replace(game, observations=()), "missing observation_id"),
            (
                "duplicate",
                replace(game, observations=(game.observations[0], duplicate)),
                "duplicate observation_id",
            ),
        )

        for label, changed, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    validate_pregame_input(changed)

    def test_unknown_available_at_returns_one_stable_flag(self) -> None:
        game = complete_forecast_game()
        unknown_provenance = replace(
            game.away_lineup.state_provenance,
            latest_available_at=None,
        )
        unknown_feature = replace(
            game,
            away_lineup=replace(
                game.away_lineup,
                state_provenance=unknown_provenance,
                projected_runs_per_9=replace(
                    game.away_lineup.projected_runs_per_9,
                    provenance=unknown_provenance,
                ),
            ),
        )
        unknown_observation = replace(
            game,
            observations=(replace(game.observations[0], available_at=None),),
        )
        both_unknown = replace(
            unknown_feature,
            observations=(replace(game.observations[0], available_at=None),),
        )

        for label, changed in (
            ("feature", unknown_feature),
            ("observation", unknown_observation),
            ("deduplicated", both_unknown),
        ):
            with self.subTest(label=label):
                result = validate_pregame_input(changed)
                self.assertFalse(result.out_of_sample_eligible)
                self.assertEqual(result.flags, ("unknown available_at",))

    def test_post_capture_audit_timestamps_do_not_prove_leakage(self) -> None:
        game = complete_forecast_game()
        observation = replace(
            game.observations[0],
            observed_at=game.identity.first_pitch + timedelta(minutes=5),
            fetched_at=game.identity.first_pitch + timedelta(minutes=10),
        )

        result = validate_pregame_input(
            replace(game, observations=(observation,))
        )

        self.assertTrue(result.out_of_sample_eligible)
        self.assertEqual(result.flags, ())

    def test_unreferenced_unknown_observation_does_not_change_eligibility(self) -> None:
        game = complete_forecast_game()
        unreferenced = source_observation(
            observation_id="obs:unreferenced",
            source_name="manual",
            source_record="unused",
            available_at=None,
            content_hash="d" * 64,
        )

        result = validate_pregame_input(
            replace(game, observations=game.observations + (unreferenced,))
        )

        self.assertTrue(result.out_of_sample_eligible)
        self.assertEqual(result.flags, ())

    def test_every_explicit_provenance_branch_is_leakage_checked(self) -> None:
        game = complete_forecast_game()
        late = FeatureProvenance(
            observation_ids=("obs:1",),
            latest_available_at=game.captured_at + timedelta(seconds=1),
            lookback_end=AVAILABLE_AT,
        )

        cases: list[tuple[str, NormalizedGameInput]] = []
        cases.append(
            (
                "away lineup state",
                replace(
                    game,
                    away_lineup=replace(
                        game.away_lineup,
                        state_provenance=late,
                    ),
                ),
            )
        )
        lineup_feature_names = (
            "projected_runs_per_9",
            "projected_traffic_per_9",
            "projected_extra_base_hits_per_9",
            "walk_rate",
            "strikeout_rate",
            "isolated_power",
            "barrel_rate",
            "recent_form_index",
            "lower_order_continuity",
            "log_standard_error",
            "recent_form_included",
        )
        for field_name in lineup_feature_names:
            feature = getattr(game.away_lineup, field_name)
            cases.append(
                (
                    f"away lineup {field_name}",
                    replace(
                        game,
                        away_lineup=replace(
                            game.away_lineup,
                            **{field_name: replace(feature, provenance=late)},
                        ),
                    ),
                )
            )

        response_slot = game.away_lineup.slots[-1]
        late_response = replace(response_slot.responses[0], provenance=late)
        cases.append(
            (
                "away lineup response",
                replace(
                    game,
                    away_lineup=replace_lineup_slot(
                        game.away_lineup,
                        8,
                        replace(response_slot, responses=(late_response,)),
                    ),
                ),
            )
        )
        cases.append(
            (
                "home lineup",
                replace(
                    game,
                    home_lineup=replace(
                        game.home_lineup,
                        projected_runs_per_9=replace(
                            game.home_lineup.projected_runs_per_9,
                            provenance=late,
                        ),
                    ),
                ),
            )
        )

        chain = game.away_pitching
        chain_pitcher = chain.pitchers[1]

        def game_with_away_pitcher(changed_pitcher: PitcherInput) -> NormalizedGameInput:
            return replace(
                game,
                away_pitching=replace_chain_pitcher(chain, 1, changed_pitcher),
            )

        pitcher_feature_names = (
            "projected_ra9",
            "log_standard_error",
            "pitch_limit",
            "command_risk",
            "home_run_risk",
            "injury_return",
            "recent_short_exit",
        )
        for field_name in pitcher_feature_names:
            feature = getattr(chain_pitcher, field_name)
            self.assertIsNotNone(feature)
            cases.append(
                (
                    f"pitcher {field_name}",
                    game_with_away_pitcher(
                        replace(
                            chain_pitcher,
                            **{field_name: replace(feature, provenance=late)},
                        )
                    ),
                )
            )

        component_names = (
            "strikeout_minus_walk_rate",
            "walk_rate",
            "home_runs_per_contact",
            "pitches_per_inning",
            "recent_pitch_count",
            "recent_workload_outs",
        )
        for field_name in component_names:
            component = getattr(chain_pitcher.projection_components, field_name)
            changed_components = replace(
                chain_pitcher.projection_components,
                **{field_name: replace(component, provenance=late)},
            )
            cases.append(
                (
                    f"projection component {field_name}",
                    game_with_away_pitcher(
                        replace(
                            chain_pitcher,
                            projection_components=changed_components,
                        )
                    ),
                )
            )

        for field_name in ("eligible_workload", "league_prior_workload"):
            history = getattr(chain_pitcher, field_name)
            cases.append(
                (
                    f"pitcher {field_name}",
                    game_with_away_pitcher(
                        replace(
                            chain_pitcher,
                            **{field_name: replace(history, provenance=late)},
                        )
                    ),
                )
            )
        cases.append(
            (
                "pitch usage",
                game_with_away_pitcher(
                    replace(
                        chain_pitcher,
                        pitch_mix=(
                            replace(chain_pitcher.pitch_mix[0], provenance=late),
                        ),
                    )
                ),
            )
        )
        home_pitcher = game.home_pitching.pitchers[0]
        cases.append(
            (
                "home pitcher",
                replace(
                    game,
                    home_pitching=replace_chain_pitcher(
                        game.home_pitching,
                        0,
                        replace(
                            home_pitcher,
                            projected_ra9=replace(
                                home_pitcher.projected_ra9,
                                provenance=late,
                            ),
                        ),
                    ),
                ),
            )
        )

        chain_feature_names = (
            "starter_confirmed",
            "starter_source_conflict",
            "handoff_verified",
            "timing_uncertain",
            "combined_pitch_limit",
        )
        for field_name in chain_feature_names:
            feature = getattr(chain, field_name)
            self.assertIsNotNone(feature)
            cases.append(
                (
                    f"chain {field_name}",
                    replace(
                        game,
                        away_pitching=replace(
                            chain,
                            **{field_name: replace(feature, provenance=late)},
                        ),
                    ),
                )
            )
        for field_name in (
            "eligible_combined_workload",
            "league_prior_combined_workload",
        ):
            history = getattr(chain, field_name)
            self.assertIsNotNone(history)
            cases.append(
                (
                    f"chain {field_name}",
                    replace(
                        game,
                        away_pitching=replace(
                            chain,
                            **{field_name: replace(history, provenance=late)},
                        ),
                    ),
                )
            )

        segment = game.away_bullpen.segments[0]
        bullpen_feature_names = (
            "projected_ra9",
            "log_standard_error",
            "available",
            "limited_performance_or_use_risk",
        )
        for field_name in bullpen_feature_names:
            feature = getattr(segment, field_name)
            changed_segment = replace(
                segment,
                **{field_name: replace(feature, provenance=late)},
            )
            cases.append(
                (
                    f"bullpen {field_name}",
                    replace(
                        game,
                        away_bullpen=replace(
                            game.away_bullpen,
                            segments=(changed_segment,),
                        ),
                    ),
                )
            )
        late_appearance = replace(segment.appearances[0], provenance=late)
        cases.append(
            (
                "bullpen appearance",
                replace(
                    game,
                    away_bullpen=replace(
                        game.away_bullpen,
                        segments=(
                            replace(segment, appearances=(late_appearance,)),
                        ),
                    ),
                ),
            )
        )
        home_segment = game.home_bullpen.segments[0]
        cases.append(
            (
                "home bullpen",
                replace(
                    game,
                    home_bullpen=replace(
                        game.home_bullpen,
                        segments=(
                            replace(
                                home_segment,
                                available=replace(
                                    home_segment.available,
                                    provenance=late,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
        cases.append(
            (
                "market",
                replace(
                    game,
                    markets=(replace(game.markets[0], provenance=late),),
                ),
            )
        )

        for label, changed in cases:
            with self.subTest(branch=label):
                with self.assertRaisesRegex(
                    ValueError,
                    "latest_available_at.*after captured_at",
                ):
                    validate_pregame_input(changed)

    def test_duplicate_markets_and_team_relationship_errors_are_rejected(self) -> None:
        game = complete_forecast_game()
        cases = (
            (
                "duplicate market",
                replace(game, markets=(game.markets[0], game.markets[0])),
                "duplicate market_key",
            ),
            (
                "away lineup",
                replace(
                    game,
                    away_lineup=replace(game.away_lineup, team="OAK"),
                ),
                "away team relationship",
            ),
            (
                "away bullpen",
                replace(
                    game,
                    away_bullpen=replace(game.away_bullpen, team="OAK"),
                ),
                "away team relationship",
            ),
            (
                "home lineup",
                replace(
                    game,
                    home_lineup=replace(game.home_lineup, team="OAK"),
                ),
                "home team relationship",
            ),
            (
                "home bullpen",
                replace(
                    game,
                    home_bullpen=replace(game.home_bullpen, team="OAK"),
                ),
                "home team relationship",
            ),
            (
                "same teams",
                replace(
                    game,
                    identity=replace(
                        game.identity,
                        home_team=game.identity.away_team,
                    ),
                ),
                "away_team and home_team must differ",
            ),
        )

        for label, changed, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    validate_pregame_input(changed)

    def test_response_pitcher_and_side_relationships_are_rejected(self) -> None:
        game = complete_forecast_game()
        away_slot = game.away_lineup.slots[0]
        home_slot = game.home_lineup.slots[0]
        away_wrong_pitcher = replace(
            away_slot,
            responses=(
                replace(away_slot.responses[0], pitcher_id="pitcher:missing"),
            ),
        )
        home_wrong_pitcher = replace(
            home_slot,
            responses=(
                replace(home_slot.responses[0], pitcher_id="pitcher:missing"),
            ),
        )
        away_wrong_side = replace(
            away_slot,
            responses=(
                replace(away_slot.responses[0], expected_side="R"),
            ),
        )
        home_wrong_side = replace(
            home_slot,
            responses=(
                replace(home_slot.responses[0], expected_side="R"),
            ),
        )
        cases = (
            (
                "away pitcher",
                replace(
                    game,
                    away_lineup=replace_lineup_slot(
                        game.away_lineup,
                        0,
                        away_wrong_pitcher,
                    ),
                ),
                "away lineup response pitcher_id",
            ),
            (
                "home pitcher",
                replace(
                    game,
                    home_lineup=replace_lineup_slot(
                        game.home_lineup,
                        0,
                        home_wrong_pitcher,
                    ),
                ),
                "home lineup response pitcher_id",
            ),
            (
                "away side",
                replace(
                    game,
                    away_lineup=replace_lineup_slot(
                        game.away_lineup,
                        0,
                        away_wrong_side,
                    ),
                ),
                "away lineup response expected_side",
            ),
            (
                "home side",
                replace(
                    game,
                    home_lineup=replace_lineup_slot(
                        game.home_lineup,
                        0,
                        home_wrong_side,
                    ),
                ),
                "home lineup response expected_side",
            ),
        )

        for label, changed, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    validate_pregame_input(changed)

    def test_switch_hitter_responses_follow_each_pitcher_handedness(self) -> None:
        game = complete_forecast_game()
        opener, bulk = game.home_pitching.pitchers
        lefty_bulk = replace(bulk, throws="L")
        home_pitching = replace(
            game.home_pitching,
            pitchers=(opener, lefty_bulk),
        )
        slot = game.away_lineup.slots[0]
        bulk_response = replace(
            slot.responses[0],
            expected_side="R",
        )
        opener_response = replace(
            bulk_response,
            pitcher_id=opener.pitcher_id,
            expected_side="L",
        )
        switch_slot = replace(
            slot,
            expected_side="R",
            responses=(opener_response, bulk_response),
        )
        away_lineup = replace_lineup_slot(
            game.away_lineup,
            0,
            switch_slot,
        )

        validate_pregame_input(
            replace(
                game,
                away_lineup=away_lineup,
                home_pitching=home_pitching,
            )
        )
