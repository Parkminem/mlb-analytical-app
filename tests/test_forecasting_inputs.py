from dataclasses import FrozenInstanceError, fields, replace
from datetime import date, datetime, timezone
import math
import unittest

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
