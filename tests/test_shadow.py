from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

from analysis.forecasting import OutcomeProbabilities, price_market, run_forecast
from analysis.forecasting.identity import (
    canonical_json,
    capture_payload,
    normalized_input_hash,
)
from analysis.ledger import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    LedgerStore,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review.classification import (
    ContradictionRule,
    PrincipalPathBinding,
)
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    ShadowComparison,
    ShadowMarketComparison,
    ShadowRunIdempotencyKeys,
    compare_shadow_forecast,
    record_shadow_run,
)
from tests.test_forecasting_engine import (
    _complete_game,
    _fast_config,
    forecast_engine_module,
)


def principal_paths() -> tuple[PrincipalPathBinding, ...]:
    return (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("home_pitcher_chain_runs_allowed",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        ),
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
            component_ids=(
                "away_full_runs",
                "away_full_traffic",
                "away_full_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        ),
    )


def candidate(
    *,
    market: str = "ml",
    score: int = 78,
    rating: str = "play",
) -> PriorityMarketCandidate:
    return PriorityMarketCandidate(
        market=market,
        score=score,
        decision_rating=rating,
        evidence_paths=frozenset(
            path.evidence_path for path in principal_paths()
        ),
    )


def binding(
    *,
    index: int = 0,
    market_key: str = "sea_ml",
    legacy_market: str = "ml",
    paths: tuple[PrincipalPathBinding, ...] | None = None,
) -> CandidateReviewBinding:
    return CandidateReviewBinding(
        market_key=market_key,
        legacy_candidate_index=index,
        legacy_market=legacy_market,
        principal_paths=principal_paths() if paths is None else paths,
    )


def legacy_snapshot(
    *,
    candidates: tuple[PriorityMarketCandidate, ...] | None = None,
    review_bindings: tuple[CandidateReviewBinding, ...] | None = None,
) -> LegacyDecisionSnapshot:
    resolved_candidates = (candidate(),) if candidates is None else candidates
    resolved_bindings = (binding(),) if review_bindings is None else review_bindings
    final_priority = tuple(
        item.market
        for item in resolved_candidates
        if isinstance(item, PriorityMarketCandidate)
    )
    return LegacyDecisionSnapshot(
        model_version="legacy-fixed-v1",
        candidates=resolved_candidates,
        final_priority=final_priority,
        review_bindings=resolved_bindings,
        notes=("legacy remains authoritative",),
    )


class TrackingStore:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store
        self.calls: list[str] = []

    def append_capture(self, *args, **kwargs):
        self.calls.append("capture")
        return self.store.append_capture(*args, **kwargs)

    def append_forecast(self, *args, **kwargs):
        self.calls.append("forecast")
        return self.store.append_forecast(*args, **kwargs)

    def link_capture_forecast(self, *args, **kwargs):
        self.calls.append("link")
        return self.store.link_capture_forecast(*args, **kwargs)


class FailOnceAfterAppendStore(TrackingStore):
    def __init__(self, store: LedgerStore, stage: str) -> None:
        super().__init__(store)
        self.stage = stage
        self.failed = False

    def _fail_once(self, stage: str) -> None:
        if self.stage == stage and not self.failed:
            self.failed = True
            raise RuntimeError(f"injected failure after {stage}")

    def append_forecast(self, *args, **kwargs):
        record = super().append_forecast(*args, **kwargs)
        self._fail_once("forecast")
        return record

    def link_capture_forecast(self, *args, **kwargs):
        record = super().link_capture_forecast(*args, **kwargs)
        self._fail_once("link")
        return record


class ShadowComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.game = _complete_game()
        cls.forecast = run_forecast(cls.game, _fast_config())

    def test_shadow_types_are_frozen(self) -> None:
        review_binding = binding()
        legacy = legacy_snapshot()
        comparison = compare_shadow_forecast(legacy, self.forecast)

        with self.assertRaises(FrozenInstanceError):
            review_binding.market_key = "other"
        with self.assertRaises(FrozenInstanceError):
            legacy.notes = ()
        with self.assertRaises(FrozenInstanceError):
            comparison.authority = "shadow"
        with self.assertRaises(FrozenInstanceError):
            comparison.market_comparisons[0].review_eligible = False

    def test_binding_validates_candidate_index_market_and_key(self) -> None:
        for overrides, message in (
            ({"index": -1}, "legacy_candidate_index"),
            ({"index": True}, "legacy_candidate_index"),
            ({"market_key": ""}, "market_key"),
            ({"legacy_market": " "}, "legacy_market"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    binding(**overrides)

    def test_binding_requires_a_tuple_of_canonical_principal_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "principal_paths.*tuple"):
            binding(paths=list(principal_paths()))
        with self.assertRaisesRegex(
            ValueError,
            "principal_paths.*PrincipalPathBinding",
        ):
            binding(paths=(object(),))

    def test_snapshot_validates_candidate_and_binding_collections(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates.*tuple"):
            legacy_snapshot(candidates=[candidate()])
        with self.assertRaisesRegex(ValueError, "PriorityMarketCandidate"):
            legacy_snapshot(candidates=(object(),))
        with self.assertRaisesRegex(ValueError, "review_bindings.*tuple"):
            legacy_snapshot(review_bindings=[binding()])

    def test_snapshot_rejects_duplicate_or_out_of_range_binding_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique.*index"):
            legacy_snapshot(review_bindings=(binding(), binding()))
        with self.assertRaisesRegex(ValueError, "candidate index.*range"):
            legacy_snapshot(review_bindings=(binding(index=1),))

    def test_snapshot_rejects_mutable_or_noncanonical_candidate_evidence(self) -> None:
        mutable_evidence = candidate()
        object.__setattr__(
            mutable_evidence,
            "evidence_paths",
            [PriorityEvidencePath.STARTER_SUPPRESSION],
        )
        with self.assertRaisesRegex(ValueError, "evidence_paths.*frozenset"):
            legacy_snapshot(candidates=(mutable_evidence,), review_bindings=())

        noncanonical_evidence = candidate()
        object.__setattr__(
            noncanonical_evidence,
            "evidence_paths",
            frozenset(("starter_suppression",)),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            legacy_snapshot(candidates=(noncanonical_evidence,), review_bindings=())

    def test_adapter_preserves_legacy_authority_candidate_identity_and_order(self) -> None:
        first = candidate(market="full_total", score=72)
        second = candidate(market="ml", score=85)
        legacy = legacy_snapshot(
            candidates=(first, second),
            review_bindings=(
                binding(
                    index=0,
                    market_key="full_under",
                    legacy_market="full_total",
                ),
                binding(index=1),
            ),
        )

        comparison = compare_shadow_forecast(legacy, self.forecast)

        self.assertEqual(comparison.authority, "legacy")
        self.assertIs(comparison.legacy, legacy)
        self.assertEqual(comparison.legacy.candidates, (first, second))
        self.assertIs(comparison.legacy.candidates[0], first)
        self.assertIs(comparison.legacy.candidates[1], second)
        self.assertEqual(
            tuple(item.legacy_candidate_index for item in comparison.market_comparisons),
            (0, 1),
        )
        self.assertFalse(hasattr(comparison, "replacement_candidates"))

    def test_forecast_probability_does_not_change_legacy_priority(self) -> None:
        legacy_candidate = candidate()
        legacy = legacy_snapshot(candidates=(legacy_candidate,))
        markets = dict(self.forecast.markets)
        markets["sea_ml"] = price_market(
            OutcomeProbabilities(win=0.32, push=0.0, loss=0.68),
            1.91,
            1.91,
        )
        low_probability_forecast = replace(self.forecast, markets=markets)

        comparison = compare_shadow_forecast(
            legacy,
            low_probability_forecast,
        )

        self.assertIs(comparison.legacy.candidates[0], legacy_candidate)
        self.assertEqual(
            comparison.market_comparisons[0].model_win_probability,
            0.32,
        )

    def test_valid_binding_copies_market_values_and_frozen_paths(self) -> None:
        comparison = compare_shadow_forecast(
            legacy_snapshot(),
            self.forecast,
        )
        market = comparison.market_comparisons[0]
        expected = self.forecast.markets["sea_ml"]

        self.assertTrue(market.review_eligible)
        self.assertEqual(market.review_ineligibility_flags, ())
        self.assertEqual(market.principal_paths, principal_paths())
        self.assertEqual(
            market.model_win_probability,
            expected.model_conditional_win_probability,
        )
        self.assertEqual(market.fair_decimal_price, expected.fair_decimal_price)
        self.assertEqual(
            market.market_no_vig_probability,
            expected.market_no_vig_probability,
        )
        self.assertEqual(market.edge, expected.edge)

    def test_missing_binding_emits_explicit_ineligible_comparison(self) -> None:
        comparison = compare_shadow_forecast(
            legacy_snapshot(review_bindings=()),
            self.forecast,
        )

        self.assertEqual(len(comparison.market_comparisons), 1)
        market = comparison.market_comparisons[0]
        self.assertFalse(market.review_eligible)
        self.assertEqual(market.principal_paths, ())
        self.assertEqual(
            market.review_ineligibility_flags,
            ("missing_pregame_binding",),
        )

    def test_invalid_binding_emits_all_flags_in_fixed_order(self) -> None:
        invalid_path = PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("missing_component",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        )
        legacy = legacy_snapshot(
            review_bindings=(
                binding(
                    legacy_market="full_total",
                    market_key="unknown_market",
                    paths=(invalid_path,),
                ),
            ),
        )

        market = compare_shadow_forecast(
            legacy,
            self.forecast,
        ).market_comparisons[0]

        self.assertFalse(market.review_eligible)
        self.assertEqual(market.principal_paths, ())
        self.assertIsNone(market.model_win_probability)
        self.assertEqual(
            market.review_ineligibility_flags,
            (
                "candidate_market_mismatch",
                "unknown_market_key",
                "invalid_path_count",
                "missing_component_distribution",
            ),
        )

    def test_duplicate_canonical_path_is_invalid_without_repair(self) -> None:
        duplicated = principal_paths()[0]
        legacy = legacy_snapshot(
            review_bindings=(binding(paths=(duplicated, duplicated)),),
        )

        market = compare_shadow_forecast(
            legacy,
            self.forecast,
        ).market_comparisons[0]

        self.assertFalse(market.review_eligible)
        self.assertEqual(market.principal_paths, ())
        self.assertEqual(
            market.review_ineligibility_flags,
            ("invalid_path_count",),
        )

    def test_missing_component_invalidates_the_whole_binding(self) -> None:
        missing_component_path = PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
            component_ids=(
                "home_bullpen_segment_missing_exposed",
                "home_pitcher_chain_workload_outs",
            ),
            contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
        )
        legacy = legacy_snapshot(
            review_bindings=(
                binding(
                    paths=(missing_component_path, principal_paths()[0]),
                ),
            ),
        )

        market = compare_shadow_forecast(
            legacy,
            self.forecast,
        ).market_comparisons[0]

        self.assertFalse(market.review_eligible)
        self.assertEqual(market.principal_paths, ())
        self.assertEqual(
            market.review_ineligibility_flags,
            ("missing_component_distribution",),
        )

    def test_market_comparison_rejects_inconsistent_eligibility_flags(self) -> None:
        values = {
            "market_key": "sea_ml",
            "legacy_candidate_index": 0,
            "legacy_market": "ml",
            "legacy_score": 78,
            "legacy_rating": "play",
            "model_win_probability": 0.55,
            "fair_decimal_price": 1.8,
            "market_no_vig_probability": 0.5,
            "edge": 0.05,
            "principal_paths": principal_paths(),
            "review_eligible": True,
            "review_ineligibility_flags": (),
        }
        with self.assertRaisesRegex(ValueError, "eligible.*flags"):
            ShadowMarketComparison(
                **{
                    **values,
                    "review_ineligibility_flags": ("unknown_market_key",),
                }
            )
        with self.assertRaisesRegex(ValueError, "ineligible.*flag"):
            ShadowMarketComparison(
                **{
                    **values,
                    "principal_paths": (),
                    "review_eligible": False,
                }
            )
        with self.assertRaisesRegex(ValueError, "fixed.*flag"):
            ShadowMarketComparison(
                **{
                    **values,
                    "principal_paths": (),
                    "review_eligible": False,
                    "review_ineligibility_flags": ("invented_flag",),
                }
            )

    def test_shadow_comparison_rejects_nonlegacy_authority(self) -> None:
        valid = compare_shadow_forecast(
            legacy_snapshot(),
            self.forecast,
        )

        with self.assertRaisesRegex(ValueError, "authority.*legacy"):
            ShadowComparison(
                authority="shadow",
                legacy=valid.legacy,
                forecast_run_id=valid.forecast_run_id,
                market_comparisons=valid.market_comparisons,
            )

    def test_comparison_has_fixed_two_argument_public_interface(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(compare_shadow_forecast).parameters),
            ("legacy", "forecast"),
        )

        comparison = compare_shadow_forecast(legacy_snapshot(), self.forecast)

        self.assertEqual(comparison.forecast_run_id, self.forecast.forecast_run_id)


class ShadowRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.game = _complete_game()
        cls.forecast = run_forecast(cls.game, _fast_config())
        cls.legacy = legacy_snapshot()

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = LedgerStore(Path(self.directory.name) / "ledger.jsonl")

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def keys(suffix: str = "a") -> ShadowRunIdempotencyKeys:
        return ShadowRunIdempotencyKeys(capture=f"capture:shadow:{suffix}")

    def record(
        self,
        *,
        store=None,
        game=None,
        legacy=None,
        forecast=None,
        keys=None,
    ):
        return record_shadow_run(
            self.store if store is None else store,
            self.game if game is None else game,
            self.legacy if legacy is None else legacy,
            self.forecast if forecast is None else forecast,
            self.keys() if keys is None else keys,
        )

    def test_idempotency_keys_are_explicit_frozen_and_validated(self) -> None:
        keys = self.keys()

        with self.assertRaises(FrozenInstanceError):
            keys.capture = "other"
        with self.assertRaisesRegex(ValueError, "capture"):
            ShadowRunIdempotencyKeys(capture=" ")
        with self.assertRaisesRegex(TypeError, "ShadowRunIdempotencyKeys"):
            self.record(keys={"capture": "capture:shadow:a"})
        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recording_same_shadow_run_is_idempotent(self) -> None:
        first = self.record()
        second = self.record()

        self.assertEqual(first.capture.capture_id, second.capture.capture_id)
        self.assertEqual(
            first.forecast.forecast_run_id,
            second.forecast.forecast_run_id,
        )
        self.assertEqual(
            first.link.envelope.record_id,
            second.link.envelope.record_id,
        )
        self.assertEqual(len(tuple(self.store.iter_records("capture"))), 1)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertEqual(
            len(tuple(self.store.iter_records("capture_forecast_link"))),
            1,
        )

    def test_second_unchanged_capture_links_to_existing_forecast(self) -> None:
        later_game = replace(
            self.game,
            captured_at=self.game.captured_at + timedelta(minutes=1),
        )
        first = self.record(keys=self.keys("first"))
        second = self.record(
            game=later_game,
            keys=self.keys("second"),
        )

        self.assertNotEqual(first.capture.capture_id, second.capture.capture_id)
        self.assertEqual(
            first.forecast.forecast_run_id,
            second.forecast.forecast_run_id,
        )
        self.assertEqual(
            first.capture.normalized_input_hash,
            second.capture.normalized_input_hash,
        )
        self.assertEqual(len(tuple(self.store.iter_records("capture"))), 2)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertEqual(
            len(tuple(self.store.iter_records("capture_forecast_link"))),
            2,
        )

    def test_recording_order_is_capture_forecast_then_link(self) -> None:
        tracking_store = TrackingStore(self.store)

        result = self.record(store=tracking_store)

        self.assertEqual(tracking_store.calls, ["capture", "forecast", "link"])
        self.assertIsInstance(result.capture, CaptureRecord)
        self.assertIsInstance(result.forecast, ForecastRecord)
        self.assertIsInstance(result.link, CaptureForecastLink)
        with self.assertRaises(FrozenInstanceError):
            result.capture = result.capture

    def test_recording_keeps_full_capture_and_semantic_hash_separate(self) -> None:
        result = self.record()
        expected_hash = normalized_input_hash(self.game)
        expected_config_hash = hashlib.sha256(
            canonical_json(self.forecast.simulation_config).encode("utf-8")
        ).hexdigest()

        self.assertEqual(result.capture.payload, capture_payload(self.game))
        self.assertEqual(result.capture.normalized_input_hash, expected_hash)
        self.assertEqual(result.forecast.normalized_input_hash, expected_hash)
        self.assertEqual(
            result.forecast.simulation_config_hash,
            expected_config_hash,
        )
        self.assertEqual(result.capture.lineup_state, "confirmed")
        self.assertTrue(result.capture.out_of_sample_eligible)

    def test_forecast_outputs_hold_copied_legacy_comparison_only(self) -> None:
        result = self.record()
        outputs = result.forecast.outputs
        shadow = outputs["shadow_comparison"]

        self.assertEqual(shadow["authority"], "legacy")
        self.assertEqual(
            shadow["legacy"]["candidates"][0]["market"],
            self.legacy.candidates[0].market,
        )
        self.assertEqual(
            shadow["market_comparisons"][0]["principal_paths"][0][
                "evidence_path"
            ],
            PriorityEvidencePath.STARTER_SUPPRESSION.value,
        )
        self.assertIn("away_full_runs", outputs["component_distributions"])
        self.assertIn("sea_ml", outputs["markets"])
        self.assertEqual(outputs["market_inputs"]["sea_ml"]["market_type"], "ml")
        self.assertEqual(
            outputs["market_inputs"]["sea_ml"]["selected_side"],
            "away",
        )
        self.assertNotIn("shadow_comparison", result.capture.payload)
        self.assertNotIn("legacy", result.capture.payload)

    def test_legacy_content_cannot_change_semantic_input_hash(self) -> None:
        first = self.record()
        changed_legacy = legacy_snapshot(
            candidates=(candidate(score=12, rating="pass"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            other_store = LedgerStore(Path(directory) / "ledger.jsonl")
            second = self.record(
                store=other_store,
                legacy=changed_legacy,
            )

        self.assertEqual(
            first.forecast.normalized_input_hash,
            second.forecast.normalized_input_hash,
        )
        self.assertNotEqual(
            first.forecast.outputs["shadow_comparison"]["legacy"],
            second.forecast.outputs["shadow_comparison"]["legacy"],
        )

    def test_invalid_pregame_input_fails_before_any_append(self) -> None:
        invalid_game = replace(
            self.game,
            captured_at=self.game.identity.first_pitch,
        )

        with self.assertRaisesRegex(ValueError, "captured_at"):
            self.record(game=invalid_game)

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_forecast_for_another_game_is_rejected_before_any_append(self) -> None:
        other_game = replace(
            self.game,
            identity=replace(
                self.game.identity,
                game_pk=self.game.identity.game_pk + 1,
            ),
        )
        other_forecast = run_forecast(other_game, _fast_config())

        with self.assertRaisesRegex(ValueError, "forecast_run_id.*game_input"):
            self.record(forecast=other_forecast)

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_forecast_with_mismatched_semantic_markets_is_rejected_before_append(
        self,
    ) -> None:
        market_inputs = dict(self.forecast.market_inputs)
        market_inputs["sea_ml"] = replace(
            market_inputs["sea_ml"],
            selected_decimal_price=(
                market_inputs["sea_ml"].selected_decimal_price + 0.01
            ),
        )
        mismatched_forecast = replace(
            self.forecast,
            market_inputs=market_inputs,
        )

        with self.assertRaisesRegex(ValueError, "market_inputs.*game_input"):
            self.record(forecast=mismatched_forecast)

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_team_alias_capture_uses_canonical_forecast_input_hash(self) -> None:
        team_game = replace(
            self.game,
            markets=tuple(
                replace(market, selected_side="SEA")
                if market.market_key == "sea_ml"
                else market
                for market in self.game.markets
            ),
        )
        team_forecast = run_forecast(team_game, _fast_config())
        resolver = getattr(
            forecast_engine_module,
            "resolve_forecast_input",
            None,
        )
        self.assertIsNotNone(
            resolver,
            "public forecast input resolver is missing",
        )
        resolved_game = resolver(team_game)

        result = self.record(
            game=team_game,
            forecast=team_forecast,
            keys=self.keys("team-alias"),
        )

        expected_hash = normalized_input_hash(resolved_game)
        self.assertNotEqual(normalized_input_hash(team_game), expected_hash)
        self.assertEqual(result.capture.payload, capture_payload(team_game))
        self.assertEqual(result.capture.normalized_input_hash, expected_hash)
        self.assertEqual(result.forecast.normalized_input_hash, expected_hash)

    def test_existing_component_with_wrong_rule_shape_is_review_ineligible(self) -> None:
        wrong_shape = PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("away_full_runs",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        )
        paths = (wrong_shape, principal_paths()[1])
        legacy = legacy_snapshot(
            review_bindings=(binding(paths=paths),),
        )
        self.assertTrue(
            {
                component_id
                for path in paths
                for component_id in path.component_ids
            }.issubset(self.forecast.component_distributions)
        )

        result = self.record(legacy=legacy, keys=self.keys("wrong-shape"))
        market = result.forecast.outputs["shadow_comparison"][
            "market_comparisons"
        ][0]

        self.assertFalse(market["review_eligible"])
        self.assertEqual(market["principal_paths"], ())
        self.assertEqual(
            market["review_ineligibility_flags"],
            ("invalid_path_count",),
        )

    def test_market_input_type_rejects_wrong_horizon_despite_misleading_key(self) -> None:
        misleading_key = "looks:like:f5_total"
        game = replace(
            self.game,
            markets=tuple(
                replace(market, market_key=misleading_key)
                if market.market_key == "sea_ml"
                else market
                for market in self.game.markets
            ),
        )
        forecast = run_forecast(game, _fast_config())
        wrong_horizon = PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
            component_ids=(
                "away_f5_runs",
                "away_f5_traffic",
                "away_f5_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        )
        paths = (principal_paths()[0], wrong_horizon)
        legacy = legacy_snapshot(
            review_bindings=(
                binding(market_key=misleading_key, paths=paths),
            ),
        )

        result = self.record(
            game=game,
            legacy=legacy,
            forecast=forecast,
            keys=self.keys("wrong-horizon"),
        )
        market = result.forecast.outputs["shadow_comparison"][
            "market_comparisons"
        ][0]

        self.assertFalse(market["review_eligible"])
        self.assertEqual(
            market["review_ineligibility_flags"],
            ("invalid_path_count",),
        )

    def test_same_side_starter_paths_are_review_ineligible(self) -> None:
        same_side_paths = (
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
                component_ids=("home_pitcher_chain_runs_allowed",),
                contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
            ),
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
                component_ids=(
                    "home_pitcher_chain_runs_allowed",
                    "home_pitcher_chain_workload_outs",
                ),
                contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
            ),
        )
        legacy = legacy_snapshot(
            review_bindings=(binding(paths=same_side_paths),),
        )

        result = self.record(legacy=legacy, keys=self.keys("same-side"))
        market = result.forecast.outputs["shadow_comparison"][
            "market_comparisons"
        ][0]

        self.assertFalse(market["review_eligible"])
        self.assertEqual(
            market["review_ineligibility_flags"],
            ("invalid_path_count",),
        )

    def test_retry_after_forecast_append_failure_completes_missing_link(self) -> None:
        failing_store = FailOnceAfterAppendStore(self.store, "forecast")

        with self.assertRaisesRegex(RuntimeError, "after forecast"):
            self.record(store=failing_store)

        self.assertEqual(len(tuple(self.store.iter_records("capture"))), 1)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertEqual(
            len(tuple(self.store.iter_records("capture_forecast_link"))),
            0,
        )

        result = self.record(store=failing_store)

        self.assertEqual(result.capture.capture_id, next(
            iter(self.store.iter_records("capture"))
        ).capture_id)
        self.assertEqual(len(tuple(self.store.iter_records("capture"))), 1)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertEqual(
            len(tuple(self.store.iter_records("capture_forecast_link"))),
            1,
        )

    def test_retry_after_link_append_failure_replays_all_records(self) -> None:
        failing_store = FailOnceAfterAppendStore(self.store, "link")

        with self.assertRaisesRegex(RuntimeError, "after link"):
            self.record(store=failing_store)

        records_after_failure = tuple(self.store.iter_records())
        result = self.record(store=failing_store)

        self.assertEqual(len(records_after_failure), 3)
        self.assertEqual(len(tuple(self.store.iter_records())), 3)
        self.assertEqual(
            result.link.envelope.record_id,
            records_after_failure[-1].envelope.record_id,
        )


if __name__ == "__main__":
    unittest.main()
