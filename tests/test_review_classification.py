from dataclasses import FrozenInstanceError, replace
import math
import unittest

from analysis.forecasting import MarketType
from analysis.model import PriorityEvidencePath
import analysis.review.classification as classification_module
from analysis.review import (
    ComponentForecast,
    ComponentObservation,
    ContradictionRule,
    ExogenousEvent,
    PathReview,
    PredictiveInterval,
    PrincipalPathBinding,
    ReviewCandidate,
    ReviewClassification,
    classify_review,
)


INTERVAL = PredictiveInterval(
    p05=0.0,
    p10=1.0,
    p20=2.0,
    median=3.0,
    p80=4.0,
    p90=5.0,
    p95=6.0,
)
POSITIVE_COUNT_INTERVAL = PredictiveInterval(
    p05=1.0,
    p10=2.0,
    p20=3.0,
    median=4.0,
    p80=5.0,
    p90=6.0,
    p95=7.0,
)
BINARY_RANGE_INTERVAL = PredictiveInterval(
    p05=0.0,
    p10=0.0,
    p20=0.0,
    median=0.5,
    p80=1.0,
    p90=1.0,
    p95=1.0,
)
BINARY_TRUE_INTERVAL = PredictiveInterval(
    p05=1.0,
    p10=1.0,
    p20=1.0,
    median=1.0,
    p80=1.0,
    p90=1.0,
    p95=1.0,
)


def binding(evidence_path, rule, *component_ids):
    return PrincipalPathBinding(
        evidence_path=evidence_path,
        component_ids=tuple(component_ids),
        contradiction_rule=rule,
    )


def candidate(
    *,
    values,
    principal_paths,
    intervals=None,
    evidence_paths=None,
    market_key="game:1:full_total:over:8.0",
    market_type=MarketType.FULL_TOTAL,
    market_result="win",
    recommended=True,
    events=(),
):
    intervals = intervals or {}
    evidence_paths = evidence_paths or {}
    default_path = (
        principal_paths[0].evidence_path
        if principal_paths
        else PriorityEvidencePath.TOP_ORDER_MATCHUP
    )
    bound_evidence_paths = {}
    for path in principal_paths:
        for component_id in path.component_ids:
            bound_evidence_paths.setdefault(component_id, path.evidence_path)
    forecasts = tuple(
        ComponentForecast(
            component_id=component_id,
            evidence_path=evidence_paths.get(
                component_id,
                bound_evidence_paths.get(component_id, default_path),
            ),
            interval=intervals.get(component_id, INTERVAL),
            direction="high_is_supportive",
        )
        for component_id in values
    )
    observations = tuple(
        ComponentObservation(component_id=component_id, value=value)
        for component_id, value in values.items()
    )
    return ReviewCandidate(
        capture_forecast_link_record_id="link:1",
        capture_id="capture:1",
        forecast_run_id="forecast:1",
        official_result_record_id="result:1",
        market_key=market_key,
        market_type=market_type,
        recommended=recommended,
        market_result=market_result,
        component_forecasts=forecasts,
        component_observations=observations,
        principal_paths=tuple(principal_paths),
        exogenous_events=tuple(events),
    )


def standard_paths():
    return (
        binding(
            PriorityEvidencePath.STARTER_SUPPRESSION,
            ContradictionRule.STARTER_SUPPRESSION,
            "away_pitcher_chain_runs_allowed",
        ),
        binding(
            PriorityEvidencePath.TOP_ORDER_MATCHUP,
            ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
        ),
    )


def standard_values(*, chain_runs=3.0, runs=3.0, traffic=3.0, xbh=3.0):
    return {
        "away_pitcher_chain_runs_allowed": chain_runs,
        "home_full_runs": runs,
        "home_full_traffic": traffic,
        "home_full_extra_base_hits": xbh,
    }


class PrincipalPathBindingValidatorTests(unittest.TestCase):
    def test_public_validator_enforces_shape_horizon_and_side_compatibility(self):
        validator = getattr(
            classification_module,
            "validate_principal_path_bindings",
            None,
        )
        self.assertIsNotNone(
            validator,
            "public principal-path binding validator is missing",
        )
        validator(MarketType.FULL_TOTAL, standard_paths())

        wrong_shape = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "away_full_runs",
            ),
        )
        with self.assertRaisesRegex(ValueError, "binding shape"):
            validator(MarketType.FULL_TOTAL, wrong_shape)

        wrong_horizon = (
            binding(
                PriorityEvidencePath.TOP_ORDER_MATCHUP,
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
                "away_f5_runs",
                "away_f5_traffic",
                "away_f5_extra_base_hits",
            ),
        )
        with self.assertRaisesRegex(ValueError, "binding shape"):
            validator(MarketType.ML, wrong_horizon)

        same_side = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "home_pitcher_chain_runs_allowed",
            ),
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "home_pitcher_chain_runs_allowed",
                "home_pitcher_chain_workload_outs",
            ),
        )
        with self.assertRaisesRegex(ValueError, "same pitching side"):
            validator(MarketType.ML, same_side)

    def test_lineup_suppression_requires_total_horizon_components(self):
        validator = classification_module.validate_principal_path_bindings
        path = binding(
            PriorityEvidencePath.LINEUP_SUPPRESSION,
            ContradictionRule.LINEUP_SUPPRESSION,
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
        )

        validator(MarketType.FULL_TOTAL, (path,))

        with self.assertRaisesRegex(ValueError, "total markets"):
            validator(MarketType.ML, (path,))

        wrong_horizon = binding(
            PriorityEvidencePath.LINEUP_SUPPRESSION,
            ContradictionRule.LINEUP_SUPPRESSION,
            "home_f5_runs",
            "home_f5_traffic",
            "home_f5_extra_base_hits",
        )
        with self.assertRaisesRegex(ValueError, "binding shape"):
            validator(MarketType.FULL_TOTAL, (wrong_horizon,))

    def test_lineup_suppression_cannot_share_conversion_components(self):
        validator = classification_module.validate_principal_path_bindings
        lineup_path = binding(
            PriorityEvidencePath.LINEUP_SUPPRESSION,
            ContradictionRule.LINEUP_SUPPRESSION,
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
        )
        conversion_path = binding(
            PriorityEvidencePath.TOP_ORDER_MATCHUP,
            ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
        )

        with self.assertRaisesRegex(ValueError, "opposite-direction"):
            validator(MarketType.FULL_TOTAL, (lineup_path, conversion_path))


class ReviewClassificationTests(unittest.TestCase):
    def test_a_requires_win_and_at_least_seventy_five_percent_inside_80(self):
        result = classify_review(
            candidate(
                values=standard_values(xbh=6.0),
                principal_paths=standard_paths(),
                market_result="win",
            )
        )

        self.assertEqual(result.classification, "A")
        self.assertEqual(result.inside_80_ratio, 0.75)

    def test_b_includes_losing_ticket_with_broadly_correct_flow(self):
        paths = (
            standard_paths()[0],
            binding(
                PriorityEvidencePath.BULLPEN_PROTECTION,
                ContradictionRule.BULLPEN_PROTECTION,
                "home_bullpen_runs_allowed",
                "home_protected_bullpen_close_entry",
            ),
        )
        result = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "home_bullpen_runs_allowed": 6.0,
                    "home_protected_bullpen_close_entry": 1.0,
                },
                intervals={
                    "home_protected_bullpen_close_entry": BINARY_TRUE_INTERVAL,
                },
                principal_paths=paths,
                market_result="loss",
            )
        )

        self.assertEqual(result.classification, "B")
        self.assertAlmostEqual(result.inside_80_ratio, 2 / 3)

    def test_winning_ticket_with_wrong_flow_is_c(self):
        result = classify_review(
            candidate(
                values=standard_values(traffic=7.0, xbh=7.0),
                principal_paths=standard_paths(),
                market_result="win",
            )
        )

        self.assertEqual(result.classification, "C")
        self.assertEqual(result.inside_80_ratio, 0.5)

    def test_documented_injury_outside_90_takes_d_precedence(self):
        result = classify_review(
            candidate(
                values=standard_values(chain_runs=7.0),
                principal_paths=standard_paths(),
                market_result="loss",
                events=(
                    ExogenousEvent(
                        event_type="injury",
                        affected_component_id="away_pitcher_chain_runs_allowed",
                        documented_source_observation_id="official:injury:1",
                    ),
                ),
            )
        )

        self.assertEqual(result.classification, "D")
        self.assertEqual(
            result.contradicted_paths,
            (PriorityEvidencePath.STARTER_SUPPRESSION,),
        )
        self.assertEqual(
            result.qualifying_exogenous_source_observation_ids,
            ("official:injury:1",),
        )

    def test_no_recommended_market_has_component_review_without_ab_class(self):
        result = classify_review(
            candidate(
                values={"away_full_runs": 3.0},
                principal_paths=(),
                market_result=None,
                recommended=False,
            )
        )

        self.assertIsNone(result.classification)
        self.assertEqual(result.inside_80_ratio, 1.0)
        self.assertEqual(result.path_reviews, ())

    def test_c_precedes_a_when_a_principal_path_is_contradicted(self):
        result = classify_review(
            candidate(
                values=standard_values(chain_runs=7.0),
                principal_paths=standard_paths(),
                market_result="win",
            )
        )

        self.assertEqual(result.inside_80_ratio, 0.75)
        self.assertEqual(result.classification, "C")

    def test_b_is_the_fallback_for_a_win_below_the_a_threshold(self):
        paths = standard_paths() + (
            binding(
                PriorityEvidencePath.BULLPEN_PROTECTION,
                ContradictionRule.BULLPEN_PROTECTION,
                "home_bullpen_runs_allowed",
                "home_protected_bullpen_close_entry",
            ),
        )
        result = classify_review(
            candidate(
                values={
                    **standard_values(xbh=6.0),
                    "home_bullpen_runs_allowed": 6.0,
                    "home_protected_bullpen_close_entry": 1.0,
                },
                principal_paths=paths,
                intervals={
                    "home_protected_bullpen_close_entry": BINARY_TRUE_INTERVAL,
                },
                market_result="win",
            )
        )

        self.assertAlmostEqual(result.inside_80_ratio, 2 / 3)
        self.assertEqual(result.classification, "B")

    def test_shared_component_across_two_paths_is_accepted_and_counted_once(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "away_pitcher_chain_runs_allowed",
                "away_pitcher_chain_workload_outs",
            ),
            binding(
                PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                ContradictionRule.LATE_BULLPEN_EXPOSURE,
                "away_bullpen_segment_bridge:1_exposed",
                "away_pitcher_chain_workload_outs",
            ),
        )

        try:
            review_candidate = candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "away_pitcher_chain_workload_outs": 6.0,
                    "away_bullpen_segment_bridge:1_exposed": 0.0,
                },
                intervals={
                    "away_bullpen_segment_bridge:1_exposed": (
                        BINARY_TRUE_INTERVAL
                    ),
                },
                principal_paths=paths,
                market_result="loss",
            )
        except ValueError as error:
            self.fail(f"shared component should be accepted: {error}")

        result = classify_review(review_candidate)

        self.assertAlmostEqual(result.inside_80_ratio, 1 / 3)
        self.assertEqual(
            tuple(
                (review.component_count, review.inside_80_count)
                for review in result.path_reviews
            ),
            ((2, 1), (2, 0)),
        )
        self.assertEqual(result.classification, "C")

    def test_inside_80_ratio_ignores_components_outside_principal_paths(self):
        values = standard_values()
        values["away_full_runs"] = 7.0
        result = classify_review(
            candidate(
                values=values,
                principal_paths=standard_paths(),
                evidence_paths={
                    "away_full_runs": PriorityEvidencePath.UNDERDOG_RUN_CREATION,
                },
                market_result="win",
            )
        )

        self.assertEqual(result.inside_80_ratio, 1.0)
        self.assertEqual(result.classification, "A")

    def test_c_threshold_is_strictly_below_sixty_percent(self):
        paths = (
            standard_paths()[0],
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "home_pitcher_chain_runs_allowed",
                "home_pitcher_chain_workload_outs",
            ),
            binding(
                PriorityEvidencePath.BULLPEN_PROTECTION,
                ContradictionRule.BULLPEN_PROTECTION,
                "away_bullpen_runs_allowed",
                "away_protected_bullpen_close_entry",
            ),
        )
        result = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "home_pitcher_chain_runs_allowed": 3.0,
                    "home_pitcher_chain_workload_outs": 6.0,
                    "away_bullpen_runs_allowed": 6.0,
                    "away_protected_bullpen_close_entry": 1.0,
                },
                principal_paths=paths,
                intervals={
                    "away_protected_bullpen_close_entry": BINARY_TRUE_INTERVAL,
                },
                market_result="loss",
            )
        )

        self.assertEqual(result.inside_80_ratio, 0.6)
        self.assertEqual(result.classification, "B")

    def test_path_outputs_are_independent_of_principal_path_input_order(self):
        intervals = {
            "home_full_runs": POSITIVE_COUNT_INTERVAL,
            "home_full_traffic": POSITIVE_COUNT_INTERVAL,
            "home_full_extra_base_hits": POSITIVE_COUNT_INTERVAL,
        }
        values = standard_values(
            chain_runs=7.0,
            runs=0.0,
            traffic=0.0,
            xbh=0.0,
        )

        forward_candidate = candidate(
            values=values,
            intervals=intervals,
            principal_paths=standard_paths(),
        )
        reverse_candidate = candidate(
            values=values,
            intervals=intervals,
            principal_paths=tuple(reversed(standard_paths())),
        )
        forward = classify_review(forward_candidate)
        reverse = classify_review(reverse_candidate)

        self.assertEqual(forward_candidate, reverse_candidate)
        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward.contradicted_paths,
            (
                PriorityEvidencePath.STARTER_SUPPRESSION,
                PriorityEvidencePath.TOP_ORDER_MATCHUP,
            ),
        )


class ContradictionRuleTests(unittest.TestCase):
    def test_starter_suppression_requires_runs_above_p90_and_outside_p95(self):
        inside_90 = classify_review(
            candidate(
                values=standard_values(chain_runs=6.0),
                principal_paths=standard_paths(),
                market_result="loss",
            )
        )
        outside_90 = classify_review(
            candidate(
                values=standard_values(chain_runs=7.0),
                principal_paths=standard_paths(),
                market_result="loss",
            )
        )

        self.assertFalse(inside_90.path_reviews[0].contradicted)
        self.assertTrue(outside_90.path_reviews[0].contradicted)

    def test_starter_collapse_requires_low_runs_and_median_workload(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "away_pitcher_chain_runs_allowed",
                "away_pitcher_chain_workload_outs",
            ),
            standard_paths()[1],
        )
        contradicted = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 0.0,
                    "away_pitcher_chain_workload_outs": 3.0,
                    "home_full_runs": 3.0,
                    "home_full_traffic": 3.0,
                    "home_full_extra_base_hits": 3.0,
                },
                intervals={
                    "away_pitcher_chain_runs_allowed": POSITIVE_COUNT_INTERVAL,
                },
                principal_paths=paths,
            )
        )
        short_workload = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 0.0,
                    "away_pitcher_chain_workload_outs": 2.0,
                    "home_full_runs": 3.0,
                    "home_full_traffic": 3.0,
                    "home_full_extra_base_hits": 3.0,
                },
                intervals={
                    "away_pitcher_chain_runs_allowed": POSITIVE_COUNT_INTERVAL,
                },
                principal_paths=paths,
            )
        )

        self.assertTrue(contradicted.path_reviews[0].contradicted)
        self.assertFalse(short_workload.path_reviews[0].contradicted)

    def test_top_order_rule_requires_all_three_low_and_one_outside_90(self):
        intervals = {
            "home_full_runs": POSITIVE_COUNT_INTERVAL,
            "home_full_traffic": POSITIVE_COUNT_INTERVAL,
            "home_full_extra_base_hits": POSITIVE_COUNT_INTERVAL,
        }
        all_low_with_tail = classify_review(
            candidate(
                values=standard_values(runs=0.0, traffic=0.0, xbh=0.0),
                intervals=intervals,
                principal_paths=standard_paths(),
            )
        )
        all_low_inside_90 = classify_review(
            candidate(
                values=standard_values(runs=1.0, traffic=1.0, xbh=1.0),
                intervals=intervals,
                principal_paths=standard_paths(),
            )
        )
        one_not_low = classify_review(
            candidate(
                values=standard_values(runs=0.0, traffic=2.0, xbh=1.0),
                intervals=intervals,
                principal_paths=standard_paths(),
            )
        )

        self.assertTrue(all_low_with_tail.path_reviews[1].contradicted)
        self.assertFalse(all_low_inside_90.path_reviews[1].contradicted)
        self.assertFalse(one_not_low.path_reviews[1].contradicted)

    def test_lineup_suppression_is_contradicted_by_an_upper_tail_result(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "home_pitcher_chain_runs_allowed",
            ),
            binding(
                PriorityEvidencePath.LINEUP_SUPPRESSION,
                ContradictionRule.LINEUP_SUPPRESSION,
                "home_full_runs",
                "home_full_traffic",
                "home_full_extra_base_hits",
            ),
        )
        result = classify_review(
            candidate(
                values={
                    "home_pitcher_chain_runs_allowed": 3.0,
                    "home_full_runs": 7.0,
                    "home_full_traffic": 3.0,
                    "home_full_extra_base_hits": 3.0,
                },
                principal_paths=paths,
                market_key="game:1:full_total:under:8.0",
                market_result="loss",
            )
        )

        self.assertFalse(result.path_reviews[0].contradicted)
        self.assertTrue(result.path_reviews[1].contradicted)
        self.assertEqual(
            result.contradicted_paths,
            (PriorityEvidencePath.LINEUP_SUPPRESSION,),
        )

    def test_lineup_suppression_requires_its_own_upper_95_tail(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "home_pitcher_chain_runs_allowed",
            ),
            binding(
                PriorityEvidencePath.LINEUP_SUPPRESSION,
                ContradictionRule.LINEUP_SUPPRESSION,
                "home_full_runs",
                "home_full_traffic",
                "home_full_extra_base_hits",
            ),
        )
        result = classify_review(
            candidate(
                values={
                    "home_pitcher_chain_runs_allowed": 3.0,
                    "home_full_runs": 7.0,
                    "home_full_traffic": 0.0,
                    "home_full_extra_base_hits": 3.0,
                },
                intervals={
                    "home_full_runs": POSITIVE_COUNT_INTERVAL,
                    "home_full_traffic": POSITIVE_COUNT_INTERVAL,
                    "home_full_extra_base_hits": POSITIVE_COUNT_INTERVAL,
                },
                principal_paths=paths,
                market_key="game:1:full_total:under:8.0",
                market_result="loss",
            )
        )

        lineup_review = next(
            review
            for review in result.path_reviews
            if review.evidence_path is PriorityEvidencePath.LINEUP_SUPPRESSION
        )
        self.assertFalse(lineup_review.contradicted)

    def test_late_bullpen_exposure_requires_nonexposure_and_p80_workload(self):
        paths = (
            standard_paths()[0],
            binding(
                PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                ContradictionRule.LATE_BULLPEN_EXPOSURE,
                "away_bullpen_segment_bridge:1_exposed",
                "away_pitcher_chain_workload_outs",
            ),
        )
        intervals = {
            "away_bullpen_segment_bridge:1_exposed": BINARY_TRUE_INTERVAL,
        }
        contradicted = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "away_bullpen_segment_bridge:1_exposed": 0.0,
                    "away_pitcher_chain_workload_outs": 4.0,
                },
                intervals=intervals,
                principal_paths=paths,
            )
        )
        short_workload = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "away_bullpen_segment_bridge:1_exposed": 0.0,
                    "away_pitcher_chain_workload_outs": 3.0,
                },
                intervals=intervals,
                principal_paths=paths,
            )
        )

        self.assertTrue(contradicted.path_reviews[1].contradicted)
        self.assertFalse(short_workload.path_reviews[1].contradicted)

    def test_bullpen_protection_requires_tail_runs_and_close_entry(self):
        paths = (
            standard_paths()[0],
            binding(
                PriorityEvidencePath.BULLPEN_PROTECTION,
                ContradictionRule.BULLPEN_PROTECTION,
                "home_bullpen_runs_allowed",
                "home_protected_bullpen_close_entry",
            ),
        )
        intervals = {
            "home_protected_bullpen_close_entry": BINARY_TRUE_INTERVAL,
        }
        contradicted = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "home_bullpen_runs_allowed": 7.0,
                    "home_protected_bullpen_close_entry": 1.0,
                },
                intervals=intervals,
                principal_paths=paths,
            )
        )
        no_close_entry = classify_review(
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "home_bullpen_runs_allowed": 7.0,
                    "home_protected_bullpen_close_entry": 0.0,
                },
                intervals=intervals,
                principal_paths=paths,
            )
        )

        self.assertTrue(contradicted.path_reviews[1].contradicted)
        self.assertFalse(no_close_entry.path_reviews[1].contradicted)

    def test_rule_specific_binding_shapes_reject_extra_cross_team_and_horizon_ids(self):
        cases = (
            (
                "starter suppression extra",
                "game:1:full_total:over:8.0",
                MarketType.FULL_TOTAL,
                binding(
                    PriorityEvidencePath.STARTER_SUPPRESSION,
                    ContradictionRule.STARTER_SUPPRESSION,
                    "away_pitcher_chain_runs_allowed",
                    "away_pitcher_chain_workload_outs",
                ),
            ),
            (
                "starter collapse cross-team",
                "game:1:full_total:over:8.0",
                MarketType.FULL_TOTAL,
                binding(
                    PriorityEvidencePath.STARTER_COLLAPSE,
                    ContradictionRule.STARTER_COLLAPSE,
                    "away_pitcher_chain_runs_allowed",
                    "home_pitcher_chain_workload_outs",
                ),
            ),
            (
                "top order cross-team",
                "game:1:full_total:over:8.0",
                MarketType.FULL_TOTAL,
                binding(
                    PriorityEvidencePath.TOP_ORDER_MATCHUP,
                    ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
                    "home_full_runs",
                    "away_full_traffic",
                    "home_full_extra_base_hits",
                ),
            ),
            (
                "top order F5 horizon",
                "opaque-market-key",
                MarketType.F5_TOTAL,
                binding(
                    PriorityEvidencePath.TOP_ORDER_MATCHUP,
                    ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
                    "home_full_runs",
                    "home_full_traffic",
                    "home_full_extra_base_hits",
                ),
            ),
            (
                "late bullpen cross-team",
                "game:1:full_total:over:8.0",
                MarketType.FULL_TOTAL,
                binding(
                    PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                    ContradictionRule.LATE_BULLPEN_EXPOSURE,
                    "away_bullpen_segment_bridge_exposed",
                    "home_pitcher_chain_workload_outs",
                ),
            ),
            (
                "bullpen protection cross-team",
                "game:1:full_total:over:8.0",
                MarketType.FULL_TOTAL,
                binding(
                    PriorityEvidencePath.BULLPEN_PROTECTION,
                    ContradictionRule.BULLPEN_PROTECTION,
                    "away_bullpen_runs_allowed",
                    "home_protected_bullpen_close_entry",
                ),
            ),
        )
        for label, market_key, market_type, path in cases:
            with self.subTest(label=label):
                binary_ids = {
                    component_id
                    for component_id in path.component_ids
                    if component_id.endswith("_exposed")
                    or component_id.endswith("_protected_bullpen_close_entry")
                }
                values = {
                    component_id: 1.0 if component_id in binary_ids else 3.0
                    for component_id in path.component_ids
                }
                with self.assertRaisesRegex(ValueError, "binding shape"):
                    candidate(
                        values=values,
                        intervals={
                            component_id: BINARY_TRUE_INTERVAL
                            for component_id in binary_ids
                        },
                        principal_paths=(path,),
                        market_key=market_key,
                        market_type=market_type,
                        recommended=False,
                    )

    def test_conversion_binding_horizon_uses_market_type_only(self):
        cases = (
            (MarketType.F5_SIDE, "f5", "looks:like:full_total"),
            (MarketType.F5_TOTAL, "f5", "looks:like:full_total"),
            (MarketType.ML, "full", "looks:like:f5_total"),
            (MarketType.PLUS_1_5, "full", "looks:like:f5_total"),
            (MarketType.FULL_TOTAL, "full", "looks:like:f5_total"),
        )
        for market_type, expected_horizon, market_key in cases:
            with self.subTest(market_type=market_type):
                component_ids = (
                    f"home_{expected_horizon}_runs",
                    f"home_{expected_horizon}_traffic",
                    f"home_{expected_horizon}_extra_base_hits",
                )
                path = binding(
                    PriorityEvidencePath.TOP_ORDER_MATCHUP,
                    ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
                    *component_ids,
                )
                try:
                    candidate(
                        values={component_id: 3.0 for component_id in component_ids},
                        principal_paths=(path,),
                        market_key=market_key,
                        market_type=market_type,
                        recommended=False,
                    )
                except (TypeError, ValueError) as error:
                    self.fail(f"market_type should define the horizon: {error}")

                wrong_horizon = "full" if expected_horizon == "f5" else "f5"
                wrong_component_ids = (
                    f"home_{wrong_horizon}_runs",
                    f"home_{wrong_horizon}_traffic",
                    f"home_{wrong_horizon}_extra_base_hits",
                )
                wrong_path = binding(
                    PriorityEvidencePath.TOP_ORDER_MATCHUP,
                    ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
                    *wrong_component_ids,
                )
                with self.assertRaisesRegex(ValueError, "binding shape"):
                    candidate(
                        values={
                            component_id: 3.0
                            for component_id in wrong_component_ids
                        },
                        principal_paths=(wrong_path,),
                        market_key=market_key,
                        market_type=market_type,
                        recommended=False,
                    )


class ExogenousEventTests(unittest.TestCase):
    def test_event_type_must_be_a_canonical_supported_string(self):
        supported_types = (
            "injury",
            "ejection",
            "defensive_error",
            "defensive_error_cluster",
        )
        for event_type in supported_types:
            with self.subTest(event_type=event_type):
                ExogenousEvent(
                    event_type=event_type,
                    affected_component_id="away_pitcher_chain_runs_allowed",
                    documented_source_observation_id="official:event:1",
                )

        for event_type in (
            "Injury",
            "defensive-error",
            "injruy",
            "weather",
            "predeclared",
        ):
            with self.subTest(event_type=event_type):
                with self.assertRaisesRegex(ValueError, "event_type"):
                    ExogenousEvent(
                        event_type=event_type,
                        affected_component_id=(
                            "away_pitcher_chain_runs_allowed"
                        ),
                        documented_source_observation_id="official:event:1",
                    )

    def test_event_must_affect_component_strictly_outside_p05_p95(self):
        for boundary in (0.0, 6.0):
            with self.subTest(boundary=boundary):
                result = classify_review(
                    candidate(
                        values=standard_values(chain_runs=boundary),
                        principal_paths=standard_paths(),
                        events=(
                            ExogenousEvent(
                                event_type="injury",
                                affected_component_id="away_pitcher_chain_runs_allowed",
                                documented_source_observation_id="official:injury:1",
                            ),
                        ),
                    )
                )
                self.assertNotEqual(result.classification, "D")

    def test_qualifying_event_source_ids_are_sorted_and_unique(self):
        result = classify_review(
            candidate(
                values=standard_values(chain_runs=7.0, runs=7.0),
                principal_paths=standard_paths(),
                events=(
                    ExogenousEvent(
                        event_type="injury",
                        affected_component_id="away_pitcher_chain_runs_allowed",
                        documented_source_observation_id="official:event:z",
                    ),
                    ExogenousEvent(
                        event_type="ejection",
                        affected_component_id="home_full_runs",
                        documented_source_observation_id="official:event:a",
                    ),
                    ExogenousEvent(
                        event_type="injury",
                        affected_component_id="away_pitcher_chain_runs_allowed",
                        documented_source_observation_id="official:event:z",
                    ),
                ),
            )
        )

        self.assertEqual(result.classification, "D")
        self.assertEqual(
            result.qualifying_exogenous_source_observation_ids,
            ("official:event:a", "official:event:z"),
        )

    def test_defensive_error_forms_require_at_least_three_unearned_runs(self):
        for event_type in ("defensive_error", "defensive_error_cluster"):
            for unearned_runs, expected in ((2, "C"), (3, "D")):
                with self.subTest(
                    event_type=event_type,
                    unearned_runs=unearned_runs,
                ):
                    result = classify_review(
                        candidate(
                            values=standard_values(chain_runs=7.0),
                            principal_paths=standard_paths(),
                            market_result="win",
                            events=(
                                ExogenousEvent(
                                    event_type=event_type,
                                    affected_component_id=(
                                        "away_pitcher_chain_runs_allowed"
                                    ),
                                    documented_source_observation_id=(
                                        "official:error:1"
                                    ),
                                    unearned_runs=unearned_runs,
                                ),
                            ),
                        )
                    )
                    self.assertEqual(result.classification, expected)

    def test_event_component_must_exist_in_forecast_and_observation(self):
        with self.assertRaisesRegex(ValueError, "affected component"):
            candidate(
                values=standard_values(),
                principal_paths=standard_paths(),
                events=(
                    ExogenousEvent(
                        event_type="injury",
                        affected_component_id="home_pitcher_chain_workload_outs",
                        documented_source_observation_id="official:injury:1",
                    ),
                ),
            )

    def test_event_component_must_be_bound_to_a_principal_path(self):
        values = standard_values()
        values["away_full_runs"] = 7.0

        with self.assertRaisesRegex(ValueError, "principal-bound"):
            candidate(
                values=values,
                principal_paths=standard_paths(),
                evidence_paths={
                    "away_full_runs": PriorityEvidencePath.UNDERDOG_RUN_CREATION,
                },
                events=(
                    ExogenousEvent(
                        event_type="injury",
                        affected_component_id="away_full_runs",
                        documented_source_observation_id="official:injury:1",
                    ),
                ),
            )


class ReviewValidationTests(unittest.TestCase):
    def test_all_public_records_are_frozen(self):
        base_candidate = candidate(
            values=standard_values(),
            principal_paths=standard_paths(),
        )
        result = classify_review(base_candidate)
        event = ExogenousEvent(
            event_type="injury",
            affected_component_id="away_pitcher_chain_runs_allowed",
            documented_source_observation_id="official:injury:1",
        )
        records = (
            INTERVAL,
            base_candidate.component_forecasts[0],
            base_candidate.component_observations[0],
            event,
            base_candidate.principal_paths[0],
            result.path_reviews[0],
            base_candidate,
            result,
        )
        for record in records:
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(FrozenInstanceError):
                    record.extra = "mutation"

    def test_interval_requires_finite_nondecreasing_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(INTERVAL, p95=math.inf)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            replace(INTERVAL, p10=-1.0)

    def test_component_records_validate_canonical_paths_directions_and_values(self):
        with self.assertRaisesRegex(ValueError, "canonical PriorityEvidencePath"):
            ComponentForecast(
                component_id="away_full_runs",
                evidence_path="top_order_matchup",
                interval=INTERVAL,
                direction="high_is_supportive",
            )
        with self.assertRaisesRegex(ValueError, "direction"):
            ComponentForecast(
                component_id="away_full_runs",
                evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
                interval=INTERVAL,
                direction="up",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ComponentObservation("away_full_runs", math.nan)

    def test_known_count_components_require_nonnegative_integral_values(self):
        component_ids = (
            "away_pitcher_chain_runs_allowed",
            "home_pitcher_chain_workload_outs",
            "away_f5_runs",
            "away_f5_traffic",
            "away_f5_extra_base_hits",
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
            "away_bullpen_runs_allowed",
        )
        negative_interval = replace(INTERVAL, p05=-1.0)
        fractional_interval = replace(INTERVAL, median=3.5)

        for component_id in component_ids:
            with self.subTest(component_id=component_id, value=-1):
                with self.assertRaisesRegex(ValueError, "count observation"):
                    ComponentObservation(component_id, -1.0)
            with self.subTest(component_id=component_id, value=1.5):
                with self.assertRaisesRegex(ValueError, "count observation"):
                    ComponentObservation(component_id, 1.5)
            with self.subTest(component_id=component_id, interval="negative"):
                with self.assertRaisesRegex(
                    ValueError,
                    "count forecast interval",
                ):
                    ComponentForecast(
                        component_id=component_id,
                        evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
                        interval=negative_interval,
                        direction="high_is_supportive",
                    )
            with self.subTest(component_id=component_id, interval="fractional"):
                with self.assertRaisesRegex(
                    ValueError,
                    "count forecast interval",
                ):
                    ComponentForecast(
                        component_id=component_id,
                        evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
                        interval=fractional_interval,
                        direction="high_is_supportive",
                    )

            ComponentObservation(component_id, 0.0)
            ComponentForecast(
                component_id=component_id,
                evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
                interval=INTERVAL,
                direction="high_is_supportive",
            )

    def test_binary_components_stay_within_the_binary_domain(self):
        component_ids = (
            "away_bullpen_segment_bridge:1_exposed",
            "home_protected_bullpen_close_entry",
        )
        invalid_intervals = (
            replace(BINARY_RANGE_INTERVAL, p05=-0.1),
            BINARY_RANGE_INTERVAL,
            replace(BINARY_RANGE_INTERVAL, p95=1.1),
        )

        for component_id in component_ids:
            for value in (-1.0, 0.5, 2.0):
                with self.subTest(component_id=component_id, value=value):
                    with self.assertRaisesRegex(ValueError, "binary observation"):
                        ComponentObservation(component_id, value)
            for interval in invalid_intervals:
                with self.subTest(component_id=component_id, interval=interval):
                    with self.assertRaisesRegex(
                        ValueError,
                        "binary forecast interval",
                    ):
                        ComponentForecast(
                            component_id=component_id,
                            evidence_path=(
                                PriorityEvidencePath.LATE_BULLPEN_EXPOSURE
                            ),
                            interval=interval,
                            direction="high_is_supportive",
                        )

            for value in (0.0, 1.0):
                ComponentObservation(component_id, value)
            ComponentForecast(
                component_id=component_id,
                evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                interval=BINARY_TRUE_INTERVAL,
                direction="high_is_supportive",
            )

    def test_direction_rejects_unhashable_non_string_values_cleanly(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            ComponentForecast(
                component_id="away_full_runs",
                evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
                interval=INTERVAL,
                direction=[],
            )

    def test_candidate_requires_unique_component_ids_and_complete_observations(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())
        with self.assertRaisesRegex(ValueError, "duplicate component forecast"):
            replace(
                base,
                component_forecasts=(
                    base.component_forecasts[0],
                    base.component_forecasts[0],
                ),
                component_observations=(base.component_observations[0],),
                principal_paths=(),
                recommended=False,
            )
        with self.assertRaisesRegex(ValueError, "duplicate component observation"):
            replace(
                base,
                component_observations=(
                    base.component_observations[0],
                    base.component_observations[0],
                ),
                principal_paths=(),
                recommended=False,
            )
        with self.assertRaisesRegex(ValueError, "observation coverage"):
            replace(base, component_observations=base.component_observations[:-1])
        with self.assertRaisesRegex(ValueError, "observation coverage"):
            replace(
                base,
                component_observations=base.component_observations
                + (ComponentObservation("away_f5_runs", 1.0),),
            )

    def test_recommended_candidate_requires_two_to_four_unique_principal_paths(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())
        for paths in ((base.principal_paths[0],), base.principal_paths * 3):
            with self.subTest(count=len(paths)):
                with self.assertRaisesRegex(ValueError, "two to four"):
                    replace(base, principal_paths=paths)
        with self.assertRaisesRegex(ValueError, "unique principal"):
            replace(
                base,
                principal_paths=(
                    base.principal_paths[0],
                    base.principal_paths[0],
                ),
            )

    def test_recommended_candidate_requires_a_market_result(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())

        with self.assertRaisesRegex(ValueError, "recommended.*market_result"):
            replace(base, market_result=None)

    def test_f5_markets_reject_full_game_bullpen_paths(self):
        cases = (
            (
                binding(
                    PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                    ContradictionRule.LATE_BULLPEN_EXPOSURE,
                    "away_bullpen_segment_bridge:1_exposed",
                    "away_pitcher_chain_workload_outs",
                ),
                {
                    "away_bullpen_segment_bridge:1_exposed": 1.0,
                    "away_pitcher_chain_workload_outs": 3.0,
                },
                {
                    "away_bullpen_segment_bridge:1_exposed": (
                        BINARY_TRUE_INTERVAL
                    ),
                },
            ),
            (
                binding(
                    PriorityEvidencePath.BULLPEN_PROTECTION,
                    ContradictionRule.BULLPEN_PROTECTION,
                    "home_bullpen_runs_allowed",
                    "home_protected_bullpen_close_entry",
                ),
                {
                    "home_bullpen_runs_allowed": 3.0,
                    "home_protected_bullpen_close_entry": 1.0,
                },
                {
                    "home_protected_bullpen_close_entry": (
                        BINARY_TRUE_INTERVAL
                    ),
                },
            ),
        )

        for market_type in (MarketType.F5_SIDE, MarketType.F5_TOTAL):
            for path, values, intervals in cases:
                with self.subTest(
                    market_type=market_type,
                    evidence_path=path.evidence_path,
                ):
                    with self.assertRaisesRegex(ValueError, "F5 markets"):
                        candidate(
                            values=values,
                            intervals=intervals,
                            principal_paths=(path,),
                            market_type=market_type,
                            recommended=False,
                        )

    def test_same_side_starter_suppression_and_collapse_are_rejected(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "away_pitcher_chain_runs_allowed",
            ),
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "away_pitcher_chain_runs_allowed",
                "away_pitcher_chain_workload_outs",
            ),
        )

        with self.assertRaisesRegex(ValueError, "same pitching side"):
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "away_pitcher_chain_workload_outs": 3.0,
                },
                principal_paths=paths,
            )

    def test_opposite_side_starter_suppression_and_collapse_are_allowed(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_SUPPRESSION,
                ContradictionRule.STARTER_SUPPRESSION,
                "away_pitcher_chain_runs_allowed",
            ),
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "home_pitcher_chain_runs_allowed",
                "home_pitcher_chain_workload_outs",
            ),
        )

        try:
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "home_pitcher_chain_runs_allowed": 3.0,
                    "home_pitcher_chain_workload_outs": 3.0,
                },
                principal_paths=paths,
            )
        except ValueError as error:
            self.fail(f"opposite pitching sides should be allowed: {error}")

    def test_principal_paths_and_rules_reject_string_aliases(self):
        with self.assertRaisesRegex(ValueError, "canonical PriorityEvidencePath"):
            PrincipalPathBinding(
                evidence_path="starter_suppression",
                component_ids=("away_pitcher_chain_runs_allowed",),
                contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
            )
        with self.assertRaisesRegex(ValueError, "ContradictionRule"):
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
                component_ids=("away_pitcher_chain_runs_allowed",),
                contradiction_rule="starter_suppression",
            )

    def test_each_evidence_path_requires_its_canonical_contradiction_rule(self):
        allowed_rules = {
            PriorityEvidencePath.STARTER_SUPPRESSION: (
                ContradictionRule.STARTER_SUPPRESSION
            ),
            PriorityEvidencePath.STARTER_COLLAPSE: (
                ContradictionRule.STARTER_COLLAPSE
            ),
            PriorityEvidencePath.TOP_ORDER_MATCHUP: (
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
            PriorityEvidencePath.RECENT_RUN_CONVERSION: (
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
            PriorityEvidencePath.UNDERDOG_RUN_CREATION: (
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
            PriorityEvidencePath.LATE_BULLPEN_EXPOSURE: (
                ContradictionRule.LATE_BULLPEN_EXPOSURE
            ),
            PriorityEvidencePath.BULLPEN_PROTECTION: (
                ContradictionRule.BULLPEN_PROTECTION
            ),
        }

        for evidence_path, allowed_rule in allowed_rules.items():
            for rule in ContradictionRule:
                with self.subTest(evidence_path=evidence_path, rule=rule):
                    if rule is allowed_rule:
                        binding(evidence_path, rule, "component")
                    else:
                        with self.assertRaisesRegex(
                            ValueError,
                            "contradiction_rule must match evidence_path",
                        ):
                            binding(evidence_path, rule, "component")

    def test_binding_component_ids_are_canonicalized(self):
        component_ids = (
            "away_pitcher_chain_runs_allowed",
            "away_pitcher_chain_workload_outs",
        )
        forward_binding = binding(
            PriorityEvidencePath.STARTER_COLLAPSE,
            ContradictionRule.STARTER_COLLAPSE,
            *component_ids,
        )
        reverse_binding = binding(
            PriorityEvidencePath.STARTER_COLLAPSE,
            ContradictionRule.STARTER_COLLAPSE,
            *reversed(component_ids),
        )
        values = {
            "away_pitcher_chain_runs_allowed": 3.0,
            "away_pitcher_chain_workload_outs": 3.0,
            "home_full_runs": 3.0,
            "home_full_traffic": 3.0,
            "home_full_extra_base_hits": 3.0,
        }
        forward_candidate = candidate(
            values=values,
            principal_paths=(forward_binding, standard_paths()[1]),
        )
        reverse_candidate = candidate(
            values=values,
            principal_paths=(reverse_binding, standard_paths()[1]),
        )

        self.assertEqual(forward_binding, reverse_binding)
        self.assertEqual(
            forward_binding.component_ids,
            tuple(sorted(component_ids)),
        )
        self.assertEqual(forward_candidate, reverse_candidate)
        self.assertEqual(
            classify_review(forward_candidate),
            classify_review(reverse_candidate),
        )

    def test_every_bound_component_must_exist_in_forecasts_and_observations(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())
        invalid_path = binding(
            PriorityEvidencePath.BULLPEN_PROTECTION,
            ContradictionRule.BULLPEN_PROTECTION,
            "away_bullpen_runs_allowed",
            "away_protected_bullpen_close_entry",
        )
        with self.assertRaisesRegex(ValueError, "bound component"):
            replace(
                base,
                principal_paths=(base.principal_paths[0], invalid_path),
            )

    def test_shared_component_path_must_match_a_referencing_binding(self):
        paths = (
            binding(
                PriorityEvidencePath.STARTER_COLLAPSE,
                ContradictionRule.STARTER_COLLAPSE,
                "away_pitcher_chain_runs_allowed",
                "away_pitcher_chain_workload_outs",
            ),
            binding(
                PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
                ContradictionRule.LATE_BULLPEN_EXPOSURE,
                "away_bullpen_segment_bridge:1_exposed",
                "away_pitcher_chain_workload_outs",
            ),
        )

        with self.assertRaisesRegex(ValueError, "binding evidence_path"):
            candidate(
                values={
                    "away_pitcher_chain_runs_allowed": 3.0,
                    "away_pitcher_chain_workload_outs": 3.0,
                    "away_bullpen_segment_bridge:1_exposed": 1.0,
                },
                principal_paths=paths,
                intervals={
                    "away_bullpen_segment_bridge:1_exposed": (
                        BINARY_TRUE_INTERVAL
                    ),
                },
                evidence_paths={
                    "away_pitcher_chain_workload_outs": (
                        PriorityEvidencePath.TOP_ORDER_MATCHUP
                    ),
                },
            )

    def test_candidate_scalars_and_event_metadata_are_validated(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())
        with self.assertRaisesRegex(ValueError, "market_type"):
            replace(base, market_type="full_total")
        with self.assertRaisesRegex(ValueError, "recommended"):
            replace(base, recommended=1)
        with self.assertRaisesRegex(ValueError, "market_result"):
            replace(base, market_result="void")
        with self.assertRaisesRegex(ValueError, "documented_source_observation_id"):
            ExogenousEvent(
                event_type="injury",
                affected_component_id="away_pitcher_chain_runs_allowed",
                documented_source_observation_id=" ",
            )
        with self.assertRaisesRegex(ValueError, "unearned_runs"):
            ExogenousEvent(
                event_type="defensive_error",
                affected_component_id="away_pitcher_chain_runs_allowed",
                documented_source_observation_id="official:error:1",
                unearned_runs=-1,
            )

    def test_market_result_rejects_unhashable_non_string_values_cleanly(self):
        base = candidate(values=standard_values(), principal_paths=standard_paths())

        with self.assertRaisesRegex(ValueError, "market_result"):
            replace(base, market_result=[])

    def test_output_records_validate_counts_ratios_and_class_labels(self):
        with self.assertRaisesRegex(ValueError, "component counts"):
            PathReview(
                evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
                component_ids=("away_pitcher_chain_runs_allowed",),
                contradicted=False,
                inside_80_count=2,
                component_count=1,
            )
        with self.assertRaisesRegex(ValueError, "classification"):
            ReviewClassification(
                classification="E",
                inside_80_ratio=1.0,
                contradicted_paths=(),
                path_reviews=(),
            )
        with self.assertRaisesRegex(ValueError, "inside_80_ratio"):
            ReviewClassification(
                classification=None,
                inside_80_ratio=1.1,
                contradicted_paths=(),
                path_reviews=(),
            )

    def test_qualifying_source_ids_preserve_positional_notes_compatibility(self):
        try:
            result = ReviewClassification(
                "A",
                0.75,
                (),
                (),
                ("existing note",),
            )
        except ValueError as error:
            self.fail(f"existing positional notes must remain valid: {error}")

        self.assertEqual(result.notes, ("existing note",))
        self.assertEqual(
            result.qualifying_exogenous_source_observation_ids,
            (),
        )

    def test_direct_classification_enforces_representable_class_invariants(self):
        contradicted_review = PathReview(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("away_pitcher_chain_runs_allowed",),
            contradicted=True,
            inside_80_count=0,
            component_count=1,
        )
        contradicted_paths = (PriorityEvidencePath.STARTER_SUPPRESSION,)
        path_reviews = (contradicted_review,)

        with self.assertRaisesRegex(ValueError, "A requires"):
            ReviewClassification("A", 0.74, (), ())
        with self.assertRaisesRegex(ValueError, "A requires"):
            ReviewClassification(
                "A",
                1.0,
                contradicted_paths,
                path_reviews,
            )
        with self.assertRaisesRegex(ValueError, "B requires"):
            ReviewClassification("B", 0.59, (), ())
        with self.assertRaisesRegex(ValueError, "B requires"):
            ReviewClassification(
                "B",
                1.0,
                contradicted_paths,
                path_reviews,
            )
        with self.assertRaisesRegex(ValueError, "C requires"):
            ReviewClassification("C", 0.60, (), ())

        try:
            with self.assertRaisesRegex(ValueError, "D requires"):
                ReviewClassification(
                    classification="D",
                    inside_80_ratio=1.0,
                    contradicted_paths=contradicted_paths,
                    path_reviews=path_reviews,
                    qualifying_exogenous_source_observation_ids=(),
                )
            with self.assertRaisesRegex(ValueError, "non-D"):
                ReviewClassification(
                    classification="C",
                    inside_80_ratio=0.59,
                    contradicted_paths=(),
                    path_reviews=(),
                    qualifying_exogenous_source_observation_ids=("source:1",),
                )
        except TypeError as error:
            self.fail(f"classification must expose qualifying source IDs: {error}")

        valid_cases = (
            ("A", 0.75, (), (), ()),
            ("B", 0.60, (), (), ()),
            ("C", 0.59, (), (), ()),
            ("C", 1.0, contradicted_paths, path_reviews, ()),
            (
                "D",
                1.0,
                contradicted_paths,
                path_reviews,
                ("source:1",),
            ),
            (None, 0.0, (), (), ()),
        )
        for classification, ratio, paths, reviews, source_ids in valid_cases:
            with self.subTest(classification=classification, ratio=ratio):
                try:
                    ReviewClassification(
                        classification=classification,
                        inside_80_ratio=ratio,
                        contradicted_paths=paths,
                        path_reviews=reviews,
                        qualifying_exogenous_source_observation_ids=source_ids,
                    )
                except TypeError as error:
                    self.fail(
                        "classification must expose qualifying source IDs: "
                        f"{error}"
                    )

    def test_classification_rejects_unhashable_non_string_values_cleanly(self):
        with self.assertRaisesRegex(ValueError, "classification"):
            ReviewClassification(
                classification=[],
                inside_80_ratio=1.0,
                contradicted_paths=(),
                path_reviews=(),
            )


if __name__ == "__main__":
    unittest.main()
