import unittest

from analysis.model import (
    AnalysisChecklist,
    BullpenSegment,
    FullGameContext,
    HandicapContext,
    LiveGameState,
    LineValueContext,
    LineupStatus,
    MarketSignalContext,
    RelieverAppearance,
    StarterRecentForm,
    StarterStructureContext,
    StarterVolatilityContext,
    TotalContext,
    UmpireContext,
    apply_umpire_adjustment,
    calculate_confidence_score,
    evaluate_pitch_velocity_sample,
    evaluate_plus_handicap,
    evaluate_line_value,
    evaluate_live_entry,
    evaluate_market_signal,
    evaluate_total,
    recommend_full_game,
    resolve_lineup_source,
    resolve_starter_structure,
    score_bullpen_segments,
    score_bullpen_availability,
    score_starter_recent_form,
    score_starter_volatility,
)


class BullpenAvailabilityTests(unittest.TestCase):
    def test_single_yesterday_appearance_at_twenty_or_fewer_pitches_is_normal(self):
        appearances = [
            RelieverAppearance("Closer", "closer", days_ago=1, pitches=20),
        ]

        result = score_bullpen_availability(appearances)

        self.assertEqual(result.score, 100)
        self.assertEqual(result.limited_arms, [])
        self.assertEqual(result.unavailable_arms, [])

    def test_single_yesterday_appearance_between_twenty_one_and_twenty_nine_is_limited(self):
        appearances = [
            RelieverAppearance("Closer", "closer", days_ago=1, pitches=25),
        ]

        result = score_bullpen_availability(appearances)

        self.assertLess(result.score, 100)
        self.assertIn("Closer", result.limited_arms)
        self.assertEqual(result.unavailable_arms, [])

    def test_single_yesterday_appearance_at_thirty_or_more_pitches_is_performance_risk(self):
        appearances = [
            RelieverAppearance("Closer", "closer", days_ago=1, pitches=30),
        ]

        result = score_bullpen_availability(appearances)

        self.assertIn("Closer", result.limited_arms)
        self.assertEqual(result.unavailable_arms, [])
        self.assertTrue(any("performance/use risk" in note for note in result.notes))

    def test_consecutive_high_leverage_usage_flags_limited_arms(self):
        appearances = [
            RelieverAppearance("Closer", "closer", days_ago=1, pitches=17),
            RelieverAppearance("Closer", "closer", days_ago=2, pitches=4),
            RelieverAppearance("Setup", "setup", days_ago=1, pitches=18),
            RelieverAppearance("Setup", "setup", days_ago=2, pitches=24),
        ]

        result = score_bullpen_availability(appearances)

        self.assertLess(result.score, 70)
        self.assertIn("Closer", result.limited_arms)
        self.assertIn("Setup", result.limited_arms)
        self.assertTrue(any("high-leverage" in note for note in result.notes))

    def test_low_leverage_usage_does_not_create_major_downgrade(self):
        appearances = [
            RelieverAppearance("Long Man", "low", days_ago=1, pitches=28),
            RelieverAppearance("Middle Arm", "middle", days_ago=2, pitches=16),
        ]

        result = score_bullpen_availability(appearances)

        self.assertGreaterEqual(result.score, 80)
        self.assertEqual(result.limited_arms, [])

    def test_three_straight_high_leverage_usage_marks_arm_unavailable(self):
        appearances = [
            RelieverAppearance("Closer", "closer", days_ago=1, pitches=12),
            RelieverAppearance("Closer", "closer", days_ago=2, pitches=15),
            RelieverAppearance("Closer", "closer", days_ago=3, pitches=11),
        ]

        result = score_bullpen_availability(appearances)

        self.assertIn("Closer", result.unavailable_arms)
        self.assertLess(result.score, 65)
        self.assertTrue(any("likely unavailable" in note for note in result.notes))


class LineupSourceTests(unittest.TestCase):
    def test_unconfirmed_lineup_uses_rotowire_projected_lineup(self):
        result = resolve_lineup_source(LineupStatus(confirmed=False, source="RotoWire"))

        self.assertEqual(result.source, "RotoWire projected lineup")
        self.assertIn("lineup not confirmed", result.notes)


class StarterStructureTests(unittest.TestCase):
    def test_official_opener_and_rotowire_bulk_resolve_source_conflict(self):
        result = resolve_starter_structure(
            StarterStructureContext(
                official_pitcher="Bryan Hudson",
                rotowire_pitcher="Sean Burke",
                official_pitcher_is_opener=True,
                rotowire_pitcher_is_bulk_candidate=True,
            )
        )

        self.assertEqual(result.opener_pitcher, "Bryan Hudson")
        self.assertEqual(result.bulk_pitcher, "Sean Burke")
        self.assertEqual(result.matchup_pitcher, "Sean Burke")
        self.assertFalse(result.starter_source_conflict)
        self.assertTrue(result.confirmed_bulk_pitcher_used_for_matchup)
        self.assertIn("bulk pitcher is used for pitch-velocity matchup", result.notes)

    def test_unresolved_pitcher_disagreement_remains_source_conflict(self):
        result = resolve_starter_structure(
            StarterStructureContext(
                official_pitcher="Pitcher A",
                rotowire_pitcher="Pitcher B",
                official_pitcher_is_opener=False,
                rotowire_pitcher_is_bulk_candidate=False,
            )
        )

        self.assertEqual(result.matchup_pitcher, "Pitcher A")
        self.assertTrue(result.starter_source_conflict)
        self.assertFalse(result.confirmed_bulk_pitcher_used_for_matchup)


class StarterRecentFormTests(unittest.TestCase):
    def test_command_risk_lowers_starter_score(self):
        form = StarterRecentForm(
            era=3.47,
            whip=1.59,
            walks=20,
            innings=23.1,
            home_runs=1,
            pitches_per_inning=18.6,
        )

        result = score_starter_recent_form(form)

        self.assertLess(result.score, 60)
        self.assertTrue(result.command_risk)
        self.assertTrue(any("walk" in note for note in result.notes))

    def test_starter_volatility_flags_early_collapse_risk(self):
        context = StarterVolatilityContext(
            recent_whip=1.69,
            recent_walks_per_nine=5.1,
            recent_home_runs_per_nine=1.7,
            first_inning_era=8.10,
            recent_pitches_per_inning=19.2,
            sample_innings=12.0,
        )

        result = score_starter_volatility(context)

        self.assertLess(result.score, 50)
        self.assertTrue(result.collapse_risk)
        self.assertIn("first-inning risk", result.notes)


class BullpenSegmentTests(unittest.TestCase):
    def test_ninth_inning_instability_caps_full_game_confidence(self):
        segments = [
            BullpenSegment("bridge", score=78, recent_pitches=12, is_high_leverage=False),
            BullpenSegment("setup", score=74, recent_pitches=16, is_high_leverage=True),
            BullpenSegment("closer", score=38, recent_pitches=8, is_high_leverage=True),
        ]

        result = score_bullpen_segments(segments)

        self.assertLess(result.full_game_score, 65)
        self.assertTrue(result.late_inning_risk)
        self.assertIn("closer instability", result.notes)

    def test_strong_segmented_bullpen_supports_full_game(self):
        segments = [
            BullpenSegment("bridge", score=76, recent_pitches=8, is_high_leverage=False),
            BullpenSegment("setup", score=82, recent_pitches=11, is_high_leverage=True),
            BullpenSegment("closer", score=80, recent_pitches=0, is_high_leverage=True),
        ]

        result = score_bullpen_segments(segments)

        self.assertGreaterEqual(result.full_game_score, 75)
        self.assertFalse(result.late_inning_risk)


class MarketSignalTests(unittest.TestCase):
    def test_market_signal_with_bullpen_gap_is_only_review_flag(self):
        context = MarketSignalContext(
            signal_side="underdog",
            model_side="favorite",
            signal_count=4,
            line_moved_toward_signal=True,
            bullpen_supports_signal=False,
        )

        result = evaluate_market_signal(context)

        self.assertEqual(result.action, "review")
        self.assertFalse(result.flip_pick)
        self.assertIn("signal conflicts with bullpen model", result.notes)


class UmpireAdjustmentTests(unittest.TestCase):
    def test_narrow_zone_downgrades_command_risk_f5(self):
        umpire = UmpireContext(
            zone_size="narrow",
            walk_tendency="high",
            favors_low_zone=False,
            favors_outer_zone=False,
        )

        result = apply_umpire_adjustment(base_f5_edge=70, umpire=umpire, starter_command_risk=True)

        self.assertLess(result.adjusted_f5_edge, 70)
        self.assertIn("narrow zone amplifies command risk", result.notes)

    def test_wide_low_zone_supports_sinker_changeup_profile(self):
        umpire = UmpireContext(
            zone_size="wide",
            walk_tendency="low",
            favors_low_zone=True,
            favors_outer_zone=False,
        )

        result = apply_umpire_adjustment(
            base_f5_edge=68,
            umpire=umpire,
            starter_command_risk=False,
            pitch_profile="sinker_changeup",
        )

        self.assertGreater(result.adjusted_f5_edge, 68)
        self.assertIn("low-zone friendly umpire supports sinker/changeup profile", result.notes)


class HandicapTests(unittest.TestCase):
    def test_plus_handicap_requires_starter_survival(self):
        context = HandicapContext(
            lineup_edge=72,
            starter_stability=42,
            bullpen_score=70,
            opponent_late_risk=True,
            decimal_odds=1.72,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter survival missing", result.notes)

    def test_plus_handicap_blocks_small_sample_starter_without_clear_survival(self):
        context = HandicapContext(
            lineup_edge=72,
            starter_stability=66,
            bullpen_score=70,
            opponent_late_risk=True,
            decimal_odds=1.72,
            starter_survival_score=64,
            starter_sample_risk=True,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("small-sample starter survival risk", result.notes)

    def test_plus_handicap_is_playable_when_lineup_and_starter_can_hold(self):
        context = HandicapContext(
            lineup_edge=68,
            starter_stability=63,
            bullpen_score=66,
            opponent_late_risk=True,
            decimal_odds=1.65,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "play")

    def test_plus_handicap_caps_small_sample_command_risk(self):
        context = HandicapContext(
            lineup_edge=72,
            starter_stability=72,
            bullpen_score=72,
            opponent_late_risk=True,
            decimal_odds=1.72,
            starter_survival_score=72,
            starter_sample_risk=True,
            starter_walks_per_nine=4.3,
            starter_strikeout_walk_ratio=1.6,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "lean")
        self.assertIn("small-sample command risk caps plus handicap", result.notes)

    def test_plus_handicap_caps_repeated_recent_hr_risk(self):
        context = HandicapContext(
            lineup_edge=72,
            starter_stability=72,
            bullpen_score=72,
            opponent_late_risk=True,
            decimal_odds=1.72,
            starter_survival_score=72,
            starter_recent_home_run_repetition=True,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "lean")
        self.assertIn("recent starter HR repetition caps plus handicap", result.notes)


class TotalTests(unittest.TestCase):
    def test_under_requires_both_starters_and_bullpens_to_be_stable(self):
        context = TotalContext(
            market_total=8.0,
            projected_total=7.2,
            starter_stability_a=78,
            starter_stability_b=52,
            lineup_contact_a=48,
            lineup_contact_b=50,
            bullpen_score_a=74,
            bullpen_score_b=76,
            lower_bullpen_risk=False,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("under requires both starters stable", result.notes)

    def test_under_does_not_use_pitch_velocity_sample_shortage_as_support(self):
        context = TotalContext(
            market_total=10.5,
            projected_total=9.4,
            starter_stability_a=76,
            starter_stability_b=74,
            lineup_contact_a=48,
            lineup_contact_b=50,
            bullpen_score_a=72,
            bullpen_score_b=70,
            lower_bullpen_risk=False,
            pitch_velocity_unknown_rate_a=0.56,
            pitch_velocity_unknown_rate_b=0.20,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("pitch-velocity sample shortage is uncertainty, not under support", result.notes)

    def test_starter_source_conflict_is_not_over_support(self):
        context = TotalContext(
            market_total=8.0,
            projected_total=8.0,
            starter_stability_a=62,
            starter_stability_b=60,
            lineup_contact_a=60,
            lineup_contact_b=61,
            bullpen_score_a=62,
            bullpen_score_b=62,
            lower_bullpen_risk=False,
            starter_source_conflict=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter source conflict lowers confidence but is not over support", result.notes)

    def test_short_starter_midgame_viability_blocks_auto_over(self):
        context = TotalContext(
            market_total=8.0,
            projected_total=8.1,
            starter_stability_a=58,
            starter_stability_b=60,
            lineup_contact_a=64,
            lineup_contact_b=58,
            bullpen_score_a=58,
            bullpen_score_b=61,
            lower_bullpen_risk=True,
            short_starter_midgame_viable=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("short starter can reach middle innings; do not auto-upgrade over", result.notes)

    def test_both_starters_recent_stability_can_support_under_at_smaller_edge(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=7.0,
            starter_stability_a=72,
            starter_stability_b=74,
            lineup_contact_a=48,
            lineup_contact_b=50,
            bullpen_score_a=70,
            bullpen_score_b=68,
            lower_bullpen_risk=False,
            both_starters_recently_stable=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "under")
        self.assertIn("both starters' recent stability supports under", result.notes)

    def test_low_total_starter_duel_prioritizes_under_over_side(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=7.55,
            starter_stability_a=76,
            starter_stability_b=73,
            lineup_contact_a=50,
            lineup_contact_b=52,
            bullpen_score_a=71,
            bullpen_score_b=69,
            lower_bullpen_risk=False,
            low_total_starter_duel=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "under")
        self.assertIn("low-total starter duel prioritizes under over side", result.notes)

    def test_low_total_under_blocks_recent_command_or_hr_volatility(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=7.1,
            starter_stability_a=76,
            starter_stability_b=73,
            lineup_contact_a=50,
            lineup_contact_b=52,
            bullpen_score_a=71,
            bullpen_score_b=69,
            lower_bullpen_risk=False,
            both_starters_recently_stable=True,
            low_total_starter_duel=True,
            recent_starter_command_or_hr_volatility=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("recent starter command/HR volatility blocks low-total under", result.notes)

    def test_neutral_heavy_traffic_blocks_low_total_under(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=7.4,
            starter_stability_a=76,
            starter_stability_b=73,
            lineup_contact_a=54,
            lineup_contact_b=55,
            bullpen_score_a=71,
            bullpen_score_b=69,
            lower_bullpen_risk=False,
            low_total_starter_duel=True,
            neutral_heavy_traffic_risk=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("neutral-heavy traffic risk blocks low-total under", result.notes)

    def test_low_total_starter_length_keeps_under_candidate_despite_plus_slots(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=7.6,
            starter_stability_a=78,
            starter_stability_b=76,
            lineup_contact_a=64,
            lineup_contact_b=60,
            bullpen_score_a=72,
            bullpen_score_b=74,
            lower_bullpen_risk=False,
            low_total_starter_duel=True,
            both_starters_recently_stable=True,
            both_starters_likely_to_reach_sixth=True,
            starter_length_tempers_plus_slots=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "under")
        self.assertIn("starter length tempers pitch-velocity plus slots at low total", result.notes)

    def test_over_triggers_on_clear_starter_collapse_and_lineup_fit(self):
        context = TotalContext(
            market_total=9.0,
            projected_total=8.7,
            starter_stability_a=38,
            starter_stability_b=58,
            lineup_contact_a=78,
            lineup_contact_b=63,
            bullpen_score_a=54,
            bullpen_score_b=62,
            lower_bullpen_risk=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "over")
        self.assertIn("starter collapse plus lineup fit supports over", result.notes)

    def test_deep_starter_workload_blocks_lineup_fit_over(self):
        context = TotalContext(
            market_total=8.0,
            projected_total=8.6,
            starter_stability_a=68,
            starter_stability_b=70,
            lineup_contact_a=72,
            lineup_contact_b=64,
            bullpen_score_a=66,
            bullpen_score_b=68,
            lower_bullpen_risk=False,
            both_starters_can_work_deep=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter workload can suppress full-game over", result.notes)

    def test_sixth_inning_starter_expectation_blocks_lineup_fit_over(self):
        context = TotalContext(
            market_total=8.5,
            projected_total=8.8,
            starter_stability_a=66,
            starter_stability_b=68,
            lineup_contact_a=72,
            lineup_contact_b=63,
            bullpen_score_a=66,
            bullpen_score_b=67,
            lower_bullpen_risk=False,
            both_starters_likely_to_reach_sixth=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter length can mute lineup-fit over", result.notes)

    def test_post_sixth_bullpen_exposure_can_override_deep_starter_over_suppression(self):
        context = TotalContext(
            market_total=9.0,
            projected_total=8.9,
            starter_stability_a=68,
            starter_stability_b=70,
            lineup_contact_a=60,
            lineup_contact_b=58,
            bullpen_score_a=62,
            bullpen_score_b=68,
            lower_bullpen_risk=False,
            both_starters_can_work_deep=True,
            post_sixth_bullpen_exposure_risk=True,
            over_validation_strict=True,
            lineup_on_base_path_a=True,
            lineup_on_base_path_b=True,
            lower_order_run_continuity_a=True,
            lower_order_run_continuity_b=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "over")
        self.assertIn("post-sixth bullpen exposure keeps over live", result.notes)

    def test_low_total_under_passes_with_late_variance_risk(self):
        context = TotalContext(
            market_total=6.5,
            projected_total=6.0,
            starter_stability_a=74,
            starter_stability_b=72,
            lineup_contact_a=47,
            lineup_contact_b=50,
            bullpen_score_a=66,
            bullpen_score_b=67,
            lower_bullpen_risk=False,
            both_starters_recently_stable=True,
            low_total_late_variance_risk=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("low total leaves no margin for late variance", result.notes)

    def test_starter_under_with_pen_risk_does_not_become_full_game_under(self):
        context = TotalContext(
            market_total=7.5,
            projected_total=6.8,
            starter_stability_a=76,
            starter_stability_b=74,
            lineup_contact_a=48,
            lineup_contact_b=49,
            bullpen_score_a=64,
            bullpen_score_b=67,
            lower_bullpen_risk=True,
            starter_under_but_full_game_pen_risk=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter under only; bullpen risk blocks full-game under", result.notes)

    def test_negative_power_bat_variance_blocks_thin_full_game_under(self):
        context = TotalContext(
            market_total=8.0,
            projected_total=7.2,
            starter_stability_a=76,
            starter_stability_b=74,
            lineup_contact_a=48,
            lineup_contact_b=50,
            bullpen_score_a=72,
            bullpen_score_b=70,
            lower_bullpen_risk=False,
            negative_power_bat_variance=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn(
            "negative pitch-velocity power bats add run variance, not under support",
            result.notes,
        )

    def test_weak_favorite_lineup_moves_under_review_ahead_of_ml(self):
        context = TotalContext(
            market_total=8.5,
            projected_total=7.8,
            starter_stability_a=78,
            starter_stability_b=72,
            lineup_contact_a=48,
            lineup_contact_b=50,
            bullpen_score_a=70,
            bullpen_score_b=71,
            lower_bullpen_risk=False,
            weak_favorite_lineup_with_starter_edge=True,
            under_validation_strict=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "under")
        self.assertIn("weak favorite lineup makes total review higher priority than ML", result.notes)
        self.assertIn("strict under validation passed", result.notes)

    def test_two_sided_volatility_can_prioritize_over_over_side(self):
        context = TotalContext(
            market_total=9.0,
            projected_total=8.9,
            starter_stability_a=42,
            starter_stability_b=56,
            lineup_contact_a=64,
            lineup_contact_b=61,
            bullpen_score_a=58,
            bullpen_score_b=62,
            lower_bullpen_risk=True,
            two_sided_over_volatility=True,
            over_validation_strict=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "over")
        self.assertIn("two-sided volatility makes over preferable to side", result.notes)

    def test_high_total_over_can_still_play_with_two_command_risk_starters(self):
        context = TotalContext(
            market_total=10.5,
            projected_total=10.3,
            starter_stability_a=52,
            starter_stability_b=54,
            lineup_contact_a=63,
            lineup_contact_b=64,
            bullpen_score_a=63,
            bullpen_score_b=64,
            lower_bullpen_risk=False,
            two_sided_over_volatility=True,
            over_validation_strict=True,
            lineup_on_base_path_a=True,
            lineup_on_base_path_b=True,
            lower_order_run_continuity_a=True,
            lower_order_run_continuity_b=True,
            both_starters_command_risk=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "over")
        self.assertIn("two command-risk starters keep high-total over live", result.notes)
        self.assertIn("two-sided volatility makes over preferable to side", result.notes)

    def test_high_total_over_requires_recent_power_or_walk_conversion(self):
        context = TotalContext(
            market_total=10.0,
            projected_total=9.9,
            starter_stability_a=52,
            starter_stability_b=54,
            lineup_contact_a=63,
            lineup_contact_b=64,
            bullpen_score_a=63,
            bullpen_score_b=64,
            lower_bullpen_risk=False,
            two_sided_over_volatility=True,
            over_validation_strict=True,
            lineup_on_base_path_a=True,
            lineup_on_base_path_b=True,
            lower_order_run_continuity_a=True,
            lower_order_run_continuity_b=True,
            both_starters_command_risk=True,
            high_total_recent_power_walk_conversion=False,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("high-total over requires recent power or walk conversion", result.notes)

    def test_starter_kbb_length_suppresses_pitch_velocity_over(self):
        context = TotalContext(
            market_total=9.5,
            projected_total=9.8,
            starter_stability_a=72,
            starter_stability_b=74,
            lineup_contact_a=72,
            lineup_contact_b=66,
            bullpen_score_a=70,
            bullpen_score_b=72,
            lower_bullpen_risk=False,
            both_starters_likely_to_reach_sixth=True,
            starter_kbb_length_suppresses_over=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("starter K/BB length suppresses pitch-velocity over", result.notes)

    def test_sixth_inning_path_requires_walk_or_bullpen_support_for_over(self):
        context = TotalContext(
            market_total=8.5,
            projected_total=8.9,
            starter_stability_a=70,
            starter_stability_b=72,
            lineup_contact_a=68,
            lineup_contact_b=66,
            bullpen_score_a=70,
            bullpen_score_b=72,
            lower_bullpen_risk=False,
            both_starters_likely_to_reach_sixth=True,
            over_lacks_walk_or_late_bullpen_path=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn(
            "sixth-inning starter path requires walk or late-bullpen support for over",
            result.notes,
        )

    def test_strict_over_validation_requires_two_sided_on_base_and_lower_order_path(self):
        context = TotalContext(
            market_total=8.5,
            projected_total=8.6,
            starter_stability_a=42,
            starter_stability_b=56,
            lineup_contact_a=68,
            lineup_contact_b=64,
            bullpen_score_a=58,
            bullpen_score_b=62,
            lower_bullpen_risk=True,
            two_sided_over_volatility=True,
            over_validation_strict=True,
            lineup_on_base_path_a=True,
            lineup_on_base_path_b=False,
            lower_order_run_continuity_a=True,
            lower_order_run_continuity_b=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn(
            "strict over validation requires two-sided on-base and lower-order continuity",
            result.notes,
        )

    def test_strict_over_validation_blocks_one_sided_volatility(self):
        context = TotalContext(
            market_total=9.0,
            projected_total=9.1,
            starter_stability_a=42,
            starter_stability_b=70,
            lineup_contact_a=74,
            lineup_contact_b=52,
            bullpen_score_a=66,
            bullpen_score_b=68,
            lower_bullpen_risk=False,
            over_validation_strict=True,
        )

        result = evaluate_total(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("strict over validation requires two-sided scoring or bullpen exposure", result.notes)


class LiveEntryTests(unittest.TestCase):
    def test_early_starter_exit_blocks_live_ml(self):
        state = LiveGameState(
            inning=2,
            run_differential=-3,
            starter_exited=True,
            bullpen_innings_needed=7.0,
            long_relief_score=52,
            decimal_odds=2.40,
            handicap=0.0,
        )

        result = evaluate_live_entry(state)

        self.assertEqual(result.recommendation, "pass")
        self.assertIn("early starter exit", result.notes)

    def test_plus_handicap_can_be_playable_after_long_relief_stabilizes(self):
        state = LiveGameState(
            inning=3,
            run_differential=-3,
            starter_exited=True,
            bullpen_innings_needed=6.0,
            long_relief_score=72,
            decimal_odds=1.72,
            handicap=3.5,
        )

        result = evaluate_live_entry(state)

        self.assertEqual(result.recommendation, "lean")
        self.assertIn("handicap cushion", result.notes)


class RecommendationTests(unittest.TestCase):
    def test_fatigued_bullpen_downgrades_full_game_ml_to_pass(self):
        context = FullGameContext(
            f5_edge=72,
            lineup_edge=65,
            bullpen_score=45,
            bullpen_quality=55,
            market_total=8.0,
            projected_total=7.4,
            run_environment=0,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_ml, "pass")
        self.assertEqual(result.f5_ml, "lean")

    def test_starter_collapse_risk_downgrades_f5_before_full_game_ml(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=66,
            bullpen_score=76,
            bullpen_quality=78,
            market_total=8.5,
            projected_total=8.0,
            run_environment=0,
            starter_collapse_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("starter collapse risk downgrades F5 first", result.notes)

    def test_late_bullpen_risk_prefers_f5_even_when_bullpen_average_is_ok(self):
        context = FullGameContext(
            f5_edge=81,
            lineup_edge=62,
            bullpen_score=72,
            bullpen_quality=70,
            market_total=7.5,
            projected_total=6.4,
            run_environment=-2,
            late_bullpen_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "play")
        self.assertEqual(result.full_game_ml, "pass")
        self.assertEqual(result.full_game_total, "pass")
        self.assertIn("late bullpen risk blocks full-game under", result.notes)

    def test_f5_only_edge_blocks_duplicate_full_game_ml_priority(self):
        context = FullGameContext(
            f5_edge=82,
            lineup_edge=68,
            bullpen_score=70,
            bullpen_quality=69,
            market_total=8.5,
            projected_total=8.2,
            run_environment=0,
            f5_only_preferred=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "play")
        self.assertEqual(result.full_game_ml, "pass")
        self.assertIn("F5-only edge blocks duplicate full-game ML priority", result.notes)

    def test_fatigued_bullpen_blocks_full_game_under(self):
        context = FullGameContext(
            f5_edge=75,
            lineup_edge=60,
            bullpen_score=48,
            bullpen_quality=60,
            market_total=9.0,
            projected_total=7.6,
            run_environment=-1,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_total, "pass")
        self.assertIn("bullpen fatigue blocks full-game under", result.notes)

    def test_underdog_pitch_velocity_plus_volume_downgrades_favorite_ml(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=72,
            bullpen_score=76,
            bullpen_quality=74,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            model_side_is_favorite=True,
            opponent_pitch_velocity_plus_slots=8,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("underdog lineup has 7+ pitch-velocity plus slots; favorite ML downgraded", result.notes)

    def test_stable_prim_bulk_pitcher_downgrades_opposing_favorite_ml(self):
        context = FullGameContext(
            f5_edge=79,
            lineup_edge=70,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.5,
            projected_total=8.1,
            run_environment=0,
            model_side_is_favorite=True,
            opponent_prim_bulk_stable=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("stable PRIM/bulk pitcher lowers favorite side confidence", result.notes)

    def test_dominant_opposing_starter_downgrades_plus_slot_lineup_edge(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=76,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=7.5,
            projected_total=7.4,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=8,
            opposing_starter_recent_dominance=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("dominant opposing starter tempers pitch-velocity plus slots", result.notes)

    def test_contact_oriented_opponent_tempers_starter_form_side(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=64,
            bullpen_score=70,
            bullpen_quality=70,
            market_total=8.5,
            projected_total=8.0,
            run_environment=0,
            opponent_contact_pressure=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("contact-oriented opponent tempers starter-form edge", result.notes)

    def test_unknown_same_handed_lineup_adds_power_variance(self):
        context = FullGameContext(
            f5_edge=82,
            lineup_edge=66,
            bullpen_score=74,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.6,
            run_environment=0,
            pitch_velocity_unknown_rate_against_model_starter=0.70,
            same_handedness_stack_against_model_starter=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("same-handed lineup with pitch-velocity uncertainty adds power variance", result.notes)

    def test_starter_source_conflict_blocks_strong_f5(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=68,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            starter_source_conflict=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("starter source conflict blocks strong F5 confidence", result.notes)

    def test_plus_slots_without_top_order_quality_downgrade_ml(self):
        context = FullGameContext(
            f5_edge=72,
            lineup_edge=76,
            bullpen_score=74,
            bullpen_quality=74,
            market_total=8.0,
            projected_total=7.7,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=8,
            top_order_pitch_velocity_plus_slots=2,
            power_pitch_velocity_plus_slots=1,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("plus slots lack top-order or power concentration", result.notes)

    def test_low_recent_run_efficiency_tempers_pitch_velocity_side(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=74,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.9,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=7,
            low_recent_run_efficiency=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("low recent run efficiency tempers pitch-velocity edge", result.notes)

    def test_high_recent_strikeout_rate_and_unfamiliar_starter_cap_plus_slot_side(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=78,
            bullpen_score=78,
            bullpen_quality=78,
            market_total=9.0,
            projected_total=8.8,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=8,
            high_recent_strikeout_rate=True,
            unfamiliar_opposing_starter=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn(
            "high recent strikeout rate plus unfamiliar starter caps pitch-velocity side",
            result.notes,
        )

    def test_plus_slots_with_low_run_efficiency_move_ml_to_pass_or_total_review(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=76,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.5,
            projected_total=8.4,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=7,
            low_recent_run_efficiency=True,
            plus_slots_low_run_efficiency_prefers_total=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_ml, "pass")
        self.assertIn("plus slots with poor run conversion move ML to pass or total review", result.notes)

    def test_short_favorite_with_starter_command_or_hr_risk_cannot_be_top_priority(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=76,
            bullpen_score=78,
            bullpen_quality=78,
            market_total=8.0,
            projected_total=7.9,
            run_environment=0,
            model_side_decimal_odds=1.57,
            starter_command_or_hr_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("short favorite with starter command/HR risk capped", result.notes)

    def test_short_favorite_f5_downgrades_when_core_hitters_have_minus_matchups(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=74,
            bullpen_score=78,
            bullpen_quality=78,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            model_side_decimal_odds=1.42,
            model_lineup_core_minus_slots=3,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertIn("core lineup pitch-velocity minus concentration caps F5", result.notes)

    def test_key_power_bat_missing_downgrades_side_priority(self):
        context = FullGameContext(
            f5_edge=82,
            lineup_edge=74,
            bullpen_score=76,
            bullpen_quality=76,
            market_total=8.0,
            projected_total=7.9,
            run_environment=0,
            key_power_bat_missing=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("key power bat missing lowers side priority", result.notes)

    def test_injury_return_or_short_exit_risk_caps_full_game_ml(self):
        context = FullGameContext(
            f5_edge=76,
            lineup_edge=74,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            injury_return_or_recent_short_exit_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("injury return or recent short exit caps full-game ML", result.notes)

    def test_two_sided_run_volatility_separates_over_from_side(self):
        context = FullGameContext(
            f5_edge=72,
            lineup_edge=70,
            bullpen_score=65,
            bullpen_quality=60,
            market_total=8.0,
            projected_total=9.2,
            run_environment=0,
            two_sided_run_volatility=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_total, "over")
        self.assertEqual(result.full_game_ml, "pass")
        self.assertIn("two-sided run volatility separates over from side", result.notes)

    def test_post_sixth_bullpen_exposure_caps_full_game_ml(self):
        context = FullGameContext(
            f5_edge=72,
            lineup_edge=68,
            bullpen_score=70,
            bullpen_quality=68,
            market_total=9.0,
            projected_total=9.1,
            run_environment=0,
            post_sixth_bullpen_exposure_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "pass")
        self.assertIn("post-sixth bullpen exposure downgrades full-game ML", result.notes)

    def test_opponent_top_order_plus_against_hr_risk_starter_blocks_model_side_ml(self):
        context = FullGameContext(
            f5_edge=72,
            lineup_edge=66,
            bullpen_score=76,
            bullpen_quality=74,
            market_total=9.0,
            projected_total=10.0,
            run_environment=0,
            opponent_top_order_pitch_velocity_plus_slots=3,
            model_starter_hr_or_collapse_risk=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.full_game_total, "over")
        self.assertEqual(result.full_game_ml, "pass")
        self.assertIn("opponent top-order pitch-velocity edge attacks starter HR/collapse risk", result.notes)

    def test_confirmed_bulk_pitcher_resolves_starter_source_conflict(self):
        context = FullGameContext(
            f5_edge=74,
            lineup_edge=68,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=8.2,
            run_environment=0,
            starter_source_conflict=True,
            confirmed_bulk_pitcher_used_for_matchup=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertNotIn("starter source conflict blocks strong F5 confidence", result.notes)
        self.assertIn("confirmed bulk pitcher resolves starter-source conflict", result.notes)

    def test_fastball_only_plus_slots_against_two_pitch_starter_downgrade_side(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=74,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.7,
            run_environment=0,
            lineup_pitch_velocity_plus_slots=7,
            fastball_only_pitch_velocity_plus_slots=6,
            opposing_starter_has_two_pitch_command=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("fastball-only plus slots are capped by starter's secondary command", result.notes)

    def test_neutral_heavy_opponent_and_negative_power_bat_cap_side(self):
        context = FullGameContext(
            f5_edge=84,
            lineup_edge=78,
            bullpen_score=78,
            bullpen_quality=78,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            opponent_neutral_pitch_velocity_slots=6,
            opponent_pitch_velocity_minus_slots=2,
            opponent_negative_power_bat_slots=1,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("neutral-heavy opponent lineup can create traffic", result.notes)
        self.assertIn(
            "negative pitch-velocity power bats still carry one-swing variance",
            result.notes,
        )

    def test_plus_handicap_needs_lower_order_run_continuity(self):
        context = HandicapContext(
            lineup_edge=70,
            starter_stability=68,
            bullpen_score=70,
            opponent_late_risk=True,
            decimal_odds=1.72,
            lower_order_run_continuity=False,
            pitch_velocity_edge_fastball_only=True,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "lean")
        self.assertIn("fastball-only edge with weak lower-order continuity caps plus handicap", result.notes)

    def test_plus_handicap_passes_when_fastball_edge_lacks_secondary_support(self):
        context = HandicapContext(
            lineup_edge=72,
            starter_stability=70,
            bullpen_score=72,
            opponent_late_risk=True,
            decimal_odds=1.72,
            lower_order_run_continuity=False,
            pitch_velocity_edge_fastball_only=True,
            secondary_pitch_support=False,
        )

        result = evaluate_plus_handicap(context)

        self.assertEqual(result.rating, "pass")
        self.assertIn("fastball edge lacks secondary-pitch support", result.notes)

    def test_stable_prim_bulk_with_weak_opponent_fit_downgrades_model_side(self):
        context = FullGameContext(
            f5_edge=76,
            lineup_edge=70,
            bullpen_score=72,
            bullpen_quality=72,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            opposing_prim_bulk_with_weak_lineup_fit=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("stable PRIM/bulk starter with weak lineup fit is not a fade", result.notes)

    def test_weak_early_scoring_support_blocks_strong_f5_only(self):
        context = FullGameContext(
            f5_edge=80,
            lineup_edge=68,
            bullpen_score=74,
            bullpen_quality=74,
            market_total=8.0,
            projected_total=7.8,
            run_environment=0,
            weak_early_scoring_support=True,
        )

        result = recommend_full_game(context)

        self.assertEqual(result.f5_ml, "lean")
        self.assertEqual(result.full_game_ml, "lean")
        self.assertIn("weak early scoring support caps F5 confidence", result.notes)

    def test_small_pitch_velocity_sample_is_weak_evidence(self):
        result = evaluate_pitch_velocity_sample(pa=4, woba=0.650)

        self.assertEqual(result.strength, "weak")
        self.assertLess(result.adjusted_woba, 0.650)

    def test_positive_line_value_requires_model_edge_over_market(self):
        result = evaluate_line_value(LineValueContext(model_probability=0.58, implied_probability=0.52))

        self.assertEqual(result.rating, "value")
        self.assertGreater(result.edge, 0)

    def test_confidence_rewards_confirmed_core_inputs_and_penalizes_flags(self):
        checklist = AnalysisChecklist(
            confirmed_lineup=True,
            starter_pitch_velocity=True,
            batter_pitch_velocity=True,
            handedness_splits=True,
            recent_form=True,
            starter_recent_form=True,
            starter_command_risk=True,
            bullpen_availability=True,
            bullpen_quality=True,
            bullpen_handedness=False,
            park_weather=True,
            defense_baserunning=True,
            catcher_battery=False,
            run_creation_style=False,
            bench_quality=False,
            umpire=False,
            market_line=True,
            line_value=True,
            f5_full_game_split=True,
            review_tags=True,
            caution_flags=["bullpen handedness unknown", "umpire unknown"],
        )

        result = calculate_confidence_score(checklist)

        self.assertEqual(result.grade, "medium")
        self.assertLess(result.score, 80)
        self.assertIn("bullpen handedness unknown", result.caution_flags)


if __name__ == "__main__":
    unittest.main()
