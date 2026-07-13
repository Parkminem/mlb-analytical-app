import unittest

from analysis.pipeline import (
    MODEL_LAYERS,
    build_analysis_manifest,
    classify_pitch_velocity_slot,
    load_model_config,
)


class ModelConfigTests(unittest.TestCase):
    def test_default_config_preserves_fixed_source_and_weather_rules(self):
        config = load_model_config()

        self.assertEqual(config["lineup_source"]["primary"], "RotoWire")
        self.assertTrue(config["lineup_source"]["use_projected_when_unconfirmed"])
        self.assertFalse(config["weather"]["included_by_default"])
        self.assertEqual(config["bullpen"]["yesterday_performance_risk_min_pitches"], 30)
        self.assertTrue(
            config["bullpen"]["lower_bullpen_usage_preserves_high_leverage_availability"]
        )
        self.assertTrue(
            config["starter_structure"][
                "official_opener_plus_rotowire_bulk_resolves_conflict"
            ]
        )
        self.assertTrue(config["starter_structure"]["use_bulk_pitcher_for_main_matchup"])
        self.assertTrue(
            config["starter_structure"]["downgrade_side_when_opener_bulk_timing_uncertain"]
        )
        self.assertTrue(
            config["starter_structure"][
                "prim_bulk_requires_recent_workload_or_explicit_confirmation"
            ]
        )
        self.assertEqual(config["starter_structure"]["stable_bulk_recent_workload_min_pitches"], 50)
        self.assertTrue(config["starter_structure"]["uncertain_prim_keeps_source_conflict_cap"])
        self.assertTrue(config["side_priority"]["downgrade_key_power_bat_missing"])
        self.assertTrue(
            config["side_priority"][
                "downgrade_ml_when_opponent_top_order_attacks_starter_hr_or_collapse_risk"
            ]
        )
        self.assertTrue(
            config["side_priority"][
                "block_ml_when_opponent_top_order_attacks_hr_risk_starter"
            ]
        )
        self.assertTrue(
            config["side_priority"]["confirmed_bulk_pitcher_resolves_starter_source_conflict"]
        )
        self.assertTrue(config["side_priority"]["cap_side_against_neutral_heavy_lineup"])
        self.assertTrue(config["side_priority"]["treat_negative_power_bats_as_variance"])
        self.assertTrue(config["total_priority"]["compete_with_sides_in_final_priority"])
        self.assertTrue(config["total_priority"]["strict_under_validation"])
        self.assertTrue(config["total_priority"]["strict_over_validation"])
        self.assertTrue(config["total_priority"]["low_total_starter_length_can_temper_plus_slots"])
        self.assertTrue(config["total_priority"]["strict_over_requires_two_sided_on_base_path"])
        self.assertTrue(config["total_priority"]["block_low_total_under_on_recent_command_or_hr_volatility"])
        self.assertTrue(config["total_priority"]["block_low_total_under_on_recent_power_hr_variance"])
        self.assertTrue(config["total_priority"]["block_low_total_under_on_neutral_heavy_traffic"])
        self.assertTrue(config["total_priority"]["allow_high_total_over_when_both_starters_have_command_risk"])
        self.assertTrue(config["total_priority"]["high_total_over_requires_recent_power_or_walk_conversion"])
        self.assertTrue(config["total_priority"]["starter_kbb_length_suppresses_pitch_velocity_over"])
        self.assertTrue(config["total_priority"]["recent_zero_walk_kbb_length_caps_thin_over"])
        self.assertTrue(config["side_priority"]["cap_plus_handicap_on_small_sample_command_risk"])
        self.assertTrue(config["side_priority"]["cap_plus_handicap_on_recent_hr_repetition"])
        self.assertTrue(config["side_priority"]["avoid_duplicate_ml_when_f5_only_edge"])
        self.assertTrue(config["side_priority"]["cap_f5_when_core_hitters_have_minus_matchups"])
        self.assertTrue(config["side_priority"]["move_plus_slots_low_run_efficiency_to_pass_or_total_review"])
        self.assertTrue(config["side_priority"]["downgrade_ml_on_post_sixth_bullpen_exposure"])
        self.assertTrue(config["side_priority"]["downgrade_ml_when_short_starter_requires_extended_bullpen"])
        self.assertTrue(config["side_priority"]["cap_plus_slots_on_high_k_rate_and_unfamiliar_starter"])
        self.assertTrue(config["total_priority"]["allow_over_on_post_sixth_bullpen_exposure"])
        self.assertTrue(config["total_priority"]["sixth_inning_path_requires_walk_or_late_bullpen_over_support"])
        self.assertTrue(config["total_priority"]["neutral_heavy_traffic_blocks_thin_full_game_under"])
        self.assertTrue(config["total_priority"]["prim_bulk_length_tempers_high_total_over"])
        self.assertTrue(config["side_priority"]["cap_recent_power_form_without_on_base_pressure"])
        self.assertTrue(config["side_priority"]["downgrade_cold_lineup_ml_toward_total_review"])
        self.assertTrue(config["side_priority"]["cap_short_favorite_ml_on_weak_confirmed_lineup_fit"])
        self.assertTrue(
            config["side_priority"][
                "cap_plus_handicap_on_sample_shortage_against_small_sample_starter"
            ]
        )
        self.assertTrue(
            config["side_priority"]["cap_fastball_plus_slots_on_secondary_command_rebound"]
        )
        self.assertTrue(
            config["side_priority"][
                "cap_plus_handicap_against_stable_deep_starter_without_top_order_support"
            ]
        )
        self.assertTrue(
            config["side_priority"]["block_plus_handicap_when_underdog_run_creation_is_cold"]
        )
        self.assertTrue(
            config["side_priority"]["plus_handicap_requires_independent_underdog_scoring_path"]
        )
        self.assertTrue(
            config["side_priority"]["block_ml_when_model_starter_walk_risk_requires_f5_only"]
        )
        self.assertTrue(
            config["side_priority"]["plus_handicap_starter_survival_overrides_lineup_edge"]
        )
        self.assertTrue(
            config["side_priority"]["cap_plus_handicap_on_general_command_or_hr_risk"]
        )
        self.assertTrue(
            config["side_priority"][
                "block_plus_handicap_on_underdog_starter_collapse_risk"
            ]
        )
        self.assertTrue(
            config["side_priority"][
                "cap_short_favorite_elite_starter_on_top_order_extra_base_route"
            ]
        )
        self.assertTrue(
            config["side_priority"]["plus_handicap_requires_late_bullpen_protection"]
        )
        self.assertTrue(
            config["total_priority"][
                "block_under_when_key_bats_missing_but_depth_remains_live"
            ]
        )
        self.assertTrue(
            config["total_priority"]["block_under_on_weak_bulk_or_lower_bullpen_collapse_risk"]
        )
        self.assertTrue(config["total_priority"]["block_over_on_one_sided_scoring_path"])
        self.assertTrue(
            config["total_priority"]["block_high_total_over_when_one_lineup_path_is_weak"]
        )
        self.assertTrue(
            config["total_priority"][
                "block_high_total_over_when_bridge_can_absorb_bullpen_fatigue"
            ]
        )
        self.assertTrue(
            config["total_priority"]["cap_fastball_only_over_on_secondary_command_rebound"]
        )
        self.assertTrue(
            config["total_priority"][
                "keep_high_total_over_live_on_single_starter_collapse_plus_late_path"
            ]
        )
        self.assertTrue(
            config["total_priority"]["block_pitch_velocity_over_on_cold_run_conversion"]
        )
        self.assertTrue(
            config["total_priority"][
                "market_nine_over_requires_more_than_five_inning_starter_floor"
            ]
        )
        self.assertTrue(config["total_priority"]["over_priority_requires_f5_scoring_path"])
        self.assertTrue(config["total_priority"]["stable_bulk_length_caps_high_total_over"])
        self.assertTrue(
            config["total_priority"][
                "bullpen_over_requires_starter_length_failure_or_post_sixth_exposure"
            ]
        )
        self.assertTrue(
            config["total_priority"][
                "extreme_total_over_requires_clear_two_sided_collapse_path"
            ]
        )
        self.assertTrue(
            config["total_priority"][
                "one_dominant_kbb_starter_prioritizes_low_total_under"
            ]
        )
        self.assertTrue(
            config["total_priority"]["traffic_over_requires_power_or_run_conversion_support"]
        )
        self.assertEqual(config["cache_policy"]["savant_statcast_historical_ttl_hours"], 24)
        self.assertEqual(config["cache_policy"]["mlb_official_schedule_probables"], "realtime")
        self.assertEqual(config["cache_policy"]["rotowire_lineups"], "realtime")
        self.assertEqual(config["cache_policy"]["market_odds_and_totals"], "realtime")
        self.assertEqual(config["cache_policy"]["boxscore_bullpen_usage"], "realtime")
        self.assertTrue(config["market"]["summary_table_single_side_market"])
        self.assertEqual(config["priority_admission"]["traffic_only_side_cap"], 74)
        self.assertEqual(config["priority_admission"]["incomplete_pitching_chain_cap"], 74)
        self.assertEqual(config["priority_admission"]["late_bullpen_protection_cap"], 71)
        self.assertEqual(config["priority_admission"]["discontinuous_lineup_over_cap"], 69)
        self.assertTrue(config["priority_admission"]["caps_only_no_positive_rescoring"])
        self.assertTrue(
            config["priority_admission"]["exclude_market_engine_pass_from_priority"]
        )
        self.assertTrue(
            config["priority_admission"]["canonical_evidence_paths_only"]
        )
        self.assertTrue(
            config["priority_admission"][
                "missing_evidence_blocks_multi_market_retention"
            ]
        )
        self.assertEqual(
            config["priority_admission"]["same_game_candidate_assembler"],
            "assemble_same_game_priority",
        )
        self.assertTrue(config["f5_total_priority"]["independent_engine"])
        self.assertFalse(
            config["f5_total_priority"]["use_full_game_bullpen_inputs"]
        )
        self.assertTrue(
            config["f5_total_priority"]["require_verified_market_line_and_price"]
        )
        self.assertTrue(
            config["f5_total_priority"]["deduplicate_correlated_full_game_total"]
        )

    def test_model_layers_keep_the_documented_order(self):
        self.assertEqual(
            MODEL_LAYERS,
            (
                "data_integrity",
                "starter_lineup_matchup",
                "lineup_form",
                "starter_stability",
                "bullpen",
                "supporting_context",
                "betting_decision_engines",
                "market_value_and_final_output",
            ),
        )


class PitchVelocitySlotTests(unittest.TestCase):
    def test_small_sample_is_labeled_as_insufficient_not_negative(self):
        result = classify_pitch_velocity_slot(pa=4, woba=0.410)

        self.assertEqual(result.label, "표본 부족")
        self.assertEqual(result.evidence_strength, "weak")

    def test_adjusted_woba_is_used_for_slot_label(self):
        result = classify_pitch_velocity_slot(pa=12, woba=0.390)

        self.assertEqual(result.evidence_strength, "moderate")
        self.assertEqual(result.label, "플러스")


class AnalysisManifestTests(unittest.TestCase):
    def test_projected_lineup_changes_confidence_not_team_strength(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=False,
            starter_confirmed=True,
            include_weather=False,
        )

        self.assertIn("projected_lineup", manifest["confidence_flags"])
        self.assertNotIn("projected_lineup", manifest["strength_adjustments"])
        self.assertEqual(manifest["lineup_source"], "RotoWire projected lineup")

    def test_manifest_exposes_cache_policy(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            include_weather=False,
        )

        self.assertEqual(
            manifest["cache_policy"]["savant_statcast_historical_ttl_hours"],
            24,
        )
        self.assertEqual(
            manifest["cache_policy"]["rotowire_lineups"],
            "realtime",
        )
        self.assertIn("F5 O/U", manifest["required_outputs"])

    def test_manifest_requires_unified_same_game_priority_assembler(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            include_weather=False,
        )

        self.assertEqual(
            manifest["priority_candidate_assembler"],
            "assemble_same_game_priority",
        )

    def test_weather_is_excluded_unless_requested(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            include_weather=False,
        )

        self.assertFalse(manifest["include_weather"])
        self.assertNotIn("weather_and_park", manifest["required_inputs"])


if __name__ == "__main__":
    unittest.main()
