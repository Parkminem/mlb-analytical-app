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
            config["starter_structure"][
                "official_opener_plus_rotowire_bulk_resolves_conflict"
            ]
        )
        self.assertTrue(config["starter_structure"]["use_bulk_pitcher_for_main_matchup"])
        self.assertTrue(config["side_priority"]["downgrade_key_power_bat_missing"])
        self.assertTrue(
            config["side_priority"][
                "downgrade_ml_when_opponent_top_order_attacks_starter_hr_or_collapse_risk"
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
        self.assertTrue(config["total_priority"]["block_low_total_under_on_neutral_heavy_traffic"])
        self.assertTrue(config["total_priority"]["allow_high_total_over_when_both_starters_have_command_risk"])
        self.assertTrue(config["total_priority"]["high_total_over_requires_recent_power_or_walk_conversion"])
        self.assertTrue(config["total_priority"]["starter_kbb_length_suppresses_pitch_velocity_over"])
        self.assertTrue(config["side_priority"]["cap_plus_handicap_on_small_sample_command_risk"])
        self.assertTrue(config["side_priority"]["cap_plus_handicap_on_recent_hr_repetition"])
        self.assertTrue(config["side_priority"]["avoid_duplicate_ml_when_f5_only_edge"])
        self.assertTrue(config["side_priority"]["cap_f5_when_core_hitters_have_minus_matchups"])
        self.assertTrue(config["side_priority"]["move_plus_slots_low_run_efficiency_to_pass_or_total_review"])
        self.assertTrue(config["side_priority"]["downgrade_ml_on_post_sixth_bullpen_exposure"])
        self.assertTrue(config["side_priority"]["cap_plus_slots_on_high_k_rate_and_unfamiliar_starter"])
        self.assertTrue(config["total_priority"]["allow_over_on_post_sixth_bullpen_exposure"])
        self.assertTrue(config["total_priority"]["sixth_inning_path_requires_walk_or_late_bullpen_over_support"])

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
