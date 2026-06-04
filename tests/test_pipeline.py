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
        self.assertTrue(config["side_priority"]["downgrade_key_power_bat_missing"])

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
