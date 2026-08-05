import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.pipeline import (
    MODEL_LAYERS,
    build_analysis_manifest,
    classify_pitch_velocity_slot,
    load_model_config,
)


EXPECTED_FORECASTING_CONFIG = {
    "enabled": True,
    "mode": "shadow",
    "model_version": "shadow-forecast-2026-07-14.1",
    "may_change_recommendations": False,
    "simulations": 20_000,
    "league_runs_per_team_game": 4.5,
    "league_woba": 0.320,
    "gamma_poisson_shape": 1.6,
    "extra_inning_multiplier": 1.8,
    "max_innings": 30,
    "offense_bounds": [0.65, 1.45],
    "matchup_bounds": [0.80, 1.20],
    "starter_ra9_bounds": [0.55, 1.75],
    "bullpen_ra9_bounds": [0.60, 1.90],
    "pitch_limit_high_bucket_transfer": 1.0,
    "pitch_limit_transfer_to_under_five": 0.4,
    "sample_shortage_log_se_penalty": 0.12,
    "unknown_bullpen_matchup_log_se_penalty": 0.08,
    "handoff_uncertainty_log_se_penalty": 0.10,
    "injury_return_log_se_penalty": 0.12,
    "recent_short_exit_log_se_penalty": 0.10,
    "limited_reliever_log_se_penalty": 0.10,
    "unscheduled_f5_relief_log_standard_error": 0.15,
    "pa_weights": [1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87],
    "confirmed_lineup_validation_games": 100,
    "market_family_tuning_decisions": 300,
}

LEGACY_CONFIG_SHA256 = "b36826bdb038e853a1188cd4743508a9af727750ec8826b934e0217422302181"


class ModelConfigTests(unittest.TestCase):
    @staticmethod
    def _load_config_payload(payload):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "model.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            return load_model_config(config_path)

    def test_forecasting_block_matches_approved_shadow_configuration(self):
        config = load_model_config()

        self.assertEqual(config["version"], "2026-08-05")
        self.assertEqual(config.get("forecasting"), EXPECTED_FORECASTING_CONFIG)
        self.assertTrue(
            config["priority_admission"][
                "require_ml_bullpen_review_path"
            ]
        )
        self.assertTrue(
            config["priority_admission"][
                "require_two_sided_under_review_path"
            ]
        )
        self.assertTrue(
            config["priority_admission"][
                "matchup_only_side_requires_recent_run_conversion"
            ]
        )
        self.assertTrue(
            config["priority_admission"][
                "full_game_bullpen_requires_performance_support"
            ]
        )
        self.assertTrue(
            config["priority_admission"][
                "ml_requires_independent_full_game_margin_path"
            ]
        )
        self.assertEqual(
            config["priority_admission"]["ml_starter_evidence_cap"],
            71,
        )
        self.assertEqual(
            config["priority_admission"]["high_total_two_sided_over_cap"],
            69,
        )
        self.assertEqual(
            config["priority_admission"][
                "high_total_two_sided_over_threshold"
            ],
            9.0,
        )
        self.assertTrue(
            config["priority_admission"][
                "fresh_review_requires_ml_starter_evidence_binding"
            ]
        )
        self.assertTrue(
            config["priority_admission"][
                "fresh_review_requires_high_total_two_sided_bindings"
            ]
        )
        self.assertTrue(
            config["f5_total_priority"][
                "require_two_sided_run_suppression"
            ]
        )
        self.assertTrue(
            config["f5_total_priority"][
                "block_under_on_long_horizon_starter_baseline_risk"
            ]
        )

    def test_priority_admission_uses_exact_approved_schema(self):
        config = load_model_config()

        self.assertEqual(
            set(config["priority_admission"]),
            {
                "allow_compatible_side_total_pair",
                "apply_family_policy_to_f5",
                "canonical_evidence_paths_only",
                "caps_only_no_positive_rescoring",
                "discontinuous_lineup_over_cap",
                "exclude_market_engine_pass_from_priority",
                "f5_market_engine_unchanged",
                "fresh_review_requires_high_total_two_sided_bindings",
                "fresh_review_requires_ml_starter_evidence_binding",
                "full_game_bullpen_requires_performance_support",
                "high_total_two_sided_over_cap",
                "high_total_two_sided_over_threshold",
                "incomplete_pitching_chain_cap",
                "late_bullpen_protection_cap",
                "matchup_only_side_requires_recent_run_conversion",
                "missing_evidence_blocks_multi_market_retention",
                "ml_requires_independent_full_game_margin_path",
                "ml_starter_evidence_cap",
                "multi_market_min_score",
                "one_candidate_per_full_game_family",
                "require_directional_coherence",
                "require_distinct_canonical_evidence_paths_for_multi_market_retention",
                "require_ml_bullpen_review_path",
                "require_two_sided_under_review_path",
                "same_game_candidate_assembler",
                "traffic_only_side_cap",
            },
        )

    def test_loader_rejects_priority_admission_schema_and_value_drift(self):
        base_config = load_model_config()
        mutations = []
        for field, value in base_config["priority_admission"].items():
            if isinstance(value, bool):
                changed = not value
            elif isinstance(value, (int, float)):
                changed = value + 1
            else:
                changed = f"{value}-changed"
            mutations.append((field, changed))

        for field, changed in mutations:
            with self.subTest(field=field, changed=changed):
                config = copy.deepcopy(base_config)
                config["priority_admission"][field] = changed
                with self.assertRaisesRegex(
                    ValueError,
                    "priority_admission configuration",
                ):
                    self._load_config_payload(config)

        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(base_config)
                if mutation == "missing":
                    config["priority_admission"].pop("ml_starter_evidence_cap")
                else:
                    config["priority_admission"]["unapproved_contract"] = True
                with self.assertRaisesRegex(
                    ValueError,
                    "priority_admission configuration",
                ):
                    self._load_config_payload(config)

    def test_forecasting_addition_preserves_every_legacy_config_value(self):
        config = load_model_config()
        legacy_config = {
            key: value for key, value in config.items() if key != "forecasting"
        }
        canonical = json.dumps(
            legacy_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

        self.assertEqual(hashlib.sha256(canonical).hexdigest(), LEGACY_CONFIG_SHA256)

    def test_loader_rejects_authority_changing_forecasting_metadata(self):
        base_config = load_model_config()

        for field, value in (
            ("mode", "production"),
            ("may_change_recommendations", True),
        ):
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(base_config)
                config["forecasting"] = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
                config["forecasting"][field] = value

                with self.assertRaises(ValueError):
                    self._load_config_payload(config)

    def test_loader_rejects_malformed_forecasting_block(self):
        base_config = load_model_config()
        missing_simulations = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
        missing_simulations.pop("simulations")
        invalid_simulations = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
        invalid_simulations["simulations"] = 0
        invalid_enabled = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
        invalid_enabled["enabled"] = "true"

        for forecasting in (
            None,
            missing_simulations,
            invalid_simulations,
            invalid_enabled,
        ):
            with self.subTest(forecasting=forecasting):
                config = copy.deepcopy(base_config)
                config["forecasting"] = forecasting

                with self.assertRaises(ValueError):
                    self._load_config_payload(config)

    def test_loader_rejects_missing_forecasting_block(self):
        config = load_model_config()
        config.pop("forecasting", None)

        with self.assertRaises(ValueError):
            self._load_config_payload(config)

    def test_loader_rejects_changed_legacy_version(self):
        config = load_model_config()
        config["version"] = "2026-07-14"
        config["forecasting"] = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)

        with self.assertRaises(ValueError):
            self._load_config_payload(config)

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
        self.assertEqual(config["priority_admission"]["multi_market_min_score"], 72)
        self.assertTrue(
            config["priority_admission"]["one_candidate_per_full_game_family"]
        )
        self.assertTrue(
            config["priority_admission"]["allow_compatible_side_total_pair"]
        )
        self.assertTrue(
            config["priority_admission"][
                "require_distinct_canonical_evidence_paths_for_multi_market_retention"
            ]
        )
        self.assertTrue(
            config["priority_admission"]["require_directional_coherence"]
        )
        self.assertTrue(config["priority_admission"]["apply_family_policy_to_f5"])
        self.assertTrue(config["priority_admission"]["f5_market_engine_unchanged"])
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
    def test_manifest_declares_legacy_as_decision_authority(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
        )

        self.assertEqual(manifest.get("decision_authority"), "legacy_fixed_model")
        forecasting = manifest.get("forecasting", {})
        self.assertEqual(forecasting.get("mode"), "shadow")
        self.assertFalse(forecasting.get("may_change_recommendations", True))

    def test_manifest_exposes_detached_forecasting_metadata(self):
        config = load_model_config()
        config["forecasting"] = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            config=config,
        )

        self.assertIn("forecasting", manifest)
        config["forecasting"]["mode"] = "production"
        config["forecasting"]["pa_weights"][0] = 99.0
        self.assertEqual(manifest["forecasting"], EXPECTED_FORECASTING_CONFIG)

    def test_shadow_metadata_does_not_change_legacy_manifest_contract(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
        )

        self.assertEqual(manifest["model_version"], "2026-08-05")
        self.assertTrue(manifest["fixed_model"])
        self.assertEqual(manifest["strength_adjustments"], [])
        self.assertNotIn("picks", manifest)
        self.assertNotIn("scores", manifest)
        self.assertNotIn("thresholds", manifest)
        self.assertEqual(
            set(manifest),
            {
                "model_version",
                "fixed_model",
                "decision_authority",
                "forecasting",
                "layers",
                "lineup_source",
                "include_weather",
                "cache_policy",
                "required_inputs",
                "confidence_flags",
                "strength_adjustments",
                "priority_candidate_assembler",
                "same_game_priority_policy",
                "required_outputs",
            },
        )

    def test_shadow_metadata_does_not_change_required_outputs(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
        )

        self.assertEqual(
            manifest["required_outputs"],
            ["F5", "F5 O/U", "ML", "+1.5", "O/U", "review"],
        )

    def test_manifest_rejects_authority_changing_inline_config(self):
        config = load_model_config()
        config["forecasting"] = copy.deepcopy(EXPECTED_FORECASTING_CONFIG)
        config["forecasting"]["mode"] = "production"

        with self.assertRaises(ValueError):
            build_analysis_manifest(
                lineup_confirmed=True,
                starter_confirmed=True,
                config=config,
            )

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

    def test_manifest_exposes_multi_market_same_game_policy(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            include_weather=False,
        )

        policy = manifest["same_game_priority_policy"]
        self.assertEqual(policy["multi_market_min_score"], 72)
        self.assertTrue(policy["one_candidate_per_full_game_family"])
        self.assertTrue(policy["allow_compatible_side_total_pair"])
        self.assertTrue(
            policy[
                "require_distinct_canonical_evidence_paths_for_multi_market_retention"
            ]
        )
        self.assertTrue(policy["require_directional_coherence"])
        self.assertTrue(policy["apply_family_policy_to_f5"])
        self.assertTrue(policy["f5_market_engine_unchanged"])

        self.assertEqual(policy["ml_starter_evidence_cap"], 71)
        self.assertEqual(policy["high_total_two_sided_over_cap"], 69)
        self.assertEqual(policy["high_total_two_sided_over_threshold"], 9.0)
        self.assertTrue(policy["fresh_review_requires_ml_starter_evidence_binding"])
        self.assertTrue(policy["fresh_review_requires_high_total_two_sided_bindings"])

    def test_weather_is_excluded_unless_requested(self):
        manifest = build_analysis_manifest(
            lineup_confirmed=True,
            starter_confirmed=True,
            include_weather=False,
        )

        self.assertFalse(manifest["include_weather"])
        self.assertNotIn("weather_and_park", manifest["required_inputs"])


class CurrentRuntimeDocumentationTests(unittest.TestCase):
    def test_current_runtime_docs_publish_the_current_legacy_version(self):
        project_root = Path(__file__).resolve().parents[1]
        expected_current_runtime_text = {
            "README.md": "고정 MLB model version은 `2026-08-05`",
            "docs/analysis-output-template.md": (
                "Use legacy model version `2026-08-05` for every fresh"
            ),
            "docs/shadow-operator.md": (
                "the current legacy model version is `2026-08-05`"
            ),
        }
        for relative_path, expected_text in expected_current_runtime_text.items():
            with self.subTest(path=relative_path):
                text = (project_root / relative_path).read_text(encoding="utf-8")
                self.assertIn(expected_text, text)

    def test_model_docs_state_inherited_review_contracts(self):
        project_root = Path(__file__).resolve().parents[1]
        model_text = (project_root / "docs/model.md").read_text(encoding="utf-8")
        template_text = (
            project_root / "docs/analysis-output-template.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "The `2026-08-05` contract inherits the prior ML bullpen review contract",
            model_text,
        )
        self.assertIn(
            "the prior two-sided under review contract",
            model_text,
        )
        self.assertIn(
            "model versions `2026-08-02`, `2026-08-03`, and `2026-08-05`",
            model_text,
        )
        self.assertIn(
            "model versions `2026-08-03` and `2026-08-05`",
            model_text,
        )
        self.assertIn(
            "model versions `2026-08-02`, `2026-08-03`, and `2026-08-05`",
            template_text,
        )
        self.assertIn(
            "model versions `2026-08-03` and `2026-08-05`",
            template_text,
        )

    def test_operator_docs_publish_current_priority_evidence_contracts(self):
        project_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "docs/analysis-output-template.md",
            "docs/shadow-operator.md",
        ):
            with self.subTest(path=relative_path):
                text = (project_root / relative_path).read_text(encoding="utf-8")
                self.assertIn("score `72` or higher", text)
                self.assertIn("capped at `71`", text)
                self.assertIn("verified full-game over line of `9.0` or higher", text)
                self.assertIn("capped at `69`", text)
                self.assertIn(
                    "suppression/protection paths do not count",
                    text.lower(),
                )


if __name__ == "__main__":
    unittest.main()
