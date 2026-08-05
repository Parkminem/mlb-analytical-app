from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import base64
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import analysis.shadow as shadow_module
import pytest
from analysis.ledger.forecast_codec import (
    compact_forecast_outputs,
    expand_component_distributions,
)
from analysis.forecasting import (
    MarketType,
    OutcomeProbabilities,
    SettlementFormat,
    price_market,
    run_forecast,
)
from analysis.forecasting.identity import (
    canonical_json,
    capture_payload,
    normalized_input_hash,
)
from analysis.ledger import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    LedgerHeadChangedError,
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
    build_shadow_capture_command,
    compare_shadow_forecast,
    record_shadow_run,
    require_recommended_review_eligibility,
    validate_recommended_review_contract,
)
from tests.test_forecasting_engine import (
    _complete_game,
    _fast_config,
    _market,
    forecast_engine_module,
)
from tests.test_forecasting_inputs import complete_forecast_game


deserialize_shadow_comparison = getattr(
    shadow_module,
    "deserialize_shadow_comparison",
    None,
)


class ForecastDistributionCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        forecast = run_forecast(_complete_game(), _fast_config())
        cls.outputs = json.loads(canonical_json(forecast))

    def test_compaction_round_trips_real_forecast_distributions(self) -> None:
        compact = compact_forecast_outputs(self.outputs)

        expanded = expand_component_distributions(
            compact["component_distributions"]
        )

        self.assertEqual(
            canonical_json(expanded),
            canonical_json(self.outputs["component_distributions"]),
        )
        self.assertNotIn(
            "values",
            next(iter(compact["component_distributions"].values())),
        )

    def test_compaction_preserves_integer_and_float_value_types(self) -> None:
        outputs = {
            "component_distributions": {
                "fixture": {
                    "component_id": "fixture",
                    "values": [0, 1.25, 2],
                    "summary": {"median": 1.25},
                }
            }
        }

        expanded = expand_component_distributions(
            compact_forecast_outputs(outputs)["component_distributions"]
        )

        values = expanded["fixture"]["values"]
        self.assertEqual(values, [0, 1.25, 2])
        self.assertEqual(tuple(type(value) for value in values), (int, float, int))

    def test_expansion_rejects_malformed_encoded_values(self) -> None:
        compact = compact_forecast_outputs(self.outputs)[
            "component_distributions"
        ]
        component_id = next(iter(compact))

        cases = (
            ("base64", "%%%"),
            (
                "zlib",
                base64.b64encode(b"not a zlib payload").decode("ascii"),
            ),
            (
                "JSON",
                base64.b64encode(
                    zlib.compress(b"{")
                ).decode("ascii"),
            ),
        )
        for expected, values_data in cases:
            with self.subTest(expected=expected):
                malformed = json.loads(canonical_json(compact))
                malformed[component_id]["values_data"] = values_data

                with self.assertRaisesRegex(ValueError, expected):
                    expand_component_distributions(malformed)

    def test_expansion_rejects_count_mismatch_unknown_fields_and_non_list_values(
        self,
    ) -> None:
        compact = compact_forecast_outputs(self.outputs)[
            "component_distributions"
        ]
        component_id = next(iter(compact))

        mismatched_count = json.loads(canonical_json(compact))
        mismatched_count[component_id]["values_count"] += 1
        unknown_field = json.loads(canonical_json(compact))
        unknown_field[component_id]["unexpected"] = True
        historical_unknown_field = json.loads(
            canonical_json(self.outputs["component_distributions"])
        )
        historical_unknown_field[component_id]["unexpected"] = True
        non_list_values = json.loads(canonical_json(compact))
        non_list_values[component_id]["values_data"] = base64.b64encode(
            zlib.compress(b'{"not":"a list"}')
        ).decode("ascii")

        for expected, malformed in (
            ("values_count", mismatched_count),
            ("unknown", unknown_field),
            ("unknown", historical_unknown_field),
            ("list", non_list_values),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    expand_component_distributions(malformed)

    def test_expansion_returns_historical_values_payload_unchanged(self) -> None:
        historical = self.outputs["component_distributions"]

        expanded = expand_component_distributions(historical)

        self.assertIs(expanded, historical)


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


def bullpen_protection_path() -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.BULLPEN_PROTECTION,
        component_ids=(
            "home_bullpen_runs_allowed",
            "home_protected_bullpen_close_entry",
        ),
        contradiction_rule=ContradictionRule.BULLPEN_PROTECTION,
    )


def late_bullpen_exposure_path() -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
        component_ids=(
            "home_bullpen_segment_missing_exposed",
            "home_pitcher_chain_workload_outs",
        ),
        contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
    )


def lineup_suppression_path(
    *,
    side: str = "home",
    horizon: str = "full",
) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.LINEUP_SUPPRESSION,
        component_ids=(
            f"{side}_{horizon}_runs",
            f"{side}_{horizon}_traffic",
            f"{side}_{horizon}_extra_base_hits",
        ),
        contradiction_rule=ContradictionRule.LINEUP_SUPPRESSION,
    )


def candidate(
    *,
    market: str = "ml",
    score: int = 78,
    rating: str = "play",
    evidence_paths: frozenset[PriorityEvidencePath] | None = None,
) -> PriorityMarketCandidate:
    return PriorityMarketCandidate(
        market=market,
        score=score,
        decision_rating=rating,
        evidence_paths=(
            frozenset(path.evidence_path for path in principal_paths())
            if evidence_paths is None
            else evidence_paths
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
    model_version: str = "legacy-fixed-v1",
    candidates: tuple[PriorityMarketCandidate, ...] | None = None,
    final_priority: tuple[str, ...] | None = None,
    review_bindings: tuple[CandidateReviewBinding, ...] | None = None,
) -> LegacyDecisionSnapshot:
    resolved_candidates = (candidate(),) if candidates is None else candidates
    resolved_bindings = (binding(),) if review_bindings is None else review_bindings
    resolved_final_priority = (
        tuple(
            item.market
            for item in resolved_candidates
            if isinstance(item, PriorityMarketCandidate)
        )
        if final_priority is None
        else final_priority
    )
    return LegacyDecisionSnapshot(
        model_version=model_version,
        candidates=resolved_candidates,
        final_priority=resolved_final_priority,
        review_bindings=resolved_bindings,
        notes=("legacy remains authoritative",),
    )


def test_current_ml_bullpen_contract_rejects_candidate_without_bullpen_path(
) -> None:
    legacy = legacy_snapshot(model_version="2026-08-02")

    with pytest.raises(
        ValueError,
        match=(
            r"candidate\[0\] ml \(play\): "
            "missing_ml_bullpen_evidence_path"
        ),
    ):
        validate_recommended_review_contract(
            legacy,
            _complete_game().markets,
        )


def test_current_ml_bullpen_contract_rejects_unbound_bullpen_path() -> None:
    bullpen_path = bullpen_protection_path()
    legacy = legacy_snapshot(
        model_version="2026-08-02",
        candidates=(
            candidate(
                evidence_paths=frozenset(
                    path.evidence_path
                    for path in principal_paths() + (bullpen_path,)
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"candidate\[0\] ml \(play\): "
            "missing_ml_bullpen_review_binding"
        ),
    ):
        validate_recommended_review_contract(
            legacy,
            _complete_game().markets,
        )


@pytest.mark.parametrize(
    "bullpen_path",
    (bullpen_protection_path(), late_bullpen_exposure_path()),
    ids=("bullpen_protection", "late_bullpen_exposure"),
)
def test_current_ml_bullpen_contract_accepts_canonical_bullpen_path(
    bullpen_path: PrincipalPathBinding,
) -> None:
    paths = principal_paths() + (bullpen_path,)
    legacy = legacy_snapshot(
        model_version="2026-08-02",
        candidates=(
            candidate(
                evidence_paths=frozenset(
                    path.evidence_path for path in paths
                ),
            ),
        ),
        review_bindings=(binding(paths=paths),),
    )

    validate_recommended_review_contract(
        legacy,
        _complete_game().markets,
    )


@pytest.mark.parametrize(
    ("market_type", "horizon", "line"),
    (
        (MarketType.F5_TOTAL, "f5", 4.5),
        (MarketType.FULL_TOTAL, "full", 8.5),
    ),
)
def test_current_under_contract_accepts_two_sided_suppression_binding(
    market_type: MarketType,
    horizon: str,
    line: float,
) -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key=f"under:{market_type.value}",
        market_type=market_type,
        selected_side="under",
        line=line,
    )
    starter_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )
    lineup_path = lineup_suppression_path(side="home", horizon=horizon)
    paths = (starter_path, lineup_path)
    legacy = legacy_snapshot(
        model_version="2026-08-03",
        candidates=(
            candidate(
                market=market_type.value,
                evidence_paths=frozenset(
                    path.evidence_path for path in paths
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=market_type.value,
                paths=paths,
            ),
        ),
    )

    validate_recommended_review_contract(legacy, (market,))


def test_current_under_contract_rejects_missing_lineup_suppression_path() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="under:full",
        selected_side="under",
        line=8.5,
    )
    legacy = legacy_snapshot(
        model_version="2026-08-03",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(
                    path.evidence_path for path in principal_paths()
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=principal_paths(),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing_under_two_sided_evidence_paths",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_current_under_contract_rejects_unbound_lineup_suppression_path() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="under:full",
        selected_side="under",
        line=8.5,
    )
    legacy = legacy_snapshot(
        model_version="2026-08-03",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(
                    (
                        PriorityEvidencePath.STARTER_SUPPRESSION,
                        PriorityEvidencePath.TOP_ORDER_MATCHUP,
                        PriorityEvidencePath.LINEUP_SUPPRESSION,
                    )
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=principal_paths(),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing_under_two_sided_review_binding",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_historical_under_contract_remains_replay_compatible() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="under:full",
        selected_side="under",
        line=8.5,
    )
    legacy = legacy_snapshot(
        model_version="2026-08-02",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(
                    path.evidence_path for path in principal_paths()
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=principal_paths(),
            ),
        ),
    )

    validate_recommended_review_contract(legacy, (market,))


def test_current_under_contract_rejects_duplicated_scoring_direction() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="under:full",
        selected_side="under",
        line=8.5,
    )
    starter_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )
    lineup_path = lineup_suppression_path(side="away")
    paths = (starter_path, lineup_path)
    legacy = legacy_snapshot(
        model_version="2026-08-03",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(
                    path.evidence_path for path in paths
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="incomplete_under_two_sided_coverage",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_lineup_suppression_contract_rejects_over_direction() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="over:full",
        selected_side="over",
        line=8.5,
    )
    starter_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )
    lineup_path = lineup_suppression_path(side="home")
    paths = (starter_path, lineup_path)
    legacy = legacy_snapshot(
        model_version="2026-08-03",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(
                    path.evidence_path for path in paths
                ),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="lineup_suppression_requires_under",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_recommended_contract_rejects_missing_lean_binding() -> None:
    legacy = legacy_snapshot(
        candidates=(candidate(rating="lean"),),
        review_bindings=(),
    )

    with pytest.raises(
        ValueError,
        match=r"candidate\[0\] ml \(lean\): missing_pregame_binding",
    ):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_contract_allows_unbound_pass() -> None:
    legacy = legacy_snapshot(
        candidates=(candidate(rating="pass"),),
        final_priority=(),
        review_bindings=(),
    )

    validate_recommended_review_contract(
        legacy,
        complete_forecast_game().markets,
    )


def _conversion_path(
    evidence_path: PriorityEvidencePath,
    side: str,
    *,
    horizon: str = "full",
) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=evidence_path,
        component_ids=(
            f"{side}_{horizon}_runs",
            f"{side}_{horizon}_traffic",
            f"{side}_{horizon}_extra_base_hits",
        ),
        contradiction_rule=(
            ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
        ),
    )


def _starter_collapse_path(side: str) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
        component_ids=(
            f"{side}_pitcher_chain_runs_allowed",
            f"{side}_pitcher_chain_workload_outs",
        ),
        contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
    )


def _ml_market():
    return replace(
        complete_forecast_game().markets[0],
        market_key="away_ml",
        market_type=MarketType.ML,
        selected_side="away",
        line=None,
    )


def test_2026_08_05_rejects_playable_ml_without_directional_starter_evidence(
) -> None:
    paths = (
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        bullpen_protection_path(),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        review_bindings=(binding(market_key="away_ml", paths=paths),),
    )

    with pytest.raises(ValueError, match="missing_ml_starter_evidence"):
        validate_recommended_review_contract(legacy, (_ml_market(),))


@pytest.mark.parametrize(
    "starter_path",
    (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("home_pitcher_chain_runs_allowed",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        ),
        _starter_collapse_path("away"),
    ),
    ids=("selected_side_opponent_suppression", "selected_side_collapse"),
)
def test_2026_08_05_rejects_ml_starter_evidence_bound_to_wrong_pitcher(
    starter_path: PrincipalPathBinding,
) -> None:
    paths = (
        starter_path,
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        bullpen_protection_path(),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        review_bindings=(binding(market_key="away_ml", paths=paths),),
    )

    with pytest.raises(ValueError, match="missing_ml_starter_evidence"):
        validate_recommended_review_contract(legacy, (_ml_market(),))


def test_2026_08_05_accepts_ml_starter_suppression_for_selected_pitcher(
) -> None:
    paths = (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("away_pitcher_chain_runs_allowed",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        ),
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        bullpen_protection_path(),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        review_bindings=(binding(market_key="away_ml", paths=paths),),
    )

    validate_recommended_review_contract(legacy, (_ml_market(),))


def test_2026_08_05_inherits_ml_bullpen_evidence_contract() -> None:
    paths = (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("away_pitcher_chain_runs_allowed",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        ),
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        review_bindings=(binding(market_key="away_ml", paths=paths),),
    )

    with pytest.raises(ValueError, match="missing_ml_bullpen_evidence_path"):
        validate_recommended_review_contract(legacy, (_ml_market(),))


def test_2026_08_05_inherits_two_sided_under_evidence_contract() -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="under:full",
        selected_side="under",
        line=8.5,
    )
    paths = principal_paths()
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        final_priority=(MarketType.FULL_TOTAL.value,),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing_under_two_sided_evidence_paths",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_2026_08_05_rejects_high_total_two_sided_over_evidence(
) -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="over:full",
        selected_side="over",
        line=9.0,
    )
    paths = (
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        _starter_collapse_path("home"),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        final_priority=(MarketType.FULL_TOTAL.value,),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing_high_total_two_sided_over_evidence",
    ):
        validate_recommended_review_contract(legacy, (market,))


@pytest.mark.parametrize(
    "opposing_pitcher_path",
    (
        _starter_collapse_path("away"),
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
            component_ids=(
                "away_bullpen_segment_missing_exposed",
                "away_pitcher_chain_workload_outs",
            ),
            contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
        ),
    ),
    ids=("starter_collapse", "late_bullpen_exposure"),
)
def test_2026_08_05_accepts_high_total_two_sided_conversion_and_opposing_pitcher_route(
    opposing_pitcher_path: PrincipalPathBinding,
) -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="over:full",
        selected_side="over",
        line=9.0,
    )
    paths = (
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        opposing_pitcher_path,
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        final_priority=(MarketType.FULL_TOTAL.value,),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    validate_recommended_review_contract(legacy, (market,))


@pytest.mark.parametrize(
    ("paths", "error"),
    (
        (
            (
                _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
                _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "home"),
            ),
            "missing_high_total_two_sided_over_evidence",
        ),
        (
            (
                _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
                _conversion_path(PriorityEvidencePath.RECENT_RUN_CONVERSION, "away"),
            ),
            "missing_high_total_two_sided_over_evidence",
        ),
        (
            (
                _conversion_path(
                    PriorityEvidencePath.TOP_ORDER_MATCHUP,
                    "away",
                    horizon="f5",
                ),
                _conversion_path(PriorityEvidencePath.RECENT_RUN_CONVERSION, "home"),
            ),
            "invalid binding shape",
        ),
    ),
    ids=("duplicate_canonical_path", "same_scoring_side", "wrong_horizon"),
)
def test_2026_08_05_high_total_two_sided_contract_rejects_invalid_routes(
    paths: tuple[PrincipalPathBinding, ...],
    error: str,
) -> None:
    market = replace(
        complete_forecast_game().markets[0],
        market_key="over:full",
        selected_side="over",
        line=9.0,
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                market=MarketType.FULL_TOTAL.value,
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        final_priority=(MarketType.FULL_TOTAL.value,),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market=MarketType.FULL_TOTAL.value,
                paths=paths,
            ),
        ),
    )

    with pytest.raises(ValueError, match=error):
        validate_recommended_review_contract(legacy, (market,))


@pytest.mark.parametrize(
    ("score", "rating"),
    ((71, "play"), (78, "pass")),
)
def test_2026_08_05_contract_leaves_pass_and_subthreshold_ml_outside_gate(
    score: int,
    rating: str,
) -> None:
    paths = (
        _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        bullpen_protection_path(),
    )
    legacy = legacy_snapshot(
        model_version="2026-08-05",
        candidates=(
            candidate(
                score=score,
                rating=rating,
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        final_priority=() if rating == "pass" else (),
        review_bindings=(
            (binding(market_key="away_ml", paths=paths),)
            if rating != "pass"
            else ()
        ),
    )

    validate_recommended_review_contract(legacy, (_ml_market(),))


def _additional_valid_full_game_paths() -> tuple[PrincipalPathBinding, ...]:
    return (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.RECENT_RUN_CONVERSION,
            component_ids=(
                "away_full_runs",
                "away_full_traffic",
                "away_full_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        ),
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.UNDERDOG_RUN_CREATION,
            component_ids=(
                "away_full_runs",
                "away_full_traffic",
                "away_full_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        ),
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
            component_ids=(
                "home_bullpen_segment_missing_exposed",
                "home_pitcher_chain_workload_outs",
            ),
            contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
        ),
    )


@pytest.mark.parametrize(
    "paths",
    (
        (principal_paths()[0],),
        principal_paths() + _additional_valid_full_game_paths(),
        (principal_paths()[0], principal_paths()[0]),
    ),
    ids=("one_path", "more_than_four_paths", "duplicate_evidence_path"),
)
def test_recommended_contract_rejects_invalid_static_path_cardinality_or_identity(
    paths: tuple[PrincipalPathBinding, ...],
) -> None:
    market = complete_forecast_game().markets[0]
    legacy = legacy_snapshot(
        candidates=(
            candidate(
                market="full_total",
                evidence_paths=frozenset(path.evidence_path for path in paths),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market="full_total",
                paths=paths,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"candidate\[0\] full_total \(play\): invalid_path_count",
    ):
        validate_recommended_review_contract(legacy, (market,))


def test_recommended_contract_rejects_duplicate_final_priority_market() -> None:
    market = complete_forecast_game().markets[0]
    legacy = legacy_snapshot(
        candidates=(candidate(market="full_total"),),
        final_priority=("full_total", "full_total"),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market="full_total",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"final_priority\[1\] full_total: "
            "duplicate_final_priority_market"
        ),
    ):
        validate_recommended_review_contract(legacy, (market,))


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        (
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
                component_ids=("home_pitcher_chain_runs_allowed",),
                contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
            ),
            "invalid binding shape for starter_collapse",
        ),
        (
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
                component_ids=(
                    "away_pitcher_chain_runs_allowed",
                    "home_pitcher_chain_runs_allowed",
                ),
                contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
            ),
            "invalid binding shape for starter_suppression",
        ),
    ),
)
def test_recommended_contract_rejects_invalid_principal_path_shape(
    path: PrincipalPathBinding,
    expected: str,
) -> None:
    market = complete_forecast_game().markets[0]
    legacy = legacy_snapshot(
        candidates=(
            candidate(
                market="full_total",
                evidence_paths=frozenset((path.evidence_path,)),
            ),
        ),
        review_bindings=(
            binding(
                market_key=market.market_key,
                legacy_market="full_total",
                paths=(path,),
            ),
        ),
    )

    with pytest.raises(ValueError, match=expected):
        validate_recommended_review_contract(legacy, (market,))


def test_recommended_contract_rejects_unknown_market_key() -> None:
    legacy = legacy_snapshot(
        review_bindings=(binding(market_key="unknown_market"),),
    )

    with pytest.raises(ValueError, match=r"candidate\[0\] ml \(play\): unknown_market_key unknown_market"):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_contract_rejects_candidate_market_mismatch() -> None:
    legacy = legacy_snapshot(
        review_bindings=(binding(legacy_market="full_total"),),
    )

    with pytest.raises(
        ValueError,
        match=r"candidate\[0\] ml \(play\): candidate_market_mismatch",
    ):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_contract_rejects_final_priority_without_candidate() -> None:
    legacy = legacy_snapshot(final_priority=("full_total",))

    with pytest.raises(
        ValueError,
        match="final_priority full_total: requires one play or lean candidate",
    ):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_contract_rejects_pass_in_final_priority() -> None:
    legacy = legacy_snapshot(
        candidates=(candidate(rating="pass"),),
        final_priority=("ml",),
        review_bindings=(),
    )

    with pytest.raises(
        ValueError,
        match="final_priority ml: requires one play or lean candidate",
    ):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_review_eligibility_allows_ineligible_pass() -> None:
    forecast = run_forecast(_complete_game(), _fast_config())
    legacy = legacy_snapshot(
        candidates=(
            candidate(),
            candidate(market="full_total", rating="pass"),
        ),
        review_bindings=(binding(),),
    )

    comparison = compare_shadow_forecast(legacy, forecast)

    assert require_recommended_review_eligibility(comparison) is None


def test_recommended_review_eligibility_rejects_lean_invalid_path_count() -> None:
    invalid_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
    )
    legacy = legacy_snapshot(
        candidates=(
            candidate(
                rating="lean",
                evidence_paths=frozenset((invalid_path.evidence_path,)),
            ),
        ),
        review_bindings=(binding(paths=(invalid_path,)),),
    )

    comparison = compare_shadow_forecast(
        legacy,
        run_forecast(_complete_game(), _fast_config()),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"candidate\[0\] ml \(lean\) sea_ml: invalid_path_count"
        ),
    ):
        require_recommended_review_eligibility(comparison)


def test_recommended_review_eligibility_rejects_play_missing_distribution() -> None:
    missing_distribution_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
        component_ids=(
            "home_bullpen_segment_missing_exposed",
            "home_pitcher_chain_workload_outs",
        ),
        contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
    )
    paths = (principal_paths()[0], missing_distribution_path)
    legacy = legacy_snapshot(
        candidates=(
            candidate(
                evidence_paths=frozenset(
                    path.evidence_path for path in paths
                ),
            ),
        ),
        review_bindings=(binding(paths=paths),),
    )

    comparison = compare_shadow_forecast(
        legacy,
        run_forecast(_complete_game(), _fast_config()),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"candidate\[0\] ml \(play\) sea_ml: "
            r"missing_component_distribution"
        ),
    ):
        require_recommended_review_eligibility(comparison)


def three_way_f5_scenario():
    market_key = "sea_f5_three_way"
    market_input = replace(
        _market(market_key, MarketType.F5_SIDE, "away", 0.0),
        settlement_format=SettlementFormat.THREE_WAY,
    )
    game = _complete_game(markets=(market_input,))
    forecast = run_forecast(game, _fast_config())
    f5_paths = (
        principal_paths()[0],
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
            component_ids=(
                "away_f5_runs",
                "away_f5_traffic",
                "away_f5_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        ),
    )
    legacy = legacy_snapshot(
        candidates=(candidate(market="f5_side"),),
        review_bindings=(
            binding(
                market_key=market_key,
                legacy_market="f5_side",
                paths=f5_paths,
            ),
        ),
    )
    return game, forecast, legacy


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


class LegacyCaptureSignatureStore(TrackingStore):
    def append_capture(self, idempotency_key, command):
        self.calls.append("capture")
        return self.store.append_capture(idempotency_key, command)


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


class AtomicOnlyStore:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store
        self.calls: list[tuple[object, ...]] = []

    def append_shadow_run_atomic(self, *args):
        self.calls.append(args)
        return self.store.append_shadow_run_atomic(*args)


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

    def test_snapshot_rejects_binding_paths_absent_from_candidate_evidence(
        self,
    ) -> None:
        incomplete_evidence = frozenset(
            (PriorityEvidencePath.STARTER_SUPPRESSION,)
        )

        with self.assertRaisesRegex(ValueError, "candidate evidence_paths"):
            legacy_snapshot(
                candidates=(
                    candidate(evidence_paths=incomplete_evidence),
                ),
            )

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

    def test_snapshot_rejects_non_boolean_pair_compatibility(self) -> None:
        invalid = replace(candidate(), same_game_pair_compatible=1)

        with self.assertRaisesRegex(
            ValueError,
            "same_game_pair_compatible must be bool",
        ):
            legacy_snapshot(candidates=(invalid,))

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

    def test_legacy_candidate_pair_compatibility_round_trips(self) -> None:
        original = candidate()
        original = replace(original, same_game_pair_compatible=True)
        legacy = legacy_snapshot(candidates=(original,))

        payload = shadow_module.canonical_legacy_decision_payload(legacy)

        self.assertTrue(payload["candidates"][0]["same_game_pair_compatible"])
        self.assertEqual(shadow_module._legacy_from_payload(payload), legacy)

    def test_historical_shadow_payload_defaults_missing_pair_compatibility(
        self,
    ) -> None:
        comparison = compare_shadow_forecast(legacy_snapshot(), self.forecast)
        payload = shadow_module._shadow_comparison_payload(comparison)
        del payload["legacy"]["candidates"][0][
            "same_game_pair_compatible"
        ]

        restored = shadow_module.deserialize_shadow_comparison(payload)

        self.assertFalse(
            restored.legacy.candidates[0].same_game_pair_compatible
        )

    def test_forecast_probability_does_not_change_legacy_priority(self) -> None:
        legacy_candidate = candidate()
        legacy = legacy_snapshot(candidates=(legacy_candidate,))
        markets = dict(self.forecast.markets)
        markets["sea_ml"] = replace(
            price_market(
                OutcomeProbabilities(win=0.32, push=0.0, loss=0.68),
                1.91,
                1.91,
            ),
            semantic_fingerprint=(
                self.forecast.markets["sea_ml"].semantic_fingerprint
            ),
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

    def test_known_unevaluated_market_remains_review_eligible(self) -> None:
        _, forecast, legacy = three_way_f5_scenario()
        evaluation = forecast.markets["sea_f5_three_way"]
        self.assertFalse(evaluation.evaluated)

        market = compare_shadow_forecast(
            legacy,
            forecast,
        ).market_comparisons[0]

        self.assertTrue(market.review_eligible)
        self.assertEqual(market.review_ineligibility_flags, ())
        self.assertEqual(len(market.principal_paths), 2)
        self.assertIsNone(market.model_win_probability)
        self.assertIsNone(market.fair_decimal_price)
        self.assertIsNone(market.market_no_vig_probability)
        self.assertIsNone(market.edge)

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

    def test_invalid_binding_emits_context_flags_in_fixed_order(self) -> None:
        invalid_path = PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("missing_component",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        )
        legacy = legacy_snapshot(
            review_bindings=(
                binding(
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
            candidates=(
                candidate(
                    evidence_paths=frozenset(
                        path.evidence_path
                        for path in (
                            missing_component_path,
                            principal_paths()[0],
                        )
                    )
                ),
            ),
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

    def test_comparison_cross_validates_binding_flags_and_identity(self) -> None:
        valid = compare_shadow_forecast(
            legacy_snapshot(),
            self.forecast,
        )
        comparison = valid.market_comparisons[0]
        forged_missing = replace(
            comparison,
            principal_paths=(),
            review_eligible=False,
            review_ineligibility_flags=("missing_pregame_binding",),
        )

        with self.assertRaisesRegex(ValueError, "binding.*missing_pregame"):
            ShadowComparison(
                authority="legacy",
                legacy=valid.legacy,
                forecast_run_id=valid.forecast_run_id,
                market_comparisons=(forged_missing,),
            )

        forged_candidate_mismatch = replace(
            comparison,
            principal_paths=(),
            review_eligible=False,
            review_ineligibility_flags=("candidate_market_mismatch",),
        )
        with self.assertRaisesRegex(ValueError, "candidate_market_mismatch"):
            ShadowComparison(
                authority="legacy",
                legacy=valid.legacy,
                forecast_run_id=valid.forecast_run_id,
                market_comparisons=(forged_candidate_mismatch,),
            )

        ineligible = replace(
            comparison,
            principal_paths=(),
            review_eligible=False,
            review_ineligibility_flags=("unknown_market_key",),
        )
        for mismatched_binding in (
            replace(binding(), market_key="other_market"),
            replace(binding(), legacy_market="full_total"),
        ):
            with self.subTest(binding=mismatched_binding):
                with self.assertRaisesRegex(ValueError, "review binding"):
                    ShadowComparison(
                        authority="legacy",
                        legacy=legacy_snapshot(
                            review_bindings=(mismatched_binding,),
                        ),
                        forecast_run_id=valid.forecast_run_id,
                        market_comparisons=(ineligible,),
                    )

        without_binding = legacy_snapshot(review_bindings=())
        with self.assertRaisesRegex(ValueError, "missing_pregame_binding"):
            ShadowComparison(
                authority="legacy",
                legacy=without_binding,
                forecast_run_id=valid.forecast_run_id,
                market_comparisons=(ineligible,),
            )

        forged_unbound_mismatch = replace(
            comparison,
            principal_paths=(),
            review_eligible=False,
            review_ineligibility_flags=(
                "missing_pregame_binding",
                "candidate_market_mismatch",
            ),
        )
        with self.assertRaisesRegex(ValueError, "candidate_market_mismatch"):
            ShadowComparison(
                authority="legacy",
                legacy=without_binding,
                forecast_run_id=valid.forecast_run_id,
                market_comparisons=(forged_unbound_mismatch,),
            )

    def test_candidate_market_mismatch_is_exact_bound_ineligible_state(
        self,
    ) -> None:
        mismatched_binding = replace(
            binding(),
            legacy_market="full_total",
        )
        legacy = legacy_snapshot(review_bindings=(mismatched_binding,))

        comparison = compare_shadow_forecast(legacy, self.forecast)
        market = comparison.market_comparisons[0]

        self.assertFalse(market.review_eligible)
        self.assertEqual(
            market.review_ineligibility_flags,
            ("candidate_market_mismatch",),
        )
        self.assertEqual(market.market_key, mismatched_binding.market_key)
        self.assertEqual(market.legacy_market, legacy.candidates[0].market)
        self.assertEqual(market.principal_paths, ())

        for forged_flags in (
            ("unknown_market_key",),
            ("missing_pregame_binding", "candidate_market_mismatch"),
        ):
            with self.subTest(forged_flags=forged_flags):
                forged = replace(
                    market,
                    review_ineligibility_flags=forged_flags,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "candidate_market_mismatch|missing_pregame_binding",
                ):
                    ShadowComparison(
                        authority="legacy",
                        legacy=legacy,
                        forecast_run_id=comparison.forecast_run_id,
                        market_comparisons=(forged,),
                    )

    def test_comparison_rejects_eligible_entry_without_binding(self) -> None:
        valid = compare_shadow_forecast(
            legacy_snapshot(),
            self.forecast,
        )

        with self.assertRaisesRegex(ValueError, "review binding"):
            ShadowComparison(
                authority="legacy",
                legacy=legacy_snapshot(review_bindings=()),
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
        supersedes_capture_id=None,
    ):
        return record_shadow_run(
            self.store if store is None else store,
            self.game if game is None else game,
            self.legacy if legacy is None else legacy,
            self.forecast if forecast is None else forecast,
            self.keys() if keys is None else keys,
            supersedes_capture_id=supersedes_capture_id,
        )

    def test_capture_command_builder_matches_recorded_capture(self) -> None:
        command = build_shadow_capture_command(self.game)

        result = self.record()

        self.assertEqual(result.capture.game_pk, command.game_pk)
        self.assertEqual(result.capture.captured_at, command.captured_at)
        self.assertEqual(result.capture.first_pitch, command.first_pitch)
        self.assertEqual(result.capture.lineup_state, command.lineup_state)
        self.assertEqual(
            result.capture.out_of_sample_eligible,
            command.out_of_sample_eligible,
        )
        self.assertEqual(
            result.capture.normalized_input_hash,
            command.normalized_input_hash,
        )
        self.assertEqual(result.capture.payload, command.payload)
        self.assertIsNone(command.supersedes_capture_id)

    def test_recording_enforces_current_under_review_contract(self) -> None:
        market = replace(
            self.game.markets[0],
            market_key="record:under:full",
            market_type=MarketType.FULL_TOTAL,
            selected_side="under",
            line=8.5,
        )
        game = replace(self.game, markets=(market,))
        forecast = run_forecast(game, _fast_config())
        legacy = legacy_snapshot(
            model_version="2026-08-03",
            candidates=(
                candidate(
                    market=MarketType.FULL_TOTAL.value,
                    evidence_paths=frozenset(
                        path.evidence_path for path in principal_paths()
                    ),
                ),
            ),
            review_bindings=(
                binding(
                    market_key=market.market_key,
                    legacy_market=MarketType.FULL_TOTAL.value,
                    paths=principal_paths(),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing_under_two_sided_evidence_paths",
        ):
            self.record(
                game=game,
                legacy=legacy,
                forecast=forecast,
                keys=self.keys("current-under-contract"),
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recording_enforces_2026_08_05_priority_evidence_contract(self) -> None:
        market = next(
            market
            for market in self.game.markets
            if market.market_type is MarketType.ML
        )
        paths = (
            _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
            bullpen_protection_path(),
        )
        legacy = legacy_snapshot(
            model_version="2026-08-05",
            candidates=(
                candidate(
                    evidence_paths=frozenset(
                        path.evidence_path for path in paths
                    ),
                ),
            ),
            review_bindings=(
                binding(market_key=market.market_key, paths=paths),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing_ml_starter_evidence",
        ):
            self.record(
                legacy=legacy,
                keys=self.keys("priority-evidence-contract"),
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recording_enforces_2026_08_05_inherited_evidence_contracts(self) -> None:
        ml_market = next(
            market
            for market in self.game.markets
            if market.market_type is MarketType.ML
        )
        ml_paths = (
            PrincipalPathBinding(
                evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
                component_ids=("away_pitcher_chain_runs_allowed",),
                contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
            ),
            _conversion_path(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        )
        missing_bullpen = legacy_snapshot(
            model_version="2026-08-05",
            candidates=(
                candidate(
                    evidence_paths=frozenset(
                        path.evidence_path for path in ml_paths
                    ),
                ),
            ),
            review_bindings=(
                binding(market_key=ml_market.market_key, paths=ml_paths),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing_ml_bullpen_evidence_path",
        ):
            self.record(
                legacy=missing_bullpen,
                keys=self.keys("missing-bullpen"),
            )

        under_market = replace(
            self.game.markets[0],
            market_key="record:under:full:2026-08-05",
            market_type=MarketType.FULL_TOTAL,
            selected_side="under",
            line=8.5,
        )
        under_game = replace(self.game, markets=(under_market,))
        missing_two_sided_under = legacy_snapshot(
            model_version="2026-08-05",
            candidates=(
                candidate(
                    market=MarketType.FULL_TOTAL.value,
                    evidence_paths=frozenset(
                        path.evidence_path for path in principal_paths()
                    ),
                ),
            ),
            final_priority=(MarketType.FULL_TOTAL.value,),
            review_bindings=(
                binding(
                    market_key=under_market.market_key,
                    legacy_market=MarketType.FULL_TOTAL.value,
                    paths=principal_paths(),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing_under_two_sided_evidence_paths",
        ):
            self.record(
                game=under_game,
                legacy=missing_two_sided_under,
                forecast=run_forecast(under_game, _fast_config()),
                keys=self.keys("missing-two-sided-under"),
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recording_preserves_historical_under_contract(self) -> None:
        market = replace(
            self.game.markets[0],
            market_key="record:historical:under:full",
            market_type=MarketType.FULL_TOTAL,
            selected_side="under",
            line=8.5,
        )
        game = replace(self.game, markets=(market,))
        forecast = run_forecast(game, _fast_config())
        legacy = legacy_snapshot(
            model_version="2026-08-02",
            candidates=(
                candidate(
                    market=MarketType.FULL_TOTAL.value,
                    evidence_paths=frozenset(
                        path.evidence_path for path in principal_paths()
                    ),
                ),
            ),
            review_bindings=(
                binding(
                    market_key=market.market_key,
                    legacy_market=MarketType.FULL_TOTAL.value,
                    paths=principal_paths(),
                ),
            ),
        )

        result = self.record(
            game=game,
            legacy=legacy,
            forecast=forecast,
            keys=self.keys("historical-under-contract"),
        )

        self.assertEqual(result.capture.game_pk, game.identity.game_pk)
        self.assertEqual(result.forecast.forecast_run_id, forecast.forecast_run_id)

    def test_recording_supports_explicit_supersession_and_retry(self) -> None:
        first = self.record(keys=self.keys("superseded"))

        second = self.record(
            keys=self.keys("successor"),
            supersedes_capture_id=first.capture.capture_id,
        )
        replay = self.record(
            keys=self.keys("successor"),
            supersedes_capture_id=first.capture.capture_id,
        )

        self.assertEqual(
            second.capture.supersedes_capture_id,
            first.capture.capture_id,
        )
        self.assertEqual(replay.capture.capture_id, second.capture.capture_id)
        self.assertEqual(
            replay.link.envelope.record_id,
            second.link.envelope.record_id,
        )
        self.assertEqual(len(tuple(self.store.iter_records("capture"))), 2)

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

    def test_historical_forecast_replays_with_compact_ledger_output(self) -> None:
        comparison = compare_shadow_forecast(self.legacy, self.forecast)
        compact_outputs = shadow_module._forecast_outputs(
            self.forecast,
            comparison,
        )
        historical_outputs = json.loads(canonical_json(compact_outputs))
        historical_outputs["component_distributions"] = (
            expand_component_distributions(
                historical_outputs["component_distributions"]
            )
        )
        historical = self.store.append_forecast(
            ForecastCommand(
                forecast_run_id=self.forecast.forecast_run_id,
                model_version=self.forecast.model_version,
                simulation_config_hash=hashlib.sha256(
                    canonical_json(self.forecast.simulation_config).encode("utf-8")
                ).hexdigest(),
                normalized_input_hash=normalized_input_hash(self.game),
                outputs=historical_outputs,
            )
        )

        replay = self.record(keys=self.keys("historical-compact-replay"))

        self.assertEqual(replay.forecast, historical)
        self.assertIn(
            "values",
            replay.forecast.outputs["component_distributions"][
                "away_full_runs"
            ],
        )
        self.assertEqual(replay.link.capture_id, replay.capture.capture_id)
        self.assertEqual(
            replay.link.forecast_run_id,
            historical.forecast_run_id,
        )
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertEqual(
            len(tuple(self.store.iter_records("capture_forecast_link"))),
            1,
        )

    def test_recording_omits_expected_head_for_legacy_store_signature(
        self,
    ) -> None:
        tracking = LegacyCaptureSignatureStore(self.store)

        result = self.record(store=tracking)

        self.assertEqual(tracking.calls, ["capture", "forecast", "link"])
        self.assertEqual(result.capture.game_pk, self.game.identity.game_pk)

    def test_recording_forwards_explicit_expected_head(self) -> None:
        existing = self.store.append_capture(
            "capture:existing-head",
            build_shadow_capture_command(self.game),
        )

        with self.assertRaisesRegex(
            LedgerHeadChangedError,
            existing.capture_id,
        ):
            record_shadow_run(
                self.store,
                self.game,
                self.legacy,
                self.forecast,
                self.keys("expected-head"),
                expected_active_capture_id=None,
            )

    def test_recording_blocks_candidate_market_mismatch(self) -> None:
        mismatched_binding = replace(
            binding(),
            legacy_market="full_total",
        )
        legacy = legacy_snapshot(review_bindings=(mismatched_binding,))

        with self.assertRaisesRegex(
            ValueError,
            r"candidate\[0\] ml \(play\) sea_ml: candidate_market_mismatch",
        ):
            self.record(
                legacy=legacy,
                keys=self.keys("candidate-market-mismatch"),
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recording_rejects_spoofed_precomputed_comparison(self) -> None:
        comparison = compare_shadow_forecast(self.legacy, self.forecast)
        market = self.forecast.markets["sea_ml"]
        spoofed_market = replace(
            price_market(
                OutcomeProbabilities(
                    win=market.probabilities.win - 0.01,
                    push=market.probabilities.push,
                    loss=market.probabilities.loss + 0.01,
                ),
                market.selected_decimal_price,
                market.opposing_decimal_price,
            ),
            semantic_fingerprint=market.semantic_fingerprint,
        )
        spoofed_forecast = replace(
            self.forecast,
            markets={
                **self.forecast.markets,
                "sea_ml": spoofed_market,
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            "shadow market evaluation model_win_probability must match",
        ):
            record_shadow_run(
                self.store,
                self.game,
                self.legacy,
                spoofed_forecast,
                self.keys("spoofed-comparison"),
                comparison=comparison,
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

    def test_recorded_payload_reconstructs_the_full_immutable_comparison(
        self,
    ) -> None:
        self.assertTrue(
            callable(deserialize_shadow_comparison),
            "shadow comparison deserializer is missing",
        )
        recorded = self.record()

        reconstructed = deserialize_shadow_comparison(
            recorded.forecast.outputs["shadow_comparison"]
        )

        self.assertEqual(
            reconstructed,
            compare_shadow_forecast(self.legacy, self.forecast),
        )

    def test_equivalent_binding_and_path_order_reuses_forecast_payload(self) -> None:
        paths = principal_paths()
        first_bindings = (
            binding(index=0, paths=paths),
            binding(
                index=1,
                market_key="full_under",
                legacy_market="full_total",
                paths=paths,
            ),
        )
        candidates = (
            candidate(market="ml"),
            candidate(market="full_total"),
        )
        first_legacy = legacy_snapshot(
            candidates=candidates,
            review_bindings=first_bindings,
        )
        second_bindings = (
            replace(
                first_bindings[1],
                principal_paths=tuple(reversed(paths)),
            ),
            replace(
                first_bindings[0],
                principal_paths=tuple(reversed(paths)),
            ),
        )
        second_legacy = legacy_snapshot(
            candidates=candidates,
            review_bindings=second_bindings,
        )
        original_bindings = second_legacy.review_bindings
        original_paths = second_legacy.review_bindings[0].principal_paths

        first = self.record(
            legacy=first_legacy,
            keys=self.keys("canonical-order-first"),
        )
        second = self.record(
            game=replace(
                self.game,
                captured_at=self.game.captured_at + timedelta(minutes=1),
            ),
            legacy=second_legacy,
            keys=self.keys("canonical-order-second"),
        )

        self.assertEqual(
            first.forecast.envelope.record_id,
            second.forecast.envelope.record_id,
        )
        self.assertEqual(first.forecast.outputs, second.forecast.outputs)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
        self.assertIs(second_legacy.review_bindings, original_bindings)
        self.assertIs(
            second_legacy.review_bindings[0].principal_paths,
            original_paths,
        )
        self.assertEqual(
            tuple(
                item.legacy_candidate_index
                for item in second_legacy.review_bindings
            ),
            (1, 0),
        )
        self.assertEqual(
            second_legacy.review_bindings[0].principal_paths,
            tuple(reversed(paths)),
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

    def test_recording_prefers_atomic_store_and_forwards_expected_head(self) -> None:
        atomic_store = AtomicOnlyStore(self.store)

        result = record_shadow_run(
            atomic_store,
            self.game,
            self.legacy,
            self.forecast,
            self.keys(),
            expected_active_capture_id=None,
        )

        self.assertEqual(len(atomic_store.calls), 1)
        args = atomic_store.calls[0]
        self.assertEqual(args[0], self.keys().capture)
        self.assertIsInstance(args[1], CaptureCommand)
        self.assertIsInstance(args[2], ForecastCommand)
        self.assertIsNone(args[3])
        self.assertEqual(result.capture.game_pk, self.game.identity.game_pk)

    def test_known_unevaluated_market_records_without_invented_pricing(self) -> None:
        game, forecast, legacy = three_way_f5_scenario()

        result = self.record(
            game=game,
            forecast=forecast,
            legacy=legacy,
            keys=self.keys("three-way-f5"),
        )
        market = result.forecast.outputs["shadow_comparison"][
            "market_comparisons"
        ][0]

        self.assertTrue(market["review_eligible"])
        self.assertIsNone(market["model_win_probability"])
        self.assertIsNone(market["fair_decimal_price"])
        self.assertIsNone(market["edge"])
        self.assertEqual(len(tuple(self.store.iter_records())), 3)

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

    def test_non_forecast_output_is_type_error_before_any_append(self) -> None:
        with self.assertRaisesRegex(TypeError, "forecast.*ForecastOutput"):
            self.record(forecast=object())

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
            selected_side="home",
        )
        mismatched_forecast = replace(self.forecast)
        # Bypass construction only to exercise record's independent guard.
        object.__setattr__(
            mismatched_forecast,
            "market_inputs",
            market_inputs,
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

        with self.assertRaisesRegex(
            ValueError,
            r"candidate\[0\] ml \(play\) sea_ml: invalid_path_count",
        ):
            self.record(legacy=legacy, keys=self.keys("wrong-shape"))

        self.assertEqual(tuple(self.store.iter_records()), ())

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

        with self.assertRaisesRegex(
            ValueError,
            r"candidate\[0\] ml \(play\) looks:like:f5_total: invalid_path_count",
        ):
            self.record(
                game=game,
                legacy=legacy,
                forecast=forecast,
                keys=self.keys("wrong-horizon"),
            )

        self.assertEqual(tuple(self.store.iter_records()), ())

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
            candidates=(
                candidate(
                    evidence_paths=frozenset(
                        path.evidence_path for path in same_side_paths
                    )
                ),
            ),
            review_bindings=(binding(paths=same_side_paths),),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"candidate\[0\] ml \(play\) sea_ml: invalid_path_count",
        ):
            self.record(legacy=legacy, keys=self.keys("same-side"))

        self.assertEqual(tuple(self.store.iter_records()), ())

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
