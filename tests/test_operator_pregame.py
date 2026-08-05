from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import warnings

import pytest

import analysis.operator.bundles as bundles_module
import analysis.operator.cli as cli_module
import analysis.operator.service as service_module
import analysis.shadow as shadow_module
from analysis.operator.bundles import (
    ManagedBundleCopy,
    ReplayOnlyPregameBundle,
    bundle_canonical_bytes,
    bundle_sha256,
    intrinsic_capture_key,
    load_bundle,
    persist_managed_bundle,
)
from analysis.forecasting import LineupState
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.forecasting.simulation import _summarize
from analysis.forecasting.validation import (
    PregameRelationshipError,
    PregameTimingError,
)
from analysis.ledger import (
    ForecastCommand,
    ForecastRecord,
    IdempotencyConflict,
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerReferenceError,
    LedgerStore,
    linked_shadow_comparison,
)
from analysis.ledger.forecast_codec import (
    compact_forecast_outputs,
    expand_component_distributions,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.operator.errors import (
    BundleValidationError,
    OperatorError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    HistoricalConfirmedBundle,
    HistoricalProjectedBundle,
    ProjectedBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from analysis.operator.service import (
    PregameRunResult,
    replay_historical_confirmed,
    replay_historical_projected,
    run_confirmed,
    run_projected,
)
from analysis.review.classification import (
    ContradictionRule,
    PrincipalPathBinding,
)
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    build_shadow_capture_command,
    canonical_legacy_decision_payload,
)
from tests.test_forecasting_engine import _complete_game
from tests.test_forecasting_inputs import complete_forecast_game
from tests.test_operator_schemas import (
    _complete_legacy_decision_mapping,
    _raw,
    confirmed_bundle_mapping,
    projected_bundle_mapping,
    review_bundle_mapping,
)


def _projected_bundle() -> ProjectedBundle:
    return parse_projected_bundle_json(_raw(projected_bundle_mapping()))


def _projected_game(**overrides: object):
    game = _complete_game()
    projected_away = replace(game.away_lineup, state=LineupState.PROJECTED)
    values = {"away_lineup": projected_away}
    values.update(overrides)
    return replace(game, **values)


def _confirmed_game(**overrides: object):
    return replace(_complete_game(), **overrides)


def _pregame_bundle(stage: str, game):
    mapping = (
        projected_bundle_mapping()
        if stage == "projected"
        else confirmed_bundle_mapping()
    )
    mapping["payload"]["game_input"] = capture_payload(game)
    raw = _raw(mapping)
    if stage == "projected":
        return parse_projected_bundle_json(raw)
    return parse_confirmed_bundle_json(raw)


def _ordered_legacy_decision() -> LegacyDecisionSnapshot:
    starter_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )
    ml_paths = (
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
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.BULLPEN_PROTECTION,
            component_ids=(
                "home_bullpen_runs_allowed",
                "home_protected_bullpen_close_entry",
            ),
            contradiction_rule=ContradictionRule.BULLPEN_PROTECTION,
        ),
    )
    under_paths = (
        starter_path,
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.LINEUP_SUPPRESSION,
            component_ids=(
                "home_full_runs",
                "home_full_traffic",
                "home_full_extra_base_hits",
            ),
            contradiction_rule=ContradictionRule.LINEUP_SUPPRESSION,
        ),
    )
    candidates = (
        PriorityMarketCandidate(
            market="ml",
            score=78,
            decision_rating="play",
            evidence_paths=frozenset(
                path.evidence_path for path in ml_paths
            ),
            same_game_pair_compatible=True,
        ),
        PriorityMarketCandidate(
            market="full_total",
            score=74,
            decision_rating="lean",
            evidence_paths=frozenset(
                path.evidence_path for path in under_paths
            ),
            same_game_pair_compatible=True,
        ),
    )
    return LegacyDecisionSnapshot(
        model_version="2026-08-03",
        candidates=candidates,
        final_priority=("ml", "full_total"),
        review_bindings=(
            CandidateReviewBinding(
                market_key="sea_ml",
                legacy_candidate_index=0,
                legacy_market="ml",
                principal_paths=ml_paths,
            ),
            CandidateReviewBinding(
                market_key="full_under",
                legacy_candidate_index=1,
                legacy_market="full_total",
                principal_paths=under_paths,
            ),
        ),
        notes=("canonical replay order",),
    )


def _compatible_ml_over_case() -> tuple[object, LegacyDecisionSnapshot]:
    game = _projected_game()
    full_over = replace(
        next(market for market in game.markets if market.market_key == "full_under"),
        market_key="full_over",
        selected_side="over",
    )
    game = replace(
        game,
        markets=tuple(
            full_over if market.market_key == "full_under" else market
            for market in game.markets
        ),
    )
    top_order_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
        component_ids=(
            "away_full_runs",
            "away_full_traffic",
            "away_full_extra_base_hits",
        ),
        contradiction_rule=ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
    )
    bullpen_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.BULLPEN_PROTECTION,
        component_ids=(
            "home_bullpen_runs_allowed",
            "home_protected_bullpen_close_entry",
        ),
        contradiction_rule=ContradictionRule.BULLPEN_PROTECTION,
    )
    starter_collapse_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
        component_ids=(
            "home_pitcher_chain_runs_allowed",
            "home_pitcher_chain_workload_outs",
        ),
        contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
    )
    recent_conversion_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.RECENT_RUN_CONVERSION,
        component_ids=(
            "home_full_runs",
            "home_full_traffic",
            "home_full_extra_base_hits",
        ),
        contradiction_rule=ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
    )
    ml_paths = (top_order_path, bullpen_path)
    total_paths = (starter_collapse_path, recent_conversion_path)
    legacy = LegacyDecisionSnapshot(
        model_version="2026-08-03",
        candidates=(
            PriorityMarketCandidate(
                market="ml",
                score=77,
                decision_rating="play",
                evidence_paths=frozenset(
                    path.evidence_path for path in ml_paths
                ),
                same_game_pair_compatible=True,
            ),
            PriorityMarketCandidate(
                market="full_total",
                score=83,
                decision_rating="play",
                evidence_paths=frozenset(
                    path.evidence_path for path in total_paths
                ),
                same_game_pair_compatible=True,
            ),
        ),
        final_priority=("full_total", "ml"),
        review_bindings=(
            CandidateReviewBinding(
                market_key="sea_ml",
                legacy_candidate_index=0,
                legacy_market="ml",
                principal_paths=ml_paths,
            ),
            CandidateReviewBinding(
                market_key="full_over",
                legacy_candidate_index=1,
                legacy_market="full_total",
                principal_paths=total_paths,
            ),
        ),
        notes=("compatible score-sorted priority",),
    )
    return game, legacy


def _run_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "ledger" / "shadow.jsonl", tmp_path / "bundles"


def _write_hash_named_managed_bundle(
    mapping: dict[str, object],
    root: Path,
) -> Path:
    raw = canonical_json(mapping).encode("utf-8")
    bundle_hash = hashlib.sha256(raw).hexdigest()
    game_dir = root / str(mapping["game_pk"])
    game_dir.mkdir(parents=True, exist_ok=True)
    source = game_dir / (
        f"{mapping['stage']}-v{mapping['schema_version']}-"
        f"{bundle_hash[:16]}.json"
    )
    source.write_bytes(raw)
    return source


def _bundle_with_legacy(stage: str, game, legacy: dict[str, object]):
    market_key = next(
        market.market_key
        for market in game.markets
        if market.market_type.value == "ml"
    )
    for binding in legacy["review_bindings"]:
        binding["market_key"] = market_key
    mapping = (
        projected_bundle_mapping()
        if stage == "projected"
        else confirmed_bundle_mapping()
    )
    mapping["payload"]["game_input"] = capture_payload(game)
    mapping["payload"]["legacy_decision"] = legacy
    raw = _raw(mapping)
    if stage == "projected":
        return parse_projected_bundle_json(raw)
    return parse_confirmed_bundle_json(raw)


def _invalid_recommended_legacy_mapping() -> dict[str, object]:
    legacy = _complete_legacy_decision_mapping()
    candidate = legacy["candidates"][0]
    binding = legacy["review_bindings"][0]
    candidate["decision_rating"] = "lean"
    candidate["evidence_paths"] = ["starter_collapse"]
    binding["principal_paths"] = [
        {
            "evidence_path": "starter_collapse",
            "component_ids": ["home_pitcher_chain_runs_allowed"],
            "contradiction_rule": "starter_collapse",
        }
    ]
    return legacy


def _valid_recommended_legacy_mapping() -> dict[str, object]:
    legacy = _complete_legacy_decision_mapping()
    candidate = legacy["candidates"][0]
    binding = legacy["review_bindings"][0]
    candidate["evidence_paths"] = [
        "starter_suppression",
        "top_order_matchup",
        "bullpen_protection",
    ]
    binding["principal_paths"].append(
        {
            "evidence_path": "top_order_matchup",
            "component_ids": [
                "away_full_runs",
                "away_full_traffic",
                "away_full_extra_base_hits",
            ],
            "contradiction_rule": "top_order_or_recent_conversion",
        }
    )
    return legacy


def test_fresh_stale_legacy_model_version_rejects_before_forecast_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _valid_recommended_legacy_mapping()
    legacy["model_version"] = "2026-07-13"
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("forecast must not run")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(
        OperatorPreconditionError,
        match=(
            "fresh pregame legacy model_version must be 2026-08-03; "
            "received 2026-07-13"
        ),
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert not ledger.parent.exists()
    assert not ledger.exists()
    assert not bundles.exists()


def test_invalid_recommended_contract_before_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        _invalid_recommended_legacy_mapping(),
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("forecast must not run")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(
        OperatorPreconditionError,
        match="recommended review contract is invalid",
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert not ledger.exists()
    assert not bundles.exists()


@pytest.mark.parametrize(
    "violation",
    ("below_threshold", "duplicate_side_family", "correlated", "incompatible"),
)
def test_fresh_pregame_rejects_priority_stack_differing_from_assembler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    game, legacy = _compatible_ml_over_case()

    if violation == "below_threshold":
        legacy = replace(
            legacy,
            candidates=(replace(legacy.candidates[0], score=71),),
            final_priority=("ml",),
            review_bindings=(legacy.review_bindings[0],),
        )
    elif violation == "duplicate_side_family":
        handicap_paths = legacy.review_bindings[1].principal_paths
        legacy = replace(
            legacy,
            candidates=(
                legacy.candidates[0],
                replace(
                    legacy.candidates[1],
                    market="plus_1_5",
                    score=76,
                    same_game_pair_compatible=False,
                ),
            ),
            final_priority=("ml", "plus_1_5"),
            review_bindings=(
                legacy.review_bindings[0],
                CandidateReviewBinding(
                    market_key="sea_plus_1_5",
                    legacy_candidate_index=1,
                    legacy_market="plus_1_5",
                    principal_paths=handicap_paths,
                ),
            ),
        )
    elif violation == "correlated":
        shared_path = legacy.review_bindings[0].principal_paths[0]
        total_binding = legacy.review_bindings[1]
        correlated_paths = (
            total_binding.principal_paths[0],
            shared_path,
        )
        legacy = replace(
            legacy,
            candidates=(
                legacy.candidates[0],
                replace(
                    legacy.candidates[1],
                    evidence_paths=frozenset(
                        path.evidence_path for path in correlated_paths
                    ),
                ),
            ),
            review_bindings=(
                legacy.review_bindings[0],
                replace(total_binding, principal_paths=correlated_paths),
            ),
        )
    else:
        legacy = replace(
            legacy,
            candidates=tuple(
                replace(candidate, same_game_pair_compatible=False)
                for candidate in legacy.candidates
            ),
        )

    base = _pregame_bundle("projected", game)
    bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("forecast must not run")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(
        OperatorPreconditionError,
        match="final_priority must exactly match assembled priority",
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert not ledger.exists()
    assert not bundles.exists()


def test_fresh_pregame_accepts_compatible_ml_and_over_in_score_order(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    game, legacy = _compatible_ml_over_case()
    base = _pregame_bundle("projected", game)
    bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )

    result = run_projected(
        bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    frozen = shadow_module.deserialize_shadow_comparison(
        linked_shadow_comparison(result.link, result.forecast)
    ).legacy
    assert frozen.final_priority == ("full_total", "ml")
    scores_by_market = {
        candidate.market: candidate.score for candidate in frozen.candidates
    }
    assert tuple(
        (market, scores_by_market[market])
        for market in frozen.final_priority
    ) == (("full_total", 83), ("ml", 77))


def test_invalid_static_path_count_rejects_before_forecast_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _complete_legacy_decision_mapping()
    candidate = legacy["candidates"][0]
    binding = legacy["review_bindings"][0]
    candidate["evidence_paths"] = ["bullpen_protection"]
    binding["principal_paths"] = binding["principal_paths"][1:]
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("forecast must not run")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(
        OperatorPreconditionError,
        match=r"candidate\[0\] ml \(play\): invalid_path_count",
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert not ledger.parent.exists()
    assert not ledger.exists()
    assert not bundles.exists()


def test_exact_replay_wins_over_a_stale_fresh_ledger_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _invalid_recommended_legacy_mapping()
    legacy["model_version"] = "legacy-fixed-v1"
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    with monkeypatch.context() as gates:
        gates.setattr(
            service_module,
            "LEGACY_MODEL_VERSION",
            "legacy-fixed-v1",
        )
        gates.setattr(
            service_module,
            "validate_recommended_review_contract",
            lambda *args, **kwargs: None,
        )
        gates.setattr(
            service_module,
            "require_recommended_review_eligibility",
            lambda *args, **kwargs: None,
        )
        gates.setattr(
            shadow_module,
            "require_recommended_review_eligibility",
            lambda *args, **kwargs: None,
        )
        historical = run_projected(
            bundle,
            ledger_path=ledger,
            bundle_root=bundles,
        )

    path_exists = Path.exists

    def stale_exists(path: Path) -> bool:
        if path == ledger:
            return False
        return path_exists(path)

    monkeypatch.setattr(Path, "exists", stale_exists)

    replay = run_projected(
        bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert replay.capture == historical.capture
    assert replay.forecast == historical.forecast
    assert replay.link == historical.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True


def test_fresh_dynamic_rejection_creates_no_ledger_artifacts(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _complete_legacy_decision_mapping()
    candidate = legacy["candidates"][0]
    binding = legacy["review_bindings"][0]
    candidate["evidence_paths"] = [
        "starter_suppression",
        "bullpen_protection",
        "late_bullpen_exposure",
    ]
    binding["principal_paths"].append(
        {
            "evidence_path": "late_bullpen_exposure",
            "component_ids": [
                "home_bullpen_segment_missing_exposed",
                "home_pitcher_chain_workload_outs",
            ],
            "contradiction_rule": "late_bullpen_exposure",
        }
    )
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="recommended review comparison is ineligible",
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert not ledger.parent.exists()
    assert not ledger.exists()
    assert not ledger.with_name(f"{ledger.name}.lock").exists()
    assert not ledger.with_name(f"{ledger.name}.checkpoint-v1.json").exists()
    assert not bundles.exists()


@pytest.mark.parametrize("rejection", ("static", "dynamic"))
@pytest.mark.parametrize("checkpoint_state", ("missing", "stale"))
def test_existing_rejection_does_not_update_checkpoint(
    tmp_path: Path,
    rejection: str,
    checkpoint_state: str,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    run_projected(
        _pregame_bundle("projected", _projected_game()),
        ledger_path=ledger,
        bundle_root=bundles,
    )
    checkpoint = ledger.with_name(f"{ledger.name}.checkpoint-v1.json")
    lock = ledger.with_name(f"{ledger.name}.lock")
    assert checkpoint.exists()
    if checkpoint_state == "missing":
        checkpoint.unlink()
        checkpoint_before = None
    else:
        ledger.touch()
        checkpoint_before = checkpoint.read_bytes()
    ledger_before = ledger.read_bytes()
    lock_before = lock.read_bytes()
    bundles_before = {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    }

    if rejection == "static":
        bundle = _bundle_with_legacy(
            "projected",
            _projected_game(),
            _invalid_recommended_legacy_mapping(),
        )
        expected = "recommended review contract is invalid"
    else:
        legacy = _complete_legacy_decision_mapping()
        candidate = legacy["candidates"][0]
        binding = legacy["review_bindings"][0]
        candidate["evidence_paths"] = [
            "starter_suppression",
            "bullpen_protection",
            "late_bullpen_exposure",
        ]
        binding["principal_paths"].append(
            {
                "evidence_path": "late_bullpen_exposure",
                "component_ids": [
                    "home_bullpen_segment_missing_exposed",
                    "home_pitcher_chain_workload_outs",
                ],
                "contradiction_rule": "late_bullpen_exposure",
            }
        )
        bundle = _bundle_with_legacy(
            "projected",
            _projected_game(),
            legacy,
        )
        expected = "recommended review comparison is ineligible"

    with pytest.raises(OperatorPreconditionError, match=expected):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert ledger.read_bytes() == ledger_before
    assert lock.read_bytes() == lock_before
    if checkpoint_before is None:
        assert not checkpoint.exists()
    else:
        assert checkpoint.read_bytes() == checkpoint_before
    assert {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    } == bundles_before


def test_successful_operator_run_constructs_one_shadow_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        _valid_recommended_legacy_mapping(),
    )
    real_compare = shadow_module.compare_shadow_forecast
    calls = 0

    def compare_spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compare(*args, **kwargs)

    monkeypatch.setattr(service_module, "compare_shadow_forecast", compare_spy)
    monkeypatch.setattr(shadow_module, "compare_shadow_forecast", compare_spy)

    run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert calls == 1


@pytest.mark.parametrize(
    ("stage", "runner", "game_factory"),
    (
        ("projected", run_projected, _projected_game),
        ("confirmed", run_confirmed, _confirmed_game),
    ),
)
def test_recommended_pregame_run_records_and_replays(
    tmp_path: Path,
    stage: str,
    runner,
    game_factory,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    bundle = _bundle_with_legacy(
        stage,
        game_factory(),
        _valid_recommended_legacy_mapping(),
    )

    first = runner(bundle, ledger_path=ledger, bundle_root=bundles)
    replay = runner(bundle, ledger_path=ledger, bundle_root=bundles)

    assert replay.capture == first.capture
    assert replay.forecast == first.forecast
    assert replay.link == first.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True
    assert len(tuple(LedgerStore(ledger).iter_records())) == 3


def test_unbound_pass_candidate_is_accepted(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _complete_legacy_decision_mapping()
    legacy["candidates"][0]["decision_rating"] = "pass"
    legacy["final_priority"] = []
    legacy["review_bindings"] = []
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    result = run_projected(
        bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert result.capture_replayed is False
    assert len(tuple(LedgerStore(ledger).iter_records())) == 3


def test_dynamic_ineligibility_leaves_ledger_and_bundle_unchanged(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    baseline = _pregame_bundle("projected", _projected_game())
    run_projected(baseline, ledger_path=ledger, bundle_root=bundles)
    ledger_before = ledger.read_bytes()
    bundles_before = {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    }
    legacy = _complete_legacy_decision_mapping()
    candidate = legacy["candidates"][0]
    binding = legacy["review_bindings"][0]
    candidate["evidence_paths"] = [
        "starter_suppression",
        "bullpen_protection",
        "late_bullpen_exposure",
    ]
    binding["principal_paths"].append(
        {
            "evidence_path": "late_bullpen_exposure",
            "component_ids": [
                "home_bullpen_segment_missing_exposed",
                "home_pitcher_chain_workload_outs",
            ],
            "contradiction_rule": "late_bullpen_exposure",
        }
    )
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="recommended review comparison is ineligible",
    ):
        run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

    assert ledger.read_bytes() == ledger_before
    assert {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    } == bundles_before


def test_historical_invalid_recommended_command_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    legacy = _invalid_recommended_legacy_mapping()
    legacy["model_version"] = "legacy-fixed-v1"
    bundle = _bundle_with_legacy(
        "projected",
        _projected_game(),
        legacy,
    )

    with monkeypatch.context() as gates:
        gates.setattr(
            service_module,
            "LEGACY_MODEL_VERSION",
            "legacy-fixed-v1",
        )
        gates.setattr(
            service_module,
            "validate_recommended_review_contract",
            lambda *args, **kwargs: None,
        )
        gates.setattr(
            service_module,
            "require_recommended_review_eligibility",
            lambda *args, **kwargs: None,
        )
        gates.setattr(
            shadow_module,
            "require_recommended_review_eligibility",
            lambda *args, **kwargs: None,
        )
        historical = run_projected(
            bundle,
            ledger_path=ledger,
            bundle_root=bundles,
        )

    replay = run_projected(
        bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert replay.capture == historical.capture
    assert replay.forecast == historical.forecast
    assert replay.link == historical.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True
    assert len(tuple(LedgerStore(ledger).iter_records())) == 3


def test_pre_change_schema_v1_bundle_and_shadow_payload_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    game, legacy = _compatible_ml_over_case()
    base = _pregame_bundle("projected", game)
    fresh_bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )
    first = run_projected(
        fresh_bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    historical_outputs = json.loads(canonical_json(first.forecast.outputs))
    for candidate in historical_outputs["shadow_comparison"]["legacy"][
        "candidates"
    ]:
        del candidate["same_game_pair_compatible"]
    _rewrite_forecast_outputs(ledger, first.forecast, historical_outputs)
    historical_forecast = next(
        record
        for record in LedgerStore(ledger).iter_records("forecast")
        if isinstance(record, ForecastRecord)
    )

    mapping = json.loads(bundle_canonical_bytes(fresh_bundle))
    for candidate in mapping["payload"]["legacy_decision"]["candidates"]:
        del candidate["same_game_pair_compatible"]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    def fail_forecast(*args, **kwargs):
        raise AssertionError("exact historical replay must not run forecast")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    replay = replay_historical_projected(
        source,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert replay.capture == first.capture
    assert replay.forecast == historical_forecast
    assert replay.link == first.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True


@pytest.mark.parametrize(
    ("stage", "runner", "game_factory"),
    (
        ("projected", run_projected, _projected_game),
        ("confirmed", run_confirmed, _confirmed_game),
    ),
)
def test_normal_runner_rejects_historical_artifact(
    tmp_path: Path,
    stage: str,
    runner,
    game_factory,
) -> None:
    _, bundles = _run_paths(tmp_path)
    empty_ledger = tmp_path / "empty-ledger" / "shadow.jsonl"
    bundle = _bundle_with_legacy(
        stage,
        game_factory(),
        _valid_recommended_legacy_mapping(),
    )
    mapping = json.loads(bundle_canonical_bytes(bundle))
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    historical_artifact = bundles_module.load_historical_managed_bundle(
        source,
        root=bundles,
    )

    with pytest.raises(
        TypeError,
        match="normal pregame runner requires a strict current bundle",
    ):
        runner(
            historical_artifact,
            ledger_path=empty_ledger,
            bundle_root=bundles,
        )

    assert not empty_ledger.exists()


@pytest.mark.parametrize(
    ("stage", "runner", "game_factory"),
    (
        ("projected", run_projected, _projected_game),
        ("confirmed", run_confirmed, _confirmed_game),
    ),
)
def test_normal_runner_rejects_historical_parsed_bundle(
    tmp_path: Path,
    stage: str,
    runner,
    game_factory,
) -> None:
    _, bundles = _run_paths(tmp_path)
    empty_ledger = tmp_path / "empty-ledger" / "shadow.jsonl"
    bundle = _bundle_with_legacy(
        stage,
        game_factory(),
        _valid_recommended_legacy_mapping(),
    )
    mapping = json.loads(bundle_canonical_bytes(bundle))
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    historical_artifact = bundles_module.load_historical_managed_bundle(
        source,
        root=bundles,
    )
    assert isinstance(historical_artifact, ReplayOnlyPregameBundle)
    managed_files_before = {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        TypeError,
        match="normal pregame runner requires a strict current bundle",
    ):
        runner(
            historical_artifact.bundle,
            ledger_path=empty_ledger,
            bundle_root=bundles,
        )

    assert not empty_ledger.exists()
    assert {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    } == managed_files_before


@pytest.mark.parametrize(
    ("stage", "runner", "historical_replay", "game_factory"),
    (
        (
            "projected",
            run_projected,
            replay_historical_projected,
            _projected_game,
        ),
        (
            "confirmed",
            run_confirmed,
            replay_historical_confirmed,
            _confirmed_game,
        ),
    ),
)
def test_historical_dedicated_replay_preserves_managed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    runner,
    historical_replay,
    game_factory,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    empty_ledger = tmp_path / "empty-ledger" / "shadow.jsonl"
    fresh_bundle = _bundle_with_legacy(
        stage,
        game_factory(),
        _valid_recommended_legacy_mapping(),
    )
    first = runner(
        fresh_bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    historical_outputs = json.loads(canonical_json(first.forecast.outputs))
    del historical_outputs["shadow_comparison"]["legacy"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    _rewrite_forecast_outputs(ledger, first.forecast, historical_outputs)
    historical_forecast = next(
        record
        for record in LedgerStore(ledger).iter_records("forecast")
        if isinstance(record, ForecastRecord)
    )

    mapping = json.loads(bundle_canonical_bytes(fresh_bundle))
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    files_before_replay = {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    }

    def fail_historical_side_effect(*args, **kwargs):
        raise AssertionError("historical replay reached a forbidden write path")

    monkeypatch.setattr(service_module, "run_forecast", fail_historical_side_effect)
    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        fail_historical_side_effect,
    )
    monkeypatch.setattr(
        service_module,
        "persist_managed_bundle",
        fail_historical_side_effect,
    )
    replay = historical_replay(
        source,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="historical replay requires an exact existing replay",
    ):
        historical_replay(
            source,
            ledger_path=empty_ledger,
            bundle_root=bundles,
        )

    assert replay.capture == first.capture
    assert replay.forecast == historical_forecast
    assert replay.link == first.link
    assert replay.managed_bundle.path == source.resolve()
    assert replay.managed_bundle.bundle_hash == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert replay.managed_bundle.replayed is True
    assert not empty_ledger.exists()
    assert {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    } == files_before_replay


@pytest.mark.parametrize(
    ("stage", "runner", "historical_replay", "historical_type"),
    (
        (
            "projected",
            run_projected,
            replay_historical_projected,
            HistoricalProjectedBundle,
        ),
        (
            "confirmed",
            run_confirmed,
            replay_historical_confirmed,
            HistoricalConfirmedBundle,
        ),
    ),
)
def test_two_market_historical_replay_cannot_substitute_runner_or_provenance(
    tmp_path: Path,
    stage: str,
    runner,
    historical_replay,
    historical_type,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    empty_ledger = tmp_path / "empty-ledger" / "shadow.jsonl"
    game, legacy = _compatible_ml_over_case()
    if stage == "confirmed":
        game = replace(
            game,
            away_lineup=replace(game.away_lineup, state=LineupState.CONFIRMED),
        )
    base = _pregame_bundle(stage, game)
    fresh_bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )
    first = runner(
        fresh_bundle,
        ledger_path=ledger,
        bundle_root=bundles,
    )
    assert len(
        first.forecast.outputs["shadow_comparison"]["legacy"]["candidates"]
    ) == 2

    historical_outputs = json.loads(canonical_json(first.forecast.outputs))
    for candidate in historical_outputs["shadow_comparison"]["legacy"][
        "candidates"
    ]:
        del candidate["same_game_pair_compatible"]
    _rewrite_forecast_outputs(ledger, first.forecast, historical_outputs)

    mapping = json.loads(bundle_canonical_bytes(fresh_bundle))
    for candidate in mapping["payload"]["legacy_decision"]["candidates"]:
        del candidate["same_game_pair_compatible"]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    artifact = bundles_module.load_historical_managed_bundle(source, root=bundles)
    assert isinstance(artifact, ReplayOnlyPregameBundle)
    assert type(artifact.bundle) is historical_type
    caller_constructed_artifact = ReplayOnlyPregameBundle(
        bundle=artifact.bundle,
        managed_bundle=ManagedBundleCopy(
            path=source,
            bundle_hash="caller-supplied-hash",
            replayed=False,
        ),
    )
    files_before = {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    }

    returned = historical_replay(
        source,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    with pytest.raises(
        TypeError,
        match="normal pregame runner requires a strict current bundle",
    ):
        runner(
            caller_constructed_artifact,
            ledger_path=empty_ledger,
            bundle_root=bundles,
        )
    with pytest.raises(
        OperatorPreconditionError,
        match="historical replay requires a pre-change managed bundle",
    ):
        historical_replay(
            first.managed_bundle.path,
            ledger_path=empty_ledger,
            bundle_root=bundles,
        )

    assert returned.managed_bundle.path == source.resolve()
    assert returned.managed_bundle.bundle_hash == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert (
        returned.managed_bundle.bundle_hash
        != caller_constructed_artifact.managed_bundle.bundle_hash
    )
    assert not empty_ledger.exists()
    assert {
        path.relative_to(bundles): path.read_bytes()
        for path in bundles.rglob("*")
        if path.is_file()
    } == files_before


def _rewrite_forecast_outputs(
    ledger_path: Path,
    forecast: ForecastRecord,
    outputs: dict[str, object],
) -> None:
    command = ForecastCommand(
        forecast_run_id=forecast.forecast_run_id,
        model_version=forecast.model_version,
        simulation_config_hash=forecast.simulation_config_hash,
        normalized_input_hash=forecast.normalized_input_hash,
        outputs=outputs,
    )
    payload_hash = hashlib.sha256(
        canonical_json(command).encode("utf-8")
    ).hexdigest()
    replacement = replace(
        forecast,
        envelope=replace(forecast.envelope, payload_hash=payload_hash),
        outputs=outputs,
    )
    records = tuple(LedgerStore(ledger_path).iter_records())
    ledger_path.write_text(
        "".join(
            canonical_json(
                replacement
                if isinstance(record, ForecastRecord)
                and record.forecast_run_id == forecast.forecast_run_id
                else record
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _fast_operator_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        version="shadow-forecast-2026-07-14.1",
        simulations=12,
    )
    monkeypatch.setattr(
        "analysis.operator.service.load_forecast_config",
        lambda version, *, registry_root=None: config,
    )


def test_capture_key_covers_schema_stage_and_all_intrinsic_command_fields() -> None:
    command = build_shadow_capture_command(complete_forecast_game())
    baseline = intrinsic_capture_key(1, "projected", command)
    mutations = (
        (2, "projected", command),
        (1, "confirmed", command),
        (1, "projected", replace(command, game_pk=command.game_pk + 1)),
        (
            1,
            "projected",
            replace(command, captured_at=command.captured_at + timedelta(seconds=1)),
        ),
        (
            1,
            "projected",
            replace(command, first_pitch=command.first_pitch + timedelta(seconds=1)),
        ),
        (1, "projected", replace(command, lineup_state="projected")),
        (
            1,
            "projected",
            replace(
                command,
                out_of_sample_eligible=not command.out_of_sample_eligible,
            ),
        ),
        (
            1,
            "projected",
            replace(command, normalized_input_hash="different-input-hash"),
        ),
        (
            1,
            "projected",
            replace(command, payload={**command.payload, "audit_marker": "new"}),
        ),
    )

    for schema_version, stage, mutated_command in mutations:
        assert intrinsic_capture_key(
            schema_version,
            stage,
            mutated_command,
        ) != baseline

    assert intrinsic_capture_key(
        1,
        "projected",
        replace(command, supersedes_capture_id="another-capture"),
    ) == baseline


def test_canonical_bundle_bytes_normalize_offsets_and_hash_exact_bytes() -> None:
    mapping = projected_bundle_mapping()
    equivalent = projected_bundle_mapping()
    equivalent["created_at"] = "2026-07-13T21:35:00-04:00"
    bundle = parse_projected_bundle_json(_raw(mapping))
    offset_bundle = parse_projected_bundle_json(_raw(equivalent))

    canonical = bundle_canonical_bytes(bundle)

    assert canonical == bundle_canonical_bytes(offset_bundle)
    assert canonical == canonical.strip()
    assert bundle_sha256(bundle) == __import__("hashlib").sha256(canonical).hexdigest()


def test_canonical_bundle_bytes_support_nonempty_legacy_evidence_paths(
    tmp_path: Path,
) -> None:
    mapping = projected_bundle_mapping()
    mapping["payload"]["legacy_decision"] = (
        _complete_legacy_decision_mapping()
    )
    bundle = parse_projected_bundle_json(_raw(mapping))

    canonical = bundle_canonical_bytes(bundle)
    decoded = json.loads(canonical)

    assert decoded["payload"]["legacy_decision"]["candidates"][0][
        "evidence_paths"
    ] == ["bullpen_protection", "starter_suppression"]
    persisted = persist_managed_bundle(bundle, root=tmp_path)
    assert load_bundle(persisted.path) == bundle


def test_canonical_bundle_bytes_do_not_emit_validated_domain_warnings() -> None:
    bundle = parse_review_bundle_json(_raw(review_bundle_mapping()))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canonical = bundle_canonical_bytes(bundle)

    assert json.loads(canonical)["stage"] == "review"


def test_load_bundle_delegates_strict_stage_schema_parsing(tmp_path: Path) -> None:
    source = tmp_path / "projected.json"
    source.write_text(_raw(projected_bundle_mapping()), encoding="utf-8")

    loaded = load_bundle(source)

    assert isinstance(loaded, ProjectedBundle)
    assert loaded == _projected_bundle()


def test_historical_schema_v1_managed_bundle_defaults_missing_compatibility(
    tmp_path: Path,
) -> None:
    mapping = projected_bundle_mapping()
    mapping["payload"]["legacy_decision"] = (
        _complete_legacy_decision_mapping()
    )
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    source = _write_hash_named_managed_bundle(mapping, tmp_path)

    loaded = bundles_module.load_historical_managed_bundle(
        source,
        root=tmp_path,
    )
    loaded_bundle = getattr(loaded, "bundle", loaded)

    assert loaded_bundle.payload.legacy_decision.candidates[
        0
    ].same_game_pair_compatible is False


def test_missing_compatibility_managed_bundle_cannot_create_fresh_records(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    game = _projected_game()
    legacy = _valid_recommended_legacy_mapping()
    legacy["review_bindings"][0]["market_key"] = "sea_ml"
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(game)
    mapping["payload"]["legacy_decision"] = legacy
    del mapping["payload"]["legacy_decision"]["candidates"][0][
        "same_game_pair_compatible"
    ]
    source = _write_hash_named_managed_bundle(mapping, bundles)
    with pytest.raises(
        OperatorPreconditionError,
        match="historical replay requires an exact existing replay",
    ):
        replay_historical_projected(
            source,
            ledger_path=ledger,
            bundle_root=bundles,
        )

    assert not ledger.exists()


def test_strict_current_managed_bundle_can_create_fresh_records(
    tmp_path: Path,
) -> None:
    ledger, bundles = _run_paths(tmp_path)
    game = _projected_game()
    bundle = _bundle_with_legacy(
        "projected",
        game,
        _valid_recommended_legacy_mapping(),
    )
    persisted = persist_managed_bundle(bundle, root=bundles)
    loaded = cli_module._load_stage_bundle(
        persisted.path,
        expected_type=ProjectedBundle,
        expected_stage="projected",
        bundle_root=bundles,
    )

    result = run_projected(
        loaded,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert result.capture_replayed is False
    assert len(tuple(LedgerStore(ledger).iter_records())) == 3


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b'\xff',
    ),
)
def test_load_bundle_rejects_malformed_duplicate_nonfinite_and_non_utf8_json(
    tmp_path: Path,
    raw: bytes,
) -> None:
    source = tmp_path / "invalid.json"
    source.write_bytes(raw)

    with pytest.raises(BundleValidationError):
        load_bundle(source)


def test_load_bundle_wraps_filesystem_errors(tmp_path: Path) -> None:
    with pytest.raises(OperatorIOError, match="read bundle"):
        load_bundle(tmp_path / "missing.json")


def test_managed_bundle_copy_is_immutable_and_replayable(tmp_path: Path) -> None:
    bundle = _projected_bundle()

    first = persist_managed_bundle(bundle, root=tmp_path)
    replay = persist_managed_bundle(bundle, root=tmp_path)

    expected_hash = bundle_sha256(bundle)
    assert isinstance(first, ManagedBundleCopy)
    assert first.bundle_hash == expected_hash
    assert first.path == (
        tmp_path
        / str(bundle.game_pk)
        / f"projected-v1-{expected_hash[:16]}.json"
    )
    assert first.path.read_bytes() == bundle_canonical_bytes(bundle)
    assert first.replayed is False
    assert replay == replace(first, replayed=True)
    assert not tuple(first.path.parent.glob("*.tmp"))
    with pytest.raises(FrozenInstanceError):
        first.replayed = True


def test_managed_bundle_rejects_conflicting_existing_bytes(tmp_path: Path) -> None:
    bundle = _projected_bundle()
    digest = bundle_sha256(bundle)
    game_dir = tmp_path / str(bundle.game_pk)
    game_dir.mkdir()
    target = game_dir / f"projected-v1-{digest[:16]}.json"
    target.write_bytes(b"different")

    with pytest.raises(OperatorIOError, match="conflicting managed bundle"):
        persist_managed_bundle(bundle, root=tmp_path)

    assert target.read_bytes() == b"different"


def test_managed_bundle_rejects_symlinked_game_directory(tmp_path: Path) -> None:
    bundle = _projected_bundle()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / str(bundle.game_pk)).symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperatorIOError, match="symbolic link"):
        persist_managed_bundle(bundle, root=tmp_path)

    assert tuple(outside.iterdir()) == ()


def test_managed_bundle_rejects_symlinked_root(tmp_path: Path) -> None:
    bundle = _projected_bundle()
    outside = tmp_path / "outside-root"
    outside.mkdir()
    root = tmp_path / "bundle-root"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperatorIOError, match="symbolic link"):
        persist_managed_bundle(bundle, root=root)

    assert tuple(outside.iterdir()) == ()


def test_managed_bundle_rejects_symlinked_target_without_mutating_it(
    tmp_path: Path,
) -> None:
    bundle = _projected_bundle()
    digest = bundle_sha256(bundle)
    game_dir = tmp_path / str(bundle.game_pk)
    game_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    target = game_dir / f"projected-v1-{digest[:16]}.json"
    target.symlink_to(outside)

    with pytest.raises(OperatorIOError, match="symbolic link"):
        persist_managed_bundle(bundle, root=tmp_path)

    assert outside.read_text(encoding="utf-8") == "outside"


def test_managed_bundle_wraps_root_io_failures(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")

    with pytest.raises(OperatorIOError, match="persist managed bundle"):
        persist_managed_bundle(_projected_bundle(), root=root)


def test_projected_run_creates_capture_forecast_link_and_bundle(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    result = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert isinstance(result, PregameRunResult)
    assert result.stage == "projected"
    assert result.lineup_state == "projected"
    assert result.capture.game_pk == bundle.game_pk
    assert result.forecast.forecast_run_id == result.link.forecast_run_id
    assert result.capture.capture_id == result.link.capture_id
    assert result.managed_bundle.path.is_file()
    assert result.capture_replayed is False
    assert result.forecast_replayed is False
    assert result.link_replayed is False
    assert result.managed_bundle.replayed is False
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 3
    with pytest.raises(FrozenInstanceError):
        result.stage = "confirmed"


def test_projected_run_uses_one_atomic_shadow_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())
    real_append = LedgerStore.append_shadow_run_atomic
    calls = 0

    def append_spy(self, *args):
        nonlocal calls
        calls += 1
        return real_append(self, *args)

    monkeypatch.setattr(
        LedgerStore,
        "append_shadow_run_atomic",
        append_spy,
    )

    result = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert calls == 1
    assert result.link.capture_id == result.capture.capture_id


def test_concurrent_same_key_reuse_reports_atomic_replay_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())
    original_record = service_module.record_shadow_run
    concurrent_records = None

    def concurrently_record_same_key(store, game_input, *args, **kwargs):
        nonlocal concurrent_records
        if concurrent_records is None:
            concurrent_records = original_record(
                LedgerStore(ledger_path),
                game_input,
                *args,
                **kwargs,
            )
        return original_record(store, game_input, *args, **kwargs)

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        concurrently_record_same_key,
    )

    result = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert concurrent_records is not None
    assert result.capture == concurrent_records.capture
    assert result.forecast == concurrent_records.forecast
    assert result.link == concurrent_records.link
    assert result.capture_replayed is True
    assert result.forecast_replayed is True
    assert result.link_replayed is True
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 3


def test_exact_projected_retry_replays_every_record_and_original_lineage(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    first = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    replay = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert replay.capture == first.capture
    assert replay.forecast == first.forecast
    assert replay.link == first.link
    assert replay.capture.supersedes_capture_id is None
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True
    assert replay.managed_bundle.replayed is True
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 3


def test_exact_confirmed_replay_returns_frozen_records_without_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError("exact replay must not run forecast")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    replay = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert replay.capture == first.capture
    assert replay.forecast == first.forecast
    assert replay.link == first.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True
    assert replay.managed_bundle.replayed is True


@pytest.mark.parametrize(
    "legacy_change",
    ("note", "candidate", "binding"),
)
def test_changed_legacy_appends_durable_link_correction_without_new_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_change: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    base = _pregame_bundle("confirmed", _confirmed_game())
    legacy = _ordered_legacy_decision()
    bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    if legacy_change == "note":
        corrected_legacy = replace(legacy, notes=("corrected operator note",))
    elif legacy_change == "candidate":
        corrected_legacy = replace(
            legacy,
            candidates=(
                replace(legacy.candidates[0], score=77),
                *legacy.candidates[1:],
            ),
        )
    else:
        first_binding = legacy.review_bindings[0]
        conversion_path = first_binding.principal_paths[0]
        corrected_legacy = replace(
            legacy,
            review_bindings=(
                replace(
                    first_binding,
                    principal_paths=(
                        replace(
                            conversion_path,
                            component_ids=(
                                "home_full_runs",
                                "home_full_traffic",
                                "home_full_extra_base_hits",
                            ),
                        ),
                        *first_binding.principal_paths[1:],
                    ),
                ),
                *legacy.review_bindings[1:],
            ),
        )
    corrected_bundle = bundle.model_copy(
        update={
            "payload": bundle.payload.model_copy(
                update={"legacy_decision": corrected_legacy}
            )
        }
    )

    corrected = run_confirmed(
        corrected_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert corrected.capture == first.capture
    assert corrected.forecast == first.forecast
    assert corrected.link.envelope.record_id != first.link.envelope.record_id
    assert (
        corrected.link.supersedes_link_record_id
        == first.link.envelope.record_id
    )
    assert corrected.link.shadow_comparison is not None
    assert corrected.link.shadow_comparison["legacy"] == (
        canonical_legacy_decision_payload(corrected_legacy)
    )
    assert corrected.capture_replayed is True
    assert corrected.forecast_replayed is True
    assert corrected.link_replayed is False
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 4

    def fail_forecast(*args, **kwargs):
        raise AssertionError("corrected exact replay must not run forecast")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)
    replay = run_confirmed(
        corrected_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert replay.capture == corrected.capture
    assert replay.forecast == corrected.forecast
    assert replay.link == corrected.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 4


def test_hash_valid_distribution_market_mismatch_replay_calls_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    outputs = json.loads(canonical_json(first.forecast.outputs))
    distributions = expand_component_distributions(
        outputs["component_distributions"]
    )
    assert isinstance(distributions, dict)
    away_full = distributions["away_full_runs"]
    home_full = distributions["home_full_runs"]
    away_values = list(away_full["values"])
    home_values = list(home_full["values"])
    changed_index = next(
        index
        for index, (away_runs, home_runs) in enumerate(
            zip(away_values, home_values, strict=True)
        )
        if away_runs < home_runs
    )
    away_values[changed_index] = home_values[changed_index] + 1
    summary = json.loads(canonical_json(_summarize(tuple(away_values))))
    away_full["values"] = away_values
    away_full["summary"] = summary
    outputs["intervals"]["away_full_runs"] = summary
    outputs["component_distributions"] = distributions
    outputs = compact_forecast_outputs(outputs)
    _rewrite_forecast_outputs(ledger_path, first.forecast, outputs)

    class ForecastCalled(RuntimeError):
        pass

    def fail_forecast(*args, **kwargs):
        raise ForecastCalled("distribution-market mismatch")

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(ForecastCalled, match="distribution-market mismatch"):
        run_confirmed(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "run_id",
        "simulation_config",
        "simulation_count",
        "market_input",
        "market_evaluation",
        "incomplete",
    ),
)
def test_inconsistent_frozen_outputs_replay_calls_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    outputs = json.loads(canonical_json(first.forecast.outputs))

    if corruption == "run_id":
        outputs["forecast_run_id"] = "0" * 64
    elif corruption == "simulation_config":
        outputs["simulation_config"]["league_woba"] += 0.001
    elif corruption == "simulation_count":
        outputs["simulation_count"] += 1
    elif corruption == "market_input":
        outputs["market_inputs"]["sea_ml"]["selected_decimal_price"] += 0.01
    elif corruption == "market_evaluation":
        outputs["markets"]["sea_ml"]["edge"] += 0.01
    else:
        outputs.pop("intervals")

    _rewrite_forecast_outputs(ledger_path, first.forecast, outputs)

    class ForecastCalled(RuntimeError):
        pass

    def fail_forecast(*args, **kwargs):
        raise ForecastCalled(corruption)

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(ForecastCalled, match=corruption):
        run_confirmed(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "offense_null",
        "offense_unknown_field",
        "pitcher_role",
        "pitcher_boolean",
        "bullpen_team",
        "bullpen_segment_role",
    ),
)
def test_malformed_component_diagnostics_replay_calls_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    outputs = json.loads(canonical_json(first.forecast.outputs))
    diagnostics = outputs["component_diagnostics"]

    if corruption == "offense_null":
        diagnostics["away_offense"] = None
    elif corruption == "offense_unknown_field":
        diagnostics["home_offense"]["unknown_field"] = "not canonical"
    elif corruption == "pitcher_role":
        diagnostics["away_pitcher_chain"]["pitchers"][0]["role"] = "closer"
    elif corruption == "pitcher_boolean":
        diagnostics["home_pitcher_chain"]["pitchers"][0][
            "command_risk"
        ] = "false"
    elif corruption == "bullpen_team":
        diagnostics["away_bullpen"]["team"] = _confirmed_game().identity.home_team
    else:
        diagnostics["home_bullpen"]["segments"][0]["role"] = "closer"

    _rewrite_forecast_outputs(ledger_path, first.forecast, outputs)

    class ForecastCalled(RuntimeError):
        pass

    def fail_forecast(*args, **kwargs):
        raise ForecastCalled(corruption)

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(ForecastCalled, match=corruption):
        run_confirmed(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    "corruption",
    ("offense_numeric_overflow", "bullpen_numeric_overflow"),
)
def test_overflowing_component_diagnostics_replay_calls_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    outputs = json.loads(canonical_json(first.forecast.outputs))
    diagnostics = outputs["component_diagnostics"]

    if corruption == "offense_numeric_overflow":
        diagnostics["away_offense"]["_context"][
            "normalized_pa_weights"
        ] = [1e308] * 9
    else:
        diagnostics["home_bullpen"]["segments"][0][
            "base_log_standard_error"
        ] = 1e308

    _rewrite_forecast_outputs(ledger_path, first.forecast, outputs)

    class ForecastCalled(RuntimeError):
        pass

    def fail_forecast(*args, **kwargs):
        raise ForecastCalled(corruption)

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(ForecastCalled, match=corruption):
        run_confirmed(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_reordered_equivalent_legacy_replay_skips_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    base = _pregame_bundle("confirmed", _confirmed_game())
    legacy = _ordered_legacy_decision()
    bundle = base.model_copy(
        update={
            "payload": base.payload.model_copy(
                update={"legacy_decision": legacy}
            )
        }
    )
    first = run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    reordered_legacy = replace(
        legacy,
        review_bindings=tuple(
            replace(
                binding,
                principal_paths=tuple(
                    reversed(binding.principal_paths)
                ),
            )
            for binding in reversed(legacy.review_bindings)
        ),
    )
    reordered_bundle = bundle.model_copy(
        update={
            "payload": bundle.payload.model_copy(
                update={"legacy_decision": reordered_legacy}
            )
        }
    )

    def fail_forecast(*args, **kwargs):
        raise AssertionError(
            "canonical-equivalent legacy replay must not run forecast"
        )

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    replay = run_confirmed(
        reordered_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert replay.capture == first.capture
    assert replay.forecast == first.forecast
    assert replay.link == first.link
    assert replay.capture_replayed is True
    assert replay.forecast_replayed is True
    assert replay.link_replayed is True


@pytest.mark.parametrize(
    "change",
    (
        "missing_link",
        "lineup",
        "provenance",
        "market",
        "config",
        "legacy",
    ),
)
def test_uncertain_or_changed_confirmed_replay_calls_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("confirmed", _confirmed_game())
    run_confirmed(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    replay_bundle = bundle

    if change == "missing_link":
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        retained = tuple(
            line
            for line in lines
            if json.loads(line)["envelope"]["record_kind"]
            != "capture_forecast_link"
        )
        ledger_path.write_text(
            "\n".join(retained) + "\n",
            encoding="utf-8",
        )
    elif change == "config":
        monkeypatch.setattr(
            service_module,
            "load_forecast_config",
            lambda version, *, registry_root=None: SimulationConfig(
                version=version,
                simulations=13,
            ),
        )
    elif change == "legacy":
        changed_legacy = replace(
            bundle.payload.legacy_decision,
            notes=("operator correction",),
        )
        replay_bundle = bundle.model_copy(
            update={
                "payload": bundle.payload.model_copy(
                    update={"legacy_decision": changed_legacy}
                )
            }
        )
    else:
        game = bundle.payload.game_input
        if change == "lineup":
            changed_lineup = replace(
                game.away_lineup,
                recent_form_index=replace(
                    game.away_lineup.recent_form_index,
                    value=1.1,
                ),
            )
            changed_game = replace(game, away_lineup=changed_lineup)
        elif change == "provenance":
            changed_observation = replace(
                game.observations[0],
                fetched_at=game.observations[0].fetched_at
                + timedelta(seconds=1),
            )
            changed_game = replace(
                game,
                observations=(
                    changed_observation,
                    *game.observations[1:],
                ),
            )
        else:
            changed_market = replace(
                game.markets[0],
                selected_decimal_price=(
                    game.markets[0].selected_decimal_price or 1.0
                )
                + 0.01,
            )
            changed_game = replace(
                game,
                markets=(changed_market, *game.markets[1:]),
            )
        replay_bundle = bundle.model_copy(
            update={
                "payload": bundle.payload.model_copy(
                    update={"game_input": changed_game}
                )
            }
        )

    class ForecastCalled(RuntimeError):
        pass

    def fail_forecast(*args, **kwargs):
        raise ForecastCalled(change)

    monkeypatch.setattr(service_module, "run_forecast", fail_forecast)

    with pytest.raises(ForecastCalled, match=change):
        run_confirmed(
            replay_bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_pregame_artifact_failure_reports_committed_progress_and_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())
    original_persist = service_module.persist_managed_bundle
    interrupted = False

    def interrupt_once(bundle, *, root):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OperatorIOError("managed bundle persistence interrupted")
        return original_persist(bundle, root=root)

    monkeypatch.setattr(
        service_module,
        "persist_managed_bundle",
        interrupt_once,
    )

    with pytest.raises(OperatorIOError) as raised:
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    records = tuple(LedgerStore(ledger_path).iter_records())
    capture = next(
        record
        for record in records
        if record.envelope.record_kind == "capture"
    )
    forecast = next(
        record
        for record in records
        if record.envelope.record_kind == "forecast"
    )
    link = next(
        record
        for record in records
        if record.envelope.record_kind == "capture_forecast_link"
    )
    progress = raised.value.partial_progress
    assert progress.stage == "projected"
    assert progress.game_pk == bundle.game_pk
    assert progress.capture_id == capture.capture_id
    assert progress.forecast_run_id == forecast.forecast_run_id
    assert progress.capture_forecast_link_record_id == link.envelope.record_id
    assert progress.capture_replayed is False
    assert progress.forecast_replayed is False
    assert progress.link_replayed is False
    assert progress.rerun_safe is True
    with pytest.raises(FrozenInstanceError):
        progress.capture_id = "changed"

    completed = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert completed.capture_replayed is True
    assert completed.forecast_replayed is True
    assert completed.link_replayed is True
    assert len(tuple(LedgerStore(ledger_path).iter_records())) == 3


def test_recovery_reread_fault_preserves_original_operator_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def interrupt_and_break_recovery(bundle, *, root):
        del bundle, root

        def fail_reread(self, kind=None):
            del self, kind
            raise RuntimeError("recovery-api-token-secret")

        monkeypatch.setattr(LedgerStore, "iter_records", fail_reread)
        raise OperatorIOError("managed bundle persistence interrupted")

    monkeypatch.setattr(
        service_module,
        "persist_managed_bundle",
        interrupt_and_break_recovery,
    )

    with pytest.raises(
        OperatorIOError,
        match="managed bundle persistence interrupted",
    ) as raised:
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    progress = raised.value.partial_progress
    assert raised.value.exit_code == 6
    assert progress.stage == "projected"
    assert progress.game_pk == bundle.game_pk
    assert progress.capture_id is None
    assert progress.rerun_safe is False


@pytest.mark.parametrize(
    "error_type",
    (
        OperatorLedgerError,
        OperatorPreconditionError,
        BundleValidationError,
    ),
)
def test_non_io_error_after_pregame_commit_is_not_claimed_rerun_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OperatorError],
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())
    original_record = service_module.record_shadow_run

    def commit_then_fail(*args, **kwargs):
        original_record(*args, **kwargs)
        raise error_type("post-commit controlled failure")

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        commit_then_fail,
    )

    with pytest.raises(error_type) as raised:
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    progress = raised.value.partial_progress
    assert progress.capture_id is not None
    assert progress.forecast_run_id is not None
    assert progress.capture_forecast_link_record_id is not None
    assert progress.rerun_safe is False


def test_partial_progress_never_exposes_managed_bundle_fields(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())
    bundle_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (bundle_root / str(bundle.game_pk)).symlink_to(
        outside,
        target_is_directory=True,
    )
    digest = bundle_sha256(bundle)
    unsafe_target = (
        outside / f"projected-v1-{digest[:16]}.json"
    )
    unsafe_target.write_bytes(bundle_canonical_bytes(bundle))

    with pytest.raises(OperatorIOError, match="symbolic link") as raised:
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    assert raised.value.partial_progress.capture_id is not None
    assert not hasattr(raised.value.partial_progress, "managed_bundle_path")
    assert not hasattr(raised.value.partial_progress, "managed_bundle_replayed")


def test_later_capture_reuses_forecast_but_supersedes_projected_capture(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    first_game = _projected_game()
    later_game = replace(
        first_game,
        captured_at=first_game.captured_at + timedelta(minutes=5),
    )

    first = run_projected(
        _pregame_bundle("projected", first_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    later_bundle = _pregame_bundle("projected", later_game)
    later = run_projected(
        later_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    replay = run_projected(
        later_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert later.capture.capture_id != first.capture.capture_id
    assert later.capture.supersedes_capture_id == first.capture.capture_id
    assert later.forecast == first.forecast
    assert later.capture_replayed is False
    assert later.forecast_replayed is True
    assert later.link_replayed is False
    assert replay.capture == later.capture
    assert replay.capture.supersedes_capture_id == first.capture.capture_id


@pytest.mark.parametrize(
    ("stage", "runner", "game_factory"),
    (
        ("projected", run_projected, _projected_game),
        ("confirmed", run_confirmed, _confirmed_game),
    ),
)
def test_new_earlier_capture_cannot_replace_latest_but_history_replays(
    tmp_path: Path,
    stage: str,
    runner,
    game_factory,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    first_game = game_factory()
    first = runner(
        _pregame_bundle(stage, first_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    historical_game = replace(
        first_game,
        captured_at=first_game.captured_at + timedelta(minutes=2),
    )
    historical_bundle = _pregame_bundle(stage, historical_game)
    historical = runner(
        historical_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    latest_game = replace(
        historical_game,
        captured_at=first_game.captured_at + timedelta(minutes=5),
    )
    latest = runner(
        _pregame_bundle(stage, latest_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    changed_lineup = replace(
        historical_game.away_lineup,
        recent_form_index=replace(
            historical_game.away_lineup.recent_form_index,
            value=1.1,
        ),
    )
    stale_bundle = _pregame_bundle(
        stage,
        replace(historical_game, away_lineup=changed_lineup),
    )

    with pytest.raises(OperatorPreconditionError, match="earlier|chronological"):
        runner(
            stale_bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    replay = runner(
        historical_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    captures = tuple(LedgerStore(ledger_path).iter_records("capture"))
    assert replay.capture == historical.capture
    assert replay.capture_replayed is True
    assert historical.capture.supersedes_capture_id == first.capture.capture_id
    assert latest.capture.supersedes_capture_id == historical.capture.capture_id
    assert len(captures) == 3


@pytest.mark.parametrize(
    ("stage", "runner", "game_factory"),
    (
        ("projected", run_projected, _projected_game),
        ("confirmed", run_confirmed, _confirmed_game),
    ),
)
def test_equal_capture_time_with_changed_semantics_becomes_latest(
    tmp_path: Path,
    stage: str,
    runner,
    game_factory,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    game = game_factory()
    first = runner(
        _pregame_bundle(stage, game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    changed_lineup = replace(
        game.away_lineup,
        recent_form_index=replace(
            game.away_lineup.recent_form_index,
            value=1.1,
        ),
    )

    changed = runner(
        _pregame_bundle(stage, replace(game, away_lineup=changed_lineup)),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert changed.capture.captured_at == first.capture.captured_at
    assert changed.capture.capture_id != first.capture.capture_id
    assert changed.capture.supersedes_capture_id == first.capture.capture_id


def test_changed_audit_provenance_creates_capture_and_reuses_forecast(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    first_game = _projected_game()
    changed_observation = replace(
        first_game.observations[0],
        fetched_at=first_game.observations[0].fetched_at + timedelta(seconds=1),
    )
    audited_game = replace(first_game, observations=(changed_observation,))

    first = run_projected(
        _pregame_bundle("projected", first_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    audited = run_projected(
        _pregame_bundle("projected", audited_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert audited.capture.capture_id != first.capture.capture_id
    assert audited.capture.supersedes_capture_id == first.capture.capture_id
    assert audited.forecast == first.forecast


def test_updated_projected_semantics_supersede_latest_active_projected(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    first_game = _projected_game()
    updated_lineup = replace(
        first_game.away_lineup,
        recent_form_index=replace(
            first_game.away_lineup.recent_form_index,
            value=1.1,
        ),
    )
    updated_game = replace(
        first_game,
        captured_at=first_game.captured_at + timedelta(minutes=5),
        away_lineup=updated_lineup,
    )

    first = run_projected(
        _pregame_bundle("projected", first_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    updated = run_projected(
        _pregame_bundle("projected", updated_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert updated.capture.supersedes_capture_id == first.capture.capture_id
    assert updated.forecast.forecast_run_id != first.forecast.forecast_run_id


def test_confirmed_supersedes_latest_active_pregame_and_retries_exactly(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    projected = run_projected(
        _pregame_bundle("projected", _projected_game()),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    confirmed_bundle = _pregame_bundle("confirmed", _confirmed_game())

    confirmed = run_confirmed(
        confirmed_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    retry = run_confirmed(
        confirmed_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert confirmed.capture.lineup_state == "confirmed"
    assert confirmed.capture.supersedes_capture_id == projected.capture.capture_id
    assert retry.capture == confirmed.capture
    assert retry.capture.supersedes_capture_id == projected.capture.capture_id
    assert retry.capture_replayed is True
    assert retry.forecast_replayed is True
    assert retry.link_replayed is True


def test_new_projected_after_confirmation_is_rejected_but_history_replays(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    old_bundle = _pregame_bundle("projected", _projected_game())
    old = run_projected(
        old_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    run_confirmed(
        _pregame_bundle("confirmed", _confirmed_game()),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    changed_game = replace(
        _projected_game(),
        captured_at=_projected_game().captured_at + timedelta(minutes=10),
    )

    with pytest.raises(OperatorPreconditionError, match="active confirmed"):
        run_projected(
            _pregame_bundle("projected", changed_game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    replay = run_projected(
        old_bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )
    assert replay.capture == old.capture
    assert replay.capture_replayed is True


@pytest.mark.parametrize(
    ("runner", "stage", "game"),
    (
        (run_projected, "projected", _confirmed_game()),
        (run_confirmed, "confirmed", _projected_game()),
    ),
)
def test_stage_and_lineup_state_mismatches_are_preconditions(
    tmp_path: Path,
    runner,
    stage: str,
    game,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)

    with pytest.raises(OperatorPreconditionError, match="lineup"):
        runner(
            _pregame_bundle(stage, game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_calling_wrong_stage_service_rejects_non_exact_bundle_type(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    confirmed = _pregame_bundle("confirmed", _confirmed_game())

    with pytest.raises(
        TypeError,
        match="normal pregame runner requires a strict current bundle",
    ):
        run_projected(
            confirmed,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    "runner,stage",
    ((run_projected, "projected"), (run_confirmed, "confirmed")),
)
def test_post_first_pitch_capture_is_a_precondition(
    tmp_path: Path,
    runner,
    stage: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    base = _projected_game() if stage == "projected" else _confirmed_game()
    game = replace(base, captured_at=base.identity.first_pitch)

    with pytest.raises(
        OperatorPreconditionError,
        match="^invalid pregame timing$",
    ):
        runner(
            _pregame_bundle(stage, game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_relationship_failure_maps_to_bundle_validation(tmp_path: Path) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    base = _projected_game()
    secret = "https://attacker.example/?api_token=pregame-secret"
    game = replace(
        base,
        away_lineup=replace(
            base.away_lineup,
            state_provenance=replace(
                base.away_lineup.state_provenance,
                observation_ids=(secret,),
            ),
        ),
    )

    with pytest.raises(BundleValidationError) as raised:
        run_projected(
            _pregame_bundle("projected", game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    assert str(raised.value) == "invalid pregame relationship"
    assert secret not in str(raised.value)


def test_active_capture_selection_uses_latest_capture_order(tmp_path: Path) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    store = LedgerStore(ledger_path)
    first_game = _projected_game()
    later_game = replace(
        first_game,
        captured_at=first_game.captured_at + timedelta(minutes=2),
    )
    first = store.append_capture(
        "manual:first",
        build_shadow_capture_command(first_game),
    )
    later = store.append_capture(
        "manual:later",
        build_shadow_capture_command(later_game),
    )
    newest_game = replace(
        first_game,
        captured_at=first_game.captured_at + timedelta(minutes=4),
    )

    result = run_projected(
        _pregame_bundle("projected", newest_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert result.capture.supersedes_capture_id == later.capture_id
    assert result.capture.supersedes_capture_id != first.capture_id


def test_projected_rejects_legacy_multi_root_with_any_active_confirmed_capture(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    store = LedgerStore(ledger_path)
    game = _projected_game()
    confirmed_game = _confirmed_game(
        captured_at=game.captured_at,
    )
    projected_game = replace(
        game,
        captured_at=game.captured_at + timedelta(minutes=1),
    )
    incoming_game = replace(
        game,
        captured_at=game.captured_at + timedelta(minutes=2),
    )
    confirmed = store.append_capture(
        "manual:legacy-confirmed-root",
        build_shadow_capture_command(confirmed_game),
    )
    latest_projected = store.append_capture(
        "manual:legacy-projected-root",
        build_shadow_capture_command(projected_game),
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="projected capture.*confirmed",
    ):
        run_projected(
            _pregame_bundle("projected", incoming_game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    captures = tuple(LedgerStore(ledger_path).iter_records("capture"))
    assert latest_projected.captured_at > confirmed.captured_at
    assert captures == (confirmed, latest_projected)


def test_postgame_successor_deactivates_pregame_supersession_target(
    tmp_path: Path,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    store = LedgerStore(ledger_path)
    game = _projected_game()
    pregame = store.append_capture(
        "manual:pregame",
        build_shadow_capture_command(game),
    )
    postgame_command = replace(
        build_shadow_capture_command(game),
        captured_at=game.identity.first_pitch + timedelta(seconds=1),
        supersedes_capture_id=pregame.capture_id,
    )
    store.append_capture("manual:postgame", postgame_command)
    updated_game = replace(
        game,
        captured_at=game.captured_at + timedelta(minutes=5),
    )

    result = run_projected(
        _pregame_bundle("projected", updated_game),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    assert result.capture.supersedes_capture_id is None


def test_projected_run_retries_when_head_changes_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    game = _projected_game()
    bundle = _pregame_bundle(
        "projected",
        replace(game, captured_at=game.captured_at + timedelta(minutes=1)),
    )
    original_record = service_module.record_shadow_run
    attempts = 0
    interloper_capture_id = None

    def change_head_then_record(store, game_input, *args, **kwargs):
        nonlocal attempts, interloper_capture_id
        attempts += 1
        if attempts == 1:
            interloper = store.append_capture(
                "manual:interloper",
                build_shadow_capture_command(game_input),
            )
            interloper_capture_id = interloper.capture_id
        return original_record(store, game_input, *args, **kwargs)

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        change_head_then_record,
    )

    result = run_projected(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
    )

    captures = tuple(LedgerStore(ledger_path).iter_records("capture"))
    superseded_ids = frozenset(
        capture.supersedes_capture_id
        for capture in captures
        if capture.supersedes_capture_id is not None
    )
    active = tuple(
        capture
        for capture in captures
        if capture.capture_id not in superseded_ids
    )
    assert attempts == 2
    assert len(captures) == 2
    assert len(active) == 1
    assert result.capture.supersedes_capture_id == interloper_capture_id
    assert sum(capture.supersedes_capture_id is None for capture in captures) == 1


def test_projected_rejects_concurrent_confirmed_root_when_head_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    game = _projected_game()
    projected_root_game = replace(
        game,
        captured_at=game.captured_at + timedelta(minutes=2),
    )
    incoming_game = replace(
        game,
        captured_at=game.captured_at + timedelta(minutes=3),
    )
    store = LedgerStore(ledger_path)
    projected_root = store.append_capture(
        "manual:projected-root",
        build_shadow_capture_command(projected_root_game),
    )
    original_record = service_module.record_shadow_run
    attempts = 0

    def add_confirmed_root_then_record(store, game_input, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            confirmed_game = _confirmed_game(
                captured_at=game.captured_at + timedelta(minutes=1),
            )
            store.append_capture(
                "manual:concurrent-confirmed-root",
                build_shadow_capture_command(confirmed_game),
            )
        return original_record(store, game_input, *args, **kwargs)

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        add_confirmed_root_then_record,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="projected capture.*confirmed",
    ):
        run_projected(
            _pregame_bundle("projected", incoming_game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    captures = tuple(LedgerStore(ledger_path).iter_records("capture"))
    assert attempts == 1
    assert len(captures) == 2
    assert max(captures, key=lambda capture: capture.captured_at) == projected_root


def test_projected_run_rejects_older_capture_after_concurrent_newer_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    game = _projected_game()
    original_record = service_module.record_shadow_run
    attempts = 0

    def change_head_then_record(store, game_input, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            concurrent_game = replace(
                game_input,
                captured_at=game_input.captured_at + timedelta(minutes=1),
            )
            store.append_capture(
                "manual:newer-interloper",
                build_shadow_capture_command(concurrent_game),
            )
        return original_record(store, game_input, *args, **kwargs)

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        change_head_then_record,
    )

    with pytest.raises(OperatorPreconditionError, match="earlier|chronological"):
        run_projected(
            _pregame_bundle("projected", game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    captures = tuple(LedgerStore(ledger_path).iter_records("capture"))
    assert attempts == 1
    assert len(captures) == 1
    assert captures[0].captured_at > game.captured_at


def test_projected_run_maps_repeated_head_changes_to_operator_ledger_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    attempts = 0

    def fail_with_head_change(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise LedgerHeadChangedError("active capture head changed")

    monkeypatch.setattr(
        service_module,
        "record_shadow_run",
        fail_with_head_change,
    )

    with pytest.raises(OperatorLedgerError, match="head changed"):
        run_projected(
            _pregame_bundle("projected", _projected_game()),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    assert attempts == 3


@pytest.mark.parametrize(
    "ledger_error",
    (
        IdempotencyConflict("conflict"),
        LedgerReferenceError("missing"),
        LedgerCorruptionError("corrupt"),
    ),
)
def test_ledger_failures_are_mapped_at_service_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_error: ValueError,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def fail_record(*args, **kwargs):
        raise ledger_error

    monkeypatch.setattr("analysis.operator.service.record_shadow_run", fail_record)

    with pytest.raises(OperatorLedgerError, match=str(ledger_error)):
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_ledger_filesystem_error_maps_to_operator_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def fail_open(path):
        raise OSError("ledger filesystem unavailable")

    monkeypatch.setattr(service_module, "LedgerStore", fail_open)

    with pytest.raises(OperatorIOError, match="filesystem unavailable"):
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_ledger_append_filesystem_error_maps_to_operator_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def fail_record(*args, **kwargs):
        raise OSError("ledger append failed")

    monkeypatch.setattr(service_module, "record_shadow_run", fail_record)

    with pytest.raises(OperatorIOError, match="append failed"):
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    ("pregame_error", "expected_error", "safe_message"),
    (
        (
            PregameRelationshipError(
                "missing observation api_token=relationship-secret"
            ),
            BundleValidationError,
            "invalid pregame relationship",
        ),
        (
            PregameTimingError("market_key=timing-secret arrived late"),
            OperatorPreconditionError,
            "invalid pregame timing",
        ),
    ),
)
def test_typed_pregame_failures_from_recording_keep_stable_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pregame_error: ValueError,
    expected_error: type[Exception],
    safe_message: str,
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def fail_record(*args, **kwargs):
        raise pregame_error

    monkeypatch.setattr("analysis.operator.service.record_shadow_run", fail_record)

    with pytest.raises(expected_error) as raised:
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )

    assert str(raised.value) == safe_message
    assert str(pregame_error) not in str(raised.value)
