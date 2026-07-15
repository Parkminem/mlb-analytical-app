from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import analysis.operator.service as service_module
from analysis.forecasting import LineupState, run_forecast
from analysis.forecasting.config import SimulationConfig
from analysis.ledger import (
    CaptureCommand,
    ForecastCommand,
    LedgerCorruptionError,
    LedgerStore,
    MarketObservationCommand,
    MarketObservationRecord,
    OfficialResultCommand,
    OfficialResultRecord,
    ReviewRecord,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    ReviewBundle,
    parse_review_bundle_json,
)
from analysis.operator.service import (
    ReviewRunResult,
    run_review,
)
from analysis.review import (
    ContradictionRule,
    PrincipalPathBinding,
    build_and_record_review as real_build_and_record_review,
)
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    ShadowRunIdempotencyKeys,
    record_shadow_run,
)
from tests.test_forecasting_engine import _complete_game
from tests.test_operator_schemas import _raw, review_bundle_mapping


UTC = timezone.utc


def _principal_paths() -> tuple[PrincipalPathBinding, ...]:
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


def _legacy_snapshot(*, include_pass: bool = True) -> LegacyDecisionSnapshot:
    paths = _principal_paths()
    candidates = [
        PriorityMarketCandidate(
            market="ml",
            score=78,
            decision_rating="play",
            evidence_paths=frozenset(path.evidence_path for path in paths),
        )
    ]
    bindings = [
        CandidateReviewBinding(
            market_key="sea_ml",
            legacy_candidate_index=0,
            legacy_market="ml",
            principal_paths=paths,
        )
    ]
    if include_pass:
        candidates.append(
            PriorityMarketCandidate(
                market="full_total",
                score=60,
                decision_rating="pass",
                evidence_paths=frozenset(path.evidence_path for path in paths),
            )
        )
        bindings.append(
            CandidateReviewBinding(
                market_key="full_under",
                legacy_candidate_index=1,
                legacy_market="full_total",
                principal_paths=paths,
            )
        )
    return LegacyDecisionSnapshot(
        model_version="legacy-fixed-v1",
        candidates=tuple(candidates),
        final_priority=("ml",),
        review_bindings=tuple(bindings),
        notes=("legacy remains authoritative",),
    )


def _record_pregame(game, legacy: LegacyDecisionSnapshot, ledger_path: Path):
    config = SimulationConfig(
        version="shadow-forecast-2026-07-14.1",
        simulations=12,
    )
    forecast = run_forecast(game, config)
    return record_shadow_run(
        LedgerStore(ledger_path),
        game,
        legacy,
        forecast,
        ShadowRunIdempotencyKeys(capture=f"capture:{game.captured_at.isoformat()}"),
    )


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ReviewContext:
    ledger_path: Path
    bundle_root: Path
    confirmed: object
    bundle: ReviewBundle


def _review_mapping(confirmed, *, with_close: bool = True) -> dict[str, object]:
    capture = confirmed.capture
    forecast = confirmed.forecast
    link = confirmed.link
    outputs = forecast.outputs
    summaries = outputs["component_distributions"]
    component_values = {
        "home_pitcher_chain_runs_allowed": 4.0,
        "away_full_runs": 4.0,
        "away_full_traffic": float(
            summaries["away_full_traffic"]["summary"]["median"]
        ),
        "away_full_extra_base_hits": float(
            summaries["away_full_extra_base_hits"]["summary"]["median"]
        ),
    }
    mapping = review_bundle_mapping()
    mapping["game_pk"] = capture.game_pk
    mapping["created_at"] = _iso(capture.first_pitch + timedelta(hours=5))
    payload = mapping["payload"]
    payload["capture_id"] = capture.capture_id
    payload["forecast_run_id"] = forecast.forecast_run_id
    payload["capture_forecast_link_record_id"] = link.envelope.record_id
    payload["official_result"] = {
        "game_pk": capture.game_pk,
        "source_name": "MLB",
        "source_record": f"game/{capture.game_pk}/boxscore",
        "source_revision": "final-1",
        "finalized_at": _iso(capture.first_pitch + timedelta(hours=4)),
        "away_f5_runs": 2,
        "home_f5_runs": 1,
        "away_full_runs": 4,
        "home_full_runs": 3,
        "component_values": component_values,
        "documented_events": [],
        "supersedes_record_id": None,
    }
    payload["closing_observations"] = (
        [
            {
                "game_pk": capture.game_pk,
                "source_name": "ExampleBook",
                "source_record": f"quote/{capture.game_pk}/sea_ml",
                "source_revision": "close-1",
                "observation_kind": "close",
                "market_key": "sea_ml",
                "market_type": "ml",
                "settlement_format": "two_way",
                "selected_side": "away",
                "line": None,
                "selected_decimal_price": 1.95,
                "opposing_decimal_price": 1.87,
                "observed_at": _iso(capture.captured_at),
                "available_at": _iso(capture.first_pitch),
                "supersedes_record_id": None,
            }
        ]
        if with_close
        else []
    )
    return mapping


def _replace_review_commands(
    bundle: ReviewBundle,
    *,
    official_result: OfficialResultCommand | None = None,
    closing_observations: tuple[MarketObservationCommand, ...] | None = None,
) -> ReviewBundle:
    payload = bundle.payload.model_copy(
        update={
            "official_result": (
                bundle.payload.official_result
                if official_result is None
                else official_result
            ),
            "closing_observations": (
                bundle.payload.closing_observations
                if closing_observations is None
                else closing_observations
            ),
        }
    )
    return bundle.model_copy(update={"payload": payload})


def _correction_bundle(
    context: ReviewContext,
    first_result: OfficialResultRecord,
    first_close: MarketObservationRecord,
) -> ReviewBundle:
    result = replace(
        context.bundle.payload.official_result,
        source_revision="final-2",
        finalized_at=(
            context.bundle.payload.official_result.finalized_at
            + timedelta(minutes=10)
        ),
        supersedes_record_id=first_result.envelope.record_id,
    )
    close = replace(
        context.bundle.payload.closing_observations[0],
        source_revision="close-2",
        selected_decimal_price=2.01,
        opposing_decimal_price=1.82,
        supersedes_record_id=first_close.envelope.record_id,
    )
    return _replace_review_commands(
        context.bundle,
        official_result=result,
        closing_observations=(close,),
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


@pytest.fixture
def review_context(tmp_path: Path) -> ReviewContext:
    ledger_path = tmp_path / "ledger" / "shadow.jsonl"
    bundle_root = tmp_path / "bundles"
    game = _complete_game()
    confirmed = _record_pregame(
        game,
        _legacy_snapshot(),
        ledger_path,
    )
    bundle = parse_review_bundle_json(_raw(_review_mapping(confirmed)))
    return ReviewContext(ledger_path, bundle_root, confirmed, bundle)


def _run(context: ReviewContext, bundle: ReviewBundle | None = None):
    return run_review(
        context.bundle if bundle is None else bundle,
        ledger_path=context.ledger_path,
        bundle_root=context.bundle_root,
    )


def _records(context: ReviewContext, kind: str) -> tuple[object, ...]:
    return tuple(LedgerStore(context.ledger_path).iter_records(kind))


def _supersede_confirmed_capture(
    context: ReviewContext,
    store: LedgerStore,
    suffix: str,
):
    capture = context.confirmed.capture
    return store.append_capture(
        f"capture:concurrent:{suffix}",
        CaptureCommand(
            game_pk=capture.game_pk,
            captured_at=capture.captured_at + timedelta(seconds=1),
            first_pitch=capture.first_pitch,
            lineup_state="confirmed",
            out_of_sample_eligible=capture.out_of_sample_eligible,
            normalized_input_hash=capture.normalized_input_hash,
            payload=capture.payload,
            supersedes_capture_id=capture.capture_id,
        ),
        expected_active_capture_id=capture.capture_id,
    )


def test_run_review_records_exact_confirmed_result_closing_and_all_eligible_markets(
    review_context: ReviewContext,
) -> None:
    result = _run(review_context)

    assert isinstance(result, ReviewRunResult)
    assert result.stage == "review"
    assert result.game_pk == review_context.confirmed.capture.game_pk
    assert result.capture == review_context.confirmed.capture
    assert result.forecast == review_context.confirmed.forecast
    assert result.link == review_context.confirmed.link
    assert len(result.closing_observations) == 1
    assert isinstance(result.closing_observations[0], MarketObservationRecord)
    assert isinstance(result.official_result, OfficialResultRecord)
    assert tuple(review.market_key for review in result.reviews) == (
        "full_under",
        "sea_ml",
    )
    assert all(isinstance(review, ReviewRecord) for review in result.reviews)
    pass_review = next(
        review for review in result.reviews if review.market_key == "full_under"
    )
    assert pass_review.classification is None
    assert result.closing_replay_count == 0
    assert result.official_result_replayed is False
    assert result.review_replay_count == 0
    assert sum(result.classification_counts.values()) == 2
    assert result.classification_counts["unclassified"] == 1
    assert result.managed_bundle.path.exists()
    assert result.managed_bundle.path.read_bytes()
    with pytest.raises(TypeError):
        result.classification_counts["A"] = 99
    with pytest.raises(FrozenInstanceError):
        result.game_pk = 1


def test_exact_review_retry_replays_every_record_and_managed_bundle(
    review_context: ReviewContext,
) -> None:
    first = _run(review_context)
    replay = _run(review_context)

    assert replay.closing_observations == first.closing_observations
    assert replay.official_result == first.official_result
    assert replay.reviews == first.reviews
    assert replay.closing_replay_count == 1
    assert replay.official_result_replayed is True
    assert replay.review_replay_count == 2
    assert replay.managed_bundle.replayed is True
    assert len(_records(review_context, "market_observation")) == 1
    assert len(_records(review_context, "official_result")) == 1
    assert len(_records(review_context, "review")) == 2


def test_completed_review_bundle_rejects_a_superseded_result_head(
    review_context: ReviewContext,
) -> None:
    first = _run(review_context)
    correction = replace(
        review_context.bundle.payload.official_result,
        source_revision="final-2",
        finalized_at=(
            review_context.bundle.payload.official_result.finalized_at
            + timedelta(minutes=10)
        ),
        supersedes_record_id=first.official_result.envelope.record_id,
    )
    store = LedgerStore(review_context.ledger_path)
    corrected_result = store.append_official_result(correction)

    with pytest.raises(
        OperatorPreconditionError,
        match="current latest|superseded",
    ):
        _run(review_context)

    assert store.latest_official_result(first.game_pk) == corrected_result
    assert len(_records(review_context, "review")) == len(first.reviews)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("capture_id", "missing-capture", "capture_id"),
        ("forecast_run_id", "missing-forecast", "forecast_run_id"),
        (
            "capture_forecast_link_record_id",
            "missing-link",
            "capture_forecast_link_record_id",
        ),
    ),
)
def test_review_requires_each_exact_frozen_record_id(
    review_context: ReviewContext,
    field_name: str,
    value: str,
    message: str,
) -> None:
    payload = review_context.bundle.payload.model_copy(update={field_name: value})
    bundle = review_context.bundle.model_copy(update={"payload": payload})

    with pytest.raises(OperatorPreconditionError, match=message):
        _run(review_context, bundle)

    assert not _records(review_context, "official_result")


def test_review_rejects_link_that_does_not_match_named_capture(
    review_context: ReviewContext,
) -> None:
    store = LedgerStore(review_context.ledger_path)
    capture = review_context.confirmed.capture
    other = store.append_capture(
        "capture:parallel-confirmed",
        CaptureCommand(
            game_pk=capture.game_pk,
            captured_at=capture.captured_at + timedelta(seconds=1),
            first_pitch=capture.first_pitch,
            lineup_state="confirmed",
            out_of_sample_eligible=capture.out_of_sample_eligible,
            normalized_input_hash=capture.normalized_input_hash,
            payload=capture.payload,
        ),
    )
    other_link = store.link_capture_forecast(
        other.capture_id,
        review_context.confirmed.forecast.forecast_run_id,
    )
    payload = review_context.bundle.payload.model_copy(
        update={
            "capture_forecast_link_record_id": other_link.envelope.record_id,
        }
    )
    bundle = review_context.bundle.model_copy(update={"payload": payload})

    with pytest.raises(OperatorPreconditionError, match="link.*capture"):
        _run(review_context, bundle)


def test_review_rejects_projected_capture(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    bundle_root = tmp_path / "bundles"
    game = _complete_game()
    game = replace(
        game,
        away_lineup=replace(game.away_lineup, state=LineupState.PROJECTED),
    )
    projected = _record_pregame(
        game,
        _legacy_snapshot(include_pass=False),
        ledger_path,
    )
    review = parse_review_bundle_json(
        _raw(_review_mapping(projected, with_close=False))
    )

    with pytest.raises(OperatorPreconditionError, match="confirmed"):
        run_review(review, ledger_path=ledger_path, bundle_root=bundle_root)


def test_review_rejects_superseded_confirmed_capture(
    review_context: ReviewContext,
) -> None:
    capture = review_context.confirmed.capture
    LedgerStore(review_context.ledger_path).append_capture(
        "capture:confirmed:replacement",
        CaptureCommand(
            game_pk=capture.game_pk,
            captured_at=capture.captured_at + timedelta(seconds=1),
            first_pitch=capture.first_pitch,
            lineup_state="confirmed",
            out_of_sample_eligible=capture.out_of_sample_eligible,
            normalized_input_hash=capture.normalized_input_hash,
            payload=capture.payload,
            supersedes_capture_id=capture.capture_id,
        ),
    )

    with pytest.raises(OperatorPreconditionError, match="active"):
        _run(review_context)


def test_review_rejects_capture_at_or_after_first_pitch(
    review_context: ReviewContext,
) -> None:
    capture = review_context.confirmed.capture
    store = LedgerStore(review_context.ledger_path)
    late = store.append_capture(
        "capture:confirmed:late",
        CaptureCommand(
            game_pk=capture.game_pk,
            captured_at=capture.first_pitch,
            first_pitch=capture.first_pitch,
            lineup_state="confirmed",
            out_of_sample_eligible=capture.out_of_sample_eligible,
            normalized_input_hash=capture.normalized_input_hash,
            payload=capture.payload,
            supersedes_capture_id=capture.capture_id,
        ),
    )
    link = store.link_capture_forecast(
        late.capture_id,
        review_context.confirmed.forecast.forecast_run_id,
    )
    payload = review_context.bundle.payload.model_copy(
        update={
            "capture_id": late.capture_id,
            "capture_forecast_link_record_id": link.envelope.record_id,
        }
    )
    bundle = review_context.bundle.model_copy(update={"payload": payload})

    with pytest.raises(OperatorPreconditionError, match="before first_pitch"):
        _run(review_context, bundle)


def test_duplicate_closing_streams_are_rejected_before_any_ledger_append(
    review_context: ReviewContext,
) -> None:
    first = review_context.bundle.payload.closing_observations[0]
    duplicate = replace(
        first,
        source_revision="close-2",
        selected_decimal_price=2.01,
        opposing_decimal_price=1.82,
    )
    bundle = _replace_review_commands(
        review_context.bundle,
        closing_observations=(first, duplicate),
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="duplicate.*closing.*stream",
    ):
        _run(review_context, bundle)

    assert not _records(review_context, "market_observation")
    assert not _records(review_context, "official_result")
    assert not _records(review_context, "review")
    assert not tuple(
        (review_context.bundle_root / str(review_context.bundle.game_pk)).glob(
            "review-*.json"
        )
    )


def test_distinct_closing_streams_in_one_bundle_still_succeed(
    review_context: ReviewContext,
) -> None:
    first = review_context.bundle.payload.closing_observations[0]
    distinct = replace(
        first,
        source_record=f"quote/{review_context.bundle.game_pk}/full_under",
        market_key="full_under",
        market_type="full_total",
        selected_side="under",
        line=8.5,
    )
    bundle = _replace_review_commands(
        review_context.bundle,
        closing_observations=(first, distinct),
    )

    result = _run(review_context, bundle)

    assert result.closing_observations == _records(
        review_context,
        "market_observation",
    )
    assert tuple(close.market_key for close in result.closing_observations) == (
        "sea_ml",
        "full_under",
    )
    assert len(_records(review_context, "official_result")) == 1
    assert len(_records(review_context, "review")) == 2


@pytest.mark.parametrize(
    ("field_name", "offset", "message"),
    (
        ("observed_at", timedelta(microseconds=-1), "observed_at"),
        ("available_at", timedelta(microseconds=-1), "available_at"),
        ("observed_at", timedelta(microseconds=1), "observed_at"),
        ("available_at", timedelta(microseconds=1), "available_at"),
    ),
)
def test_close_timestamps_must_stay_inside_frozen_capture_interval(
    review_context: ReviewContext,
    field_name: str,
    offset: timedelta,
    message: str,
) -> None:
    capture = review_context.confirmed.capture
    close = review_context.bundle.payload.closing_observations[0]
    boundary = (
        capture.captured_at if offset < timedelta(0) else capture.first_pitch
    )
    bad_close = replace(close, **{field_name: boundary + offset})
    bundle = _replace_review_commands(
        review_context.bundle,
        closing_observations=(bad_close,),
    )

    with pytest.raises(OperatorPreconditionError, match=message):
        _run(review_context, bundle)

    assert not _records(review_context, "market_observation")


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"available_at": None}, "available_at"),
        ({"observation_kind": "entry"}, "observation_kind"),
        ({"selected_decimal_price": None}, "selected_decimal_price"),
        ({"opposing_decimal_price": None}, "opposing_decimal_price"),
    ),
)
def test_run_review_rechecks_complete_close_terms_before_append(
    review_context: ReviewContext,
    changes: dict[str, object],
    message: str,
) -> None:
    close = replace(
        review_context.bundle.payload.closing_observations[0],
        **changes,
    )
    bundle = _replace_review_commands(
        review_context.bundle,
        closing_observations=(close,),
    )

    with pytest.raises(OperatorPreconditionError, match=message):
        _run(review_context, bundle)

    assert not _records(review_context, "market_observation")


@pytest.mark.parametrize(
    ("source_name", "finalized_at_factory", "message"),
    (
        ("mlb", None, "source_name"),
        ("MLB", lambda first_pitch: first_pitch, "finalized_at"),
    ),
)
def test_result_source_and_post_pitch_timing_are_checked_before_any_append_or_replay(
    review_context: ReviewContext,
    source_name: str,
    finalized_at_factory,
    message: str,
) -> None:
    first = _run(review_context)
    command = review_context.bundle.payload.official_result
    bad = replace(
        command,
        source_name=source_name,
        finalized_at=(
            command.finalized_at
            if finalized_at_factory is None
            else finalized_at_factory(review_context.confirmed.capture.first_pitch)
        ),
    )
    bundle = _replace_review_commands(review_context.bundle, official_result=bad)

    with pytest.raises(OperatorPreconditionError, match=message):
        _run(review_context, bundle)

    assert _records(review_context, "official_result") == (first.official_result,)
    assert _records(review_context, "review") == first.reviews


def test_close_and_result_corrections_require_explicit_current_latest_and_replay(
    review_context: ReviewContext,
) -> None:
    first = _run(review_context)
    correction = _correction_bundle(
        review_context,
        first.official_result,
        first.closing_observations[0],
    )

    corrected = _run(review_context, correction)
    replay = _run(review_context, correction)

    assert corrected.closing_observations[0].supersedes_record_id == (
        first.closing_observations[0].envelope.record_id
    )
    assert corrected.official_result.supersedes_record_id == (
        first.official_result.envelope.record_id
    )
    assert corrected.closing_replay_count == 0
    assert corrected.official_result_replayed is False
    assert replay.closing_replay_count == 1
    assert replay.official_result_replayed is True
    assert replay.review_replay_count == len(corrected.reviews)
    assert len(_records(review_context, "market_observation")) == 2
    assert len(_records(review_context, "official_result")) == 2


@pytest.mark.parametrize("record_kind", ("close", "result"))
@pytest.mark.parametrize("target_kind", ("missing", "wrong"))
def test_new_corrections_reject_missing_or_wrong_supersession_targets(
    review_context: ReviewContext,
    record_kind: str,
    target_kind: str,
) -> None:
    first = _run(review_context)
    correction = _correction_bundle(
        review_context,
        first.official_result,
        first.closing_observations[0],
    )
    if record_kind == "close":
        command = correction.payload.closing_observations[0]
        target = {
            "missing": None,
            "wrong": first.official_result.envelope.record_id,
        }[target_kind]
        bad = replace(command, supersedes_record_id=target)
        correction = _replace_review_commands(
            correction,
            closing_observations=(bad,),
        )
    else:
        command = correction.payload.official_result
        target = {
            "missing": None,
            "wrong": first.closing_observations[0].envelope.record_id,
        }[target_kind]
        bad = replace(command, supersedes_record_id=target)
        correction = _replace_review_commands(
            correction,
            official_result=bad,
        )

    with pytest.raises(OperatorPreconditionError, match="supersed|target|latest"):
        _run(review_context, correction)


@pytest.mark.parametrize("record_kind", ("close", "result"))
def test_existing_correction_key_rejects_a_self_supersession_target(
    review_context: ReviewContext,
    record_kind: str,
) -> None:
    first = _run(review_context)
    correction = _correction_bundle(
        review_context,
        first.official_result,
        first.closing_observations[0],
    )
    corrected = _run(review_context, correction)
    if record_kind == "close":
        bad = replace(
            correction.payload.closing_observations[0],
            supersedes_record_id=(
                corrected.closing_observations[0].envelope.record_id
            ),
        )
        correction = _replace_review_commands(
            correction,
            closing_observations=(bad,),
        )
    else:
        bad = replace(
            correction.payload.official_result,
            supersedes_record_id=corrected.official_result.envelope.record_id,
        )
        correction = _replace_review_commands(
            correction,
            official_result=bad,
        )

    with pytest.raises(OperatorPreconditionError, match="itself|self"):
        _run(review_context, correction)


def test_existing_key_still_uses_store_hash_validation_for_close_and_result(
    review_context: ReviewContext,
) -> None:
    _run(review_context)
    close = replace(
        review_context.bundle.payload.closing_observations[0],
        selected_decimal_price=2.15,
    )
    changed_close = _replace_review_commands(
        review_context.bundle,
        closing_observations=(close,),
    )
    with pytest.raises(OperatorLedgerError, match="idempotency|payload"):
        _run(review_context, changed_close)

    result = replace(
        review_context.bundle.payload.official_result,
        documented_events=({"result_label": "corrected"},),
    )
    changed_result = _replace_review_commands(
        review_context.bundle,
        official_result=result,
    )
    with pytest.raises(OperatorLedgerError, match="idempotency|payload"):
        _run(review_context, changed_result)


def test_changed_payload_same_result_key_is_not_claimed_rerun_safe(
    review_context: ReviewContext,
) -> None:
    committed = _run(review_context)
    changed_command = replace(
        review_context.bundle.payload.official_result,
        documented_events=({"result_label": "changed-same-key"},),
    )
    changed_bundle = _replace_review_commands(
        review_context.bundle,
        official_result=changed_command,
    )

    messages: list[str] = []
    for _ in range(2):
        with pytest.raises(
            OperatorLedgerError,
            match="idempotency|payload",
        ) as raised:
            _run(review_context, changed_bundle)
        messages.append(str(raised.value))
        progress = raised.value.partial_progress
        assert progress.official_result_record_id == (
            committed.official_result.envelope.record_id
        )
        assert progress.rerun_safe is False

    assert messages[1] == messages[0]
    assert len(_records(review_context, "official_result")) == 1


def test_concurrent_closing_head_change_is_retried_without_forking(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = LedgerStore.append_market_observation
    injected = False

    def append_after_competing_close(
        self,
        command,
        idempotency_key=None,
        **kwargs,
    ):
        nonlocal injected
        if not injected:
            injected = True
            original(
                self,
                replace(
                    command,
                    source_revision="concurrent-close",
                    selected_decimal_price=2.05,
                    opposing_decimal_price=1.79,
                ),
            )
        return original(self, command, idempotency_key, **kwargs)

    monkeypatch.setattr(
        LedgerStore,
        "append_market_observation",
        append_after_competing_close,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="explicitly supersede|current latest",
    ):
        _run(review_context)

    closes = _records(review_context, "market_observation")
    assert len(closes) == 1
    assert closes[0].source_revision == "concurrent-close"
    assert not _records(review_context, "official_result")
    assert not _records(review_context, "review")


def test_capture_supersession_before_result_append_prevents_result_write(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = LedgerStore.append_official_result
    injected = False
    bundle = parse_review_bundle_json(
        _raw(_review_mapping(review_context.confirmed, with_close=False))
    )

    def append_after_capture_supersession(
        self,
        command,
        idempotency_key=None,
        **kwargs,
    ):
        nonlocal injected
        if not injected:
            injected = True
            _supersede_confirmed_capture(review_context, self, "before-result")
        return original(self, command, idempotency_key, **kwargs)

    monkeypatch.setattr(
        LedgerStore,
        "append_official_result",
        append_after_capture_supersession,
    )

    with pytest.raises(OperatorPreconditionError, match="active.*capture"):
        _run(review_context, bundle)

    assert not _records(review_context, "official_result")
    assert not _records(review_context, "review")


def test_capture_supersession_before_review_append_prevents_review_write(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = LedgerStore.append_review
    injected = False

    def append_after_capture_supersession(
        self,
        command,
        idempotency_key=None,
        **kwargs,
    ):
        nonlocal injected
        if not injected:
            injected = True
            _supersede_confirmed_capture(review_context, self, "before-review")
        return original(self, command, idempotency_key, **kwargs)

    monkeypatch.setattr(
        LedgerStore,
        "append_review",
        append_after_capture_supersession,
    )

    with pytest.raises(
        OperatorPreconditionError,
        match="active.*capture",
    ) as raised:
        _run(review_context)

    progress = raised.value.partial_progress
    assert progress.rerun_safe is False
    assert len(_records(review_context, "official_result")) == 1
    assert not _records(review_context, "review")

    monkeypatch.setattr(LedgerStore, "append_review", original)
    with pytest.raises(OperatorPreconditionError, match="latest active"):
        _run(review_context)


def test_partial_retry_after_close_append_completes_without_duplicates(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = LedgerStore.append_official_result

    def interrupt(self, command, idempotency_key=None, **kwargs):
        raise OSError("interrupted after close")

    monkeypatch.setattr(LedgerStore, "append_official_result", interrupt)
    with pytest.raises(OperatorIOError, match="official result") as raised:
        _run(review_context)
    monkeypatch.setattr(LedgerStore, "append_official_result", original)

    closing_records = _records(review_context, "market_observation")
    assert len(closing_records) == 1
    assert not _records(review_context, "official_result")
    assert not _records(review_context, "review")
    progress = raised.value.partial_progress
    assert progress.capture_id == review_context.confirmed.capture.capture_id
    assert progress.forecast_run_id == review_context.confirmed.forecast.forecast_run_id
    assert progress.capture_forecast_link_record_id == (
        review_context.confirmed.link.envelope.record_id
    )
    assert progress.closing_observation_record_ids == (
        closing_records[0].envelope.record_id,
    )
    assert progress.official_result_record_id is None
    assert progress.review_record_ids == ()
    assert progress.rerun_safe is True
    assert not tuple(
        (review_context.bundle_root / str(review_context.bundle.game_pk)).glob(
            "review-*.json"
        )
    )

    completed = _run(review_context)
    assert completed.closing_replay_count == 1
    assert completed.official_result_replayed is False
    assert completed.review_replay_count == 0
    assert len(_records(review_context, "market_observation")) == 1


def test_partial_retry_after_result_and_one_review_completes_all_sorted_reviews(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def interrupt_second(
        store,
        link_record_id,
        result_record_id,
        market_key,
        **kwargs,
    ):
        calls.append(market_key)
        if len(calls) == 2:
            raise OSError("interrupted during reviews")
        return real_build_and_record_review(
            store,
            link_record_id,
            result_record_id,
            market_key,
            **kwargs,
        )

    monkeypatch.setattr(
        service_module,
        "build_and_record_review",
        interrupt_second,
    )
    with pytest.raises(OperatorIOError, match="review") as raised:
        _run(review_context)

    assert calls == ["full_under", "sea_ml"]
    result_records = _records(review_context, "official_result")
    review_records = _records(review_context, "review")
    assert len(result_records) == 1
    assert tuple(review.market_key for review in review_records) == ("full_under",)
    progress = raised.value.partial_progress
    assert progress.official_result_record_id == (
        result_records[0].envelope.record_id
    )
    assert progress.review_record_ids == (
        review_records[0].envelope.record_id,
    )
    assert progress.rerun_safe is True
    assert not tuple(
        (review_context.bundle_root / str(review_context.bundle.game_pk)).glob(
            "review-*.json"
        )
    )

    monkeypatch.setattr(
        service_module,
        "build_and_record_review",
        real_build_and_record_review,
    )
    completed = _run(review_context)
    assert completed.official_result_replayed is True
    assert completed.review_replay_count == 1
    assert tuple(review.market_key for review in completed.reviews) == (
        "full_under",
        "sea_ml",
    )
    assert len(_records(review_context, "review")) == 2


def test_report_failure_reports_all_progress_and_exact_retry_replays(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = service_module._publish_calibration_report
    interrupted = False

    def interrupt_once(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OperatorIOError("calibration publication interrupted")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        interrupt_once,
    )

    with pytest.raises(OperatorIOError, match="publication") as raised:
        _run(review_context)

    closing_records = _records(review_context, "market_observation")
    result_records = _records(review_context, "official_result")
    review_records = _records(review_context, "review")
    progress = raised.value.partial_progress
    assert progress.closing_observation_record_ids == tuple(
        record.envelope.record_id for record in closing_records
    )
    assert progress.official_result_record_id == (
        result_records[0].envelope.record_id
    )
    assert progress.review_record_ids == tuple(
        record.envelope.record_id for record in review_records
    )
    assert not hasattr(progress, "managed_bundle_path")
    assert not hasattr(progress, "managed_bundle_replayed")
    assert progress.rerun_safe is True

    completed = _run(review_context)

    assert completed.closing_replay_count == len(closing_records)
    assert completed.official_result_replayed is True
    assert completed.review_replay_count == len(review_records)
    assert completed.managed_bundle.replayed is True
    assert len(_records(review_context, "market_observation")) == len(
        closing_records
    )
    assert len(_records(review_context, "official_result")) == 1
    assert len(_records(review_context, "review")) == len(review_records)


def test_result_correction_after_report_failure_makes_retry_unsafe(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = service_module._publish_calibration_report
    corrected_record_id = None

    def correct_result_then_interrupt(store, *args, **kwargs):
        nonlocal corrected_record_id
        del args, kwargs
        original = next(
            record
            for record in store.iter_records("official_result")
            if isinstance(record, OfficialResultRecord)
            and record.source_revision == "final-1"
        )
        correction = replace(
            review_context.bundle.payload.official_result,
            source_revision="final-competing-correction",
            finalized_at=(
                review_context.bundle.payload.official_result.finalized_at
                + timedelta(minutes=20)
            ),
            documented_events=({"result_label": "official correction"},),
            supersedes_record_id=original.envelope.record_id,
        )
        corrected = store.append_official_result(
            correction,
            required_active_capture_id=(
                review_context.confirmed.capture.capture_id
            ),
        )
        corrected_record_id = corrected.envelope.record_id
        raise OperatorIOError("calibration publication interrupted")

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        correct_result_then_interrupt,
    )

    with pytest.raises(OperatorIOError) as raised:
        _run(review_context)

    assert corrected_record_id is not None
    assert raised.value.partial_progress.rerun_safe is False
    assert len(_records(review_context, "official_result")) == 2

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        original_publish,
    )
    with pytest.raises(
        OperatorPreconditionError,
        match="current latest|superseded",
    ):
        _run(review_context)
    assert len(_records(review_context, "official_result")) == 2


def test_closing_correction_after_report_failure_makes_retry_unsafe(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = service_module._publish_calibration_report
    corrected_record_id = None

    def correct_close_then_interrupt(store, *args, **kwargs):
        nonlocal corrected_record_id
        del args, kwargs
        original = next(
            record
            for record in store.iter_records("market_observation")
            if isinstance(record, MarketObservationRecord)
            and record.source_revision == "close-1"
        )
        correction = replace(
            review_context.bundle.payload.closing_observations[0],
            source_revision="close-competing-correction",
            selected_decimal_price=2.07,
            opposing_decimal_price=1.79,
            supersedes_record_id=original.envelope.record_id,
        )
        corrected = store.append_market_observation(
            correction,
            expected_latest_record_id=original.envelope.record_id,
        )
        corrected_record_id = corrected.envelope.record_id
        raise OperatorIOError("calibration publication interrupted")

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        correct_close_then_interrupt,
    )

    with pytest.raises(OperatorIOError) as raised:
        _run(review_context)

    assert corrected_record_id is not None
    assert raised.value.partial_progress.rerun_safe is False
    assert len(_records(review_context, "market_observation")) == 2

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        original_publish,
    )
    with pytest.raises(
        OperatorPreconditionError,
        match="current latest|superseded",
    ):
        _run(review_context)
    assert len(_records(review_context, "market_observation")) == 2


def test_post_persistence_replacement_is_never_exposed_in_progress(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "replacement-review-bundle.json"

    def replace_bundle_with_symlink(*args, **kwargs):
        del args, kwargs
        game_dir = review_context.bundle_root / str(review_context.bundle.game_pk)
        (managed_path,) = tuple(game_dir.glob("review-*.json"))
        outside.write_bytes(managed_path.read_bytes())
        managed_path.unlink()
        managed_path.symlink_to(outside)
        raise OperatorIOError("calibration publication interrupted")

    monkeypatch.setattr(
        service_module,
        "_publish_calibration_report",
        replace_bundle_with_symlink,
    )

    with pytest.raises(OperatorIOError, match="publication") as raised:
        _run(review_context)

    progress = raised.value.partial_progress
    assert not hasattr(progress, "managed_bundle_path")
    assert not hasattr(progress, "managed_bundle_replayed")
    assert progress.rerun_safe is True


def test_partial_recovery_has_no_managed_artifact_discovery_api() -> None:
    assert not hasattr(service_module, "_committed_managed_bundle_path")


def test_review_bundle_is_persisted_only_after_all_ledger_operations_succeed(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = service_module.persist_managed_bundle
    observed: dict[str, int] = {}

    def assert_ledger_complete(bundle, *, root):
        store = LedgerStore(review_context.ledger_path)
        observed["close"] = len(tuple(store.iter_records("market_observation")))
        observed["result"] = len(tuple(store.iter_records("official_result")))
        observed["review"] = len(tuple(store.iter_records("review")))
        return original(bundle, root=root)

    monkeypatch.setattr(
        service_module,
        "persist_managed_bundle",
        assert_ledger_complete,
    )

    result = _run(review_context)

    assert observed == {"close": 1, "result": 1, "review": 2}
    assert result.managed_bundle.path.exists()


def test_expected_ledger_read_failures_map_to_operator_ledger_error(
    review_context: ReviewContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupt(self, kind=None):
        raise LedgerCorruptionError("broken ledger")

    monkeypatch.setattr(LedgerStore, "iter_records", corrupt)

    with pytest.raises(OperatorLedgerError, match="broken ledger"):
        _run(review_context)


def test_review_result_score_conservation_errors_remain_bundle_validation_errors(
) -> None:
    mapping = review_bundle_mapping()
    result = mapping["payload"]["official_result"]
    result["component_values"] = {
        "home_pitcher_chain_runs_allowed": 2.0,
        "home_bullpen_runs_allowed": 1.0,
    }

    with pytest.raises(
        BundleValidationError,
        match="payload.official_result: invalid_value",
    ):
        parse_review_bundle_json(_raw(mapping))
