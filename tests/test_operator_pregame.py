from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import json
from pathlib import Path
import warnings

import pytest

import analysis.operator.service as service_module
from analysis.operator.bundles import (
    ManagedBundleCopy,
    bundle_canonical_bytes,
    bundle_sha256,
    intrinsic_capture_key,
    load_bundle,
    persist_managed_bundle,
)
from analysis.forecasting import LineupState
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import capture_payload
from analysis.forecasting.validation import (
    PregameRelationshipError,
    PregameTimingError,
)
from analysis.ledger import (
    IdempotencyConflict,
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerReferenceError,
    LedgerStore,
)
from analysis.operator.errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from analysis.operator.schemas import (
    ProjectedBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)
from analysis.operator.service import (
    PregameRunResult,
    run_confirmed,
    run_projected,
)
from analysis.shadow import build_shadow_capture_command
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


def _run_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "ledger" / "shadow.jsonl", tmp_path / "bundles"


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
    ] == ["starter_suppression"]
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


def test_partial_progress_never_claims_symlinked_managed_bundle(
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
    assert raised.value.partial_progress.managed_bundle_path is None


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


def test_calling_wrong_stage_service_is_a_precondition(tmp_path: Path) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    confirmed = _pregame_bundle("confirmed", _confirmed_game())

    with pytest.raises(OperatorPreconditionError, match="stage"):
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

    with pytest.raises(OperatorPreconditionError, match="captured_at"):
        runner(
            _pregame_bundle(stage, game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


def test_relationship_failure_maps_to_bundle_validation(tmp_path: Path) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    base = _projected_game()
    game = replace(base, away_lineup=replace(base.away_lineup, team="WRONG"))

    with pytest.raises(BundleValidationError, match="away team"):
        run_projected(
            _pregame_bundle("projected", game),
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )


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
    ("pregame_error", "expected_error"),
    (
        (PregameRelationshipError("relationship"), BundleValidationError),
        (PregameTimingError("timing"), OperatorPreconditionError),
    ),
)
def test_typed_pregame_failures_from_recording_keep_stable_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pregame_error: ValueError,
    expected_error: type[Exception],
) -> None:
    ledger_path, bundle_root = _run_paths(tmp_path)
    bundle = _pregame_bundle("projected", _projected_game())

    def fail_record(*args, **kwargs):
        raise pregame_error

    monkeypatch.setattr("analysis.operator.service.record_shadow_run", fail_record)

    with pytest.raises(expected_error, match=str(pregame_error)):
        run_projected(
            bundle,
            ledger_path=ledger_path,
            bundle_root=bundle_root,
        )
