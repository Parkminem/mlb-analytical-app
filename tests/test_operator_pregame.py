from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path

import pytest

from analysis.operator.bundles import (
    ManagedBundleCopy,
    bundle_canonical_bytes,
    bundle_sha256,
    intrinsic_capture_key,
    load_bundle,
    persist_managed_bundle,
)
from analysis.operator.errors import BundleValidationError, OperatorIOError
from analysis.operator.schemas import ProjectedBundle, parse_projected_bundle_json
from analysis.shadow import build_shadow_capture_command
from tests.test_forecasting_inputs import complete_forecast_game
from tests.test_operator_schemas import _raw, projected_bundle_mapping


def _projected_bundle() -> ProjectedBundle:
    return parse_projected_bundle_json(_raw(projected_bundle_mapping()))


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
