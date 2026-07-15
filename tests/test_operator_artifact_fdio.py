import fcntl
import os
from pathlib import Path
import shutil

import pytest

import analysis.operator.bundles as bundles_module
import analysis.operator.reporting as reporting_module
from analysis.operator.bundles import persist_managed_bundle
from analysis.operator.errors import OperatorIOError
from analysis.operator.reporting import write_calibration_reports
from tests.test_forecasting_engine import _complete_game
from tests.test_operator_pregame import _projected_bundle
from tests.test_operator_reporting import (
    _ledger_record,
    _mapping,
    _unique_row,
)


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes | str], ...]:
    snapshot: list[tuple[str, str, bytes | str]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot.append((relative, "link", os.readlink(path)))
        elif path.is_dir():
            snapshot.append((relative, "directory", ""))
        else:
            snapshot.append((relative, "file", path.read_bytes()))
    return tuple(snapshot)


def _substitute_after_exclusive_lock(
    monkeypatch: pytest.MonkeyPatch,
    module,
    substitute,
) -> None:
    real_flock = fcntl.flock
    substituted = False

    def adversarial_flock(descriptor: int, operation: int) -> None:
        nonlocal substituted
        real_flock(descriptor, operation)
        if operation & fcntl.LOCK_EX and not substituted:
            substituted = True
            substitute()

    monkeypatch.setattr(module.fcntl, "flock", adversarial_flock)


def test_bundle_root_substitution_after_game_lock_never_writes_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _projected_bundle()
    root = tmp_path / "bundles"
    root.mkdir()
    moved_root = tmp_path / "bundles-opened"
    external = tmp_path / "external"
    external_game = external / str(bundle.game_pk)
    external_game.mkdir(parents=True)
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)

    def substitute() -> None:
        root.rename(moved_root)
        root.symlink_to(external, target_is_directory=True)

    _substitute_after_exclusive_lock(
        monkeypatch,
        bundles_module,
        substitute,
    )

    with pytest.raises(OperatorIOError) as error:
        persist_managed_bundle(bundle, root=root)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert not tuple(external_game.glob("*.json"))


def test_bundle_game_substitution_on_replay_never_reads_or_writes_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _projected_bundle()
    root = tmp_path / "bundles"
    first = persist_managed_bundle(bundle, root=root)
    game_dir = first.path.parent
    moved_game = root / f"{bundle.game_pk}-opened"
    external = tmp_path / "external-game"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)

    def substitute() -> None:
        game_dir.rename(moved_game)
        game_dir.symlink_to(external, target_is_directory=True)

    _substitute_after_exclusive_lock(
        monkeypatch,
        bundles_module,
        substitute,
    )

    with pytest.raises(OperatorIOError) as error:
        persist_managed_bundle(bundle, root=root)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert (moved_game / first.path.name).is_file()


def test_bundle_game_substitution_during_destination_rename_is_external_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _projected_bundle()
    root = tmp_path / "bundles"
    game_dir = root / str(bundle.game_pk)
    moved_game = root / f"{bundle.game_pk}-opened"
    external = tmp_path / "external-game"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)
    real_replace = os.replace
    substituted = False

    def adversarial_replace(source, destination, **kwargs):
        nonlocal substituted
        if Path(destination).suffix == ".json" and not substituted:
            substituted = True
            game_dir.rename(moved_game)
            game_dir.symlink_to(external, target_is_directory=True)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(bundles_module.os, "replace", adversarial_replace)

    with pytest.raises(OperatorIOError) as error:
        persist_managed_bundle(bundle, root=root)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert not tuple(external.glob("*.json"))
    assert len(tuple(moved_game.glob("*.json"))) == 1


def test_report_root_substitution_on_new_publication_never_writes_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    moved_report_dir = tmp_path / "reports-opened"
    external = tmp_path / "external-reports"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)

    def substitute() -> None:
        report_dir.rename(moved_report_dir)
        report_dir.symlink_to(external, target_is_directory=True)

    _substitute_after_exclusive_lock(
        monkeypatch,
        reporting_module,
        substitute,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(mapping, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert not (external / "current").exists()


def test_report_root_substitution_on_replay_preserves_opened_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(mapping, report_dir=report_dir)
    current_target = os.readlink(published.current_generation_path)
    moved_report_dir = tmp_path / "reports-opened"
    external = tmp_path / "external-reports"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)

    def substitute() -> None:
        report_dir.rename(moved_report_dir)
        report_dir.symlink_to(external, target_is_directory=True)

    _substitute_after_exclusive_lock(
        monkeypatch,
        reporting_module,
        substitute,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(mapping, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert os.readlink(moved_report_dir / "current") == current_target


def test_report_root_substitution_after_replay_reads_fails_without_redirecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(mapping, report_dir=report_dir)
    current_target = os.readlink(published.current_generation_path)
    moved_report_dir = tmp_path / "reports-opened"
    external = tmp_path / "external-reports"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)
    real_verify = reporting_module._require_equal_revision_content

    def adversarial_verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        report_dir.rename(moved_report_dir)
        report_dir.symlink_to(external, target_is_directory=True)
        return result

    monkeypatch.setattr(
        reporting_module,
        "_require_equal_revision_content",
        adversarial_verify,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(mapping, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert os.readlink(moved_report_dir / "current") == current_target


def test_report_root_substitution_during_legacy_migration_is_external_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy_json = b'{"legacy":"json"}\n'
    legacy_markdown = b"# legacy markdown\n"
    (report_dir / "calibration-latest.json").write_bytes(legacy_json)
    (report_dir / "calibration-latest.md").write_bytes(legacy_markdown)
    moved_report_dir = tmp_path / "reports-opened"
    external = tmp_path / "external-reports"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)

    def substitute() -> None:
        report_dir.rename(moved_report_dir)
        report_dir.symlink_to(external, target_is_directory=True)

    _substitute_after_exclusive_lock(
        monkeypatch,
        reporting_module,
        substitute,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(mapping, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert (moved_report_dir / "calibration-latest.json").read_bytes() == (
        legacy_json
    )
    assert (moved_report_dir / "calibration-latest.md").read_bytes() == (
        legacy_markdown
    )


def test_report_root_substitution_between_legacy_reads_is_external_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy_json = b'{"legacy":"json"}\n'
    legacy_markdown = b"# legacy markdown\n"
    (report_dir / "calibration-latest.json").write_bytes(legacy_json)
    (report_dir / "calibration-latest.md").write_bytes(legacy_markdown)
    moved_report_dir = tmp_path / "reports-opened"
    external = tmp_path / "external-reports"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)
    real_read = reporting_module.read_regular_file
    substituted = False

    def adversarial_read(descriptor, filename, *, label):
        nonlocal substituted
        content = real_read(descriptor, filename, label=label)
        if label == "legacy calibration JSON" and not substituted:
            substituted = True
            report_dir.rename(moved_report_dir)
            report_dir.symlink_to(external, target_is_directory=True)
        return content

    monkeypatch.setattr(
        reporting_module,
        "read_regular_file",
        adversarial_read,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(mapping, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert (moved_report_dir / "calibration-latest.json").read_bytes() == (
        legacy_json
    )
    assert (moved_report_dir / "calibration-latest.md").read_bytes() == (
        legacy_markdown
    )
    assert not (moved_report_dir / "current").exists()


def test_report_generations_substitution_preserves_current_and_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(appended_at, sequence=2)
    older = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
        ledger_records=(first_record,),
    )
    newer = _mapping(
        tmp_path,
        (
            _unique_row(1, lineup_state="projected"),
            _unique_row(2, lineup_state="projected"),
        ),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(older, report_dir=report_dir)
    current_target = os.readlink(published.current_generation_path)
    generation_name = Path(current_target).name
    generations_dir = report_dir / "generations"
    moved_generations = report_dir / "generations-opened"
    external = tmp_path / "external-generations"
    external.mkdir()
    shutil.copytree(
        generations_dir / generation_name,
        external / generation_name,
    )
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)
    real_open = reporting_module._open_directory_without_following_links
    substituted = False

    def adversarial_open(path, *args, **kwargs):
        nonlocal substituted
        descriptor = real_open(path, *args, **kwargs)
        if Path(path).name == "generations" and not substituted:
            substituted = True
            generations_dir.rename(moved_generations)
            generations_dir.symlink_to(external, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(
        reporting_module,
        "_open_directory_without_following_links",
        adversarial_open,
    )

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(newer, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert os.readlink(report_dir / "current") == current_target
    assert (moved_generations / generation_name).is_dir()


def test_report_generations_substitution_during_pointer_swap_restores_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(appended_at, sequence=2)
    older = _mapping(tmp_path, (), ledger_records=(first_record,))
    newer = _mapping(
        tmp_path,
        (),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(older, report_dir=report_dir)
    current_target = os.readlink(published.current_generation_path)
    generation_name = Path(current_target).name
    generations_dir = report_dir / "generations"
    moved_generations = report_dir / "generations-opened"
    external = tmp_path / "external-generations"
    external.mkdir()
    (external / "sentinel").write_bytes(b"outside")
    before = _tree_snapshot(external)
    real_replace = os.replace
    substituted = False

    def adversarial_replace(source, destination, **kwargs):
        nonlocal substituted
        if Path(destination).name == "current" and not substituted:
            substituted = True
            generations_dir.rename(moved_generations)
            generations_dir.symlink_to(external, target_is_directory=True)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(reporting_module.os, "replace", adversarial_replace)

    with pytest.raises(OperatorIOError) as error:
        write_calibration_reports(newer, report_dir=report_dir)

    assert error.value.exit_code == 6
    assert _tree_snapshot(external) == before
    assert os.readlink(report_dir / "current") == current_target
    assert {path.name for path in moved_generations.iterdir()} == {
        generation_name
    }
