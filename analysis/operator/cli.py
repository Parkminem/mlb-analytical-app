"""Typer command boundary for the three-stage shadow operator workflow."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console
from rich.text import Text

from .bundles import (
    DEFAULT_BUNDLE_ROOT,
    ReplayOnlyPregameBundle,
    load_bundle,
    load_historical_managed_bundle,
)
from .batch import (
    PregameBatchFailure,
    PregameBatchResult,
    PregameBatchSuccess,
    run_pregame_batch,
)
from .errors import (
    BundleValidationError,
    OperatorError,
    OperatorPartialProgress,
    OperatorPreconditionError,
)
from .schemas import (
    ConfirmedBundle,
    HistoricalRelaxedConfirmedBundleV1,
    HistoricalRelaxedProjectedBundleV1,
    HistoricalStrictConfirmedBundleV1,
    HistoricalStrictProjectedBundleV1,
    ProjectedBundle,
    ReviewBundle,
)
from .service import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_REPORT_DIR,
    PregameRunResult,
    ReviewRunResult,
    replay_historical_confirmed,
    replay_historical_projected,
    run_confirmed,
    run_projected,
    run_review,
)


app = typer.Typer(
    help="Run the projected, confirmed, and review shadow stages.",
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

_OUTPUT = Console(highlight=False)
_ERROR_OUTPUT = Console(stderr=True, highlight=False)
_BundleT = TypeVar(
    "_BundleT",
    ProjectedBundle,
    ConfirmedBundle,
    ReviewBundle,
)
_SummaryRows = Iterable[tuple[str, object]]


@dataclass(frozen=True)
class _PregameBundleLoad:
    """A strict current bundle or an authenticated historical source path."""

    bundle: ProjectedBundle | ConfirmedBundle | None
    historical_path: Path | None

    def __post_init__(self) -> None:
        if (self.bundle is None) == (self.historical_path is None):
            raise ValueError(
                "pregame load must contain exactly one dispatch target"
            )


def _scalar_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _print_summary(title: str, rows: _SummaryRows) -> None:
    _OUTPUT.print(Text(title, style="bold"), soft_wrap=True)
    for label, value in rows:
        line = Text()
        line.append(f"{label}: ", style="bold cyan")
        line.append(_scalar_text(value))
        _OUTPUT.print(line, soft_wrap=True)


def _partial_progress_rows(
    progress: OperatorPartialProgress,
) -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = [
        ("partial.stage", progress.stage),
        ("partial.game_pk", progress.game_pk),
    ]
    optional_rows = (
        ("partial.capture_id", progress.capture_id),
        ("partial.forecast_run_id", progress.forecast_run_id),
        (
            "partial.capture_forecast_link_record_id",
            progress.capture_forecast_link_record_id,
        ),
        ("partial.official_result_record_id", progress.official_result_record_id),
        ("partial.capture_replayed", progress.capture_replayed),
        ("partial.forecast_replayed", progress.forecast_replayed),
        ("partial.link_replayed", progress.link_replayed),
        ("partial.closing_replay_count", progress.closing_replay_count),
        (
            "partial.official_result_replayed",
            progress.official_result_replayed,
        ),
        ("partial.review_replay_count", progress.review_replay_count),
    )
    rows.extend((label, value) for label, value in optional_rows if value is not None)
    rows.extend(
        (f"partial.closing_observation_record_id.{index}", record_id)
        for index, record_id in enumerate(
            progress.closing_observation_record_ids,
            start=1,
        )
    )
    rows.extend(
        (f"partial.review_record_id.{index}", record_id)
        for index, record_id in enumerate(progress.review_record_ids, start=1)
    )
    rows.append(("partial.rerun_safe", progress.rerun_safe))
    return tuple(rows)


def _print_partial_progress(progress: OperatorPartialProgress) -> None:
    _ERROR_OUTPUT.print(Text("Partial progress", style="bold"), soft_wrap=True)
    for label, value in _partial_progress_rows(progress):
        line = Text()
        line.append(f"{label}: ", style="bold cyan")
        line.append(_scalar_text(value))
        _ERROR_OUTPUT.print(line, soft_wrap=True)


def _concise_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message or error.__class__.__name__


def _run_at_cli_boundary(operation: Callable[[], None]) -> None:
    try:
        operation()
    except OperatorError as error:
        _ERROR_OUTPUT.print(
            Text(f"error: {_concise_error(error)}", style="bold red"),
            soft_wrap=True,
        )
        if error.partial_progress is not None:
            _print_partial_progress(error.partial_progress)
        raise typer.Exit(code=error.exit_code) from None
    except Exception as error:
        _ERROR_OUTPUT.print(
            Text(
                f"internal error: {error.__class__.__name__}",
                style="bold red",
            ),
            soft_wrap=True,
        )
        raise typer.Exit(code=1) from None


def _load_stage_bundle(
    path: Path,
    *,
    expected_type: type[_BundleT],
    expected_stage: str,
    bundle_root: Path,
) -> _BundleT:
    del bundle_root
    validated_bundle = load_bundle(path)
    if (
        type(validated_bundle) is not expected_type
        or validated_bundle.stage != expected_stage
    ):
        raise OperatorPreconditionError(
            "bundle stage/type mismatch: "
            f"{expected_stage} requires {expected_type.__name__} with stage "
            f"{expected_stage!r}"
        )
    return validated_bundle


def _load_pregame_bundle(
    path: Path,
    *,
    expected_type: type[ProjectedBundle] | type[ConfirmedBundle],
    expected_stage: str,
    bundle_root: Path,
) -> _PregameBundleLoad:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = bundle_root.expanduser().resolve(strict=False)
    try:
        bundle = _load_stage_bundle(
            path,
            expected_type=expected_type,
            expected_stage=expected_stage,
            bundle_root=bundle_root,
        )
    except BundleValidationError:
        if resolved_path.parent.parent != resolved_root:
            raise
        historical = load_historical_managed_bundle(path, root=bundle_root)
        expected_historical_types = (
            (
                HistoricalRelaxedProjectedBundleV1,
                HistoricalStrictProjectedBundleV1,
            )
            if expected_type is ProjectedBundle
            else (
                HistoricalRelaxedConfirmedBundleV1,
                HistoricalStrictConfirmedBundleV1,
            )
        )
        if (
            not isinstance(historical, ReplayOnlyPregameBundle)
            or type(historical.bundle) not in expected_historical_types
            or historical.bundle.schema_version != 1
            or historical.bundle.stage != expected_stage
        ):
            raise OperatorPreconditionError(
                "schema-v1 pregame replay stage/type mismatch"
            )
        return _PregameBundleLoad(
            bundle=None,
            historical_path=historical.managed_bundle.path,
        )
    return _PregameBundleLoad(bundle=bundle, historical_path=None)


def _print_pregame(result: PregameRunResult) -> None:
    _print_summary(
        "Shadow pregame stage",
        (
            ("stage", result.stage),
            ("game_pk", result.game_pk),
            ("lineup_state", result.lineup_state),
            ("capture_id", result.capture.capture_id),
            ("forecast_run_id", result.forecast.forecast_run_id),
            (
                "capture_forecast_link_record_id",
                result.link.envelope.record_id,
            ),
            ("managed_bundle_path", result.managed_bundle.path),
            ("managed_bundle_hash", result.managed_bundle.bundle_hash),
            ("managed_bundle_replayed", result.managed_bundle.replayed),
            ("capture_replayed", result.capture_replayed),
            ("forecast_replayed", result.forecast_replayed),
            ("link_replayed", result.link_replayed),
        ),
    )


def _batch_rows(
    result: PregameBatchResult,
    receipt_path: Path,
) -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = [
        ("stage", result.stage),
        ("complete", result.complete),
        ("target_count", result.target_count),
        ("success_count", result.success_count),
        ("failure_count", result.failure_count),
        ("max_workers", result.max_workers),
        ("receipt_path", receipt_path),
    ]
    for index, entry in enumerate(result.entries, start=1):
        prefix = f"game.{index}"
        rows.extend(
            (
                (f"{prefix}.game_pk", entry.game_pk),
                (
                    f"{prefix}.status",
                    (
                        "success"
                        if isinstance(entry, PregameBatchSuccess)
                        else "failure"
                    ),
                ),
                (f"{prefix}.bundle_path", entry.bundle_path),
            )
        )
        if isinstance(entry, PregameBatchSuccess):
            rows.extend(
                (
                    (f"{prefix}.capture_id", entry.capture_id),
                    (f"{prefix}.forecast_run_id", entry.forecast_run_id),
                    (
                        f"{prefix}.capture_forecast_link_record_id",
                        entry.capture_forecast_link_record_id,
                    ),
                    (
                        f"{prefix}.managed_bundle_path",
                        entry.managed_bundle_path,
                    ),
                    (f"{prefix}.capture_replayed", entry.capture_replayed),
                    (f"{prefix}.forecast_replayed", entry.forecast_replayed),
                    (f"{prefix}.link_replayed", entry.link_replayed),
                )
            )
        else:
            assert isinstance(entry, PregameBatchFailure)
            rows.extend(
                (
                    (f"{prefix}.error_type", entry.error_type),
                    (f"{prefix}.message", entry.message),
                    (f"{prefix}.exit_code", entry.exit_code),
                    (f"{prefix}.rerun_safe", entry.rerun_safe),
                )
            )
    return tuple(rows)


def _print_batch(result: PregameBatchResult, receipt_path: Path) -> None:
    _print_summary(
        "Shadow pregame batch",
        _batch_rows(result, receipt_path),
    )


def _run_pregame_batch_command(
    *,
    stage: str,
    manifest: Path,
    ledger: Path,
    bundle_root: Path,
    receipt: Path,
    max_workers: int,
) -> None:
    completed: list[PregameBatchResult] = []

    def execute() -> None:
        result = run_pregame_batch(
            manifest,
            expected_stage=stage,
            ledger_path=ledger,
            bundle_root=bundle_root,
            receipt_path=receipt,
            max_workers=max_workers,
        )
        completed.append(result)
        _print_batch(result, receipt)

    _run_at_cli_boundary(execute)
    if not completed[0].complete:
        raise typer.Exit(code=7)


def _review_rows(result: ReviewRunResult) -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = [
        ("stage", result.stage),
        ("game_pk", result.game_pk),
        ("capture_id", result.capture.capture_id),
        ("forecast_run_id", result.forecast.forecast_run_id),
        (
            "capture_forecast_link_record_id",
            result.link.envelope.record_id,
        ),
        (
            "official_result_record_id",
            result.official_result.envelope.record_id,
        ),
        ("closing_observation_count", len(result.closing_observations)),
        ("closing_replay_count", result.closing_replay_count),
        ("official_result_replayed", result.official_result_replayed),
        ("review_count", len(result.reviews)),
        ("review_replay_count", result.review_replay_count),
    ]
    fixed_classifications = ("A", "B", "C", "D", "unclassified")
    rows.extend(
        (
            f"classification.{classification}",
            result.classification_counts.get(classification, 0),
        )
        for classification in fixed_classifications
    )
    rows.extend(
        (f"classification.{classification}", count)
        for classification, count in sorted(result.classification_counts.items())
        if classification not in fixed_classifications
    )
    rows.extend(
        (
            ("managed_bundle_path", result.managed_bundle.path),
            ("managed_bundle_replayed", result.managed_bundle.replayed),
            ("calibration_sample_size", result.calibration_sample_size),
            ("confirmed_game_count", result.confirmed_game_count),
            ("confirmed_decision_count", result.confirmed_decision_count),
        )
    )
    rows.extend(
        (f"confirmed_decision_count.{market_type}", count)
        for market_type, count in sorted(result.confirmed_decision_counts.items())
    )
    rows.extend(
        (
            ("calibration_json_path", result.calibration_json_path),
            (
                "calibration_markdown_path",
                result.calibration_markdown_path,
            ),
            (
                "calibration_generation_dir",
                result.calibration_generation_dir,
            ),
            (
                "calibration_current_generation_path",
                result.calibration_current_generation_path,
            ),
            (
                "calibration_report_replayed",
                result.calibration_report_replayed,
            ),
        )
    )
    return tuple(rows)


def _print_review(result: ReviewRunResult) -> None:
    _print_summary("Shadow review stage", _review_rows(result))


@app.command("batch-projected")
def batch_projected(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Projected batch manifest JSON.",
    ),
    ledger: Path = typer.Option(
        DEFAULT_LEDGER_PATH,
        "--ledger",
        help="Append-only shadow ledger path.",
    ),
    bundle_root: Path = typer.Option(
        DEFAULT_BUNDLE_ROOT,
        "--bundle-root",
        help="Managed bundle directory.",
    ),
    receipt: Path = typer.Option(
        ...,
        "--receipt",
        help="Atomic projected batch receipt JSON.",
    ),
    max_workers: int = typer.Option(
        3,
        "--max-workers",
        min=1,
        max=4,
        help="Bounded forecast worker process count.",
    ),
) -> None:
    """Record or replay a complete projected-lineup batch."""

    _run_pregame_batch_command(
        stage="projected",
        manifest=manifest,
        ledger=ledger,
        bundle_root=bundle_root,
        receipt=receipt,
        max_workers=max_workers,
    )


@app.command("batch-confirmed")
def batch_confirmed(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        help="Confirmed batch manifest JSON.",
    ),
    ledger: Path = typer.Option(
        DEFAULT_LEDGER_PATH,
        "--ledger",
        help="Append-only shadow ledger path.",
    ),
    bundle_root: Path = typer.Option(
        DEFAULT_BUNDLE_ROOT,
        "--bundle-root",
        help="Managed bundle directory.",
    ),
    receipt: Path = typer.Option(
        ...,
        "--receipt",
        help="Atomic confirmed batch receipt JSON.",
    ),
    max_workers: int = typer.Option(
        3,
        "--max-workers",
        min=1,
        max=4,
        help="Bounded forecast worker process count.",
    ),
) -> None:
    """Record or replay a complete confirmed-lineup batch."""

    _run_pregame_batch_command(
        stage="confirmed",
        manifest=manifest,
        ledger=ledger,
        bundle_root=bundle_root,
        receipt=receipt,
        max_workers=max_workers,
    )


@app.command()
def projected(
    bundle: Path = typer.Option(..., "--bundle", help="Projected bundle JSON."),
    ledger: Path = typer.Option(
        DEFAULT_LEDGER_PATH,
        "--ledger",
        help="Append-only shadow ledger path.",
    ),
    bundle_root: Path = typer.Option(
        DEFAULT_BUNDLE_ROOT,
        "--bundle-root",
        help="Managed bundle directory.",
    ),
) -> None:
    """Record or replay a projected-lineup shadow forecast."""

    def execute() -> None:
        loaded = _load_pregame_bundle(
            bundle,
            expected_type=ProjectedBundle,
            expected_stage="projected",
            bundle_root=bundle_root,
        )
        if loaded.historical_path is not None:
            result = replay_historical_projected(
                loaded.historical_path,
                ledger_path=ledger,
                bundle_root=bundle_root,
            )
        else:
            assert type(loaded.bundle) is ProjectedBundle
            result = run_projected(
                loaded.bundle,
                ledger_path=ledger,
                bundle_root=bundle_root,
            )
        _print_pregame(result)

    _run_at_cli_boundary(execute)


@app.command()
def confirmed(
    bundle: Path = typer.Option(..., "--bundle", help="Confirmed bundle JSON."),
    ledger: Path = typer.Option(
        DEFAULT_LEDGER_PATH,
        "--ledger",
        help="Append-only shadow ledger path.",
    ),
    bundle_root: Path = typer.Option(
        DEFAULT_BUNDLE_ROOT,
        "--bundle-root",
        help="Managed bundle directory.",
    ),
) -> None:
    """Record or replay a confirmed-lineup shadow forecast."""

    def execute() -> None:
        loaded = _load_pregame_bundle(
            bundle,
            expected_type=ConfirmedBundle,
            expected_stage="confirmed",
            bundle_root=bundle_root,
        )
        if loaded.historical_path is not None:
            result = replay_historical_confirmed(
                loaded.historical_path,
                ledger_path=ledger,
                bundle_root=bundle_root,
            )
        else:
            assert type(loaded.bundle) is ConfirmedBundle
            result = run_confirmed(
                loaded.bundle,
                ledger_path=ledger,
                bundle_root=bundle_root,
            )
        _print_pregame(result)

    _run_at_cli_boundary(execute)


@app.command()
def review(
    bundle: Path = typer.Option(..., "--bundle", help="Review bundle JSON."),
    ledger: Path = typer.Option(
        DEFAULT_LEDGER_PATH,
        "--ledger",
        help="Append-only shadow ledger path.",
    ),
    bundle_root: Path = typer.Option(
        DEFAULT_BUNDLE_ROOT,
        "--bundle-root",
        help="Managed bundle directory.",
    ),
    report_dir: Path = typer.Option(
        DEFAULT_REPORT_DIR,
        "--report-dir",
        help="Calibration report directory.",
    ),
) -> None:
    """Record or replay an official result review and calibration report."""

    def execute() -> None:
        loaded = _load_stage_bundle(
            bundle,
            expected_type=ReviewBundle,
            expected_stage="review",
            bundle_root=bundle_root,
        )
        result = run_review(
            loaded,
            ledger_path=ledger,
            bundle_root=bundle_root,
            report_dir=report_dir,
        )
        _print_review(result)

    _run_at_cli_boundary(execute)


def main() -> None:
    """Run the Typer application."""

    app()


__all__ = ("app", "main")
