"""Typer command boundary for the three-stage shadow operator workflow."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console
from rich.text import Text

from .bundles import DEFAULT_BUNDLE_ROOT, load_bundle
from .errors import (
    OperatorError,
    OperatorPartialProgress,
    OperatorPreconditionError,
)
from .schemas import ConfirmedBundle, ProjectedBundle, ReviewBundle
from .service import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_REPORT_DIR,
    PregameRunResult,
    ReviewRunResult,
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
) -> _BundleT:
    bundle = load_bundle(path)
    if type(bundle) is not expected_type or bundle.stage != expected_stage:
        raise OperatorPreconditionError(
            "bundle stage/type mismatch: "
            f"{expected_stage} requires {expected_type.__name__} with stage "
            f"{expected_stage!r}"
        )
    return bundle


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
        loaded = _load_stage_bundle(
            bundle,
            expected_type=ProjectedBundle,
            expected_stage="projected",
        )
        result = run_projected(
            loaded,
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
        loaded = _load_stage_bundle(
            bundle,
            expected_type=ConfirmedBundle,
            expected_stage="confirmed",
        )
        result = run_confirmed(
            loaded,
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
