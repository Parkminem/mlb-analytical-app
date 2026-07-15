"""Stable exception categories for shadow operator commands."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperatorPartialProgress:
    """Authoritative committed state discovered after an interrupted stage."""

    stage: str
    game_pk: int
    capture_id: str | None = None
    forecast_run_id: str | None = None
    capture_forecast_link_record_id: str | None = None
    closing_observation_record_ids: tuple[str, ...] = ()
    official_result_record_id: str | None = None
    review_record_ids: tuple[str, ...] = ()
    managed_bundle_path: Path | None = None
    capture_replayed: bool | None = None
    forecast_replayed: bool | None = None
    link_replayed: bool | None = None
    closing_replay_count: int | None = None
    official_result_replayed: bool | None = None
    review_replay_count: int | None = None
    managed_bundle_replayed: bool | None = None
    rerun_safe: bool = False


class OperatorError(Exception):
    """Base class for expected operator failures."""

    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        partial_progress: OperatorPartialProgress | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_progress = partial_progress

    def with_partial_progress(
        self,
        progress: OperatorPartialProgress,
    ) -> "OperatorError":
        """Return the same error category carrying immutable recovery state."""

        if self.partial_progress is not None:
            return self
        return self.__class__(
            self.args[0],
            partial_progress=progress,
        )


class BundleValidationError(OperatorError):
    """Invalid config or external bundle content."""

    exit_code = 3


class OperatorPreconditionError(OperatorError):
    """Valid input that cannot run in the current operator state."""

    exit_code = 4


class OperatorLedgerError(OperatorError):
    """Ledger reference, idempotency, or corruption failure."""

    exit_code = 5


class OperatorIOError(OperatorError):
    """Managed artifact or report I/O failure."""

    exit_code = 6


__all__ = (
    "BundleValidationError",
    "OperatorError",
    "OperatorIOError",
    "OperatorLedgerError",
    "OperatorPartialProgress",
    "OperatorPreconditionError",
)
