"""Stable exception categories for shadow operator commands."""


class OperatorError(Exception):
    """Base class for expected operator failures."""

    exit_code = 1


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
