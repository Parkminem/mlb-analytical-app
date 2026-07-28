"""Postgame review classification API."""

from importlib import import_module

from .classification import (
    ComponentForecast,
    ComponentObservation,
    ContradictionRule,
    ExogenousEvent,
    PathReview,
    PredictiveInterval,
    PrincipalPathBinding,
    ReviewCandidate,
    ReviewClassification,
    classify_review,
)
from .service import (
    RecordedReview,
    build_and_record_review,
    build_review_candidate,
)


_COVERAGE_AUDIT_EXPORTS = frozenset(
    ("build_coverage_audit", "render_coverage_audit_markdown")
)


def __getattr__(name: str):
    if name not in _COVERAGE_AUDIT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.coverage_audit"), name)
    globals()[name] = value
    return value


__all__ = (
    "ComponentForecast",
    "ComponentObservation",
    "ContradictionRule",
    "ExogenousEvent",
    "PathReview",
    "PredictiveInterval",
    "PrincipalPathBinding",
    "ReviewCandidate",
    "ReviewClassification",
    "RecordedReview",
    "build_and_record_review",
    "build_coverage_audit",
    "build_review_candidate",
    "classify_review",
    "render_coverage_audit_markdown",
)
