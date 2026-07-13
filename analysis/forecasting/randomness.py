"""Injected pseudo-random primitives for deterministic forecast simulations."""

from collections.abc import Sequence
import math
from typing import TypeVar


_T = TypeVar("_T")


def _require_method(rng: object, method_name: str):
    method = getattr(rng, method_name, None)
    if not callable(method):
        raise TypeError(f"rng must provide callable {method_name}()")
    return method


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _require_finite_number(value: object, field_name: str) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite non-boolean number")


def draw_bounded_lognormal(
    rng: object,
    point: float,
    sigma: float,
    lower: float,
    upper: float,
) -> float:
    """Draw a mean-preserving lognormal multiplier and clip it to bounds."""

    gauss = _require_method(rng, "gauss")
    for value, field_name in (
        (point, "point"),
        (sigma, "sigma"),
        (lower, "lower"),
        (upper, "upper"),
    ):
        _require_finite_number(value, field_name)
    if point <= 0 or lower <= 0 or upper <= 0:
        raise ValueError("point, lower, and upper must be positive")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper")

    standard_normal = gauss(0, 1)
    _require_finite_number(standard_normal, "rng.gauss result")
    log_latent = (
        math.log(point)
        - sigma * sigma / 2
        + sigma * standard_normal
    )
    if math.isnan(log_latent):
        log_latent = math.log(point) + sigma * (standard_normal - sigma / 2)
    try:
        latent = math.exp(log_latent)
    except OverflowError:
        latent = math.inf
    return float(min(max(latent, lower), upper))


def draw_gamma_poisson(
    rng: object,
    *,
    mean: float,
    shape: float,
) -> int:
    """Draw a gamma-mixed Poisson count using Knuth's product algorithm."""

    gammavariate = _require_method(rng, "gammavariate")
    uniform = _require_method(rng, "random")
    _require_finite_number(mean, "mean")
    _require_finite_number(shape, "shape")
    if mean < 0:
        raise ValueError("mean must be nonnegative")
    if shape <= 0:
        raise ValueError("shape must be positive")
    if mean == 0:
        return 0

    latent = gammavariate(shape, mean / shape)
    if not _is_finite_number(latent) or latent < 0:
        raise ValueError("gamma latent must be finite and nonnegative")
    if latent == 0:
        return 0

    limit = math.exp(-latent)
    product = 1.0
    count = 0
    while product > limit:
        sample = uniform()
        if not _is_finite_number(sample) or not 0.0 <= sample <= 1.0:
            raise ValueError("rng.random result must be finite and in [0, 1]")
        product *= sample
        count += 1
    return count - 1


def weighted_choice(
    rng: object,
    values: Sequence[_T],
    weights: Sequence[float],
) -> _T:
    """Select by cumulative nonnegative weights using only the injected RNG."""

    uniform = _require_method(rng, "random")
    if not isinstance(values, Sequence) or not isinstance(weights, Sequence):
        raise TypeError("values and weights must be finite sequences")
    if not values:
        raise ValueError("values must be nonempty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    for weight in weights:
        if not _is_finite_number(weight) or weight < 0:
            raise ValueError("weights must be finite nonnegative numbers")
    try:
        total = math.fsum(weights)
    except OverflowError as error:
        raise ValueError("weight total must be finite and positive") from error
    if not math.isfinite(total) or total <= 0:
        raise ValueError("weight total must be finite and positive")

    sample = uniform()
    if not _is_finite_number(sample) or not 0.0 <= sample <= 1.0:
        raise ValueError("rng.random result must be finite and in [0, 1]")
    threshold = sample * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += weight
        if threshold < cumulative:
            return value
    return values[-1]


def central_interval(
    values: Sequence[float],
    mass: float = 0.80,
) -> tuple[float, float]:
    """Return equal-tail nearest-rank endpoints from finite observations.

    A nearest-rank quantile at probability ``q`` uses one-based rank
    ``ceil(q * n)``, clamped to the first and final observations. This makes
    singleton and small-sample behavior deterministic without interpolation.
    """

    _require_finite_number(mass, "mass")
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must be strictly between 0 and 1")
    if not isinstance(values, Sequence) or not values:
        raise ValueError("values must be a nonempty finite sequence")

    ordered: list[float] = []
    for value in values:
        _require_finite_number(value, "values")
        ordered.append(float(value))
    ordered.sort()

    tail_mass = (1.0 - mass) / 2.0
    lower_rank = max(1, math.ceil(tail_mass * len(ordered)))
    upper_rank = min(
        len(ordered),
        max(1, math.ceil((1.0 - tail_mass) * len(ordered))),
    )
    return ordered[lower_rank - 1], ordered[upper_rank - 1]
