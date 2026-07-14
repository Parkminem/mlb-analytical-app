"""Push-aware settlement and pricing for simulated betting markets."""

from dataclasses import dataclass, replace
import hashlib
import math
import re
import sys

from .identity import canonical_json
from .inputs import MarketInput, MarketType, SettlementFormat
from .simulation import SimulatedGame, SimulationResult


_PROBABILITY_TOLERANCE = 1e-12
_MAX_FINITE_DECIMAL_PRICE = sys.float_info.max
_SIDE_MARKETS = frozenset(
    {MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5}
)
_TOTAL_MARKETS = frozenset({MarketType.F5_TOTAL, MarketType.FULL_TOTAL})
_DERIVED_PRICING_FIELDS = (
    "model_conditional_win_probability",
    "fair_decimal_price",
    "market_no_vig_probability",
    "edge",
    "expected_value",
)
_SEMANTIC_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_optional_price(value: object, field_name: str) -> None:
    if value is None:
        return
    if not _is_finite_number(value) or value <= 1.0:
        raise ValueError(
            f"{field_name} must be a finite decimal price above one when present"
        )


def _require_optional_finite(value: object, field_name: str) -> None:
    if value is not None and not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite when present")


def market_semantic_fingerprint(
    *,
    market_key: str,
    market_type: MarketType,
    settlement_format: SettlementFormat,
    selected_side: str,
    line: float | None,
    selected_decimal_price: float | None,
    opposing_decimal_price: float | None,
) -> str:
    """Hash the complete canonical meaning of one selected market."""

    _require_nonempty_string(market_key, "market_key")
    if not isinstance(market_type, MarketType):
        raise ValueError("market_type must be MarketType")
    if not isinstance(settlement_format, SettlementFormat):
        raise ValueError("settlement_format must be SettlementFormat")
    _require_nonempty_string(selected_side, "selected_side")
    if line is not None and not _is_finite_number(line):
        raise ValueError("line must be finite when present")
    _require_optional_price(
        selected_decimal_price,
        "selected_decimal_price",
    )
    _require_optional_price(
        opposing_decimal_price,
        "opposing_decimal_price",
    )
    payload = {
        "market_key": market_key,
        "market_type": market_type,
        "settlement_format": settlement_format,
        "selected_side": selected_side,
        "line": line,
        "selected_decimal_price": selected_decimal_price,
        "opposing_decimal_price": opposing_decimal_price,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_derived_probability(value: float, field_name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")
    if 0.0 <= value <= 1.0:
        return value
    boundary = 0.0 if value < 0.0 else 1.0
    if math.isclose(
        value,
        boundary,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        return boundary
    raise ValueError(f"{field_name} must be in [0, 1]")


def _fair_decimal_price(probabilities: "OutcomeProbabilities") -> float | None:
    """Return 1 + loss / win, capping binary64 overflow at its finite maximum."""

    if probabilities.win == 0.0:
        return None
    fair_decimal_price = 1.0 + probabilities.loss / probabilities.win
    if not math.isfinite(fair_decimal_price):
        return _MAX_FINITE_DECIMAL_PRICE
    return fair_decimal_price


def _require_consistent_probability_endpoints(
    values: dict[str, float],
) -> None:
    for endpoint_field, endpoint_value in values.items():
        if endpoint_value != 1.0:
            continue
        if any(
            value != 0.0
            for field_name, value in values.items()
            if field_name != endpoint_field
        ):
            raise ValueError(
                f"{endpoint_field} exact-one endpoint requires all other "
                "probabilities to be zero"
            )


def _derive_market_pricing(
    probabilities: "OutcomeProbabilities",
    selected_decimal_price: float | None,
    opposing_decimal_price: float | None,
) -> dict[str, float | None]:
    all_push = probabilities.push == 1.0
    if all_push:
        model_conditional = None
        fair_decimal_price = None
    else:
        model_conditional = _canonical_derived_probability(
            probabilities.win / (1.0 - probabilities.push),
            "model_conditional_win_probability",
        )
        fair_decimal_price = _fair_decimal_price(probabilities)

    market_no_vig = None
    if selected_decimal_price is not None and opposing_decimal_price is not None:
        selected_implied = 1.0 / selected_decimal_price
        opposing_implied = 1.0 / opposing_decimal_price
        market_no_vig = _canonical_derived_probability(
            selected_implied / (selected_implied + opposing_implied),
            "market_no_vig_probability",
        )

    edge = (
        None
        if model_conditional is None or market_no_vig is None
        else model_conditional - market_no_vig
    )
    if selected_decimal_price is None:
        expected_value = None
    elif all_push:
        expected_value = 0.0
    else:
        expected_value = (
            probabilities.win * (selected_decimal_price - 1.0)
            - probabilities.loss
        )
    return {
        "model_conditional_win_probability": model_conditional,
        "fair_decimal_price": fair_decimal_price,
        "market_no_vig_probability": market_no_vig,
        "edge": edge,
        "expected_value": expected_value,
    }


@dataclass(frozen=True)
class OutcomeProbabilities:
    win: float
    push: float
    loss: float

    def __post_init__(self) -> None:
        values: dict[str, float] = {}
        for field_name in ("win", "push", "loss"):
            value = getattr(self, field_name)
            if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be a finite probability in [0, 1]"
                )
            values[field_name] = float(value)
        _require_consistent_probability_endpoints(values)
        total = math.fsum(values.values())
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("win, push, and loss probabilities must sum to one")
        if total != 1.0:
            adjusted_field = max(values, key=values.__getitem__)
            values[adjusted_field] = 1.0 - math.fsum(
                value
                for field_name, value in values.items()
                if field_name != adjusted_field
            )
        _require_consistent_probability_endpoints(values)
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class MarketEvaluation:
    probabilities: OutcomeProbabilities | None
    model_conditional_win_probability: float | None
    fair_decimal_price: float | None
    market_no_vig_probability: float | None
    edge: float | None
    expected_value: float | None
    selected_decimal_price: float | None = None
    opposing_decimal_price: float | None = None
    evaluated: bool = True
    reason: str | None = None
    semantic_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.semantic_fingerprint is not None and (
            not isinstance(self.semantic_fingerprint, str)
            or _SEMANTIC_FINGERPRINT.fullmatch(self.semantic_fingerprint) is None
        ):
            raise ValueError(
                "semantic_fingerprint must be exactly 64 lowercase hex characters"
            )
        if not isinstance(self.evaluated, bool):
            raise ValueError("evaluated must be a boolean")
        if not self.evaluated:
            _require_nonempty_string(self.reason, "reason")
            for field_name in (
                "probabilities",
                "selected_decimal_price",
                "opposing_decimal_price",
                *_DERIVED_PRICING_FIELDS,
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"unevaluated markets cannot contain {field_name}"
                    )
            return

        if not isinstance(self.probabilities, OutcomeProbabilities):
            raise ValueError("evaluated markets must contain OutcomeProbabilities")
        if self.reason is not None:
            raise ValueError("evaluated markets cannot contain a reason")

        for field_name in (
            "selected_decimal_price",
            "opposing_decimal_price",
        ):
            value = getattr(self, field_name)
            _require_optional_price(value, field_name)
            if value is not None:
                object.__setattr__(self, field_name, float(value))

        for field_name in _DERIVED_PRICING_FIELDS:
            _require_optional_finite(getattr(self, field_name), field_name)
        for field_name in (
            "model_conditional_win_probability",
            "market_no_vig_probability",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1] when present")

        expected = _derive_market_pricing(
            self.probabilities,
            self.selected_decimal_price,
            self.opposing_decimal_price,
        )
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"{field_name} must match probabilities, "
                    "selected_decimal_price, and opposing_decimal_price"
                )
            object.__setattr__(self, field_name, expected_value)


def price_market(
    probabilities: OutcomeProbabilities,
    selected_price: float | None,
    opposing_price: float | None,
) -> MarketEvaluation:
    """Price one selected market side from win/push/loss probabilities."""

    if not isinstance(probabilities, OutcomeProbabilities):
        raise TypeError("probabilities must be OutcomeProbabilities")
    _require_optional_price(selected_price, "selected_price")
    _require_optional_price(opposing_price, "opposing_price")
    selected_decimal_price = (
        None if selected_price is None else float(selected_price)
    )
    opposing_decimal_price = (
        None if opposing_price is None else float(opposing_price)
    )
    pricing = _derive_market_pricing(
        probabilities,
        selected_decimal_price,
        opposing_decimal_price,
    )

    return MarketEvaluation(
        probabilities=probabilities,
        selected_decimal_price=selected_decimal_price,
        opposing_decimal_price=opposing_decimal_price,
        **pricing,
    )


def _require_market_contract(market_input: MarketInput) -> None:
    if market_input.market_type in _SIDE_MARKETS:
        if market_input.selected_side not in {"away", "home"}:
            raise ValueError(
                "side market selected_side must be away or home"
            )
    elif market_input.market_type in _TOTAL_MARKETS:
        if market_input.selected_side not in {"over", "under"}:
            raise ValueError(
                "total market selected_side must be over or under"
            )
    else:
        raise ValueError("market_type is unsupported")

    if market_input.market_type is MarketType.ML:
        if market_input.line is not None:
            raise ValueError("ML line must be absent")
    else:
        if not _is_finite_number(market_input.line):
            raise ValueError(
                f"{market_input.market_type.value} line must be present and finite"
            )
        if market_input.market_type in _TOTAL_MARKETS and market_input.line < 0:
            raise ValueError("total market line must be nonnegative")
        if (
            market_input.market_type is MarketType.PLUS_1_5
            and market_input.line != 1.5
        ):
            raise ValueError("plus_1_5 line must be exactly 1.5")

    if (
        market_input.market_type is not MarketType.F5_SIDE
        and market_input.settlement_format is not SettlementFormat.TWO_WAY
    ):
        raise ValueError("settlement_format must be two_way for this market")
    _require_optional_price(
        market_input.selected_decimal_price,
        "selected_decimal_price",
    )
    _require_optional_price(
        market_input.opposing_decimal_price,
        "opposing_decimal_price",
    )


def _side_margin(game: SimulatedGame, selected_side: str, period: str) -> int:
    opposing_side = "home" if selected_side == "away" else "away"
    selected_runs = getattr(game, f"{selected_side}_{period}_runs")
    opposing_runs = getattr(game, f"{opposing_side}_{period}_runs")
    return selected_runs - opposing_runs


def _settlement_value(game: SimulatedGame, market_input: MarketInput) -> float:
    market_type = market_input.market_type
    if market_type is MarketType.F5_SIDE:
        return _side_margin(game, market_input.selected_side, "f5") + market_input.line
    if market_type is MarketType.F5_TOTAL:
        total = game.away_f5_runs + game.home_f5_runs
        return (
            total - market_input.line
            if market_input.selected_side == "over"
            else market_input.line - total
        )
    if market_type is MarketType.ML:
        return float(_side_margin(game, market_input.selected_side, "full"))
    if market_type is MarketType.PLUS_1_5:
        return (
            _side_margin(game, market_input.selected_side, "full")
            + market_input.line
        )
    if market_type is MarketType.FULL_TOTAL:
        total = game.away_full_runs + game.home_full_runs
        return (
            total - market_input.line
            if market_input.selected_side == "over"
            else market_input.line - total
        )
    raise ValueError("market_type is unsupported")


def _outcome(value: float, market_type: MarketType) -> str:
    if value > 0.0:
        return "win"
    if value < 0.0 or market_type is MarketType.ML:
        return "loss"
    return "push"


def evaluate_market(
    simulation: SimulationResult,
    market_input: MarketInput,
) -> MarketEvaluation:
    """Settle a verified normalized market over every shared simulation path."""

    if not isinstance(simulation, SimulationResult):
        raise TypeError("simulation must be SimulationResult")
    if not isinstance(market_input, MarketInput):
        raise TypeError("market_input must be MarketInput")
    if (
        not isinstance(simulation.games, tuple)
        or not simulation.games
        or len(simulation.games) != simulation.simulation_count
    ):
        raise ValueError("simulation games must match a positive simulation_count")
    if any(not isinstance(game, SimulatedGame) for game in simulation.games):
        raise TypeError("simulation games must contain SimulatedGame values")

    _require_market_contract(market_input)
    semantic_fingerprint = market_semantic_fingerprint(
        market_key=market_input.market_key,
        market_type=market_input.market_type,
        settlement_format=market_input.settlement_format,
        selected_side=market_input.selected_side,
        line=market_input.line,
        selected_decimal_price=market_input.selected_decimal_price,
        opposing_decimal_price=market_input.opposing_decimal_price,
    )
    if (
        market_input.market_type is MarketType.F5_SIDE
        and market_input.settlement_format is SettlementFormat.THREE_WAY
    ):
        return MarketEvaluation(
            probabilities=None,
            model_conditional_win_probability=None,
            fair_decimal_price=None,
            market_no_vig_probability=None,
            edge=None,
            expected_value=None,
            evaluated=False,
            reason="three-way F5 settlement is unsupported",
            semantic_fingerprint=semantic_fingerprint,
        )

    counts = {"win": 0, "push": 0, "loss": 0}
    for game in simulation.games:
        value = _settlement_value(game, market_input)
        counts[_outcome(value, market_input.market_type)] += 1
    denominator = len(simulation.games)
    probabilities = OutcomeProbabilities(
        win=counts["win"] / denominator,
        push=counts["push"] / denominator,
        loss=counts["loss"] / denominator,
    )
    return replace(
        price_market(
            probabilities,
            selected_price=market_input.selected_decimal_price,
            opposing_price=market_input.opposing_decimal_price,
        ),
        semantic_fingerprint=semantic_fingerprint,
    )


__all__ = (
    "OutcomeProbabilities",
    "MarketEvaluation",
    "market_semantic_fingerprint",
    "price_market",
    "evaluate_market",
)
