"""Push-aware settlement and pricing for simulated betting markets."""

from dataclasses import dataclass
import math

from .inputs import MarketInput, MarketType, SettlementFormat
from .simulation import SimulatedGame, SimulationResult


_PROBABILITY_TOLERANCE = 1e-12
_SIDE_MARKETS = frozenset(
    {MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5}
)
_TOTAL_MARKETS = frozenset({MarketType.F5_TOTAL, MarketType.FULL_TOTAL})


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
    evaluated: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluated, bool):
            raise ValueError("evaluated must be a boolean")
        if self.evaluated:
            if not isinstance(self.probabilities, OutcomeProbabilities):
                raise ValueError(
                    "evaluated markets must contain OutcomeProbabilities"
                )
            if self.reason is not None:
                raise ValueError("evaluated markets cannot contain a reason")
        else:
            _require_nonempty_string(self.reason, "reason")
            if self.probabilities is not None or any(
                value is not None
                for value in (
                    self.model_conditional_win_probability,
                    self.fair_decimal_price,
                    self.market_no_vig_probability,
                    self.edge,
                    self.expected_value,
                )
            ):
                raise ValueError(
                    "unevaluated markets cannot contain probabilities or prices"
                )

        for field_name in (
            "model_conditional_win_probability",
            "fair_decimal_price",
            "market_no_vig_probability",
            "edge",
            "expected_value",
        ):
            _require_optional_finite(getattr(self, field_name), field_name)
        for field_name in (
            "model_conditional_win_probability",
            "market_no_vig_probability",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1] when present")
        if self.fair_decimal_price is not None and self.fair_decimal_price < 1.0:
            raise ValueError("fair_decimal_price must be at least one when present")
        if self.edge is not None and not -1.0 <= self.edge <= 1.0:
            raise ValueError("edge must be in [-1, 1] when present")
        if self.edge is not None and (
            self.model_conditional_win_probability is None
            or self.market_no_vig_probability is None
        ):
            raise ValueError("edge requires model and market probabilities")


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

    all_push = probabilities.push == 1.0
    if all_push:
        model_conditional = None
        fair_decimal_price = None
    else:
        model_conditional = probabilities.win / (1.0 - probabilities.push)
        fair_decimal_price = (
            None
            if probabilities.win == 0.0
            else 1.0 + probabilities.loss / probabilities.win
        )

    market_no_vig = None
    if selected_price is not None and opposing_price is not None:
        selected_implied = 1.0 / selected_price
        opposing_implied = 1.0 / opposing_price
        market_no_vig = selected_implied / (
            selected_implied + opposing_implied
        )

    edge = (
        None
        if model_conditional is None or market_no_vig is None
        else model_conditional - market_no_vig
    )
    if selected_price is None:
        expected_value = None
    elif all_push:
        expected_value = 0.0
    else:
        expected_value = (
            probabilities.win * (selected_price - 1.0) - probabilities.loss
        )

    return MarketEvaluation(
        probabilities=probabilities,
        model_conditional_win_probability=model_conditional,
        fair_decimal_price=fair_decimal_price,
        market_no_vig_probability=market_no_vig,
        edge=edge,
        expected_value=expected_value,
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
    return price_market(
        probabilities,
        selected_price=market_input.selected_decimal_price,
        opposing_price=market_input.opposing_decimal_price,
    )


__all__ = (
    "OutcomeProbabilities",
    "MarketEvaluation",
    "price_market",
    "evaluate_market",
)
