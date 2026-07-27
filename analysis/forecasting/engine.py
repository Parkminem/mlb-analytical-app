"""Pure orchestration for one deterministic MLB forecast run."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import math
import re
from types import MappingProxyType

from .bullpen import BullpenForecast, forecast_bullpen
from .config import SimulationConfig
from .identity import build_forecast_run_id_for_input as build_forecast_run_id
from .inputs import (
    MarketInput,
    MarketType,
    NormalizedGameInput,
    SettlementFormat,
)
from .market import (
    MarketEvaluation,
    evaluate_market,
    market_is_push_capable,
    market_semantic_fingerprint,
)
from .offense import OffenseForecast, forecast_offense
from .simulation import (
    ComponentDistribution,
    DistributionSummary,
    GameSimulationInput,
    simulate_game_distribution,
    validate_component_distribution_domain,
)
from .starter import PitcherChainForecast, forecast_pitcher_chain
from .validation import validate_pregame_input


_FORECAST_RUN_ID = re.compile(r"[0-9a-f]{64}\Z")
_DIAGNOSTIC_TYPES = {
    "away_pitcher_chain": PitcherChainForecast,
    "home_pitcher_chain": PitcherChainForecast,
    "away_offense": OffenseForecast,
    "home_offense": OffenseForecast,
    "away_bullpen": BullpenForecast,
    "home_bullpen": BullpenForecast,
}
_BASE_COMPONENT_SUFFIXES = (
    "f5_runs",
    "full_runs",
    "f5_traffic",
    "full_traffic",
    "f5_extra_base_hits",
    "full_extra_base_hits",
    "pitcher_chain_runs_allowed",
    "bullpen_runs_allowed",
    "pitcher_chain_workload_outs",
    "protected_bullpen_close_entry",
)
_REQUIRED_COMPONENT_IDS = frozenset(
    f"{side}_{suffix}"
    for side in ("away", "home")
    for suffix in _BASE_COMPONENT_SUFFIXES
)
_SIDE_MARKET_TYPES = frozenset(
    {MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5}
)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _freeze_simulation_config(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("simulation_config must be a mapping")
    expected_keys = {field.name for field in fields(SimulationConfig)}
    if set(value) != expected_keys:
        raise ValueError("simulation_config must contain every configuration field")
    try:
        rebuilt = SimulationConfig.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"simulation_config is invalid: {error}") from error
    canonical = rebuilt.to_mapping()
    frozen: dict[str, object] = {}
    for key, canonical_value in canonical.items():
        supplied = value[key]
        if isinstance(canonical_value, list):
            if (
                not isinstance(supplied, (list, tuple))
                or list(supplied) != canonical_value
            ):
                raise ValueError("simulation_config must be canonical")
            frozen[key] = tuple(canonical_value)
        else:
            if supplied != canonical_value:
                raise ValueError("simulation_config must be canonical")
            frozen[key] = canonical_value
    return MappingProxyType(frozen)


def _freeze_component_diagnostics(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("component_diagnostics must be a mapping")
    copied = dict(value)
    if set(copied) != set(_DIAGNOSTIC_TYPES):
        raise ValueError("component_diagnostics must contain all six components")
    for component_id, expected_type in _DIAGNOSTIC_TYPES.items():
        if not isinstance(copied[component_id], expected_type):
            raise TypeError(
                f"component_diagnostics[{component_id!r}] must be "
                f"{expected_type.__name__}"
            )
    return MappingProxyType(
        {key: copied[key] for key in sorted(copied)}
    )


def _freeze_component_distributions(
    value: object,
    simulation_count: int,
    diagnostics: Mapping[str, object],
) -> Mapping[str, ComponentDistribution]:
    if not isinstance(value, Mapping):
        raise TypeError("component_distributions must be a mapping")
    copied = dict(value)
    expected_ids = set(_REQUIRED_COMPONENT_IDS)
    for side in ("away", "home"):
        bullpen = diagnostics[f"{side}_bullpen"]
        expected_ids.update(
            f"{side}_bullpen_segment_{segment.segment_id}_exposed"
            for segment in bullpen.available_segments
        )
    if set(copied) != expected_ids:
        raise ValueError(
            "component_distributions must contain every fixed component ID"
        )
    for component_id, distribution in copied.items():
        _require_nonempty_string(component_id, "component distribution key")
        if not isinstance(distribution, ComponentDistribution):
            raise TypeError(
                "component_distributions must contain ComponentDistribution values"
            )
        if distribution.component_id != component_id:
            raise ValueError("component distribution key and ID must match")
        validate_component_distribution_domain(distribution)
        if len(distribution.values) != simulation_count:
            raise ValueError("component values must match simulation_count")
    return MappingProxyType(
        {key: copied[key] for key in sorted(copied)}
    )


def _freeze_intervals(
    value: object,
    distributions: Mapping[str, ComponentDistribution],
) -> Mapping[str, DistributionSummary]:
    if not isinstance(value, Mapping):
        raise TypeError("intervals must be a mapping")
    copied = dict(value)
    if set(copied) != set(distributions):
        raise ValueError("interval keys must exactly match component distributions")
    for component_id, interval in copied.items():
        if not isinstance(interval, DistributionSummary):
            raise TypeError("intervals must contain DistributionSummary values")
        if interval != distributions[component_id].summary:
            raise ValueError("interval must match its complete component distribution")
    return MappingProxyType(
        {
            key: distributions[key].summary
            for key in sorted(distributions)
        }
    )


def _freeze_markets(value: object) -> Mapping[str, MarketEvaluation]:
    if not isinstance(value, Mapping):
        raise TypeError("markets must be a mapping")
    copied = dict(value)
    for market_key, evaluation in copied.items():
        _require_nonempty_string(market_key, "market key")
        if not isinstance(evaluation, MarketEvaluation):
            raise TypeError("markets must contain MarketEvaluation values")
    return MappingProxyType(
        {key: copied[key] for key in sorted(copied)}
    )


def _require_optional_finite_number(value: object, field_name: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite number when present")


@dataclass(frozen=True)
class SemanticMarketInput:
    """Capture-independent market terms required for pricing and settlement."""

    market_key: str
    market_type: MarketType
    settlement_format: SettlementFormat
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.market_key, "market_key")
        if not isinstance(self.market_type, MarketType):
            raise ValueError("market_type must be MarketType")
        if not isinstance(self.settlement_format, SettlementFormat):
            raise ValueError("settlement_format must be SettlementFormat")
        _require_nonempty_string(self.selected_side, "selected_side")
        _require_optional_finite_number(self.line, "line")
        for field_name in (
            "selected_decimal_price",
            "opposing_decimal_price",
        ):
            value = getattr(self, field_name)
            _require_optional_finite_number(value, field_name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{field_name} must be positive when present")

    @classmethod
    def from_market_input(cls, market: MarketInput) -> "SemanticMarketInput":
        if not isinstance(market, MarketInput):
            raise TypeError("market must be MarketInput")
        return cls(
            market_key=market.market_key,
            market_type=market.market_type,
            settlement_format=market.settlement_format,
            selected_side=market.selected_side,
            line=market.line,
            selected_decimal_price=market.selected_decimal_price,
            opposing_decimal_price=market.opposing_decimal_price,
        )

    @property
    def semantic_fingerprint(self) -> str:
        return market_semantic_fingerprint(
            market_key=self.market_key,
            market_type=self.market_type,
            settlement_format=self.settlement_format,
            selected_side=self.selected_side,
            line=self.line,
            selected_decimal_price=self.selected_decimal_price,
            opposing_decimal_price=self.opposing_decimal_price,
        )


def build_semantic_market_inputs(
    markets: tuple[MarketInput, ...],
) -> Mapping[str, SemanticMarketInput]:
    """Project resolved market inputs into canonical forecast semantics."""

    if not isinstance(markets, tuple) or any(
        not isinstance(market, MarketInput) for market in markets
    ):
        raise TypeError("markets must be a tuple of MarketInput values")
    projected = {
        market.market_key: SemanticMarketInput.from_market_input(market)
        for market in markets
    }
    if len(projected) != len(markets):
        raise ValueError("markets must have unique market_key values")
    return MappingProxyType(
        {key: projected[key] for key in sorted(projected)}
    )


def _freeze_market_inputs(
    value: object,
    markets: Mapping[str, MarketEvaluation],
) -> Mapping[str, SemanticMarketInput]:
    if not isinstance(value, Mapping):
        raise TypeError("market_inputs must be a mapping")
    copied = dict(value)
    if set(copied) != set(markets):
        raise ValueError("market_inputs keys must exactly match markets")
    for market_key, market in copied.items():
        if not isinstance(market, SemanticMarketInput):
            raise TypeError(
                "market_inputs must contain SemanticMarketInput values"
            )
        if market.market_key != market_key:
            raise ValueError("market_inputs key and market_key must match")
        evaluation = markets[market_key]
        if evaluation.evaluated:
            for field_name in (
                "selected_decimal_price",
                "opposing_decimal_price",
            ):
                if getattr(evaluation, field_name) != getattr(market, field_name):
                    raise ValueError(
                        f"{field_name} must match market_inputs for evaluated "
                        "markets"
                    )
        if evaluation.semantic_fingerprint != market.semantic_fingerprint:
            raise ValueError(
                "evaluation semantic_fingerprint must match market_inputs"
            )
        if (
            evaluation.evaluated
            and evaluation.probabilities is not None
            and not market_is_push_capable(
                market.market_type,
                market.settlement_format,
                market.line,
            )
            and evaluation.probabilities.push != 0.0
        ):
            raise ValueError(
                "non-push-capable market cannot contain push probability"
            )
    return MappingProxyType(
        {key: copied[key] for key in sorted(copied)}
    )


def _resolve_market_side(
    market: MarketInput,
    away_team: str,
    home_team: str,
) -> MarketInput:
    if market.market_type not in _SIDE_MARKET_TYPES:
        return market
    if market.selected_side in {"away", "home"}:
        return market
    team_sides = {away_team: "away", home_team: "home"}
    canonical_side = team_sides.get(market.selected_side)
    if canonical_side is None:
        raise ValueError(
            "side market selected_side must be away, home, or match the "
            "away or home team"
        )
    return replace(market, selected_side=canonical_side)


def resolve_forecast_input(
    normalized_game: NormalizedGameInput,
) -> NormalizedGameInput:
    """Return the canonical market input used for forecast identity and pricing."""

    if not isinstance(normalized_game, NormalizedGameInput):
        raise TypeError("normalized_game must be NormalizedGameInput")
    resolved_markets = tuple(
        sorted(
            (
                _resolve_market_side(
                    market,
                    normalized_game.identity.away_team,
                    normalized_game.identity.home_team,
                )
                for market in normalized_game.markets
            ),
            key=lambda market: market.market_key,
        )
    )
    return replace(normalized_game, markets=resolved_markets)


@dataclass(frozen=True)
class ForecastOutput:
    forecast_run_id: str
    config_version: str
    simulation_config: Mapping[str, object]
    simulation_count: int
    component_diagnostics: Mapping[str, object]
    component_distributions: Mapping[str, ComponentDistribution]
    intervals: Mapping[str, DistributionSummary]
    market_inputs: Mapping[str, SemanticMarketInput]
    markets: Mapping[str, MarketEvaluation]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.forecast_run_id, str)
            or _FORECAST_RUN_ID.fullmatch(self.forecast_run_id) is None
        ):
            raise ValueError(
                "forecast_run_id must be exactly 64 lowercase hex characters"
            )
        _require_nonempty_string(self.config_version, "config_version")
        config = _freeze_simulation_config(self.simulation_config)
        if config["version"] != self.config_version:
            raise ValueError("config_version must match simulation_config")
        if (
            isinstance(self.simulation_count, bool)
            or not isinstance(self.simulation_count, int)
            or self.simulation_count <= 0
        ):
            raise ValueError("simulation_count must be a positive integer")
        if config["simulations"] != self.simulation_count:
            raise ValueError("simulation_count must match simulation_config")

        diagnostics = _freeze_component_diagnostics(
            self.component_diagnostics
        )
        distributions = _freeze_component_distributions(
            self.component_distributions,
            self.simulation_count,
            diagnostics,
        )
        intervals = _freeze_intervals(self.intervals, distributions)
        markets = _freeze_markets(self.markets)
        market_inputs = _freeze_market_inputs(self.market_inputs, markets)
        object.__setattr__(self, "simulation_config", config)
        object.__setattr__(self, "component_diagnostics", diagnostics)
        object.__setattr__(self, "component_distributions", distributions)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "market_inputs", market_inputs)
        object.__setattr__(self, "markets", markets)

    @property
    def model_version(self) -> str:
        return self.config_version


def run_forecast(
    normalized_game: NormalizedGameInput,
    config: SimulationConfig,
) -> ForecastOutput:
    """Build one deterministic forecast without external writes."""

    validate_pregame_input(normalized_game)
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")
    try:
        validated_config = SimulationConfig.from_mapping(config.to_mapping())
    except (TypeError, ValueError) as error:
        raise ValueError(f"config is invalid: {error}") from error
    if validated_config != config:
        raise ValueError("config is invalid or noncanonical")

    resolved_game = resolve_forecast_input(normalized_game)
    config_mapping = config.to_mapping()
    forecast_run_id = build_forecast_run_id(
        resolved_game,
        config,
    )

    away_pitcher_chain = forecast_pitcher_chain(
        resolved_game.away_pitching,
        config,
    )
    home_pitcher_chain = forecast_pitcher_chain(
        resolved_game.home_pitching,
        config,
    )
    away_offense = forecast_offense(
        resolved_game.away_lineup,
        resolved_game.home_pitching.pitchers,
        config,
    )
    home_offense = forecast_offense(
        resolved_game.home_lineup,
        resolved_game.away_pitching.pitchers,
        config,
    )
    away_bullpen = forecast_bullpen(resolved_game.away_bullpen, config)
    home_bullpen = forecast_bullpen(resolved_game.home_bullpen, config)

    simulation_input = GameSimulationInput(
        away_offense=away_offense,
        home_offense=home_offense,
        away_pitcher_chain=away_pitcher_chain,
        home_pitcher_chain=home_pitcher_chain,
        away_bullpen=away_bullpen,
        home_bullpen=home_bullpen,
    )
    simulation = simulate_game_distribution(
        simulation_input,
        forecast_run_id,
        config,
    )
    markets = {
        market.market_key: evaluate_market(simulation, market)
        for market in resolved_game.markets
    }
    market_inputs = build_semantic_market_inputs(resolved_game.markets)
    component_diagnostics = {
        "away_pitcher_chain": away_pitcher_chain,
        "home_pitcher_chain": home_pitcher_chain,
        "away_offense": away_offense,
        "home_offense": home_offense,
        "away_bullpen": away_bullpen,
        "home_bullpen": home_bullpen,
    }
    component_distributions = simulation.component_distributions
    intervals = {
        component_id: distribution.summary
        for component_id, distribution in component_distributions.items()
    }
    return ForecastOutput(
        forecast_run_id=forecast_run_id,
        config_version=config.version,
        simulation_config=config_mapping,
        simulation_count=simulation.simulation_count,
        component_diagnostics=component_diagnostics,
        component_distributions=component_distributions,
        intervals=intervals,
        market_inputs=market_inputs,
        markets=markets,
    )


__all__ = (
    "ForecastOutput",
    "SemanticMarketInput",
    "build_semantic_market_inputs",
    "resolve_forecast_input",
    "run_forecast",
)
