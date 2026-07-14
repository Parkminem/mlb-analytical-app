"""Pure orchestration for one deterministic MLB forecast run."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import re
from types import MappingProxyType

from .bullpen import BullpenForecast, forecast_bullpen
from .config import SimulationConfig
from .identity import build_forecast_run_id
from .inputs import MarketInput, MarketType, NormalizedGameInput
from .market import MarketEvaluation, evaluate_market
from .offense import OffenseForecast, forecast_offense
from .simulation import (
    ComponentDistribution,
    DistributionSummary,
    GameSimulationInput,
    simulate_game_distribution,
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


@dataclass(frozen=True)
class ForecastOutput:
    forecast_run_id: str
    config_version: str
    simulation_config: Mapping[str, object]
    simulation_count: int
    component_diagnostics: Mapping[str, object]
    component_distributions: Mapping[str, ComponentDistribution]
    intervals: Mapping[str, DistributionSummary]
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
        object.__setattr__(self, "simulation_config", config)
        object.__setattr__(self, "component_diagnostics", diagnostics)
        object.__setattr__(self, "component_distributions", distributions)
        object.__setattr__(self, "intervals", intervals)
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
    resolved_game = replace(normalized_game, markets=resolved_markets)
    config_mapping = config.to_mapping()
    forecast_run_id = build_forecast_run_id(
        resolved_game,
        config.version,
        config_mapping,
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
        markets=markets,
    )


__all__ = ("ForecastOutput", "run_forecast")
