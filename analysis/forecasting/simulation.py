"""Deterministic one-generator MLB score simulation."""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
import math
import random
import re
from types import MappingProxyType

from .bullpen import (
    BullpenForecast,
    select_bullpen_segment,
)
from .config import SimulationConfig
from .inputs import BullpenRole
from .offense import OffenseForecast
from .randomness import draw_bounded_lognormal, draw_gamma_poisson
from .starter import PitcherChainForecast, SampledPitcherPlan


_FORECAST_RUN_ID = re.compile(r"[0-9a-f]{64}\Z")
_SIDES = ("away", "home")
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

__all__ = (
    "GameSimulationInput",
    "SimulatedGame",
    "DistributionSummary",
    "ComponentDistribution",
    "validate_component_distribution_domain",
    "SimulationResult",
    "mean_runs_for_half_inning",
    "simulate_game_distribution",
)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _require_finite_positive(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")


@dataclass(frozen=True)
class GameSimulationInput:
    away_offense: OffenseForecast
    home_offense: OffenseForecast
    away_pitcher_chain: PitcherChainForecast
    home_pitcher_chain: PitcherChainForecast
    away_bullpen: BullpenForecast
    home_bullpen: BullpenForecast

    def __post_init__(self) -> None:
        for side in _SIDES:
            if not isinstance(getattr(self, f"{side}_offense"), OffenseForecast):
                raise TypeError(f"{side}_offense must be OffenseForecast")
            if not isinstance(
                getattr(self, f"{side}_pitcher_chain"),
                PitcherChainForecast,
            ):
                raise TypeError(
                    f"{side}_pitcher_chain must be PitcherChainForecast"
                )
            if not isinstance(getattr(self, f"{side}_bullpen"), BullpenForecast):
                raise TypeError(f"{side}_bullpen must be BullpenForecast")

        opposing = (
            (self.away_offense, self.home_pitcher_chain, self.home_bullpen),
            (self.home_offense, self.away_pitcher_chain, self.away_bullpen),
        )
        for offense, chain, bullpen in opposing:
            pitcher_ids = tuple(pitcher.pitcher_id for pitcher in chain.pitchers)
            missing = set(pitcher_ids).difference(offense.matchup_multipliers)
            if missing:
                raise ValueError(
                    "offense matchup multipliers must include every opposing "
                    "pitcher-chain ID"
                )
            all_ids = pitcher_ids + tuple(
                segment.segment_id for segment in bullpen.segments
            ) + (bullpen.unscheduled_f5_relief.segment_id,)
            if len(set(all_ids)) != len(all_ids):
                raise ValueError(
                    "pitcher-chain and bullpen segment IDs must be unique per side"
                )


@dataclass(frozen=True)
class DistributionSummary:
    """Nearest-rank component quantiles without interpolation."""

    p05: float
    p10: float
    p20: float
    median: float
    p80: float
    p90: float
    p95: float

    def __post_init__(self) -> None:
        values = (
            self.p05,
            self.p10,
            self.p20,
            self.median,
            self.p80,
            self.p90,
            self.p95,
        )
        if any(not _is_finite_number(value) for value in values):
            raise ValueError("distribution summary values must be finite numbers")
        if any(first > second for first, second in zip(values, values[1:])):
            raise ValueError("distribution summary values must be nondecreasing")


@dataclass(frozen=True)
class ComponentDistribution:
    component_id: str
    values: tuple[int | float, ...]
    summary: DistributionSummary

    def __post_init__(self) -> None:
        _require_nonempty_string(self.component_id, "component_id")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("values must be a nonempty tuple")
        if any(not _is_finite_number(value) for value in self.values):
            raise ValueError("values must contain only finite numbers")
        if not isinstance(self.summary, DistributionSummary):
            raise TypeError("summary must be DistributionSummary")
        if self.summary != _summarize(self.values):
            raise ValueError("summary must match the nearest-rank values")


def validate_component_distribution_domain(
    distribution: ComponentDistribution,
) -> None:
    """Validate the count domain shared by every forecast component."""

    if not isinstance(distribution, ComponentDistribution):
        raise TypeError("distribution must be ComponentDistribution")
    if any(
        type(value) is not int or value < 0
        for value in distribution.values
    ):
        raise ValueError(
            f"component {distribution.component_id} count domain requires "
            "nonnegative integers"
        )
    binary_component = distribution.component_id.endswith(
        "_protected_bullpen_close_entry"
    ) or (
        "_bullpen_segment_" in distribution.component_id
        and distribution.component_id.endswith("_exposed")
    )
    if binary_component and any(
        value not in (0, 1) for value in distribution.values
    ):
        raise ValueError(
            f"binary component {distribution.component_id} requires values "
            "in {{0, 1}}"
        )


@dataclass(frozen=True)
class SimulatedGame:
    away_f5_runs: int
    home_f5_runs: int
    away_full_runs: int
    home_full_runs: int
    away_f5_traffic: int
    home_f5_traffic: int
    away_full_traffic: int
    home_full_traffic: int
    away_f5_extra_base_hits: int
    home_f5_extra_base_hits: int
    away_full_extra_base_hits: int
    home_full_extra_base_hits: int
    away_pitcher_chain_runs_allowed: int
    home_pitcher_chain_runs_allowed: int
    away_bullpen_runs_allowed: int
    home_bullpen_runs_allowed: int
    away_pitcher_chain_workload_outs: int
    home_pitcher_chain_workload_outs: int
    away_bullpen_segment_ids_exposed: tuple[str, ...]
    home_bullpen_segment_ids_exposed: tuple[str, ...]
    away_protected_bullpen_close_entry: bool
    home_protected_bullpen_close_entry: bool
    bottom_ninth_played: bool
    final_inning: int
    max_inning_tiebreak_used: bool

    def __post_init__(self) -> None:
        numeric_fields = (
            "away_f5_runs",
            "home_f5_runs",
            "away_full_runs",
            "home_full_runs",
            "away_f5_traffic",
            "home_f5_traffic",
            "away_full_traffic",
            "home_full_traffic",
            "away_f5_extra_base_hits",
            "home_f5_extra_base_hits",
            "away_full_extra_base_hits",
            "home_full_extra_base_hits",
            "away_pitcher_chain_runs_allowed",
            "home_pitcher_chain_runs_allowed",
            "away_bullpen_runs_allowed",
            "home_bullpen_runs_allowed",
            "away_pitcher_chain_workload_outs",
            "home_pitcher_chain_workload_outs",
        )
        for field_name in numeric_fields:
            _require_nonnegative_integer(getattr(self, field_name), field_name)
        for side in _SIDES:
            workload_outs = getattr(
                self,
                f"{side}_pitcher_chain_workload_outs",
            )
            if workload_outs > 27:
                raise ValueError(
                    "pitcher-chain workload outs must be within [0, 27]"
                )
        for side in _SIDES:
            if getattr(self, f"{side}_full_runs") < getattr(
                self, f"{side}_f5_runs"
            ):
                raise ValueError("full-game runs cannot be below F5 runs")
            if getattr(self, f"{side}_full_traffic") < getattr(
                self, f"{side}_f5_traffic"
            ):
                raise ValueError("full-game traffic cannot be below F5 traffic")
            if getattr(self, f"{side}_full_extra_base_hits") < getattr(
                self, f"{side}_f5_extra_base_hits"
            ):
                raise ValueError(
                    "full-game extra-base hits cannot be below F5 extra-base hits"
                )
            if getattr(self, f"{side}_f5_extra_base_hits") > getattr(
                self, f"{side}_f5_traffic"
            ) or getattr(self, f"{side}_full_extra_base_hits") > getattr(
                self, f"{side}_full_traffic"
            ):
                raise ValueError("extra-base hits cannot exceed traffic")
            exposed = getattr(self, f"{side}_bullpen_segment_ids_exposed")
            if not isinstance(exposed, tuple):
                raise ValueError(
                    "exposed bullpen IDs must be a canonical sorted unique tuple"
                )
            for segment_id in exposed:
                _require_nonempty_string(segment_id, "exposed bullpen segment ID")
            if (
                exposed != tuple(sorted(exposed))
                or len(exposed) != len(set(exposed))
            ):
                raise ValueError(
                    "exposed bullpen IDs must be a canonical sorted unique tuple"
                )
            protected = getattr(
                self, f"{side}_protected_bullpen_close_entry"
            )
            if not isinstance(protected, bool):
                raise TypeError("protected bullpen entry flags must be booleans")
        if not isinstance(self.bottom_ninth_played, bool):
            raise TypeError("bottom_ninth_played must be a boolean")
        if not isinstance(self.max_inning_tiebreak_used, bool):
            raise TypeError("max_inning_tiebreak_used must be a boolean")
        if (
            isinstance(self.final_inning, bool)
            or not isinstance(self.final_inning, int)
            or self.final_inning < 9
        ):
            raise ValueError("final_inning must be an integer of at least nine")
        if self.away_full_runs == self.home_full_runs:
            raise ValueError("a simulated full game must have a winner")
        if self.final_inning > 9 and not self.bottom_ninth_played:
            raise ValueError(
                "game state with extra innings requires the bottom ninth"
            )
        if (
            not self.bottom_ninth_played
            and self.home_full_runs <= self.away_full_runs
        ):
            raise ValueError(
                "game state with a skipped bottom ninth requires a regulation "
                "home win"
            )
        if (
            self.bottom_ninth_played
            and self.home_full_runs > self.away_full_runs
            and self.home_full_runs - self.away_full_runs != 1
        ):
            raise ValueError(
                "walkoff home wins must have an exact one-run margin"
            )
        if self.max_inning_tiebreak_used and (
            self.final_inning < 10
            or not self.bottom_ninth_played
            or abs(self.away_full_runs - self.home_full_runs) != 1
        ):
            raise ValueError(
                "game state with a max-inning tiebreak requires a one-run "
                "extra-inning terminal score"
            )
        if (
            self.home_pitcher_chain_runs_allowed
            + self.home_bullpen_runs_allowed
            != self.away_full_runs
            or self.away_pitcher_chain_runs_allowed
            + self.away_bullpen_runs_allowed
            != self.home_full_runs
        ):
            raise ValueError("pitching run attribution must conserve all runs")

    @property
    def away_exposed_bullpen_segment_ids(self) -> tuple[str, ...]:
        return self.away_bullpen_segment_ids_exposed

    @property
    def home_exposed_bullpen_segment_ids(self) -> tuple[str, ...]:
        return self.home_bullpen_segment_ids_exposed


@dataclass(frozen=True)
class SimulationResult:
    forecast_run_id: str
    simulation_count: int
    games: tuple[SimulatedGame, ...]
    component_distributions: Mapping[str, ComponentDistribution]

    def __post_init__(self) -> None:
        max_innings, available_entries = _require_simulation_result_context()
        available_by_side = {
            side: dict(segment_entries)
            for side, segment_entries in available_entries
        }
        protected_roles = {
            BullpenRole.HIGH_LEVERAGE.value,
            BullpenRole.CLOSER.value,
        }
        _validate_forecast_run_id(self.forecast_run_id)
        if (
            isinstance(self.simulation_count, bool)
            or not isinstance(self.simulation_count, int)
            or self.simulation_count <= 0
        ):
            raise ValueError("simulation_count must be a positive integer")
        if not isinstance(self.games, tuple) or len(self.games) != self.simulation_count:
            raise ValueError("games must be a tuple matching simulation_count")
        if any(not isinstance(game, SimulatedGame) for game in self.games):
            raise TypeError("games must contain SimulatedGame values")
        for game in self.games:
            if game.final_inning > max_innings:
                raise ValueError(
                    "game final_inning cannot exceed configured max_innings"
                )
            if (
                game.max_inning_tiebreak_used
                and game.final_inning != max_innings
            ):
                raise ValueError(
                    "max-inning tiebreak final_inning must equal config.max_innings"
                )
            for side in _SIDES:
                exposed = getattr(
                    game,
                    f"{side}_bullpen_segment_ids_exposed",
                )
                if not set(exposed).issubset(available_by_side[side]):
                    raise ValueError(
                        "game exposure IDs must reference available bullpen "
                        "segments"
                    )
                if getattr(
                    game,
                    f"{side}_protected_bullpen_close_entry",
                ) and not any(
                    available_by_side[side][segment_id] in protected_roles
                    for segment_id in exposed
                ):
                    raise ValueError(
                        "protected bullpen entry requires an exposed available "
                        "high-leverage or closer segment"
                    )
        if not isinstance(self.component_distributions, Mapping):
            raise TypeError("component_distributions must be a mapping")
        copied = dict(self.component_distributions)
        for component_id, distribution in copied.items():
            if not isinstance(distribution, ComponentDistribution):
                raise TypeError(
                    "component_distributions must contain ComponentDistribution values"
                )
            if distribution.component_id != component_id:
                raise ValueError("component distribution key and ID must match")
            if not isinstance(distribution.values, tuple):
                raise ValueError("component values must be a tuple")
            if len(distribution.values) != self.simulation_count:
                raise ValueError("component values must match simulation_count")
            if not isinstance(distribution.summary, DistributionSummary):
                raise TypeError("component summary must be DistributionSummary")
        expected = _expected_component_distributions(
            self.games,
            available_entries,
        )
        if set(copied) != set(expected):
            raise ValueError(
                "component distributions must contain exactly the canonical IDs"
            )
        for component_id, expected_distribution in expected.items():
            if not _component_distribution_matches_exactly(
                copied[component_id],
                expected_distribution,
            ):
                raise ValueError(
                    "component distributions must exactly match game-derived "
                    "values and summaries"
                )
        object.__setattr__(
            self,
            "component_distributions",
            MappingProxyType(copied),
        )


@dataclass(frozen=True)
class _OffenseLatents:
    base_offense_multiplier: float
    matchup_multipliers: Mapping[str, float]
    diagnostic_uncertainty: float


@dataclass(frozen=True)
class _PitchingLatents:
    pitcher_run_prevention: Mapping[str, float]
    bullpen_run_prevention: Mapping[str, float]


@dataclass(frozen=True)
class _PitchingEffect:
    segment_id: str
    out_fraction: float
    matchup_multiplier: float
    run_prevention_multiplier: float
    pitcher_chain: bool
    exposed_bullpen: bool
    protected_close_entry: bool


@dataclass(frozen=True)
class _HalfInningOutcome:
    runs: int
    pitcher_chain_runs: int
    bullpen_runs: int
    traffic: int
    extra_base_hits: int
    mean_runs: float
    exposed_bullpen_ids: tuple[str, ...]
    protected_close_entry: bool


def mean_runs_for_half_inning(
    *,
    league_inning_rate: float,
    offense_multiplier: float,
    matchup_multiplier: float,
    run_prevention_multiplier: float,
    out_fraction: float = 1.0,
) -> float:
    """Return one segment's contribution to a half-inning run mean."""

    for value, field_name in (
        (league_inning_rate, "league_inning_rate"),
        (offense_multiplier, "offense_multiplier"),
        (matchup_multiplier, "matchup_multiplier"),
        (run_prevention_multiplier, "run_prevention_multiplier"),
    ):
        _require_finite_positive(value, field_name)
    if not _is_finite_number(out_fraction) or not 0.0 <= out_fraction <= 1.0:
        raise ValueError("out_fraction must be a finite number in [0, 1]")
    if out_fraction == 0:
        return 0.0
    mean = (
        league_inning_rate
        * offense_multiplier
        * out_fraction
        * matchup_multiplier
        * run_prevention_multiplier
    )
    if not math.isfinite(mean):
        raise ValueError("half-inning mean must remain finite")
    return mean


class _ScriptedGammaPoissonRng:
    """Replay one pre-sampled Gamma-Poisson outcome through the shared helper."""

    def __init__(self, latent: float, count: int) -> None:
        self._latent = latent
        self._count = count
        self._random_calls = 0

    def gammavariate(self, _shape: float, _scale: float) -> float:
        return self._latent

    def random(self) -> float:
        self._random_calls += 1
        return 1.0 if self._random_calls <= self._count else 0.0


def _poisson_quantile(mean: float, sample: float) -> int:
    """Invert a Poisson CDF using one uniform sample."""

    if mean == 0:
        return 0
    probability = math.exp(-mean)
    if probability > 0:
        cumulative = probability
        count = 0
        while sample >= cumulative:
            count += 1
            probability *= mean / count
            cumulative += probability
        return count

    mode = math.floor(mean)
    if mode > 100_000:
        raise ValueError("gamma latent is too large for a stable Poisson draw")
    mode_probability = math.exp(
        -mean + mode * math.log(mean) - math.lgamma(mode + 1)
    )
    lower_probabilities = [(mode, mode_probability)]
    probability = mode_probability
    for count in range(mode, 0, -1):
        probability *= count / mean
        lower_probabilities.append((count - 1, probability))
    cumulative = math.fsum(
        probability for _count, probability in lower_probabilities
    )
    if sample < cumulative:
        running = 0.0
        for count, probability in reversed(lower_probabilities):
            running += probability
            if sample < running:
                return count
        return mode

    count = mode
    probability = mode_probability
    while sample >= cumulative:
        count += 1
        probability *= mean / count
        cumulative += probability
        if probability == 0:
            return count
    return count


def _binomial_quantile(trials: int, probability: float, sample: float) -> int:
    """Aggregate independent Bernoulli shares using one uniform sample."""

    if trials == 0 or probability == 0:
        return 0
    if probability == 1:
        return trials
    if sample <= 0:
        return 0
    if sample >= 1:
        return trials

    mode = min(trials, math.floor((trials + 1) * probability))
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    log_weights = [-math.inf] * (trials + 1)
    log_weights[mode] = 0.0
    for successes in range(mode, 0, -1):
        log_weights[successes - 1] = (
            log_weights[successes]
            + math.log(successes)
            - math.log(trials - successes + 1)
            + log_complement
            - log_probability
        )
    for successes in range(mode, trials):
        log_weights[successes + 1] = (
            log_weights[successes]
            + math.log(trials - successes)
            - math.log(successes + 1)
            + log_probability
            - log_complement
        )

    weights = tuple(math.exp(log_weight) for log_weight in log_weights)
    threshold = sample * math.fsum(weights)
    cumulative = 0.0
    for successes, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return successes
    return trials


def _fixed_gamma_poisson_source(
    rng: random.Random,
    *,
    mean: float,
    shape: float,
) -> _ScriptedGammaPoissonRng:
    if not _is_finite_number(mean) or mean < 0:
        raise ValueError("mean must be a finite nonnegative number")
    _require_finite_positive(shape, "shape")
    scale = mean / shape if mean else 1.0
    latent_draw = rng.gammavariate(shape, scale)
    if not _is_finite_number(latent_draw) or latent_draw < 0:
        raise ValueError("gamma latent must be finite and nonnegative")
    sample = rng.random()
    if not _is_finite_number(sample) or not 0.0 <= sample <= 1.0:
        raise ValueError("rng.random result must be finite and in [0, 1]")
    latent = latent_draw if mean else 0.0
    return _ScriptedGammaPoissonRng(
        latent,
        _poisson_quantile(latent, sample),
    )


def _draw_run_count(rng: object, *, mean: float, shape: float) -> int:
    return draw_gamma_poisson(
        _fixed_gamma_poisson_source(rng, mean=mean, shape=shape),
        mean=mean,
        shape=shape,
    )


def _draw_diagnostic_counts(
    rng: object,
    *,
    traffic_mean: float,
    extra_base_hit_mean: float,
    shape: float,
) -> tuple[int, int]:
    traffic = draw_gamma_poisson(
        _fixed_gamma_poisson_source(rng, mean=traffic_mean, shape=shape),
        mean=traffic_mean,
        shape=shape,
    )
    raw_extra_base_hits = draw_gamma_poisson(
        _fixed_gamma_poisson_source(
            rng,
            mean=extra_base_hit_mean,
            shape=shape,
        ),
        mean=extra_base_hit_mean,
        shape=shape,
    )
    return traffic, min(traffic, raw_extra_base_hits)


def _validate_forecast_run_id(forecast_run_id: object) -> None:
    if not isinstance(forecast_run_id, str):
        raise TypeError("forecast_run_id must be a string")
    if _FORECAST_RUN_ID.fullmatch(forecast_run_id) is None:
        raise ValueError("forecast_run_id must be exactly 64 lowercase hex characters")


def _draw_offense_latents(
    offense: OffenseForecast,
    opposing_chain: PitcherChainForecast,
    opposing_bullpen: BullpenForecast,
    rng: random.Random,
    diagnostic_rng: random.Random,
    config: SimulationConfig,
) -> _OffenseLatents:
    base = draw_bounded_lognormal(
        rng,
        offense.base_offense_multiplier,
        offense.log_standard_error,
        *config.offense_bounds,
    )
    applicable_ids = tuple(
        pitcher.pitcher_id for pitcher in opposing_chain.pitchers
    ) + tuple(
        segment.segment_id
        for segment in opposing_bullpen.segments
    ) + (opposing_bullpen.unscheduled_f5_relief.segment_id,)
    matchups = {
        segment_id: draw_bounded_lognormal(
            rng,
            offense.multiplier_for(segment_id),
            offense.log_standard_error_for(segment_id),
            *config.matchup_bounds,
        )
        for segment_id in applicable_ids
    }
    diagnostic_uncertainty = draw_bounded_lognormal(
        diagnostic_rng,
        1.0,
        offense.lineup_log_standard_error,
        *config.offense_bounds,
    )
    return _OffenseLatents(
        base_offense_multiplier=base,
        matchup_multipliers=MappingProxyType(matchups),
        diagnostic_uncertainty=diagnostic_uncertainty,
    )


def _draw_pitching_latents(
    chain: PitcherChainForecast,
    bullpen: BullpenForecast,
    rng: random.Random,
    config: SimulationConfig,
) -> _PitchingLatents:
    pitcher_values = {
        pitcher.pitcher_id: draw_bounded_lognormal(
            rng,
            pitcher.run_prevention_multiplier,
            pitcher.log_standard_error,
            *config.starter_ra9_bounds,
        )
        for pitcher in chain.pitchers
    }
    bullpen_segments = bullpen.segments + (bullpen.unscheduled_f5_relief,)
    bullpen_values = {
        segment.segment_id: draw_bounded_lognormal(
            rng,
            segment.run_prevention_multiplier,
            segment.log_standard_error,
            *config.bullpen_ra9_bounds,
        )
        for segment in bullpen_segments
    }
    return _PitchingLatents(
        pitcher_run_prevention=MappingProxyType(pitcher_values),
        bullpen_run_prevention=MappingProxyType(bullpen_values),
    )


def _pitching_effects_for_half_inning(
    *,
    inning: int,
    margin: int,
    chain: PitcherChainForecast,
    plan: SampledPitcherPlan,
    bullpen: BullpenForecast,
    offense_latents: _OffenseLatents,
    pitching_latents: _PitchingLatents,
) -> tuple[_PitchingEffect, ...]:
    start_out = (inning - 1) * 3
    end_out = start_out + 3
    effects: list[_PitchingEffect] = []
    covered_outs = 0
    handoff_verified = "handoff_unverified" not in chain.diagnostics
    active_segments = (
        plan.segments
        if len(chain.pitchers) == 1 or handoff_verified
        else plan.segments[:1]
    )

    for segment in active_segments:
        overlap = max(
            0,
            min(end_out, segment.end_out) - max(start_out, segment.start_out),
        )
        if overlap == 0:
            continue
        covered_outs += overlap
        effects.append(
            _PitchingEffect(
                segment_id=segment.pitcher_id,
                out_fraction=overlap / 3,
                matchup_multiplier=offense_latents.matchup_multipliers[
                    segment.pitcher_id
                ],
                run_prevention_multiplier=(
                    pitching_latents.pitcher_run_prevention[segment.pitcher_id]
                ),
                pitcher_chain=True,
                exposed_bullpen=False,
                protected_close_entry=False,
            )
        )

    remaining_outs = 3 - covered_outs
    if remaining_outs:
        if inning <= 5:
            relief = bullpen.unscheduled_f5_relief
        else:
            relief = select_bullpen_segment(
                bullpen,
                inning=inning,
                margin=margin,
                bridge_needed=(
                    active_segments[-1].end_out < 18 and inning <= 7
                ),
                verified_opener_bulk=(
                    len(chain.pitchers) > 1 and handoff_verified
                ),
            )
        close_entry = (
            relief.role in (BullpenRole.HIGH_LEVERAGE, BullpenRole.CLOSER)
            and abs(margin) <= 3
            and not relief.synthetic
        )
        effects.append(
            _PitchingEffect(
                segment_id=relief.segment_id,
                out_fraction=remaining_outs / 3,
                matchup_multiplier=offense_latents.matchup_multipliers[
                    relief.segment_id
                ],
                run_prevention_multiplier=(
                    pitching_latents.bullpen_run_prevention[relief.segment_id]
                ),
                pitcher_chain=False,
                exposed_bullpen=not relief.synthetic,
                protected_close_entry=close_entry,
            )
        )
    if not math.isclose(
        math.fsum(effect.out_fraction for effect in effects),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("pitching effects must cover exactly three outs")
    return tuple(effects)


def _simulate_half_inning(
    *,
    inning: int,
    margin: int,
    chain: PitcherChainForecast,
    plan: SampledPitcherPlan,
    bullpen: BullpenForecast,
    offense: OffenseForecast,
    offense_latents: _OffenseLatents,
    pitching_latents: _PitchingLatents,
    rng: random.Random,
    diagnostic_rng: random.Random,
    config: SimulationConfig,
    walkoff_runs_needed: int | None = None,
) -> _HalfInningOutcome:
    effects = _pitching_effects_for_half_inning(
        inning=inning,
        margin=margin,
        chain=chain,
        plan=plan,
        bullpen=bullpen,
        offense_latents=offense_latents,
        pitching_latents=pitching_latents,
    )
    league_inning_rate = config.league_runs_per_team_game / 9
    if inning >= 10:
        league_inning_rate *= config.extra_inning_multiplier
    contributions = tuple(
        mean_runs_for_half_inning(
            league_inning_rate=league_inning_rate,
            offense_multiplier=offense_latents.base_offense_multiplier,
            matchup_multiplier=effect.matchup_multiplier,
            run_prevention_multiplier=effect.run_prevention_multiplier,
            out_fraction=effect.out_fraction,
        )
        for effect in effects
    )
    mean_runs = math.fsum(contributions)
    raw_runs = _draw_run_count(
        rng,
        mean=mean_runs,
        shape=config.gamma_poisson_shape,
    )
    runs = (
        raw_runs
        if walkoff_runs_needed is None
        else min(raw_runs, walkoff_runs_needed)
    )
    attribution_sample = rng.random()

    chain_intensity = math.fsum(
        contribution
        for effect, contribution in zip(effects, contributions)
        if effect.pitcher_chain
    )
    if chain_intensity == 0:
        pitcher_chain_runs = 0
    elif math.isclose(chain_intensity, mean_runs, rel_tol=0.0, abs_tol=1e-15):
        pitcher_chain_runs = runs
    else:
        chain_share = chain_intensity / mean_runs
        pitcher_chain_runs = _binomial_quantile(
            runs,
            chain_share,
            attribution_sample,
        )

    diagnostic_matchup = math.fsum(
        effect.out_fraction * effect.matchup_multiplier for effect in effects
    )
    traffic, extra_base_hits = _draw_diagnostic_counts(
        diagnostic_rng,
        traffic_mean=(
            offense.projected_traffic_per_9
            / 9
            * diagnostic_matchup
            * offense_latents.diagnostic_uncertainty
        ),
        extra_base_hit_mean=(
            offense.projected_extra_base_hits_per_9
            / 9
            * diagnostic_matchup
            * offense_latents.diagnostic_uncertainty
        ),
        shape=config.gamma_poisson_shape,
    )
    exposed_ids = tuple(
        effect.segment_id for effect in effects if effect.exposed_bullpen
    )
    return _HalfInningOutcome(
        runs=runs,
        pitcher_chain_runs=pitcher_chain_runs,
        bullpen_runs=runs - pitcher_chain_runs,
        traffic=traffic,
        extra_base_hits=extra_base_hits,
        mean_runs=mean_runs,
        exposed_bullpen_ids=exposed_ids,
        protected_close_entry=any(
            effect.protected_close_entry for effect in effects
        ),
    )


def _consume_unused_half_inning_draws(
    rng: random.Random,
    diagnostic_rng: random.Random,
    shape: float,
) -> None:
    # Every possible half-inning reserves the same parent-stream footprint.
    rng.gammavariate(shape, 1.0)
    rng.random()
    rng.random()
    for _ in range(2):
        diagnostic_rng.gammavariate(shape, 1.0)
        diagnostic_rng.random()


def _simulate_one_game(
    simulation_input: GameSimulationInput,
    rng: random.Random,
    diagnostic_rng: random.Random,
    config: SimulationConfig,
) -> SimulatedGame:
    away_plan = simulation_input.away_pitcher_chain.sample_plan(rng)
    home_plan = simulation_input.home_pitcher_chain.sample_plan(rng)
    away_offense_latents = _draw_offense_latents(
        simulation_input.away_offense,
        simulation_input.home_pitcher_chain,
        simulation_input.home_bullpen,
        rng,
        diagnostic_rng,
        config,
    )
    home_offense_latents = _draw_offense_latents(
        simulation_input.home_offense,
        simulation_input.away_pitcher_chain,
        simulation_input.away_bullpen,
        rng,
        diagnostic_rng,
        config,
    )
    away_pitching_latents = _draw_pitching_latents(
        simulation_input.away_pitcher_chain,
        simulation_input.away_bullpen,
        rng,
        config,
    )
    home_pitching_latents = _draw_pitching_latents(
        simulation_input.home_pitcher_chain,
        simulation_input.home_bullpen,
        rng,
        config,
    )

    scores = {"away": 0, "home": 0}
    traffic = {"away": 0, "home": 0}
    extra_base_hits = {"away": 0, "home": 0}
    chain_runs_allowed = {"away": 0, "home": 0}
    bullpen_runs_allowed = {"away": 0, "home": 0}
    exposed_ids = {"away": set(), "home": set()}
    protected_entry = {"away": False, "home": False}
    f5_scores = {"away": 0, "home": 0}
    f5_traffic = {"away": 0, "home": 0}
    f5_extra_base_hits = {"away": 0, "home": 0}
    bottom_ninth_played = True
    max_inning_tiebreak_used = False
    max_tiebreak_intensities: tuple[float, float] | None = None
    played_half_innings = 0

    def play_half(inning: int, batting_side: str, walkoff: bool = False):
        nonlocal played_half_innings
        pitching_side = "home" if batting_side == "away" else "away"
        offense = getattr(simulation_input, f"{batting_side}_offense")
        offense_latents = (
            away_offense_latents if batting_side == "away" else home_offense_latents
        )
        chain = getattr(simulation_input, f"{pitching_side}_pitcher_chain")
        plan = home_plan if pitching_side == "home" else away_plan
        bullpen = getattr(simulation_input, f"{pitching_side}_bullpen")
        pitching_latents = (
            home_pitching_latents
            if pitching_side == "home"
            else away_pitching_latents
        )
        runs_needed = (
            scores[pitching_side] - scores[batting_side] + 1
            if walkoff
            else None
        )
        outcome = _simulate_half_inning(
            inning=inning,
            margin=scores[batting_side] - scores[pitching_side],
            chain=chain,
            plan=plan,
            bullpen=bullpen,
            offense=offense,
            offense_latents=offense_latents,
            pitching_latents=pitching_latents,
            rng=rng,
            diagnostic_rng=diagnostic_rng,
            config=config,
            walkoff_runs_needed=runs_needed,
        )
        played_half_innings += 1
        scores[batting_side] += outcome.runs
        traffic[batting_side] += outcome.traffic
        extra_base_hits[batting_side] += outcome.extra_base_hits
        chain_runs_allowed[pitching_side] += outcome.pitcher_chain_runs
        bullpen_runs_allowed[pitching_side] += outcome.bullpen_runs
        exposed_ids[pitching_side].update(outcome.exposed_bullpen_ids)
        protected_entry[pitching_side] = (
            protected_entry[pitching_side] or outcome.protected_close_entry
        )
        return outcome

    for inning in range(1, 10):
        play_half(inning, "away")
        if inning == 9 and scores["home"] > scores["away"]:
            bottom_ninth_played = False
            break
        play_half(inning, "home", walkoff=inning >= 9)
        if inning == 5:
            f5_scores = dict(scores)
            f5_traffic = dict(traffic)
            f5_extra_base_hits = dict(extra_base_hits)

    final_inning = 9
    if scores["away"] == scores["home"]:
        for inning in range(10, config.max_innings + 1):
            away_outcome = play_half(inning, "away")
            home_outcome = play_half(inning, "home", walkoff=True)
            final_inning = inning
            if scores["away"] != scores["home"]:
                break
            if inning == config.max_innings:
                max_inning_tiebreak_used = True
                max_tiebreak_intensities = (
                    away_outcome.mean_runs,
                    home_outcome.mean_runs,
                )

    for _ in range(config.max_innings * 2 - played_half_innings):
        _consume_unused_half_inning_draws(
            rng,
            diagnostic_rng,
            config.gamma_poisson_shape,
        )
    tiebreak_sample = rng.random()
    if max_tiebreak_intensities is not None:
        away_intensity, home_intensity = max_tiebreak_intensities
        total_intensity = away_intensity + home_intensity
        if tiebreak_sample < away_intensity / total_intensity:
            scores["away"] += 1
            bullpen_runs_allowed["home"] += 1
        else:
            scores["home"] += 1
            bullpen_runs_allowed["away"] += 1

    return SimulatedGame(
        away_f5_runs=f5_scores["away"],
        home_f5_runs=f5_scores["home"],
        away_full_runs=scores["away"],
        home_full_runs=scores["home"],
        away_f5_traffic=f5_traffic["away"],
        home_f5_traffic=f5_traffic["home"],
        away_full_traffic=traffic["away"],
        home_full_traffic=traffic["home"],
        away_f5_extra_base_hits=f5_extra_base_hits["away"],
        home_f5_extra_base_hits=f5_extra_base_hits["home"],
        away_full_extra_base_hits=extra_base_hits["away"],
        home_full_extra_base_hits=extra_base_hits["home"],
        away_pitcher_chain_runs_allowed=chain_runs_allowed["away"],
        home_pitcher_chain_runs_allowed=chain_runs_allowed["home"],
        away_bullpen_runs_allowed=bullpen_runs_allowed["away"],
        home_bullpen_runs_allowed=bullpen_runs_allowed["home"],
        away_pitcher_chain_workload_outs=away_plan.combined_outs,
        home_pitcher_chain_workload_outs=home_plan.combined_outs,
        away_bullpen_segment_ids_exposed=tuple(sorted(exposed_ids["away"])),
        home_bullpen_segment_ids_exposed=tuple(sorted(exposed_ids["home"])),
        away_protected_bullpen_close_entry=protected_entry["away"],
        home_protected_bullpen_close_entry=protected_entry["home"],
        bottom_ninth_played=bottom_ninth_played,
        final_inning=final_inning,
        max_inning_tiebreak_used=max_inning_tiebreak_used,
    )


def _nearest_rank(values: tuple[int | float, ...], probability: float):
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(probability * len(ordered))))
    return ordered[rank - 1]


def _summarize(values: tuple[int | float, ...]) -> DistributionSummary:
    return DistributionSummary(
        p05=_nearest_rank(values, 0.05),
        p10=_nearest_rank(values, 0.10),
        p20=_nearest_rank(values, 0.20),
        median=_nearest_rank(values, 0.50),
        p80=_nearest_rank(values, 0.80),
        p90=_nearest_rank(values, 0.90),
        p95=_nearest_rank(values, 0.95),
    )


def _summary_values(summary: DistributionSummary) -> tuple[int | float, ...]:
    return (
        summary.p05,
        summary.p10,
        summary.p20,
        summary.median,
        summary.p80,
        summary.p90,
        summary.p95,
    )


def _component_distribution_matches_exactly(
    actual: ComponentDistribution,
    expected: ComponentDistribution,
) -> bool:
    try:
        validate_component_distribution_domain(actual)
    except (TypeError, ValueError):
        return False
    if actual.component_id != expected.component_id:
        return False
    if len(actual.values) != len(expected.values):
        return False
    if any(
        type(actual_value) is not type(expected_value)
        or actual_value != expected_value
        for actual_value, expected_value in zip(actual.values, expected.values)
    ):
        return False
    return all(
        type(actual_value) is type(expected_value)
        and actual_value == expected_value
        for actual_value, expected_value in zip(
            _summary_values(actual.summary),
            _summary_values(expected.summary),
        )
    )


def _available_bullpen_entries(
    simulation_input: GameSimulationInput,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return tuple(
        (
            side,
            tuple(
                (segment.segment_id, segment.role.value)
                for segment in getattr(
                    simulation_input,
                    f"{side}_bullpen",
                ).available_segments
            ),
        )
        for side in _SIDES
    )


def _expected_component_distributions(
    games: tuple[SimulatedGame, ...],
    available_entries: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
) -> Mapping[str, ComponentDistribution]:
    available_by_side = dict(available_entries)
    values_by_id: dict[str, tuple[int | float, ...]] = {}
    for side in _SIDES:
        for suffix in _BASE_COMPONENT_SUFFIXES:
            component_id = f"{side}_{suffix}"
            values_by_id[component_id] = tuple(
                int(getattr(game, component_id))
                if suffix == "protected_bullpen_close_entry"
                else getattr(game, component_id)
                for game in games
            )
        exposed_field = f"{side}_bullpen_segment_ids_exposed"
        for segment_id, _role in available_by_side[side]:
            component_id = f"{side}_bullpen_segment_{segment_id}_exposed"
            values_by_id[component_id] = tuple(
                int(segment_id in getattr(game, exposed_field))
                for game in games
            )
    return MappingProxyType(
        {
            component_id: ComponentDistribution(
                component_id=component_id,
                values=values,
                summary=_summarize(values),
            )
            for component_id, values in values_by_id.items()
        }
    )


def _component_distributions(
    games: tuple[SimulatedGame, ...],
    simulation_input: GameSimulationInput,
) -> Mapping[str, ComponentDistribution]:
    return _expected_component_distributions(
        games,
        _available_bullpen_entries(simulation_input),
    )


def _simulate_game_distribution_impl(
    input: GameSimulationInput,
    forecast_run_id: str,
    config: SimulationConfig,
    issue_result,
) -> SimulationResult:
    """Simulate one distribution whose seeded game prefix is batch-size stable."""

    if not isinstance(input, GameSimulationInput):
        raise TypeError("input must be GameSimulationInput")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")
    try:
        validated_config = SimulationConfig.from_mapping(config.to_mapping())
    except (TypeError, ValueError) as error:
        raise ValueError(f"config is invalid: {error}") from error
    if validated_config != config:
        raise ValueError("config is invalid or noncanonical")
    _validate_forecast_run_id(forecast_run_id)
    seed = int(forecast_run_id[:16], 16)
    score_rng = random.Random(seed)
    # Diagnostics share the canonical seed but never consume the score stream.
    diagnostic_rng = random.Random(seed)
    games = tuple(
        _simulate_one_game(
            input,
            score_rng,
            diagnostic_rng,
            config,
        )
        for _ in range(config.simulations)
    )
    available_entries = _available_bullpen_entries(input)
    return issue_result(
        forecast_run_id=forecast_run_id,
        simulation_count=config.simulations,
        games=games,
        component_distributions=_component_distributions(games, input),
        max_innings=config.max_innings,
        available_entries=available_entries,
    )


def _build_simulate_game_distribution():
    active_context = ContextVar("simulation_result_context", default=None)

    def issue_result(
        *,
        forecast_run_id: str,
        simulation_count: int,
        games: tuple[SimulatedGame, ...],
        component_distributions: Mapping[str, ComponentDistribution],
        max_innings: int,
        available_entries: tuple[
            tuple[str, tuple[tuple[str, str], ...]],
            ...,
        ],
    ) -> SimulationResult:
        if (
            isinstance(max_innings, bool)
            or not isinstance(max_innings, int)
            or max_innings < 10
        ):
            raise ValueError("result context max_innings must be at least ten")
        canonical_entries = tuple(
            (side, tuple(sorted(segment_entries)))
            for side, segment_entries in available_entries
        )
        if tuple(side for side, _ids in canonical_entries) != _SIDES:
            raise ValueError("result context must contain away and home metadata")
        valid_roles = {role.value for role in BullpenRole}
        for _side, segment_entries in canonical_entries:
            segment_ids = tuple(
                segment_id for segment_id, _role in segment_entries
            )
            if len(segment_ids) != len(set(segment_ids)):
                raise ValueError("result context bullpen IDs must be unique")
            for segment_id, role in segment_entries:
                _require_nonempty_string(segment_id, "result context bullpen ID")
                if role not in valid_roles:
                    raise ValueError("result context bullpen role must be canonical")
        context = (max_innings, canonical_entries)
        token = active_context.set(context)
        try:
            return SimulationResult(
                forecast_run_id=forecast_run_id,
                simulation_count=simulation_count,
                games=games,
                component_distributions=component_distributions,
            )
        finally:
            active_context.reset(token)

    def require_context():
        context = active_context.get()
        if context is None:
            raise ValueError(
                "SimulationResult must be constructed by the simulation factory"
            )
        return context

    def simulate_game_distribution(
        input: GameSimulationInput,
        forecast_run_id: str,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Simulate a seeded distribution with a batch-size-stable prefix."""

        return _simulate_game_distribution_impl(
            input,
            forecast_run_id,
            config,
            issue_result,
        )

    return simulate_game_distribution, require_context


simulate_game_distribution, _require_simulation_result_context = (
    _build_simulate_game_distribution()
)
del _build_simulate_game_distribution
