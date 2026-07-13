"""Starter and opener/bulk workload forecasts."""

from dataclasses import dataclass
import math
from statistics import median_low

from .config import SimulationConfig
from .inputs import PitcherChainInput, PitcherInput
from .randomness import weighted_choice


_BUCKET_NAMES = ("under_five", "five", "six_plus")
_WORKLOAD_SOURCES = frozenset({"pitcher_history", "combined_chain_history"})
_PITCHER_ROLES = frozenset({"starter", "opener", "bulk", "prim"})
_PROBABILITY_TOLERANCE = 1e-12


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_finite_nonnegative(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite nonnegative number")


def _require_diagnostics(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    for diagnostic in value:
        _require_nonempty_string(diagnostic, field_name)


def _require_out(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must contain only integer outs")
    if not 0 <= value <= 27:
        raise ValueError(f"{field_name} outs must be within [0, 27]")


def _bucket_index(outs: int) -> int:
    if outs < 15:
        return 0
    if outs < 18:
        return 1
    return 2


@dataclass(frozen=True)
class WorkloadDistribution:
    probabilities: tuple[float, float, float]
    supports: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    source: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, tuple) or len(self.probabilities) != 3:
            raise ValueError("probabilities must be a tuple of exactly three values")
        for probability in self.probabilities:
            if not _is_finite_number(probability) or probability < 0:
                raise ValueError(
                    "probabilities must contain finite nonnegative numbers"
                )
        if not math.isclose(
            math.fsum(self.probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("probabilities must sum to one")

        if not isinstance(self.supports, tuple) or len(self.supports) != 3:
            raise ValueError("supports must be a tuple of exactly three buckets")
        for bucket_index, support in enumerate(self.supports):
            if not isinstance(support, tuple) or not support:
                raise ValueError("each workload support must be a nonempty tuple")
            for outs in support:
                _require_out(outs, "supports")
                if _bucket_index(outs) != bucket_index:
                    raise ValueError(
                        f"supports[{bucket_index}] must contain only "
                        f"{_BUCKET_NAMES[bucket_index]} outs"
                    )

        if not isinstance(self.source, str) or self.source not in _WORKLOAD_SOURCES:
            raise ValueError(
                "source must be pitcher_history or combined_chain_history"
            )
        _require_diagnostics(self.notes, "notes")

    @property
    def bucket_probabilities(self) -> tuple[float, float, float]:
        return self.probabilities

    @property
    def bucket_supports(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.supports

    @property
    def probability_under_five(self) -> float:
        return self.probabilities[0]

    @property
    def probability_five(self) -> float:
        return self.probabilities[1]

    @property
    def probability_six_plus(self) -> float:
        return self.probabilities[2]

    @property
    def support_under_five(self) -> tuple[int, ...]:
        return self.supports[0]

    @property
    def support_five(self) -> tuple[int, ...]:
        return self.supports[1]

    @property
    def support_six_plus(self) -> tuple[int, ...]:
        return self.supports[2]

    @property
    def expected_outs(self) -> float:
        bucket_expectations = (
            math.fsum(support) / len(support)
            for support in self.supports
        )
        return math.fsum(
            probability * expectation
            for probability, expectation in zip(
                self.probabilities,
                bucket_expectations,
            )
        )

    def sample_outs(self, rng: object) -> int:
        bucket_index = weighted_choice(
            rng,
            (0, 1, 2),
            self.probabilities,
        )
        support = self.supports[bucket_index]
        return weighted_choice(rng, support, (1.0,) * len(support))


@dataclass(frozen=True)
class PitcherForecast:
    pitcher_id: str
    role: str
    workload: WorkloadDistribution
    run_prevention_multiplier: float
    log_standard_error: float
    command_risk: bool
    home_run_risk: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        if not isinstance(self.role, str) or self.role not in _PITCHER_ROLES:
            raise ValueError("role must be starter, opener, bulk, or prim")
        if not isinstance(self.workload, WorkloadDistribution):
            raise ValueError("workload must be WorkloadDistribution")
        if (
            not _is_finite_number(self.run_prevention_multiplier)
            or self.run_prevention_multiplier <= 0
        ):
            raise ValueError(
                "run_prevention_multiplier must be a finite positive number"
            )
        _require_finite_nonnegative(
            self.log_standard_error,
            "log_standard_error",
        )
        if not isinstance(self.command_risk, bool):
            raise ValueError("command_risk must be a boolean")
        if not isinstance(self.home_run_risk, bool):
            raise ValueError("home_run_risk must be a boolean")
        _require_diagnostics(self.diagnostics, "diagnostics")

    @property
    def multiplier(self) -> float:
        return self.run_prevention_multiplier


@dataclass(frozen=True)
class PitcherPlanSegment:
    pitcher_id: str
    role: str
    start_out: int
    end_out: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        if not isinstance(self.role, str) or self.role not in _PITCHER_ROLES:
            raise ValueError("role must be starter, opener, bulk, or prim")
        _require_out(self.start_out, "start_out")
        _require_out(self.end_out, "end_out")
        if self.end_out < self.start_out:
            raise ValueError("end_out must be greater than or equal to start_out")

    @property
    def outs(self) -> int:
        return self.end_out - self.start_out


@dataclass(frozen=True)
class SampledPitcherPlan:
    segments: tuple[PitcherPlanSegment, ...]
    combined_outs: int

    def __post_init__(self) -> None:
        _require_out(self.combined_outs, "combined_outs")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("segments must be a nonempty tuple")
        for segment in self.segments:
            if not isinstance(segment, PitcherPlanSegment):
                raise ValueError("segments must contain PitcherPlanSegment values")

        if self.segments[0].start_out != 0:
            raise ValueError("the first segment must start at out zero")
        previous_end = 0
        for segment in self.segments:
            if segment.start_out != previous_end:
                raise ValueError("segments must be ordered and contiguous")
            previous_end = segment.end_out
        if previous_end != self.combined_outs:
            raise ValueError("final segment end_out must equal combined_outs")
        if sum(segment.outs for segment in self.segments) != self.combined_outs:
            raise ValueError("segment outs must sum exactly to combined_outs")


@dataclass(frozen=True)
class PitcherChainForecast:
    pitchers: tuple[PitcherForecast, ...]
    combined_workload: WorkloadDistribution
    log_standard_error: float
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.pitchers, tuple) or not self.pitchers:
            raise ValueError("pitchers must be a nonempty tuple")
        for pitcher in self.pitchers:
            if not isinstance(pitcher, PitcherForecast):
                raise ValueError("pitchers must contain PitcherForecast values")
        pitcher_ids = tuple(pitcher.pitcher_id for pitcher in self.pitchers)
        if len(set(pitcher_ids)) != len(pitcher_ids):
            raise ValueError("pitchers must have unique pitcher_id values")

        if len(self.pitchers) == 1:
            if self.pitchers[0].role != "starter":
                raise ValueError("a single pitcher forecast must have starter role")
        else:
            roles = tuple(pitcher.role for pitcher in self.pitchers)
            if roles[0] != "opener":
                raise ValueError("a multi-pitcher forecast must start with opener")
            if any(role not in {"bulk", "prim"} for role in roles[1:]):
                raise ValueError("later pitcher roles must be bulk or prim")
            if len(set(roles[1:])) != len(roles[1:]):
                raise ValueError("later pitcher roles must be distinct")

        if not isinstance(self.combined_workload, WorkloadDistribution):
            raise ValueError("combined_workload must be WorkloadDistribution")
        _require_finite_nonnegative(
            self.log_standard_error,
            "log_standard_error",
        )
        _require_diagnostics(self.diagnostics, "diagnostics")

    @property
    def pitcher_forecasts(self) -> tuple[PitcherForecast, ...]:
        return self.pitchers

    @property
    def component_forecasts(self) -> tuple[PitcherForecast, ...]:
        return self.pitchers

    @property
    def run_prevention_multipliers(self) -> tuple[float, ...]:
        return tuple(
            pitcher.run_prevention_multiplier for pitcher in self.pitchers
        )

    def sample_plan(self, rng: object) -> SampledPitcherPlan:
        combined_outs = self.combined_workload.sample_outs(rng)
        remaining_outs = combined_outs
        start_out = 0
        segments: list[PitcherPlanSegment] = []

        for index, pitcher in enumerate(self.pitchers):
            is_final = index == len(self.pitchers) - 1
            if is_final:
                pitcher_outs = remaining_outs
            else:
                pitcher_outs = min(
                    pitcher.workload.sample_outs(rng),
                    remaining_outs,
                )
            end_out = start_out + pitcher_outs
            segments.append(
                PitcherPlanSegment(
                    pitcher_id=pitcher.pitcher_id,
                    role=pitcher.role,
                    start_out=start_out,
                    end_out=end_out,
                )
            )
            start_out = end_out
            remaining_outs -= pitcher_outs

        return SampledPitcherPlan(
            segments=tuple(segments),
            combined_outs=combined_outs,
        )


def _normalize_probabilities(values: list[float]) -> tuple[float, float, float]:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("workload probability total must be finite and positive")
    normalized = [value / total for value in values]
    correction = 1.0 - math.fsum(normalized)
    correction_index = max(range(len(normalized)), key=normalized.__getitem__)
    normalized[correction_index] += correction
    return normalized[0], normalized[1], normalized[2]


def _build_workload_distribution(
    *,
    eligible_outs: tuple[int, ...],
    league_prior_outs: tuple[int, ...],
    source: str,
    prior_field_name: str,
    pitch_limit: int | None,
    config: SimulationConfig,
) -> WorkloadDistribution:
    eligible_buckets: list[list[int]] = [[], [], []]
    prior_buckets: list[list[int]] = [[], [], []]
    for outs in eligible_outs:
        eligible_buckets[_bucket_index(outs)].append(outs)
    for outs in league_prior_outs:
        prior_buckets[_bucket_index(outs)].append(outs)

    missing_buckets = tuple(
        bucket_name
        for bucket_name, prior_bucket in zip(_BUCKET_NAMES, prior_buckets)
        if not prior_bucket
    )
    if missing_buckets:
        raise ValueError(
            f"{prior_field_name} must contain every workload bucket; missing "
            f"{', '.join(missing_buckets)}"
        )

    supports = tuple(
        tuple(eligible_bucket) + (median_low(prior_bucket),)
        for eligible_bucket, prior_bucket in zip(
            eligible_buckets,
            prior_buckets,
        )
    )
    total_count = sum(len(support) for support in supports)
    probabilities = [len(support) / total_count for support in supports]
    notes: list[str] = []

    if pitch_limit is not None:
        high_mass = probabilities[2]
        transfer_fraction = config.pitch_limit_high_bucket_transfer
        under_five_share = config.pitch_limit_transfer_to_under_five
        moved_mass = high_mass * transfer_fraction
        probabilities[0] += moved_mass * under_five_share
        probabilities[1] += moved_mass * (1.0 - under_five_share)
        probabilities[2] = (
            0.0
            if transfer_fraction == 1.0
            else high_mass - moved_mass
        )
        notes.append(
            f"pitch_limit={pitch_limit}: transferred {moved_mass:.12g} "
            "six_plus probability "
            f"({under_five_share:.0%} under_five, "
            f"{1.0 - under_five_share:.0%} five)"
        )

    return WorkloadDistribution(
        probabilities=_normalize_probabilities(probabilities),
        supports=(supports[0], supports[1], supports[2]),
        source=source,
        notes=tuple(notes),
    )


def forecast_pitcher(
    pitcher: PitcherInput,
    config: SimulationConfig,
) -> PitcherForecast:
    if not isinstance(pitcher, PitcherInput):
        raise TypeError("pitcher must be PitcherInput")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")

    workload = _build_workload_distribution(
        eligible_outs=pitcher.eligible_workload.recorded_outs,
        league_prior_outs=pitcher.league_prior_workload.recorded_outs,
        source="pitcher_history",
        prior_field_name="league_prior_workload",
        pitch_limit=(
            None if pitcher.pitch_limit is None else pitcher.pitch_limit.value
        ),
        config=config,
    )

    base_log_standard_error = pitcher.log_standard_error.value
    _require_finite_nonnegative(
        base_log_standard_error,
        "pitcher.log_standard_error.value",
    )
    variance_terms = [base_log_standard_error ** 2]
    diagnostics: list[str] = []
    if pitcher.injury_return.value:
        variance_terms.append(config.injury_return_log_se_penalty ** 2)
        diagnostics.append("injury_return")
    if pitcher.recent_short_exit.value:
        variance_terms.append(config.recent_short_exit_log_se_penalty ** 2)
        diagnostics.append("recent_short_exit")
    if pitcher.command_risk.value:
        diagnostics.append("command_risk")
    if pitcher.home_run_risk.value:
        diagnostics.append("home_run_risk")

    unbounded_multiplier = (
        pitcher.projected_ra9.value / config.league_runs_per_team_game
    )
    lower_bound, upper_bound = config.starter_ra9_bounds
    run_prevention_multiplier = min(
        upper_bound,
        max(lower_bound, unbounded_multiplier),
    )

    return PitcherForecast(
        pitcher_id=pitcher.pitcher_id,
        role=pitcher.role.value,
        workload=workload,
        run_prevention_multiplier=run_prevention_multiplier,
        log_standard_error=math.sqrt(math.fsum(variance_terms)),
        command_risk=pitcher.command_risk.value,
        home_run_risk=pitcher.home_run_risk.value,
        diagnostics=tuple(diagnostics),
    )


def forecast_pitcher_chain(
    chain: PitcherChainInput,
    config: SimulationConfig,
) -> PitcherChainForecast:
    if not isinstance(chain, PitcherChainInput):
        raise TypeError("chain must be PitcherChainInput")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")

    pitcher_forecasts = tuple(
        forecast_pitcher(pitcher, config) for pitcher in chain.pitchers
    )
    if len(pitcher_forecasts) == 1:
        combined_workload = pitcher_forecasts[0].workload
    else:
        if (
            chain.eligible_combined_workload is None
            or chain.league_prior_combined_workload is None
        ):
            raise ValueError(
                "multi-pitcher chains require combined workload histories"
            )
        combined_workload = _build_workload_distribution(
            eligible_outs=chain.eligible_combined_workload.recorded_outs,
            league_prior_outs=(
                chain.league_prior_combined_workload.recorded_outs
            ),
            source="combined_chain_history",
            prior_field_name="league_prior_combined_workload",
            pitch_limit=(
                None
                if chain.combined_pitch_limit is None
                else chain.combined_pitch_limit.value
            ),
            config=config,
        )

    total_expected_outs = math.fsum(
        pitcher.workload.expected_outs for pitcher in pitcher_forecasts
    )
    if not math.isfinite(total_expected_outs) or total_expected_outs <= 0:
        raise ValueError("component total expected outs must be positive")
    weighted_variance = math.fsum(
        pitcher.workload.expected_outs * pitcher.log_standard_error ** 2
        for pitcher in pitcher_forecasts
    ) / total_expected_outs

    diagnostics: list[str] = []
    if not chain.starter_confirmed.value:
        diagnostics.append("starter_unconfirmed")
    if chain.starter_source_conflict.value:
        diagnostics.append("starter_source_conflict")
    if not chain.handoff_verified.value:
        diagnostics.append("handoff_unverified")
    if chain.timing_uncertain.value:
        diagnostics.append("handoff_timing_uncertain")
    if not chain.handoff_verified.value or chain.timing_uncertain.value:
        weighted_variance += config.handoff_uncertainty_log_se_penalty ** 2

    return PitcherChainForecast(
        pitchers=pitcher_forecasts,
        combined_workload=combined_workload,
        log_standard_error=math.sqrt(weighted_variance),
        diagnostics=tuple(diagnostics),
    )
