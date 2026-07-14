"""Pregame leakage and relationship validation for normalized forecast inputs."""

from collections.abc import Iterator
from dataclasses import dataclass

from .inputs import (
    BullpenInput,
    FeatureProvenance,
    LineupInput,
    NormalizedGameInput,
    PitcherChainInput,
    SourceObservation,
)


@dataclass(frozen=True)
class ValidationResult:
    out_of_sample_eligible: bool
    flags: tuple[str, ...]


def _iter_lineup_provenance(lineup: LineupInput) -> Iterator[FeatureProvenance]:
    yield lineup.state_provenance
    yield lineup.projected_runs_per_9.provenance
    yield lineup.projected_traffic_per_9.provenance
    yield lineup.projected_extra_base_hits_per_9.provenance
    yield lineup.walk_rate.provenance
    yield lineup.strikeout_rate.provenance
    yield lineup.isolated_power.provenance
    yield lineup.barrel_rate.provenance
    yield lineup.recent_form_index.provenance
    yield lineup.lower_order_continuity.provenance
    yield lineup.log_standard_error.provenance
    yield lineup.recent_form_included.provenance
    for slot in lineup.slots:
        for response in slot.responses:
            yield response.provenance


def _iter_pitching_provenance(
    chain: PitcherChainInput,
) -> Iterator[FeatureProvenance]:
    for pitcher in chain.pitchers:
        yield pitcher.projected_ra9.provenance
        components = pitcher.projection_components
        yield components.strikeout_minus_walk_rate.provenance
        yield components.walk_rate.provenance
        yield components.home_runs_per_contact.provenance
        yield components.pitches_per_inning.provenance
        yield components.recent_pitch_count.provenance
        yield components.recent_workload_outs.provenance
        yield pitcher.log_standard_error.provenance
        yield pitcher.eligible_workload.provenance
        yield pitcher.league_prior_workload.provenance
        if pitcher.pitch_limit is not None:
            yield pitcher.pitch_limit.provenance
        for pitch_usage in pitcher.pitch_mix:
            yield pitch_usage.provenance
        yield pitcher.command_risk.provenance
        yield pitcher.home_run_risk.provenance
        yield pitcher.injury_return.provenance
        yield pitcher.recent_short_exit.provenance

    yield chain.starter_confirmed.provenance
    yield chain.starter_source_conflict.provenance
    yield chain.handoff_verified.provenance
    yield chain.timing_uncertain.provenance
    if chain.eligible_combined_workload is not None:
        yield chain.eligible_combined_workload.provenance
    if chain.league_prior_combined_workload is not None:
        yield chain.league_prior_combined_workload.provenance
    if chain.combined_pitch_limit is not None:
        yield chain.combined_pitch_limit.provenance


def _iter_bullpen_provenance(
    bullpen: BullpenInput,
) -> Iterator[FeatureProvenance]:
    for segment in bullpen.segments:
        yield segment.projected_ra9.provenance
        yield segment.log_standard_error.provenance
        for appearance in segment.appearances:
            yield appearance.provenance
        yield segment.available.provenance
        yield segment.limited_performance_or_use_risk.provenance


def iter_feature_provenance(
    game: NormalizedGameInput,
) -> Iterator[FeatureProvenance]:
    """Yield every used feature provenance through explicit domain traversal."""

    if not isinstance(game, NormalizedGameInput):
        raise TypeError("game must be NormalizedGameInput")
    yield from _iter_lineup_provenance(game.away_lineup)
    yield from _iter_lineup_provenance(game.home_lineup)
    yield from _iter_pitching_provenance(game.away_pitching)
    yield from _iter_pitching_provenance(game.home_pitching)
    yield from _iter_bullpen_provenance(game.away_bullpen)
    yield from _iter_bullpen_provenance(game.home_bullpen)
    for market in game.markets:
        yield market.provenance


def _require_unique_market_keys(game: NormalizedGameInput) -> None:
    seen: set[str] = set()
    for market in game.markets:
        if market.market_key in seen:
            raise ValueError(f"duplicate market_key: {market.market_key}")
        seen.add(market.market_key)


def _validate_team_relationships(game: NormalizedGameInput) -> None:
    identity = game.identity
    if identity.away_team == identity.home_team:
        raise ValueError("away_team and home_team must differ")
    if not (
        identity.away_team
        == game.away_lineup.team
        == game.away_bullpen.team
    ):
        raise ValueError(
            "away team relationship mismatch: identity, lineup, and bullpen "
            "teams must match"
        )
    if not (
        identity.home_team
        == game.home_lineup.team
        == game.home_bullpen.team
    ):
        raise ValueError(
            "home team relationship mismatch: identity, lineup, and bullpen "
            "teams must match"
        )


def _validate_response_relationships(
    lineup: LineupInput,
    opposing_chain: PitcherChainInput,
    team_label: str,
) -> None:
    opposing_pitchers = {
        pitcher.pitcher_id: pitcher for pitcher in opposing_chain.pitchers
    }
    matchup_pitcher = opposing_pitchers[opposing_chain.matchup_pitcher_id]
    for slot in lineup.slots:
        matchup_side = (
            "L" if matchup_pitcher.throws == "R" else "R"
        ) if slot.bats == "S" else slot.bats
        if slot.expected_side != matchup_side:
            raise ValueError(
                f"{team_label} lineup expected_side must match the batter side "
                "against the matchup pitcher"
            )
        for response in slot.responses:
            if response.pitcher_id not in opposing_pitchers:
                raise ValueError(
                    f"{team_label} lineup response pitcher_id must reference "
                    "the opposing pitcher chain"
                )
            pitcher = opposing_pitchers[response.pitcher_id]
            expected_side = (
                "L" if pitcher.throws == "R" else "R"
            ) if slot.bats == "S" else slot.bats
            if response.expected_side != expected_side:
                raise ValueError(
                    f"{team_label} lineup response expected_side must match the "
                    "batter side resolved for its opposing pitcher"
                )


def _observations_by_id(
    game: NormalizedGameInput,
) -> dict[str, SourceObservation]:
    observations: dict[str, SourceObservation] = {}
    for observation in game.observations:
        if observation.observation_id in observations:
            raise ValueError(
                f"duplicate observation_id: {observation.observation_id}"
            )
        observations[observation.observation_id] = observation
    return observations


def validate_pregame_input(game: NormalizedGameInput) -> ValidationResult:
    """Validate pregame timing, provenance references, and domain relationships."""

    if not isinstance(game, NormalizedGameInput):
        raise TypeError("game must be NormalizedGameInput")
    if game.captured_at >= game.identity.first_pitch:
        raise ValueError("captured_at must be before identity.first_pitch")

    _validate_team_relationships(game)
    _require_unique_market_keys(game)
    _validate_response_relationships(
        game.away_lineup,
        game.home_pitching,
        "away",
    )
    _validate_response_relationships(
        game.home_lineup,
        game.away_pitching,
        "home",
    )
    observations = _observations_by_id(game)

    unknown_available_at = False
    for feature_provenance in iter_feature_provenance(game):
        latest_available_at = feature_provenance.latest_available_at
        if latest_available_at is None:
            unknown_available_at = True
        elif latest_available_at > game.captured_at:
            raise ValueError(
                "feature latest_available_at is after captured_at"
            )

        lookback_end = feature_provenance.lookback_end
        if lookback_end is not None and lookback_end > game.captured_at:
            raise ValueError("feature lookback_end is after captured_at")

        for observation_id in feature_provenance.observation_ids:
            if observation_id not in observations:
                raise ValueError(
                    "missing observation_id referenced by feature: "
                    f"{observation_id}"
                )
            observation = observations[observation_id]
            available_at = observation.available_at
            if available_at is None:
                unknown_available_at = True
            elif available_at > game.captured_at:
                raise ValueError(
                    f"observation {observation_id} available_at is after "
                    "captured_at"
                )

    if unknown_available_at:
        return ValidationResult(False, ("unknown available_at",))
    return ValidationResult(True, ())
