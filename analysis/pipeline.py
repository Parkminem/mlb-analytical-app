from __future__ import annotations

import argparse
from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.forecasting.config import SimulationConfig
from analysis.model import evaluate_pitch_velocity_sample


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model.json"
LEGACY_MODEL_VERSION = "2026-08-02"
DECISION_AUTHORITY = "legacy_fixed_model"
FORECASTING_MODEL_VERSION = "shadow-forecast-2026-07-14.1"

_SIMULATION_CONFIG_KEYS = frozenset(SimulationConfig().to_mapping())
_FORECASTING_REQUIRED_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "model_version",
        "may_change_recommendations",
        "confirmed_lineup_validation_games",
        "market_family_tuning_decisions",
    }
    | (_SIMULATION_CONFIG_KEYS - {"version"})
)

MODEL_LAYERS = (
    "data_integrity",
    "starter_lineup_matchup",
    "lineup_form",
    "starter_stability",
    "bullpen",
    "supporting_context",
    "betting_decision_engines",
    "market_value_and_final_output",
)

REQUIRED_INPUTS = (
    "official_schedule_and_starters",
    "rotowire_lineup",
    "starter_pitch_mix_by_batter_side",
    "all_hitters_pitch_velocity_response",
    "recent_lineup_form",
    "starter_recent_form_and_risk",
    "bullpen_availability_and_quality",
    "umpire_when_available",
    "market_odds_and_total",
)


@dataclass(frozen=True)
class PitchVelocitySlotGrade:
    label: str
    evidence_strength: str
    adjusted_woba: float


def _validate_forecasting_config(forecasting: object) -> None:
    if not isinstance(forecasting, dict):
        raise ValueError("model configuration must contain a forecasting object")
    if set(forecasting) != _FORECASTING_REQUIRED_KEYS:
        raise ValueError("forecasting configuration must use the complete approved schema")
    if forecasting["enabled"] is not True:
        raise ValueError("forecasting must declare enabled=true")
    if forecasting["mode"] != "shadow":
        raise ValueError("forecasting mode must remain shadow")
    if forecasting["model_version"] != FORECASTING_MODEL_VERSION:
        raise ValueError(
            f"forecasting model_version must remain {FORECASTING_MODEL_VERSION}"
        )
    if forecasting["may_change_recommendations"] is not False:
        raise ValueError("shadow forecasting cannot change recommendations")

    simulation_config = {
        key: forecasting[key]
        for key in _SIMULATION_CONFIG_KEYS
        if key != "version"
    }
    simulation_config["version"] = forecasting["model_version"]
    try:
        SimulationConfig.from_mapping(simulation_config)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid forecasting configuration: {error}") from error

    for key in (
        "confirmed_lineup_validation_games",
        "market_family_tuning_decisions",
    ):
        value = forecasting[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"forecasting {key} must be a positive integer")


def _validate_model_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a JSON object")
    if config.get("version") != LEGACY_MODEL_VERSION:
        raise ValueError(f"legacy model version must remain {LEGACY_MODEL_VERSION}")
    if config.get("fixed_model") is not True:
        raise ValueError("model configuration must declare fixed_model=true")
    _validate_forecasting_config(config.get("forecasting"))


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    _validate_model_config(config)
    return config


def classify_pitch_velocity_slot(
    pa: int,
    woba: float,
    config: dict[str, Any] | None = None,
) -> PitchVelocitySlotGrade:
    model_config = config or load_model_config()
    thresholds = model_config["pitch_velocity"]
    evidence = evaluate_pitch_velocity_sample(pa=pa, woba=woba)

    if evidence.strength == "weak":
        label = "표본 부족"
    elif evidence.adjusted_woba >= thresholds["plus_adjusted_woba"]:
        label = "플러스"
    elif evidence.adjusted_woba < thresholds["minus_adjusted_woba"]:
        label = "마이너스"
    else:
        label = "중립"

    return PitchVelocitySlotGrade(
        label=label,
        evidence_strength=evidence.strength,
        adjusted_woba=evidence.adjusted_woba,
    )


def build_analysis_manifest(
    *,
    lineup_confirmed: bool,
    starter_confirmed: bool,
    include_weather: bool | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        model_config = load_model_config()
    else:
        _validate_model_config(config)
        model_config = config
    weather_default = model_config["weather"]["included_by_default"]
    weather_enabled = weather_default if include_weather is None else include_weather

    confidence_flags: list[str] = []
    if not lineup_confirmed:
        confidence_flags.append("projected_lineup")
    if not starter_confirmed:
        confidence_flags.append("unconfirmed_starter")

    required_inputs = list(REQUIRED_INPUTS)
    if weather_enabled:
        required_inputs.append("weather_and_park")

    lineup_source = (
        "RotoWire confirmed lineup"
        if lineup_confirmed
        else "RotoWire projected lineup"
    )

    return {
        "model_version": model_config["version"],
        "fixed_model": model_config["fixed_model"],
        "decision_authority": DECISION_AUTHORITY,
        "forecasting": deepcopy(model_config["forecasting"]),
        "layers": list(MODEL_LAYERS),
        "lineup_source": lineup_source,
        "include_weather": weather_enabled,
        "cache_policy": model_config.get("cache_policy", {}),
        "required_inputs": required_inputs,
        "confidence_flags": confidence_flags,
        "strength_adjustments": [],
        "priority_candidate_assembler": "assemble_same_game_priority",
        "required_outputs": ["F5", "F5 O/U", "ML", "+1.5", "O/U", "review"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the fixed MLB analysis pipeline manifest."
    )
    parser.add_argument(
        "--lineup",
        choices=("confirmed", "projected"),
        default="projected",
    )
    parser.add_argument(
        "--starter",
        choices=("confirmed", "unconfirmed"),
        default="confirmed",
    )
    parser.add_argument(
        "--weather",
        action="store_true",
        help="Include weather and park inputs for this analysis.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = build_analysis_manifest(
        lineup_confirmed=args.lineup == "confirmed",
        starter_confirmed=args.starter == "confirmed",
        include_weather=args.weather,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
