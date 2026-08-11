from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from analysis.review.watchlist import (
    DEFAULT_WATCHLIST_CONFIG_PATH,
    WatchlistResult,
    build_review_watchlist,
    load_review_watchlist_config,
)
from analysis.review.calibration import CalibrationRow


APPROVED_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "targets": [
        {
            "target_id": "f5_over_early_conversion",
            "label": "F5 오버 초기 득점 전환",
            "kind": "f5_over_conversion",
            "minimum_sample": 20,
            "hit_rate_below": 0.45,
            "b_loss_count_min": 5,
            "b_loss_rate_min": 0.25,
        },
        {
            "target_id": "bullpen_protection_contradiction",
            "label": "불펜 보호 경로 모순",
            "kind": "path_contradiction",
            "minimum_sample": 20,
            "evidence_path": "bullpen_protection",
            "contradiction_count_min": 3,
            "contradiction_rate_min": 0.15,
        },
        {
            "target_id": "lineup_suppression_contradiction",
            "label": "언더 라인업 억제 경로 모순",
            "kind": "path_contradiction",
            "minimum_sample": 20,
            "evidence_path": "lineup_suppression",
            "contradiction_count_min": 3,
            "contradiction_rate_min": 0.15,
        },
    ],
}


def write_config(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "review-watchlist.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_raw_config(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "review-watchlist.json"
    path.write_text(value, encoding="utf-8")
    return path


def test_loads_approved_watchlist_configuration(tmp_path: Path) -> None:
    config = load_review_watchlist_config(write_config(tmp_path, APPROVED_CONFIG))

    assert config.schema_version == 1
    assert tuple(target.target_id for target in config.targets) == (
        "f5_over_early_conversion",
        "bullpen_protection_contradiction",
        "lineup_suppression_contradiction",
    )
    assert config.targets[0].evidence_path is None
    assert config.targets[1].hit_rate_below is None
    with pytest.raises(FrozenInstanceError):
        config.schema_version = 2


def test_default_watchlist_configuration_is_the_approved_version() -> None:
    config = load_review_watchlist_config()

    assert DEFAULT_WATCHLIST_CONFIG_PATH.name == "review-watchlist.json"
    assert config.schema_version == APPROVED_CONFIG["schema_version"]
    assert [target.target_id for target in config.targets] == [
        target["target_id"] for target in APPROVED_CONFIG["targets"]
    ]


def test_config_read_oserror_remains_filesystem_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review-watchlist.json"

    def unreadable(*args, **kwargs):
        raise PermissionError("watchlist permission denied")

    monkeypatch.setattr(Path, "read_text", unreadable)

    with pytest.raises(PermissionError, match="permission denied"):
        load_review_watchlist_config(path)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = json.dumps(APPROVED_CONFIG, ensure_ascii=False)
    raw = raw.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1)

    with pytest.raises(ValueError, match="duplicate"):
        load_review_watchlist_config(write_raw_config(tmp_path, raw))


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_rejects_nonfinite_json_constants(tmp_path: Path, constant: str) -> None:
    raw = json.dumps(APPROVED_CONFIG, ensure_ascii=False).replace("0.45", constant, 1)

    with pytest.raises(ValueError, match="number|finite|JSON"):
        load_review_watchlist_config(write_raw_config(tmp_path, raw))


@pytest.mark.parametrize(
    "raw",
    (
        "{",
        "null",
        "[]",
    ),
)
def test_rejects_malformed_or_nonobject_roots(tmp_path: Path, raw: str) -> None:
    with pytest.raises(ValueError, match="JSON|root"):
        load_review_watchlist_config(write_raw_config(tmp_path, raw))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value.pop("schema_version"),
        lambda value: value.pop("targets"),
        lambda value: value.update(extra=True),
        lambda value: value.update(targets={}),
        lambda value: value.update(targets=[]),
    ),
)
def test_rejects_malformed_root_contract(tmp_path: Path, mutator) -> None:
    value = deepcopy(APPROVED_CONFIG)
    mutator(value)

    with pytest.raises(ValueError, match="root|target|unknown|missing"):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize("schema_version", (True, 1.0, 0, 2))
def test_rejects_unsupported_or_malformed_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        load_review_watchlist_config(write_config(tmp_path, value))


def test_rejects_duplicate_target_ids(tmp_path: Path) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["targets"].append(deepcopy(value["targets"][0]))

    with pytest.raises(ValueError, match="duplicate"):
        load_review_watchlist_config(write_config(tmp_path, value))


def test_rejects_unsupported_target_kinds(tmp_path: Path) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["targets"][0]["kind"] = "unknown"

    with pytest.raises(ValueError, match="kind"):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["targets"][0].update(extra=True),
        lambda value: value["targets"][0].update(evidence_path="bullpen_protection"),
        lambda value: value["targets"][1].update(hit_rate_below=0.45),
    ),
)
def test_rejects_unknown_or_inapplicable_target_keys(
    tmp_path: Path,
    mutator,
) -> None:
    value = deepcopy(APPROVED_CONFIG)
    mutator(value)

    with pytest.raises(ValueError, match="unknown|inapplicable|keys"):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["targets"][0].pop("label"),
        lambda value: value["targets"][0].pop("hit_rate_below"),
        lambda value: value["targets"][1].pop("evidence_path"),
        lambda value: value["targets"].__setitem__(0, "not-an-object"),
    ),
)
def test_rejects_malformed_target_contracts(tmp_path: Path, mutator) -> None:
    value = deepcopy(APPROVED_CONFIG)
    mutator(value)

    with pytest.raises(ValueError, match="target|missing|keys|object"):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize(
    ("target_index", "field_name", "invalid_value"),
    (
        (0, "minimum_sample", True),
        (0, "minimum_sample", 20.0),
        (0, "minimum_sample", 0),
        (0, "b_loss_count_min", -1),
        (0, "b_loss_count_min", 5.0),
        (1, "contradiction_count_min", False),
        (1, "contradiction_count_min", 0),
    ),
)
def test_rejects_invalid_integer_thresholds(
    tmp_path: Path,
    target_index: int,
    field_name: str,
    invalid_value: object,
) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["targets"][target_index][field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize(
    ("target_index", "field_name", "invalid_value"),
    (
        (0, "hit_rate_below", True),
        (0, "hit_rate_below", "0.45"),
        (0, "hit_rate_below", -0.01),
        (0, "b_loss_rate_min", 1.01),
        (1, "contradiction_rate_min", False),
        (1, "contradiction_rate_min", 1.01),
    ),
)
def test_rejects_invalid_rate_thresholds(
    tmp_path: Path,
    target_index: int,
    field_name: str,
    invalid_value: object,
) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["targets"][target_index][field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        load_review_watchlist_config(write_config(tmp_path, value))


@pytest.mark.parametrize(
    ("target_index", "field_name", "invalid_value"),
    (
        (0, "target_id", ""),
        (0, "label", " "),
        (1, "evidence_path", ""),
    ),
)
def test_rejects_empty_required_strings(
    tmp_path: Path,
    target_index: int,
    field_name: str,
    invalid_value: str,
) -> None:
    value = deepcopy(APPROVED_CONFIG)
    value["targets"][target_index][field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        load_review_watchlist_config(write_config(tmp_path, value))


WATCHLIST_CONFIG = load_review_watchlist_config()


def watchlist_result(**overrides: object) -> WatchlistResult:
    values: dict[str, object] = {
        "target_id": "f5_over_early_conversion",
        "label": "F5 오버 초기 득점 전환",
        "status": "관찰",
        "eligible_count": 3,
        "excluded_count": 0,
        "win_count": 2,
        "loss_count": 1,
        "push_count": 0,
        "classification_counts": {"A": 2, "B": 1, "C": 0, "D": 0},
        "b_loss_count": 1,
        "contradiction_count": None,
        "contradiction_rate": None,
        "non_push_hit_rate": 2 / 3,
        "minimum_sample": 20,
        "trigger_reasons": (),
    }
    values.update(overrides)
    return WatchlistResult(**values)


def path_watchlist_result(**overrides: object) -> WatchlistResult:
    values: dict[str, object] = {
        "target_id": "bullpen_protection_contradiction",
        "label": "불펜 보호 경로 모순",
        "status": "관찰",
        "eligible_count": 4,
        "excluded_count": 0,
        "win_count": 0,
        "loss_count": 4,
        "push_count": 0,
        "classification_counts": {"A": 0, "B": 0, "C": 4, "D": 0},
        "b_loss_count": None,
        "contradiction_count": 1,
        "contradiction_rate": 0.25,
        "non_push_hit_rate": None,
        "minimum_sample": 20,
        "trigger_reasons": (),
    }
    values.update(overrides)
    return WatchlistResult(**values)


def test_watchlist_result_requires_derived_rates() -> None:
    with pytest.raises(ValueError, match="non_push_hit_rate"):
        watchlist_result(non_push_hit_rate=0.5)
    with pytest.raises(ValueError, match="contradiction_rate"):
        path_watchlist_result(contradiction_rate=0.5)


def test_watchlist_result_requires_b_losses_to_be_losses() -> None:
    with pytest.raises(ValueError, match="b_loss_count.*loss_count"):
        watchlist_result(b_loss_count=2)


@pytest.mark.parametrize(
    "overrides",
    (
        {"contradiction_count": 0, "contradiction_rate": 0.0},
        {"b_loss_count": None, "non_push_hit_rate": None},
        {"contradiction_rate": 0.0},
    ),
)
def test_watchlist_result_requires_exactly_one_metric_family(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="metric family"):
        watchlist_result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    (
        {"status": "정상 범위"},
        {
            "status": "관찰",
            "trigger_reasons": ("non_push_hit_rate_below",),
        },
        {"minimum_sample": 3, "status": "관찰"},
        {
            "minimum_sample": 3,
            "status": "정상 범위",
            "trigger_reasons": ("non_push_hit_rate_below",),
        },
    ),
)
def test_watchlist_result_status_matches_minimum_and_triggers(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="status|trigger_reasons|minimum_sample"):
        watchlist_result(**overrides)


def test_watchlist_result_recognizes_triggers_for_each_metric_family() -> None:
    with pytest.raises(ValueError, match="trigger_reasons"):
        watchlist_result(
            minimum_sample=3,
            status="정밀 검토",
            trigger_reasons=("contradiction_count_and_rate",),
        )
    with pytest.raises(ValueError, match="trigger_reasons"):
        path_watchlist_result(
            minimum_sample=4,
            status="정밀 검토",
            trigger_reasons=("b_loss_count_and_rate",),
        )


def watchlist_row(
    *,
    lineup_state: str = "confirmed",
    market_type: str = "f5_total",
    selected_side: str = "over",
    classification: str | None = "A",
    market_result: str = "win",
    principal_paths: tuple[str, ...] = (),
    contradicted_paths: tuple[str, ...] = (),
) -> CalibrationRow:
    line = {
        "f5_side": 0.0,
        "f5_total": 8.0,
        "plus_1_5": 1.5,
        "full_total": 8.0,
    }.get(market_type)
    push_probability = 0.10 if market_type in ("f5_total", "full_total") else 0.0
    residual_probability = (1.0 - push_probability) / 2.0
    return CalibrationRow(
        game_pk=900001,
        capture_id="capture-watchlist",
        forecast_run_id="forecast-watchlist",
        market_key=f"game:900001:{market_type}:{selected_side}",
        model_version="2026-08-05",
        market_type=market_type,
        settlement_format="two_way",
        line=line,
        lineup_state=lineup_state,
        starter_structure="starter_vs_starter",
        legacy_priority_score=80,
        selected_side=selected_side,
        classification=classification,
        principal_paths=principal_paths,
        contradicted_paths=contradicted_paths,
        p_win=residual_probability,
        p_push=push_probability,
        p_loss=residual_probability,
        market_result=market_result,
        selected_decimal_price=1.91,
        closing_selected_decimal_price=1.87,
        entry_no_vig_probability=0.52,
        closing_no_vig_probability=0.54,
    )


def f5_over_rows(
    *,
    wins: int = 0,
    losses: int = 0,
    pushes: int = 0,
    b_losses: int = 0,
) -> tuple[CalibrationRow, ...]:
    return (
        *(watchlist_row(market_result="win") for _ in range(wins)),
        *(watchlist_row(market_result="loss") for _ in range(losses)),
        *(
            watchlist_row(market_result="loss", classification="B")
            for _ in range(b_losses)
        ),
        *(watchlist_row(market_result="push") for _ in range(pushes)),
    )


def path_rows(
    *,
    evidence_path: str,
    eligible: int,
    contradicted: int,
    market_type: str = "ml",
    selected_side: str = "away",
) -> tuple[CalibrationRow, ...]:
    assert 0 <= contradicted <= eligible
    return tuple(
        watchlist_row(
            market_type=market_type,
            selected_side=selected_side,
            classification="C",
            market_result="loss",
            principal_paths=(evidence_path,),
            contradicted_paths=(evidence_path,) if index < contradicted else (),
        )
        for index in range(eligible)
    )


def evaluate(rows: tuple[CalibrationRow, ...]) -> tuple[WatchlistResult, ...]:
    return build_review_watchlist(rows, WATCHLIST_CONFIG)


def result_for(
    rows: tuple[CalibrationRow, ...],
    target_id: str,
) -> WatchlistResult:
    return next(result for result in evaluate(rows) if result.target_id == target_id)


def test_f5_over_counts_outcomes_classes_and_b_losses() -> None:
    rows = (
        watchlist_row(market_result="win", classification="A"),
        watchlist_row(market_result="loss", classification="B"),
        watchlist_row(market_result="push", classification="C"),
        watchlist_row(market_result="loss", classification="D"),
    )

    result = result_for(rows, "f5_over_early_conversion")

    assert result.eligible_count == 4
    assert (result.win_count, result.loss_count, result.push_count) == (1, 2, 1)
    assert dict(result.classification_counts) == {"A": 1, "B": 1, "C": 1, "D": 1}
    assert result.b_loss_count == 1
    assert result.non_push_hit_rate == pytest.approx(1 / 3)
    assert result.contradiction_count is None
    assert result.contradiction_rate is None


def test_f5_over_stays_observe_below_twenty() -> None:
    result = result_for(
        f5_over_rows(losses=19),
        "f5_over_early_conversion",
    )

    assert result.status == "관찰"
    assert result.eligible_count == 19
    assert result.trigger_reasons == ()


def test_f5_over_triggers_on_hit_rate_strictly_below_threshold() -> None:
    triggered = result_for(
        f5_over_rows(wins=8, losses=12),
        "f5_over_early_conversion",
    )
    boundary = result_for(
        f5_over_rows(wins=9, losses=11),
        "f5_over_early_conversion",
    )

    assert triggered.status == "정밀 검토"
    assert triggered.non_push_hit_rate == 0.4
    assert "non_push_hit_rate_below" in triggered.trigger_reasons
    assert boundary.status == "정상 범위"
    assert boundary.non_push_hit_rate == 0.45


def test_f5_over_hit_rate_excludes_pushes() -> None:
    result = result_for(
        f5_over_rows(wins=9, pushes=11),
        "f5_over_early_conversion",
    )

    assert result.eligible_count == 20
    assert result.non_push_hit_rate == 1.0
    assert result.status == "정상 범위"


def test_f5_over_triggers_when_b_loss_count_and_rate_both_reach_thresholds() -> None:
    result = result_for(
        f5_over_rows(wins=15, b_losses=5),
        "f5_over_early_conversion",
    )

    assert result.status == "정밀 검토"
    assert result.b_loss_count == 5
    assert result.trigger_reasons == ("b_loss_count_and_rate",)


@pytest.mark.parametrize(
    "rows",
    (
        f5_over_rows(wins=16, b_losses=4),
        f5_over_rows(wins=16, b_losses=5),
    ),
)
def test_f5_over_requires_both_b_loss_gates(rows: tuple[CalibrationRow, ...]) -> None:
    result = result_for(rows, "f5_over_early_conversion")

    assert result.status == "정상 범위"


def test_f5_over_selector_is_exact_and_excludes_missing_classification() -> None:
    rows = (
        watchlist_row(),
        watchlist_row(lineup_state="projected"),
        watchlist_row(selected_side="under"),
        watchlist_row(classification=None),
        watchlist_row(market_type="full_total"),
    )

    result = result_for(rows, "f5_over_early_conversion")

    assert result.eligible_count == 1
    assert result.excluded_count == 1


def test_path_contradiction_stays_observe_below_twenty() -> None:
    result = result_for(
        path_rows(
            evidence_path="bullpen_protection",
            eligible=19,
            contradicted=19,
        ),
        "bullpen_protection_contradiction",
    )

    assert result.status == "관찰"
    assert result.eligible_count == 19
    assert result.trigger_reasons == ()


def test_bullpen_protection_excludes_unrecommended_review() -> None:
    row = watchlist_row(
        market_type="ml",
        selected_side="away",
        classification=None,
        market_result="loss",
        principal_paths=("bullpen_protection",),
        contradicted_paths=("bullpen_protection",),
    )

    result = result_for((row,), "bullpen_protection_contradiction")

    assert result.eligible_count == 0
    assert result.excluded_count == 1
    assert result.contradiction_count == 0


def test_path_contradiction_triggers_at_exact_count_and_rate_thresholds() -> None:
    result = result_for(
        path_rows(
            evidence_path="bullpen_protection",
            eligible=20,
            contradicted=3,
        ),
        "bullpen_protection_contradiction",
    )

    assert result.status == "정밀 검토"
    assert result.contradiction_count == 3
    assert result.contradiction_rate == 0.15
    assert result.trigger_reasons == ("contradiction_count_and_rate",)


@pytest.mark.parametrize(
    ("eligible", "contradicted"),
    (
        (20, 2),
        (21, 3),
    ),
)
def test_path_contradiction_requires_both_count_and_rate_gates(
    eligible: int,
    contradicted: int,
) -> None:
    result = result_for(
        path_rows(
            evidence_path="bullpen_protection",
            eligible=eligible,
            contradicted=contradicted,
        ),
        "bullpen_protection_contradiction",
    )

    assert result.status == "정상 범위"


def test_path_contradiction_requires_exact_principal_and_contradicted_paths() -> None:
    rows = tuple(
        watchlist_row(
            market_type="ml",
            selected_side="away",
            classification="C",
            market_result="loss",
            principal_paths=("bullpen_protection", "starter_suppression"),
            contradicted_paths=("starter_suppression",),
        )
        for _ in range(20)
    ) + (
        watchlist_row(
            market_type="ml",
            selected_side="away",
            classification="C",
            principal_paths=("bullpen_protection_late",),
            contradicted_paths=("bullpen_protection_late",),
        ),
        watchlist_row(
            lineup_state="projected",
            market_type="ml",
            selected_side="away",
            classification="C",
            principal_paths=("bullpen_protection",),
            contradicted_paths=("bullpen_protection",),
        ),
    )

    result = result_for(rows, "bullpen_protection_contradiction")

    assert result.eligible_count == 20
    assert result.contradiction_count == 0
    assert result.status == "정상 범위"


def test_lineup_suppression_requires_exact_under_total_markets() -> None:
    rows = (
        watchlist_row(
            market_type="f5_total",
            selected_side="under",
            principal_paths=("lineup_suppression",),
        ),
        watchlist_row(
            market_type="full_total",
            selected_side="under",
            principal_paths=("lineup_suppression",),
        ),
        watchlist_row(
            market_type="f5_total",
            selected_side="over",
            principal_paths=("lineup_suppression",),
        ),
        watchlist_row(
            market_type="ml",
            selected_side="under",
            principal_paths=("lineup_suppression",),
        ),
        watchlist_row(
            market_type="full_total",
            selected_side="under",
            principal_paths=("lineup_suppression_late",),
        ),
    )

    result = result_for(rows, "lineup_suppression_contradiction")

    assert result.eligible_count == 2


def test_results_are_deterministic_deeply_immutable_and_config_ordered() -> None:
    rows = f5_over_rows(wins=8, losses=7, b_losses=5)

    first = evaluate(rows)
    second = evaluate(tuple(reversed(rows)))

    assert first == second
    assert tuple(result.target_id for result in first) == tuple(
        target.target_id for target in WATCHLIST_CONFIG.targets
    )
    assert first[0].trigger_reasons == tuple(sorted(first[0].trigger_reasons))
    assert first[0].to_mapping() == second[0].to_mapping()
    with pytest.raises(FrozenInstanceError):
        first[0].status = "changed"
    with pytest.raises(TypeError):
        first[0].classification_counts["A"] = 99
    mapping = first[0].to_mapping()
    mapping["classification_counts"]["A"] = 99
    assert first[0].classification_counts["A"] == 15


class SelectorOnlyRow:
    lineup_state = "confirmed"
    market_type = "f5_total"
    selected_side = "over"
    classification = "A"
    market_result = "win"
    principal_paths: tuple[str, ...] = ()
    contradicted_paths: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> object:
        if name in {
            "p_win",
            "p_push",
            "p_loss",
            "probabilities",
            "component_distributions",
            "forecast_draws",
        }:
            raise AssertionError(f"watchlist inspected forbidden forecast field: {name}")
        raise AttributeError(name)


def test_evaluation_never_reads_forecast_probabilities_or_draws() -> None:
    results = build_review_watchlist((SelectorOnlyRow(),), WATCHLIST_CONFIG)

    assert results[0].eligible_count == 1
