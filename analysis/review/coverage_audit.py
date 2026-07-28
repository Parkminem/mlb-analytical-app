"""Deterministic, read-only coverage audit for excluded shadow recommendations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final

from analysis.forecasting.engine import SemanticMarketInput
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType, SettlementFormat
from analysis.forecasting.market import settle_observed_market
from analysis.ledger import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    LedgerStore,
    OfficialResultRecord,
    linked_shadow_comparison,
)
from analysis.ledger.store import LedgerCorruptionError
from analysis.shadow import deserialize_shadow_comparison


_MANIFEST_KEYS: Final = frozenset(("schema_version", "audit_id", "entries"))
_ENTRY_KEYS: Final = frozenset(
    (
        "game_pk",
        "capture_id",
        "forecast_run_id",
        "capture_forecast_link_record_id",
        "market_key",
    )
)
_CALIBRATION_KEYS: Final = frozenset(
    (
        "schema_version",
        "ledger_as_of",
        "ledger_path",
        "ledger_revision",
        "migration",
        "overall",
        "group_by",
        "groups",
    )
)
_LEDGER_REVISION_KEYS: Final = frozenset(("record_count", "state_hash"))
_OVERALL_METRIC_KEYS: Final = frozenset(
    (
        "sample_size",
        "wins",
        "losses",
        "pushes",
        "binary_sample_size",
        "multiclass_sample_size",
        "priced_stake_count",
        "probability_clv_sample_size",
        "price_clv_sample_size",
        "hit_rate",
        "push_rate",
        "binary_brier",
        "multiclass_brier",
        "log_loss",
        "profit_units",
        "roi",
        "mean_probability_clv",
        "mean_price_clv",
    )
)
_SEMANTIC_MARKET_KEYS: Final = frozenset(
    (
        "market_key",
        "market_type",
        "settlement_format",
        "selected_side",
        "line",
        "selected_decimal_price",
        "opposing_decimal_price",
    )
)
_RECOMMENDED_RATINGS: Final = frozenset(("play", "lean"))


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    keys = frozenset(value)
    if keys != expected:
        missing = ", ".join(sorted(expected - keys))
        unknown = ", ".join(sorted(keys - expected))
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise ValueError(f"{name} has invalid fields: {'; '.join(parts)}")


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    return _require_mapping(value, name)


def _load_manifest(path: Path) -> tuple[str, tuple[Mapping[str, object], ...]]:
    manifest = _load_json(path, "manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("manifest schema_version must be 1")
    audit_id = _require_nonempty_string(manifest["audit_id"], "audit_id")
    entries_value = manifest["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise ValueError("manifest entries must be a nonempty array")
    entries: list[Mapping[str, object]] = []
    seen: set[tuple[int, str, str, str, str]] = set()
    for index, entry_value in enumerate(entries_value):
        entry = _require_mapping(entry_value, f"manifest entries[{index}]")
        _require_exact_keys(entry, _ENTRY_KEYS, f"manifest entries[{index}]")
        game_pk = _require_positive_integer(entry["game_pk"], "game_pk")
        key = (
            game_pk,
            _require_nonempty_string(entry["capture_id"], "capture_id"),
            _require_nonempty_string(entry["forecast_run_id"], "forecast_run_id"),
            _require_nonempty_string(
                entry["capture_forecast_link_record_id"],
                "capture_forecast_link_record_id",
            ),
            _require_nonempty_string(entry["market_key"], "market_key"),
        )
        if key in seen:
            raise ValueError("manifest entries must be unique")
        seen.add(key)
        entries.append(entry)
    return audit_id, tuple(entries)


def _load_calibration(path: Path) -> Mapping[str, object]:
    calibration = _load_json(path, "calibration")
    _require_exact_keys(calibration, _CALIBRATION_KEYS, "calibration")
    if calibration["schema_version"] != 1:
        raise ValueError("calibration schema_version must be 1")
    revision = _require_mapping(calibration["ledger_revision"], "ledger_revision")
    _require_exact_keys(revision, _LEDGER_REVISION_KEYS, "ledger_revision")
    _require_nonnegative_integer(revision["record_count"], "record_count")
    _require_nonempty_string(revision["state_hash"], "state_hash")
    overall = _require_mapping(calibration["overall"], "overall metrics")
    _require_exact_keys(overall, _OVERALL_METRIC_KEYS, "overall metrics")
    for field_name in (
        "sample_size",
        "wins",
        "losses",
        "pushes",
        "binary_sample_size",
        "multiclass_sample_size",
        "priced_stake_count",
        "probability_clv_sample_size",
        "price_clv_sample_size",
    ):
        _require_nonnegative_integer(overall.get(field_name), f"overall.{field_name}")
    for field_name in (
        "hit_rate",
        "push_rate",
        "binary_brier",
        "multiclass_brier",
        "log_loss",
        "profit_units",
        "roi",
        "mean_probability_clv",
        "mean_price_clv",
    ):
        if overall.get(field_name) is not None:
            _require_finite_number(overall[field_name], f"overall.{field_name}")
    return calibration


def _ledger_index(records: tuple[object, ...]) -> tuple[
    Mapping[str, CaptureRecord],
    Mapping[str, ForecastRecord],
    Mapping[str, CaptureForecastLink],
    Mapping[int, OfficialResultRecord],
]:
    captures: dict[str, CaptureRecord] = {}
    forecasts: dict[str, ForecastRecord] = {}
    links: dict[str, CaptureForecastLink] = {}
    results_by_game: dict[int, list[OfficialResultRecord]] = {}
    for record in records:
        if isinstance(record, CaptureRecord):
            if captures.setdefault(record.capture_id, record) is not record:
                raise LedgerCorruptionError("duplicate capture_id in ledger")
        elif isinstance(record, ForecastRecord):
            if forecasts.setdefault(record.forecast_run_id, record) is not record:
                raise LedgerCorruptionError("duplicate forecast_run_id in ledger")
        elif isinstance(record, CaptureForecastLink):
            record_id = record.envelope.record_id
            if links.setdefault(record_id, record) is not record:
                raise LedgerCorruptionError("duplicate link record_id in ledger")
        elif isinstance(record, OfficialResultRecord):
            results_by_game.setdefault(record.game_pk, []).append(record)
    latest_results: dict[int, OfficialResultRecord] = {}
    for game_pk, results in results_by_game.items():
        superseded = {
            result.supersedes_record_id
            for result in results
            if result.supersedes_record_id is not None
        }
        latest = [
            result for result in results if result.envelope.record_id not in superseded
        ]
        if len(latest) != 1:
            raise LedgerCorruptionError(
                "official result corrections do not have one latest record"
            )
        latest_results[game_pk] = latest[0]
    return captures, forecasts, links, latest_results


def _semantic_market(forecast: ForecastRecord, market_key: str) -> SemanticMarketInput:
    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    inputs = _require_mapping(outputs.get("market_inputs"), "forecast market_inputs")
    payload = _require_mapping(inputs.get(market_key), f"market_inputs[{market_key}]")
    _require_exact_keys(payload, _SEMANTIC_MARKET_KEYS, f"market_inputs[{market_key}]")
    if payload["market_key"] != market_key:
        raise ValueError("market input key must match market_key")
    return SemanticMarketInput(
        market_key=market_key,
        market_type=MarketType(_require_nonempty_string(payload["market_type"], "market_type")),
        settlement_format=SettlementFormat(
            _require_nonempty_string(payload["settlement_format"], "settlement_format")
        ),
        selected_side=_require_nonempty_string(payload["selected_side"], "selected_side"),
        line=payload["line"],
        selected_decimal_price=payload["selected_decimal_price"],
        opposing_decimal_price=payload["opposing_decimal_price"],
    )


def _audit_entry(
    entry: Mapping[str, object],
    captures: Mapping[str, CaptureRecord],
    forecasts: Mapping[str, ForecastRecord],
    links: Mapping[str, CaptureForecastLink],
    latest_results: Mapping[int, OfficialResultRecord],
) -> Mapping[str, object]:
    game_pk = _require_positive_integer(entry["game_pk"], "game_pk")
    capture_id = _require_nonempty_string(entry["capture_id"], "capture_id")
    forecast_run_id = _require_nonempty_string(entry["forecast_run_id"], "forecast_run_id")
    link_record_id = _require_nonempty_string(
        entry["capture_forecast_link_record_id"],
        "capture_forecast_link_record_id",
    )
    market_key = _require_nonempty_string(entry["market_key"], "market_key")
    capture = captures.get(capture_id)
    forecast = forecasts.get(forecast_run_id)
    link = links.get(link_record_id)
    if capture is None or capture.game_pk != game_pk:
        raise ValueError(f"manifest capture does not resolve for {market_key}")
    if forecast is None:
        raise ValueError(f"manifest forecast does not resolve for {market_key}")
    if link is None or (
        link.capture_id != capture_id or link.forecast_run_id != forecast_run_id
    ):
        raise ValueError(f"manifest link relationship does not resolve for {market_key}")
    comparison = deserialize_shadow_comparison(linked_shadow_comparison(link, forecast))
    if comparison.forecast_run_id != forecast_run_id:
        raise ValueError(f"linked comparison forecast does not match for {market_key}")
    markets = [
        market for market in comparison.market_comparisons if market.market_key == market_key
    ]
    if len(markets) != 1:
        raise ValueError(f"named market comparison does not resolve for {market_key}")
    market_comparison = markets[0]
    if market_comparison.legacy_rating not in _RECOMMENDED_RATINGS:
        raise ValueError(f"named market is not recommended: {market_key}")
    if market_comparison.review_eligible:
        raise ValueError(f"named market is not review-ineligible: {market_key}")
    market = _semantic_market(forecast, market_key)
    result = latest_results.get(game_pk)
    payload: dict[str, object] = {
        "game_pk": game_pk,
        "capture_id": capture_id,
        "forecast_run_id": forecast_run_id,
        "capture_forecast_link_record_id": link_record_id,
        "market_key": market_key,
        "legacy_rating": market_comparison.legacy_rating,
        "ineligibility_flags": list(market_comparison.review_ineligibility_flags),
        "market_type": market.market_type.value,
        "settlement_format": market.settlement_format.value,
        "selected_side": market.selected_side,
        "line": market.line,
        "selected_decimal_price": market.selected_decimal_price,
    }
    if result is None:
        payload.update(
            {
                "status": "unresolved",
                "official_result_record_id": None,
                "market_result": None,
                "profit_units": None,
            }
        )
        return payload
    market_result = settle_observed_market(market, result)
    price = market.selected_decimal_price
    profit = (
        None
        if price is None
        else {"win": price - 1.0, "loss": -1.0, "push": 0.0}[market_result]
    )
    payload.update(
        {
            "status": "settled",
            "official_result_record_id": result.envelope.record_id,
            "market_result": market_result,
            "profit_units": profit,
        }
    )
    return payload


def _audit_only_metrics(
    official_metrics: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    metrics = dict(official_metrics)
    wins = _require_nonnegative_integer(metrics["wins"], "official wins")
    losses = _require_nonnegative_integer(metrics["losses"], "official losses")
    pushes = _require_nonnegative_integer(metrics["pushes"], "official pushes")
    sample_size = _require_nonnegative_integer(metrics["sample_size"], "official sample_size")
    priced = _require_nonnegative_integer(
        metrics["priced_stake_count"],
        "official priced_stake_count",
    )
    profit = metrics.get("profit_units")
    if profit is not None:
        profit = _require_finite_number(profit, "official profit_units")
    for entry in entries:
        if entry["status"] != "settled":
            continue
        sample_size += 1
        result = entry["market_result"]
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "push":
            pushes += 1
        else:
            raise ValueError("settled audit entry has an invalid market_result")
        entry_profit = entry["profit_units"]
        if entry_profit is not None:
            priced += 1
            profit = (0.0 if profit is None else profit) + _require_finite_number(
                entry_profit,
                "entry profit_units",
            )
    metrics.update(
        {
            "sample_size": sample_size,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "priced_stake_count": priced,
            "profit_units": profit,
            "roi": None if profit is None or priced == 0 else profit / priced,
        }
    )
    return metrics


def build_coverage_audit(
    ledger_path: Path,
    calibration_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    """Build a deterministic audit without modifying ledger or calibration inputs."""

    if not all(isinstance(path, Path) for path in (ledger_path, calibration_path, manifest_path)):
        raise TypeError("ledger_path, calibration_path, and manifest_path must be Paths")
    audit_id, manifest_entries = _load_manifest(manifest_path)
    calibration = _load_calibration(calibration_path)
    records = tuple(LedgerStore(ledger_path).iter_records())
    captures, forecasts, links, latest_results = _ledger_index(records)
    entries = tuple(
        _audit_entry(entry, captures, forecasts, links, latest_results)
        for entry in manifest_entries
    )
    official_metrics = dict(_require_mapping(calibration["overall"], "overall metrics"))
    return {
        "schema_version": 1,
        "audit_id": audit_id,
        "official_calibration_unchanged": True,
        "official_ledger_revision": dict(
            _require_mapping(calibration["ledger_revision"], "ledger_revision")
        ),
        "official_metrics": official_metrics,
        "audit_only_metrics": _audit_only_metrics(official_metrics, entries),
        "resolved_omission_count": sum(entry["status"] == "settled" for entry in entries),
        "unresolved_omission_count": sum(entry["status"] == "unresolved" for entry in entries),
        "entries": list(entries),
    }


def render_coverage_audit_markdown(audit: Mapping[str, object]) -> str:
    """Render the stable Korean operator summary for an audit payload."""

    audit_id = _require_nonempty_string(audit.get("audit_id"), "audit_id")
    official = _require_mapping(audit.get("official_metrics"), "official_metrics")
    adjusted = _require_mapping(audit.get("audit_only_metrics"), "audit_only_metrics")
    entries = audit.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    lines = [
        f"# Review Coverage Audit: {audit_id}",
        "",
        "> 이 조정치는 감사용이며 공식 calibration row가 아닙니다.",
        "",
        "## Official Calibration",
        f"- Sample: {official['sample_size']}",
        f"- Record: {official['wins']}W-{official['losses']}L-{official['pushes']}P",
        f"- Priced stake: {official['priced_stake_count']}",
        f"- Profit: {official['profit_units']}u",
        f"- ROI: {official['roi']}",
        "",
        "## Audit-only Coverage",
        f"- Resolved omissions: {audit['resolved_omission_count']}",
        f"- Unresolved omissions: {audit['unresolved_omission_count']}",
        f"- Sample: {adjusted['sample_size']}",
        f"- Record: {adjusted['wins']}W-{adjusted['losses']}L-{adjusted['pushes']}P",
        f"- Priced stake: {adjusted['priced_stake_count']}",
        f"- Profit: {adjusted['profit_units']}u",
        f"- ROI: {adjusted['roi']}",
        "",
        "## Omitted Markets",
    ]
    for entry in entries:
        item = _require_mapping(entry, "audit entry")
        flags = ", ".join(item["ineligibility_flags"])
        lines.append(
            f"- `{item['market_key']}`: {item['status']}; flags={flags}; "
            f"result={item['market_result']}"
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("data/ledger/shadow.jsonl"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("reports/private/shadow/current/calibration-latest.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/audits/2026-07-28-review-coverage-manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/private/shadow/audits/2026-07-28-review-coverage"),
    )
    args = parser.parse_args(argv)
    audit = build_coverage_audit(args.ledger, args.calibration, args.manifest)
    _atomic_write(args.output_dir / "audit.json", canonical_json(audit) + "\n")
    _atomic_write(args.output_dir / "audit.md", render_coverage_audit_markdown(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
