from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.forecasting import (
    LineupState,
    MarketType,
    SettlementFormat,
    run_forecast,
)
from analysis.forecasting.identity import (
    build_forecast_run_id_from_capture_payload,
    canonical_json,
)
from analysis.forecasting.market import (
    OutcomeProbabilities,
    market_semantic_fingerprint,
    price_market,
)
from analysis.forecasting.simulation import _summarize
from analysis.ledger.forecast_codec import (
    compact_forecast_outputs,
    expand_component_distributions,
)
from analysis.ledger import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    LedgerReferenceError,
    LedgerStore,
    MarketObservationCommand,
    OfficialResultCommand,
    OfficialResultRecord,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review import ContradictionRule, PrincipalPathBinding
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    ShadowRunIdempotencyKeys,
    deserialize_forecast_context,
    record_shadow_run,
)
from tests.test_forecasting_engine import _complete_game, _fast_config


try:
    from analysis.review.service import (
        build_and_record_review,
        build_review_candidate,
    )
except (ImportError, ModuleNotFoundError):
    build_and_record_review = None
    build_review_candidate = None


REVIEW_SERVICE_AVAILABLE = all(
    callable(value)
    for value in (build_review_candidate, build_and_record_review)
)
MARKET_KEY = "sea_ml"


def _principal_paths() -> tuple[PrincipalPathBinding, ...]:
    return (
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
            component_ids=("home_pitcher_chain_runs_allowed",),
            contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
        ),
        PrincipalPathBinding(
            evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
            component_ids=(
                "away_full_runs",
                "away_full_traffic",
                "away_full_extra_base_hits",
            ),
            contradiction_rule=(
                ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
            ),
        ),
    )


def _legacy_snapshot() -> LegacyDecisionSnapshot:
    paths = _principal_paths()
    candidate = PriorityMarketCandidate(
        market="ml",
        score=78,
        decision_rating="play",
        evidence_paths=frozenset(path.evidence_path for path in paths),
    )
    return LegacyDecisionSnapshot(
        model_version="legacy-fixed-v1",
        candidates=(candidate,),
        final_priority=("ml",),
        review_bindings=(
            CandidateReviewBinding(
                market_key=MARKET_KEY,
                legacy_candidate_index=0,
                legacy_market="ml",
                principal_paths=paths,
            ),
        ),
        notes=("legacy remains authoritative",),
    )


@dataclass(frozen=True)
class RecordedFixture:
    store: LedgerStore
    capture: CaptureRecord
    forecast: ForecastRecord
    link: CaptureForecastLink
    result: OfficialResultRecord
    caller_outputs: dict[str, object]


class FailOnceAfterReviewAppend:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store
        self.fail = True

    def iter_records(self, kind=None):
        return self.store.iter_records(kind)

    def append_review(self, command, idempotency_key=None):
        record = self.store.append_review(command, idempotency_key)
        if self.fail:
            self.fail = False
            raise RuntimeError("after review append")
        return record


class CorrectResultBeforeReviewAppend:
    def __init__(self, store: LedgerStore, correction: OfficialResultCommand):
        self.store = store
        self.correction = correction
        self.corrected = False

    def iter_records(self, kind=None):
        return self.store.iter_records(kind)

    def append_review(self, command, idempotency_key=None):
        if not self.corrected:
            self.store.append_official_result(self.correction)
            self.corrected = True
        return self.store.append_review(command, idempotency_key)


class RequireActiveCaptureReviewAppend:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store
        self.required_active_capture_id = None

    def iter_records(self, kind=None):
        return self.store.iter_records(kind)

    def append_review(
        self,
        command,
        idempotency_key=None,
        *,
        required_active_capture_id,
    ):
        self.required_active_capture_id = required_active_capture_id
        return self.store.append_review(
            command,
            idempotency_key,
            required_active_capture_id=required_active_capture_id,
        )


class ReviewServiceApiAvailabilityTests(unittest.TestCase):
    def test_review_service_api_is_available(self):
        self.assertTrue(
            REVIEW_SERVICE_AVAILABLE,
            "postgame review service API is missing",
        )


@unittest.skipUnless(
    REVIEW_SERVICE_AVAILABLE,
    "postgame review service API is missing",
)
class ReviewServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.game = _complete_game()
        cls.engine_forecast = run_forecast(cls.game, _fast_config())
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            records = record_shadow_run(
                store,
                cls.game,
                _legacy_snapshot(),
                cls.engine_forecast,
                ShadowRunIdempotencyKeys(capture="capture:review:base"),
            )
            cls.base_outputs = json.loads(
                canonical_json(records.forecast.outputs)
            )
            cls.capture_payload = json.loads(
                canonical_json(records.capture.payload)
            )
            cls.normalized_input_hash = records.capture.normalized_input_hash
            cls.simulation_config_hash = (
                records.forecast.simulation_config_hash
            )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = LedgerStore(Path(self.directory.name) / "ledger.jsonl")

    def tearDown(self) -> None:
        self.directory.cleanup()

    @classmethod
    def _fresh_outputs(cls) -> dict[str, object]:
        outputs = json.loads(canonical_json(cls.base_outputs))
        outputs["component_distributions"] = expand_component_distributions(
            outputs["component_distributions"]
        )
        return outputs

    @classmethod
    def _compact_outputs(cls) -> dict[str, object]:
        return compact_forecast_outputs(cls._fresh_outputs())

    @classmethod
    def _default_component_values(cls) -> dict[str, float]:
        values = {}
        for path in _principal_paths():
            for component_id in path.component_ids:
                summary = cls.base_outputs["component_distributions"][
                    component_id
                ]["summary"]
                values[component_id] = float(summary["median"])
        return values

    def fixture(
        self,
        *,
        outputs=None,
        mutate_outputs=None,
        component_values=None,
        documented_events=None,
        score=(2, 1, 4, 3),
        result_game_pk=None,
        result_finalized_at=None,
        market_observation_side="home",
        forecast_run_id=None,
        model_version=None,
        stored_normalized_input_hash=None,
        simulation_config_hash=None,
        mutate_capture_payload=None,
        record_captured_at=None,
        record_first_pitch=None,
        record_lineup_state="confirmed",
    ) -> RecordedFixture:
        outputs = (
            self._fresh_outputs()
            if outputs is None
            else json.loads(canonical_json(outputs))
        )
        if mutate_outputs is not None:
            mutate_outputs(outputs)
        capture_payload = json.loads(canonical_json(self.capture_payload))
        if mutate_capture_payload is not None:
            mutate_capture_payload(capture_payload)
        game_pk = self.game.identity.game_pk
        record_input_hash = (
            self.normalized_input_hash
            if stored_normalized_input_hash is None
            else stored_normalized_input_hash
        )
        record_forecast_run_id = (
            self.engine_forecast.forecast_run_id
            if forecast_run_id is None
            else forecast_run_id
        )
        capture = self.store.append_capture(
            "capture:review:test",
            CaptureCommand(
                game_pk=game_pk,
                captured_at=(
                    self.game.captured_at
                    if record_captured_at is None
                    else record_captured_at
                ),
                first_pitch=(
                    self.game.identity.first_pitch
                    if record_first_pitch is None
                    else record_first_pitch
                ),
                lineup_state=record_lineup_state,
                out_of_sample_eligible=True,
                normalized_input_hash=record_input_hash,
                payload=capture_payload,
            ),
        )
        forecast = self.store.append_forecast(
            ForecastCommand(
                forecast_run_id=record_forecast_run_id,
                model_version=(
                    self.engine_forecast.model_version
                    if model_version is None
                    else model_version
                ),
                simulation_config_hash=(
                    self.simulation_config_hash
                    if simulation_config_hash is None
                    else simulation_config_hash
                ),
                normalized_input_hash=record_input_hash,
                outputs=outputs,
            )
        )
        link = self.store.link_capture_forecast(
            capture.capture_id,
            forecast.forecast_run_id,
        )
        self.store.append_market_observation(
            MarketObservationCommand(
                game_pk=game_pk,
                source_name="sportsbook",
                source_record=f"event-{game_pk}-{MARKET_KEY}",
                source_revision="1",
                observation_kind="entry",
                market_key=MARKET_KEY,
                market_type="ml",
                settlement_format="two_way",
                selected_side=market_observation_side,
                line=None,
                selected_decimal_price=1.91,
                opposing_decimal_price=1.91,
                observed_at=self.game.captured_at - timedelta(minutes=5),
                available_at=self.game.captured_at - timedelta(minutes=4),
            )
        )
        away_f5, home_f5, away_full, home_full = score
        result_component_values = dict(
            self._default_component_values()
            if component_values is None
            else component_values
        )
        official_scores = {
            "away_f5_runs": away_f5,
            "home_f5_runs": home_f5,
            "away_full_runs": away_full,
            "home_full_runs": home_full,
        }
        for component_id, score_value in official_scores.items():
            if component_id in result_component_values:
                result_component_values[component_id] = float(score_value)
        result = self.store.append_official_result(
            OfficialResultCommand(
                game_pk=game_pk if result_game_pk is None else result_game_pk,
                source_name="MLB",
                source_record=(
                    f"game/{game_pk if result_game_pk is None else result_game_pk}"
                    "/boxscore"
                ),
                source_revision="1",
                finalized_at=(
                    self.game.identity.first_pitch + timedelta(hours=4)
                    if result_finalized_at is None
                    else result_finalized_at
                ),
                away_f5_runs=away_f5,
                home_f5_runs=home_f5,
                away_full_runs=away_full,
                home_full_runs=home_full,
                component_values=result_component_values,
                documented_events=(
                    ({"result_label": "loss"},)
                    if documented_events is None
                    else documented_events
                ),
            )
        )
        return RecordedFixture(
            store=self.store,
            capture=capture,
            forecast=forecast,
            link=link,
            result=result,
            caller_outputs=outputs,
        )

    def correction_command(
        self,
        fixture: RecordedFixture,
        *,
        score=(2, 2, 3, 4),
    ) -> OfficialResultCommand:
        away_f5, home_f5, away_full, home_full = score
        component_values = dict(fixture.result.component_values)
        component_values["away_full_runs"] = float(away_full)
        return OfficialResultCommand(
            game_pk=fixture.result.game_pk,
            source_name=fixture.result.source_name,
            source_record=fixture.result.source_record,
            source_revision="2",
            finalized_at=fixture.result.finalized_at + timedelta(minutes=10),
            away_f5_runs=away_f5,
            home_f5_runs=home_f5,
            away_full_runs=away_full,
            home_full_runs=home_full,
            component_values=component_values,
            documented_events=fixture.result.documented_events,
            supersedes_record_id=fixture.result.envelope.record_id,
        )

    def append_correction(
        self,
        fixture: RecordedFixture,
        *,
        score=(2, 2, 3, 4),
    ) -> OfficialResultRecord:
        return self.store.append_official_result(
            self.correction_command(fixture, score=score)
        )

    def test_service_joins_exact_link_capture_forecast_result_and_market(self):
        fixture = self.fixture()

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(
            candidate.capture_forecast_link_record_id,
            fixture.link.envelope.record_id,
        )
        self.assertEqual(candidate.capture_id, fixture.capture.capture_id)
        self.assertEqual(
            candidate.forecast_run_id,
            fixture.forecast.forecast_run_id,
        )
        self.assertEqual(
            candidate.official_result_record_id,
            fixture.result.envelope.record_id,
        )
        self.assertIs(candidate.market_type, MarketType.ML)
        self.assertEqual(candidate.market_result, "win")

        with self.assertRaisesRegex(ValueError, "link_record_id"):
            build_review_candidate(
                self.store,
                fixture.capture.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )
        with self.assertRaisesRegex(ValueError, "result_record_id"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.source_revision,
                MARKET_KEY,
            )

    def test_service_rejects_capture_and_result_game_mismatch(self):
        fixture = self.fixture(result_game_pk=self.game.identity.game_pk + 1)

        with self.assertRaisesRegex(ValueError, "game_pk"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_service_rejects_outcome_finalized_before_capture_or_first_pitch(
        self,
    ):
        cases = (
            (self.game.captured_at, "captured_at"),
            (
                self.game.identity.first_pitch - timedelta(seconds=1),
                "first_pitch",
            ),
        )

        for finalized_at, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(
                        result_finalized_at=finalized_at
                    )

                    with self.assertRaisesRegex(ValueError, expected):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_settlement_uses_frozen_semantics_not_narrative_or_market_record(self):
        fixture = self.fixture(
            documented_events=({"result_label": "loss"},),
            market_observation_side="home",
            score=(0, 2, 4, 3),
        )

        result = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(result.record.market_result, "win")
        self.assertEqual(result.candidate.market_result, "win")

    def test_candidate_reuses_complete_frozen_distribution_and_interval(self):
        component_values = self._default_component_values()
        component_values["home_pitcher_chain_runs_allowed"] = 99.0
        fixture = self.fixture(
            component_values=component_values,
            score=(2, 1, 100, 3),
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )
        forecast = next(
            item
            for item in candidate.component_forecasts
            if item.component_id == "home_pitcher_chain_runs_allowed"
        )
        observation = next(
            item
            for item in candidate.component_observations
            if item.component_id == "home_pitcher_chain_runs_allowed"
        )
        frozen_summary = fixture.forecast.outputs[
            "component_distributions"
        ]["home_pitcher_chain_runs_allowed"]["summary"]

        for quantile in (
            "p05",
            "p10",
            "p20",
            "median",
            "p80",
            "p90",
            "p95",
        ):
            with self.subTest(quantile=quantile):
                self.assertEqual(
                    getattr(forecast.interval, quantile),
                    frozen_summary[quantile],
                )
        self.assertEqual(observation.value, 99.0)
        self.assertNotEqual(forecast.interval.p90, observation.value)

    def test_compact_and_historical_forecasts_build_equivalent_reviews(self):
        historical = self.fixture()
        historical_context = deserialize_forecast_context(
            historical.forecast.outputs,
            away_team=self.game.identity.away_team,
            home_team=self.game.identity.home_team,
        )
        historical_review = build_and_record_review(
            self.store,
            historical.link.envelope.record_id,
            historical.result.envelope.record_id,
            MARKET_KEY,
        )

        with tempfile.TemporaryDirectory() as directory:
            compact_store = LedgerStore(Path(directory) / "ledger.jsonl")
            original_store = self.store
            self.store = compact_store
            try:
                compact = self.fixture(outputs=self._compact_outputs())
                compact_context = deserialize_forecast_context(
                    compact.forecast.outputs,
                    away_team=self.game.identity.away_team,
                    home_team=self.game.identity.home_team,
                )
                compact_review = build_and_record_review(
                    self.store,
                    compact.link.envelope.record_id,
                    compact.result.envelope.record_id,
                    MARKET_KEY,
                )
            finally:
                self.store = original_store

        self.assertEqual(historical_context, compact_context)
        self.assertEqual(
            canonical_json(
                {
                    "classification": historical_review.command.classification,
                    "component_reviews": historical_review.command.component_reviews,
                    "contradicted_paths": historical_review.command.contradicted_paths,
                    "inside_80_ratio": historical_review.command.inside_80_ratio,
                    "market_result": historical_review.command.market_result,
                    "qualifying_exogenous_source_observation_ids": (
                        historical_review.command.qualifying_exogenous_source_observation_ids
                    ),
                }
            ),
            canonical_json(
                {
                    "classification": compact_review.command.classification,
                    "component_reviews": compact_review.command.component_reviews,
                    "contradicted_paths": compact_review.command.contradicted_paths,
                    "inside_80_ratio": compact_review.command.inside_80_ratio,
                    "market_result": compact_review.command.market_result,
                    "qualifying_exogenous_source_observation_ids": (
                        compact_review.command.qualifying_exogenous_source_observation_ids
                    ),
                }
            ),
        )

    def test_missing_exact_pregame_binding_is_not_reconstructed(self):
        def remove_exact_comparison(outputs):
            shadow = outputs["shadow_comparison"]
            shadow["legacy"]["review_bindings"] = []
            comparison = shadow["market_comparisons"][0]
            for field_name in (
                "model_win_probability",
                "fair_decimal_price",
                "market_no_vig_probability",
                "edge",
            ):
                comparison[field_name] = None
            comparison["principal_paths"] = []
            comparison["review_eligible"] = False
            comparison["review_ineligibility_flags"] = [
                "missing_pregame_binding"
            ]

        fixture = self.fixture(mutate_outputs=remove_exact_comparison)
        self.assertFalse(
            fixture.forecast.outputs["shadow_comparison"]["legacy"][
                "review_bindings"
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing_pregame_binding",
        ):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_review_ineligibility_stops_with_stored_flags(self):
        def make_ineligible(outputs):
            comparison = outputs["shadow_comparison"]["market_comparisons"][0]
            comparison["principal_paths"] = []
            comparison["review_eligible"] = False
            comparison["review_ineligibility_flags"] = [
                "missing_component_distribution",
            ]
            outputs["component_distributions"].pop(
                "home_pitcher_chain_runs_allowed"
            )

        fixture = self.fixture(mutate_outputs=make_ineligible)

        with self.assertRaisesRegex(
            ValueError,
            "missing_component_distribution",
        ):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_missing_distribution_or_values_is_not_reconstructed(self):
        for mutation, expected in (
            (
                lambda outputs: outputs["component_distributions"].pop(
                    "home_pitcher_chain_runs_allowed"
                ),
                "missing_component_distribution",
            ),
            (
                lambda outputs: outputs["component_distributions"][
                    "home_pitcher_chain_runs_allowed"
                ].pop("values"),
                "values",
            ),
        ):
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)
                    with self.assertRaisesRegex(ValueError, expected):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_all_component_distributions_and_intervals_are_revalidated(self):
        component_id = "home_f5_runs"

        def replace_values(outputs, target_component_id, value):
            distribution = outputs["component_distributions"][
                target_component_id
            ]
            distribution["values"] = [value] * outputs["simulation_count"]
            for summary in (
                distribution["summary"],
                outputs["intervals"][target_component_id],
            ):
                for field_name in summary:
                    summary[field_name] = value

        def change_internal_component_id(outputs):
            outputs["component_distributions"][component_id][
                "component_id"
            ] = "away_f5_runs"

        def change_unrelated_value_count(outputs):
            outputs["component_distributions"][component_id]["values"].pop()

        def change_unrelated_summary(outputs):
            summary = outputs["component_distributions"][component_id][
                "summary"
            ]
            for field_name in summary:
                summary[field_name] = 999

        def remove_unrelated_interval(outputs):
            outputs["intervals"].pop(component_id)

        def change_unrelated_interval(outputs):
            summary = outputs["intervals"][component_id]
            for field_name in summary:
                summary[field_name] = -1

        def add_fabricated_component(outputs):
            fabricated_id = "away_fabricated_component"
            distribution = json.loads(
                canonical_json(outputs["component_distributions"][component_id])
            )
            distribution["component_id"] = fabricated_id
            outputs["component_distributions"][fabricated_id] = distribution
            outputs["intervals"][fabricated_id] = json.loads(
                canonical_json(distribution["summary"])
            )

        def change_bullpen_diagnostic_segment_id(outputs):
            outputs["component_diagnostics"]["away_bullpen"]["segments"][0][
                "segment_id"
            ] = "bullpen:SEA:tampered"

        def use_negative_unrelated_count(outputs):
            replace_values(outputs, component_id, -1)

        def use_nonbinary_unrelated_exposure(outputs):
            exposure_id = next(
                candidate_id
                for candidate_id in outputs["component_distributions"]
                if "_bullpen_segment_" in candidate_id
                and candidate_id.endswith("_exposed")
            )
            replace_values(outputs, exposure_id, 2)

        mutations = (
            change_internal_component_id,
            change_unrelated_value_count,
            change_unrelated_summary,
            remove_unrelated_interval,
            change_unrelated_interval,
            add_fabricated_component,
            change_bullpen_diagnostic_segment_id,
            use_negative_unrelated_count,
            use_nonbinary_unrelated_exposure,
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)

                    with self.assertRaisesRegex(
                        ValueError,
                        "component|simulation_count|summary|interval|diagnostic",
                    ):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_missing_official_component_is_rejected_even_when_score_exists(self):
        component_values = self._default_component_values()
        component_values.pop("away_full_runs")
        fixture = self.fixture(
            component_values=component_values,
            score=(2, 1, 12, 3),
        )

        with self.assertRaisesRegex(ValueError, "away_full_runs"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_captured_team_alias_resolves_to_away_namespace(self):
        def use_team_alias(outputs):
            semantic = outputs["market_inputs"][MARKET_KEY]
            semantic["selected_side"] = "SEA"
            outputs["markets"][MARKET_KEY]["semantic_fingerprint"] = (
                market_semantic_fingerprint(
                    market_key=MARKET_KEY,
                    market_type=MarketType.ML,
                    settlement_format=SettlementFormat.TWO_WAY,
                    selected_side="SEA",
                    line=None,
                    selected_decimal_price=semantic[
                        "selected_decimal_price"
                    ],
                    opposing_decimal_price=semantic[
                        "opposing_decimal_price"
                    ],
                )
            )

        fixture = self.fixture(
            mutate_outputs=use_team_alias,
            score=(0, 2, 4, 3),
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(candidate.market_result, "win")

    def test_team_abbreviation_component_ids_are_rejected(self):
        def use_team_component_id(outputs):
            shadow = outputs["shadow_comparison"]
            for paths in (
                shadow["market_comparisons"][0]["principal_paths"],
                shadow["legacy"]["review_bindings"][0]["principal_paths"],
            ):
                paths[1]["component_ids"][0] = "SEA_full_extra_base_hits"

        fixture = self.fixture(mutate_outputs=use_team_component_id)

        with self.assertRaisesRegex(ValueError, "away/home namespace"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_only_supported_documented_events_with_source_ids_are_converted(self):
        target = "home_pitcher_chain_runs_allowed"
        component_values = self._default_component_values()
        component_values[target] = 99.0
        fixture = self.fixture(
            component_values=component_values,
            score=(2, 1, 100, 3),
            documented_events=(
                {
                    "event_type": "injury",
                    "affected_component_id": target,
                    "documented_source_observation_id": "official:injury:1",
                },
                {
                    "event_type": "ejection",
                    "affected_component_id": target,
                },
                {
                    "event_type": "generic_predeclared",
                    "affected_component_id": target,
                    "documented_source_observation_id": "official:generic:1",
                },
                {
                    "event_type": "defensive_error_cluster",
                    "affected_component_id": target,
                    "documented_source_observation_id": "official:error:1",
                    "unearned_runs": 3,
                },
                {
                    "event_type": "home_run",
                    "source_observation_id": "official:home-run:1",
                },
            ),
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )
        result = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(
            tuple(
                (
                    event.event_type,
                    event.documented_source_observation_id,
                )
                for event in candidate.exogenous_events
            ),
            (
                ("defensive_error_cluster", "official:error:1"),
                ("injury", "official:injury:1"),
            ),
        )
        self.assertEqual(result.record.classification, "D")
        expected_source_ids = (
            "official:error:1",
            "official:injury:1",
        )
        self.assertEqual(
            result.classification.qualifying_exogenous_source_observation_ids,
            expected_source_ids,
        )
        self.assertTrue(
            hasattr(
                result.command,
                "qualifying_exogenous_source_observation_ids",
            ),
            "ReviewCommand drops qualifying exogenous source IDs",
        )
        self.assertEqual(
            result.command.qualifying_exogenous_source_observation_ids,
            expected_source_ids,
        )
        self.assertEqual(
            result.record.qualifying_exogenous_source_observation_ids,
            expected_source_ids,
        )
        reloaded = next(
            LedgerStore(self.store.path).iter_records("review")
        )
        self.assertEqual(
            reloaded.qualifying_exogenous_source_observation_ids,
            expected_source_ids,
        )

    def test_event_without_source_id_cannot_persist_as_classification_d(self):
        target = "home_pitcher_chain_runs_allowed"
        component_values = self._default_component_values()
        component_values[target] = 99.0
        fixture = self.fixture(
            component_values=component_values,
            score=(2, 1, 100, 3),
            documented_events=(
                {
                    "event_type": "injury",
                    "affected_component_id": target,
                },
            ),
        )

        result = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertNotEqual(result.record.classification, "D")
        self.assertEqual(
            result.classification.qualifying_exogenous_source_observation_ids,
            (),
        )
        self.assertTrue(
            hasattr(
                result.record,
                "qualifying_exogenous_source_observation_ids",
            ),
            "ReviewRecord drops the empty qualifying source-ID contract",
        )
        self.assertEqual(
            result.record.qualifying_exogenous_source_observation_ids,
            (),
        )

    def test_same_review_retry_returns_one_immutable_record(self):
        fixture = self.fixture()

        first = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )
        second = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(first.record.review_id, second.record.review_id)
        self.assertEqual(len(tuple(self.store.iter_records("review"))), 1)
        with self.assertRaises(FrozenInstanceError):
            first.record.market_result = "loss"

    def test_required_active_capture_is_forwarded_only_when_supplied(self):
        fixture = self.fixture()
        guarded_store = RequireActiveCaptureReviewAppend(self.store)

        result = build_and_record_review(
            guarded_store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
            required_active_capture_id=fixture.capture.capture_id,
        )

        self.assertEqual(
            guarded_store.required_active_capture_id,
            fixture.capture.capture_id,
        )
        self.assertEqual(result.record.capture_id, fixture.capture.capture_id)

    def test_retry_after_partial_failure_reuses_the_appended_review(self):
        fixture = self.fixture()
        fail_once = FailOnceAfterReviewAppend(self.store)

        with self.assertRaisesRegex(RuntimeError, "after review append"):
            build_and_record_review(
                fail_once,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )
        self.assertEqual(len(tuple(self.store.iter_records("review"))), 1)

        retried = build_and_record_review(
            fail_once,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(len(tuple(self.store.iter_records("review"))), 1)
        self.assertEqual(
            retried.record.review_id,
            next(self.store.iter_records("review")).review_id,
        )

    def test_correction_between_build_and_append_rejects_stale_review(self):
        fixture = self.fixture()
        racing_store = CorrectResultBeforeReviewAppend(
            self.store,
            self.correction_command(fixture),
        )

        with self.assertRaisesRegex(LedgerReferenceError, "latest"):
            build_and_record_review(
                racing_store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

        self.assertEqual(len(tuple(self.store.iter_records("review"))), 0)
        self.assertEqual(
            len(tuple(self.store.iter_records("official_result"))),
            2,
        )

    def test_result_correction_creates_new_review_and_keeps_old_immutable(self):
        fixture = self.fixture()
        first = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )
        first_payload = canonical_json(first.record)
        correction = self.append_correction(fixture)

        second = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            correction.envelope.record_id,
            MARKET_KEY,
        )

        self.assertNotEqual(first.record.review_id, second.record.review_id)
        self.assertEqual(first.record.market_result, "win")
        self.assertEqual(second.record.market_result, "loss")
        self.assertEqual(canonical_json(first.record), first_payload)
        self.assertEqual(len(tuple(self.store.iter_records("review"))), 2)
        with self.assertRaisesRegex(ValueError, "superseded"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_review_accepts_exact_mlb_original_and_correction_results(self):
        fixture = self.fixture()

        original = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )
        correction = self.append_correction(fixture)
        corrected = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            correction.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(fixture.result.source_name, "MLB")
        self.assertEqual(correction.source_name, "MLB")
        self.assertEqual(
            original.official_result_record_id,
            fixture.result.envelope.record_id,
        )
        self.assertEqual(
            corrected.official_result_record_id,
            correction.envelope.record_id,
        )

    def test_review_rejects_latest_result_from_noncanonical_source(self):
        for source_name in ("unofficial-blog", "unofficial-MLB-blog"):
            with self.subTest(source_name=source_name):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture()
                    correction = self.store.append_official_result(
                        replace(
                            self.correction_command(fixture),
                            source_name=source_name,
                        )
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "source_name.*MLB|official MLB",
                    ):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            correction.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_malformed_forecast_schema_and_fingerprint_are_rejected(self):
        mutations = (
            (
                lambda outputs: outputs.update(forecast_run_id=123),
                "forecast_run_id",
            ),
            (
                lambda outputs: outputs.pop("market_inputs"),
                "market_inputs",
            ),
            (
                lambda outputs: outputs["markets"][MARKET_KEY].update(
                    semantic_fingerprint="0" * 64
                ),
                "semantic_fingerprint",
            ),
            (
                lambda outputs: outputs["market_inputs"][MARKET_KEY].update(
                    market_type="invented_market"
                ),
                "market_type",
            ),
        )

        for mutation, expected in mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)
                    with self.assertRaisesRegex(ValueError, expected):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_market_inputs_and_evaluations_require_exact_key_sets(self):
        def add_extra_market_input(outputs):
            semantic = json.loads(
                canonical_json(outputs["market_inputs"][MARKET_KEY])
            )
            semantic["market_key"] = "ghost_market"
            outputs["market_inputs"]["ghost_market"] = semantic

        def add_extra_market_evaluation(outputs):
            outputs["markets"]["ghost_market"] = json.loads(
                canonical_json(outputs["markets"][MARKET_KEY])
            )

        for mutation in (add_extra_market_input, add_extra_market_evaluation):
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)

                    with self.assertRaisesRegex(ValueError, "keys.*exactly match"):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_forecast_markets_must_exactly_match_canonical_capture_markets(self):
        def forge_selected_side_and_consistent_metrics(outputs):
            semantic = outputs["market_inputs"][MARKET_KEY]
            semantic["selected_side"] = "home"
            fingerprint = market_semantic_fingerprint(
                market_key=MARKET_KEY,
                market_type=MarketType.ML,
                settlement_format=SettlementFormat.TWO_WAY,
                selected_side="home",
                line=None,
                selected_decimal_price=semantic["selected_decimal_price"],
                opposing_decimal_price=semantic["opposing_decimal_price"],
            )
            evaluation = replace(
                price_market(
                    OutcomeProbabilities(win=0.5, push=0.0, loss=0.5),
                    semantic["selected_decimal_price"],
                    semantic["opposing_decimal_price"],
                ),
                semantic_fingerprint=fingerprint,
            )
            outputs["markets"][MARKET_KEY] = json.loads(
                canonical_json(evaluation)
            )
            comparison = outputs["shadow_comparison"][
                "market_comparisons"
            ][0]
            comparison.update(
                model_win_probability=(
                    evaluation.model_conditional_win_probability
                ),
                fair_decimal_price=evaluation.fair_decimal_price,
                market_no_vig_probability=(
                    evaluation.market_no_vig_probability
                ),
                edge=evaluation.edge,
            )

        fixture = self.fixture(
            mutate_outputs=forge_selected_side_and_consistent_metrics
        )

        with self.assertRaisesRegex(ValueError, "capture.*markets|market_inputs"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_capture_team_alias_matches_canonical_forecast_market(self):
        def use_capture_team_alias(payload):
            market = next(
                market
                for market in payload["markets"]
                if market["market_key"] == MARKET_KEY
            )
            market["selected_side"] = "SEA"

        fixture = self.fixture(
            mutate_capture_payload=use_capture_team_alias,
            score=(0, 2, 4, 3),
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(candidate.market_result, "win")

    def test_capture_payload_and_record_timing_must_match_and_be_pregame(self):
        def canonical_timestamp(value):
            return value.isoformat().replace("+00:00", "Z")

        def shift_payload_capture(payload):
            payload["captured_at"] = canonical_timestamp(
                self.game.captured_at + timedelta(minutes=1)
            )

        def move_payload_capture_post_pitch(payload):
            payload["captured_at"] = canonical_timestamp(
                self.game.identity.first_pitch + timedelta(minutes=1)
            )

        cases = (
            ({"mutate_capture_payload": shift_payload_capture}, "captured_at"),
            (
                {"mutate_capture_payload": move_payload_capture_post_pitch},
                "captured_at|first_pitch",
            ),
            (
                {
                    "record_captured_at": (
                        self.game.identity.first_pitch + timedelta(minutes=1)
                    )
                },
                "captured_at|first_pitch",
            ),
        )
        for fixture_kwargs, expected in cases:
            with self.subTest(fixture_kwargs=fixture_kwargs):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(**fixture_kwargs)

                    with self.assertRaisesRegex(ValueError, expected):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_capture_lineup_state_must_match_frozen_payload(self):
        def project_away_lineup(payload):
            payload["away_lineup"]["state"] = "projected"

        cases = (
            ({"record_lineup_state": "projected"}, "confirmed payload"),
            (
                {"mutate_capture_payload": project_away_lineup},
                "projected payload",
            ),
        )
        for fixture_kwargs, label in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(**fixture_kwargs)

                    with self.assertRaisesRegex(ValueError, "lineup_state"):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_review_accepts_confirmed_mixed_and_projected_lineups(self):
        cases = (
            (LineupState.CONFIRMED, LineupState.CONFIRMED, "confirmed"),
            (LineupState.CONFIRMED, LineupState.PROJECTED, "projected"),
            (LineupState.PROJECTED, LineupState.PROJECTED, "projected"),
        )
        for index, (away_state, home_state, expected) in enumerate(cases):
            with self.subTest(away_state=away_state, home_state=home_state):
                with tempfile.TemporaryDirectory() as directory:
                    store = LedgerStore(Path(directory) / "ledger.jsonl")
                    game = replace(
                        self.game,
                        away_lineup=replace(
                            self.game.away_lineup,
                            state=away_state,
                        ),
                        home_lineup=replace(
                            self.game.home_lineup,
                            state=home_state,
                        ),
                    )
                    forecast = run_forecast(game, _fast_config())
                    records = record_shadow_run(
                        store,
                        game,
                        _legacy_snapshot(),
                        forecast,
                        ShadowRunIdempotencyKeys(
                            capture=f"capture:review:lineup:{index}"
                        ),
                    )
                    component_values = {}
                    for path in _principal_paths():
                        for component_id in path.component_ids:
                            summary = records.forecast.outputs[
                                "component_distributions"
                            ][component_id]["summary"]
                            component_values[component_id] = float(
                                summary["median"]
                            )
                    component_values["away_full_runs"] = 4.0
                    result = store.append_official_result(
                        OfficialResultCommand(
                            game_pk=game.identity.game_pk,
                            source_name="MLB",
                            source_record=(
                                f"game/{game.identity.game_pk}/boxscore"
                            ),
                            source_revision="1",
                            finalized_at=(
                                game.identity.first_pitch + timedelta(hours=4)
                            ),
                            away_f5_runs=2,
                            home_f5_runs=1,
                            away_full_runs=4,
                            home_full_runs=3,
                            component_values=component_values,
                            documented_events=(),
                        )
                    )

                    candidate = build_review_candidate(
                        store,
                        records.link.envelope.record_id,
                        result.envelope.record_id,
                        MARKET_KEY,
                    )

                    self.assertEqual(records.capture.lineup_state, expected)
                    self.assertEqual(candidate.capture_id, records.capture.capture_id)

    def test_market_evaluation_revalidates_probability_derived_fields(self):
        def forge_probability_and_edge(outputs):
            evaluation = outputs["markets"][MARKET_KEY]
            probabilities = evaluation["probabilities"]
            probabilities["win"] = 0.8
            probabilities["push"] = 0.0
            probabilities["loss"] = 0.2
            evaluation["edge"] = 0.123456
            outputs["shadow_comparison"]["market_comparisons"][0][
                "edge"
            ] = 0.123456

        fixture = self.fixture(mutate_outputs=forge_probability_and_edge)

        with self.assertRaisesRegex(
            ValueError,
            "probabilities|derived|edge",
        ):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_review_rejects_distribution_market_output_mismatch_before_append(
        self,
    ):
        def change_score_distribution_without_changing_market(outputs):
            away = outputs["component_distributions"]["away_full_runs"]
            home_values = outputs["component_distributions"][
                "home_full_runs"
            ]["values"]
            for index, (away_runs, home_runs) in enumerate(
                zip(away["values"], home_values)
            ):
                if away_runs <= home_runs:
                    away["values"][index] = home_runs + 1
                    break
            else:
                self.fail("fixture requires one non-away-win simulation")
            summary = json.loads(
                canonical_json(_summarize(tuple(away["values"])))
            )
            away["summary"] = summary
            outputs["intervals"]["away_full_runs"] = summary

        fixture = self.fixture(
            mutate_outputs=(
                change_score_distribution_without_changing_market
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "market outputs|score distributions",
        ):
            build_and_record_review(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

        self.assertEqual(tuple(self.store.iter_records("review")), ())

    def test_non_push_market_rejects_frozen_push_probability(self):
        def add_consistent_ml_push_probability(outputs):
            semantic = outputs["market_inputs"][MARKET_KEY]
            evaluation = replace(
                price_market(
                    OutcomeProbabilities(win=0.5, push=0.1, loss=0.4),
                    semantic["selected_decimal_price"],
                    semantic["opposing_decimal_price"],
                ),
                semantic_fingerprint=outputs["markets"][MARKET_KEY][
                    "semantic_fingerprint"
                ],
            )
            outputs["markets"][MARKET_KEY] = json.loads(
                canonical_json(evaluation)
            )
            comparison = outputs["shadow_comparison"][
                "market_comparisons"
            ][0]
            comparison.update(
                model_win_probability=(
                    evaluation.model_conditional_win_probability
                ),
                fair_decimal_price=evaluation.fair_decimal_price,
                market_no_vig_probability=(
                    evaluation.market_no_vig_probability
                ),
                edge=evaluation.edge,
            )

        fixture = self.fixture(
            mutate_outputs=add_consistent_ml_push_probability
        )

        with self.assertRaisesRegex(ValueError, "push"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_ordinary_market_cannot_be_swapped_to_arbitrary_unevaluated(self):
        def swap_to_unevaluated(outputs):
            evaluation = outputs["markets"][MARKET_KEY]
            for field_name in (
                "probabilities",
                "model_conditional_win_probability",
                "fair_decimal_price",
                "market_no_vig_probability",
                "edge",
                "expected_value",
                "selected_decimal_price",
                "opposing_decimal_price",
            ):
                evaluation[field_name] = None
            evaluation["evaluated"] = False
            evaluation["reason"] = "arbitrary unevaluated replacement"
            comparison = outputs["shadow_comparison"]["market_comparisons"][0]
            for field_name in (
                "model_win_probability",
                "fair_decimal_price",
                "market_no_vig_probability",
                "edge",
            ):
                comparison[field_name] = None

        fixture = self.fixture(mutate_outputs=swap_to_unevaluated)

        with self.assertRaisesRegex(ValueError, "unevaluated|evaluated"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_shadow_schema_rejects_missing_and_additional_fields(self):
        mutations = (
            lambda outputs: outputs["shadow_comparison"]["legacy"].pop(
                "notes"
            ),
            lambda outputs: outputs["shadow_comparison"][
                "market_comparisons"
            ][0].update(unexpected_field=True),
        )

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)

                    with self.assertRaisesRegex(ValueError, "schema"):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_shadow_rating_must_match_the_frozen_legacy_candidate(self):
        def change_play_to_pass(outputs):
            outputs["shadow_comparison"]["market_comparisons"][0][
                "legacy_rating"
            ] = "pass"

        fixture = self.fixture(mutate_outputs=change_play_to_pass)

        with self.assertRaisesRegex(ValueError, "legacy candidate"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_shadow_candidate_and_comparison_order_must_match(self):
        def change_comparison_index(outputs):
            outputs["shadow_comparison"]["market_comparisons"][0][
                "legacy_candidate_index"
            ] = 1

        fixture = self.fixture(mutate_outputs=change_comparison_index)

        with self.assertRaisesRegex(ValueError, "candidate order"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_root_forecast_run_id_must_match_record_and_shadow(self):
        def change_root_forecast_run_id(outputs):
            outputs["forecast_run_id"] = "0" * 64

        fixture = self.fixture(mutate_outputs=change_root_forecast_run_id)
        self.assertEqual(
            fixture.forecast.outputs["shadow_comparison"]["forecast_run_id"],
            fixture.forecast.forecast_run_id,
        )

        with self.assertRaisesRegex(ValueError, "outputs forecast_run_id"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_forecast_model_version_must_match_frozen_config_version(self):
        forged_model_version = "forged-model-v2"
        forged_run_id = build_forecast_run_id_from_capture_payload(
            self.capture_payload,
            forged_model_version,
            self.engine_forecast.simulation_config,
        )

        def forge_version_bound_ids(outputs):
            outputs["forecast_run_id"] = forged_run_id
            outputs["shadow_comparison"]["forecast_run_id"] = forged_run_id

        fixture = self.fixture(
            mutate_outputs=forge_version_bound_ids,
            forecast_run_id=forged_run_id,
            model_version=forged_model_version,
        )

        with self.assertRaisesRegex(
            ValueError,
            "model_version.*config_version|simulation_config.*version",
        ):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_simulation_config_hash_must_match_frozen_config(self):
        fixture = self.fixture(simulation_config_hash="f" * 64)

        with self.assertRaisesRegex(ValueError, "simulation_config_hash"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_forged_record_output_shadow_ids_are_rejected_by_capture_semantics(
        self,
    ):
        forged_id = "f" * 64

        def forge_all_ids(outputs):
            outputs["forecast_run_id"] = forged_id
            outputs["shadow_comparison"]["forecast_run_id"] = forged_id

        fixture = self.fixture(
            mutate_outputs=forge_all_ids,
            forecast_run_id=forged_id,
            stored_normalized_input_hash=forged_id,
            simulation_config_hash=forged_id,
        )

        with self.assertRaisesRegex(
            ValueError,
            "forecast_run_id.*capture semantics|normalized_input_hash",
        ):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_selected_comparison_must_match_exact_frozen_review_binding(self):
        def change_binding_market_key(outputs):
            outputs["shadow_comparison"]["legacy"]["review_bindings"][0][
                "market_key"
            ] = "other_market"

        def change_binding_index(outputs):
            outputs["shadow_comparison"]["legacy"]["review_bindings"][0][
                "legacy_candidate_index"
            ] = 1

        def change_binding_legacy_market(outputs):
            outputs["shadow_comparison"]["legacy"]["review_bindings"][0][
                "legacy_market"
            ] = "full_total"

        def change_binding_paths(outputs):
            outputs["shadow_comparison"]["legacy"]["review_bindings"][0][
                "principal_paths"
            ][0]["component_ids"] = ["home_f5_runs"]

        for mutation in (
            change_binding_market_key,
            change_binding_index,
            change_binding_legacy_market,
            change_binding_paths,
        ):
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(mutate_outputs=mutation)

                    with self.assertRaisesRegex(ValueError, "review binding"):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_malformed_unrelated_shadow_comparison_is_not_silently_skipped(self):
        def add_malformed_comparison(outputs):
            outputs["shadow_comparison"]["market_comparisons"].insert(
                0,
                "not-a-comparison",
            )

        fixture = self.fixture(mutate_outputs=add_malformed_comparison)

        with self.assertRaisesRegex(ValueError, "shadow market comparison"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    @staticmethod
    def _append_ghost_comparison(outputs, *, eligible):
        shadow = outputs["shadow_comparison"]
        candidate = json.loads(
            canonical_json(shadow["legacy"]["candidates"][0])
        )
        binding = json.loads(
            canonical_json(shadow["legacy"]["review_bindings"][0])
        )
        comparison = json.loads(
            canonical_json(shadow["market_comparisons"][0])
        )
        binding["legacy_candidate_index"] = 1
        binding["market_key"] = "ghost_ml"
        comparison["legacy_candidate_index"] = 1
        comparison["market_key"] = "ghost_ml"
        if not eligible:
            for field_name in (
                "model_win_probability",
                "fair_decimal_price",
                "market_no_vig_probability",
                "edge",
            ):
                comparison[field_name] = None
            comparison["principal_paths"] = []
            comparison["review_eligible"] = False
            comparison["review_ineligibility_flags"] = [
                "unknown_market_key"
            ]
        shadow["legacy"]["candidates"].append(candidate)
        shadow["legacy"]["final_priority"].append(candidate["market"])
        shadow["legacy"]["review_bindings"].append(binding)
        shadow["market_comparisons"].append(comparison)

    def test_unrelated_eligible_ghost_market_invalidates_normal_review(self):
        fixture = self.fixture(
            mutate_outputs=lambda outputs: self._append_ghost_comparison(
                outputs,
                eligible=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "unknown_market_key"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_unrelated_unknown_ineligible_ghost_market_remains_valid(self):
        fixture = self.fixture(
            mutate_outputs=lambda outputs: self._append_ghost_comparison(
                outputs,
                eligible=False,
            )
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(candidate.market_key, MARKET_KEY)

    def test_shadow_metrics_must_match_full_forecast_context(self):
        def forge_shadow_metric(outputs):
            outputs["shadow_comparison"]["market_comparisons"][0][
                "edge"
            ] = 0.123456

        fixture = self.fixture(mutate_outputs=forge_shadow_metric)

        with self.assertRaisesRegex(ValueError, "market evaluation.*edge"):
            build_review_candidate(
                self.store,
                fixture.link.envelope.record_id,
                fixture.result.envelope.record_id,
                MARKET_KEY,
            )

    def test_ineligible_flags_must_match_full_forecast_context(self):
        def claim_flag(flag):
            def mutate(outputs):
                comparison = outputs["shadow_comparison"][
                    "market_comparisons"
                ][0]
                comparison["principal_paths"] = []
                comparison["review_eligible"] = False
                comparison["review_ineligibility_flags"] = [flag]

            return mutate

        for flag in (
            "unknown_market_key",
            "missing_component_distribution",
            "invalid_path_count",
        ):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as directory:
                    self.store = LedgerStore(Path(directory) / "ledger.jsonl")
                    fixture = self.fixture(
                        mutate_outputs=claim_flag(flag)
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        f"inconsistent.*{flag}",
                    ):
                        build_review_candidate(
                            self.store,
                            fixture.link.envelope.record_id,
                            fixture.result.envelope.record_id,
                            MARKET_KEY,
                        )

    def test_nonstring_unsupported_event_type_is_ignored(self):
        fixture = self.fixture(
            documented_events=(
                {
                    "event_type": [],
                    "affected_component_id": (
                        "home_pitcher_chain_runs_allowed"
                    ),
                    "documented_source_observation_id": "official:event:1",
                },
            ),
        )

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(candidate.exogenous_events, ())

    def test_review_command_copies_complete_classification_deterministically(self):
        fixture = self.fixture()

        result = build_and_record_review(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(
            result.command.classification,
            result.classification.classification,
        )
        self.assertEqual(
            result.command.contradicted_paths,
            tuple(
                path.value for path in result.classification.contradicted_paths
            ),
        )
        self.assertEqual(
            tuple(
                review["evidence_path"]
                for review in result.command.component_reviews
            ),
            tuple(
                review.evidence_path.value
                for review in result.classification.path_reviews
            ),
        )
        self.assertEqual(
            result.record.component_reviews,
            result.command.component_reviews,
        )

    def test_ledger_snapshot_isolated_from_caller_output_mutation(self):
        fixture = self.fixture()
        fixture.caller_outputs["market_inputs"][MARKET_KEY][
            "selected_side"
        ] = "home"
        fixture.caller_outputs["component_distributions"][
            "home_pitcher_chain_runs_allowed"
        ]["summary"]["p90"] = 999

        candidate = build_review_candidate(
            self.store,
            fixture.link.envelope.record_id,
            fixture.result.envelope.record_id,
            MARKET_KEY,
        )

        self.assertEqual(candidate.market_result, "win")
        interval = next(
            forecast.interval
            for forecast in candidate.component_forecasts
            if forecast.component_id == "home_pitcher_chain_runs_allowed"
        )
        self.assertNotEqual(interval.p90, 999)


if __name__ == "__main__":
    unittest.main()
