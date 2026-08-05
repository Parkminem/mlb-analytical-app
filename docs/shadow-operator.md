# Shadow Operator CLI 운영 가이드

Shadow Operator는 외부에서 준비한 버전 고정 JSON 번들을 받아 예상 라인업, 확정 라인업, 경기 후 복기를 append-only 원장에 기록합니다. 라이브 출처 수집과 정규화는 이 CLI 밖에서 수동 또는 외부 절차로 수행합니다.

## 요청과 명령 매핑

| 자연어 요청 | 번들 stage | 실행 명령 |
|---|---|---|
| "예상 라인업으로 경기 전 분석을 기록해줘" | `projected` | `.venv/bin/python -m analysis.operator projected --bundle <path>` |
| "확정 라인업으로 다시 기록하고 동결해줘" | `confirmed` | `.venv/bin/python -m analysis.operator confirmed --bundle <path>` |
| "공식 결과로 복기하고 보고서를 갱신해줘" | `review` | `.venv/bin/python -m analysis.operator review --bundle <path>` |

세 명령 모두 strict JSON bundle만 받으며 알 수 없는 필드, 잘못된 stage, 타입 강제 변환, 시간대 없는 timestamp를 거부합니다. 새 `projected`와 `confirmed` 실행은 JSON integer `schema_version: 2`와 명시적인 `authority: "current"`를 모두 요구합니다. `review` 계약은 `schema_version: 1`로 유지됩니다.

새 확정 라인업 pregame envelope의 축약 예시는 다음과 같습니다. 예상 라인업은 `stage`만 `projected`로 바꿉니다.

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "confirmed"
}
```

## 번들 역할

- `projected`: schema-v2/current envelope에 적어도 한 팀이 예상 라인업인 완전한 `game_input`, 동결된 `legacy_decision`, `forecast_config_version`을 담습니다. `captured_at`은 첫 투구 전이어야 합니다.
- `confirmed`: schema-v2/current envelope에 양 팀이 확정 라인업인 같은 종류의 pregame payload를 담습니다. 성공한 capture, forecast, link가 경기 후 review의 동결 기준입니다.
- `review`: schema v1을 유지하며 confirmed 성공 출력의 정확한 세 ID, 공식 MLB 결과 revision, 선택적 closing observation을 담습니다. 새 pregame 입력이나 forecast를 받지 않습니다.

각 버전 번들은 역할이 섞이지 않습니다. `projected` 번들을 `confirmed` 명령에 넘기는 식의 stage/type 불일치는 실패합니다. 원본 MLB, RotoWire, Savant/Statcast, market 데이터를 가져와 domain 값과 provenance로 정규화하는 작업은 계속 외부 또는 수동 책임입니다. 전체 schema를 문서에 복제하지 않으며 실제 계약은 `analysis/operator/schemas.py`와 domain dataclass가 기준입니다.

Fresh pregame 실행 객체는 raw mapping 또는 JSON을 public `ProjectedBundle`/`ConfirmedBundle` validation 경로로 검증해 반환된 정확한 객체 graph만 지원합니다. `model_construct`, `copy.copy`, `copy.deepcopy`, `model_copy`로 만든 bundle/payload/decision graph와 미리 만든 nested decision input은 새 실행 권한을 이어받지 않습니다. 별도 실행 객체가 필요하면 원본을 canonical current JSON으로 serialize한 뒤 public parser로 다시 검증해야 합니다. Runner는 caller graph의 권한과 불변식을 확인하고 같은 canonical bytes를 strict parse한 내부 snapshot만 `_run_pregame`에 전달하므로 snapshot 생성 뒤 caller 변경은 실행이나 저장 결과에 반영되지 않습니다. 임의의 private `object.__setattr__` 또는 `__class__` 변경은 지원 caller API가 아닙니다.

모든 schema-v1 `projected`/`confirmed` shape은 candidate compatibility 필드가 없는 relaxed v1과 필드가 있는 strict v1 모두 역사적 exact replay 전용입니다. 입력은 설정된 관리 번들 루트의 인증된 불변 파일이어야 하고 ledger에 정확히 일치하는 capture, forecast, link가 이미 있어야 합니다. 관리 루트 밖의 v1, exact ledger 관계가 없는 v1, 아직 실행하지 않은 pending v1은 fresh 실행으로 승격되지 않습니다. Pending v1은 schema v2/current 번들로 다시 생성해야 하며, 거부 시 ledger나 관리 번들 bytes를 쓰거나 바꾸지 않습니다.

고정 MLB 모델 버전 `2026-08-05`은 번들 schema version과 독립적입니다. Schema v2 전환은 MLB 점수화, F5 side/F5 O/U, full-game market engine, same-game assembler, legacy decision authority, Shadow의 비권위 비교 모드를 변경하지 않습니다.

Pregame 번들의 `forecast_config_version`은 `config/forecasting/<version>.json`의 완전한 불변 설정을 선택합니다. CLI에서 임의 설정 override나 precomputed forecast를 주입할 수 없습니다.

## Recommended Review Eligibility Gate

- every new projected or confirmed `play`/`lean` candidate must be review-eligible;
- the current legacy model version is `2026-08-05`, and every fresh recommended full-game ML must declare and bind `bullpen_protection` or `late_bullpen_exposure`;
- every fresh recommended F5 or full-game under must bind same-namespace `starter_suppression` and `lineup_suppression` paths for two-sided review coverage;
- a current full-game ML `play`/`lean` candidate with score `72` or higher needs a directionally valid starter path: `starter_suppression` binds the selected side's own starter run prevention, while `starter_collapse` binds opposing-starter failure. Missing evidence is capped at `71`, and a fresh submission above the cap is rejected;
- a current `play`/`lean` candidate on a verified full-game over line of `9.0` or higher needs two distinct canonical routes covering away and home scoring directions. Conversion paths score for their batting side; `starter_collapse` and `late_bullpen_exposure` score for the side opposing the bound pitching side. Suppression/protection paths do not count as over scoring routes. Missing coverage is capped at `69`, and a fresh submission above the cap is rejected;
- the Operator does not infer or inject the required bullpen path;
- `pass` candidates are excluded from the hard gate;
- static validation runs before simulation and dynamic validation before append;
- managed-root 인증과 exact ledger 관계가 있는 schema-v1 pregame artifact만 historical replay로 반환됩니다;
- a stale legacy model version cannot create a new projected or confirmed record;
- a rejection writes no pregame ledger or managed-bundle artifact.

## 경로와 Override

경로는 저장소 루트 기준이며 실제 출력에는 해석된 경로가 표시됩니다.

| 용도 | 기본값 | Override |
|---|---|---|
| 입력 번들 | 없음, 필수 | `--bundle <path>` |
| append-only ledger | `data/ledger/shadow.jsonl` | 모든 명령의 `--ledger <path>` |
| 관리 번들 루트 | `data/shadow/bundles` | 모든 명령의 `--bundle-root <dir>` |
| calibration 보고서 디렉터리 | `reports/private/shadow` | `review`의 `--report-dir <dir>` |

전체 override 형태는 다음과 같습니다.

```bash
.venv/bin/python -m analysis.operator projected --bundle <path> --ledger <path> --bundle-root <dir>
.venv/bin/python -m analysis.operator confirmed --bundle <path> --ledger <path> --bundle-root <dir>
.venv/bin/python -m analysis.operator review --bundle <path> --ledger <path> --bundle-root <dir> --report-dir <dir>
```

관리 번들은 `<bundle-root>/<game_pk>/<stage>-v<schema_version>-<canonical_sha256_앞16자리>.json`에 canonical bytes로 불변 저장됩니다. Fresh pregame 성공은 v2 관리 번들을 만들고, 역사적 pregame v1은 이미 존재하는 해당 경로에서만 인증합니다. 같은 내용은 재사용하고 같은 경로의 bytes가 다르면 I/O 오류로 중단합니다. Ledger가 유일한 authoritative state이며 관리 번들과 보고서는 파생 운영 산출물입니다.

## Pregame 성공과 ID 인계

`projected`와 `confirmed` 성공 출력은 다음 이름을 그대로 사용합니다.

- `capture_id`: 출처 상태와 audit provenance를 포함한 immutable capture 식별자
- `forecast_run_id`: 정규화된 forecast 의미와 고정 config로 결정되는 deterministic forecast 식별자
- `capture_forecast_link_record_id`: 위 capture와 forecast의 정확한 관계 record 식별자
- `managed_bundle_path`, `managed_bundle_hash`, `managed_bundle_replayed`
- `capture_replayed`, `forecast_replayed`, `link_replayed`

경기 후에는 반드시 `confirmed` 출력의 세 ID를 review 번들에 그대로 넣습니다. `projected` ID나 더 최신으로 보이는 다른 record를 대신 선택하지 않습니다.

```json
{
  "schema_version": 1,
  "stage": "review",
  "payload": {
    "capture_id": "<confirmed의 capture_id>",
    "forecast_run_id": "<confirmed의 forecast_run_id>",
    "capture_forecast_link_record_id": "<confirmed의 capture_forecast_link_record_id>",
    "official_result": {},
    "closing_observations": []
  }
}
```

Review는 이 세 ID가 같은 game의 active confirmed capture, 그 capture에 연결된 forecast와 exact link인지 검증합니다. Confirmed capture는 첫 투구 전에 동결되어 있어야 하며 review는 다른 capture를 추론하거나 forecast를 다시 실행하지 않습니다.

## Closing과 결과 정정

`closing_observations`는 선택 사항입니다. 각 항목은 `observation_kind: "close"`와 `source_name`, `source_record`, `source_revision`, `market_key`, `market_type`, `settlement_format`, `selected_side`, `line`, 양쪽 decimal price, `observed_at`, `available_at`, nullable `supersedes_record_id`를 완전하게 제공해야 합니다. 두 가격은 `1.0`보다 커야 하고 두 timestamp는 모두 동결 capture 시각 이상 첫 투구 시각 이하여야 합니다. Closing 가격은 entry 가격으로 사용하지 않습니다.

최초 closing quote의 `supersedes_record_id`는 `null`입니다. 정정 quote는 같은 logical stream의 현재 최신 observation record ID를 명시해야 하며 operator가 target을 추론하지 않습니다. Observation record ID는 authoritative ledger에서 해당 source/market stream의 최신 non-superseded record를 확인해 사용합니다.

공식 결과는 `source_name: "MLB"`여야 하고 `finalized_at`은 첫 투구보다 늦어야 합니다. 최초 결과의 `supersedes_record_id`는 `null`입니다. 결과 정정은 새 `source_revision`을 사용하고, 직전 성공 출력의 `official_result_record_id`를 새 review 번들의 `payload.official_result.supersedes_record_id`에 넣습니다. 점수와 선발 체인/불펜 실점 보존 규칙을 통과해야 하며 이전 결과 record는 수정되지 않습니다.

## Replay와 중단 복구

Operator capture key는 `(schema_version, stage, intrinsic CaptureCommand)`에서 자동 생성합니다. Intrinsic 내용에는 game, `captured_at`, first pitch, lineup state, eligibility, normalized input hash, canonical capture payload가 포함되고 `supersedes_capture_id`만 제외됩니다.

- Fresh schema-v2/current command의 같은 exact command는 기존 capture와 원래 supersession target을 먼저 찾고 capture, deterministic forecast, link를 replay하며, 관리 번들이 이미 저장되어 있으면 그 bytes도 replay합니다. 성공 또는 exact replay는 exit `0`입니다.
- Schema-v1 pregame 입력은 fresh 경로에 들어가지 않습니다. CLI가 관리 루트의 filename, stage, game, hash, bytes를 인증한 뒤 relaxed/strict v1 shape을 보존하고, ledger의 exact capture, forecast, link를 모두 찾은 경우에만 기존 records를 반환합니다. 하나라도 없으면 시뮬레이션, append, supersession, report, 관리 번들 저장 없이 거부합니다.
- `forecast_run_id`가 같은 상태에서 운영자 note, legacy candidate 또는 review binding만 정정되면 immutable forecast record는 바꾸지 않습니다. 대신 정정된 `shadow_comparison`을 담은 새 capture-forecast link가 직전 exact link를 append-only로 supersede합니다. 이후 exact replay와 review에는 이 최신 관계 record ID를 사용합니다.
- `captured_at`이나 audit provenance가 달라지면 forecast 의미가 같아도 새 capture가 생길 수 있고 같은 `forecast_run_id`를 재사용할 수 있습니다.
- 새 projected/confirmed 상태는 최신 active pregame capture를 새 record로 supersede합니다. Active confirmed 이후 새 projected 상태는 거부되지만 이미 기록된 exact projected command의 역사적 replay는 허용됩니다.
- Result idempotency는 game, 공식 source identity, source revision에서 도출합니다. Closing과 review도 exact source/revision 및 frozen record 관계로 재사용됩니다.
- 어떤 append 뒤 작업이 중단되어도 같은 번들을 같은 경로 override로 다시 실행하면 기존 record를 수정하지 않고 남은 단계를 완성합니다.

실패 후 stderr에 `Partial progress`가 있으면 `partial.capture_id`, `partial.forecast_run_id`, `partial.capture_forecast_link_record_id`, result/review/closing ID와 replay count 중 확인된 값만 authoritative ledger 재조회 결과로 표시됩니다.

- Pregame의 `partial.rerun_safe: true`는 I/O 오류 뒤 ledger 재조회가 성공했고 해당 capture가 durable하게 존재함을 뜻합니다.
- Review의 `partial.rerun_safe: true`는 I/O 오류 뒤에도 지정한 frozen capture가 최신 active capture임을 재확인했음을 뜻합니다.
- `partial.rerun_safe: true`이면 같은 command와 override를 그대로 재실행합니다. `false`는 rollback을 뜻하지 않으며 자동 재실행 안전성을 확인하지 못했다는 뜻이므로 ledger와 active lineage를 먼저 점검합니다.

## Checkpoint와 Forecast 압축 호환성

Pregame 명령은 ledger 옆의 `<ledger 파일명>.checkpoint-v1.json`을 재구축 가능한 조회 가속 상태로 사용합니다.

- JSONL ledger가 유일한 authoritative state입니다. Checkpoint는 ledger bytes를 수정하거나 대체하지 않으며 언제든 삭제할 수 있습니다. 없거나 손상됐거나 ledger의 device, inode, size, mtime, 전체 SHA-256 또는 row metadata와 맞지 않으면 strict JSONL 검증으로 다시 만듭니다.
- 첫 checkpoint 구축은 전체 historical ledger를 엄격하게 읽는 일회성 비용입니다. 이후 warm load는 전체 fingerprint와 각 canonical JSON row의 envelope 및 관계 metadata를 검증하지만 historical forecast domain object 전체를 재구성하지 않습니다.
- 한 pregame 명령 안에서는 immutable checkpoint가 full ledger fingerprint와 정확히 같을 때만 다음 lock-scoped session에서 재사용됩니다. Ledger가 바뀌면 재사용하지 않고 authoritative state를 다시 검증합니다.
- 구형 forecast record의 평문 `values` 배열은 계속 읽을 수 있습니다. 신규 forecast는 각 component의 전체 20,000개 값을 canonical JSON, zlib level 6, base64 형식으로 결정론적·무손실 압축합니다. Decode 후 값과 정수/실수 타입은 구형 표현과 같습니다.
- Review와 calibration은 압축 여부와 무관하게 같은 decoded forecast context와 report mapping을 사용합니다. 압축은 확률, interval, classification, migration count 또는 보고서 값을 바꾸지 않습니다.
- 완전한 exact replay는 frozen capture, forecast output, 최신 exact link의 legacy snapshot, market context를 모두 검증한 뒤에만 20,000회 시뮬레이션을 생략합니다. 저장된 네 score path에서 각 시장의 win/push/loss 확률과 가격을 같은 settlement 로직으로 다시 계산해 frozen market output과 정확히 일치하는지도 확인합니다. 하나라도 없거나 다르거나 해석할 수 없으면 기존 전체 forecast 경로로 보수적으로 돌아갑니다.

Checkpoint를 수동으로 지우는 것은 안전하지만 다음 pregame 명령에서 strict rebuild 비용이 다시 발생합니다. JSONL ledger나 기존 forecast row를 checkpoint 복구 목적으로 수정하거나 재작성하지 않습니다.

## Exit Code

| Code | 의미 |
|---:|---|
| `0` | 성공 또는 exact replay |
| `1` | 예상하지 못한 내부 오류 |
| `2` | Typer 사용법, 필수 option, 알 수 없는 option 오류 |
| `3` | 잘못된 JSON, 미지원 schema, 엄격한 field/domain 관계 검증 실패 |
| `4` | stage, lineup, timestamp, source, frozen-record precondition 실패 |
| `5` | ledger reference, idempotency conflict, corruption 실패 |
| `6` | bundle, ledger, report I/O 실패 |

Expected operator 오류는 traceback 없이 간결한 원인과 가능한 partial progress를 출력합니다.

## Calibration 보고서

`review`는 전체 ledger에서 leakage-safe primary row를 다시 선택하고 동일 mapping으로 JSON과 한국어 Markdown을 만듭니다. 각 ledger revision은 다음 불변 generation에 저장됩니다.

```text
reports/private/shadow/generations/
  generation-v1-<ledger_path_hash16>-<record_count_20자리>-<ledger_state_sha256>/
    calibration-latest.json
    calibration-latest.md
```

`reports/private/shadow/current`는 현재 generation을 가리키는 atomic symbolic link입니다. 호환 경로 `reports/private/shadow/calibration-latest.json`과 `reports/private/shadow/calibration-latest.md`는 각각 `current` 아래 파일을 가리킵니다. `--report-dir`을 쓰면 같은 구조가 지정한 디렉터리 아래에 생성됩니다. Ledger revision이 같으면 기존 generation bytes를 검증하고 `calibration_report_replayed: true`를 반환합니다. 이전 generation은 덮어쓰지 않습니다.

성공 출력과 보고서의 수치는 다음처럼 구분합니다.

- `calibration_sample_size`: 전체 ledger에서 선택된 leakage-safe primary calibration row 수
- `confirmed_game_count`: review가 존재하고 exact frozen comparison이 review-eligible인 confirmed primary row의 고유 `game_pk` 수, 기준 `100`
- `confirmed_decision_count`: 아래 시장군별 confirmed decision count의 합계
- `confirmed_decision_count.<market_type>`: confirmed eligible row의 고유 `(forecast_run_id, market_key)` 수, 시장군별 기준 `300`
- Legacy comparison이 관계 link에서 정정된 경우 review와 migration eligibility는 forecast에 내장된 과거 값이 아니라 최신 active exact link를 사용합니다.
- Review-eligible legacy `pass`는 migration count에 포함되지만 review-ineligible market은 제외됩니다.
- 표본이나 필요한 가격이 없으면 JSON metric은 `null`, Markdown은 `사용 불가 (unavailable)`로 표시합니다. 이를 `0`으로 해석하지 않습니다.

JSON에는 resolved `ledger_path`, ledger 최대 append 시각인 `ledger_as_of`, record count/state hash, overall/group metrics와 migration progress가 들어갑니다. Wall-clock 생성 시각은 넣지 않으므로 ledger가 같으면 결과 bytes도 같습니다.

## 안전 경계

- CLI는 MLB, RotoWire, Savant/Statcast, market site를 live fetch하지 않습니다.
- CLI와 보고서는 자동 model tuning, online learning, production promotion을 수행하지 않습니다.
- 기존 legacy authority, `play`/`lean`/`pass`, `0-100` rank score, same-game assembler, 최종 priority와 model threshold를 변경하지 않습니다.
- 고정 model version `2026-08-05`은 pregame schema v2 및 review schema v1과 별도로 버전 관리됩니다.
- Forecast는 반드시 첫 투구 전에 `confirmed` 단계에서 동결합니다. `review`는 공식 결과를 받은 뒤에도 forecast를 다시 만들거나 바꾸지 않습니다.
