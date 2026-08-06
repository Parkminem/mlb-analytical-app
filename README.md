# MLB Analytical App

채팅과 독립적으로 재사용할 수 있도록 고정한 MLB 경기 분석 모델입니다. 선발, 전체 라인업의 구종+구속 대응력, 최근 타격 흐름, 불펜, 심판, 시장 가격을 분리해서 평가합니다.

## Authoritative Files

- `AGENTS.md`: 이 저장소를 여는 미래 Codex 세션의 기본 지침
- `docs/model.md`: 상세 분석 모델의 단일 기준 문서
- `analysis/model.py`: 현재 점수화 로직
- `config/model.json`: 기계가 읽는 고정 설정
- `analysis/pipeline.py`: 분석 단계와 설정을 검증하는 도구
- `docs/shadow-operator.md`: 3단계 Shadow Operator CLI 운영 가이드

모델은 사용자가 명시적으로 수정 요청을 하지 않는 한 변경하지 않습니다.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m analysis.pipeline --lineup projected --starter confirmed
python3 -m analysis.pipeline --lineup confirmed --starter confirmed --weather
.venv/bin/python -m analysis.operator projected --bundle <path>
.venv/bin/python -m analysis.operator confirmed --bundle <path>
.venv/bin/python -m analysis.operator batch-projected --manifest <path> --receipt <path>
.venv/bin/python -m analysis.operator batch-confirmed --manifest <path> --receipt <path>
.venv/bin/python -m analysis.operator review --bundle <path>
```

`analysis.pipeline`은 라이브 데이터를 스크래핑하지 않습니다. 분석에 필요한 단계, 출처 상태, 신뢰도 플래그가 고정 규칙과 맞는지 확인하는 재현성 도구입니다.
`analysis.operator`도 라이브 데이터를 수집하거나 정규화하지 않습니다. 외부에서 준비한 엄격한 버전 번들을 받아 append-only 원장과 불변 운영 산출물을 기록합니다.

Fresh `projected`/`confirmed` 번들은 JSON integer `schema_version: 2`와 명시적인 `authority: "current"`를 사용합니다. `review` 번들은 `schema_version: 1`을 유지합니다.

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "confirmed"
}
```

## Daily Workflow

이 전달에서 라이브 출처 수집과 정규화는 운영자 또는 외부 수집 절차가 담당합니다. 자동 수집을 가정하지 않으며, 예측에 쓰는 모든 정규화 값은 원본 출처 관측치로 추적할 수 있어야 합니다.

1. **예상 라인업:** 공식 MLB 선발과 오프너/벌크 구조, RotoWire 예상 라인업, Savant/Statcast 매치업, 최근 흐름, 불펜, 심판, 현재 소수 배당을 사용해 기존 고정 모델 분석을 수행합니다. F5, F5 O/U, ML, `+1.5`, O/U의 `play`/`lean`/`pass`와 최종 우선순위를 정한 뒤 schema-v2/current `projected` 번들을 만들어 `.venv/bin/python -m analysis.operator projected --bundle <path>`로 같은 상태를 기록합니다. 예상 라인업은 팀 강도가 아니라 신뢰도만 낮춥니다.
2. **확정 라인업:** RotoWire 확정 라인업과 최신 선발 및 시장 상태로 schema-v2/current `confirmed` 번들을 새로 만들고 `.venv/bin/python -m analysis.operator confirmed --bundle <path>`를 실행합니다. CLI는 번들의 intrinsic capture 내용에서 idempotency를 도출하고 최신 active pregame capture를 immutable supersession lineage로 연결합니다. 의미가 같은 forecast는 재사용될 수 있으며 정확한 재실행은 원래 capture, forecast, link를 반환합니다.
3. **경기 후:** `confirmed` 성공 출력의 정확한 `capture_id`, `forecast_run_id`, `capture_forecast_link_record_id`와 공식 MLB 결과로 schema-v1 `review` 번들을 만들고 `.venv/bin/python -m analysis.operator review --bundle <path>`를 실행합니다. CLI는 동결된 pregame 상태만 복기하고 calibration report를 갱신하며 새 forecast를 만들지 않습니다.

여러 경기를 처리할 때는 각 단계의 schema-v2/current 번들을 먼저 모두 준비한 뒤 `batch-projected` 또는 `batch-confirmed`를 사용합니다. 배치 명령은 기본 세 worker process로 기존 단일 경기 Operator를 제한 병렬 실행하고, 마지막에 원장의 capture/forecast/link 세 ID를 한 번 감사해 atomic JSON receipt를 만듭니다. 응답에서 `동결 완료`라고 표시하려면 receipt의 `complete`가 `true`이고 `success_count == target_count`여야 합니다. 일부 경기만 성공하면 immutable 성공 기록은 유지되지만 명령은 exit `7`로 끝나며 같은 manifest를 재실행해 안전하게 이어갑니다.

모든 pregame schema-v1 shape은 relaxed/strict candidate 형식과 관계없이 역사적 exact replay 전용입니다. 설정된 관리 번들 루트의 인증된 파일과 ledger의 정확한 capture, forecast, link가 모두 있을 때만 replay할 수 있습니다. 아직 실행하지 않은 pending v1 파일은 fresh 상태를 만들 수 없으므로 schema v2/current로 다시 생성해야 합니다. V1 거부는 ledger나 관리 번들 bytes를 변경하지 않습니다.

번들 역할, 기본 경로, 재실행과 복구, 결과 정정, 보고서 해석은 `docs/shadow-operator.md`를 따릅니다.

Shadow forecasting은 필수 비교 모드이며 기존 추천의 방향, `play`/`lean`/`pass`, `0-100` 순위 점수, 최종 우선순위를 변경하거나 재정렬하거나 상향/하향할 수 없습니다. `0-100`은 확률이 아닌 legacy rank score입니다. 최소 100개의 확정 라인업 경기 전에는 production 의사결정에 영향을 줄 수 없고, 가능하면 시장군별 300개 decision 전에는 튜닝하지 않습니다. Production 우선권, 규칙 삭제, 가중치 변경은 별도 승인된 새 모델 버전과 설계가 필요합니다.

고정 MLB model version은 `2026-08-05`이며 bundle schema version과 독립적입니다. Pregame schema v2 전환은 MLB scoring, F5 side/F5 O/U, full-game market engine, same-game assembler, legacy decision authority 또는 Shadow 권한을 변경하지 않습니다.

## Local Shadow Ledger

- 운영자는 설정 가능한 로컬 디렉터리를 선택하고 그 안의 JSONL 파일 경로를 `LedgerStore`에 전달합니다. 예: `data/ledger/shadow.jsonl`. `data/ledger/`는 Git에서 제외되며 커밋하지 않는 private local runtime state입니다.
- Ledger는 append-only JSONL이 유일한 authoritative state입니다. 각 record envelope에 `record_kind`, idempotency key, canonical payload hash가 영구 기록되고, `LedgerStore`는 로그를 읽어 durable idempotency index를 재구성합니다.
- 일반 운영 진입점은 `python -m analysis.operator` CLI입니다. `projected`, `confirmed`, `review` 명령이 capture, deterministic forecast, exact link, 결과, review, calibration 산출물을 단계별로 append하거나 replay합니다.
- Operator는 schema version, stage, supersession target을 제외한 intrinsic `CaptureCommand`에서 capture idempotency key를 도출합니다. 새 상태는 최신 active capture를 명시적으로 supersede하고, 정확한 재실행은 원래 target과 record를 재사용하므로 lineage가 불변입니다.
- Lower-level `analysis.shadow.record_shadow_run`은 여전히 caller-supplied capture key로 append/replay하며, 선택적 `supersedes_capture_id`와 active-head guard를 받습니다. 이 API와 `LedgerStore`의 동작은 append-only입니다. supersession은 이전 record를 수정하거나 삭제하지 않고 새 record로 관계만 추가합니다.
- 공식 결과 정정은 새 `source_revision`과 최신 결과를 가리키는 `supersedes_record_id`로 새 immutable result를 append합니다. 이전 결과는 변경하지 않으며 review와 calibration은 최신 non-superseded official result만 사용합니다.
- 공식 결과가 선발 체인 실점과 불펜 실점을 모두 제공하면 각 투수진의 두 값은 음이 아닌 정수여야 하며 상대 최종 득점과 정확히 일치해야 합니다. 보존되지 않는 구성요소는 review 또는 calibration에 들어가기 전에 거부됩니다.
- Pregame 조회는 ledger 옆의 rebuildable `checkpoint-v1` 파생 파일을 사용합니다. JSONL만 권위가 있으며 checkpoint는 삭제하거나 손상돼도 strict ledger scan으로 복구됩니다.
- Historical 평문 forecast 배열은 계속 읽히고, 신규 20,000개 component 배열은 canonical JSON 기반으로 결정론적·무손실 압축됩니다. Decode 결과, review, calibration report mapping은 동일합니다.
- Fresh schema-v2/current의 완전한 exact projected/confirmed replay만 일반 실행에서 시뮬레이션을 건너뜁니다. Historical schema-v1 pregame은 인증된 관리 번들과 exact ledger triplet을 조회해 기존 records만 반환합니다. 동일 forecast의 legacy note/candidate/review binding 정정은 forecast를 바꾸지 않고 최신 exact link에 append-only로 기록합니다. 입력, config, 최신 link의 legacy snapshot, 저장된 score path에서 재계산한 market output 또는 frozen output 검증이 하나라도 다르면 v2 일반 실행은 기존 20,000회 forecast를 그대로 실행합니다.

## Documentation

- 분석 출력 형식: `docs/analysis-output-template.md`
- 새 채팅용 프롬프트: `docs/new-chat-prompt.md`
- 데이터 출처와 권한: `docs/data-sources.md`
- 복기 템플릿: `docs/reviews/README.md`
- Shadow Operator CLI: `docs/shadow-operator.md`

임시 HTML, Statcast 캐시, 로그인 정보, 개인 베팅 기록은 Git에 포함하지 않습니다.
