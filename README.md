# MLB Analytical App

채팅과 독립적으로 재사용할 수 있도록 고정한 MLB 경기 분석 모델입니다. 선발, 전체 라인업의 구종+구속 대응력, 최근 타격 흐름, 불펜, 심판, 시장 가격을 분리해서 평가합니다.

## Authoritative Files

- `AGENTS.md`: 이 저장소를 여는 미래 Codex 세션의 기본 지침
- `docs/model.md`: 상세 분석 모델의 단일 기준 문서
- `analysis/model.py`: 현재 점수화 로직
- `config/model.json`: 기계가 읽는 고정 설정
- `analysis/pipeline.py`: 분석 단계와 설정을 검증하는 도구

모델은 사용자가 명시적으로 수정 요청을 하지 않는 한 변경하지 않습니다.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m analysis.pipeline --lineup projected --starter confirmed
python3 -m analysis.pipeline --lineup confirmed --starter confirmed --weather
```

`analysis.pipeline`은 라이브 데이터를 스크래핑하지 않습니다. 분석에 필요한 단계, 출처 상태, 신뢰도 플래그가 고정 규칙과 맞는지 확인하는 재현성 도구입니다.

## Daily Workflow

이 전달에서 라이브 출처 수집과 정규화는 운영자 또는 외부 수집 절차가 담당합니다. 자동 수집을 가정하지 않으며, 예측에 쓰는 모든 정규화 값은 원본 출처 관측치로 추적할 수 있어야 합니다.

1. **예상 라인업:** 공식 MLB 선발과 오프너/벌크 구조, RotoWire 예상 라인업, Savant/Statcast 매치업, 최근 흐름, 불펜, 심판, 현재 소수 배당을 사용해 기존 고정 모델 분석을 수행합니다. F5, F5 O/U, ML, `+1.5`, O/U의 `play`/`lean`/`pass`와 최종 우선순위를 정한 뒤 같은 상태를 shadow capture와 forecast로 기록합니다. 예상 라인업은 팀 강도가 아니라 신뢰도만 낮춥니다.
2. **확정 라인업:** RotoWire 확정 라인업과 최신 선발 및 시장 상태로 새로고침합니다. 실제 출처 상태 변화는 새 capture로 남기고, 정규화된 야구 입력이나 의미 있는 시장 조건이 바뀐 경우에만 새 deterministic forecast run을 만듭니다. 비의미적 시각 메타데이터만 달라진 capture는 같은 forecast에 연결할 수 있습니다. 정확한 재시도가 기존 capture를 반환하려면 호출자가 같은 idempotency key를 재사용해야 합니다. 같은 상태라도 새 key를 쓰면 capture가 하나 더 추가될 수 있지만 deterministic forecast는 재사용됩니다.
3. **경기 후:** 공식 MLB 결과를 불변 기록으로 수집하고, 동결된 pregame capture/forecast에 연결해 객관적 A/B/C/D 흐름 복기를 수행한 뒤 calibration report를 갱신합니다.

Shadow forecasting은 필수 비교 모드이며 기존 추천의 방향, `play`/`lean`/`pass`, `0-100` 순위 점수, 최종 우선순위를 변경하거나 재정렬하거나 상향/하향할 수 없습니다. `0-100`은 확률이 아닌 legacy rank score입니다. 최소 100개의 확정 라인업 경기 전에는 production 의사결정에 영향을 줄 수 없고, 가능하면 시장군별 300개 decision 전에는 튜닝하지 않습니다. Production 우선권, 규칙 삭제, 가중치 변경은 별도 승인된 새 모델 버전과 설계가 필요합니다.

## Local Shadow Ledger

- 운영자는 설정 가능한 로컬 디렉터리를 선택하고 그 안의 JSONL 파일 경로를 `LedgerStore`에 전달합니다. 예: `data/ledger/shadow.jsonl`. `data/ledger/`는 Git에서 제외되며 커밋하지 않는 private local runtime state입니다.
- Ledger는 append-only JSONL이 유일한 authoritative state입니다. 각 record envelope에 `record_kind`, idempotency key, canonical payload hash가 영구 기록되고, `LedgerStore`는 로그를 읽어 durable idempotency index를 재구성합니다.
- 일반적인 shadow 기록의 public application API는 `analysis.shadow.record_shadow_run`입니다. 이 API는 capture, deterministic forecast, capture-to-forecast link를 append하거나 replay하며 별도 CLI를 제공하지 않습니다.
- 같은 record kind와 같은 caller-supplied idempotency key에 같은 canonical payload를 다시 보내면 기존 record를 반환합니다. 같은 key에 바뀐 payload를 보내면 idempotency conflict입니다. 같은 semantic state라도 새 key를 사용하면 새 capture가 append될 수 있고 deterministic forecast는 재사용될 수 있습니다. 실제 상태 변화에는 새 key를 사용하며, 이전 capture를 대체하는 경우 supersession 관계와 새 capture-to-forecast link를 함께 기록합니다.
- `record_shadow_run`은 capture supersession 인자를 받지 않습니다. 이전 capture를 명시적으로 대체해야 하면 lower-level `LedgerStore.append_capture`에 `CaptureCommand.supersedes_capture_id`를 설정하고 forecast/link API로 lineage를 완성합니다. 기존 record를 수정하지 않습니다.
- 공식 결과 정정은 새 `source_revision`과 최신 결과를 가리키는 `supersedes_record_id`로 새 immutable result를 append합니다. 이전 결과는 변경하지 않으며 review와 calibration은 최신 non-superseded official result만 사용합니다.
- 공식 결과가 선발 체인 실점과 불펜 실점을 모두 제공하면 각 투수진의 두 값은 음이 아닌 정수여야 하며 상대 최종 득점과 정확히 일치해야 합니다. 보존되지 않는 구성요소는 review 또는 calibration에 들어가기 전에 거부됩니다.

## Documentation

- 분석 출력 형식: `docs/analysis-output-template.md`
- 새 채팅용 프롬프트: `docs/new-chat-prompt.md`
- 데이터 출처와 권한: `docs/data-sources.md`
- 복기 템플릿: `docs/reviews/README.md`

임시 HTML, Statcast 캐시, 로그인 정보, 개인 베팅 기록은 Git에 포함하지 않습니다.
