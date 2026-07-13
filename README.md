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

## Workflow

1. 공식 MLB 선발과 오프너/벌크 구조를 확인합니다.
2. RotoWire 확정 라인업을 사용하고, 미확정이면 예상 라인업을 사용합니다.
3. Baseball Savant/Statcast로 양 팀 모든 타자의 구종+구속 대응력을 예상 타석과 투수 좌우 기준으로 평가합니다.
4. 최근 타선 흐름, 선발 안정성, 불펜 가용성과 품질, 심판을 결합합니다.
5. F5, F5 O/U, ML, `+1.5`, O/U를 서로 다른 가중치로 판단하고 시장 가격을 마지막에 적용합니다. 시장 엔진의 `pass`를 제외한 뒤 동일경기 후보의 증거 경로 중복을 제거합니다.
6. 경기 후에는 `docs/reviews/README.md` 형식으로 복기합니다.

## Documentation

- 분석 출력 형식: `docs/analysis-output-template.md`
- 새 채팅용 프롬프트: `docs/new-chat-prompt.md`
- 데이터 출처와 권한: `docs/data-sources.md`
- 복기 템플릿: `docs/reviews/README.md`

임시 HTML, Statcast 캐시, 로그인 정보, 개인 베팅 기록은 Git에 포함하지 않습니다.
