# New Chat Prompt

Use this prompt if a future environment does not automatically load the repository's `AGENTS.md`.

```text
이 저장소의 MLB 분석 모델을 사용해줘.

먼저 AGENTS.md와 docs/model.md를 읽고, 사용자가 명시적으로 모델 수정을 요청하지 않는 한 규칙을 절대 수정하지 마.

라인업은 RotoWire 확정 라인업을 우선하고, 미확정이면 RotoWire 예상 라인업을 사용해. 공식 MLB 선발과 오프너/벌크 구조를 확인하고, Baseball Savant/Statcast로 양 팀 모든 타자의 구종+구속 대응력을 투수 좌우 및 예상 타석 기준으로 분석해.

기본 출력은 F5 / ML / +1.5 / O/U이며, 소수배당을 사용해. 불펜 연투·휴식·필승조 가용성과 구간별 품질, 최근 타격 흐름, 선발 안정성, 심판을 포함해. 날씨는 내가 제공하거나 요청할 때만 반영해.

애매한 추측은 실제 데이터처럼 말하지 말고, 표본 부족은 타자 약점으로 처리하지 마.
```

