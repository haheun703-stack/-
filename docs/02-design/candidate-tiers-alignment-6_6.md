# FLOWX Market OS v1 — 4단계 candidate_tiers 정합 검증 (2026-06-06)

선행: 3단계 `morning_plan_07` (커밋 `d903980`)
판정: **PASS — plan ↔ ledger 등급 정합 확인. 새 코드/스키마 없음(검증 단계).**

## 목표

`morning_plan_07`(계획서)의 CORE/WATCH/CONTROL 등급과 `paper_smart_entry`/`paper_ledger`
(장부)의 등급이 **같은 기준(`classify_tier` SSOT)**으로 매겨지는지 검증한다.
지시서 핵심: 새 분류기 발명 금지, "계획서 등급 == 장부 등급"만 확인.

## 결론: 불일치 여지 로직상 0

plan과 ledger **둘 다** `_latest_candidate_logs` + 같은 candidate dict의
floor/market/supply를 **같은 `classify_tier`**에 넣는다. 순수 결정론 함수 →
같은 입력 같은 출력 → 등급이 갈릴 수 없다.

| 항목 | 상태 | 근거 (파일:라인) |
|---|---|---|
| `classify_tier` 단일 정의(SSOT) | ✅ | `scripts/paper_smart_entry.py:33-53` 유일. `scripts/archive/`·`_etf_ref/` 제외 복붙 0 |
| plan 경로 | ✅ | `morning_plan_07._load_candidates`(`src/use_cases/morning_plan_07.py:235`) → `build_picks_from_candidate_log`(`paper_smart_entry.py:65`, line71 `_latest_candidate_logs`) → `classify_tier`(line78) |
| ledger CORE/WATCH | ✅ | `record_entries`(`paper_smart_entry.py:142`) → `cand_map=_candidate_feature_map`(line155, line134 `_latest_candidate_logs`) → `tier=classify_tier(c)`(line181) → `shadow_observations[].tier` |
| ledger CONTROL | ✅ | `record_control_pool`(line105) → `shadow_control[].tier="CONTROL"` |

## 검증 테스트 (`tests/test_candidate_tiers_alignment.py`)

같은 candidate_log(CORE/WATCH/CONTROL 3등급)를 tmp 파일로 격리
(`monkeypatch` `pse.LEDGER` → 실제 `data/paper_ledger.json` **미접촉**)하고:

- **plan 측**: `build_picks_from_candidate_log()` → `{ticker: _tier}`
- **ledger 측**: `record_entries`(SHADOW) + `record_control_pool` → `{ticker: tier}`
- **단언**: `plan_tier == ledger_tier` (전 ticker 일치)

케이스: ①3등급 완전 일치 ②CORE↔WATCH 경계(단독수급 유무)가 양 경로 동일 ③후보 0건
④SHADOW_OPEN vs PAPER_OPEN은 status/key만 다르고 tier 동일(5단계 경계 못박기).

## Explore 오류 정정

초기 조사가 "candidate_log에 floor_quality 없어 전부 CONTROL"이라 봤으나 **오류**.
Grep으로 feature 328회 등장 확인 + 3단계 CLI 스모크 CORE1/WATCH2/CONTROL9가 반증.
candidate_log 연동은 정상.

## 범위 밖 (지시서 "새 기준 만들지 마라")

- candidate_log 스키마 보강 / `paper_trades` tier 백필 (초기 조사 제안) → **4단계 아님**. 필요 시 별도 P1.
- `paper_trades`에 tier 미존재는 결함 아님 — PAPER_OPEN 실행 이력이 없을 뿐,
  `record_entries:181`이 박게 돼 있음(6/8 관찰 시 자동 기록).
- 새 점수체계/분류 로직 일절 추가 없음.

## 안전선 (불변)

실주문 0 / scheduler 변경 0 / SAJANG 변경 0 / PAPER_OPEN 기본 금지 / 매도 자동화 BLOCKED.
candidate_tiers는 검증 코드일 뿐 엔진 실행권 없음. 테스트는 monkeypatch로 실제 ledger 미접촉.

## 다음

5단계 `smart_entry_adapter` — "관찰 후보"를 SmartEntry 실행 인터페이스 근처로 연결하는 구간.
**SHADOW_OPEN(기본) ↔ PAPER_OPEN(`--paper-open`, 6/8 금지) 경계를 절대 혼동 금지.**
