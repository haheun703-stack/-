# FLOWX Market OS v1 3단계 지시서 — morning_plan_07

작성일: 2026-06-06  
대상: 퀀트봇  
선행 완료: `29b4e0d` 라우터 히스테리시스, `6cddef9` engine_policy_map

## 0. 2단계 검증 결과

2단계 `engine_policy_map`는 진행 가능 상태다.

검증한 내용:

- `regime_router_v1`에 C60 2거래일 히스테리시스가 실제 반영됨.
- `engine_policy_map`는 라우터의 `effective_regime`을 받아 정책을 만든다.
- R1이면 가설C 차단, SmartEntry/B·C/Event는 shadow, 매도는 BLOCKED.
- R4이면 가설C는 `PAPER_ONLY`, SmartEntry는 `ALLOWED_SHADOW`, B·C/Event는 `SHADOW_ONLY`, 매도는 BLOCKED.
- `PAPER_ONLY`는 `PAPER_OPEN` 허용이 아니다.
- `paper_open_allowed=False` 유지.
- R0/R5는 `shadow_advisories`, 엔진 전환권 0.
- 데이터 없음이면 `DATA_UNAVAILABLE`로 보수적 차단.

재현 검증:

- `tests/test_engine_policy_map.py`
- `tests/test_regime_router_v1.py`
- `tests/test_regime_monitor.py`
- 결과: 20개 통과

로컬 실데이터 관찰:

- 로컬 네트워크 제한으로 488080이 `DATA_UNAVAILABLE` 처리됨.
- 삼성전자·SK하이닉스는 R4였으나, 488080 데이터 없음 때문에 시장 종합 정책은 `DATA_UNAVAILABLE`.
- 이 동작은 정상이다. 데이터가 없으면 신규진입 차단이 맞다.

## 1. 3단계 목표

`morning_plan_07`은 아침 7시에 사람이 보는 당일 작전계획을 만든다.

핵심 역할:

1. `regime_router_v1` 결과 확인
2. `engine_policy_map` 결과 확인
3. 최신 `candidate_log` 확인
4. CORE/WATCH/CONTROL 후보 수와 후보 목록 정리
5. 오늘 허용 엔진과 차단 엔진을 사람이 바로 읽을 수 있게 표시
6. 실주문 0, 매도 BLOCKED, `PAPER_OPEN` 금지를 명시

이 단계는 **계획 생성만** 한다. 스캐너 실행, SmartEntry 실행, 주문, 스케줄러 등록은 하지 않는다.

## 2. 구현 위치

권장 파일:

- `src/use_cases/morning_plan_07.py`
- `scripts/morning_plan_07.py`
- `tests/test_morning_plan_07.py`

산출물:

- `data_store/plans/plan_YYYY-MM-DD.json`
- `data_store/plans/plan_YYYY-MM-DD.md`

주의:

- `data_store/` 산출물은 커밋하지 않는다.
- 코드는 커밋한다.
- 산출물은 실행 검증용으로만 둔다.

## 3. 입력

### 3.1 정책 입력

1순위:

- `engine_policy_map.run_policy_map(write=False)` 직접 호출

허용:

- 이미 생성된 `data_store/policies/policy_YYYY-MM-DD.json` 읽기

주의:

- 3단계에서 정책을 임의로 다시 해석하지 않는다.
- 엔진 허용/차단 판단은 2단계 정책표가 단일진실이다.

### 3.2 후보 입력

입력 파일:

- `data/paper_ledger.json`

사용 섹션:

- 최신 `candidate_log`

실제 구조:

- `as_of_date`
- `source`
- `source_label`
- `total`
- `enter_count`
- `avoid_count`
- `candidates`

후보 필드:

- `ticker`
- `name`
- `decision`
- `reason`
- `date`
- `weekly_gate`
- `daily_setup`
- `floor_quality`
- `market_context`
- `supply_confirmation`
- `risk_reward`

주의:

- 최신 `candidate_log.as_of_date`만 사용한다.
- stale 후보는 계획서에서 제외한다.
- `decision="진입"`만 CORE/WATCH/CONTROL 분류 대상으로 삼는다.
- `decision="회피"`는 회피 사유 요약에만 사용한다.

### 3.3 CORE/WATCH/CONTROL 분류

가능하면 기존 `scripts/paper_smart_entry.py`의 순수 분류 로직을 재사용한다.

재사용 가능 함수:

- `_latest_candidate_logs`
- `classify_tier`
- `build_picks_from_candidate_log`

주의:

- `record_control_pool` 호출 금지. 이 함수는 ledger에 기록한다.
- `paper_smart_entry.main()` 호출 금지. 이 함수는 picks 파일을 쓰고 SmartEntry 경로로 이어질 수 있다.
- 3단계는 계획 생성만 해야 한다.

## 4. 정책 적용 규칙

### DATA_UNAVAILABLE

- 신규진입 금지
- 가설C 차단
- SmartEntry `SHADOW_ONLY`
- B/C 로테이션 `SHADOW_ONLY`
- Event/DART `SHADOW_ONLY`
- 매도 `BLOCKED`
- 계획서 상단에 데이터 없음 경고 표시

### R1 약세/위험

- 신규진입 금지
- 가설C 차단
- SmartEntry `SHADOW_ONLY`
- 후보가 있더라도 "관찰 후보"로만 표시
- 매도 자동화 금지, 신호만 허용

### R4 정상/강세

- 가설C 후보선정 `PAPER_ONLY`
- SmartEntry `ALLOWED_SHADOW`
- CORE/WATCH 후보를 장중 관찰 대상으로 표시
- CONTROL은 진입 대상이 아니라 missed-winner 비교군으로 표시
- `PAPER_OPEN`은 여전히 금지

### R0/R5

- 표시만 한다.
- 엔진 허용/차단 권한 없음.
- R5 과열은 "주의 라벨"로만 표시한다.

## 5. JSON 산출물 필수 스키마

`plan_YYYY-MM-DD.json`에는 최소 아래 필드를 포함한다.

```json
{
  "version": "morning_plan_07_v1",
  "generated_at": "...",
  "as_of_date": "YYYY-MM-DD",
  "policy_source": "engine_policy_map_v1",
  "market_regime": "...",
  "engines": {},
  "paper_open_allowed": false,
  "sell_automation": "BLOCKED",
  "candidate_log": {
    "as_of_date": "...",
    "source": "...",
    "total": 0,
    "enter_count": 0,
    "avoid_count": 0
  },
  "tiers": {
    "CORE": [],
    "WATCH": [],
    "CONTROL": []
  },
  "blocked_or_shadow_reason": [],
  "data_warnings": [],
  "safety": {
    "real_order": false,
    "scheduler_changed": false,
    "sajang_changed": false,
    "paper_open_default": false,
    "sell_automation": "BLOCKED"
  }
}
```

## 6. Markdown 산출물 필수 내용

`plan_YYYY-MM-DD.md`는 사람이 바로 읽는 문서다.

필수 섹션:

1. 오늘 한 줄 결론
2. 기준일
3. 시장 종합 정책
4. 기초자산별 C60 상태
5. 허용 엔진
6. 차단 엔진
7. CORE 후보
8. WATCH 후보
9. CONTROL 비교군
10. 데이터 경고
11. 안전선

문장 원칙:

- "매수"라는 표현은 쓰지 않는다.
- "관찰", "후보", "차단", "비교군"으로 표현한다.
- `PAPER_OPEN`을 열었다고 표현하지 않는다.
- "실주문 0 / 매도 BLOCKED / scheduler 변경 0 / SAJANG 변경 0"을 하단에 반복한다.

## 7. 완료 조건

테스트 필수:

1. DATA_UNAVAILABLE이면 계획서가 생성되고 신규진입이 차단된다.
2. R1이면 CORE/WATCH가 있어도 SmartEntry가 `SHADOW_ONLY`로 표시된다.
3. R4이면 CORE/WATCH가 관찰 후보로 표시된다.
4. CONTROL은 진입 후보가 아니라 비교군으로 표시된다.
5. 후보 0건이어도 JSON/MD가 생성된다.
6. `PAPER_OPEN`이 기본 금지로 표시된다.
7. 매도는 `BLOCKED`로 표시된다.
8. 주문 관련 문자열과 주문 어댑터 import가 없다.

검증 명령:

```powershell
.\venv\Scripts\python.exe -u -X utf8 -m py_compile src\use_cases\morning_plan_07.py scripts\morning_plan_07.py tests\test_morning_plan_07.py
.\venv\Scripts\python.exe -u -X utf8 -m pytest tests/test_morning_plan_07.py tests/test_engine_policy_map.py tests/test_regime_router_v1.py -q
.\venv\Scripts\python.exe -u -X utf8 scripts\morning_plan_07.py --no-remote --no-write
```

## 8. 금지

- 실주문 금지
- KIS 주문 어댑터 import 금지
- SmartEntry 실행 금지
- `paper_smart_entry.main()` 호출 금지
- `record_control_pool()` 호출 금지
- scheduler 등록 금지
- systemctl 금지
- SAJANG 변경 금지
- VPS 배포 금지
- `data_store/` 산출물 커밋 금지

## 9. 커밋 기준

3단계 구현 커밋은 아래 파일만 포함하는 것을 원칙으로 한다.

- `src/use_cases/morning_plan_07.py`
- `scripts/morning_plan_07.py`
- `tests/test_morning_plan_07.py`

문서 보강이 필요하면 별도 커밋으로 분리한다.

커밋 전 확인:

- `git diff --cached --stat`
- `git status --short`
- `data_store/` 산출물 미포함
- 미추적 `hybrid_design_6_1.bundle`, `quant_bot (1).py` 미접촉

## 10. 다음 단계 연결

3단계 완료 후 다음은 4단계 `candidate_tiers plan 연동`이다.

단, 3단계 구현 중 CORE/WATCH/CONTROL 수가 plan과 ledger에서 이미 일치한다면 4단계는 짧은 정합 검증으로 축소할 수 있다.

