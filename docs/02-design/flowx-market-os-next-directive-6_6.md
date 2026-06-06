# FLOWX Market OS v1 다음 진행 지시서 — 2026-06-06

## 0. 공통 시작 규칙

작업 시작 전 반드시 확인한다.

1. `docs/02-design/flowx_market_os_v1.md`와 최신 커밋 `2ae96e6`, `63cdae7`을 확인한다.
2. 실주문 0, 봇 OFF, 스케줄러 변경 0, SAJANG 변경 0을 유지한다.
3. `scripts/archive/`, `_etf_ref/`는 참조·실행·복사 금지.
4. 산출물은 `data_store/` 또는 `data/`의 shadow/paper 영역에만 둔다.
5. 각 단계는 구현 → 회귀 테스트 → 보고 → 커밋 → 푸시 순서로 닫는다.
6. VPS 배포와 systemctl 재시작은 별도 승인 전 금지.

## 1단계 완료 상태

완료 커밋: `2ae96e6 feat: add FLOWX regime router v1`

완료 내용:

- `regime_router_v1` 구현
- C60 보고서를 Market OS 정책 JSON으로 번역
- R1/R4만 실제 `HARD_GATE`
- R0/R5는 `SHADOW_LABEL`, 엔진 전환권 0
- R1이면 신규진입·가설C 차단
- R4이면 SmartEntry shadow 관찰만 허용
- 매도 자동화 `BLOCKED`
- `paper_open` 기본값 `False`

1단계 후속 확인:

- 설계도 §3의 **C60 BULL/BEAR 2거래일 히스테리시스**가 실제 라우터/정책 경로에 반영됐는지 2단계 착수 전에 확인한다.
- 1단계 라우터에 히스테리시스가 없다면, 2단계 `engine_policy_map`에서 정책 확정 전 2거래일 확인을 적용하거나 별도 P1로 보강한다.
- 하루짜리 C60 깜빡임으로 엔진 허용/차단이 바뀌면 안 된다.

검증 기준:

- `tests/test_regime_router_v1.py`
- `tests/test_regime_monitor.py`
- `tests/test_paper_smart_entry_guardrails.py`
- `tests/test_collect_foreign_exhaustion_naver.py`
- 위 묶음 17개 통과

## 2단계 — engine_policy_map 구현

다음에 가장 먼저 진행한다. 3단계보다 먼저다.

목표:

`regime_router_v1`의 route 결과를 읽어, 오늘 어떤 엔진이 어떤 모드로 허용되는지 단일 정책표로 만든다.

입력:

- `regime_router_v1` route document
- 각 ticker별 `hard_gate_regime`
- 각 ticker별 `data_available`
- 각 ticker별 `shadow_labels`

출력:

- 정책 딕셔너리 또는 JSON
- 권장 경로: `data_store/policies/policy_YYYY-MM-DD.json`

정책 규칙:

| 조건 | 가설C | SmartEntry | B/C 로테이션 | Event/DART | 매도 |
|---|---|---|---|---|---|
| 데이터 없음 | 차단 | SHADOW_ONLY | SHADOW_ONLY | SHADOW_ONLY | BLOCKED |
| R1 약세/위험 | 차단 | SHADOW_ONLY | SHADOW_ONLY | SHADOW_ONLY | BLOCKED |
| R4 정상/강세 | PAPER_ONLY | ALLOWED_SHADOW | SHADOW_ONLY | SHADOW_ONLY | BLOCKED |
| R0 위험이벤트 | 권한 없음 | 권한 없음 | 권한 없음 | 권한 없음 | 권한 없음 |
| R5 과열 | 권고 라벨만 | 권고 라벨만 | 권고 라벨만 | 권고 라벨만 | 권한 없음 |

주의:

- R0/R5가 엔진을 직접 끄거나 켜면 안 된다.
- 외국인/KOSPI/변동성 보조 관측은 hard gate가 아니다.
- `PAPER_ONLY`는 **paper/shadow 트랙에서만 후보선정·관찰을 허용한다는 뜻**이다.
- `PAPER_ONLY`는 `PAPER_OPEN` 허용이 아니다.
- `PAPER_OPEN`은 기본 금지다. 6/8 관찰 가동은 기본 `SHADOW_OPEN`이다.
- 매도는 실행 금지, 신호만 허용한다.

완료 조건:

- R1에서 모든 신규진입 엔진이 shadow 또는 차단으로 내려가는 테스트
- R4에서 가설C와 SmartEntry shadow만 허용되는 테스트
- C60 전환은 2거래일 히스테리시스 적용 또는 미적용 사유가 명시되는 테스트
- R0/R5가 정책 권한을 갖지 않는 테스트
- 주문 관련 문자열 누출 금지 테스트

## 3단계 — morning_plan_07 구현

2단계 완료 후 진행한다.

목표:

아침 7시에 사람이 볼 수 있는 당일 작전계획을 생성한다.

입력:

- `regime_router_v1` 산출물
- `engine_policy_map` 산출물
- 최신 `candidate_log`
- `paper_smart_entry`의 CORE/WATCH/CONTROL 분류 로직

출력:

- `data_store/plans/plan_YYYY-MM-DD.json`
- `data_store/plans/plan_YYYY-MM-DD.md`

계획서 필수 항목:

1. 기준일
2. 각 기초자산 C60 국면
3. 오늘 허용 엔진
4. 오늘 차단 엔진
5. CORE/WATCH/CONTROL 후보 수
6. SmartEntry 관찰 가능 여부
7. 데이터 없음 또는 stale 경고
8. 매도 자동화 차단 문구
9. 실주문 0 증빙

완료 조건:

- R1이면 계획서에 신규진입 차단이 명시된다.
- R4이면 계획서에 SmartEntry shadow 관찰만 표시된다.
- 후보가 0건이어도 정상 계획서가 생성된다.
- 계획서가 paper/shadow만 말하고 실매매 지시를 만들지 않는다.

## 4단계 — candidate_tiers 정합화

3단계와 붙어서 진행하되, 별도 커밋이 좋다.

목표:

CORE/WATCH/CONTROL 분류를 plan과 ledger가 같은 기준으로 사용하게 만든다.

필수 기준:

- 최신 `candidate_log.as_of_date`만 사용
- CONTROL은 진입하지 않고 missed-winner 비교군으로만 기록
- CONTROL 기준일은 실행일이 아니라 후보 `as_of_date`
- 같은 날 같은 종목 중복 기록 금지

현재 관련 커밋:

- `63cdae7 fix: harden paper smart entry and foreign exhaustion guards`

완료 조건:

- `tests/test_paper_smart_entry_guardrails.py` 통과
- plan에 CORE/WATCH/CONTROL 숫자가 ledger와 일치

## 5단계 — smart_entry_adapter 관찰 연결

목표:

아침 계획에서 허용된 경우에만 SmartEntry 관찰을 실행한다.

원칙:

- 기본은 `SHADOW_OPEN`
- `--paper-open`은 6/8 관찰일 사용 금지
- KIS는 장중 데이터 조회만
- 주문 어댑터는 `None`
- `dry_run=True`

완료 조건:

- R1 또는 데이터 없음이면 SmartEntry 실행 대신 shadow-only 기록
- R4이면 CORE/WATCH만 SmartEntry 관찰
- CONTROL은 SmartEntry 대상 아님, missed-winner 추적만
- 실주문 관련 호출 0

## 6단계 — exit_signal_observer

목표:

매도 신호를 기록만 한다. 자동매도는 계속 금지한다.

기록 가능한 신호:

- VWAP 이탈
- 고점 대비 하락
- 체결강도 약화
- 시장 급변
- C60 약세전환

금지:

- 자동매도
- 주문 intent 생성
- sell adapter 호출
- 보유수량 변경

완료 조건:

- 신호 JSON만 생성
- `sell_automation=BLOCKED` 유지
- 주문 관련 import 없음

## 7단계 — daily_review

목표:

장마감 후 후보선정 성능과 SmartEntry 실행 성능을 분리해서 복기한다.

분리 기준:

- 후보선정 성능: 후보 `as_of_date` 종가 기준 D+10
- SmartEntry 실행 성능: 실제 관찰 진입가 기준 D+1/D+3/D+10

필수 지표:

- MFE
- MAE
- missed_winner
- false_positive
- CORE/WATCH/CONTROL별 성과
- SmartEntry 관찰 성공/실패

완료 조건:

- 후보선정과 실행성과가 섞이지 않는다.
- CONTROL도 같은 기준으로 비교된다.
- 매매 판단이 아니라 복기 리포트만 생성한다.

## 8단계 — SHOW ME 리포트

목표:

사장님이 눈으로 판단할 수 있게 그림 또는 표로 보여준다.

필수 화면:

1. C60 국면 그래프
2. CORE/WATCH/CONTROL 후보 흐름
3. 후보 D+성과 표
4. SmartEntry 관찰 가격 위치
5. missed_winner와 false_positive 비교

완료 조건:

- 숫자만 보고하지 않는다.
- 차트 또는 표로 한눈에 보이게 한다.
- 실주문 0 상태를 리포트 하단에 명시한다.

## 최종 순서

1. `engine_policy_map`
2. `morning_plan_07`
3. `candidate_tiers` plan 연동
4. `smart_entry_adapter` 관찰 연결
5. `exit_signal_observer`
6. `daily_review`
7. `SHOW ME` 리포트

현재 다음 작업은 무조건 2단계 `engine_policy_map`이다.
