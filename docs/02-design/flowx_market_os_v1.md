# FLOWX Market Operating System v1 — 설계도

> **코드 0, 설계도 1장 / 실주문 금지 / paper·shadow 운영체제**
> 작성 2026-06-06 · 상태: 설계 확정(구현 전) · 다음 작업: 구현순서 1단계(regime_router_v1)부터 별도 승인 후 착수

---

## 0. 핵심 철학

이 시스템의 목표는 "좋은 종목 찾기"가 아니라 **매일 시장국면을 판단하고 그 국면에 맞는 엔진을 자동 선택하는 운영체제**다.

**6국면을 전부 hard gate로 쓰지 않는다.** 검증된 R1/R4(C60 BULL/BEAR)만 진짜 게이트로 두고, 나머지 4개 국면(R0·R2·R3·R5)은 기록·분석용 관찰 라벨이다. 검증 안 된 분류에 엔진 자동전환 권한을 주면, 그 자체가 "자동으로 잘못된 판단을 실행하는 경로"가 된다.

> **근거**: 5/27 `owner_rule_monitor`가 검증 안 된 매도 로직으로 원익IPS를 자동 손절매도한 사고. 교훈 = **검증 안 된 엔진을 자동 파이프라인에 배선하면 = 자동으로 돈 잃는 경로.** 그래서 이 OS는 매수 로직보다 **안전장치(국면 게이트 제한·히스테리시스·preflight·매도 BLOCKED)를 먼저 박는다.**

---

## 1. 최상위 7단계 파이프라인 (고정)

| 단계 | 내용 | 산출/상태 |
|---|---|---|
| 1 | 전날/장전 **시장국면 판단** (C60 SSOT) | regime json |
| 2 | **국면별 엔진 선택** (R4→가설C 허용, R1→전부 shadow) | policy map |
| 3 | 아침 7시 **작전계획 생성** | plan json + md |
| 4 | 후보 **CORE / WATCH / CONTROL** 분리 | tier 목록 |
| 5 | 장중 **SmartEntry 관찰** (VWAP/체결강도) | shadow 기록 |
| 6 | **매도/리스크 시그널 관리** | 신호 기록만 (BLOCKED) |
| 7 | 장마감 **복기·학습** (MFE/MAE, missed/false) | daily_review |

**실주문 금지. 1차 목표 = paper/shadow 운영체제 완성.** 코드가 매수하는 것이 완료가 아니다.

---

## 2. 상태 라벨 4종 (문서 전체 표기 규약)

| 라벨 | 의미 |
|---|---|
| **HARD_GATE** | 검증됨. 실제 차단/허용에 사용 |
| **SHADOW_LABEL** | 미검증. 기록만, 엔진 자동전환 권한 0 |
| **PAPER_ONLY** | paper에서만 실행 |
| **BLOCKED** | 자동화 금지 |

---

## 3. 국면 라우터 R0~R5 (★가장 중요)

**검증된 R1/R4만 HARD_GATE. 나머지 4개는 SHADOW_LABEL(엔진 전환권 없음).**

| 국면 | 정의 (계산식) | 상태 | 행동 권한 |
|---|---|---|---|
| **R0 위험이벤트** | C60과 **별개**. 지수 급락 / 환율 급등 / EWY 급락 / preflight 실패 / 대외충격 | SHADOW_LABEL | "오늘 왜 위험했나" 설명 태그. **엔진 차단권 없음** |
| **R1 약세/위험** | **C60 BEAR** (close ≤ MA60) | **HARD_GATE** | 신규진입 차단 · 현금화/방어 · 가설C(C-rule) 비활성 |
| **R2 급락후 반등** | `market_context.drop_context` = market_selloff / resilient_pullback | SHADOW_LABEL | 회복후보만 관찰, stock_specific_drop 금지. 계산식 있되 권한 0 |
| **R3 횡보** | KOSPI N일 변동성 낮음 + 박스권 | SHADOW_LABEL | **횡보장 엔진 미구현** → research/shadow만 |
| **R4 정상/강세** | **C60 BULL** (close > MA60) | **HARD_GATE** | 가설C CORE + 섹터눌림 B + 이벤트 A 관찰 허용 |
| **R5 과열** | C60 BULL + 60선 이격도 과열 (실측 +57~62%) | SHADOW_LABEL | 추격금지/보유관리 **권고 라벨** — 자동권한 0 |

> **R0 ≠ R1 (중요)**: C60 BEAR 자체는 R1이다. R0는 C60과 무관하게 "급락·충격이 있었다"를 설명하는 태그일 뿐 엔진을 끄지 않는다. (R0가 C60 BEAR와 겹치면 게이트가 이중으로 작동해 의미가 흐려진다 — 사장님 지적 반영)

### 엔진 역할 분리 (섞지 말 것)
| 엔진 | 역할 | 검증 |
|---|---|---|
| **가설C** | 강세/정상장(R4) **종목선별** | 강세 OOS PASS / 약세 부적합 |
| **C60** | 약세장(R1) **방어** | PASS (HARD_GATE) |
| **SmartEntry** | 장중 **진입 타이밍** | PAPER_ONLY |

### 히스테리시스 (휩쏘 방지)
C60 BULL/BEAR 전환은 **2거래일 연속 확인 후 확정**(대칭). 하루짜리 깜빡임으로 엔진 전환 금지.
> **근거**: C60이 강세장 V자(3/31~4/2)에서 휩쏘로 BULL↔BEAR 깜빡인 이력. 매도가 BLOCKED라 늦은 방어로 인한 위험 없음 → 대칭 2일로 시작.

### C60 단일진실원천 (SSOT)
`regime_router`는 **`src/etf/regime_monitor.py`의 C60만 단일 참조**(CLI 진입점 `scripts/regime_monitor.py`). C60 = 종목별 `close > MA60 → BULL / ≤ → BEAR_TRANSITION`. 외국인/KOSPI/변동성은 동 모듈 `OBSERVATION_GATE_STATUS`에서 명시적 로그-only(gate 미사용).
> C60 복붙 중복 제거는 **후속 P2** (AUDIT_BACKLOG).

---

## 4. 데이터 헬스 preflight (파이프라인 맨 앞 전제조건)

| 체크 | 도구/근거 |
|---|---|
| 거래일 가드 | `src/trading_calendar.py` `is_kr_trading_day` |
| KOSPI index stale | 최신 거래일 종가 존재 확인 |
| parquet stale | processed 최신일 확인 |
| EWY/미국장 stale | (EWY는 현재 5/15부터 끊김 = 알려진 결함) |
| 0행·결측·휴장 ghost | OHLCV=0 행 차단 |

**preflight 실패 시: 신규진입 금지(shadow only), 보유 포지션도 "데이터 불신" 플래그로 관찰만.**
> **근거**: 6/3 휴장 유령행(OHLCV=0)이 전종목 processed parquet을 오염시킨 사고. 오염된 C60을 읽으면 국면 자체가 오판된다.

---

## 5. 성과 측정 (★2개로 분리)

| 측정 | 기준 | 용도 |
|---|---|---|
| **후보선정 성능** | `as_of_date`(후보 생성일) 종가 → **D+10** | CORE/WATCH/CONTROL **공통**(공정 비교) |
| **SmartEntry 실행 성능** | 실제 SmartEntry 진입가 → **D+1/D+3/D+10** | 실행 품질 별도 평가 |

> 후보선정과 실행은 기준 시점이 다르다. 섞으면 사과 vs 오렌지 비교가 된다.

### 사전 정의 (결과 보기 전에 고정)
- **missed_winner**: CONTROL/탈락군이 D+10 **≥ +8%**
- **false_positive**: CORE/WATCH가 손절선 도달 또는 D+10 **음수**

### 승격 조건
paper **10거래일 이상** + base 대비 승률/평균수익 개선 + **MDD −15% 이내** + missed_winner/false_positive 사전정의 통과.
**단, 자동승격 절대 금지. 최종 승격 = 사장님 승인.**
> **근거**: CORE 표본이 하루 1~3개라 10일 모아도 통계검정엔 부족. 숫자는 *전제조건*일 뿐, 숫자가 사람을 대체하면 사후합리화가 된다.

---

## 6. 부품 매핑 상태표

| 부품 | 파일 경로 | 역할 | 상태 | 근거 |
|---|---|---|---|---|
| C60 국면판정 | `src/etf/regime_monitor.py` (CLI `scripts/regime_monitor.py`) | 라우터 SSOT | **HARD_GATE** | 검증됨, 보조지표 로그-only |
| 가설C 선별 | `paper_track.py` build_floor_quality/market_context/supply_confirmation | R4 종목선별 | **PAPER_ONLY** (R4) / R1 비활성 | in 73.7% vs 56.1% · OOS 61.3% vs 49.4%(KOSPI+0.74%p) · **2022 42.6% vs 47.5% 미달** |
| 기존 4중결합 | (paper_track 구 로직) | — | **폐기** | in-sample base 미달(55.9 vs 56.1) + 2022 진입 0건 |
| 3-tier 분류 | `paper_smart_entry.py` classify_tier / record_control_pool | CORE/WATCH/CONTROL | PAPER_ONLY | look-ahead 차단(asof 슬라이스) 확인 |
| SmartEntry | `src/use_cases/smart_entry.py` `vwap_gate.py` `vwap_eye_advisor.py` `intraday_entry_trigger.py` `kis_intraday_adapter.py` | 장중 타이밍 | **PAPER_ONLY** | 코드 有, 6/8 관찰 |
| 미국장/EWY/거시 | `scripts/regime_macro_signal.py` `collect_foreign_exhaustion.py` `macro_leadlag_adversarial.py` | 맥락 | **SHADOW_LABEL** | EWY 5/15 끊김 · 거시 NOISE 검증 → hard gate 금지 |
| 매도/smart_sell | `src/use_cases/smart_sell.py` `src/agents/sell_brain.py` | 청산 | **BLOCKED** | 5/27 자동매도 사고. 실주문 경로(`_market_immediate`) 존재 |
| 횡보장(R3) 엔진 | — | — | **없음(연구대상)** | 미구현 |
| 오케스트레이터 | — | 국면→엔진 배선 | **없음(이번 설계 대상)** | 부품은 있으나 지휘자 부재 |
| 아침 브리핑 | `src/agents/reporter.py`(06:00) `data_integrity.py`(07:00) `flowx_uploader.py` | 점호/적재 | 가동 中 | — |
| 분할 작전 | `docs/02-design/leverage-split-buy-table-6_3.md` 분할표C | 진입/약세전환 | 참조 | 강세 +740%/−45% · 2022 −9.6% |
| 거래일 가드 | `src/trading_calendar.py` | preflight | 사용 | — |

---

## 7. 구현 순서 7단계 (다음 작업 — 이번엔 설계만, 코드 0)

| # | 모듈 | 역할 | 출력 | 레이어 |
|---|---|---|---|---|
| 1 | `regime_router_v1` | C60 읽어 오늘 국면 출력 | `data_store/regime/regime_YYYY-MM-DD.json` | use_case |
| 2 | `engine_policy_map` | 국면별 허용 엔진 결정 (R4→가설C, R1→shadow) | (정책 dict) | use_case |
| 3 | `morning_plan_07` | 아침 7시 작전계획 | `data_store/plans/plan_YYYY-MM-DD.json` + md | use_case |
| 4 | `candidate_tiers` | CORE/WATCH/CONTROL 분리 (6/8 = CORE1/WATCH2/CONTROL9) | tier 목록 | use_case |
| 5 | `smart_entry_adapter` | 후보 관찰만. 기본 SHADOW, PAPER_OPEN은 명시 옵션 없으면 금지 | shadow 기록 | adapter |
| 6 | `exit_signal_observer` | 자동매도 금지. VWAP 이탈/고점대비 −3~−5%/체결강도 하락/시장 급변을 **신호 기록만** | exit 신호 | use_case/adapter |
| 7 | `daily_review` | 장마감 후 MFE/MAE, missed_winner, false_positive, 손절/익절 시뮬 | 복기 기록 | use_case |

**클린아키텍처**: entities(국면/플랜 모델) → use_cases(라우터·플랜·tier·review) → adapters(data_store json·KIS read-only) → agents(조율). **안쪽→바깥 import 금지** (CLAUDE.md 규칙).

---

## 8. 기존 보유 × 국면전환 정책

신규진입뿐 아니라 보유 포지션도 정책에 포함한다.

- R4에서 산 종목이 **R1로 전환**되면 → **자동매도하지 말고 shadow exit signal만 기록.**
- 실제 청산 로직은 **별도 검증 전까지 관찰만**(매도 BLOCKED 유지).
- 약세전환 처리는 분할표C의 약세전환 로직과 연결(shadow).

---

## 9. 타임라인 의존성

```
전날 저녁 US/EWY 수집  →  새벽 적재  →  06:00 점호(reporter)
  →  07:00 plan(morning_plan_07)  →  09:00~ 장중 SmartEntry 관찰
  →  15:30 마감  →  daily_review
```
국면 판단은 **전날 종가 C60** 기준(점프 없음). 장중 시장 급변은 exit_signal_observer가 신호로 기록.

---

## 10. 6/8 운영 원칙

- 실주문 **0** / scheduler 연결 **금지** / SAJANG 변경 **금지**
- PAPER_OPEN **기본 금지** (SHADOW가 기본, `--paper-open` 명시할 때만)
- CORE · WATCH · CONTROL **모두 추적**
- C60이 **약세·위험이면 전부 shadow**
- C60이 **정상·강세일 때만 SmartEntry 관찰**
- 매도는 **자동실행 금지, 신호만 기록**

---

## 11. 완료 기준

1. `flowx_market_os_v1.md` 설계도 생성 ✅
2. 현재 코드 부품 매핑 완료 ✅
3. 국면별 엔진 선택표 확정 ✅
4. 6/8 관찰 플랜 자동 생성 가능 (설계상)
5. 실주문 0 / scheduler 0 / SAJANG 0 확인
6. 매도 자동화는 BLOCKED로 명시 ✅

---

> 이 문서는 **설계도이며 코드를 포함하지 않는다.** 다음 작업은 구현 순서 1단계(`regime_router_v1`)부터, **별도 승인 후** 착수한다.
