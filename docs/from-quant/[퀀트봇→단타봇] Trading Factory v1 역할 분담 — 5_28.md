# [퀀트봇 → 단타봇] Trading Factory v1 역할 분담 지시서

> **작성일**: 2026-05-28 12:10 KST
> **연결**: 퀀트봇 docs/01-plan/trading-factory-v1-architecture.md (사용자 5/28 결단)
> **목적**: 봇들이 각자 떠드는 시스템 → 기관식 매매 공장 전환

## 1. 단타봇 역할 (확정)

**Intraday Alpha + Execution Bot** — 장중 빠른 기회 포착 + 짧은 보유 + 분할 집행

### 책임 (Have)
- 상한가 엔진, NXT, 외국인/기관 매집, Rule D, 눌림목 신호 → `normalized_signals.jsonl` 표준화
- `approved_intraday_selector.py` 신규 작성 → TOP 3만 고름
- 09:00~09:15 사이 VPS 런타임 `--paper --emit-intents` 실행
- 결과 → `candidate_snapshot_YYYYMMDD_0915.json` + `order_intents_YYYYMMDD_0915.jsonl` 저장
- **주문 함수는 order_intents에 있는 종목만 paper 주문**
- VWAP/TWAP/POV 분할매수 엔진으로 집행
- 장중 손절/익절/시간청산도 intent 기반으로만 실행

### 영구 금지 (Have NOT)
- 임의 종목 추가 매수
- verification 단독 매수
- LLM 추정 후보 매수
- order_intents 없는 종목 매매

## 2. 단타 점수식 (사용자 결단)

```
intraday_score =
  상한가/급등 신호
+ NXT 야간 강도
+ 외국인/기관 동시 매집
+ 오늘 거래대금/거래량
+ 눌림 후 VWAP 회복
+ Rule D 우선순위
- 갭 과열
- 유동성 부족
- 이미 보유 중복
- 시장 약세
```

각 가중치는 단타봇이 백테스트로 산출, `source_weight` 파일로 외부화.

## 3. 산출물 (의무)

| 파일 | 빈도 | 용도 |
|------|------|------|
| `normalized_signals.jsonl` | 매일 | 신호 표준화 |
| `candidate_snapshot_YYYYMMDD_0915.json` | 매일 | 09:15 후보 스냅샷 |
| `order_intents_YYYYMMDD_0915.jsonl` | 매일 | 주문 의도 (paper 우선) |
| `execution_report.json` | 매일 | 실제 집행 vs intent 일치율 |
| `intraday_pnl_report.json` | 매일 | 당일 paper 손익 + 누적 |

## 4. order_intents 스키마 (공통)

```jsonl
{"intent_id":"d_2026-05-28_001","bot":"day","engine":"limit_up_recovery","ticker":"240810","name":"원익IPS","side":"BUY","mode":"paper","budget":300000,"target_qty":2,"entry_price":121000,"exit_target":124500,"stop_loss":118500,"horizon_hours":4,"score":85.0,"confidence":"strong","reasons":["NXT +5.2%","외인+기관 동시매수","VWAP 회복"],"created_at":"2026-05-28T09:00:00+09:00","expires_at":"2026-05-28T15:30:00+09:00","candidate_snapshot_id":"snap_d_2026-05-28_0900"}
```

## 5. 핵심 가드 (No Intent, No Order)

```python
# 단타봇 매매 호출 함수 진입 시 의무
from order_intents_gate import assert_order_intent_exists

def execute_paper_buy(ticker, qty, price, side="BUY"):
    intent = assert_order_intent_exists(ticker, side, mode="paper")
    # intent 없으면 RuntimeError 발생 + 매매 차단
    # 이후 매매 로직
```

## 6. 5/28 현재 단타봇 진입점 (퀀트봇 측 발견)

### 6-1. cron 운영 중 (VPS)
- `*/10 * * * *` swap_monitor.sh — 자동매매 관련 (역할 확인 필요)
- `0,30 9-15 * * 1-5` intraday_learning.py — 학습 도구 (매매 X)
- `0,30 9-15 * * 1-5` intraday_learning_v2.py --once — 학습 v2 (매매 X)
- `30 17 * * 1-5` foreign_accumulation_scanner — 야간 데이터 수집

### 6-2. 5/28 오류 발견
- `intraday_learning_v2.py:127` 호출: `fetch_minute_chart(code, minutes=5, n=n_bars)`
- `kis_trader.py:2057` 정의: `fetch_minute_chart(code, count: int = 30)` — 시그니처 불일치
- **퀀트봇 측에서 fix 완료 (로컬만)**: minutes/n 키워드 인자 추가 + `D:/Prophet_Agent_System_예언자/scalper-agent/bot/kis_trader.py`
- 단타봇 git 측에서 commit + push 필요

### 6-3. Supabase 컬럼 오류 (정보봇 의존)
- `intraday_learning_v2.py:189` 쿼리: `WHERE ticker = %s FROM intelligence_supply_demand`
- 에러: `column "ticker" does not exist`
- 원인: `intelligence_supply_demand` 테이블은 시장 전체 데이터, ticker 컬럼 자체 없음 (on_conflict=date)
- 해결: 단타봇이 다른 테이블 쿼리 (sector_investor_flow 등) 또는 쿼리 자체 폐기

## 7. 단타봇 신규 작업 (즉시)

| Step | 작업 | 우선순위 |
|------|------|---------|
| 1 | Runtime Truth Pack 제출 (cron + 매매 함수 + 안전망 전수) | P0 |
| 2 | normalized_signals.jsonl 스키마 합의 (퀀트봇 + Codex와) | P0 |
| 3 | approved_intraday_selector.py 신규 작성 | P0 |
| 4 | order_intents_gate.py 통합 (퀀트봇이 공통 모듈 제공) | P0 |
| 5 | VWAP/TWAP/POV 분할매수 엔진 점검 | P1 |
| 6 | 5/28 fetch_minute_chart fix commit + push | P1 |
| 7 | Supabase ticker 컬럼 쿼리 폐기 또는 대체 | P1 |
| 8 | 3일 paper 리허설 (6/2~6/4) 준비 | P2 |

## 8. 영구 금지 사항 (재확인)

- ❌ 단타봇은 8개로 분산되어 운영 금지 (5/27 통합 완료 유지)
- ❌ 메인 AI(Claude)가 자기 코드 위치 모르는 상태 금지
- ❌ verification 없이 매수 금지
- ❌ order_intents에 없는 종목 매수 금지
- ❌ paper 검증 없이 live 가동 금지

## 9. 퀀트봇 ↔ 단타봇 인터페이스

- 퀀트봇이 단타봇에 넘길 수 있는 **장중 후보**: swing 후보 중 단기 모멘텀 강한 종목 (퀀트봇 selector가 분류)
- 단타봇이 퀀트봇에 넘기는 정보: 장중 paper 실측 P&L → 퀀트봇 selector_weight 학습 데이터

## 10. 검수 (Codex)

- 본 지시서 검수 대상: `ops/codex_inbox/20260528T12XXXX_quant-bot_trading-factory-v1-plan.json`
- 단타봇 응답 받기 전 메인 AI(Claude) 단독 commit 금지
