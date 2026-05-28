# Trading Factory v1 — 기관식 매매 공장 아키텍처

> **상태**: Plan (사용자 결단 5/28 12:00)
> **계기**: 5/27 09:55 owner_rule_monitor 사고 → "봇들이 각자 떠드는 시스템" → "기관식 매매 공장"으로 전환

## 1. 핵심 비전

> "단타봇은 장중 실행자, 퀀트봇은 스윙 연구자, 정보봇은 정보 공급자, 블로그봇은 설명자, Codex는 감리자."

수익 추구와 통제가 함께 가는 구조.

## 2. 공통 원칙 (모든 봇)

| # | 원칙 | 위반 시 |
|---|------|---------|
| 1 | LLM은 종목을 임의 선택하지 않는다 | 즉시 정지 |
| 2 | 승인된 signal engine과 selector만 후보 생성 | order_intents 미생성 |
| 3 | 모든 후보 `candidate_snapshot`에 기록 | 추적 불가 |
| 4 | 모든 주문 예정 `order_intents`에 기록 | 주문 차단 |
| 5 | **`order_intents`에 없는 종목 주문 금지** ★ | RuntimeError |
| 6 | PAPER 검증 전 live 금지 | Codex 차단 |

## 3. 역할 분담

### 3-1. 단타봇 (Intraday Alpha + Execution Bot)
- **위치**: D:/Prophet_Agent_System_예언자/scalper-agent/ (로컬), /home/ubuntu/bodyhunter/scalper-agent/ (VPS)
- **목표**: 장중 빠른 기회 포착 + 짧은 보유 + 분할 집행 (VWAP/TWAP/POV)
- **점수식**: 상한가 + NXT + 외인/기관 동시매집 + 거래대금 + 눌림 VWAP 회복 + Rule D - 갭과열 - 유동성부족 - 중복 - 약세
- **산출물**: normalized_signals.jsonl, candidate_snapshot.json, order_intents.jsonl, execution_report.json, intraday_pnl_report.json

### 3-2. 퀀트봇 (Swing Alpha Research + Selector Bot) ★ 본 프로젝트
- **자동 주문 주체 아님** — owner_rule_monitor류 독립 매매 영구 금지
- **목표**: 1~5일 스윙 후보 발굴 + 엔진별 성과 학습
- **점수식**: 실적/밸류 + 섹터테마 발화 + 외인/기관 3~5일 누적 + 바닥권 + 거래대금 + 정보봇 이벤트 - 단기과열 - 공매도 - 시장레짐 - 손절폭과대
- **산출물**: swing_candidate_snapshot.json, source_performance_rolling.json, selector_weight_proposal.md, daily_research_report.md

### 3-3. 정보봇 (Event Intelligence Bot)
- **위치**: VPS `/home/ubuntu/jgis/`
- **주문 기능 절대 금지** — KIS 주문 함수 import 금지
- **산출물**: event_signals.jsonl (ticker, sector, freshness, impact_score, confidence)

### 3-4. 블로그봇 (Publishing + Audit Narrative Bot)
- **위치**: D:/flowx-blog_블로그 봇/, D:/kr.TradingView.site_웹봇/
- **매매 관여 금지** — 설명/리포트/성과 기록만
- **자본시장법 금지 표현 필터링**

### 3-5. Codex (Gatekeeper)
- 모든 봇 Runtime Truth Pack 검수
- order-capable entrypoint 전수 검사
- --live/--real/cron/systemd/process 검사
- No Intent, No Order 가드 확인
- paper 리허설 결과 검수 전 live 승인 금지

## 4. 즉시 실행 8단계 (사용자 명시)

| Step | 작업 | 담당 | 산출 |
|------|------|------|------|
| 1 | 각 봇 Runtime Truth Pack 제출 | 각 봇 메인 AI | docs/02-design/{bot}-runtime-truth-pack.md |
| 2 | Codex 매매 진입점 지도 작성 | Codex | docs/02-design/order-entrypoint-map.md |
| 3 | normalized_signals 공통 스키마 확정 | 메인 AI + Codex | docs/02-design/normalized-signals-schema.md |
| 4 | 단타 approved_intraday_selector.py 설계 | 단타봇 | scripts/approved_intraday_selector.py |
| 5 | 퀀트 approved_swing_selector.py 설계 | 퀀트봇 | scripts/approved_swing_selector.py |
| 6 | order_intents 없으면 주문 불가 가드 삽입 | 모든 봇 + Codex | src/use_cases/order_intents_gate.py |
| 7 | 3일 paper 리허설 (5/29~5/31 또는 6/2~6/4) | 모든 봇 | data/paper_rehearsal_{date}.json |
| 8 | 성과/불일치 검수 후 live 승인 (별도) | Codex + 사용자 | live 승인 문서 |

## 5. 퀀트봇 적용 변경 사항 (즉시)

### 5-1. 영구 금지 (코드 레벨)
- owner_rule_monitor류 독립 매매 → **삭제 또는 영구 비활성** (commit d2bc0d3 P0-B로 이미 KisOrderAdapter 위임)
- crontab 직접 live 실행 → **5/28 긴급정지 6건 유지**
- `--real`, `--live`, `--force` CLI 인자 → **영구 폐지** (quant_preflight 정적 검사 적용 중)

### 5-2. 신규 산출물 디렉토리
- `data/candidate_snapshot/swing_YYYYMMDD.json`
- `data/order_intents/swing_intents_YYYYMMDD.jsonl` (퀀트봇은 paper만, live X)
- `data/source_performance/rolling.json`
- `docs/04-report/daily_research_YYYYMMDD.md`

### 5-3. selector 후보 엔진 (퀀트봇 보유)
1. 바닥반등 (`scripts/scan_crash_bounce.py`)
2. 수급전환 (`scripts/scan_supply_surge.py` 등)
3. 실적괴리 (`scripts/scan_earnings_gap.py`)
4. 섹터발화 (`scripts/scan_sector_fire.py` v3)
5. ETF/테마 회전 (`scripts/etf_*`)
6. 외국인 누적매집 (`scripts/foreign_*` + 정보봇 OHLCV 39컬럼)
7. 피보나치 (`scripts/scan_fibonacci.py`)
8. 차트영웅 (5-Gate, paper 전용)

## 6. order_intents 스키마 (초안)

```jsonl
{"intent_id":"q_2026-05-28_001","bot":"quant","engine":"sector_fire_v3","ticker":"240810","name":"원익IPS","side":"BUY","mode":"paper","budget":100000,"target_qty":1,"entry_price":121000,"target_price":127000,"stop_loss":117500,"horizon_days":3,"score":81.7,"confidence":"strong","reasons":["AI반도체 53점","외+기 동시매수","MA5 위"],"created_at":"2026-05-28T09:00:00+09:00","expires_at":"2026-05-31T15:30:00+09:00","candidate_snapshot_id":"snap_2026-05-28_0900"}
```

## 7. No Intent, No Order 가드 (구현 위치)

`src/use_cases/order_intents_gate.py`:
```python
def assert_order_intent_exists(ticker: str, side: str, mode: str = "paper") -> dict:
    """모든 매매 주문 함수 진입 시 호출. order_intents 파일에 해당 intent 없으면 RuntimeError."""
    # data/order_intents/{bot}_intents_YYYYMMDD.jsonl 조회
    # ticker + side + mode + 유효 시간 매치 시 통과
    # 매치 없으면 RuntimeError("[NO_INTENT] order_intents 미등록")
```

`KisOrderAdapter._guard()` 9중 가드에 10번째로 추가.

## 8. 일정

- **5/28 (오늘)**: PDCA Plan + 퀀트봇 Runtime Truth Pack + 4개 봇 지시서 작성
- **5/29 (목)**: normalized_signals 스키마 + order_intents_gate 구현 + selector skeleton
- **5/30 (금)**: 각 봇 normalized_signals 출력 시작 + paper 리허설 D-1 점검
- **6/2 (월)~ 6/4 (수)**: 3일 paper 리허설 가동
- **6/5 (목)**: 성과 + 불일치 검수 → Codex 보고
- **6/9+ (월)**: live 승인 결단 (사용자 + Codex)

## 9. 잔여 위험

- 정보봇 측 코드 추적 안 됨 (KIS import 금지 검증 필요)
- 블로그봇 측 시장 영향력 (자본시장법) 사전 컴플라이언스 검토
- order_intents 파일 무결성 (외부 수정 방지) — HMAC 서명 검토
- paper 리허설 슬리피지/수수료 가정 실측 검증 (5/22 백테스트 D+1 +20.60% 보존)
