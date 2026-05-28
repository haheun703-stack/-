# 퀀트봇 Runtime Truth Pack (5/28 12:00 기준)

> **목적**: Codex 감리용 매매 진입점 전수 + 안전망 + paper 가동 현황 보고
> **연결**: docs/01-plan/trading-factory-v1-architecture.md Step 1

## 1. order-capable entrypoint 전수 (5/28 12:00 기준)

### 1-1. 매매 호출 함수 (KIS API → 실주문)

| 파일 | 라인 | 함수 | 호출 경로 | 상태 (5/28) |
|------|------|------|----------|-------------|
| `src/adapters/kis_order_adapter.py` | 229 | `buy_limit` | KisOrderAdapter (어댑터) | ✅ _guard 9중 가드 + assert_runtime_orders_allowed |
| `src/adapters/kis_order_adapter.py` | 255 | `sell_limit` | KisOrderAdapter | ✅ 동일 |
| `src/adapters/kis_order_adapter.py` | 280 | `buy_market` | KisOrderAdapter | ✅ 동일 |
| `src/adapters/kis_order_adapter.py` | 301 | `sell_market` | KisOrderAdapter | ✅ 동일 |
| `src/adapters/kis_order_adapter.py` | 321 | `cancel` | KisOrderAdapter | ✅ P0-A 5/28 가드 추가 |
| `src/adapters/kis_order_adapter.py` | 341 | `modify` | KisOrderAdapter | ✅ assert_runtime_orders_allowed |
| `src/adapters/paper_order_adapter.py` | 157-228 | buy/sell × limit/market 4종 | PaperOrderAdapter (시뮬) | ✅ 실주문 X (가상만) |

### 1-2. 매매 호출 use_cases (어댑터 경유)

| 파일 | 함수 | 가드 | 상태 |
|------|------|------|------|
| `scripts/owner_rule_monitor.py` | `execute_sell` | P0-B fix: KisOrderAdapter 위임 | ✅ raw broker fallback 제거 |
| `scripts/sell_monitor.py` | `_execute_sell` | P0-D fix: KisOrderAdapter 위임 | ✅ |
| `src/use_cases/adaptive_stop_loss.py` | `execute_stop_loss_sell` | hasattr sell_market 강제 + raw fallback RuntimeError | ✅ |
| `src/use_cases/adaptive_quick_profit.py` | `execute_quick_sell` | 동일 | ✅ |
| `src/use_cases/adaptive_position_manager.py` | `execute_auto_sell` | 동일 (MVP-1) | ✅ |
| `src/use_cases/adaptive_reentry.py` | `execute_auto_reentry` | 동일 (MVP-4) | ✅ |
| `src/use_cases/smart_entry.py` | `SmartEntryEngine.run` | KisOrderAdapter 경유 | ✅ |
| `src/use_cases/smart_sell.py` | `SmartSellExecutor.execute` | 어댑터 경유 | ✅ |
| `src/telegram_command_handler.py` | `_execute_buy/sell/liquidate` | P0-C fix: assert_runtime_orders_allowed 명시 | ✅ |

### 1-3. 텔레그램 수동 매매 입력

- `_execute_buy(ticker, qty, chat_id, msg_id)` (line 505) — chat_id 화이트리스트 + 가드
- `_execute_sell` (line 549) — 동일
- `_execute_liquidate(chat_id, msg_id)` (line 575) — 전체 보유 시장가 매도

## 2. 자동 실행 트리거 (cron + systemd)

### 2-1. systemd
| 서비스 | 상태 |
|--------|------|
| `quantum-scheduler.service` | **mask /dev/null** ✅ (5/27 결단 유지) |

### 2-2. VPS crontab 매매 진입점 (5/28 긴급정지 적용)

| cron | 스크립트 | 정지 차수 | 상태 |
|------|---------|-----------|------|
| `*/5 9-15 1-5` | owner_rule_monitor.py | 1차 (5/28 08:44) | 🛑 정지 |
| `30 9 * 1-5` | chart_hero_morning_monitor.py --real | 1차 | 🛑 정지 |
| `55 14 * 1-5` | chart_hero_close_cycle.py --real | 1차 | 🛑 정지 |
| `*/5 14 * 1-5` | auto_buy_executor.py | 1차 | 🛑 정지 |
| `*/5 9-15 1-5` | run_adaptive_cycle.py --real | 1차 | 🛑 정지 |
| `50 8 * 1-5` | bash $CRON E (smart_entry_runner.py --live --force) | 2차 (5/28 09:19) | 🛑 정지 + --live --force 제거 |

### 2-3. 살아있는 cron (안전 확인)

- BAT A/B/K_safety/M_US/N/I/LU/H/L/O/D/J/F/PICKV2/HEALTH — 데이터 수집/모니터링
- `counter_trade_monitor` (`*/5 9-15`) — 코덱스 무죄 확인 (KIS 주문 호출 0건)
- `paper_warmup_daily --open` (`15 9 * 1-5`) — 시초가 기록
- `paper_warmup_daily --close` (`30 15 * 1-5`) — **5/28 P&L 시뮬 통합** ✅
- `snapshot_session`, `market_regime`, `daily_winners`, `surge_pattern_collector`
- 검수팀 5명 (EnvChecker, CodeAuditor, FlowMonitor, DataIntegrity, MarketRegimeGate, MarketScanner)

## 3. 안전망 (5/28 commit d2bc0d3 시점)

| 레이어 | 상태 | 효과 |
|--------|------|------|
| L1 `.env AUTO_TRADING_ENABLED=0` | ✅ 로컬+VPS 동일 | _guard line 111 BLOCK |
| L2 `AUTO_TRADE_5_20=false` | ✅ | auto_buy_executor 비활성 |
| L3 `data/KILL_SWITCH` 파일 | ✅ 로컬+VPS 존재 | assert_runtime_orders_allowed BLOCK |
| L4 systemd mask | ✅ | BAT 직접 실행 X |
| L5 `quant_preflight.py` | ✅ 9/9 PASS | cron 인자 + raw 호출 정적 검사 |
| L6 `_guard()` 9중 (KisOrderAdapter) | ✅ | AUTO_TRADING_ENABLED + MAX_QTY + 거래시간 등 |
| L7 raw mojito broker fallback 차단 | ✅ 5/28 적용 | RuntimeError |
| L8 텔레그램 chat_id 화이트리스트 | ✅ | 외부 사용자 차단 |

**아직 미구현 (Trading Factory v1 요구)**:
- L9 **No Intent, No Order 가드** (`order_intents_gate.py`) — Step 6 작업 대상
- L10 normalized_signals 스키마 검증 — Step 3 작업 대상

## 4. paper 가동 현황 (5/28 12:00)

| 항목 | 상태 |
|------|------|
| `paper_warmup_daily --open` (09:15) | ✅ 매일 가동 (5/18~) — 시초가 9건 기록 |
| `paper_warmup_daily --close` (15:30) | ✅ 5/28부터 P&L 시뮬 보고 가동 |
| `paper_order_adapter.py` | ✅ 구현 완료 (5/19) — 슬리피지/수수료/세금 가정 |
| `data/paper_trader_history.json` | ⏰ 오늘 15:30 첫 기록 예정 |
| paper P&L 텔레그램 보고 | ⏰ 오늘 15:30 첫 발송 예정 |

**구체 가동 정보 (오늘 5/28)**:
- 시초가 기록 9건: warmup_20260528.json (11:56 복구)
- 09:56 dry-run 결과: 적중률 11.1% (1/9), 평균 -3.79%, 당일 -19,285원 (-4.59%)
- 정식 15:30 결과: 대기 중

## 5. 5/27 사고 재발 방지 검증

### 5-1. owner_rule_monitor.py line 84 raw 호출
- **변경 전**: `broker.create_market_sell_order(symbol=ticker, quantity=qty)` (mojito raw)
- **변경 후 (commit d2bc0d3)**: `KisOrderAdapter.sell_market(ticker, qty)` 위임 — _guard 9중 가드 통과 강제
- **현재 상태**: AUTO_TRADING_ENABLED=0 → PermissionError로 차단 ✅

### 5-2. raw mojito broker 호출 정적 검사 (quant_preflight)
- 5개 use_cases + 1개 sell_monitor → raw fallback RuntimeError
- preflight 결과: "raw mojito broker calls (use_cases/scripts): no raw calls" ✅

### 5-3. cron --live --force 인자 영구 폐지
- run_bat.sh BAT-E `--live --force` 제거 (commit d2bc0d3)
- preflight 결과: "cron/shell danger args (run_bat.sh): no danger args" ✅

## 6. Codex 검수 요청 사항

1. 본 Runtime Truth Pack의 매매 호출 함수 전수 누락 여부
2. systemd / cron / textarea / 텔레그램 외 추가 진입점 (예: API 엔드포인트, 파일 기반 신호) 존재 여부
3. paper_warmup_daily 가상 매매 P&L 계산 산식 검증 (수수료 0.015% × 2 + 거래세 0.18%)
4. `order_intents_gate.py` 설계 안전성 (Step 6 작업)
5. normalized_signals 스키마 (Step 3 작업) — 봇 간 공통 합의 도출
6. 3일 paper 리허설 (5/29 또는 6/2~) 기준 met 시 live 승인 가능 조건 명확화

## 7. 잔여 작업 (퀀트봇 5/28 이후)

| Step | 작업 | 일정 |
|------|------|------|
| 3 | normalized_signals 스키마 초안 작성 | 5/29 |
| 5 | approved_swing_selector.py 설계 (기존 8개 엔진 통합) | 5/29~30 |
| 6 | order_intents_gate.py 구현 + _guard 10번째 가드 추가 | 5/29 |
| 7 | 3일 paper 리허설 가동 | 6/2~6/4 |
| 다른 봇 지시서 | 단타봇/정보봇/블로그봇 측 보낼 역할 분담 문서 | 5/28 ~ |
