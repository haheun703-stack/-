# Trading Factory v1 — 호출처 마이그레이션 Plan (5/28)

> **상태**: Plan (코덱스 4차 응답 검수 대기)
> **목적**: 어댑터 매매 메서드의 mode/executor_bot 명시 전달로 점진 마이그레이션
> **연결**: docs/01-plan/trading-factory-v1-architecture.md §11 next steps

## 1. 마이그레이션 대상

KisOrderAdapter / PaperOrderAdapter 매매 메서드 호출처 — `mode` + `executor_bot` 인자 추가 의무.

### 1-1. 호출처 전수 (5/28 시점)

| # | 파일 | 라인 | 호출 메서드 | 현재 인자 | 마이그레이션 후 |
|---|------|------|------------|----------|----------------|
| 1 | scripts/owner_rule_monitor.py:80 | execute_sell | adapter.sell_market(ticker, qty) | 동일 + mode + executor_bot |
| 2 | scripts/sell_monitor.py:289 | _execute_sell | adapter.sell_limit / sell_market | 동일 + mode + executor_bot |
| 3 | scripts/smart_entry_runner.py | smart_entry | adapter.buy_limit | 동일 + mode + executor_bot |
| 4 | src/use_cases/adaptive_buy_queue.py | execute_auto_buy | adapter.buy_limit/market | 동일 |
| 5 | src/use_cases/adaptive_position_manager.py | execute_auto_sell | adapter.sell_limit | 동일 |
| 6 | src/use_cases/adaptive_quick_profit.py | execute_quick_sell | adapter.sell_limit/market | 동일 |
| 7 | src/use_cases/adaptive_stop_loss.py | execute_stop_loss_sell | adapter.sell_market | 동일 |
| 8 | src/use_cases/adaptive_reentry.py | execute_auto_reentry | adapter.buy_market | 동일 |
| 9 | src/telegram_command_handler.py:537 | _execute_buy | adapter.buy_limit | 동일 |
| 10 | src/telegram_command_handler.py:563 | _execute_sell | adapter.sell_limit | 동일 |
| 11 | src/telegram_command_handler.py:592 | _execute_liquidate | adapter.sell_market | 동일 |
| 12 | src/use_cases/smart_entry.py | run() | adapter.buy_limit | 동일 |
| 13 | src/use_cases/smart_sell.py | execute() | adapter.sell_market/limit | 동일 |
| 14 | src/use_cases/live_trading.py | (다수) | adapter.buy_limit/sell_limit | 동일 |
| 15 | scripts/chart_hero_*.py | morning/close | adapter.buy_limit/sell_market | 동일 |
| 16 | scripts/auto_buy_executor.py | execute_auto_buy | adapter.buy_limit | 동일 |
| 17 | scripts/paper_warmup_daily.py | cmd_close (5/28 추가) | (현 PaperOrderAdapter 미사용) | mode="paper" + executor_bot="quant" |

## 2. 마이그레이션 단계

### Phase 1 (5/28~5/29): paper 호출처 우선
- 우선순위: paper_warmup_daily, 5/28 신규 paper 시뮬 코드
- 영향: paper 모드에서 동작 확인 (실 KIS 영향 X)

### Phase 2 (5/30~6/1): live 호출처 (단타봇 매매 경로)
- 우선순위: telegram_command_handler (수동), smart_entry_runner, auto_buy_executor
- 영향: live executor_bot="day" 인자 명시
- 단, 현재 cron 6개 정지 상태라 실제 동작 X (안전)

### Phase 3 (6/2~6/4): paper 리허설 가동
- 모든 호출처 마이그레이션 완료
- 3일 paper 리허설 수행
- order_intents gate 통과 빈도 + NoIntentError 발생 로그 수집

### Phase 4 (6/5): 코덱스 검수
- 마이그레이션 완료 + paper 결과 + gate 동작 검수
- 잔여 fail 패턴 fix
- live 가동 결단 (사용자 + 코덱스)

### Phase 5 (6/9+): live 가동 전 추가 검수
- modify_intent v2 도입 결단 (필요시)
- approved_swing_selector / approved_intraday_selector 가동
- live 6개 cron 재가동 결단

## 3. 마이그레이션 패턴 (코드 예시)

### Before
```python
order = adapter.buy_limit(ticker, price, qty)
```

### After (paper)
```python
# 1. selector가 intent 등록
intent = {
    "intent_id": f"q_{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    "bot": "quant", "engine": "sector_fire_v3",
    "ticker": ticker, "side": "BUY", "mode": "paper",
    "score": 81.7, "created_at": now_seoul.isoformat(),
    "expires_at": (now_seoul + timedelta(hours=4)).isoformat(),
}
register_intent(intent, bot="quant")

# 2. 매매 실행 (paper)
order = adapter.buy_limit(ticker, price, qty, mode="paper", executor_bot="quant")
```

### After (live, day bot만 가능)
```python
intent = {..., "bot": "day", "mode": "live", ...}
register_intent(intent, bot="day")
order = adapter.buy_limit(ticker, price, qty, mode="live", executor_bot="day")
```

## 4. 잔여 위험

### 4-1. modify/cancel 정책 미확정
- 별도 docs: docs/02-design/order-intents-cancel-modify-policy.md
- 6/9+ live 전 확정

### 4-2. OrderPort 인터페이스 업데이트 또는 IntentBoundOrderPort
- 현재 OrderPort 시그니처에 mode/executor_bot 없음 (backward compat)
- 향후 IntentBoundOrderPort 별도 정의 검토 (Phase 4+)

### 4-3. assert_runtime_orders_allowed 우회 위험
- L8 가드는 KILL_SWITCH 등 환경 기반
- L10 강제 후에도 L8 통과 필수 (defense in depth 유지)

## 5. 검수 (Codex)

1. Phase 분할의 안전성
2. 호출처 우선순위 합리성
3. OrderPort 시그니처 업데이트 시점
4. IntentBoundOrderPort 별도 정의 필요성
5. live executor_bot="day" 인자 명시의 효과 (현재 cron 정지 중)
