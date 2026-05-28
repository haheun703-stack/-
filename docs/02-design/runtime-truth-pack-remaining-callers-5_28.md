# 잔여 매매 호출자 Runtime Truth Pack (5/28 코덱스 16:57 결정 응답)

> **상태**: Runtime 분류 보고 (Phase 1 paper cron 재가동 결단용)
> **HEAD**: f97711c
> **계기**: 코덱스 5/28 16:57 — "잔여 호출자 Runtime Truth Pack 제출 후 paper cron 재개 재검수"

## 1. 분류 기준

| 분류 | 정의 |
|------|------|
| A. 실행 중 | cron/systemd 활성 → 매매 실제 발생 가능 |
| B. cron 등록 (정지) | crontab 라인 존재 but 5/28 긴급정지로 주석 |
| C. 수동 명령 | 텔레그램/CLI 등 사용자 직접 트리거만 |
| D. 폐기 대상 | 호출자 없음 또는 archive 대상 |

## 2. 잔여 호출자 매트릭스

### 2-1. scripts/auto_buy_executor.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 443 `kis_order.buy_limit(tk, current_price, 1)` |
| cron 상태 | B. 정지 (`# [긴급정지 5/28] */5 14 * * 1-5`) |
| L10 마이그레이션 | 미적용 (kwargs 없음) |
| 실행 빈도 (정지 전) | 매 5분 14:00~14:55 |
| 우선순위 | **P0** (Phase 1 paper 재가동 전 필수) |

### 2-2. scripts/sell_monitor.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 305 sell_limit, 326 sell_market (PaperOrderAdapter 위임 — 5/28 P0-D fix) |
| cron 상태 | BAT-E 08:50 내부 `--dry-run` 강제 (BAT-E 정지) |
| L10 마이그레이션 | 미적용 |
| 우선순위 | **P1** (dry-run으로 위험 낮음) |

### 2-3. scripts/owner_rule_monitor.py ★

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 95 `adapter.sell_market` (5/28 P0-B로 KisOrderAdapter 위임 적용) |
| cron 상태 | B. 정지 (`# [긴급정지 5/28] */5 9-15`) |
| L10 마이그레이션 | 미적용 (kwargs 없음) |
| 우선순위 | **P0** ★ 5/27 사고 직접 원인 + Phase 1 자동 청산 |

### 2-4. src/telegram_command_handler.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 540 buy_limit, 565 sell_limit, 593 sell_market |
| cron 상태 | C. 수동 명령 (telegram bot 24/7 active service) |
| L10 마이그레이션 | 미적용 (P0-C로 `assert_runtime_orders_allowed()` 호출은 추가) |
| 우선순위 | **P0** ★ 24/7 활성, chat_id 화이트리스트 외 추가 가드 필요 |

### 2-5. src/use_cases/smart_entry.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 644 buy_limit, 1284 modify, 1310 cancel, 1696 추매 buy_limit |
| cron 상태 | B. 정지 (BAT-E 08:50 → smart_entry_runner → SmartEntryEngine.run) |
| L10 마이그레이션 | 미적용 |
| 우선순위 | **P0** (BAT-E 재가동 전 필수) |

### 2-6. src/use_cases/smart_sell.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | line 91 sell_market, 130 sell_limit, 227 sell_limit |
| cron 상태 | D. 호출자 추적 필요 (smart_entry/live_trading 위임 추정) |
| L10 마이그레이션 | 미적용 |
| 우선순위 | **P1** (호출자 검증 후) |

### 2-7. src/use_cases/live_trading.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | (별도 grep 필요) |
| 호출자 | main.py stock_buy/stock_sell 추정 |
| L10 마이그레이션 | 미적용 |
| 우선순위 | **P1** |

### 2-8. src/use_cases/adaptive_trend_exit.py / adaptive_position_manager.py

| 항목 | 값 |
|------|-----|
| 매매 호출 | adaptive_trend_exit:350/364, adaptive_position_manager:340 |
| cron 상태 | run_adaptive_cycle 안에서 호출 (5/28 정지) |
| L10 마이그레이션 | 미적용 |
| 우선순위 | **P0** (run_adaptive_cycle 재가동 전 필수) |

### 2-9. src/split_order.py

| 항목 | 값 |
|------|-----|
| L10 마이그레이션 | b76dc5e SplitOrderExecutor.__init__에 mode/executor_bot 인자 추가됨 |
| 호출자 측 전달 | 미적용 (호출자에서 SplitOrderExecutor 생성 시 명시 필요) |
| 우선순위 | **P1** (호출자 측 검증) |

### 2-10. src/use_cases/limit_up_scanner.py — 완료 ✓

| 항목 | 값 |
|------|-----|
| L10 마이그레이션 | f97711c 완료 (LimitUpScanner.__init__ + buy_limit kwargs + 호출자 측 mode 전달) |
| 회귀 | test_limit_up_scanner_forwards_kwargs_on_buy PASS |

## 3. 분류 종합

| 분류 | 호출자 |
|------|--------|
| A. 실행 중 | (없음 — 5/28 긴급정지로 매매 cron 모두 정지) |
| B. cron 등록 (정지) | owner_rule_monitor / auto_buy_executor / smart_entry / adaptive_trend_exit / adaptive_position_manager |
| B. (--dry-run sub) | sell_monitor |
| C. 수동 명령 | telegram_command_handler (24/7) |
| D. 호출자 추적 | smart_sell / live_trading |
| 완료 | limit_up_scanner (f97711c) |

## 4. Phase 1 paper cron 재가동 전 필수 P0 (5건)

| # | 호출자 | 이유 |
|---|--------|------|
| 1 | owner_rule_monitor.py | 5/27 사고 직접 원인 + Phase 1 자동 청산 |
| 2 | telegram_command_handler.py | 24/7 활성, 사용자 매매 입력 |
| 3 | auto_buy_executor.py | 14:00~14:55 자동 매수 |
| 4 | smart_entry.py | BAT-E 08:50 자동 매수 |
| 5 | adaptive_trend_exit / adaptive_position_manager | run_adaptive_cycle 매도 경로 |

## 5. P1 (재가동 후 별도) + P0 승격 조건

| 호출자 | 현재 | P0 승격 조건 (코덱스 5/28 17:13 보정) |
|--------|------|--------------------------------------|
| sell_monitor.py | P1 (--dry-run 강제) | **수동 실행 또는 인자 누락으로 실매도 가능 시 즉시 P0** |
| smart_sell.py | P1 (호출자 미확정) | **active entrypoint 발견 시 즉시 P0** |
| live_trading.py | P1 (main.py stock_buy/sell 추정) | **active entrypoint 확인 시 즉시 P0** |
| split_order.py 호출자 측 | P1 | (SplitOrderExecutor 자체는 b76dc5e 완료) |

## 6. 정책 리스크 (코덱스 6번)

KisOrderAdapter backward compat — mode/executor_bot None 시 L10 미강제.

옵션 A) backward compat 폐지 (mode 필수화) — 모든 호출자 완료 후
옵션 B) 환경변수 `ORDER_INTENTS_GATE_STRICT=1` 시 backward compat 거부

권장: A. 단 모든 호출자 마이그레이션 후 적용. 현재 단계 적용 X (코덱스 결단 대기).

## 7. 다음 step (사용자 + 코덱스 결단)

1. P0 5건 마이그레이션 (owner_rule + telegram + auto_buy + smart_entry + adaptive_*)
2. P0 회귀 (fake broker kwargs 검증)
3. Phase 1 paper cron 재가동 검수 의뢰
4. paper 리허설 일정 결단
5. Phase 2 (live, 6/9+) 별도

## 8. 표현 금지 준수

- "전체 주문 경로 L10 완료" 사용 X
- "Phase 1 paper 재가동 가능" 사용 X
- 사용 가능: "run_adaptive_cycle paper 경로 L10 통합 완료 (해당 범위 한정)"
- 사용 가능: "잔여 호출자 5건 마이그레이션 필요"
