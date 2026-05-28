# Phase 1 Paper 호출처 마이그레이션 추적표 (5/28)

> **상태**: Phase 1 진행 중 (코덱스 5차 PASS 직후)
> **연결**: docs/02-design/order-intents-migration-plan.md
> **원칙**: "속도보다 추적표가 더 중요. 1건씩 남아야 함." (코덱스 5차 응답)

## 1. 추적표 v1

| # | 파일 | 함수 | 라인 | 호출 메서드 | side | mode | executor_bot | register_intent 위치 | assert 통과 | 테스트명 |
|---|------|------|------|------------|------|------|--------------|----------------------|-------------|---------|
| 1 | `scripts/paper_warmup_daily.py` | `cmd_paper_trade_open` | 297~ | `PaperOrderAdapter.buy_limit` | BUY | paper | quant | 같은 함수 내 직전 (intent dict + register_intent) | `PaperOrderAdapter.buy_limit` 내부에서 자동 호출 (P0-2 검증) | `test_phase1_paper_trade.py::test_paper_trade_open_*` (6 tests) |
| 2 | `scripts/paper_warmup_daily.py` | `cmd_paper_trade_close` | 405~ | `PaperOrderAdapter.sell_limit` | SELL | paper | quant | open filled records 순회 + 같은 함수 내 직전 | `PaperOrderAdapter.sell_limit` 내부 자동 호출 | `test_phase1_paper_trade.py::test_paper_trade_close_*` (7 tests) |

## 2. 마이그레이션 패턴

```python
# Step 1: intent 생성 (timezone-aware, Asia/Seoul)
intent = {
    "intent_id": f"q_{ticker}_{datetime.now(tz=seoul).strftime('%Y%m%d%H%M%S')}",
    "bot": "quant", "engine": "phase1_paper_warmup",
    "ticker": ticker, "side": "BUY", "mode": "paper",
    "score": 80.0, "created_at": now_kst.isoformat(),
    "expires_at": (now_kst + timedelta(hours=8)).isoformat(),
}

# Step 2: register_intent (HMAC 서명 자동 추가)
register_intent(intent, bot="quant")

# Step 3: PaperOrderAdapter 호출 (mode + executor_bot 명시)
order = paper_adapter.buy_limit(
    ticker=ticker, price=entry, quantity=qty,
    mode="paper", executor_bot="quant",
)
# 내부에서 assert_order_intent_exists 자동 호출 → P0-2 + P0-3 검증 통과
```

## 3. 가드 체크리스트 (각 호출처마다)

- [x] `register_intent` 호출 → HMAC 서명 추가
- [x] intent.bot = executor_bot (quant↔quant 매치)
- [x] intent.mode = "paper" (P0-2 PaperOrderAdapter만 통과)
- [x] timezone-aware expires_at (Asia/Seoul +09:00)
- [x] PaperOrderAdapter.buy_limit 호출 시 mode + executor_bot 명시
- [x] NoIntentError 발생 시 records에 status="intent_blocked" 기록 (추적성)

## 4. 회귀 (Phase 1 마이그레이션 후)

| 회귀 | 결과 |
|------|------|
| 기존 57/57 PASS | 유지 |
| Phase 1 신규 테스트 | 작성 진행 |

## 5. 결과 파일 (실행 후)

- `data/phase1_paper_trades/{YYYYMMDD}_open.json`
  - records: 종목별 status (filled / intent_blocked / register_failed / qty_zero / fetch_failed)
  - n_registered, n_filled, n_blocked 카운터

## 6. cron 등록 (선택)

```bash
# 09:15 paper trade open (Phase 1, mode='paper')
20 9 * * 1-5 cd /home/ubuntu/quantum-master && \
  PYTHONPATH=. ./venv/bin/python3.11 \
  scripts/paper_warmup_daily.py --paper-trade-open --top 9 \
  >> /tmp/phase1_paper_trade.log 2>&1
```

**주의**: cron 등록은 코덱스 6차 검수 통과 후 진행. 현재는 수동 트리거만.

## 7. 다음 호출처 (Phase 1 잔여)

| # | 후보 | 비고 |
|---|------|------|
| 2 | `cmd_paper_trade_close` (sell) | 신규 작성 필요 (BUY → SELL 페어) |
| 3 | `chart_hero_picker_cycle.py` paper 모드 | cron 정지 중, 별도 paper 가동 |
| 4 | `chart_hero_morning_monitor.py` paper 모드 | 동일 |
| 5 | `chart_hero_close_cycle.py` paper 모드 | 동일 |

각 호출처마다 본 추적표 v1과 동일한 row 추가 후 회귀 테스트 통과 의무.

## 8. live 호출처 (금지 — 코덱스 6차+ 승인 후)

- `scripts/owner_rule_monitor.py`
- `scripts/sell_monitor.py`
- `scripts/smart_entry_runner.py`
- `scripts/auto_buy_executor.py`
- `src/use_cases/adaptive_*.py`
- `src/telegram_command_handler.py`

→ Phase 2 이후 (5/30~6/1 예정), 코덱스 별도 승인 필수.

## 9. 코덱스 검수 요청

1. 추적표 v1 형식 적절성 (필드 누락 여부)
2. paper_warmup_daily.cmd_paper_trade_open 구현의 안전성
3. Phase 1 잔여 호출처 우선순위
4. cron 등록 시점 (코덱스 6차 검수 후?)
