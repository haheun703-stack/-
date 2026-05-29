# P1 Truth Pack — 5/29(금) 운영 재가동 심사 준비

> **상태**: 정적 검수 (코드 수정 0건 / cron 0건 / systemctl 0건)
> **HEAD (로컬)**: 35718cb
> **HEAD (VPS)**: db5d731 (docs 1건 차이, 매매 영향 0)
> **검수 일시**: 2026-05-29 09:30~10:00 KST
> **검수자**: 메인 AI (security-architect 미수행 — 정적 grep+read 기반)
> **계기**: 5/28 사장님 결정문 P1 4건 blocker → 5/29 재가동 심사 1단계

## 0. 한 줄 결론

> **KILL_SWITCH 가드는 작동한다 (1차 안전망 OK). 그러나 `sell_monitor` + `smart_sell` + `live_trading` 3 호출자는 `mode`/`executor_bot` 미전달로 `order_intents_gate` 10중 가드를 우회 가능하다. 재가동 전 마이그레이션 의무.**

---

## 1. 실행 경로 (각 진입점 → 어댑터 → KIS)

### 1-1. 매매 진입점 5종

| # | 진입점 | 1차 가드 (assert_runtime) | 2차 가드 (order_intents_gate) | 어댑터 |
|---|---|---|---|---|
| 1 | `src/telegram_command_handler.py` (buy/sell/liquidate) | ✅ L512/L562/L593 | ⚠️ default `"live"` (L541/L573/L608) | KisOrderAdapter |
| 2 | `src/strategies/chart_hero_executor.py` | ✅ (adapter 내부) | ✅ L293/L444/L484 명시 | KisOrderAdapter |
| 3 | `scripts/owner_rule_monitor.py` (5/28 P0-B fix) | ✅ (adapter 내부) | ✅ L290-291 `mode=owner_mode, executor_bot="quant"` | KisOrderAdapter |
| 4 | `src/use_cases/adaptive_stop_loss.py` (5/28 P0-2 fix) | ✅ | ✅ L173 | adapter 주입형 |
| 5 | `src/use_cases/adaptive_quick_profit.py` (5/28 P0-2 fix) | ✅ | ✅ L232 | adapter 주입형 |

### 1-2. 미마이그레이션 호출자 3종 ★ 위험

| # | 진입점 | 1차 가드 | 2차 가드 | 비고 |
|---|---|---|---|---|
| 6 | `scripts/sell_monitor.py` | ✅ L36 import + L46 KILL_SWITCH 직접 체크 | ❌ **L305 `sell_limit(ticker, limit_price, qty)`**, L326 `sell_market(ticker, qty)` | mode/executor_bot 미전달 |
| 7 | `src/use_cases/smart_sell.py` | ✅ (adapter 내부) | ❌ **L91/L130/L227 `self.order.sell_*(ticker, ...)`** | mode/executor_bot 미전달 |
| 8 | `src/use_cases/live_trading.py` | ✅ (adapter 내부) | ❌ **L201/L203/L267/L304/L307/L312 `order_port.*`** | mode/executor_bot 미전달 |

---

## 2. 증거 로그 (라인 인용)

### 2-1. 1차 가드 — KILL_SWITCH 작동 확인 (`src/utils/trade_runtime_safety.py`)

```python
KILL_SWITCH_PATHS = (
    PROJECT_ROOT / "data" / "KILL_SWITCH",          # L14
    PROJECT_ROOT / "data" / "kill_switch.flag",     # L15
)

def runtime_order_block_reasons() -> list[str]:
    for name in (
        "QUANT_AUTO_TRADE_DISABLED", "AUTO_TRADE_DISABLED",
        "AUTO_TRADING_DISABLED", "PAPER_ONLY", "QUANT_PAPER_ONLY",
    ):
        if _env_true(name):
            reasons.append(f"{name}=true")          # L37
    for path in KILL_SWITCH_PATHS:
        if path.exists():
            reasons.append(f"{path.name} exists")   # L41

def assert_runtime_orders_allowed() -> None:
    reasons = runtime_order_block_reasons()
    if reasons:
        raise PermissionError(...)                  # L54
```

**결론**: KILL_SWITCH 파일 또는 5개 env var 중 하나라도 true면 `PermissionError` raise → 모든 텔레그램/cron 매매 차단.

### 2-2. 2차 가드 — KisOrderAdapter `_guard` 이중 (`src/adapters/kis_order_adapter.py`)

```python
# L138: 1차 호출
assert_runtime_orders_allowed()

# L142-147: backward compat 분기 ★ 위험 지점
if mode is not None and executor_bot is not None:
    from src.use_cases.order_intents_gate import assert_order_intent_exists
    assert_order_intent_exists(
        ticker=ticker, side=side, mode=mode, executor_bot=executor_bot,
    )
# else: backward compat — order_intents_gate 우회
```

**결론**: mode/executor_bot **둘 다** 명시되어야 order_intents_gate 통과 강제. 둘 중 하나라도 None이면 1차 가드만 적용.

### 2-3. 미마이그레이션 3 호출자 — 실측 증거

```python
# scripts/sell_monitor.py
L305: order = _adapter.sell_limit(ticker, limit_price, qty)          # mode/executor_bot 없음
L326: order = _adapter.sell_market(ticker, qty)                       # mode/executor_bot 없음

# src/use_cases/smart_sell.py
L91:  result = self.order.sell_market(ticker, qty)                    # 없음
L130: order = self.order.sell_limit(ticker, limit_price, qty)         # 없음
L227: order = self.order.sell_limit(ticker, limit_price, qty)         # 없음

# src/use_cases/live_trading.py
L201: self.order_port.buy_limit(ticker, order_price, shares)          # 없음
L267: self.order_port.cancel(order)                                   # 없음
L304: self.order_port.sell_market(ticker, quantity)                   # 없음
L307: self.order_port.sell_limit(ticker, sell_price, quantity)        # 없음
```

### 2-4. TELEGRAM_TRADING_MODE 기본값 `"live"` (5/28 paper 정책 잠재 위반)

```python
# src/telegram_command_handler.py
L541: tg_mode = os.getenv("TELEGRAM_TRADING_MODE", "live")  # _execute_buy
L573: tg_mode = os.getenv("TELEGRAM_TRADING_MODE", "live")  # _execute_sell
L608: tg_mode = os.getenv("TELEGRAM_TRADING_MODE", "live")  # _execute_liquidate
```

**결론**: env var 미설정 시 텔레그램 매수/매도/청산이 `mode="live"` intent 등록 시도. paper 정책 위반 가능성.

---

## 3. 차단 여부 매트릭스

| 경로 | KILL_SWITCH 파일 | env var (QUANT_AUTO_TRADE_DISABLED 등) | order_intents_gate | HMAC 검증 | 종합 |
|---|---|---|---|---|---|
| telegram /매수 | ✅ | ✅ | ⚠️ default "live" | ✅ | **3.5/4** |
| telegram /매도 | ✅ | ✅ | ⚠️ default "live" | ✅ | **3.5/4** |
| telegram /청산 | ✅ | ✅ | ⚠️ default "live" | ✅ | **3.5/4** |
| chart_hero_executor | ✅ | ✅ | ✅ | ✅ | **4/4** |
| owner_rule_monitor | ✅ | ✅ | ✅ | ✅ | **4/4** |
| adaptive_stop_loss | ✅ | ✅ | ✅ | ✅ | **4/4** |
| adaptive_quick_profit | ✅ | ✅ | ✅ | ✅ | **4/4** |
| **sell_monitor** | ✅ | ✅ | ❌ | ❌ | **2/4 ★** |
| **smart_sell** | ✅ | ✅ | ❌ | ❌ | **2/4 ★** |
| **live_trading** | ✅ | ✅ | ❌ | ❌ | **2/4 ★** |

---

## 4. 남은 위험 (P1 4건 + 신규 발견)

### 4-1. P1-1 Telegram KILL_SWITCH gap — **부분 해소**
- ✅ `assert_runtime_orders_allowed()` 3 경로 모두 적용 (L512/L562/L593)
- ⚠️ `TELEGRAM_TRADING_MODE` default `"live"` 3곳 — env var 미설정 시 live intent 등록

### 4-2. P1-2 §9 4단계 dry-run — **미수행**
- 별도 단계 (이번 검수 범위 밖)

### 4-3. P1-3 호출자 Runtime Truth — **미해소 ★**
- sell_monitor / smart_sell / live_trading 3종 `mode`/`executor_bot` 미전달
- backward compat 분기로 `order_intents_gate` 10중 가드 우회
- 1차 가드 (KILL_SWITCH + env var)만 작동

### 4-4. P1-4 filelock + HMAC rotation — **부분 해소**
- ✅ `register_intent` append-only (L421 `"a"` mode)
- ✅ HMAC 서명 작동 (32+ 문자 강제, IntentSignatureError raise)
- ❌ filelock 부재 (Linux append는 보통 atomic이나 정책 명시 필요)
- ❌ HMAC key rotation playbook 부재 (.env 노출 시 위조 가능)

### 4-5. 신규 발견 (이번 검수)
- ⚠️ `TELEGRAM_TRADING_MODE` default `"live"` (위 4-1)
- ⚠️ `order_intents_gate` 참조 파일 24개 — 호출자 광범위, 마이그레이션 범위 재산정 필요

---

## 5. 승인 필요 작업 (사장님 명시 지시 후만)

### 우선순위 A (재가동 전 의무)
1. **sell_monitor.py** `mode`/`executor_bot` 인자 마이그레이션 (L305, L326)
2. **live_trading.py** `mode`/`executor_bot` 인자 마이그레이션 (L201/L203/L267/L304/L307/L312)
3. **smart_sell.py** `mode`/`executor_bot` 인자 마이그레이션 (L91/L130/L227)
4. **TELEGRAM_TRADING_MODE** default `"live"` → `"paper"` 또는 default 제거 (L541/L573/L608)

### 우선순위 B (재가동 전 권장)
5. **filelock 정책 문서화** — Linux append atomicity 근거 + 미적용 결단 / portalocker 도입 결단
6. **HMAC rotation playbook 문서화** — 키 갱신 절차 + .env 노출 시 대응

### 우선순위 C (별도 단계)
7. **§9 4단계 dry-run** (P1-2)
8. **bkit:code-analyzer 검수** (어제 미수행)
9. **bkit:security-architect 검수** (이 산출물 기반)

---

## 6. 적용 금지 (재가동 심사 통과 전까지)

- ❌ paper cron 재가동 X
- ❌ quantum-scheduler unmask/start X
- ❌ 매매 cron 6개 주석 해제 X
- ❌ live 어댑터 호출 코드 추가 X
- ❌ backward compat 폐지 X (단계적 마이그레이션 후 별도 결단)
- ❌ MEMORY.md 갱신 X (사장님 명시 지시 후만)

---

## 7. 다음 단계 권장

1. 이 산출물 사장님 확인 → 우선순위 A 4건 마이그레이션 결단
2. 결단 후 bkit:security-architect + code-analyzer 분업 검수 (어제 공백 보충)
3. 검수 통과 시 재가동 심사서 작성 (§9 4단계 + one-line crontab)
4. Codex 외부 검증
5. 사장님 최종 승인 → 제한적 paper cron만 가동

---

## 8. 연결 문서
- `docs/02-design/runtime-truth-pack-remaining-callers-5_28.md` (어제 1차)
- `docs/02-design/quant-runtime-truth-pack.md` (5/28 원본)
- `docs/02-design/paper-cron-mode-policy-5_28.md` (5/28 정책)
- `docs/02-design/security/phase1-comprehensive-review-5_28.md` (5/28 종합 검수)
- `memory/decision_5_28_p1_blockers.md` (사장님 결정문)
