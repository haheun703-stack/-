# 자동매매 안전망 보안 아키텍처 리뷰 (5/27 사고 후속)

> **분류**: 보안 아키텍처 권장안 (코드 수정 없음, 설계 문서만)
> **계기**: 2026-05-27 09:55:11 `owner_rule_monitor.py`가 5겹 안전망을 모두 우회하여
> 원익IPS 1주 시장가 매도 (-6,200원 손실)
> **검수 관점**: OWASP 2021 + KISA 금융보안 + Defense-in-Depth
> **상태**: 메인 AI 단독 결정 금지. 코덱스 외부 검증 후 적용.

---

## 1. 사건 요약 (보안 인시던트 분류)

| 항목 | 값 |
|------|-----|
| 인시던트 ID | INC-2026-05-27-001 |
| 분류 | A01 Broken Access Control (OWASP) + A04 Insecure Design |
| 심각도 | **Critical** (자금 손실 + 안전망 우회 + 재발 가능) |
| 영향 | 사용자 의도와 다른 실거래 발생 (계좌 47339014) |
| 탐지 | 사용자가 KIS 잔고 변경을 사후 발견 |
| 평균 탐지 시간 (MTTD) | 수 시간 (실시간 알람 없음) — A09 Logging Failure |

---

## 2. 권한 경계 매트릭스 (진입점 × 안전망)

각 매매 진입점이 호출 시 어떤 안전망을 통과하는지 코드 라인 기준 정리.

### 2-1. 매매 진입점 5개

| # | 진입점 (cron 직접 실행) | KIS 호출 경로 | 모드 결정 |
|---|----------------------|---------------|----------|
| 1 | `scripts/owner_rule_monitor.py` | `broker.create_market_sell_order` (line 84) — **mojito raw 직접 호출** | 코드 내장 (CLI 인자 없음) |
| 2 | `scripts/chart_hero_morning_monitor.py` | `KisOrderAdapter` 어댑터 | `--real` / `--paper` CLI |
| 3 | `scripts/chart_hero_close_cycle.py` | `KisOrderAdapter` 어댑터 | `--real` / `--paper` CLI |
| 4 | `scripts/chart_hero_picker_cycle.py` | (picker만, 주문은 다른 사이클) | `--real` / `--paper` CLI |
| 5 | `scripts/auto_buy_executor.py` | `KisOrderAdapter` 어댑터 | dry-run 분기 (CLI) |
| 6 | `scripts/run_adaptive_cycle.py` | `KisOrderAdapter` + `adaptive_position_manager` | `--real` / `--paper` CLI + `validate_real_mode_gate` |

### 2-2. 안전망 5개

| 안전망 | 구현 위치 | 작동 방식 |
|-------|----------|----------|
| L1: `.env AUTO_TRADING_ENABLED` | `kis_order_adapter._guard` line 111 | `os.getenv != "1"` → `PermissionError` |
| L2: `AUTO_TRADE_5_20` | `owner_rule_monitor.main` line 107 + `auto_buy_executor` line 270 | `os.environ != "true"` → `return 0` (스킵) |
| L3: `data/KILL_SWITCH` 파일 | `trade_runtime_safety.runtime_order_block_reasons` line 39 + 진입점별 산발 | 파일 존재 시 차단 |
| L4: `quantum-scheduler` service mask | systemd `/dev/null` 마스크 | BAT만 차단 — cron 직접 실행에 무효 |
| L5: `quant_preflight.py` | `tools/quant_preflight.py` | 사람이 명시적으로 호출해야만 실행 — cron 경로에 미연결 |

### 2-3. 우회 매트릭스 (각 진입점 × 안전망)

`O` = 차단 가능, `X` = 차단 불가 (우회), `?` = 부분/조건부

| 진입점 | L1 .env=0 | L2 AUTO_TRADE_5_20 | L3 KILL_SWITCH | L4 service mask | L5 quant_preflight |
|--------|-----------|-------------------|----------------|-----------------|-------------------|
| `owner_rule_monitor` | **X** (mojito raw 직접 호출, `_guard` 미통과) | O (line 107) | **X** (`assert_runtime_orders_allowed` line 83이 호출되긴 하나 `execute_sell` 내부) | X (cron 직접) | X (호출 안 됨) |
| `chart_hero_morning_monitor` | O (어댑터 통과) | X (다른 환경변수) | ? (executor line 211~213 한정) | X (cron 직접) | X (호출 안 됨) |
| `chart_hero_close_cycle` | O (어댑터 통과) | X | ? | X | X |
| `auto_buy_executor` | O (어댑터 통과) | O (line 270 첫 체크) | O (line 261) | X | X |
| `run_adaptive_cycle` | O (어댑터 + `validate_real_mode_gate`) | X | ? | X | X |

### 2-4. 5/27 사고의 본질적 원인

`owner_rule_monitor.py` line 84:
```python
resp = broker.create_market_sell_order(symbol=ticker, quantity=qty)
```

- `broker`는 `KisStockDataAdapter().broker` (line 120~122) — **mojito 원시 객체**
- `KisOrderAdapter._guard()`를 거치지 않음 → **L1 (AUTO_TRADING_ENABLED) 우회**
- line 83에서 `assert_runtime_orders_allowed()` 호출은 있으나 → L3만 부분 차단
- L2 (`AUTO_TRADE_5_20`)는 line 107에서 체크되나 — 사고 당시 `true`였거나, **이 변수의 의미가 "5/20용"인데 5/27까지도 살아있음 = 시간 기반 가드 미작동**

**보안 결론**: L1과 L3 사이에 비대칭이 존재. `kis_order_adapter._guard`에는 `AUTO_TRADING_ENABLED` 체크가 있으나 `trade_runtime_safety`에는 없음. mojito raw 호출 경로는 L3만 통과하면 매매 가능.

---

## 3. 단일 실패점 (SPOF) 식별

### SPOF-1: mojito raw 객체 노출 (Critical)

`KisStockDataAdapter().broker` 및 `KisOrderAdapter().broker` 속성을 통해 **mojito 원시 객체에 무가드 접근 가능**.

- 영향 진입점: `owner_rule_monitor` (현재), `paper_order_adapter` 내부 (간접), 향후 신규 코드 모두
- 회귀 테스트 `tests/test_no_raw_mojito_order_bypass.py` 존재 — 그러나 ALLOWLIST에 `kis_order_adapter.py`만 등록되어 있고 `owner_rule_monitor`는 `assert_runtime_orders_allowed`만 호출하면 통과되도록 설계됨
- **본질**: `broker` 속성을 public으로 노출한 어댑터 설계 자체가 SPOF

### SPOF-2: `.env` 평문 환경변수 (High)

`AUTO_TRADING_ENABLED=0`이 단일 텍스트 값. cron이 환경변수 주입 시점에 무효화 가능:
- cron의 `env -i` 또는 `cd ... && python ...` 형태에서 `.env`를 명시적으로 로드하지 않으면 셸 환경의 빈 값으로 폴백
- `owner_rule_monitor`는 line 28~30에서 `load_dotenv(PROJECT_ROOT / ".env")` 호출 — 그러나 **`AUTO_TRADING_ENABLED` 자체를 코드에서 체크하지 않음**
- 코덱스 의심: "cron 인자 `--real`이 덮어쓴다" — 검수 결과 `owner_rule_monitor`에는 `--real` 인자 자체가 없음. 진짜 원인은 **그 변수를 안 읽음**

### SPOF-3: KILL_SWITCH 파일 기반 가드 (Medium-High)

파일 존재 = 차단, 부재 = 허용. **fail-open** 패턴:
- 파일 시스템 오류 시 (디스크 풀, 권한 변경) → 부재로 인식되어 자동매매 활성
- 누군가 `rm data/KILL_SWITCH` → 즉시 매매 가능 (감사 로그 없음)
- Race condition: 진입점 A가 KILL_SWITCH 체크 후, 진입점 B가 cron으로 동시 실행 시 — 별도 프로세스 격리는 있으나 **동시 매매 차단 메커니즘 없음**

### SPOF-4: 단일 자격증명 (Critical for ATO 시나리오)

`.env`의 `KIS_APP_KEY` / `KIS_APP_SECRET` / `KIS_ACC_NO` 평문 저장. 파일 권한만이 보호 장벽:
- 백업 파일 `.env.bak.20260514_1622` 등 평문 보존 (CLAUDE.md 메모리에 명시)
- VPS `/home/ubuntu/quantum-master/.env` 같은 위치 동일 패턴 추정
- 키 회전 자동화 없음

---

## 4. OWASP 2021 매핑 + 자동매매 보안 패턴

### 4-1. 본 사건의 OWASP 분류

| OWASP | 본 사건 적용 |
|-------|------------|
| **A01 Broken Access Control** | `owner_rule_monitor`가 권한 없이 (`AUTO_TRADING_ENABLED=0`임에도) `create_market_sell_order` 호출 — 인가 결함 |
| **A04 Insecure Design** | 5겹 안전망이 OR(or-of-ANDs)로 구성되지 않고 진입점별로 다른 부분집합만 체크. 원자성 부재 |
| **A05 Security Misconfiguration** | `.env` 변수명이 진입점별로 다름 (`AUTO_TRADING_ENABLED`, `AUTO_TRADE_5_20`, `QUANT_PAPER_ONLY` 등 5개+) → 운영자가 정확한 OFF 상태 보장 불가 |
| **A07 Authentication Failures** | KIS API 키와 자동매매 활성 권한이 분리되지 않음 (키 있으면 활성). 별도 권한 토큰 부재 |
| **A08 Software Integrity Failures** | preflight 도구는 있으나 강제 호출 메커니즘 없음. CI/CD 게이트도 없음 |
| **A09 Logging Failures** | 사용자가 KIS 잔고 변경을 사후 발견. 실거래 즉시 텔레그램 알람은 있으나 **사용자가 "정상"으로 오인** — 알람 차별화 부재 |

### 4-2. 적용 권장 보안 원칙

1. **Fail-Closed (Fail-Safe Defaults)**
   - 환경변수 미정의 → **차단 (현재는 일부 OK, 일부 fail-open)**
   - 파일 시스템 오류 시 → 차단
   - 어댑터 초기화 실패 → 차단 (mojito 객체 부재 시 절대 raw 호출 불가)

2. **Defense in Depth (다층 방어)**
   - L1~L5가 진입점마다 다른 부분집합이 아니라 **모든 진입점이 동일 5층 통과 의무**
   - 어댑터 레이어가 아닌 **broker 호출 직전** 단일 게이트

3. **Least Privilege (최소 권한)**
   - paper 모드 객체는 **실제 broker에 접근 불가** (현재 `PaperOrderAdapter._get_market_data`가 `KisOrderAdapter`를 lazy import — 권한 누수 위험)
   - 매도 전용 진입점은 매수 권한 토큰 부재
   - 종목 화이트리스트 외 종목은 모듈 레벨에서 차단

4. **Separation of Duties (직무 분리)**
   - **결정** (use_cases) ↔ **주문** (adapters) ↔ **권한** (utils/auth) 명확 분리
   - `owner_rule_monitor`가 결정+권한+주문을 모두 가진 현재 구조는 위반

5. **Audit Trail (불변 감사)**
   - 모든 broker 호출은 호출 직전 append-only 로그 (현재 일부 누락)
   - 사용자에게 "실거래 발생" 텔레그램은 **다른 채널 + 다른 톤** (warning 채널)

---

## 5. 통합 가드 모듈 권장 아키텍처

### 5-1. 단일 진입 게이트 설계

```
┌──────────────────────────────────────────────────────────┐
│ 모든 매매 호출 경로 (5개 진입점)                          │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ TradeAuthority (신규 모듈, src/utils/trade_authority.py)  │
│                                                          │
│  - issue_order_permit(side, ticker, qty, mode) → Permit │
│     • 모든 L1~L5 검증 수행                                │
│     • 검증 실패 시 PermissionError (Fail-Closed)         │
│     • 통과 시 짧은 수명 (60초) Permit 객체 반환           │
│                                                          │
│  Permit 객체:                                            │
│    - permit_id (UUID, append-only 로그 키)               │
│    - expires_at (60초 후)                                │
│    - scope: BUY/SELL + ticker + qty                      │
│    - signature: HMAC(.env SECRET, fields)                │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ KisOrderAdapter (모든 broker 호출 wrapping)              │
│                                                          │
│  - 모든 메서드 첫 줄에 verify_permit(permit) 의무         │
│  - mojito broker 객체는 private (__broker)               │
│  - permit 없으면 절대 broker.create_*_order 호출 안 됨   │
└──────────────────────────────────────────────────────────┘
```

### 5-2. 핵심 발상

1. **Capability-based Security**: 권한을 호출자가 "주장"하는 게 아니라 **gate가 발급한 토큰을 제시**해야 호출 가능. 토큰 없으면 어떤 경로로도 실거래 불가.

2. **mojito broker private화**: `KisOrderAdapter.broker` 속성 → `KisOrderAdapter._KisOrderAdapter__broker` (mangling) 또는 별도 모듈로 격리. **외부에서 절대 직접 접근 불가**.

3. **Permit 만료**: 60초 짧은 수명. 발급 → 사용 사이 시간차에서 KILL_SWITCH 작동 가능.

4. **HMAC 서명**: Permit 위조 방지. `.env`에 `TRADE_AUTHORITY_SECRET` 추가.

### 5-3. 검증 체크리스트 (TradeAuthority가 강제 호출하는 L1~L7)

| Layer | 검증 항목 | 실패 시 |
|-------|----------|--------|
| L1 | `AUTO_TRADING_ENABLED == "1"` | PermissionError + 텔레그램 |
| L2 | `os.getenv("PAPER_ONLY") != "1"` (실거래만) | PermissionError |
| L3 | `data/KILL_SWITCH` 부재 | PermissionError |
| L4 | `quant_preflight.runtime_order_block_reasons()` 빈 리스트 | PermissionError |
| L5 | 거래시간 09:00~15:30 + `is_kr_trading_day()` | RuntimeError |
| L6 | `AUTO_TRADING_WHITELIST_ONLY=1` + 화이트리스트 통과 | PermissionError |
| L7 | 일일 누적 한도 (`check_daily_limits`) | ValueError |

**중요**: 모든 진입점은 `issue_order_permit`만 호출. 진입점이 L1~L7 일부만 체크하는 현재 패턴 완전 제거.

### 5-4. 회귀 방지 정적 분석 (재테스트 강화)

현재 `tests/test_no_raw_mojito_order_bypass.py`가 mojito raw 호출을 detect — 그러나 ALLOWLIST에 `kis_order_adapter.py`만 있고 owner_rule_monitor는 통과됨.

**권장 강화**:
- AST 기반 검사: `create_*_order` 호출 직전 라인에 `verify_permit(permit)` 호출 의무
- ALLOWLIST 폐지. `kis_order_adapter`도 `__broker` mangled 호출만 허용
- pre-commit hook + GitHub Actions CI 게이트

---

## 6. KILL_SWITCH의 본질적 한계

### 6-1. 파일 기반 가드의 위험

| 위험 | 시나리오 | 영향 |
|------|---------|-----|
| Race condition | 진입점 A가 line 112에서 `KILL_SWITCH.exists()` 체크 후, 사이에 cron으로 진입점 B가 실행. A는 통과 후 KILL_SWITCH가 활성화돼도 line 84 도달하기 전 검사 없음 | TOCTOU |
| 외부 삭제 | `rm data/KILL_SWITCH` 누구나 가능 (감사 없음) | 즉시 활성화 |
| 파일 시스템 오류 | 디스크 풀, NFS 끊김, 권한 변경 → `exists()` False 반환 | Fail-open으로 실거래 |
| 경로 변동 | `KILL_SWITCH_PATHS` 두 개 등록되어 있으나 (`KILL_SWITCH`, `kill_switch.flag`) — 신규 진입점이 다른 이름 사용 시 누락 | 일부만 차단 |
| 동기화 (VPS vs 로컬) | 로컬에 KILL_SWITCH 있어도 VPS에 없으면 VPS cron 활성 | 환경 불일치 |

### 6-2. 권장 대체안

1. **DB 기반 unified state**: SQLite의 `trade_state` 테이블 — `enabled` 칼럼 (BOOL) + `last_changed_by` + `last_changed_at` + `change_reason`. **트랜잭션 격리** 자동 보장.
2. **HMAC 서명된 활성 토큰**: `.env`의 `TRADE_ACTIVE_TOKEN`이 valid HMAC이어야 활성. 위조 불가.
3. **Heartbeat 기반 자동 OFF**: 사용자가 1시간마다 명시적 갱신 안 하면 자동 OFF (Heartbeat 패턴).
4. **다중 서명**: 활성화에 사용자 + 코덱스 검수 PASS 둘 다 필요 (2-key system, 핵 발사 모델).

### 6-3. 현실적 즉시 적용안 (Quick-Win)

- `KILL_SWITCH` 파일 → 내용에 timestamp + reason + signature 작성. 파일 내용 무결성 검증 추가.
- `KILL_SWITCH_ACTIVE_DEFAULT=true` 환경변수 — 파일이 **없어도** 차단 (whitelist 모드). 명시 활성화는 별도 토큰.

---

## 7. `--real` CLI 인자가 `.env`를 덮어쓰는 패턴 위험

### 7-1. 현재 패턴

```python
# chart_hero_close_cycle.py 등
mode_group.add_argument("--real", action="store_true", help="실전 모드 (KIS 주문)")
```

- cron이 `--real`로 호출하면 코드는 `args.real`만 보고 KIS 어댑터 진입
- `.env AUTO_TRADING_ENABLED=0`이어도 → 어댑터 진입 (어댑터 `_guard`가 막긴 함, 그러나 어댑터를 거치지 않는 코드 경로 존재)
- 사용자가 `.env`를 OFF로 설정해도 cron 명령줄을 못 봐서 안심하는 **인지적 함정**

### 7-2. 근본 위험성

1. **신뢰 경계 모호**: `.env`는 시스템 운영자의 의도, cron 인자는 자동화 시스템의 결정. 후자가 전자를 덮어쓰면 운영자 통제 불가
2. **감사 누락**: cron이 무엇으로 실행됐는지 추적하려면 crontab + systemd + 로그 3곳 확인 필요
3. **테스트 어려움**: dev/staging/prod 모두 같은 `.env` + cron 인자만 다르면 prod에서만 발현되는 버그

### 7-3. 권장 패턴: 단일 신뢰 원천 (Single Source of Truth)

```python
# 모든 진입점 시작부
mode = TradeAuthority.resolve_mode()  # .env만 본다, CLI 무시
if mode != "REAL":
    return paper_simulation()
# CLI에 --paper/--real 인자 자체 제거
```

원칙:
- `.env`가 **유일한** 모드 결정자
- CLI 인자는 **읽기 전용** 파라미터만 (예: `--once`, `--verbose`)
- 모드 변경 → `.env` 수정 → 명시적 git commit → 운영자가 본 흔적 남김
- crontab은 단순 invoker, 결정자가 아님

---

## 8. `AUTO_TRADING_ENABLED=0`인데 실전 어댑터가 초기화되는 모순

### 8-1. 현재 동작

`KisOrderAdapter.__init__` (line 71~84):
```python
is_mock = os.getenv("MODEL") != "REAL"
self.broker = mojito.KoreaInvestment(...)  # 무조건 초기화
logger.info("KisOrderAdapter 초기화 (모드: %s, 가드레일: %s)", ...)
```

- `AUTO_TRADING_ENABLED=0`이어도 broker 객체 (mojito)는 정상 생성
- KIS 토큰 발급 + API 연결 성립
- `_guard()`에서만 차단 — 그러나 **broker 객체가 살아있으면 다른 코드 경로에서 우회 가능**

### 8-2. Fail-Open vs Fail-Closed

| 패턴 | 현재 동작 | 위험 |
|------|---------|-----|
| 어댑터 초기화 | broker 무조건 살아있음 | mojito raw 접근 누구나 가능 |
| `_guard` 호출 | 어댑터 메서드 호출 시만 | 누락 진입점은 차단 안 됨 |
| 모드 분기 | `_is_mock` 속성만 확인 | mojito는 mock 플래그로만 분리 — bypass 가능 |

**이것이 Fail-Open**. `AUTO_TRADING_ENABLED=0`이 의도하는 바는 "실거래 절대 불가"이지만, 현재 코드는 "실거래 시도하면 막음"에 불과.

### 8-3. Fail-Closed 권장 아키텍처

```python
class KisOrderAdapter:
    def __init__(self):
        if not TradeAuthority.real_mode_authorized():
            raise PermissionError(
                "KisOrderAdapter 초기화 거부: AUTO_TRADING_ENABLED != 1. "
                "PaperOrderAdapter를 사용하세요."
            )
        # 이 라인 이후만 mojito 초기화
        self.__broker = mojito.KoreaInvestment(...)
```

- `AUTO_TRADING_ENABLED=0` → 어댑터 import 시점 또는 `__init__` 시점에 즉시 거부
- mojito 객체 자체가 메모리에 없음 → **어떤 코드 경로로도 raw 호출 불가**
- 사용자가 .env OFF인 채로 코드 변경해도, 실거래 시도 시 import 단계에서 멈춤

이게 진짜 의미의 Fail-Closed.

---

## 9. `assert_runtime_orders_allowed()` 강제 호출 메커니즘

### 9-1. 현재 한계

5/27 20:24 추가 후 다음 위치에 호출:
- `kis_order_adapter.modify` line 343
- `kis_order_adapter._guard` line 109
- `owner_rule_monitor.execute_sell` line 83
- `adaptive_position_manager` line 355
- 5개 다른 use_cases 파일

문제:
- **사람이 잊으면 누락**. 신규 코드에서 호출 안 하면 우회 가능
- 정적 분석 강제 없음 (테스트는 있으나 ALLOWLIST 우회 가능)
- 호출자가 `try/except`로 묻으면 무력화

### 9-2. 권장: 데코레이터 + 메타클래스 강제

```python
# src/utils/trade_authority.py
def requires_order_permit(func):
    """모든 broker 변형 호출 메서드에 적용. permit 없으면 raise."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        permit = kwargs.pop("permit", None)
        if not TradeAuthority.verify_permit(permit, scope=func.__name__):
            raise PermissionError(f"{func.__name__} requires valid permit")
        return func(self, *args, **kwargs)
    return wrapper

class _OrderAdapterMeta(type):
    """주문 메서드는 무조건 데코레이터 강제. 누락 시 클래스 정의 자체 실패."""
    REQUIRED_METHODS = {"buy_limit", "sell_limit", "buy_market", "sell_market", "modify", "cancel"}
    def __init__(cls, name, bases, ns):
        for method_name in cls.REQUIRED_METHODS:
            method = ns.get(method_name)
            if method and not getattr(method, "__permit_required__", False):
                raise TypeError(f"{name}.{method_name} must use @requires_order_permit")
        super().__init__(name, bases, ns)

class KisOrderAdapter(metaclass=_OrderAdapterMeta):
    @requires_order_permit
    def buy_limit(self, ...): ...
```

**효과**: 클래스 정의 시점에 강제. import만 해도 위반 시 ImportError. 우회 불가.

### 9-3. CI/CD 게이트

GitHub Actions에 정적 분석 단계:
```yaml
- name: Trade safety static check
  run: |
    python tests/test_no_raw_mojito_order_bypass.py
    python tools/check_permit_decorators.py  # 신규 — ALLOWLIST 폐지
```

PR에 broker 호출 추가 시 자동 fail. 사람의 기억에 의존하지 않음.

---

## 10. KISA / 금융보안 권장사항 (자동매매 표준)

### 10-1. KISA 전자금융감독규정 요지 (자동매매 관련)

- **거래 인증 분리**: 매매 자격 ≠ API 키 보유. 별도 매매 활성화 토큰 필요
- **이중 통제**: 임계금액 초과 거래는 2단계 인증 (사람 개입)
- **감사 로그 보존**: 모든 매매 시도/실행 5년 보존 (현재 텔레그램+로그 일부만)
- **이상 거래 탐지 (FDS)**: 패턴 이탈 시 자동 차단

### 10-2. 본 시스템 적용 권장

| KISA 원칙 | 본 시스템 매핑 |
|----------|--------------|
| 거래 인증 분리 | `TradeAuthority` 권한 토큰 vs `.env` KIS 키 분리 |
| 이중 통제 | 일일 누적 30만원 초과 → 사용자 텔레그램 응답 필요 (현재 자동) |
| 감사 로그 보존 | `data/audit/orders/YYYY-MM-DD.jsonl` append-only + S3 백업 |
| 이상 거래 탐지 | 비정상 시간대(09:55:11 등) + 비정상 종목(원익IPS, 화이트리스트 외) 자동 차단 |
| 키 관리 | `.env` 평문 → AWS Secrets Manager 또는 OS keyring 권장 |
| 변경 통제 | 자동매매 코드 PR → 코덱스 검수 PASS 필수 (이미 5/27 결단) |

### 10-3. KIS API 자체 안전망 활용

KIS API 화이트리스트 IP 등록 (13.209.153.221) — 이미 적용됨. 추가:
- KIS 일일 매매 한도 (계좌별) — 본인 계좌 한도 확인
- KIS 매매 가능 종목 제한 — 가능하면 ETF/대형주만 허용 신청
- KIS 모의투자 환경 — paper mode = mock=True로 이미 사용 중

---

## 11. 단계적 적용 로드맵 (우선순위)

### Phase 1: Critical (24시간 내, 5/28~29)

1. **KILL_SWITCH 활성화 확인**: VPS + 로컬 양쪽 `data/KILL_SWITCH` 파일 존재 및 내용 검증
2. **`.env AUTO_TRADING_ENABLED=0` 강제 + 검증**: `quant_preflight.py --expect blocked` PASS 의무
3. **owner_rule_monitor.py line 84 긴급 보강**: `assert_runtime_orders_allowed()` + `os.getenv("AUTO_TRADING_ENABLED")` 추가 체크 (코드 수정은 별도 PR + 코덱스 검수)
4. **5/20~5/27 owner_rule_monitor 로그 전수 점검**: 다른 자동매도 발생 여부 확인 (감사)

### Phase 2: High (1주 내, 6/3까지)

5. **TradeAuthority 모듈 설계 + 구현 + 코덱스 검수**
6. **KisOrderAdapter Fail-Closed 전환**: `AUTO_TRADING_ENABLED != "1"`이면 `__init__` 거부
7. **paper_order_adapter의 KisOrderAdapter lazy import 제거**: 권한 누수 차단
8. **`tests/test_no_raw_mojito_order_bypass.py` ALLOWLIST 폐지** + 메타클래스 강제

### Phase 3: Medium (1개월 내, 6월 말)

9. **DB 기반 trade_state 도입 + 파일 KILL_SWITCH 단계적 폐지**
10. **HMAC 서명된 활성 토큰 + 60초 Permit 발급 시스템**
11. **GitHub Actions CI 게이트**: 매매 관련 PR 자동 검증
12. **AWS Secrets Manager 통합**: `.env` 평문 키 폐지

### Phase 4: Low (Defense-in-Depth)

13. **이상 거래 탐지 (FDS) 룰셋**: 시간/종목/금액/빈도 이상 패턴
14. **2-key system**: 활성화에 사용자 + 코덱스 검수 PASS 둘 다 필요
15. **외부 감사 로그 백업**: S3 또는 별도 VPS

---

## 12. 메인 AI 자가 평가 (코덱스 검수 대상)

본 문서의 한계 자인:
1. **mojito 라이브러리 내부 동작 미실측**: `broker.create_market_sell_order`의 실제 호출 경로/예외 처리 미검증 → 코덱스 라이브러리 소스 확인 요청
2. **5/27 사고 당시 실제 환경변수 값 미확보**: `AUTO_TRADE_5_20`이 사고 시점에 `true`였는지 확정 불가 → VPS 로그 추출 필요
3. **메타클래스 강제 패턴은 Python 우회 가능**: monkey-patching, `__class__` 변경 등 → 정적 분석 보강 필요
4. **`TradeAuthority` Permit 객체 만료/재사용 공격 미검토**: replay attack, time skew → 코덱스 위협 모델링 요청
5. **본 권장사항은 백테스트/성능 영향 미검토**: 매 호출에 Permit 발급 시 latency 영향 → 측정 필요

본 문서는 메인 AI가 단독으로 결단할 사항이 아닙니다. **코덱스 외부 검증 + 사용자 결정** 후 적용해야 합니다.

---

## 13. 핵심 질문에 대한 직접 답변 (요약)

| 질문 | 답변 |
|------|-----|
| KILL_SWITCH 파일 가드의 본질적 한계 | TOCTOU race, 외부 삭제, 파일시스템 오류 시 fail-open. DB 기반 상태 + HMAC 토큰으로 대체 권장 |
| cron 인자가 `.env` 덮어쓰는 패턴 위험 | 신뢰 경계 모호 + 운영자 통제 무력화. `.env`만 신뢰 원천, CLI 인자 폐지 권장 |
| `AUTO_TRADING_ENABLED=0`인데 어댑터 초기화 모순 | Fail-Open 패턴. `__init__`에서 즉시 거부하는 Fail-Closed 전환 권장 |
| `assert_runtime_orders_allowed()` 강제 메커니즘 | 데코레이터 + 메타클래스로 클래스 정의 시점 강제 + CI 정적 분석 게이트 |
| 통합 가드 모듈 핵심 발상 | Capability-based: 짧은 수명 HMAC Permit 발급/검증 + mojito broker private화 |

---

> **다음 액션 (사용자 결정 대기)**:
> 1. 본 문서 코덱스 검수 의뢰 (`ops/codex_inbox/`)
> 2. Phase 1 (24시간 내) 즉시 적용 여부
> 3. `docs/02-design/unified-trader-integration-5_28.md`와 본 문서 통합 여부
