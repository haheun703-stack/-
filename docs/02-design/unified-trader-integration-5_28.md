# 통합 매매봇 설계 (5/28 긴급정지 후속)

> **상태**: DRAFT (메인 AI 초안, 코덱스 검수 대기)
> **연결 inbox**: `ops/codex_inbox/20260528T085000_quant-bot_P0-emergency-stop-and-integration.json`
> **계기**: 5/27 09:55 원익IPS 1주 자동매도 (-5.12%, -6,200원) — 5개 진입점 분산 + 안전망 우회

## 1. 문제 정의

### 1-1. 5개 진입점 분산 (단타봇 8개와 동일 패턴)

| # | 스크립트 | cron | 모드 인자 | 호출 경로 |
|---|---------|------|----------|----------|
| 1 | `owner_rule_monitor.py` | `*/5 9-15` | mode=REAL (코드 내장) | `create_market_sell_order` (line 80) |
| 2 | `chart_hero_morning_monitor.py` | `30 9` | `--real --max-qty 1` | KIS 어댑터 직접 |
| 3 | `chart_hero_close_cycle.py` | `55 14` | `--real --max-qty 1` | KIS 어댑터 직접 |
| 4 | `auto_buy_executor.py` | `*/5 14` | 자체 매수 실행 | KIS 어댑터 직접 |
| 5 | `run_adaptive_cycle.py` | `*/5 9-15` | `--real` | adaptive_position_manager |

### 1-2. 안전망 우회 메커니즘

5겹 안전망이 owner_rule_monitor를 **하나도** 막지 못함:

```
[.env AUTO_TRADING_ENABLED=0] ← cron --real 인자가 덮어씀 (의심, 코덱스 검증 필요)
[AUTO_TRADE_5_20=false]        ← 이름만 5/20 가동용, 무관
[KILL_SWITCH 파일]             ← chart_hero 매수만 차단, 매도/owner_rule 통과
[service mask /dev/null]       ← BAT만 차단, cron 직접 실행 차단 X
[quant_preflight.py PASS]      ← cron 실행 경로에 호출 없음
```

## 2. 통합 설계 목표

1. **단일 진입점**: `scripts/unified_trader.py` 1개만 KIS 주문 호출
2. **단일 가드 통과 후만 실거래**: 5겹 안전망 **모두 PASS** 필수
3. **mode 인자 폐기**: `.env`만 신뢰. cron `--real` 인자 제거
4. **기존 5개 LOCK**: `scripts/archive/legacy_traders/`로 이동, import 금지
5. **클린 아키텍처 준수**: entities → use_cases → adapters → agents

## 3. 통합 아키텍처

```
[cron] → scripts/unified_trader.py (단일 진입점)
           ↓
         [Phase 1: Pre-flight Guard] ★ 모두 PASS만 통과
           - quant_preflight.py 호출 (필수)
           - trade_runtime_safety.py 호출 (필수)
           - .env AUTO_TRADING_ENABLED=1 (or PAPER_ONLY=1)
           - data/KILL_SWITCH 부재
           - 시장 게이트 (market_regime, MarketScanner)
           ↓
         [Phase 2: Strategy Router] (use_cases)
           - chart_hero (D0 picker / D+1 morning / D+1 close)
           - owner_rule (SELL_STOP_LOSS / SELL_TAKE_PROFIT)
           - auto_buy (시간대별 매수)
           - adaptive_cycle (split / reentry / quick_profit / time_exit)
           - counter_trade (역매수 신호만)
           ↓
         [Phase 3: Order Adapter] (adapters)
           - paper_order_adapter (PAPER_ONLY=1)
           - kis_order_adapter (실거래, 단일 진입점)
           ↓
         [Phase 4: Logger + Audit]
           - structured_decision_log
           - signal_snapshot (5/25 P1-2 미저장 사고 재발 방지)
           - 텔레그램 통지
```

## 4. crontab 재설계 (예시)

기존 5개 분산 → 1개 통합:

```bash
# 5분 단위 통합 매매봇 (장중 09:00~15:30)
*/5 9-15 * * 1-5 cd /home/ubuntu/quantum-master && \
  PYTHONPATH=/home/ubuntu/quantum-master \
  ./venv/bin/python3.11 -u -X utf8 scripts/unified_trader.py \
  >> /home/ubuntu/quantum-master/logs/unified_trader.log 2>&1
```

`unified_trader.py` 내부에서 시간대별 분기:
- 09:00~09:25: pre-market check + owner_rule (어제 종가 -3% 손절 체크)
- 09:30: chart_hero morning monitor
- 09:30~14:55: adaptive_cycle (split / reentry / quick_profit)
- 14:00~14:55: auto_buy
- 14:55: chart_hero close
- 15:00~15:30: 종가 정리

## 5. 메인 AI 자체 오판 인정

| 오판 | 사실 |
|------|------|
| "자동매매 OFF (mask + .env=0)" | service mask + .env=0이지만 cron 직접 실행 5개 활성 |
| "수동 paper만" | crontab 출력에 5개 진입점 명시되어 있었는데도 잘못 보고 |
| "counter_trade가 매도 주범 강함" | 코덱스 검수: counter_trade는 매도 호출 0건 |

**근본 원인**: 메인 AI가 자기가 만든 코드의 위치/동작/진입점을 파악하지 못함. 단타봇 8개 분산 패턴과 동일.

## 6. 코덱스 검수 요청 사항

1. 본 설계의 안전성 검증 (5겹 가드 누락 여부)
2. 5개 진입점 각각의 안전망 체크 로직 라인 단위 검수
3. owner_rule_monitor가 어떤 안전망을 우회했는지 추적 (line 80 create_market_sell_order 호출 경로 + 가드 호출 여부)
4. mode=REAL이 .env를 어떻게 덮어쓰는지 (cron 인자 vs 코드 내장 vs 환경변수)
5. 통합 시 백테스트 데이터 호환성 (메모리: 5/22 백테스트 결과 D+1 +20.60% 보존 필요)
6. 단타봇 8개 통합 패턴 참조 가능 여부 (어제 5/27 진행한 통합 작업)

## 7. 결단 보류 사항 (사용자 승인 후)

- 통합 매매봇 가동 시점
- 단계적 paper → real 전환 일정
- 기존 5개 archive 이동 시점 (안전 확인 후)
- KIS 잔고 실측 (원익IPS 매도 체결 내역 + 현 잔고)

## 8. 즉시 확인 필요한 잔여 위험

- 어제(5/27) 매도 외 추가 자동 거래 발생 여부 (코덱스 단타봇 forensics는 클린이지만 퀀트봇 측 KIS 거래내역 미실측)
- 5/27 이전 일자에도 동일 패턴 발생 가능성 (5/20~5/26 owner_rule_monitor 로그 전수 검증)
- VPS .env가 로컬과 다를 가능성 (오늘 확인 결과 동일하지만 다른 환경변수는 미확인)

---

## 9. 5/28 메인 AI 직접 검수 결과 (3개 서브에이전트 병렬)

### 9-1. code-analyzer 검수 (`bkit:code-analyzer`)

- **BAT-I 안심 확인**: `vwap_monitor.py` + `intraday_eye.py` 매매 호출 0건 (조회/알림 전용). 정지 결단 보류.
- **5/27 사고 진짜 원인 확정**: `owner_rule_monitor.py line 84` raw `broker.create_market_sell_order`가 `KisOrderAdapter._guard()` 9중 가드 전부 우회. 코덱스 의심 "cron --real이 .env 덮어씀"은 사실과 다름 (owner_rule_monitor에 --real 인자 자체 없음). **진짜 원인은 `AUTO_TRADING_ENABLED`를 owner_rule_monitor가 읽지 않음**.
- **추가 누락 발견**: `kis_order_adapter.py line 321 cancel 함수`에 `assert_runtime_orders_allowed()` 없음.
- **`_KisLite`/`KisOrderAdapter` broker 객체 public 노출 위험**: 향후 한 줄 추가만으로 매매 가능 → read-only wrapper 권장.

### 9-2. security-architect 검수 (`bkit:security-architect`)

- 권장안 문서: `docs/02-design/security/trading-safety-architecture-review.md`
- **SPOF 4건 식별**:
  1. 어댑터 `broker` 속성 public 노출 → mojito raw 누구나 접근
  2. `.env AUTO_TRADING_ENABLED`를 진입점이 안 읽으면 무력화
  3. `KILL_SWITCH` 파일 기반 가드 (TOCTOU race + 외부 삭제 + 파일시스템 오류)
  4. KIS 키 평문 저장 + 키 보유 = 매매 권한 (분리 안 됨)
- **권장 아키텍처**:
  - `TradeAuthority` 모듈 (Capability-based): 60초 수명 HMAC 서명 Permit 발급 → 모든 broker 호출이 Permit 검증 필수
  - mojito broker private화 (`__broker` name mangling) — public 노출 금지
  - 메타클래스 + 데코레이터 강제: 클래스 정의 시점에 `@requires_order_permit` 누락 시 ImportError
  - KisOrderAdapter Fail-Closed: `AUTO_TRADING_ENABLED != "1"`이면 `__init__` 자체 거부 → mojito 객체가 메모리에 존재조차 못 함
- **`.env` 단일 신뢰 원천 정책**: CLI `--real`/`--paper` 인자 완전 폐지
- **KISA/금융보안 관점**: TradeAuthority 토큰 vs KIS 키 거래 인증 분리 / 이중 통제 (임계금액) / 감사 로그 5년 보존 / 이상 거래 탐지 / `.env` 평문 → AWS Secrets Manager

### 9-3. Explore 전수 탐색

- **22개 매매 호출 파일 + 10개+ 엔드포인트** 식별
- **의존성 체인**: 사용자 입력 → scripts → use_cases → adapters → `trade_runtime_safety.assert_runtime_orders_allowed()` → `KisOrderAdapter._guard()` 9중 → mojito → KIS API
- **핵심 차단점이 `assert_runtime_orders_allowed()` + `_guard()` 2단계인데 raw broker 호출이 둘 다 우회 가능** → SPOF
- **텔레그램 24/7 능동 매매 입력**: `_execute_buy/_sell/_liquidate` (chat_id 화이트리스트 있음, KisOrderAdapter._guard 통과)

## 10. P0-A/B/C 적용 완료 (5/28 메인 AI, 코덱스 사후 검수 의무)

| # | 변경 | 파일 | 라인 |
|---|------|------|------|
| **P0-A** | `cancel()` 함수에 `assert_runtime_orders_allowed()` 1줄 추가 | `src/adapters/kis_order_adapter.py` | line 323 (신규) |
| **P0-B** | owner_rule_monitor의 raw broker 호출 → `KisOrderAdapter.sell_market()` 위임 (5/27 사고 근본 fix) | `scripts/owner_rule_monitor.py` | line 80~100 |
| **P0-C** | `_execute_buy/_sell/_liquidate` 첫 줄에 `assert_runtime_orders_allowed()` 명시 호출 (defense in depth) | `src/telegram_command_handler.py` | line 505~ / 549~ / 575~ |

**검수 의무**: 본 변경은 메인 AI 단독 commit 금지 (5/27 코덱스 협업 결단). 코덱스 outbox 검수 결과 수신 후 commit + VPS 반영.

## 11. 추가 통합 설계 사항 (next steps)

- 22개 매매 호출 파일을 단일 `unified_trader.py`로 통합 시 의존성 체인 보존
- `TradeAuthority` 모듈 신설 (security-architect 권장 그대로 적용)
- `quant_preflight.py`에 cron/run_bat.sh 인자 검사 추가 (영구 규칙화 — `--live`, `--real`, `--force`, `--no-dry-run` 등장 시 BLOCK)
- `KisOrderAdapter.__init__` Fail-Closed 전환 (`AUTO_TRADING_ENABLED != "1"`이면 mojito 객체 생성 자체 거부)
- 통합 가동 일정 (paper 1주 → real 검증 → 정식) 사용자 결단
