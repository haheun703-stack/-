# 퀀트봇 Deletion/Quarantine Audit 1차 — 5/29(금)

> **상태**: read-only 정적 감사 (코드 수정 0건 / 파일 삭제 0건 / 결단 대기)
> **HEAD**: 35718cb (작업 전) + 5/29 종일 미커밋 변경
> **계기**: 5/29 사장님 추가 지시 — "퀀트봇도 삭제/격리 후보가 있을 가능성 큼. 단타봇 다음으로 deletion/quarantine audit 수행"
> **범위**: 조사 대상 8 카테고리 + 5분류 (DELETE/MERGE/KEEP+GUARD/QUARANTINE/KEEP) + canonical 경로 부합 검증

---

## 0. 한 줄 결론

> **scripts/archive 152 파일 + _etf_ref 7 파일은 격리 정책 100% 준수 (외부 import 0건). 실제 운영 코드에서 캐노니컬 매매 경로 우회 가능 진입점 1차 식별: dead-letter Flask 큐 1건 + 노골적 삭제 후보 2건 + canonical 흡수 후보 4건. 단, 5/28 P0 + 5/29 P1-A + 5/29 P2 마이그레이션으로 mode/executor_bot 우회 가능 호출자 0건 확인.**

---

## 1. Canonical 경로 명세 (사장님 5/29 결단)

```
signal/decision → order_intent → order_intents_gate → adapter/executor → audit/report
```

**이 밖에서 매매/청산/긴급청산/텔레그램 매도/live adapter 호출 가능하면 전부 조사 대상**.

---

## 2. 8 카테고리 조사 결과 요약

### 2-1. 카테고리 1+5+8 — 오래된 스크립트 + masked cron + 폐기 문서 실행파일 (Explore 1)

| 영역 | 파일 수 | 외부 import | 매매 호출 | 결론 |
|---|---|---|---|---|
| scripts/archive/ 9 폴더 | 152 | **0건** | 6줄 (KisOrderAdapter, 모두 archive 내부) | 격리 정책 100% 준수 |
| _etf_ref/ | 7 | **0건** | 0줄 | 격리 정책 100% 준수 |
| 폐기 마커 13건 | 13 | (활성) | 0줄 | dashboard.py 만 진정 폐기, 나머지 12건 운영 중 |
| VPS 살아있는 cron 21 | 21 | (활성) | 2건 (paper_warmup + chart_hero_picker, gate 통과) | 모두 paper/scanner/learning |
| VPS 주석 매매 cron 5 | 5 | (정지) | 3건 (auto_buy + owner_rule + adaptive) | 정지 유지 |

→ **archive/ 폐기 정책은 작동 중**. 추가 삭제 대상 0건.

### 2-2. 카테고리 2 — 중복 매매 entrypoint (Explore 2)

22개 매매 entrypoint 매트릭스 분류:
- **canonical 완전 부합 (4건)**: auto_buy_executor / owner_rule_monitor / sell_monitor / paper_warmup_daily
- **canonical 간접 부합 (8건)**: smart_entry_runner / run_adaptive_cycle / run_limit_up_scanner / telegram / live_trading / safety_guard / chart_hero_picker_cycle / chart_hero_executor (라이브러리 호출자 경유)
- **분석/모니터링 (3건)**: counter_trade_monitor / intraday_eye / position_monitor (fetch_balance만, 매매 0)
- **1회성 (1건)**: `scripts/one_off/integration_dryrun_5_20.py` (5/20 가동 전 1회 시뮬, 완료)
- **canonical 흡수 후보 (4건)**: paper_warmup_daily / paper_trading_unified / run_limit_up_scanner / run_quant_3day_pilot
- **라이브러리 (13건)**: src/use_cases/* + src/strategies/* + src/adapters/*

### 2-3. 카테고리 3 — 긴급청산 경로 (직접 grep)

| 위치 | 라인 | 상태 |
|---|---|---|
| `main.py:611-645` cmd_emergency_stop | L632/L637 | ✅ 5/29 P2 마이그레이션 완료 (LIVE_TRADING_MODE env default paper) |
| `src/use_cases/safety_guard.py:115` emergency_liquidate | L135 | ✅ 5/29 P2 (keyword-only mode/executor_bot) |
| `src/use_cases/live_trading.py:110, 414` | L114, L416 | ✅ 5/29 P2 (Critical 봉합 완료) |
| `src/entities/trading_models.py:161` must_liquidate | 모델 속성 | (속성만, 매매 호출 X) |
| **`src/etf/index_engine.py:107` _close_all** | (의사결정만) | ⚠️ 의사결정 dict 생성, broker 호출 0 |
| **`src/etf/leverage_engine.py:258` _emergency_close** | (의사결정만) | ⚠️ 의사결정 dict 생성, broker 호출 0 |
| **`src/etf/orchestrator.py:252` action_type close_all** | (의사결정만) | ⚠️ orders dict 생성, broker 호출 0 |
| **`src/etf/risk_manager.py:152` action=close_all** | (의사결정만) | ⚠️ adjustments dict 생성, broker 호출 0 |
| `scripts/archive/*/backtest_*.py` force_close | (backtest) | ✅ archive 내부 + 시뮬레이션만 |

→ **canonical 청산 경로는 5/29 P2까지 완료**. ETF 엔진은 의사결정 dict만 생성 (실주문 X).

### 2-4. 카테고리 4 — telegram sell/close/liquidate 경로 (직접 grep)

| 라인 | 함수 | 5/29 상태 |
|---|---|---|
| L337 `_run_sell_analysis` | AI 분석 → confirm 버튼 | callback이 L394 _execute_sell 호출 |
| L394 `_execute_sell` (callback) | confirm 버튼 → 매도 | ✅ P1-A4 (default "paper") |
| L396 `_execute_liquidate` (callback) | 전량 청산 | ✅ P1-A4 |
| L432 `_execute_sell` (AI sell button) | AI 추천 매도 | ✅ P1-A4 (동일 함수 경로) |
| L441 `_execute_sell` (50% sell button) | 부분 매도 | ✅ P1-A4 (동일 함수 경로) |
| L589 `_execute_liquidate` 정의 | 청산 실행 | ✅ P1-A4 |
| L926 `_cmd_liquidate` 정의 | "청산" 명령 진입점 | ✅ confirm 버튼 → L396 |
| L1442 "청산" 명령 매핑 | 텔레그램 키워드 → cmd | ✅ |

→ **telegram 매도/청산 경로 전부 P1-A4 마이그레이션 완료**. 정책 매칭 OK.

### 2-5. 카테고리 6 — order_intents_gate 우회 legacy adapter 호출 (Explore 2)

→ **0건** (5/28 P0 5건 + 5/29 P1-A 5 파일 + 5/29 P2 1건 마이그레이션으로 모든 매매 진입점 mode/executor_bot 명시).

### 2-6. 카테고리 7 — 테스트/수동용 + 실주문 객체 생성 가능 (Explore 2)

| 파일 | 위험도 | 분류 |
|---|---|---|
| `scripts/one_off/integration_dryrun_5_20.py` | 낮음 (dry-run 전용) | **DELETE 후보** |
| `scripts/smart_entry_runner.py --live` | 중간 (--live + confirm) | KEEP+GUARD (이미 가드 적용) |
| `scripts/run_limit_up_scanner.py --live` | 중간 (--live 플래그) | KEEP+GUARD (이미 가드 적용) |
| `tools/manual_order.py` | (검출 0건, 파일 없음) | - |
| `tests/` 외 `scripts/test_*.py` | (검출 0건) | - |

---

## 3. ★ 신규 발견 (메인 AI 직접 grep)

### 3-1. ★ Flask 웹 "청산" 큐 — dead-letter queue (소비자 0건)

**위치**: `website/flask_app.py:956-968`

```python
elif cmd == '청산':
    q = _load_remote_queue()
    q.append({'cmd': 'liquidate', 'ts': datetime.now().isoformat()})
    _save_remote_queue(q)
    return {
        'title': '💀 전량 청산 명령',
        'lines': [
            '🛑 청산 명령이 큐에 등록되었습니다.',
            '로컬 시스템이 명령을 수신하면 전량 시장가 매도됩니다.',
            ...
        ]
    }
```

**검증**:
- `remote_queue.json` 파일 grep → flask_app.py에서만 생성/조회 (외부 polling 클라이언트 0건)
- `cmd == 'liquidate'` 처리 grep → telegram callback 1건만 (flask 큐와 별개 경로)
- `_load_remote_queue` polling 코드 검색 결과 0건

**결론**: 큐에 등록은 되지만 **소비자(consumer)가 없음**. 사용자에게 "로컬 시스템이 명령을 수신하면..."이라 안내하지만 실제 polling 코드 부재 = **dead-letter queue**.

**위험**: 누군가 polling 클라이언트를 추가하면 **canonical 우회 청산 경로**가 즉시 활성화됨.

**분류 후보**: **QUARANTINE** (Flask 라우트 자체는 운영 중이라 즉시 삭제 위험. 큐 등록 + UI 표시 부분만 비활성화 권장) 또는 **DELETE** (사장님 결단).

### 3-2. ETF 엔진 close_all 경로 (의사결정만)

**위치**: `src/etf/{index_engine,leverage_engine,orchestrator,risk_manager}.py`

- `_close_all` / `_emergency_close` 메서드는 의사결정 dict 생성만 (broker 호출 0건)
- `ETFOrchestrator.decide()` → `orders` dict 반환 (`axis/code/action/target_weight_pct/reason`)
- 출력 소비자: `scripts/run_etf_rotation.py` (entrypoint) + `src/etf/data_bridge.py`

**검증 필요 (P3 권장)**: `scripts/run_etf_rotation.py`가 orders dict를 실제 KIS 주문으로 변환하는지 별도 확인. 현 시점 grep 결과 broker 직접 호출 0건 + register_intent 호출 0건 → **canonical 미연결 상태로 추정**.

**분류 후보**: **KEEP** (의사결정 라이브러리, 매매 직접 0). 단, **run_etf_rotation.py 별도 P3 감사 권장**.

---

## 4. 5분류 종합표

### 4-1. DELETE 후보 (2건)

| # | 파일 | 사유 |
|---|---|---|
| 1 | `scripts/dashboard.py` | "DEPRECATED — flowx.kr/quant 페이지가 본진" 명시. Streamlit 미사용. 모든 기능 Supabase 이관 |
| 2 | `scripts/one_off/integration_dryrun_5_20.py` | 5/20 가동 전 1회 시뮬 (2026-05-18 작성). dry-run 전용. 가동 9일 누적으로 용도 종료 |

### 4-2. MERGE 후보 (4건)

| # | 파일 | 흡수 대상 |
|---|---|---|
| 1 | `scripts/paper_warmup_daily.py` | canonical paper evaluation 파이프라인 (별도 PDCA) |
| 2 | `scripts/paper_trading_unified.py` | FLOWX portfolio tracking 모듈 |
| 3 | `scripts/run_limit_up_scanner.py` | `run_adaptive_cycle` integrand (`--limit-up` 플래그) |
| 4 | `scripts/run_quant_3day_pilot.py` | `run_adaptive_cycle --paper --pilot 5-27:5-29` |

**조건**: 별도 PDCA 진입 (현 사이클 외 — 9월 v2 리팩 권장).

### 4-3. KEEP+GUARD 후보 (4건 — 기존 가드 유지)

| # | 파일 | 가드 상태 |
|---|---|---|
| 1 | `scripts/auto_buy_executor.py` | ✅ AUTO_TRADE_5_20 + KILL_SWITCH + P0-2 mode/executor_bot |
| 2 | `scripts/owner_rule_monitor.py` | ✅ KILL_SWITCH + P0-B mode/executor_bot |
| 3 | `scripts/sell_monitor.py` | ✅ KILL_SWITCH + P1-A mode/executor_bot + --dry-run |
| 4 | `scripts/smart_entry_runner.py` | ✅ --live 플래그 + confirm 프롬프트 + KILL_SWITCH |

### 4-4. QUARANTINE 후보 (1건 ★ 신규)

| # | 파일 | 사유 | 격리 방안 |
|---|---|---|---|
| 1 | **`website/flask_app.py:956-968` "청산" 큐 라우트** | 소비자 0건 dead-letter queue. canonical 우회 청산 경로 잠재 위험 | "청산" 라우트 비활성화 + 큐 등록 코드 주석 처리 + UI 표시 삭제 (Flask 자체는 유지) |

### 4-5. KEEP 후보 (13+ 건)

- `src/use_cases/*` 라이브러리 함수 (매매 로직은 별도 gate 처리)
- `src/adapters/` (KisOrderAdapter / PaperOrderAdapter — gate 강제)
- `src/strategies/chart_hero_executor.py`
- `src/telegram_command_handler.py` (P1-A4 완료)
- `src/etf/{index_engine,leverage_engine,orchestrator,risk_manager,sector_engine,data_bridge}.py` (의사결정 라이브러리, broker 호출 0)
- VPS 살아있는 cron 21건의 비매매 스크립트 (alert_foreign_surge / auto_backtest_weekly / auto_regression / 등 19건)
- `scripts/archive/*` 152 파일 (격리 정책 100% 준수)
- `_etf_ref/*` 7 파일

---

## 5. order_intents_gate 우회 가능성 — 5/29 종일 후 종합 재판정

| 진입점 | 상태 |
|---|---|
| chart_hero / owner_rule / adaptive_stop_loss / adaptive_quick_profit (5/28 P0) | ✅ 4/4 |
| telegram 3종 (5/29 P1-A) | ✅ 4/4 |
| sell_monitor / smart_sell / live_trading (5/29 P1-A) | ✅ 4/4 |
| safety_guard.emergency_liquidate (5/29 P2) | ✅ 4/4 |
| main.py cmd_emergency_stop (5/29 P2) | ✅ 4/4 |
| ETF orchestrator orders | ⚠️ broker 미연결 (의사결정만, 사용 시 별도 결단 필요) |
| **Flask 큐 'liquidate' 명령** | ⚠️ 소비자 0건 dead-letter (QUARANTINE 권장) |

→ **현재 운영 중 매매 진입점 우회 0건**. Flask 큐는 잠재 위험 (소비자 0).

---

## 6. 사장님 결단 요청 사항 (5건)

### 6-1. DELETE 2건 처리 결단
- **dashboard.py**: 즉시 삭제 vs scripts/archive/deprecated/로 이동
- **one_off/integration_dryrun_5_20.py**: 즉시 삭제 vs scripts/archive/orphan_/로 이동

### 6-2. QUARANTINE 1건 처리 결단 ★
- **Flask "청산" 큐**: 코드 제거 vs 라우트 비활성화 (응답 "기능 중단됨") vs 그대로 유지 (현 dead-letter 상태)
- 권장: **라우트 비활성화** (큐 등록 코드 주석 + UI 메뉴 제거, Flask 자체는 유지) — 향후 소비자 추가 시 canonical gate 통과 의무 명시

### 6-3. MERGE 4건 시점 결단
- 별도 PDCA 진입 시점 (즉시 vs 9월 v2 리팩 vs 보류)
- 권장: **9월 v2 리팩** (현 사이클은 P1/P2 안전망 우선)

### 6-4. ETF 엔진 별도 P3 감사 결단
- `scripts/run_etf_rotation.py` orders → broker 변환 경로 추가 감사 (단독 deletion/quarantine audit 사이클)
- 권장: **별도 P3 PDCA 진입** (지금은 broker 호출 0건이라 즉시 위험 0)

### 6-5. commit 시점 결단
- 본 audit 보고서를 P2 + 3 문서 + Codex 의뢰서 ②와 함께 commit vs 별도 commit
- 권장: **본 보고서만 별도 commit** (audit 결과는 코드 수정 0건이라 commit 가능)

---

## 7. 후속 PDCA (현 사이클 외)

| # | 항목 | 우선순위 | 작업량 |
|---|---|---|---|
| 1 | `run_etf_rotation.py` 단독 audit | P3 | ~30분 |
| 2 | DELETE 2건 처리 + 회귀 | P2 | ~30분 |
| 3 | QUARANTINE 1건 (Flask 큐) 처리 + 회귀 | P2 | ~1시간 |
| 4 | MERGE 4건 v2 리팩 | P3 (9월) | ~1주 |
| 5 | `tests/test_live_trading_import.py` 신규 회귀 (SyntaxError 자동 검출) | P3 | ~20분 |

---

## 8. 적용 금지 (본 audit 후)

- ❌ 본 audit 결과로 즉시 파일 삭제 X (사장님 결단 후만)
- ❌ Flask 라우트 즉시 비활성화 X (결단 후만)
- ❌ cron 복원 / scheduler unmask / 실거래 실행 X
- ❌ 단독 commit X

---

## 9. 표현 룰

### 사용 가능
- "삭제/격리 후보 1차 식별 완료"
- "scripts/archive 정책 100% 준수 확인"
- "Flask 청산 큐 — dead-letter 잠재 위험 식별"
- "canonical 부합 검증 완료 (현재 우회 진입점 0건)"

### 사용 금지
- "운영 안전 완성" X
- "Phase 1 paper 재가동 가능" X
- "전체 코드베이스 청소 완료" X
- "재가동 가능" X

---

## 10. 연결 문서
- `docs/02-design/p1-truth-pack-5-29.md`
- `docs/02-design/p1-residual-plan-5-29.md`
- `docs/02-design/filelock-policy-5_29.md`
- `docs/02-design/hmac-rotation-playbook-5_29.md`
- `docs/02-design/restart-dry-run-spec-5_29.md`
- `ops/codex_outbox/20260529T101341_..._p1-a4-callers-migration_review-requested.md`
- `ops/codex_outbox/20260529T194250_..._p2-residual-4items-and-critical-fix_review-requested.md`
- `CLAUDE.md` (scripts/archive + _etf_ref 참조 금지 정책)
