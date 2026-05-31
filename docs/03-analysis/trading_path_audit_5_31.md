# 퀀트봇 실매매 경로 검수 결과 (2026-05-31)

> 방법: 단타봇 직접 정독 + 적대적 서브에이전트 1대(무료, Codex 미사용) → **발견 전건을
> 단타봇이 코드로 재확인**. 환각/과대평가 2건 기각. 결정적 검증(실행·테스트)만 신뢰.
> 대상: `live_trading.py`, `kis_order_adapter.py`, `smart_sell.py`, `split_order.py`,
> `paper_order_adapter.py` + 의존 모듈(position_tracker, order_intents_gate, sell_monitor).

## 전제: live 매매는 현재 3중 잠금(OFF)

모든 결함은 **LATENT** — 아래 게이트가 다 풀려야 발현. 지금은 사고 아님.
1. `.env`에 `LIVE_TRADING_MODE` 없음 → 기본 `paper`
2. `create_live_engine`: mode!="live"면 RuntimeError raise
3. `order_intents_gate`가 quant+live 차단
4. `settings.yaml: dry_run: true`

→ 단, **live 재개 시 즉시 발현**하므로 재개 전 수정 의무.

## 검증 결과 (단타봇 재확인 기준)

| # | 발견 | 판정 | 조치 | 커밋 |
|---|---|---|---|---|
| C-1 | `_wait_for_fill`가 PARTIAL 미분기 → 타임아웃 cancel 시 체결분 유실 = 유령 포지션 | ✅ 사실 (HIGH) | 부분체결 보존+FILLED 등록 | `c7f8bb0` |
| C-3 | `cancel()` 반환 bool 무시하고 CANCELLED 단정 | ✅ 사실 (MED-HIGH) | 반환 확인+reconcile 경고 | `c7f8bb0` |
| H-2 | 시장가 매수 시 `_estimate_price`=0이면 일일 금액한도 우회 | ✅ 사실 (HIGH) | est_price<=0 fail-safe 거부 | `3f547d7` |
| M-4 | 잔고 조회 실패→cash=0→총손실 과대→긴급청산 오발동 | ✅ 사실 (**MED→HIGH 상향**) | fetch_balance `ok` 플래그+청산 게이트 | `3f547d7` |
| M-2 | split_order가 PENDING(0원)을 평균가에 섞어 왜곡 | ✅ 사실 (MED) | 가격확인 체결분만 avg 반영 | (이 커밋) |
| H-1 | `filled_quantity or quantity` 단정 + 매수후 잔고 reconcile 없음 | ◐ 일부 (C-1으로 체결수량 정확화. 매수후 브로커 재대조는 후속 권장) | C-1로 완화 | `c7f8bb0` |
| **C-2** | "smart_sell/split_order가 PositionTracker reconcile 안 함 → 재매도" | ❌ **기각(과대평가)** | — | — |
| **M-1** | smart_sell `cumulative_filled` 덮어쓰기 → 초과매도 | ❌ **기각** | — | — |
| L-1 | `_adjust_to_tick` return 뒤 죽은코드 | ⚪ 미관(선택) | 미수정 | — |
| H-3 | record_buy가 PENDING 시점 요청수량으로 기록 | 🟡 사실이나 한도엔 보수적 | 미수정(저위험) | — |
| H-4/H-5/L-2/L-3 | 호가보정 이중·retry 카운터·grade F·avg_prvs 의미 | ⚪ 저위험/확인필요 | 미수정 | — |

## 기각 사유 (재논쟁 방지 — 중요)

- **C-2 기각**: `sell_monitor.py`는 매 사이클 `broker.fetch_balance()`의 실보유(output1)를
  직접 읽음(116-157). PositionTracker/positions.json을 안 씀. 매도 완료가 다음 사이클
  브로커 잔고에 자동 반영 → 재매도 안 일어남. `live_trading`(PositionTracker)와 `sell_monitor`
  (브로커-진실)는 **별개의, 각자 일관된 시스템**. 곧이곧대로 묶었으면 오히려 더 견고한
  브로커-진실 패턴을 훼손할 뻔함.
- **M-1 기각**: KIS `tot_ccld_qty`는 단조증가 누계라, "부분체결분이 사라져 잔여 전량
  재매도" 시나리오는 실현 불가.

## 근본원인 (수정 완료분)

(a) PARTIAL을 일급으로 안 다룸 → C-1로 해소. (b) silent-fail이 파괴적 행동(긴급청산)으로
전이 → M-4로 차단. (c) 한도/평균가 가드의 0값 우회 → H-2/M-2로 fail-safe.

## 회귀 검증

- 신규 영구 테스트 8개: `test_wait_for_fill_partial.py`(4), `test_balance_ok_flag.py`(2),
  `test_split_order_avg_price.py`(2) — 전부 PASS.
- 전체 스위트 stash 비교: 본 수정이 실패 **0개 추가**(32→30, 신규 통과분).

## ⚠️ 본 검수 범위 밖 — 사전 기존 테스트 깨짐 (별도 처리 필요)

전체 스위트 ~30건 실패는 본 수정과 **무관한 사전 깨짐 + 테스트 순서 오염**:
- `test_backtest_mechanics.py` 단독 14/18 실패 (backtest_engine — 본 검수 무관 파일)
- `test_protected_tickers.py`, `test_simulate_5_26_launch.py`, `test_preflight_simulate_paper.py`
- `test_phase1_paper_trade.py` stale 날짜 intent fixture 2건(expires 5/29) + 순서 오염
→ 퀀트봇 테스트 스위트 건전성은 별도 작업으로 다룰 것.
