# RISK_ENGINE Phase 2 + Phase 3(G8) 작업 보고서

- 작성: 2026-06-14 · 브랜치: `feature/risk-engine` (worktree `D:\quant-bot-risk`)
- 연계 스펙: `docs/01-plan/RISK_ENGINE_SPEC_v2.md` · 인수인계: `SESSION_HANDOFF.md`
- 상태: ★freeze 유지 · 실주문 0 · 매매로직 무손상

---

## 0. 한 줄 요약

> RISK_ENGINE의 **L2 VaR 게이트(Phase 2 전체)** + **L3 드로다운 디레버리징 사다리 G8(Phase 3a/3b/3c)**를
> 완성했다. 동적 게이트의 두 축 — **시장 변동성(VaR)** 과 **누적 손실(드로다운)** — 을 모두 막았다.
> 매매 심장(pre_trade_gate) 31개 테스트를 보존하며 엔진→게이트→배선→검증을 닫았고, 검증 중
> 잠복버그 2개를 덤으로 포착했다.

## 1. 배경 & 목표

- 사장님 지시 순차 트랙 ①②③: C-ii 적대검증 → fx-liquidity P0 → RISK_ENGINE Phase 2(VaR 게이트).
- 같은 세션 추가: Phase 3 G8 드로다운 사다리(엔진·게이트·배선).
- 원칙: freeze 유지(실주문 0). 게이트는 "주문을 *더* 막는" 추가라 freeze와 양립(Phase 1a 정신).

## 2. 작업 내역 (커밋 순)

### ① C-ii 적대 자기검증 — `0bb6d21` (main)
- C-ii(6개 BUY 호출처 게이트 배선)의 독립 검증.
- **발견(P2, 안전측)**: 시장가 경로(adaptive_reentry·live_trading market)에서 게이트 발급가와 어댑터
  검증가(`_estimate_price` 주문시점 현재가)가 어긋나 유효 토큰이 거부될 수 있음. **fail-closed(매수 차단)
  라 리스크 한도는 지켜짐 = freeze blocker 아님.** → unfreeze-checklist에 박제(시장가 실탄 전 완화).
- 견고 확인 3축: nonce 이중소비(재발급+verify 1회)·enforce 우회불가(어댑터 `_is_mock` 2차 방어)·
  테스트 사각지대(enforce 12케이스 전부 지정가 시뮬).

### ② fx-liquidity P0-1 — `1b8a481` (main)
- four_signal 사망원인 규명: 생산자 `four_signal_gate.save_daily_record`가 `__main__` 전용 **고아**
  (BAT 28개·cron 전수 미배선) → 5/19 수동 1회가 마지막.
- 재발방지: `stale_warning()` + preflight가 stale을 `[WARN]`로 노출(매매 게이트 checks와 분리, RESULT 불변).
- **★부수 수정**: `four_signal_gate` import가 `kis_weekly_kit` 체인의 `load_dotenv`로 `.env` 76키를
  `os.environ`에 로드 → 테스트 격리 파괴(`adaptive_buy_queue.SPLIT_MAX_QTY` 캐시 오염) → lazy import로 격리.

### ② fx-liquidity P0-2 — `b8e8ac1` (main)
- 환율 신호(usdkrw 1일 +1% / 20일 신고가)의 KOSPI 하락 선행성 적대검증(ECOS 환율 + Yahoo ^KS11, 1083일).
- 결과: 환율 1일 +1% = **NOISE**(lift 0.86) / 20일 신고가 = **미약**(lift 1.17). 국면 의존(2022 약세장
  lift 1.19 / 2024+ 현재장 0.21). → MacroRiskScore 환율 컴포넌트 **"참고" 강등** 권고.

### Phase 2 — L2 VaR 게이트 (main)
| 단계 | 커밋 | 내용 |
|---|---|---|
| 2a 엔진 | `40f110e` | `risk/var_engine.py` — FHS(EWMA σ→표준화→리스케일→VaR95/ES95+스트레스) |
| 2b 게이트 | `e687af3` | `pre_trade_gate` G1(VaR95≤2.5%)/G2(스트레스≤4%) 활성화 + RESIZE 일반화(G1/G2/G3) |
| 2c 배선 | `c27a54f` | `gate_wiring` 보유+신규 종목 수익률 주입 → G1/G2 실활성화(데이터 부족 시 G3~G6 폴백) |
| 2d 검증 | `f923323` | `risk/var_backtest.py` — 롤링 VaR95 초과율 + Kupiec POF. ★실데이터 삼성005930 **초과율 5.35%**(밴드 3~8%)+Kupiec LR 0.26(통과) = 모델 calibrated |

### Phase 3 — L3 드로다운 사다리 G8 (worktree)
| 단계 | 커밋 | 내용 |
|---|---|---|
| 3a 엔진 | `497a199` | `risk/drawdown_ladder.py` — 계좌 고점 대비 DD로 노출 사다리(0/-4/-7/-10%) + 히스테리시스(복귀 +1.5%p, 점진) |
| 3b 게이트 | `1a2c4d9` | `pre_trade_gate` G8 — 신규 금지(step2/3)→REJECT, 사이즈 축소(step1)→사전 50%(original=proposed 보존) |
| 3c 배선 | `dcbb10c` | `risk/account_peak.py` EquityPeakStore(고점 영속·graceful) + `gate_wiring`에 `equity_peak_store` 주입 → DD→ladder→G8 완전 활성 |

## 3. 핵심 산출물 (모듈)

신규: `risk/var_engine.py` · `risk/var_backtest.py` · `risk/drawdown_ladder.py` · `risk/account_peak.py` ·
`scripts/backtest_fx_signal_precision.py`
수정: `risk/pre_trade_gate.py`(G1/G2/G8 + RESIZE 일반화) · `src/use_cases/gate_wiring.py`(수익률·DD 배선) ·
`src/macro/four_signal_gate.py`(staleness 가드 + lazy import) · `tools/quant_preflight.py`(staleness WARN)

## 4. 검증

- **신규 테스트 60+**: var_engine 13 · var_gate 9 · var_backtest 6 · four_signal_staleness 9 ·
  drawdown_ladder 11 · drawdown_gate 6 · account_peak 6.
- **전체 회귀**: main 30 failed/1453 passed · worktree 27 failed/1476 passed(7 skipped). **신규 실패 0** —
  failed는 전부 기존 베이스라인(날짜만료 intent·stash 깨짐: backtest_mechanics·phase1_paper_trade·
  protected_tickers·simulate_5_26·auto_buy_decider).
- **실데이터**: VaR 백테스트 삼성 초과율 5.35% calibrated. ★pykrx는 KRX_DATA_PW 만료로 차단 → Yahoo ^KS11 우회.
- ★매매 심장(pre_trade_gate 31) + 게이트 호출처(gate_wiring·var_gate 등) 전부 보존.

## 5. 게이트 현황

| 게이트 | 내용 | 상태 |
|---|---|---|
| G1 / G2 | 포트폴리오 VaR95 / 스트레스 VaR95 | ✅ Phase 2 |
| G3~G6 | 단일종목·섹터·상관클러스터·유동성 | ✅ Phase 1 |
| **G8** | **드로다운 디레버리징 사다리** | ✅ **Phase 3(오늘)** |
| G7 | Component VaR 기여 | ⬜ Phase 4 (상관행렬 배선 필요) |

→ **남은 게이트는 G7 하나.** 리스크의 두 축(시장 변동성 VaR + 누적 손실 드로다운)은 모두 막았다.

## 6. 부수 발견 (검증 중 포착)

1. **시장가 사이즈 바인딩**(C-ii): fail-closed 안전측 — 시장가 실탄 켜기 직전 완화 필요(박제).
2. **four_signal_gate `.env` import 부작용**(P0-1): 테스트 격리 파괴 잠복버그 → lazy import 수정.

## 7. 남은 것 / 다음 (fresh 세션 권장)

- ⚠️ **G8 실배선**: 호출처(live_trading 등)가 `build_gate_result(equity_peak_store=EquityPeakStore())`를
  넘기면 production 활성. **현재 엔진·게이트·배선 능력은 완비, 호출처 주입만 미적용**(VaR의 G1/G2도 동일 — 능력 추가지 강제 활성 아님 = freeze 무손상).
- Phase 3 나머지(게이트 아닌 노출/모니터 레이어): 변동성 타겟팅(§4.3 `vol_targeting.py`)·스트레스 테스트
  (§4.1 `stress_test.py` 역사5+가상5)·크라우딩(§4.4 `crowding.py`).
- Phase 4: Component VaR(G7 §3.3)·이중상관(G5 스트레스 §3.4).
- unfreeze: **E(페이퍼 20일)만** 남음. ★Phase 2/3은 unfreeze 조건이 아니다(리스크 엔진 심화·추가 보호층).

## 8. 운영 메모

- 작업트리 분리: **main(`D:\sub-agent-project_퀀트봇`)** = 운영·봇 자동화·웹봇·ewy / **worktree
  (`D:\quant-bot-risk`, feature/risk-engine)** = risk 개발. 폴더가 달라 git staging 독립(섞임 방지).
- 다음 퀀트봇 세션은 `D:\quant-bot-risk`에서 시작 → CLAUDE.md → `SESSION_HANDOFF.md` 자동 인식.
- 상세 메모리(main 세션): `...\D--sub-agent-project----\memory\` (session_state_6_14_phase2_var 등).
