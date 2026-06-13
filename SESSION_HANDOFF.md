# 🔁 SESSION HANDOFF — 퀀트봇 risk 개발 worktree (작성 2026-06-14)

> **이 폴더 `D:\quant-bot-risk`는 `feature/risk-engine` worktree입니다 — 퀀트봇 risk 개발 전용.**
> 운영·봇 자동화(BAT/scheduler/codex_inbox)·웹봇·ewy는 main(`D:\sub-agent-project_퀀트봇`)에서 돕니다.
> 새 퀀트봇 세션은 이 HANDOFF를 먼저 읽고 시작하세요.
> 📄 **오늘 작업 상세 보고서**: `docs/04-report/risk-engine-phase2-3-report-6_14.md`

## 왜 분리?
멀티 세션(퀀트봇·웹봇·ewy)이 같은 폴더의 main을 공유 → git staging 섞임 사고(Phase 2b 커밋 e687af3에
사장님 다른 세션 ewy 2파일 동봉). **폴더가 다르면 staging이 독립** → 해결. (feedback_worktree_separation 참조)

## 이번 세션(6/14) 작업 = 순차 트랙 ①②③, 7커밋 (HEAD `f923323` = origin/main)
- **①** C-ii 적대검증: 시장가 사이즈 바인딩 P2 갭 박제 + 견고 3축(nonce·enforce·테스트 사각지대) — `0bb6d21`
- **②** fx-liquidity P0-1: four_signal 사망원인(`save_daily_record` 고아=BAT/cron 미배선) 규명 +
  preflight staleness 가드 + ★`four_signal_gate` import의 `.env` 로드 부작용(테스트 격리 파괴) lazy 수정 — `1b8a481`
- **②** fx-liquidity P0-2: 환율 신호(1일+1%/20일 신고가)→KOSPI 하락 선행성 = NOISE/미약 입증 → "참고" 강등 — `b8e8ac1`
- **③** ★**RISK_ENGINE Phase 2 (L2 VaR 게이트) 전체 완료**:
  - 2a 엔진(FHS: EWMA σ→표준화→리스케일→VaR95/ES95+스트레스) — `risk/var_engine.py` (`40f110e`)
  - 2b 게이트 G1(VaR95≤2.5%)/G2(스트레스≤4%) 활성화 + RESIZE 일반화 — `risk/pre_trade_gate.py` (`e687af3`)
  - 2c 수익률 배선(보유+신규 종목 → returns_by_ticker 주입 = 실활성화) — `src/use_cases/gate_wiring.py` (`c27a54f`)
  - 2d 모델 검증(롤링 VaR95 초과율 + Kupiec POF) — `risk/var_backtest.py` (`f923323`).
    ★실데이터 삼성005930(평가1009일) 초과율 **5.35%**(밴드 3~8%) + Kupiec LR 0.26(통과) = 모델 calibrated.

## 현재 상태 (불변)
- freeze 유지 · 실주문 0 · 매매로직 무손상 · 전체 **30 failed / 1453 passed**(신규 실패 0; 30은 기존 날짜만료/stash).
- ⚠️ **Phase 2 완료 ≠ unfreeze 조건**. unfreeze는 여전히 **E(페이퍼 20일)만** 남음(시간 누적, 코드 0).
- production은 로컬 parquet 충분(≥60) 종목만 G1/G2, 부족하면 G3~G6 폴백(graceful, 과차단 방지) = 점진 활성.

## 셋업 (이미 완료)
- `venv` junction → main venv 공유 / `data\raw` junction → main parquet(1198종목) / `.env` 복사.
- 검증: 새 worktree에서 test 28 passed + 실데이터 백테스트 작동 확인.

## 다음 후보
- RISK_ENGINE **Phase 3**(드로다운 사다리 G8·변동성 타겟팅·크라우딩) / **Phase 4**(Component VaR G7·이중상관 G5
  스트레스). 스펙 `docs/01-plan/RISK_ENGINE_SPEC_v2.md` §4·§3.3-3.4.
- unfreeze 별트랙: E(페이퍼 20일 누적).

## git 운영 규칙 (이 worktree)
- `git add`는 항상 명시 파일만. 신규 파일은 `git add` 후 `git commit -- <pathspec>`(staged 섞임 방지).
- 작업 → `git push origin feature/risk-engine`. ★freeze라 risk가 feature에 있어도 봇 매매 0 → 즉시 main
  머지 불필요. unfreeze 전 검증 끝나면 main 머지(봇은 main에서 돌므로).
- 상세 컨텍스트 = main 세션 메모리(`C:\Users\ASUS\.claude\projects\D--sub-agent-project----\memory\`:
  MEMORY.md · session_state_6_14_phase2_var · session_state_6_14_fx_liquidity_p0 · feedback_worktree_separation).
  이 worktree 세션이 별도 메모리를 쓰면 이 HANDOFF가 요약본 역할.

---

## Phase 3a/3b 완료 (6/14 같은 세션 연장, worktree 커밋)
- ✅ **드로다운 사다리 G8** — 엔진 `risk/drawdown_ladder.py`(`497a199`) + 게이트 `pre_trade_gate` 연결(`1a2c4d9`).
  계좌 고점 대비 DD로 노출 사다리(0/-4/-7/-10%) + 히스테리시스(복귀 +1.5%p, 점진 한 단계). 게이트 G8: `ladder`
  주입 시 신규 금지(step2/3)→REJECT, 사이즈 축소(step1)→사전 50%(★original=proposed 보존). `ladder=None`이면
  not_active. test_drawdown_ladder 11 + test_drawdown_gate 6 + 기존 40(pre_trade_gate 31·var_gate 9) 보존, 회귀 신규 0.

## 다음 후보 (Phase 3 나머지 + Phase 4)
- ✅ **3c 완료** (`dcbb10c`): `risk/account_peak.py` EquityPeakStore(계좌 고점 영속, nonce_store graceful 패턴)
  + `gate_wiring`에 `equity_peak_store` 주입 → DD → `ladder_state()` → `evaluate(ladder=)`. **G8 완전 활성**(주입 시).
  `equity_peak_store=None`이면 ladder=None=not_active → gate_wiring 16 보존. test_account_peak 6.
  ★실배선 = live_trading/호출처가 `build_gate_result(equity_peak_store=EquityPeakStore())` 넘기면 끝(아직 호출처는 미주입).
- Phase 3 나머지(게이트 아닌 노출 조절/모니터): 변동성 타겟팅(§4.3 `vol_targeting.py`)·스트레스 테스트(§4.1
  `stress_test.py` 역사5+가상5)·크라우딩(§4.4 `crowding.py`).
- unfreeze 별트랙: E(페이퍼 20일).

## Phase 4 게이트 G5 + G7 완료 (6/14 이어지는 세션, worktree 커밋) → ★게이트 G1~G8 전부 구현
- ✅ **G7 Component VaR** (`c6dc909`): 엔진 `risk/component_var.py` 신규 + `pre_trade_gate` 연결.
  Euler 공분산 분해를 **FHS 필터링 수익률** 위에서(기여율_i = w_i·Cov(r̃_i,r̃_p)/Var(r̃_p), Σ=1 정확).
  꼬리 조건부 추정 대신 공분산 분해 채택(표본 60~수백일이면 95% 꼬리 표본 부족=노이즈). near-zero
  분산(완전 헤지) 상대 floor 가드로 기여율 발산 차단. 신규 종목 기여 ≤ component_var_limit(25%) →
  RESIZE. ★≥2 리스크 포지션 시만 활성(단일=pass, 부트스트랩 과차단 방지). returns 미주입=not_active.
  ★`var_engine`에서 `_build_fhs_panel/FhsPanel` 추출 → VaR와 Component VaR 동일 패널 공유(로직 1곳),
  compute_portfolio_var 수식 불변(var_backtest 5.35% 보존). gate_wiring 무변경(기존 returns 배선 자동 활용).
  test_component_var 10 + test_component_gate 6.
- ✅ **G5 이중 상관 클러스터 실활성화** (`ee40fd7`): 엔진 `risk/correlation.py` 신규 + `gate_wiring` 배선.
  ρ_normal(최근 252일 EWMA 가중 피어슨) → ρ_stress = ρ_normal + 0.5(1-ρ_normal) 1방향 슈링크(음의
  상관도 1쪽으로 당겨 헤지 과신 차단). gate_wiring이 VaR용으로 이미 로드한 returns를 **재사용**해 신규 vs
  보유 ρ_stress 계산 → `Holding.corr_with_new` 주입 → **G5 실활성화**. 그전엔 corr_with_new=None만 넘겨
  production에서 G5가 한 번도 군집 안 됨(unknown_corr_count로 노출만). 게이트 로직 무변경(ρ_stress ≥ 0.8
  보유 합산 = 유효 단일 포지션 → G3 한도 → REJECT). 계산 불가(공통표본<60)는 None 유지(과차단 방지).
  test_correlation 9(엔진 7 + G5 REJECT/비활성 2).
- 현재 상태: risk/gate **191 passed** · 전체 **27 failed / 1505 passed**(신규 실패 0; 27=기존 날짜만료/stash).
  freeze 유지 · 실주문 0 · G5=REJECT만 / G7=RESIZE만.
- ⚠️ **활성 조건**(G1/G2와 동일 graceful): G5/G7은 returns_by_ticker 가용(로컬 parquet ≥60obs) 종목만 실작동.
  데이터 부족 시 G7 not_active·G5 군집 제외(unknown_corr_count). pykrx KRX 만료라 로컬 parquet 있는 종목만.
- 다음 후보(별 세션): Phase 4 **크라우딩**(§4.4 `crowding.py`) / Phase 3 나머지(`stress_test.py` §4.1·
  `vol_targeting.py` §4.3). ★G5 정밀화 옵션 = ρ_stress 슈링크 대신 **VKOSPI 위기구간 별도 추정**(§3.4
  "데이터 충분하면 이쪽 우선") — 현재는 슈링크 공식(스펙 1순위 기재) 사용.
- ★worktree HEAD `ee40fd7` = origin/feature/risk-engine (0/0).
