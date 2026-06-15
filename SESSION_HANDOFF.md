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

## Phase 4 크라우딩 모니터 완료 (6/14 순차 트랙 ①, worktree 커밋 `1c923ac`)
- ✅ **크라우딩/동질화 모니터** `risk/crowding.py` 신규 (스펙 §4.4). ★게이트(G1~G8)가 아니라 **L3 노출 조절
  모니터** — pre_trade_gate의 PASS/RESIZE/REJECT가 아니라 `gross_exposure_mult`를 낸다(호출처가 사다리·
  변동성 타겟팅 계수와 곱). drawdown_ladder와 동형(순수 계산, 데이터는 호출처 주입).
  - C1 보유 평균 쌍상관(60일) > crowding_corr(0.70) / C2 VKOSPI 20초과 **AND** 5일 30%급등(동시) / C3 외인선물
    2년 분위수 상·하위 5%. 경고 **2개 이상** → `gross_exposure_mult` 0.80(-20%p, 사다리와 별도). 3개여도 -20%p 고정.
  - ★**C1 vs G5 구분**(혼동 주의): C1=평시 동질성(균등가중 60일 피어슨), G5=위기 보수화 ρ_stress(1방향 슈링크).
    목적이 달라 C1은 슈링크 미적용(평시 측정에 위기 가정 주입 = 이중 보수 방지). 주석에 박제.
  - C2/C3 시계열은 호출처 주입(VKOSPI·외인선물 미배선) → 미주입=미평가(graceful, 과경고 차단). 평가 가능 경고만
    카운트(evaluable=False는 '위험 없음'이 아니라 '판단 불가' = warning_count 제외). **G8과 동일 production 휴면 = freeze 유지**.
  - 견고성(component_var 자가검출 교훈 적용): C1 상수→corr NaN→미평가, C3 상수→`nunique` 가드(`(s<=cur).mean()`이
    1.0 반환해 거짓 경고 되는 것 차단), C2 past≤0/표본<6 가드. **config 무변경**(crowding_corr 선등록 활용, C2/C3
    세부 임계는 모듈 상수+스펙 §4.4 수치 박제 = 분기1회 변경 규칙 존중).
  - test_crowding **20 passed** + risk 핵심 **222 passed**(신규 실패 0). 격리: risk.config만 import, write 0, 실주문 0.
- 다음 후보(순차 ②③, 별 세션): **② Phase 3 나머지** — `stress_test.py`(§4.1 역사 H1~H5 + 가상 S1~S5)·
  `vol_targeting.py`(§4.3 EWMA 실현변동성 타겟 15% → scale). **③ G5 정밀화** — ρ_stress 슈링크 대신 VKOSPI
  위기구간 별도 추정(§3.4 "데이터 충분하면 이쪽 우선"). unfreeze 별트랙: E(페이퍼 20일).
- ⚠️ **Phase 4 크라우딩 완료 ≠ unfreeze**. 여전히 E(페이퍼 20일)만 남음(시간 누적, 코드 0).
- ★worktree HEAD `1c923ac` = origin/feature/risk-engine (push 후 0/0).

## Phase 3 나머지 완료 (6/14 순차 트랙 ②, worktree 커밋 `347a27b`·`b1a3b7d`)
- ✅ **변동성 타겟팅** `risk/vol_targeting.py` (`347a27b`, 스펙 §4.3). EWMA halflife 20일 실현변동성 ×√252 →
  scale = min(1.0, target_vol 15% / realized). ★scale ≤ 1.0(변동성 높으면 축소, 낮아도 **레버리지 확대 금지**).
  총노출 = 기본 × scale × 사다리(§4.2) × 크라우딩(§4.4). 미주입/표본<20/변동성0 → 중립 1.0. test 7.
- ✅ **스트레스 시나리오** `risk/stress_test.py` (`b1a3b7d`, 스펙 §4.1). ★사장님 승인 = **순수 집계 엔진**(시나리오
  충격을 호출처 주입). 역사 H1~H5(2008/2011/2020/2022/2024) + 가상 S1~S5. S3(KOSPI -8% 갭, 시장 베타 1 폴백)·
  S4(최대비중 하한가 -30%) = 포트 비중만으로 **항상 평가**. S1(FX)·S2(반도체)·S5(복합)·H1~H5 = 팩터 베타/역사
  충격 주입 시 평가, 미주입=미평가(과소평가 방지). `report.worst` = 최악 시나리오. test 18.
- ★**발견(중요)**: `factor_exposure.py`(§3.1 팩터 노출 EWMA 회귀)가 **미구현**(게이트 G1~G8엔 안 쓰여 빠져 있음).
  stress_test의 H1~H5 역사재생·S1/S2 팩터 시나리오가 여기 의존 → production 휴면. **factor_exposure가
  stress_test 완전 활성의 선행 트랙**(스펙 §4.1 "팩터 노출 × 당시 팩터 충격으로 근사").
- 세 L3 모니터(crowding·vol_targeting·stress_test) 전부 격리 순수계산 · production 배선 0(src import grep 0,
  매칭은 risk_models.stress_tests 필드·ports.stress_test 메서드 이름 겹침뿐) = freeze 유지 · 실주문 0.
- 전체 회귀 **27 failed / 1556 passed / 7 skipped**(신규 실패 0; 27=기존 backtest_mechanics·phase1_paper_trade·
  protected_tickers·simulate_5_26 = 날짜만료/stash, 신규 모듈 미import).
- 다음 후보: **③ G5 정밀화**(ρ_stress 슈링크 대신 VKOSPI 위기구간 별도 추정, §3.4) / **factor_exposure §3.1**
  (stress_test 완전체 선행) / **세 L3 모니터 실배선**(노출 관리 계층 = 사다리·vol·크라우딩 계수 곱, unfreeze 직전).
- ⚠️ **Phase 3 완료 ≠ unfreeze**. 여전히 E(페이퍼 20일)만 남음(시간 누적, 코드 0).
- ★worktree HEAD `b1a3b7d` = origin/feature/risk-engine (push 후 0/0).

## ③ G5 정밀화 완료 (6/14 순차 트랙 ③ = ★①②③ 전부 완료, worktree 커밋 `2a6d8a4`)
- ✅ **ρ_crisis 위기구간 실측** `risk/correlation.py` + `gate_wiring` 배선 (스펙 §3.4 "데이터 충분하면 이쪽
  우선"). 슈링크 공식(ρ_stress)의 대안 — VKOSPI 상위 10% 위기일만 모아 종목 간 동조화를 *실측*(가정 아님).
  - correlation.py: `crisis_correlation_with` + `_pearson` 추가. ★기존 stress_shrink/stress_correlation_with **무변경**.
  - gate_wiring: `vkospi_series`(graceful, equity_peak_store 패턴) 주입 시 ρ_effective = **max(ρ_stress, ρ_crisis)**
    = "둘 중 나쁜 값"(§3.4 line 181). ★정밀화는 G5를 **절대 약화 안 함**(REJECT→PASS 불가, 강화만 가능).
  - ★**동작 불변 입증**: VKOSPI 미주입(현 production 미배선) → crisis_map 비어 슈링크만 = G5 calibrated 보존.
    `test_g5_precision_crisis_tightens_cluster`가 r_no(미주입) cluster==[] / r_vk(주입) cluster=[H1]로 격리 입증.
  - graceful 폴백: VKOSPI 표본<30 / 위기 공통표본<_MIN_CRISIS_OBS(30) → 슈링크 유지. test_correlation 9→15.
- ⚠️ pre-commit hook의 영향모듈 자동회귀가 **시스템 python(pytest 미설치)** 사용해 실질 스킵(`[OK]`만 찍힘) —
  worktree 회귀는 반드시 `PYTHONPATH=D:/quant-bot-risk venv/Scripts/python.exe -m pytest`로 수동 실행(activate 안 먹힘).
  이번엔 수동으로 게이트 98 + 전체 1562 passed 검증함.
- ★**①②③ 순차 트랙 전부 완료**: ① 크라우딩(crowding) / ② Phase 3 나머지(vol_targeting·stress_test) / ③ G5 정밀화(ρ_crisis).
- 다음 후보: **factor_exposure §3.1**(stress_test 완전체 선행 + 게이트 G1~G8의 빠진 퍼즐 = EWMA 팩터 회귀) /
  **L3 모니터 실배선**(crowding·vol_targeting·stress_test → 노출 관리 계층, unfreeze 직전) /
  **VKOSPI·외인선물 데이터 파이프라인**(crowding C2/C3 + ρ_crisis 정밀화 활성화 전제).
- ⚠️ **①②③ 완료 ≠ unfreeze**. 여전히 E(페이퍼 20일)만 남음(시간 누적, 코드 0).
- ★worktree HEAD `2a6d8a4` = origin/feature/risk-engine (push 후 0/0).

## 6/15 factor_exposure §3.1 + factor_stress 오케스트레이터 (커밋 `645563a`·`8caf832`)
- ✅ **factor_exposure §3.1** (`645563a`, 6/15 1세션): EWMA 가중 다변량 팩터 회귀. 팩터 5종
  (market/smallcap/fx/semi/rate, smallcap·semi 시장잔차화로 공선성 제거). `betas_for(fx/semi/market)`
  → stress_test 주입용 연료. (6/14 "factor_exposure 미구현 발견" 해소.) test 13.
- ✅ **factor_stress 오케스트레이터** (`8caf832`, 6/15 2세션 = 순차 ①): factor_exposure ↔ stress_test
  연결 = 빠진 호출처. `run_factor_stress_test`가 베타 자동주입 → 휴면이던 S1(FX)/S2(반도체)/S5(복합)
  + 역사 H1~H5를 실평가. `factor_historical_shocks` = 스펙 §4.1 line230 "팩터노출 × 당시 팩터충격"
  근사 구현(역사 일자 팩터충격을 `build_factor_panel` 잔차화+센터링 패널에서 실측 → 종목 베타와
  단위 정합 = raw 충격을 잔차 베타에 곱하는 시장성분 이중계산 회피; 상장돼 그날 실제 종목수익률
  있으면 우선=스펙 1순위). ★factor_exposure·stress_test 자체 무변경(이 모듈만 둘을 import=격리).
  graceful: factor_returns가 H 일자 미포함(시계열 짧음)이면 그 H 생략. test 10(잔차화 정합·실측우선·graceful).
- 현재: freeze 유지·실주문0·write0·production 미배선(factor_returns 데이터 파이프라인은 별 트랙).
  전체 **28 failed/1588 passed/7 skipped**(신규 실패0; 28=베이스라인 645563a 측정과 동일 = 무관 입증).
- 다음 후보(순차 ②③): **② L3 실배선**(crowding·vol_targeting·stress_test/factor_stress →
  gross_exposure 노출관리 계층 = 사다리·vol·크라우딩 계수 곱, unfreeze 직전) / **③ VKOSPI·외인선물
  데이터 파이프라인**(crowding C2/C3 + ρ_crisis + factor_returns 일부의 공통 연료) / E(페이퍼 20일).
- ★worktree HEAD `8caf832` = origin/feature/risk-engine (push 후 0/0).
