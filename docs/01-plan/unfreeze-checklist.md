# Freeze 해제 룰 + Unfreeze 체크리스트

작성: 2026-06-11 (사장님 확정) · 상태: **활성 룰** · 연계: [RISK_ENGINE_SPEC_v2.md](RISK_ENGINE_SPEC_v2.md)

---

## 0. 룰 (협상 불가)

> **리스크 엔진(RISK_ENGINE_SPEC_v2 Phase 1)이 깔리고, 썩은 데이터가 매매 경로에 안 들어감이 확인되기 전엔 freeze를 해제하지 않는다.**

- **급하지 않다. 대신 freeze 해제의 전제조건이다.**
- 캘린더 데드라인 없음 → **실주문 재개(unfreeze)라는 *사건*이 데드라인**이다. 서두를 필요 없으나 건너뛸 수 없다.
- "나중에 하자"가 아니라 **"이거 없이는 실탄 장전 자체가 안 된다."**
- 토대(킬스위치·sizing·게이트)는 "있으면 좋은" 게 아니라 freeze 해제 *전에 반드시* 깔려 있어야 하는 바닥이다.

---

## 1. 의존 순서 (날짜가 아니라 의존관계)

| 시점 | 작업 | 비고 |
|------|------|------|
| 오늘 저녁 | 운영 런북만 ([evening_run]) | 확정, 변경 없음 |
| **6/12** | LIVE 7조건 판정 | **날짜에 묶인 유일 작업 → 무조건 선순위** |
| 다음 집중세션 | **Phase 1a** (freeze 완전 양립 구간) | 킬스위치 + sizing + 게이트 로직. ★6/12 직후로 너무 미루지 말 것(§3) |
| unfreeze 직전 | **Phase 1b** (실주문 경로) | execution 배선 + 우회불가 테스트 + fx-liquidity P0 |

**Phase 1a vs 1b 분리 기준** = 위험도. 1a는 "주문을 더 막는" 순수 추가(freeze 양립), 1b는 실주문 경로 수술.

---

## 2. ★ Unfreeze 체크리스트 (전부 ✅ 되어야 freeze 해제 — 하나라도 ❌면 실탄 장전 금지)

- [x] **A. Phase 1a 구현 완료** ✅ (6/12, 커밋 `5fffb4e`, 적대리뷰 14발견 수정·105 passed)
  - [x] `kill_switch/` 별도 프로세스 + `state.json` 영속화 (메인 봇 생사 무관 watchdog)
  - [x] `sizing.py` 리스크 예산 사이징 + 한국 갭 보정 + 하한가 생존 조건(`limit_down_survival_ok`)
  - [x] `pre_trade_gate.py` G3~G6 정적 한도 로직 (VaR 게이트 G1/G2는 Phase 2)
- [x] **B. 킬스위치 실제 발동 검증** ✅ (6/11~12 격리 경로, pytest + CLI 모의발동)
  - [x] K1 모의발동(전량청산+거래정지) 실제 실행 성공 — actions LIQUIDATE_ALL/HALT 선언(실행 배선=1b)
  - [x] `state.json` 영속 + **재부팅 후에도 발동 상태 유지** 확인 (test_full_scenario)
  - [x] 수동 해제 절차(확인코드) 작동 — 자동 복구 코드 0, ★키 부재 시 fail-closed raise
- [x] **C. Phase 1b-i: execution 배선(강제 자물쇠)** ✅ (6/12, 1안=REAL 라이브 경로만 강제 + mock 경고)
  - [x] 주문 함수가 `gate_result` PASS 토큰(서명+타임스탬프) 없이는 호출 불가하도록 구조적 강제
    — `KisOrderAdapter._guard`(4개 주문 진입점의 단일 초크포인트)의 BUY 최종 관문에
    `_enforce_gate_token` 추가. **REAL(_is_mock=False) BUY**는 검증된 통행증 없으면
    `PermissionError`. ★우회불가의 코드적 근거 = 실서버/모의서버를 가르는 `self._is_mock`과
    토큰 강제 여부를 가르는 변수가 **동일**(둘 다 `__init__`의 `MODEL!=REAL`) → '실서버로
    나가는데 토큰 미검사' 상태가 존재 불가. 검증 = 서명·타임스탬프·만료(300s)·미래발급·
    nonce replay(인스턴스 `_seen_gate_nonces`)·종목 일치·주문금액 ≤ 승인 사이즈. SELL은 면제
    (사전 게이트는 신규 리스크용, 매도는 리스크 감소 + 킬스위치 매도 지속 불변식 보존).
  - [x] **게이트 우회 주문이 코드상 불가능함을 테스트로 증명** — `tests/test_gate_token_enforcement.py`
    12케이스(무토큰/위조/만료/replay/종목불일치/사이즈초과/REJECT verdict 전부 raise + mock 경고만
    + SELL 미호출 구조 + `create_*_buy_order`가 `_guard` 경유 진입점에서만 호출되는 우회경로 grep 0).
    기존 `test_no_raw_mojito_order_bypass`도 무손상. 1안 채택 = mock/paper 경로는 무토큰 BUY 시
    "REAL이었다면 차단" 경고만 → 페이퍼 20일(E)이 게이트 배선 드라이런 증거가 됨.
  - 검증: 신규 12 + 가드레일 13 passed, 전체 회귀 **30 failed/1384 passed**(stash 베이스라인
    30 failed/1372 passed 대비 신규 실패 0 = 기존 날짜만료/stash 깨짐과 동일). freeze 무손상
    (REAL+ENABLED 둘 다 OFF라 추가된 raise는 현재 도달 불가 경로 위의 추가 차단).
  - [ ] **C-ii. Phase 1b-ii: 라이브 호출처 토큰 발급 배선** ⬅ 실제 unfreeze 전 남은 통합
    — 자물쇠(1b-i)는 완성. 단 **현재 어떤 라이브 호출처도 토큰을 발급·전달하지 않으므로**
    MODEL=REAL이면 모든 BUY가 fail-closed로 막힌다(안전하나 매수 불능). unfreeze 실작동을
    위해 live_trading / smart_entry(real) 경로가 `evaluate_pre_trade`로 통행증을 발급해
    `gate_result`로 넘기도록 배선해야 함. 다음 집중세션.
  - 비고: 토큰은 ★감사 로그 기록 성공 후에만 발급 + nonce replay 방지(verify에 seen_nonces)
    설계가 Phase 1a에 완료돼 있어 1b-i는 이 토큰을 주문 함수 필수 인자로 강제만 하면 됐음.
- [x] **D. fx-liquidity P0 — 썩은 데이터 매매경로 진입 여부 확정** ✅ (6/12, 21에이전트 read-only 조사)
  - [x] `chart_hero_executor.py`가 four_signal을 **어떻게 쓰는지 확정 = unused(휴면)**: stale CSV(`macro_four_signal_daily.csv` 5/19)를 읽는 코드 0개(write-only 로그) + chart_hero 파이프라인이 BAT/cron 어디에도 미배선·산출물 5/20 이후 정지. executor 자체는 four_signal 미참조(docstring만), 게이트는 surge_d1_picker 내부 hard gate지만 파이프라인 미가동.
  - [x] **결론: staleness 복구는 unfreeze blocker 아님**(GIGO 마실 입 닫힘). ⚠️단 향후 chart_hero **재배선 시** Gate 1이 hard gate로 살아나므로 그때 재점검(live API 3종 가용성 + CSV 공백 5/20~). fx-liquidity P0-1(파이프라인 사망원인 규명)은 별도 관측 인프라 과제.
- [ ] **E. (스펙 §8 공통)** 페이퍼 트레이딩 최소 20거래일 + 게이트/킬스위치 로그가 빠짐없이 남는지 확인 ⬅ 진행 중(FLOWX 관측 누적)

**현재 상태**: A·B·C(1b-i 자물쇠)·D ✅ / **C-ii(라이브 호출처 토큰 발급 배선)·E ⬅ 미완** → **unfreeze 아직 금지**. 게이트 자물쇠는 완성됐고, 남은 건 ① 라이브 매수 경로가 통행증을 발급·전달하도록 배선(C-ii) ② 페이퍼 20일 누적(E). E는 시간 누적.

> 흩어져 있던 미결(fx-liquidity P0, 킬스위치 테스트)이 여기 한 곳에 모인다. 따로 굴러다니다 잊히지 않게.

---

## 3. ★ 타이밍의 숨은 이점 — 킬스위치 모의발동은 freeze 중이 가장 안전

실주문 0인 **지금이 "전량 청산 + 거래 정지"를 실제로 발동시켜 볼 유일하게 부작용 없는 창**이다.
state.json 영속성·재부팅 생존·수동 해제 절차를 — 실탄 없이 — 검증할 수 있다.
**unfreeze 후에 킬스위치를 처음 테스트하는 건 소방훈련을 불난 뒤에 하는 것.**
→ Phase 1a(특히 킬스위치 B항목)를 **6/12 직후로 너무 미루지 말 것.** freeze 창이 닫히기 전에 소방훈련을 마쳐야 한다.

---

## 4. 한 줄

리스크 엔진은 수익 기능이 아니라 **freeze 해제의 자물쇠**다. 이 체크리스트가 그 열쇠의 이빨들이다.
