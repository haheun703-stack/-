# 5/20 자동매매 첫 가동 실전 매매 시나리오 + 대응 매트릭스

**작성일**: 2026-05-19 (화) D-21h
**대상일**: 2026-05-20 (수) 14:00 ~ 15:30 KST 가동
**사장님 지시 (5/19 13:50)**: "필드에서 매매할 때 어떻게 대처를 해야하는지 잘 확인하고 미리 생각하고 계산을 해놔야된다 이게 지금 해야되는 일이다"
**선행 문서**: `docs/01-plan/quant-auto-trading-test.md`, `docs/02-design/quant-auto-trading-p2-autonomy.md`
**검증 코드 베이스**: 커밋 `dd7d74f` (VWAP 게이트 + snapshot regime 저장)

---

## 0. 핵심 요약 (사장님이 5/20 가동 전 알아두면 좋은 3가지)

1. **사장님 손 0번이 기본** — 7명 검수팀 + Layer 7 KILL_SWITCH가 매수/매도/차단을 자동 결정. 사장님은 카톡을 받기만 함. 단, 핵심 RED 메시지가 오면 "수동 ssh 해제" 1회만 필요.
2. **최악 시나리오 손실 한도는 -3,000원 ± 수수료** (1주 10만원 한도 × -3% 절대 손절). 일일 1건 한도이므로 누적 손실도 같음.
3. **14:55까지 매수 0건도 정상** — 안전선 9건 ALL 통과해야 매수. 0건 마감 카톡이 오면 "TOP 3 후보 점수 어디까지 갔는지" 확인하면 됨. 회피한 매수가 -3%로 떨어졌으면 검수팀이 일을 잘한 것.

---

## 1. 가동 환경 사양 (불변)

| 항목 | 값 | 비고 |
|---|---|---|
| 가동일 | 2026-05-20 (수) | KRX 정규장 |
| 매수 가동 시간 | 14:00 ~ 14:55 (12회 cron) | 5분 간격 |
| 청산 모니터링 | 09:00 ~ 15:30 (5분 간격) | owner_rule_monitor |
| 강제 청산 시각 | 15:20 KST | 룰 ③, NXT 안전마진 |
| 수량 한도 | 1주 / 종목 | AUTO_TRADING_MAX_QTY=1 |
| 금액 한도 | 100,000원 / 1주 | should_auto_buy 안전선 ⑤ |
| 일일 매수 한도 | 1건 | MAX_DAILY_BUYS=1 |
| 계좌 | KIS 47339014-01 | MODEL=REAL |
| 화이트리스트 | OFF | AUTO_TRADING_WHITELIST_ONLY=0 (5/18 결단) |
| 가격 편차 | ±5% | AUTO_TRADING_PRICE_RANGE_PCT=5 |
| paper mirror | 병행 ON | PAPER_MIRROR_MODE=true (옵션 B) |

### 수수료/세금 가정 (1주 ≤ 10만원)
- KIS 수수료: 0.015% (한투 우대 가정)
- 매도 시 거래세: 0.18% (코스피 0.18%, 코스닥 0.18%, 5/20 기준)
- 슬리피지: paper mirror 가정 +0.05% (매수) / -0.05% (매도)
- **왕복 비용**: 약 0.36% (수수료 0.03% 왕복 + 세금 0.18% 매도 + 슬리피지 0.10% 왕복 추정)
- **손익분기**: 약 +0.36% 이상 상승 시 수익 발생

---

## 2. 사장님 룰 (청산 규칙, 불변)

| 룰 | 조건 | 동작 | 코드 |
|---|---|---|---|
| ① 절대 손절 | 진입가 대비 ≤ -3.00% | SELL_STOP_LOSS (시장가 매도) | `src/use_cases/owner_rule.py:106` |
| ② 트레일링 | 활성화 후 peak 대비 ≤ -3.00% | SELL_TRAILING (시장가 매도) | `src/use_cases/owner_rule.py:119` |
| ③ 강제 청산 | 현재시각 ≥ 15:20 | SELL_FORCE_CLOSE (시장가 매도) | `src/use_cases/owner_rule.py:136` |
| ④ 이월 | 룰 ③ 직전 5조건 ALL 통과 | HOLD_OVERNIGHT (청산 SKIP) | `src/use_cases/owner_rule.py:164` |

**트레일링 활성화 조건**: 진입가 대비 +3.00% 도달 시 (한 번 활성화되면 종료까지 유지)

**룰 ④ 이월 5조건** (15:20 시점):
1. 현재가 ≥ 진입가 +3.00% 이상 양봉
2. 외인+기관+연기금 매수 누적 ≥ +1.0억
3. EYE 필터 5종 PASS
4. peak 대비 -3% 이내 (트레일링 안전)
5. 보유 일수 < 5일

---

## 3. 7명 검수팀 자동 흐름 (사장님 개입 0건 기준)

| 시각 | 워커 | 동작 | FAIL 시 |
|---|---|---|---|
| 06:00 | (cron) | KILL_SWITCH 자동 삭제 (전날 잔재 제거) | — |
| 06:02 | EnvChecker | 환경변수 19개 점검 | KILL_SWITCH 자동 활성화 → 매수 차단 |
| 06:30 | DataIntegrity | BAT/정보봇/KIS 16개 데이터 점검 | KILL_SWITCH 자동 활성화 |
| 13:55 | EnvChecker | 가동 5분 전 재점검 | KILL_SWITCH 자동 활성화 |
| 13:55 | CodeAuditor | 코드 무결성 점검 | KILL_SWITCH 자동 활성화 |
| 13:55 | MarketRegimeGate | KODEX 200/인버스/2X 폭락장 검출 | BEARISH → KILL_SWITCH 자동 활성화 |
| 14:00~14:55 | auto_buy_executor | 매 5분 매수 평가 (12회) | KILL_SWITCH 있으면 즉시 종료 |
| 14:00~15:55 | FlowMonitor | 매매 흐름 8단계 추적 | 이상 검출 시 KILL_SWITCH |
| 09:00~15:30 | MarketScanner | 5분 시장 스캔 (KILL_SWITCH 활성화 안 함, 정보용) | — |
| 14:35~15:30 | owner_rule_monitor | 매 5분 청산 평가 | — |
| 15:20 | owner_rule_monitor | 룰 ④ 이월 평가 + 룰 ③ 강제 청산 | — |
| 16:00 | (cron) | KILL_SWITCH 자동 복구 (5/21 06:00까지) | — |

**카톡 통제 (5/19 결단 C)**:
- `AGENT_TELEGRAM_ENABLED=false` (디폴트) → 워커 평상시 카톡 OFF
- KILL_SWITCH RED 활성화 → `kill_switch_manager`가 단일 카톡 발송
- 매수/매도 실주문 체결 → `KisOrderAdapter._send_telegram_alert` 별도 카톡

---

## 4. 안전선 9건 (auto_buy_decider)

| # | 안전선 | 기준 | 검증 위치 |
|---|---|---|---|
| ① | 통합 점수 STRONG 90+ | `integrated_score >= 90.0` | should_auto_buy |
| ② | EYE 필터 5종 PASS | `eye_should_skip == False` | should_auto_buy |
| ③ | 14:00 이후 진입 | `now >= "14:00"` | should_auto_buy |
| ④ | 일일 1건 한도 | `today_entries < 1` | should_auto_buy |
| ⑤ | 1주 10만원 한도 | `current_price <= 100,000` | should_auto_buy |
| ⑥ | 시장 regime ∈ {MILD_BULL, NEUTRAL, STRONG_BULL} | snapshot regime | should_auto_buy |
| ⑦ | AUTO_TRADE_5_20=true | env var | should_auto_buy |
| ⑧ | 막내 NEGA 0건 | 5/20은 미적용 (5/21+) | should_auto_buy |
| ⑨ | 지정가 현재가 ±5% | `KisOrderAdapter._guard` | kis_order_adapter |
| (별도) | VWAP 정상 | 과열 -2.5%↑ 차단 | auto_buy_executor.check_vwap_gate |

---

## 5. 시나리오 매트릭스 (총 24건)

### 5-1. 정상 흐름 (S-01 ~ S-05) — 5건

---

#### S-01: 14:00 매수 후 +5% 상승 → 14:45 peak -3% 트레일링 청산 → 수익 +2%

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 → 14:25 peak +5% → 14:45 peak 대비 -3% 도달 |
| 시스템 자동 대응 | (1) 14:00 auto_buy_executor: 안전선 9건 ALL PASS → buy_limit → PENDING → 체결<br>(2) 14:25 owner_rule_monitor: pnl_pct=+5% ≥ +3% → trailing_active=True<br>(3) 14:45 owner_rule_monitor: peak 대비 -3% → SELL_TRAILING → create_market_sell_order |
| 손익 계산 | 진입 50,000원 × 1주 = 50,000원<br>peak 52,500원 → 청산 50,925원 (peak -3%)<br>매수 수수료: 50,000 × 0.015% = 7.5원<br>매도 수수료: 50,925 × 0.015% = 7.6원<br>거래세: 50,925 × 0.18% = 91.7원<br>**손익: +925원 - 106원 = +819원 (수익률 +1.64%)** |
| 사장님 카톡 | (1) 14:00 ✅ 매수 성공 (지정가 50,000원 × 1주 = 50,000원)<br>(2) 14:45 🟡 SELL_TRAILING 청산 (peak 52,500 → 현재 50,925, peak -3.00%, 이익 보존 +1.85%)<br>(3) KisOrderAdapter `매도 접수` 카톡 |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | paper mirror 동기 시뮬 (별도 데이터). 16:30 BAT-D 후 D+1 학습 데이터 라벨링. |

**paper mirror 시뮬 결과**: 슬리피지 +25원 매수, -25원 매도 가정 → paper 손익 약 +769원
**자기반성 학습 데이터**: D+1/D+3/D+5 라벨링 가능 (5/21~5/26 종가 추적)
**5/20 가동 후 분석 포인트**: peak 대비 -3% 룰이 너무 타이트한지(+5% 익절 권장 vs 트레일링) 비교

---

#### S-02: 14:30 매수 후 -2% 하락 → 15:20 강제 청산 → 손실 -2%

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:30 매수 → 15:15까지 -2% 박스권 → 15:20 도달 → 룰 ④ 이월 조건 미달 |
| 시스템 자동 대응 | (1) 14:30 매수 (안전선 9건 ALL PASS)<br>(2) 15:00~15:15 owner_rule_monitor: pnl_pct=-2% > -3% → HOLD<br>(3) 15:20 owner_rule_monitor: current_time >= "15:20" → SELL_FORCE_CLOSE 평가<br>(4) evaluate_hold_overnight: pnl_pct < +3% → ❌ 이월 미달 → 강제 청산 진행 |
| 손익 계산 | 진입 80,000원 × 1주<br>청산 78,400원 (-2%)<br>수수료 합 24원, 거래세 141원<br>**손익: -1,600원 - 165원 = -1,765원 (수익률 -2.21%)** |
| 사장님 카톡 | (1) 14:30 ✅ 매수 성공<br>(2) 15:20 ⏰ SELL_FORCE_CLOSE (현재 -2.00%, 이월 조건 미달) |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 학습 데이터: "강제 청산 후 5/21 시초가 vs 청산가" 비교 → 청산이 옳았는지 검증 |

**paper mirror 시뮬 결과**: 동일 흐름, 슬리피지 가정 차이로 -1,820원 추정
**자기반성 학습 데이터**: D+1 시초가 추적 시 strong gap-up이면 룰 ③ 너무 일찍이라는 지표
**5/20 가동 후 분석 포인트**: 15:20~15:30 동안 회복했을 가능성 (NXT 안전마진이 과한지)

---

#### S-03: 14:00 매수 후 +3% 상승 후 횡보 → 15:20 강제 청산 → 수익 +3%

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 → 14:30 +3% 도달 (트레일링 활성화) → 15:15 peak +3.2% → 15:20 도달 → 이월 미달 |
| 시스템 자동 대응 | (1) 14:00 매수<br>(2) 14:30 trailing_active=True<br>(3) 15:00~15:15 peak -1~-2% (트레일링 미발동, -3% 미달)<br>(4) 15:20 SELL_FORCE_CLOSE 평가 → 이월 미달 (pnl_pct +2.8% < +3.0% OR 수급 데이터 부재) → 강제 청산 |
| 손익 계산 | 진입 70,000원, 청산 72,100원 (+3.0%)<br>수수료 합 21원, 거래세 130원<br>**손익: +2,100원 - 151원 = +1,949원 (수익률 +2.78%)** |
| 사장님 카톡 | (1) 14:00 ✅ 매수<br>(2) 15:20 ⏰ SELL_FORCE_CLOSE (+3.00%, 이월 미달) |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 첫 출격 수익! paper mirror 비교, D+1~D+5 추적. |

**paper mirror 시뮬 결과**: 슬리피지 영향 약 -50원 → paper 약 +1,899원
**자기반성 학습 데이터**: 트레일링 활성화 + 강제 청산 시 D+1 갭다운 빈도 추적
**5/20 가동 후 분석 포인트**: 룰 ④ 이월 5조건이 너무 빡빡한지 (대부분 강제 청산으로 가는지)

---

#### S-04: 14:00 매수 후 수급 지속 → 15:20 룰 ④ 이월 → 5/21 09:00 재가동

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 → 15:15 peak +5% → 외인+기관 매수 +2.5억 → EYE PASS → 이월 5조건 ALL OK |
| 시스템 자동 대응 | (1) 14:00 매수<br>(2) 15:20 owner_rule_monitor: SELL_FORCE_CLOSE 진입<br>(3) evaluate_hold_overnight 5조건 ALL PASS → can_hold=True<br>(4) positions[ticker]["entry_date"] 유지, days_held 갱신, last_hold_check 기록<br>(5) 청산 SKIP, continue 다음 종목 |
| 손익 계산 | 5/20 시점: 미실현 +5% (체결 X) |
| 사장님 카톡 | (1) 14:00 ✅ 매수 성공<br>(2) 15:20 🌙 [사장님 룰 ④ — 익일 이월] PnL +5.00%, 수급 +2.5억, EYE PASS, → 5/21 익일 보유 (최대 4일 더) |
| 사장님 개입 | **0건 (완전 자동)** — 단, 5/21 09:00 시초가 갭다운 -3% 가능성 인지 |
| 후속 대응 | 5/21 09:00부터 owner_rule_monitor 자동 가동 (S-18 시나리오로 분기) |

**paper mirror 시뮬 결과**: 동일 이월 결정. 5/21 시초가 시뮬으로 검증.
**자기반성 학습 데이터**: 이월 5조건 적중률 추적 (D+1 수익률 +X% 이상 가능성)
**5/20 가동 후 분석 포인트**: 이월 결정의 적중률. 5/21 갭다운 빈도가 높으면 조건 강화 필요.

---

#### S-05: 14:00~14:55 모든 cron 안전선 미충족 → 매수 0건 → 손실 0

| 항목 | 내용 |
|---|---|
| 발동 조건 | 12회 cron 모두 안전선 9건 중 1개 이상 미달 (예: 점수 88, regime CAUTION, VWAP 과열) |
| 시스템 자동 대응 | (1) 14:00~14:50: 매 cron마다 후보 9건 평가 → 모두 SKIP<br>(2) 14:55 마지막 cron: now_hhmm >= "14:55" + buy_executed=False → 0건 마감 카톡 발송 |
| 손익 계산 | **손익 0원** (매수 X) |
| 사장님 카톡 | 14:55 ⏭️ [자동매수 0건] 14:00~14:55 통과 후보 없음<br>TOP 3 (참고): HPSP 점수 88 (regime CAUTION), 삼성E&A 점수 87 (VWAP 과열), ...<br>+ 정보봇 ETF 신뢰도 어필 |
| 사장님 개입 | **0건 (완전 자동)** — TOP 3 메시지만 참고. 5/21 picks 기대. |
| 후속 대응 | 5/21 picks_v2 갱신 후 다시 평가. **0건 마감은 정상**(검수팀이 위험을 회피한 것). |

**paper mirror 시뮬 결과**: paper도 0건. 시뮬 비교 데이터 X.
**자기반성 학습 데이터**: TOP 3 후보의 5/20 D+1 종가 추적 → 회피가 옳았는지 (예: -2%로 떨어졌으면 검수팀 적중)
**5/20 가동 후 분석 포인트**: 안전선 9건 중 어떤 게 가장 자주 차단했는지 통계 → 향후 조정 후보

---

### 5-2. 사고 대응 (S-06 ~ S-13) — 8건

---

#### S-06: 14:00 매수 직후 즉시 -3% 폭락 → 룰 ① 손절 → 손실 -3%

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 후 14:05 단일 봉 -3% 도달 (예: 외인 대량 매도, 거래소 공시 악재) |
| 시스템 자동 대응 | (1) 14:00 매수<br>(2) 14:05 owner_rule_monitor: pnl_pct=-3.05% <= -3% → SELL_STOP_LOSS → 시장가 매도 |
| 손익 계산 | 진입 50,000원, 청산 48,500원 (-3%)<br>수수료 합 15원, 거래세 87원<br>**손익: -1,500원 - 102원 = -1,602원 (수익률 -3.20%)**<br>**최악 시나리오 손실 한도** |
| 사장님 카톡 | (1) 14:00 ✅ 매수 성공<br>(2) 14:05 🔴 [사장님 룰 자동 청산] SELL_STOP_LOSS (-3.05%, "진입가 -3% 절대 손절") |
| 사장님 개입 | **0건 (완전 자동)** — 단, 사장님 멘탈 관리 필요 (첫 매매에서 손절은 자연스러움) |
| 후속 대응 | 5/20 일일 1건 한도 도달했으므로 추가 매수 없음. 5/21 picks 갱신 대기. |

**paper mirror 시뮬 결과**: 동일 손절 결정, 슬리피지 가정으로 약 -1,650원
**자기반성 학습 데이터**: 진입 후 5분 내 -3% 폭락의 패턴 분석 (시간대/섹터/수급 신호)
**5/20 가동 후 분석 포인트**: 즉시 폭락 종목의 14:00 직전 신호 (VWAP 과열, 거래량 급증, 호가 이상)

---

#### S-07: 14:00 매수 후 거래정지 (current_price=0) → E2 가드 HOLD → 다음 cron 재시도

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 → 14:10 거래정지 (예: 거래소 공시, 조회공시 요구) → fetch_price `stck_prpr=0` |
| 시스템 자동 대응 | (1) 14:00 매수 정상<br>(2) 14:10 owner_rule_monitor: current_price=0 → owner_rule.evaluate_owner_rule 가드 발동 (line 82) → HOLD ("현재가 0 (거래정지/응답 누락) — 매도 보류, 다음 cron 재시도")<br>(3) 14:15, 14:20, ... 매 5분 재시도<br>(4) 15:20 도달 시에도 거래정지면 SELL_FORCE_CLOSE도 0원에 매도 불가 → HOLD 지속<br>(5) 16:00 KILL_SWITCH 자동 복구 — 5/21로 이월 (의도치 않은) |
| 손익 계산 | 5/20 미실현 손익 = 진입가 대비 거래정지 시점 (정확값 불명)<br>5/21 거래 재개 시 시초가 결정 → 손익 확정 |
| 사장님 카톡 | (1) 14:00 ✅ 매수 성공<br>(2) 거래정지 동안 카톡 없음 (HOLD는 알림 X)<br>(3) 15:20 ⏰ SELL_FORCE_CLOSE 시도 → 실패 (현재가 0)<br>(4) ★ **사장님 수동 확인 필요** — 다음날 시세 회복 후 처리 |
| 사장님 개입 | **수동 확인 권장** — 거래정지 사유 (악재? 권리락? 공시?) 파악. 5/21 거래 재개 후 청산 가능. |
| 후속 대응 | 5/21 09:00 거래 재개 시 owner_rule_monitor가 자동 청산 (룰 ① -3% 또는 시초가 청산) |

**paper mirror 시뮬 결과**: paper도 동일하게 HOLD (가격 0). 시뮬 의미 X.
**자기반성 학습 데이터**: 거래정지 종목의 사전 시그널 (공시 캘린더, 호가 이상)
**5/20 가동 후 분석 포인트**: 거래정지가 발생할 경우 사장님 수동 알림이 필요한지 (자동 텔레그램 추가 검토)

---

#### S-08: 매수 시도 후 KIS 응답 PERMISSION/RATE → 텔레그램 알람 + 다음 cron 재시도

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 buy_limit 호출 → KIS API `rt_cd=1, msg1="요청 횟수 초과"` 또는 `"권한 없음"` |
| 시스템 자동 대응 | (1) `_parse_order_response`: rt_cd != "0" → status=FAILED → message=KIS 응답<br>(2) auto_buy_executor: ok=False → 텔레그램 발송 `❌ [자동매수 실패]`<br>(3) continue → 다음 종목 평가 (해당 cron 내)<br>(4) 모든 후보 실패 시 break 없이 다음 5분 cron 자동 재시도 |
| 손익 계산 | **손익 0원** (체결 X) |
| 사장님 카톡 | 14:00 ❌ [자동매수 실패] HPSP(403030)<br>지정가 52,300원 1주<br>사유: 요청 횟수 초과<br>→ 다음 5분 cron 재시도 |
| 사장님 개입 | **수동 확인 권장** — RATE LIMIT은 5분 후 자동 해소. PERMISSION은 KIS 키 또는 IP 화이트리스트 확인 필요. |
| 후속 대응 | 14:05/14:10 자동 재시도. 14:55까지 모두 실패 시 0건 마감 카톡. |

**paper mirror 시뮬 결과**: paper는 KIS 무관하므로 동일 시그널로 정상 시뮬 진행
**자기반성 학습 데이터**: KIS RATE LIMIT 빈도 추적 → CachedBroker TTL=60s 효과 검증
**5/20 가동 후 분석 포인트**: PERMISSION 오류 시 즉시 KILL_SWITCH 활성화 추가 필요할지

---

#### S-09: 매수 성공인데 positions.json 저장 실패 → FlowMonitor critical → KILL_SWITCH

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 KIS 매수 PENDING 성공 → save_positions_state 시 디스크 I/O 실패 (권한, 디스크 풀) |
| 시스템 자동 대응 | (1) buy_limit 성공 → order.status=PENDING<br>(2) positions[tk] = {...} 인메모리만<br>(3) save_positions_state: 예외 발생 → auto_buy_executor 종료<br>(4) 14:05 FlowMonitor: KIS 체결은 확인되는데 owner_rule_positions.json에 미기록 → 불일치 감지<br>(5) FlowMonitor → activate_kill_switch ("FlowMonitor: KIS 체결 vs positions.json 불일치") |
| 손익 계산 | 미실현 (체결은 됐는데 시스템이 모름)<br>**즉시 owner_rule_monitor 청산 불가** — entry_price 없음 |
| 사장님 카톡 | (1) 14:00 ✅ KisOrderAdapter `매수 접수` 카톡 (별도 경로, 정상 발송)<br>(2) 14:05 🚨 [자동 차단 발동] FlowMonitor FAIL — 사유: KIS 체결 vs positions.json 불일치 → KILL_SWITCH 활성화됨<br>(3) **즉시 ssh 필요** |
| 사장님 개입 | **즉시 ssh 필요** — 1) KIS 잔고 확인 2) positions.json 수동 INSERT 또는 시장가 매도 수동 처리 |
| 후속 대응 | 5/21 가동 전 디스크 용량 점검, 권한 점검 |

**paper mirror 시뮬 결과**: paper는 별도 디렉터리(`data/paper_mirror/`)이므로 디스크 문제 동시 발생 가능
**자기반성 학습 데이터**: 디스크 I/O 오류 빈도 (드물지만 발생 시 critical)
**5/20 가동 후 분석 포인트**: positions.json 저장을 트랜잭션(임시파일 → rename)으로 강화할지

---

#### S-10: KIS 토큰 만료 (장중 발생) → fetch_price 실패 → MarketRegimeGate fail-safe → KILL_SWITCH

| 항목 | 내용 |
|---|---|
| 발동 조건 | 13:55 MarketRegimeGate 가동 시 KIS 토큰 23h59m 경과로 만료 직전 → fetch_price 401 |
| 시스템 자동 대응 | (1) MarketRegimeGate.check_market_regime: 3종목 중 ≥2건 fetch 실패<br>(2) C3-B fail-safe 발동 (`market_regime_gate.py:189`) → status=FAIL, regime=UNKNOWN<br>(3) activate_kill_switch ("MarketRegimeGate fail-safe: 3/3 fetch 실패")<br>(4) 14:00 auto_buy_executor: KILL_SWITCH 존재 → return 0 (매수 SKIP) |
| 손익 계산 | **손익 0원** (매수 X) |
| 사장님 카톡 | 13:55 🚨 [자동 차단 발동] MarketRegimeGate FAIL — 사유: 3/3 fetch 실패 → KILL_SWITCH 활성화됨 |
| 사장님 개입 | **즉시 ssh 권장** — `python -c "from src.adapters.kis_stock_data_adapter import KisStockDataAdapter; KisStockDataAdapter()"` 로 토큰 갱신 + `rm ~/quantum-master/data/KILL_SWITCH` |
| 후속 대응 | mojito 라이브러리가 자동 토큰 갱신 — 일반적으로 6시간마다 갱신. 5/21 가동 전 점검. |

**paper mirror 시뮬 결과**: paper도 fetch 실패 시 entry 시뮬 불가
**자기반성 학습 데이터**: 토큰 만료 빈도 추적 (실측 6시간 갱신 패턴)
**5/20 가동 후 분석 포인트**: KIS 토큰 자동 갱신 헬스체크 추가 (06:00, 13:55)

---

#### S-11: 14:00 매수 후 +10% 폭등 → peak 갱신 → -3% 트레일링 → 수익 +7%

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 매수 → 14:30 +10% peak → 14:50 peak 대비 -3% (절대 +7%) |
| 시스템 자동 대응 | (1) 14:00 매수<br>(2) 14:15 pnl +5% → trailing_active=True<br>(3) 14:30 peak 55,000원 갱신<br>(4) 14:50 현재가 53,350원 → peak 대비 -3.00% → SELL_TRAILING |
| 손익 계산 | 진입 50,000원, peak 55,000원, 청산 53,350원<br>수수료 합 15원, 거래세 96원<br>**손익: +3,350원 - 111원 = +3,239원 (수익률 +6.48%)**<br>**최선 시나리오 1주 기준** |
| 사장님 카톡 | (1) 14:00 ✅ 매수<br>(2) 14:50 🟡 SELL_TRAILING (peak 55,000 → 현재 53,350, peak -3.00%, 이익 보존 +6.70%) |
| 사장님 개입 | **0건 (완전 자동)** — 첫 매매 수익! |
| 후속 대응 | 트레일링이 너무 일찍 끝났는지 D+1 종가 확인 (계속 상승했으면 ✓ 사람 직관 + 트레일링은 보수적) |

**paper mirror 시뮬 결과**: 슬리피지 영향으로 약 +3,150원
**자기반성 학습 데이터**: +10% 폭등 종목의 14:00 직전 시그널 (역으로 학습)
**5/20 가동 후 분석 포인트**: 트레일링 -3%가 너무 빠른지 (peak에서 -5% 트레일링 vs +상승 추적 비교)

---

#### S-12: 1주 10만원 한도 초과 → KIS 어댑터 _guard 차단 → 다음 종목 재평가

| 항목 | 내용 |
|---|---|
| 발동 조건 | tomorrow_picks 1순위 종목 현재가 110,000원 (예: 삼성바이오로직스, 한미반도체) |
| 시스템 자동 대응 | (1) 14:00 auto_buy_executor: candidate=A 평가<br>(2) should_auto_buy 안전선 ⑤: estimated_amount 110,000 > 100,000 → SKIP<br>(3) decisions_log에 SKIP 기록 → continue<br>(4) candidate=B 평가 (다음 종목) |
| 손익 계산 | candidate A: 손익 0원 (매수 X) |
| 사장님 카톡 | candidate A는 SKIP 콘솔 로그만 (카톡 X). 14:55까지 매수 0건이면 0건 마감 카톡에 TOP 3로 표시. |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 가격이 ≤10만원인 강력포착 종목이 후보에 있어야 매수 가능. tomorrow_picks 생성 시 가격 필터 추가 검토. |

**paper mirror 시뮬 결과**: paper도 동일 SKIP
**자기반성 학습 데이터**: 10만원 한도로 SKIP된 종목 비율 → 향후 한도 조정 근거
**5/20 가동 후 분석 포인트**: 강력포착 9건 중 ≤10만원 종목 비율 통계

---

#### S-13: 13:55 EnvChecker FAIL (예: AUTO_TRADE_5_20=false) → KILL_SWITCH → 14:00 매수 0건

| 항목 | 내용 |
|---|---|
| 발동 조건 | 사장님이 .env에 `AUTO_TRADE_5_20=false`로 토글하고 가동 시작 (또는 PAPER_MIRROR_MODE 미설정) |
| 시스템 자동 대응 | (1) 13:55 EnvChecker: AUTO_TRADE_5_20=false 검출 → FAIL<br>(2) activate_kill_switch ("EnvChecker FAIL: AUTO_TRADE_5_20=false")<br>(3) 14:00 auto_buy_executor: KILL_SWITCH 존재 → return 0<br>(4) 14:05~14:55 모든 cron 동일하게 즉시 종료 |
| 손익 계산 | **손익 0원** |
| 사장님 카톡 | 13:55 🚨 [자동 차단 발동] EnvChecker FAIL — 사유: AUTO_TRADE_5_20=false (현재 false) → KILL_SWITCH 활성화됨 |
| 사장님 개입 | **즉시 ssh 권장** — .env에서 `AUTO_TRADE_5_20=true` 설정 + `rm ~/quantum-master/data/KILL_SWITCH` |
| 후속 대응 | EnvChecker 14:00 재가동 시 자동 통과 → KILL_SWITCH 자동 활성화 안 함. 매수 가능. |

**paper mirror 시뮬 결과**: paper도 동일하게 0건 (env 가드)
**자기반성 학습 데이터**: 환경변수 토글 휴먼 에러 빈도
**5/20 가동 후 분석 포인트**: EnvChecker 점검 항목이 12개 → 실제 매수 차단의 결정 요인 비율

---

### 5-3. 시장 상황별 (S-14 ~ S-17) — 4건

---

#### S-14: 5/19 같은 폭락장 (코스피 -4%) → MarketRegimeGate BEARISH → KILL_SWITCH → 매수 0건

| 항목 | 내용 |
|---|---|
| 발동 조건 | 5/20 13:55 시점: KODEX 200 -2.5%, KODEX 인버스 +2.5%, KODEX 200선물인버스2X +5.0% |
| 시스템 자동 대응 | (1) 13:55 MarketRegimeGate.check_market_regime:<br>  - KODEX 200(069500) -2.5% ≤ -2.0% → trigger ✓<br>  - KODEX 200선물인버스2X(252670) +5.0% ≥ +5.0% → trigger ✓<br>  - triggered=2 → regime=STRONG_BEARISH<br>(2) activate_kill_switch ("시장 약세 검출: KODEX 200 -2.50%, KODEX 200선물인버스2X +5.00%")<br>(3) 14:00~14:55 모든 cron 즉시 종료 |
| 손익 계산 | **손익 0원** (매수 차단됨) |
| 사장님 카톡 | 13:55 🚨 [자동 차단 발동] MarketRegimeGate FAIL — 사유: 시장 약세 검출: KODEX 200 -2.50%, KODEX 200선물인버스2X +5.00% → KILL_SWITCH 활성화됨 |
| 사장님 개입 | **0건 (완전 자동)** — 차단이 정상 동작. 사장님이 "오 검수팀이 일하네" 안심. |
| 후속 대응 | 5/19 사례처럼 폭락장 회피 → 손실 -3,000원 방지 |

**paper mirror 시뮬 결과**: paper도 매수 시뮬 X (KILL_SWITCH 가드 동일)
**자기반성 학습 데이터**: 폭락장 검출 정확도 — 5/20 종가 KODEX 200 회복 여부로 검증
**5/20 가동 후 분석 포인트**: BEARISH 임계값 (KODEX 200 -2%, 인버스 +3%, 2X +5%)이 너무 보수적인지

---

#### S-15: 강세장 (코스피 +2%) → 정상 매수 → +3~+5% 수익 가능성

| 항목 | 내용 |
|---|---|
| 발동 조건 | 5/20 14:00 KODEX 200 +1.5%, KODEX 인버스 -1.5%, regime=MILD_BULL |
| 시스템 자동 대응 | (1) 13:55 MarketRegimeGate: triggered=0 → NORMAL<br>(2) snapshot regime=MILD_BULL 저장 (5/18 자아성찰 #5 fix)<br>(3) 14:00 auto_buy_executor: 안전선 ⑥ regime ∈ MILD_BULL OK<br>(4) 안전선 9건 PASS → buy_limit → 정상 매수<br>(5) 종일 +3~+5% 상승 → S-01 또는 S-04 (이월)로 분기 |
| 손익 계산 | S-01 또는 S-04 시나리오와 동일 |
| 사장님 카톡 | S-01 또는 S-04와 동일 |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 강세장 추적, 첫 매매 성공 패턴 학습 |

**paper mirror 시뮬 결과**: 동일 시그널로 시뮬 (슬리피지 차이만)
**자기반성 학습 데이터**: 강세장 환경에서 안전선 9건 통과율
**5/20 가동 후 분석 포인트**: regime=MILD_BULL/STRONG_BULL 시 적중률 비교

---

#### S-16: 횡보장 (코스피 ±0.5%) → 매수 후 ±1% 등락 → 15:20 강제 청산 또는 이월

| 항목 | 내용 |
|---|---|
| 발동 조건 | 5/20 종일 KODEX 200 ±0.5%, 매수 종목도 ±1% 박스권 |
| 시스템 자동 대응 | (1) 14:00 매수 (regime=NEUTRAL → 안전선 ⑥ PASS)<br>(2) 15:00~15:15 owner_rule_monitor: pnl_pct ±1% → HOLD (-3% 미달)<br>(3) 15:20 SELL_FORCE_CLOSE → 이월 미달 (pnl_pct < +3%) → 강제 청산 |
| 손익 계산 | ±1% 수준 → 수수료/거래세 차감 시 **약 -200~+700원** |
| 사장님 카톡 | (1) 14:00 ✅ 매수<br>(2) 15:20 ⏰ SELL_FORCE_CLOSE (pnl ±1%) |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 횡보장에서 수익 가능성 낮음 → MarketRegimeGate에 NEUTRAL 차단 옵션 검토 |

**paper mirror 시뮬 결과**: 슬리피지로 paper는 -500원 정도 (왕복 비용 영향 큼)
**자기반성 학습 데이터**: 횡보장에서 자동매매 EV (기댓값) → 음수면 차단 로직 추가 검토
**5/20 가동 후 분석 포인트**: regime=NEUTRAL 매수의 손익분기점 추적

---

#### S-17: 갭다운 시초가 (-5%) → 매수 시점 이미 -3% → 룰 ① 즉시 손절

| 항목 | 내용 |
|---|---|
| 발동 조건 | 5/20 09:00 시초가 -5% (밤사이 미국 -3% 폭락) → 13:55 회복 -2% → 14:00 매수 → 14:10 -3% 재하락 |
| 시스템 자동 대응 | (1) 13:55 MarketRegimeGate: KODEX 200 -2.0% ≤ -2.0% → trigger ✓ (-2.0% 경계, 정확히 -2.0%이면 ≤ 만족) → BEARISH → KILL_SWITCH<br>(2) 14:00 매수 SKIP (KILL_SWITCH 존재) |
| 손익 계산 | **손익 0원** (매수 차단됨, KILL_SWITCH 덕분) |
| 사장님 카톡 | 13:55 🚨 [자동 차단 발동] MarketRegimeGate FAIL — 사유: KODEX 200 -2.00% ≤ -2.0% → KILL_SWITCH 활성화됨 |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 갭다운 차단이 BEARISH 임계값 결정 요인. -2.0% 경계가 너무 빡빡하면 -2.5%로 조정 검토. |

**paper mirror 시뮬 결과**: paper도 KILL_SWITCH 가드 동일
**자기반성 학습 데이터**: 갭다운 후 회복 빈도 (5분/30분/1시간) → 차단 임계값 검증
**5/20 가동 후 분석 포인트**: KODEX 200 -2.0% 경계 케이스 — 14:00 시점 -1.5%까지 회복 시 BEARISH 유지가 옳은지

---

### 5-4. 이월 시나리오 (S-18 ~ S-20) — 3건

---

#### S-18: 5/20 이월 → 5/21 09:00 보유 시작 → 09:30 +2% → 15:20 매도 → 수익

| 항목 | 내용 |
|---|---|
| 발동 조건 | S-04 후속. 5/20 15:20 이월 결정 → 5/21 09:00 owner_rule_monitor 가동 |
| 시스템 자동 대응 | (1) 5/21 09:00 owner_rule_monitor: positions[tk] 로드 (entry_price=5/20 매수가)<br>(2) 09:30 현재가 +2% (vs 5/20 진입가) → HOLD<br>(3) 14:00 +3.5% peak<br>(4) 15:20 SELL_FORCE_CLOSE 평가 → 이월 5조건 재평가:<br>  - days_held=1 < 5 ✓<br>  - pnl_pct +3.5% ≥ +3.0% ✓<br>  - 수급 ... (BAT-D 후 정확값)<br>(5) 이월 가능하면 5/22로 이월, 미달이면 청산 |
| 손익 계산 | 5/20 진입 50,000원 → 5/21 청산 51,750원 (+3.5%)<br>**손익: +1,750원 - 111원 = +1,639원 (+3.28%)** |
| 사장님 카톡 | 5/21 15:20 ⏰ SELL_FORCE_CLOSE (또는 🌙 추가 이월) |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 다일 보유 패턴 검증, 5일 한도 도달 시 강제 청산 |

**paper mirror 시뮬 결과**: 5/20~5/21 동기 시뮬, 슬리피지 누적
**자기반성 학습 데이터**: D+1 수익률 분포 (이월 결정의 적중률)
**5/20 가동 후 분석 포인트**: 이월 후 D+1 갭다운 빈도 vs D+1 추가 상승 빈도

---

#### S-19: 5/20 이월 → 5/21 갭다운 (-5%) → 룰 ① 손절

| 항목 | 내용 |
|---|---|
| 발동 조건 | S-04 후속. 5/21 09:00 시초가가 5/20 진입가 대비 -5% |
| 시스템 자동 대응 | (1) 5/21 09:00 owner_rule_monitor: pnl_pct=-5.0% ≤ -3.0% → SELL_STOP_LOSS<br>(2) 시장가 매도 → -5% 부근 체결 |
| 손익 계산 | 진입 50,000원, 청산 47,500원 (-5%)<br>수수료 합 15원, 거래세 86원<br>**손익: -2,500원 - 101원 = -2,601원 (-5.20%)**<br>**이월의 위험 시나리오** |
| 사장님 카톡 | 5/21 09:00 🔴 [사장님 룰 자동 청산] SELL_STOP_LOSS (-5.00%) |
| 사장님 개입 | **0건 (완전 자동)** — 단, 이월 결정 재검토 멘탈 관리 |
| 후속 대응 | 이월 5조건이 너무 후한 게 아닌지 검증 (예: 외인+기관 수급 임계 +1억 → +3억으로 강화) |

**paper mirror 시뮬 결과**: 동일 손절
**자기반성 학습 데이터**: 이월 후 갭다운 빈도 (D+1 종가 기준)
**5/20 가동 후 분석 포인트**: 이월 결정에서 야간 美 증시/환율 영향 추가 검토 가능

---

#### S-20: 5/20 이월 → 5/21~5/23 매일 평가 → 5/25 5일 한도 도달 청산

| 항목 | 내용 |
|---|---|
| 발동 조건 | S-04 후속. 매일 15:20 이월 5조건 ALL PASS → 5/21~5/24 4회 이월 → 5/25 (월) 15:20 days_held=5 |
| 시스템 자동 대응 | (1) 5/25 15:20 evaluate_hold_overnight: holding_days_ok = (5 < 5) → False<br>(2) can_hold=False → 강제 청산 진행 |
| 손익 계산 | 5/20 진입 50,000원 → 5/25 청산 (시장 상태에 따라 ±10% 가능)<br>**예: +8% 종가 → 54,000원 청산 → 손익 +4,000원 - 116원 = +3,884원 (+7.77%)** |
| 사장님 카톡 | 5/21~5/24 매일 🌙 이월 알림<br>5/25 ⏰ SELL_FORCE_CLOSE (보유 5일 한도) |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 5일 한도가 적절한지 검증 (수익 곡선 분석). 5일 = 자비스 기존 패턴. |

**paper mirror 시뮬 결과**: 5일 누적 슬리피지 영향 작음 (홀딩 동안 거래 X)
**자기반성 학습 데이터**: 5일 보유 시 D+5 수익률 분포
**5/20 가동 후 분석 포인트**: 5일 한도 vs 10일 한도 비교 백테스트 (P3 자율 매매로 확장 시)

---

### 5-5. paper mirror 단독 시나리오 (S-21 ~ S-22) — 2건

---

#### S-21: 실주문 KIS RATE 실패인데 paper는 정상 시뮬 → 실측 vs 가정 갭 데이터

| 항목 | 내용 |
|---|---|
| 발동 조건 | S-08 같은 KIS RATE LIMIT 발생 |
| 시스템 자동 대응 | (1) 실주문 buy_limit FAILED<br>(2) auto_buy_executor: continue (다음 종목) — paper_record_entry는 buy_limit 성공한 종목만 호출되므로 동일 종목에 대한 paper 시뮬도 발생 X<br>(3) ★ **실주문 실패 시 paper만 진행 옵션은 없음** (현재 설계) |
| 손익 계산 | paper 손익 0원 (시뮬 X) |
| 사장님 카톡 | (1) ❌ 실주문 실패 카톡<br>(2) paper 관련 카톡 없음 |
| 사장님 개입 | **0건** |
| 후속 대응 | paper-only 시뮬 모드 추가 검토 (KIS 실패 시 paper만 진행 — 학습 데이터 확보) |

**paper mirror 시뮬 결과**: 시뮬 X
**자기반성 학습 데이터**: KIS 실패율 vs paper-only 시뮬 가치
**5/20 가동 후 분석 포인트**: paper-only fallback 모드 설계 필요성

---

#### S-22: paper mirror 슬리피지 가정 (+0.05% 매수 / -0.05% 매도) 실측 보정

| 항목 | 내용 |
|---|---|
| 발동 조건 | 정상 매수+매도 (S-01) 완료 후 paper 결과 비교 |
| 시스템 자동 대응 | (1) S-01 정상 흐름 — paper도 동일 시점 매수/매도 시뮬<br>(2) data/paper_mirror/2026-05-20_positions.json에 paper 손익 기록<br>(3) 5/21 06:30 paper 일일 정산 cron (또는 수동) — 실주문 손익 vs paper 손익 비교 |
| 손익 계산 | 실주문 +1,949원 (S-03 예시) vs paper +1,899원 → 갭 -50원 (슬리피지 가정 영향) |
| 사장님 카톡 | paper 단독 카톡 없음 (관찰자 역할). 5/21 정산 리포트만. |
| 사장님 개입 | **0건** |
| 후속 대응 | paper 슬리피지 가정 -50원이 실측 +25원과 비교 → 가정 수정 |

**paper mirror 시뮬 결과**: S-01과 동일 흐름 + 슬리피지 차이
**자기반성 학습 데이터**: 매수/매도 슬리피지 분포 실측 → paper 가정 보정
**5/20 가동 후 분석 포인트**: paper 가정 +0.05% 매수가 실측 ±0.X%인지

---

### 5-6. 엣지 케이스 (S-23 ~ S-24) — 2건

---

#### S-23: tomorrow_picks.json 어제 데이터 (max_age_hours > 24) → DataIntegrity FAIL → KILL_SWITCH

| 항목 | 내용 |
|---|---|
| 발동 조건 | 5/19 BAT-D 실패 → tomorrow_picks.json 마지막 갱신 5/18 → 5/20 06:30 점검 시 > 24h |
| 시스템 자동 대응 | (1) 06:30 DataIntegrity: max_age_hours 초과 검출 → FAIL<br>(2) activate_kill_switch ("DataIntegrity FAIL: tomorrow_picks.json stale (48h)")<br>(3) 14:00~14:55 매수 SKIP |
| 손익 계산 | **손익 0원** |
| 사장님 카톡 | 06:30 🚨 [자동 차단 발동] DataIntegrity FAIL — 사유: tomorrow_picks.json stale (48h) |
| 사장님 개입 | **즉시 ssh 권장** — BAT-D 수동 재실행 또는 picks 수동 생성 후 KILL_SWITCH 해제 |
| 후속 대응 | BAT-D 실패 원인 분석 (5/19 사례) — 정보봇 회신 지연, KIS API 장애 등 |

**paper mirror 시뮬 결과**: paper도 KILL_SWITCH 가드 동일
**자기반성 학습 데이터**: BAT-D 실패 빈도 → 신뢰성 지표
**5/20 가동 후 분석 포인트**: DataIntegrity 점검이 BAT 실패를 정확히 검출하는지

---

#### S-24: 일일 1건 한도 도달 후 같은 cron에서 추가 후보 평가 → 안전선 ④ 차단

| 항목 | 내용 |
|---|---|
| 발동 조건 | 14:00 cron 내 후보 9건 중 첫 번째 매수 성공 → 두 번째 종목도 점수 95 STRONG |
| 시스템 자동 대응 | (1) 14:00 candidate 1: BUY 성공 → positions[tk1] = {...}<br>(2) buy_executed=True → **break** (line 380, 일일 1건 한도)<br>(3) candidate 2~9 평가 안 됨<br>(4) 14:05 cron: positions에 1건 기록됨 → should_auto_buy 안전선 ④: today_entries=1 ≥ MAX_DAILY_BUYS(1) → SKIP<br>(5) 모든 후속 cron SKIP |
| 손익 계산 | candidate 1만 손익, candidate 2는 0원 |
| 사장님 카톡 | (1) 14:00 ✅ candidate 1 매수<br>(2) 14:05+ candidate 2는 SKIP 콘솔만 |
| 사장님 개입 | **0건 (완전 자동)** |
| 후속 대응 | 일일 1건 한도가 5/20 가동의 핵심 안전선. 검증 후 5월 말 한도 2건 확대 검토. |

**paper mirror 시뮬 결과**: paper도 일일 1건 한도 (`paper_mirror.py:82`)
**자기반성 학습 데이터**: candidate 2가 더 좋은 수익을 냈을 가능성 검토
**5/20 가동 후 분석 포인트**: 1순위 매수 vs 2~9순위 종목의 D+1 수익률 비교

---

## 6. 사장님이 받을 수 있는 카톡 6대 메시지 패턴

### 6-1. ✅ 매수 성공

```
✅ [자동매수 성공] HPSP(403030)
  지정가 52,300원 × 1주 = 52,300원
  점수: 100 (STRONG)
  주문번호: 1234567890
  ─────────────
  사장님 룰 모니터: 14:35부터 매 5분
  - 룰 ① -3% 절대 손절
  - 룰 ② peak -3% 트레일링
  - 룰 ③ 15:20 강제 청산
  - 룰 ④ 수급 지속 시 5/21 이월
  ─────────────
  📊 정보봇 ETF 신뢰도 (5/15 누적, 6일 잠정):
    theme 73% · global 63% · sector 46%
    direction 41% · bond_commodity 29%
```

**+ KisOrderAdapter 별도 카톡**:
```
[자동매매] 매수 접수
종목: 403030
수량: 1주
가격: 52,300원 (지정가)
금액: 52,300원
일일 누적: 52,300원 / 100,000원 (52.3%)
일일 횟수: 1회 / 1회
시각: 14:00:15
```

**사장님 권장 대응**: 무대응. 15:20까지 자동 진행됨. 14:35부터 5분 간격 청산 평가가 시작됨.

---

### 6-2. ❌ 매수 실패

```
❌ [자동매수 실패] HPSP(403030)
  지정가 52,300원 1주
  사유: 요청 횟수 초과
  → 다음 5분 cron 재시도
```

**사장님 권장 대응**: 5분 대기. 자동 재시도. 14:55까지 모두 실패하면 0건 마감 카톡으로 다시 알림.

---

### 6-3. 🌙 이월 결정

```
🌙 [사장님 룰 ④ — 익일 이월] HPSP(403030)
  PnL +4.50% | 보유 0일
  수급 +2.5억 | EYE PASS
  → 5/21 익일 보유 (최대 4일 더)
```

**사장님 권장 대응**: 무대응. 5/21 09:00부터 owner_rule_monitor가 자동 평가. 갭다운 -5% 가능성은 인지.

---

### 6-4. ✅ 매도 (룰 ①/②/③ 발동)

**룰 ① 절대 손절**:
```
🔴 [사장님 룰 자동 청산] HPSP(403030) SELL_STOP_LOSS
  진입 52,300 → 현재 50,700 (-3.06%)
  peak 52,500 대비 -3.43%
  사유: 진입가 -3% 절대 손절 (현재 -3.06%)
  주문: 성공
```

**룰 ② 트레일링**:
```
🟡 [사장님 룰 자동 청산] HPSP(403030) SELL_TRAILING
  진입 52,300 → 현재 53,500 (+2.29%)
  peak 55,200 대비 -3.08%
  사유: 트레일링 청산 (peak 55,200 → 현재 53,500, 하락 -3.08%, 이익 보존 +2.29%)
  주문: 성공
```

**룰 ③ 강제 청산**:
```
⏰ [사장님 룰 자동 청산] HPSP(403030) SELL_FORCE_CLOSE
  진입 52,300 → 현재 52,900 (+1.15%)
  peak 53,200 대비 -0.56%
  사유: 15:20 강제 청산 (NXT 안전마진, 현재 +1.15%)
  주문: 성공
```

**사장님 권장 대응**: 무대응. 첫 매매 결과 확인만. 손실 -3,000원 한도는 사전 인지된 최악 시나리오.

---

### 6-5. 🚨 KILL_SWITCH RED 활성화

```
🚨 [자동 차단 발동]
검수팀 [MarketRegimeGate]가 FAIL 검출
━━━━━━━━━━━━━━━━━
사유: 시장 약세 검출: KODEX 200 -2.50%, KODEX 200선물인버스2X +5.00%
data/KILL_SWITCH 자동 활성화됨
→ 자동매매 cron 즉시 중단
━━━━━━━━━━━━━━━━━
사장님 확인 후 수동 해제:
  rm ~/quantum-master/data/KILL_SWITCH
```

**워커별 발생 사유 예시**:
| 워커 | 사유 예시 |
|---|---|
| EnvChecker | AUTO_TRADE_5_20=false, KIS_APP_KEY 누락 |
| CodeAuditor | 핵심 import 실패, owner_rule.py 변조 |
| FlowMonitor | KIS 체결 vs positions.json 불일치 |
| DataIntegrity | tomorrow_picks.json stale (>24h), Supabase 데이터 누락 |
| MarketRegimeGate | KODEX 200 -2%, 인버스 +3%, 2X +5% |

**사장님 권장 대응**:
- **EnvChecker/CodeAuditor/DataIntegrity FAIL**: 즉시 ssh 권장. 원인 파악 후 수정 + KILL_SWITCH 해제.
- **MarketRegimeGate FAIL**: 무대응 권장. 폭락장이면 그대로 보호 받음. 5/20 매수 0건 마감.
- **FlowMonitor FAIL**: 즉시 ssh 필요 (체결 불일치는 critical).

---

### 6-6. ⏭️ 14:55 매수 0건 마감

```
⏭️ [자동매수 0건] 14:00~14:55 통과 후보 없음
TOP 3 (참고):
  HPSP 점수 88 (regime CAUTION)
  삼성E&A 점수 87 (VWAP 과열 +2.8%)
  SK하이닉스 점수 85 (1주 110,000원 > 10만 한도)
─────────────
📊 정보봇 ETF 신뢰도 (5/15 누적, 6일 잠정):
  theme 73% · global 63% · sector 46%
```

**사장님 권장 대응**: 무대응. 0건 마감은 정상. 검수팀이 위험을 회피한 것일 수 있음. 5/21 picks 갱신 대기. 회피한 매수가 실제로 -X%로 떨어졌으면 다음날 학습 데이터로 확인.

---

## 7. 자체 검수 (작성 후 점검)

### 7-1. 코드 흐름 정확성 검증

| 검증 항목 | 코드 위치 | 매트릭스 반영 |
|---|---|---|
| `auto_buy_executor` KILL_SWITCH 가드 | `scripts/auto_buy_executor.py:203` | ✅ S-13, S-14, S-23 |
| `auto_buy_executor` AUTO_TRADE_5_20 가드 | `scripts/auto_buy_executor.py:211` | ✅ S-13 |
| `should_auto_buy` 안전선 9건 | `src/use_cases/auto_buy_decider.py:104` | ✅ S-05, S-12 |
| `evaluate_owner_rule` 룰 ① 손절 | `src/use_cases/owner_rule.py:106` | ✅ S-06, S-19 |
| `evaluate_owner_rule` 룰 ② 트레일링 | `src/use_cases/owner_rule.py:119` | ✅ S-01, S-11 |
| `evaluate_owner_rule` 룰 ③ 강제 청산 | `src/use_cases/owner_rule.py:136` | ✅ S-02, S-03, S-16 |
| `evaluate_hold_overnight` 룰 ④ 이월 | `src/use_cases/owner_rule.py:164` | ✅ S-04, S-18 |
| `KisOrderAdapter._guard` 9건 검사 | `src/adapters/kis_order_adapter.py:88` | ✅ S-12 (수량/금액) |
| `KisOrderAdapter._adjust_to_tick` 호가 단위 | `src/adapters/kis_order_adapter.py:167` | (간접 적용) |
| `MarketRegimeGate.check_market_regime` | `src/agents/market_regime_gate.py:68` | ✅ S-14, S-17 |
| `MarketRegimeGate` fail-safe (MODEL/fetch) | `src/agents/market_regime_gate.py:87, 189` | ✅ S-10 |
| `activate_kill_switch` | `src/agents/kill_switch_manager.py:37` | ✅ S-13, S-14, S-23 |
| `check_vwap_gate` VWAP 과열 차단 | `scripts/auto_buy_executor.py:132` | ✅ S-05 (TOP 3 표시) |
| `paper_record_entry` 일일 1건 한도 | `src/use_cases/paper_mirror.py:82` | ✅ S-21, S-22 |

### 7-2. 손익 계산 검증

수수료 0.015% + 거래세 0.18% 적용 확인:

| 시나리오 | 매수가 | 매도가 | 수수료 합 | 거래세 | 손익 |
|---|---|---|---|---|---|
| S-01 +5%→트레일링 | 50,000 | 50,925 | 15원 | 91.7원 | +819원 |
| S-06 즉시 -3% | 50,000 | 48,500 | 15원 | 87.3원 | **-1,602원 (최악)** |
| S-11 +10%→트레일링 | 50,000 | 53,350 | 15원 | 96.0원 | **+3,239원 (최선)** |
| S-19 갭다운 -5% | 50,000 | 47,500 | 15원 | 85.5원 | -2,601원 |

수식 확인:
- 매수 수수료 = price × 0.00015
- 매도 수수료 = price × 0.00015
- 매도 거래세 = price × 0.0018
- 손익 = (매도가 - 매수가) - 매수 수수료 - 매도 수수료 - 거래세

### 7-3. 7명 검수팀 자동 대응 검증

| 워커 | KILL_SWITCH 자동 활성화 | 매트릭스 반영 |
|---|---|---|
| EnvChecker | ✅ FAIL 시 (kill_switch_manager) | S-13 |
| CodeAuditor | ✅ FAIL 시 | (잠재 — S-09 변형) |
| FlowMonitor | ✅ critical 시 | S-09 |
| DataIntegrity | ✅ FAIL 시 | S-23 |
| MarketRegimeGate | ✅ FAIL 시 (1건+ trigger) | S-14, S-17 |
| Reporter | ❌ (수동 호출만, 활성화 X) | — |
| MarketScanner | ❌ (정보용, 활성화 X) | — |

---

## 8. 5/20 가동 후 사장님 1차 점검 체크리스트

가동 종료 후 (5/20 16:00+) 사장님이 확인할 항목:

### 8-1. KIS 계좌 잔고 확인
- 매수 1건 체결 여부
- 평균 단가 / 보유 수량
- 평가 손익

### 8-2. 카톡 로그 정리
- 총 카톡 수: 정상이면 2~5건 (매수 1 + 매도 1 + KIS 어댑터 2 + 기타)
- 5건 초과 시 패턴 분석

### 8-3. 핵심 로그 위치
- `sudo journalctl -u quantum-scheduler --no-pager -n 200 | grep -E "auto_buy|owner_rule|MarketRegime"`
- `data/owner_rule_positions.json` (당일 매수 기록)
- `data/paper_mirror/2026-05-20_positions.json` (paper 시뮬 결과)
- `data/agent_reports/*_latest.json` (7명 워커 보고)

### 8-4. 학습 데이터 라벨링 (5/21 16:30 BAT-D 후)
- 매수 종목 D+1 종가 확인
- 사장님 룰 결정의 적중률 (-3% 손절이 맞았는지, 트레일링이 빠른지)
- 회피한 후보들의 D+1 종가 (검수팀 적중률)

---

## 9. 마무리 — 5/20 가동 직전 사장님 멘탈 가이드

**최선 시나리오** (S-11): 첫 매매 +3,239원 수익 (1주 기준)
**최악 시나리오** (S-06): 첫 매매 -1,602원 손실 (1주 기준)
**가장 가능성 높은 시나리오** (S-02, S-03, S-05): 횡보 또는 0건 마감

### 사장님이 가동 전 인지해야 할 3대 사실
1. **첫 매매에서 손절은 자연스럽다** — 최악 -1,602원 (-3.20%)은 사전 인지된 한도. 멘탈 안정성 유지.
2. **0건 마감도 정상이다** — 안전선 9건 ALL 통과 못 하면 매수 X. 검수팀이 일을 잘하면 0건이 안전한 결과.
3. **KILL_SWITCH RED 카톡 = 검수팀 칭찬 사인** — 자동 차단이 발동했다는 건 시스템이 일하고 있다는 뜻. 사장님은 카톡 받고 확인만 하면 됨.

### 5/20 가동 직후 (14:00 ~ 15:30) 사장님 행동 지침
- **14:00 직전 (13:55~14:00)**: MarketRegimeGate 카톡 대기. 무소식 = NORMAL (정상).
- **14:00 ~ 14:55**: 매수 성공 카톡 (1건) 또는 0건 마감 카톡 대기. 그 외 카톡은 KIS 어댑터 별도 알림.
- **15:00 ~ 15:30**: 청산 카톡 (룰 ①/②/③) 또는 이월 카톡 (룰 ④) 대기.
- **15:30 ~ 16:00**: KIS 잔고 + 카톡 로그 정리 + paper mirror 결과 비교.

**사장님 손 0번 — 모든 결정은 자동. 사장님은 결과 확인만 하면 됨.**
