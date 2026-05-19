# PDCA Plan: 차트영웅 D+1 매매법 자동화

**기능명**: `surge-d1-chart-hero`
**작성일**: 2026-05-19 (화)
**작성자**: 퀀트봇 (사장님 결단 반영)
**가동 목표**: 2026-05-22(금) 09:00 paper mirror, 2026-05-27(화) 실전 검토

---

## 1. 목적 (Why)

차트영웅 단타 트레이더의 73연승 매매법을 퀀트봇에 완전 구현.
우리 백테스트 "저변동+눌림+D+1양봉" (WR 91.7%, PF 31.65, D+3 +5.56%) 패턴과
차트영웅 룰이 100% 일치 — 두 검증의 교집합으로 자동매매 시스템 구축.

## 2. 사장님 확정 룰 (옵션 A — 차트영웅 원본 100%)

### 종목 선정 5-Gate
- Gate 1: 매크로 4-시그널 3/4 GO
- Gate 2: 전날 상한가 (D0)
- Gate 3: 저변동 + 눌림 (MA20 -3~0%)
- Gate 4: 주도 섹터 (fire_score ≥ 60)
- Gate 5: 펀더(PER<20, ROE>5) + 주봉 스토캐스틱<30
- **최종 진입 트리거**: D+1 양봉 확인 (D+1 종가 > D+1 시가 + D+1 종가 > D0×0.95)

### 분할매수
- 1차: 1.5% (D+1 양봉 + 종가 진입)
- 2차: -10% → 1.5% 추매 (누적 3.0%)
- 3차: -20% → 1.5% 추매 (누적 4.5%)
- 4차: -30% → 1.5% 추매 (누적 6.0%)

### 익절
- +5% → 50% 부분 익절
- +10% → 나머지 50%
- D+5 미달 → 강제 청산

### 현금 보유: 자연 비축 (강제 룰 없음)

## 3. 데이터 소스

| 데이터 | 출처 | 보유 여부 |
|---|---|---|
| 종목 일봉 OHLCV | KIS, 정보봇 OHLCV CSV | ✅ |
| **종목 주봉 OHLCV** | KIS `inquire-daily-itemchartprice` (W) | 🆕 5/20 구현 |
| **KOSPI 지수 주봉** | KIS `inquire-daily-indexchartprice` (W) | 🆕 5/20 구현 |
| 상한가 종목 풀 | KIS 등락률 순위 API, surge_pullback | ✅ |
| 섹터 fire_score | quant_sector_fire (Supabase) | ✅ |
| MA20 / 볼린저 / 스토캐스틱 | `src/indicators.py` | ✅ |
| **종목 펀더멘털 카드** | 정보봇 `quant_company_card` | 🤝 정보봇 |
| **상한가 catalyst** | 정보봇 `quant_surge_catalyst` | 🤝 정보봇 |
| 미국 10년 국채 수익률 | yfinance `^TNX` | 🆕 5/20 구현 |
| 공포탐욕지수 | CNN Fear & Greed API | 🆕 5/20 구현 |
| 환율 USD/KRW | KIS 또는 yfinance | ✅ 부분 |

## 4. 작업 분해 (Work Breakdown)

### Phase 1: 데이터 인프라 (5/20 수)

#### 1-1. KIS 주봉 어댑터 [P0]
- 파일: `src/adapters/kis_weekly_kit.py` (신규)
- 함수:
  - `get_stock_weekly(code, from_date, to_date)` — 종목 주봉
  - `get_kospi_weekly(from_date, to_date)` — KOSPI 지수 주봉
  - `get_kosdaq_weekly(from_date, to_date)` — KOSDAQ 지수 주봉
- 검증: 005930, 000660, KOSPI(0001), KOSDAQ(1001) 6개월

#### 1-2. 매크로 4-시그널 모듈 [P0]
- 파일: `src/macro/four_signal_gate.py` (신규)
- 시그널:
  1. KOSPI 주봉 스토캐스틱 K < 30
  2. 미국 10년물 < 4.5% (yfinance ^TNX)
  3. USD/KRW < 1450원
  4. 공포탐욕지수 < 25 (CNN API)
- 출력: `{date, signal_1, signal_2, signal_3, signal_4, gate_pass: bool, score: int/4}`
- 저장: `data/macro_four_signal_daily.csv`

#### 1-3. 정보봇 데이터 수신 핸들러 [P0]
- 파일: `src/adapters/jgis_intel_adapter.py` (확장)
- Supabase `quant_company_card`, `quant_surge_catalyst` 조회
- 캐싱: 30분 TTL

### Phase 2: 매매 엔진 (5/21 목)

#### 2-1. 5-Gate 종목 선정기 [P0]
- 파일: `scripts/surge_d1_picker.py` (신규)
- 입력: 매일 15:30 (장 마감 후)
- 처리: Gate 1~5 통과 종목 추출 → TOP N (1~3종목)
- 출력: `data/picks_surge_d1_YYYYMMDD.csv`

#### 2-2. KIS 예약매수 어댑터 확장 [P0]
- 파일: `src/adapters/kis_order_adapter.py` (확장)
- 기능 추가:
  - `place_reserve_order()` — 다음날 예약 매수
  - `place_split_buy()` — 추매 트리거 (-10%/-20%/-30%)
  - `place_take_profit()` — 부분 익절 (+5%, +10%)
  - `place_force_exit()` — D+5 강제 청산

#### 2-3. D+1 양봉 확인 로직 [P0]
- 파일: `src/strategies/d1_confirm.py` (신규)
- 09:00~15:00 모니터링
- 확인 조건:
  - D+1 종가 > D+1 시가
  - D+1 종가 > D0 종가 × 0.95
- 만족 시: 다음날 종가 예약매수 발주

#### 2-4. 분할매수 + 익절 룰 엔진 [P0]
- 파일: `src/strategies/chart_hero_split_rule.py` (신규)
- 보유 종목 평단가 대비 -10%/-20%/-30% 도달 시 추매 발주
- 보유 종목 평단가 대비 +5%/+10% 도달 시 부분 익절 발주
- D+5 종가까지 +5% 미달 시 강제 청산

### Phase 3: 백테스트 + Paper Mirror (5/21 목)

#### 3-1. 6개월 백테스트
- 파일: `scripts/backtest_chart_hero.py` (신규)
- 기간: 2025-11-19 ~ 2026-05-19
- 정보봇 catalyst 데이터 결합
- 목표 지표:
  - WR ≥ 85% (메모리 91.7% 대비 보수적)
  - PF ≥ 10 (메모리 31.65 대비 보수적)
  - MDD ≤ -10%

#### 3-2. Paper Mirror 가동 [P0]
- 5/22(금) 09:00 시작
- 5/22 ~ 5/26 (1주)
- 잔고 시뮬: 2,500만원 (실제 계좌 격리)
- 매일 18:00 일일 보고

### Phase 4: 실전 진입 검토 (5/27 화)

- 1주 paper mirror 성과 분석
- WR ≥ 60%, MDD ≤ -5% 만족 시 사장님 결단
- 실전 진입 사이즈: 보수적 (최소 1주 ~ 5주)

## 5. 위험 관리

### 5-1. KILL_SWITCH 연동
- Layer 7 자동 KILL_SWITCH 발동 시 신규 진입 중지
- 보유 종목은 차트영웅 룰 그대로 (분할매수가 안전망)

### 5-2. 매크로 RED 시
- Shield=RED + 매크로 4-시그널 1/4 이하 → 신규 진입 일시 중지
- 보유 종목 익절 룰만 유지

### 5-3. 정보봇 데이터 결손 시
- catalyst 데이터 없으면 Gate 5 통과 불가 → 매수 X
- 펀더멘털만 있으면 보수적 진입 (1차만, 추매 X) 옵션

## 6. 검증 (Check)

### 6-1. 일일 검증 (paper mirror 기간)
- 5-Gate 통과 종목 수 (목표: 일 1~3종목)
- D+1 양봉 확인율
- 매수 시점/사이즈/추매 정확성

### 6-2. 주간 검증 (5/26 일요일)
- 1주 paper mirror 누적 손익
- WR / PF / MDD
- 실전 진입 가능 여부

## 7. 일정 (Timeline)

| 일자 | 작업 | 담당 |
|---|---|---|
| 5/19 22:00 | 본 Plan 작성 + 정보봇 의뢰서 발송 | 메인 |
| **5/20(수)** | Phase 1 (KIS 주봉 + 매크로 + 정보봇 수신) | 메인 + Bash |
| 5/20 22:00 | 정보봇 6개월 과거 데이터 수신 | 정보봇 |
| **5/21(목)** | Phase 2 (매매 엔진) + Phase 3-1 (백테스트) | 메인 + 서브에이전트 |
| 5/21 18:00 | 정보봇 1차 펀더+catalyst 수신 | 정보봇 |
| **5/22(금) 09:00** | Phase 3-2 (Paper Mirror 가동) | 자동 |
| 5/22~5/26 | 1주 paper mirror 검증 | 자동 + 일일 보고 |
| 5/27(화) | Phase 4 (실전 진입 검토) | 사장님 결단 |

## 8. 산출물 (Deliverables)

1. `src/adapters/kis_weekly_kit.py`
2. `src/macro/four_signal_gate.py`
3. `src/adapters/jgis_intel_adapter.py` (확장)
4. `scripts/surge_d1_picker.py`
5. `src/adapters/kis_order_adapter.py` (확장)
6. `src/strategies/d1_confirm.py`
7. `src/strategies/chart_hero_split_rule.py`
8. `scripts/backtest_chart_hero.py`
9. 6개월 백테스트 결과 리포트 (`docs/03-analysis/backtest-chart-hero-6m.md`)
10. Paper Mirror 일일 보고 (`docs/04-report/paper-mirror-chart-hero-*.md`)

## 9. 성공 기준

- ✅ 5/22 09:00 paper mirror 가동 성공
- ✅ 일일 1~3종목 자동 선정 + 예약매수 발주 정상
- ✅ 분할매수 + 익절 자동 트리거 정상
- ✅ 1주 paper mirror WR ≥ 60%
- ✅ 5/27 실전 진입 사장님 GO 사인
