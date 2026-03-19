# 퀀트봇 전수조사 결과 보고서
> **작성일**: 2026-03-19
> **대상**: Quantum Master 한국 주식 자동매매 시스템
> **목적**: ALPHA ENGINE V2 (멀티팩터) 전환을 위한 현행 시스템 전수 점검
> **범위**: 131개 질문 / 9개 PART / 코드 기반 실사

---

## 목차
1. [PART 1: 데이터 수집 능력 (Q01-Q30)](#part-1)
2. [PART 2: 분석/스코어링 엔진 (Q31-Q50)](#part-2)
3. [PART 3: 포지션 관리 (Q51-Q74)](#part-3)
4. [PART 4: 매수/매도 실행 (Q75-Q88)](#part-4)
5. [PART 5: 실행 흐름 (Q89-Q96)](#part-5)
6. [PART 6: 출력/통신 (Q97-Q107)](#part-6)
7. [PART 7: 학습/피드백 (Q108-Q121)](#part-7)
8. [PART 8: 코드 구조 (Q122-Q131)](#part-8)
9. [PART 9: V2 전환 진단](#part-9)

---

<a id="part-1"></a>
## PART 1: 데이터 수집 능력 (Q01-Q30)

### 1-2. KIS API 사용 현황

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q01 | 현재가 조회 | ✅ 실시간. `broker.fetch_price(ticker)` via mojito2 | `src/adapters/kis_stock_data_adapter.py:28` |
| Q02 | 일봉/분봉 | ✅ 일봉(120일)+1분봉(당일). N분봉 집계 가능 | `kis_stock_data_adapter.py:45`, `kis_intraday_adapter.py:116` |
| Q03 | 투자자별 매매동향 | ✅ 외인/기관/개인 순매수(당일). pykrx fallback | `kis_stock_data_adapter.py:183`, `pykrx_supply_adapter.py:150` |
| Q04 | 프로그램 매매 | ❌ 미구현. `program_net_buy=0` 더미값 | `kis_intraday_adapter.py:321` |
| Q05 | 호가창 10호가 | ✅ 매수/매도 10호가 + 잔량 | `kis_intraday_adapter.py:435` |
| Q06 | 시간외 데이터 | ❌ 미구현. 08:30~15:30만 인정 | `kis_intraday_adapter.py:496` |
| Q07 | 잔고/주문가능금액 | ✅ `fetch_balance()` + `get_available_cash()` | `kis_order_adapter.py:199,237` |
| Q08 | 주문 실행 | ✅ 지정가/시장가/정정/취소/체결확인 | `kis_order_adapter.py:45-313` |

### 1-3. pykrx / yfinance / 기타

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q09 | pykrx 사용 | ✅ OHLCV + 투자자 수급 + PER/PBR/EPS/BPS | `src/data_collector.py` |
| Q10 | yfinance | ✅ 미국 ETF 위주 (SOXX 등). FDR fallback | `src/adapters/macro_adapter.py:82` |
| Q11 | 가격 히스토리 | 3년치 기본. `--backfill-years`로 조절 | `scripts/rebuild_universe.py:148` |
| Q12 | 재무제표 | ✅ DART API(우선) + pykrx(fallback). PER/PBR/ROE/EBITDA/FCF | `src/adapters/dart_adapter.py` |
| Q13 | 컨센서스 | ✅ wisereport 스크래핑. 목표가/Forward EPS/PER | `src/adapters/consensus_scraper.py:64` |

### 1-4. 매크로/해외

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q14 | FRED | ⚠️ API키 있으나 비활성. 일일 스윙에 미사용 | `.env` |
| Q15 | VIX | ✅ FDR 수집. 일별 종가. US Overnight에서 활용 | `macro_adapter.py:49` |
| Q16 | 미국 선물 | ✅ yfinance(SPY/QQQ/DIA/SOXX) ETF 형태 | `scripts/us_data_backfill.py` |
| Q17 | USD/KRW | ✅ FDR 일별 종가 | `macro_adapter.py:60` |
| Q18 | 원자재 | ⚠️ yfinance 가능하나 자동화 스크립트 최소 | - |
| Q19 | CFTC COT | ✅ S&P500/Gold/10Y/WTI 주간 수집 | `scripts/fetch_cot_weekly.py` |
| Q20 | CNN F&G | ❌ 미구현. VIX로 대체 | - |

### 1-5. 뉴스/공시/이벤트

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q21 | DART 공시 | ✅ OpenAPI. 재무제표+신규공시 감지 | `src/adapters/dart_adapter.py` |
| Q22 | EDGAR | ❌ 미구현 | - |
| Q23 | GDELT | ❌ 미구현. RSS+Perplexity로 대체 | - |
| Q24 | 뉴스 감성분석 | ✅ Perplexity Sonar + Claude. 호재/악재 분류 | `src/adapters/perplexity_news_adapter.py` |
| Q25 | 이벤트 캘린더 | ⚠️ 네 마녀의 날만. 실적/배당락일 미추적 | `src/use_cases/market_calendar.py` |

### 1-6. 섹터/유니버스

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q26 | 유니버스 크기 | 84종목. 시총 2,000억↑, 거래대금 5억↑ | `data/universe.csv` |
| Q27 | 갱신 주기 | 매월 자동 | `scripts/rebuild_universe.py` |
| Q28 | 섹터 분류 | 자체 (KIS 업종코드→20개 섹터) | `src/sector_classifier.py` |
| Q29 | 섹터 자금흐름 | ✅ 상위 5종목 외인/기관 합산. 스텔스 매집 감지 | `scripts/sector_investor_flow.py` |
| Q30 | 상관관계 매트릭스 | ❌ 미구현 | - |

### PART 1 요약

| 분류 | 구현 | 미구현 |
|------|------|--------|
| KIS API | 7/8 (88%) | 시간외(Q06) |
| 시장 데이터 | 4/5 (80%) | 프로그램매매(Q04) |
| 매크로 | 5/7 (71%) | CNN F&G, EDGAR |
| 뉴스/공시 | 3/5 (60%) | EDGAR, GDELT |
| 섹터/유니버스 | 4/5 (80%) | 상관관계(Q30) |

---

<a id="part-2"></a>
## PART 2: 분석/스코어링 엔진 (Q31-Q50)

### 2-1. 스코어링 시스템

**Q31. 메인 함수/파일**
- `src/v8_scorers.py` → `ScoringEngine.score_all()` (라인 72-100)
- 진입점: `src/v8_pipeline.py` → `QuantumPipelineV8.scan_single()`

**Q32. 배점 체계 — 중요 정정**
> 현재 코드는 **100점 4축이 아니라 5축 가중합 (0.0~1.0)** 체계

| 축 | 이름 | 가중치 | 설명 |
|----|------|--------|------|
| S1 | 에너지소진 | 0.30 | RSI + 거래량감소 + BB위치 + 패닉보너스 |
| S2 | 밸류에이션 | 0.20 | PER할인 + PBR백분위 + Forward PER + 턴어라운드 |
| S3 | OU평균회귀 | 0.20 | z-score + half-life + theta강도 |
| S4 | 모멘텀감속 | 0.15 | 기울기감속 + 곡률양전환 + MACD히스토그램 |
| S5 | 수급/스마트머니 | 0.15 | OBV + 외인연속 + 기관(ETF보정) + DRS + 공매도 + 연기금 |

등급: A≥0.65 / B≥0.50 / C≥0.35 / F<0.35

**Q33. 세부 계산 로직** — 각 축별 4~7개 구성요소 (총 21개 서브팩터)
- 상세: `src/v8_scorers.py` 라인 115-504

**Q34. 컨센서스 축 항상 0점 이유**
- **컨센서스 축 자체가 없음**. Forward PER은 S2(밸류)에 통합
- 뉴스/이벤트는 `zone_score`에 직접 부스트 (별도 축 아님)
- `scan_consensus.py`는 독립 스크립트 (스코어링 미연동)

### 2-2. 종목 발굴 소스

**Q35-Q37. 실제 작동 소스 5개**

| # | 소스 | 파일 | 기여 방식 |
|---|------|------|-----------|
| 1 | Parquet 유니버스 (84종목) | `data/processed/*.parquet` | v8 Gate+Score+Trigger |
| 2 | DART 공시 촉매 | `data/dart_event_signals.json` | catalyst_boost ×1.10 |
| 3 | 뉴스 (Perplexity) | 실시간 API | 실적서프라이즈 ×1.10 |
| 4 | US Overnight Signal | `data/us_market/overnight_signal.json` | 섹터별 us_mult |
| 5 | CSV 백필 종목 | `stock_data_daily/*.csv` | quick_csv_score() |

- 중복 투표: **곱셈 방식** (`rr × zone × catalyst × sd × density × us × freshness`)
- 소스별 적중률: **미구현** (daily_market_learner에서 추적만, 가중치 미반영)

### 2-3. 게이트/필터

| # | 게이트 | 조건 | 파일 |
|---|--------|------|------|
| Q39 | G1 ADX | ADX≥14 + MA60>MA120 + 현재가>MA120 | `src/v8_gates.py:78` |
| Q40 | G2 풀백 | **20일 최고가** 대비 0.8~4.0 ATR (52주 아님) | `src/v8_gates.py:119` |
| Q41 | G3 과열 | 현재가 < 52주최고가 × 0.95 | `src/v8_gates.py:156` |
| Q42 | SAR 필터 | `sar_reversal_up==1` + close > sar | `src/v8_triggers.py:152` |
| Q43 | KOSPI 레짐 | MA20↑+RV<50%→BULL(5) / MA20↑→CAUTION(3) / MA60↑→BEAR(2) / ↓→CRISIS(0) | `scan_buy_candidates.py:793` |
| Q44 | NORMAL vs MOMENTUM | **없음**. 슬롯 수로 공격성 조절 | - |

### 2-4. 레짐 판별

| # | 질문 | 답변 |
|---|------|------|
| Q45 | 레짐 판별 로직 | KOSPI 종가 기준 MA20/MA60 + RV20 백분위 |
| Q46 | 레짐 종류 | 4개: BULL / CAUTION / BEAR / CRISIS |
| Q47 | 슬롯 제한 | 5 / 3 / 2 / 0 |
| Q48 | 전환 조건 | Alpha: 업그레이드 3일 확인, 다운그레이드 즉시. KOSPI: 매일 재계산 |
| Q49 | NIGHTWATCH 연동 | **독립**. US 야간 기반 3색 게이트(GREEN/YELLOW/RED). KOSPI 레짐과 직접 연동 없음 |
| Q50 | BRAIN 5D 연동 | ✅ `swing_pct`→슬롯 환산. 최종 = min(KOSPI슬롯, BRAIN슬롯) 교집합 |

---

<a id="part-3"></a>
## PART 3: 포지션 관리 (Q51-Q74)

### 3-1. 포지션 사이징

| # | 질문 | 답변 | 파일 |
|---|------|------|------|
| Q51 | 결정 로직 | ATR 기반 동적 사이징. 5가지 조정요소 | `src/position_sizer.py` |
| Q52 | 고정 금액? | ❌ 동적 (ATR 변동성 기반) | - |
| Q53 | 고정 비율? | 기본 2% 리스크, 5요소로 조정 | - |
| Q54 | ATR 기반? | ✅ `risk_amount / (ATR × 2.0)` | `position_sizer.py:51` |
| Q55 | 확신도 반영? | ✅ A=1.0 / B=0.67 / C=0.33 | `position_sizer.py` |
| Q56 | 상관관계 조정? | ❌ 미구현 | - |
| Q57 | 섹터 비중 제한? | ❌ 미구현 | - |
| Q58 | 최대 동시 보유 | 5종목(BULL). 레짐별 3→2→0 축소 | `settings.yaml` |
| Q59 | 종목당 최대 비중 | 40% (VETO V7에서 추가 체크) | `position_sizer.py:19` |

### 3-2. 손절/익절/출구 (X1~X5)

| 규칙 | 우선순위 | 조건 | 동작 |
|------|---------|------|------|
| X1 Hard Stop | 1순위 | 진입가 - ATR×2.0 | 전량 청산 |
| X5 Target | 2순위 | 진입가 + ATR×4.0 | 50% 부분청산 |
| X3 Trailing | 3순위 | 최고가 - ATR×2.5 (수익구간만) | 전량 청산 |
| X4 Time | 4순위 | 10일↑ + 수익<2% | 전량 청산 |
| X2 Flow | 5순위 | 기관+외인 3일 연속 순매도 | 전량 청산 |

모든 규칙: `src/alpha/risk_manager.py:196-285`

### 3-3. 포트폴리오 리스크

| # | 질문 | 설정값 | 효과 |
|---|------|--------|------|
| Q67 | 일간 손실 한도 | -3% | V5 VETO + 당일 매수 중단 |
| Q68 | 주간 손실 한도 | -5% | V6 VETO + 포트 1/3 감축 |
| Q69 | MDD 서킷브레이커 | -15% | V4 VETO + 전량 청산 |
| Q70 | 최대 노출 | 레짐별 현금최소: BULL 25% / CAUTION 30% / BEAR 40% / CRISIS 65% |
| Q71 | 현금 비중 관리 | V2 VETO (매수후 현금<최소값 → 거부) |
| Q72 | 주문가능금액 확인 | ✅ KIS API 조회. 최소 50만원 |
| Q73 | 일간 최대 주문 수 | 5종목 (`smart_entry.live.max_stocks`) |
| Q74 | 당일 재매수 금지 | ❌ 미구현 (SmartEntry 1회 실행으로 간접 방지) |

---

<a id="part-4"></a>
## PART 4: 매수/매도 실행 (Q75-Q88)

### 4-1. 매수 실행

| # | 질문 | 답변 |
|---|------|------|
| Q75 | 자동매수 | ✅ SmartEntry (장중) + Paper Trader (장후) |
| Q76 | 매수 시점 | Phase 1(08:55)→Phase 3(09:05)→Phase 4(09:15~10:30)→Phase 5(10:30 취소) |
| Q77 | 매수 방법 | 지정가 (전일종가 -0.5%). 긴급시만 시장가 |
| Q78 | 분할매수 | 백테스트: 3단계(40/40/20%). 라이브: 1회성 |
| Q79 | VETO 체크 | ✅ 8개 조건 (V1~V8). ANY 해당 시 거부 |
| Q80 | 매수후 손절주문 | 백테스트: 자동기록. 라이브: 실시간 모니터링 |

### 4-2. 매도 실행

| # | 질문 | 답변 |
|---|------|------|
| Q81 | 자동매도 | ✅ X1~X5 구현. 현재 dry_run=True |
| Q82 | 매도 트리거 | X1→X5→X3→X4→X2 우선순위 + XP_MDD/WEEKLY 포트폴리오 레벨 |
| Q83 | 매도 시점 | 백테스트: 일일종가. 라이브: BAT-J(17:00) AI 매도판단 |
| Q84 | 매도 방법 | 지정가 기본. 긴급시 시장가 |

### 4-3. 페이퍼 트레이딩

| # | 질문 | 답변 |
|---|------|------|
| Q85 | PAPER 모드 | ✅ SmartEntry paper_mode + paper_trader.py |
| Q86 | 가상 자본금 | 1,500만원 (`paper_trader.py:76`) |
| Q87 | 거래 기록 | `data/order_audit.db` + `data/paper_portfolio.json` |
| Q88 | Paper vs 실전 비교 | ❌ 미구현 (수동 비교만) |

---

<a id="part-5"></a>
## PART 5: 실행 흐름 (Q89-Q96)

### Q89. 하루 실행 타임라인

```
06:10 [BAT-A] 미장 마감 데이터 반영 (7단계)
  └─ US Overnight → COT → 유동성 → 릴레이 → AI Brain → 추천 재스캔

07:00 [BAT-B] 장전 통합 브리핑 (2단계)
  └─ 증권사 리포트 → 테마+뉴스+ETF 텔레그램

08:50 [BAT-E] SmartEntry LIVE (3단계)
  └─ v3 2종목 + TOP7 3종목 = 최대 5종목 자동 지정가

08:55 [BAT-I] VWAP 모니터 (5종목 추적)

16:30 [BAT-D] 장마감 전체 파이프라인 (29단계, ~50분)
  └─ Phase 1: 기초 데이터 → Phase 2: 지표 계산
  └─ Phase 3: 섹터+ETF → Phase 4: 종목 스캔
  └─ Phase 5: 성과+추천 → Phase 6: 아카이브+리포트

17:00 [BAT-J] 포트폴리오 방향 예측
```

### Q90-Q96 요약

| # | 질문 | 답변 |
|---|------|------|
| Q90 | 스케줄러 | Windows Task Scheduler (schtasks) |
| Q91 | BAT 파일 수 | 20개 (A~P) |
| Q92 | Orchestrator | `src/etf/orchestrator.py` (ETF 3축 전용) |
| Q93 | 시작 방법 | Task Scheduler 자동 + 수동 python 실행 |
| Q94 | Kill Switch | `data/KILL_SWITCH` 파일 생성 → 즉시 중단 |
| Q95 | 원격 C2 | ❌ 미구현. Flask 대시보드(읽기전용)만 |
| Q96 | 헬스체크 | 로그 + JSON 존재여부 + 텔레그램(성공만). CPU/메모리 모니터링 없음 |

---

<a id="part-6"></a>
## PART 6: 출력/통신 (Q97-Q107)

### Q97-Q99 메시지 유형 및 포맷

| 유형 | 예시 | 발송 시간 | 채널 |
|------|------|---------|------|
| 아침브리핑 | 테마+뉴스+섹터릴레이+ETF | 07:00 (BAT-B) | Telegram |
| SmartEntry 진입 | 지정가 접수/체결/취소 | 08:50~10:30 (BAT-E) | Telegram |
| VWAP 모니터 | 5종목 실시간 추적 | 08:55 (BAT-I) | Telegram |
| 세력/수급 감지 | 기관+외인 동시매수 | 불규칙 (BAT-F/N) | Telegram |
| 저녁 통합리포트 | 보유종목+DART+추천+밸류체인 | 17:20 (BAT-D [24/29]) | Telegram |
| 완료 알림 | "✅ BAT-D 완료" | 17:30 (BAT-D [27/31]) | Telegram |
| SHIELD 경보 | GREEN→ORANGE 위험 상향 | BAT-D [11.1/29] | Telegram |

### Q100-Q107 플랫폼별 출력

| 플랫폼 | 대상 | 데이터 | 상태 |
|--------|------|--------|------|
| 개인 Telegram | 사용자 1인 | 신호+추천+경보 전체 | ✅ 운영 중 |
| JARVIS (ppwangga.com) | 개인 웹 대시보드 | 12개 배너 (보유/BRAIN/SHIELD/추천 등) | ✅ 운영 중 |
| FLOWX (Supabase) | 구독자 | ETF+수급+섹터 모멘텀 | ✅ 구현 (v13) |

---

<a id="part-7"></a>
## PART 7: 학습/피드백 (Q108-Q121)

### Q108-Q115 학습 시스템

| # | 질문 | 답변 |
|---|------|------|
| Q108 | 학습 모듈 | ✅ `scripts/daily_market_learner.py` (6-Phase) |
| Q109 | 파라미터 보정 | ⚠️ 적중률 추적 O, 가중치 자동반영 ❌ |
| Q110 | 학습 결과 파일 | `data/market_learning/signal_accuracy.json` (10개 신호별 hit_rate) |
| Q111 | 소스별 적중률 | ✅ 10개 신호 추적 (tomorrow_picks, whale_detect 등) |
| Q112 | 수익률 추적 | ✅ `scripts/track_pick_results.py` → `data/daily_performance.json` |
| Q113 | 트레이드 P&L | ⚠️ 백테스트만. 실매매 기록 없음 |
| Q114 | 성과 메트릭 | ✅ PF, 승률, MDD, Sharpe 자동 계산 (`src/quant_metrics.py`) |
| Q115 | 리포트 자동생성 | ✅ 일간(SQLite) + 주간(금요일) |

### Q116-Q121 백테스트/검증

| # | 질문 | 답변 |
|---|------|------|
| Q116 | 백테스트 엔진 | ✅ `src/backtest_engine.py` (1,300줄, 4단계 부분청산) |
| Q117 | 백테스트 기간 | 3~6년 (`data/processed/*.parquet`) |
| Q118 | 성과 산출 | ✅ Sharpe/PF/MDD/승률 자동 |
| Q119 | Walk-Forward | ✅ `src/walk_forward.py` (Train/Test 윈도우) |
| Q120 | Out-of-Sample | ✅ 3년 IS + 1년 OOS + Monte Carlo |
| Q121 | 과적합 감지 | ⚠️ IS vs OOS Sharpe 비교만 |

---

<a id="part-8"></a>
## PART 8: 코드 구조 (Q122-Q131)

### Q122. 프로젝트 구조

```
D:\sub-agent-project\
├── config/                     설정 (settings.yaml 1,460줄)
├── data/                       실시간 데이터 + 학습 결과
│   ├── processed/              parquet 84종목
│   ├── us_market/              US 데이터
│   ├── sector_rotation/        섹터 시그널
│   ├── market_learning/        학습 결과
│   └── *.json                  30+ 시그널 파일
├── scripts/                    20+ BAT + Python 실행 스크립트
├── src/
│   ├── entities/               도메인 모델 (15개)
│   ├── use_cases/              비즈니스 로직 (21개)
│   ├── adapters/               외부 연동 (19개)
│   ├── agents/                 AI 에이전트 (15개)
│   ├── etf/                    ETF 엔진 (12개)
│   ├── alpha/                  Alpha Engine (4개)
│   ├── relay/                  섹터 릴레이 (5개)
│   └── *.py                    핵심 모듈 (~15개)
├── website/                    Flask 대시보드
├── stock_data_daily/           CSV 종목 데이터 (~1,000+)
└── logs/                       실행 로그
```

### Q123. 핵심 파일 TOP 15

| 파일 | 줄수 | 역할 |
|------|------|------|
| `src/signal_engine.py` | 1,660 | 10-Layer 시그널 판정 |
| `src/brain.py` | 1,300 | 9-ARM 자본배분 (BRAIN) |
| `src/html_report.py` | 1,500 | HTML 리포트 생성 |
| `src/backtest_engine.py` | 1,300 | 백테스트 시뮬레이션 |
| `src/indicators.py` | 1,200 | 기술지표 35개 |
| `src/shield.py` | 980 | 포트폴리오 방어 (SHIELD) |
| `scripts/scan_buy_candidates.py` | 1,100+ | v5.0 종목 스캔 |
| `src/v8_scorers.py` | 504 | 5축 가중합 스코어러 |
| `src/walk_forward.py` | 500 | WF 검증 + Monte Carlo |
| `src/daily_archive.py` | 500 | SQLite 아카이브 |
| `src/quant_metrics.py` | 400 | PF/승률/MDD 산출 |
| `src/v8_gates.py` | 300 | G1~G3 Hard Gate |
| `src/v8_triggers.py` | 250 | T1~T4 Trigger |
| `src/etf/orchestrator.py` | 300 | ETF 3축 조율 |
| `src/alpha/risk_manager.py` | 350 | 8-VETO + X1~X5 청산 |

### Q124-Q131 기타

| # | 질문 | 답변 |
|---|------|------|
| Q124 | config 파일 | `config/settings.yaml` (1,460줄, 8개 섹션) |
| Q125 | .env 키 | 15개 API 키 (Anthropic/KIS/Telegram/DART/Perplexity 등) |
| Q126 | 하드코딩 | ADX=14, ATR×2.0 손절, 4단계 R배수, 수수료 0.185% 등 |
| Q127 | SQLite DB | 3개 (daily_archive / us_kr_history / nationality) |
| Q128 | JSON 파일 | 30+ (시그널 15+ / 시장상황 8+ / 학습 3+) |
| Q129 | CSV 파일 | 1,000+ (종목별 일봉) + kospi_index.csv |
| Q130 | Supabase | 7개 테이블 (etf_signals, sector_momentum 등) |
| Q131 | 데이터 보존 | Parquet 전체(~5년) / JSON 최신만(덮어쓰기) / SQLite 전체(~90일) |

---

<a id="part-9"></a>
## PART 9: V2 전환 진단

### 테이블 1: 10개 종목 발굴 소스 V2 처분 판단

| # | 소스명 | 현재 상태 | V2 처분 | 사유 |
|---|--------|----------|---------|------|
| 1 | Parquet v8 Pipeline | ✅ 운영 (84종목) | **유지+리팩터** | L2 Factor 레이어로 재편. Gate/Score/Trigger 구조 보존 |
| 2 | DART 공시 촉매 | ✅ 운영 | **유지** | 이벤트 드리븐 부스트. V2에서 Quality 팩터 입력으로 승격 |
| 3 | Perplexity 뉴스 | ✅ 운영 | **유지** | 감성 점수 → V2 Momentum 팩터의 서브팩터로 통합 |
| 4 | US Overnight Signal | ✅ 운영 | **유지** | L1 Regime의 외부 입력으로 격상 |
| 5 | CSV 백필 (quick_score) | ⚠️ 보조 | **폐기** | parquet 유니버스 확대로 대체 |
| 6 | Consensus Pool | ✅ 독립 운영 | **통합** | V2 Value 팩터(Forward PER/EPS)로 직접 연동 |
| 7 | Relay/Group Relay | ✅ ETF용 | **유지** | ETF 축과 별도 운영. 개별종목 V2와 무관 |
| 8 | SmallCap Explosion | ⚠️ 보조 | **유지** | 유니버스 외 종목 발굴 채널 유지 |
| 9 | Whale Detect | ✅ 운영 | **통합** | V2 Supply/Demand 팩터의 서브팩터로 흡수 |
| 10 | Accumulation Tracker | ✅ 운영 | **통합** | V2 Supply/Demand 팩터의 서브팩터로 흡수 |

### 테이블 2: V2 Layer별 현황 매핑

| Layer | 구성요소 | 현재 존재? | 재사용 가능? | 신규 개발? | 비고 |
|-------|---------|-----------|-------------|-----------|------|
| **L1 REGIME** | KOSPI 레짐 (4등급) | ✅ | ✅ | - | `scan_buy_candidates.py:793` |
| | Alpha Regime (Hysteresis) | ✅ | ✅ | - | `src/alpha/regime.py` (Phase I 구현 완료) |
| | US Overnight 연동 | ✅ | ✅ | - | 3색 매크로게이트 (GREEN/YELLOW/RED) |
| | BRAIN ARM 배분 | ✅ | ⚠️ 부분 | 리팩터 | 9-ARM→L1 제공 파라미터로 재정의 필요 |
| **L2 SD** | 외인 연속매수 | ✅ | ✅ | - | S5 스마트머니 서브팩터 |
| | 기관 연속매수 (ETF보정) | ✅ | ✅ | - | S5 서브팩터 |
| | OBV 다이버전스 | ✅ | ✅ | - | S5 서브팩터 |
| | 프로그램 매매 | ❌ | - | **필요** | KIS API 미지원. 대안 필요 |
| **L2 M** | 기울기 감속 | ✅ | ✅ | - | S4 서브팩터 |
| | 곡률 양전환 | ✅ | ✅ | - | S4 서브팩터 |
| | MACD 히스토그램 | ✅ | ✅ | - | S4 서브팩터 |
| | 이익 모멘텀 (EPS YoY) | ⚠️ | 부분 | **보완** | 분기 EPS 있지만 YoY 자동비교 없음 |
| **L2 V** | PER 할인 | ✅ | ✅ | - | S2 서브팩터 |
| | PBR 백분위 | ✅ | ✅ | - | S2 서브팩터 |
| | EBITDA/EV | ❌ | - | **신규** | DART에 raw 있지만 EV 계산 파이프라인 없음 |
| | FCF Yield | ❌ | - | **신규** | FCF 수집 파이프라인 없음 |
| **L2 Q** | ROE 안정성 | ❌ | - | **신규** | 분기 ROE 시계열 비교 미구현 |
| | 부채 건전성 | ❌ | - | **신규** | 부채비율 DART 수집 가능하지만 자동화 없음 |
| | 이익 품질 | ❌ | - | **신규** | 영업이익/순이익 비율, 발생액 분석 없음 |
| | 배당 지속성 | ❌ | - | **신규** | 배당 히스토리 자동 수집 없음 |
| **L3 SIZE** | ATR 기반 사이징 | ✅ | ✅ | - | `src/position_sizer.py` |
| | 등급별 배수 | ✅ | ✅ | - | A=1.0/B=0.67/C=0.33 |
| | Kelly 기준 | ❌ | - | **신규** | 적중률×배당률 기반 최적 배분 미구현 |
| | 상관관계 기반 | ❌ | - | **신규** | 종목간 상관관계 매트릭스 미구현 |
| **L4 RISK** | 8-VETO 시스템 | ✅ | ✅ | - | `src/alpha/risk_manager.py` (Phase I 완료) |
| | X1~X5 청산 규칙 | ✅ | ✅ | - | Hard/Target/Trailing/Time/Flow |
| | 포트폴리오 방어선 | ✅ | ✅ | - | 일일-3%/주간-5%/MDD-15% |
| | 섹터 집중 리스크 | ❌ | - | **신규** | 동일 섹터 비중 제한 없음 |
| **L5 EXECUTE** | KIS 자동매수 | ✅ | ✅ | - | SmartEntry (08:55~10:30) |
| | KIS 자동매도 | ✅ | ⚠️ | 보완 | 규칙 있지만 dry_run=True |
| | 텔레그램 승인게이트 | ⚠️ | 부분 | 보완 | 기본 구조만 |
| | 체결 모니터링 | ✅ | ✅ | - | 주문상태 조회 + VWAP 모니터 |

### 테이블 3: 4-팩터 데이터 갭 분석

| 팩터 | 서브팩터 | 있나? | 소스 | GAP |
|------|---------|-------|------|-----|
| **SD (수급)** | SD1: 외인 순매수 | ✅ | pykrx + KIS API | - |
| | SD2: 기관 순매수 | ✅ | pykrx + KIS API | ETF 왜곡 보정 필요 (구현됨) |
| | SD3: OBV 다이버전스 | ✅ | 자체 계산 | - |
| | SD4: 프로그램 매매 | ❌ | - | **GAP: API 미지원** |
| **M (모멘텀)** | M1: 가격 모멘텀 | ✅ | 기울기+곡률+MACD | - |
| | M2: 이익 모멘텀 | ⚠️ | DART EPS | **GAP: YoY 자동비교 미구현** |
| | M3: RSI/TRIX | ✅ | 자체 계산 | - |
| | M4: SAR 트렌드 | ✅ | 자체 계산 | - |
| **V (밸류)** | V1: EBITDA/EV | ❌ | - | **GAP: EV 계산 파이프라인 없음** |
| | V2: FCF Yield | ❌ | - | **GAP: FCF 수집 없음** |
| | V3: PER 할인 | ✅ | pykrx + wisereport | - |
| | V4: PBR 백분위 | ✅ | pykrx 3년 히스토리 | - |
| **Q (퀄리티)** | Q1: ROE 안정성 | ❌ | - | **GAP: 분기 ROE 시계열 없음** |
| | Q2: 부채 건전성 | ❌ | - | **GAP: 부채비율 자동화 없음** |
| | Q3: 이익 품질 | ❌ | - | **GAP: 발생액 분석 없음** |
| | Q4: 배당 지속성 | ❌ | - | **GAP: 배당 히스토리 없음** |

**GAP 요약**: 16개 서브팩터 중 **7개 GAP** (44%)
- SD: 1개 GAP (프로그램 매매)
- M: 1개 부분 GAP (이익 모멘텀)
- V: 2개 GAP (EBITDA/EV, FCF)
- Q: 4개 전부 GAP (**퀄리티 팩터 전면 신규 개발 필요**)

### 테이블 4: V2 전환 타임라인 영향도

| Phase | 작업 | 난이도 | 기존 영향 | 소요 추정 |
|-------|------|--------|----------|-----------|
| **Phase I** | L1 REGIME + L4 RISK | ⭐⭐ | 최소 (토글) | ✅ 완료 (`4b4bf22`) |
| **Phase II** | L2 SD+M 팩터 정리 | ⭐⭐ | 중간 (스코어러 리팩터) | 중 (기존 S4/S5 재편) |
| **Phase III** | L2 V 팩터 (EBITDA/FCF) | ⭐⭐⭐ | 낮음 (신규 데이터) | 높 (DART 파이프라인 구축) |
| **Phase IV** | L2 Q 팩터 (전면 신규) | ⭐⭐⭐⭐ | 낮음 (신규 데이터) | **최고** (4개 서브팩터 전부) |
| **Phase V** | L3 SIZE (Kelly+상관) | ⭐⭐⭐ | 중간 (사이징 교체) | 중 (적중률 DB 필요) |
| **Phase VI** | 백테스트 통합 검증 | ⭐⭐ | 높음 (엔진 수정) | 중 (4개 통합포인트 확장) |

### 테이블 5: 핵심 파일 V2 역할 매핑

| 현재 파일 | V2 역할 | 처분 |
|----------|---------|------|
| `src/alpha/regime.py` | L1 REGIME 엔진 | **유지** |
| `src/alpha/risk_manager.py` | L4 RISK VETO+EXIT | **유지** |
| `src/alpha/engine.py` | L1+L4 통합 인터페이스 | **확장** (L2/L3 추가) |
| `src/v8_scorers.py` | → L2 FACTOR 엔진으로 리팩터 | **리팩터** |
| `src/v8_gates.py` | → L1 REGIME의 종목레벨 필터 | **유지** |
| `src/v8_triggers.py` | → L2 MOMENTUM 서브팩터 | **통합** |
| `src/position_sizer.py` | → L3 SIZE 엔진 | **확장** (Kelly 추가) |
| `src/backtest_engine.py` | 검증 엔진 | **확장** (L2/L3 통합포인트) |
| `config/settings.yaml` | 전체 설정 | **확장** (V2 팩터 섹션) |
| `scripts/scan_buy_candidates.py` | 종목 스캔 메인 | **리팩터** (L2 팩터 호출) |

---

## 종합 진단 요약

### 현재 시스템 능력 평가

| 영역 | 점수 | 상세 |
|------|------|------|
| 데이터 수집 | 75/100 | KIS+pykrx+DART 강력. 프로그램매매/EDGAR/상관관계 부재 |
| 스코어링 | 70/100 | 5축 가중합 잘 작동. 컨센서스/퀄리티 축 부재 |
| 포지션 관리 | 85/100 | ATR 사이징+8 VETO+X1~X5 체계적. 상관관계/섹터제한 부재 |
| 실행/자동화 | 80/100 | 20 BAT 완전 자동. Kill Switch/원격제어 미흡 |
| 학습/검증 | 65/100 | 적중률 추적 O, 가중치 반영 X. WF/MC 검증 있음 |
| 코드 품질 | 75/100 | 클린 아키텍처. 일부 1,000줄+ 파일 분할 필요 |

### V2 전환 핵심 판단

1. **L1 REGIME + L4 RISK**: ✅ Phase I 완료. 재활용 가능
2. **L2 SD+M**: ⚠️ 기존 S4/S5 재편으로 70% 커버. 30% 보완
3. **L2 V**: ❌ EBITDA/EV + FCF 파이프라인 신규 구축 필요
4. **L2 Q**: ❌ **전면 신규 개발**. 4개 서브팩터 데이터+로직 전부
5. **L3 SIZE**: ⚠️ ATR 기반 유지 + Kelly/상관관계 추가
6. **L5 EXECUTE**: ✅ 기존 SmartEntry + KIS API 재활용

### V2 전환 우선순위 (추천)

```
Phase I  (완료) ─── L1 REGIME + L4 RISK ──── 방어 레이어 ✅
Phase II (다음) ─── L2 SD + M 재편 ────────── 기존 코드 리팩터
Phase III ──────── L2 V (EBITDA/FCF) ──────── DART 파이프라인
Phase IV ──────── L2 Q (퀄리티 전면) ─────── 최대 난이도
Phase V ───────── L3 SIZE (Kelly+상관) ──── 적중률 DB 기반
Phase VI ──────── 백테스트 통합 검증 ─────── A/B 비교
```

---

*이 문서는 코드 기반 실사 결과입니다. 추측이나 계획 문서가 아닌 실제 파일 내용과 git history를 기반으로 작성되었습니다.*
