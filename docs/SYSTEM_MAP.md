# Quantum Master v13.3 — 시스템 전체 맵

> 최종 업데이트: 2026-03-07 (v13.3-arm-expansion)

---

## 시스템 아키텍처 전체도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BRAIN v13.0 — 중앙 두뇌                         │
│  NightWatch + VIX + KOSPI레짐 + SHIELD → 9-ARM 자본배분                │
│                                                                         │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐ ┌────────────────┐ │
│  │ swing   │ │etf_sector│ │etf_lever- │ │etf_index│ │  패시브 4 ARM  │ │
│  │  30%    │ │  모멘텀  │ │  age      │ │ KOSPI/  │ │ gold │small_cap│ │
│  │(개별종목)│ │  20섹터  │ │ 인버스/2x │ │  MSCI   │ │ bonds│ dollar │ │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └────┬────┘ └───────┬────────┘ │
│       │           │             │             │              │          │
│       ▼           └──────┬──────┘             │              │          │
│  SignalEngine        ETF Orchestrator ◄───────┘              │          │
│  + SmartEntry        + AI Filter                PassiveETFEngine        │
│  + 추천 스캔         + Predator                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                      SHIELD v13.2 — 포트폴리오 방어                     │
│  S1 섹터집중 │ S2 MDD킬스위치 │ S3 동시하락 → GREEN/YELLOW/ORANGE/RED  │
│  → BRAIN arms 보정 + frozen_sectors                                    │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                     │
    ┌────▼────┐         ┌────▼────┐           ┌────▼────┐
    │ US Over-│         │ AI Brain│           │ 데이터  │
    │  night  │         │   v3    │           │  수집   │
    │ Signal  │         │ 7-Phase │           │ 29단계  │
    └────┬────┘         └────┬────┘           └────┬────┘
         │                   │                      │
         └───────────┬───────┘                      │
                     ▼                              ▼
              ┌─────────────┐              ┌──────────────┐
              │  텔레그램    │              │ JARVIS 대시보드│
              │  알림/승인   │              │ ppwangga.com │
              └─────────────┘              └──────────────┘
```

---

## 1. BRAIN v13.0 — 중앙 자본배분 두뇌

- **파일**: `src/brain.py`
- **실행**: `scripts/run_brain.py`
- **역할**: 9개 ARM에 자본 배분 결정 (총합 100%)
- **9개 ARM**:

| ARM | 키 | 역할 | 엔진 |
|-----|-----|------|------|
| 개별스윙 | `swing` | v8 시그널 기반 개별종목 | SignalEngine |
| 섹터ETF | `etf_sector` | 20개 섹터 모멘텀 로테이션 | SectorEngine |
| 레버리지 | `etf_leverage` | BULL시 2x / BEAR시 인버스 | LeverageEngine |
| 지수ETF | `etf_index` | KOSPI/MSCI 광역 분산 | IndexEngine |
| 금 | `etf_gold` | 안전자산 헤지 (KODEX 골드선물H, 132030) | PassiveETFEngine |
| 소형주 | `etf_small_cap` | BULL 시 초과수익 (KODEX 코스닥150, 229200) | PassiveETFEngine |
| 채권 | `etf_bonds` | 금리하락 수익/방어 (KODEX 국고채10년, 114820) | PassiveETFEngine |
| 달러 | `etf_dollar` | 환율 헤지 (KODEX 미국달러선물, 261240) | PassiveETFEngine |
| 현금 | `cash` | 무위험 대기 | — |

- **입력 시그널 4개**: NightWatch(-1~+1), VIX(LOW/MID/HIGH), KOSPI레짐(7단계), SHIELD(방어등급)
- **레짐별 비중 매트릭스**:

| 레짐 | sector | lever | index | gold | small | bonds | dollar | cash |
|------|--------|-------|-------|------|-------|-------|--------|------|
| BULL | 30 | 15 | 20 | 5 | 10 | 0 | 0 | 20 |
| PRE_BULL | 25 | 10 | 20 | 5 | 5 | 5 | 0 | 30 |
| CAUTION | 20 | 0 | 20 | 10 | 0 | 10 | 5 | 35 |
| PRE_BEAR | 5 | 15 | 5 | 10 | 0 | 15 | 10 | 40 |
| BEAR | 0 | 15 | 5 | 15 | 0 | 15 | 10 | 40 |
| PRE_CRISIS | 0 | 20 | 0 | 10 | 0 | 15 | 15 | 40 |
| CRISIS | 0 | 20 | 0 | 10 | 0 | 15 | 15 | 40 |

- **충격 보정 (SHOCK_ARM_ADJUSTMENTS)**:
  - GEOPOLITICAL: 레버리지-10, 금+5, 현금+5
  - RATE: 지수-5, 섹터-5, 채권+5, 현금+5
  - LIQUIDITY: 레버리지-15, 섹터-5, 달러+5, 현금+15
  - COMPOUND: 레버리지-15, 섹터-10, 금+5, 달러+5, 현금+15
- **출력**: `data/brain_decision.json`

---

## 2. SHIELD v13.2 — 포트폴리오 방어 시스템

- **파일**: `src/shield.py`
- **실행**: `scripts/run_shield.py`
- **3축 방어**:
  - S1 섹터 집중 리스크: 단일 섹터 30%+ 초과 시 경고
  - S2 MDD 킬스위치: 포트폴리오 MDD 추적 → 임계치 초과 시 자동 차단
  - S3 동시하락 감지: N개 종목 동시 -5%+ 하락 → 시스템 리스크 경고
- **등급**: GREEN → YELLOW → ORANGE → RED
- **BRAIN 연동**: SHIELD 결과를 BRAIN에 주입 → ARM 비중 자동 조정
- **설정**: `config/settings.yaml` → `shield` 섹션
- **출력**: `data/shield_report.json`

---

## 3. SignalEngine — 개별종목 시그널 판정

- **파일**: `src/signal_engine.py` (1,660줄, 10-Layer Pipeline)
- **모드**: v8_hybrid — Gate(3개) → Scoring → Trigger(4개)
- **Gate**: G1 ADX≥14, G2 pullback≤0.8, G3 overheat≤0.95
- **Trigger**: T1 TRIX, T2 Volume/RSI(OR), T3 Curvature/OBV, T4 SAR Reversal
- **4축 점수 (100점)**: Quant(30) + Supply/Demand(25) + News(25) + Consensus(20)
- **설정**: `config/settings.yaml` → `signal_engine` 섹션
- **관련 파일**: `src/v8_pipeline.py`, `src/v8_triggers.py`, `src/v8_gates.py`, `src/v8_scorers.py`

---

## 4. US Overnight Signal — 미국장 선행 시그널

- **실행**: `scripts/us_overnight_signal.py --update`
- **NightWatch 5-Layer** (v12.1):
  - L0 선행: HYG 5D + VIX Term Structure
  - L1 채권자경단: 10Y 금리 변동
  - L4 환율삼각: JPY + KRW + 엔캐리
- **L1 Score**: EWY(25%) > NASDAQ(20%) > SP500(15%) = VIX(15%) = SOXX(15%) > Dollar(10%)
- **L2 패턴매칭**: SQLite 기반 유클리드 거리
- **섹터 Kill**: 20개 한국 섹터별 US ETF 급락 시 매수 차단
- **특수 룰 6개**: VIX_SPIKE, VIX_HIGH, SOXX_CRASH, NASDAQ_CIRCUIT, TRIPLE_BULL, MARKET_CRASH
- **5등급**: STRONG_BULL → STRONG_BEAR
- **선행 레짐** (v12.1): PRE_BEAR / PRE_CRISIS → 인버스/레버리지 2~3일 전 선제 경보
- **데이터**: `data/us_market/overnight_signal.json`, `us_daily.parquet`, `us_kr_history.db`

---

## 5. AI Brain v3 — Claude API 뉴스/전략 판단 (v12.0)

- **실행**: `scripts/ai_news_brain.py`, `scripts/run_v3_brain.py`
- **7단계 파이프라인**:
  - Phase 0: GPT o1 Deep Thinking → `data/o1_deep_analysis.json`
  - Phase 1: StrategicBrain (Opus) + o1 컨텍스트 주입
  - Phase 2: SectorStrategist
  - Phase 3+4: DeepAnalyst + Claude Vision (차트 이미지 분석)
  - Phase 5: PortfolioBrain
  - Phase 6: Perplexity 교차검증 → `data/perplexity_verification.json`
  - Phase 7: 학습 루프
- **뉴스 소스 5개**: Perplexity, RSS, DART, sector_outlook, US Overnight
- **출력**: `data/ai_brain_judgment.json`, `data/ai_v3_picks.json`, `data/ai_sector_focus.json`
- **핵심 에이전트 파일**:
  - `src/agents/strategic_brain.py`, `src/agents/sector_strategist.py`
  - `src/agents/deep_analyst.py`, `src/agents/portfolio_brain.py`
  - `src/agents/o1_strategist.py`, `src/agents/perplexity_verifier.py`
  - `src/chart_renderer.py`: parquet→matplotlib→base64 PNG 차트

---

## 6. ETF 3축 로테이션 + 패시브 4 ARM

- **실행**: `scripts/run_etf_rotation.py`
- **오케스트레이터**: `src/etf/orchestrator.py` (5단계 실행)
  - Step 1: BRAIN 비중 수신
  - Step 2: 축1 섹터 로테이션 (`src/etf/sector_engine.py`)
  - Step 3: 축2 레버리지/인버스 (`src/etf/leverage_engine.py`)
  - Step 4: 축3 지수 ETF (`src/etf/index_engine.py`)
  - Step 4.5: 패시브 ETF 금/소형주/채권/달러 (`src/etf/passive_engine.py`)
  - Step 5: 통합 시그널 출력
- **프레데터 모드**: `src/etf/predator_engine.py` — 모멘텀 가속도 + 확신 집중
- **AI 필터**: `src/etf/ai_filter.py` — KILL/HOLD/PASS (공격=룰, 방어=AI)
- **블라인드 테스트**: `data/etf_rotation_blind/YYYY-MM-DD.json`
- **설정**: `config/settings.yaml` → `etf_rotation` 섹션

---

## 7. 섹터 릴레이 엔진 — US→KR 4단계 경보

- **4단계**: INACTIVE → WATCH → CONFIRM → KR_READY → EXECUTE
- **5개 섹터**: AI반도체, 방산, 에너지, 배터리/ESS, 조선/LNG
- **핵심 파일**:
  - `config/relay_sectors.yaml`: 섹터 정의
  - `src/relay/relay_engine.py`: 판정 코어
  - `src/relay/us_tracker.py`: US 가격 추적
  - `scripts/run_relay_engine.py`: 통합 실행
- **데이터**: `data/relay/`

---

## 8. SmartEntry — AI 실시간 진입 판단

- **파일**: `src/use_cases/smart_entry.py` (1,424줄)
- **3축 AI 판단**: 호가창(10호가) + 5분봉 패턴 + 수급 → 0~30점
- **판정**: ≥18 BUY, ≥12 WAIT, <12 SKIP
- **BAT-E**: `scripts/schedule_E_smart_entry.bat`

---

## 9. 종목 스캔 시스템 (8개)

| 스캔 | 스크립트 | 출력 | 대시보드 배너 |
|------|---------|------|--------------|
| 내일 추천 | `scan_tomorrow_picks.py` | `tomorrow_picks.json` | 내일추천종목 |
| 세력감지 | `scan_whale_detect.py` | `whale_detect.json` | 세력감지 |
| 동반매수 | `scan_dual_buying.py` | `dual_buying_watch.json` | 동반매수 |
| 눌림목 | `scan_pullback.py` | `pullback_scan.json` | 눌림목 |
| 수급폭발 | `scan_volume_spike.py` | `volume_spike_watchlist.json` | — |
| 밸류체인 | `scan_value_chain.py` | `value_chain_relay.json` | — |
| 매집추적 | `scan_accumulation_tracker.py` | `accumulation_tracker.json` | — |
| 밸류트랩 | `scan_value_trap.py` | `value_trap_candidates.json` | — |

---

## 10. 차이나머니 수급 감지 (v12.2)

- **실행**: `scripts/crawl_china_money.py`
- **방법**: KIS API 외국인 국적별 매매 + EWY 프록시
- **등급**: SURGE / INFLOW / OUTFLOW
- **출력**: `data/china_money/china_money_signal.json`

---

## 11. JARVIS 대시보드 — ppwangga.com

- **Flask 앱**: `website/flask_app.py`
- **템플릿**: `website/templates/dashboard.html` (~1,300줄)
- **데이터 빌드**: `scripts/build_brain_upload.py` → `website/data/brain_data_upload.json`
- **업로드**: `src/adapters/jarvis_uploader.py` (5개 API)
  - `/api/upload` — HTML 리포트
  - `/api/metrics` — 포트폴리오 메트릭스
  - `/api/market` — 추천종목 + US시그널 + 레짐
  - `/api/holdings` — 보유주식 (KIS 잔고)
  - `/api/brain` — AI Brain + BRAIN배분 + SHIELD 전체
- **대시보드 배너**: 보유종목, BRAIN 9-ARM 차트, SHIELD 방어, 추천종목, 세력감지, 동반매수, 눌림목, 그룹릴레이, 차이나머니, US Overnight, 섹터모멘텀, 매매일지

---

## KOSPI 레짐 (7단계)

| 레짐 | 조건 | 스윙 슬롯 |
|------|------|----------|
| BULL | MA20↑ + RV20<50%ile | 5 |
| PRE_BULL | NW 선행 시그널 | 4 |
| CAUTION | MA20↑ + RV20≥50%ile | 3 |
| PRE_BEAR | NW 하락 선행 시그널 | 2 |
| BEAR | MA20~MA60 | 2 |
| PRE_CRISIS | NW 급락 선행 시그널 | 1 |
| CRISIS | MA60↓ | 0 |

---

## BAT 스케줄 (Windows Task Scheduler)

| BAT | 시간 | 역할 | 단계 수 |
|-----|------|------|--------|
| A (`schedule_A_us_close.bat`) | 06:10 | US Overnight + 릴레이 경보 + AI Brain 재스캔 + 추천 | ~7 |
| B (`schedule_B_morning.bat`) | 08:00 | 아침 데이터 갱신 | ~5 |
| C (`schedule_C_etf_alert.bat`) | 15:30 | 장마감 후 ETF 알림 | ~3 |
| D (`schedule_D_after_close.bat`) | 16:30 | **전체 파이프라인** (29단계) — 데이터→지표→스캔→추천→BRAIN→SHIELD→ETF→대시보드 | 29 |
| E (`schedule_E_smart_entry.bat`) | 08:50 | SmartEntry 실행 | ~3 |
| F (`schedule_F_sniper_watch.bat`) | 장중 | 스나이퍼 감시 | ~2 |
| G (`schedule_G_friday_dip.bat`) | 금요일 | 금요 눌림목 | ~2 |

---

## BAT-D 29단계 파이프라인 상세

```
PHASE 1: 기초 데이터 (~15분)
  [1] CSV 전종목 종가 업데이트 (FinanceDataReader)
  [2] Parquet 유니버스 증분 업데이트 (pykrx)
  [3] 수급 데이터 수집 (외인/기관 매매동향)
  [4] KOSPI 인덱스 업데이트

PHASE 2: 지표 계산 (~10분)
  [5] 기술지표 재계산 (35개 지표)
  [6] US Overnight Signal 갱신
  [7] US-KR 패턴매칭 DB 일일 누적

PHASE 3: 섹터 + ETF (~10분)
  [8] 섹터 ETF 시세 업데이트
  [9] 섹터 모멘텀 + z-score + 수급 + 통합 리포트
  [9c.5] 차이나머니 수급 크롤링
  [9c.7] ETF 투자자별 수급 수집
  [10] ETF 마스터 데이터 빌드
  [11] ETF 매매 시그널 생성
  [11.1] SHIELD 포트폴리오 방어 점검
  [11.2] BRAIN 자본배분 결정 (9-ARM)
  [11.3] ETF 3축 로테이션 (블라인드 테스트)

PHASE 4: 종목 스캔 (~10분)
  [11.5] 레버리지/인버스 ETF 스캔
  [12] 눌림목 스캔
  [12.5] 수급폭발 스캐너
  [12.6] 소형주 급등 포착
  [12.7] 밸류체인 릴레이 스캔
  [13] DART 전자공시 크롤링
  [13.5] 레짐 매크로 시그널
  [14] 시장 뉴스 크롤링
  [15] 세력감지 스캔
  [16] 세력감지 하이브리드
  [17] 동반매수 스캔
  [17.5] 섹터 릴레이 시그널
  [18] 그룹 릴레이 감지
  [18.5] 매집 추적 스캔

PHASE 5: 성과추적 + 추천 (~5분)
  [19] 추천 성과 추적
  [19.3] DART 이벤트 드리븐 시그널
  [19.5] 기관 추정 목표가 계산
  [19.6] 보유종목 동적 목표가 재판정
  [19.7] Perplexity 시장 인텔리전스
  [19.8] AI 두뇌 뉴스 분석
  [19.9] v3 AI Brain 5단계 깔때기
  [20] 내일 추천 종목 통합 스캔 ★
  [20.7] 멀티전략 포트폴리오 배분
  [20.8] 3단 예측 체인

PHASE 6: 아카이브 + 보고서 (~1분)
  [21] 일일 아카이브 (JSON → SQLite)
  [21.5] v3 Brain 일일 성과 리뷰
  [22] 주간 보고서 (금요일만)
  [23] JARVIS 대시보드 업로드
  [23.5] Brain 대시보드 빌드 + API 업로드
  [24] 저녁 통합 텔레그램
```

---

## 클린 아키텍처 디렉토리

```
src/
├── entities/          # 도메인 모델 (15개 모델 파일)
├── use_cases/         # 비즈니스 로직 (21개)
│   ├── smart_entry.py        # AI 실시간 진입
│   ├── live_trading.py       # 실매매 실행
│   └── ...
├── adapters/          # 외부 인터페이스 (19개)
│   ├── kis_order_adapter.py  # 한투 주문 API
│   ├── jarvis_uploader.py    # 대시보드 업로드
│   └── ...
├── agents/            # AI 에이전트 (20개)
│   ├── strategic_brain.py    # Opus 전략
│   ├── sell_brain.py         # 매도 AI
│   ├── etf_brain.py          # ETF 전용 AI
│   └── ...
├── etf/               # ETF 엔진 (12개)
│   ├── orchestrator.py       # 통합 오케스트레이터
│   ├── sector_engine.py      # 축1 섹터
│   ├── leverage_engine.py    # 축2 레버리지
│   ├── index_engine.py       # 축3 지수
│   ├── passive_engine.py     # 금/소형주/채권/달러
│   ├── predator_engine.py    # 프레데터 모드
│   └── ai_filter.py          # AI 방어 필터
├── relay/             # 릴레이 엔진 (5개)
├── brain.py           # BRAIN 중앙 두뇌
├── shield.py          # SHIELD 방어
├── signal_engine.py   # 시그널 엔진 (1,660줄)
└── indicators.py      # 기술지표 (35개)
```

---

## 주요 데이터 경로

| 경로 | 내용 |
|------|------|
| `data/processed/*.parquet` | 84종목 일봉 (기계적 유니버스) |
| `data/us_market/` | US 시장 데이터 + overnight_signal |
| `data/relay/` | 릴레이 엔진 결과 |
| `data/sector_rotation/` | 섹터 모멘텀, ETF 시그널, 수급 |
| `data/etf_rotation_blind/` | ETF 블라인드 테스트 로그 |
| `data/china_money/` | 차이나머니 수급 시그널 |
| `data/brain_decision.json` | BRAIN 자본배분 결과 |
| `data/shield_report.json` | SHIELD 방어 리포트 |
| `website/data/` | 대시보드 업로드 데이터 |
| `config/settings.yaml` | 전체 설정 (1,460+ 줄) |
| `config/relay_sectors.yaml` | 릴레이 5개 섹터 정의 |

---

## 기술 스택

- Python 3.13 + asyncio
- Anthropic Claude API (claude-sonnet-4-5 / claude-haiku-4-5)
- 한국투자증권 API (mojito2) — REAL 모드
- yfinance (US 시장 데이터)
- Flask (ppwangga.com 대시보드)
- Playwright (HTML→PNG 변환)
- SQLite (US-KR 패턴DB, 일일 아카이브)

---

## 운영 방침 (2026-03-07 기준)

- **자동매매 전면 금지**: 매수/매도 모두 수동 (텔레그램 승인 게이트)
- **SmartEntry**: dry_run=True 기본
- **데이터 수집/스캔/추천**: BAT-D 자동 실행 유지
- **ETF 3축**: 소액 실전 진입 중 (BRAIN 배분 참조)

---

## 버전 히스토리 (최근)

| 태그 | 날짜 | 내용 |
|------|------|------|
| v13.3-arm-expansion | 03-07 | 금/소형주/채권/달러 4개 패시브 ARM 추가 |
| v13.2 | 03-07 | SHIELD 포트폴리오 방어 시스템 |
| v13.1 | 03-07 | 스윙 ARM-BRAIN 연결 (슬롯 동적 조절) |
| v13.0 | 03-07 | BRAIN 자본배분 중앙 두뇌 (5-ARM) |
| v12.2 | 03-04 | 차이나머니 수급 감지 |
| v12.1 | 03-04 | 레짐 선행 시그널 (PRE_BEAR/PRE_CRISIS) |
| v12.0 | 03-04 | AI 역할 업그레이드 (Vision+o1+Perplexity) |
| v11.2 | 02-26 | 기관 추정 목표가 역산 엔진 |
| v10.5-sar | 02-28 | Parabolic SAR 통합 |
