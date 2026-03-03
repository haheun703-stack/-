# Quantum Master — 시스템 전체 맵

> 이 파일은 CLAUDE.md에서 분리된 상세 참조 문서입니다.
> 필요할 때 `Read docs/SYSTEM_MAP.md`로 확인하세요.

---

## 1. SignalEngine — 개별종목 시그널 판정

- **파일**: `src/signal_engine.py` (1,660줄, 10-Layer Pipeline)
- **모드**: v8_hybrid — Gate(3개) → Scoring → Trigger(4개)
- **Gate**: G1 ADX≥14, G2 pullback≤0.8, G3 overheat≤0.95
- **Trigger**: T1 TRIX, T2 Volume/RSI(OR), T3 Curvature/OBV, T4 SAR Reversal
- **4축 점수 (100점)**: Quant(30) + Supply/Demand(25) + News(25) + Consensus(20)
- **설정**: `config/settings.yaml` → `signal_engine` 섹션
- **관련 파일**: `src/v8_pipeline.py`, `src/v8_triggers.py`

## 2. US Overnight Signal — 미국장 선행 시그널

- **실행**: `scripts/us_overnight_signal.py --update`
- **L1 Score**: EWY(25%) > NASDAQ(20%) > SP500(15%) = VIX(15%) = SOXX(15%) > Dollar(10%)
- **L2 패턴매칭**: SQLite 기반 유클리드 거리
- **섹터 Kill**: 20개 한국 섹터별 US ETF 급락 시 매수 차단
- **5등급**: STRONG_BULL → STRONG_BEAR
- **데이터**: `data/us_market/overnight_signal.json`, `us_daily.parquet`, `us_kr_history.db`

## 3. AI Brain v3 — Claude API 뉴스/전략 판단 (v12.0 업그레이드)

- **실행**: `scripts/ai_news_brain.py`, `scripts/run_v3_brain.py`
- **7단계 파이프라인** (v12.0):
  - Phase 0 (NEW): GPT o1 Deep Thinking → `data/o1_deep_analysis.json`
  - Phase 1: StrategicBrain (Opus) + o1 컨텍스트 주입
  - Phase 2: SectorStrategist
  - Phase 3+4: DeepAnalyst + Claude Vision (차트 이미지 분석)
  - Phase 5: PortfolioBrain
  - Phase 6 (NEW): Perplexity 교차검증 → `data/perplexity_verification.json`
  - Phase 7: 학습 루프
- **뉴스 소스 5개**: Perplexity, RSS, DART, sector_outlook, US Overnight
- **출력**: `data/ai_brain_judgment.json`, `data/ai_v3_picks.json`, `data/ai_sector_focus.json`
- **v12.0 신규 파일**:
  - `src/chart_renderer.py`: parquet→matplotlib→base64 PNG 차트 렌더링
  - `src/agents/o1_strategist.py`: GPT o1 거시/미시 Deep Thinking
  - `src/agents/perplexity_verifier.py`: Perplexity 팩트체크 교차검증
- **설정**: `config/settings.yaml` → `ai_upgrade` 섹션

## 4. ETF 3축 로테이션

- **실행**: `scripts/run_etf_rotation.py`
- **축1 섹터**: 20개 섹터 ETF 모멘텀 순위 (2주 리밸런싱)
- **축2 레버리지**: BULL + 모멘텀 1위 반도체 → 반도체레버리지(488080) 정밀 타격
- **축3 지수**: KOSPI/MSCI 광역 분산
- **프레데터 모드**: `src/etf/predator_engine.py` — 모멘텀 가속도 + 확신 집중 + 이벤트 트리거
- **AI 필터**: `src/etf/ai_filter.py` — KILL/HOLD/PASS 방어 (공격=룰, 방어=AI)
- **블라인드 테스트**: `data/etf_rotation_blind/YYYY-MM-DD.json`
- **핵심 파일**: `src/etf/sector_engine.py`, `src/etf/orchestrator.py`, `src/etf/config.py`, `src/etf/data_bridge.py`

## 5. 섹터 릴레이 엔진 — US→KR 4단계 릴레이 경보 (2026-03-03)

- **핵심 개념**: US 대장주 확인 → US 2차 연동주 확산 → KR 대장주 확인 → KR 2차 연동주 진입
- **4단계 Phase**: INACTIVE(0) → WATCH(1) → CONFIRM(2) → KR_READY(3) → EXECUTE(4)
- **5개 섹터**:
  - AI반도체 (persistent): NVDA, AVGO, AMD, MU → SK하이닉스, 삼성전자 → 한미반도체, HPSP, ISC
  - 방산 (event): LMT, RTX → 한화에어로, LIG넥스원 → 한화시스템
  - 에너지 (event): XOM, CVX → S-Oil, SK이노베이션 → 한국석유, 중앙에너비스, 흥구석유
  - 배터리/ESS (conditional): TSLA, ENPH → LG에너지솔루션, 삼성SDI → 에코프로비엠, 포스코퓨처엠
  - 조선/LNG (conditional): LNG, GLNG → HD현대중공업, 한화오션 → HSD엔진, STX중공업
- **경보 3개+ 동시 충족 시만 거래**
- **섹터 유형별 차등**: persistent(지속형), event(이벤트형=뉴스 필수), conditional(조건부)
- **실행 규칙**: A형(전일고가 재돌파), B형(VWAP+15분고가 눌림), 손절=VWAP종가이탈/돌파캔들저점이탈
- **핵심 파일**:
  - `config/relay_sectors.yaml`: 5개 섹터 정의
  - `src/relay/us_tracker.py`: US 대장주 가격/레벨 추적 (yfinance)
  - `src/relay/relay_engine.py`: 4단계 릴레이 판정 코어
  - `src/relay/alert_classifier.py`: 경보 분류기
  - `src/relay/execution_rules.py`: 매수/매도 실행 규칙
  - `scripts/run_relay_engine.py`: 통합 실행 (`--update`, `--signal`, `--telegram`, `--all`)
- **데이터**: `data/relay/us_leaders.json`, `relay_signal.json`, `relay_history.json`
- **BAT-A (06:10)**: [3/7] 단계로 통합

## 6. SmartEntry — AI 실시간 진입 판단

- **파일**: `src/use_cases/smart_entry.py` (1,424줄)
- **실행**: `scripts/smart_entry_runner.py` (`--analysis` / `--live`)
- **3축 AI 판단**: 호가창(10호가) + 5분봉 패턴 + 수급 → 0~30점 합산
- **판정**: ≥18 BUY, ≥12 WAIT, <12 SKIP
- **BAT-E**: `scripts/schedule_E_smart_entry.bat`

## 7. 듀얼 AI 매도 시스템 (2026-03-03)

- Claude + GPT 역할 분담: Claude=보수적, GPT=공격적
- REDUCE 원칙 + 점수 합산 투명성

## 8. 밸류트랩 사냥

- **스캐너**: `scripts/scan_value_trap.py`
- **후보**: `data/value_trap_candidates.json`
- **전략**: 저PBR + 현금부자 + 대주주50%+ + 오너리스크 → M&A 카탈리스트

---

## KOSPI 레짐 캡

- BULL(5슬롯): MA20↑ + RV20<50%ile
- CAUTION(3슬롯): MA20↑ + RV20≥50%ile
- BEAR(2슬롯): MA20~MA60
- CRISIS(0슬롯): MA60↓
- 데이터: `data/kospi_index.csv`

---

## BAT 스케줄 (Windows Task Scheduler)

| BAT | 시간 | 역할 |
| --- | --- | --- |
| A (`schedule_A_us_close.bat`) | 06:10 | US Overnight + 릴레이 경보 + AI Brain 재스캔 + 추천종목 + 텔레그램 |
| B (`schedule_B_morning.bat`) | 08:00 | 아침 데이터 갱신 |
| C (`schedule_C_scan.bat`) | 15:30 | 장마감 후 스캔 |
| D (`schedule_D_evening.bat`) | 17:00 | 저녁 종합 (ETF 로테이션 + 블라인드테스트 + 뉴스 + 학습) |
| E (`schedule_E_smart_entry.bat`) | 08:50 | SmartEntry 실행 |

---

## 주요 데이터 경로

| 경로 | 내용 |
| --- | --- |
| `data/processed/*.parquet` | 84종목 일봉 (기계적 유니버스) |
| `data/us_market/` | US 시장 데이터 + overnight_signal |
| `data/relay/` | 릴레이 엔진 결과 |
| `data/sector_rotation/` | 섹터 모멘텀, ETF 시그널, 수급 |
| `data/etf_rotation_blind/` | ETF 블라인드 테스트 로그 |
| `config/settings.yaml` | 전체 설정 (1,460+ 줄) |
| `config/relay_sectors.yaml` | 릴레이 5개 섹터 정의 |

---

## 운영 방침 (2026-03 기준)

- **ETF 3축**: 소액 실전 진입 중 (40% 배분 중 15%만 투입)
- **개별종목 자동매수**: OFF (사용자 수동 매수 중)
- **SmartEntry**: dry_run=True 기본
- **데이터 수집/스캔/추천**: 자동 실행 유지

## 기술 스택

- Python 3.13 + asyncio
- Anthropic Claude API (sonnet-4-5 / haiku-4-5)
- 한국투자증권 API (mojito2)
- yfinance (US 시장 데이터)
- Playwright (HTML→PNG 변환)
- SQLite (US-KR 패턴DB)
