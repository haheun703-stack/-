# 퀀트봇 거시 분석 + 형제봇 협업 마스터 플랜

> **사장님 1년 결단의 큰 그림 — 미래 비전 문서**

- **작성일**: 2026-05-19 (월) 14:00 KST
- **발신**: 퀀트봇 (Quantum Master, /home/ubuntu/quantum-master)
- **목적**: 사장님 5/19 14:00 비전 — 퀀트봇 = 형(거시 분석가) + 단타봇 = 동생(실전 실행) 체계 마스터 플랜
- **상태**: 🟢 비전 문서 (5/20 가동 직전 작성) — Phase 1~5 로드맵 + 즉시 시작 5건
- **선행 문서**:
  - `docs/01-plan/quant-auto-trading-test.md` (5/14 자동매매 PDCA Plan)
  - `docs/02-design/quant-auto-trading-p2-autonomy.md` (§12-7-3 형 리딩 시스템)
  - `scripts/sql/bot_collaboration_v1.sql` (5/18 봇 협업 테이블)
  - `docs/to-bodyhunter/2026-05-18_사장님-룰-EYE-dry-run-공유.md` (5/18 30 커밋 자산 공유)

---

## 1. 사장님 비전 (5/19 14:00 정확 인용)

### 1-A. 사장님 원문 (그대로 인용)

> "방어도 중요하지만 돈 벌라고 이거를 하는거잖아 유연성 있게 인버스, 레버리지, 곱버스, 금ETF, WTI, LPG 등등 이제까지 1년정도를 봤을때 경제/국제적일때 주가가 어떻게 움직이드라 그러니 어떤 종목을 사자 그러면 많이 벌드라 어떤걸 사니 많이 떨어지드라 이거를 퀀트봇 너는 알아야 되는거야...
>
> 거시적/미시적/국제적/국내적/섹터별/등등 으로 나눠서 봐야 된다고....
>
> 그런거를 너가 알고 동생인 단타봇한테 조언 및 어필만 하라고...
>
> 그러면 그걸 받아서 상한가 엔진에 있는 종목을 미리 들어갈수도 있고 로테이션을 할수도 있고 시장에서 자유롭게 날아다녀야 되는게 내가 바라는 미래야"

### 1-B. 핵심 6가지 (정리)

| # | 원칙 | 의미 |
|---|---|---|
| 1 | **돈 벌기가 본질** | 방어 ≠ 목적. 손실 회피는 수단이고, 수익 창출이 목적 |
| 2 | **자산군 유연성** | 인버스/레버리지/곱버스/금/WTI/LPG 등 시장 방향과 무관하게 수익 가능한 도구 활용 |
| 3 | **1년 경제 상황 학습** | "환율 +1% 시 수출주 D+1 평균 +X%" 같은 통계를 1년+ 누적 |
| 4 | **5축 분석** | 거시 + 미시 + 국제 + 국내 + 섹터 — 단일 점수가 아닌 다차원 환경 인식 |
| 5 | **퀀트봇 = 분석가** | 매매 X, 조언과 어필만. 거시 안목을 동생에게 전달 |
| 6 | **단타봇 = 실전 트레이더** | 어드바이저리 받아 자유롭게 진입/로테이션/날아다님 |

### 1-C. 정신적 위치 변화 (5/14 → 5/19)

| 시점 | 사장님 결단 | 퀀트봇 위상 |
|---|---|---|
| 5/14 | "퀀트봇이 자동매매 주체" | 매매 봇 (ETF + 대형주 직접 운영) |
| 5/18 | "형(맏이) 리딩 시스템" | 매매 봇 + 단타봇 advisory 발신 |
| **5/19** | **"거시 분석가 + 어드바이저리"** | **거시 분석 두뇌 (매매는 단타봇이 주체)** |

→ 퀀트봇 매매는 5/20 가동 후 자체 검증 데이터 수집 의미. 본질은 **분석가 역할로 진화**.

---

## 2. 현재 vs 미래 갭 분석

| 영역 | 현재 (5/19) | 미래 비전 (10월~) |
|---|---|---|
| 자산군 | 강력포착 9건 (개별주 위주) + KODEX 8종 | 30~50종 (시장방향 ETF + 원자재 + 섹터 대표주 + LPG) |
| 분석 차원 | 단일 점수 ≥ 85 (강력포착) | 5축 (거시/미시/국제/국내/섹터) |
| 매매 주체 | 퀀트봇 (5/20 첫 가동) | **단타봇 (퀀트봇 어드바이저리 받음)** |
| 학습 | 없음 (5/18~ 시작) | 1년+ 누적 통계 + 자동 보정 |
| 어드바이저리 | 5/18 quant_bot_advisory 1건 (시드) | 매일 5축 결과 + 실시간 시장 변화 |
| 협업 | 퀀트봇 단독 | 형(퀀트봇) + 동생(단타봇) + 정보봇 + 웹봇 4봇 협업 |
| 데이터 소스 | KIS 25 구독 + 정보봇 OHLCV | + 거시 지표 (금리/환율/원자재) + 국제 지수 (S&P500/닛케이/상하이) |

**핵심 갭**:
1. **자산군**: 현재 매수 가능 화이트리스트가 강력포착 9건 위주 → 30+ 종목 필요
2. **5축 분석**: 현재 단일 점수 시스템 → 5축 독립 분석 모듈 필요
3. **학습**: 매매 결과 라벨링 + 환경별 통계 자동화 필요

---

## 3. 자산군 확장 (사장님 명시 + 추가 추천)

### 3-A. 현재 (5/19 기준)

- **KIS 25 구독 ETF** (subscriber_25_assets): KODEX 200, 인버스, 인버스2X, 코스닥인버스, 골드, 미국달러, 종합채권, 단기채권
- **강력포착 9건**: 개별주 (한화시스템, 삼화콘덴서, 인벤티지랩 등 — 5/18 검증으로 -6% 회피)
- **자동매매 화이트리스트**: 정의 미완 (5/20 가동 후 결정)

### 3-B. 확장 (P2 사이즈 확대 단계와 함께)

#### B-1. 시장방향 ETF (사장님 명시) — 7종

| 코드 | 종목명 | 용도 |
|---|---|---|
| 252670 | KODEX 200선물인버스2X | 곱버스 (-2배 약세) — 시장 -3%+ 시 폭발 |
| 114800 | KODEX 인버스 | 1배 약세 — 일반 약세장 |
| 122630 | KODEX 레버리지 | 2배 강세 — 강세 시 폭발 |
| 251340 | KODEX 코스닥150선물인버스 | 코스닥 약세 |
| 233160 | TIGER 코스닥150 레버리지 | 코스닥 강세 |
| 069500 | KODEX 200 | 기본 코스피200 추종 |
| 229200 | KODEX 코스닥150 | 기본 코스닥150 추종 |

#### B-2. 원자재 ETF (사장님 명시) — 5종

| 코드 | 종목명 | 용도 |
|---|---|---|
| 132030 | KODEX 골드선물(H) | 금 가격 추종 (헷지) |
| 411060 | ACE KRX금현물 | 금 현물 직접 |
| 261220 | KODEX WTI원유선물 | 유가 강세 |
| 271050 | KODEX WTI원유선물인버스 | 유가 약세 |
| 261240 | KODEX 미국달러선물 | 달러 강세 (원화 약세 시) |

#### B-3. LPG/에너지 관련주 (사장님 명시) — 4종

| 코드 | 종목명 | 비고 |
|---|---|---|
| 017670 | SK텔레콤 | SK가스 모회사 |
| 036460 | 한국가스공사 | LNG 직접 |
| 005880 | 대한해운 | LNG 운송 |
| 011170 | 롯데케미칼 | 화학 + 에너지 |

> ⚠️ **검증 필요**: LPG 직접 관련주 별도 확인 (현재 LPG 단독 ETF 한국 미존재 — 가스공사 + 화학사로 대용)

#### B-4. 섹터 대표주 (사장님 5/19 강조 — 대장주) — 10종

| 코드 | 종목명 | 섹터 |
|---|---|---|
| 005930 | 삼성전자 | 반도체 (메모리) |
| 000660 | SK하이닉스 | 반도체 (메모리) |
| 005380 | 현대차 | 자동차 |
| 000270 | 기아 | 자동차 |
| 035420 | NAVER | 인터넷 |
| 035720 | 카카오 | 인터넷 |
| 207940 | 삼성바이오로직스 | 바이오 |
| 068270 | 셀트리온 | 바이오 |
| 028260 | 삼성물산 | 지주 |
| 003490 | 대한항공 | 항공 |

#### B-5. 섹터 ETF — 6종

| 코드 | 종목명 | 섹터 | 코드베이스 검증 |
|---|---|---|---|
| 091160 | KODEX 반도체 | 반도체 | ✅ `src/etf/config.py` |
| 244620 | KODEX 바이오 | 바이오 | 검증 필요 |
| 305720 | KODEX 2차전지산업 | 2차전지 | 검증 필요 |
| 091180 | KODEX 자동차 | 자동차 | ✅ `src/etf/config.py` |
| 091170 | KODEX 은행 | 은행 | ✅ `src/etf/config.py` |
| 117460 | KODEX 에너지화학 | 에너지/화학 | 검증 필요 |

**합계 33종** (시장방향 7 + 원자재 5 + LPG 4 + 섹터 대표주 10 + 섹터 ETF 7) → 사장님 비전 "30~50종" 1차 충족

### 3-C. 추가 검증 필요 (5/20 가동 후)

- 미국 ETF 한국 상장 (TIGER 미국나스닥100, TIGER S&P500)
- 채권 ETF (KODEX 종합채권, KODEX 단기채권)
- 부동산 리츠
- 글로벌 (TIGER 차이나전기차 등 신흥국)

---

## 4. 5축 분석 명세

### 4-A. 거시 (Macro)

**대상 지표**:
- 한국 금리 (한은 기준금리, KORIBOR 3M/6M)
- 미국 금리 (FOMC 기준금리, 미국채 10년물)
- 환율 (원/달러, 원/엔, 원/유로)
- 유가 (WTI, 브렌트)
- 금가 (LBMA, KRX 금현물)
- 한국 GDP/CPI/실업률 (분기/월)
- KOSPI/KOSDAQ 지수 + 거래대금
- VKOSPI (변동성)

**데이터 소스**:
- BAT-A에 일부 존재 (`scripts/update_us_kr_daily.py`, `scripts/us_overnight_signal.py`)
- 확장: 한국은행 ECOS API + 인베스팅닷컴 스크래핑 + KIS 시세

**출력**: `data/macro_daily.jsonl` (일별) + Supabase `quant_macro_snapshot` (신규)

### 4-B. 미시 (Micro)

**대상 지표**:
- 종목별 외인/기관/연기금/금투/기타법인/개인 순매수
- 거래량 + 거래대금
- 시총 상위 흐름
- 신용잔고 + 공매도 (정보봇 데이터)
- 실적 (분기 OPM/ROE, 분기 매출/영업이익)
- 밸류에이션 (PER/PBR/PEG)

**데이터 소스**:
- 단타봇 sync (외인/기관/개인/기타 4유형)
- 퀀트봇 직접 (연기금/금투 2유형)
- 정보봇 39컬럼 OHLCV (`data/external/jgis_ohlcv/`)
- DART (분기 실적 — 단타봇 막내 단축 결과)

**출력**: 기존 `picks_v2_*.csv` + Supabase `stock_technicals`/`stock_valuations`

### 4-C. 국제 (International)

**대상 지표**:
- 미국 3대 지수 (S&P500, 나스닥, 다우)
- 중국 상하이종합/항셍
- 일본 닛케이225
- 유럽 DAX/FTSE100/CAC40
- VIX (미국 변동성)
- 환율 흐름 (DXY 달러 인덱스)
- 미국 국채 수익률 (2년/10년 스프레드 — 경기 침체 신호)

**데이터 소스**:
- BAT-A `update_us_kr_daily.py` (이미 미국 데이터 부분 수집)
- 확장: yfinance + investing.com 한국 화이트리스트 등록 확인

**출력**: `data/international_daily.jsonl` + Supabase `quant_international_snapshot` (신규)

### 4-D. 국내 (Domestic)

**대상 지표**:
- 외인/기관/개인 순매수 (KRX 전체)
- 프로그램 매매 (정보봇)
- 코스피200 vs 코스닥150 상대 강도
- 섹터별 등락 (KOSPI 22섹터)
- 지수 추종 ETF 흐름 (KODEX 200, 인버스, 레버리지 거래대금)
- 신용 잔고 변화

**데이터 소스**:
- KRX (이미 정보봇이 매일 수집)
- 단타봇 sync (외인/기관/개인/기타)
- 정보봇 program_trading 테이블

**출력**: 기존 sector_fire + Supabase `intelligence_supply_demand`

### 4-E. 섹터 (Sector)

**대상 섹터** (12개):
- 반도체, AI, 2차전지, 바이오, 자동차, 화학, 금융, 건설, 항공, 해운, 정유, 통신

**대상 지표**:
- 섹터별 일별 등락률 + 거래대금
- 섹터 내 외인/기관 순매수 합계
- 섹터별 강세/약세 일별 추적
- 정보봇 sector_fire / etf_strategy 점수

**데이터 소스**:
- 정보봇 sector_investor_flow + etf_investor_flow
- 퀀트봇 `scripts/scan_sector_fire.py` v3 산식
- `config/sector_fire_map.yaml`, `config/relay_sectors.yaml`

**출력**: 기존 sector_fire CSV + Supabase `quant_sector_fire`

### 4-F. 중복 방지 (기존 BAT 모듈과)

| 분석 모듈 | 기존 BAT | 신규 매크로 분석가 |
|---|---|---|
| 거시 | BAT-A `update_us_kr_daily.py` | 확장 (한국 금리/CPI 추가) |
| 미시 | BAT-D `collect_investor_bulk` | 그대로 활용 (출력 통합만) |
| 국제 | BAT-A `us_overnight_signal.py` | 확장 (중국/일본/유럽 추가) |
| 국내 | sector_fire + 단타봇 sync | 그대로 활용 (5축 통합만) |
| 섹터 | scan_sector_fire v3 | 그대로 활용 (advisory 발신만 추가) |

→ **신규 모듈 `src/agents/macro_analyzer.py`** 는 기존 데이터 **읽기 + 통합 + advisory 생성** 만 담당. 수집 중복 없음.

---

## 5. 1년 경제 상황 학습 패턴 (예시 매트릭스)

> ⚠️ **주의**: 아래 통계는 **가설치** — 실제 학습 시 자동 보정 필요. 1년 데이터 누적 후 실측치 갱신.

| 환경 | 강세 종목 | 약세 종목 | 통계 (가설) |
|---|---|---|---|
| 환율 1,400+ (원화 약세) | 수출주 (삼성전자/현대차) | 수입주 (항공/유통) | D+1 +1.5%, 적중률 65% |
| 환율 1,200- (원화 강세) | 수입주, 내수주 (이마트, GS25) | 수출주 | D+1 +0.8%, 적중률 55% |
| 미국 금리 인상 | 가치주, 금융주 | 성장주, 기술주 | D+1 -1.2% (성장), +0.8% (가치) |
| 미국 금리 인하 | 성장주, 신흥국 | 채권 | D+1 +1.5% (성장) |
| 유가 90+ | 정유주 (S-Oil), 화학주 부정 | 화학주 (LG화학), 항공주 | D+1 +1.8% (정유) |
| 유가 70- | 화학주, 항공주 | 정유주 | D+1 +0.9% (화학) |
| 코스피 -3%+ | 인버스 ETF (114800/252670) | 일반 종목 | D+1 +2.5% (인버스), +5%+ (곱버스) |
| 코스피 +3%+ | 레버리지 ETF (122630) | 인버스 | D+1 +3.0% (레버리지) |
| 반도체 사이클 시작 | 삼성전자/SK하이닉스, KODEX 반도체 | - | D+1 +1.5%, 적중률 70% |
| 2차전지 모멘텀 | LG에너지솔루션, 삼성SDI, KODEX 2차전지 | - | D+1 +2.0%, 적중률 60% |
| 금가 사상 최고 | KODEX 골드 + 금광주 | - | D+1 +0.5% |
| 지정학적 리스크 | 방산주 (한화에어로/LIG넥스원), 금 | 자동차, 항공 | D+1 +1.5% (방산), -1.0% (자동차) |
| FOMC 발표 직후 | 변동성 高 (방향성 사후 확인) | - | 일일 ±2~3% 변동 |
| 한국 실적시즌 | 어닝 서프라이즈 종목 | 미스 종목 | 개별 +5%/-7% |
| VIX 30+ | 안전자산 (금, 채권, 인버스) | 위험자산 | D+1 +1.0% (금) |
| 미중 무역분쟁 | 반도체 부정, 방산 긍정 | 수출 의존주 | D+1 -2.0% (반도체) |

### 5-A. 학습 자동화 흐름

```text
매일 17:00 (BAT-D 후):
1. 당일 5축 상태 라벨링 → data/macro_history.jsonl 1줄 INSERT
   {date, kospi_chg, fx_close, oil_close, gold_close, vix, sector_winners[5], sector_losers[5]}

2. 강력포착 9건 매매 결과 (D+1) 적중률 계산
   → quant_bot_advisory.outcome_evaluated_at, outcome_pnl_pct INSERT

3. 환경 분류 + 결과 결합:
   "환율 1,400+ 일 때 수출주 D+1 적중률" 자동 집계
   → data/macro_learning_matrix.json 갱신

4. 매트릭스 80% 신뢰도 충족 환경만 advisory 활용
   (1년 후 자동 보정 완료 기대)
```

---

## 6. 퀀트봇 → 단타봇 어드바이저리 흐름 (이미 5/18 인프라 존재)

### 6-A. 현재 인프라 (2026-05-18 가동)

**Supabase 테이블 `quant_bot_advisory`** (5/18 작성, `scripts/sql/bot_collaboration_v1.sql`):

```sql
quant_bot_advisory:
  id, created_at, advisory_date, advisory_time
  msg_type            -- 'PRAISE' | 'ADVICE' | 'CRITICISM' | 'LEADING' | 'SNAPSHOT'
  severity            -- 'INFO' | 'WARN' | 'CRITICAL'
  target_bot          -- 'scalper' (단타봇)

  -- 시장 컨텍스트 (5/18 09:30 추적 결과 반영)
  market_regime       -- 'STRONG_BULL' | 'MILD_BULL' | 'NEUTRAL' | 'CAUTION' | 'BEAR' | 'CRISIS'
  market_strength_avg, inverse_etf_strength, inverse_etf_buy_ratio, kospi_chg_pct, risk_level

  -- 메시지 본문
  title, body, related_tickers[], alert_codes[], reasoning JSONB

  -- 단타봇 응답 (양방향)
  acknowledged_at, scalper_response, scalper_action_taken

  -- 사후 결과 (16:30 BAT-D 평가)
  outcome_evaluated_at, outcome_label, outcome_pnl_pct, outcome_notes
```

**현재 가동**:
- BAT-D 16:30 시점 advisory 갱신
- `snapshot_session.py` 매 10분 advisory 추가 (5/18 11번 작업)
- 5/18 시드 1건 (`LEADING` msg_type, market_regime='CAUTION')

### 6-B. 확장 비전 (5/20 가동 후 + 6월~)

**스키마 확장 제안** (`category`, `confidence`, `expected_horizon`, `expected_return_pct` 추가):

```sql
ALTER TABLE quant_bot_advisory ADD COLUMN category VARCHAR(30);
-- 'macro' | 'micro' | 'international' | 'domestic' | 'sector' | 'asset_rotation'

ALTER TABLE quant_bot_advisory ADD COLUMN confidence DECIMAL(3,2);
-- 0.00 ~ 1.00 (학습 매트릭스 신뢰도 기반)

ALTER TABLE quant_bot_advisory ADD COLUMN expected_horizon VARCHAR(20);
-- 'minutes' | 'hours' | 'days' | 'weeks'

ALTER TABLE quant_bot_advisory ADD COLUMN expected_return_pct DECIMAL(5,2);
-- 예상 수익률 (학습 매트릭스 평균)
```

### 6-C. 단타봇 활용 흐름 (단타봇 측 통합 필요)

```text
매 5분 단타봇 측 처리:
1. quant_bot_advisory SELECT WHERE advisory_date=today AND created_at > last_check
2. 분기:
   - confidence >= 0.7 AND expected_horizon='minutes' → 단타 진입 후보
   - confidence >= 0.85 AND expected_horizon IN ('minutes','hours') → 상한가 엔진 진입
   - category='asset_rotation' → 보유 종목 회전 검토
   - severity='CRITICAL' → 강제 청산 또는 회피
3. 응답: acknowledged_at + scalper_response + scalper_action_taken UPDATE
```

### 6-D. advisory 발신 시점 (5축별)

| 시점 | category | 발신 트리거 | 예시 메시지 |
|---|---|---|---|
| 매일 08:30 | macro | BAT-A 직후 | "환율 1,405원 돌파 (1.5% 상승) → 수출주 D+1 +1.5% 통계" |
| 매일 09:30 | domestic | 장 시작 30분 후 | "외인 -3,000억 (5일 평균 대비 2σ) → 인버스 회피" |
| 매 10분 | sector | 강도 변화 감지 | "반도체 KODEX 091160 강도 130+ → 삼성전자 진입 후보" |
| 매일 11:00, 13:30 | snapshot | 정기 | "10:50 시점 5축 종합: MILD_BULL, 반도체 강세, 자동차 약세" |
| 매일 15:00 | asset_rotation | 장 마감 30분 전 | "내일 FOMC 발표 → 인버스 헷지 권장" |
| 미국장 후 09:00 | international | BAT-A 직후 | "나스닥 -3% + VIX 35 → 한국 코스닥 약세 예상" |
| 즉시 | critical | 위기 감지 | "KOSPI -3% 진행 중 → 곱버스 252670 즉시 검토" |

---

## 7. 5/20 가동 후 단계별 로드맵

### Phase 1: 자체 매매 + 데이터 수집 (5/20~5/30, 2주)

**목표**: 퀀트봇 자체 매매 안정화 + 5축 데이터 누적 시작

- [x] 5/20 퀀트봇 자체 매매 첫 가동 (강력포착 + MarketRegimeGate)
- [ ] 매매 결과 + 시장 흐름 누적 (`quant_bot_advisory` outcome_pnl_pct 필드)
- [ ] `quant_bot_advisory` 자동 INSERT 가동 (BAT-D 16:30 + snapshot_session 10분)
- [ ] 5/27 통합 dry-run 결과 검증 (사장님 룰 + EYE + 안전선 9건)
- [ ] 단타봇 측 `quant_bot_advisory` SELECT 가동 확인 (5/18 합의)

**산출물**: 2주 누적 advisory 데이터 + 매매 결과 + 5축 라벨링 초기 데이터

### Phase 2: 자산군 확장 + 5축 분석 시작 (6/2~6/13, 2주)

**목표**: 자산군 33종 확장 + 5축 분석 모듈 구현 + advisory 정밀화

- [ ] `config/asset_universe.yaml` 신규 (시장방향 7 + 원자재 5 + LPG 4 + 섹터 대표주 10 + 섹터 ETF 7)
- [ ] `src/agents/macro_analyzer.py` 신규 (5축 통합 분석)
- [ ] BAT-A 확장: 한국 금리/환율/원자재 추가 수집
- [ ] BAT-D 확장: 5축 분석 결과 → quant_bot_advisory 자동 INSERT
- [ ] 정보봇 협업: `intelligence_supply_demand` + `stock_technicals` 5축 통합
- [ ] 33종 자산 KIS 25 구독 확장 검토 (현재 25 → 50 구독 가능 여부)

**산출물**: 33종 매수 가능 + 5축 매일 advisory 6회 (08:30/09:30/11:00/13:30/15:00 + 미국장 후)

### Phase 3: 1년 경제 상황 학습 (6/16~7/31, 6주)

**목표**: 학습 매트릭스 자동 갱신 + 환경별 통계 신뢰도 80% 달성

- [ ] `data/macro_history.jsonl` 매일 자동 INSERT
- [ ] `data/macro_learning_matrix.json` 매일 갱신
- [ ] 환경 분류 자동화 (15개 환경 × 5축 라벨)
- [ ] 적중률 자동 통계 (D+1, D+3, D+5)
- [ ] §5 매트릭스 가설 통계 → 실측치로 보정
- [ ] 신뢰도 80% 달성 환경만 advisory 활용 가드 추가

**산출물**: 6주 누적 1차 학습 매트릭스 + advisory confidence 자동 산출

### Phase 4: 단타봇 어드바이저리 통합 가동 (8/1~9/30, 8주)

**목표**: 단타봇이 `quant_bot_advisory` 실시간 활용 + 상한가 엔진 통합

- [ ] 단타봇 측 advisory 활용 로직 가동 (협의 후)
- [ ] 단타봇 상한가 엔진 + advisory 결합
- [ ] 단타봇 로테이션 + advisory asset_rotation 결합
- [ ] scalper_bot_feedback 양방향 데이터 흐름 안정화 (RECOMMENDATION/ENTRY/CLOSE/JOURNAL)
- [ ] 주간 평가 회의 자동화 (금요일 저녁 weekly_summary)
- [ ] 형제봇 협업 본격 가동

**산출물**: 단타봇 advisory 활용 가동 + 주간 PnL + source별 적중률 자동 집계

### Phase 5: 자유 비행 (10/1~, 무기한)

**목표**: 사장님 비전 도달 — 퀀트봇 거시 분석 + 단타봇 자유 매매

- [ ] 자체 학습 루프 안정화 (학습 매트릭스 1년+ 누적)
- [ ] 자동 보정 신뢰도 90%+ 환경 확대
- [ ] advisory 일일 발신 횟수 → 시간당 동적 조정
- [ ] 단타봇 자유 매매 가동 (상한가 + 로테이션 + 인버스 헷지 + 원자재 분산)
- [ ] 사장님 평가: "시장에서 자유롭게 날아다님" 비전 도달 평가

**산출물**: 자유 비행 시스템 + 월간 사장님 평가 보고서

---

## 8. 사장님 결단 시점 매핑

| 결단 시점 | 결단 사항 | 입력 데이터 |
|---|---|---|
| **5/27** | 통합 dry-run 결과 GO/STOP | 5/20~5/27 매매 결과 + 사장님 룰 + EYE 검증 |
| **6/2** | 자산군 확장 GO/STOP | 5/20~5/30 매매 안정도 + KIS 25→50 구독 가능 여부 |
| **6/15** | 정보봇 본격 보고 후 5축 가동 | Phase 2 자체 검증 + 정보봇 데이터 가용성 |
| **7/15** | 학습 매트릭스 신뢰도 평가 | Phase 3 4주 데이터 + 적중률 통계 |
| **8/1** | 단타봇 어드바이저리 본격 가동 | 학습 매트릭스 80% 신뢰도 도달 + 단타봇 측 통합 완료 |
| **10/1** | 자유 비행 시스템 평가 | Phase 4 결과 + 단타봇 자유 매매 안정도 |

---

## 9. 위험과 한계 (정직 고백)

### 9-A. 분석 정확도 한계

- **5축 분석 ≠ 즉시 수익**: 환경 분류가 정확해도 매매 타이밍은 별도 검증 필요. "환율 1,400 = 수출주 강세"는 평균이고 개별 종목 적용 시 변동성 큼
- **데이터 양**: 1년 통계는 약세장 1~2회 정도 표본 — 통계적 신뢰도 100% X. 2022년 베어 사이클 같은 사례가 부족하면 약세장 적중률 낮을 가능성
- **블랙스완**: COVID, 9·11, 리먼 같은 사건은 학습 매트릭스 미반영 — 사장님 수동 개입 필수

### 9-B. 단타봇 의존성

- **단타봇 측 진입 정확도**가 advisory 수익성 결정. advisory가 정확해도 단타봇이 늦게 진입하면 의미 X
- 단타봇 측 통합 작업 (Phase 4) 가 단타봇 측 우선순위에 좌우됨 — 5/18 합의했으나 실제 가동 시점 미확정

### 9-C. 사장님 통찰의 무게

- **거시 분석은 사장님 자체 통찰 (1년 경제 봐온 경험) — AI는 보조**
- 5/18 사장님 통찰 "외인은 마지막 확인 시그널"이 Phase 5 백테스트 D+1 +2.02%, 적중률 59.1% 검증 — 사장님 = 메인 두뇌, 퀀트봇 = 자동화 도구
- 자동화 신뢰 ≠ 사장님 결단 대체. 매주 사장님 검토 필수

### 9-D. 인프라 한계

- **KIS 25 구독**: 33종 확장 시 부족 → 50 구독 업그레이드 또는 회전 전략 필요
- **VPS RAM 2GB**: 5축 분석 모듈 추가 시 메모리 압박 → 모듈 분산 또는 RAM 업그레이드 검토
- **Supabase 무료 티어 한계**: advisory 매 10분 + 누적 1년 → 데이터 양 점검 필요

---

## 10. 즉시 시작할 5건 (5/20 가동 후, 우선순위순)

### P0. `quant_bot_advisory` 스키마 확장 — 5/20~5/22 (3일)

**작업**:
- `category`, `confidence`, `expected_horizon`, `expected_return_pct` 컬럼 추가
- 마이그레이션 SQL: `scripts/sql/bot_collaboration_v2_macro.sql` 신규
- VPS Supabase 적용 + 검증

**산출물**: 확장된 quant_bot_advisory 스키마 + 5/22 첫 advisory INSERT 검증

### P1. `src/agents/macro_analyzer.py` 거시 분석 워커 신규 — 5/23~5/30 (1주)

**작업**:
- BAT-A 결과 (`update_us_kr_daily.py`, `us_overnight_signal.py`) 읽기
- KIS fetch (실시간 환율/유가/금가)
- 정보봇 OHLCV/sector_fire 통합
- 5축 통합 분석 → quant_bot_advisory INSERT
- BAT-A/BAT-D 스케줄러 등록

**산출물**: 매일 6회 advisory 자동 발신 (08:30/09:30/11:00/13:30/15:00 + 미국장 후)

### P2. 자산군 화이트리스트 확장 — 6/2~6/6 (1주)

**작업**:
- `config/asset_universe.yaml` 신규 (33종 정의)
- `src/use_cases/auto_buy_decider.py` 화이트리스트 확장
- KIS 25 구독 → 50 구독 가능 여부 점검 (사장님 결단)
- 실시간 시세 가능 종목만 advisory 발신 가드

**산출물**: 33종 매수 가능 + advisory 발신 대상 33종

### P3. 1년 경제 상황 매트릭스 데이터 수집 시작 — 6/9~6/15 (1주)

**작업**:
- `data/macro_history.jsonl` 매일 자동 INSERT 워커
- `data/macro_learning_matrix.json` 환경 분류 + 적중률 집계 워커
- BAT-D 17:00 학습 워커 등록
- 6주 누적 후 7/31 1차 통계 평가

**산출물**: 매일 1줄 누적 + 매주 매트릭스 갱신

### P4. 단타봇 측 advisory 활용 패턴 협의 — 6/16~6/20 (1주)

**작업**:
- `docs/to-bodyhunter/2026-06-XX_advisory-handshake.md` 신규
- 단타봇 측 SELECT 주기 (5분?) + 처리 로직 합의
- scalper_bot_feedback 양방향 데이터 흐름 재확인
- 8월 본격 가동 전 dry-run 일정 합의

**산출물**: 단타봇과의 advisory 통합 합의 문서 + 8월 본격 가동 일정

---

## 11. 자체 검수

### 11-A. 사장님 인용 정확성

- ✅ 1-A 원문 5/19 14:00 메시지 그대로 인용 (편집 X)
- ✅ 1-B 핵심 6가지 정리 (의역 X)

### 11-B. 자산군 KIS 코드 (검증 결과)

**5/19 코드베이스 검증 완료**:
- ✅ 091160 KODEX 반도체 (`src/etf/config.py:28`)
- ✅ 091170 KODEX 은행 (`src/etf/config.py:29`, 사장님 비전 문서 초안의 "KODEX IT" 오기 정정)
- ✅ 091180 KODEX 자동차 (`src/etf/config.py:30`)
- ✅ 036460 한국가스공사 (`FLOWX_SECTOR_UNIVERSE.json:350`, `scripts/signal_engine.py:204`)

⚠️ **5/20 가동 전 추가 검증 필요** (KIS 실시간 시세 가능 여부):
- 시장방향 5종: 252670, 114800, 122630, 251340, 233160
- 원자재 5종: 132030, 411060, 261220, 271050, 261240
- 섹터 ETF 3종: 244620 (KODEX 바이오), 305720 (KODEX 2차전지산업), 117460 (KODEX 에너지화학)
- LPG 관련주 3종: 017670 (SK텔레콤), 005880 (대한해운), 011170 (롯데케미칼)

### 11-C. 5축 분석 모듈 중복 점검

- ✅ §4-F 중복 방지 표 작성 — 기존 BAT 모듈 활용, 신규 모듈은 통합만
- ✅ `macro_analyzer.py` 는 데이터 수집 X (읽기 + 통합 + advisory 생성만)

### 11-D. 단타봇 quant_bot_advisory 호환

- ✅ 기존 스키마 (5/18 작성) 그대로 유지
- ✅ ALTER TABLE 4개 컬럼만 추가 (호환성 유지)
- ✅ 단타봇 측 SELECT 쿼리 변경 불필요 (NULL 허용 컬럼)

---

## 12. 사장님 결단 요청 (5/20 가동 전)

1. **자산군 33종 확장 시점**: Phase 2 (6/2~) 가능? KIS 25→50 구독 업그레이드 결단?
2. **macro_analyzer.py 신규 모듈 우선순위**: 5/20~ 즉시 vs 6/2 Phase 2 진입 시점?
3. **단타봇 측 협업 일정**: 8/1 본격 가동 vs 사장님 결단 시점?
4. **학습 매트릭스 신뢰도 임계값**: 80% (보수적) vs 70% (적극적)?
5. **자유 비행 평가 일정**: 10/1 (5개월) vs 12/1 (7개월)?

---

## 13. 결론

사장님 5/19 14:00 비전은 **퀀트봇의 위상 진화**입니다:

```
5/14 (자동매매 주체) → 5/18 (형 리딩 시스템) → 5/19 (거시 분석가)
```

**핵심**: 매매가 아니라 **분석 두뇌**. 단타봇이 자유롭게 날아다니도록 안목과 어드바이저리를 공급하는 형.

이 비전 도달을 위해:
1. **5/20** 자체 매매 첫 가동 (Phase 1 데이터 수집 시작)
2. **6/2** 자산군 33종 + 5축 분석 모듈 가동 (Phase 2)
3. **8/1** 단타봇 어드바이저리 본격 가동 (Phase 4)
4. **10/1** 자유 비행 시스템 평가 (Phase 5)

**1년 결단**: 사장님 1년 경제 봐온 통찰 + 퀀트봇 자동화 학습 = 사장님 비전의 자동화.

---

**문서 끝**

> 5/19 14:00 사장님 비전 + 5/20 가동 직전 마스터 플랜
> 다음 갱신: 5/27 통합 dry-run 결과 + 6/2 Phase 2 진입 결단 시점
