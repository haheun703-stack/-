# Plan: TIER2 — 섹터 로테이션 + 기관 수급 + 펀드 플로우 통합

> Feature: tier2-sector-rotation-institutional-flow
> Created: 2026-03-27
> Status: Draft
> PDCA Phase: Plan

---

## 1. 문제 정의 (Problem Statement)

### 현재 상황

Quantum Master v13.3은 **개별종목 시그널** 중심으로 설계되어 있으며, 섹터/기관 차원의 자금 흐름 분석은 TIER1에서 기초 인프라만 구축된 상태이다.

| 영역 | 현재 상태 | 갭 |
|------|----------|-----|
| 섹터 로테이션 | Phase 1-5 완성 (모멘텀/z-score/수급/리포트) | 자동 매매 시그널 미연결, BRAIN 직접 주입 없음 |
| 기관 수급 | 6D 프레임워크 (L1/L5) — 개별종목 레벨 | **기관 유형별 세분화 없음** (금투/보험/투신/연기금/은행) |
| 펀드 플로우 | **미구현** | 연기금·투신 대형 매매 추적 없음 |
| ETF 수급 왜곡 | flow_distortion.py 구현 완료 | 섹터 시그널에 미통합 |
| 파생 선행 | derivatives_collector.py 구현 | BRAIN/섹터에 미연결 |
| COT/유동성 | cot_tracker + liquidity_tracker 완성 | BRAIN에 부분 연결, 섹터 미연결 |

### 핵심 문제

1. **기관 유형별 행동 차이를 무시**: 연기금(장기)과 금투(단기)의 매매 의미가 다른데, 합산 "기관합계"만 사용
2. **섹터→종목 시그널 단절**: 섹터 모멘텀이 강해도 개별종목 추천에 반영 안 됨
3. **펀드 플로우 사각지대**: 연기금/투신 대규모 매집은 2~4주 선행 신호인데 추적 불가
4. **Top-Down ↔ Bottom-Up 단절**: BRAIN 자본배분(Top-Down)과 종목 시그널(Bottom-Up) 사이에 섹터 연결고리 부재

---

## 2. 목표 (Objectives)

### 정량 목표

| 지표 | 현재 | 목표 | 비고 |
|------|------|------|------|
| 섹터 로테이션 승률 | 미측정 | >= 55% | 20일 보유 기준 |
| 기관 매집 선행성 | 미측정 | 1~3일 선행 | 기관 순매수 5일 연속 → 종목 상승 |
| 펀드 플로우 감지율 | 0% | >= 80% | 연기금 대규모 매집 이벤트 포착 |
| 추천 종목 중 섹터 부스트 반영률 | 0% | >= 40% | 강세 섹터 종목 비중 |

### 정성 목표

1. **Top-Down → Bottom-Up 연결**: BRAIN 섹터배분 → 섹터 모멘텀 → 종목 시그널 파이프라인 완성
2. **기관 행동 프로파일링**: 연기금/금투/외인 각각의 매매 패턴을 시그널로 변환
3. **"스마트 머니 따라가기"**: 연기금+외인 동시 매집 섹터를 자동 감지하여 우선 추천

---

## 3. TIER2 아키텍처 설계

### 3.1 전체 구조: 3-Layer Institutional Flow Pipeline

```
Layer 3: BRAIN Integration (자본배분 보정)
  ├── sector_momentum_signal → BRAIN arms 가중치 조정
  ├── institutional_regime → 기관 매집/매도 레짐
  └── liquidity_macro → COT/유동성 통합

Layer 2: Sector Aggregator (섹터 수급 통합)
  ├── sector_fund_flow → 연기금/투신 섹터별 순매수
  ├── sector_institutional_map → 기관 유형별 섹터 집중도
  ├── etf_flow_leading → ETF 수급 왜곡 보정 선행 신호
  └── sector_momentum_enhanced → 모멘텀 + 수급 + 펀드 통합 점수

Layer 1: Data Collection (데이터 수집)
  ├── [기존] pykrx_supply_adapter → 종목별 투자자 수급
  ├── [기존] kis_investor_adapter → KIS API 수급
  ├── [신규] institutional_flow_collector → 기관 유형별 세분화 수집
  ├── [신규] fund_flow_tracker → 연기금/투신 추적
  └── [기존] collect_etf_investor_flow → ETF 투자자 플로우
```

### 3.2 모듈별 상세

#### Module A: 기관 유형별 수급 세분화 (Institutional Flow Decomposition)

**현재**: KIS/PyKRX → "기관합계", "외국인합계"만 수집
**목표**: 6개 기관 유형 분리

```
기관 유형 (KRX WICS 기준):
  1. 금융투자 (증권사 자기매매 + HFT) — 단기 노이즈 多
  2. 보험 — 중기 (3~6개월), 배당주/방어주 선호
  3. 투신 (자산운용사) — 중기, 펀드 설정/환매에 따른 기계적 매매
  4. 연기금 (국민연금 등) — 장기, 가장 스마트한 매매 (선행성 높음)
  5. 은행 — 소규모, 무시 가능
  6. 기타금융 (사모펀드 등) — 이벤트 드리븐
```

**핵심 데이터 소스**: PyKRX `stock.get_market_trading_value_by_date()`
- 이미 기관 유형별 분리 데이터 제공
- 현재 `fetch_investor_flow()`에서 "기관합계"만 사용 → 세분화 확장

**가중치 체계**:
```python
INSTITUTIONAL_WEIGHTS = {
    "pension_fund":    0.35,  # 연기금: 가장 높은 선행성
    "insurance":       0.20,  # 보험: 중기 방향성
    "asset_mgmt":      0.20,  # 투신: 펀드 플로우 반영
    "foreign":         0.15,  # 외국인: 이미 별도 추적
    "securities":      0.07,  # 금투: 노이즈 多, 낮은 가중치
    "other_financial": 0.03,  # 기타: 최소 가중치
}
```

#### Module B: 펀드 플로우 트래커 (Fund Flow Tracker)

**현재**: 미구현
**목표**: 연기금/투신의 섹터별 자금 흐름 추적

**데이터 소스 3가지**:

1. **PyKRX 기관 유형별 매매동향** (가장 현실적)
   - `stock.get_market_trading_value_by_date(start, end, ticker, detail=True)`
   - 기관 유형별(금투/보험/투신/연기금 등) 일별 순매수 데이터
   - 유니버스 상위 100종목에 대해 수집 → 섹터별 합산

2. **금융투자협회 펀드 설정/환매** (중기 신호)
   - 주식형 펀드 설정액 증가 → 강세 신호
   - 환매 급증 → 바닥 근처 역발상 신호
   - 크롤링: `kofia.or.kr` 펀드 순자산 변동

3. **ETF 설정/환매 프록시** (이미 부분 구현)
   - 섹터 ETF 순자산 변동 = 기관 자금 유입/유출 프록시
   - `collect_etf_investor_flow.py` 확장

**핵심 지표**:
```python
# 섹터별 펀드 플로우 점수
sector_fund_score = {
    "pension_5d_net":   pension_cum_5d,     # 연기금 5일 누적 순매수
    "pension_20d_net":  pension_cum_20d,    # 연기금 20일 누적 (추세)
    "asset_mgmt_flow":  fund_setting_ratio, # 투신 설정/환매 비율
    "etf_creation":     etf_nav_change,     # ETF 순자산 변동
    "smart_money_index": (pension_5d + insurance_5d) / total_volume  # 스마트머니 비중
}
```

#### Module C: 섹터 로테이션 자동화 (Enhanced Sector Rotation)

**현재**: sector_momentum.py → sector_daily_report.py (수동 참조)
**목표**: 자동 매매 시그널 + BRAIN 직접 주입

**통합 섹터 점수 (0~100)**:
```python
sector_composite_score = (
    0.30 * momentum_score          # 기존: 5/20/60일 모멘텀
    + 0.25 * institutional_score   # 신규: 기관 유형별 가중 수급
    + 0.20 * fund_flow_score       # 신규: 펀드 플로우
    + 0.15 * relative_strength     # 기존: KRX300 대비 상대강도
    + 0.10 * technical_score       # 기존: RSI/MA 기술지표
)
```

**섹터 레짐 분류**:
```
STRONG_ROTATION (>= 80): 모멘텀 + 수급 + 펀드 모두 일치 → 적극 매수
MODERATE_ROTATION (60~80): 2개 이상 일치 → 관심
NEUTRAL (40~60): 혼조 → 기존 보유 유지
WEAK_ROTATION (20~40): 이탈 신호 → 비중 축소
EXODUS (<20): 수급 이탈 + 모멘텀 꺾임 → 즉시 청산
```

**BRAIN 연동**:
- `sector_composite_score` → BRAIN의 `etf_sector` ARM 비중 조정
- STRONG_ROTATION 섹터가 3개+ → 전체 섹터 비중 +5%
- EXODUS 섹터가 3개+ → 섹터 비중 -10%, 현금 +10%

#### Module D: 종목 시그널 부스트 (Signal Boost from Sector)

**현재**: SignalEngine 4축 100점 = Quant(30) + Supply(25) + News(25) + Consensus(20)
**목표**: 섹터 컨텍스트 부스트 추가

```python
# 기존 4축 점수에 섹터 부스트 적용 (가산이 아닌 승수)
if sector_regime == "STRONG_ROTATION":
    sector_boost = 1.15   # +15% 부스트
elif sector_regime == "MODERATE_ROTATION":
    sector_boost = 1.05   # +5% 부스트
elif sector_regime == "WEAK_ROTATION":
    sector_boost = 0.90   # -10% 페널티
elif sector_regime == "EXODUS":
    sector_boost = 0.75   # -25% 강한 페널티
else:
    sector_boost = 1.00   # 중립

final_score = base_score * sector_boost
```

---

## 4. 실행 로드맵

### Phase 0: 즉시 적용 (1~2일, 기대효과: HIGH)

**0-1. 기관 유형별 수급 세분화 수집**
- `pykrx_supply_adapter.py` → `fetch_investor_flow()` 확장
- PyKRX `detail=True` 옵션으로 6개 기관 유형 분리
- 저장: `data/supply_demand/institutional_detail/YYYY-MM-DD.json`
- 예상 소요: 3~4시간

**0-2. 연기금 매집 감지기 (Pension Fund Detector)**
- 연기금 5일 연속 순매수 + 금액 임계치(50억+) → "연기금 매집" 알림
- 텔레그램 알림 + morning_recommendation에 "[연기금]" 태그
- 예상 소요: 2~3시간

### Phase 1: 핵심 기능 (3~5일, 기대효과: HIGH)

**1-1. 섹터별 기관 플로우 집계**
- 23개 섹터별 기관 유형 순매수 합산
- `institutional_flow_collector.py` 신규 스크립트
- BAT-D Phase 3에 추가 (섹터 관련 단계 이후)
- 예상 소요: 4~5시간

**1-2. 섹터 통합 점수 엔진 (Sector Composite Scorer)**
- `src/sector_composite.py` 신규
- 모멘텀 + 기관수급 + 상대강도 + 기술지표 통합
- 섹터 레짐 분류 (STRONG → EXODUS)
- 예상 소요: 6~8시간

**1-3. BRAIN 섹터 시그널 주입**
- `brain.py`에 `sector_composite_signal` 입력 추가 (5번째 시그널)
- 섹터 레짐에 따른 `etf_sector` ARM 동적 조정
- 예상 소요: 3~4시간

### Phase 2: 확장 기능 (5~7일, 기대효과: MEDIUM-HIGH)

**2-1. 펀드 플로우 트래커**
- PyKRX 기관 유형별 → 섹터 합산 → 5/20일 누적 추세
- ETF 순자산 변동 프록시 추가
- `scripts/fund_flow_tracker.py` 신규
- 예상 소요: 6~8시간

**2-2. 종목 시그널 섹터 부스트**
- `signal_engine.py`에 sector_boost 승수 적용
- scan_tomorrow_picks에 섹터 레짐 표시
- 예상 소요: 4~5시간

**2-3. JARVIS 대시보드 — 섹터 히트맵**
- 23개 섹터 기관 수급 히트맵 (연기금/외인/금투)
- 대시보드 배너 추가: "기관 수급 히트맵"
- 예상 소요: 5~6시간

### Phase 3: 고도화 (1~2주, 기대효과: MEDIUM)

**3-1. 섹터 로테이션 백테스트 프레임워크**
- 2019~2026 섹터 모멘텀 + 수급 기반 로테이션 성과 검증
- 최적 가중치/임계값 도출
- 예상 소요: 10~12시간

**3-2. 파생 선행 → 섹터 연결**
- derivatives_collector의 풋/콜 프록시 → BRAIN 입력
- 선물 베이시스 → 레버리지 ARM 조정
- 예상 소요: 5~6시간

**3-3. 기관 행동 프로파일러**
- 연기금의 섹터별 매매 패턴 학습 (역발상 vs 추세추종)
- 과거 매집 이벤트 → 이후 수익률 통계
- 예상 소요: 8~10시간

---

## 5. ROI 순위 정리

| 순위 | 항목 | 난이도 | 기대효과 | ROI |
|------|------|--------|---------|-----|
| 1 | 0-1. 기관 유형별 수급 세분화 | LOW | HIGH | **VERY HIGH** |
| 2 | 0-2. 연기금 매집 감지기 | LOW | HIGH | **VERY HIGH** |
| 3 | 1-1. 섹터별 기관 플로우 집계 | MEDIUM | HIGH | HIGH |
| 4 | 1-2. 섹터 통합 점수 엔진 | MEDIUM | HIGH | HIGH |
| 5 | 1-3. BRAIN 섹터 시그널 주입 | MEDIUM | MEDIUM-HIGH | HIGH |
| 6 | 2-1. 펀드 플로우 트래커 | MEDIUM-HIGH | MEDIUM-HIGH | MEDIUM |
| 7 | 2-2. 종목 시그널 섹터 부스트 | MEDIUM | MEDIUM | MEDIUM |
| 8 | 2-3. JARVIS 히트맵 | MEDIUM | LOW-MEDIUM | LOW-MEDIUM |
| 9 | 3-1. 백테스트 프레임워크 | HIGH | MEDIUM | LOW-MEDIUM |
| 10 | 3-2. 파생 선행 연결 | MEDIUM | MEDIUM | MEDIUM |
| 11 | 3-3. 기관 행동 프로파일러 | HIGH | MEDIUM | LOW |

**Phase 0만 적용해도**: 연기금/기관 매집 조기 감지 → 추천 품질 즉시 개선
**Phase 0+1 적용 시**: 섹터 로테이션 자동화 + BRAIN Top-Down 연결 완성

---

## 6. 범위 (Scope)

### In Scope

- 기관 유형별(6개) 수급 데이터 수집 및 세분화
- 연기금 매집 감지기 + 텔레그램 알림
- 섹터별 기관 플로우 집계 시스템
- 섹터 통합 점수 엔진 (Sector Composite Scorer)
- BRAIN 5번째 시그널 연동
- 종목 시그널 섹터 부스트
- BAT-D 파이프라인 통합

### Out of Scope (이번 이터레이션)

- 금융투자협회(KOFIA) 펀드 설정/환매 크롤링 (API 없음, 추후 고려)
- KRX 옵션 실시간 P/C Ratio (실시간 데이터 필요)
- 섹터 ETF 자동매매 실행 (수동 참조까지만)
- 해외 기관 투자자 세분화 (미국/유럽/아시아 등)
- 프론트엔드 섹터 히트맵 (Phase 2-3로 후순위)

---

## 7. 영향 파일 (Affected Files)

### 수정 예정

| 파일 | 변경 내용 |
|------|----------|
| `src/adapters/pykrx_supply_adapter.py` | `fetch_investor_flow()` 기관 유형별 세분화 |
| `src/supply_demand_analyzer.py` | 기관 유형별 가중치 적용 |
| `src/entities/supply_demand_models.py` | `InstitutionalDetailData` 모델 추가 |
| `src/brain.py` | 5번째 시그널 (sector_composite) 입력 추가 |
| `src/etf/sector_engine.py` | 통합 점수 엔진 연결 |
| `scripts/sector_daily_report.py` | 기관 유형별 수급 포함 |
| `config/settings.yaml` | tier2 설정 섹션 추가 |
| BAT-D 스케줄 | Phase 3에 기관 수급 수집 단계 추가 |

### 신규 생성 예정

| 파일 | 역할 |
|------|------|
| `scripts/institutional_flow_collector.py` | 기관 유형별 섹터 수급 수집 |
| `scripts/fund_flow_tracker.py` | 펀드 플로우 트래커 |
| `src/sector_composite.py` | 섹터 통합 점수 엔진 |
| `src/entities/institutional_models.py` | 기관 데이터 모델 |

---

## 8. 제약 조건 (Constraints)

### 기술적 제약

- PyKRX rate limit: 초당 2~3 요청 (23개 섹터 × 상위 종목 → 대량 요청 주의)
- KIS API 일일 호출 제한: 초당 20회
- 기관 유형별 데이터: PyKRX `detail=True`로 가능하나, 과거 데이터 backfill 시간 필요
- 클린 아키텍처 유지: entities → use_cases → adapters (안쪽→바깥 import 금지)

### 비즈니스 제약

- 기존 시그널 파이프라인 성과 악화 금지 (섹터 부스트로 인한 오판 방지)
- 섹터 부스트는 승수(×) 방식 → 기존 점수 체계 유지
- BAT-D 전체 소요시간 60분 이내 유지 (현재 ~50분)

---

## 9. 성공 기준 (Success Criteria)

| # | 기준 | 측정 방법 |
|---|------|----------|
| 1 | 기관 유형별 수급 일 1회 자동 수집 | BAT-D 로그 확인 |
| 2 | 연기금 매집 감지 → 텔레그램 알림 | 연기금 5일+ 순매수 종목 알림 발송 |
| 3 | 섹터 통합 점수 23개 섹터 산출 | sector_composite.json 일일 생성 |
| 4 | BRAIN sector ARM 동적 조정 | brain_decision.json에 sector_signal 반영 |
| 5 | 추천 종목에 섹터 부스트 표시 | tomorrow_picks.json에 sector_boost 필드 |
| 6 | BAT-D 총 소요시간 60분 이내 | 파이프라인 로그 |

---

## 10. 다음 단계 (Next Steps)

1. **이 Plan 문서의 리뷰 및 승인**
2. **Phase 0 즉시 착수**
   - 0-1: pykrx 기관 유형별 수급 세분화
   - 0-2: 연기금 매집 감지기
3. **Phase 1 순차 진행**
   - 1-1 → 1-2 → 1-3 순서
4. **각 Phase 완료 후 Gap Analysis** (`/pdca analyze`)
