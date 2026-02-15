# Phase 3: 5D Geometry Engine — 설계 문서

> 1D~5D 다차원 분석 프레임워크를 통한 "자비스급" 보유종목 실시간 판단 엔진

## 1. 사고 프레임워크

```
1D: 현상을 본다  (뭐가 일어났지?)
2D: 관계를 본다  (뭐랑 뭐가 연결돼있지?)
3D: 구조를 본다  (숨겨진 입체 구조가 뭐지?)
4D: 변화를 본다  (그 구조가 어떻게 움직이지?)
5D: 게임을 본다  (누가 이걸 보고 있고, 그게 뭘 바꾸지?)
```

## 2. 배경 및 의존성

### 완료된 Phase
- **Phase 1**: 장중 데이터 수집 파이프라인 (KIS API → SQLite, 7개 테이블)
- **Phase 2**: 상황보고서 생성기 (SituationReporter → Claude API 입력용 텍스트)
- **v8.1**: Gate + Score Hybrid 진입 시스템 (parquet 기반 일봉 지표 50+개)

### 핵심 의존성
- `src/entities/intraday_models.py` — Phase 1~2 데이터 모델
- `src/use_cases/situation_reporter.py` — Phase 2 보고서 생성기
- `src/use_cases/ports.py` — IntradayStorePort (SQLite 데이터 조회)
- `src/indicators.py` — IndicatorEngine (50+개 기술지표)
- `data/processed/*.parquet` — 일봉 지표 데이터

## 3. 아키텍처: 15개 모듈, 5개 차원

```
┌─────────────────────────────────────────────────┐
│          뉴스 채널 (별도 입력)                    │
│  Grok API / DART → 각 Layer에 이벤트 공급        │
└──────────────────┬──────────────────────────────┘
                   │
═══════════════════╪═══════════════════════════════

1D 현상 (4 data streams) ← Phase 1~2 + v8.1 기존
  ① Price & Technicals  (RSI, ADX, BB, MACD, TRIX, OU)
  ② Volume              (거래량, OBV, 거래대금)
  ③ Investor Flow       (외인/기관 순매수)
  ④ Market & Sector     (KOSPI, 업종, V-KOSPI)

2D 관계 (2 analyzers) ← Phase 3 신규
  ⑤ Lead-Lag Network    (종목간 선행-후행 관계)
  ⑥ Smart Money Detector (기관 매집/분산 패턴)

3D 구조 (3 engines) ← Phase 3-A 신규
  ⑦ Confluence Scorer   (N팩터 조합 적중률 + 조건부 분석)
  ⑧ Cycle Clock         (3주파수 위상 + 위상 정렬도)
  ⑨ Divergence Detector (팩터간 모순/발산 감지)

4D 변화 (2 engines) ← Phase 3-B 신규
  ⑩ Simultaneity Meter  (신호 동시 발생 + 수렴/발산)
  ⑪ Dynamic Equilibrium (OU μ 동적화 + 스프링 에너지)

5D 게임 (3 monitors) ← Phase 3-C 신규
  ⑫ Crowd Meter         (군중 포화도 측정)
  ⑬ Strategy Health     (알파 디케이 + Phase A~D + 가짜패턴)
  ⑭ Scenario Engine     (다중 시나리오 + 켈리 사이징)

═══════════════════╪═══════════════════════════════
                   │
         ⑮ Claude API 종합 판단
           (레이어별 신뢰도 가중)
                   │
            텔레그램 알림
                   │
            학습 루프 → 진화
```

### 신뢰도 원칙
차원이 올라갈수록 신뢰도는 내려간다:
- **L1~L2**: 높은 신뢰도 (숫자/상관관계) → 주요 판단 근거
- **L3~L4**: 중간 신뢰도 (계산/추정) → 보조 판단 근거
- **L5**: 낮은 신뢰도 (추측) → 경고용, 강한 신호일 때만 판단 뒤집기

## 4. 노이즈 제거 근거

| 제거 항목 | 원래 위치 | 이유 | 흡수처 |
|-----------|----------|------|--------|
| 뉴스 | L1 | 팩터가 아닌 데이터 소스 | 별도 입력 채널 |
| DNA매칭 | L1 | 조건부 분석의 하위 기능 | ⑦ Confluence |
| 멀티주파수 | L2 | 나선시계와 100% 중복 | ⑧ Cycle Clock |
| 내접구 균형 | L3 | 가중평균의 기하학적 포장 | v8 스코어링 |
| 4D 단면 | L4 | 조건부 분석의 다른 이름 | ⑦ Confluence |
| 도넛 포지셔닝 | L4 | Cycle Clock + alignment | ⑧ Cycle Clock |
| 가짜패턴 감지 | L5 | 전략수명 Phase D의 증상 | ⑬ Strategy Health |

**22개 → 15개 모듈. 정보 손실 = 없음.**

## 5. 구현 단계

### Phase 3-A: 3D 구조 엔진 (즉시 구현 가능)

#### ⑦ Confluence Scorer (`src/geometry/confluence_scorer.py`)

**역할**: N개 팩터의 조합별 역사적 적중률 계산 + 실시간 매칭

**수학적 근거**: (a+b+c)³ 전개에서 3중 교차항 6abc
- 개별 팩터보다 3팩터 동시 충족의 예측력이 최대 6배

**입력**: parquet 일봉 데이터 (indicators.py 결과)

**핵심 로직**:
```python
class ConfluenceScorer:
    """N-팩터 조합 적중률 분석기"""

    # 7개 기본 팩터 (이진화 조건)
    FACTORS = {
        "rsi_oversold":    lambda r: r["rsi_14"] < 40,
        "adx_trending":    lambda r: r["adx_14"] > 20,
        "bb_lower":        lambda r: r["bb_position"] < 0.25,
        "ou_undervalued":  lambda r: r["ou_z"] < -1.0,
        "volume_surge":    lambda r: r["volume_surge_ratio"] > 1.5,
        "macd_recovering": lambda r: (r["macd_histogram"] < 0) &
                                     (r["macd_histogram"] > r["macd_histogram_prev"]),
        "smart_money_buy": lambda r: r["smart_z"] > 1.0,
    }

    def build_hit_rate_db(self, df, forward_days=5, min_return=0.03):
        """과거 데이터에서 모든 C(7,3)=35개 트리플의 적중률 DB 구축"""
        # 1. 각 팩터를 이진화 (True/False)
        # 2. 모든 3-팩터 조합 생성
        # 3. 조합 충족일의 forward return 통계
        # 4. 적중률 = (forward_return > min_return) / total

    def score_current(self, row) -> dict:
        """현재 행에서 발동 중인 트리플과 적중률 반환"""
        # 1. 현재 충족 팩터 목록
        # 2. 활성 트리플 조합 추출
        # 3. DB에서 적중률 조회
        # 반환: {"active_triples": [...], "best_hit_rate": 0.78, ...}
```

**출력 형식** (Claude 프롬프트용):
```
[3D 교차 분석]
  활성 트리플 2개:
    RSI과매도 × BB하단 × OU저평가 → 적중률 78% (과거 23건)
    거래량급증 × 스마트머니매수 × MACD회복 → 적중률 65% (과거 17건)
  최고 적중률: 78%
```

#### ⑧ Cycle Clock (`src/geometry/cycle_clock.py`)

**역할**: 3개 주파수대의 사이클 위치 추적 + 위상 정렬도

**수학적 근거**: sin(θ), sin(2θ), sin(θ/2) — 같은 시점에서 다른 주파수의 위상

**입력**: parquet 일봉 close 데이터

**핵심 로직**:
```python
from scipy.signal import hilbert

class CycleClock:
    """3주파수 사이클 위치 추적기"""

    BANDS = {
        "long":  (40, 120),   # 장기: 40~120일 주기
        "mid":   (10, 40),    # 중기: 10~40일 주기
        "short": (3, 10),     # 단기: 3~10일 주기
    }

    def extract_phase(self, prices, band):
        """힐베르트 변환으로 즉시 위상 추출"""
        # 1. 밴드패스 필터 적용
        # 2. 힐베르트 변환 → analytic signal
        # 3. np.angle() → 위상 (-π ~ +π)
        # 4. 시계 위치로 변환 (0~12시)

    def get_clock_position(self, prices) -> dict:
        """현재 3주파수 시계 위치 + 정렬도"""
        positions = {}
        for name, band in self.BANDS.items():
            positions[name] = self.extract_phase(prices, band)

        # 위상 정렬도: cos(long_phase - mid_phase)
        # +1 = 같은 방향 (강한 추세), -1 = 반대 방향 (조정/전환)
        alignment = cos(positions["long"]["phase"] - positions["mid"]["phase"])

        return {
            "long_clock": positions["long"]["clock"],   # 8시
            "mid_clock": positions["mid"]["clock"],      # 5시
            "short_clock": positions["short"]["clock"],  # 7시
            "phase_alignment": alignment,  # 0.85 (높은 정렬)
            "interpretation": self._interpret(positions, alignment),
        }
```

**시계 해석 규칙**:
```
5~7시 = 바닥~반등 시작 = 매수 구간
7~9시 = 상승 초중반 = 보유 구간
9~11시 = 상승 후반 = 이익실현 준비
11~1시 = 고점 = 매도 구간
1~3시 = 하락 초기 = 관망
3~5시 = 하락 후반 = 바닥 탐색
```

**출력 형식**:
```
[사이클 시계]
  장기(40~120일): 8시 (상승 초반)
  중기(10~40일):  5시 (바닥 근처)
  단기(3~10일):   7시 (반등 시작)
  위상 정렬: 0.85 (장기-중기 같은 방향 → 강한 추세)
  해석: 장기 상승 + 중기 조정 막바지 → 매수 최적 구간 접근 중
```

#### ⑨ Divergence Detector (`src/geometry/divergence_detector.py`)

**역할**: 서로 다른 팩터 쌍이 모순되는 신호를 낼 때 감지

**수학적 근거**: "회전하면 보이는 구조" — 다른 각도에서 보면 다른 것이 보임

**입력**: L1 데이터 (기술지표 + 거래량 + 수급)

**핵심 로직**:
```python
class DivergenceDetector:
    """팩터간 모순/발산 감지기"""

    # 4개 주요 축의 방향 (-1, 0, +1)
    def classify_direction(self, row) -> dict:
        return {
            "price":  sign(row["ret1"]),                    # 가격 방향
            "volume": sign(row["volume_surge_ratio"] - 1),  # 거래량 방향
            "flow":   sign(row["foreign_net"] + row["inst_net"]),  # 수급 방향
            "momentum": sign(row["macd_histogram"]),        # 모멘텀 방향
        }

    def detect(self, row) -> dict:
        dirs = self.classify_direction(row)

        divergences = []
        # 가격↑ + 거래량↓ = "허약한 상승" (위험)
        # 가격↓ + 수급↑ = "매집 중" (기회)
        # 모멘텀↓ + 수급↑ = "바닥 다지기" (기회)
        # 가격↑ + 모멘텀↓ = "상승 피로" (위험)
        ...

        return {
            "divergences": divergences,
            "risk_count": ...,    # 위험 발산 개수
            "opportunity_count": ...,  # 기회 발산 개수
            "net_signal": ...,    # 종합 (-1 ~ +1)
        }
```

**출력 형식**:
```
[발산 감지]
  ⚠ 가격↑ + 모멘텀↓ = "상승 피로" (주의)
  ✅ 가격↓ + 수급↑ = "매집 중" (기회)
  종합: 기회 1건 > 위험 1건 → 중립~소폭 긍정
```

### Phase 3-B: 4D 변화 엔진 (데이터 축적 후)

#### ⑩ Simultaneity Meter (`src/geometry/simultaneity_meter.py`)

**역할**: 여러 신호가 동시에 발생하는 "정보 폭발" 감지

**수학적 근거**: (a+b+c+d)⁴의 4중 교차항 24abcd — 동시성이 높을수록 배수 효과

**입력**: Phase 1 SQLite tick 데이터 (1분 단위 타임스탬프)

**핵심 로직**:
```python
class SimultaneityMeter:
    """신호 동시 발생 감지기"""

    # 감시 대상 이벤트 유형
    EVENTS = [
        "foreign_surge",      # 외인 대량 체결
        "volume_breakout",    # 거래량 돌파
        "price_breakout",     # 가격 돌파
        "sector_move",        # 섹터 동반 움직임
    ]

    def measure(self, events: list[dict]) -> dict:
        """이벤트 목록에서 동시성 계수 계산"""
        if len(events) < 2:
            return {"simultaneity": 0, "type": "none"}

        # 1. 이벤트 간 시간 간격 계산
        timestamps = sorted([e["timestamp"] for e in events])
        max_gap = (timestamps[-1] - timestamps[0]).total_seconds()

        # 2. 동시성 계수 (0~1)
        # 5분 이내 = 0.9+, 30분 이내 = 0.5+, 2시간+ = 0.1
        sim = max(0, 1.0 - max_gap / 7200)

        # 3. 수렴/발산 분류
        directions = [e["direction"] for e in events]
        convergent = all(d == directions[0] for d in directions)

        return {
            "simultaneity": sim,
            "event_count": len(events),
            "max_gap_seconds": max_gap,
            "type": "convergent" if convergent else "divergent",
            "strength": sim * len(events),  # 24abcd 근사
        }
```

**선행 조건**: Phase 1 tick 데이터가 최소 5거래일 축적

#### ⑪ Dynamic Equilibrium (`src/geometry/dynamic_equilibrium.py`)

**역할**: OU 프로세스의 μ를 고정→동적으로 업그레이드, 스프링 에너지 측정

**수학적 근거**: 4D 초구 = 현재 팩터 + 시장 기대를 모두 반영한 균형점

**입력**: OU 파라미터 (indicators.py) + V-KOSPI / 선물 베이시스 / 신용잔고

**핵심 로직**:
```python
class DynamicEquilibrium:
    """OU μ 동적화 + 스프링 에너지 계산"""

    def calculate(self, static_mu, current_price, volatility,
                  vkospi=None, basis=None, credit_change=None,
                  foreign_flow_accel=None) -> dict:
        # 1. 기대 벡터 계산
        expectation = 0.0
        if vkospi is not None:
            expectation += w1 * self._vkospi_signal(vkospi)
        if basis is not None:
            expectation += w2 * self._basis_signal(basis)
        if credit_change is not None:
            expectation += w3 * self._credit_signal(credit_change)
        if foreign_flow_accel is not None:
            expectation += w4 * foreign_flow_accel

        # 2. 동적 균형가
        dynamic_mu = static_mu * (1 + expectation)

        # 3. 스프링 에너지 = (현재가 - 동적균형)² / (2σ²)
        spring_energy = (current_price - dynamic_mu) ** 2 / (2 * volatility ** 2)

        # 4. 방향: 동적균형 > 현재가 → 스프링이 위로 → 매수 신호
        spring_direction = "up" if dynamic_mu > current_price else "down"

        return {
            "static_mu": static_mu,
            "dynamic_mu": dynamic_mu,
            "expectation_pct": expectation * 100,
            "spring_energy": spring_energy,
            "spring_direction": spring_direction,
            "gap_pct": (dynamic_mu - current_price) / current_price * 100,
        }
```

**선행 조건**: V-KOSPI200, 선물 베이시스 데이터 조회 구현 (KIS API)

### Phase 3-C: 5D 게임 엔진 (시스템 운용 경험 축적 후)

#### ⑫ Crowd Meter (`src/geometry/crowd_meter.py`)

**역할**: 군중 포화도 측정 — 같은 방향에 몰린 참여자 비율 추정

**데이터 소스**:
- 신용잔고 변화율 (KRX 일 1회)
- 대차잔고 / 공매도 비율
- ETF 순자산 변동 (섹터별)
- 개인 순매수 누적 추이

**출력**: crowd_score (-1 ~ +1)
- +1 = 극도의 낙관 (모두가 매수 → 추가 매수세 고갈)
- -1 = 극도의 비관 (모두가 매도 → 역발상 기회)
- 0 = 중립

#### ⑬ Strategy Health (`src/geometry/strategy_health.py`)

**역할**: 전략의 알파 디케이 모니터링 + 수명 주기 추적

**4단계 수명 주기**:
```
Phase A (발견): IR > 1.5, 신호→반응 시차 > 2h, 적중률 안정
Phase B (확산): IR 1.0~1.5, 시차 감소, 신호일 거래량 증가
Phase C (포화): IR 0.5~1.0, 적중률 하락
Phase D (역전): IR < 0.5, 역반응 감지, 가짜패턴 출현
```

**자동 행동 규칙**:
- A→B: 진입 기준 10% 상향
- B→C: 포지션 사이즈 30% 축소 + 대체 전략 탐색
- C→D: 해당 전략 비활성화

**측정 지표** (매주 업데이트):
1. 신호→시장반응 시차 (초)
2. 6개월 롤링 Information Ratio
3. 신호 발생일 거래량 이상치 비율
4. 최근 20건 적중률 vs 전체 평균

#### ⑭ Scenario Engine (`src/geometry/scenario_engine.py`)

**역할**: 다중 시나리오 스트레스 테스트 + 최적 포지션 사이징

**핵심 개념**:
- 균형가가 하나가 아니라 여러 시나리오에 분포
- 시나리오 분산이 클수록 포지션 축소
- 겸손 계수(15%): 항상 "모르는 시나리오" 여유분

**다중 시나리오 켈리 사이징**:
```python
def optimal_size(self, scenarios: list[dict], max_position=0.40, humility=0.15):
    """
    scenarios = [
        {"name": "호재", "probability": 0.30, "return_pct": 5.0},
        {"name": "현상유지", "probability": 0.40, "return_pct": 0.5},
        {"name": "악재", "probability": 0.25, "return_pct": -4.0},
        {"name": "블랙스완", "probability": 0.05, "return_pct": -15.0},
    ]
    """
    # 1. 기대 수익률
    expected = sum(s["probability"] * s["return_pct"] for s in scenarios)

    # 2. 시나리오 분산
    variance = sum(s["probability"] * (s["return_pct"] - expected)**2
                   for s in scenarios)

    # 3. 켈리 비율 (단순화)
    if variance == 0:
        kelly = 0
    else:
        kelly = expected / variance

    # 4. 겸손 계수 적용 + 상한
    size = min(kelly * (1 - humility), max_position)
    return max(size, 0)  # 음수면 0
```

## 6. 파일 구조

```
src/geometry/              ← Phase 3 신규 패키지
  __init__.py
  confluence_scorer.py     ← ⑦ 3D: N팩터 조합 적중률
  cycle_clock.py           ← ⑧ 3D: 3주파수 사이클 위치
  divergence_detector.py   ← ⑨ 3D: 팩터간 발산 감지
  simultaneity_meter.py    ← ⑩ 4D: 신호 동시 발생
  dynamic_equilibrium.py   ← ⑪ 4D: OU μ 동적화
  crowd_meter.py           ← ⑫ 5D: 군중 포화도
  strategy_health.py       ← ⑬ 5D: 전략 수명
  scenario_engine.py       ← ⑭ 5D: 시나리오 사이징
  engine.py                ← ⑮ 통합 GeometryEngine (상황보고서에 주입)

tests/
  test_phase3_confluence.py
  test_phase3_cycle_clock.py
  test_phase3_divergence.py
  test_phase3_geometry_engine.py
```

## 7. Phase 2 통합: 상황보고서 확장

`SituationReporter.generate()` → `GeometryEngine.analyze()` 호출:

```python
# situation_reporter.py 수정
from src.geometry.engine import GeometryEngine

class SituationReporter:
    def __init__(self, config, store_port, ...):
        self.geometry = GeometryEngine(config)

    def generate(self, holdings, report_type="regular"):
        report = ...  # 기존 Phase 2 로직

        # Phase 3: 기하학 분석 추가
        for stock in report.stocks:
            ticker = stock["ticker"]
            geo_result = self.geometry.analyze(ticker, self.store_port)
            stock["geometry"] = geo_result

        # 보고서에 기하학 섹션 추가
        report.geometry_summary = self.geometry.summarize_all(report.stocks)
        return report
```

`SituationReport.to_prompt_text()` 확장:
```
=== 상황보고서 ===

### 핵심 경고
  ...

### 기하학 분석 (3D~5D)          ← 신규 섹션
  [3D 교차 분석]
    활성 트리플: RSI과매도×BB하단×OU저평가 → 적중률 78%
  [사이클 시계]
    장기 8시(상승초반), 중기 5시(바닥근처), 정렬 0.85
  [발산 감지]
    가격↓ + 수급↑ = "매집 중" (기회)

### 시장 환경
  ...

### 종목별 상황
  ...
```

## 8. 구현 우선순위

| 단계 | 모듈 | 선행 조건 | 난이도 |
|------|------|-----------|--------|
| **3-A** (즉시) | ⑦ Confluence Scorer | parquet 데이터 | 중간 |
| **3-A** (즉시) | ⑧ Cycle Clock | parquet close | 중간 |
| **3-A** (즉시) | ⑨ Divergence Detector | indicators | 낮음 |
| **3-A** (즉시) | ⑮ GeometryEngine 통합 | 위 3개 | 낮음 |
| 3-B (2~4주 후) | ⑩ Simultaneity Meter | tick 데이터 축적 | 중간 |
| 3-B (2~4주 후) | ⑪ Dynamic Equilibrium | V-KOSPI API | 중간 |
| 3-C (운용 후) | ⑫ Crowd Meter | 신용잔고 API | 낮음 |
| 3-C (운용 후) | ⑬ Strategy Health | 백테스트 이력 | 높음 |
| 3-C (운용 후) | ⑭ Scenario Engine | Claude 시나리오 | 중간 |

## 9. 설정 (`config/settings.yaml`)

```yaml
# Phase 3: 5D Geometry Engine
geometry:
  enabled: true

  # 3D: Confluence Scorer
  confluence:
    n_factors: 3            # 조합할 팩터 수 (기본 3)
    min_samples: 10         # 최소 과거 사례 수
    min_hit_rate: 0.60      # 보고 기준 최소 적중률
    forward_days: 5         # 적중 판단 기간 (거래일)
    min_return: 0.03        # 적중 기준 수익률 (3%)

  # 3D: Cycle Clock
  cycle:
    long_band: [40, 120]    # 장기 주기 대역 (일)
    mid_band: [10, 40]      # 중기 주기 대역
    short_band: [3, 10]     # 단기 주기 대역
    min_data_days: 120      # 최소 필요 데이터

  # 3D: Divergence Detector
  divergence:
    enabled: true

  # 4D: Simultaneity Meter (Phase 3-B)
  simultaneity:
    enabled: false          # tick 데이터 축적 후 활성화
    window_seconds: 300     # 동시 판정 윈도우 (5분)

  # 4D: Dynamic Equilibrium (Phase 3-B)
  dynamic_eq:
    enabled: false
    weights:
      vkospi: 0.30
      basis: 0.25
      credit: 0.20
      flow_accel: 0.25

  # 5D: Crowd / Strategy / Scenario (Phase 3-C)
  crowd:
    enabled: false
  strategy_health:
    enabled: false
  scenario:
    enabled: false
    humility_factor: 0.15
    max_position: 0.40
```

## 10. 테스트 계획

### Phase 3-A 단위 테스트 (즉시)
- `test_phase3_confluence.py`: 팩터 이진화, 조합 생성, 적중률 계산, 현재 매칭
- `test_phase3_cycle_clock.py`: 위상 추출, 시계 변환, 정렬도, 해석
- `test_phase3_divergence.py`: 방향 분류, 발산 감지, 4가지 패턴
- `test_phase3_geometry_engine.py`: 통합 엔진, 프롬프트 텍스트 생성

### 검증 명령
```bash
source venv/Scripts/activate
python -m pytest tests/test_phase3_confluence.py tests/test_phase3_cycle_clock.py tests/test_phase3_divergence.py tests/test_phase3_geometry_engine.py -v
python -m pytest tests/ -v --tb=short  # 전체 회귀
```

## 11. 향후 발전 방향

### Phase 4: 텔레그램 등급별 알림
- GREEN/YELLOW/RED/BLUE 4단계
- 기하학 분석 결과 포함

### Phase 5: 뉴스/DART 모니터링
- 뉴스 채널을 통한 각 Layer 이벤트 트리거

### Phase 6: 학습 루프
- 매 거래 후 적중률 DB 업데이트
- 가중치 자동 진화
- Strategy Health 본격 가동
