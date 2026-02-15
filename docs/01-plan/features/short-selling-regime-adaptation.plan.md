# Plan: 공매도 재개장 시장 체제 적응 전략

> Feature: short-selling-regime-adaptation
> Created: 2026-02-16
> Status: Draft
> PDCA Phase: Plan

---

## 1. 문제 정의 (Problem Statement)

### 현재 상황

Quantum Master v8.2 시스템은 공매도 금지장(2023.11~2025.03)에서 PF 5.185, 승률 61.5%의 탁월한 성과를 보였으나,
공매도 재개장(2025.04~)에서 PF 0.485, 승률 23.1%로 급격한 성과 저하를 겪고 있다.

| 기간 | 거래 | 승률 | PF | 총 PnL |
|------|------|------|-----|--------|
| 정상장 (~2023.10) | 74건 | 47.3% | 1.462 | +2,607K |
| 금지장 (2023.11~2025.03) | 26건 | 61.5% | 5.185 | +6,871K |
| 재개장 (2025.04~) | 26건 | 23.1% | 0.485 | -1,217K |

### 근본 원인 분석

1. **구조적 원인: 공매도 세력 복귀**
   - "기대가 식은 자리" 패턴 = 과매도 구간에서 반등을 기대하는 전략
   - 공매도 세력이 돌아오면서 과매도 구간에서 추가 하락 압력 발생
   - 금지장에서의 "쉬운 반등"이 사라짐 -> 시스템 전제 붕괴

2. **데이터 결손: 외국인합계 전량 0**
   - 2025+ parquet 데이터에서 외국인합계 컬럼이 전량 0
   - Regime Gate의 Foreign Flow 신호가 항상 중립(0.50) 반환
   - 3개 base 신호 중 1개가 사실상 비활성화됨

3. **Regime Gate v3 구조적 한계**
   - base = breadth(0~1) + foreign_norm(0.50 고정) + volatility(0~1) = 최대 2.50
   - SA가 0.06이어도 sa_mult = 0.55 + 0.45 * 0.06 = 0.577
   - composite = 2.00 * 0.577 = 1.154 -> caution (hostile인 0.8 도달 불가가 아닌, hostile 근접하지만 도달 어려움)
   - 재개장 환경에서 hostile을 판정하지 못해 손실 거래를 차단하지 못함

4. **SA Death Spiral 위험**
   - sa_floor을 낮추면 손실 -> SA 하락 -> hostile -> 거래 없음 -> SA 회복 불가
   - 현재 sa_floor=0.55로 보수적 설정하여 death spiral 방지하지만, hostile 진입 불가 부작용

---

## 2. 목표 (Objectives)

### 정량 목표

| 지표 | 현재 (재개장) | 목표 | 비고 |
|------|-------------|------|------|
| PF (Profit Factor) | 0.485 | >= 0.80 | 손익분기 접근 |
| 승률 | 23.1% | >= 35% | 최소 3건 중 1건 |
| 총 PnL | -1,217K | >= -200K | 손실 최소화 |
| 최대 연속 손실 | 미측정 | <= 5건 | 리스크 관리 |

### 정성 목표

1. 시스템이 공매도 재개 환경을 자동으로 감지하는 능력 확보
2. 시장 체제별 적응형 파라미터 조정 메커니즘 구축
3. 외국인 데이터 결손 문제의 근본 해결 또는 효과적인 우회
4. 사망 나선(death spiral)을 방지하면서도 hostile 체제 도달 가능한 구조 확보

---

## 3. 전략적 연구 결과

### 3.1 공매도 체제 감지 (Short-Selling Regime Indicator)

#### 3.1.1 감지 가능한 시장 신호

공매도 재개/금지를 감지하는 방법은 크게 3가지이다.

**방법 A: 직접 공매도 데이터 감시 (가장 정확)**

이미 `src/entities/supply_demand_models.py`에 `ShortSellingData` 모델이 존재하고,
`src/adapters/pykrx_supply_adapter.py`에 `fetch_short_selling()` 메서드가 구현되어 있다.

- `short_ratio` (공매도 비중): 금지장에서는 0%, 재개 후 급증
- `short_balance` (공매도 잔고): 재개 시 잔고 급증
- `lending_balance` (대차잔고): 대차잔고 증가 = 공매도 준비

감지 로직:
```
if 유니버스 평균 short_ratio > 0 for 연속 5일:
    regime = "short_selling_active"
elif 유니버스 평균 short_ratio == 0 for 연속 20일:
    regime = "short_selling_banned"
```

**방법 B: 간접 시장 미시구조 변화 감지 (데이터 없을 때 대안)**

공매도가 재개되면 시장 미시구조가 변화한다:
- 하락일의 거래량 패턴 변화 (매도압력 증가)
- Bid-Ask 스프레드 확대 경향
- 하방 변동성(downside volatility) 증가
- 과매도 구간에서의 반등 강도 약화

이를 감지하는 composite 지표:
```
short_proxy = (
    0.4 * downside_vol_ratio    # 하방 변동성 / 전체 변동성
    + 0.3 * bounce_failure_rate # 과매도 후 반등 실패 비율
    + 0.3 * sell_volume_ratio   # 하락일 거래량 / 상승일 거래량
)
```

**방법 C: 정책 캘린더 하드코딩 (확실하지만 수동)**

알려진 공매도 금지/재개 일자를 config에 등록:
```yaml
short_selling_calendar:
  - { start: "2023-11-06", end: "2025-03-30", status: "banned" }
  - { start: "2025-03-31", end: null, status: "active" }
```

#### 3.1.2 권장 접근법

**방법 A + C 조합**을 권장한다.

- 단기: 방법 C (캘린더)로 즉시 적용하여 재개장 감지
- 중기: 방법 A로 실시간 데이터 기반 자동 감지 전환
- 방법 B는 parquet 데이터로도 계산 가능하므로 백테스트 검증용으로 활용

### 3.2 Regime Gate 구조 개선

#### 3.2.1 현재 구조의 한계 진단

현재 `RegimeGate`(d:\sub-agent-project\src\regime_gate.py)의 공식:

```
base = breadth(0~1) + foreign_norm(0~1) + volatility(0~1)    -> 0~3
sa_mult = sa_floor(0.55) + (1 - 0.55) * sa_raw(0~1)          -> 0.55~1.0
composite = base * sa_mult                                     -> 0~3
```

문제점:
1. **Foreign 신호 고착**: 외국인 데이터 전량 0 -> foreign_norm = 0.50 고정
2. **3개 additive 신호의 상한**: base 최대 = 3.0이지만, 1개 고정이면 실질 최대 2.50
3. **SA multiplier의 floor 딜레마**: floor 높으면 hostile 불가, 낮으면 death spiral
4. **공매도 환경 반영 없음**: 공매도 세력의 존재를 감지하는 신호 부재

#### 3.2.2 개선안: Regime Gate v4 제안

**핵심 변경: 4번째 base 신호 추가 + 비대칭 SA 구조**

```
base_v4 = breadth(0~1) + foreign_norm(0~1) + volatility(0~1) + short_regime(0~1)
                                                                 ^^^^^^^^^^^^^^^^
                                                                 [신규 4번째 신호]

sa_mult_v4 = asymmetric_floor(regime_context) + (1 - floor) * sa_raw

composite_v4 = base_v4 * sa_mult_v4    -> 0~4 범위 (임계값도 재조정)
```

**신규 신호 1: Short-Selling Regime Signal (0~1)**

공매도 환경의 위험도를 정량화:
```python
def _calc_short_regime(self, data_dict, idx):
    """공매도 환경 위험도. 0=위험(공매도 활발), 1=안전(금지/미미)"""
    # 방법 1: 직접 데이터
    if short_data_available:
        avg_short_ratio = universe_avg_short_ratio
        if avg_short_ratio > 5.0:   return 0.0   # 공매도 매우 활발
        elif avg_short_ratio > 2.0: return 0.3   # 활발
        elif avg_short_ratio > 0.5: return 0.6   # 보통
        else:                       return 1.0   # 미미/금지

    # 방법 2: 캘린더
    if date in banned_period:
        return 1.0  # 금지장 -> 안전
    else:
        return 0.3  # 재개장 -> 기본 경계 (데이터 수집 전까지)

    # 방법 3: proxy
    return short_proxy_score
```

**비대칭 SA Floor 구조**

공매도 환경에 따라 SA floor을 동적으로 조정:
```python
if short_regime < 0.3:  # 공매도 활발 환경
    sa_floor = 0.30     # 낮은 floor -> hostile 도달 가능
else:                   # 금지장 또는 미미
    sa_floor = 0.55     # 기존 높은 floor 유지
```

이렇게 하면:
- 공매도 활발 환경: base=2.0 * sa_mult(0.30)=0.60 -> hostile(0.8 미만) 도달 가능
- 금지장: 기존과 동일하게 death spiral 방지

**신규 신호 2 (선택): Bounce Quality Signal (0~1)**

과매도 후 반등 품질을 측정. 우리 시스템의 핵심 전제가 유효한지 직접 측정:
```python
def _calc_bounce_quality(self, data_dict, idx):
    """최근 N일간 과매도 반등 성공률. 높으면 우리 전략에 유리."""
    # RSI<30 이후 5일 내 +3% 반등한 종목 비율
    bounce_success_rate = count_successful_bounces / count_oversold_events
    return min(bounce_success_rate / 0.60, 1.0)  # 60% 이상이면 1.0
```

#### 3.2.3 임계값 재조정

base가 0~4 범위로 확장되므로 임계값도 조정:

| 체제 | v3 (0~3) | v4 (0~4) | 비고 |
|------|----------|----------|------|
| favorable | >= 2.2 | >= 2.8 | 4/3 스케일 |
| neutral | >= 1.5 | >= 1.9 | |
| caution | >= 0.8 | >= 1.0 | |
| hostile | < 0.8 | < 1.0 | |

### 3.3 재개장 적응 전략

#### 3.3.1 Adaptive Threshold 전략

시장 체제에 따라 entry trigger의 임계값을 동적으로 조정:

```python
class AdaptiveThresholdManager:
    """공매도 환경에 따른 진입 임계값 동적 조정"""

    def adjust_thresholds(self, short_regime_score: float) -> dict:
        if short_regime_score < 0.3:  # 공매도 활발
            return {
                "min_rr_ratio": 2.5,        # 2.0 -> 2.5 (높은 손익비 요구)
                "min_zone_score_A": 0.90,    # 0.85 -> 0.90 (엄격한 A등급)
                "min_zone_score_B": 0.78,    # 0.70 -> 0.78
                "impulse_min_conditions": 3, # 2 -> 3 (3개 전부 충족)
                "confirm_rsi_above": 55,     # 50 -> 55
                "max_positions": 3,          # 5 -> 3 (포지션 수 축소)
                "stop_loss_tighter": 0.8,    # 손절 20% 타이트
            }
        elif short_regime_score < 0.6:  # 보통
            return {
                "min_rr_ratio": 2.2,
                "min_zone_score_A": 0.87,
                "min_zone_score_B": 0.73,
                "impulse_min_conditions": 2,
                "confirm_rsi_above": 52,
                "max_positions": 4,
                "stop_loss_tighter": 0.9,
            }
        else:  # 안전 (금지장)
            return {}  # 기본값 유지
```

#### 3.3.2 체제별 전략 모듈 분기

재개장에서는 "기대가 식은 자리" 패턴의 성공률이 낮으므로,
**전략 강조점을 이동**해야 한다:

| 전략 요소 | 금지장 (안전) | 재개장 (위험) |
|-----------|-------------|-------------|
| 핵심 전략 | 과매도 반등 | 추세 추종 (Trend Continuation) |
| Trigger 우선순위 | Impulse > Confirm | Confirm > Trend Cont |
| 공매도 체크 | 불필요 | 필수 (L1 공매도 게이트) |
| 포지션 크기 | 정상 | 축소 (60~80%) |
| 손절 폭 | 정상 | 타이트 (80%) |
| 최대 보유일 | 20일 | 10~15일 |
| 최소 손익비 | 1.5~2.0 | 2.0~2.5 |

**구현 방안**: `config/settings.yaml`에 `regime_profiles` 섹션 추가

```yaml
regime_profiles:
  short_selling_active:
    max_positions: 3
    min_rr_impulse: 2.5
    min_rr_confirm: 2.5
    stop_loss_scale: 0.8       # 20% 타이트
    max_hold_days: 12
    disable_impulse: false
    require_short_gate: true   # 공매도 L1 게이트 필수
    position_scale: 0.7        # 포지션 70%

  short_selling_banned:
    max_positions: 5
    min_rr_impulse: 1.5
    min_rr_confirm: 2.0
    stop_loss_scale: 1.0
    max_hold_days: 20
    disable_impulse: false
    require_short_gate: false
    position_scale: 1.0
```

### 3.4 외국인 데이터 문제 해결

#### 3.4.1 데이터 수집 방법

**방법 1: PyKRX (가장 현실적)**

이미 `pykrx_supply_adapter.py`에 투자자별 매매동향 수집 코드가 존재한다.
`InvestorFlowData` 모델에 `foreign_net` 필드가 있다.

```python
# pykrx_supply_adapter.py의 fetch_investor_flow() 활용
from pykrx import stock
df = stock.get_market_trading_value_by_date(start, end, ticker)
# 컬럼: 기관합계, 기타법인, 개인, 외국인합계, 전체
```

이 데이터를 parquet에 backfill하면 외국인합계 문제 해결 가능.

**방법 2: KIS API (한국투자증권)**

실시간 + 일별 투자자별 수급 데이터 제공. 이미 프로젝트에 KIS API 연동이 있다.

**방법 3: KRX 정보데이터시스템 직접 크롤링**

data.krx.co.kr 에서 CSV 다운로드 자동화.

#### 3.4.2 권장 접근법

**단기 (즉시 적용)**: PyKRX로 2025년+ 외국인 데이터 backfill
- `pykrx_supply_adapter.py`의 `fetch_investor_flow()`를 확장
- 기존 parquet 파일에 외국인합계 컬럼 업데이트
- 예상 소요: 1~2시간 (스크립트 작성 + 실행)

**중기**: 일일 자동 업데이트 파이프라인 구축
- `scripts/daily_scheduler.py`에 통합
- 장마감 후 자동으로 당일 수급 데이터 수집 -> parquet 갱신

#### 3.4.3 외국인 데이터 없이 proxy 추정 (백테스트 보완용)

외국인 데이터를 직접 수집하기 전 백테스트에서 사용할 proxy:

```python
def estimate_foreign_proxy(df, idx, lookback=20):
    """외국인 수급 proxy 추정
    - 대형주 동반 상승/하락 패턴
    - 환율(USD/KRW) 역상관
    - KOSPI200 추종 강도
    """
    # 간단 버전: breadth의 대형주 가중 버전
    # 대형주(시총 상위 30%)의 추세 정렬 비율이 높으면 외국인 매수 추정
    large_cap_aligned = count_large_cap_aligned / total_large_cap
    return (large_cap_aligned - 0.5) * 2  # -1 ~ 1
```

### 3.5 실행 가능한 우선순위 로드맵

#### Phase 0: 즉시 적용 (난이도: LOW, 기대효과: HIGH)

**0-1. 공매도 캘린더 하드코딩 + Regime Profile 적용**
- `config/settings.yaml`에 `short_selling_calendar` + `regime_profiles` 추가
- `backtest_engine.py`에서 날짜 기반 profile 전환 로직 추가
- 예상 효과: 재개장에서 포지션 축소 + 엄격한 진입 -> 손실 거래 감소
- 소요: 2~3시간

**0-2. SA Floor 비대칭 적용**
- `regime_gate.py`에서 공매도 환경에 따른 sa_floor 동적 조정
- 재개장: sa_floor=0.30 (hostile 도달 가능)
- 금지장: sa_floor=0.55 (기존 유지)
- 소요: 1시간

#### Phase 1: 단기 개선 (난이도: MEDIUM, 기대효과: HIGH)

**1-1. 외국인 데이터 backfill**
- PyKRX로 2025+ 외국인합계 데이터 수집 -> parquet 업데이트
- Regime Gate Foreign Flow 신호 정상화
- 소요: 3~4시간

**1-2. Short-Selling Regime Signal (4번째 base 신호) 추가**
- `regime_gate.py`에 `_calc_short_regime()` 메서드 추가
- base = breadth + foreign + volatility + short_regime (0~4)
- 임계값 재조정
- 소요: 4~5시간

#### Phase 2: 중기 개선 (난이도: MEDIUM-HIGH, 기대효과: MEDIUM)

**2-1. Adaptive Threshold Manager 구현**
- 공매도 환경에 따른 진입 임계값 동적 조정
- trigger, zone score, position sizing 모두 연동
- 소요: 6~8시간

**2-2. Bounce Quality Signal (5번째 base 신호) 추가**
- 과매도 반등 품질 측정 -> 전략 전제 유효성 직접 평가
- 소요: 4~5시간

#### Phase 3: 장기 개선 (난이도: HIGH, 기대효과: MEDIUM)

**3-1. 공매도 데이터 실시간 연동**
- PyKRX 공매도 잔고/거래 데이터를 parquet에 통합
- `supply_demand_analyzer.py`의 L1 레이어 활성화
- 소요: 8~10시간

**3-2. 체제별 독립 전략 모듈**
- 공매도 활발 환경 전용 전략 (추세 추종 강화, 반등 전략 억제)
- 기존 파이프라인과 분기 구조
- 소요: 12~15시간

### ROI 순위 정리

| 순위 | 항목 | 난이도 | 기대 PF 개선 | ROI |
|------|------|--------|-------------|-----|
| 1 | 0-1. 캘린더 + Profile | LOW | +0.10~0.15 | VERY HIGH |
| 2 | 0-2. SA Floor 비대칭 | LOW | +0.05~0.10 | VERY HIGH |
| 3 | 1-1. 외국인 backfill | MEDIUM | +0.05~0.15 | HIGH |
| 4 | 1-2. Short Regime 신호 | MEDIUM | +0.10~0.20 | HIGH |
| 5 | 2-1. Adaptive Threshold | MED-HIGH | +0.05~0.10 | MEDIUM |
| 6 | 2-2. Bounce Quality | MEDIUM | +0.03~0.08 | MEDIUM |
| 7 | 3-1. 공매도 실시간 | HIGH | +0.05~0.10 | LOW-MED |
| 8 | 3-2. 독립 전략 모듈 | HIGH | +0.10~0.20 | LOW |

**Phase 0만 적용해도 PF 0.485 -> 0.60~0.70 개선 예상**
**Phase 0+1 적용 시 PF 0.70~0.85 도달 가능성 높음**

---

## 4. 범위 (Scope)

### In Scope

- Regime Gate v4 설계 및 구현 (4번째 base 신호 추가)
- SA Floor 비대칭 구조 적용
- 공매도 캘린더 기반 체제 감지
- 외국인 데이터 backfill (PyKRX)
- Regime Profile 기반 적응형 파라미터 조정
- 전체 기간(2019.07~2026.01) 백테스트 재검증

### Out of Scope (이번 이터레이션)

- 완전히 새로운 독립 전략 모듈 개발
- KIS API 실시간 공매도 연동
- 옵션/선물 이상 신호 (L6)
- 프론트엔드 대시보드

---

## 5. 제약 조건 (Constraints)

### 기술적 제약

- Python 3.13, 클린 아키텍처 유지 (entities -> use_cases -> adapters)
- 기존 v8 파이프라인과의 하위호환 필수
- 데이터: data/processed/*.parquet (100+ 종목)
- PyKRX rate limit: 초당 약 2~3 요청

### 비즈니스 제약

- 백테스트 전체 기간 성과가 기존보다 악화되면 안 됨 (정상장+금지장 PF 유지)
- 재개장 PF 목표: >= 0.80 (또는 해당 기간 효과적 차단)
- 거래 빈도가 지나치게 줄어들면 안 됨 (0건이 되면 시스템 무의미)

---

## 6. 성공 기준 (Success Criteria)

| # | 기준 | 측정 방법 |
|---|------|----------|
| 1 | 재개장 PF >= 0.80 | 백테스트 결과 |
| 2 | 전체 기간 PF >= 1.30 | 백테스트 결과 (기존 ~1.46 대비 소폭 하락 허용) |
| 3 | 재개장 승률 >= 35% | 백테스트 결과 |
| 4 | 공매도 환경 자동 감지 | Regime Gate 로그에서 체제 전환 확인 |
| 5 | SA death spiral 미발생 | SA가 0으로 고착되지 않음 확인 |
| 6 | 기존 코드 하위호환 | v8_hybrid.enabled=false 시 기존과 동일 동작 |

---

## 7. 영향 파일 (Affected Files)

### 수정 예정

| 파일 | 변경 내용 |
|------|----------|
| `d:\sub-agent-project\config\settings.yaml` | regime_profiles, short_selling_calendar 추가 |
| `d:\sub-agent-project\src\regime_gate.py` | v4 확장 (4번째 신호, 비대칭 SA) |
| `d:\sub-agent-project\src\backtest_engine.py` | regime profile 전환 로직 |

### 신규 생성 예정

| 파일 | 역할 |
|------|------|
| `src/use_cases/regime_profile_manager.py` | 체제별 파라미터 프로파일 관리 |
| `scripts/backfill_foreign_data.py` | 외국인 데이터 backfill 스크립트 |

### 관련 참조 (수정 불필요)

| 파일 | 역할 |
|------|------|
| `d:\sub-agent-project\src\supply_demand_analyzer.py` | 수급 분석 (L1 공매도 활용 가능) |
| `d:\sub-agent-project\src\adapters\pykrx_supply_adapter.py` | PyKRX 데이터 수집 |
| `d:\sub-agent-project\src\entities\supply_demand_models.py` | 공매도/수급 데이터 모델 |
| `d:\sub-agent-project\src\signal_engine.py` | 시그널 파이프라인 |
| `d:\sub-agent-project\src\v8_pipeline.py` | v8 파이프라인 |

---

## 8. 리스크 (Risks)

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|----------|
| 재개장 개선이 금지장 성과를 악화시킴 | 중 | 높음 | 체제별 분기 처리로 금지장 로직 보존 |
| PyKRX 외국인 데이터 부정확 | 낮 | 중 | KIS API로 교차 검증 |
| 과최적화 (재개장에 맞추면 미래 성과 저하) | 중 | 높음 | Walk-Forward 검증 + Bootstrap |
| SA death spiral 재발 | 낮 | 높음 | 비대칭 floor + 최소 거래 보장 |
| 공매도 재금지 시 시스템 재조정 필요 | 낮 | 중 | 캘린더 + 자동 감지 이중 구조 |

---

## 9. 다음 단계 (Next Steps)

1. **이 Plan 문서의 리뷰 및 승인**
2. **Design 단계 진행** (`/pdca design short-selling-regime-adaptation`)
   - Phase 0 + Phase 1 상세 설계
   - 클래스 다이어그램, 시퀀스 다이어그램
   - 테스트 계획
3. **구현** (Do 단계)
   - Phase 0 즉시 적용 -> 백테스트 -> 효과 확인
   - Phase 1 구현 -> 누적 효과 확인
4. **검증** (Check 단계)
   - 전 기간 백테스트 비교
   - Walk-Forward + Bootstrap 검증
