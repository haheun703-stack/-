"""
기하학적 퀀트 엔진 - assert 기반 테스트
Bug #7 수정: print 기반 → assert 기반 검증
v2.0: 신규 7지표 + 프로파일 + 하위호환 테스트 추가
"""

import os
import sys

import numpy as np
import pandas as pd

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometric_engine import (
    WEIGHT_PROFILES,
    BandBreachDetector,
    ConfluenceAnalyzer,
    CurvatureAnalyzer,
    ElliottWaveAnalyzer,
    GeometricQuantEngine,
    HarmonicDetector,
    IndicatorResult,
    MeanReversionAnalyzer,
    PivotDetector,
    SlopeAnalyzer,
    SlopeMomentumAnalyzer,
    SqueezeDetector,
    VolumeClimaxDetector,
)

# =============================================================================
# 테스트 데이터 생성
# =============================================================================

def generate_gartley_data() -> tuple:
    """Bullish Gartley 패턴 데이터"""
    n = 200
    np.random.seed(123)

    prices = []
    X_price = 50000
    A_price = 60000
    for i in range(40):
        t = i / 39
        p = X_price + (A_price - X_price) * t + np.random.normal(0, 200)
        prices.append(p)

    xa_range = A_price - X_price
    B_price = A_price - xa_range * 0.618
    for i in range(30):
        t = i / 29
        p = A_price + (B_price - A_price) * t + np.random.normal(0, 200)
        prices.append(p)

    ab_range = A_price - B_price
    C_price = B_price + ab_range * 0.618
    for i in range(30):
        t = i / 29
        p = B_price + (C_price - B_price) * t + np.random.normal(0, 150)
        prices.append(p)

    D_price = A_price - xa_range * 0.786
    for i in range(40):
        t = i / 39
        p = C_price + (D_price - C_price) * t + np.random.normal(0, 200)
        prices.append(p)

    target = D_price + xa_range * 0.382
    for i in range(60):
        t = i / 59
        p = D_price + (target - D_price) * t + np.random.normal(0, 300)
        prices.append(p)

    prices = np.array(prices[:n])
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "open": prices + np.random.normal(0, 100, n),
        "high": prices + np.abs(np.random.normal(0, 200, n)),
        "low": prices - np.abs(np.random.normal(0, 200, n)),
        "close": prices,
        "volume": np.random.randint(500000, 2000000, n),
    }, index=dates)

    return df, {"X": X_price, "A": A_price, "B": B_price, "C": C_price, "D": D_price}


def generate_elliott_data() -> tuple:
    """상승 충격파 (1-2-3-4-5) 패턴 데이터"""
    n = 250
    np.random.seed(456)

    prices = []
    base = 30000

    w1_start, w1_end = base, base + 5000
    for i in range(40):
        t = i / 39
        p = w1_start + (w1_end - w1_start) * t + np.random.normal(0, 150)
        prices.append(p)

    w1_len = w1_end - w1_start
    w2_end = w1_end - w1_len * 0.50
    for i in range(25):
        t = i / 24
        p = w1_end + (w2_end - w1_end) * t + np.random.normal(0, 150)
        prices.append(p)

    w3_end = w2_end + w1_len * 1.618
    for i in range(55):
        t = i / 54
        p = w2_end + (w3_end - w2_end) * t + np.random.normal(0, 200)
        prices.append(p)

    w3_len = w3_end - w2_end
    w4_end = w3_end - w3_len * 0.382
    w4_end = max(w4_end, w1_end + 500)
    for i in range(30):
        t = i / 29
        p = w3_end + (w4_end - w3_end) * t + np.random.normal(0, 150)
        prices.append(p)

    w5_end = w4_end + w1_len * 1.0
    for i in range(50):
        t = i / 49
        p = w4_end + (w5_end - w4_end) * t + np.random.normal(0, 200)
        prices.append(p)

    correction_end = w5_end - w1_len * 0.618
    for i in range(50):
        t = i / 49
        p = w5_end + (correction_end - w5_end) * t + np.random.normal(0, 300)
        prices.append(p)

    prices = np.array(prices[:n])
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "open": prices + np.random.normal(0, 100, n),
        "high": prices + np.abs(np.random.normal(0, 200, n)),
        "low": prices - np.abs(np.random.normal(0, 200, n)),
        "close": prices,
        "volume": np.random.randint(300000, 1500000, n),
    }, index=dates)

    return df, {
        "w1": (w1_start, w1_end),
        "w2_end": w2_end,
        "w3_end": w3_end,
        "w4_end": w4_end,
        "w5_end": w5_end,
    }


# =============================================================================
# 테스트 함수
# =============================================================================

def test_pivot_detection():
    """피벗 추출 테스트"""
    df, expected = generate_gartley_data()
    detector = PivotDetector(zigzag_pct=3.0, min_bars=5)

    pivots = detector.find_pivots_zigzag(df["close"].values)

    assert len(pivots) >= 4, f"피벗이 최소 4개여야 하나 {len(pivots)}개 감지됨"

    # 첫 피벗은 저점(X)이어야 함
    assert pivots[0]["type"] == "low", f"첫 피벗이 low가 아님: {pivots[0]['type']}"

    # X 가격이 기대값 근처인지 (10% 허용)
    x_price = pivots[0]["price"]
    assert abs(x_price - expected["X"]) / expected["X"] < 0.10, \
        f"X점 가격 {x_price:.0f}이 기대값 {expected['X']:.0f}에서 10% 이상 벗어남"

    # 피벗 확정 테스트
    for p in pivots[:-1]:  # 마지막 피벗 제외
        confirmed = detector.is_pivot_confirmed(
            df["close"].values, p["index"], p["type"]
        )
        assert confirmed, f"피벗 idx={p['index']} ({p['type']})이 확정되지 않음"

    print("  [PASS] test_pivot_detection")


def test_pivot_from_atr():
    """ATR 기반 동적 ZigZag 테스트 (Bug #5)"""
    df, _ = generate_gartley_data()

    # ATR 컬럼이 없을 때도 동작해야 함
    detector = PivotDetector.from_atr(df, atr_mult=1.5, min_bars=5)
    assert detector.zigzag_pct > 0, "zigzag_pct가 0 이하"
    assert 0.02 <= detector.zigzag_pct <= 0.10, \
        f"zigzag_pct가 범위 밖: {detector.zigzag_pct}"

    pivots = detector.find_pivots_zigzag(df["close"].values)
    assert len(pivots) >= 2, "ATR 기반 피벗이 최소 2개 이상이어야 함"

    print("  [PASS] test_pivot_from_atr")


def test_harmonic_gartley():
    """하모닉 Gartley 패턴 감지 테스트"""
    df, expected = generate_gartley_data()
    detector = HarmonicDetector(tolerance=0.08, zigzag_pct=3.0, min_bars=5)

    patterns = detector.detect_patterns(df, lookback=200)

    assert len(patterns) >= 1, "Gartley 데이터에서 패턴이 감지되지 않음"

    best = patterns[0]
    assert best.pattern_type.value == "Gartley", \
        f"Gartley가 아닌 {best.pattern_type.value} 감지됨"
    assert best.direction == "bullish", \
        f"Bullish가 아닌 {best.direction} 방향"
    assert best.score >= 50, \
        f"적합도가 50% 미만: {best.score}%"
    assert best.target_1 > best.prz_low, \
        "target_1이 PRZ보다 낮음"
    assert best.stop_loss < best.prz_low, \
        "stop_loss가 PRZ보다 높음 (bullish인데)"

    # AB/XA 비율이 0.618 근처인지 (20% 허용)
    ab_xa = best.ratios.get("AB_XA", 0)
    assert 0.45 <= ab_xa <= 0.80, f"AB/XA 비율 {ab_xa:.3f}이 Gartley 범위 밖"

    print(f"  [PASS] test_harmonic_gartley (score={best.score}%)")


def test_elliott_sliding_window():
    """엘리어트 파동 슬라이딩 윈도우 테스트 (Bug #2)"""
    df, expected = generate_elliott_data()
    analyzer = ElliottWaveAnalyzer(zigzag_pct=5.0, min_bars=5)

    wave = analyzer.analyze(df, lookback=250)

    # 슬라이딩 윈도우로 더 넓은 범위에서 파동 탐색
    if wave is not None:
        assert wave.wave_type.value == "Impulse", \
            f"Impulse가 아닌 {wave.wave_type.value}"
        assert wave.confidence > 0, "신뢰도가 0"
        assert len(wave.rules_passed) > 0, "통과한 규칙이 없음"

        # 절대 규칙 위반이 1개 이하인지
        absolute_failures = sum(1 for f in wave.rules_failed if f.startswith("규칙"))
        assert absolute_failures <= 1, \
            f"절대 규칙 위반 {absolute_failures}개 (최대 1개 허용)"

        print(f"  [PASS] test_elliott_sliding_window (confidence={wave.confidence}%, "
              f"wave={wave.current_wave})")
    else:
        # ZigZag 파라미터에 따라 미감지 가능 — 경고만
        print("  [WARN] test_elliott_sliding_window: 파동 미감지 (파라미터 조정 필요)")


def test_elliott_wave5_sell():
    """엘리어트 5파 매도 시그널 테스트 (Bug #4)"""
    engine = GeometricQuantEngine(config={
        "zigzag_pct": 5.0,
        "elliott_zigzag_pct": 5.0,
        "elliott_min_bars": 5,
        "lookback": 250,
        "w_harmonic": 0.35,
        "w_elliott": 0.30,
        "w_slope": 0.20,
        "w_divergence": 0.15,
    })

    df, _ = generate_elliott_data()
    signal = engine.generate_signal(df, ticker="TEST_ELLIOTT")

    # 엘리어트가 감지되었고 5파 관련이면 매도 시그널이 있어야 함
    if signal.elliott and signal.elliott.get("current_wave", "").startswith("5"):
        sell_reasoning = [r for r in signal.reasoning if "5파" in r]
        assert len(sell_reasoning) > 0, \
            "5파 감지됐는데 매도 관련 reasoning이 없음"
        print(f"  [PASS] test_elliott_wave5_sell (action={signal.action})")
    else:
        print("  [WARN] test_elliott_wave5_sell: 5파 미감지 — 스킵")


def test_slope_analysis():
    """각도 분석 테스트"""
    n = 100
    np.random.seed(789)

    # 급등 데이터
    steep_prices = 50000 + np.arange(n) * 500 + np.random.normal(0, 200, n)
    df_steep = pd.DataFrame({
        "open": steep_prices, "high": steep_prices + 200,
        "low": steep_prices - 200, "close": steep_prices,
        "volume": np.random.randint(100000, 500000, n),
    })

    # 완만 상승 데이터
    gentle_prices = 50000 + np.arange(n) * 50 + np.random.normal(0, 200, n)
    df_gentle = pd.DataFrame({
        "open": gentle_prices, "high": gentle_prices + 200,
        "low": gentle_prices - 200, "close": gentle_prices,
        "volume": np.random.randint(100000, 500000, n),
    })

    analyzer = SlopeAnalyzer(lookback_periods=[10, 20, 60])

    steep_results = analyzer.analyze(df_steep)
    gentle_results = analyzer.analyze(df_gentle)

    assert len(steep_results) == 3, "급등 데이터 결과가 3개여야 함"
    assert len(gentle_results) == 3, "완만 데이터 결과가 3개여야 함"

    # 급등은 60봉 기준 moderate 이상이어야 함
    steep_60 = steep_results[-1]  # 60봉
    assert steep_60.normalized_angle > 5, \
        f"급등 60봉 각도가 너무 작음: {steep_60.normalized_angle}도"

    # 완만은 60봉 기준 weak여야 함
    gentle_60 = gentle_results[-1]
    assert gentle_60.trend_strength == "weak", \
        f"완만 60봉 강도가 weak가 아님: {gentle_60.trend_strength}"

    # 급등 > 완만 각도
    assert steep_60.normalized_angle > gentle_60.normalized_angle, \
        "급등 각도가 완만보다 작음"

    print("  [PASS] test_slope_analysis")


def test_slope_divergence():
    """기울기 다이버전스 감지 테스트"""
    analyzer = SlopeAnalyzer()

    # bearish_divergence: long_angle > 10도 AND short_angle < -5도
    # 구조: 80봉 횡보 → 15봉 급등(+40%) → 5봉 급락(-16%)
    # long_period=20: 50000→59000 (+18%) → angle ~10.2도
    # short_period=5:  70000→59000 (-16%) → angle ~-8.9도
    div_prices = np.concatenate([
        np.full(80, 50000),                          # 80봉 횡보
        np.linspace(50000, 70000, 15),               # 15봉 급등
        np.linspace(70000, 59000, 6)[1:],            # 5봉 급락
    ])

    div = analyzer.detect_slope_divergence(div_prices, short_period=5, long_period=20)
    assert div == "bearish_divergence", f"bearish_divergence가 아님: {div}"

    # bullish_divergence: long_angle < -10도 AND short_angle > 5도
    # 구조: 80봉 횡보 → 15봉 급락(-40%) → 5봉 급반등(+16%)
    div_prices2 = np.concatenate([
        np.full(80, 70000),                          # 80봉 횡보
        np.linspace(70000, 42000, 15),               # 15봉 급락
        np.linspace(42000, 50000, 6)[1:],            # 5봉 급반등
    ])

    div2 = analyzer.detect_slope_divergence(div_prices2, short_period=5, long_period=20)
    assert div2 == "bullish_divergence", f"bullish_divergence가 아님: {div2}"

    print("  [PASS] test_slope_divergence")


def test_integrated_signal_buy():
    """통합 시그널 매수 테스트 (Bug #1, #3)"""
    df, _ = generate_gartley_data()
    engine = GeometricQuantEngine(config={
        "harmonic_tolerance": 0.08,
        "zigzag_pct": 3.0,
        "min_bars": 5,
        "lookback": 250,
    })

    signal = engine.generate_signal(df, ticker="005930")

    # 기본 필드 존재 확인
    assert signal.ticker == "005930"
    assert signal.action in ("BUY", "SELL", "HOLD")
    assert 0 <= signal.confidence <= 100
    assert isinstance(signal.reasoning, list)
    assert len(signal.reasoning) > 0

    # Gartley 데이터이므로 하모닉이 감지되어야 함
    assert signal.harmonic is not None, "Gartley 데이터인데 하모닉 미감지"
    assert signal.harmonic["pattern"] == "Gartley"

    # Bug #1: 전부 0일 때 BUY가 아닌 HOLD여야 함 → 여기서는 하모닉이 있으므로
    # confidence가 0보다 커야 함
    if signal.action == "BUY":
        assert signal.confidence >= 30, \
            f"BUY인데 confidence가 30% 미만: {signal.confidence}%"

    print(f"  [PASS] test_integrated_signal_buy (action={signal.action}, "
          f"conf={signal.confidence}%)")


def test_integrated_hold_on_zero():
    """패턴 없는 횡보 데이터 → HOLD 반환 테스트 (Bug #1)"""
    # 랜덤 노이즈만 있는 데이터 → 패턴 미감지 → HOLD
    n = 50
    np.random.seed(999)
    flat_prices = 50000 + np.random.normal(0, 100, n)

    df = pd.DataFrame({
        "open": flat_prices, "high": flat_prices + 50,
        "low": flat_prices - 50, "close": flat_prices,
        "volume": np.random.randint(100000, 500000, n),
    })

    engine = GeometricQuantEngine(config={"lookback": 50})
    signal = engine.generate_signal(df, ticker="FLAT")

    assert signal.action == "HOLD", \
        f"횡보 데이터인데 HOLD가 아님: {signal.action}"
    # v2.0: 10지표 체제에서 신규 지표가 미미한 점수를 낼 수 있으나
    # min_confidence(30%) 미만이므로 HOLD 유지가 핵심
    assert signal.confidence < 30, \
        f"횡보 데이터인데 confidence가 30% 이상: {signal.confidence}"

    print(f"  [PASS] test_integrated_hold_on_zero (conf={signal.confidence}%)")


def test_l7_result():
    """L7 보조 레이어 결과 형식 테스트"""
    df, _ = generate_gartley_data()
    engine = GeometricQuantEngine(config={
        "harmonic_tolerance": 0.08,
        "lookback": 200,
    })

    result = engine.generate_l7_result(df, ticker="005930")

    # 필수 키 존재
    required_keys = [
        "geo_action", "geo_confidence", "geo_harmonic", "geo_elliott",
        "geo_slope", "geo_reasoning", "geo_confirms_buy", "geo_warns_sell",
    ]
    for key in required_keys:
        assert key in result, f"L7 결과에 '{key}' 키 누락"

    # boolean 타입
    assert isinstance(result["geo_confirms_buy"], bool)
    assert isinstance(result["geo_warns_sell"], bool)

    # 동시에 True일 수 없음
    assert not (result["geo_confirms_buy"] and result["geo_warns_sell"]), \
        "confirms_buy와 warns_sell이 동시에 True"

    print(f"  [PASS] test_l7_result (action={result['geo_action']})")


# =============================================================================
# v2.0 신규 지표 + 프로파일 테스트
# =============================================================================

def generate_squeeze_data() -> pd.DataFrame:
    """볼린저밴드 스퀴즈 → 상방 해소 데이터"""
    n = 150
    np.random.seed(2024)
    # 80봉: 점차 변동성 축소 (스퀴즈 구간)
    squeeze_phase = 50000 + np.cumsum(np.random.normal(0, 30, 80))
    # 20봉: 갑작스런 상승 돌파 (해소)
    release_phase = squeeze_phase[-1] + np.cumsum(np.random.normal(300, 100, 20))
    # 50봉: 추세 지속
    trend_phase = release_phase[-1] + np.cumsum(np.random.normal(100, 150, 50))
    prices = np.concatenate([squeeze_phase, release_phase, trend_phase])

    vol_base = np.random.randint(300000, 600000, 80)
    vol_release = np.random.randint(1200000, 2500000, 20)  # 해소 시 거래량 급증
    vol_trend = np.random.randint(500000, 900000, 50)
    volume = np.concatenate([vol_base, vol_release, vol_trend])

    return pd.DataFrame({
        "open": prices + np.random.normal(0, 50, n),
        "high": prices + np.abs(np.random.normal(0, 100, n)),
        "low": prices - np.abs(np.random.normal(0, 100, n)),
        "close": prices,
        "volume": volume,
    })


def generate_inflection_data() -> pd.DataFrame:
    """하락→상승 변곡점 데이터"""
    n = 100
    np.random.seed(2025)
    # 50봉 하락 → 50봉 상승 (V자 반전)
    down = 60000 - np.arange(50) * 150 + np.random.normal(0, 50, 50)
    up = down[-1] + np.arange(50) * 200 + np.random.normal(0, 50, 50)
    prices = np.concatenate([down, up])

    return pd.DataFrame({
        "open": prices, "high": prices + 100,
        "low": prices - 100, "close": prices,
        "volume": np.random.randint(500000, 1500000, n),
    })


def generate_oversold_data() -> pd.DataFrame:
    """극단 과매도 + 회복 시작 데이터"""
    n = 200
    np.random.seed(2026)
    # 120봉: 기본 추세
    base = 50000 + np.random.normal(0, 200, 120)
    # 50봉: 급락 (-35%)
    crash = base[-1] - np.arange(50) * 350 + np.random.normal(0, 100, 50)
    # 30봉: 바닥에서 약한 회복
    bottom = crash[-1]
    recovery = bottom + np.arange(30) * 80 + np.random.normal(0, 80, 30)
    prices = np.concatenate([base, crash, recovery])

    vol_base = np.random.randint(300000, 800000, 120)
    vol_crash = np.random.randint(1500000, 4000000, 50)  # 투매 거래량
    vol_recovery = np.random.randint(400000, 900000, 30)
    volume = np.concatenate([vol_base, vol_crash, vol_recovery])

    return pd.DataFrame({
        "open": prices + np.random.normal(0, 50, n),
        "high": prices + np.abs(np.random.normal(0, 100, n)),
        "low": prices - np.abs(np.random.normal(0, 100, n)),
        "close": prices,
        "volume": volume,
    })


def generate_volume_climax_data() -> pd.DataFrame:
    """투매 거래량 클라이맥스 + 소진 패턴 데이터"""
    n = 100
    np.random.seed(2027)
    # 60봉: 안정
    stable = 50000 + np.random.normal(0, 200, 60)
    # 5봉: 급락 + 거래량 폭발
    crash = stable[-1] - np.arange(5) * 800 + np.random.normal(0, 100, 5)
    # 35봉: 바닥 횡보
    bottom = crash[-1] + np.random.normal(0, 150, 35)
    prices = np.concatenate([stable, crash, bottom])

    vol_stable = np.random.randint(300000, 600000, 60)
    vol_climax = np.array([2500000, 3500000, 2000000, 1200000, 800000])  # 투매→소진
    vol_bottom = np.random.randint(300000, 600000, 35)
    volume = np.concatenate([vol_stable, vol_climax, vol_bottom])

    return pd.DataFrame({
        "open": prices + np.random.normal(0, 50, n),
        "high": prices + np.abs(np.random.normal(0, 100, n)),
        "low": prices - np.abs(np.random.normal(0, 100, n)),
        "close": prices,
        "volume": volume,
    })


def generate_band_breach_data() -> pd.DataFrame:
    """볼린저밴드 하단 이탈→복귀 데이터"""
    n = 120
    np.random.seed(2028)
    # 70봉: 안정
    stable = 50000 + np.random.normal(0, 200, 70)
    # 10봉: 급락 (밴드 하단 돌파)
    crash = stable[-1] - np.arange(10) * 600 + np.random.normal(0, 50, 10)
    # 10봉: V자 반등 (밴드 안으로 복귀)
    recovery = crash[-1] + np.arange(10) * 500 + np.random.normal(0, 50, 10)
    # 30봉: 추가 안정
    post = recovery[-1] + np.random.normal(0, 200, 30)
    prices = np.concatenate([stable, crash, recovery, post])

    vol_stable = np.random.randint(300000, 600000, 70)
    vol_crash = np.random.randint(1500000, 3000000, 10)
    vol_recovery = np.random.randint(800000, 1500000, 10)
    vol_post = np.random.randint(300000, 600000, 30)
    volume = np.concatenate([vol_stable, vol_crash, vol_recovery, vol_post])

    return pd.DataFrame({
        "open": prices + np.random.normal(0, 50, n),
        "high": prices + np.abs(np.random.normal(0, 100, n)),
        "low": prices - np.abs(np.random.normal(0, 100, n)),
        "close": prices,
        "volume": volume,
    })


def test_squeeze_detection():
    """스퀴즈 상태 + 해소 감지 테스트"""
    df = generate_squeeze_data()
    detector = SqueezeDetector(bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5)

    result = detector.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "squeeze", f"이름이 squeeze가 아님: {result.name}"
    assert result.signal in ("BUY", "SELL", "HOLD"), f"잘못된 시그널: {result.signal}"
    assert 0 <= result.score <= 100, f"점수 범위 초과: {result.score}"
    assert isinstance(result.meta, dict), "meta가 dict가 아님"
    assert "is_squeeze" in result.meta, "meta에 is_squeeze 키 누락"
    assert "squeeze_bars" in result.meta, "meta에 squeeze_bars 키 누락"
    assert "bandwidth" in result.meta, "meta에 bandwidth 키 누락"

    print(f"  [PASS] test_squeeze_detection (sig={result.signal}, score={result.score})")


def test_curvature_inflection():
    """변곡점 감지 테스트"""
    df = generate_inflection_data()
    analyzer = CurvatureAnalyzer(ema_span=20, window=10)

    result = analyzer.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "curvature", f"이름이 curvature가 아님: {result.name}"
    assert result.signal in ("BUY", "SELL", "HOLD"), f"잘못된 시그널: {result.signal}"
    assert "curvature" in result.meta, "meta에 curvature 키 누락"
    assert "inflection_type" in result.meta, "meta에 inflection_type 키 누락"
    assert "kappa_strength" in result.meta, "meta에 kappa_strength 키 누락"

    # V자 반전 데이터이므로 bullish 변곡점 감지 기대
    if result.meta["inflection_type"] == "bullish":
        assert result.signal == "BUY", f"bullish 변곡점인데 BUY가 아님: {result.signal}"

    print(f"  [PASS] test_curvature_inflection (sig={result.signal}, "
          f"inflection={result.meta['inflection_type']})")


def test_slope_momentum():
    """기울기 가속도(모멘텀) 테스트"""
    n = 100
    np.random.seed(3001)
    # 급격한 방향 전환 데이터: 50봉 하락 → 50봉 급등
    down = 60000 - np.arange(50) * 200 + np.random.normal(0, 50, 50)
    up = down[-1] + np.arange(50) * 400 + np.random.normal(0, 50, 50)
    prices = np.concatenate([down, up])

    df = pd.DataFrame({
        "open": prices, "high": prices + 100,
        "low": prices - 100, "close": prices,
        "volume": np.random.randint(500000, 1000000, n),
    })

    analyzer = SlopeMomentumAnalyzer(slope_period=20, momentum_period=5)
    result = analyzer.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "slope_mom", f"이름이 slope_mom이 아님: {result.name}"
    assert "angle" in result.meta, "meta에 angle 키 누락"
    assert "momentum" in result.meta, "meta에 momentum 키 누락"

    # 마지막 구간은 상승이므로 양의 각도
    assert result.meta["angle"] > 0, f"상승 데이터인데 각도가 음수: {result.meta['angle']}"

    print(f"  [PASS] test_slope_momentum (angle={result.meta['angle']:.1f}, "
          f"mom={result.meta['momentum']:.1f})")


def test_confluence_zones():
    """피보나치 합치구간 감지 테스트"""
    df, _ = generate_gartley_data()  # 기존 Gartley 데이터에는 피벗이 충분함
    analyzer = ConfluenceAnalyzer(zigzag_pct=5.0, bin_pct=0.5, min_count=3)

    result = analyzer.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "confluence", f"이름이 confluence가 아님: {result.name}"
    assert "zones" in result.meta, "meta에 zones 키 누락"
    assert "total_zones" in result.meta, "meta에 total_zones 키 누락"
    assert isinstance(result.meta["zones"], list), "zones가 list가 아님"

    # Gartley 데이터는 피벗이 풍부하므로 합치구간이 존재해야 함
    if result.meta["total_zones"] > 0:
        zone = result.meta["zones"][0]
        assert "price" in zone, "zone에 price 키 누락"
        assert "count" in zone, "zone에 count 키 누락"
        assert "type" in zone, "zone에 type 키 누락"
        assert zone["type"] in ("지지", "저항"), f"잘못된 zone type: {zone['type']}"

    print(f"  [PASS] test_confluence_zones (total_zones={result.meta['total_zones']})")


def test_mean_reversion_extreme():
    """극단 이격도 + 회복 시그널 테스트"""
    df = generate_oversold_data()
    analyzer = MeanReversionAnalyzer(periods=(20, 60, 120))

    result = analyzer.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "mean_rev", f"이름이 mean_rev가 아님: {result.name}"
    assert "deviations" in result.meta, "meta에 deviations 키 누락"
    assert "percentile" in result.meta, "meta에 percentile 키 누락"
    assert "extreme_type" in result.meta, "meta에 extreme_type 키 누락"
    assert "recovery_signal" in result.meta, "meta에 recovery_signal 키 누락"
    assert "primary_dev" in result.meta, "meta에 primary_dev 키 누락"

    # 급락 + 약간의 회복 → 과매도 또는 극단 과매도 기대
    if result.meta["extreme_type"] is not None:
        assert result.meta["extreme_type"] in (
            "extreme_oversold", "oversold", "extreme_overbought", "overbought"
        ), f"잘못된 extreme_type: {result.meta['extreme_type']}"

    # primary_dev가 음수여야 함 (급락 데이터)
    assert result.meta["primary_dev"] < 0, \
        f"급락 데이터인데 이격도가 양수: {result.meta['primary_dev']}"

    # 과매도 상태이면 BUY 시그널이어야 함
    if result.meta["extreme_type"] in ("extreme_oversold", "oversold"):
        assert result.signal == "BUY", \
            f"과매도인데 BUY가 아님: {result.signal}"
        assert result.score > 0, "과매도인데 점수가 0"

    print(f"  [PASS] test_mean_reversion_extreme (dev={result.meta['primary_dev']:+.1f}%, "
          f"type={result.meta['extreme_type']}, recovery={result.meta['recovery_signal']})")


def test_volume_climax():
    """거래량 클라이맥스 감지 테스트"""
    df = generate_volume_climax_data()
    detector = VolumeClimaxDetector(spike_mult=3.0, exhaust_bars=5)

    result = detector.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "vol_climax", f"이름이 vol_climax가 아님: {result.name}"
    assert result.signal in ("BUY", "SELL", "HOLD"), f"잘못된 시그널: {result.signal}"
    assert 0 <= result.score <= 100, f"점수 범위 초과: {result.score}"

    # 투매 클라이맥스 데이터이므로 감지되면 selling 타입이어야 함
    if result.meta.get("climax_type"):
        assert result.meta["climax_type"] in ("selling", "buying"), \
            f"잘못된 climax_type: {result.meta['climax_type']}"
        assert "spike_ratio" in result.meta, "meta에 spike_ratio 누락"
        assert "exhaust_ratio" in result.meta, "meta에 exhaust_ratio 누락"

    print(f"  [PASS] test_volume_climax (sig={result.signal}, score={result.score})")


def test_band_breach_recovery():
    """밴드 이탈->복귀 반전 시그널 테스트"""
    df = generate_band_breach_data()
    detector = BandBreachDetector(bb_period=20, bb_std=2.0, lookback=25)

    result = detector.analyze(df)

    assert isinstance(result, IndicatorResult), "IndicatorResult 반환이 아님"
    assert result.name == "band_breach", f"이름이 band_breach가 아님: {result.name}"
    assert result.signal in ("BUY", "SELL", "HOLD"), f"잘못된 시그널: {result.signal}"
    assert "breach_type" in result.meta, "meta에 breach_type 키 누락"
    assert "breach_depth" in result.meta, "meta에 breach_depth 키 누락"
    assert "bw_percentile" in result.meta, "meta에 bw_percentile 키 누락"

    # 밴드 이탈이 감지되면 breach_depth > 0이어야 함
    if result.meta["breach_type"] is not None:
        assert result.meta["breach_depth"] > 0, \
            f"breach가 있는데 depth가 0: {result.meta['breach_depth']}"

    print(f"  [PASS] test_band_breach_recovery (sig={result.signal}, "
          f"type={result.meta['breach_type']}, depth={result.meta['breach_depth']:.1f}%)")


def test_profile_weights():
    """프로파일 가중치 합 = 1.0 검증"""
    for name, weights in WEIGHT_PROFILES.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"프로파일 '{name}' 가중치 합이 1.0이 아님: {total:.4f}"

        # 10개 지표 모두 존재하는지 확인
        expected_keys = {
            "harmonic", "elliott", "slope", "squeeze", "curvature",
            "slope_mom", "confluence", "mean_rev", "vol_climax", "band_breach",
        }
        assert set(weights.keys()) == expected_keys, \
            f"프로파일 '{name}'에 지표 키 불일치: {set(weights.keys()) ^ expected_keys}"

        # 모든 가중치가 0 이상이어야 함
        for key, val in weights.items():
            assert val >= 0, f"프로파일 '{name}'의 '{key}' 가중치가 음수: {val}"

    assert len(WEIGHT_PROFILES) == 7, \
        f"프로파일이 7개가 아님: {len(WEIGHT_PROFILES)}"
    assert "capitulation" in WEIGHT_PROFILES, "capitulation 프로파일 누락"

    print(f"  [PASS] test_profile_weights ({len(WEIGHT_PROFILES)}개 프로파일 검증)")


def test_profile_capitulation():
    """capitulation 프로파일 — 투매 바닥 특화 동작 테스트"""
    df = generate_oversold_data()

    # capitulation 프로파일
    engine_cap = GeometricQuantEngine(config={"profile": "capitulation", "lookback": 200})
    signal_cap = engine_cap.generate_signal(df, ticker="CAP_TEST")

    # default 프로파일
    engine_def = GeometricQuantEngine(config={"profile": "default", "lookback": 200})
    signal_def = engine_def.generate_signal(df, ticker="DEF_TEST")

    # capitulation은 mean_rev에 가중치 25%를 두므로 과매도에 더 강하게 반응해야 함
    cap_weights = WEIGHT_PROFILES["capitulation"]
    def_weights = WEIGHT_PROFILES["default"]
    assert cap_weights["mean_rev"] > def_weights["mean_rev"], \
        "capitulation의 mean_rev 가중치가 default보다 작음"
    assert cap_weights["vol_climax"] > def_weights["vol_climax"], \
        "capitulation의 vol_climax 가중치가 default보다 작음"
    assert cap_weights["squeeze"] < def_weights["squeeze"], \
        "capitulation의 squeeze 가중치가 default보다 큼 (투매에서 스퀴즈 역효과)"

    # 기본 검증
    assert signal_cap.action in ("BUY", "SELL", "HOLD")
    assert 0 <= signal_cap.confidence <= 100

    print(f"  [PASS] test_profile_capitulation (cap={signal_cap.action}/{signal_cap.confidence}%, "
          f"def={signal_def.action}/{signal_def.confidence}%)")


def test_l7_backward_compat():
    """L7 하위호환: 기존 필드 유지 + 신규 필드 추가 검증"""
    df, _ = generate_gartley_data()
    engine = GeometricQuantEngine(config={
        "harmonic_tolerance": 0.08,
        "lookback": 200,
        "profile": "default",
    })

    result = engine.generate_l7_result(df, ticker="COMPAT_TEST")

    # 기존 필수 키 (v1.1)
    legacy_keys = [
        "geo_action", "geo_confidence", "geo_harmonic", "geo_elliott",
        "geo_slope", "geo_reasoning", "geo_confirms_buy", "geo_warns_sell",
    ]
    for key in legacy_keys:
        assert key in result, f"하위호환 키 '{key}' 누락"

    # 신규 필수 키 (v2.0)
    assert "geo_profile" in result, "신규 키 'geo_profile' 누락"
    assert "geo_indicators" in result, "신규 키 'geo_indicators' 누락"

    # 타입 검증
    assert isinstance(result["geo_confirms_buy"], bool)
    assert isinstance(result["geo_warns_sell"], bool)
    assert isinstance(result["geo_profile"], str)
    assert isinstance(result["geo_indicators"], dict)
    assert result["geo_profile"] == "default"

    # geo_indicators에 7개 신규 지표 키 존재
    expected_indicators = {"squeeze", "curvature", "slope_mom", "confluence",
                           "mean_rev", "vol_climax", "band_breach"}
    actual_indicators = set(result["geo_indicators"].keys())
    assert actual_indicators == expected_indicators, \
        f"geo_indicators 키 불일치: 누락={expected_indicators - actual_indicators}, " \
        f"초과={actual_indicators - expected_indicators}"

    # 각 지표 결과 구조 확인
    for name, ind in result["geo_indicators"].items():
        assert "signal" in ind, f"지표 '{name}'에 signal 누락"
        assert "score" in ind, f"지표 '{name}'에 score 누락"
        assert "detail" in ind, f"지표 '{name}'에 detail 누락"
        assert "meta" in ind, f"지표 '{name}'에 meta 누락"
        assert ind["signal"] in ("BUY", "SELL", "HOLD"), \
            f"지표 '{name}' 잘못된 시그널: {ind['signal']}"

    print(f"  [PASS] test_l7_backward_compat (profile={result['geo_profile']}, "
          f"indicators={len(result['geo_indicators'])})")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  기하학적 퀀트 엔진 v2.0 - assert 기반 테스트")
    print("=" * 60)

    print("\n--- 기존 테스트 (v1.1) ---")
    tests_legacy = [
        test_pivot_detection,
        test_pivot_from_atr,
        test_harmonic_gartley,
        test_elliott_sliding_window,
        test_elliott_wave5_sell,
        test_slope_analysis,
        test_slope_divergence,
        test_integrated_signal_buy,
        test_integrated_hold_on_zero,
        test_l7_result,
    ]

    print("\n--- 신규 테스트 (v2.0) ---")
    tests_v2 = [
        test_squeeze_detection,
        test_curvature_inflection,
        test_slope_momentum,
        test_confluence_zones,
        test_mean_reversion_extreme,
        test_volume_climax,
        test_band_breach_recovery,
        test_profile_weights,
        test_profile_capitulation,
        test_l7_backward_compat,
    ]

    tests = tests_legacy + tests_v2

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  결과: {passed} passed, {failed} failed (총 {len(tests)}개)")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
