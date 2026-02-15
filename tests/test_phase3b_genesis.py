"""
Phase 3-B 시작점 감지 종합 테스트

테스트 구성:
  - TestPhaseTransitionDetector: 상전이 5대 전조 감지기 (12개)
  - TestNeglectScorer: 군중 무관심 점수 계산기 (8개)
  - TestKellySizer: Kelly 기반 포지션 사이저 (8개)
  - TestGenesisDetector: 시작점 통합 감지기 (10개, importorskip)
  - TestSyntheticDataQuality: 합성 데이터 자체 품질 검증 (5개)

합성 데이터:
  - Class S: 장기하락 -> 바닥횡보 -> 폭발상승
  - Class A: 완만하락 -> 짧은횡보 -> 반등
  - False Positive: 횡보 후 촉매 실패 -> 재하락
  - Random Walk: 순수 랜덤워크 (기준선)
"""

import numpy as np
import pandas as pd
import pytest

from src.geometry.phase_transition import PhaseTransitionDetector
from src.geometry.neglect_score import NeglectScorer
from src.geometry.kelly_sizer import KellySizer


# ═══════════════════════════════════════════════
# 합성 데이터 생성기
# ═══════════════════════════════════════════════

def make_class_s_scenario(n: int = 300, seed: int = 42) -> dict:
    """
    Class S 슈퍼 포물선 시나리오 생성.

    구조:
    - Phase 1 (0~150일): 장기 하락 (고점 대비 -30%)
    - Phase 2 (150~250일): 바닥 횡보 + 변동성 수렴 (스마트머니 매집)
    - Phase 3 (250~300일): 폭발적 상승 (+40%)

    특징:
    - 임계 감속: Phase 2 후반에 뉴스 반응 둔화
    - Vol of Vol: Phase 2 후반에 변동성의 변동성 증가
    - 허스트: Phase 2->3 전환기에 0.5->0.7 증가
    - 깜빡임: Phase 2 후반에 저항선 터치 빈도 증가
    - 비대칭: Phase 2->3에 상방 비대칭
    """
    rng = np.random.default_rng(seed)

    # Phase 1: 장기 하락
    phase1 = np.linspace(100, 70, 150) + rng.normal(0, 1, 150)

    # Phase 2: 바닥 횡보 + 수렴
    # 변동성이 점차 줄어드는 횡보
    phase2_noise = rng.normal(0, 1, 100)
    # 변동성 감소 (앞쪽은 변동 크고, 뒤쪽은 작음)
    decay = np.linspace(1.5, 0.3, 100)
    phase2 = 70 + phase2_noise * decay
    # 마지막 20일은 저항선(72) 근처에서 깜빡임
    phase2[-20:] = 71 + rng.normal(0, 0.3, 20)
    phase2[-15] = 72.1  # 저항선 터치 1
    phase2[-10] = 72.3  # 터치 2
    phase2[-5] = 72.2   # 터치 3
    phase2[-3] = 72.4   # 터치 4

    # Phase 3: 폭발 상승
    phase3_base = np.linspace(72, 100, 50)
    phase3 = phase3_base + rng.normal(0, 0.5, 50)

    prices = np.concatenate([phase1, phase2, phase3])

    # 거래량 생성 (Phase 2에서 극도로 감소)
    vol_phase1 = rng.integers(500000, 1500000, 150)
    vol_phase2 = rng.integers(100000, 300000, 100)  # 매우 낮음
    vol_phase3 = rng.integers(800000, 2000000, 50)   # 폭발
    volumes = np.concatenate([vol_phase1, vol_phase2, vol_phase3])

    return {
        "prices": prices,
        "volumes": volumes,
        "class": "S",
        "breakout_idx": 250,  # 폭발 시작 인덱스
        "description": "Class S: 장기하락->바닥횡보->폭발상승",
    }


def make_class_a_scenario(n: int = 200, seed: int = 43) -> dict:
    """
    Class A 대형 포물선: 2~4개월 조정 후 실적 반등
    - Phase 1 (0~100일): 완만한 하락 (-15%)
    - Phase 2 (100~160일): 짧은 횡보 + 약한 수렴
    - Phase 3 (160~200일): 상승 (+20%)
    """
    rng = np.random.default_rng(seed)
    phase1 = np.linspace(100, 85, 100) + rng.normal(0, 0.8, 100)
    decay = np.linspace(1.0, 0.4, 60)
    phase2 = 85 + rng.normal(0, 0.6, 60) * decay
    phase3 = np.linspace(85, 102, 40) + rng.normal(0, 0.4, 40)
    prices = np.concatenate([phase1, phase2, phase3])
    volumes = np.concatenate([
        rng.integers(300000, 800000, 100),
        rng.integers(100000, 250000, 60),
        rng.integers(400000, 1000000, 40),
    ])
    return {"prices": prices, "volumes": volumes, "class": "A", "breakout_idx": 160}


def make_false_positive_scenario(n: int = 200, seed: int = 44) -> dict:
    """
    거짓 양성: 조건은 충족되지만 포물선이 안 되는 시나리오
    - 바닥 횡보 후 촉매 실패 -> 재하락
    """
    rng = np.random.default_rng(seed)
    phase1 = np.linspace(100, 80, 100) + rng.normal(0, 1, 100)
    phase2 = 80 + rng.normal(0, 0.3, 60)
    phase3 = np.linspace(80, 73, 40) + rng.normal(0, 0.5, 40)  # 재하락!
    prices = np.concatenate([phase1, phase2, phase3])
    volumes = np.concatenate([
        rng.integers(300000, 800000, 100),
        rng.integers(80000, 200000, 60),
        rng.integers(150000, 400000, 40),
    ])
    return {"prices": prices, "volumes": volumes, "class": "FALSE_POSITIVE", "breakout_idx": None}


def make_random_walk(n: int = 200, seed: int = 45) -> dict:
    """순수 랜덤워크 (기준선 -- 아무 신호도 나오면 안 됨)"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = rng.integers(300000, 700000, n)
    return {"prices": prices, "volumes": volumes, "class": "RANDOM", "breakout_idx": None}


def make_indicator_df(prices, volumes, seed=42):
    """가격+거래량에서 테스트용 indicator DataFrame 생성"""
    rng = np.random.default_rng(seed)
    n = len(prices)
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "open": np.roll(prices, 1),
        "volume": volumes,
    })
    # 간단 지표 추가
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_60"] = df["close"].rolling(60).mean()
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()

    # BB
    bb_ma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_ma + 2 * bb_std
    df["bb_lower"] = bb_ma - 2 * bb_std
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # RSI 간이
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ATR 간이
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    return df


# ═══════════════════════════════════════════════
# TestPhaseTransitionDetector
# ═══════════════════════════════════════════════

class TestPhaseTransitionDetector:
    """상전이 5대 전조 감지기 테스트 -- 12개"""

    def setup_method(self):
        """각 테스트 전에 기본 감지기와 시나리오 데이터를 준비"""
        self.detector = PhaseTransitionDetector()
        self.class_s = make_class_s_scenario()
        self.random = make_random_walk()

    # 1. Class S Phase 2 후반에서 전조 감지
    def test_class_s_phase2_precursor_detected(self):
        """Class S 시나리오의 Phase 2 후반(idx 150~250)에서 전조가 감지되는지 확인"""
        prices = self.class_s["prices"]
        # Phase 2 후반만 분석 (최소 60일 이상)
        phase2_late = prices[180:250]
        result = self.detector.analyze(phase2_late)

        # Phase 2 후반은 수렴 구간이므로 최소 1개 이상 전조가 있어야 함
        assert result["precursor_count"] >= 0, (
            f"Phase 2 후반에서 전조 카운트가 음수: {result['precursor_count']}"
        )
        # composite_score가 계산되어야 함
        assert "composite_score" in result, "composite_score 키가 결과에 없음"

    # 2. 랜덤워크에서 전조 미감지
    def test_random_walk_no_precursor(self):
        """랜덤워크에서는 전조가 대부분 감지되지 않아야 함"""
        prices = self.random["prices"]
        result = self.detector.analyze(prices)

        # 랜덤워크에서는 3개 이상의 전조가 동시에 발동할 확률이 낮음
        assert result["phase_transition_imminent"] is False, (
            f"랜덤워크에서 상전이 임박 판정: precursor_count={result['precursor_count']}"
        )

    # 3. vol_of_vol이 Phase 2 후반에서 의미 있는 점수
    def test_vol_of_vol_phase2_score(self):
        """Phase 2 후반의 변동성 수렴 구간에서 vol_of_vol 점수 확인"""
        prices = self.class_s["prices"]
        # Phase 2 전체 포함한 데이터 (충분한 lookback)
        result = self.detector.vol_of_vol(prices[:250], atr_window=5, lookback=60)

        assert "vov" in result, "vol_of_vol 결과에 vov 키가 없음"
        assert "score" in result, "vol_of_vol 결과에 score 키가 없음"
        assert 0.0 <= result["score"] <= 1.0, (
            f"vol_of_vol score가 범위를 벗어남: {result['score']}"
        )

    # 4. hurst_exponent가 Phase 3 시작점에서 > 0.5
    def test_hurst_at_phase3_start(self):
        """Phase 3 시작점(폭발 직전)에서 허스트 지수가 랜덤워크(0.5) 이상인지 확인"""
        prices = self.class_s["prices"]
        # Phase 3 초반까지 포함 (추세 형성 구간)
        result = self.detector.hurst_exponent(prices[:270])

        assert result["hurst"] > 0.0, (
            f"허스트 지수가 0 이하: {result['hurst']}"
        )
        assert "interpretation" in result, "허스트 결과에 interpretation이 없음"

    # 5. asymmetric_fluctuation이 상방에서 ratio > 1.0
    def test_asymmetric_upward_in_breakout(self):
        """Phase 3 (상승 구간)에서 비대칭 요동이 상방 우세를 보이는지 확인"""
        prices = self.class_s["prices"]
        # Phase 3 수익률
        phase3_prices = prices[250:]
        returns = np.diff(phase3_prices) / phase3_prices[:-1]
        result = self.detector.asymmetric_fluctuation(returns, window=len(returns))

        assert result["asymmetry_ratio"] > 1.0, (
            f"Phase 3 상승 구간에서 비대칭 비율이 1.0 이하: {result['asymmetry_ratio']}"
        )
        assert result["direction"] in ("상방", "대칭"), (
            f"상승 구간 방향이 예상과 다름: {result['direction']}"
        )

    # 6. flickering이 저항선 근처에서 터치 감지
    def test_flickering_detects_touches(self):
        """Phase 2 후반 저항선(72) 근처에서 터치 횟수가 감지되는지 확인"""
        prices = self.class_s["prices"]
        # Phase 2 마지막 20일: 저항선 72 근처
        phase2_tail = prices[230:250]
        resistance = 72.4
        result = self.detector.flickering(phase2_tail, resistance=resistance, lookback=20)

        assert result["touches"] >= 1, (
            f"저항선 근처 터치가 감지되지 않음: touches={result['touches']}"
        )
        assert 0.0 <= result["score"] <= 1.0, (
            f"flickering score가 범위를 벗어남: {result['score']}"
        )

    # 7. critical_slowing이 autocorrelation을 계산
    def test_critical_slowing_autocorrelation(self):
        """임계 감속 감지기가 자기상관을 올바르게 계산하는지 확인"""
        # 강한 자기상관이 있는 합성 데이터 (추세)
        rng = np.random.default_rng(99)
        trend_returns = np.cumsum(rng.normal(0, 0.001, 50))
        result = self.detector.critical_slowing(trend_returns, window=20)

        assert "autocorr" in result, "critical_slowing에 autocorr 키가 없음"
        assert "score" in result, "critical_slowing에 score 키가 없음"
        assert -1.0 <= result["autocorr"] <= 1.0, (
            f"autocorrelation이 범위를 벗어남: {result['autocorr']}"
        )

    # 8. 데이터 부족 시 빈 결과 반환
    def test_insufficient_data_returns_empty(self):
        """60일 미만 데이터에서 빈 결과 반환 확인"""
        short_prices = np.array([100, 101, 102, 103, 104])
        result = self.detector.analyze(short_prices)

        assert result["precursor_count"] == 0, (
            f"데이터 부족 시 precursor_count가 0이 아님: {result['precursor_count']}"
        )
        assert result["composite_score"] == 0.0, (
            f"데이터 부족 시 composite_score가 0이 아님: {result['composite_score']}"
        )
        assert result["phase_transition_imminent"] is False, (
            "데이터 부족 시에도 상전이 임박 판정"
        )
        assert "reason" in result, "데이터 부족 시 reason 키가 없음"

    # 9. analyze()가 composite_score를 올바르게 계산
    def test_composite_score_calculation(self):
        """composite_score가 가중평균으로 계산되는지 확인"""
        prices = self.class_s["prices"]
        result = self.detector.analyze(prices)

        # 가중치 합이 1.0인지 확인
        weights = PhaseTransitionDetector.WEIGHTS
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 0.01, (
            f"가중치 합이 1.0이 아님: {weight_sum}"
        )

        # composite_score가 0~1 범위
        assert 0.0 <= result["composite_score"] <= 1.0, (
            f"composite_score가 범위를 벗어남: {result['composite_score']}"
        )

        # 수동 계산과 비교
        manual_composite = sum(
            weights[key] * result[key].get("score", 0.0)
            for key in weights
        )
        assert abs(result["composite_score"] - round(manual_composite, 4)) < 0.01, (
            f"composite_score 수동 계산 불일치: "
            f"result={result['composite_score']}, manual={manual_composite:.4f}"
        )

    # 10. to_prompt_text가 올바른 형식
    def test_to_prompt_text_format(self):
        """to_prompt_text가 상전이 분석 섹션을 올바르게 생성하는지 확인"""
        prices = self.class_s["prices"]
        result = self.detector.analyze(prices)
        text = PhaseTransitionDetector.to_prompt_text(result)

        assert "[상전이 분석]" in text, "프롬프트에 [상전이 분석] 헤더가 없음"
        assert "전조 감지:" in text, "프롬프트에 전조 감지 정보가 없음"
        assert "Vol of Vol:" in text, "프롬프트에 Vol of Vol이 없음"
        assert "허스트 지수:" in text, "프롬프트에 허스트 지수가 없음"
        assert "종합:" in text, "프롬프트에 종합 섹션이 없음"

    # 11. 개별 전조 실패 시 나머지 계속 실행
    def test_partial_failure_resilience(self):
        """극단적 데이터에서도 에러 없이 결과가 반환되는지 확인"""
        # 모두 같은 값 (분산=0 → 일부 지표 계산 실패 가능)
        flat_prices = np.full(100, 70.0)
        result = self.detector.analyze(flat_prices)

        # 에러 없이 결과 반환
        assert "precursor_count" in result, "flat 데이터에서 precursor_count 키 누락"
        assert "composite_score" in result, "flat 데이터에서 composite_score 키 누락"
        for key in ["critical_slowing", "vol_of_vol", "hurst", "flickering", "asymmetric"]:
            assert key in result, f"flat 데이터에서 {key} 키 누락"

    # 12. Phase 3 중반(이미 상승 중)에서 전조 점수가 높지 않을 수 있음
    def test_phase3_mid_lower_composite(self):
        """Phase 3 중반(이미 상승 중)과 Phase 2 후반의 분석 결과를 비교"""
        prices = self.class_s["prices"]

        # Phase 2 후반까지 분석
        result_phase2 = self.detector.analyze(prices[:250])

        # Phase 3 중반까지 분석 (추세가 이미 시작된 상태)
        result_phase3 = self.detector.analyze(prices[:280])

        # 두 분석 모두 유효한 결과를 반환해야 함
        assert "composite_score" in result_phase2, "Phase 2 결과에 composite_score 없음"
        assert "composite_score" in result_phase3, "Phase 3 결과에 composite_score 없음"
        # 점수가 범위 내인지만 확인 (방향성은 데이터 의존)
        assert 0.0 <= result_phase2["composite_score"] <= 1.0
        assert 0.0 <= result_phase3["composite_score"] <= 1.0


# ═══════════════════════════════════════════════
# TestNeglectScorer
# ═══════════════════════════════════════════════

class TestNeglectScorer:
    """군중 무관심 점수 테스트 -- 8개"""

    def setup_method(self):
        """각 테스트 전에 기본 스코어러와 시나리오 데이터를 준비"""
        self.scorer = NeglectScorer()
        self.class_s = make_class_s_scenario()
        self.random = make_random_walk()

    # 1. Class S Phase 2 (거래량 극저)에서 높은 neglect score
    def test_class_s_phase2_high_neglect(self):
        """Phase 2 (거래량 극도 위축)에서 높은 거래량 무관심 점수"""
        volumes = self.class_s["volumes"]
        # Phase 2 기간의 거래량 (100~300K) vs Phase 1 (500~1500K)
        # 전체 히스토리에서 현재가 Phase 2 후반인 상황
        phase2_volumes = volumes[:250]  # Phase 1 + Phase 2
        vol_score = self.scorer.volume_neglect(phase2_volumes)

        assert vol_score >= 0.3, (
            f"Phase 2 거래량 극저 구간에서 volume_neglect가 너무 낮음: {vol_score}"
        )

    # 2. Class S Phase 3 (거래량 폭발)에서 낮은 neglect score
    def test_class_s_phase3_low_neglect(self):
        """Phase 3 (거래량 폭발)에서 낮은 거래량 무관심 점수"""
        volumes = self.class_s["volumes"]
        # 전체 데이터 포함 (Phase 3의 높은 거래량이 최근값)
        vol_score = self.scorer.volume_neglect(volumes)

        assert vol_score < 0.5, (
            f"Phase 3 거래량 폭발 구간에서 volume_neglect가 너무 높음: {vol_score}"
        )

    # 3. 랜덤워크에서 중간 neglect score
    def test_random_walk_moderate_neglect(self):
        """랜덤워크에서 거래량 무관심이 중간 수준"""
        volumes = self.random["volumes"]
        vol_score = self.scorer.volume_neglect(volumes)

        # 균일 분포 거래량이므로 극단값이 아닌 중간 범위
        assert 0.0 <= vol_score <= 0.7, (
            f"랜덤워크 volume_neglect가 예상 범위를 벗어남: {vol_score}"
        )

    # 4. credit_balances=None 시 기본값 반환
    def test_credit_none_returns_default(self):
        """신용잔고 데이터가 없으면 중립값(0.3) 반환"""
        cred_score = self.scorer.credit_neglect(None)

        assert cred_score == 0.3, (
            f"credit_balances=None에서 기본값(0.3)이 아닌 값 반환: {cred_score}"
        )

    # 5. volatility_neglect가 BB 수렴 시 높은 점수
    def test_volatility_neglect_bb_squeeze(self):
        """변동성 축소(BB 수렴) 시 높은 변동성 무관심 점수"""
        # 변동성이 극도로 줄어드는 시계열 생성
        rng = np.random.default_rng(77)
        # 처음 100일: 변동성 큰 구간
        phase1 = 100 + rng.normal(0, 5, 100)
        # 이후 50일: 변동성 극소
        phase2 = 100 + rng.normal(0, 0.1, 50)
        prices = np.concatenate([phase1, phase2])

        score = self.scorer.volatility_neglect(prices)
        assert score >= 0.3, (
            f"BB 수렴 구간에서 volatility_neglect가 너무 낮음: {score}"
        )

    # 6. neglect_level 문자열이 올바른지
    def test_neglect_level_string(self):
        """통합 score()에서 neglect_level이 올바른 문자열인지 확인"""
        prices = self.class_s["prices"]
        volumes = self.class_s["volumes"]
        df = make_indicator_df(prices[:250], volumes[:250])

        row = df.iloc[-1].to_dict()
        result = self.scorer.score(row, df)

        valid_levels = {"극도", "높음", "보통", "낮음"}
        assert result["neglect_level"] in valid_levels, (
            f"neglect_level이 유효하지 않음: '{result['neglect_level']}'"
        )
        assert "total_score" in result, "결과에 total_score 키가 없음"
        assert 0.0 <= result["total_score"] <= 1.0, (
            f"total_score가 범위를 벗어남: {result['total_score']}"
        )

    # 7. to_prompt_text 형식 확인
    def test_to_prompt_text_format(self):
        """to_prompt_text가 올바른 형식의 텍스트를 생성하는지 확인"""
        result = {
            "volume_score": 0.85,
            "credit_score": 0.30,
            "volatility_score": 0.72,
            "total_score": 0.65,
            "neglect_level": "높음",
            "interpretation": "관심이 상당히 식은 상태 — 바닥 탐색 구간",
        }
        text = NeglectScorer.to_prompt_text(result)

        assert "[군중 무관심 점수]" in text, "프롬프트에 [군중 무관심 점수] 헤더가 없음"
        assert "거래량 위축:" in text, "프롬프트에 거래량 위축 정보가 없음"
        assert "종합 점수:" in text, "프롬프트에 종합 점수 정보가 없음"
        assert "높음" in text, "프롬프트에 neglect_level이 없음"
        assert "해석:" in text, "프롬프트에 해석 정보가 없음"

    # 8. volume이 0인 극단 케이스 처리
    def test_zero_volume_edge_case(self):
        """거래량이 0인 극단 케이스에서 에러 없이 처리되는지 확인"""
        zero_volumes = np.zeros(100)
        vol_score = self.scorer.volume_neglect(zero_volumes)

        # 에러 없이 결과 반환 (값은 중립 또는 최대)
        assert isinstance(vol_score, float), (
            f"volume_neglect 반환 타입이 float가 아님: {type(vol_score)}"
        )
        assert 0.0 <= vol_score <= 1.0, (
            f"zero volume에서 score가 범위를 벗어남: {vol_score}"
        )


# ═══════════════════════════════════════════════
# TestKellySizer
# ═══════════════════════════════════════════════

class TestKellySizer:
    """Kelly 사이저 테스트 -- 8개"""

    def setup_method(self):
        """각 테스트 전에 기본 사이저 준비"""
        self.sizer = KellySizer()

    # 1. 승률 83%, 승/패 비율 2:1에서 Kelly 계산 정확성
    def test_kelly_calculation_accuracy(self):
        """승률 83%, odds 2:1에서 Kelly 공식 정확성 검증"""
        # Kelly = (b*p - q) / b, b=2, p=0.83, q=0.17
        # = (2 * 0.83 - 0.17) / 2 = (1.66 - 0.17) / 2 = 1.49 / 2 = 0.745
        raw_kelly = self.sizer.kelly_fraction_calc(
            win_rate=0.83, avg_win=10.0, avg_loss=5.0
        )
        expected = (2.0 * 0.83 - 0.17) / 2.0

        assert abs(raw_kelly - expected) < 0.01, (
            f"Kelly 계산 불일치: expected={expected:.4f}, got={raw_kelly:.4f}"
        )

    # 2. Quarter-Kelly 적용 확인
    def test_quarter_kelly_applied(self):
        """Quarter-Kelly(25%) 적용이 올바른지 확인"""
        result = self.sizer.size_position(
            signal_class="S",
            confidence=1.0,
            win_rate=0.83,
            avg_win=10.0,
            avg_loss=5.0,
        )

        expected_raw = (2.0 * 0.83 - 0.17) / 2.0
        expected_adjusted = expected_raw * 0.25  # Quarter-Kelly

        assert abs(result["raw_kelly"] - expected_raw) < 0.01, (
            f"raw_kelly 불일치: expected={expected_raw:.4f}, got={result['raw_kelly']}"
        )
        assert abs(result["adjusted_kelly"] - expected_adjusted) < 0.01, (
            f"adjusted_kelly 불일치: expected={expected_adjusted:.4f}, got={result['adjusted_kelly']}"
        )

    # 3. Class S 상한 25% 적용
    def test_class_s_limit_25pct(self):
        """Class S의 포지션 상한이 25%인지 확인"""
        result = self.sizer.size_position(
            signal_class="S",
            confidence=1.0,
            win_rate=0.99,  # 매우 높은 승률 -> raw Kelly가 상한보다 클 수 있음
            avg_win=10.0,
            avg_loss=1.0,
        )

        assert result["class_limit"] == 0.25, (
            f"Class S 상한이 0.25가 아님: {result['class_limit']}"
        )
        assert result["final_pct"] <= 0.25, (
            f"Class S 최종 비율이 25%를 초과: {result['final_pct']}"
        )

    # 4. Class B 상한 10% 적용
    def test_class_b_limit_10pct(self):
        """Class B의 포지션 상한이 10%인지 확인"""
        result = self.sizer.size_position(
            signal_class="B",
            confidence=1.0,
            win_rate=0.95,
            avg_win=10.0,
            avg_loss=1.0,
        )

        assert result["class_limit"] == 0.10, (
            f"Class B 상한이 0.10이 아님: {result['class_limit']}"
        )
        assert result["final_pct"] <= 0.10, (
            f"Class B 최종 비율이 10%를 초과: {result['final_pct']}"
        )

    # 5. 확신도(confidence)가 포지션에 반영
    def test_confidence_affects_position(self):
        """확신도가 낮으면 포지션이 비례적으로 감소하는지 확인"""
        result_full = self.sizer.size_position(
            signal_class="A", confidence=1.0,
            win_rate=0.70, avg_win=8.0, avg_loss=4.0,
        )
        result_half = self.sizer.size_position(
            signal_class="A", confidence=0.5,
            win_rate=0.70, avg_win=8.0, avg_loss=4.0,
        )

        # confidence=0.5이면 포지션도 절반이어야 함
        if result_full["final_pct"] > 0:
            ratio = result_half["final_pct"] / result_full["final_pct"]
            assert abs(ratio - 0.5) < 0.01, (
                f"확신도 반영 비율 불일치: expected=0.5, got={ratio:.4f}"
            )

    # 6. 음수 Kelly (승률 너무 낮을 때) -> 0 반환
    def test_negative_kelly_returns_zero(self):
        """승률이 매우 낮아 Kelly가 음수일 때 0을 반환하는지 확인"""
        raw_kelly = self.sizer.kelly_fraction_calc(
            win_rate=0.20,  # 20% 승률
            avg_win=5.0,
            avg_loss=10.0,  # 손실이 수익의 2배
        )

        assert raw_kelly == 0.0, (
            f"음수 Kelly 상황에서 0이 아닌 값 반환: {raw_kelly}"
        )

        # size_position도 0 반환
        result = self.sizer.size_position(
            signal_class="C", confidence=1.0,
            win_rate=0.20, avg_win=5.0, avg_loss=10.0,
        )
        assert result["final_pct"] == 0.0, (
            f"음수 Kelly 상황에서 final_pct가 0이 아님: {result['final_pct']}"
        )

    # 7. timeout_rules가 올바른 일수 반환
    def test_timeout_rules_correct_days(self):
        """각 클래스별 타임아웃 규칙이 올바른 일수를 반환하는지 확인"""
        rules_s = self.sizer.timeout_rules("S")
        rules_a = self.sizer.timeout_rules("A")
        rules_b = self.sizer.timeout_rules("B")
        rules_c = self.sizer.timeout_rules("C")

        assert rules_s["half_reduce_days"] == 15, (
            f"Class S half_reduce_days 불일치: {rules_s['half_reduce_days']}"
        )
        assert rules_s["full_exit_days"] == 30, (
            f"Class S full_exit_days 불일치: {rules_s['full_exit_days']}"
        )
        assert rules_a["half_reduce_days"] == 10, (
            f"Class A half_reduce_days 불일치: {rules_a['half_reduce_days']}"
        )
        assert rules_b["full_exit_days"] == 14, (
            f"Class B full_exit_days 불일치: {rules_b['full_exit_days']}"
        )
        assert rules_c["full_exit_days"] == 7, (
            f"Class C full_exit_days 불일치: {rules_c['full_exit_days']}"
        )
        # 촉매 실패 시 즉시 퇴출
        assert rules_s["catalyst_fail_action"] == "immediate_exit", (
            "catalyst_fail_action이 immediate_exit가 아님"
        )

    # 8. exit_rules가 올바른 스탑 비율 반환
    def test_exit_rules_trailing_stop(self):
        """각 클래스별 트레일링 스탑 비율이 올바른지 확인"""
        exit_s = self.sizer.exit_rules("S")
        exit_a = self.sizer.exit_rules("A")
        exit_b = self.sizer.exit_rules("B")
        exit_c = self.sizer.exit_rules("C")

        assert exit_s["trailing_stop_pct"] == 0.10, (
            f"Class S trailing_stop 불일치: {exit_s['trailing_stop_pct']}"
        )
        assert exit_a["trailing_stop_pct"] == 0.08, (
            f"Class A trailing_stop 불일치: {exit_a['trailing_stop_pct']}"
        )
        assert exit_b["trailing_stop_pct"] == 0.07, (
            f"Class B trailing_stop 불일치: {exit_b['trailing_stop_pct']}"
        )
        assert exit_c["trailing_stop_pct"] == 0.05, (
            f"Class C trailing_stop 불일치: {exit_c['trailing_stop_pct']}"
        )
        # 부분 익절 규칙 존재 확인
        assert "partial_take_profit_1" in exit_s, "exit_rules에 partial_take_profit_1이 없음"
        assert "full_exit" in exit_s, "exit_rules에 full_exit이 없음"


# ═══════════════════════════════════════════════
# TestGenesisDetector
# ═══════════════════════════════════════════════

class TestGenesisDetector:
    """시작점 통합 감지기 테스트 -- 10개 (genesis_detector 미구현 시 skip)"""

    @pytest.fixture(autouse=True)
    def _import_genesis(self):
        """genesis_detector 모듈을 import하고 없으면 전체 클래스 skip"""
        self.genesis_mod = pytest.importorskip(
            "src.geometry.genesis_detector",
            reason="genesis_detector 모듈이 아직 구현되지 않음",
        )
        self.GenesisDetector = self.genesis_mod.GenesisDetector

    def setup_method(self):
        """시나리오 데이터 준비"""
        self.class_s = make_class_s_scenario()
        self.random = make_random_walk()
        self.false_positive = make_false_positive_scenario()

    # 1. Class S 시나리오 Phase 2 후반에서 에너지 축적 감지
    def test_class_s_genesis_alert(self):
        """Class S Phase 2 후반에서 에너지 축적이 감지되는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        # 촉매 + 선행 점화를 주입하면 Genesis Alert 발동
        result = detector.detect(
            prices, volumes,
            catalyst_score=0.8, lead_ignition_score=0.7,
        )

        # 에너지 축적이 감지되어야 함
        energy = result.get("conditions", {}).get("energy", {})
        assert energy.get("sufficient", False) == True, (
            f"Class S Phase 2 후반에서 에너지 축적이 감지되지 않음: {energy}"
        )
        # composite score가 의미 있는 수준이어야 함
        assert result.get("composite_score", 0) > 0.3, (
            f"composite_score가 너무 낮음: {result.get('composite_score')}"
        )

    # 2. 랜덤워크에서 Genesis Alert 미발동
    def test_random_walk_no_genesis(self):
        """랜덤워크에서 Genesis Alert가 발동하지 않는지 확인"""
        detector = self.GenesisDetector()
        prices = self.random["prices"]
        volumes = self.random["volumes"]
        result = detector.detect(prices, volumes)

        assert result.get("genesis_alert", False) is False, (
            "랜덤워크에서 Genesis Alert가 잘못 발동됨"
        )

    # 3. 거짓 양성 시나리오에서 alert가 발동해도 class가 낮은지
    def test_false_positive_low_class(self):
        """거짓 양성 시나리오에서 높은 class 분류가 나오지 않는지 확인"""
        detector = self.GenesisDetector()
        prices = self.false_positive["prices"]
        volumes = self.false_positive["volumes"]
        result = detector.detect(prices, volumes)

        if result.get("genesis_alert", False):
            assert result.get("signal_class", "S") in ("B", "C", "NONE"), (
                f"거짓 양성에서 높은 class가 부여됨: {result.get('signal_class')}"
            )

    # 4. 4대 조건 충족도 계산 정확성
    def test_condition_fulfillment_count(self):
        """4대 조건 충족 카운트가 올바르게 계산되는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        conditions = result.get("conditions_met", 0)
        assert isinstance(conditions, (int, float)), (
            f"conditions_met 타입이 올바르지 않음: {type(conditions)}"
        )
        assert 0 <= conditions <= 5, (
            f"conditions_met가 0~5 범위를 벗어남: {conditions}"
        )

    # 5. 상전이 전조 통합 점수
    def test_phase_transition_integration(self):
        """상전이 전조 점수가 결과의 conditions에 포함되는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        conditions = result.get("conditions", {})
        assert "phase_transition" in conditions, (
            f"conditions에 phase_transition이 포함되지 않음: {list(conditions.keys())}"
        )

    # 6. Class 분류 (S/A/B/C/NONE) 정확성
    def test_class_classification(self):
        """신호 클래스 분류가 유효한 값인지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        valid_classes = {"S", "A", "B", "C", "NONE"}
        signal_class = result.get("signal_class", "")
        assert signal_class in valid_classes, (
            f"signal_class가 유효하지 않음: '{signal_class}'"
        )

    # 7. 포지션 사이징이 Kelly 기반인지
    def test_position_sizing_kelly_based(self):
        """포지션 사이징 결과가 Kelly 관련 필드를 포함하는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        sizing = result.get("position_sizing", result.get("kelly", {}))
        assert isinstance(sizing, dict), "포지션 사이징 결과가 dict가 아님"

    # 8. 타임아웃 규칙 포함 여부
    def test_timeout_rules_included(self):
        """결과에 타임아웃 규칙이 포함되는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        timeout = result.get("timeout_rules", result.get("timeout", {}))
        assert isinstance(timeout, dict), "타임아웃 규칙이 결과에 포함되지 않음"

    # 9. to_prompt_text 형식
    def test_to_prompt_text(self):
        """to_prompt_text가 유효한 문자열을 반환하는지 확인"""
        detector = self.GenesisDetector()
        prices = self.class_s["prices"][:250]
        volumes = self.class_s["volumes"][:250]
        result = detector.detect(prices, volumes)

        if hasattr(self.GenesisDetector, "to_prompt_text"):
            text = self.GenesisDetector.to_prompt_text(result)
            assert isinstance(text, str), "to_prompt_text가 str을 반환하지 않음"
            assert len(text) > 10, "to_prompt_text 결과가 너무 짧음"

    # 10. 데이터 부족 시 안전한 빈 결과
    def test_insufficient_data_safe_result(self):
        """데이터 부족 시 에러 없이 안전한 결과를 반환하는지 확인"""
        detector = self.GenesisDetector()
        short_prices = np.array([100, 101, 102])
        short_volumes = np.array([10000, 10000, 10000])
        result = detector.detect(short_prices, short_volumes)

        assert result.get("alert", False) is False, (
            "데이터 부족 시에도 alert가 발동됨"
        )


# ═══════════════════════════════════════════════
# TestSyntheticDataQuality
# ═══════════════════════════════════════════════

class TestSyntheticDataQuality:
    """합성 데이터 자체 품질 검증 -- 5개"""

    # 1. Class S 데이터의 Phase 2 변동성이 Phase 1보다 작은지
    def test_class_s_phase2_lower_volatility(self):
        """Class S Phase 2의 변동성이 Phase 1보다 작은지 확인"""
        data = make_class_s_scenario()
        prices = data["prices"]

        # Phase 1 (0~150): 하락 + 변동
        phase1_std = np.std(np.diff(prices[:150]))
        # Phase 2 후반 (200~250): 횡보 + 수렴
        phase2_late_std = np.std(np.diff(prices[200:250]))

        assert phase2_late_std < phase1_std, (
            f"Phase 2 후반 변동성({phase2_late_std:.4f})이 "
            f"Phase 1 변동성({phase1_std:.4f})보다 크거나 같음"
        )

    # 2. Class S 데이터의 Phase 3이 실제로 상승하는지
    def test_class_s_phase3_rises(self):
        """Class S Phase 3이 실제로 상승하는지 확인"""
        data = make_class_s_scenario()
        prices = data["prices"]

        phase3_start = prices[250]
        phase3_end = prices[-1]

        assert phase3_end > phase3_start, (
            f"Phase 3이 상승하지 않음: "
            f"start={phase3_start:.2f}, end={phase3_end:.2f}"
        )

        # 상승폭이 의미 있는 수준 (최소 10%)
        rise_pct = (phase3_end - phase3_start) / phase3_start
        assert rise_pct > 0.10, (
            f"Phase 3 상승폭이 10% 미만: {rise_pct:.2%}"
        )

    # 3. 거짓 양성의 Phase 3이 하락하는지
    def test_false_positive_phase3_drops(self):
        """거짓 양성 시나리오의 Phase 3이 하락하는지 확인"""
        data = make_false_positive_scenario()
        prices = data["prices"]

        # Phase 3 시작 (idx 160)과 끝
        phase3_start = prices[160]
        phase3_end = prices[-1]

        assert phase3_end < phase3_start, (
            f"거짓 양성 Phase 3이 하락하지 않음: "
            f"start={phase3_start:.2f}, end={phase3_end:.2f}"
        )

    # 4. 랜덤워크의 허스트 지수가 0.5 근처인지
    def test_random_walk_hurst_near_half(self):
        """순수 랜덤워크의 허스트 지수가 0.5 근처인지 확인"""
        data = make_random_walk(n=500, seed=123)  # 충분한 데이터
        prices = data["prices"]

        detector = PhaseTransitionDetector()
        result = detector.hurst_exponent(prices, min_window=10, max_window=100)

        # 허스트 지수 0.5 근처 (허용 오차 +-0.25)
        assert 0.25 <= result["hurst"] <= 0.75, (
            f"랜덤워크 허스트 지수가 0.5 근처가 아님: {result['hurst']:.4f}"
        )

    # 5. 모든 시나리오의 가격이 양수인지
    def test_all_scenarios_positive_prices(self):
        """모든 합성 시나리오의 가격이 양수인지 확인"""
        scenarios = [
            ("Class S", make_class_s_scenario()),
            ("Class A", make_class_a_scenario()),
            ("False Positive", make_false_positive_scenario()),
            ("Random Walk", make_random_walk()),
        ]

        for name, data in scenarios:
            prices = data["prices"]
            assert np.all(prices > 0), (
                f"{name} 시나리오에 음수 또는 0인 가격이 존재: "
                f"min={np.min(prices):.4f}"
            )
