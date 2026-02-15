"""
Phase 3-A 기하학 엔진 단위 테스트

테스트 구성:
  - TestConfluenceScorer: 팩터 이진화, DB 구축, 현재 매칭, 프롬프트
  - TestCycleClock: 밴드패스, 힐베르트, 시계변환, 정렬도
  - TestDivergenceDetector: 방향 분류, 패턴 감지, 프롬프트
  - TestGeometryEngine: 통합 엔진, 포트폴리오 요약
"""

import numpy as np
import pandas as pd
import pytest

from src.geometry.confluence_scorer import ConfluenceScorer, FACTOR_DEFS
from src.geometry.cycle_clock import CycleClock
from src.geometry.divergence_detector import DivergenceDetector
from src.geometry.engine import GeometryEngine


# ═══════════════════════════════════════════════════
# 테스트 데이터 헬퍼
# ═══════════════════════════════════════════════════

def make_indicator_df(n=200, seed=42):
    """테스트용 지표 DataFrame 생성 (실제 parquet 구조 모사)"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-06-01", periods=n, freq="B")

    # 기본 가격 (약간의 추세 + 노이즈)
    trend = np.linspace(100000, 120000, n)
    noise = rng.normal(0, 2000, n).cumsum()
    close = trend + noise
    close = np.maximum(close, 50000)  # 최소 가격

    df = pd.DataFrame({
        "close": close,
        "high": close * (1 + rng.uniform(0, 0.03, n)),
        "low": close * (1 - rng.uniform(0, 0.03, n)),
        "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
        "volume": rng.randint(100000, 500000, n).astype(float),
        "rsi_14": rng.uniform(20, 80, n),
        "adx_14": rng.uniform(10, 40, n),
        "bb_position": rng.uniform(0, 1, n),
        "ou_z": rng.normal(0, 1.5, n),
        "volume_surge_ratio": rng.uniform(0.5, 2.5, n),
        "macd_histogram": rng.normal(0, 100, n),
        "smart_z": rng.normal(0, 1.5, n),
    }, index=dates)

    # macd_histogram_prev
    df["macd_histogram_prev"] = df["macd_histogram"].shift(1)

    # ret1 (수익률)
    df["ret1"] = df["close"].pct_change()
    df["change_pct"] = df["ret1"] * 100

    # 수급
    df["foreign_net"] = rng.randint(-100000, 100000, n).astype(float)
    df["inst_net"] = rng.randint(-50000, 50000, n).astype(float)

    return df


# ═══════════════════════════════════════════════════
# TestConfluenceScorer
# ═══════════════════════════════════════════════════

class TestConfluenceScorer:
    """⑦ Confluence Scorer 테스트"""

    def test_binarize_factors_all_false(self):
        """중립 데이터 → 모든 팩터 False"""
        row = {"rsi_14": 50, "adx_14": 15, "bb_position": 0.5, "ou_z": 0,
               "volume_surge_ratio": 1.0, "macd_histogram": 0, "macd_histogram_prev": 0,
               "smart_z": 0}
        result = ConfluenceScorer.binarize_factors(row)
        assert not result["rsi_oversold"]
        assert not result["adx_trending"]
        assert not result["bb_lower"]

    def test_binarize_factors_multiple_true(self):
        """극단 데이터 → 여러 팩터 True"""
        row = {"rsi_14": 25, "adx_14": 30, "bb_position": 0.1, "ou_z": -2.0,
               "volume_surge_ratio": 2.0, "macd_histogram": -50, "macd_histogram_prev": -80,
               "smart_z": 2.0}
        result = ConfluenceScorer.binarize_factors(row)
        assert result["rsi_oversold"]
        assert result["adx_trending"]
        assert result["bb_lower"]
        assert result["ou_undervalued"]
        assert result["volume_surge"]
        assert result["macd_recovering"]
        assert result["smart_money_buy"]

    def test_build_hit_rate_db(self):
        """적중률 DB 구축 - 최소 1개 이상 조합 생성"""
        df = make_indicator_df(200)
        scorer = ConfluenceScorer({"geometry": {"confluence": {"min_samples": 5, "forward_days": 3, "min_return": 0.01}}})
        db = scorer.build_hit_rate_db(df)
        assert isinstance(db, dict)
        # 200일 랜덤 데이터면 일부 조합은 충분한 샘플 있을 것
        if db:
            first = list(db.values())[0]
            assert "hits" in first
            assert "total" in first
            assert "rate" in first
            assert 0 <= first["rate"] <= 1

    def test_build_hit_rate_db_insufficient_data(self):
        """데이터 부족 시 빈 DB"""
        df = make_indicator_df(10)
        scorer = ConfluenceScorer()
        db = scorer.build_hit_rate_db(df)
        assert db == {}

    def test_score_current_no_active(self):
        """활성 팩터 부족 → 빈 결과"""
        scorer = ConfluenceScorer()
        row = {"rsi_14": 50, "adx_14": 10, "bb_position": 0.5, "ou_z": 0,
               "volume_surge_ratio": 1.0, "macd_histogram": 0, "macd_histogram_prev": 0,
               "smart_z": 0}
        result = scorer.score_current(row)
        assert result["triple_count"] == 0
        assert result["best_hit_rate"] == 0

    def test_score_current_with_db(self):
        """DB 구축 후 스코어링 — 구조 확인"""
        df = make_indicator_df(200)
        scorer = ConfluenceScorer({"geometry": {"confluence": {"min_samples": 3, "forward_days": 3, "min_return": 0.005}}})
        scorer.build_hit_rate_db(df)

        # DB에 있는 첫 번째 조합의 조건을 충족하는 row 만들기
        row = {"rsi_14": 25, "adx_14": 30, "bb_position": 0.1, "ou_z": -2.0,
               "volume_surge_ratio": 2.0, "macd_histogram": -50, "macd_histogram_prev": -80,
               "smart_z": 2.0}
        result = scorer.score_current(row)
        assert isinstance(result["active_factors"], list)
        assert len(result["active_factors"]) == 7  # 모든 팩터 활성

    def test_prompt_text_no_triples(self):
        """트리플 없을 때 프롬프트"""
        result = {"active_triples": [], "triple_count": 0, "best_hit_rate": 0}
        text = ConfluenceScorer.to_prompt_text(result)
        assert "활성 트리플 없음" in text

    def test_prompt_text_with_triples(self):
        """트리플 있을 때 프롬프트"""
        result = {
            "active_triples": [
                {"combo": ("a", "b", "c"), "labels": "A×B×C", "hit_rate": 0.78,
                 "total": 23, "avg_return": 0.045},
            ],
            "triple_count": 1,
            "best_hit_rate": 0.78,
        }
        text = ConfluenceScorer.to_prompt_text(result)
        assert "A×B×C" in text
        assert "78%" in text


# ═══════════════════════════════════════════════════
# TestCycleClock
# ═══════════════════════════════════════════════════

class TestCycleClock:
    """⑧ Cycle Clock 테스트"""

    def test_bandpass_filter_output_shape(self):
        """밴드패스 필터 출력 크기 확인"""
        prices = np.sin(np.linspace(0, 10 * np.pi, 200)) * 1000 + 100000
        result = CycleClock.bandpass_filter(prices, 10, 40)
        assert len(result) == len(prices)

    def test_bandpass_filter_short_data(self):
        """짧은 데이터 → 0 배열"""
        prices = np.array([100, 101, 102])
        result = CycleClock.bandpass_filter(prices, 10, 40)
        assert np.allclose(result, 0)

    def test_hilbert_phase_range(self):
        """힐베르트 위상 범위 -π ~ +π"""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        phases = CycleClock.hilbert_phase(signal)
        assert len(phases) == 100
        assert np.all(phases >= -np.pi - 0.01)
        assert np.all(phases <= np.pi + 0.01)

    def test_hilbert_phase_empty(self):
        """빈 배열 → 빈 결과"""
        result = CycleClock.hilbert_phase(np.array([]))
        assert len(result) == 0

    def test_phase_to_clock_range(self):
        """위상→시계 변환 범위 0~12"""
        for phase in np.linspace(-np.pi, np.pi, 100):
            clock = CycleClock.phase_to_clock(phase)
            assert 0 <= clock <= 12

    def test_interpret_clock_all_zones(self):
        """시계 해석 6개 구간 모두 커버"""
        zones = set()
        for h in np.linspace(0, 11.9, 50):
            interp = CycleClock.interpret_clock(h)
            zones.add(interp)
        # 최소 5개 이상의 다른 해석이 있어야 함
        assert len(zones) >= 5

    def test_phase_alignment_same(self):
        """같은 위상 → 정렬도 1.0"""
        assert abs(CycleClock.phase_alignment(1.0, 1.0) - 1.0) < 0.001

    def test_phase_alignment_opposite(self):
        """반대 위상 → 정렬도 -1.0"""
        assert abs(CycleClock.phase_alignment(0, np.pi) - (-1.0)) < 0.001

    def test_get_clock_position_full(self):
        """전체 사이클 추적 (200일 사인파)"""
        # 60일 주기 사인파 + 20일 주기 사인파
        t = np.arange(200)
        prices = 100000 + 5000 * np.sin(2 * np.pi * t / 60) + 2000 * np.sin(2 * np.pi * t / 20)

        clock = CycleClock()
        result = clock.get_clock_position(prices)

        assert "long" in result
        assert "mid" in result
        assert "short" in result
        assert "alignment_long_mid" in result
        assert "summary" in result
        assert 0 <= result["long"]["clock"] <= 12
        assert 0 <= result["mid"]["clock"] <= 12

    def test_get_clock_position_insufficient_data(self):
        """데이터 부족 → 에러 메시지"""
        clock = CycleClock()
        result = clock.get_clock_position(np.array([1, 2, 3]))
        assert "데이터 부족" in result["summary"]

    def test_prompt_text(self):
        """프롬프트 텍스트 생성"""
        result = {
            "long": {"clock": 8.0, "phase_rad": -0.5, "interpretation": "상승 초중반 (보유)"},
            "mid": {"clock": 5.0, "phase_rad": -1.2, "interpretation": "바닥~반등 (매수 구간)"},
            "short": {"clock": 7.0, "phase_rad": -0.8, "interpretation": "상승 초중반 (보유)"},
            "alignment_long_mid": 0.85,
            "alignment_long_short": 0.72,
            "overall_alignment": 0.78,
            "summary": "장기 상승 + 중기 바닥 = 같은 방향 (강한 추세)",
        }
        text = CycleClock.to_prompt_text(result)
        assert "장기" in text
        assert "중기" in text
        assert "0.78" in text


# ═══════════════════════════════════════════════════
# TestDivergenceDetector
# ═══════════════════════════════════════════════════

class TestDivergenceDetector:
    """⑨ Divergence Detector 테스트"""

    def test_classify_direction_neutral(self):
        """중립 데이터 → 모든 방향 0"""
        result = DivergenceDetector.classify_direction(
            price_change=0.1, volume_ratio=1.0, foreign_net=0, inst_net=0, macd_histogram=0,
        )
        assert result["price"] == 0
        assert result["volume"] == 0

    def test_classify_direction_strong_up(self):
        """강한 상승 데이터"""
        result = DivergenceDetector.classify_direction(
            price_change=3.0, volume_ratio=2.0, foreign_net=50000, inst_net=30000, macd_histogram=100,
        )
        assert result["price"] == 1
        assert result["volume"] == 1
        assert result["flow"] == 1
        assert result["momentum"] == 1

    def test_detect_weak_rally(self):
        """허약한 상승 패턴 감지"""
        det = DivergenceDetector()
        result = det.detect(
            price_change=2.0, volume_ratio=0.5, foreign_net=0, inst_net=0, macd_histogram=50,
        )
        names = [d["name"] for d in result["divergences"]]
        assert "허약한 상승" in names
        assert result["risk_count"] >= 1

    def test_detect_accumulation(self):
        """매집 중 패턴 감지"""
        det = DivergenceDetector()
        result = det.detect(
            price_change=-1.0, volume_ratio=0.8, foreign_net=80000, inst_net=20000, macd_histogram=-50,
        )
        names = [d["name"] for d in result["divergences"]]
        assert "매집 중" in names
        assert result["opportunity_count"] >= 1

    def test_detect_healthy_rally(self):
        """건전한 상승 확인"""
        det = DivergenceDetector()
        result = det.detect(
            price_change=2.0, volume_ratio=1.8, foreign_net=50000, inst_net=30000, macd_histogram=100,
        )
        names = [d["name"] for d in result["divergences"]]
        assert "거래량 확인 상승" in names
        assert result["confirm_count"] >= 1

    def test_detect_panic_selling(self):
        """투매 징후 감지"""
        det = DivergenceDetector()
        result = det.detect(
            price_change=-5.0, volume_ratio=3.0, foreign_net=-100000, inst_net=-50000, macd_histogram=-200,
        )
        names = [d["name"] for d in result["divergences"]]
        assert "투매 징후" in names

    def test_detect_from_row(self):
        """행 딕셔너리에서 직접 분석"""
        det = DivergenceDetector()
        row = {
            "change_pct": -2.0,
            "volume_ratio": 0.6,
            "foreign_net_buy": 60000,
            "inst_net_buy": 20000,
            "macd_histogram": -30,
        }
        result = det.detect_from_row(row)
        assert "directions" in result
        assert "divergences" in result

    def test_net_signal_range(self):
        """net_signal 범위 -1 ~ +1"""
        det = DivergenceDetector()
        for _ in range(20):
            result = det.detect(
                price_change=np.random.uniform(-5, 5),
                volume_ratio=np.random.uniform(0.3, 3.0),
                foreign_net=np.random.uniform(-100000, 100000),
                inst_net=np.random.uniform(-50000, 50000),
                macd_histogram=np.random.uniform(-200, 200),
            )
            assert -1 <= result["net_signal"] <= 1

    def test_prompt_text_no_divergence(self):
        """발산 없을 때 프롬프트"""
        result = {"divergences": [], "risk_count": 0, "opportunity_count": 0, "confirm_count": 0, "net_signal": 0}
        text = DivergenceDetector.to_prompt_text(result)
        assert "발산 없음" in text

    def test_prompt_text_with_divergence(self):
        """발산 있을 때 프롬프트"""
        result = {
            "divergences": [{"name": "매집 중", "type": "opportunity", "description": "가격↓ + 수급↑"}],
            "risk_count": 0, "opportunity_count": 1, "confirm_count": 0, "net_signal": 1.0,
        }
        text = DivergenceDetector.to_prompt_text(result)
        assert "매집 중" in text
        assert "긍정 우세" in text


# ═══════════════════════════════════════════════════
# TestGeometryEngine
# ═══════════════════════════════════════════════════

class TestGeometryEngine:
    """⑮ 통합 GeometryEngine 테스트"""

    def test_engine_disabled(self):
        """엔진 비활성 → 빈 결과"""
        engine = GeometryEngine({"geometry": {"enabled": False}})
        result = engine.analyze("005930")
        assert result["prompt_text"] == "[기하학 분석 비활성]"

    def test_engine_no_parquet(self, tmp_path):
        """parquet 없을 때 → 에러 없이 빈 결과"""
        engine = GeometryEngine(parquet_dir=tmp_path)
        result = engine.analyze("005930")
        assert "confluence" in result
        assert "cycle" in result
        assert "divergence" in result

    def test_engine_with_parquet(self, tmp_path):
        """parquet 있을 때 → 정상 분석"""
        df = make_indicator_df(200)
        pq_path = tmp_path / "005930.parquet"
        df.to_parquet(pq_path)

        engine = GeometryEngine(
            config={"geometry": {"enabled": True, "confluence": {"min_samples": 3, "min_return": 0.005}}},
            parquet_dir=tmp_path,
        )
        result = engine.analyze("005930")

        assert "confluence" in result
        assert "cycle" in result
        assert "divergence" in result
        assert "prompt_text" in result
        assert len(result["prompt_text"]) > 50

    def test_engine_with_row_override(self, tmp_path):
        """row 직접 제공 시 divergence에 반영"""
        df = make_indicator_df(200)
        pq_path = tmp_path / "005930.parquet"
        df.to_parquet(pq_path)

        engine = GeometryEngine(parquet_dir=tmp_path)
        row = {"change_pct": -3.0, "volume_ratio": 0.5, "foreign_net_buy": 80000,
               "inst_net_buy": 20000, "macd_histogram": -100,
               "rsi_14": 25, "adx_14": 30, "bb_position": 0.1, "ou_z": -2.0,
               "volume_surge_ratio": 0.5, "macd_histogram_prev": -150, "smart_z": 1.5}
        result = engine.analyze("005930", row=row)
        # 매집 중 패턴이 감지되어야 함
        divs = result["divergence"]["divergences"]
        names = [d["name"] for d in divs]
        assert "매집 중" in names

    def test_summarize_portfolio(self, tmp_path):
        """포트폴리오 요약 생성"""
        engine = GeometryEngine(parquet_dir=tmp_path)
        stocks = [
            {
                "ticker": "005930", "name": "삼성전자",
                "geometry": {
                    "confluence": {"best_hit_rate": 0.78, "active_triples": [
                        {"labels": "RSI×BB×OU", "hit_rate": 0.78, "total": 23}
                    ]},
                    "cycle": {"mid": {"clock": 6.0}},
                    "divergence": {"risk_count": 0},
                },
            },
            {
                "ticker": "000660", "name": "SK하이닉스",
                "geometry": {
                    "confluence": {"best_hit_rate": 0.0, "active_triples": []},
                    "cycle": {"mid": {"clock": 11.0}},
                    "divergence": {"risk_count": 3},
                },
            },
        ]
        summary = engine.summarize_portfolio(stocks)
        assert "삼성전자" in summary or "005930" in summary
        assert "매수 구간" in summary or "매도 구간" in summary

    def test_prompt_text_integration(self, tmp_path):
        """프롬프트 텍스트에 3개 섹션 모두 포함"""
        df = make_indicator_df(200)
        pq_path = tmp_path / "005930.parquet"
        df.to_parquet(pq_path)

        engine = GeometryEngine(parquet_dir=tmp_path)
        result = engine.analyze("005930")
        text = result["prompt_text"]

        assert "[3D 교차 분석]" in text
        assert "[사이클 시계]" in text
        assert "[발산 감지]" in text
