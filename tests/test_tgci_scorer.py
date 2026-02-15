"""TGCI (TRIX Golden Cross Indicator) 독립 스코어러 단위 테스트."""
from src.tgci_scorer import TGCIScorer, _score_to_grade


def _base_row(**overrides):
    """기본 row dict 생성. 골든크로스=X, RSI=50, 거래량 보통."""
    row = {
        "trix": 0.05,
        "trix_signal": 0.06,
        "trix_golden_cross": 0,
        "rsi_14": 50.0,
        "volume": 200_000,
        "volume_ma20": 200_000,
        "obv": 1_000_000,
        "obv_prev": 1_000_000,
        "bb_position": 0.5,
        "macd_histogram": 0.1,
        "macd_histogram_prev": 0.05,
    }
    row.update(overrides)
    return row


# ─── 골든크로스 점수 ───

class TestTrixCross:
    def test_golden_cross_gives_35(self):
        row = _base_row(trix_golden_cross=1, trix=0.10, trix_signal=0.05)
        result = TGCIScorer.score(row)
        assert result["details"]["trix_cross"] is True
        # 35(cross) + slope bonus + 25(RSI 40-60) + 10(vol 1x=10) + ...
        assert result["score"] >= 35

    def test_no_cross_no_points(self):
        row = _base_row(trix_golden_cross=0)
        result = TGCIScorer.score(row)
        assert result["details"]["trix_cross"] is False
        # RSI 50 = 25pts, vol 1x = 10pts, no OBV rise → max ~35
        assert result["score"] < 40

    def test_slope_bonus_capped_at_8(self):
        row = _base_row(trix_golden_cross=1, trix=1.0, trix_signal=0.0)
        result = TGCIScorer.score(row)
        assert result["details"]["trix_slope_bonus"] <= 8


# ─── RSI 점수 ───

class TestRSI:
    def test_optimal_range_25pts(self):
        row = _base_row(rsi_14=50.0)
        result = TGCIScorer.score(row)
        assert result["details"]["rsi_pts"] == 25

    def test_near_range_15pts(self):
        row = _base_row(rsi_14=35.0)
        result = TGCIScorer.score(row)
        assert result["details"]["rsi_pts"] == 15

    def test_extreme_0pts(self):
        row = _base_row(rsi_14=80.0)
        result = TGCIScorer.score(row)
        assert result["details"]["rsi_pts"] == 0

    def test_rsi_boundary_low(self):
        row = _base_row(rsi_14=40.0)
        result = TGCIScorer.score(row)
        assert result["details"]["rsi_pts"] == 25

    def test_rsi_boundary_high(self):
        row = _base_row(rsi_14=60.0)
        result = TGCIScorer.score(row)
        assert result["details"]["rsi_pts"] == 25


# ─── 거래량 점수 ───

class TestVolume:
    def test_explosion_full_25(self):
        row = _base_row(volume=500_000, volume_ma20=200_000)
        result = TGCIScorer.score(row)
        assert result["details"]["vol_pts"] == 25

    def test_normal_vol_10(self):
        row = _base_row(volume=200_000, volume_ma20=200_000)
        result = TGCIScorer.score(row)
        assert result["details"]["vol_pts"] == 10

    def test_zero_vol_0(self):
        row = _base_row(volume=0, volume_ma20=200_000)
        result = TGCIScorer.score(row)
        assert result["details"]["vol_pts"] == 0


# ─── OBV ───

class TestOBV:
    def test_obv_rising_10pts(self):
        row = _base_row(obv=1_100_000, obv_prev=1_000_000)
        result = TGCIScorer.score(row)
        assert result["details"]["obv_rising"] is True

    def test_obv_flat_0pts(self):
        row = _base_row(obv=1_000_000, obv_prev=1_000_000)
        result = TGCIScorer.score(row)
        assert result["details"]["obv_rising"] is False


# ─── 방어 감점 ───

class TestPenalties:
    def test_bb_overheat_minus10(self):
        row = _base_row(bb_position=0.98)
        result = TGCIScorer.score(row)
        assert result["details"]["bb_penalty"] == -10

    def test_bb_normal_no_penalty(self):
        row = _base_row(bb_position=0.50)
        result = TGCIScorer.score(row)
        assert result["details"]["bb_penalty"] == 0

    def test_macd_weakening_minus8(self):
        row = _base_row(macd_histogram=-0.1, macd_histogram_prev=0.1)
        result = TGCIScorer.score(row)
        assert result["details"]["macd_penalty"] == -8

    def test_macd_normal_no_penalty(self):
        row = _base_row(macd_histogram=0.1, macd_histogram_prev=0.05)
        result = TGCIScorer.score(row)
        assert result["details"]["macd_penalty"] == 0


# ─── 종합 ───

class TestOverall:
    def test_full_score_above_85(self):
        """골든크로스 + RSI 최적 + 거래량 3x + OBV 상승 → S등급"""
        row = _base_row(
            trix_golden_cross=1, trix=0.10, trix_signal=0.05,
            rsi_14=50.0,
            volume=600_000, volume_ma20=200_000,
            obv=1_100_000, obv_prev=1_000_000,
        )
        result = TGCIScorer.score(row)
        assert result["score"] >= 85
        assert result["grade"] == "S"

    def test_score_clamped_0_100(self):
        # 최악의 경우: 모든 감점 + 0점
        row = _base_row(
            trix_golden_cross=0, rsi_14=80.0,
            volume=0, volume_ma20=200_000,
            obv=900_000, obv_prev=1_000_000,
            bb_position=0.99, macd_histogram=-0.1, macd_histogram_prev=0.1,
        )
        result = TGCIScorer.score(row)
        assert 0 <= result["score"] <= 100

    def test_grade_mapping(self):
        assert _score_to_grade(90) == "S"
        assert _score_to_grade(85) == "S"
        assert _score_to_grade(75) == "A"
        assert _score_to_grade(70) == "A"
        assert _score_to_grade(60) == "B"
        assert _score_to_grade(55) == "B"
        assert _score_to_grade(45) == "C"
        assert _score_to_grade(40) == "C"
        assert _score_to_grade(30) == "D"
        assert _score_to_grade(0) == "D"

    def test_result_structure(self):
        row = _base_row()
        result = TGCIScorer.score(row)
        assert "score" in result
        assert "grade" in result
        assert "details" in result
        assert isinstance(result["score"], int)
        assert result["grade"] in ("S", "A", "B", "C", "D")

    def test_nan_handling(self):
        row = _base_row(rsi_14=float("nan"), volume=float("nan"))
        result = TGCIScorer.score(row)
        assert 0 <= result["score"] <= 100

    def test_config_override(self):
        """사용자 config로 골든크로스 배점 변경"""
        row = _base_row(trix_golden_cross=1, trix=0.10, trix_signal=0.05)
        cfg = {"golden_cross_pts": 50}
        result = TGCIScorer.score(row, config=cfg)
        assert result["score"] >= 50
