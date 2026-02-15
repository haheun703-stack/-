"""Master Controller 단위 테스트."""
import pytest
from src.master_controller import MasterController, Action, EntryMode, DEFAULT_WEIGHTS


def _all_scores(val):
    """모든 서브시스템에 동일 점수."""
    return {k: val for k in DEFAULT_WEIGHTS}


class TestWeights:
    def test_weight_sum_is_1(self):
        mc = MasterController()
        assert abs(sum(mc.weights.values()) - 1.0) < 1e-6

    def test_all_100_gives_100(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(100))
        assert result["master_score"] == 100

    def test_all_0_gives_0(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(0))
        assert result["master_score"] == 0

    def test_missing_key_treated_as_0(self):
        mc = MasterController()
        result = mc.evaluate({"pipeline": 80})
        # 80 * 0.40 = 32
        assert result["master_score"] == 32


class TestStrongBuy:
    def test_all_high_is_strong_buy(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(80))
        assert result["action"] == Action.STRONG_BUY
        assert result["entry_mode"] == EntryMode.CONSERVATIVE
        assert result["contributing_systems"] == 5

    def test_high_score_but_few_contributors(self):
        """점수는 높지만 contributing < 4 → STRONG_BUY 아님."""
        mc = MasterController()
        scores = {"pipeline": 100, "tgci": 100, "smart_money": 0, "regime": 0, "geometric": 0}
        result = mc.evaluate(scores)
        # 100*0.4 + 100*0.2 = 60, contributing=2
        assert result["action"] != Action.STRONG_BUY


class TestConservativeBuy:
    def test_three_systems_high(self):
        mc = MasterController()
        scores = {"pipeline": 80, "tgci": 80, "smart_money": 80, "regime": 30, "geometric": 30}
        result = mc.evaluate(scores)
        # 80*0.4 + 80*0.2 + 80*0.15 + 30*0.15 + 30*0.10 = 32+16+12+4.5+3 = 67.5
        assert result["action"] == Action.BUY
        assert result["entry_mode"] == EntryMode.CONSERVATIVE


class TestAggressiveBuy:
    def test_two_systems_with_tgci(self):
        mc = MasterController()
        scores = {"pipeline": 80, "tgci": 60, "smart_money": 20, "regime": 20, "geometric": 20}
        result = mc.evaluate(scores)
        # 80*0.4 + 60*0.2 + 20*0.15 + 20*0.15 + 20*0.10 = 32+12+3+3+2 = 52
        # contributing: pipeline(80>50), tgci(60>50) = 2, tgci_above_50=True
        assert result["action"] == Action.BUY
        assert result["entry_mode"] == EntryMode.AGGRESSIVE

    def test_tgci_required_for_aggressive(self):
        """TGCI < 50이면 aggressive 불가."""
        mc = MasterController()
        scores = {"pipeline": 80, "tgci": 40, "smart_money": 60, "regime": 20, "geometric": 20}
        result = mc.evaluate(scores)
        # pipeline(80>50), smart_money(60>50) = 2 contributing, tgci_above_50=False
        # score = 32 + 8 + 9 + 3 + 2 = 54 → >= 50 but no tgci → not aggressive
        assert result["entry_mode"] != EntryMode.AGGRESSIVE


class TestWatch:
    def test_moderate_score_is_watch(self):
        mc = MasterController()
        scores = {"pipeline": 60, "tgci": 30, "smart_money": 30, "regime": 30, "geometric": 30}
        result = mc.evaluate(scores)
        # 60*0.4 + 30*0.2 + 30*0.15 + 30*0.15 + 30*0.10 = 24+6+4.5+4.5+3 = 42
        assert result["action"] == Action.WATCH
        assert result["entry_mode"] is None


class TestHold:
    def test_low_score_is_hold(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(20))
        # 20*1.0 = 20
        assert result["action"] == Action.HOLD
        assert result["entry_mode"] is None

    def test_all_zero_is_hold(self):
        mc = MasterController()
        result = mc.evaluate({})
        assert result["action"] == Action.HOLD
        assert result["master_score"] == 0


class TestContributing:
    def test_count_above_50(self):
        mc = MasterController()
        scores = {"pipeline": 51, "tgci": 51, "smart_money": 49, "regime": 51, "geometric": 50}
        result = mc.evaluate(scores)
        # pipeline(51>50), tgci(51>50), regime(51>50) = 3
        assert result["contributing_systems"] == 3

    def test_boundary_50_not_counted(self):
        mc = MasterController()
        scores = {"pipeline": 50, "tgci": 50, "smart_money": 50, "regime": 50, "geometric": 50}
        result = mc.evaluate(scores)
        assert result["contributing_systems"] == 0


class TestResultStructure:
    def test_all_fields_present(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(50))
        assert "master_score" in result
        assert "action" in result
        assert "entry_mode" in result
        assert "contributing_systems" in result
        assert "weighted_breakdown" in result

    def test_weighted_breakdown_keys(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(50))
        for key in DEFAULT_WEIGHTS:
            assert key in result["weighted_breakdown"]

    def test_score_clamped(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(150))  # 초과 입력
        assert result["master_score"] <= 100

    def test_negative_clamped(self):
        mc = MasterController()
        result = mc.evaluate(_all_scores(-50))
        assert result["master_score"] >= 0


class TestCustomConfig:
    def test_custom_thresholds(self):
        cfg = {
            "master_controller": {
                "thresholds": {
                    "strong_buy": 90,
                    "buy_conservative": 80,
                    "buy_aggressive": 60,
                    "watch": 50,
                },
            }
        }
        mc = MasterController(cfg)
        result = mc.evaluate(_all_scores(75))
        # score=75, contributing=5 → under strong_buy(90), under buy_conservative(80)
        # → buy_aggressive(60)? contributing>=2, tgci>50 → BUY aggressive
        assert result["action"] == Action.BUY
        assert result["entry_mode"] == EntryMode.AGGRESSIVE
