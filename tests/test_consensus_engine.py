"""
v5.0 ConsensusVerifier 단위 테스트

테스트 항목:
  1. 전부 통과 → consistency = 1.0
  2. 60% 통과 → consistency = 0.75 (tau=0.8)
  3. C=1.0, R=0.8, D=0.7 → strong 등급
  4. DB 없으면 confidence 평균 fallback
  5. consensus_mode=False → 기존 결과에 영향 없음
  6. 빈 투표 → 기본값 반환
  7. 등급 경계값 테스트
"""

import pytest

from src.entities.consensus_models import (
    AnchorDatabase,
    ConsensusResult,
    LayerVote,
)
from src.use_cases.consensus_engine import ConsensusVerifier


# ── 헬퍼 ──

def make_votes(passed_list: list, confidence_list: list = None) -> list:
    """테스트용 LayerVote 리스트 생성"""
    # v6.0: L-1_rttp, L1_extreme_vol, L3_martin 추가
    names = [
        "L-1_rttp", "L0_grade", "L1_regime", "L1_extreme_vol",
        "L2_ou", "L3_momentum", "L3_martin", "L4_smart_money",
        "L4.5_probability", "L6_trigger", "L5_risk", "L7_geometric",
    ]
    if confidence_list is None:
        confidence_list = [0.7] * len(passed_list)
    votes = []
    for i, (passed, conf) in enumerate(zip(passed_list, confidence_list)):
        name = names[i] if i < len(names) else f"L{i}"
        votes.append(LayerVote(
            layer_name=name,
            passed=passed,
            confidence=conf,
        ))
    return votes


# ── Consistency 테스트 ──

class TestConsistency:
    def test_all_pass(self):
        """전부 통과 → consistency = 1.0"""
        verifier = ConsensusVerifier(tau=0.8)
        votes = make_votes([True] * 9)
        c = verifier.calc_consistency(votes)
        assert c == 1.0

    def test_partial_pass_below_tau(self):
        """60% 통과 (5/9 ≈ 0.556) → 0.556/0.8 = 0.694"""
        verifier = ConsensusVerifier(tau=0.8)
        passed = [True, True, True, True, True, False, False, False, False]
        votes = make_votes(passed)
        c = verifier.calc_consistency(votes)
        expected = (5 / 9) / 0.8
        assert abs(c - expected) < 0.01

    def test_above_tau_capped(self):
        """통과율 >= tau → capped at 1.0"""
        verifier = ConsensusVerifier(tau=0.8)
        passed = [True] * 8 + [False]  # 8/9 ≈ 0.889 > 0.8
        votes = make_votes(passed)
        c = verifier.calc_consistency(votes)
        assert c == 1.0

    def test_none_pass(self):
        """전부 실패 → 0.0"""
        verifier = ConsensusVerifier(tau=0.8)
        votes = make_votes([False] * 5)
        c = verifier.calc_consistency(votes)
        assert c == 0.0

    def test_empty_votes(self):
        """빈 투표 → 0.0"""
        verifier = ConsensusVerifier()
        assert verifier.calc_consistency([]) == 0.0


# ── Reliability 테스트 ──

class TestReliability:
    def test_no_anchor_db_fallback(self):
        """앵커 DB 없으면 confidence 평균"""
        verifier = ConsensusVerifier(anchor_db=None)
        confs = [0.8, 0.6, 0.9, 0.7, 0.5]
        votes = make_votes([True] * 5, confs)
        r = verifier.calc_reliability(votes)
        expected = sum(confs) / len(confs)
        assert abs(r - expected) < 0.01

    def test_no_anchor_only_passed(self):
        """실패한 레이어의 confidence는 제외"""
        verifier = ConsensusVerifier(anchor_db=None)
        passed = [True, True, False, True, False]
        confs = [0.8, 0.6, 0.9, 0.7, 0.5]
        votes = make_votes(passed, confs)
        r = verifier.calc_reliability(votes)
        expected = (0.8 + 0.6 + 0.7) / 3
        assert abs(r - expected) < 0.01

    def test_anchor_db_exact_match(self):
        """특성이 중심점과 동일 → reliability = 1.0"""
        db = AnchorDatabase(
            success_centroid={"L0_grade": 0.8, "L1_regime": 0.7},
        )
        verifier = ConsensusVerifier(anchor_db=db)
        votes = [
            LayerVote("L0_grade", True, 0.8),
            LayerVote("L1_regime", True, 0.7),
        ]
        r = verifier.calc_reliability(votes)
        assert abs(r - 1.0) < 0.01

    def test_anchor_db_far_from_centroid(self):
        """중심점에서 먼 경우 → reliability < 0.5"""
        db = AnchorDatabase(
            success_centroid={"L0_grade": 0.9, "L1_regime": 0.9},
        )
        verifier = ConsensusVerifier(anchor_db=db)
        votes = [
            LayerVote("L0_grade", True, 0.1),
            LayerVote("L1_regime", True, 0.1),
        ]
        r = verifier.calc_reliability(votes)
        assert r < 0.5

    def test_no_passed_votes(self):
        """통과한 레이어 없음 → 0.0"""
        verifier = ConsensusVerifier()
        votes = make_votes([False] * 3, [0.5, 0.5, 0.5])
        r = verifier.calc_reliability(votes)
        assert r == 0.0


# ── Diversity 테스트 ──

class TestDiversity:
    def test_no_indicators_default(self):
        """지표 없으면 중립 0.5 (indicator 파트)"""
        verifier = ConsensusVerifier()
        votes = make_votes([True] * 5)
        d = verifier.calc_diversity(votes, None)
        # 0.6 * 0.5 (indicator default) + 0.4 * layer_div
        assert 0.0 <= d <= 1.0

    def test_all_same_signal_low_diversity(self):
        """모든 지표가 BUY → 낮은 다양성"""
        verifier = ConsensusVerifier()
        geo = {f"ind_{i}": {"signal": "BUY", "score": 80} for i in range(10)}
        votes = make_votes([True] * 5)
        d = verifier.calc_diversity(votes, geo)
        # 모두 동일 신호 → entropy ≈ 0
        assert d < 0.5

    def test_balanced_signals_high_diversity(self):
        """BUY/SELL/HOLD 균형 → 높은 다양성"""
        verifier = ConsensusVerifier()
        geo = {}
        for i in range(3):
            geo[f"buy_{i}"] = {"signal": "BUY", "score": 60}
        for i in range(3):
            geo[f"sell_{i}"] = {"signal": "SELL", "score": 60}
        for i in range(4):
            geo[f"hold_{i}"] = {"signal": "HOLD", "score": 30}
        votes = make_votes([True] * 12)  # v6.0: 8개 카테고리 커버
        d = verifier.calc_diversity(votes, geo)
        assert d > 0.5


# ── Geometric Reward 테스트 ──

class TestGeometricReward:
    def test_strong_grade(self):
        """C=1.0, R=0.8, D=0.7 → strong"""
        verifier = ConsensusVerifier()
        reward = verifier.calc_geometric_reward(1.0, 0.8, 0.7)
        # 1.0 * 0.8 * 0.7^(1/3) ≈ 0.8 * 0.8879 ≈ 0.710
        assert reward >= 0.6
        grade = verifier._determine_grade(reward)
        assert grade == "strong"

    def test_moderate_grade(self):
        """중간 수준 → moderate"""
        verifier = ConsensusVerifier()
        # 0.8 * 0.7 * 0.8^(1/3) ≈ 0.56 * 0.928 ≈ 0.52 → moderate
        reward = verifier.calc_geometric_reward(0.8, 0.7, 0.8)
        grade = verifier._determine_grade(reward)
        assert grade in ("moderate", "strong")

    def test_reject_when_zero(self):
        """한 축이 0이면 → reject"""
        verifier = ConsensusVerifier()
        reward = verifier.calc_geometric_reward(0.0, 0.8, 0.7)
        assert reward == 0.0
        grade = verifier._determine_grade(reward)
        assert grade == "reject"

    def test_reject_low_scores(self):
        """모두 낮으면 → reject"""
        verifier = ConsensusVerifier()
        reward = verifier.calc_geometric_reward(0.2, 0.2, 0.2)
        grade = verifier._determine_grade(reward)
        assert grade == "reject"


# ── 통합 verify() 테스트 ──

class TestVerifyIntegration:
    def test_full_pipeline_strong(self):
        """모든 레이어 통과 + 높은 confidence → strong"""
        verifier = ConsensusVerifier(tau=0.8)
        # v6.0: 12개 레이어 (L-1_rttp ~ L7_geometric)
        votes = make_votes(
            [True] * 12,
            [0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.75, 0.6, 0.75, 0.85, 0.7, 0.65],
        )
        geo = {
            "harmonic": {"signal": "BUY", "score": 80},
            "elliott": {"signal": "BUY", "score": 70},
            "slope": {"signal": "HOLD", "score": 40},
            "squeeze": {"signal": "BUY", "score": 90},
            "curvature": {"signal": "BUY", "score": 60},
            "slope_mom": {"signal": "HOLD", "score": 30},
            "confluence": {"signal": "BUY", "score": 50},
            "mean_rev": {"signal": "SELL", "score": 40},
            "vol_climax": {"signal": "HOLD", "score": 20},
            "band_breach": {"signal": "BUY", "score": 55},
        }
        result = verifier.verify(votes, geo)
        assert isinstance(result, ConsensusResult)
        assert result.consistency == 1.0
        assert result.reliability > 0.5
        assert result.diversity > 0.3
        assert result.geometric_reward > 0.4
        assert result.consensus_grade in ("strong", "moderate")
        assert result.passed_voters == 12
        assert result.total_voters == 12

    def test_empty_votes_returns_default(self):
        """빈 투표 → 기본값"""
        verifier = ConsensusVerifier()
        result = verifier.verify([])
        assert result.consistency == 0.0
        assert result.geometric_reward == 0.0
        assert result.consensus_grade == "reject"

    def test_partial_failure_weak(self):
        """많은 레이어 실패 → weak/reject"""
        verifier = ConsensusVerifier(tau=0.8)
        votes = make_votes(
            [True, True, False, False, False, False, False, False, False],
            [0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        result = verifier.verify(votes)
        assert result.consensus_grade in ("weak", "reject")
        assert result.passed_voters == 2
