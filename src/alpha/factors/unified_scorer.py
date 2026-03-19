"""Alpha Engine V2 — 통합 4팩터 스코어러 (STEP 6-1)

기존 5축(S1~S5)을 4개 매크로 팩터로 재편:
  SD (Supply/Demand) = S5 (수급)
  M  (Momentum)      = S4(0.45) + S3(0.30) + S1(0.25)
  V  (Value)         = ValueComposite(S2 + EBITDA/EV + FCF Yield)
  Q  (Quality)       = QualityComposite(ROE + Debt + Accruals + Dividend)

최종: total = SD×w_sd + M×w_m + V×w_v + Q×w_q (레짐별 w)

settings.yaml factor_weights:
  bull:    { sd: 0.30, momentum: 0.35, value: 0.15, quality: 0.20 }
  caution: { sd: 0.25, momentum: 0.20, value: 0.25, quality: 0.30 }
  bear:    { sd: 0.15, momentum: 0.10, value: 0.40, quality: 0.35 }
  crisis:  { sd: 0.00, momentum: 0.00, value: 0.00, quality: 0.00 }
"""

from __future__ import annotations

import logging

import pandas as pd

from src.alpha.models import AlphaRegimeLevel
from src.alpha.factors.quality_composite import QualityComposite
from src.alpha.factors.value_composite import ValueComposite
from src.v8_scorers import GradeResult, ScoreResult, ScoringEngine

logger = logging.getLogger(__name__)

# M (Momentum) 내부 서브팩터 가중치 (STEP 1 Sharpe 비율 기반)
# S4(0.53) > S3(0.47) > S1(0.29) → 정규화 후 조정
_M_SUB_WEIGHTS = {"S4": 0.45, "S3": 0.30, "S1": 0.25}

# 4팩터 기본 가중치 (settings.yaml factor_weights 없을 때 폴백)
_DEFAULT_FACTOR_WEIGHTS = {
    "BULL":    {"sd": 0.30, "momentum": 0.35, "value": 0.15, "quality": 0.20},
    "CAUTION": {"sd": 0.25, "momentum": 0.20, "value": 0.25, "quality": 0.30},
    "BEAR":    {"sd": 0.15, "momentum": 0.10, "value": 0.40, "quality": 0.35},
    "CRISIS":  {"sd": 0.00, "momentum": 0.00, "value": 0.00, "quality": 0.00},
}


class UnifiedV2Scorer:
    """4팩터 통합 스코어러: SD + M + V + Q"""

    def __init__(self, config: dict):
        self._config = config
        self._scorer = ScoringEngine(config)

        # 4팩터 가중치 로드
        v2_cfg = config.get("alpha_v2", {})
        self._factor_weights: dict[str, dict[str, float]] = {}
        fw_cfg = v2_cfg.get("factor_weights", {})
        for regime in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
            regime_lower = regime.lower()
            if regime_lower in fw_cfg:
                self._factor_weights[regime] = fw_cfg[regime_lower]
            else:
                self._factor_weights[regime] = _DEFAULT_FACTOR_WEIGHTS[regime]

        # 등급 커트라인
        v8_cfg = config.get("v8_hybrid", {})
        scoring_cfg = v8_cfg.get("scoring", {})
        self._cutoffs = scoring_cfg.get("grade_cutoffs", {
            "A": 0.65, "B": 0.50, "C": 0.35,
        })

        # Q/V 컴포지트 (lazy init — 첫 score() 호출 시 로드)
        self._quality: QualityComposite | None = None
        self._value: ValueComposite | None = None

    def _ensure_composites(self):
        """Q/V 컴포지트 lazy 초기화."""
        if self._quality is None:
            self._quality = QualityComposite()
        if self._value is None:
            self._value = ValueComposite()

    def score(
        self,
        row: pd.Series,
        ticker: str,
        regime_level: AlphaRegimeLevel,
    ) -> GradeResult:
        """4팩터 통합 스코어링.

        Args:
            row: parquet DataFrame의 한 행 (S1~S5 계산용)
            ticker: 종목코드 (Q/V 조회용)
            regime_level: 현재 시장 레짐

        Returns:
            GradeResult (기존 인터페이스 100% 호환)
        """
        self._ensure_composites()
        regime_str = regime_level.value

        # 1. 기존 5축 개별 계산
        s1 = self._scorer.score_energy_depletion(row)
        s2 = self._scorer.score_valuation(row)
        s3 = self._scorer.score_ou_reversion(row)
        s4 = self._scorer.score_momentum_deceleration(row)
        s5 = self._scorer.score_smart_money(row)

        # 2. 4팩터 합성
        sd_score = s5.score

        m_score = (
            s4.score * _M_SUB_WEIGHTS["S4"]
            + s3.score * _M_SUB_WEIGHTS["S3"]
            + s1.score * _M_SUB_WEIGHTS["S1"]
        )

        v_score = self._value.score(ticker, regime_str, s2_score=s2.score)
        q_score = self._quality.score(ticker, regime_str)

        # 3. 레짐별 가중합
        weights = self._factor_weights.get(
            regime_str, _DEFAULT_FACTOR_WEIGHTS["CAUTION"]
        )
        total = (
            sd_score * weights["sd"]
            + m_score * weights["momentum"]
            + v_score * weights["value"]
            + q_score * weights["quality"]
        )

        # 4. ScoreResult 리스트 (디버깅/표시용)
        scores = [
            ScoreResult(
                name="SD_수급",
                score=sd_score,
                weight=weights["sd"],
                breakdown={"S5": round(s5.score, 4)},
            ),
            ScoreResult(
                name="M_모멘텀",
                score=round(m_score, 4),
                weight=weights["momentum"],
                breakdown={
                    "S4": round(s4.score, 4),
                    "S3": round(s3.score, 4),
                    "S1": round(s1.score, 4),
                },
            ),
            ScoreResult(
                name="V_밸류",
                score=round(v_score, 4),
                weight=weights["value"],
                breakdown={"S2": round(s2.score, 4)},
            ),
            ScoreResult(
                name="Q_퀄리티",
                score=round(q_score, 4),
                weight=weights["quality"],
            ),
        ]

        # 5. 등급 결정
        grade = self._determine_grade(total)

        pos_cfg = self._config.get("v8_hybrid", {}).get("position", {})
        pos_pct = 0.0
        tradeable = False
        if grade == "A":
            pos_pct = pos_cfg.get("A_grade_pct", 0.20)
            tradeable = True
        elif grade == "B":
            pos_pct = pos_cfg.get("B_grade_pct", 0.10)
            tradeable = True

        return GradeResult(
            total_score=round(total, 4),
            grade=grade,
            scores=scores,
            position_size_pct=pos_pct,
            tradeable=tradeable,
        )

    def rescore_signals(
        self,
        signals: list[dict],
        data_dict: dict,
        idx: int,
        regime_level: AlphaRegimeLevel,
    ) -> list[dict]:
        """SignalEngine 출력을 V2 스코어로 재채점.

        백테스트/스캔에서 SignalEngine 결과를 받아 V2로 재채점할 때 사용.
        게이트/트리거는 유지하고 V2 스코어로 포지션 사이징만 조정.
        V1이 통과시킨 시그널은 제거하지 않음 (sizing-only mode).

        포지션 비율 조정:
          A등급 → 1.0× (100%)
          B등급 → 0.8× (80%)
          C등급 → 0.5× (50% — 방어적 감축)
          F등급 → 0.3× (30% — 최소 배분)

        Args:
            signals: SignalEngine.scan_universe() 반환 리스트
            data_dict: {ticker: DataFrame} 전체 데이터
            idx: 현재 날짜 인덱스
            regime_level: 현재 레짐

        Returns:
            재채점된 signal 리스트 (원본 시그널 100% 유지, 사이징만 조정)
        """
        _GRADE_RATIO = {"A": 1.0, "B": 0.8, "C": 0.5, "F": 0.3}

        rescored = []
        for sig in signals:
            ticker = sig["ticker"]
            if ticker not in data_dict:
                rescored.append(sig)
                continue

            df = data_dict[ticker]
            if idx >= len(df):
                rescored.append(sig)
                continue

            row = df.iloc[idx]
            result = self.score(row, ticker, regime_level)

            # 기존 signal dict 업데이트 (gate/trigger 정보는 유지)
            sig = dict(sig)  # shallow copy
            sig["v2_total"] = result.total_score
            sig["v2_grade"] = result.grade

            # V2 등급 기반 포지션 비율 조정 (필터링 아님)
            v2_ratio = _GRADE_RATIO.get(result.grade, 0.5)
            orig_ratio = sig.get("position_ratio", 1.0)
            sig["position_ratio"] = round(orig_ratio * v2_ratio, 3)

            sig["v2_scores"] = {
                s.name: {"score": s.score, "weight": s.weight, "weighted": s.weighted}
                for s in result.scores
            }
            rescored.append(sig)

        return rescored

    def _determine_grade(self, total: float) -> str:
        if total >= self._cutoffs.get("A", 0.65):
            return "A"
        elif total >= self._cutoffs.get("B", 0.50):
            return "B"
        elif total >= self._cutoffs.get("C", 0.35):
            return "C"
        else:
            return "F"

    def get_weights(self, regime_level: AlphaRegimeLevel) -> dict:
        """현재 레짐의 4팩터 가중치 반환."""
        return self._factor_weights.get(
            regime_level.value, _DEFAULT_FACTOR_WEIGHTS["CAUTION"]
        )
