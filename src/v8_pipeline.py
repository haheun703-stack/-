"""
Quantum Master v8.0 — Main Pipeline
"포물선 초점에 진입하여 추세를 먹는다"

Architecture: Gate + Score Hybrid
  Phase 1: Hard Gates (AND) -> 자격 없는 종목 빠른 제거
  Phase 2: Scoring (가중합) -> 초점 근접도 측정 -> 등급 결정
  Phase 3: Triggers (OR) -> B등급+ 종목 진입 타이밍

v7.0 -> v8.0 핵심 변경:
  - L2 OU: Gate -> Score (98% 차단 문제 해결)
  - L3 Momentum: martin_dead_zone 제거 -> 곡률 전환 감지
  - C등급: 진입 불가 (워치리스트만)
"""

import logging

import pandas as pd

from .v8_gates import GateEngine
from .v8_scorers import ScoringEngine
from .v8_triggers import TriggerEngine

logger = logging.getLogger(__name__)


class QuantumPipelineV8:
    """
    Quantum Master v8.0 Main Pipeline

    Usage:
        pipeline = QuantumPipelineV8(config)
        signal = pipeline.scan_single(row, ticker='005930', date='2024-06-15')
    """

    def __init__(self, config: dict):
        self.config = config
        self.gate_engine = GateEngine(config)
        self.scoring_engine = ScoringEngine(config)
        self.trigger_engine = TriggerEngine(config)

        v8_cfg = config.get('v8_hybrid', {})
        pos_cfg = v8_cfg.get('position', {})
        self.stop_loss_atr = pos_cfg.get('stop_loss_atr', 2.0)
        self.target_atr_A = pos_cfg.get('target_atr_A', 5.0)
        self.target_atr_B = pos_cfg.get('target_atr_B', 3.5)
        self.min_rr_ratio = pos_cfg.get('min_rr_ratio', 1.5)

    def scan_single(self, row: pd.Series, ticker: str = "", date: str = "") -> dict:
        """
        단일 종목을 파이프라인에 통과시킵니다.

        Returns:
            backtest_engine.py 호환 dict:
            - signal=True/False
            - zone_score, grade, trigger_type, entry_price 등
        """
        ticker = ticker or str(row.get('ticker', 'UNKNOWN'))
        close = row.get('close', 0)
        atr = row.get('atr_14', row.get('atr', 1))
        if atr <= 0:
            atr = 1

        # 기본 결과 (signal=False)
        result = {
            "ticker": ticker,
            "date": date,
            "signal": False,
            "zone_score": 0.0,
            "bes_score": 0.0,
            "grade": "F",
            "trigger_type": "none",
            "trigger_confidence": 0.0,
            "entry_price": int(close),
            "stop_loss": int(close - atr * self.stop_loss_atr),
            "target_price": int(close + atr * self.target_atr_B),
            "risk_reward_ratio": 0.0,
            "atr_value": round(float(atr), 1),
            "position_ratio": 0.0,
            "entry_stage_pct": 1.0,
            "stop_loss_pct": 0.0,
            "v8_action": "SKIP",
        }

        # ═══ Phase 1: Hard Gates ═══
        gate_passed, gate_results = self.gate_engine.run_all_gates(row)
        result["v8_gate_results"] = [
            {"name": g.gate_name, "passed": g.passed, "reason": g.reason}
            for g in gate_results
        ]

        if not gate_passed:
            result["v8_action"] = "SKIP"
            result["v8_skip_reason"] = gate_results[-1].reason
            return result

        # ═══ Phase 2: Scoring ═══
        grade_result = self.scoring_engine.score_all(row)
        result["zone_score"] = round(grade_result.total_score, 4)
        result["bes_score"] = result["zone_score"]
        result["grade"] = grade_result.grade
        result["v8_score_details"] = [
            {
                "name": s.name,
                "score": round(s.score, 3),
                "weight": s.weight,
                "weighted": round(s.weighted, 4),
                "breakdown": s.breakdown,
            }
            for s in grade_result.scores
        ]

        if not grade_result.tradeable:
            result["v8_action"] = "WATCH" if grade_result.grade == "C" else "SKIP"
            return result

        # 등급별 포지션 비율
        position_ratio = 1.0 if grade_result.grade == "A" else 0.5
        result["position_ratio"] = position_ratio

        # 등급별 타겟 설정
        target_atr_mult = self.target_atr_A if grade_result.grade == "A" else self.target_atr_B
        stop_price = close - atr * self.stop_loss_atr
        target_price = close + atr * target_atr_mult

        # RR 비율 체크
        risk = close - stop_price
        if risk <= 0:
            risk = 1
        rr_ratio = (target_price - close) / risk

        if rr_ratio < self.min_rr_ratio:
            result["v8_action"] = "WATCH"
            result["v8_skip_reason"] = f"RR {rr_ratio:.2f} < {self.min_rr_ratio}"
            return result

        result["stop_loss"] = int(stop_price)
        result["target_price"] = int(target_price)
        result["risk_reward_ratio"] = round(rr_ratio, 2)
        result["stop_loss_pct"] = round(risk / close, 4)

        # ═══ Phase 3: Triggers ═══
        triggers_fired = self.trigger_engine.check_all(row)
        result["v8_triggers"] = [
            {"name": t.trigger_name, "fired": t.fired, "reason": t.reason, "strength": t.strength}
            for t in triggers_fired
        ]

        if triggers_fired:
            # 트리거 발동 → BUY
            best_trigger = max(triggers_fired, key=lambda t: t.strength)
            result["signal"] = True
            result["trigger_type"] = best_trigger.trigger_name
            result["trigger_confidence"] = round(best_trigger.strength, 3)
            result["v8_action"] = "BUY"

            # martin_momentum 호환 (backtest_engine.py에서 vol_weight 참조)
            result["martin_momentum"] = {"vol_weight": 1.0}
        else:
            # B등급+이지만 트리거 대기
            result["v8_action"] = "WATCH"

        return result
