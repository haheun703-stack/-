"""
Quantum Master v8.5 — Main Pipeline
"포물선의 초점에 진입하여 추세를 먹는다"

Architecture: Gate + Score Hybrid + Ratio Boost + Horizon
  Phase 1: Hard Gates (AND) -> 자격 없는 종목 빠른 제거
  Phase 2: Scoring (가중합) -> 초점 근접도 측정 -> 등급 결정
  Phase 2.5: Ratio Boost (v8.5) -> 91건 역도출 비율 지표 → 등급 부스트
  Phase 3: Triggers (OR) -> B등급+ 종목 진입 타이밍
  Phase 4: Horizon (v8.1) -> 단기/중기/장기 보유기간 판정

v8.1 -> v8.5 변경:
  - 91건 시작점 역도출 비율 지표 3종 (BB/ATR, Vol/Price, OBV/Price) 부스트
"""

import logging

import pandas as pd

from .holding_horizon import HoldingHorizonClassifier
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
        self.horizon_classifier = HoldingHorizonClassifier(config)

        v8_cfg = config.get('v8_hybrid', {})
        pos_cfg = v8_cfg.get('position', {})
        self.stop_loss_atr = pos_cfg.get('stop_loss_atr', 2.0)
        self.target_atr_A = pos_cfg.get('target_atr_A', 5.0)
        self.target_atr_B = pos_cfg.get('target_atr_B', 3.5)
        self.min_rr_ratio = pos_cfg.get('min_rr_ratio', 1.5)

        # v8.5: 비율 지표 부스트 (91건 역도출)
        boost_cfg = v8_cfg.get('scoring', {}).get('ratio_boost', {})
        self.ratio_boost_enabled = boost_cfg.get('enabled', False)
        self.ratio_boost_cfg = boost_cfg

    def _calc_ratio_boost(self, row: pd.Series) -> tuple[float, dict]:
        """v8.5 비율 지표 부스트 — 91건 시작점 역도출.

        3개 조건의 동시 충족 정도 × max_boost → zone_score 가산.
        최소 2개 이상 충족해야 부스트 적용 (선택적 필터).
          R1: BB/ATR 비율 → 변동성 압축 구간 (91건 중앙 4.05)
          R2: 조용한 거래량 + 가격 평탄 → 은닉 매집 (91건 69% vol < avg)
          R3: OBV 선행 → 가격보다 수급 먼저 전환 (91건 50%+)
        """
        cfg = self.ratio_boost_cfg
        max_boost = cfg.get('max_boost', 0.05)
        details = {}
        conditions_met = 0

        close = row.get('close', 0)
        if close <= 0:
            return 0.0, details

        # ─── R1: BB/ATR 변동성 압축 ───
        bb_width = row.get('bb_width', 0) or 0
        atr = row.get('atr_14', 1)
        atr_pct = atr / close
        bb_atr = bb_width / max(atr_pct, 0.001)

        r1_range = cfg.get('bb_atr_optimal', [3.0, 8.0])
        r1_hit = r1_range[0] <= bb_atr <= r1_range[1]
        if r1_hit:
            conditions_met += 1
        details['r1_bb_atr'] = round(bb_atr, 2)

        # ─── R2: 조용한 매집 (거래량 비급등 + 가격 비변동) ───
        vol_surge = row.get('volume_surge_ratio', 1.0) or 1.0
        vol_contraction = row.get('volume_contraction_ratio', 1.0) or 1.0
        price_trend = abs(row.get('price_trend_5d', 0) or 0)

        # 91건 패턴: 69% 거래량 < 20d 평균, 가격 5일 변동 작음
        quiet_vol = vol_surge < 1.3 and vol_contraction < 1.5
        small_price_move = price_trend < 0.05  # 5일 5% 미만
        r2_hit = quiet_vol and small_price_move
        if r2_hit:
            conditions_met += 1
        details['r2_quiet'] = r2_hit

        # ─── R3: OBV 선행 (수급 전환) ───
        obv_trend = row.get('obv_trend_5d', 0) or 0
        price_trend_raw = row.get('price_trend_5d', 0) or 0

        # OBV 양성이면서 가격이 강하게 상승하지 않는 경우
        r3_hit = obv_trend > 0 and price_trend_raw < 0.03
        if r3_hit:
            conditions_met += 1
        details['r3_obv_lead'] = round(obv_trend, 4)

        # ── 최소 2개 조건 충족 시에만 부스트 ──
        min_conditions = cfg.get('min_conditions', 2)
        if conditions_met < min_conditions:
            details['conditions'] = conditions_met
            details['total_boost'] = 0.0
            return 0.0, details

        # 조건 충족 비율에 따른 부스트 크기
        ratio = conditions_met / 3.0  # 2/3 = 0.667, 3/3 = 1.0
        boost = round(ratio * max_boost, 4)
        details['conditions'] = conditions_met
        details['total_boost'] = boost
        return boost, details

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
        base_score = grade_result.total_score
        result["zone_score"] = round(base_score, 4)
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

        # ═══ Phase 2.5: Ratio Boost (v8.5) ═══
        # 91건 역도출 비율 지표로 zone_score 가산 → 순위 차별화
        # 등급은 변경하지 않음 (같은 A등급 내 우선순위만 조정)
        if self.ratio_boost_enabled:
            ratio_boost, ratio_details = self._calc_ratio_boost(row)
            if ratio_boost > 0:
                result["zone_score"] = round(min(base_score + ratio_boost, 1.0), 4)
                result["bes_score"] = result["zone_score"]
                result["v8_ratio_boost"] = ratio_details

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

            # ═══ Phase 4: Holding Horizon (v8.1) ═══
            horizon = self.horizon_classifier.classify(row)
            result["holding_horizon"] = horizon.horizon
            result["holding_horizon_days"] = horizon.horizon_days
            result["holding_horizon_confidence"] = horizon.confidence
            result["holding_horizon_label"] = self.horizon_classifier.horizon_label(horizon.horizon)
            result["holding_horizon_factors"] = horizon.factors
        else:
            # B등급+이지만 트리거 대기
            result["v8_action"] = "WATCH"

        return result
