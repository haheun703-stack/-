"""
5D-⑤ Kelly Sizer — Kelly Criterion 기반 포지션 사이저

시작점 전략의 Class S/A/B/C별 포지션 크기를 결정.
Kelly 공식으로 최적 베팅 비율을 산출하고,
Quarter-Kelly + 클래스별 상한으로 리스크를 제어.

핵심 공식:
  Kelly % = (b*p - q) / b
  b = avg_win / avg_loss (odds ratio)
  p = win_rate, q = 1 - p

계층 규칙:
  Class S → 최대 25% (확신 종목)
  Class A → 최대 15% (우수 종목)
  Class B → 최대 10% (보통 종목)
  Class C → 최대  5% (관찰 종목)

의존성: numpy만 사용
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class KellySizer:
    """Kelly Criterion 기반 포지션 사이저"""

    # 클래스별 기본 포지션 상한
    DEFAULT_MAX_POSITION = {
        "S": 0.25,
        "A": 0.15,
        "B": 0.10,
        "C": 0.05,
    }

    # 클래스별 타임아웃 규칙 (일 단위)
    TIMEOUT_RULES = {
        "S": {"half_reduce_days": 15, "full_exit_days": 30},
        "A": {"half_reduce_days": 10, "full_exit_days": 20},
        "B": {"half_reduce_days": 7, "full_exit_days": 14},
        "C": {"half_reduce_days": 3, "full_exit_days": 7},
    }

    # 클래스별 트레일링 스탑 비율
    TRAILING_STOP = {
        "S": 0.10,
        "A": 0.08,
        "B": 0.07,
        "C": 0.05,
    }

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("kelly", {})
        self.kelly_fraction = cfg.get("kelly_fraction", 0.25)  # Quarter-Kelly
        self.max_position_pct = {
            k: cfg.get("max_position_pct", {}).get(k, v)
            for k, v in self.DEFAULT_MAX_POSITION.items()
        }
        self.min_rr_ratio = cfg.get("min_rr_ratio", 1.5)

    # ─── Kelly Fraction 계산 ───────────────────────

    def kelly_fraction_calc(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly 비율 계산.

        Kelly % = (b*p - q) / b
        b = avg_win / avg_loss (odds ratio)
        p = win_rate, q = 1 - p

        Parameters:
            win_rate: 승률 (0.0~1.0)
            avg_win: 평균 수익 (양수)
            avg_loss: 평균 손실 (양수, 절대값)

        Returns:
            raw Kelly fraction (음수면 0)
        """
        if avg_loss <= 0:
            logger.warning("avg_loss가 0 이하: %.4f → Kelly 0 반환", avg_loss)
            return 0.0

        if avg_win <= 0:
            return 0.0

        b = avg_win / avg_loss  # odds ratio
        p = np.clip(win_rate, 0.0, 1.0)
        q = 1.0 - p

        kelly = (b * p - q) / b

        if kelly < 0:
            logger.debug("Kelly 음수: %.4f (승률=%.2f, b=%.2f) → 베팅 불가", kelly, p, b)
            return 0.0

        return float(kelly)

    # ─── 포지션 사이징 ─────────────────────────────

    def size_position(
        self,
        signal_class: str,
        confidence: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_capital: float = 1.0,
    ) -> dict:
        """
        포지션 크기 결정.

        Kelly → Quarter-Kelly → 클래스 상한 → 확신도 보정 → 최종 비율.

        Parameters:
            signal_class: "S", "A", "B", "C"
            confidence: 전조 충족도 (0.0~1.0)
            win_rate: 승률 (0.0~1.0)
            avg_win: 평균 수익 (양수)
            avg_loss: 평균 손실 (양수, 절대값)
            current_capital: 현재 자본금 (기본 1.0)

        Returns:
            {
                "raw_kelly": float,
                "adjusted_kelly": float,
                "class_limit": float,
                "final_pct": float,
                "position_amount": float,
                "signal_class": str,
                "rationale": str,
            }
        """
        signal_class = signal_class.upper()
        if signal_class not in self.max_position_pct:
            logger.warning("알 수 없는 신호 클래스: %s → C로 처리", signal_class)
            signal_class = "C"

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # 1. Raw Kelly
        raw_kelly = self.kelly_fraction_calc(win_rate, avg_win, avg_loss)

        # 2. Fraction 적용 (Quarter-Kelly)
        adjusted_kelly = raw_kelly * self.kelly_fraction

        # 3. 클래스별 상한
        class_limit = self.max_position_pct[signal_class]

        # 4. 상한 적용 + 확신도 보정
        capped = min(adjusted_kelly, class_limit)
        final_pct = capped * confidence

        # 5. 포지션 금액
        position_amount = current_capital * final_pct

        # 6. 근거 문자열
        rationale = (
            f"Kelly {raw_kelly * 100:.1f}%"
            f" → {self.kelly_fraction:.0%} 적용 {adjusted_kelly * 100:.1f}%"
            f" → Class {signal_class} 상한 {class_limit * 100:.0f}%"
            f" → 확신도 {confidence * 100:.0f}%"
            f" → 최종 {final_pct * 100:.1f}%"
        )

        return {
            "raw_kelly": round(raw_kelly, 4),
            "adjusted_kelly": round(adjusted_kelly, 4),
            "class_limit": round(class_limit, 4),
            "final_pct": round(final_pct, 4),
            "position_amount": round(position_amount, 2),
            "signal_class": signal_class,
            "rationale": rationale,
        }

    # ─── 타임아웃 규칙 ─────────────────────────────

    def timeout_rules(self, signal_class: str) -> dict:
        """
        클래스별 타임아웃 규칙 반환.

        Parameters:
            signal_class: "S", "A", "B", "C"

        Returns:
            {
                "half_reduce_days": int,
                "full_exit_days": int,
                "catalyst_fail_action": "immediate_exit",
            }
        """
        signal_class = signal_class.upper()
        if signal_class not in self.TIMEOUT_RULES:
            logger.warning("알 수 없는 신호 클래스: %s → C로 처리", signal_class)
            signal_class = "C"

        rules = self.TIMEOUT_RULES[signal_class]

        return {
            "half_reduce_days": rules["half_reduce_days"],
            "full_exit_days": rules["full_exit_days"],
            "catalyst_fail_action": "immediate_exit",
        }

    # ─── 출구 전략 규칙 ────────────────────────────

    def exit_rules(self, signal_class: str) -> dict:
        """
        클래스별 출구 전략 규칙 반환.

        Parameters:
            signal_class: "S", "A", "B", "C"

        Returns:
            {
                "trailing_stop_pct": float,
                "partial_take_profit_1": {...},
                "partial_take_profit_2": {...},
                "full_exit": {...},
            }
        """
        signal_class = signal_class.upper()
        if signal_class not in self.TRAILING_STOP:
            logger.warning("알 수 없는 신호 클래스: %s → C로 처리", signal_class)
            signal_class = "C"

        return {
            "trailing_stop_pct": self.TRAILING_STOP[signal_class],
            "partial_take_profit_1": {"trigger": "clock_12h", "pct": 0.30},
            "partial_take_profit_2": {"trigger": "crowd_overheat", "pct": 0.30},
            "full_exit": {"trigger": "phase_transition_down", "pct": 1.0},
        }
