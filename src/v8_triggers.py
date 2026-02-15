"""
Quantum Master v8.0 — Phase 3: Entry Triggers
"B등급 이상 종목의 정확한 진입 타이밍을 결정한다"

3가지 독립 트리거 — OR 조건 (하나 이상 발동 시 진입)
"""

import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class TriggerResult:
    """트리거 발동 결과"""
    fired: bool
    trigger_name: str
    reason: str = ""
    strength: float = 0.0  # 0.0~1.0 (트리거 강도)


class TriggerEngine:
    """Phase 3: Entry Trigger Engine"""

    def __init__(self, config: dict):
        v8_cfg = config.get('v8_hybrid', {})
        self.cfg = v8_cfg.get('triggers', {})

    def check_all(self, row: pd.Series) -> List[TriggerResult]:
        """
        모든 트리거를 체크합니다.
        Returns: 발동된 트리거 리스트 (빈 리스트 = 미발동 → 대기)
        """
        all_triggers = [
            self.trigger_trix_golden(row),
            self.trigger_volume_rsi(row),
            self.trigger_curvature_obv(row),
        ]

        return [t for t in all_triggers if t.fired]

    # ─── T1: TRIX 골든크로스 ───
    def trigger_trix_golden(self, row: pd.Series) -> TriggerResult:
        """
        TRIX(12) > Signal(9) 상향교차
        중기 모멘텀의 방향 전환을 확인하는 가장 안정적인 트리거
        """
        trix = row.get('trix', 0)
        trix_signal = row.get('trix_signal', 0)
        trix_prev = row.get('trix_prev', 0)
        trix_signal_prev = row.get('trix_signal_prev', 0)

        golden = (trix > trix_signal) and (trix_prev <= trix_signal_prev)

        strength = 0.0
        if golden:
            diff = abs(trix - trix_signal)
            strength = min(diff / 0.1, 1.0)

        return TriggerResult(
            fired=golden,
            trigger_name="T1_TRIX_Golden",
            reason=f"TRIX({trix:.4f}) > Signal({trix_signal:.4f})" if golden else "",
            strength=strength,
        )

    # ─── T2: 거래량 + RSI 돌파 ───
    def trigger_volume_rsi(self, row: pd.Series) -> TriggerResult:
        """
        거래량 > 20MA * 1.5 AND RSI 45 상향돌파
        에너지 유입과 강도 회복을 동시에 확인
        """
        cfg = self.cfg.get('volume_rsi', {})
        vol_mult = cfg.get('vol_multiplier', 1.5)
        rsi_thresh = cfg.get('rsi_threshold', 45)

        volume = row.get('volume', 0)
        vol_20ma = row.get('volume_ma20', row.get('vol_20ma', 1))
        rsi = row.get('rsi_14', 0)
        rsi_prev = row.get('rsi_prev', rsi)

        vol_surge = volume > vol_20ma * vol_mult
        rsi_cross = rsi > rsi_thresh and rsi_prev <= rsi_thresh

        fired = vol_surge and rsi_cross

        strength = 0.0
        if fired:
            vol_ratio = volume / max(vol_20ma, 1)
            strength = min((vol_ratio - vol_mult) / vol_mult + 0.5, 1.0)

        reason = ""
        if fired:
            reason = f"Vol({volume:.0f}) > 20MA*{vol_mult}({vol_20ma*vol_mult:.0f}), RSI {rsi_prev:.1f}->{rsi:.1f}"

        return TriggerResult(
            fired=fired,
            trigger_name="T2_Volume_RSI",
            reason=reason,
            strength=strength,
        )

    # ─── T3: 곡률 + OBV ───
    def trigger_curvature_obv(self, row: pd.Series) -> TriggerResult:
        """
        곡률 양전환 + OBV 5일 상승 추세
        구조적 반전과 매집을 동시에 확인하는 트리거
        """
        curvature = row.get('ema_curvature', 0)
        obv_trend = row.get('obv_trend_5d', 0)

        curv_positive = curvature > 0
        obv_positive = obv_trend > 0

        fired = curv_positive and obv_positive

        strength = 0.0
        if fired:
            strength = min(curvature * 100 + 0.5, 1.0)

        reason = ""
        if fired:
            reason = f"Curvature({curvature:.4f}) > 0, OBV trend({obv_trend:.4f}) > 0"

        return TriggerResult(
            fired=fired,
            trigger_name="T3_Curvature_OBV",
            reason=reason,
            strength=strength,
        )
