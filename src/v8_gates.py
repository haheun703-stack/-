"""
Quantum Master v10.0 — Phase 1: Hard Gates
"자격 없는 종목을 빠르게 제거한다"

4개의 필수 게이트: 하나라도 미달이면 즉시 탈락 (AND 조건)
G1: 추세 | G2: 조정폭 | G3: 과열 방지 | G4: 공매도 압력 (레짐별 활성)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class GateResult:
    """게이트 통과 결과"""
    passed: bool
    gate_name: str
    reason: str = ""
    values: dict = None

    def __post_init__(self):
        if self.values is None:
            self.values = {}


class GateEngine:
    """Phase 1: Hard Gate Engine"""

    def __init__(self, config: dict):
        v8_cfg = config.get('v8_hybrid', {})
        self.cfg = v8_cfg.get('gates', {})
        # 동적 오버라이드 (레짐 프로파일에서 설정)
        self._pullback_max_override: float | None = None
        self._short_gate_active: bool = False  # G4는 공매도 활성기에만 작동

        # v10.1: 마스터 스위치 — use_short_selling_filter: false면 G4 영구 비활성
        self._short_filter_enabled = config.get('use_short_selling_filter', False)

    def set_pullback_max(self, value: float | None):
        """공매도 체제에 따라 G2 pullback 상한을 동적 조정."""
        self._pullback_max_override = value

    def set_short_gate_active(self, active: bool):
        """공매도 체제에 따라 G4 활성/비활성."""
        self._short_gate_active = active

    def run_all_gates(self, row: pd.Series) -> tuple[bool, list]:
        """
        모든 게이트를 순차 실행.
        Returns: (통과 여부, [GateResult 리스트])
        """
        results = []

        g1 = self.gate_trend(row)
        results.append(g1)
        if not g1.passed:
            return False, results

        g2 = self.gate_pullback(row)
        results.append(g2)
        if not g2.passed:
            return False, results

        g3 = self.gate_overheat(row)
        results.append(g3)
        if not g3.passed:
            return False, results

        g4 = self.gate_short_pressure(row)
        results.append(g4)
        if not g4.passed:
            return False, results

        return True, results

    # ─── G1: 추세 게이트 ───
    def gate_trend(self, row: pd.Series) -> GateResult:
        """
        상승 추세 확인 — 추세가 없으면 눌림목이 아니라 하락

        조건:
        - 60MA > 120MA (중기 > 장기)
        - ADX > 18 (추세 강도)
        - 현재가 > 120MA (장기 이평 위)
        """
        cfg = self.cfg.get('trend', {})
        ma_fast_period = cfg.get('ma_fast', 60)
        ma_slow_period = cfg.get('ma_slow', 120)
        adx_min = cfg.get('adx_min', 18)

        ma_fast = row.get(f'sma_{ma_fast_period}', row.get(f'ma{ma_fast_period}', 0))
        ma_slow = row.get(f'sma_{ma_slow_period}', row.get(f'ma{ma_slow_period}', 0))
        adx = row.get('adx_14', row.get('adx', 0))
        close = row.get('close', 0)

        cond_ma = ma_fast > ma_slow
        cond_adx = adx > adx_min
        cond_above = close > ma_slow

        passed = cond_ma and cond_adx and cond_above

        reasons = []
        if not cond_ma:
            reasons.append(f"MA{ma_fast_period}({ma_fast:.0f}) <= MA{ma_slow_period}({ma_slow:.0f})")
        if not cond_adx:
            reasons.append(f"ADX({adx:.1f}) < {adx_min}")
        if not cond_above:
            reasons.append(f"Close({close:.0f}) <= MA{ma_slow_period}({ma_slow:.0f})")

        return GateResult(
            passed=passed,
            gate_name="G1_Trend",
            reason=" | ".join(reasons) if reasons else "추세 확인됨",
            values={'ma_fast': ma_fast, 'ma_slow': ma_slow, 'adx': adx, 'close': close}
        )

    # ─── G2: 조정폭 게이트 ───
    def gate_pullback(self, row: pd.Series) -> GateResult:
        """
        충분한 눌림 확인 — "기대가 식었는가?"

        조건: 20일 최고가 대비 현재가의 ATR 단위 하락폭 0.8~4.0
        """
        cfg = self.cfg.get('pullback', {})
        min_atr = cfg.get('min_atr_pullback', 0.8)
        max_atr = self._pullback_max_override or cfg.get('max_atr_pullback', 4.0)
        lookback = cfg.get('high_lookback_days', 20)

        high_recent = row.get(f'high_{lookback}d', row.get(f'high_{lookback}', 0))
        close = row.get('close', 0)
        atr = row.get('atr_14', row.get('atr', 1))

        if atr <= 0:
            atr = 1

        pullback_atr = (high_recent - close) / atr

        passed = min_atr <= pullback_atr <= max_atr

        if pullback_atr < min_atr:
            reason = f"조정 부족: {pullback_atr:.2f} ATR < {min_atr} ATR"
        elif pullback_atr > max_atr:
            reason = f"과도한 하락: {pullback_atr:.2f} ATR > {max_atr} ATR (추세 전환 의심)"
        else:
            reason = f"적정 조정: {pullback_atr:.2f} ATR"

        return GateResult(
            passed=passed,
            gate_name="G2_Pullback",
            reason=reason,
            values={'pullback_atr': pullback_atr, 'high_recent': high_recent, 'atr': atr}
        )

    # ─── G3: 과열 방지 게이트 ───
    def gate_overheat(self, row: pd.Series) -> GateResult:
        """
        고점 근처 종목 제거 — 이미 많이 오른 종목 진입 방지

        조건: 52주 고점 대비 현재가 < 92%
        """
        cfg = self.cfg.get('overheat', {})
        max_ratio = cfg.get('max_52w_ratio', 0.92)

        close = row.get('close', 0)
        high_52w = row.get('high_252', row.get('high_52w', close))

        if high_52w <= 0:
            high_52w = close

        ratio = close / high_52w if high_52w > 0 else 1.0

        passed = ratio < max_ratio

        if not passed:
            reason = f"고점 근처: {ratio:.1%} >= {max_ratio:.0%} (52주 고점 {high_52w:.0f})"
        else:
            reason = f"52주 고점 대비 {ratio:.1%}"

        return GateResult(
            passed=passed,
            gate_name="G3_Overheat",
            reason=reason,
            values={'ratio_52w': ratio, 'high_52w': high_52w}
        )

    # ─── G4: 공매도 압력 게이트 ───
    def gate_short_pressure(self, row: pd.Series) -> GateResult:
        """
        공매도 집중 + 증가 중인 종목 제거 — 공매도 재개기에만 활성.

        조건: short_interest_pct > 5% AND short_balance_chg_5d > 0
        공매도 금지기에는 자동 통과 (self._short_gate_active=False).

        v10.1: use_short_selling_filter=false 시 영구 비활성 (마스터 스위치)
        """
        # v10.1 마스터 스위치: 공매도 필터 비활성 → 항상 통과
        if not self._short_filter_enabled:
            return GateResult(
                passed=True,
                gate_name="G4_ShortPressure",
                reason="마스터 스위치 OFF — 공매도 필터 비활성",
            )

        if not self._short_gate_active:
            return GateResult(
                passed=True,
                gate_name="G4_ShortPressure",
                reason="공매도 금지기 — 게이트 비활성",
            )

        cfg = self.cfg.get('short_pressure', {})
        max_short_pct = cfg.get('max_short_pct', 5.0)

        short_pct = row.get('short_interest_pct', 0) or 0
        short_chg = row.get('short_balance_chg_5d', 0) or 0

        # 공매도 5% 이상 + 증가 중 = 진입 금지
        blocked = short_pct > max_short_pct and short_chg > 0

        if blocked:
            reason = (
                f"공매도 압력: {short_pct:.2f}% > {max_short_pct}% "
                f"& 5D 변화 +{short_chg:.1f}%"
            )
        elif short_pct > max_short_pct:
            reason = f"공매도 {short_pct:.2f}% (높지만 감소 중)"
        else:
            reason = f"공매도 {short_pct:.2f}%"

        return GateResult(
            passed=not blocked,
            gate_name="G4_ShortPressure",
            reason=reason,
            values={'short_pct': short_pct, 'short_chg_5d': short_chg}
        )
