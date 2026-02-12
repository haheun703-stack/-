"""
Step 4: screener.py — Pre-screening + Mandatory Gate 필터

2단계 필터:
1. Pre-screening: 매출 1,000억+, 거래대금 5억+, 2분기 흑자
2. Gate: 추세 필터(60MA>120MA) + 분배 리스크(DRS<0.60)
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class Screener:
    """Pre-screening + Gate 체크"""

    def __init__(self, config: dict, fundamental_engine):
        self.config = config
        self.fundamental = fundamental_engine
        self.strategy = config["strategy"]

    # ──────────────────────────────────────────────
    # Pre-screening
    # ──────────────────────────────────────────────

    def pre_screen(self, ticker: str, df: pd.DataFrame, idx: int) -> bool:
        """
        Pre-screening 조건 (AND):
        1. 매출 >= 1,000억원 (sector_map 기반)
        2. 60일 평균 거래대금 >= 5억원
        3. 최근 2분기 연속 영업이익 > 0
        """
        # 1. 매출 필터
        if not self.fundamental.check_revenue_filter(ticker):
            return False

        # 2. 거래대금 필터
        min_tv = self.strategy["min_daily_trading_value_억"] * 1e8  # 억→원
        if "trading_value_ma60" in df.columns:
            tv = df["trading_value_ma60"].iloc[idx]
            if pd.notna(tv) and tv < min_tv:
                return False
        elif "trading_value" in df.columns:
            recent_tv = df["trading_value"].iloc[max(0, idx - 60):idx + 1].mean()
            if recent_tv < min_tv:
                return False

        # 3. 수익성 필터
        if not self.fundamental.check_profitability(df, idx):
            return False

        return True

    # ──────────────────────────────────────────────
    # Trend Gate (추세 필터)
    # ──────────────────────────────────────────────

    def check_trend_gate(self, df: pd.DataFrame, idx: int) -> bool:
        """
        추세 필터 (Mandatory Gate):
        조건 A: 60MA > 120MA (중기 추세 상승)
        조건 B: ADX(14) > 20 (추세 존재)
        조건 C: 현재가 > 200MA (장기 추세)
        통과: A AND (B OR C)
        """
        row = df.iloc[idx]

        sma60 = row.get("sma_60", np.nan)
        sma120 = row.get("sma_120", np.nan)
        adx = row.get("adx_14", np.nan)
        close = row.get("close", np.nan)
        sma200 = row.get("sma_200", np.nan)

        # 데이터 부족 시 실패
        if pd.isna(sma60) or pd.isna(sma120):
            return False

        # 조건 A: 중기 추세 상승
        cond_a = sma60 > sma120

        if not cond_a:
            return False

        # 조건 B: ADX > threshold
        cond_b = (not pd.isna(adx)) and adx > self.strategy["adx_threshold"]

        # 조건 C: 장기 추세
        cond_c = (not pd.isna(sma200)) and (not pd.isna(close)) and close > sma200

        return cond_b or cond_c

    # ──────────────────────────────────────────────
    # Distribution Risk Score (DRS)
    # ──────────────────────────────────────────────

    def calc_distribution_risk(self, df: pd.DataFrame, idx: int) -> float:
        """
        와이코프 분배 리스크 점수 (DRS)
        DRS = 0.30×f(OBV) + 0.25×f(거래량패턴) + 0.25×f(기관수급) + 0.20×f(반등강도)
        범위: 0.0 (안전) ~ 1.0 (분배 위험)
        """
        lookback = self.strategy["drs_lookback"]
        weights = self.strategy["drs_weights"]

        start = max(0, idx - lookback)
        window = df.iloc[start:idx + 1]

        if len(window) < 10:
            return 0.3  # 데이터 부족 시 중립

        # ── f1: OBV 방향 (선형회귀 기울기) ──
        f_obv = 0.0
        if "obv" in window.columns:
            obv_vals = window["obv"].dropna()
            if len(obv_vals) >= 5:
                try:
                    slope = linregress(range(len(obv_vals)), obv_vals.values).slope
                    f_obv = 0.0 if slope > 0 else 1.0
                except Exception:
                    f_obv = 0.5

        # ── f2: 거래량 패턴 (하락일 vs 상승일 평균 거래량) ──
        f_vol = 0.0
        close_diff = window["close"].diff()
        down_vol = window.loc[close_diff < 0, "volume"]
        up_vol = window.loc[close_diff >= 0, "volume"]

        if len(down_vol) > 0 and len(up_vol) > 0:
            avg_down = down_vol.mean()
            avg_up = up_vol.mean()
            if avg_down > avg_up * 1.1:  # 하락일 거래량이 10% 이상 많으면
                f_vol = 1.0
            elif avg_down > avg_up:
                f_vol = 0.6
            else:
                f_vol = 0.0

        # ── f3: 기관+외국인 순매수 ──
        f_inst = 0.0
        inst_col = "기관합계"
        foreign_col = "외국인합계"

        if inst_col in window.columns and foreign_col in window.columns:
            inst_net = window[inst_col].sum() + window[foreign_col].sum()
            if inst_net < 0:
                f_inst = 1.0
            elif inst_net < window["trading_value"].sum() * 0.01:
                f_inst = 0.4
            else:
                f_inst = 0.0

        # ── f4: 반등 강도 (최근 반등들의 크기 변화) ──
        f_bounce = self._calc_bounce_trend(window)

        # ── DRS 종합 ──
        drs = (weights["obv_slope"] * f_obv +
               weights["volume_pattern"] * f_vol +
               weights["institutional_flow"] * f_inst +
               weights["bounce_strength"] * f_bounce)

        return round(min(max(drs, 0.0), 1.0), 3)

    def _calc_bounce_trend(self, window: pd.DataFrame) -> float:
        """
        반등 강도 추세 분석.
        최근 반등 피크들의 크기가 축소되면 분배 징후 (1.0)
        확대되면 축적 징후 (0.0)
        """
        close = window["close"].values
        if len(close) < 10:
            return 0.5

        # 간단한 피크 탐지 (3일 연속 상승 → 피크)
        peaks = []
        for i in range(2, len(close) - 1):
            if close[i] > close[i - 1] and close[i] > close[i + 1]:
                peaks.append(close[i])

        if len(peaks) < 2:
            return 0.5

        # 최근 피크 vs 이전 피크
        recent = peaks[-1]
        prev = peaks[-2]

        if recent < prev * 0.98:  # 반등 축소
            return 0.8
        elif recent > prev * 1.02:  # 반등 확대
            return 0.2
        else:
            return 0.5

    def check_distribution_gate(self, df: pd.DataFrame, idx: int) -> tuple:
        """
        DRS Gate 판정.
        반환: (통과여부, DRS값)
        """
        drs = self.calc_distribution_risk(df, idx)
        passed = drs < self.strategy["drs_max_threshold"]
        return passed, drs

    # ──────────────────────────────────────────────
    # 종합 Gate 체크
    # ──────────────────────────────────────────────

    def check_all_gates(self, ticker: str, df: pd.DataFrame, idx: int) -> dict:
        """
        모든 Gate를 종합 체크.
        반환: {
            "passed": bool,
            "pre_screen": bool,
            "trend_gate": bool,
            "drs_gate": bool,
            "drs_value": float,
            "fail_reason": str or None,
        }
        """
        result = {
            "passed": False,
            "pre_screen": False,
            "trend_gate": False,
            "drs_gate": False,
            "drs_value": 0.0,
            "fail_reason": None,
        }

        # 1. Pre-screening
        if not self.pre_screen(ticker, df, idx):
            result["fail_reason"] = "pre_screen_fail"
            return result
        result["pre_screen"] = True

        # 2. Trend Gate
        if not self.check_trend_gate(df, idx):
            result["fail_reason"] = "trend_gate_fail"
            return result
        result["trend_gate"] = True

        # 3. DRS Gate
        drs_passed, drs_val = self.check_distribution_gate(df, idx)
        result["drs_value"] = drs_val
        if not drs_passed:
            result["fail_reason"] = f"drs_too_high({drs_val:.2f})"
            return result
        result["drs_gate"] = True

        result["passed"] = True
        return result
