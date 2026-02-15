"""
7D Market Regime Gate v3 — 시장 체제 감지

"지금이 우리 시스템이 작동하는 시장인가?"를 판단.
개별 종목이 아닌 유니버스 전체의 cross-sectional 신호를 사용.

3가지 시장 신호 + 1가지 메타 신호(승률 곱셈 계수):
  Base Signal (additive, 0~3):
    1. Market Breadth: 추세 정렬 종목 비율 (0~1)
    2. Foreign Flow: 외국인 수급 방향성 (0~1, 정규화)
    3. Volatility Regime: 변동성 환경 변화 (0~1)
  Meta Signal (multiplicative):
    4. Self-Adaptive: 최근 거래 성공률 (sa_mult 0.3~1.0)

composite = base * sa_mult → 0~3 범위

출력: RegimeState (position_scale 0.0~1.0)
  - favorable (1.0): 전체 포지션 허용
  - neutral   (0.75): 포지션 75%
  - caution   (0.50): 포지션 50%
  - hostile   (0.0): 신규 진입 차단

v3 변경 (v2 대비):
  - SA를 additive → multiplicative로 전환 (사망 나선 방지)
  - sa_floor=0.30 (최악 시에도 base의 30% 반영 → hostile 진입 가능)
  - SA window 20→15 (빠른 반응/회복)
  - 임계값 원복: favorable≥2.2, neutral≥1.5, caution≥0.8 (0~3 범위)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """체제 판정 결과"""
    regime: str                 # favorable / neutral / caution / hostile
    position_scale: float       # 0.0 ~ 1.0
    breadth_score: float        # 0.0 ~ 1.0 (추세 정렬 비율)
    foreign_score: float        # 0.0 ~ 1.0 (외국인 방향, 정규화)
    volatility_score: float     # 0.0 ~ 1.0 (변동성 안정도)
    composite_score: float      # 0.0 ~ 3.0 (합산 점수)
    details: str = ""


class RegimeGate:
    """7D 시장 체제 감지 Gate v3"""

    def __init__(self, config: dict):
        cfg = config.get("regime_gate", {})
        self.enabled = cfg.get("enabled", True)
        self.lookback = cfg.get("lookback", 20)

        # 1. Market Breadth
        br_cfg = cfg.get("breadth", {})
        self.breadth_favorable = br_cfg.get("favorable_pct", 0.55)
        self.breadth_hostile = br_cfg.get("hostile_pct", 0.35)

        # 2. Foreign Flow
        ff_cfg = cfg.get("foreign_flow", {})
        self.foreign_lookback = ff_cfg.get("lookback", 20)
        self.foreign_strong_ratio = ff_cfg.get("strong_buy_ratio", 0.60)
        self.foreign_weak_ratio = ff_cfg.get("weak_sell_ratio", 0.40)

        # 3. Volatility
        vol_cfg = cfg.get("volatility", {})
        self.vol_lookback_short = vol_cfg.get("short_lookback", 10)
        self.vol_lookback_long = vol_cfg.get("long_lookback", 60)
        self.vol_spike_threshold = vol_cfg.get("spike_threshold", 1.5)

        # 4. Self-Adaptive (multiplicative — 최근 거래 성공률)
        sa_cfg = cfg.get("self_adaptive", {})
        self.sa_window = sa_cfg.get("window", 15)
        self.sa_min_trades = sa_cfg.get("min_trades", 8)
        self.sa_floor = sa_cfg.get("floor", 0.30)
        self.sa_hostile_winrate = sa_cfg.get("hostile_winrate", 0.25)
        self.sa_caution_winrate = sa_cfg.get("caution_winrate", 0.35)
        self.sa_neutral_winrate = sa_cfg.get("neutral_winrate", 0.50)
        self._recent_trades: deque[int] = deque(maxlen=self.sa_window)

        # 체제 임계값 (0~3 범위, SA multiplier 적용 후)
        thresholds = cfg.get("thresholds", {})
        self.favorable_min = thresholds.get("favorable", 2.2)
        self.neutral_min = thresholds.get("neutral", 1.5)
        self.caution_min = thresholds.get("caution", 0.8)

        # 캐시 (같은 날 반복 계산 방지)
        self._cache_idx = -1
        self._cache_result: RegimeState | None = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def update_trade_outcome(self, is_win: bool) -> None:
        """거래 완료 후 승패 기록. backtest_engine에서 호출."""
        self._recent_trades.append(1 if is_win else 0)

    def detect(self, data_dict: dict, idx: int) -> RegimeState:
        """유니버스 전체 데이터로 현재 시장 체제 판정"""
        if not self.enabled:
            return RegimeState(
                regime="favorable", position_scale=1.0,
                breadth_score=1.0, foreign_score=1.0,
                volatility_score=1.0, composite_score=3.0,
                details="regime_gate disabled",
            )

        # 캐시 히트
        if idx == self._cache_idx and self._cache_result is not None:
            return self._cache_result

        breadth = self._calc_breadth(data_dict, idx)
        foreign_raw = self._calc_foreign_flow(data_dict, idx)
        volatility = self._calc_volatility(data_dict, idx)
        sa_raw = self._calc_self_adaptive()

        # foreign을 0~1로 정규화: -1→0, 0→0.5, +1→1.0
        foreign_norm = (foreign_raw + 1.0) / 2.0

        # Base signal (0~3)
        base = breadth + foreign_norm + volatility

        # SA multiplier: sa_floor ~ 1.0
        # SA=0.0 → mult=0.30, SA=0.5 → mult=0.65, SA=1.0 → mult=1.0
        sa_mult = self.sa_floor + (1.0 - self.sa_floor) * sa_raw

        # Final composite (0~3)
        composite = base * sa_mult

        # 체제 판정
        if composite >= self.favorable_min:
            regime, scale = "favorable", 1.0
        elif composite >= self.neutral_min:
            regime, scale = "neutral", 0.75
        elif composite >= self.caution_min:
            regime, scale = "caution", 0.50
        else:
            regime, scale = "hostile", 0.0

        details = (
            f"breadth={breadth:.2f} foreign={foreign_norm:.2f} "
            f"vol={volatility:.2f} sa={sa_raw:.2f}(×{sa_mult:.2f}) "
            f"base={base:.2f} → {composite:.2f} [{regime}]"
        )

        result = RegimeState(
            regime=regime,
            position_scale=scale,
            breadth_score=breadth,
            foreign_score=foreign_norm,
            volatility_score=volatility,
            composite_score=composite,
            details=details,
        )

        self._cache_idx = idx
        self._cache_result = result
        return result

    # ──────────────────────────────────────────────
    # Signal 1: Market Breadth
    # ──────────────────────────────────────────────

    def _calc_breadth(self, data_dict: dict, idx: int) -> float:
        """추세 정렬(Close>SMA60>SMA120) 종목 비율. Returns 0.0~1.0"""
        aligned = 0
        total = 0

        for ticker, df in data_dict.items():
            if idx >= len(df):
                continue
            row = df.iloc[idx]

            sma60 = row.get("sma_60", np.nan)
            sma120 = row.get("sma_120", np.nan)
            close = row.get("close", np.nan)

            if pd.isna(sma60) or pd.isna(sma120) or pd.isna(close):
                continue

            total += 1
            if close > sma60 and sma60 > sma120:
                aligned += 1

        if total == 0:
            return 0.5

        ratio = aligned / total

        if ratio >= self.breadth_favorable:
            return 1.0
        elif ratio <= self.breadth_hostile:
            return 0.0
        else:
            return (ratio - self.breadth_hostile) / (
                self.breadth_favorable - self.breadth_hostile
            )

    # ──────────────────────────────────────────────
    # Signal 2: Foreign Flow
    # ──────────────────────────────────────────────

    def _calc_foreign_flow(self, data_dict: dict, idx: int) -> float:
        """유니버스 전체 외국인 순매수 방향. Returns -1.0~1.0.

        데이터가 전량 0이면 중립(0.0) 반환 — 데이터 없음 ≠ 매도.
        """
        if idx < self.foreign_lookback:
            return 0.0

        buy_days = 0
        total_days = 0

        for day_offset in range(self.foreign_lookback):
            check_idx = idx - day_offset
            if check_idx < 0:
                break

            net_buy_count = 0
            total_count = 0
            nonzero_count = 0

            for ticker, df in data_dict.items():
                if check_idx >= len(df):
                    continue
                foreign_col = None
                for col_name in ["외국인합계", "foreign_net"]:
                    if col_name in df.columns:
                        foreign_col = col_name
                        break
                if foreign_col is None:
                    continue

                val = df[foreign_col].iloc[check_idx]
                if pd.isna(val):
                    continue

                total_count += 1
                if val != 0:
                    nonzero_count += 1
                if val > 0:
                    net_buy_count += 1

            if total_count > 0:
                # 데이터 전량 0 감지: nonzero가 5% 미만이면 "데이터 없음"
                if nonzero_count / total_count < 0.05:
                    continue  # 이 날은 카운트하지 않음

                day_ratio = net_buy_count / total_count
                total_days += 1
                if day_ratio >= self.foreign_strong_ratio:
                    buy_days += 1
                elif day_ratio <= self.foreign_weak_ratio:
                    buy_days -= 1

        if total_days == 0:
            return 0.0

        return np.clip(buy_days / total_days, -1.0, 1.0)

    # ──────────────────────────────────────────────
    # Signal 3: Volatility Regime
    # ──────────────────────────────────────────────

    def _calc_volatility(self, data_dict: dict, idx: int) -> float:
        """단기 변동성 vs 장기 변동성. Returns 0.0~1.0"""
        if idx < self.vol_lookback_long:
            return 0.5

        vol_ratios = []

        for ticker, df in data_dict.items():
            if idx >= len(df):
                continue
            if "atr_14" not in df.columns or "close" not in df.columns:
                continue

            close = df["close"].iloc[idx]
            if close <= 0 or pd.isna(close):
                continue

            start_short = max(0, idx - self.vol_lookback_short + 1)
            short_atr = df["atr_14"].iloc[start_short: idx + 1].mean()

            start_long = max(0, idx - self.vol_lookback_long + 1)
            long_atr = df["atr_14"].iloc[start_long: idx + 1].mean()

            if long_atr > 0 and not pd.isna(short_atr) and not pd.isna(long_atr):
                vol_ratios.append(short_atr / long_atr)

        if not vol_ratios:
            return 0.5

        median_ratio = np.median(vol_ratios)

        if median_ratio >= self.vol_spike_threshold:
            return 0.0
        elif median_ratio >= 1.2:
            return max(0.0, 1.0 - (median_ratio - 1.0) / (
                self.vol_spike_threshold - 1.0
            ))
        else:
            return 1.0

    # ──────────────────────────────────────────────
    # Signal 4: Self-Adaptive (multiplicative 계수)
    # ──────────────────────────────────────────────

    def _calc_self_adaptive(self) -> float:
        """최근 N거래의 승률로 시스템 적합성 직접 측정.

        충분한 거래 수(min_trades) 이전에는 중립(0.5) 반환.
        Returns 0.0~1.0 → sa_mult = sa_floor + (1-sa_floor) * return_value
        """
        if len(self._recent_trades) < self.sa_min_trades:
            return 0.5  # 데이터 부족 → 중립

        win_rate = sum(self._recent_trades) / len(self._recent_trades)

        # 승률 → 0.0~1.0 스코어 (선형 보간)
        if win_rate <= self.sa_hostile_winrate:
            return 0.0
        elif win_rate <= self.sa_caution_winrate:
            t = (win_rate - self.sa_hostile_winrate) / (
                self.sa_caution_winrate - self.sa_hostile_winrate
            )
            return t * 0.35
        elif win_rate <= self.sa_neutral_winrate:
            t = (win_rate - self.sa_caution_winrate) / (
                self.sa_neutral_winrate - self.sa_caution_winrate
            )
            return 0.35 + t * 0.30
        else:
            t = min(1.0, (win_rate - self.sa_neutral_winrate) / 0.20)
            return 0.65 + t * 0.35
