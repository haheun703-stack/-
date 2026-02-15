"""
v4.1 적응형 청산 모듈 — 건강한 조정 vs 위험한 조정 판별

문제: Impulse -3% 고정 손절이 건강한 조정도 잘라버림
해결: 조정 발생 시 건강도를 평가하여 손절을 동적으로 조정

건강한 조정 (추세 지속):
  - 주가가 MA20 위 또는 MA20 근처에서 반등
  - ADX >= 25 (추세 유지)
  - +DI > -DI (방향성 유지)
  - OBV 기울기 유지/상승 (매집 계속)
  - 거래량 감소 (매도 압력 없음)
  - 기관/외국인 순매수 유지

위험한 조정 (추세 전환):
  - MA20 이탈 + MA60 접근
  - ADX 하락 추세
  - +DI < -DI 교차 (역방향)
  - OBV 급락 (분배)
  - 거래량 증가하며 하락 (패닉)
  - 기관/외국인 동시 매도

엔티티(trading_models.PullbackHealth)에만 의존.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.entities.trading_models import PullbackHealth

_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("adaptive_exit", {})


class AdaptiveExitManager:
    """
    조정 건강도 기반 적응형 손절 관리.

    backtest_engine / position_tracker에서 기존 pct_stop 대신 호출:
      health = adaptive_exit.evaluate_pullback(df, idx, entry_price, highest_price)
      if health.classification == "dangerous":
          → 즉시 청산
      elif health.classification == "healthy":
          → 손절을 MA20 기반 / ATR*3 으로 완화
    """

    def __init__(self, config_path: Path | str | None = None):
        cfg = _load_config(config_path)

        # 점수 배점 (총 100점)
        self.w_ma20 = cfg.get("weight_ma20_support", 15)
        self.w_ma60 = cfg.get("weight_ma60_support", 10)
        self.w_adx = cfg.get("weight_adx_trending", 12)
        self.w_di = cfg.get("weight_di_positive", 10)
        self.w_obv = cfg.get("weight_obv_intact", 15)
        self.w_vol = cfg.get("weight_volume_declining", 10)
        self.w_inst = cfg.get("weight_inst_flow", 8)
        self.w_foreign = cfg.get("weight_foreign_flow", 8)
        self.w_rsi = cfg.get("weight_rsi_healthy", 7)
        self.w_pullback_depth = cfg.get("weight_pullback_depth", 5)

        # 분류 임계값
        self.healthy_threshold = cfg.get("healthy_threshold", 65)
        self.caution_threshold = cfg.get("caution_threshold", 40)
        self.dangerous_threshold = cfg.get("dangerous_threshold", 25)

        # 적응형 손절 배율
        self.healthy_stop_mult = cfg.get("healthy_stop_mult", 2.0)     # ATR*2
        self.caution_stop_mult = cfg.get("caution_stop_mult", 1.5)     # ATR*1.5
        self.dangerous_stop_pct = cfg.get("dangerous_stop_pct", 0.02)  # 즉시 -2%

        # MA20 기반 손절 버퍼
        self.ma20_stop_buffer = cfg.get("ma20_stop_buffer", 0.015)  # MA20 아래 1.5%

        # 최대 허용 pullback
        self.max_healthy_pullback_pct = cfg.get("max_healthy_pullback_pct", 0.08)

    def evaluate_pullback(
        self,
        df: pd.DataFrame,
        idx: int,
        entry_price: float,
        highest_price: float,
        trigger_type: str = "impulse",
    ) -> PullbackHealth:
        """
        현재 시점의 조정 건강도 평가.

        Args:
            df: 기술적 지표가 계산된 DataFrame
            idx: 분석 기준 행 인덱스
            entry_price: 진입가
            highest_price: 보유 중 최고가
            trigger_type: 트리거 유형 (impulse/confirm/breakout)
        """
        if df.empty or idx >= len(df) or idx < 0:
            idx = len(df) - 1

        row = df.iloc[idx]
        col = self._normalize_columns(df)
        close = float(row[col["close"]]) if col.get("close") else 0

        if close <= 0 or highest_price <= 0:
            return PullbackHealth(reason="데이터 부족")

        # 조정 컨텍스트
        pullback_pct = (highest_price - close) / highest_price
        pullback_days = self._count_pullback_days(df, col, idx, highest_price)

        # 현재 수익률
        pnl_pct = (close / entry_price - 1)

        score = 0.0
        reasons = []

        # ── 1. MA20 지지 (15점) ──
        ma20_support = False
        if col.get("ma20"):
            ma20 = self._safe_float(row.get(col["ma20"]))
            if ma20 and ma20 > 0:
                distance_to_ma20 = (close - ma20) / ma20
                if distance_to_ma20 >= 0:
                    # MA20 위에 있음 → 풀점
                    score += self.w_ma20
                    ma20_support = True
                    reasons.append(f"MA20 위 ({distance_to_ma20*100:+.1f}%)")
                elif distance_to_ma20 >= -0.02:
                    # MA20 바로 아래 (2% 이내) → 부분 점수
                    score += self.w_ma20 * 0.5
                    ma20_support = True
                    reasons.append(f"MA20 근처 지지 ({distance_to_ma20*100:+.1f}%)")
                else:
                    reasons.append(f"MA20 이탈 ({distance_to_ma20*100:+.1f}%)")

        # ── 2. MA60 지지 (10점) ──
        ma60_support = False
        if col.get("ma60"):
            ma60 = self._safe_float(row.get(col["ma60"]))
            if ma60 and ma60 > 0:
                distance_to_ma60 = (close - ma60) / ma60
                if distance_to_ma60 >= 0:
                    score += self.w_ma60
                    ma60_support = True
                    reasons.append(f"MA60 위 ({distance_to_ma60*100:+.1f}%)")
                elif distance_to_ma60 >= -0.03:
                    score += self.w_ma60 * 0.3
                    reasons.append(f"MA60 근처 ({distance_to_ma60*100:+.1f}%)")
                else:
                    reasons.append(f"MA60 이탈 ({distance_to_ma60*100:+.1f}%)")

        # ── 3. ADX 추세 유지 (12점) ──
        adx_val = 0.0
        adx_trending = False
        if col.get("adx"):
            adx_val = self._safe_float(row.get(col["adx"])) or 0.0
            if adx_val >= 25:
                score += self.w_adx
                adx_trending = True
                reasons.append(f"ADX {adx_val:.0f} (추세 유지)")
            elif adx_val >= 20:
                score += self.w_adx * 0.5
                reasons.append(f"ADX {adx_val:.0f} (추세 약화)")
            else:
                reasons.append(f"ADX {adx_val:.0f} (추세 없음)")

        # ── 4. +DI > -DI (10점) ──
        di_positive = False
        if col.get("plus_di") and col.get("minus_di"):
            pdi = self._safe_float(row.get(col["plus_di"])) or 0
            mdi = self._safe_float(row.get(col["minus_di"])) or 0
            if pdi > mdi:
                score += self.w_di
                di_positive = True
                reasons.append(f"+DI({pdi:.0f}) > -DI({mdi:.0f})")
            elif pdi > mdi * 0.85:
                score += self.w_di * 0.3
                reasons.append(f"+DI({pdi:.0f}) ≈ -DI({mdi:.0f}) 근접")
            else:
                reasons.append(f"+DI({pdi:.0f}) < -DI({mdi:.0f}) 역전")

        # ── 5. OBV 기울기 유지 (15점) ──
        obv_intact = False
        if col.get("obv") and len(df) > 10:
            obv_slope = self._calc_obv_slope(df, col["obv"], idx, days=10)
            if obv_slope > 0:
                score += self.w_obv
                obv_intact = True
                reasons.append("OBV 상승 유지 (매집 계속)")
            elif obv_slope > -0.001:
                score += self.w_obv * 0.5
                obv_intact = True
                reasons.append("OBV 횡보 (매집 유지)")
            else:
                reasons.append("OBV 하락 (분배 징후)")

        # ── 6. 거래량 감소 (10점) ──
        volume_declining = False
        if col.get("volume") and len(df) > 5:
            vol_trend = self._calc_volume_trend(df, col["volume"], idx, pullback_days)
            if vol_trend < 0:
                score += self.w_vol
                volume_declining = True
                reasons.append("거래량 감소 (건강한 조정)")
            elif vol_trend < 0.3:
                score += self.w_vol * 0.3
                reasons.append("거래량 소폭 증가")
            else:
                reasons.append("거래량 증가 (매도 압력)")

        # ── 7. 기관 수급 (8점) ──
        inst_ok = False
        if col.get("inst_net"):
            inst_streak = self._calc_recent_streak(df, col["inst_net"], idx, days=3)
            if inst_streak >= 0:
                score += self.w_inst
                inst_ok = True
                if inst_streak > 0:
                    reasons.append(f"기관 순매수 {inst_streak}일")
            else:
                reasons.append(f"기관 순매도 {abs(inst_streak)}일")

        # ── 8. 외국인 수급 (8점) ──
        foreign_ok = False
        if col.get("foreign_net"):
            foreign_streak = self._calc_recent_streak(df, col["foreign_net"], idx, days=3)
            if foreign_streak >= 0:
                score += self.w_foreign
                foreign_ok = True
                if foreign_streak > 0:
                    reasons.append(f"외국인 순매수 {foreign_streak}일")
            else:
                reasons.append(f"외국인 순매도 {abs(foreign_streak)}일")

        # ── 9. RSI 건강도 (7점) ──
        rsi_val = 50.0
        rsi_ok = True
        if col.get("rsi"):
            rsi_val = self._safe_float(row.get(col["rsi"])) or 50.0
            if 30 <= rsi_val <= 70:
                score += self.w_rsi
                reasons.append(f"RSI {rsi_val:.0f} (정상)")
            elif rsi_val > 25:
                score += self.w_rsi * 0.5
                reasons.append(f"RSI {rsi_val:.0f} (과매도 접근)")
            else:
                rsi_ok = False
                reasons.append(f"RSI {rsi_val:.0f} (과매도)")

        # ── 10. 조정 깊이 보정 (5점) ──
        if pullback_pct <= 0.03:
            score += self.w_pullback_depth
            reasons.append(f"조정 {pullback_pct*100:.1f}% (얕은 조정)")
        elif pullback_pct <= 0.05:
            score += self.w_pullback_depth * 0.6
            reasons.append(f"조정 {pullback_pct*100:.1f}% (적정)")
        elif pullback_pct <= self.max_healthy_pullback_pct:
            score += self.w_pullback_depth * 0.2
            reasons.append(f"조정 {pullback_pct*100:.1f}% (깊은 조정)")
        else:
            reasons.append(f"조정 {pullback_pct*100:.1f}% (과도)")

        # ── 분류 ──
        if score >= self.healthy_threshold:
            classification = "healthy"
        elif score >= self.caution_threshold:
            classification = "caution"
        elif score >= self.dangerous_threshold:
            classification = "dangerous"
        else:
            classification = "critical"

        # ── 적응형 손절가 계산 ──
        adjusted_stop_pct, adjusted_stop_price = self._calc_adaptive_stop(
            df, col, idx, row, entry_price, highest_price,
            classification, trigger_type,
        )

        return PullbackHealth(
            health_score=round(score, 1),
            classification=classification,
            ma20_support=ma20_support,
            ma60_support=ma60_support,
            adx_value=adx_val,
            adx_trending=adx_trending,
            di_positive=di_positive,
            obv_intact=obv_intact,
            volume_declining=volume_declining,
            inst_not_selling=inst_ok,
            foreign_not_selling=foreign_ok,
            rsi_value=rsi_val,
            rsi_not_oversold=rsi_ok,
            pullback_pct=round(pullback_pct * 100, 2),
            pullback_days=pullback_days,
            reasons=reasons,
            adjusted_stop_pct=round(adjusted_stop_pct, 4),
            adjusted_stop_price=round(adjusted_stop_price, 0),
        )

    # ──────────────────────────────────────────
    # 적응형 손절가 계산
    # ──────────────────────────────────────────

    def _calc_adaptive_stop(
        self,
        df: pd.DataFrame,
        col: dict,
        idx: int,
        row,
        entry_price: float,
        highest_price: float,
        classification: str,
        trigger_type: str,
    ) -> tuple[float, float]:
        """
        건강도에 따른 적응형 손절가 계산.

        healthy → MA20 기반 (-1.5%) 또는 ATR*2 중 더 넓은 값
        caution → 기존 손절 유지 (impulse -3%, confirm -5%)
        dangerous/critical → 타이트 -2% 또는 즉시 청산
        """
        close = float(row[col["close"]]) if col.get("close") else entry_price

        # 기본 손절 (기존 로직)
        base_stop_pct = 0.03 if trigger_type == "impulse" else 0.05

        if classification == "healthy":
            # 건강한 조정 = 추세 유지 → 넓은 손절로 버텨야 이후 상승 포착
            #
            # 핵심 철학: 건강한 조정이 확인되면 MA20 이탈도 허용
            # MA20은 "지지 구간"이지 "절대 손절선"이 아님
            #
            # 방법1: 진입가 기준 넓은 손절 (impulse: -6%, confirm: -8%)
            healthy_pct = 0.06 if trigger_type == "impulse" else 0.08
            pct_stop_price = entry_price * (1 - healthy_pct)

            # 방법2: ATR 기반 넓은 손절 (entry - ATR*3)
            atr_stop_price = 0
            if col.get("atr"):
                atr = self._safe_float(row.get(col["atr"]))
                if atr and atr > 0:
                    atr_stop_price = entry_price - atr * 3.0

            # 두 방법 중 더 넓은(낮은) 값 채택 → 추세에 더 많은 여유
            stop_candidates = [p for p in [pct_stop_price, atr_stop_price] if p > 0]
            if stop_candidates:
                stop_price = min(stop_candidates)  # 더 넓은 손절
                stop_pct = max(0.02, (entry_price - stop_price) / entry_price)
                stop_pct = min(stop_pct, self.max_healthy_pullback_pct)
                return stop_pct, entry_price * (1 - stop_pct)

            # fallback
            return healthy_pct, entry_price * (1 - healthy_pct)

        elif classification == "caution":
            # 기존 손절 유지
            return base_stop_pct, entry_price * (1 - base_stop_pct)

        elif classification == "dangerous":
            # 타이트 손절
            tight_pct = self.dangerous_stop_pct
            return tight_pct, entry_price * (1 - tight_pct)

        else:  # critical
            # 즉시 청산 (현재가 기준)
            return 0.0, close

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> dict:
        mapping = {}
        cols = {c.lower(): c for c in df.columns}
        for key, candidates in {
            "close": ["close"],
            "high": ["high"],
            "low": ["low"],
            "volume": ["volume"],
            "ma20": ["sma_20", "ma20", "ma_20"],
            "ma60": ["sma_60", "ma60", "ma_60"],
            "adx": ["adx", "adx_14"],
            "plus_di": ["plus_di", "+di", "di_plus", "pdi"],
            "minus_di": ["minus_di", "-di", "di_minus", "mdi"],
            "obv": ["obv"],
            "rsi": ["rsi", "rsi_14"],
            "atr": ["atr", "atr_14"],
            "inst_net": ["inst_net", "instnet"],
            "foreign_net": ["foreign_net", "foreignnet"],
        }.items():
            for c in candidates:
                if c in cols:
                    mapping[key] = cols[c]
                    break
        return mapping

    @staticmethod
    def _safe_float(val) -> float | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _count_pullback_days(
        df: pd.DataFrame, col: dict, idx: int, highest_price: float,
    ) -> int:
        """고점 이후 조정 지속일 계산"""
        if not col.get("high"):
            return 0
        days = 0
        for i in range(idx, max(idx - 30, -1), -1):
            h = float(df.iloc[i][col["high"]])
            if h >= highest_price * 0.998:
                break
            days += 1
        return days

    @staticmethod
    def _calc_obv_slope(
        df: pd.DataFrame, obv_col: str, idx: int, days: int = 10,
    ) -> float:
        """최근 N일 OBV 선형 회귀 기울기"""
        start = max(0, idx - days + 1)
        obv = df[obv_col].iloc[start:idx + 1].astype(float).values
        if len(obv) < 5 or np.isnan(obv).any():
            return 0.0
        x = np.arange(len(obv))
        slope = np.polyfit(x, obv, 1)[0]
        mean_obv = abs(np.mean(obv)) + 1
        return slope / mean_obv

    @staticmethod
    def _calc_volume_trend(
        df: pd.DataFrame, vol_col: str, idx: int, pullback_days: int,
    ) -> float:
        """조정 기간 중 거래량 추세 (-1=감소, 0=유지, +1=증가)"""
        days = max(pullback_days, 3)
        start = max(0, idx - days + 1)
        vol = df[vol_col].iloc[start:idx + 1].astype(float).values
        if len(vol) < 3:
            return 0.0
        x = np.arange(len(vol))
        slope = np.polyfit(x, vol, 1)[0]
        mean_vol = np.mean(vol) + 1
        return slope / mean_vol

    @staticmethod
    def _calc_recent_streak(
        df: pd.DataFrame, col: str, idx: int, days: int = 3,
    ) -> int:
        """최근 N일 기준 연속 양수/음수 일수"""
        if col not in df.columns:
            return 0
        start = max(0, idx - days + 1)
        vals = df[col].iloc[start:idx + 1].astype(float).values
        streak = 0
        for v in reversed(vals):
            if v > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif v < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        return streak
