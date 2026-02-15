"""
v4.1 일일 보유 점수 시스템 (DailyHoldScorer)

매일 포지션 건강 상태를 종합 평가하여 보유/청산 결정:
  70+ → STRONG_HOLD: 강력 보유 (손절 완화 + 보유일 연장)
  50~70 → HOLD: 보유 유지 (현행 파라미터)
  30~50 → TIGHTEN: 경계 (손절 강화 + 트레일링 타이트)
  <30 → EXIT: 청산 시그널 (다음 봉 매도)

평가 항목 (총 100점):
  기술적 건강 (40점): MA 정렬, ADX, RSI, MACD
  수급 건강 (30점): OBV, 거래량, 기관/외국인 수급
  뉴스/이슈 건강 (15점): 뉴스 등급, 살아있는 이슈
  포지션 건강 (15점): PnL 모멘텀, 조정 건강도

엔티티(trading_models.HoldScore)에만 의존.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.entities.trading_models import HoldScore
from src.use_cases.adaptive_exit import AdaptiveExitManager

_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("daily_hold_scorer", {})


class DailyHoldScorer:
    """
    매일 포지션 건강 상태를 종합 평가.

    사용법:
      scorer = DailyHoldScorer()
      hold = scorer.score(df, idx, entry_price, highest_price, news_grade, trigger_type)
      if hold.action == "exit":
          → 청산 실행
      elif hold.action == "tighten":
          → 손절 강화 (stop_adjustment < 1.0)
      elif hold.action == "strong_hold":
          → 손절 완화 (stop_adjustment > 1.0), 보유일 연장
    """

    def __init__(self, config_path: Path | str | None = None):
        cfg = _load_config(config_path)
        self.adaptive_exit = AdaptiveExitManager(config_path)

        # 등급별 임계값
        self.strong_hold_threshold = cfg.get("strong_hold_threshold", 70)
        self.hold_threshold = cfg.get("hold_threshold", 50)
        self.tighten_threshold = cfg.get("tighten_threshold", 30)

        # 파라미터 조정 배율
        self.strong_hold_stop_mult = cfg.get("strong_hold_stop_mult", 1.5)
        self.strong_hold_days_bonus = cfg.get("strong_hold_days_bonus", 5)
        self.strong_hold_trailing_mult = cfg.get("strong_hold_trailing_mult", 1.3)
        self.tighten_stop_mult = cfg.get("tighten_stop_mult", 0.6)
        self.tighten_trailing_mult = cfg.get("tighten_trailing_mult", 0.7)

    def score(
        self,
        df: pd.DataFrame,
        idx: int,
        entry_price: float,
        highest_price: float,
        trigger_type: str = "impulse",
        news_grade: str = "",
        living_issues: list | None = None,
        hold_days: int = 0,
    ) -> HoldScore:
        """
        일일 보유 점수 산출.

        Args:
            df: 기술적 지표 DataFrame
            idx: 기준 행
            entry_price: 진입가
            highest_price: 보유 중 최고가
            trigger_type: impulse/confirm/breakout
            news_grade: 뉴스 등급 (A/B/C)
            living_issues: 살아있는 이슈 리스트
            hold_days: 현재 보유일수
        """
        if df.empty or idx >= len(df):
            return HoldScore(total_score=50, action="hold", reasons=["데이터 부족"])

        row = df.iloc[idx]
        col = self._normalize_columns(df)
        close = float(row[col["close"]]) if col.get("close") else 0
        reasons = []

        # ═══════════════════════════════════════
        # 1. 기술적 건강 (40점)
        # ═══════════════════════════════════════
        tech_score = self._score_technical(df, col, row, idx, reasons)

        # ═══════════════════════════════════════
        # 2. 수급 건강 (30점)
        # ═══════════════════════════════════════
        sd_score = self._score_supply_demand(df, col, row, idx, reasons)

        # ═══════════════════════════════════════
        # 3. 뉴스/이슈 건강 (15점)
        # ═══════════════════════════════════════
        news_score = self._score_news_issues(news_grade, living_issues, reasons)

        # ═══════════════════════════════════════
        # 4. 포지션 건강 (15점)
        # ═══════════════════════════════════════
        pos_score = self._score_position_health(
            df, col, idx, entry_price, highest_price, trigger_type, hold_days, reasons,
        )

        total = round(tech_score + sd_score + news_score + pos_score, 1)

        # ── 판단 ──
        if total >= self.strong_hold_threshold:
            action = "strong_hold"
            stop_adj = self.strong_hold_stop_mult
            days_adj = self.strong_hold_days_bonus
            trail_adj = self.strong_hold_trailing_mult
            reasons.insert(0, f"★ STRONG HOLD ({total:.0f}점) — 추세 지속, 손절 완화")
        elif total >= self.hold_threshold:
            action = "hold"
            stop_adj = 1.0
            days_adj = 0
            trail_adj = 1.0
            reasons.insert(0, f"● HOLD ({total:.0f}점) — 현행 유지")
        elif total >= self.tighten_threshold:
            action = "tighten"
            stop_adj = self.tighten_stop_mult
            days_adj = -3
            trail_adj = self.tighten_trailing_mult
            reasons.insert(0, f"▲ TIGHTEN ({total:.0f}점) — 손절 강화, 경계")
        else:
            action = "exit"
            stop_adj = 0.0
            days_adj = 0
            trail_adj = 0.0
            reasons.insert(0, f"✕ EXIT ({total:.0f}점) — 청산 시그널")

        return HoldScore(
            total_score=total,
            action=action,
            technical_score=round(tech_score, 1),
            supply_demand_score=round(sd_score, 1),
            news_issue_score=round(news_score, 1),
            position_health_score=round(pos_score, 1),
            reasons=reasons,
            stop_adjustment=stop_adj,
            hold_days_adjustment=days_adj,
            trailing_tightness=trail_adj,
        )

    # ──────────────────────────────────────────
    # 1. 기술적 건강 (40점)
    # ──────────────────────────────────────────

    def _score_technical(
        self, df: pd.DataFrame, col: dict, row, idx: int, reasons: list,
    ) -> float:
        score = 0.0

        # MA 정렬 (12점)
        ma_score = 0.0
        if col.get("ma20") and col.get("ma60") and col.get("close"):
            close = float(row[col["close"]])
            ma20 = self._sf(row.get(col["ma20"])) or 0
            ma60 = self._sf(row.get(col["ma60"])) or 0
            ma120 = self._sf(row.get(col.get("ma120", ""))) if col.get("ma120") else 0

            if close > ma20 > 0:
                ma_score += 4
            if ma20 > ma60 > 0:
                ma_score += 4
            if ma120 and ma60 > ma120 > 0:
                ma_score += 4
            elif ma120 is None or ma120 == 0:
                # MA120 없으면 비율 보정
                if close > ma20 > ma60 > 0:
                    ma_score += 4

            if ma_score >= 8:
                reasons.append(f"MA정렬 우수 ({ma_score:.0f}/12)")
            elif ma_score >= 4:
                reasons.append(f"MA정렬 부분 ({ma_score:.0f}/12)")
        score += ma_score

        # ADX 강도 (10점)
        if col.get("adx"):
            adx = self._sf(row.get(col["adx"])) or 0
            if adx >= 30:
                score += 10
                reasons.append(f"ADX {adx:.0f} (강한 추세)")
            elif adx >= 25:
                score += 7
            elif adx >= 20:
                score += 4
            else:
                reasons.append(f"ADX {adx:.0f} (추세 약함)")

        # RSI 위치 (10점)
        if col.get("rsi"):
            rsi = self._sf(row.get(col["rsi"])) or 50
            if 45 <= rsi <= 65:
                score += 10
            elif 35 <= rsi <= 75:
                score += 7
            elif 25 <= rsi <= 80:
                score += 4
                if rsi > 75:
                    reasons.append(f"RSI {rsi:.0f} (과매수 주의)")
            else:
                if rsi < 25:
                    reasons.append(f"RSI {rsi:.0f} (과매도)")
                else:
                    reasons.append(f"RSI {rsi:.0f} (극단)")

        # MACD 방향 (8점)
        if col.get("macd") and col.get("macd_signal"):
            macd = self._sf(row.get(col["macd"])) or 0
            signal = self._sf(row.get(col["macd_signal"])) or 0
            if macd > signal:
                score += 8
            elif macd > signal * 0.95:
                score += 4
                reasons.append("MACD 교차 임박")
            else:
                reasons.append("MACD 하향")

        return min(40, score)

    # ──────────────────────────────────────────
    # 2. 수급 건강 (30점)
    # ──────────────────────────────────────────

    def _score_supply_demand(
        self, df: pd.DataFrame, col: dict, row, idx: int, reasons: list,
    ) -> float:
        score = 0.0

        # OBV 추세 (10점)
        if col.get("obv") and len(df) > 10:
            obv_slope = self._calc_obv_slope(df, col["obv"], idx, 10)
            if obv_slope > 0.001:
                score += 10
                reasons.append("OBV 상승 (매집)")
            elif obv_slope > -0.001:
                score += 5
            else:
                reasons.append("OBV 하락 (분배)")

        # 거래량 패턴 (8점)
        if col.get("volume") and len(df) > 20:
            vol = df[col["volume"]].astype(float)
            cur_vol = vol.iloc[idx]
            avg_vol = vol.iloc[max(0, idx-20):idx].mean()
            if avg_vol > 0:
                vol_ratio = cur_vol / avg_vol
                if 0.5 <= vol_ratio <= 1.5:
                    score += 8
                elif vol_ratio < 0.5:
                    score += 4
                    reasons.append("거래량 급감 (관심 이탈)")
                else:
                    score += 4
                    reasons.append(f"거래량 급증 ({vol_ratio:.1f}배)")

        # 기관 수급 (6점)
        if col.get("inst_net"):
            inst_streak = self._calc_streak(df, col["inst_net"], idx)
            if inst_streak >= 3:
                score += 6
                reasons.append(f"기관 {inst_streak}일 연속 순매수")
            elif inst_streak >= 0:
                score += 3
            else:
                if inst_streak <= -3:
                    reasons.append(f"기관 {abs(inst_streak)}일 연속 순매도")

        # 외국인 수급 (6점)
        if col.get("foreign_net"):
            foreign_streak = self._calc_streak(df, col["foreign_net"], idx)
            if foreign_streak >= 3:
                score += 6
                reasons.append(f"외국인 {foreign_streak}일 연속 순매수")
            elif foreign_streak >= 0:
                score += 3
            else:
                if foreign_streak <= -3:
                    reasons.append(f"외국인 {abs(foreign_streak)}일 연속 순매도")

        return min(30, score)

    # ──────────────────────────────────────────
    # 3. 뉴스/이슈 건강 (15점)
    # ──────────────────────────────────────────

    def _score_news_issues(
        self, news_grade: str, living_issues: list | None, reasons: list,
    ) -> float:
        score = 7.5  # 뉴스 없으면 중립 (15점 만점의 절반)

        # 뉴스 등급 (7점)
        if news_grade == "A":
            score = 12
            reasons.append("A급 뉴스 (확정 호재)")
        elif news_grade == "B":
            score = 10
            reasons.append("B급 뉴스 (루머 호재)")

        # 살아있는 이슈 (최대 3점 추가)
        if living_issues:
            active_positive = sum(
                1 for li in living_issues
                if getattr(li, "status", "") == "active"
                and getattr(li, "sentiment", "") == "positive"
            )
            high_impact = sum(
                1 for li in living_issues
                if getattr(li, "impact_score", 0) >= 8
                and getattr(li, "status", "") == "active"
            )
            if high_impact > 0:
                score += 3
                reasons.append(f"고영향 살아있는 이슈 {high_impact}건")
            elif active_positive > 0:
                score += 1.5
                reasons.append(f"긍정 이슈 {active_positive}건 진행 중")

        # 부정 뉴스 감점
        if living_issues:
            negative_active = sum(
                1 for li in living_issues
                if getattr(li, "status", "") == "active"
                and getattr(li, "sentiment", "") == "negative"
                and getattr(li, "impact_score", 0) >= 7
            )
            if negative_active > 0:
                score -= 5
                reasons.append(f"부정 이슈 {negative_active}건 (위험)")

        return max(0, min(15, score))

    # ──────────────────────────────────────────
    # 4. 포지션 건강 (15점)
    # ──────────────────────────────────────────

    def _score_position_health(
        self,
        df: pd.DataFrame,
        col: dict,
        idx: int,
        entry_price: float,
        highest_price: float,
        trigger_type: str,
        hold_days: int,
        reasons: list,
    ) -> float:
        score = 0.0
        close = float(df.iloc[idx][col["close"]]) if col.get("close") else 0

        if close <= 0 or entry_price <= 0:
            return 7.5

        pnl_pct = (close / entry_price - 1)

        # PnL 상태 (8점)
        if pnl_pct >= 0.10:
            score += 8
            reasons.append(f"수익 {pnl_pct*100:+.1f}% (안전 구간)")
        elif pnl_pct >= 0.05:
            score += 6
        elif pnl_pct >= 0:
            score += 4
        elif pnl_pct >= -0.03:
            score += 2
            reasons.append(f"수익 {pnl_pct*100:+.1f}% (손실 접근)")
        else:
            reasons.append(f"수익 {pnl_pct*100:+.1f}% (손실 구간)")

        # 고점 대비 되돌림 (4점)
        if highest_price > 0:
            drawdown = (highest_price - close) / highest_price
            if drawdown <= 0.02:
                score += 4
            elif drawdown <= 0.05:
                score += 2
            elif drawdown <= 0.08:
                score += 1
            else:
                reasons.append(f"고점 대비 -{drawdown*100:.1f}% 되돌림")

        # 보유일 리스크 (3점)
        max_hold = 10
        if hold_days <= max_hold * 0.5:
            score += 3
        elif hold_days <= max_hold * 0.8:
            score += 1.5
        else:
            reasons.append(f"보유 {hold_days}일 (만기 접근)")

        # v6.0 WaveLSFormer: Sharpe 보너스/패널티
        sharpe_adj = self._calc_sharpe_bonus(df, col, idx, hold_days, reasons)
        score += sharpe_adj

        return min(15, max(0, score))

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
            "volume": ["volume"],
            "ma20": ["sma_20", "ma20", "ma_20"],
            "ma60": ["sma_60", "ma60", "ma_60"],
            "ma120": ["sma_120", "ma120", "ma_120"],
            "adx": ["adx", "adx_14"],
            "rsi": ["rsi", "rsi_14"],
            "macd": ["macd"],
            "macd_signal": ["macd_signal", "macdsignal", "signal"],
            "obv": ["obv"],
            "inst_net": ["inst_net", "instnet"],
            "foreign_net": ["foreign_net", "foreignnet"],
        }.items():
            for c in candidates:
                if c in cols:
                    mapping[key] = cols[c]
                    break
        return mapping

    @staticmethod
    def _sf(val) -> float | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _calc_obv_slope(df, obv_col, idx, days=10) -> float:
        start = max(0, idx - days + 1)
        obv = df[obv_col].iloc[start:idx + 1].astype(float).values
        if len(obv) < 5 or np.isnan(obv).any():
            return 0.0
        x = np.arange(len(obv))
        slope = np.polyfit(x, obv, 1)[0]
        return slope / (abs(np.mean(obv)) + 1)

    def _calc_sharpe_bonus(
        self,
        df: pd.DataFrame,
        col: dict,
        idx: int,
        hold_days: int,
        reasons: list,
    ) -> float:
        """v6.0 WaveLSFormer: Sharpe 기반 보너스/패널티.

        보유 구간 수익률의 Sharpe ratio로 포지션 건강 보정:
          Sharpe > 1.5  → +3점 (강한 양의 리스크 조정 수익)
          Sharpe > 0.5  → +1.5점 (적절한 리스크 조정 수익)
          Sharpe < -0.5 → -2점 (리스크 대비 손실)
        """
        if hold_days < 5 or not col.get("close"):
            return 0.0

        lookback = min(hold_days, 20)
        start = max(0, idx - lookback + 1)
        prices = df[col["close"]].iloc[start:idx + 1].astype(float)

        if len(prices) < 5:
            return 0.0

        returns = prices.pct_change().dropna()
        if len(returns) < 4 or returns.std() == 0:
            return 0.0

        sharpe = float(returns.mean() / returns.std()) * (252 ** 0.5)

        if sharpe > 1.5:
            reasons.append(f"Sharpe {sharpe:.2f} (우수 → +3)")
            return 3.0
        elif sharpe > 0.5:
            reasons.append(f"Sharpe {sharpe:.2f} (양호 → +1.5)")
            return 1.5
        elif sharpe < -0.5:
            reasons.append(f"Sharpe {sharpe:.2f} (부진 → -2)")
            return -2.0

        return 0.0

    @staticmethod
    def _calc_streak(df, col, idx) -> int:
        if col not in df.columns:
            return 0
        vals = df[col].astype(float).values[:idx + 1]
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
