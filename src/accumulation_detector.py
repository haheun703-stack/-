"""
v3.1 매집 3단계 감지기 (SMAD)

Phase 1 (초기): OBV 다이버전스 + 주가 횡보 → +5점
Phase 2 (중기): 기관/외국인 5일+ 연속 순매수 + 120MA 지지 → +10점
Phase 3 (점화): 기관+외국인 동시 10일+ 순매수 + 거래량 폭발 → +15점
Dumping: 기관+외국인 동시 5일+ 순매도 → -20점 (최우선 체크)

엔티티(news_models.AccumulationSignal)에만 의존.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.entities.news_models import AccumulationSignal

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_sm_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("smart_money_v2", {})


class AccumulationDetector:
    """매집 3단계 + 투매 감지 엔진"""

    def __init__(self, config_path: Path | str | None = None):
        cfg = _load_sm_config(config_path)
        self.inst_days = cfg.get("inst_net_buy_days", 5)
        self.foreign_days = cfg.get("foreign_net_buy_days", 5)
        self.strong_days = cfg.get("inst_net_buy_strong_days", 10)
        self.phase3_vol_ratio = cfg.get("phase3_vol_ratio", 3.0)
        self.ma120_buffer = cfg.get("ma120_support_buffer", 0.03)
        self.bonus_p1 = cfg.get("bonus_phase1", 5)
        self.bonus_p2 = cfg.get("bonus_phase2", 10)
        self.bonus_p3 = cfg.get("bonus_phase3", 15)
        self.penalty_dump = cfg.get("penalty_dump", -20)

    def detect(self, df: pd.DataFrame, idx: int = -1) -> AccumulationSignal:
        """
        매집 단계 판정.

        Args:
            df: 기술적 지표가 계산된 DataFrame.
                필요 컬럼: Close, MA120, OBV, Volume, Foreign_Net, Inst_Net
            idx: 분석 기준 행 인덱스 (기본: 마지막 행)
        """
        if df.empty or len(df) < 20:
            return AccumulationSignal(reason="데이터 부족")

        # 컬럼명 정규화 (대소문자 모두 지원)
        col_map = self._normalize_columns(df)
        if not col_map.get("close"):
            return AccumulationSignal(reason="Close 컬럼 없음")

        row = df.iloc[idx]
        current_price = float(row[col_map["close"]])

        # 수급 스트릭 계산
        inst_streak = self._calc_streak(df, col_map.get("inst_net")) if col_map.get("inst_net") else 0
        foreign_streak = self._calc_streak(df, col_map.get("foreign_net")) if col_map.get("foreign_net") else 0
        flow_pattern = self._get_flow_pattern(inst_streak, foreign_streak)

        # 120MA 근처 확인
        near_ma120 = False
        ma120_val = None
        if col_map.get("ma120"):
            ma120_val = self._safe_float(row.get(col_map["ma120"]))
            if ma120_val and ma120_val > 0:
                distance = abs(current_price - ma120_val) / ma120_val
                near_ma120 = distance <= self.ma120_buffer

        # ═══════════════════════════════════════
        # 투매 감지 (최우선)
        # ═══════════════════════════════════════
        if flow_pattern == "both_selling":
            if abs(inst_streak) >= 5 and abs(foreign_streak) >= 5:
                return AccumulationSignal(
                    phase="dumping",
                    phase_name="투매 감지",
                    confidence=85.0,
                    reasons=[
                        f"기관 {abs(inst_streak)}일 연속 순매도",
                        f"외국인 {abs(foreign_streak)}일 연속 순매도",
                        "기관+외국인 동시 투매 — 즉시 경고",
                    ],
                    inst_streak=inst_streak,
                    foreign_streak=foreign_streak,
                    flow_pattern=flow_pattern,
                    score_modifier=self.penalty_dump,
                    action="danger",
                )

        # ═══════════════════════════════════════
        # Phase 3: 점화 직전
        # ═══════════════════════════════════════
        if (flow_pattern == "both_buying"
                and inst_streak >= self.strong_days
                and foreign_streak >= self.strong_days):
            confidence = 40.0
            reasons = [
                f"기관+외국인 동시 순매수 {min(inst_streak, foreign_streak)}일+",
            ]

            # 거래량 폭발 확인
            if col_map.get("volume"):
                vol_ratio = self._calc_vol_ratio(df, col_map["volume"], idx)
                if vol_ratio >= self.phase3_vol_ratio:
                    confidence += 20.0
                    reasons.append(f"거래량 {vol_ratio:.1f}배 — 점화")

            if near_ma120:
                confidence += 10.0
                reasons.append(f"120MA 지지 구간에서 점화 ({current_price:,.0f} vs {ma120_val:,.0f})")

            return AccumulationSignal(
                phase="phase3",
                phase_name="점화 직전",
                confidence=min(100, confidence),
                reasons=reasons,
                inst_streak=inst_streak,
                foreign_streak=foreign_streak,
                flow_pattern=flow_pattern,
                score_modifier=self.bonus_p3,
                action="buy_ready",
            )

        # ═══════════════════════════════════════
        # Phase 2: 본격 축적
        # ═══════════════════════════════════════
        inst_buying = inst_streak >= self.inst_days
        foreign_buying = foreign_streak >= self.foreign_days

        if inst_buying or foreign_buying:
            confidence = 25.0
            reasons = []

            if inst_buying:
                reasons.append(f"기관 {inst_streak}일 연속 순매수")
            if foreign_buying:
                reasons.append(f"외국인 {foreign_streak}일 연속 순매수")
            if near_ma120:
                confidence += 15.0
                reasons.append(f"120MA 근처 매집 ({current_price:,.0f} vs {ma120_val:,.0f})")
            if flow_pattern == "both_buying":
                confidence += 10.0
                reasons.append("기관+외국인 동시 순매수 — 고신뢰")

            return AccumulationSignal(
                phase="phase2",
                phase_name="본격 축적",
                confidence=min(100, confidence),
                reasons=reasons,
                inst_streak=inst_streak,
                foreign_streak=foreign_streak,
                flow_pattern=flow_pattern,
                score_modifier=self.bonus_p2,
                action="prepare",
            )

        # ═══════════════════════════════════════
        # Phase 1: 초기 매집 (OBV 다이버전스)
        # ═══════════════════════════════════════
        if col_map.get("obv") and col_map.get("close") and len(df) >= 20:
            price_20d_change = self._calc_price_change(df, col_map["close"], 20)
            obv_rising = self._is_obv_rising(df, col_map["obv"], 20)

            # OBV 상승인데 주가 횡보(-2%~+2%) 또는 하락 = 조용한 매집
            if obv_rising and price_20d_change < 0.02:
                confidence = 20.0
                reasons = [
                    f"OBV 다이버전스: 주가 20일 {price_20d_change * 100:+.1f}% vs OBV 상승",
                ]
                if near_ma120:
                    confidence += 10.0
                    reasons.append("120MA 근처 — 장기 지지 구간에서 매집")

                return AccumulationSignal(
                    phase="phase1",
                    phase_name="초기 매집",
                    confidence=min(100, confidence),
                    reasons=reasons,
                    inst_streak=inst_streak,
                    foreign_streak=foreign_streak,
                    flow_pattern=flow_pattern,
                    score_modifier=self.bonus_p1,
                    action="watch",
                )

        # ═══════════════════════════════════════
        # 매집 미감지
        # ═══════════════════════════════════════
        return AccumulationSignal(
            inst_streak=inst_streak,
            foreign_streak=foreign_streak,
            flow_pattern=flow_pattern,
        )

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> dict:
        """컬럼명을 정규화하여 매핑 반환"""
        mapping = {}
        cols = {c.lower(): c for c in df.columns}
        for key, candidates in {
            "close": ["close"],
            "volume": ["volume"],
            "obv": ["obv"],
            "ma120": ["ma120", "ma_120", "sma_120"],
            "inst_net": ["inst_net", "instnet", "inst_buy_sell"],
            "foreign_net": ["foreign_net", "foreignnet", "foreign_buy_sell"],
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
    def _calc_streak(df: pd.DataFrame, col: str) -> int:
        """마지막 행 기준 연속 양수/음수 일수 계산"""
        if col not in df.columns:
            return 0
        vals = df[col].astype(float).values
        streak = 0
        for i in range(len(vals) - 1, -1, -1):
            if vals[i] > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif vals[i] < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        return streak

    @staticmethod
    def _get_flow_pattern(inst_streak: int, foreign_streak: int) -> str:
        if inst_streak > 0 and foreign_streak > 0:
            return "both_buying"
        elif inst_streak > 0 and foreign_streak <= 0:
            return "inst_only"
        elif inst_streak <= 0 and foreign_streak > 0:
            return "foreign_only"
        elif inst_streak < 0 and foreign_streak < 0:
            return "both_selling"
        return "unknown"

    @staticmethod
    def _calc_vol_ratio(df: pd.DataFrame, vol_col: str, idx: int) -> float:
        """최근 거래량 vs 20일 평균 비율"""
        if vol_col not in df.columns or len(df) < 21:
            return 1.0
        vol = df[vol_col].astype(float)
        ma20 = vol.rolling(20).mean()
        cur = vol.iloc[idx]
        avg = ma20.iloc[idx - 1] if abs(idx) < len(df) else ma20.iloc[-2]
        return cur / avg if avg and avg > 0 else 1.0

    @staticmethod
    def _calc_price_change(df: pd.DataFrame, close_col: str, days: int) -> float:
        if len(df) < days + 1:
            return 0.0
        c = df[close_col].astype(float)
        return (c.iloc[-1] - c.iloc[-days]) / c.iloc[-days]

    @staticmethod
    def _is_obv_rising(df: pd.DataFrame, obv_col: str, days: int) -> bool:
        if len(df) < days:
            return False
        import numpy as np
        obv = df[obv_col].astype(float).tail(days).values
        if np.isnan(obv).any():
            return False
        x = np.arange(len(obv))
        slope = np.polyfit(x, obv, 1)[0]
        return slope > 0
