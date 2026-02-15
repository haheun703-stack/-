"""
v3.2 시장 시그널 스캐너

골든크로스 임박/확정, 스마트머니 최적, 추세 가속, 정배열 등
경쟁 AI 매매 시스템 수준의 핵심 인사이트를 감지하여 알림 생성.

엔티티(news_models)에만 의존 — 외부 API 없음.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.entities.news_models import MarketSignal


class MarketSignalScanner:
    """시장 시그널 통합 스캐너"""

    def scan_all(self, df: pd.DataFrame, idx: int = -1) -> list[MarketSignal]:
        """전체 시그널 스캔. 중요도 높은 순 정렬."""
        if df.empty or len(df) < 60:
            return []

        signals = []

        signals.extend(self._scan_golden_cross(df, idx))
        signals.extend(self._scan_ma_alignment(df, idx))
        signals.extend(self._scan_smart_money(df, idx))
        signals.extend(self._scan_macd_signals(df, idx))
        signals.extend(self._scan_bollinger_squeeze(df, idx))
        signals.extend(self._scan_trend_acceleration(df, idx))

        # 중요도 순 정렬
        importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        signals.sort(key=lambda s: (importance_order.get(s.importance, 9), -s.confidence))

        return signals

    # ──────────────────────────────────────────
    # 1. 골든크로스 / 데드크로스 감지
    # ──────────────────────────────────────────

    def _scan_golden_cross(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """이동평균선 골든크로스/데드크로스 감지."""
        signals = []
        row = df.iloc[idx]

        ma20 = self._get_val(row, "MA20", "sma_20")
        ma60 = self._get_val(row, "MA60", "sma_60")
        ma120 = self._get_val(row, "MA120", "sma_120")

        if pd.isna(ma20) or pd.isna(ma60):
            return signals

        gap_pct = (ma20 - ma60) / ma60 * 100  # 양수면 MA20 > MA60

        # MA20 슬로프 (5일)
        ma20_5ago = self._get_val_at(df, idx - 5, "MA20", "sma_20")
        ma60_5ago = self._get_val_at(df, idx - 5, "MA60", "sma_60")

        ma20_slope = 0
        ma60_slope = 0
        if not pd.isna(ma20_5ago) and ma20_5ago > 0:
            ma20_slope = (ma20 - ma20_5ago) / ma20_5ago * 100
        if not pd.isna(ma60_5ago) and ma60_5ago > 0:
            ma60_slope = (ma60 - ma60_5ago) / ma60_5ago * 100

        converging = ma20_slope > ma60_slope  # MA20이 더 빨리 상승

        data = {
            "ma20": round(ma20, 0),
            "ma60": round(ma60, 0),
            "gap_pct": round(gap_pct, 2),
            "ma20_slope_5d": round(ma20_slope, 3),
            "ma60_slope_5d": round(ma60_slope, 3),
        }

        # ── 골든크로스 확정 (MA20가 최근 5일 내 MA60 상향 돌파) ──
        if gap_pct > 0 and gap_pct < 3.0:
            # 5일 전에는 MA20 < MA60 이었는지 확인
            if not pd.isna(ma20_5ago) and not pd.isna(ma60_5ago):
                gap_5ago = (ma20_5ago - ma60_5ago) / ma60_5ago * 100
                if gap_5ago <= 0:
                    signals.append(MarketSignal(
                        signal_type="golden_cross_confirmed",
                        title="골든크로스 확정!",
                        description=(
                            f"MA20({ma20:,.0f})이 MA60({ma60:,.0f})을 "
                            f"상향 돌파 완료 (현재 갭: +{gap_pct:.1f}%). "
                            f"본격 상승 추세 진입 신호."
                        ),
                        importance="critical",
                        confidence=85 + min(gap_pct * 5, 10),
                        data=data,
                    ))
                    return signals

        # ── 골든크로스 임박 (MA20 < MA60이지만 빠르게 수렴 중) ──
        if gap_pct < 0 and gap_pct > -3.0 and converging:
            abs_gap = abs(gap_pct)
            if abs_gap < 1.0:
                importance = "critical"
                confidence = 90 - abs_gap * 20
                title = "골든크로스 임박!"
                desc = (
                    f"MA20({ma20:,.0f})이 MA60({ma60:,.0f})에 "
                    f"급속 수렴 중 (갭: {gap_pct:.2f}%). "
                    f"MA20 상승속도({ma20_slope:.2f}%/5d) > "
                    f"MA60({ma60_slope:.2f}%/5d). "
                    f"1~3일 내 골든크로스 예상."
                )
            elif abs_gap < 2.0:
                importance = "high"
                confidence = 75 - abs_gap * 10
                title = "골든크로스 접근"
                desc = (
                    f"MA20({ma20:,.0f})과 MA60({ma60:,.0f}) 수렴 진행 "
                    f"(갭: {gap_pct:.2f}%). 수렴 가속 중."
                )
            else:
                importance = "medium"
                confidence = 60
                title = "이평선 수렴 감지"
                desc = (
                    f"MA20-MA60 갭({gap_pct:.2f}%) 축소 중. "
                    f"아직 골든크로스까지 거리 있음."
                )

            signals.append(MarketSignal(
                signal_type="golden_cross_imminent",
                title=title,
                description=desc,
                importance=importance,
                confidence=max(confidence, 50),
                data=data,
            ))

        # ── 데드크로스 임박 (MA20 > MA60이지만 하락 수렴 중) ──
        if gap_pct > 0 and gap_pct < 2.0 and not converging:
            signals.append(MarketSignal(
                signal_type="dead_cross_warning",
                title="데드크로스 경고",
                description=(
                    f"MA20({ma20:,.0f})-MA60({ma60:,.0f}) 갭 축소 중 "
                    f"({gap_pct:.2f}%). MA20 둔화 추세."
                ),
                importance="high",
                confidence=70,
                data=data,
            ))

        # ── MA20-MA120 골든크로스 (장기 추세 전환) ──
        if not pd.isna(ma120):
            gap_120 = (ma20 - ma120) / ma120 * 100
            if gap_120 > 0 and gap_120 < 2.0:
                ma20_10ago = self._get_val_at(df, idx - 10, "MA20", "sma_20")
                ma120_10ago = self._get_val_at(df, idx - 10, "MA120", "sma_120")
                if not pd.isna(ma20_10ago) and not pd.isna(ma120_10ago):
                    gap_10ago = (ma20_10ago - ma120_10ago) / ma120_10ago * 100
                    if gap_10ago <= 0:
                        signals.append(MarketSignal(
                            signal_type="golden_cross_120_confirmed",
                            title="장기 골든크로스 확정! (MA20×MA120)",
                            description=(
                                "MA20이 MA120을 상향 돌파. "
                                "장기 추세 전환 확인. 강력 매수 시그널."
                            ),
                            importance="critical",
                            confidence=90,
                            data={"ma20": round(ma20, 0), "ma120": round(ma120, 0),
                                  "gap_pct": round(gap_120, 2)},
                        ))

        return signals

    # ──────────────────────────────────────────
    # 2. 이동평균 정배열/역배열
    # ──────────────────────────────────────────

    def _scan_ma_alignment(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """이동평균선 정배열/역배열 감지."""
        signals = []
        row = df.iloc[idx]
        close = row.get("Close", row.get("close", 0))

        ma5 = self._get_val(row, "MA5", "sma_5")
        ma20 = self._get_val(row, "MA20", "sma_20")
        ma60 = self._get_val(row, "MA60", "sma_60")
        ma120 = self._get_val(row, "MA120", "sma_120")

        if any(pd.isna(v) for v in [ma5, ma20, ma60]):
            return signals

        # 트리플 정배열: 가격 > MA5 > MA20 > MA60
        triple = close > ma5 > ma20 > ma60
        # 쿼드러플 정배열: + MA120
        quad = triple and not pd.isna(ma120) and ma60 > ma120

        if quad:
            signals.append(MarketSignal(
                signal_type="quad_alignment",
                title="완전 정배열 (5>20>60>120)",
                description=(
                    f"모든 이평선이 정배열. 가장 이상적인 상승 구도. "
                    f"주가({close:,.0f}) > MA5({ma5:,.0f}) > MA20({ma20:,.0f}) > "
                    f"MA60({ma60:,.0f}) > MA120({ma120:,.0f})"
                ),
                importance="high",
                confidence=85,
                data={"close": close, "ma5": round(ma5, 0), "ma20": round(ma20, 0),
                      "ma60": round(ma60, 0), "ma120": round(ma120, 0)},
            ))
        elif triple:
            signals.append(MarketSignal(
                signal_type="triple_alignment",
                title="정배열 (5>20>60)",
                description=(
                    f"주요 이평선 정배열. 상승 추세 지속. "
                    f"주가({close:,.0f}) > MA5({ma5:,.0f}) > MA20({ma20:,.0f}) > MA60({ma60:,.0f})"
                ),
                importance="medium",
                confidence=75,
                data={"close": close, "ma5": round(ma5, 0),
                      "ma20": round(ma20, 0), "ma60": round(ma60, 0)},
            ))

        return signals

    # ──────────────────────────────────────────
    # 3. 스마트머니 최적 감지
    # ──────────────────────────────────────────

    def _scan_smart_money(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """스마트머니(세력) 최적 조건 감지."""
        signals = []
        row = df.iloc[idx]
        close = row.get("Close", row.get("close", 0))

        adx = self._get_val(row, "ADX", "adx_14")
        plus_di = self._get_val(row, "Plus_DI", "plus_di")
        minus_di = self._get_val(row, "Minus_DI", "minus_di")

        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return signals

        # 조건 체크
        conditions = {}
        score = 0

        # 1. ADX 강세 (30 이상 = 강한 추세)
        conditions["adx_strong"] = adx >= 30
        if adx >= 40:
            score += 3
            conditions["adx_very_strong"] = True
        elif adx >= 30:
            score += 2
        elif adx >= 25:
            score += 1

        # 2. 방향성 (+DI >> -DI)
        di_ratio = plus_di / minus_di if minus_di > 0 else 99
        conditions["plus_di_dominant"] = di_ratio >= 2.0
        if di_ratio >= 5.0:
            score += 3
        elif di_ratio >= 3.0:
            score += 2
        elif di_ratio >= 2.0:
            score += 1

        # 3. OBV 상승 추세
        obv_rising = False
        if idx >= 20:
            obv_col = "OBV" if "OBV" in df.columns else "obv"
            if obv_col in df.columns:
                obv_now = df[obv_col].iloc[idx]
                obv_20ago = df[obv_col].iloc[idx - 20]
                if not pd.isna(obv_now) and not pd.isna(obv_20ago):
                    obv_change = (obv_now - obv_20ago) / abs(obv_20ago) * 100 if obv_20ago != 0 else 0
                    obv_rising = obv_change > 2
                    conditions["obv_rising"] = obv_rising
                    if obv_rising:
                        score += 2

        # 4. 거래량 확대
        vol_col = "Volume" if "Volume" in df.columns else "volume"
        vol_surge = False
        if vol_col in df.columns and idx >= 20:
            vol_now = df[vol_col].iloc[idx]
            vol_avg = df[vol_col].iloc[max(0, idx - 19):idx + 1].mean()
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
            vol_surge = vol_ratio >= 1.3
            conditions["volume_surge"] = vol_surge
            if vol_ratio >= 2.0:
                score += 2
            elif vol_surge:
                score += 1

        # 5. 수급 동향 (기관/외국인)
        inst_buying = False
        foreign_buying = False
        inst_streak = self._calc_streak(df, "Inst_Net", "inst_net", idx)
        foreign_streak = self._calc_streak(df, "Foreign_Net", "foreign_net", idx)

        if inst_streak >= 3:
            inst_buying = True
            score += 1
        if foreign_streak >= 3:
            foreign_buying = True
            score += 1
        if inst_streak >= 5 and foreign_streak >= 5:
            score += 2  # 동시 강력 매수

        conditions["inst_buying"] = inst_buying
        conditions["foreign_buying"] = foreign_buying
        conditions["inst_streak"] = inst_streak
        conditions["foreign_streak"] = foreign_streak

        # ── 스마트머니 최적 판정 ──
        # 최대 14점, 8점 이상 = 최적, 5점 이상 = 양호
        data = {
            "adx": round(adx, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "di_ratio": round(di_ratio, 1),
            "score": score,
            "inst_streak": inst_streak,
            "foreign_streak": foreign_streak,
            "conditions": conditions,
        }

        if score >= 8:
            signals.append(MarketSignal(
                signal_type="smart_money_optimal",
                title="스마트머니 최적!",
                description=self._build_sm_description(adx, plus_di, minus_di,
                                                       inst_streak, foreign_streak,
                                                       obv_rising, vol_surge),
                importance="critical",
                confidence=min(85 + (score - 8) * 3, 98),
                data=data,
            ))
        elif score >= 5:
            signals.append(MarketSignal(
                signal_type="smart_money_favorable",
                title="스마트머니 양호",
                description=self._build_sm_description(adx, plus_di, minus_di,
                                                       inst_streak, foreign_streak,
                                                       obv_rising, vol_surge),
                importance="high",
                confidence=65 + (score - 5) * 5,
                data=data,
            ))
        elif score >= 3:
            signals.append(MarketSignal(
                signal_type="smart_money_building",
                title="스마트머니 축적 중",
                description=(
                    f"세력 매집 초기 신호. ADX={adx:.1f}, "
                    f"+DI/-DI={plus_di:.1f}/{minus_di:.1f}"
                ),
                importance="medium",
                confidence=50 + score * 3,
                data=data,
            ))

        return signals

    @staticmethod
    def _build_sm_description(adx, plus_di, minus_di, inst_streak, foreign_streak,
                              obv_rising, vol_surge) -> str:
        parts = []
        parts.append(f"ADX={adx:.1f}(강한 추세)")
        parts.append(f"+DI={plus_di:.1f} >> -DI={minus_di:.1f}(상승 지배)")
        if obv_rising:
            parts.append("OBV 상승(매집)")
        if vol_surge:
            parts.append("거래량 확대")
        if inst_streak > 0:
            parts.append(f"기관 {inst_streak}일 연속 순매수")
        if foreign_streak > 0:
            parts.append(f"외국인 {foreign_streak}일 연속 순매수")
        return " | ".join(parts)

    # ──────────────────────────────────────────
    # 4. MACD 시그널
    # ──────────────────────────────────────────

    def _scan_macd_signals(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """MACD 골든크로스/제로선 돌파 감지."""
        signals = []
        row = df.iloc[idx]

        macd = self._get_val(row, "MACD", "macd")
        macd_sig = self._get_val(row, "MACD_Signal", "macd_signal")

        if pd.isna(macd) or pd.isna(macd_sig):
            return signals

        hist = macd - macd_sig

        # 전일 데이터
        if idx >= 1:
            prev = df.iloc[idx - 1]
            prev_macd = self._get_val(prev, "MACD", "macd")
            prev_sig = self._get_val(prev, "MACD_Signal", "macd_signal")

            if not pd.isna(prev_macd) and not pd.isna(prev_sig):
                prev_hist = prev_macd - prev_sig

                # MACD 골든크로스 (매수 시그널)
                if hist > 0 and prev_hist <= 0:
                    signals.append(MarketSignal(
                        signal_type="macd_golden_cross",
                        title="MACD 골든크로스",
                        description=(
                            f"MACD({macd:.1f})가 Signal({macd_sig:.1f}) 상향 돌파. "
                            f"모멘텀 상승 전환."
                        ),
                        importance="high",
                        confidence=75,
                        data={"macd": round(macd, 1), "signal": round(macd_sig, 1),
                              "histogram": round(hist, 1)},
                    ))

                # MACD 데드크로스 (경고)
                elif hist < 0 and prev_hist >= 0:
                    signals.append(MarketSignal(
                        signal_type="macd_dead_cross",
                        title="MACD 데드크로스 경고",
                        description=(
                            f"MACD({macd:.1f})가 Signal({macd_sig:.1f}) 하향 이탈. "
                            f"모멘텀 하락 전환 주의."
                        ),
                        importance="high",
                        confidence=70,
                        data={"macd": round(macd, 1), "signal": round(macd_sig, 1)},
                    ))

        # MACD 제로선 돌파
        if idx >= 1:
            prev_macd = self._get_val(df.iloc[idx - 1], "MACD", "macd")
            if not pd.isna(prev_macd):
                if macd > 0 and prev_macd <= 0:
                    signals.append(MarketSignal(
                        signal_type="macd_zero_cross_up",
                        title="MACD 제로선 상향 돌파",
                        description=f"MACD가 0선 돌파 ({macd:.1f}). 추세 전환 확인.",
                        importance="medium",
                        confidence=65,
                        data={"macd": round(macd, 1)},
                    ))

        return signals

    # ──────────────────────────────────────────
    # 5. 볼린저 밴드 스퀴즈
    # ──────────────────────────────────────────

    def _scan_bollinger_squeeze(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """볼린저 밴드 스퀴즈 (큰 변동 예고) 감지."""
        signals = []
        row = df.iloc[idx]

        upper = self._get_val(row, "Upper_Band")
        lower = self._get_val(row, "Lower_Band")
        close = row.get("Close", row.get("close", 0))

        if pd.isna(upper) or pd.isna(lower) or close == 0:
            return signals

        bb_width = (upper - lower) / close * 100

        # 20일 평균 밴드폭과 비교
        if idx >= 20:
            widths = []
            for i in range(max(0, idx - 19), idx + 1):
                r = df.iloc[i]
                u = self._get_val(r, "Upper_Band")
                l = self._get_val(r, "Lower_Band")
                c = r.get("Close", r.get("close", 0))
                if not pd.isna(u) and not pd.isna(l) and c > 0:
                    widths.append((u - l) / c * 100)

            if widths:
                avg_width = np.mean(widths)
                width_ratio = bb_width / avg_width if avg_width > 0 else 1.0

                if width_ratio < 0.6:
                    signals.append(MarketSignal(
                        signal_type="bollinger_squeeze",
                        title="볼린저 밴드 스퀴즈!",
                        description=(
                            f"밴드폭({bb_width:.1f}%)이 20일 평균({avg_width:.1f}%)의 "
                            f"{width_ratio:.0%}로 극도로 수축. 큰 변동 임박 예상."
                        ),
                        importance="high",
                        confidence=75,
                        data={"bb_width": round(bb_width, 2),
                              "avg_width": round(avg_width, 2),
                              "ratio": round(width_ratio, 2)},
                    ))

        return signals

    # ──────────────────────────────────────────
    # 6. 추세 가속 감지
    # ──────────────────────────────────────────

    def _scan_trend_acceleration(self, df: pd.DataFrame, idx: int) -> list[MarketSignal]:
        """추세 가속/감속 감지."""
        signals = []
        row = df.iloc[idx]

        adx = self._get_val(row, "ADX", "adx_14")
        if pd.isna(adx):
            return signals

        # ADX 5일 전 비교
        if idx >= 5:
            adx_5ago = self._get_val(df.iloc[idx - 5], "ADX", "adx_14")
            if not pd.isna(adx_5ago):
                adx_change = adx - adx_5ago

                if adx >= 30 and adx_change >= 5:
                    signals.append(MarketSignal(
                        signal_type="trend_acceleration",
                        title="추세 가속!",
                        description=(
                            f"ADX가 5일간 {adx_5ago:.1f}→{adx:.1f} (+{adx_change:.1f}) 급등. "
                            f"추세 강도 급속 증가."
                        ),
                        importance="high",
                        confidence=70 + min(adx_change * 2, 20),
                        data={"adx": round(adx, 1), "adx_5ago": round(adx_5ago, 1),
                              "adx_change": round(adx_change, 1)},
                    ))

        return signals

    # ──────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────

    @staticmethod
    def _get_val(row, *col_names) -> float:
        """여러 컬럼명 후보 중 첫 번째 유효값 반환."""
        for col in col_names:
            val = row.get(col, np.nan)
            if not pd.isna(val):
                return float(val)
        return np.nan

    @staticmethod
    def _get_val_at(df: pd.DataFrame, idx: int, *col_names) -> float:
        """특정 인덱스의 값을 가져옴."""
        if idx < 0 or idx >= len(df):
            return np.nan
        row = df.iloc[idx]
        for col in col_names:
            val = row.get(col, np.nan)
            if not pd.isna(val):
                return float(val)
        return np.nan

    @staticmethod
    def _calc_streak(df: pd.DataFrame, col1: str, col2: str, idx: int) -> int:
        """마지막 행 기준 연속 양수 일수."""
        col = col1 if col1 in df.columns else (col2 if col2 in df.columns else None)
        if col is None:
            return 0

        streak = 0
        for i in range(idx, max(idx - 30, -1), -1):
            val = df.iloc[i].get(col, 0)
            if pd.isna(val) or val <= 0:
                break
            streak += 1

        return streak
