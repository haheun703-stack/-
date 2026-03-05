"""
3단 예측 체인 — 유럽장 오픈 기반 미국장 방향 선행 예측

[1단] 09:00~15:30  아시안 리스크 스코어
      AUD/JPY + CNH + KOSPI 외국인 순매매
      → 리스크 온/오프 사전 판정

[2단] 16:00~16:30  유럽 오픈 스코어 ★핵심★
      DAX 첫 30분 + EUR/USD + S&P선물 변화
      → 1단과 방향 일치 여부 → 진입 결정

[3단] 상시 괴리 감지
      HYG vs 선물 / VIX 커브 / 채권 vs 금
      → 괴리 발견 시 신호 무효화 또는 강화

최종: 1단+2단 방향 일치 + 3단 괴리 없음 → 16:30 시그널 발생

실행 시점: 16:35 KST (유럽장 오픈 30분 후)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 타임존
KST = timezone(timedelta(hours=9))
CET = timezone(timedelta(hours=1))  # 유럽 중앙시간 (서머타임 시 +2)


class PredictChainEngine:
    """3단 예측 체인 엔진."""

    # 기본 설정
    DEFAULT_CFG = {
        "asian_risk": {
            "audjpy_threshold_pct": 0.3,    # AUD/JPY ±0.3% = 방향 판정
            "cnh_threshold_pct": 0.2,        # USD/CNH ±0.2%
            "weight_audjpy": 0.50,
            "weight_cnh": 0.30,
            "weight_futures": 0.20,
        },
        "europe_open": {
            "dax_30m_threshold_pct": 0.3,    # DAX 30분 ±0.3%
            "eurusd_threshold_pct": 0.15,    # EUR/USD ±0.15%
            "weight_dax": 0.55,
            "weight_eurusd": 0.25,
            "weight_futures": 0.20,
        },
        "divergence": {
            "hyg_futures_gap_pct": 0.5,      # HYG vs 선물 괴리 임계
            "bond_gold_diverge": True,        # 채권↑+금↓ = 리스크온
            "vix_curve_shift": True,          # VIX 커브 전환 감지
        },
        "signal": {
            "min_agreement_score": 0.4,       # 1단+2단 합산 최소 동의 점수
            "direction_match_required": True,  # 1단/2단 같은 방향이어야 함
        },
    }

    def __init__(self, settings: dict | None = None):
        pc = (settings or {}).get("predict_chain", {})
        self.cfg = {
            "asian_risk": {**self.DEFAULT_CFG["asian_risk"], **pc.get("asian_risk", {})},
            "europe_open": {**self.DEFAULT_CFG["europe_open"], **pc.get("europe_open", {})},
            "divergence": {**self.DEFAULT_CFG["divergence"], **pc.get("divergence", {})},
            "signal": {**self.DEFAULT_CFG["signal"], **pc.get("signal", {})},
        }

    def compute(self, intraday: dict[str, pd.DataFrame]) -> dict:
        """3단 예측 체인 실행.

        Args:
            intraday: 티커별 5분봉 DataFrame
                필수: AUDJPY=X, USDCNH=X, ^GDAXI, EURUSD=X, ES=F, NQ=F
                선택: HYG, ^TNX, GC=F

        Returns:
            {
                "stage1": {...},  # 아시안 리스크
                "stage2": {...},  # 유럽 오픈
                "stage3": {...},  # 괴리 감지
                "final_signal": "BULL" / "BEAR" / "NEUTRAL",
                "confidence": 0.0~1.0,
                "direction_match": bool,
                "timestamp": str,
            }
        """
        s1 = self._stage1_asian_risk(intraday)
        s2 = self._stage2_europe_open(intraday)
        s3 = self._stage3_divergence(intraday)

        final = self._combine(s1, s2, s3)

        return {
            "version": "1.0",
            "stage1_asian_risk": s1,
            "stage2_europe_open": s2,
            "stage3_divergence": s3,
            "final_signal": final["signal"],
            "final_score": final["score"],
            "confidence": final["confidence"],
            "direction_match": final["direction_match"],
            "divergence_alert": final["divergence_alert"],
            "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        }

    # ──────────────────────────────────────────
    # Stage 1: 아시안 리스크 스코어 (09:00~15:30)
    # ──────────────────────────────────────────
    def _stage1_asian_risk(self, data: dict) -> dict:
        """AUD/JPY + CNH + 미선물로 아시안 세션 리스크 판정."""
        cfg = self.cfg["asian_risk"]
        score = 0.0
        details: dict[str, Any] = {}

        # AUD/JPY: 리스크 온도계
        audjpy = self._get_session_return(data, "AUDJPY=X", hours_back=7)
        details["audjpy_ret_pct"] = _r4(audjpy)

        if audjpy is not None:
            th = cfg["audjpy_threshold_pct"] / 100
            w = cfg["weight_audjpy"]
            if audjpy > th:
                score += w  # 리스크 온
                details["audjpy_signal"] = "RISK_ON"
            elif audjpy < -th:
                score -= w  # 리스크 오프
                details["audjpy_signal"] = "RISK_OFF"
            else:
                details["audjpy_signal"] = "NEUTRAL"

        # USD/CNH: 위안 약세 = 중국 리스크 = 아시아 risk-off
        cnh = self._get_session_return(data, "USDCNH=X", hours_back=7)
        details["usdcnh_ret_pct"] = _r4(cnh)

        if cnh is not None:
            th = cfg["cnh_threshold_pct"] / 100
            w = cfg["weight_cnh"]
            if cnh > th:
                # USD/CNH 상승 = 위안 약세 = 아시아 risk-off
                score -= w
                details["cnh_signal"] = "CNH_WEAK"
            elif cnh < -th:
                score += w * 0.5  # 위안 강세는 약한 긍정
                details["cnh_signal"] = "CNH_STRONG"
            else:
                details["cnh_signal"] = "STABLE"

        # S&P500 선물 (아시안 세션 변화)
        es_ret = self._get_session_return(data, "ES=F", hours_back=7)
        details["es_futures_ret_pct"] = _r4(es_ret)

        if es_ret is not None:
            w = cfg["weight_futures"]
            score += _clamp(es_ret * 100, -1, 1) * w  # 비례 반영

        score = _clamp(score, -1.0, 1.0)

        if score > 0.15:
            direction = "BULL"
        elif score < -0.15:
            direction = "BEAR"
        else:
            direction = "NEUTRAL"

        details["score"] = round(score, 6)  # raw -1~1 (combine에서 사용)
        details["direction"] = direction
        return details

    # ──────────────────────────────────────────
    # Stage 2: 유럽 오픈 스코어 (16:00~16:30)
    # ──────────────────────────────────────────
    def _stage2_europe_open(self, data: dict) -> dict:
        """DAX 첫 30분 + EUR/USD + 선물 변화로 유럽 세션 방향 판정."""
        cfg = self.cfg["europe_open"]
        score = 0.0
        details: dict[str, Any] = {}

        # DAX 첫 30분 수익률 (핵심!)
        dax_30m = self._get_open_30m_return(data, "^GDAXI")
        details["dax_30m_ret_pct"] = _r4(dax_30m)

        if dax_30m is not None:
            th = cfg["dax_30m_threshold_pct"] / 100
            w = cfg["weight_dax"]

            if dax_30m > th:
                score += w
                details["dax_signal"] = "BULL"
            elif dax_30m < -th:
                score -= w
                details["dax_signal"] = "BEAR"
            else:
                # 임계값 미만이라도 방향성 반영 (절반 가중)
                score += _clamp(dax_30m * 100, -1, 1) * w * 0.5
                details["dax_signal"] = "WEAK_" + ("BULL" if dax_30m > 0 else "BEAR")
        else:
            details["dax_signal"] = "NO_DATA"

        # EUR/USD: 유로 강세 = 리스크온 (달러 약세)
        eurusd = self._get_open_30m_return(data, "EURUSD=X")
        details["eurusd_30m_ret_pct"] = _r4(eurusd)

        if eurusd is not None:
            th = cfg["eurusd_threshold_pct"] / 100
            w = cfg["weight_eurusd"]
            if eurusd > th:
                score += w  # 유로 강세 = 리스크온
                details["eurusd_signal"] = "RISK_ON"
            elif eurusd < -th:
                score -= w  # 유로 약세 = 리스크오프
                details["eurusd_signal"] = "RISK_OFF"
            else:
                details["eurusd_signal"] = "NEUTRAL"

        # S&P 선물 변화 (유럽 오픈 30분간)
        es_30m = self._get_open_30m_return(data, "ES=F")
        details["es_futures_30m_pct"] = _r4(es_30m)

        if es_30m is not None:
            w = cfg["weight_futures"]
            score += _clamp(es_30m * 100, -1, 1) * w

        score = _clamp(score, -1.0, 1.0)

        if score > 0.15:
            direction = "BULL"
        elif score < -0.15:
            direction = "BEAR"
        else:
            direction = "NEUTRAL"

        details["score"] = round(score, 6)  # raw -1~1 (combine에서 사용)
        details["direction"] = direction
        return details

    # ──────────────────────────────────────────
    # Stage 3: 괴리 감지 (상시)
    # ──────────────────────────────────────────
    def _stage3_divergence(self, data: dict) -> dict:
        """자산 간 괴리 감지 → 신호 무효화 또는 강화."""
        cfg = self.cfg["divergence"]
        alerts: list[dict] = []
        details: dict[str, Any] = {}
        invalidate = False
        boost = False

        # 1. 미선물 평온 + HYG 하락 = 숨은 스트레스
        es_ret = self._get_session_return(data, "ES=F", hours_back=2)
        hyg_ret = self._get_session_return(data, "HYG", hours_back=2)

        if es_ret is not None and hyg_ret is not None:
            gap = abs((es_ret or 0) * 100) - abs((hyg_ret or 0) * 100)
            details["hyg_vs_futures_gap"] = _r4(gap)

            # 선물 평온(-0.3~+0.3%) 인데 HYG가 -0.5% 이상 하락
            if abs(es_ret) < 0.003 and hyg_ret < -0.005:
                alerts.append({
                    "type": "HYG_HIDDEN_STRESS",
                    "desc": f"선물 평온({es_ret*100:+.2f}%) + HYG 하락({hyg_ret*100:+.2f}%)",
                    "action": "매수 금지",
                })
                invalidate = True

        # 2. 채권↑ + 금↓ = 스마트머니 리스크온 전환
        tnx_ret = self._get_session_return(data, "^TNX", hours_back=4)
        gold_ret = self._get_session_return(data, "GC=F", hours_back=4)
        details["tnx_ret_pct"] = _r4(tnx_ret)
        details["gold_ret_pct"] = _r4(gold_ret)

        if cfg["bond_gold_diverge"] and tnx_ret is not None and gold_ret is not None:
            # 금리↑(채권↓) + 금↓ = 리스크온 (강매수 시그널)
            if tnx_ret > 0.002 and gold_ret < -0.003:
                alerts.append({
                    "type": "SMART_MONEY_RISK_ON",
                    "desc": f"금리↑({tnx_ret*100:+.2f}%) + 금↓({gold_ret*100:+.2f}%)",
                    "action": "강매수",
                })
                boost = True
            # 금리↓(채권↑) + 금↑ = 리스크오프 (방어)
            elif tnx_ret < -0.002 and gold_ret > 0.003:
                alerts.append({
                    "type": "FLIGHT_TO_SAFETY",
                    "desc": f"금리↓({tnx_ret*100:+.2f}%) + 금↑({gold_ret*100:+.2f}%)",
                    "action": "방어",
                })

        # 3. AUD/JPY ↓ + 구리 ↑ = 지정학 긴장 (경제 아님)
        audjpy_ret = self._get_session_return(data, "AUDJPY=X", hours_back=4)
        copper_ret = self._get_session_return(data, "CL=F", hours_back=4)  # 원유로 대체

        if audjpy_ret is not None and copper_ret is not None:
            if audjpy_ret < -0.003 and copper_ret > 0.005:
                alerts.append({
                    "type": "GEOPOLITICAL_TENSION",
                    "desc": f"AUD/JPY↓({audjpy_ret*100:+.2f}%) + 원유↑({copper_ret*100:+.2f}%)",
                    "action": "방산 매수",
                })

        details["alerts"] = alerts
        details["alert_count"] = len(alerts)
        details["invalidate"] = invalidate
        details["boost"] = boost
        return details

    # ──────────────────────────────────────────
    # 최종 결합
    # ──────────────────────────────────────────
    def _combine(self, s1: dict, s2: dict, s3: dict) -> dict:
        cfg = self.cfg["signal"]

        s1_score = s1.get("score", 0) or 0
        s2_score = s2.get("score", 0) or 0
        s1_dir = s1.get("direction", "NEUTRAL")
        s2_dir = s2.get("direction", "NEUTRAL")

        # 방향 일치 확인
        direction_match = (
            s1_dir == s2_dir and s1_dir != "NEUTRAL"
        )

        # 합산 점수
        combined = s1_score * 0.35 + s2_score * 0.65  # 유럽 오픈에 더 큰 가중

        # 괴리 감지 적용
        divergence_alert = False
        if s3.get("invalidate"):
            combined = min(combined, 0)  # BULL 무효화
            divergence_alert = True
        if s3.get("boost"):
            combined = combined * 1.3  # 강화
        combined = _clamp(combined, -1.0, 1.0)

        # 신뢰도 계산
        confidence = abs(combined)
        if direction_match:
            confidence = min(1.0, confidence * 1.5)  # 일치 시 신뢰도 보너스

        # 최종 판정
        min_score = cfg["min_agreement_score"]
        if cfg["direction_match_required"] and not direction_match:
            signal = "NEUTRAL"
            confidence *= 0.5
        elif combined >= min_score:
            signal = "BULL"
        elif combined <= -min_score:
            signal = "BEAR"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "score": _r4(combined),
            "confidence": _r4(confidence),
            "direction_match": direction_match,
            "divergence_alert": divergence_alert,
        }

    # ──────────────────────────────────────────
    # 데이터 유틸리티
    # ──────────────────────────────────────────
    def _get_session_return(
        self, data: dict, ticker: str, hours_back: int = 7
    ) -> float | None:
        """최근 N시간 동안의 수익률."""
        df = data.get(ticker)
        if df is None or df.empty:
            return None

        now = df.index[-1]
        start = now - pd.Timedelta(hours=hours_back)
        session = df[df.index >= start]

        if len(session) < 2:
            return None

        first_close = session.iloc[0]["Close"]
        last_close = session.iloc[-1]["Close"]

        if first_close == 0 or pd.isna(first_close):
            return None

        return (last_close - first_close) / first_close

    def _get_open_30m_return(
        self, data: dict, ticker: str
    ) -> float | None:
        """오늘의 시가 대비 30분 후 수익률.

        유럽장 기준: 오늘 첫 봉(09:00 CET) ~ 6봉 후(09:30 CET)
        """
        df = data.get(ticker)
        if df is None or df.empty:
            return None

        # 오늘 날짜의 데이터만
        today = df.index[-1].date()
        today_data = df[df.index.date == today]

        if len(today_data) < 6:  # 최소 30분 (5분봉 × 6)
            return None

        open_price = today_data.iloc[0]["Open"]
        # 30분 후 (6번째 봉의 종가)
        close_30m = today_data.iloc[min(5, len(today_data) - 1)]["Close"]

        if open_price == 0 or pd.isna(open_price):
            return None

        return (close_30m - open_price) / open_price


# ──────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────

# 3단 체인에 필요한 티커
CHAIN_TICKERS = {
    # Stage 1: 아시안 리스크
    "AUDJPY=X": "AUD/JPY",
    "USDCNH=X": "USD/CNH",
    "ES=F": "S&P500 선물",
    "NQ=F": "NASDAQ 선물",
    # Stage 2: 유럽 오픈
    "^GDAXI": "DAX",
    "EURUSD=X": "EUR/USD",
    # Stage 3: 괴리 감지
    "HYG": "하이일드",
    "^TNX": "10Y금리",
    "GC=F": "금선물",
    "CL=F": "원유선물",
}


def fetch_intraday(tickers: dict | None = None, period: str = "5d") -> dict[str, pd.DataFrame]:
    """yfinance로 5분봉 인트라데이 데이터 수집."""
    import yfinance as yf

    if tickers is None:
        tickers = CHAIN_TICKERS

    result = {}
    for ticker, label in tickers.items():
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(period=period, interval="5m")
            if df is not None and not df.empty:
                result[ticker] = df
                logger.debug(f"  {ticker} ({label}): {len(df)}봉")
            else:
                logger.warning(f"  {ticker} ({label}): 데이터 없음")
        except Exception as e:
            logger.warning(f"  {ticker} ({label}): 실패 — {e}")

    return result


# ──────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────

def _r4(v: float | None) -> float | None:
    """소수점 4자리 반올림. None-safe."""
    if v is None:
        return None
    return round(v * 100, 4)  # 퍼센트로 변환


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
