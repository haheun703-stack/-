"""
Shakeout Detector — 세력 털기 vs 진짜 하락 판별

백테스트 결과 기반 최적 조건:
  - 횡보 후 급락: 직전 20일 수익률 std < 0.020
  - 하락폭: -3% ~ -7% (sweet spot)
  - 거래량: 20일 평균 대비 2배 이상
  - MA120 위 유지

판정:
  3~4점: SHAKEOUT (손절 보류, 1일 대기)
  0~1점: REAL_DROP (기존대로 손절)
  2점:   UNCERTAIN (1일 대기 후 재판정)

글로벌 리스크 필터:
  VIX SPIKE / STRONG_BEAR 시 → 무조건 REAL_DROP (shakeout 오판 방지)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 설정 ──
STD_THRESHOLD = 0.020       # 횡보 판별 (일간수익률 std)
DROP_MIN = -0.07            # 하락폭 하한 (-7%)
DROP_MAX = -0.03            # 하락폭 상한 (-3%)
VOL_MULT = 2.0              # 거래량 배수 기준
LOOKBACK = 20               # 횡보/거래량 판별 기간


@dataclass
class ShakeoutResult:
    """Shakeout 판정 결과"""
    verdict: str = "REAL_DROP"       # SHAKEOUT / REAL_DROP / UNCERTAIN
    score: int = 0                   # 0~4
    conditions: dict = field(default_factory=dict)
    global_risk: bool = False        # 글로벌 리스크 발동 여부
    message: str = ""

    def to_alert_text(self, ticker: str, name: str, drop_pct: float) -> str:
        """텔레그램 알림용 텍스트 생성"""
        icon = {"SHAKEOUT": "🔍", "REAL_DROP": "🔴", "UNCERTAIN": "🟡"}.get(
            self.verdict, "❓"
        )

        cond_lines = []
        for key, val in self.conditions.items():
            mark = "✅" if val.get("pass") else "❌"
            cond_lines.append(f"  {mark} {val.get('label', key)}: {val.get('detail', '')}")

        action = {
            "SHAKEOUT": "→ 손절 보류 (1일 대기). 내일 재판정.",
            "REAL_DROP": "→ 기존대로 손절 실행.",
            "UNCERTAIN": "→ 판단 보류 (1일 대기). 내일 재판정.",
        }.get(self.verdict, "")

        lines = [
            f"{icon} [Shakeout 감지] {name} ({ticker})",
            f"하락: {drop_pct:+.1f}% | 판정: {self.verdict} ({self.score}/4점)",
            "",
            *cond_lines,
        ]
        if self.global_risk:
            lines.append("  ⚠️ 글로벌 리스크 → SHAKEOUT 무효화")
        lines.append("")
        lines.append(action)
        return "\n".join(lines)


def _load_parquet_data(ticker: str) -> pd.DataFrame | None:
    """parquet에서 종목 데이터 로드 (있으면)."""
    path = PROJECT_ROOT / "data" / "processed" / f"{ticker}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.debug("parquet 로드 실패 (%s): %s", ticker, e)
    return None


def _load_pykrx_data(ticker: str, days: int = 400) -> pd.DataFrame | None:
    """pykrx에서 종목 데이터 수집 (parquet 없을 때 fallback)."""
    try:
        from pykrx import stock as pykrx_stock
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        df = pykrx_stock.get_market_ohlcv(start_date, end_date, ticker)
        if df.empty:
            return None

        # 컬럼 표준화
        col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
        df = df.rename(columns=col_map)
        df["ret1"] = df["close"].pct_change()
        df["sma_120"] = df["close"].rolling(120).mean()
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        return df
    except Exception as e:
        logger.warning("pykrx 수집 실패 (%s): %s", ticker, e)
        return None


def _check_global_risk() -> bool:
    """
    글로벌 리스크 체크 (VIX SPIKE / STRONG_BEAR).
    US overnight signal이 STRONG_BEAR 또는 특수 룰 발동이면 True.
    """
    sig_path = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
    if not sig_path.exists():
        return False
    try:
        sig = json.loads(sig_path.read_text(encoding="utf-8"))
        grade = sig.get("final_grade", "NEUTRAL")
        special = sig.get("special_rules_fired", [])

        # STRONG_BEAR 또는 위험 특수 룰이면 글로벌 리스크
        if grade == "STRONG_BEAR":
            return True
        risk_rules = {"VIX_SPIKE", "SOXX_CRASH", "NASDAQ_CIRCUIT", "MARKET_CRASH"}
        if set(special) & risk_rules:
            return True
        return False
    except Exception:
        return False


def detect_shakeout(
    ticker: str,
    current_price: float,
    entry_price: float,
) -> ShakeoutResult:
    """
    급락 발생 시 세력 털기 vs 진짜 하락 판별.

    Args:
        ticker: 종목 코드
        current_price: 현재가 (장중 실시간)
        entry_price: 진입가

    Returns:
        ShakeoutResult (verdict, score, conditions)
    """
    result = ShakeoutResult()
    conditions = {}

    # ── 0. 글로벌 리스크 필터 ──
    global_risk = _check_global_risk()
    result.global_risk = global_risk

    # ── 1. 데이터 로드 ──
    df = _load_parquet_data(ticker)
    if df is None:
        df = _load_pykrx_data(ticker)
    if df is None or len(df) < LOOKBACK + 130:
        result.message = "데이터 부족"
        return result

    # 최근 데이터
    recent = df.iloc[-LOOKBACK - 5:]  # 약간의 여유
    if len(recent) < LOOKBACK:
        result.message = "최근 데이터 부족"
        return result

    # ── 조건 1: 횡보 구간 (직전 20일 std) ──
    if "ret1" not in df.columns:
        df["ret1"] = df["close"].pct_change()

    last_20_rets = df["ret1"].iloc[-LOOKBACK - 1 : -1]  # 오늘 제외, 직전 20일
    std_val = last_20_rets.std()
    cond1_pass = std_val < STD_THRESHOLD if not pd.isna(std_val) else False
    conditions["sideways"] = {
        "label": "횡보 구간 (std<0.020)",
        "pass": cond1_pass,
        "detail": f"std={std_val:.4f}" if not pd.isna(std_val) else "N/A",
    }

    # ── 조건 2: 하락폭 (-3% ~ -7%) ──
    yesterday_close = float(df["close"].iloc[-1])  # 가장 최근 parquet 종가
    # 진입가 대비 현재가 하락률
    drop_pct = (current_price / entry_price) - 1
    # 전일 종가 대비 하락률도 계산
    drop_vs_yesterday = (current_price / yesterday_close) - 1 if yesterday_close > 0 else 0

    # 더 큰 하락폭 기준 사용
    effective_drop = min(drop_pct, drop_vs_yesterday)
    cond2_pass = DROP_MIN <= effective_drop <= DROP_MAX
    conditions["drop_range"] = {
        "label": "하락폭 -3%~-7%",
        "pass": cond2_pass,
        "detail": f"진입대비 {drop_pct*100:+.1f}%, 전일대비 {drop_vs_yesterday*100:+.1f}%",
    }

    # ── 조건 3: 거래량 급증 ──
    if "volume_ma20" not in df.columns:
        df["volume_ma20"] = df["volume"].rolling(20).mean()

    vol_ma20 = float(df["volume_ma20"].iloc[-1])
    # 오늘 거래량은 장중이라 정확하지 않을 수 있음
    # parquet의 마지막 거래량 사용 (전일)
    last_volume = float(df["volume"].iloc[-1])
    vol_ratio = last_volume / vol_ma20 if vol_ma20 > 0 else 0
    cond3_pass = vol_ratio >= VOL_MULT
    conditions["volume_spike"] = {
        "label": f"거래량 {VOL_MULT}배+",
        "pass": cond3_pass,
        "detail": f"{vol_ratio:.1f}x (전일 기준)",
    }

    # ── 조건 4: MA120 위 유지 ──
    if "sma_120" not in df.columns:
        df["sma_120"] = df["close"].rolling(120).mean()

    ma120 = float(df["sma_120"].iloc[-1])
    cond4_pass = current_price > ma120 if ma120 > 0 and not pd.isna(ma120) else False
    ma120_margin = ((current_price / ma120) - 1) * 100 if ma120 > 0 else 0
    conditions["above_ma120"] = {
        "label": "MA120 위 유지",
        "pass": cond4_pass,
        "detail": f"MA120={ma120:,.0f}, 현재가={current_price:,.0f} ({ma120_margin:+.1f}%)",
    }

    # ── 점수 합산 ──
    score = sum(1 for c in conditions.values() if c["pass"])
    result.score = score
    result.conditions = conditions

    # ── 판정 ──
    if global_risk:
        # 글로벌 리스크 시 무조건 REAL_DROP
        result.verdict = "REAL_DROP"
        result.message = "글로벌 리스크 발동 → SHAKEOUT 무효화"
    elif score >= 3:
        result.verdict = "SHAKEOUT"
        result.message = f"세력 털기 가능성 높음 ({score}/4점)"
    elif score == 2:
        result.verdict = "UNCERTAIN"
        result.message = f"판단 보류 ({score}/4점)"
    else:
        result.verdict = "REAL_DROP"
        result.message = f"진짜 하락 ({score}/4점)"

    logger.info(
        "[Shakeout] %s: %s (%d/4) — %s",
        ticker, result.verdict, score, result.message,
    )
    return result
