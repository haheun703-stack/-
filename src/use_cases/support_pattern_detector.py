"""적응형 포지션 매매법 MVP-3 — 받침 패턴 감지 (아래꼬리 + 양봉 + 거래량).

배경 (퐝가님 5/23 흐름):
  6단계 흐름 중 [5단계 받침 형성 확인]:
    "조정의 시작점에서 다시 사고 — 단, 받침 확인 후."

  분할매수만으로는 부족. 바닥 형성 = "받침 패턴"이 확정되어야 재진입.

  받침 패턴 = "물량을 던졌다가 곧바로 회수하는" 3가지 시그널 ALL 충족:

    1. ✅ 어제 캔들: 아래꼬리 길이 ≥ 몸통 × 2.0  (개미 털기 후 회복)
    2. ✅ 오늘 양봉: close > open + ATR × 0.5   (확정 회복)
    3. ✅ 오늘 거래량 ≥ 5일 평균 × 2.0          (관심 재진입)

  ALL → 받침 확정 → 텔레그램 알림 (MVP-4가 자동 매수 의사결정)

MVP-3 기능:
  1. detect_support_pattern(broker, ticker) — 종목별 3 조건 검증
  2. scan_pool_for_support(broker, candidates) — 후보 풀 일괄 스캔
  3. format_support_signal_for_telegram

5/17 자기반성 #1 적용: import + 함수 호출 + 실제 OHLCV 흐름 검증.

사용:
  from src.use_cases.support_pattern_detector import detect_support_pattern
  sig = detect_support_pattern(broker, "240810")
  if sig.trigger:
      # 텔레그램 알림 → MVP-4가 재진입 판정
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


# === 임계 (.env 동적) ===
SHADOW_RATIO_MIN = float(os.getenv("ADAPTIVE_SUPPORT_SHADOW_RATIO", "2.0"))   # 아래꼬리/몸통
BULLISH_ATR_MULT = float(os.getenv("ADAPTIVE_SUPPORT_BULLISH_ATR", "0.5"))    # 양봉 강도
VOLUME_MULT = float(os.getenv("ADAPTIVE_SUPPORT_VOLUME_MULT", "2.0"))         # 거래량 배수
ATR_PERIOD = int(os.getenv("ADAPTIVE_SUPPORT_ATR_PERIOD", "14"))
LOOKBACK_DAYS = int(os.getenv("ADAPTIVE_SUPPORT_LOOKBACK_DAYS", "20"))


@dataclass
class SupportSignal:
    """받침 패턴 시그널."""

    ticker: str
    # 어제 캔들
    yesterday_open: int = 0
    yesterday_close: int = 0
    yesterday_low: int = 0
    yesterday_high: int = 0
    yesterday_body: int = 0
    yesterday_low_shadow: int = 0
    yesterday_shadow_ratio: float = 0.0
    # 오늘 캔들
    today_open: int = 0
    today_close: int = 0
    today_bullish_threshold: int = 0
    today_volume: int = 0
    avg_volume_5d: float = 0.0
    volume_ratio: float = 0.0
    atr_14: float = 0.0
    # 결과
    trigger: bool = False
    reasons_pass: list[str] = field(default_factory=list)
    reasons_fail: list[str] = field(default_factory=list)
    error: Optional[str] = None


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _fetch_recent_ohlcv(broker, ticker: str, days: int = LOOKBACK_DAYS) -> list[dict]:
    """최근 N일 OHLCV (KIS API output2 최신순)."""
    try:
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
        res = broker.fetch_ohlcv(ticker, timeframe="D", start_day=start_day, end_day=end_day)
        rows = res.get("output2", []) if res else []
        return rows[:days]
    except Exception as e:
        logger.warning("OHLCV fetch %s 실패: %s", ticker, e)
        return []


def _safe_int(v) -> int:
    try:
        return int(str(v).replace(",", "") or 0)
    except (ValueError, TypeError):
        return 0


def _safe_float(v) -> float:
    try:
        return float(str(v).replace(",", "") or 0)
    except (ValueError, TypeError):
        return 0.0


def _compute_atr(rows: list[dict], period: int = ATR_PERIOD) -> float:
    """단순 ATR — 최근 period 일의 TR 평균.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    rows: 최신순 (rows[0]=오늘 또는 가장 최신).
    """
    if len(rows) < 2:
        return 0.0

    sample = rows[: min(period + 1, len(rows))]
    trs: list[float] = []
    for i in range(len(sample) - 1):
        cur = sample[i]
        prev = sample[i + 1]
        high = _safe_float(cur.get("stck_hgpr"))
        low = _safe_float(cur.get("stck_lwpr"))
        prev_close = _safe_float(prev.get("stck_clpr"))
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if not trs:
        return 0.0
    return sum(trs) / len(trs)


def detect_support_pattern(
    broker,
    ticker: str,
    shadow_ratio_min: float = SHADOW_RATIO_MIN,
    bullish_atr_mult: float = BULLISH_ATR_MULT,
    volume_mult: float = VOLUME_MULT,
) -> SupportSignal:
    """받침 패턴 감지 (어제 아래꼬리 + 오늘 양봉 + 거래량 폭증).

    Args:
        broker: KIS broker
        ticker: 종목 코드
        shadow_ratio_min: 어제 아래꼬리/몸통 최소 비율 (기본 2.0)
        bullish_atr_mult: 오늘 양봉 강도 (close > open + ATR × N, 기본 0.5)
        volume_mult: 오늘 거래량 ÷ 5일 평균 (기본 2.0)

    Returns:
        SupportSignal
    """
    sig = SupportSignal(ticker=ticker)

    # KILL_SWITCH 확인
    if _is_kill_switch_active():
        sig.reasons_fail.append("KILL_SWITCH 발동 중 → 본 흐름 정지")
        return sig

    rows = _fetch_recent_ohlcv(broker, ticker)
    if len(rows) < 7:
        sig.error = f"OHLCV {len(rows)}건 부족 (받침 판정 7건 필요)"
        sig.reasons_fail.append(sig.error)
        return sig

    today = rows[0]
    yesterday = rows[1]

    # === 1. 어제 캔들: 아래꼬리 / 몸통 비율 ===
    y_open = _safe_int(yesterday.get("stck_oprc"))
    y_close = _safe_int(yesterday.get("stck_clpr"))
    y_low = _safe_int(yesterday.get("stck_lwpr"))
    y_high = _safe_int(yesterday.get("stck_hgpr"))

    sig.yesterday_open = y_open
    sig.yesterday_close = y_close
    sig.yesterday_low = y_low
    sig.yesterday_high = y_high

    body = abs(y_close - y_open)
    low_shadow = min(y_open, y_close) - y_low
    sig.yesterday_body = body
    sig.yesterday_low_shadow = max(low_shadow, 0)

    # body=0 (도지) 처리: 몸통 1원으로 가정 → ratio 매우 큼 → 통과 가능
    body_for_ratio = max(body, 1)
    shadow_ratio = sig.yesterday_low_shadow / body_for_ratio
    sig.yesterday_shadow_ratio = round(shadow_ratio, 2)

    if sig.yesterday_low_shadow <= 0:
        sig.reasons_fail.append(
            f"아래꼬리 없음 (low_shadow={sig.yesterday_low_shadow}, 시·종가가 저가)"
        )
    elif shadow_ratio >= shadow_ratio_min:
        sig.reasons_pass.append(
            f"아래꼬리/몸통 = {shadow_ratio:.2f} ≥ {shadow_ratio_min:.1f}배 (개미털기)"
        )
    else:
        sig.reasons_fail.append(
            f"아래꼬리/몸통 = {shadow_ratio:.2f} < {shadow_ratio_min:.1f}배"
        )

    # === 2. 오늘 양봉: close > open + ATR × N ===
    t_open = _safe_int(today.get("stck_oprc"))
    t_close = _safe_int(today.get("stck_clpr"))
    sig.today_open = t_open
    sig.today_close = t_close

    atr = _compute_atr(rows, ATR_PERIOD)
    sig.atr_14 = round(atr, 2)
    bullish_threshold = t_open + atr * bullish_atr_mult
    sig.today_bullish_threshold = int(bullish_threshold)

    if t_close > bullish_threshold:
        sig.reasons_pass.append(
            f"양봉 강도 OK: 종가 {t_close:,} > 시가+{bullish_atr_mult}ATR "
            f"({int(bullish_threshold):,}, ATR={int(atr):,})"
        )
    else:
        sig.reasons_fail.append(
            f"양봉 약함: 종가 {t_close:,} ≤ {int(bullish_threshold):,}"
        )

    # === 3. 오늘 거래량 ≥ 5일 평균 × N ===
    t_vol = _safe_int(today.get("acml_vol"))
    sig.today_volume = t_vol

    # 어제~5일전 (오늘 제외)
    avg_vol_5d = 0.0
    sample_5d = rows[1:6]  # 어제~5일전
    if sample_5d:
        vols = [_safe_int(r.get("acml_vol")) for r in sample_5d]
        avg_vol_5d = sum(vols) / len(vols) if vols else 0.0
    sig.avg_volume_5d = round(avg_vol_5d, 1)

    vol_ratio = (t_vol / avg_vol_5d) if avg_vol_5d > 0 else 0.0
    sig.volume_ratio = round(vol_ratio, 2)

    if avg_vol_5d <= 0:
        sig.reasons_fail.append("5일 평균 거래량 0 (휴장 또는 데이터 누락)")
    elif vol_ratio >= volume_mult:
        sig.reasons_pass.append(
            f"거래량 {t_vol:,} ÷ 5일평균 {int(avg_vol_5d):,} "
            f"= {vol_ratio:.2f}배 ≥ {volume_mult:.1f}배 (관심 폭증)"
        )
    else:
        sig.reasons_fail.append(
            f"거래량 {vol_ratio:.2f}배 < {volume_mult:.1f}배 (관심 부족)"
        )

    # 3 조건 ALL 통과
    sig.trigger = len(sig.reasons_fail) == 0

    return sig


def scan_pool_for_support(broker, tickers: list[str]) -> list[SupportSignal]:
    """후보 풀 일괄 스캔 — trigger=True 우선 정렬."""
    results: list[SupportSignal] = []
    for ticker in tickers:
        sig = detect_support_pattern(broker, ticker)
        results.append(sig)
    # trigger 우선 + 거래량 비율 높은 순
    results.sort(key=lambda s: (not s.trigger, -s.volume_ratio))
    return results


def format_support_signal_for_telegram(sig: SupportSignal, name: str = "") -> str:
    """텔레그램 알림용 포맷."""
    head = f"🌱 받침 패턴 감지 [{name or sig.ticker}]"

    if not sig.trigger:
        return f"{head} (미트리거)\n  실패 사유: {' / '.join(sig.reasons_fail[:2])}"

    lines = [
        head,
        f"  어제: 아래꼬리/몸통 = {sig.yesterday_shadow_ratio:.2f}배 "
        f"(저가 {sig.yesterday_low:,})",
        f"  오늘: 시가 {sig.today_open:,} → 종가 {sig.today_close:,} "
        f"(ATR-14 {int(sig.atr_14):,})",
        f"  거래량: {sig.today_volume:,} ({sig.volume_ratio:.2f}배 폭증)",
        f"  ➜ MVP-4 재진입 판정 대기",
    ]
    return "\n".join(lines)
