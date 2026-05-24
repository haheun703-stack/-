"""적응형 포지션 매매법 MVP-1 — 천장 감지 + -3% 트리거 매도.

배경 (퐝가님 5/23 흐름):
  "내리면 사고, 끝까지 오르고 -3%면 팔고, 조정의 시작점에서 다시 사고"

  4 차트 검증으로 본 흐름이 중소형 소부장에서 가장 잘 작동.
  MVP-1은 6단계 흐름의 [3단계 천장 -3% 트리거 매도]만 구현.
  MVP-2~4 (분할매수/받침/재진입)는 별도 모듈로 단계별.

MVP-1 기능:
  1. 보유 종목 + 후보 풀의 N일 최고가 감지 (천장 라인)
  2. 현재가 ≥ 천장 × 0.97 + 신선도 검증 (최근 K일 내 천장 도달)
  3. 트리거 시:
     - 텔레그램 알림 (천장 가격, 현재가, 차이)
     - 자동 매도 (옵션 ADAPTIVE_AUTO_SELL=1일 때만)

5/22 인사이트 통합:
  - 자비스 9 안전선 통과한 종목만 본 흐름 적용
  - STEP 5 ★★★ 이상 화이트리스트 (soubujang_pool 결과)
  - KILL_SWITCH 발동 시 본 흐름 자동 정지

사용:
  from src.use_cases.adaptive_position_manager import detect_peak_signal
  signal = detect_peak_signal(broker, "067310")
  if signal["trigger"]:
      # 텔레그램 알림 + (옵션) 매도
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# === 임계 (.env 동적, 1주차는 보수적) ===
PEAK_LOOKBACK_DAYS = int(os.getenv("ADAPTIVE_PEAK_LOOKBACK_DAYS", "30"))
PEAK_TRIGGER_PCT = float(os.getenv("ADAPTIVE_PEAK_TRIGGER_PCT", "0.97"))      # 천장 -3%
PEAK_FRESHNESS_DAYS = int(os.getenv("ADAPTIVE_PEAK_FRESHNESS_DAYS", "5"))     # 최근 5일 내 천장 도달
AUTO_SELL = os.getenv("ADAPTIVE_AUTO_SELL", "0") == "1"                       # 1주차는 알림만
SELL_RATIO = float(os.getenv("ADAPTIVE_SELL_RATIO", "1.0"))                   # 전량 vs 부분

# KILL_SWITCH 연동
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"


@dataclass
class PeakSignal:
    """천장 감지 시그널."""

    ticker: str
    current_price: int = 0
    peak_price: int = 0
    peak_date: Optional[str] = None
    days_since_peak: int = -1
    pct_from_peak: float = 0.0          # 천장 대비 % (예: -1.8 = -1.8%)
    trigger: bool = False
    reasons_pass: list[str] = field(default_factory=list)
    reasons_fail: list[str] = field(default_factory=list)
    auto_sell_eligible: bool = False
    error: Optional[str] = None


def _is_kill_switch_active() -> bool:
    """KILL_SWITCH 발동 여부."""
    return KILL_SWITCH_PATH.exists()


def _fetch_recent_ohlcv(broker, ticker: str, days: int = PEAK_LOOKBACK_DAYS) -> list[dict]:
    """최근 N일 OHLCV (KIS API)."""
    try:
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
        res = broker.fetch_ohlcv(ticker, timeframe="D", start_day=start_day, end_day=end_day)
        rows = res.get("output2", []) if res else []
        # output2는 최신순 → 오래된 순
        return rows[:days]  # 최신 N일
    except Exception as e:
        logger.warning("OHLCV fetch %s 실패: %s", ticker, e)
        return []


def _fetch_current_price(broker, ticker: str) -> int:
    """현재가 fetch."""
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}
        return int(str(output.get("stck_prpr", 0)).replace(",", "") or 0)
    except Exception as e:
        logger.warning("price fetch %s 실패: %s", ticker, e)
        return 0


def detect_peak_signal(
    broker,
    ticker: str,
    current_price: Optional[int] = None,
    lookback_days: int = PEAK_LOOKBACK_DAYS,
    trigger_pct: float = PEAK_TRIGGER_PCT,
    freshness_days: int = PEAK_FRESHNESS_DAYS,
) -> PeakSignal:
    """천장 감지 + -3% 트리거 매도 신호.

    Args:
        broker: KIS broker
        ticker: 종목 코드
        current_price: 현재가 (None이면 자동 fetch)
        lookback_days: 천장 감지 기간 (기본 30일)
        trigger_pct: 천장 대비 트리거 비율 (기본 0.97 = -3%)
        freshness_days: 천장 신선도 (기본 5일 내 도달)

    Returns:
        PeakSignal
    """
    sig = PeakSignal(ticker=ticker)

    # KILL_SWITCH 확인
    if _is_kill_switch_active():
        sig.reasons_fail.append("KILL_SWITCH 발동 중 → 본 흐름 정지")
        return sig

    # 현재가
    if current_price is None:
        current_price = _fetch_current_price(broker, ticker)
    sig.current_price = current_price
    if current_price <= 0:
        sig.error = "현재가 fetch 실패"
        sig.reasons_fail.append(sig.error)
        return sig

    # OHLCV 30일
    rows = _fetch_recent_ohlcv(broker, ticker, lookback_days)
    if len(rows) < 5:
        sig.error = f"OHLCV {len(rows)}건 부족 (5 미만)"
        sig.reasons_fail.append(sig.error)
        return sig

    # 천장 (최근 lookback_days 일 중 최고가)
    peak_price = 0
    peak_date_str = ""
    for r in rows:
        try:
            high = float(r.get("stck_hgpr", 0))
            d_str = r.get("stck_bsop_date", "")
            if high > peak_price:
                peak_price = high
                peak_date_str = d_str
        except (ValueError, TypeError):
            continue

    if peak_price <= 0:
        sig.error = "천장 추출 실패"
        sig.reasons_fail.append(sig.error)
        return sig

    sig.peak_price = int(peak_price)
    sig.peak_date = peak_date_str

    # 천장 도달 신선도
    try:
        peak_dt = datetime.strptime(peak_date_str, "%Y%m%d").date()
        days_since = (date.today() - peak_dt).days
        sig.days_since_peak = days_since
    except Exception:
        days_since = 999
        sig.days_since_peak = -1

    # 천장 대비 위치
    pct_from_peak = (current_price / peak_price - 1) * 100
    sig.pct_from_peak = round(pct_from_peak, 2)

    # 조건 1: 현재가가 천장 × 0.97 이상 (= 천장 -3% 이내)
    threshold_price = peak_price * trigger_pct
    if current_price >= threshold_price:
        sig.reasons_pass.append(
            f"천장 -3% 진입: 현재 {current_price:,} ≥ {threshold_price:,.0f} "
            f"(천장 {peak_price:,}, {pct_from_peak:+.2f}%)"
        )
    else:
        sig.reasons_fail.append(
            f"천장 -3% 미진입: 현재 {current_price:,} < {threshold_price:,.0f} "
            f"({pct_from_peak:+.2f}%)"
        )

    # 조건 2: 천장 도달 신선도 (최근 K일 내)
    if 0 <= days_since <= freshness_days:
        sig.reasons_pass.append(f"천장 신선도 OK: {days_since}일 전 도달 (≤ {freshness_days}일)")
    elif days_since > freshness_days:
        sig.reasons_fail.append(f"천장 묵힘: {days_since}일 전 (> {freshness_days}일, 이미 조정 완료)")
    else:
        sig.reasons_fail.append("천장 도달일 추출 실패")

    # 트리거 여부 (두 조건 ALL 통과)
    sig.trigger = len(sig.reasons_fail) == 0

    # 자동 매도 가능 여부
    if sig.trigger and AUTO_SELL:
        sig.auto_sell_eligible = True

    return sig


def format_peak_signal_for_telegram(sig: PeakSignal, name: str = "") -> str:
    """텔레그램 알림용 포맷."""
    head = f"🔔 천장 -3% 트리거 [{name or sig.ticker}]"
    if not sig.trigger:
        return f"{head}\n  미진입: {' / '.join(sig.reasons_fail[:2])}"

    lines = [
        head,
        f"  현재가: {sig.current_price:,}",
        f"  30일 천장: {sig.peak_price:,} ({sig.days_since_peak}일 전)",
        f"  천장 대비: {sig.pct_from_peak:+.2f}%",
        f"  자동매도: {'★ ON (즉시 실행)' if sig.auto_sell_eligible else 'OFF (알림만)'}",
    ]
    for r in sig.reasons_pass[:3]:
        lines.append(f"  ✓ {r}")
    return "\n".join(lines)


def scan_holdings_for_peaks(
    broker,
    holdings: dict[str, dict],
    soubujang_pool: Optional[dict] = None,
) -> list[PeakSignal]:
    """보유 종목 + 후보 풀 일괄 스캔.

    Args:
        broker: KIS broker
        holdings: 보유 종목 dict {ticker: {qty, avg_price, ...}}
        soubujang_pool: 후보 풀 (선택, 통과 종목만 추가 스캔)

    Returns:
        PeakSignal 리스트 (trigger=True인 것 우선)
    """
    results = []
    tickers = set(holdings.keys())

    # 후보 풀의 통과 종목도 포함 (천장 임박 종목 조기 알림)
    if soubujang_pool:
        for t, info in soubujang_pool.items():
            if info.get("passed"):
                tickers.add(t)

    for ticker in tickers:
        sig = detect_peak_signal(broker, ticker)
        results.append(sig)

    # 트리거된 것부터 정렬
    results.sort(key=lambda s: (not s.trigger, -s.pct_from_peak))
    return results


def execute_auto_sell(broker, sig: PeakSignal, holdings_qty: int) -> dict:
    """자동 매도 실행 (ADAPTIVE_AUTO_SELL=1일 때만).

    매도 성공 시 → MVP-2 분할매수 큐 자동 등록 (peak_price + 가용 현금 기반).
    AUTO_SELL=0 (1주차)이면 매도 안 됨 → 큐 등록도 자동으로 미발생.

    Returns:
        {"success": bool, "order_id": str, "qty": int, "price": int, "error": str,
         "queue_registered": bool, "queue_error": str}
    """
    if not sig.auto_sell_eligible:
        return {"success": False, "error": "auto_sell_eligible=False"}
    if not AUTO_SELL:
        return {"success": False, "error": "ADAPTIVE_AUTO_SELL=0"}
    if holdings_qty <= 0:
        return {"success": False, "error": f"보유 수량 {holdings_qty}"}

    sell_qty = max(1, int(holdings_qty * SELL_RATIO))

    try:
        # 시장가 매도
        res = broker.create_market_sell_order(
            symbol=sig.ticker,
            quantity=sell_qty,
        )
        result = {
            "success": True,
            "order_id": res.get("output", {}).get("ODNO", ""),
            "qty": sell_qty,
            "price": sig.current_price,
            "peak_price": sig.peak_price,
            "pct_from_peak": sig.pct_from_peak,
        }

        # MVP-2 연동: 매도 후 분할매수 큐 자동 등록
        try:
            from src.use_cases.adaptive_buy_queue import register_buy_queue

            cash = 0
            if hasattr(broker, "get_available_cash"):
                try:
                    cash = int(broker.get_available_cash() or 0)
                except Exception:
                    cash = 0

            if cash >= 100_000 and sig.peak_price > 0:
                queue_result = register_buy_queue(
                    ticker=sig.ticker,
                    peak_price=sig.peak_price,
                    available_cash=cash,
                    name=sig.ticker,
                )
                result["queue_registered"] = bool(queue_result.get("success"))
                if not queue_result.get("success"):
                    result["queue_error"] = queue_result.get("error", "")
            else:
                result["queue_registered"] = False
                result["queue_error"] = f"가용 현금 {cash:,} 미달 또는 peak 부재"
        except Exception as e:
            logger.warning("MVP-2 큐 자동 등록 실패: %s", e)
            result["queue_registered"] = False
            result["queue_error"] = str(e)

        return result
    except Exception as e:
        logger.error("auto sell %s 실패: %s", sig.ticker, e)
        return {"success": False, "error": str(e)}
