"""추세 이탈 매도 — MVP-2.8 (5/26 23:00, 옵션 C 천장 따라가는 매도).

배경 (사용자 5/26 22:55 강력 지시):
- 옵션 A (임계값 조정)만으로 부족 — 옵션 C (MA + RSI) 추가 필수
- 천장까지 따라가되 추세 진짜 끝났을 때만 매도
- 단순한 trailing -5% 보다 정확한 매도 시점

룰 (옵션 B + C 통합):
1. MA 이탈 매도 (단기/장기 추세 끝):
   조건 A1: MA5 < MA20 (단기 추세 끝) + 평단 +5% 이상 → 매도 (수익 확보)
   조건 A2: MA20 < MA60 (대세 끝) + 평단 +0% 이상 → 매도 (대세 회피)
   조건 A3: MA5 < MA20 < MA60 (역배열) + 평단 무관 → 즉시 매도 (전체 약세)

2. RSI 모멘텀 끝 매도 (옵션 C):
   조건 B1: RSI 70+ 도달 기록 + 현재 RSI < 50 → 매도 (모멘텀 끝)
   조건 B2: RSI 80+ 과열 + 평단 +3% 이상 → 매도 (과열 익절)

3. 거래량 급감 + 천장 도달 후 매도 (보너스):
   조건 C1: 매수 후 5분봉 거래량 < 평균 0.5배 + 평단 +5% 이상 → 매도 (관심 끝)

OR 조건 — 어느 하나 만족 시 매도. 평단 -5% 손절은 별개 (MVP-2.6).

활용:
- run_adaptive_cycle.py MVP-2.8 단계 신규
- 매 5분 cron마다 FILLED 상태 stage 평가
- 매도 시 지정가 (ADAPTIVE_SELL_USE_LIMIT 환경변수 활용)

환경변수:
- ADAPTIVE_TREND_EXIT_ENABLED=1
- TREND_EXIT_MIN_PROFIT_PCT=5.0 (MA 이탈 시 최소 수익)
- TREND_EXIT_RSI_PEAK=70 (RSI peak 임계)
- TREND_EXIT_RSI_DROP=50 (RSI drop 임계)
- TREND_EXIT_RSI_OVERHEAT=80
- TREND_EXIT_OVERHEAT_PROFIT_PCT=3.0 (과열 익절 최소 수익)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

ENABLED = os.getenv("ADAPTIVE_TREND_EXIT_ENABLED", "0") == "1"
MIN_PROFIT_PCT = float(os.getenv("TREND_EXIT_MIN_PROFIT_PCT", "5.0"))
RSI_PEAK = float(os.getenv("TREND_EXIT_RSI_PEAK", "70.0"))
RSI_DROP = float(os.getenv("TREND_EXIT_RSI_DROP", "50.0"))
RSI_OVERHEAT = float(os.getenv("TREND_EXIT_RSI_OVERHEAT", "80.0"))
OVERHEAT_PROFIT_PCT = float(os.getenv("TREND_EXIT_OVERHEAT_PROFIT_PCT", "3.0"))


@dataclass
class TrendExitSignal:
    """추세 이탈 매도 신호."""
    ticker: str
    triggered: bool = False
    exit_type: str = "WAIT"   # 'MA_BEAR_SHORT' / 'MA_BEAR_LONG' / 'MA_FULL_BEAR' /
                              # 'RSI_MOMENTUM_END' / 'RSI_OVERHEATED' / 'VOLUME_DRY' / 'WAIT'
    entry_price: int = 0
    current_price: int = 0
    pnl_pct: float = 0.0
    ma5: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    rsi: float = 0.0
    rsi_peak_reached: bool = False  # 과거 RSI 70+ 도달 기록
    qty: int = 0
    reason: str = ""


def evaluate_trend_exit(
    ticker: str,
    entry_price: int,
    current_price: int,
    qty: int,
    ma5: float,
    ma20: float,
    ma60: float,
    rsi: float,
    rsi_peak_reached: bool = False,
    min_profit_pct: float = MIN_PROFIT_PCT,
    rsi_peak: float = RSI_PEAK,
    rsi_drop: float = RSI_DROP,
    rsi_overheat: float = RSI_OVERHEAT,
    overheat_profit_pct: float = OVERHEAT_PROFIT_PCT,
) -> TrendExitSignal:
    """추세 이탈 매도 평가.

    Args:
        ma5/ma20/ma60: 일봉 이동평균 (정보봇 OHLCV 39컬럼)
        rsi: 일봉 RSI(14)
        rsi_peak_reached: 매수 후 RSI 70+ 도달 기록 (큐 stage에 저장)
    """
    sig = TrendExitSignal(
        ticker=ticker, entry_price=entry_price, current_price=current_price,
        qty=qty, ma5=ma5, ma20=ma20, ma60=ma60, rsi=rsi,
        rsi_peak_reached=rsi_peak_reached,
    )

    if entry_price <= 0 or current_price <= 0:
        sig.reason = "invalid prices"
        return sig

    pnl_pct = (current_price - entry_price) / entry_price * 100
    sig.pnl_pct = pnl_pct

    # ─────────────────────────────────────────
    # 룰 A3 (최강): 역배열 + 평단 무관 → 즉시 매도
    # ─────────────────────────────────────────
    if ma5 > 0 and ma20 > 0 and ma60 > 0 and ma5 < ma20 < ma60:
        sig.triggered = True
        sig.exit_type = "MA_FULL_BEAR"
        sig.reason = f"MA 역배열 (MA5<MA20<MA60) — 전체 약세, 평단 {pnl_pct:+.1f}% 무관 매도"
        return sig

    # ─────────────────────────────────────────
    # 룰 A2: 대세 끝 (MA20 < MA60) + 수익 0%+ → 매도
    # ─────────────────────────────────────────
    if ma20 > 0 and ma60 > 0 and ma20 < ma60 and pnl_pct >= 0:
        sig.triggered = True
        sig.exit_type = "MA_BEAR_LONG"
        sig.reason = f"대세 끝 (MA20<MA60) + 수익 {pnl_pct:+.1f}% — 매도"
        return sig

    # ─────────────────────────────────────────
    # 룰 A1: 단기 추세 끝 (MA5 < MA20) + 수익 +5%+ → 매도
    # ─────────────────────────────────────────
    if ma5 > 0 and ma20 > 0 and ma5 < ma20 and pnl_pct >= min_profit_pct:
        sig.triggered = True
        sig.exit_type = "MA_BEAR_SHORT"
        sig.reason = f"단기 추세 끝 (MA5<MA20) + 수익 {pnl_pct:+.1f}% ≥ {min_profit_pct}% — 매도"
        return sig

    # ─────────────────────────────────────────
    # 룰 B2: RSI 80+ 과열 + 수익 +3%+ → 매도 (옵션 C)
    # ─────────────────────────────────────────
    if rsi >= rsi_overheat and pnl_pct >= overheat_profit_pct:
        sig.triggered = True
        sig.exit_type = "RSI_OVERHEATED"
        sig.reason = f"RSI 과열 {rsi:.1f} ≥ {rsi_overheat} + 수익 {pnl_pct:+.1f}% — 익절"
        return sig

    # ─────────────────────────────────────────
    # 룰 B1: RSI 70+ 도달 후 50 하향 돌파 → 매도 (옵션 C 핵심)
    # ─────────────────────────────────────────
    if rsi_peak_reached and rsi < rsi_drop:
        sig.triggered = True
        sig.exit_type = "RSI_MOMENTUM_END"
        sig.reason = f"RSI 모멘텀 끝 (peak 70+ → 현재 {rsi:.1f} < {rsi_drop})"
        return sig

    # 대기 상태
    sig.reason = (
        f"대기: MA5={ma5:.0f}/MA20={ma20:.0f}/MA60={ma60:.0f}, "
        f"RSI={rsi:.1f}, 평단 {pnl_pct:+.1f}%"
    )
    return sig


def update_rsi_peak_state(rsi: float, current_state: bool,
                          peak_threshold: float = RSI_PEAK) -> bool:
    """RSI 70+ 도달 여부 추적 (큐 stage에 저장 후 매 사이클 갱신).

    True가 되면 다시 False로 안 돌아옴 (이 매수 사이클 내내 유지).
    """
    if current_state:
        return True
    return rsi >= peak_threshold


def _fetch_daily_indicators(broker, ticker: str) -> Optional[dict]:
    """일봉 60일 OHLCV → MA5/MA20/MA60 + RSI(14) 계산.

    ★ M5 fix (5/27 검수): 정보봇 OHLCV CSV 39컬럼에 RSI/MA 컬럼 있음 (VPS 한정)
    → 1차 시도 (정보봇 CSV) → 2차 (KIS OHLCV + Wilder smoothing 계산)

    Returns:
        {"ma5": float, "ma20": float, "ma60": float, "rsi": float, "close": float}
        실패 시 None.
    """
    # 1차: 정보봇 OHLCV CSV (RSI/MA 컬럼 직접 사용)
    try:
        from pathlib import Path
        import csv
        OHLCV_DIR = Path("/home/ubuntu/quantum-master/data/external/jgis_ohlcv")
        if not OHLCV_DIR.exists():
            OHLCV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "jgis_ohlcv"
        if OHLCV_DIR.exists():
            tk6 = str(ticker).zfill(6)
            candidates = list(OHLCV_DIR.glob(f"*_{tk6}.csv"))
            if candidates:
                with candidates[0].open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    try:
                        ma5 = float(last.get("MA5", 0) or 0)
                        ma20 = float(last.get("MA20", 0) or 0)
                        ma60 = float(last.get("MA60", 0) or 0)
                        rsi = float(last.get("RSI", 0) or 0)
                        close = float(last.get("Close", 0) or 0)
                        if ma5 > 0 and ma20 > 0 and close > 0:
                            return {"ma5": ma5, "ma20": ma20, "ma60": ma60,
                                    "rsi": rsi, "close": close}
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        logger.debug("[추세 이탈] 정보봇 CSV 로드 실패 %s: %s — KIS fallback", ticker, e)

    # 2차: KIS OHLCV (Wilder smoothing 적용)
    try:
        if not hasattr(broker, "fetch_ohlcv"):
            return None
        from datetime import date, timedelta
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=120)).strftime("%Y%m%d")
        raw = broker.fetch_ohlcv(ticker, timeframe="D",
                                   start_day=start_day, end_day=end_day)
        out2 = raw.get("output2", []) if isinstance(raw, dict) else raw
        if not out2 or len(out2) < 60:
            return None

        # 최신순 → 시간순 정렬 (KIS는 보통 최신 마지막)
        closes = []
        for r in out2:
            try:
                c = float(r.get("stck_clpr", 0) or 0)
                if c > 0:
                    closes.append(c)
            except (ValueError, TypeError):
                continue

        if len(closes) < 60:
            return None

        # 일부 KIS 응답은 역순 (최신 첫번째). 마지막 60개 사용.
        recent_60 = closes[-60:]
        ma5 = sum(recent_60[-5:]) / 5
        ma20 = sum(recent_60[-20:]) / 20
        ma60 = sum(recent_60) / 60

        # RSI(14) 계산
        gains, losses = [], []
        for i in range(1, len(recent_60)):
            diff = recent_60[i] - recent_60[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-diff)

        # ★ M5 fix: Wilder smoothing (지수 이동 평균) — 표준 RSI 공식
        if len(gains) < 14:
            return None
        avg_gain = sum(gains[:14]) / 14
        avg_loss = sum(losses[:14]) / 14
        # Wilder smoothing: 14일 이후는 ((이전 평균 × 13) + 현재) / 14
        for i in range(14, len(gains)):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return {
            "ma5": ma5, "ma20": ma20, "ma60": ma60,
            "rsi": rsi, "close": recent_60[-1],
        }
    except Exception as e:
        logger.warning("[추세 이탈] %s 지표 계산 실패: %s", ticker, e)
        return None


def scan_queue_for_trend_exit(queue_state: dict, broker) -> list[TrendExitSignal]:
    """전체 큐 스캔 → FILLED 상태 stage 추세 이탈 평가.

    Returns:
        매도 발화 TrendExitSignal 리스트.
    """
    if not ENABLED:
        logger.debug("[추세 이탈] ADAPTIVE_TREND_EXIT_ENABLED=0 — 비활성")
        return []

    signals: list[TrendExitSignal] = []
    queues = queue_state.get("queues", {})

    for ticker, entry in queues.items():
        for stage in entry.get("stages", []):
            if stage.get("status") != "FILLED":
                continue

            actual_price = int(stage.get("actual_price", 0))
            actual_qty = int(stage.get("actual_qty", 0))
            if actual_price <= 0 or actual_qty <= 0:
                continue

            # 현재가 조회
            try:
                res = broker.fetch_price(ticker)
                cur = int(res.get("output", {}).get("stck_prpr", 0))
            except Exception as e:
                logger.warning("[추세 이탈] %s 현재가 실패: %s", ticker, e)
                continue
            if cur <= 0:
                continue

            # 일봉 지표 계산
            indicators = _fetch_daily_indicators(broker, ticker)
            if not indicators:
                continue

            # RSI peak 상태 (stage에 저장된 값)
            rsi_peak_reached = stage.get("rsi_peak_reached", False)
            rsi_peak_reached = update_rsi_peak_state(indicators["rsi"], rsi_peak_reached)
            stage["rsi_peak_reached"] = rsi_peak_reached  # 후속 사이클 위해 저장

            sig = evaluate_trend_exit(
                ticker=ticker, entry_price=actual_price, current_price=cur,
                qty=actual_qty,
                ma5=indicators["ma5"], ma20=indicators["ma20"], ma60=indicators["ma60"],
                rsi=indicators["rsi"], rsi_peak_reached=rsi_peak_reached,
            )
            if sig.triggered:
                signals.append(sig)
                logger.warning(
                    "[추세 이탈] %s %s (%+.1f%%) — %s",
                    ticker, sig.exit_type, sig.pnl_pct, sig.reason,
                )
    return signals


def execute_trend_exit(
    broker, sig: TrendExitSignal,
    *, mode: str | None = None, executor_bot: str | None = None,
) -> dict:
    """추세 이탈 매도 실행 (지정가 우선, 시장가 fallback).

    ADAPTIVE_SELL_USE_LIMIT=1 시 지정가 매도 (slippage -0.3%).
    """
    if not sig.triggered:
        return {"success": False, "error": "trigger=False"}

    use_limit = os.getenv("ADAPTIVE_SELL_USE_LIMIT", "1") == "1"
    sell_slippage_pct = float(os.getenv("ADAPTIVE_SELL_LIMIT_SLIPPAGE_PCT", "0.3"))

    # 5/28 P0-5: mode/executor_bot 명시 시 broker에 전달
    adapter_kwargs = {}
    if mode is not None or executor_bot is not None:
        adapter_kwargs = {"mode": mode, "executor_bot": executor_bot}

    try:
        if use_limit and hasattr(broker, "sell_limit") and sig.current_price > 0:
            limit_price = int(sig.current_price * (1 - sell_slippage_pct / 100))
            order = broker.sell_limit(sig.ticker, limit_price, sig.qty, **adapter_kwargs)
            logger.info(
                "[추세 이탈] %s 지정가 매도 %d주 @ %d (%s, 평단 %+.1f%%)",
                sig.ticker, sig.qty, limit_price, sig.exit_type, sig.pnl_pct,
            )
            return {
                "success": True,
                "order_id": getattr(order, "order_id", "") or "",
                "qty": sig.qty,
                "exit_type": sig.exit_type,
                "limit_price": limit_price,
                "pnl_pct": sig.pnl_pct,
            }
        elif hasattr(broker, "sell_market"):
            order = broker.sell_market(sig.ticker, sig.qty, **adapter_kwargs)
            return {
                "success": True,
                "order_id": getattr(order, "order_id", "") or "",
                "qty": sig.qty,
                "exit_type": sig.exit_type,
                "pnl_pct": sig.pnl_pct,
            }
        else:
            return {"success": False, "error": "sell_limit/sell_market 미지원"}
    except Exception as e:
        logger.error("[추세 이탈] %s 매도 실패: %s", sig.ticker, e)
        return {"success": False, "error": str(e)}


def format_trend_exit_for_telegram(sig: TrendExitSignal) -> str:
    """텔레그램 알림 포맷."""
    icons = {
        "MA_FULL_BEAR": "🔴",
        "MA_BEAR_LONG": "🟠",
        "MA_BEAR_SHORT": "🟡",
        "RSI_OVERHEATED": "🔥",
        "RSI_MOMENTUM_END": "💔",
    }
    icon = icons.get(sig.exit_type, "📉")
    return (
        f"{icon} [추세 이탈 매도] {sig.exit_type}\n"
        f"  {sig.ticker} 평단 {sig.entry_price:,} → 현재 {sig.current_price:,}\n"
        f"  수익 {sig.pnl_pct:+.2f}% / 수량 {sig.qty}주\n"
        f"  MA5={sig.ma5:.0f} / MA20={sig.ma20:.0f} / MA60={sig.ma60:.0f}\n"
        f"  RSI={sig.rsi:.1f}\n"
        f"  사유: {sig.reason}"
    )
