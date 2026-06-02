"""C60 shadow forward ledger for 488080.

This module is intentionally read-only with respect to trading. It computes a
daily close/MA60 shadow ledger and never imports or calls order adapters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TICKER = "488080"
C60_MA_PERIOD = 60
C60_REENTRY_MA_PERIOD = 5
C60_REENTRY_RETURN_PCT = 0.02
C60_REENTRY_MA60_SLOPE_DAYS = 5
DEFAULT_SEED_EQUITY = 1.0
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SHADOW_DIR = PROJECT_ROOT / "data" / "shadow"
LEDGER_PATH = SHADOW_DIR / f"{DEFAULT_TICKER}_c60_shadow_ledger.json"
REPORT_PATH = SHADOW_DIR / f"{DEFAULT_TICKER}_c60_shadow_report.json"


@dataclass(frozen=True)
class C60LedgerRow:
    date: str
    ticker: str
    close: float
    ma60: float
    signal: str
    c60_position_state: str
    c60_equity_curve: float
    buyhold_equity_curve: float
    drawdown_c60: float
    drawdown_buyhold: float
    delta_vs_buyhold: float
    days_in_cash: int
    whipsaw_count: int
    missed_upside_after_exit: float
    avoided_drawdown_after_exit: float


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Korean/English OHLCV columns into lowercase English columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ("open", "시가"):
            col_map[col] = "open"
        elif key in ("high", "고가"):
            col_map[col] = "high"
        elif key in ("low", "저가"):
            col_map[col] = "low"
        elif key in ("close", "종가"):
            col_map[col] = "close"
        elif key in ("volume", "거래량"):
            col_map[col] = "volume"

    out = df.rename(columns=col_map).copy()
    if "close" not in out.columns:
        return pd.DataFrame()

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    return out


def load_daily_close(ticker: str = DEFAULT_TICKER, days: int = 260) -> pd.DataFrame:
    """Load daily OHLCV from local parquet first, then pykrx as a data-only fallback."""
    local_path = RAW_DATA_DIR / f"{ticker}.parquet"
    if local_path.exists():
        try:
            local_df = normalize_ohlcv(pd.read_parquet(local_path))
            if not local_df.empty:
                return local_df.tail(days)
        except Exception as exc:
            logger.warning("local parquet load failed for %s: %s", ticker, exc)

    try:
        from pykrx import stock as pykrx_stock

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        krx_df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)
        return normalize_ohlcv(krx_df).tail(days)
    except Exception as exc:
        logger.warning("pykrx daily load failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _sum_completed_or_open_exit_cycles(rows: list[C60LedgerRow]) -> tuple[float, float]:
    """Sum one missed/avoided value per exit cycle.

    Ledger rows store the current open-cycle missed/avoided value while in cash.
    For report-level accounting, use the REENTER row as the completed cycle's
    final value and include the latest row only when the final cycle is still
    open in cash.
    """
    missed_total = 0.0
    avoided_total = 0.0
    last_exit_seen = False

    for row in rows:
        if row.signal == "EXIT":
            last_exit_seen = True
        elif row.signal == "REENTER" and last_exit_seen:
            missed_total += row.missed_upside_after_exit
            avoided_total += row.avoided_drawdown_after_exit
            last_exit_seen = False

    last = rows[-1]
    if last_exit_seen and last.c60_position_state == "CASH":
        missed_total += last.missed_upside_after_exit
        avoided_total += last.avoided_drawdown_after_exit

    return missed_total, avoided_total


def _equity_mdd(values: Sequence[float]) -> float:
    """Return max drawdown for a local equity segment."""
    if not values:
        return 0.0

    peak = float(values[0])
    worst = 0.0
    for value in values:
        current = float(value)
        peak = max(peak, current)
        if peak > 0:
            worst = min(worst, current / peak - 1.0)
    return worst


def _segment_return(values: Sequence[float]) -> float:
    if len(values) < 2 or float(values[0]) == 0:
        return 0.0
    return float(values[-1]) / float(values[0]) - 1.0


def build_accelerated_c60_validation(
    rows: Iterable[C60LedgerRow],
    windows: Sequence[int] = (20, 40, 60, 120),
) -> dict:
    """Summarize many historical forward-like windows from one C60 ledger.

    This does not replace live forward observation, but it pulls the first
    decision point forward by replaying the already-known history in rolling
    windows. It remains shadow-only and does not touch trading paths.
    """
    ledger = list(rows)
    if not ledger:
        return {
            "status": "NO_DATA",
            "order_count": 0,
            "live_trading_state": "HOLD",
            "windows": {},
        }

    report = build_c60_report(ledger)
    window_summary: dict[str, dict] = {}

    for window in windows:
        if window <= 1 or len(ledger) < window:
            continue

        segments = []
        for start in range(0, len(ledger) - window + 1):
            segment = ledger[start : start + window]
            c60_values = [row.c60_equity_curve for row in segment]
            buyhold_values = [row.buyhold_equity_curve for row in segment]
            c60_return = _segment_return(c60_values)
            buyhold_return = _segment_return(buyhold_values)
            c60_mdd = _equity_mdd(c60_values)
            buyhold_mdd = _equity_mdd(buyhold_values)
            mdd_edge = c60_mdd - buyhold_mdd
            return_delta = c60_return - buyhold_return

            segments.append(
                {
                    "start_date": segment[0].date,
                    "end_date": segment[-1].date,
                    "c60_return": _round(c60_return),
                    "buyhold_return": _round(buyhold_return),
                    "return_delta": _round(return_delta),
                    "c60_mdd": _round(c60_mdd),
                    "buyhold_mdd": _round(buyhold_mdd),
                    "mdd_edge": _round(mdd_edge),
                    "days_in_cash_delta": segment[-1].days_in_cash - segment[0].days_in_cash,
                    "whipsaw_delta": segment[-1].whipsaw_count - segment[0].whipsaw_count,
                    "c60_state_end": segment[-1].c60_position_state,
                    "signal_end": segment[-1].signal,
                }
            )

        c60_mdd_better = [s for s in segments if s["mdd_edge"] > 0]
        c60_return_better = [s for s in segments if s["return_delta"] > 0]
        stress_segments = [s for s in segments if s["buyhold_mdd"] <= -0.2]
        latest = segments[-1]
        worst_buyhold = min(segments, key=lambda s: s["buyhold_mdd"])
        worst_c60 = min(segments, key=lambda s: s["c60_mdd"])

        window_summary[str(window)] = {
            "window_days": window,
            "segment_count": len(segments),
            "c60_mdd_better_count": len(c60_mdd_better),
            "c60_mdd_better_rate": _round(len(c60_mdd_better) / len(segments)),
            "c60_return_better_count": len(c60_return_better),
            "c60_return_better_rate": _round(len(c60_return_better) / len(segments)),
            "stress_segment_count_buyhold_mdd_20pct": len(stress_segments),
            "avg_return_delta": _round(sum(s["return_delta"] for s in segments) / len(segments)),
            "avg_mdd_edge": _round(sum(s["mdd_edge"] for s in segments) / len(segments)),
            "latest_segment": latest,
            "worst_buyhold_segment": worst_buyhold,
            "worst_c60_segment": worst_c60,
        }

    return {
        "ticker": report.get("ticker", DEFAULT_TICKER),
        "status": "ACCELERATED_SHADOW_REPLAY",
        "basis": "historical rolling windows from C60 shadow ledger",
        "ledger_start": ledger[0].date,
        "ledger_end": ledger[-1].date,
        "ledger_rows": len(ledger),
        "latest_signal": report.get("latest_signal"),
        "latest_c60_position_state": report.get("latest_c60_position_state"),
        "base_report": report,
        "windows": window_summary,
        "order_count": 0,
        "live_trading_state": "HOLD",
        "safety_note": "실주문 0건/HOLD 유지 — accelerated replay is analytics only",
    }


def build_c60_shadow_ledger(
    prices: pd.DataFrame,
    ticker: str = DEFAULT_TICKER,
    ma_period: int = C60_MA_PERIOD,
    seed_equity: float = DEFAULT_SEED_EQUITY,
) -> list[C60LedgerRow]:
    """Build C60 shadow ledger rows from daily close prices.

    C60 uses the close/MA60 decision after market close. EXIT and strict
    REENTER are recorded on the observation day, while the state change is
    applied from the next observed trading day.

    Strict REENTER:
    - close > MA5
    - close > MA60
    - daily return >= +2%
    - volume > previous observed volume
    - MA60 is rising versus five observations ago
    """
    df = normalize_ohlcv(prices)
    if df.empty or len(df) < ma_period + 1:
        return []

    df = df.copy()
    if "volume" not in df.columns:
        df["volume"] = pd.NA
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["ma5"] = df["close"].rolling(C60_REENTRY_MA_PERIOD).mean()
    df["ma60"] = df["close"].rolling(ma_period).mean()
    df["prev_volume"] = df["volume"].shift(1)
    df["ma60_up"] = df["ma60"] > df["ma60"].shift(C60_REENTRY_MA60_SLOPE_DAYS)
    df = df.dropna(subset=["ma60"])
    if len(df) < 2:
        return []

    first_close = float(df["close"].iloc[0])
    c60_equity = float(seed_equity)
    state = "HOLD" if float(df["close"].iloc[0]) > float(df["ma60"].iloc[0]) else "CASH"
    c60_peak = c60_equity
    buyhold_peak = float(seed_equity)
    days_in_cash = 0
    whipsaw_count = 0
    last_exit_close: float | None = None
    exit_equity: float | None = None
    rows: list[C60LedgerRow] = []
    prev_close = first_close

    for dt, row in df.iterrows():
        close = float(row["close"])
        ma60 = float(row["ma60"])
        ma5 = float(row["ma5"]) if pd.notna(row["ma5"]) else 0.0
        volume = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
        prev_volume = float(row["prev_volume"]) if pd.notna(row["prev_volume"]) else 0.0

        daily_return = 0.0
        if rows:
            daily_return = close / prev_close - 1.0
            if state == "HOLD":
                c60_equity *= 1.0 + daily_return
            else:
                days_in_cash += 1

        buyhold_equity = seed_equity * (close / first_close)
        c60_peak = max(c60_peak, c60_equity)
        buyhold_peak = max(buyhold_peak, buyhold_equity)

        signal = "HOLD" if state == "HOLD" else "CASH"
        next_state = state
        missed = 0.0
        avoided = 0.0

        if state == "HOLD" and close <= ma60:
            signal = "EXIT"
            next_state = "CASH"
            last_exit_close = close
            exit_equity = c60_equity
        elif state == "CASH":
            strict_reenter = (
                close > ma5
                and close > ma60
                and daily_return >= C60_REENTRY_RETURN_PCT
                and volume > prev_volume
                and bool(row["ma60_up"])
            )
            if strict_reenter:
                signal = "REENTER"
                next_state = "HOLD"
                if last_exit_close is not None and close > last_exit_close:
                    whipsaw_count += 1

        if state == "CASH" and last_exit_close and exit_equity is not None:
            missed = max(0.0, close / last_exit_close - 1.0) * exit_equity
            avoided = max(0.0, 1.0 - close / last_exit_close) * exit_equity

        drawdown_c60 = c60_equity / c60_peak - 1.0
        drawdown_buyhold = buyhold_equity / buyhold_peak - 1.0
        ledger_date = pd.Timestamp(dt).strftime("%Y-%m-%d")

        rows.append(
            C60LedgerRow(
                date=ledger_date,
                ticker=ticker,
                close=_round(close, 4),
                ma60=_round(ma60, 4),
                signal=signal,
                c60_position_state=state,
                c60_equity_curve=_round(c60_equity),
                buyhold_equity_curve=_round(buyhold_equity),
                drawdown_c60=_round(drawdown_c60),
                drawdown_buyhold=_round(drawdown_buyhold),
                delta_vs_buyhold=_round(c60_equity - buyhold_equity),
                days_in_cash=days_in_cash,
                whipsaw_count=whipsaw_count,
                missed_upside_after_exit=_round(missed),
                avoided_drawdown_after_exit=_round(avoided),
            )
        )

        state = next_state
        prev_close = close

    return rows


def build_c60_report(rows: Iterable[C60LedgerRow]) -> dict:
    ledger = list(rows)
    if not ledger:
        return {
            "ticker": DEFAULT_TICKER,
            "status": "NO_DATA",
            "order_count": 0,
            "live_trading_state": "HOLD",
        }

    last = ledger[-1]
    c60_final_return = last.c60_equity_curve - DEFAULT_SEED_EQUITY
    buyhold_final_return = last.buyhold_equity_curve - DEFAULT_SEED_EQUITY
    mdd_c60 = min(r.drawdown_c60 for r in ledger)
    mdd_buyhold = min(r.drawdown_buyhold for r in ledger)
    total_missed, total_avoided = _sum_completed_or_open_exit_cycles(ledger)
    insurance_value = total_avoided - total_missed
    conclusion = (
        "보험료 가치 있음"
        if mdd_c60 > mdd_buyhold and insurance_value >= 0
        else "보험료 관찰 지속"
    )

    return {
        "ticker": last.ticker,
        "status": "SHADOW_ONLY",
        "latest_date": last.date,
        "latest_signal": last.signal,
        "latest_c60_position_state": last.c60_position_state,
        "c60_mdd": _round(mdd_c60),
        "buyhold_mdd": _round(mdd_buyhold),
        "c60_final_return": _round(c60_final_return),
        "buyhold_final_return": _round(buyhold_final_return),
        "days_in_cash": last.days_in_cash,
        "whipsaw_count": last.whipsaw_count,
        "avoided_drawdown_amount": _round(total_avoided),
        "missed_upside_amount": _round(total_missed),
        "insurance_value": _round(insurance_value),
        "one_line_conclusion": conclusion,
        "order_count": 0,
        "live_trading_state": "HOLD",
        "safety_note": "실주문 0건/HOLD 유지 — 외부 주문 함수 미사용",
    }


def save_shadow_outputs(
    rows: list[C60LedgerRow],
    ledger_path: Path = LEDGER_PATH,
    report_path: Path = REPORT_PATH,
) -> tuple[Path, Path, dict]:
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    report = build_c60_report(rows)

    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return ledger_path, report_path, report
