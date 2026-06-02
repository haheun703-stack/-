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
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TICKER = "488080"
C60_MA_PERIOD = 60
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


def build_c60_shadow_ledger(
    prices: pd.DataFrame,
    ticker: str = DEFAULT_TICKER,
    ma_period: int = C60_MA_PERIOD,
    seed_equity: float = DEFAULT_SEED_EQUITY,
) -> list[C60LedgerRow]:
    """Build C60 shadow ledger rows from daily close prices.

    C60 uses the close/MA60 decision after market close. EXIT and REENTER are
    recorded on the observation day, while the state change is applied from the
    next observed trading day.
    """
    df = normalize_ohlcv(prices)
    if df.empty or len(df) < ma_period + 1:
        return []

    df = df.copy()
    df["ma60"] = df["close"].rolling(ma_period).mean()
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
        elif state == "CASH" and close > ma60:
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
    total_avoided = max(r.avoided_drawdown_after_exit for r in ledger)
    total_missed = max(r.missed_upside_after_exit for r in ledger)
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
