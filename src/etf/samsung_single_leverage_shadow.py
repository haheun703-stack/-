"""Samsung single-stock leverage shadow tracking.

Analytics only:
- Signal source: Samsung Electronics common stock (005930).
- Tracked product: Samsung single-stock 2x leverage ETF/ETN ticker.
- No broker/order/scheduler imports.
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
UNDERLYING_TICKER = "005930"
DEFAULT_LEVERAGE_TICKER = "0193W0"
SEMI_LEVERAGE_TICKER = "488080"
DEFAULT_LEVERAGE_MULTIPLIER = 2.0
DEFAULT_SEED_EQUITY = 1.0
MA20_PERIOD = 20
MA60_PERIOD = 60
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SHADOW_DIR = PROJECT_ROOT / "data" / "shadow"
LEDGER_PATH = SHADOW_DIR / f"{DEFAULT_LEVERAGE_TICKER}_samsung_single_lev_shadow_ledger.json"
REPORT_PATH = SHADOW_DIR / f"{DEFAULT_LEVERAGE_TICKER}_samsung_single_lev_shadow_report.json"


@dataclass(frozen=True)
class SamsungLeverageLedgerRow:
    date: str
    signal_ticker: str
    leverage_ticker: str
    underlying_close: float
    leverage_close: float
    ma20: float
    ma60: float
    c60_signal: str
    c60_state: str
    sajang_signal: str
    sajang_state: str
    c60_equity_curve: float
    sajang_equity_curve: float
    leverage_buyhold_equity_curve: float
    underlying_buyhold_equity_curve: float
    drawdown_c60: float
    drawdown_sajang: float
    drawdown_leverage_buyhold: float
    drawdown_underlying_buyhold: float
    c60_days_in_cash: int
    sajang_days_in_cash: int
    c60_whipsaw_count: int
    sajang_whipsaw_count: int
    c60_delta_vs_leverage_buyhold: float
    sajang_delta_vs_leverage_buyhold: float


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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


def _load_pykrx_daily_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    try:
        from pykrx import stock as pykrx_stock

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        krx_df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)
        return normalize_ohlcv(krx_df).tail(days)
    except Exception as exc:
        logger.warning("pykrx daily load failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def load_daily_ohlcv(ticker: str, days: int = 1260, prefer_remote: bool = False) -> pd.DataFrame:
    if prefer_remote:
        remote_df = _load_pykrx_daily_ohlcv(ticker, days)
        if not remote_df.empty:
            return remote_df

    local_path = RAW_DATA_DIR / f"{ticker}.parquet"
    if local_path.exists():
        try:
            local_df = normalize_ohlcv(pd.read_parquet(local_path))
            if not local_df.empty:
                return local_df.tail(days)
        except Exception as exc:
            logger.warning("local parquet load failed for %s: %s", ticker, exc)

    return _load_pykrx_daily_ohlcv(ticker, days)


def _synthetic_leverage_close(underlying: pd.Series, multiplier: float) -> pd.Series:
    returns = (underlying.pct_change().fillna(0.0) * multiplier).clip(lower=-0.99)
    synthetic = (1.0 + returns).cumprod()
    return synthetic * float(underlying.iloc[0])


def prepare_shadow_prices(
    underlying_prices: pd.DataFrame,
    leverage_prices: pd.DataFrame | None = None,
    multiplier: float = DEFAULT_LEVERAGE_MULTIPLIER,
) -> pd.DataFrame:
    underlying = normalize_ohlcv(underlying_prices)
    if underlying.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=underlying.index)
    out["underlying_close"] = underlying["close"]
    out["underlying_return"] = out["underlying_close"].pct_change().fillna(0.0)

    leverage = normalize_ohlcv(leverage_prices) if leverage_prices is not None else pd.DataFrame()
    if not leverage.empty:
        lev_close = leverage["close"].reindex(out.index)
        if lev_close.notna().sum() >= 2:
            synthetic_all = _synthetic_leverage_close(out["underlying_close"], multiplier)
            first_valid = lev_close.dropna().index[0]
            first_actual = float(lev_close.loc[first_valid])
            if first_actual > 0:
                lev_close = lev_close * (float(synthetic_all.loc[first_valid]) / first_actual)
            lev_close = lev_close.combine_first(synthetic_all)
            lev_close = lev_close.ffill()
            out["leverage_close"] = lev_close
        else:
            out["leverage_close"] = _synthetic_leverage_close(out["underlying_close"], multiplier)
    else:
        out["leverage_close"] = _synthetic_leverage_close(out["underlying_close"], multiplier)

    out = out.dropna(subset=["underlying_close", "leverage_close"])
    out["leverage_return"] = out["leverage_close"].pct_change().fillna(0.0)
    out["ma20"] = out["underlying_close"].rolling(MA20_PERIOD).mean()
    out["ma60"] = out["underlying_close"].rolling(MA60_PERIOD).mean()
    out["ma20_up"] = out["ma20"] > out["ma20"].shift(1)
    out = out.dropna(subset=["ma20", "ma60"])
    return out


def _mdd(current: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return current / peak - 1.0


def _segment_return(values: list[float]) -> float:
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def _segment_mdd(values: list[float]) -> float:
    if not values:
        return 0.0

    base = values[0]
    if base <= 0:
        return 0.0

    peak = 1.0
    worst = 0.0
    for value in values:
        normalized = value / base
        peak = max(peak, normalized)
        worst = min(worst, normalized / peak - 1.0)
    return worst


def _series_metrics(rows: list[object], equity_attr: str) -> dict:
    values = [float(getattr(row, equity_attr)) for row in rows]
    return {
        "return": _round(_segment_return(values)),
        "mdd": _round(_segment_mdd(values)),
    }


def _next_rule_state(
    rule: str,
    state: str,
    close: float,
    ma20: float,
    ma60: float,
    ma20_up: bool,
) -> tuple[str, str]:
    if rule == "c60":
        should_hold = close > ma60
    elif rule == "sajang":
        should_hold = close > ma60 and close > ma20 and ma20_up
    else:
        raise ValueError(f"unknown rule: {rule}")

    if state == "HOLD":
        if should_hold:
            return "HOLD", "HOLD"
        return "EXIT", "CASH"

    if should_hold:
        return "REENTER", "HOLD"
    return "CASH", "CASH"


def build_samsung_single_leverage_shadow_ledger(
    underlying_prices: pd.DataFrame,
    leverage_prices: pd.DataFrame | None = None,
    leverage_ticker: str = DEFAULT_LEVERAGE_TICKER,
    multiplier: float = DEFAULT_LEVERAGE_MULTIPLIER,
    seed_equity: float = DEFAULT_SEED_EQUITY,
) -> list[SamsungLeverageLedgerRow]:
    df = prepare_shadow_prices(underlying_prices, leverage_prices, multiplier)
    if df.empty or len(df) < 2:
        return []

    first_underlying = float(df["underlying_close"].iloc[0])
    first_leverage = float(df["leverage_close"].iloc[0])

    c60_state = "HOLD" if float(df["underlying_close"].iloc[0]) > float(df["ma60"].iloc[0]) else "CASH"
    sajang_state = (
        "HOLD"
        if (
            float(df["underlying_close"].iloc[0]) > float(df["ma60"].iloc[0])
            and float(df["underlying_close"].iloc[0]) > float(df["ma20"].iloc[0])
            and bool(df["ma20_up"].iloc[0])
        )
        else "CASH"
    )

    c60_equity = float(seed_equity)
    sajang_equity = float(seed_equity)
    c60_peak = c60_equity
    sajang_peak = sajang_equity
    lev_peak = float(seed_equity)
    underlying_peak = float(seed_equity)
    c60_cash_days = 0
    sajang_cash_days = 0
    c60_whipsaws = 0
    sajang_whipsaws = 0
    c60_last_exit = None
    sajang_last_exit = None
    rows: list[SamsungLeverageLedgerRow] = []

    for idx, (dt, row) in enumerate(df.iterrows()):
        leverage_return = float(row["leverage_return"])
        if idx > 0:
            if c60_state == "HOLD":
                c60_equity *= 1.0 + leverage_return
            else:
                c60_cash_days += 1

            if sajang_state == "HOLD":
                sajang_equity *= 1.0 + leverage_return
            else:
                sajang_cash_days += 1

        leverage_buyhold = seed_equity * (float(row["leverage_close"]) / first_leverage)
        underlying_buyhold = seed_equity * (float(row["underlying_close"]) / first_underlying)

        c60_peak = max(c60_peak, c60_equity)
        sajang_peak = max(sajang_peak, sajang_equity)
        lev_peak = max(lev_peak, leverage_buyhold)
        underlying_peak = max(underlying_peak, underlying_buyhold)

        c60_signal, c60_next = _next_rule_state(
            "c60",
            c60_state,
            float(row["underlying_close"]),
            float(row["ma20"]),
            float(row["ma60"]),
            bool(row["ma20_up"]),
        )
        sajang_signal, sajang_next = _next_rule_state(
            "sajang",
            sajang_state,
            float(row["underlying_close"]),
            float(row["ma20"]),
            float(row["ma60"]),
            bool(row["ma20_up"]),
        )

        if c60_signal == "EXIT":
            c60_last_exit = float(row["underlying_close"])
        elif c60_signal == "REENTER" and c60_last_exit is not None:
            if float(row["underlying_close"]) > c60_last_exit:
                c60_whipsaws += 1
            c60_last_exit = None

        if sajang_signal == "EXIT":
            sajang_last_exit = float(row["underlying_close"])
        elif sajang_signal == "REENTER" and sajang_last_exit is not None:
            if float(row["underlying_close"]) > sajang_last_exit:
                sajang_whipsaws += 1
            sajang_last_exit = None

        rows.append(
            SamsungLeverageLedgerRow(
                date=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                signal_ticker=UNDERLYING_TICKER,
                leverage_ticker=leverage_ticker,
                underlying_close=_round(row["underlying_close"], 4),
                leverage_close=_round(row["leverage_close"], 4),
                ma20=_round(row["ma20"], 4),
                ma60=_round(row["ma60"], 4),
                c60_signal=c60_signal,
                c60_state=c60_state,
                sajang_signal=sajang_signal,
                sajang_state=sajang_state,
                c60_equity_curve=_round(c60_equity),
                sajang_equity_curve=_round(sajang_equity),
                leverage_buyhold_equity_curve=_round(leverage_buyhold),
                underlying_buyhold_equity_curve=_round(underlying_buyhold),
                drawdown_c60=_round(_mdd(c60_equity, c60_peak)),
                drawdown_sajang=_round(_mdd(sajang_equity, sajang_peak)),
                drawdown_leverage_buyhold=_round(_mdd(leverage_buyhold, lev_peak)),
                drawdown_underlying_buyhold=_round(_mdd(underlying_buyhold, underlying_peak)),
                c60_days_in_cash=c60_cash_days,
                sajang_days_in_cash=sajang_cash_days,
                c60_whipsaw_count=c60_whipsaws,
                sajang_whipsaw_count=sajang_whipsaws,
                c60_delta_vs_leverage_buyhold=_round(c60_equity - leverage_buyhold),
                sajang_delta_vs_leverage_buyhold=_round(sajang_equity - leverage_buyhold),
            )
        )

        c60_state = c60_next
        sajang_state = sajang_next

    return rows


def build_common_period_comparison(
    samsung_rows: Iterable[SamsungLeverageLedgerRow],
    c60_488080_rows: Iterable[object],
) -> dict:
    """Compare Samsung single leverage and 488080 only on shared dates.

    Full-period Samsung history is useful for 2022-style stress validation, but
    it must not be compared directly with 488080 because 488080 has a much
    shorter listed history. This block is the fair allocation comparison.
    """
    samsung_by_date = {row.date: row for row in samsung_rows}
    semi_by_date = {str(getattr(row, "date")): row for row in c60_488080_rows}
    common_dates = sorted(set(samsung_by_date) & set(semi_by_date))
    if len(common_dates) < 2:
        return {
            "status": "NO_COMMON_PERIOD",
            "note": "Samsung and 488080 ledgers do not have enough shared dates for fair comparison.",
        }

    samsung_segment = [samsung_by_date[date] for date in common_dates]
    semi_segment = [semi_by_date[date] for date in common_dates]
    metrics = {
        "samsung_c60": _series_metrics(samsung_segment, "c60_equity_curve"),
        "samsung_sajang": _series_metrics(samsung_segment, "sajang_equity_curve"),
        "samsung_leverage_buyhold": _series_metrics(samsung_segment, "leverage_buyhold_equity_curve"),
        "samsung_underlying_buyhold": _series_metrics(samsung_segment, "underlying_buyhold_equity_curve"),
        "etf_488080_c60": _series_metrics(semi_segment, "c60_equity_curve"),
        "etf_488080_buyhold": _series_metrics(semi_segment, "buyhold_equity_curve"),
    }
    return_winner = max(metrics, key=lambda key: metrics[key]["return"])
    defensive_winner = max(metrics, key=lambda key: metrics[key]["mdd"])
    leveraged_keys = [key for key in metrics if key != "samsung_underlying_buyhold"]
    leveraged_return_winner = max(leveraged_keys, key=lambda key: metrics[key]["return"])
    leveraged_defensive_winner = max(leveraged_keys, key=lambda key: metrics[key]["mdd"])

    return {
        "status": "FAIR_COMMON_PERIOD",
        "period_start": common_dates[0],
        "period_end": common_dates[-1],
        "trading_days": len(common_dates),
        "basis": "All return and MDD values are recomputed from equity curves normalized at the shared start date.",
        "metrics": metrics,
        "winner_by_return": return_winner,
        "winner_by_mdd_defense": defensive_winner,
        "winner_by_return_leveraged_only": leveraged_return_winner,
        "winner_by_mdd_defense_leveraged_only": leveraged_defensive_winner,
        "comparison_warning": "Do not compare full-period Samsung stress metrics directly with 488080. Use this common-period block for allocation decisions.",
        "latest_states": {
            "samsung_c60": samsung_segment[-1].c60_state,
            "samsung_sajang": samsung_segment[-1].sajang_state,
            "etf_488080_c60": getattr(semi_segment[-1], "c60_position_state", None),
        },
    }


def build_samsung_single_leverage_report(
    rows: Iterable[SamsungLeverageLedgerRow],
    c60_488080_reference: dict | None = None,
    common_period_comparison: dict | None = None,
) -> dict:
    ledger = list(rows)
    if not ledger:
        return {
            "status": "NO_DATA",
            "order_count": 0,
            "live_trading_state": "HOLD",
        }

    last = ledger[-1]
    report = {
        "status": "SHADOW_ONLY",
        "signal_ticker": last.signal_ticker,
        "leverage_ticker": last.leverage_ticker,
        "comparison_basis": "full_period_samsung_validation_plus_common_period_488080_fair_compare",
        "ledger_start": ledger[0].date,
        "ledger_end": last.date,
        "ledger_rows": len(ledger),
        "full_period_note": "Samsung full-period metrics are stress-validation numbers, not a direct 488080 allocation comparison.",
        "latest_date": last.date,
        "latest_c60_signal": last.c60_signal,
        "latest_c60_state": last.c60_state,
        "latest_sajang_signal": last.sajang_signal,
        "latest_sajang_state": last.sajang_state,
        "c60_final_return": _round(last.c60_equity_curve - 1.0),
        "sajang_final_return": _round(last.sajang_equity_curve - 1.0),
        "leverage_buyhold_final_return": _round(last.leverage_buyhold_equity_curve - 1.0),
        "underlying_buyhold_final_return": _round(last.underlying_buyhold_equity_curve - 1.0),
        "c60_mdd": min(row.drawdown_c60 for row in ledger),
        "sajang_mdd": min(row.drawdown_sajang for row in ledger),
        "leverage_buyhold_mdd": min(row.drawdown_leverage_buyhold for row in ledger),
        "underlying_buyhold_mdd": min(row.drawdown_underlying_buyhold for row in ledger),
        "c60_days_in_cash": last.c60_days_in_cash,
        "sajang_days_in_cash": last.sajang_days_in_cash,
        "c60_whipsaw_count": last.c60_whipsaw_count,
        "sajang_whipsaw_count": last.sajang_whipsaw_count,
        "winner_by_final_return": max(
            {
                "c60": last.c60_equity_curve,
                "sajang": last.sajang_equity_curve,
                "leverage_buyhold": last.leverage_buyhold_equity_curve,
                "underlying_buyhold": last.underlying_buyhold_equity_curve,
            },
            key={
                "c60": last.c60_equity_curve,
                "sajang": last.sajang_equity_curve,
                "leverage_buyhold": last.leverage_buyhold_equity_curve,
                "underlying_buyhold": last.underlying_buyhold_equity_curve,
            }.get,
        ),
        "one_line_conclusion": "Samsung single leverage should be tracked as shadow only until live liquidity/spread and rule stability are proven.",
        "order_count": 0,
        "live_trading_state": "HOLD",
        "safety_note": "real orders 0 / HOLD maintained / no broker order functions used",
    }
    if c60_488080_reference:
        report["c60_488080_reference_full_available_period"] = c60_488080_reference
    if common_period_comparison:
        report["common_period_fair_comparison"] = common_period_comparison
    return report


def save_samsung_single_leverage_outputs(
    rows: list[SamsungLeverageLedgerRow],
    ledger_path: Path = LEDGER_PATH,
    report_path: Path = REPORT_PATH,
    c60_488080_reference: dict | None = None,
    common_period_comparison: dict | None = None,
) -> tuple[Path, Path, dict]:
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    report = build_samsung_single_leverage_report(
        rows,
        c60_488080_reference=c60_488080_reference,
        common_period_comparison=common_period_comparison,
    )
    ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger_path, report_path, report
