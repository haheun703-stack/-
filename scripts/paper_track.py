"""PAPER 모의 원장 추적/스키마 보강.

data/paper_ledger.json의 PAPER_OPEN 포지션을 processed parquet 기준으로 추적한다.

보강 필드:
  - candidate: 주봉 게이트, 일봉 눌림/지지, 과열, 손익비, 4주체 수급, 진입/회피 사유
  - entry: T0 종가 진입 + T+1 시가 진입 병행 기록
  - tracking: 일별 close/high/low, MFE/MAE, MA10/MA20, exit_check
  - exit: 지지이탈/MA10/MA20/D+10/손절 발생 시 paper 청산 기록

실주문 0 / KIS 미접촉 / scheduler·SAJANG 무변경.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

LEDGER = PROJECT_ROOT / "data" / "paper_ledger.json"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SCHEMA_VERSION = "quant_paper_ledger_v2"


def _date(value: Any) -> str:
    if value in (None, ""):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 2) -> float:
    return round(_float(value), digits)


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator - 1) * 100, 2)


def load_price_df(ticker: str) -> pd.DataFrame | None:
    """processed/{ticker}.parquet 로드. OHLCV 0 행은 제거한다."""
    f = PROCESSED / f"{ticker}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return None
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    if df.empty:
        return None
    return df


def latest_close(ticker: str) -> tuple[float | None, str | None]:
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None, None
    return float(df["close"].iloc[-1]), _date(df.index[-1])


def _row_on_or_before(df: pd.DataFrame, date_str: str) -> tuple[pd.Timestamp, pd.Series] | None:
    hist = df.loc[df.index <= pd.Timestamp(date_str)]
    if hist.empty:
        return None
    return hist.index[-1], hist.iloc[-1]


def _row_after(df: pd.DataFrame, date_str: str) -> tuple[pd.Timestamp, pd.Series] | None:
    future = df.loc[df.index > pd.Timestamp(date_str)]
    if future.empty:
        return None
    return future.index[0], future.iloc[0]


def _ma(row: pd.Series, hist: pd.DataFrame, col: str, window: int) -> float:
    if col in row and _float(row.get(col)) > 0:
        return _float(row.get(col))
    if len(hist) < window:
        return 0.0
    return _float(hist["close"].tail(window).mean())


def build_weekly_gate(df: pd.DataFrame, asof_date: str) -> dict:
    """주봉 20/60선 기반 LONG_ALLOWED/WATCH/AVOID 관찰 게이트."""
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    if hist.empty:
        return {"data_available": False, "gate": "WATCH", "weekly_LONG_ALLOWED": False}

    weekly = hist["close"].resample("W-FRI").last().dropna()
    if len(weekly) < 20:
        return {
            "data_available": False,
            "gate": "WATCH",
            "weekly_LONG_ALLOWED": False,
            "reason": "weekly_data_insufficient",
            "weeks": int(len(weekly)),
        }

    ma20 = weekly.rolling(20).mean()
    ma60 = weekly.rolling(60).mean()
    close = _float(weekly.iloc[-1])
    ma20_now = _float(ma20.iloc[-1])
    ma60_now = _float(ma60.iloc[-1]) if len(weekly) >= 60 else 0.0
    ma20_prev = _float(ma20.iloc[-4]) if len(weekly) >= 24 else ma20_now
    ma60_prev = _float(ma60.iloc[-4]) if len(weekly) >= 64 else ma60_now
    slope20 = _pct(ma20_now, ma20_prev) if ma20_prev else 0.0
    slope60 = _pct(ma60_now, ma60_prev) if ma60_prev else 0.0
    vs20 = _pct(close, ma20_now) if ma20_now else 0.0
    vs60 = _pct(close, ma60_now) if ma60_now else 0.0

    if close >= ma20_now and slope20 >= -1.0 and (not ma60_now or close >= ma60_now * 0.95 or slope60 >= 0):
        gate = "LONG_ALLOWED"
    elif close >= ma20_now * 0.95 or slope20 >= 0:
        gate = "WATCH"
    else:
        gate = "AVOID"

    return {
        "data_available": True,
        "gate": gate,
        "weekly_LONG_ALLOWED": gate == "LONG_ALLOWED",
        "weekly_close": int(close),
        "weekly_close_vs_ma20_pct": round(vs20, 2),
        "weekly_close_vs_ma60_pct": round(vs60, 2),
        "weekly_ma20": round(ma20_now, 2),
        "weekly_ma60": round(ma60_now, 2) if ma60_now else None,
        "weekly_ma20_slope_pct_4w": round(slope20, 2),
        "weekly_ma60_slope_pct_4w": round(slope60, 2) if ma60_now else None,
    }


def build_daily_setup(df: pd.DataFrame, asof_date: str) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    if hist.empty:
        return {"data_available": False, "daily_pullback_support": False}

    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else row
    close = _float(row["close"])
    open_ = _float(row["open"])
    low = _float(row["low"])
    high = _float(row["high"])
    ma5 = _ma(row, hist, "sma_5", 5)
    ma10 = _float(hist["close"].tail(10).mean()) if len(hist) >= 10 else 0.0
    ma20 = _ma(row, hist, "sma_20", 20)
    ma60 = _ma(row, hist, "sma_60", 60)
    rsi = _float(row.get("rsi_14"))
    adx = _float(row.get("adx_14"))
    distance_to_ma20 = _pct(close, ma20) if ma20 else 0.0
    support_test = bool(ma20 and low <= ma20 * 1.03 and close >= ma20 * 0.97)
    bounce_confirmed = bool(support_test and close >= open_ and close >= _float(prev["close"]) * 0.995)
    overheated = bool(distance_to_ma20 >= 12.0 or rsi >= 75.0)

    if bounce_confirmed:
        setup = "BOUNCE_CONFIRMED"
    elif support_test:
        setup = "SUPPORT_TEST"
    elif ma20 and close < ma20 * 1.08:
        setup = "PULLBACK"
    else:
        setup = "NONE"

    return {
        "data_available": True,
        "daily_setup": setup,
        "daily_pullback_support": setup in ("SUPPORT_TEST", "BOUNCE_CONFIRMED"),
        "support_price": int(ma20) if ma20 else int(low),
        "ma5": round(ma5, 2) if ma5 else None,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "adx": round(adx, 2) if adx else None,
        "rsi": round(rsi, 2) if rsi else None,
        "distance_to_ma20_pct": round(distance_to_ma20, 2),
        "overheated": overheated,
        "candle": {
            "open": int(open_),
            "high": int(high),
            "low": int(low),
            "close": int(close),
            "is_bullish": close >= open_,
        },
    }


def build_supply_4_actor(df: pd.DataFrame, asof_date: str, lookback: int = 5) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)].tail(lookback)
    cols = {
        "foreign": "외국인합계",
        "institution": "기관합계",
        "individual": "개인",
        "other_corp": "기타법인",
    }
    if hist.empty or not any(col in hist.columns for col in cols.values()):
        return {
            "data_available": False,
            "lookback_days": lookback,
            "score": 0,
            "hard_gate_used": False,
            "reason": "4_actor_flow_missing",
        }

    sums = {key: int(_float(hist[col].sum())) if col in hist.columns else 0 for key, col in cols.items()}
    score = 0
    score += 2 if sums["foreign"] > 0 else -2 if sums["foreign"] < 0 else 0
    score += 2 if sums["institution"] > 0 else -2 if sums["institution"] < 0 else 0
    score += 1 if sums["other_corp"] > 0 else -1 if sums["other_corp"] < 0 else 0
    if sums["individual"] > 0 and (sums["foreign"] < 0 or sums["institution"] < 0):
        score -= 1

    buyers = [k for k, v in sums.items() if v > 0]
    if sums["foreign"] > 0 and sums["institution"] > 0:
        alignment = "FOREIGN_INST"
    elif sums["institution"] > 0:
        alignment = "INST_ONLY"
    elif sums["foreign"] > 0:
        alignment = "FOREIGN_ONLY"
    elif sums["individual"] > 0:
        alignment = "RETAIL_ABSORB"
    else:
        alignment = "NONE"

    return {
        "data_available": True,
        "lookback_days": int(len(hist)),
        "score": int(score),
        "hard_gate_used": False,
        "alignment": alignment,
        "buyers": buyers,
        "net_shares": sums,
    }


def build_risk_reward(trade: dict, df: pd.DataFrame, asof_date: str) -> dict:
    hist = df.loc[df.index <= pd.Timestamp(asof_date)]
    entry = _float(trade.get("entry_price"))
    stop = _float(trade.get("stop_loss_price"))
    if entry <= 0:
        return {"data_available": False, "rr": 0.0}
    if stop <= 0:
        daily = build_daily_setup(df, asof_date)
        stop = _float(daily.get("support_price")) * 0.97 if daily.get("support_price") else entry * 0.92

    recent_high = _float(hist["high"].tail(60).max()) if not hist.empty else entry
    risk = max(entry - stop, 1.0)
    target = recent_high if recent_high > entry else entry + risk * 2.0
    reward = max(target - entry, 0.0)
    rr = reward / risk if risk > 0 else 0.0
    return {
        "data_available": True,
        "entry_price": int(entry),
        "stop_loss_price": int(stop),
        "target_reference_price": int(target),
        "risk_per_share": int(risk),
        "reward_per_share": int(reward),
        "rr": round(rr, 2),
    }


def build_candidate(trade: dict, df: pd.DataFrame) -> dict:
    asof_date = trade.get("entry_date") or _date(df.index[-1])
    weekly = build_weekly_gate(df, asof_date)
    daily = build_daily_setup(df, asof_date)
    supply = build_supply_4_actor(df, asof_date)
    rr = build_risk_reward(trade, df, asof_date)

    blockers: list[str] = []
    if weekly.get("gate") != "LONG_ALLOWED":
        blockers.append(f"weekly_gate={weekly.get('gate')}")
    if not daily.get("daily_pullback_support"):
        blockers.append(f"daily_setup={daily.get('daily_setup')}")
    if daily.get("overheated"):
        blockers.append("overheated")
    if rr.get("rr", 0) < 1.5:
        blockers.append(f"rr<{1.5}")

    decision = "진입" if not blockers else "회피"
    decision_shadow = "ENTER_CANDIDATE" if not blockers else "AVOID"
    reason = "조건 충족: LONG_ALLOWED + 눌림/지지 + 과열아님 + RR>=1.5" if not blockers else "; ".join(blockers)

    return {
        "ticker": trade.get("ticker", ""),
        "name": trade.get("name", ""),
        "date": asof_date,
        "weekly_LONG_ALLOWED": bool(weekly.get("weekly_LONG_ALLOWED", False)),
        "weekly_gate": weekly,
        "daily_pullback_support": bool(daily.get("daily_pullback_support", False)),
        "daily_setup": daily,
        "overheated": bool(daily.get("overheated", False)),
        "risk_reward": rr,
        "supply_4주체점수": supply.get("score", 0),
        "stock_flow_4_actor": supply,
        "decision": decision,
        "decision_shadow": decision_shadow,
        "reason": reason,
        "hard_gate_notes": "수급은 feature/log only. hard gate 미사용.",
    }


def build_entry(trade: dict, df: pd.DataFrame) -> dict:
    entry_date = trade.get("entry_date") or _date(df.index[-1])
    t0 = _row_on_or_before(df, entry_date)
    t1 = _row_after(df, entry_date)
    t0_date, t0_row = t0 if t0 else (None, None)
    t1_date, t1_row = t1 if t1 else (None, None)
    entry_price = _float(trade.get("entry_price")) or (_float(t0_row["close"]) if t0_row is not None else 0.0)
    stop_price = _float(trade.get("stop_loss_price")) or entry_price * 0.92
    qty = int(_float(trade.get("qty")))
    return {
        "variant_recorded": ["T0_CLOSE", "T1_OPEN"],
        "selected_variant": "T0_CLOSE",
        "entry_date": entry_date,
        "t0_date": _date(t0_date) if t0_date is not None else entry_date,
        "t0_close_price": int(_float(t0_row["close"])) if t0_row is not None else int(entry_price),
        "t1_date": _date(t1_date) if t1_date is not None else None,
        "t1_open_price": int(_float(t1_row["open"])) if t1_row is not None else None,
        "actual_entry_price": int(entry_price),
        "qty": qty,
        "capital_won": int(entry_price * qty),
        "stop_loss": {
            "price": int(stop_price),
            "pct": _round(trade.get("stop_loss_pct", -8.0), 2),
        },
    }


def build_tracking_record(trade: dict, df: pd.DataFrame) -> dict:
    entry_date = trade.get("entry_date") or _date(df.index[-1])
    entry_price = _float(trade.get("entry_price"))
    qty = int(_float(trade.get("qty")))
    period = df.loc[df.index >= pd.Timestamp(entry_date)]
    if period.empty:
        period = df.tail(1)
    row = period.iloc[-1]
    hist = df.loc[df.index <= period.index[-1]]
    close = _float(row["close"])
    high = _float(row["high"])
    low = _float(row["low"])
    mfe_price = _float(period["high"].max())
    mae_price = _float(period["low"].min())
    ma10 = _float(hist["close"].tail(10).mean()) if len(hist) >= 10 else 0.0
    ma20 = _ma(row, hist, "sma_20", 20)
    support_price = ma20 if ma20 else _float(period["low"].tail(5).min())
    pnl_pct = _pct(close, entry_price) if entry_price else 0.0
    pnl_won = int((close - entry_price) * qty) if entry_price else 0
    mfe_pct = _pct(mfe_price, entry_price) if entry_price else 0.0
    mae_pct = _pct(mae_price, entry_price) if entry_price else 0.0
    days_held = max(0, len(period) - 1)
    exit_check = check_exit(trade, close, low, ma10, ma20, support_price, days_held)
    return {
        "date": _date(period.index[-1]),
        "close": int(close),
        "high": int(high),
        "low": int(low),
        "qty": qty,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_won": pnl_won,
        "MFE_pct": round(mfe_pct, 2),
        "MAE_pct": round(mae_pct, 2),
        "MFE_price": int(mfe_price),
        "MAE_price": int(mae_price),
        "days_held": days_held,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "support_price": int(support_price) if support_price else None,
        "exit_check": exit_check,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def check_exit(
    trade: dict,
    close: float,
    low: float,
    ma10: float,
    ma20: float,
    support_price: float,
    days_held: int,
) -> dict:
    stop = _float(trade.get("stop_loss_price"))
    if stop and low <= stop:
        return {"should_exit": True, "reason": "STOP_LOSS", "detail": "손절선 터치"}
    if support_price and close < support_price * 0.97:
        return {"should_exit": True, "reason": "SUPPORT_BREAK", "detail": "지지선 3% 이상 이탈"}
    if ma20 and close < ma20:
        return {"should_exit": True, "reason": "MA20_BREAK", "detail": "MA20 이탈"}
    if ma10 and close < ma10:
        return {"should_exit": True, "reason": "MA10_BREAK", "detail": "MA10 이탈"}
    if days_held >= 10:
        return {"should_exit": True, "reason": "D+10_REVIEW", "detail": "D+10 점검 기준"}
    return {"should_exit": False, "reason": "HOLD", "detail": "보유 관찰"}


def apply_exit_if_needed(trade: dict, record: dict) -> None:
    if trade.get("status") != "PAPER_OPEN":
        return
    exit_check = record.get("exit_check", {})
    if not exit_check.get("should_exit"):
        return
    entry_price = _float(trade.get("entry_price"))
    exit_price = _float(record.get("close"))
    qty = int(_float(trade.get("qty")))
    pnl_won = int((exit_price - entry_price) * qty)
    pnl_pct = _pct(exit_price, entry_price) if entry_price else 0.0
    trade["status"] = "PAPER_CLOSED"
    trade["exit"] = {
        "date": record.get("date"),
        "reason": exit_check.get("reason"),
        "detail": exit_check.get("detail"),
        "price": int(exit_price),
        "qty": qty,
        "pnl_won": pnl_won,
        "pnl_pct": round(pnl_pct, 2),
        "real_order": False,
    }


def upsert_tracking(trade: dict, record: dict) -> None:
    tracking = trade.get("tracking")
    if not isinstance(tracking, list):
        tracking = []
    tracking = [r for r in tracking if r.get("date") != record.get("date")]
    tracking.append(record)
    tracking.sort(key=lambda x: x.get("date", ""))
    trade["tracking"] = tracking


def enrich_trade(trade: dict) -> tuple[dict, list[str]]:
    issues: list[str] = []
    ticker = trade.get("ticker", "")
    df = load_price_df(ticker)
    if df is None:
        issues.append(f"{ticker}: processed parquet 없음")
        trade.setdefault("candidate", {"ticker": ticker, "decision": "회피", "reason": "price_data_missing"})
        return trade, issues

    trade["schema_version"] = SCHEMA_VERSION
    trade["candidate"] = build_candidate(trade, df)
    trade["entry"] = build_entry(trade, df)

    if trade.get("status") == "PAPER_OPEN":
        record = build_tracking_record(trade, df)
        upsert_tracking(trade, record)
        apply_exit_if_needed(trade, record)
    elif trade.get("tracking"):
        # 닫힌 거래도 최신 MFE/MAE 필드가 없는 과거 tracking은 그대로 두되 스키마만 표시한다.
        pass

    return trade, issues


def validate_trade(trade: dict) -> list[str]:
    missing: list[str] = []
    candidate = trade.get("candidate") or {}
    entry = trade.get("entry") or {}
    tracking = trade.get("tracking") or []
    for key in ("ticker", "weekly_LONG_ALLOWED", "daily_pullback_support", "overheated", "risk_reward", "decision", "reason"):
        if key not in candidate:
            missing.append(f"candidate.{key}")
    for key in ("t0_close_price", "t1_open_price", "qty", "stop_loss"):
        if key not in entry:
            missing.append(f"entry.{key}")
    if not tracking:
        missing.append("tracking[]")
    else:
        last = tracking[-1]
        for key in ("close", "MFE_pct", "MAE_pct", "exit_check"):
            if key not in last:
                missing.append(f"tracking[-1].{key}")
    if trade.get("status") == "PAPER_CLOSED" and "exit" not in trade:
        missing.append("exit")
    return missing


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {
            "_note": "PAPER 전용 모의 원장. 실주문 0 / KIS 미접촉.",
            "_created": datetime.now().strftime("%Y-%m-%d"),
            "paper_trades": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_trade(trade: dict) -> None:
    name = trade.get("name", trade.get("ticker", ""))
    ticker = trade.get("ticker", "")
    sector = trade.get("sector", "")
    candidate = trade.get("candidate", {})
    entry = trade.get("entry", {})
    tracking = trade.get("tracking", [])
    latest = tracking[-1] if tracking else {}
    status = trade.get("status", "")
    status_icon = "🔴 청산" if status == "PAPER_CLOSED" else "🟢 보유중" if status == "PAPER_OPEN" else "⚪ 기타"

    print(f"{name}({ticker}) [{sector}] | {status_icon}")
    print(
        f"  후보: {candidate.get('decision', '?')} "
        f"| weekly={candidate.get('weekly_gate', {}).get('gate', '?')} "
        f"| daily={candidate.get('daily_setup', {}).get('daily_setup', '?')} "
        f"| RR={candidate.get('risk_reward', {}).get('rr', 0)} "
        f"| 4주체={candidate.get('supply_4주체점수', 0)}"
    )
    print(f"  사유: {candidate.get('reason', '')}")
    print(
        f"  entry: T0 {entry.get('t0_close_price')}원 / "
        f"T+1 open {entry.get('t1_open_price')}원 / 실제 {entry.get('actual_entry_price')}원 x{entry.get('qty')}주"
    )
    if latest:
        print(
            f"  latest {latest.get('date')}: close {latest.get('close'):,}원 "
            f"| PnL {latest.get('pnl_pct'):+.2f}% ({latest.get('pnl_won'):+,}원) "
            f"| MFE {latest.get('MFE_pct'):+.2f}% / MAE {latest.get('MAE_pct'):+.2f}%"
        )
        ex = latest.get("exit_check", {})
        print(f"  exit_check: {ex.get('reason')} — {ex.get('detail')}")
    if trade.get("exit"):
        ex = trade["exit"]
        print(f"  EXIT: {ex.get('date')} {ex.get('reason')} @{ex.get('price'):,}원 PnL {ex.get('pnl_pct'):+.2f}%")
    print()


def run(path: Path, write: bool, check_only: bool) -> int:
    data = load_ledger(path)
    data["_schema_version"] = SCHEMA_VERSION
    data["_last_tracked_at"] = datetime.now().isoformat(timespec="seconds")
    data["_safety"] = {
        "real_order": False,
        "kis_touch": False,
        "scheduler_changed": False,
        "sajang_changed": False,
    }

    print("=== PAPER 모의 원장 추적 / v2 스키마 점검 (실주문 0) ===\n")
    issues: list[str] = []
    trades = data.get("paper_trades", [])
    for idx, trade in enumerate(trades):
        if not check_only:
            enriched, trade_issues = enrich_trade(trade)
            trades[idx] = enriched
            issues.extend(trade_issues)
            trade = enriched
        missing = validate_trade(trade)
        if missing:
            issues.append(f"{trade.get('ticker', '?')}: " + ", ".join(missing))
        print_trade(trade)

    data["paper_trades"] = trades
    if write and not check_only:
        save_ledger(path, data)
        print(f"[WRITE] ledger 저장 완료: {path}")
    else:
        print("[NO-WRITE] ledger 파일은 변경하지 않음")

    if issues:
        print("\n[LEDGER CHECK] FAIL")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("\n[LEDGER CHECK] PASS — candidate/entry/tracking/exit 스키마 충족")
    print("★ 실주문 0 / KIS 미접촉 / AUTO_TRADING_ENABLED=0 불변.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PAPER ledger v2 추적/점검")
    parser.add_argument("--ledger", default=str(LEDGER), help="ledger JSON 경로")
    parser.add_argument("--no-write", action="store_true", help="미리보기만 수행")
    parser.add_argument("--check-only", action="store_true", help="현재 ledger 스키마만 점검")
    args = parser.parse_args()

    return run(Path(args.ledger), write=not args.no_write, check_only=args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
