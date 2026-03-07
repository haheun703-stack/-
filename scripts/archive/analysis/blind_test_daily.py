"""
블라인드 테스트 일일 트래커

기능:
  1. 시장 데이터 수집 (KOSPI 레짐, EWY, US Signal)
  2. 그룹순환매 z-score 라이브 스캔
  3. 보유 종목 청산 조건 체크
  4. daily_log JSON 생성
  5. trades.csv / equity_curve.csv 업데이트
  6. 텔레그램 일일 리포트

사용법:
  python scripts/blind_test_daily.py                        # 일일 기록
  python scripts/blind_test_daily.py --enter v10 005930 삼성전자 55000 100
  python scripts/blind_test_daily.py --exit 005930 56000 target
  python scripts/blind_test_daily.py --weekly
  python scripts/blind_test_daily.py --no-send              # 텔레그램 미전송
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
BLIND_DIR = BASE_DIR / "blind_test"
LOG_DIR = BLIND_DIR / "daily_log"
PQ_DIR = BASE_DIR / "data" / "processed"
ROT_DIR = BASE_DIR / "data" / "group_rotation"

SAMSUNG_EXCLUDE = {"032830"}  # 삼성생명

# 거래 비용 (backtest_v2.py와 동일)
SLIPPAGE = 0.005   # 0.5%
COMMISSION = 0.00015  # 편도
TAX = 0.0018        # 매도세


# ── 설정 로드 ──

def load_config():
    with open(BLIND_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_positions():
    with open(BLIND_DIR / "positions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_positions(positions):
    positions["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(BLIND_DIR / "positions.json", "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


# ── 시장 데이터 ──

def get_kospi_regime():
    """KOSPI 레짐 판정 (scan_buy_candidates.py와 동일 로직)"""
    kospi_path = BASE_DIR / "data" / "kospi_index.csv"
    if not kospi_path.exists():
        return {"regime": "UNKNOWN", "slots": 0, "close": 0, "ma20": 0, "ma60": 0}

    df = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["rv20_pct"] = df["rv20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    if len(df) < 60:
        return {"regime": "UNKNOWN", "slots": 0, "close": 0, "ma20": 0, "ma60": 0}

    row = df.iloc[-1]
    close = float(row["close"])
    ma20 = float(row["ma20"]) if not pd.isna(row["ma20"]) else 0
    ma60 = float(row["ma60"]) if not pd.isna(row["ma60"]) else 0
    rv_pct = float(row.get("rv20_pct", 0.5)) if not pd.isna(row.get("rv20_pct", 0.5)) else 0.5
    change_pct = float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100) if len(df) >= 2 else 0

    if ma20 == 0 or ma60 == 0:
        regime, slots = "CAUTION", 3
    elif close > ma20:
        regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
    elif close > ma60:
        regime, slots = "BEAR", 2
    else:
        regime, slots = "CRISIS", 0

    return {
        "regime": regime, "slots": slots,
        "close": close, "ma20": ma20, "ma60": ma60,
        "rv_pct": rv_pct, "change_pct": change_pct,
        "date": df.index[-1].strftime("%Y-%m-%d"),
    }


def get_us_signal():
    """US Overnight Signal 로드"""
    signal_path = BASE_DIR / "data" / "us_market" / "overnight_signal.json"
    if not signal_path.exists():
        return {"grade": "N/A", "score": 0, "ewy_change": 0, "vix": 0}
    try:
        with open(signal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "grade": data.get("composite", "N/A"),
            "score": data.get("score", 0),
            "ewy_change": data.get("ewy_change_pct", 0),
            "vix": data.get("vix_close", 0),
        }
    except Exception:
        return {"grade": "N/A", "score": 0, "ewy_change": 0, "vix": 0}


# ── 그룹순환매 라이브 스캔 ──

def load_rotation_data():
    """그룹 멤버 + ETF + 종목 데이터 로드"""
    members_path = ROT_DIR / "members.json"
    if not members_path.exists():
        return None, None, None, None

    with open(members_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    # 삼성생명 제외
    for gname, gdata in groups.items():
        gdata["members"] = [m for m in gdata["members"]
                            if m["ticker"] not in SAMSUNG_EXCLUDE]

    # ETF
    etfs = {}
    for gname, gdata in groups.items():
        etf_path = ROT_DIR / gdata["etf_file"]
        if etf_path.exists():
            etfs[gname] = pd.read_csv(etf_path, index_col="Date", parse_dates=True).sort_index()

    # EWY
    ewy_path = ROT_DIR / "etf_ewy.csv"
    ewy = pd.read_csv(ewy_path, index_col="Date", parse_dates=True).sort_index() if ewy_path.exists() else None

    # 종목
    stocks = {}
    for gname, gdata in groups.items():
        for m in gdata["members"]:
            pq = PQ_DIR / f"{m['ticker']}.parquet"
            if pq.exists():
                stocks[m["ticker"]] = pd.read_parquet(pq)

    return groups, etfs, ewy, stocks


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def scan_rotation():
    """그룹순환매 z-score 라이브 스캔

    반환: {
      "ewy_above_ma20": bool,
      "groups": {group_name: {"etf_above_ma20": bool, "ret_20": float}},
      "candidates": [{ticker, name, group, z_20, z_5, rel_ret_20, close, entry_ok, reason}],
    }
    """
    groups, etfs, ewy, stocks = load_rotation_data()
    if not groups or not etfs:
        return {"error": "데이터 없음", "candidates": []}

    result = {"ewy_above_ma20": None, "groups": {}, "candidates": []}

    # EWY 상태
    if ewy is not None and len(ewy) > 20:
        ewy["ma20"] = ewy["close"].rolling(20).mean()
        ewy["ret_20"] = ewy["close"].pct_change(20) * 100
        ewy_close = float(ewy["close"].iloc[-1])
        ewy_ma20 = float(ewy["ma20"].iloc[-1])
        result["ewy_above_ma20"] = ewy_close > ewy_ma20
        result["ewy_close"] = ewy_close
        result["ewy_ma20"] = ewy_ma20

    # 그룹별 처리
    for gname, gdata in groups.items():
        etf = etfs.get(gname)
        if etf is None or len(etf) < 25:
            continue

        etf["ret_20"] = etf["close"].pct_change(20) * 100
        etf["ret_5"] = etf["close"].pct_change(5) * 100
        etf["ma20"] = etf["close"].rolling(20).mean()

        etf_close = float(etf["close"].iloc[-1])
        etf_ma20 = float(etf["ma20"].iloc[-1])
        etf_ret_20 = float(etf["ret_20"].iloc[-1]) if not pd.isna(etf["ret_20"].iloc[-1]) else 0
        etf_ret_5 = float(etf["ret_5"].iloc[-1]) if not pd.isna(etf["ret_5"].iloc[-1]) else 0
        etf_above = etf_close > etf_ma20

        result["groups"][gname] = {
            "etf_above_ma20": etf_above,
            "etf_close": etf_close,
            "etf_ma20": etf_ma20,
            "ret_20": etf_ret_20,
        }

        # 종목 상대 수익률 수집
        members = gdata["members"]
        raw = {}
        for m in members:
            ticker = m["ticker"]
            if ticker not in stocks:
                continue
            df = stocks[ticker]
            if len(df) < 25:
                continue

            close = float(df["close"].iloc[-1])
            ret_20 = float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100) if len(df) > 21 else 0
            ret_5 = float((df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0

            rsi_series = calc_rsi(df["close"])
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2]) else 50

            ma60 = float(df["close"].rolling(60).mean().iloc[-1]) if len(df) > 60 else 0
            high_60 = float(df["close"].rolling(60).max().iloc[-1]) if len(df) > 60 else 0

            raw[ticker] = {
                "name": m["name"],
                "close": close,
                "ret_20": ret_20,
                "ret_5": ret_5,
                "rel_ret_20": ret_20 - etf_ret_20,
                "rel_ret_5": ret_5 - etf_ret_5,
                "rsi": rsi,
                "rsi_prev": rsi_prev,
                "ma60": ma60,
                "high_60": high_60,
            }

        if len(raw) < 2:
            continue

        # Z-score
        rel_20_vals = [v["rel_ret_20"] for v in raw.values()]
        rel_5_vals = [v["rel_ret_5"] for v in raw.values()]
        mean_20, std_20 = np.mean(rel_20_vals), max(np.std(rel_20_vals), 0.01)
        mean_5, std_5 = np.mean(rel_5_vals), max(np.std(rel_5_vals), 0.01)

        for ticker, data in raw.items():
            z_20 = (data["rel_ret_20"] - mean_20) / std_20
            z_5 = (data["rel_ret_5"] - mean_5) / std_5

            # 진입 조건 체크
            entry_ok = True
            reason = "OK"

            if not etf_above:
                entry_ok, reason = False, "ETF<MA20"
            elif z_20 >= -0.8:
                entry_ok, reason = False, f"z20={z_20:.2f}"
            elif data["rel_ret_20"] >= -3.0:
                entry_ok, reason = False, f"rel20={data['rel_ret_20']:.1f}%"
            else:
                # 반전 체크
                reversal = z_5 > z_20 or (data["rsi_prev"] < 40 and data["rsi"] > data["rsi_prev"])
                if not reversal:
                    entry_ok, reason = False, "no_reversal"
                elif data["ma60"] > 0 and data["close"] < data["ma60"]:
                    entry_ok, reason = False, "close<MA60"
                elif data["high_60"] > 0 and (data["close"] / data["high_60"] - 1) < -0.25:
                    entry_ok, reason = False, "drop>25%"

            result["candidates"].append({
                "ticker": ticker,
                "name": data["name"],
                "group": gname,
                "z_20": round(z_20, 3),
                "z_5": round(z_5, 3),
                "rel_ret_20": round(data["rel_ret_20"], 2),
                "close": data["close"],
                "rsi": round(data["rsi"], 1),
                "entry_ok": entry_ok,
                "reason": reason,
            })

    # z_20 기준 정렬
    result["candidates"].sort(key=lambda x: x["z_20"])

    return result


# ── 포지션 관리 ──

def check_position_exits(positions, config):
    """보유 종목 청산 조건 체크 (종가 기준)"""
    alerts = []
    today = date.today()

    for strategy in ["v10", "rotation"]:
        rules = config.get(f"{strategy.replace('rotation', 'rotation')}_rules",
                           config.get("rotation_rules", {}))
        if strategy == "v10":
            rules = config.get("v10_rules", {})

        for pos in positions.get(strategy, []):
            ticker = pos["ticker"]
            pq = PQ_DIR / f"{ticker}.parquet"
            if not pq.exists():
                continue

            df = pd.read_parquet(pq)
            if len(df) < 1:
                continue

            current_price = float(df["close"].iloc[-1])
            entry_price = pos["entry_price"]
            gross_pnl = current_price / entry_price - 1
            cost_pct = SLIPPAGE * 2 + COMMISSION * 2 + TAX
            pnl_pct = (gross_pnl - cost_pct) * 100

            entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
            hold_days = (today - entry_date).days

            pos["current_price"] = current_price
            pos["unrealized_pnl_pct"] = round(pnl_pct, 2)
            pos["hold_days"] = hold_days

            # 손절 체크
            stop_pct = rules.get("stop_loss_pct", -7.0)
            if pnl_pct <= stop_pct:
                alerts.append({
                    "strategy": strategy,
                    "ticker": ticker,
                    "name": pos["name"],
                    "signal": "STOP_LOSS",
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                })

            # v10: 목표가 체크
            if strategy == "v10":
                target_pct = rules.get("target_pct", 10.0)
                if pnl_pct >= target_pct:
                    alerts.append({
                        "strategy": strategy,
                        "ticker": ticker,
                        "name": pos["name"],
                        "signal": "TARGET",
                        "pnl_pct": round(pnl_pct, 2),
                        "hold_days": hold_days,
                    })

            # 시간 정지
            max_days = rules.get("max_hold_days", 20)
            time_stop = rules.get("time_stop_days", 15)
            if hold_days >= max_days:
                alerts.append({
                    "strategy": strategy,
                    "ticker": ticker,
                    "name": pos["name"],
                    "signal": "MAX_TIME",
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                })
            elif hold_days >= time_stop and pnl_pct < 3.0:
                alerts.append({
                    "strategy": strategy,
                    "ticker": ticker,
                    "name": pos["name"],
                    "signal": "TIME_STOP",
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                })

    return alerts


def enter_position(strategy, ticker, name, price, shares, grade="", z20=0, group=""):
    """포지션 진입 기록"""
    positions = load_positions()
    config = load_config()

    pos = {
        "ticker": ticker,
        "name": name,
        "entry_date": date.today().strftime("%Y-%m-%d"),
        "entry_price": price,
        "shares": shares,
        "allocated": round(price * shares, 0),
        "current_price": price,
        "unrealized_pnl_pct": 0,
    }

    if strategy == "v10":
        pos["grade"] = grade
        pos["stop_loss"] = round(price * (1 + config["v10_rules"]["stop_loss_pct"] / 100), 0)
        pos["target"] = round(price * (1 + config["v10_rules"]["target_pct"] / 100), 0)
    else:
        pos["group"] = group
        pos["z20_entry"] = z20

    positions[strategy].append(pos)
    save_positions(positions)
    print(f"  진입 기록: [{strategy}] {name}({ticker}) @ {price:,}원 × {shares}주")


def exit_position(ticker, exit_price, reason):
    """포지션 청산 기록 → trades.csv 추가"""
    positions = load_positions()
    today_str = date.today().strftime("%Y-%m-%d")

    for strategy in ["v10", "rotation"]:
        for pos in positions[strategy]:
            if pos["ticker"] == ticker:
                gross_pnl_pct = exit_price / pos["entry_price"] - 1
                cost_pct = SLIPPAGE * 2 + COMMISSION * 2 + TAX  # 왕복 비용
                pnl_pct = (gross_pnl_pct - cost_pct) * 100
                net_exit = exit_price * (1 - SLIPPAGE - COMMISSION - TAX)
                net_entry = pos["entry_price"] * (1 + SLIPPAGE + COMMISSION)
                pnl_krw = round((net_exit - net_entry) * pos["shares"])

                # trades.csv에 추가
                row = {
                    "date": today_str,
                    "strategy": strategy,
                    "ticker": ticker,
                    "name": pos["name"],
                    "direction": "LONG",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": today_str,
                    "hold_days": (date.today() - datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()).days,
                    "reason": reason,
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_krw": pnl_krw,
                    "grade": pos.get("grade", ""),
                    "z20_at_entry": pos.get("z20_entry", ""),
                    "group": pos.get("group", ""),
                }

                trades_path = BLIND_DIR / "trades.csv"
                with open(trades_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writerow(row)

                positions[strategy].remove(pos)
                save_positions(positions)
                print(f"  청산 기록: [{strategy}] {pos['name']}({ticker}) "
                      f"@ {exit_price:,}원 ({pnl_pct:+.1f}%, {reason})")
                return

    print(f"  경고: {ticker} 보유 포지션 없음")


# ── 일일 로그 ──

def generate_daily_log(kospi, us_signal, rotation_scan, exit_alerts, positions, config):
    """일별 로그 JSON 생성"""
    today_str = date.today().strftime("%Y-%m-%d")

    # 포지션 평가
    v10_positions = positions.get("v10", [])
    rot_positions = positions.get("rotation", [])

    v10_equity = config["initial_capital"] * config["allocation"]["v10_swing"]
    rot_equity = config["initial_capital"] * config["allocation"]["group_rotation"]

    # 미실현 손익
    v10_unrealized = sum(
        p.get("unrealized_pnl_pct", 0) * p.get("allocated", 0) / 100
        for p in v10_positions
    )
    rot_unrealized = sum(
        p.get("unrealized_pnl_pct", 0) * p.get("allocated", 0) / 100
        for p in rot_positions
    )

    # 실현 손익 (trades.csv에서)
    trades_path = BLIND_DIR / "trades.csv"
    realized_pnl = 0
    if trades_path.exists():
        try:
            df = pd.read_csv(trades_path)
            if len(df) > 0:
                realized_pnl = df["pnl_krw"].sum()
        except Exception:
            pass

    equity = config["initial_capital"] + realized_pnl + v10_unrealized + rot_unrealized

    # 후보 분류
    rot_recommended = [c for c in rotation_scan.get("candidates", []) if c.get("entry_ok")]

    log = {
        "date": today_str,
        "day_number": None,  # 나중에 계산
        "market": {
            "kospi_close": kospi.get("close"),
            "kospi_change_pct": kospi.get("change_pct"),
            "kospi_regime": kospi.get("regime"),
            "kospi_slots": kospi.get("slots"),
            "ewy_close": rotation_scan.get("ewy_close"),
            "ewy_above_ma20": rotation_scan.get("ewy_above_ma20"),
            "us_signal_grade": us_signal.get("grade"),
            "us_signal_score": us_signal.get("score"),
            "vix": us_signal.get("vix"),
        },
        "v10_scan": {
            "recommended": [],  # 사용자가 scan_buy_candidates.py 결과 확인 후 기록
            "action_taken": [],
        },
        "rotation_scan": {
            "ewy_above_ma20": rotation_scan.get("ewy_above_ma20"),
            "groups": rotation_scan.get("groups", {}),
            "all_stocks": rotation_scan.get("candidates", []),
            "recommended": rot_recommended,
            "action_taken": [],
        },
        "positions": {
            "v10": v10_positions,
            "rotation": rot_positions,
        },
        "exit_alerts": exit_alerts,
        "equity": round(equity),
        "realized_pnl": round(realized_pnl),
        "unrealized_pnl": round(v10_unrealized + rot_unrealized),
    }

    # 일차 계산
    start = datetime.strptime(config["start_date"], "%Y-%m-%d").date()
    log["day_number"] = (date.today() - start).days + 1

    # 저장
    log_path = LOG_DIR / f"{today_str}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)

    # equity_curve.csv 업데이트
    eq_path = BLIND_DIR / "equity_curve.csv"
    cum_ret = (equity / config["initial_capital"] - 1) * 100

    # MDD 계산
    mdd = 0
    if eq_path.exists():
        try:
            eq_df = pd.read_csv(eq_path)
            if len(eq_df) > 0:
                all_eq = list(eq_df["equity"]) + [equity]
                peak = np.maximum.accumulate(all_eq)
                dd = (np.array(all_eq) - peak) / peak * 100
                mdd = round(float(np.min(dd)), 2)
        except Exception:
            pass

    # 일별 수익률
    daily_ret = 0
    if eq_path.exists():
        try:
            eq_df = pd.read_csv(eq_path)
            if len(eq_df) > 0:
                prev_eq = eq_df["equity"].iloc[-1]
                daily_ret = round((equity / prev_eq - 1) * 100, 2)
        except Exception:
            pass

    with open(eq_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            today_str, round(equity), "", "",
            round(equity - v10_unrealized - rot_unrealized),
            kospi.get("close", ""),
            daily_ret, round(cum_ret, 2), mdd,
        ])

    return log


# ── 텔레그램 리포트 ──

def send_telegram_report(log, rotation_scan, no_send=False):
    """텔레그램 일일 리포트"""
    config = load_config()
    day = log.get("day_number", "?")
    mkt = log["market"]
    equity = log["equity"]
    init_cap = config["initial_capital"]
    cum_ret = (equity / init_cap - 1) * 100

    # 레짐 이모지
    regime_emoji = {
        "BULL": "🟢", "CAUTION": "🟡", "BEAR": "🔴", "CRISIS": "🚨"
    }.get(mkt.get("kospi_regime", ""), "⬜")

    lines = []
    lines.append("━━━━━━━━━━━━━━━━━")
    lines.append(f"📊 블라인드 테스트 Day {day}")
    lines.append("━━━━━━━━━━━━━━━━━")

    # 시장
    kospi_close = mkt.get("kospi_close", 0)
    kospi_chg = mkt.get("kospi_change_pct", 0)
    lines.append(f"📈 KOSPI: {kospi_close:,.0f} ({kospi_chg:+.1f}%)")
    lines.append(f"🌐 레짐: {mkt.get('kospi_regime', '?')} {regime_emoji} ({mkt.get('kospi_slots', '?')}슬롯)")

    us_grade = mkt.get("us_signal_grade", "N/A")
    us_score = mkt.get("us_signal_score", 0)
    lines.append(f"🇺🇸 US Signal: {us_grade} ({us_score})")
    lines.append("")

    # v10 스윙
    lines.append("[v10.3 스윙]")
    v10_recs = log["v10_scan"].get("recommended", [])
    if v10_recs:
        for r in v10_recs:
            lines.append(f"  추천: {r}")
    else:
        lines.append("  추천: 없음 (스캔 결과 확인)")
    v10_pos = log["positions"]["v10"]
    v10_pnl = sum(p.get("unrealized_pnl_pct", 0) for p in v10_pos)
    lines.append(f"  보유: {len(v10_pos)}종목 | PnL: {v10_pnl:+.1f}%")
    lines.append("")

    # 그룹 순환매
    lines.append("[그룹 순환매]")
    ewy_ok = rotation_scan.get("ewy_above_ma20")
    lines.append(f"  EWY > MA20: {'O' if ewy_ok else 'X' if ewy_ok is not None else '?'}")
    for gname, ginfo in rotation_scan.get("groups", {}).items():
        lines.append(f"  {gname}: ETF{'>' if ginfo.get('etf_above_ma20') else '<'}MA20, ret20={ginfo.get('ret_20', 0):.1f}%")

    rot_recs = [c for c in rotation_scan.get("candidates", []) if c.get("entry_ok")]
    if rot_recs:
        for r in rot_recs[:3]:
            lines.append(f"  추천: {r['name']}({r['group']}) z20={r['z_20']:.2f}")
    else:
        lines.append("  추천: 없음")
    rot_pos = log["positions"]["rotation"]
    rot_pnl = sum(p.get("unrealized_pnl_pct", 0) for p in rot_pos)
    lines.append(f"  보유: {len(rot_pos)}종목 | PnL: {rot_pnl:+.1f}%")
    lines.append("")

    # 청산 경고
    alerts = log.get("exit_alerts", [])
    if alerts:
        lines.append("[청산 경고]")
        for a in alerts:
            lines.append(f"  {a['signal']}: {a['name']}({a['ticker']}) {a['pnl_pct']:+.1f}% ({a['hold_days']}일)")
        lines.append("")

    # 총자산
    lines.append(f"💰 총 자산: {equity/1e4:,.0f}만원 ({cum_ret:+.1f}%)")

    # MDD
    eq_path = BLIND_DIR / "equity_curve.csv"
    mdd = 0
    if eq_path.exists():
        try:
            eq_df = pd.read_csv(eq_path)
            if len(eq_df) > 0 and "drawdown" in eq_df.columns:
                mdd = eq_df["drawdown"].min()
        except Exception:
            pass
    lines.append(f"📉 MDD: {mdd:.1f}%")
    lines.append("━━━━━━━━━━━━━━━━━")

    text = "\n".join(lines)
    print(text)

    if not no_send:
        try:
            from src.telegram_sender import send_message
            send_message(text)
            print("\n  텔레그램 전송 완료")
        except Exception as e:
            print(f"\n  텔레그램 전송 실패: {e}")

    return text


# ── 주간 리포트 ──

def generate_weekly_report():
    """주간 리포트 생성"""
    config = load_config()
    start = datetime.strptime(config["start_date"], "%Y-%m-%d").date()
    today = date.today()
    week_num = ((today - start).days // 7) + 1

    # trades.csv 로드
    trades_path = BLIND_DIR / "trades.csv"
    trades_df = pd.DataFrame()
    if trades_path.exists():
        try:
            trades_df = pd.read_csv(trades_path)
        except Exception:
            pass

    # equity_curve.csv 로드
    eq_path = BLIND_DIR / "equity_curve.csv"
    eq_df = pd.DataFrame()
    if eq_path.exists():
        try:
            eq_df = pd.read_csv(eq_path)
        except Exception:
            pass

    # 이번 주 거래
    week_start = today - pd.Timedelta(days=today.weekday())
    week_trades = pd.DataFrame()
    if len(trades_df) > 0:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        week_trades = trades_df[trades_df["date"] >= str(week_start)]

    # 리포트 생성
    lines = []
    lines.append(f"# 블라인드 테스트 Week {week_num}")
    lines.append(f"기간: {week_start.strftime('%m/%d')} ~ {today.strftime('%m/%d')}")
    lines.append("")

    lines.append("## 성과 요약")
    lines.append("| 지표 | 이번주 | 누적 |")
    lines.append("|------|--------|------|")

    if len(eq_df) > 0:
        cum_ret = eq_df["cumulative_return"].iloc[-1]
        lines.append(f"| 수익률 | - | {cum_ret:+.1f}% |")
    else:
        lines.append("| 수익률 | - | 0% |")

    total_trades = len(trades_df) if len(trades_df) > 0 else 0
    week_trade_count = len(week_trades) if len(week_trades) > 0 else 0
    lines.append(f"| 거래 수 | {week_trade_count} | {total_trades} |")

    if total_trades > 0:
        wins = (trades_df["pnl_pct"] > 0).sum()
        wr = wins / total_trades * 100
        lines.append(f"| 승률 | - | {wr:.0f}% |")
    else:
        lines.append("| 승률 | - | - |")

    if len(eq_df) > 0 and "drawdown" in eq_df.columns:
        mdd = eq_df["drawdown"].min()
        lines.append(f"| MDD | - | {mdd:.1f}% |")

    lines.append("")

    if len(week_trades) > 0:
        lines.append("## 이번 주 거래")
        for _, t in week_trades.iterrows():
            lines.append(f"- {t['name']}({t['ticker']}): {t['pnl_pct']:+.1f}% ({t['reason']})")
        lines.append("")

    lines.append("## 교훈/메모")
    lines.append("(수동 기록)")

    report_text = "\n".join(lines)
    report_path = BLIND_DIR / "weekly_report" / f"week_{week_num:02d}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  주간 리포트 생성: {report_path}")
    print(report_text)


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="블라인드 테스트 일일 트래커")
    parser.add_argument("--enter", nargs=5,
                        metavar=("STRATEGY", "TICKER", "NAME", "PRICE", "SHARES"),
                        help="포지션 진입 (v10/rotation ticker name price shares)")
    parser.add_argument("--exit", nargs=3,
                        metavar=("TICKER", "PRICE", "REASON"),
                        help="포지션 청산 (ticker price reason)")
    parser.add_argument("--weekly", action="store_true", help="주간 리포트 생성")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미전송")
    parser.add_argument("--z20", type=float, default=0, help="순환매 진입 z20 값")
    parser.add_argument("--group", default="", help="순환매 그룹명")
    parser.add_argument("--grade", default="", help="v10 등급")
    args = parser.parse_args()

    # 포지션 진입
    if args.enter:
        strategy, ticker, name, price, shares = args.enter
        enter_position(strategy, ticker, name, float(price), int(shares),
                       grade=args.grade, z20=args.z20, group=args.group)
        return

    # 포지션 청산
    if args.exit:
        ticker, price, reason = args.exit
        exit_position(ticker, float(price), reason)
        return

    # 주간 리포트
    if args.weekly:
        generate_weekly_report()
        return

    # ── 일일 실행 ──
    print("=" * 50)
    print(f"  블라인드 테스트 — {date.today()}")
    print("=" * 50)

    config = load_config()
    positions = load_positions()

    # 1. 시장 데이터
    print("\n[1] 시장 데이터 수집...")
    kospi = get_kospi_regime()
    us_signal = get_us_signal()
    print(f"  KOSPI: {kospi.get('close', 0):,.0f} (레짐: {kospi.get('regime', '?')}, {kospi.get('slots', '?')}슬롯)")
    print(f"  US Signal: {us_signal.get('grade', 'N/A')} ({us_signal.get('score', 0)})")

    # 2. 그룹순환매 스캔
    print("\n[2] 그룹순환매 z-score 스캔...")
    rotation_scan = scan_rotation()

    ewy_ok = rotation_scan.get("ewy_above_ma20")
    print(f"  EWY > MA20: {'O' if ewy_ok else 'X' if ewy_ok is not None else 'N/A'}")

    for gname, ginfo in rotation_scan.get("groups", {}).items():
        above = ">" if ginfo.get("etf_above_ma20") else "<"
        print(f"  {gname}: ETF{above}MA20, 20일수익 {ginfo.get('ret_20', 0):+.1f}%")

    print(f"\n  전체 종목 z-score:")
    for c in rotation_scan.get("candidates", []):
        mark = " [진입가능]" if c["entry_ok"] else ""
        print(f"    {c['name']:>8}({c['group']:<8}): z20={c['z_20']:+.3f} "
              f"rel20={c['rel_ret_20']:+.1f}% RSI={c['rsi']:.0f} "
              f"{c['reason']}{mark}")

    rec_count = sum(1 for c in rotation_scan.get("candidates", []) if c["entry_ok"])
    print(f"\n  진입 후보: {rec_count}종목")

    # 3. 보유 종목 체크
    print("\n[3] 보유 종목 청산 체크...")
    exit_alerts = check_position_exits(positions, config)
    v10_count = len(positions.get("v10", []))
    rot_count = len(positions.get("rotation", []))
    print(f"  v10: {v10_count}종목 | 순환매: {rot_count}종목")

    if exit_alerts:
        for a in exit_alerts:
            print(f"  ⚠ {a['signal']}: {a['name']}({a['ticker']}) {a['pnl_pct']:+.1f}% ({a['hold_days']}일)")
    else:
        print("  청산 경고 없음")

    # 4. 일일 로그 생성
    print("\n[4] 일일 로그 생성...")
    log = generate_daily_log(kospi, us_signal, rotation_scan, exit_alerts, positions, config)
    print(f"  저장: blind_test/daily_log/{date.today()}.json")

    # 5. 텔레그램 리포트
    print("\n[5] 텔레그램 리포트...")
    send_telegram_report(log, rotation_scan, no_send=args.no_send)

    # 6. v10.3 스캔 안내
    print(f"\n{'=' * 50}")
    print("  v10.3 스캔은 별도 실행:")
    print("  python scripts/scan_buy_candidates.py --grade AB --no-send")
    print(f"{'=' * 50}")

    save_positions(positions)


if __name__ == "__main__":
    main()
