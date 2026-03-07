"""
소형주 급등 패턴 Forward Return 백테스트

과거 N 거래일 동안 패턴 발생 시점을 찾고,
발생 후 5일/10일/20일 수익률을 측정하여 패턴별 실전 유효성 검증.

Usage:
    python scripts/backtest_smallcap_forward.py
    python scripts/backtest_smallcap_forward.py --days 120
    python scripts/backtest_smallcap_forward.py --days 60 --top 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "smallcap_forward_backtest.json"


def load_settings() -> dict:
    defaults = {
        "min_trading_value": 5e8,
        "min_price": 1000,
        "vol_breakout": {"min_vol_ratio": 2.5},
        "shakeout": {"min_drop_pct": -5, "min_bounce_pct": 3, "min_vol_ratio": 1.0},
        "consecutive": {"min_streak": 3, "min_cum_return": 8},
    }
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("smallcap_explosion", defaults)
    except Exception:
        return defaults


# ── 패턴 감지 함수 (scan_smallcap_explosion.py에서 복사) ──

def detect_volume_breakout(closes, highs, volumes, rsis, bb_upper, idx, cfg):
    min_vr = cfg.get("vol_breakout", {}).get("min_vol_ratio", 2.5)
    vol = volumes[idx]
    vol_ma20 = np.mean(volumes[max(0, idx - 19):idx])
    if vol_ma20 <= 0:
        return None
    vol_ratio = vol / vol_ma20
    if vol_ratio < min_vr:
        return None

    close = closes[idx]
    high_20 = np.max(highs[max(0, idx - 20):idx]) if idx > 0 else close
    high_60 = np.max(highs[max(0, idx - 60):idx]) if idx > 0 else close

    bb = bb_upper[idx] if bb_upper is not None and not np.isnan(bb_upper[idx]) else high_20

    breakout_type = None
    if close > high_60:
        breakout_type = "60d"
    elif close > high_20:
        breakout_type = "20d"
    elif bb > 0 and close > bb:
        breakout_type = "BB"
    else:
        return None

    score = 50 + min(int(vol_ratio * 5), 25)
    if breakout_type == "60d":
        score += 15
    elif breakout_type == "20d":
        score += 10
    else:
        score += 5

    rsi = rsis[idx] if rsis is not None and not np.isnan(rsis[idx]) else 50
    if rsi < 75:
        score += 5

    return {"pattern": "A", "score": min(score, 100), "vol_ratio": round(vol_ratio, 1)}


def detect_shakeout_reversal(closes, volumes, idx, cfg):
    if idx < 5:
        return None
    sh_cfg = cfg.get("shakeout", {})
    min_drop = sh_cfg.get("min_drop_pct", -5)
    min_bounce = sh_cfg.get("min_bounce_pct", 3)
    min_vr = sh_cfg.get("min_vol_ratio", 1.0)

    today_chg = (closes[idx] / closes[idx - 1] - 1) * 100 if closes[idx - 1] > 0 else 0
    yest_chg = (closes[idx - 1] / closes[idx - 2] - 1) * 100 if closes[idx - 2] > 0 else 0

    shakeout = None
    if yest_chg <= min_drop and today_chg >= min_bounce:
        shakeout = {"drop_pct": yest_chg, "bounce_pct": today_chg}

    if not shakeout and idx >= 3 and closes[idx - 3] > 0:
        day2_chg = (closes[idx - 2] / closes[idx - 3] - 1) * 100
        bounce_2d = (closes[idx] / closes[idx - 2] - 1) * 100
        if day2_chg <= min_drop and bounce_2d >= min_bounce * 1.5:
            shakeout = {"drop_pct": day2_chg, "bounce_pct": bounce_2d}

    if not shakeout:
        return None

    close_5d_ago = closes[max(0, idx - 5)]
    trend_5d = (closes[idx] / close_5d_ago - 1) * 100 if close_5d_ago > 0 else 0
    if trend_5d < 0:
        return None

    vol = volumes[idx]
    vol_ma20 = np.mean(volumes[max(0, idx - 19):idx])
    vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio < min_vr:
        return None

    score = 55 + min(int(abs(shakeout["drop_pct"]) * 2), 15)
    score += min(int(shakeout["bounce_pct"] * 2), 15)
    score += min(int(vol_ratio * 5), 10)
    if trend_5d >= 10:
        score += 5
    return {"pattern": "B", "score": min(score, 100), "vol_ratio": round(vol_ratio, 1)}


def detect_consecutive_momentum(closes, volumes, idx, cfg):
    if idx < 5:
        return None
    con_cfg = cfg.get("consecutive", {})
    min_streak = con_cfg.get("min_streak", 3)
    min_cum = con_cfg.get("min_cum_return", 8)

    streak = 0
    for i in range(idx, max(idx - 10, 0), -1):
        if closes[i] > closes[i - 1]:
            streak += 1
        else:
            break
    if streak < min_streak:
        return None

    base = closes[idx - streak]
    if base <= 0:
        return None
    cum_ret = (closes[idx] / base - 1) * 100
    if cum_ret < min_cum:
        return None

    vol_recent = np.mean(volumes[max(0, idx - 2):idx + 1])
    vol_ma20 = np.mean(volumes[max(0, idx - 19):idx + 1])
    vol_accel = vol_recent / vol_ma20 if vol_ma20 > 0 else 1
    if vol_accel < 0.8:
        return None

    score = 50 + min(streak * 5, 20) + min(int(cum_ret), 20) + min(int(vol_accel * 5), 10)
    return {"pattern": "C", "score": min(score, 100), "streak": streak,
            "cum_return": round(cum_ret, 1)}


def run_paper_sim(args):
    """과거 PRIMARY 시그널에 ATR×1.0 SL / ATR×3.0 TP 적용 시뮬레이션.

    2주 기다릴 필요 없이 과거 데이터로 즉시 검증.
    """
    cfg = load_settings()
    min_tv = float(cfg.get("min_trading_value", 5e8))
    min_price = float(cfg.get("min_price", 1000))
    hold_days = args.hold_days
    sl_mult = args.sl_mult
    tp_mult = args.tp_mult

    print(f"[페이퍼 트레이딩 시뮬레이션]")
    print(f"  스캔 범위: 과거 {args.days} 거래일")
    print(f"  보유기간: {hold_days}일, SL: ATR×{sl_mult}, TP: ATR×{tp_mult}")
    print(f"  대상: PRIMARY (C, A+C) 시그널만")
    print("=" * 75)

    # ── 1단계: CSV 로딩 ──
    t0 = time.time()
    name_map = {}
    stock_data = {}

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    print(f"  CSV 로딩: {len(csv_files)}개...")

    for csv_path in csv_files:
        parts = csv_path.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts
        name_map[ticker] = name
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
        except Exception:
            continue
        if len(df) < 80:
            continue
        stock_data[ticker] = df

    print(f"  로딩 완료: {len(stock_data)}종목 ({time.time() - t0:.1f}s)")

    # ── 2단계: 날짜 범위 설정 ──
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    # scan_end = 마지막 - hold_days (forward 확인용)
    scan_end_idx = len(all_dates) - 1 - hold_days
    scan_start_idx = max(scan_end_idx - args.days + 1, 60)
    scan_dates = all_dates[scan_start_idx:scan_end_idx + 1]
    print(f"  스캔: {scan_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{scan_dates[-1].strftime('%Y-%m-%d')} ({len(scan_dates)}일)")

    # ── 3단계: 시뮬레이션 ──
    trades = []
    t1 = time.time()

    for scan_i, scan_date in enumerate(scan_dates):
        if (scan_i + 1) % 20 == 0:
            print(f"  진행: {scan_i + 1}/{len(scan_dates)} "
                  f"({(scan_i + 1) / len(scan_dates) * 100:.0f}%)")

        for ticker, df in stock_data.items():
            if scan_date not in df.index:
                continue
            idx = df.index.get_loc(scan_date)
            if isinstance(idx, slice):
                idx = idx.start
            if idx < 60:
                continue

            close = float(df["Close"].iloc[idx])
            if close < min_price:
                continue
            vol_20d = float(df["Volume"].iloc[max(0, idx - 19):idx + 1].mean())
            if close * vol_20d < min_tv:
                continue

            # 패턴 감지
            closes = df["Close"].values.astype(float)
            highs_arr = df["High"].values.astype(float)
            volumes = df["Volume"].values.astype(float)
            rsis = df["RSI"].values.astype(float) if "RSI" in df.columns else None
            bb = df["Upper_Band"].values.astype(float) if "Upper_Band" in df.columns else None

            a = detect_volume_breakout(closes, highs_arr, volumes, rsis, bb, idx, cfg)
            b = detect_shakeout_reversal(closes, volumes, idx, cfg)
            c = detect_consecutive_momentum(closes, volumes, idx, cfg)

            pattern_set = set()
            for det in [a, b, c]:
                if det:
                    pattern_set.add(det["pattern"])

            if not pattern_set:
                continue

            # PRIMARY만 (C 포함)
            has_c = "C" in pattern_set
            has_a = "A" in pattern_set
            has_b = "B" in pattern_set
            if has_a and has_b and not has_c:
                continue  # WARNING — 스킵
            if not has_c:
                continue  # ALERT, WATCH — 스킵

            # OBV 수급
            obv_data = df["OBV"].values.astype(float) if "OBV" in df.columns else None
            supply_grade = "NONE"
            if obv_data is not None and idx >= 2:
                obv_rising = all(obv_data[i] > obv_data[i - 1]
                                 for i in range(max(1, idx - 2), idx + 1))
                vol_5 = np.mean(volumes[max(0, idx - 4):idx + 1])
                vol_20 = np.mean(volumes[max(0, idx - 19):idx + 1])
                vr_5_20 = vol_5 / vol_20 if vol_20 > 0 else 1.0
                if obv_rising and vr_5_20 >= 1.5:
                    supply_grade = "CONFIRMED"
                elif obv_rising or vr_5_20 >= 1.2:
                    supply_grade = "PARTIAL"

            # ATR 계산
            if "ATR" in df.columns and not pd.isna(df["ATR"].iloc[idx]):
                atr = float(df["ATR"].iloc[idx])
            else:
                # fallback: 14일 True Range 평균
                if idx >= 14:
                    tr_list = []
                    for k in range(idx - 13, idx + 1):
                        h = float(df["High"].iloc[k])
                        l = float(df["Low"].iloc[k])
                        pc = float(df["Close"].iloc[k - 1]) if k > 0 else l
                        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
                    atr = np.mean(tr_list)
                else:
                    atr = close * 0.05

            if atr <= 0:
                atr = close * 0.05

            sl_price = close - atr * sl_mult
            tp_price = close + atr * tp_mult

            # Forward 확인 (hold_days일)
            fwd_end = min(idx + hold_days, len(df) - 1)
            if fwd_end <= idx:
                continue

            # 일별 SL/TP 체크
            result = "TIME_EXIT"
            exit_price = float(df["Close"].iloc[fwd_end])
            exit_day = hold_days
            sl_hit_day = None
            tp_hit_day = None

            for d in range(1, fwd_end - idx + 1):
                day_low = float(df["Low"].iloc[idx + d])
                day_high = float(df["High"].iloc[idx + d])

                if day_low <= sl_price and sl_hit_day is None:
                    sl_hit_day = d
                if day_high >= tp_price and tp_hit_day is None:
                    tp_hit_day = d

            # 판정: SL/TP 동시 히트 → SL 우선 (보수적)
            if sl_hit_day is not None and (tp_hit_day is None or sl_hit_day <= tp_hit_day):
                result = "SL_HIT"
                exit_price = sl_price
                exit_day = sl_hit_day
            elif tp_hit_day is not None:
                result = "TP_HIT"
                exit_price = tp_price
                exit_day = tp_hit_day

            pnl_pct = (exit_price / close - 1) * 100
            pnl_net = pnl_pct - 0.5  # 수수료+슬리피지 0.5%

            group = "A+C" if (has_a and has_c) else "C_only"

            trades.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "date": scan_date.strftime("%Y-%m-%d"),
                "entry": int(close),
                "atr": round(atr, 1),
                "sl": round(sl_price),
                "tp": round(tp_price),
                "sl_pct": round(-atr * sl_mult / close * 100, 1),
                "tp_pct": round(atr * tp_mult / close * 100, 1),
                "result": result,
                "exit_price": round(exit_price),
                "exit_day": exit_day,
                "pnl_pct": round(pnl_pct, 2),
                "pnl_net": round(pnl_net, 2),
                "group": group,
                "supply": supply_grade,
            })

    elapsed = time.time() - t1
    print(f"\n  시뮬레이션 완료: {len(trades)}건 ({elapsed:.1f}s)")

    if not trades:
        print("  시그널 없음!")
        return

    df_t = pd.DataFrame(trades)

    # ── 4단계: 전체 통계 ──
    print(f"\n{'═' * 75}")
    print(f"  페이퍼 시뮬레이션 결과 (ATR×{sl_mult} SL / ATR×{tp_mult} TP / {hold_days}일)")
    print(f"{'═' * 75}")

    total = len(df_t)
    sl_hits = len(df_t[df_t["result"] == "SL_HIT"])
    tp_hits = len(df_t[df_t["result"] == "TP_HIT"])
    time_exits = len(df_t[df_t["result"] == "TIME_EXIT"])
    time_wins = len(df_t[(df_t["result"] == "TIME_EXIT") & (df_t["pnl_net"] > 0)])
    time_losses = len(df_t[(df_t["result"] == "TIME_EXIT") & (df_t["pnl_net"] <= 0)])

    wins = tp_hits + time_wins
    losses = sl_hits + time_losses
    win_rate = wins / total * 100 if total > 0 else 0

    avg_win = df_t[df_t["pnl_net"] > 0]["pnl_net"].mean() if wins > 0 else 0
    avg_loss = df_t[df_t["pnl_net"] <= 0]["pnl_net"].mean() if losses > 0 else 0
    avg_pnl = df_t["pnl_net"].mean()
    total_pnl = df_t["pnl_net"].sum()

    # Profit Factor
    gross_profit = df_t[df_t["pnl_net"] > 0]["pnl_net"].sum()
    gross_loss = abs(df_t[df_t["pnl_net"] <= 0]["pnl_net"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 기대값 (Expectancy)
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

    print(f"\n  총 거래: {total}건")
    print(f"  TP 히트: {tp_hits}건 ({tp_hits / total * 100:.1f}%)")
    print(f"  SL 히트: {sl_hits}건 ({sl_hits / total * 100:.1f}%)")
    print(f"  시간만료: {time_exits}건 (승 {time_wins} / 패 {time_losses})")
    print(f"  ───────────────────────────")
    print(f"  승률:    {win_rate:.1f}%")
    print(f"  평균 승: +{avg_win:.2f}%")
    print(f"  평균 패: {avg_loss:.2f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}% (수수료 후)")
    print(f"  총 PnL:  {total_pnl:+.1f}%")
    print(f"  PF:      {pf:.2f}")
    print(f"  기대값:  {expectancy:+.2f}% (거래당)")

    # ── 5단계: 그룹별 (C_only vs A+C) ──
    print(f"\n{'═' * 75}")
    print(f"  패턴별 시뮬레이션 비교")
    print(f"{'═' * 75}")

    header = (f"  {'그룹':>8} {'건수':>5} │ {'TP':>5} {'SL':>5} {'시간':>5} │"
              f" {'승률':>5} │ {'평균PnL':>8} {'PF':>6} │ {'기대값':>7}")
    print(header)
    print(f"  {'─' * 8} {'─' * 5} ┼ {'─' * 5} {'─' * 5} {'─' * 5} ┼"
          f" {'─' * 5} ┼ {'─' * 8} {'─' * 6} ┼ {'─' * 7}")

    for group in ["C_only", "A+C"]:
        gdf = df_t[df_t["group"] == group]
        n = len(gdf)
        if n < 3:
            continue
        tp_n = len(gdf[gdf["result"] == "TP_HIT"])
        sl_n = len(gdf[gdf["result"] == "SL_HIT"])
        te_n = len(gdf[gdf["result"] == "TIME_EXIT"])
        w = tp_n + len(gdf[(gdf["result"] == "TIME_EXIT") & (gdf["pnl_net"] > 0)])
        wr = w / n * 100
        avg = gdf["pnl_net"].mean()
        gp = gdf[gdf["pnl_net"] > 0]["pnl_net"].sum()
        gl = abs(gdf[gdf["pnl_net"] <= 0]["pnl_net"].sum())
        g_pf = gp / gl if gl > 0 else float("inf")
        aw = gdf[gdf["pnl_net"] > 0]["pnl_net"].mean() if w > 0 else 0
        al = gdf[gdf["pnl_net"] <= 0]["pnl_net"].mean() if (n - w) > 0 else 0
        exp = (wr / 100 * aw) + ((100 - wr) / 100 * al)
        print(f"  {group:>8} {n:>5} │ {tp_n:>5} {sl_n:>5} {te_n:>5} │"
              f" {wr:>4.1f}% │ {avg:>+7.2f}% {g_pf:>5.2f} │ {exp:>+6.2f}%")

    # ── 6단계: 수급별 ──
    print(f"\n{'═' * 75}")
    print(f"  수급 등급별 시뮬레이션")
    print(f"{'═' * 75}")

    print(header)
    print(f"  {'─' * 8} {'─' * 5} ┼ {'─' * 5} {'─' * 5} {'─' * 5} ┼"
          f" {'─' * 5} ┼ {'─' * 8} {'─' * 6} ┼ {'─' * 7}")

    for sg in ["CONFIRMED", "PARTIAL", "NONE"]:
        sdf = df_t[df_t["supply"] == sg]
        n = len(sdf)
        if n < 3:
            continue
        tp_n = len(sdf[sdf["result"] == "TP_HIT"])
        sl_n = len(sdf[sdf["result"] == "SL_HIT"])
        te_n = len(sdf[sdf["result"] == "TIME_EXIT"])
        w = tp_n + len(sdf[(sdf["result"] == "TIME_EXIT") & (sdf["pnl_net"] > 0)])
        wr = w / n * 100
        avg = sdf["pnl_net"].mean()
        gp = sdf[sdf["pnl_net"] > 0]["pnl_net"].sum()
        gl = abs(sdf[sdf["pnl_net"] <= 0]["pnl_net"].sum())
        g_pf = gp / gl if gl > 0 else float("inf")
        aw = sdf[sdf["pnl_net"] > 0]["pnl_net"].mean() if w > 0 else 0
        al = sdf[sdf["pnl_net"] <= 0]["pnl_net"].mean() if (n - w) > 0 else 0
        exp = (wr / 100 * aw) + ((100 - wr) / 100 * al)
        print(f"  {sg:>8} {n:>5} │ {tp_n:>5} {sl_n:>5} {te_n:>5} │"
              f" {wr:>4.1f}% │ {avg:>+7.2f}% {g_pf:>5.2f} │ {exp:>+6.2f}%")

    # ── 7단계: 평균 청산일 분석 ──
    print(f"\n{'═' * 75}")
    print(f"  결과별 평균 청산일")
    print(f"{'═' * 75}")
    for r in ["TP_HIT", "SL_HIT", "TIME_EXIT"]:
        rdf = df_t[df_t["result"] == r]
        if len(rdf) > 0:
            avg_day = rdf["exit_day"].mean()
            med_day = rdf["exit_day"].median()
            print(f"  {r:>10}: 평균 {avg_day:.1f}일, 중앙값 {med_day:.0f}일 ({len(rdf)}건)")

    # ── 8단계: ATR 범위별 분석 ──
    print(f"\n{'═' * 75}")
    print(f"  ATR 비율별 성과 (SL% 기준)")
    print(f"{'═' * 75}")

    df_t["sl_pct_abs"] = df_t["sl_pct"].abs()
    bins = [(0, 5, "~5%"), (5, 8, "5~8%"), (8, 12, "8~12%"), (12, 100, "12%+")]
    for lo, hi, label in bins:
        bdf = df_t[(df_t["sl_pct_abs"] >= lo) & (df_t["sl_pct_abs"] < hi)]
        n = len(bdf)
        if n < 3:
            continue
        w = len(bdf[bdf["pnl_net"] > 0])
        wr = w / n * 100
        avg = bdf["pnl_net"].mean()
        print(f"  SL {label:>5}: {n:>4}건, 승률 {wr:.1f}%, 평균PnL {avg:+.2f}%")

    # ── 9단계: 보유일수별 분석 ──
    print(f"\n{'═' * 75}")
    print(f"  보유일수 효과 분석 (시간만료 건)")
    print(f"{'═' * 75}")

    te_df = df_t[df_t["result"] == "TIME_EXIT"]
    if len(te_df) >= 10:
        for day_limit in [3, 5, 7, 10]:
            sub = te_df[te_df["exit_day"] <= day_limit]
            if len(sub) > 0:
                w = len(sub[sub["pnl_net"] > 0])
                wr = w / len(sub) * 100
                avg = sub["pnl_net"].mean()
                print(f"  {day_limit}일 이내: {len(sub)}건, 승률 {wr:.1f}%, 평균PnL {avg:+.2f}%")

    # ── 10단계: 저장 ──
    sim_output = {
        "mode": "paper_sim",
        "hold_days": hold_days,
        "scan_days": args.days,
        "scan_range": {
            "start": scan_dates[0].strftime("%Y-%m-%d"),
            "end": scan_dates[-1].strftime("%Y-%m-%d"),
        },
        "total_trades": total,
        "summary": {
            "win_rate": round(win_rate, 1),
            "avg_pnl_net": round(avg_pnl, 2),
            "pf": round(pf, 2),
            "expectancy": round(expectancy, 2),
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
        },
        "trades": trades[:200],  # 상위 200건
    }

    sim_path = PROJECT_ROOT / "data" / "smallcap_paper_sim.json"
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {sim_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=120,
                        help="과거 N 거래일 스캔 (기본: 120)")
    parser.add_argument("--top", type=int, default=50,
                        help="결과 출력 상위 N건")
    parser.add_argument("--forward", type=str, default="5,10,20",
                        help="Forward return 기간 (기본: 5,10,20)")
    parser.add_argument("--paper-sim", action="store_true",
                        help="ATR 기반 SL/TP 페이퍼 시뮬레이션 모드")
    parser.add_argument("--hold-days", type=int, default=5,
                        help="페이퍼 시뮬레이션 보유일수 (기본: 5)")
    parser.add_argument("--sl-mult", type=float, default=1.0,
                        help="SL ATR 배수 (기본: 1.0)")
    parser.add_argument("--tp-mult", type=float, default=3.0,
                        help="TP ATR 배수 (기본: 3.0)")
    args = parser.parse_args()

    if args.paper_sim:
        run_paper_sim(args)
        return

    forward_days = [int(x) for x in args.forward.split(",")]
    cfg = load_settings()
    min_tv = float(cfg.get("min_trading_value", 5e8))
    min_price = float(cfg.get("min_price", 1000))

    print(f"[소형주 패턴 Forward Return 백테스트]")
    print(f"  스캔 범위: 과거 {args.days} 거래일")
    print(f"  Forward: {forward_days}일")
    print(f"  필터: 거래대금≥{min_tv/1e8:.0f}억, 종가≥{min_price:.0f}원")
    print("=" * 70)

    # ── 1단계: CSV 전종목 로딩 ──
    t0 = time.time()
    name_map = {}
    stock_data = {}  # ticker → DataFrame

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    print(f"  CSV 로딩: {len(csv_files)}개...")

    for csv_path in csv_files:
        parts = csv_path.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts
        name_map[ticker] = name

        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
        except Exception:
            continue

        if len(df) < 80:  # 60일 데이터 + 20일 forward 최소
            continue

        stock_data[ticker] = df

    print(f"  로딩 완료: {len(stock_data)}종목 ({time.time() - t0:.1f}s)")

    # ── 2단계: 공통 날짜 인덱스 구축 ──
    # 가장 많은 종목이 공유하는 날짜를 기준으로
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    if len(all_dates) < args.days + max(forward_days) + 60:
        print(f"  ERROR: 데이터 부족 ({len(all_dates)}일 < {args.days + max(forward_days) + 60}일)")
        return

    # 스캔 범위: [scan_start, scan_end]
    # scan_end = 마지막 날짜 - max(forward_days) (forward 계산 가능하도록)
    max_fwd = max(forward_days)
    scan_end_idx = len(all_dates) - 1 - max_fwd
    scan_start_idx = max(scan_end_idx - args.days + 1, 60)

    scan_dates = all_dates[scan_start_idx:scan_end_idx + 1]
    print(f"  스캔 날짜: {scan_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{scan_dates[-1].strftime('%Y-%m-%d')} ({len(scan_dates)}일)")

    # ── 3단계: 날짜별 스캔 + Forward Return 계산 ──
    all_signals = []  # 전체 시그널 기록
    t1 = time.time()

    for scan_i, scan_date in enumerate(scan_dates):
        if (scan_i + 1) % 20 == 0:
            pct = (scan_i + 1) / len(scan_dates) * 100
            print(f"  진행: {scan_i + 1}/{len(scan_dates)} ({pct:.0f}%)")

        for ticker, df in stock_data.items():
            # 해당 날짜가 이 종목 데이터에 있는지
            if scan_date not in df.index:
                continue

            idx = df.index.get_loc(scan_date)
            if isinstance(idx, slice):
                idx = idx.start
            if idx < 60:
                continue

            # 기본 필터: 종가, 거래대금
            close = float(df["Close"].iloc[idx])
            if close < min_price:
                continue
            vol_20d = float(df["Volume"].iloc[max(0, idx - 19):idx + 1].mean())
            if close * vol_20d < min_tv:
                continue

            # 패턴 감지
            closes = df["Close"].values.astype(float)
            highs = df["High"].values.astype(float)
            volumes = df["Volume"].values.astype(float)
            rsis = df["RSI"].values.astype(float) if "RSI" in df.columns else None
            bb = df["Upper_Band"].values.astype(float) if "Upper_Band" in df.columns else None

            patterns_found = []
            a = detect_volume_breakout(closes, highs, volumes, rsis, bb, idx, cfg)
            b = detect_shakeout_reversal(closes, volumes, idx, cfg)
            c = detect_consecutive_momentum(closes, volumes, idx, cfg)

            for det in [a, b, c]:
                if det:
                    patterns_found.append(det)

            if not patterns_found:
                continue

            # 복합 패턴 분류
            pattern_set = set(p["pattern"] for p in patterns_found)

            if "A" in pattern_set and "B" in pattern_set and "C" in pattern_set:
                group = "A+B+C"
            elif "A" in pattern_set and "B" in pattern_set:
                group = "A+B"
            elif "A" in pattern_set and "C" in pattern_set:
                group = "A+C"
            elif "B" in pattern_set and "C" in pattern_set:
                group = "B+C"
            elif "A" in pattern_set:
                group = "A_only"
            elif "B" in pattern_set:
                group = "B_only"
            else:
                group = "C_only"

            # 등급 (v2 기준)
            has_c = "C" in pattern_set
            has_a = "A" in pattern_set
            has_b = "B" in pattern_set
            if has_a and has_b and not has_c:
                grade = "WARNING"
            elif has_c:
                grade = "PRIMARY"
            elif has_b:
                grade = "ALERT"
            else:
                grade = "WATCH"

            # OBV 수급 확인
            obv_data = df["OBV"].values.astype(float) if "OBV" in df.columns else None
            supply_grade = "N/A"
            if obv_data is not None and idx >= 2:
                obv_rising = all(obv_data[i] > obv_data[i-1]
                                 for i in range(max(1, idx-2), idx+1))
                vol_5 = np.mean(volumes[max(0, idx-4):idx+1])
                vol_20 = np.mean(volumes[max(0, idx-19):idx+1])
                vr_5_20 = vol_5 / vol_20 if vol_20 > 0 else 1.0
                if obv_rising and vr_5_20 >= 1.5:
                    supply_grade = "CONFIRMED"
                elif obv_rising or vr_5_20 >= 1.2:
                    supply_grade = "PARTIAL"
                else:
                    supply_grade = "NONE"

            # Forward Return 계산
            fwd_returns = {}
            valid_fwd = True
            for fd in forward_days:
                fwd_idx = idx + fd
                if fwd_idx >= len(df):
                    valid_fwd = False
                    break
                fwd_close = float(df["Close"].iloc[fwd_idx])
                fwd_ret = (fwd_close / close - 1) * 100
                fwd_returns[f"fwd_{fd}d"] = round(fwd_ret, 2)

            if not valid_fwd:
                continue

            # Max drawdown in forward period (최대 낙폭)
            max_fwd_idx = min(idx + max(forward_days), len(df) - 1)
            fwd_lows = df["Low"].values[idx + 1:max_fwd_idx + 1].astype(float)
            max_dd = ((np.min(fwd_lows) / close - 1) * 100) if len(fwd_lows) > 0 else 0

            # Max upside in forward period (최대 상승)
            fwd_highs = df["High"].values[idx + 1:max_fwd_idx + 1].astype(float)
            max_up = ((np.max(fwd_highs) / close - 1) * 100) if len(fwd_highs) > 0 else 0

            all_signals.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "date": scan_date.strftime("%Y-%m-%d"),
                "close": int(close),
                "group": group,
                "grade": grade,
                "supply": supply_grade,
                "patterns": [p["pattern"] for p in patterns_found],
                **fwd_returns,
                "max_dd_pct": round(max_dd, 2),
                "max_up_pct": round(max_up, 2),
            })

    elapsed = time.time() - t1
    print(f"\n  스캔 완료: {len(all_signals)}건 시그널 ({elapsed:.1f}s)")

    if not all_signals:
        print("  시그널 없음!")
        return

    # ── 4단계: 패턴 그룹별 통계 ──
    df_sig = pd.DataFrame(all_signals)

    print(f"\n{'═' * 75}")
    print(f"  패턴 그룹별 Forward Return 분석")
    print(f"{'═' * 75}")

    groups_order = ["A+B", "A+B+C", "A+C", "B+C", "A_only", "B_only", "C_only"]
    existing_groups = [g for g in groups_order if g in df_sig["group"].unique()]
    # 누락된 그룹 추가
    for g in df_sig["group"].unique():
        if g not in existing_groups:
            existing_groups.append(g)

    summary_rows = []

    for group in existing_groups:
        gdf = df_sig[df_sig["group"] == group]
        n = len(gdf)
        if n < 3:
            continue

        row = {"group": group, "n": n}

        for fd in forward_days:
            col = f"fwd_{fd}d"
            vals = gdf[col].values
            avg_ret = np.mean(vals)
            med_ret = np.median(vals)
            win_rate = np.sum(vals > 0) / len(vals) * 100
            win_rate_5 = np.sum(vals >= 5) / len(vals) * 100  # +5% 이상 승률

            row[f"avg_{fd}d"] = round(avg_ret, 2)
            row[f"med_{fd}d"] = round(med_ret, 2)
            row[f"wr_{fd}d"] = round(win_rate, 1)
            row[f"wr5_{fd}d"] = round(win_rate_5, 1)

        row["avg_max_dd"] = round(gdf["max_dd_pct"].mean(), 2)
        row["avg_max_up"] = round(gdf["max_up_pct"].mean(), 2)

        summary_rows.append(row)

    # 전체 통계
    total_row = {"group": "ALL", "n": len(df_sig)}
    for fd in forward_days:
        col = f"fwd_{fd}d"
        vals = df_sig[col].values
        total_row[f"avg_{fd}d"] = round(np.mean(vals), 2)
        total_row[f"med_{fd}d"] = round(np.median(vals), 2)
        total_row[f"wr_{fd}d"] = round(np.sum(vals > 0) / len(vals) * 100, 1)
        total_row[f"wr5_{fd}d"] = round(np.sum(vals >= 5) / len(vals) * 100, 1)
    total_row["avg_max_dd"] = round(df_sig["max_dd_pct"].mean(), 2)
    total_row["avg_max_up"] = round(df_sig["max_up_pct"].mean(), 2)
    summary_rows.append(total_row)

    # 출력 테이블
    header = (f"  {'그룹':>8} {'건수':>5} │"
              f" {'5d평균':>7} {'5d승률':>6} │"
              f" {'10d평균':>7} {'10d승률':>7} │"
              f" {'20d평균':>7} {'20d승률':>7} │"
              f" {'최대DD':>7} {'최대UP':>7}")
    print(header)
    print(f"  {'─' * 8} {'─' * 5} ┼ {'─' * 7} {'─' * 6} ┼ {'─' * 7} {'─' * 7} ┼"
          f" {'─' * 7} {'─' * 7} ┼ {'─' * 7} {'─' * 7}")

    for row in summary_rows:
        if row["group"] == "ALL":
            print(f"  {'─' * 80}")
        line = f"  {row['group']:>8} {row['n']:>5} │"
        for fd in forward_days:
            avg = row.get(f"avg_{fd}d", 0)
            wr = row.get(f"wr_{fd}d", 0)
            line += f" {avg:>+6.1f}% {wr:>5.1f}% │"
        line += f" {row.get('avg_max_dd', 0):>+6.1f}% {row.get('avg_max_up', 0):>+6.1f}%"
        print(line)

    # ── 5단계: 등급별 분석 (v2 핵심) ──
    print(f"\n{'═' * 75}")
    print(f"  등급별 Forward Return (v2 검증)")
    print(f"{'═' * 75}")

    grade_order = ["PRIMARY", "WARNING", "ALERT", "WATCH"]
    print(f"  {'등급':>10} {'건수':>5} │"
          f" {'5d평균':>7} {'5d승률':>6} │"
          f" {'10d평균':>7} {'10d승률':>7} │"
          f" {'20d평균':>7} {'20d승률':>7} │"
          f" {'MaxDD':>7}")
    print(f"  {'─' * 10} {'─' * 5} ┼ {'─' * 7} {'─' * 6} ┼ {'─' * 7} {'─' * 7} ┼"
          f" {'─' * 7} {'─' * 7} ┼ {'─' * 7}")

    for grade in grade_order:
        gdf = df_sig[df_sig["grade"] == grade]
        n = len(gdf)
        if n < 3:
            continue
        line = f"  {grade:>10} {n:>5} │"
        for fd in forward_days:
            col = f"fwd_{fd}d"
            vals = gdf[col].values
            avg = np.mean(vals)
            wr = np.sum(vals > 0) / len(vals) * 100
            line += f" {avg:>+6.1f}% {wr:>5.1f}% │"
        line += f" {gdf['max_dd_pct'].mean():>+6.1f}%"
        print(line)

    # ── 6단계: PRIMARY 수급별 분석 (핵심 질문: 수급 확인이 성과 개선?) ──
    primary_sig = df_sig[df_sig["grade"] == "PRIMARY"]
    if len(primary_sig) >= 10:
        print(f"\n{'═' * 75}")
        print(f"  PRIMARY 수급 필터 효과 검증")
        print(f"{'═' * 75}")

        supply_groups = ["CONFIRMED", "PARTIAL", "NONE"]
        print(f"  {'수급':>12} {'건수':>5} │"
              f" {'5d평균':>7} {'5d승률':>6} │"
              f" {'10d평균':>7} {'10d승률':>7} │"
              f" {'20d평균':>7} {'20d승률':>7} │"
              f" {'MaxDD':>7}")
        print(f"  {'─' * 12} {'─' * 5} ┼ {'─' * 7} {'─' * 6} ┼ {'─' * 7} {'─' * 7} ┼"
              f" {'─' * 7} {'─' * 7} ┼ {'─' * 7}")

        for sg in supply_groups:
            sdf = primary_sig[primary_sig["supply"] == sg]
            n = len(sdf)
            if n < 3:
                continue
            line = f"  {sg:>12} {n:>5} │"
            for fd in forward_days:
                col = f"fwd_{fd}d"
                vals = sdf[col].values
                avg = np.mean(vals)
                wr = np.sum(vals > 0) / len(vals) * 100
                line += f" {avg:>+6.1f}% {wr:>5.1f}% │"
            line += f" {sdf['max_dd_pct'].mean():>+6.1f}%"
            print(line)

        # PRIMARY 전체
        n = len(primary_sig)
        line = f"  {'ALL_PRIMARY':>12} {n:>5} │"
        for fd in forward_days:
            col = f"fwd_{fd}d"
            vals = primary_sig[col].values
            avg = np.mean(vals)
            wr = np.sum(vals > 0) / len(vals) * 100
            line += f" {avg:>+6.1f}% {wr:>5.1f}% │"
        line += f" {primary_sig['max_dd_pct'].mean():>+6.1f}%"
        print(f"  {'─' * 80}")
        print(line)

    # ── 7단계: 종목 중복 시그널 빈도 ──
    ticker_counts = df_sig["ticker"].value_counts()
    repeat_tickers = ticker_counts[ticker_counts >= 3]
    if len(repeat_tickers) > 0:
        print(f"\n{'═' * 75}")
        print(f"  3회+ 반복 감지 종목 (추세 주도주)")
        print(f"{'═' * 75}")
        for ticker, cnt in repeat_tickers.head(15).items():
            name = name_map.get(ticker, ticker)
            subset = df_sig[df_sig["ticker"] == ticker]
            avg_10d = subset["fwd_10d"].mean()
            grade_dist = subset["grade"].value_counts().to_dict()
            print(f"    {name}({ticker}): {cnt}회, 10d {avg_10d:+.1f}%, 등급:{grade_dist}")

    # ── 9단계: 저장 ──
    output = {
        "scan_days": args.days,
        "forward_days": forward_days,
        "total_signals": len(all_signals),
        "scan_range": {
            "start": scan_dates[0].strftime("%Y-%m-%d"),
            "end": scan_dates[-1].strftime("%Y-%m-%d"),
        },
        "group_summary": summary_rows,
        "top_signals": all_signals[:100],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
