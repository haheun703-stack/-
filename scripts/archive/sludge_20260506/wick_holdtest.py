"""강한마감 + 쌍끌이 패턴: 시초가 매도 vs 보유 타이밍 백테스트"""
import pandas as pd
import numpy as np
import glob
import os

files = sorted(glob.glob("data/raw/*.parquet"))
print(f"universe: {len(files)} stocks")

all_signals = []

for f in files:
    ticker = os.path.basename(f).replace(".parquet", "")
    try:
        df = pd.read_parquet(f)
        if len(df) < 60:
            continue
        has_supply = "기관합계" in df.columns
        df = df[df["close"] > 0].copy()

        rng = df["high"] - df["low"]
        close_pos = np.where(rng > 0, (df["close"] - df["low"]) / rng, 0.5)
        day_ret = df["close"].pct_change() * 100
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()

        df["d1_open"] = df["open"].shift(-1)
        df["d1_close"] = df["close"].shift(-1)
        df["d1_high"] = df["high"].shift(-1)
        df["d1_low"] = df["low"].shift(-1)
        df["d2_close"] = df["close"].shift(-2)
        df["d3_close"] = df["close"].shift(-3)
        df["d5_close"] = df["close"].shift(-5)

        df["gap_ret"] = (df["d1_open"] / df["close"] - 1) * 100
        df["d1_open_to_close"] = (df["d1_close"] / df["d1_open"] - 1) * 100
        df["d1_ret"] = (df["d1_close"] / df["close"] - 1) * 100
        df["d1_max_from_open"] = (df["d1_high"] / df["d1_open"] - 1) * 100
        df["d1_min_from_open"] = (df["d1_low"] / df["d1_open"] - 1) * 100
        df["d2_ret"] = (df["d2_close"] / df["close"] - 1) * 100
        df["d3_ret"] = (df["d3_close"] / df["close"] - 1) * 100
        df["d5_ret"] = (df["d5_close"] / df["close"] - 1) * 100
        df["d2_from_d1open"] = (df["d2_close"] / df["d1_open"] - 1) * 100

        df["close_pos"] = close_pos
        df["day_ret"] = day_ret
        df["vol_ratio"] = vol_ratio

        if has_supply:
            mask = (
                (df["close_pos"] >= 0.80) &
                (df["day_ret"] >= 2.0) &
                (df["vol_ratio"] >= 1.2) &
                (df["기관합계"] > 0) &
                (df["외국인합계"] > 0) &
                (df["gap_ret"].notna()) &
                (df["d5_ret"].notna())
            )
        else:
            mask = (
                (df["close_pos"] >= 0.80) &
                (df["day_ret"] >= 2.0) &
                (df["vol_ratio"] >= 1.2) &
                (df["gap_ret"].notna()) &
                (df["d5_ret"].notna())
            )

        cols = ["close", "day_ret", "vol_ratio", "close_pos",
                "gap_ret", "d1_open_to_close", "d1_ret",
                "d1_max_from_open", "d1_min_from_open",
                "d2_ret", "d3_ret", "d5_ret", "d2_from_d1open"]
        signals = df[mask][cols].copy()
        signals["ticker"] = ticker
        all_signals.append(signals)
    except Exception:
        pass

result = pd.concat(all_signals) if all_signals else pd.DataFrame()
print(f"strong close + dual buy signals: {len(result):,}\n")

if len(result) == 0:
    print("NO SIGNALS")
    exit()

sep = "=" * 70

# A. Timing Analysis
print(sep)
print("A. TIMING ANALYSIS: when to sell?")
print(sep)
print(f"  {'Metric':<35} {'Avg':>8} {'Med':>8} {'WR':>6} {'PF':>6}")
print(f"  {'-'*65}")

for label, col in [
    ("Overnight Gap (close->D+1 open)", "gap_ret"),
    ("D+1 Intraday (open->close)", "d1_open_to_close"),
    ("D+1 Total (close->close)", "d1_ret"),
    ("D+2 Total", "d2_ret"),
    ("D+3 Total", "d3_ret"),
    ("D+5 Total", "d5_ret"),
    ("Hold D+1open->D+2close", "d2_from_d1open"),
]:
    s = result[col].dropna()
    wr = (s > 0).mean() * 100
    w = s[s > 0].sum()
    l = abs(s[s < 0].sum())
    pf = w / l if l > 0 else 999
    print(f"  {label:<35} {s.mean():>+7.2f}% {s.median():>+7.2f}% {wr:>5.1f}% {pf:>5.2f}")

# B. D+1 Intraday detail
print(f"\n{sep}")
print("B. D+1 INTRADAY: after open, what happens?")
print(sep)
d1_otc = result["d1_open_to_close"]
d1_max = result["d1_max_from_open"]
d1_min = result["d1_min_from_open"]
print(f"  D+1 open->close avg: {d1_otc.mean():+.2f}% (median {d1_otc.median():+.2f}%)")
print(f"  D+1 open->high  avg: {d1_max.mean():+.2f}% (best from open)")
print(f"  D+1 open->low   avg: {d1_min.mean():+.2f}% (worst from open)")
print(f"  D+1 open->close < -3%: {(d1_otc < -3).mean()*100:.1f}%")
print(f"  D+1 open->close < -5%: {(d1_otc < -5).mean()*100:.1f}%")
print(f"  D+1 open->low   < -5%: {(d1_min < -5).mean()*100:.1f}%")
print(f"  D+1 open->low   < -10%: {(d1_min < -10).mean()*100:.1f}%")

# C. Gap-up then Fade
print(f"\n{sep}")
print("C. GAP-UP then FADE (gap>=2% then intraday negative)")
print(sep)
gap_up = result[result["gap_ret"] >= 2]
if len(gap_up) > 10:
    otc = gap_up["d1_open_to_close"]
    fade = (otc < 0).mean() * 100
    crash = (otc < -3).mean() * 100
    deep = (otc < -5).mean() * 100
    print(f"  Gap >= +2%: n={len(gap_up):,}")
    print(f"  -> D+1 intraday negative (fade): {fade:.1f}%")
    print(f"  -> D+1 intraday < -3% (crash): {crash:.1f}%")
    print(f"  -> D+1 intraday < -5% (deep crash): {deep:.1f}%")
    print(f"  -> D+1 total avg: {gap_up['d1_ret'].mean():+.2f}%")
    print(f"  -> D+5 total avg: {gap_up['d5_ret'].mean():+.2f}%")

gap_big = result[result["gap_ret"] >= 5]
if len(gap_big) > 10:
    otc2 = gap_big["d1_open_to_close"]
    fade2 = (otc2 < 0).mean() * 100
    crash2 = (otc2 < -3).mean() * 100
    print(f"\n  Gap >= +5%: n={len(gap_big):,}")
    print(f"  -> fade: {fade2:.1f}% | crash(-3%+): {crash2:.1f}%")
    print(f"  -> D+1 total avg: {gap_big['d1_ret'].mean():+.2f}%")

# D. By day_return size
print(f"\n{sep}")
print("D. HOLDING RISK by same-day gain size")
print(sep)
hdr = f"  {'DayGain':<10} {'n':>6} | {'Gap':>7} {'D1otc':>7} {'D+1':>7} {'D+5':>7} | {'Fade':>6} {'Crash':>6}"
print(hdr)
print(f"  {'-'*75}")
for lo, hi, label in [(2, 5, "2~5%"), (5, 10, "5~10%"), (10, 20, "10~20%"), (20, 50, "20%+")]:
    sub = result[(result["day_ret"] >= lo) & (result["day_ret"] < hi)]
    if len(sub) > 10:
        gap = sub["gap_ret"].mean()
        otc = sub["d1_open_to_close"].mean()
        d1 = sub["d1_ret"].mean()
        d5 = sub["d5_ret"].mean()
        fade = (sub["d1_open_to_close"] < 0).mean() * 100
        crash = (sub["d1_open_to_close"] < -3).mean() * 100
        print(f"  {label:<10} {len(sub):>6} | {gap:>+6.2f}% {otc:>+6.2f}% {d1:>+6.2f}% {d5:>+6.2f}% | {fade:>5.1f}% {crash:>5.1f}%")

# E. Strategy comparison
print(f"\n{sep}")
print("E. STRATEGY COMPARISON (per trade avg)")
print(sep)
s1 = result["gap_ret"]
s2 = result["d1_ret"]
s3 = result["d1_open_to_close"]
s4 = result["d5_ret"]

for lbl, s in [
    ("S1 overnight (close->D+1 open)", s1),
    ("S2 1-day (close->D+1 close)", s2),
    ("S3 D+1 intraday (open->close)", s3),
    ("S4 5-day (close->D+5 close)", s4),
]:
    w = s[s > 0].sum()
    l = abs(s[s < 0].sum())
    pf = w / l if l > 0 else 999
    print(f"  {lbl}: avg={s.mean():+.2f}% WR={(s>0).mean()*100:.0f}% PF={pf:.2f}")

best = "S1 overnight" if s1.mean() > s2.mean() else "S2 1-day"
verdict = "HOLD through day OK" if s3.mean() > 0 else "SELL AT OPEN!"
print(f"\n  -> BEST: {best}")
print(f"  -> D+1 intraday direction: {s3.mean():+.2f}% = {verdict}")

# F. Crash distribution
print(f"\n{sep}")
print("F. WORST CASE SCENARIOS (if you hold to D+5)")
print(sep)
d5 = result["d5_ret"]
for thr, lbl in [(-20, "< -20%"), (-15, "< -15%"), (-10, "< -10%"), (-5, "< -5%")]:
    cnt = (d5 < thr).sum()
    pct = (d5 < thr).mean() * 100
    print(f"  D+5 {lbl}: {pct:.1f}% ({cnt:,} trades)")

for thr, lbl in [(5, "> +5%"), (10, "> +10%"), (20, "> +20%")]:
    pct = (d5 > thr).mean() * 100
    print(f"  D+5 {lbl}: {pct:.1f}%")

worst10 = result.nsmallest(10, "d5_ret")
print(f"\n  WORST 10 (D+5 return):")
for idx, row in worst10.iterrows():
    dt = str(idx)[:10]
    print(f"    {dt} {row['ticker']} day={row['day_ret']:+.1f}% gap={row['gap_ret']:+.1f}% D+1={row['d1_ret']:+.1f}% D+5={row['d5_ret']:+.1f}%")

print(f"\n{sep}")
print("DONE")
