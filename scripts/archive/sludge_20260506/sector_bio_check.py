"""4/15 섹터별 수익률 + 바이오 섹터 분석"""
import pandas as pd
import numpy as np
import json
import os
import glob

name_map = json.load(open("data/universe/name_map.json"))
sector_map = pd.read_csv("data/universe/sector_map.csv")

tk_sector = {}
for _, row in sector_map.iterrows():
    t = str(row.get("ticker", "")).zfill(6)
    s = row.get("sector", "")
    tk_sector[t] = s

files = sorted(glob.glob("data/raw/*.parquet"))
sector_data = {}
bio_stocks = []

for f in files:
    ticker = os.path.basename(f).replace(".parquet", "")
    try:
        df = pd.read_parquet(f)
        if len(df) < 20:
            continue
        df = df[df["close"] > 0]
        last_date = str(df.index[-1])[:10]
        if "2026-04-15" not in last_date:
            continue

        close = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        ret = (close / prev - 1) * 100
        vol = df["volume"].iloc[-1]
        vol_20 = df["volume"].tail(20).mean()
        vol_ratio = vol / vol_20 if vol_20 > 0 else 0

        has_supply = "기관합계" in df.columns
        inst = df["기관합계"].iloc[-1] / 1e8 if has_supply else 0
        frgn = df["외국인합계"].iloc[-1] / 1e8 if has_supply else 0

        sector = tk_sector.get(ticker, "Unknown")
        name = name_map.get(ticker, ticker)

        if sector not in sector_data:
            sector_data[sector] = []
        sector_data[sector].append({
            "ticker": ticker, "name": name, "ret": ret,
            "vol_ratio": vol_ratio, "inst": inst, "frgn": frgn,
            "close": int(close), "trading_val": close * vol
        })

        is_bio = ("바이오" in sector or "제약" in sector or "의약" in sector or
                  "헬스" in sector or "바이오" in name or "제약" in name or
                  "셀" in name or "젠" in name or "메디" in name or
                  "팜" in name or "랩" in name or "텍" in name and "바이오" in name)
        if is_bio:
            bio_stocks.append({
                "ticker": ticker, "name": name, "ret": ret,
                "vol_ratio": vol_ratio, "inst": inst, "frgn": frgn,
                "close": int(close), "sector": sector,
                "trading_val": close * vol
            })
    except Exception:
        pass

# Sector ranking
sep = "=" * 70
print(sep)
print("4/15 SECTOR RETURN RANKING")
print(sep)
sector_stats = []
for s, stocks in sector_data.items():
    if len(stocks) < 3:
        continue
    rets = [x["ret"] for x in stocks]
    avg_ret = np.mean(rets)
    med_ret = np.median(rets)
    up_pct = sum(1 for r in rets if r > 0) / len(rets) * 100
    inst_sum = sum(x["inst"] for x in stocks)
    frgn_sum = sum(x["frgn"] for x in stocks)
    count = len(stocks)
    sector_stats.append((s, avg_ret, med_ret, up_pct, count, inst_sum, frgn_sum))

sector_stats.sort(key=lambda x: x[1], reverse=True)
print(f"  {'Sector':<14} {'Avg':>7} {'Med':>7} {'UpPct':>6} {'n':>4} {'Inst':>8} {'Frgn':>8}")
print(f"  {'-'*58}")
for s, avg, med, up, n, ist, frg in sector_stats[:25]:
    mark = " <<<" if any(k in s for k in ["바이오", "제약", "의약", "헬스"]) else ""
    print(f"  {s:<14} {avg:>+6.2f}% {med:>+6.2f}% {up:>5.0f}% {n:>4} {ist:>+7.0f}a {frg:>+7.0f}a{mark}".replace("a", "억"))

print(f"\n  ... bottom 5:")
for s, avg, med, up, n, ist, frg in sector_stats[-5:]:
    print(f"  {s:<14} {avg:>+6.2f}% {med:>+6.2f}% {up:>5.0f}% {n:>4}")

# Bio detail
print(f"\n{sep}")
print(f"BIO/PHARMA STOCKS (n={len(bio_stocks)})")
print(sep)
bio_stocks.sort(key=lambda x: x["ret"], reverse=True)
avg_bio = np.mean([x["ret"] for x in bio_stocks]) if bio_stocks else 0
med_bio = np.median([x["ret"] for x in bio_stocks]) if bio_stocks else 0
up_bio = sum(1 for x in bio_stocks if x["ret"] > 0) / len(bio_stocks) * 100 if bio_stocks else 0
cnt3 = sum(1 for x in bio_stocks if x["ret"] >= 3)
cnt5 = sum(1 for x in bio_stocks if x["ret"] >= 5)
print(f"  avg: {avg_bio:+.2f}% | median: {med_bio:+.2f}% | up: {up_bio:.0f}%")
print(f"  +3%+: {cnt3}stocks | +5%+: {cnt5}stocks")
print()

print(f"  [TOP 25 bio gainers]")
print(f"  {'Name':<14} {'Close':>8} {'Ret':>7} {'Vol':>5} {'Inst':>7} {'Frgn':>7} {'Sector':<10}")
print(f"  {'-'*65}")
for s in bio_stocks[:25]:
    print(f"  {s['name']:<14} {s['close']:>8,} {s['ret']:>+6.1f}% {s['vol_ratio']:>4.1f}x {s['inst']:>+6.0f}억 {s['frgn']:>+6.0f}억 {s['sector']:<10}")

bio_losers = [x for x in bio_stocks if x["ret"] < -1]
if bio_losers:
    print(f"\n  [bio losers (< -1%): {len(bio_losers)}]")
    for s in sorted(bio_losers, key=lambda x: x["ret"])[:8]:
        print(f"  {s['name']:<14} {s['close']:>8,} {s['ret']:>+6.1f}%")

# Market comparison
print(f"\n{sep}")
print("SECTOR-WIDE CHECK")
print(sep)
all_rets = []
for stocks in sector_data.values():
    all_rets.extend([x["ret"] for x in stocks])
mkt_avg = np.mean(all_rets)
mkt_med = np.median(all_rets)
mkt_up = sum(1 for r in all_rets if r > 0) / len(all_rets) * 100
print(f"  Market: avg {mkt_avg:+.2f}% med {mkt_med:+.2f}% up {mkt_up:.0f}%")
print(f"  Bio:    avg {avg_bio:+.2f}% med {med_bio:+.2f}% up {up_bio:.0f}%")
print(f"  Diff:   avg {avg_bio - mkt_avg:+.2f}% med {med_bio - mkt_med:+.2f}%")

bio_inst = sum(x["inst"] for x in bio_stocks)
bio_frgn = sum(x["frgn"] for x in bio_stocks)
print(f"\n  Bio total flow: Inst {bio_inst:+,.0f}억 | Frgn {bio_frgn:+,.0f}억")

# 5-day trend
print(f"\n{sep}")
print("BIO SECTOR 5-DAY TREND")
print(sep)
bio_tickers = [x["ticker"] for x in bio_stocks]
for day_offset in range(5, 0, -1):
    day_rets = []
    for t in bio_tickers[:80]:
        try:
            df = pd.read_parquet(f"data/raw/{t}.parquet")
            if len(df) >= day_offset + 1:
                c = df["close"].iloc[-day_offset]
                p = df["close"].iloc[-day_offset - 1]
                if p > 0:
                    day_rets.append((c / p - 1) * 100)
        except Exception:
            pass
    if day_rets:
        dt_idx = -day_offset
        try:
            sample_df = pd.read_parquet(f"data/raw/{bio_tickers[0]}.parquet")
            dt_str = str(sample_df.index[dt_idx])[:10]
        except Exception:
            dt_str = f"D-{day_offset}"
        avg = np.mean(day_rets)
        up = sum(1 for r in day_rets if r > 0) / len(day_rets) * 100
        bar = "+" * int(max(0, avg) * 5) + "-" * int(max(0, -avg) * 5)
        print(f"  {dt_str}: avg {avg:>+5.2f}% up {up:>3.0f}% {bar}")
