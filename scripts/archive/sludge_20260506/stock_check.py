"""현대차/한국단자 4/15 기준 D+1 예측 분석"""
import pandas as pd
import numpy as np
import json
import os

name_map = json.load(open("data/universe/name_map.json"))

# 한국단자 ticker 찾기
hkd_ticker = None
for k, v in name_map.items():
    if "한국단자" in v:
        hkd_ticker = k
        print(f"한국단자 ticker: {k} = {v}")

targets = [("005380", "현대차")]
if hkd_ticker:
    targets.append((hkd_ticker, "한국단자"))
else:
    print("한국단자 not found in name_map, trying 025540")
    targets.append(("025540", "한국단자"))

sep70 = "=" * 70

for ticker, name in targets:
    f = f"data/raw/{ticker}.parquet"
    if not os.path.exists(f):
        print(f"\n{name} ({ticker}): parquet not found!")
        continue

    df = pd.read_parquet(f)
    df = df[df["close"] > 0].copy()
    if len(df) < 60:
        print(f"\n{name}: data too short ({len(df)} rows)")
        continue

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_date = str(df.index[-1])[:10]

    close = int(last["close"])
    high = int(last["high"])
    low = int(last["low"])
    opn = int(last["open"])
    vol = int(last["volume"])
    rng = high - low

    close_pos = (close - low) / rng * 100 if rng > 0 else 50
    body_pct = (close - opn) / opn * 100 if opn > 0 else 0
    upper_wick = high - max(close, opn)
    upper_wick_pct = upper_wick / close * 100
    lower_wick = min(close, opn) - low
    lower_wick_pct = lower_wick / close * 100
    day_ret = (close / prev["close"] - 1) * 100

    vol_20 = df["volume"].tail(20).mean()
    vol_ratio = vol / vol_20 if vol_20 > 0 else 0

    has_supply = "기관합계" in df.columns
    inst = int(last.get("기관합계", 0)) if has_supply else 0
    frgn = int(last.get("외국인합계", 0)) if has_supply else 0
    indv = int(last.get("개인", 0)) if has_supply else 0

    inst_5d = int(df["기관합계"].tail(5).sum()) if has_supply else 0
    frgn_5d = int(df["외국인합계"].tail(5).sum()) if has_supply else 0
    inst_20d = int(df["기관합계"].tail(20).sum()) if has_supply else 0
    frgn_20d = int(df["외국인합계"].tail(20).sum()) if has_supply else 0

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss_s
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    ma5 = df["close"].tail(5).mean()
    ma20 = df["close"].tail(20).mean()
    ma60 = df["close"].tail(60).mean()
    ma120 = df["close"].tail(120).mean()

    r250 = df.tail(250)
    high_52w = int(r250["high"].max())
    low_52w = int(r250["low"].min())
    fib_diff = high_52w - low_52w
    fib_pos = (close - low_52w) / fib_diff * 100 if fib_diff > 0 else 50

    is_strong_close = close_pos >= 80
    is_dual_buy = inst > 0 and frgn > 0
    is_vol_up = vol_ratio >= 1.2
    pattern_match = is_strong_close and is_dual_buy and is_vol_up and day_ret >= 2

    # Historical similar pattern for this stock
    df["_rng"] = df["high"] - df["low"]
    df["_cp"] = np.where(df["_rng"] > 0, (df["close"] - df["low"]) / df["_rng"], 0.5)
    df["_dr"] = df["close"].pct_change() * 100
    df["_vr"] = df["volume"] / df["volume"].rolling(20).mean()
    df["_d1"] = (df["close"].shift(-1) / df["close"] - 1) * 100
    df["_gap"] = (df["open"].shift(-1) / df["close"] - 1) * 100
    df["_otc"] = (df["close"].shift(-1) / df["open"].shift(-1) - 1) * 100
    df["_d5"] = (df["close"].shift(-5) / df["close"] - 1) * 100

    # Similar condition: close position, return range, volume
    sim_mask = (
        (df["_cp"] >= 0.70) &
        (df["_dr"] >= max(day_ret * 0.5, 1.0)) &
        (df["_dr"] <= day_ret * 2.0) &
        (df["_vr"] >= vol_ratio * 0.5) &
        (df["_d1"].notna())
    )
    similar = df[sim_mask]

    # Also get: strong close specifically
    sc_mask = (
        (df["_cp"] >= 0.80) &
        (df["_dr"] >= 2.0) &
        (df["_vr"] >= 1.0) &
        (df["_d1"].notna())
    )
    strong_days = df[sc_mask]

    print(f"\n{sep70}")
    print(f"  {name} ({ticker}) -- {last_date}")
    print(sep70)
    print(f"  종가: {close:,} | 시가: {opn:,} | 고가: {high:,} | 저가: {low:,}")
    print(f"  등락: {day_ret:+.1f}% | 거래량: {vol:,} ({vol_ratio:.1f}x)")
    print()

    # Candle
    candle = "STRONG CLOSE" if close_pos >= 80 and body_pct > 0 else \
             "UPPER WICK" if upper_wick_pct >= 2 else \
             "WEAK CLOSE" if close_pos <= 30 else "NEUTRAL"
    print(f"  [candle] 종가위치: {close_pos:.0f}% | 몸통: {body_pct:+.1f}% | 윗꼬리: {upper_wick_pct:.1f}% | 아랫꼬리: {lower_wick_pct:.1f}%")
    print(f"  type: {candle}")
    print()

    # Supply
    dual_str = "DUAL BUY" if is_dual_buy else \
               ("FRGN only" if frgn > 0 else ("INST only" if inst > 0 else "SELL"))
    print(f"  [supply] 기관: {inst/1e8:+,.0f}억 | 외인: {frgn/1e8:+,.0f}억 | 개인: {indv/1e8:+,.0f}억 => {dual_str}")
    print(f"  5d  기관: {inst_5d/1e8:+,.0f}억 | 외인: {frgn_5d/1e8:+,.0f}억")
    print(f"  20d 기관: {inst_20d/1e8:+,.0f}억 | 외인: {frgn_20d/1e8:+,.0f}억")
    print()

    # Technical
    print(f"  [tech] RSI: {rsi:.1f} | MA20: {int(ma20):,}({(close/ma20-1)*100:+.1f}%) | MA60: {int(ma60):,}({(close/ma60-1)*100:+.1f}%)")
    print(f"  52w: {low_52w:,}~{high_52w:,} | FIB: {fib_pos:.1f}%")

    # Fib levels near current price
    for pct, label in [(0.382, "38.2%"), (0.5, "50%"), (0.618, "61.8%"), (0.786, "78.6%")]:
        level = low_52w + fib_diff * pct
        if abs(close - level) / close < 0.05:
            print(f"  ** 피보나치 {label} = {int(level):,} (현재가 근접!)")
    print()

    # Pattern check
    print(f"  [PATTERN CHECK: strong close + dual buy]")
    checks = [
        ("close >= 80%", is_strong_close, f"{close_pos:.0f}%"),
        ("dual buy", is_dual_buy, dual_str),
        ("vol >= 1.2x", is_vol_up, f"{vol_ratio:.1f}x"),
        ("day >= +2%", day_ret >= 2, f"{day_ret:+.1f}%"),
    ]
    all_pass = True
    for label, passed, val in checks:
        mark = "O" if passed else "X"
        print(f"    [{mark}] {label}: {val}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"  >>> PATTERN MATCH! overnight hold -> sell at open recommended")
    else:
        print(f"  >>> PATTERN NOT MATCHED")
    print()

    # Historical stats
    if len(similar) > 3:
        d1s = similar["_d1"].dropna()
        gaps = similar["_gap"].dropna()
        otcs = similar["_otc"].dropna()
        d5s = similar["_d5"].dropna()
        print(f"  [HISTORY: {name} similar days] n={len(d1s)}")
        print(f"  D+1 gap: {gaps.mean():+.2f}% | intraday: {otcs.mean():+.2f}% | total: {d1s.mean():+.2f}%")
        print(f"  D+1 WR: {(d1s>0).mean()*100:.0f}% | 5%+: {(d1s>=5).mean()*100:.1f}%")
        print(f"  D+5 avg: {d5s.mean():+.2f}% | WR: {(d5s>0).mean()*100:.0f}%")
        fade = (otcs < 0).mean() * 100
        crash = (otcs < -3).mean() * 100
        print(f"  Fade(intraday-): {fade:.0f}% | Crash(-3%+): {crash:.0f}%")

    if len(strong_days) > 3:
        d1s2 = strong_days["_d1"].dropna()
        gaps2 = strong_days["_gap"].dropna()
        otcs2 = strong_days["_otc"].dropna()
        print(f"\n  [HISTORY: {name} all strong-close days] n={len(d1s2)}")
        print(f"  D+1 gap: {gaps2.mean():+.2f}% | intraday: {otcs2.mean():+.2f}% | total: {d1s2.mean():+.2f}%")
        print(f"  D+1 WR: {(d1s2>0).mean()*100:.0f}%")

    # Recent 5 days
    print(f"\n  [recent 5 days]")
    for idx, row in df.tail(5).iterrows():
        dt = str(idx)[:10]
        c = int(row["close"])
        cp = row.get("_cp", 0) * 100
        dr = row.get("_dr", 0)
        vr = row.get("_vr", 0)
        i_e = row.get("기관합계", 0) / 1e8 if has_supply else 0
        f_e = row.get("외국인합계", 0) / 1e8 if has_supply else 0
        print(f"    {dt}: {c:>8,} {dr:>+5.1f}% cp={cp:>3.0f}% vol={vr:.1f}x | I{i_e:>+5.0f}억 F{f_e:>+5.0f}억")

    print()
