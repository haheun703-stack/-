"""scripts/backtest_holding_nav_grid.py — NAV 디스카운트 신호 그리드 스캔 (적대적 검증).

1차 가설(할인확대+NAV모멘텀)이 기각돼, 신호 '방향'이 틀렸는지 확인하기 위해 8개 신호 ×
2개 보유기간을 그리드로 돌려 어디에 엣지가 있는지(없는지) 한눈에 본다. 오버피팅 경계 —
파라미터 최적화가 목적이 아니라 '방향성이 존재하는가' 판정용.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.use_cases.holding_nav import EOK, Holding  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
ROLL = 252
FWDS = [20, 60]


def load_caps():
    caps = {}
    with open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                caps[str(row["ticker"]).zfill(6)] = float(row["market_cap"])
            except (ValueError, KeyError):
                continue
    return caps


def load_close(tk):
    p = os.path.join(PROC, f"{tk}.parquet")
    return pd.read_parquet(p, columns=["close"])["close"].astype(float) if os.path.exists(p) else None


def row(fwd, mask, label):
    sel = fwd[mask].dropna()
    if len(sel) < 10:
        return f"  {label:26s} | 표본 {len(sel):4d}  (부족)"
    return (f"  {label:26s} | 표본 {len(sel):4d} | 승률 {(sel > 0).mean() * 100:5.1f}% | "
            f"평균 {sel.mean() * 100:+6.2f}% | 중앙 {sel.median() * 100:+6.2f}%")


def main():
    with open(os.path.join(ROOT, "config", "holding_nav.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    holdings = [Holding.from_dict(tk, d) for tk, d in cfg["holdings"].items()]
    caps = load_caps()

    need = set()
    for h in holdings:
        need.add(h.ticker)
        need.update(s.ticker for s in h.listed_stakes)
    closes, shares = {}, {}
    for tk in need:
        c = load_close(tk)
        if c is not None and tk in caps and c.iloc[-1] > 0:
            closes[tk] = c
            shares[tk] = caps[tk] / c.iloc[-1]
    mcap = pd.DataFrame({tk: closes[tk] * shares[tk] for tk in closes}).sort_index()

    print("=" * 80)
    print(f"NAV 디스카운트 신호 그리드 (적대적)  기간 {mcap.index.min().date()}~{mcap.index.max().date()}")
    print("=" * 80)

    for h in holdings:
        stake = pd.Series(0.0, index=mcap.index)
        for s in h.listed_stakes:
            if s.ticker in mcap.columns:
                stake = stake.add(mcap[s.ticker] * (s.pct / 100.0), fill_value=0.0)
        fixed = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = stake + fixed
        df = pd.DataFrame({"nav": nav, "hold": mcap[h.ticker]}).dropna()
        df = df[df["nav"] > 0]

        disc = (df["hold"] - df["nav"]) / df["nav"]
        z = (disc - disc.rolling(ROLL).median()) / disc.rolling(ROLL).std()
        nav_m5 = df["nav"].pct_change(5)
        nav_m20 = df["nav"].pct_change(20)
        disc_rank = disc.rolling(ROLL).rank(pct=True)  # 0=역대최저할인폭(가장쌈) ... 1=가장비쌈

        signals = {
            "전체(벤치마크)":              z.notna(),
            "A 할인확대 z≤-1":             z <= -1.0,
            "C 할인축소 z≥+1":             z >= 1.0,
            "D NAV모멘텀5d>0(할인무관)":    nav_m5 > 0,
            "E NAV모멘텀20d>0":           nav_m20 > 0,
            "F 절대최저할인 rank≤0.2":      disc_rank <= 0.2,
            "B 할인확대+NAVmom":           (z <= -1.0) & (nav_m5 > 0),
            "G 할인축소+NAVmom(추격)":      (z >= 1.0) & (nav_m5 > 0),
        }
        fwd = {n: df["hold"].shift(-n) / df["hold"] - 1 for n in FWDS}

        print(f"\n■ {h.name} ({h.ticker})  현 z {z.iloc[-1]:+.2f} / 할인 {disc.iloc[-1]*100:+.1f}%")
        for n in FWDS:
            print(f"  ── 포워드 D+{n} ──")
            for lbl, m in signals.items():
                print(row(fwd[n], m, lbl))

    print("\n" + "=" * 80)
    print("해석: 벤치마크 대비 승률+평균 동시 우위 신호만 엣지. 없으면 NAV할인은 매매신호 부적합.")
    print("=" * 80)


if __name__ == "__main__":
    main()
