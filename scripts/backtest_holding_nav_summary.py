"""scripts/backtest_holding_nav_summary.py — 표본확대 일반화 검증 (8 지주사 종합).

각 지주사의 G신호(할인축소 z≥+1 & NAV모멘텀5d>0) D+60 성과를 한 줄로 요약하고,
가격모멘텀 대비 순수 부가가치(G한정=P제외)와 2025~26 시기집중도를 함께 본다.
마지막에 전 종목 신호를 합친 pooled 통계로 '일반화 여부'를 판정한다.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.use_cases.holding_nav import EOK, Holding  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
ROLL = 252
FWD = 60


def load_caps():
    caps = {}
    with open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                caps[str(r["ticker"]).zfill(6)] = float(r["market_cap"])
            except (ValueError, KeyError):
                continue
    return caps


def load_close(tk):
    p = os.path.join(PROC, f"{tk}.parquet")
    return pd.read_parquet(p, columns=["close"])["close"].astype(float) if os.path.exists(p) else None


def wr(s):
    s = s.dropna()
    return (len(s), (s > 0).mean() * 100 if len(s) else 0, s.mean() * 100 if len(s) else 0)


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
            closes[tk], shares[tk] = c, caps[tk] / c.iloc[-1]
    mcap = pd.DataFrame({tk: closes[tk] * shares[tk] for tk in closes}).sort_index()

    print("=" * 100)
    print(f"지주사 NAV G신호 일반화 검증 (D+{FWD})  G=할인축소(z≥+1)&NAVmom5d>0  G한정=가격mom20 미충족")
    print("=" * 100)
    hdr = f"{'지주사':10s}{'현z':>6s}{'벤치WR':>7s} | {'G표본':>5s}{'G승률':>7s}{'G평균':>8s} | {'G한정표본':>8s}{'G한정승률':>9s}{'G한정평균':>9s} | {'25~26집중':>9s}"
    print(hdr)
    print("-" * 100)

    pool_g, pool_gonly, pool_base = [], [], []
    for h in holdings:
        if h.ticker not in mcap.columns:
            continue
        stake = pd.Series(0.0, index=mcap.index)
        for s in h.listed_stakes:
            if s.ticker in mcap.columns:
                stake = stake.add(mcap[s.ticker] * (s.pct / 100.0), fill_value=0.0)
        fixed = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = stake + fixed
        df = pd.DataFrame({"nav": nav, "hold": mcap[h.ticker]}).dropna()
        df = df[df["nav"] > 0]
        if len(df) < ROLL + FWD:
            continue
        disc = (df["hold"] - df["nav"]) / df["nav"]
        z = (disc - disc.rolling(ROLL).median()) / disc.rolling(ROLL).std()
        nav_m5 = df["nav"].pct_change(5)
        price_m20 = df["hold"].pct_change(20)
        fwd = df["hold"].shift(-FWD) / df["hold"] - 1

        g = (z >= 1.0) & (nav_m5 > 0)
        gonly = g & ~(price_m20 > 0)
        base = z.notna()
        recent = g & (df.index.year >= 2025)
        conc = recent.sum() / max(g.sum(), 1) * 100

        gn, gw, gm = wr(fwd[g])
        on, ow, om = wr(fwd[gonly])
        bn, bw, bm = wr(fwd[base])
        znow = z.iloc[-1]
        print(f"{h.name:10s}{znow:>6.2f}{bw:>6.1f}% | {gn:>5d}{gw:>6.1f}%{gm:>+7.2f}% |"
              f" {on:>8d}{ow:>8.1f}%{om:>+8.2f}% | {conc:>8.0f}%")

        pool_g.append(fwd[g].dropna())
        pool_gonly.append(fwd[gonly].dropna())
        pool_base.append(fwd[base].dropna())

    print("-" * 100)
    pg, po, pb = pd.concat(pool_g), pd.concat(pool_gonly), pd.concat(pool_base)
    print(f"{'POOLED(전체)':10s}{'':>6s}{(pb>0).mean()*100:>6.1f}% | "
          f"{len(pg):>5d}{(pg>0).mean()*100:>6.1f}%{pg.mean()*100:>+7.2f}% | "
          f"{len(po):>8d}{(po>0).mean()*100:>8.1f}%{po.mean()*100:>+8.2f}% |")
    print("=" * 100)
    print("판정: POOLED G승률>벤치WR & G한정승률>50% → NAV신호 일반화/가격모멘텀 초과정보 인정.")
    print("      종목별 25~26집중이 대부분 70%+면 '재평가 테마 의존'(상시알파 아님).")
    print("=" * 100)


if __name__ == "__main__":
    main()
