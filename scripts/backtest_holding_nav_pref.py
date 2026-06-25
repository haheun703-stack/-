"""scripts/backtest_holding_nav_pref.py — 지주사 NAV 신호 → 보통주 vs 우선주 비교.

가설(퐝가님): 지주사 우선주는 같은 NAV를 공유하지만 보통주보다 더 할인 + 고배당이라,
보통주 NAV 신호(G=할인축소 z≥+1 & NAVmom>0)가 뜰 때 보통주 대신 우선주를 사면 같거나
더 나은 반사이익 + 배당 보너스를 얻을 수 있다.

검증: 보통주 기준으로 NAV·할인율·G신호를 계산하고, 그 신호일에 보통주 D+60 수익률과
      우선주 D+60 수익률을 나란히 비교한다. (우선주 시총 불필요 — 종가 수익률만 사용)
★연구 전용. 배당수익률 차이(우선주 통상 1.5~2배)는 가격수익률에 미반영 → 우선주 추가 보너스.
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
FWD = 60

# 보통주 → 우선주 매핑 (parquet 보유분)
PREF = {
    "034730": ("03473K", "SK우"),
    "003550": ("003555", "LG우"),
    "028260": ("02826K", "삼성물산우"),
    "000880": ("00088K", "한화우"),
    "001040": ("001045", "CJ우"),
}


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

    print("=" * 92)
    print(f"NAV G신호 → 보통주 vs 우선주 D+{FWD} 비교  (G=할인축소 z≥+1 & NAVmom5d>0)")
    print("=" * 92)
    print(f"{'지주사':12s} | {'G표본':>5s} {'보통주승률':>9s}{'보통주평균':>9s} | {'우선주승률':>9s}{'우선주평균':>9s} | {'우선주우위':>9s}")
    print("-" * 92)

    pool_com, pool_pref = [], []
    for h in holdings:
        if h.ticker not in PREF or h.ticker not in mcap.columns:
            continue
        pref_tk, pref_nm = PREF[h.ticker]
        pref_close = load_close(pref_tk)
        if pref_close is None:
            continue

        stake = pd.Series(0.0, index=mcap.index)
        for s in h.listed_stakes:
            if s.ticker in mcap.columns:
                stake = stake.add(mcap[s.ticker] * (s.pct / 100.0), fill_value=0.0)
        fixed = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = stake + fixed
        df = pd.DataFrame({"nav": nav, "hold": mcap[h.ticker]}).dropna()
        df = df[df["nav"] > 0]
        df["pref"] = pref_close.reindex(df.index)

        disc = (df["hold"] - df["nav"]) / df["nav"]
        z = (disc - disc.rolling(ROLL).median()) / disc.rolling(ROLL).std()
        nav_m5 = df["nav"].pct_change(5)
        g = (z >= 1.0) & (nav_m5 > 0)

        fwd_com = df["hold"].shift(-FWD) / df["hold"] - 1
        fwd_pref = df["pref"].shift(-FWD) / df["pref"] - 1
        # 우선주 결측일 제외 위해 둘 다 존재하는 행만
        valid = g & df["pref"].notna()

        cn, cw, cm = wr(fwd_com[valid])
        pn, pw, pm = wr(fwd_pref[valid])
        edge = pm - cm
        print(f"{h.name+'/'+pref_nm:12s} | {cn:>5d} {cw:>8.1f}%{cm:>+8.2f}% | {pw:>8.1f}%{pm:>+8.2f}% | {edge:>+8.2f}%p")
        pool_com.append(fwd_com[valid].dropna())
        pool_pref.append(fwd_pref[valid].dropna())

    print("-" * 92)
    pc, pp = pd.concat(pool_com), pd.concat(pool_pref)
    print(f"{'POOLED':12s} | {len(pc):>5d} {(pc>0).mean()*100:>8.1f}%{pc.mean()*100:>+8.2f}% | "
          f"{(pp>0).mean()*100:>8.1f}%{pp.mean()*100:>+8.2f}% | {(pp.mean()-pc.mean())*100:>+8.2f}%p")
    print("=" * 92)
    print("해석: 우선주평균 ≥ 보통주평균이면, 같은 NAV신호에 우선주가 더 싸게 같은 반등 → 매력.")
    print("      여기에 우선주 고배당(통상 보통주 1.5~2배)은 미반영 → 실제 우위는 더 큼.")
    print("=" * 92)


if __name__ == "__main__":
    main()
