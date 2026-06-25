"""scripts/backtest_holding_nav_validate.py — NAV 신호 적대적 2차 검증.

그리드에서 'C/G(할인축소+NAV모멘텀=모멘텀 방향)'가 강한 엣지를 보였다. 진짜인지 두 함정을 친다:
  (1) 시기 편중: 연도별로 신호일·평균수익을 쪼개 특정 강세장 착시인지 본다.
  (2) NAV 부가가치: 지주사 '자체 가격 20일 모멘텀'(NAV 불필요)과 비교해, NAV를 계산하는
      수고가 순수 가격모멘텀 대비 실제 우위를 주는지 검증. 우위 없으면 NAV 인프라는 과잉.
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


def summ(fwd, mask):
    sel = fwd[mask].dropna()
    if len(sel) < 10:
        return f"표본 {len(sel)} (부족)"
    return f"표본 {len(sel):4d} | 승률 {(sel>0).mean()*100:5.1f}% | 평균 {sel.mean()*100:+6.2f}% | 중앙 {sel.median()*100:+6.2f}%"


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

    print("=" * 78)
    print(f"NAV 신호 적대적 2차 검증 (D+{FWD})")
    print("=" * 78)

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
        price_m20 = df["hold"].pct_change(20)        # 순수 가격모멘텀(NAV 불필요)
        fwd = df["hold"].shift(-FWD) / df["hold"] - 1

        sig_G = (z >= 1.0) & (nav_m5 > 0)            # 할인축소+NAV모멘텀
        sig_P = price_m20 > 0                         # 순수 가격 20일 모멘텀
        sig_GP = sig_G & sig_P
        base = z.notna()

        print(f"\n■ {h.name} ({h.ticker})")
        print(f"  [NAV 부가가치 비교]")
        print(f"    벤치마크         : {summ(fwd, base)}")
        print(f"    P 순수가격mom20  : {summ(fwd, sig_P)}")
        print(f"    G 할인축소+NAVmom: {summ(fwd, sig_G)}")
        print(f"    G∩P 교집합       : {summ(fwd, sig_GP)}")
        # G가 P보다 나은가? P에 없고 G에만 있는 날 vs 그 반대
        print(f"    G한정(P제외)     : {summ(fwd, sig_G & ~sig_P)}")
        print(f"    P한정(G제외)     : {summ(fwd, sig_P & ~sig_G)}")

        print(f"  [시기 편중 — G 신호 연도별]")
        yrs = df.index.year
        g_fwd = fwd[sig_G].dropna()
        for y in sorted(set(yrs)):
            ym = sig_G & (df.index.year == y)
            sub = fwd[ym].dropna()
            if len(sub) > 0:
                print(f"    {y} | 신호 {len(sub):3d}일 | 승률 {(sub>0).mean()*100:5.1f}% | 평균 {sub.mean()*100:+6.2f}%")

    print("\n" + "=" * 78)
    print("판정: G한정(P제외)이 양호하면 NAV가 가격모멘텀에 없는 정보 추가. 특정연도 편중이면 신뢰↓.")
    print("=" * 78)


if __name__ == "__main__":
    main()
