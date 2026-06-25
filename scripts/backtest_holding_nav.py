"""scripts/backtest_holding_nav.py — 지주사 NAV 디스카운트 반사이익 백테스트 (POC).

가설: NAV(보유자산 가치)는 오르는데(NAV 5일 모멘텀 > 0) 지주사 주가는 역사적 할인밴드보다
      더 싸게 거래(할인율 z ≤ −1)되면, 곧 자산을 따라 반등한다(반사이익).

검증: 2019~2026 일봉으로 일별 NAV·할인율을 재구성하고, 신호일의 포워드 수익률(D+10/D+20)을
      무신호 전체평균과 비교한다. 밸류트랩(할인 영구화) 회피의 핵심 — 엣지 없으면 기각.

데이터: data/processed/{ticker}.parquet (종가). 발행주식수 = universe.csv 현재시총 / 최근종가
        (대형주 증자/감자 거의 없어 고정 가정). 비상장·자체사업·순부채는 현재값 고정(POC 한계).
★관측/연구 전용 — 실주문 0.
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

ROLL = 252       # 할인율 z-score 롤링창 (≈1년)
Z_THR = -1.0     # 할인확대 임계 (역사평균보다 1σ 더 싸다)
NAV_MOM_DAYS = 5
FWDS = [10, 20]  # 포워드 수익률 측정일


def load_caps() -> dict[str, float]:
    caps: dict[str, float] = {}
    with open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                caps[str(row["ticker"]).zfill(6)] = float(row["market_cap"])
            except (ValueError, KeyError):
                continue
    return caps


def load_close(ticker: str) -> pd.Series | None:
    p = os.path.join(PROC, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p, columns=["close"])["close"].astype(float)


def stats(fwd: pd.Series, mask: pd.Series, label: str) -> str:
    sel = fwd[mask].dropna()
    if len(sel) == 0:
        return f"  {label:24s} | 표본 0"
    win = (sel > 0).mean() * 100
    return (f"  {label:24s} | 표본 {len(sel):4d} | 승률 {win:5.1f}% | "
            f"평균 {sel.mean() * 100:+5.2f}% | 중앙 {sel.median() * 100:+5.2f}%")


def main() -> None:
    with open(os.path.join(ROOT, "config", "holding_nav.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    holdings = [Holding.from_dict(tk, d) for tk, d in cfg.get("holdings", {}).items()]
    caps = load_caps()

    # 필요한 모든 티커 종가 + 주식수 역산
    need: set[str] = set()
    for h in holdings:
        need.add(h.ticker)
        need.update(s.ticker for s in h.listed_stakes)
    closes, shares, dropped = {}, {}, []
    for tk in need:
        c = load_close(tk)
        if c is None or tk not in caps or c.iloc[-1] <= 0:
            dropped.append(tk)
            continue
        closes[tk] = c
        shares[tk] = caps[tk] / c.iloc[-1]   # 발행주식수 ≈ 현재시총/최근종가

    mcap = pd.DataFrame({tk: closes[tk] * shares[tk] for tk in closes}).sort_index()

    print("=" * 72)
    print(f"지주사 NAV 디스카운트 반사이익 백테스트 (POC)  | 롤z{ROLL} z≤{Z_THR} NAVmom{NAV_MOM_DAYS}d")
    print(f"기간 {mcap.index.min().date()} ~ {mcap.index.max().date()} | 결측티커 {dropped or '없음'}")
    print("=" * 72)

    for h in holdings:
        if h.ticker not in mcap.columns:
            print(f"\n■ {h.name}: 지주사 종가 결측 → 스킵")
            continue
        # 일별 NAV = Σ(자회사 시총 × 지분율) + (비상장+자체사업−순부채) 고정
        stake = pd.Series(0.0, index=mcap.index)
        used, miss = [], []
        for s in h.listed_stakes:
            if s.ticker in mcap.columns:
                stake = stake.add(mcap[s.ticker] * (s.pct / 100.0), fill_value=0.0)
                used.append(s.name)
            else:
                miss.append(s.name)
        fixed = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = stake + fixed
        hold_cap = mcap[h.ticker]
        df = pd.DataFrame({"nav": nav, "hold": hold_cap}).dropna()
        df = df[df["nav"] > 0]

        disc = (df["hold"] - df["nav"]) / df["nav"]               # 음수=할인
        med = disc.rolling(ROLL).median()
        std = disc.rolling(ROLL).std()
        z = (disc - med) / std
        nav_mom = df["nav"].pct_change(NAV_MOM_DAYS)

        fwd = {n: df["hold"].shift(-n) / df["hold"] - 1 for n in FWDS}

        sig_wide = z <= Z_THR                       # A: 할인확대만
        sig_full = (z <= Z_THR) & (nav_mom > 0)     # B: 할인확대 + NAV 모멘텀(반사이익 가설)
        base = z.notna()                            # 전체(벤치마크)

        print(f"\n■ {h.name} ({h.ticker}) [{h.kind}]  현 할인율 {disc.iloc[-1] * 100:+.1f}%"
              f"  z {z.iloc[-1]:+.2f}")
        print(f"  NAV 구성: {', '.join(used)}" + (f"  (결측:{','.join(miss)})" if miss else ""))
        for n in FWDS:
            print(f"  — 포워드 D+{n} —")
            print(stats(fwd[n], base, "전체(벤치마크)"))
            print(stats(fwd[n], sig_wide, f"A 할인확대(z≤{Z_THR})"))
            print(stats(fwd[n], sig_full, "B 할인확대+NAV모멘텀★"))

    print("\n" + "=" * 72)
    print("판정 기준: B(가설)가 전체 벤치마크보다 승률·평균 모두 유의하게 높아야 엣지 인정.")
    print("표본 부족(<30)이면 신뢰 낮음. 비상장/순부채 고정은 POC 한계(자회사 변동만 반영).")
    print("=" * 72)


if __name__ == "__main__":
    main()
