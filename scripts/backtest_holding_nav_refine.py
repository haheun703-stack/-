"""scripts/backtest_holding_nav_refine.py — NAV 신호 정련 (페이퍼 배선 전제조건).

현재 신호 점검에서 드러난 2결함을 고친 '현실적 진입 규칙'을 재측정한다:
  (1) z 상한: z≥+1만 보면 SK z+3.13처럼 '할인 거의 다 좁혀진 과열'도 잡혀 추격매수.
      → z 밴드 [zlo, zhi]로 과열 제외.
  (2) 할증종목 필터: CJ 할인율 +104%(NAV<시총=데이터버그/사업지주 과소평가) 제외.
      → discount < 0(실제 할인거래)인 날만 신호 허용.
  (3) 표본중첩 보정: D+60 신호가 연속일 겹쳐 독립 거래기회를 과대계상.
      → 진입 후 COOLDOWN일 내 재신호 무시(독립 진입만 카운트).

순수지주·검증통과 종목(㈜LG·SK·두산·삼성물산) 중심. 한화·SK스퀘어(부적합)·CJ(버그) 제외.
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
COOLDOWN = 20  # 진입 후 재신호 무시일(독립 거래기회만)
INCLUDE = {"003550", "034730", "000150", "028260"}  # 검증통과 종목만
Z_BANDS = [(1.0, 99.0), (1.0, 2.0), (1.0, 2.5), (0.5, 2.0)]


def load_caps():
    caps = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "universe.csv"), encoding="utf-8")):
        try:
            caps[str(r["ticker"]).zfill(6)] = float(r["market_cap"])
        except (ValueError, KeyError):
            pass
    return caps


def load_close(tk):
    p = os.path.join(PROC, f"{tk}.parquet")
    return pd.read_parquet(p, columns=["close"])["close"].astype(float) if os.path.exists(p) else None


def dedup_signals(sig: pd.Series, cooldown: int) -> pd.Series:
    """연속 신호 클러스터에서 독립 진입만 남긴다(직전 진입 +cooldown 내 재신호 제거)."""
    out = pd.Series(False, index=sig.index)
    last = -10 ** 9
    pos = {ts: i for i, ts in enumerate(sig.index)}
    for ts in sig.index[sig.values]:
        i = pos[ts]
        if i - last >= cooldown:
            out.iloc[i] = True
            last = i
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "holding_nav.yaml"), encoding="utf-8"))
    holdings = [Holding.from_dict(tk, d) for tk, d in cfg["holdings"].items() if h_in(tk)]
    caps = load_caps()
    need = set()
    for h in holdings:
        need.add(h.ticker)
        need.update(s.ticker for s in h.listed_stakes)
    cl, sh = {}, {}
    for tk in need:
        c = load_close(tk)
        if c is not None and tk in caps and c.iloc[-1] > 0:
            cl[tk], sh[tk] = c, caps[tk] / c.iloc[-1]
    mc = pd.DataFrame({tk: cl[tk] * sh[tk] for tk in cl}).sort_index()

    # 종목별 (z, nav_mom, disc, fwd) 미리 계산
    per = {}
    for h in holdings:
        st = pd.Series(0.0, index=mc.index)
        for s in h.listed_stakes:
            if s.ticker in mc.columns:
                st = st.add(mc[s.ticker] * (s.pct / 100), fill_value=0.0)
        fx = (h.other_nav_eok + h.own_business_eok - h.net_debt_eok) * EOK
        nav = st + fx
        df = pd.DataFrame({"nav": nav, "hold": mc[h.ticker]}).dropna()
        df = df[df["nav"] > 0]
        disc = (df["hold"] - df["nav"]) / df["nav"]
        z = (disc - disc.rolling(ROLL).median()) / disc.rolling(ROLL).std()
        nm = df["nav"].pct_change(5)
        fwd = df["hold"].shift(-FWD) / df["hold"] - 1
        per[h.name] = (z, nm, disc, fwd)

    print("=" * 84)
    print(f"NAV 신호 정련 백테스트 (D+{FWD}, 쿨다운 {COOLDOWN}일, 할인거래限, 검증통과 4종목)")
    print("=" * 84)
    print(f"{'z밴드':14s} | {'진입수':>6s} {'승률':>7s} {'평균':>8s} {'중앙':>8s} {'최악':>8s}")
    print("-" * 84)
    for zlo, zhi in Z_BANDS:
        all_fwd = []
        for name, (z, nm, disc, fwd) in per.items():
            raw = (z >= zlo) & (z <= zhi) & (nm > 0) & (disc < 0)  # 할인거래 + z밴드 + NAVmom
            ent = dedup_signals(raw.fillna(False), COOLDOWN)
            all_fwd.append(fwd[ent].dropna())
        s = pd.concat(all_fwd)
        band = f"[{zlo:.1f},{zhi:.0f}]" if zhi > 90 else f"[{zlo:.1f},{zhi:.1f}]"
        if len(s):
            print(f"{band:14s} | {len(s):>6d} {(s>0).mean()*100:>6.1f}% "
                  f"{s.mean()*100:>+7.2f}% {s.median()*100:>+7.2f}% {s.min()*100:>+7.1f}%")
        else:
            print(f"{band:14s} | 표본 0")
    print("-" * 84)
    print("해석: 과열(높은 z) 제외하면 표본↓ 품질↑가 정상. '최악'(최대낙폭 진입)이 견딜만해야 실전가치.")
    print("=" * 84)


def h_in(tk):
    return str(tk).zfill(6) in INCLUDE


if __name__ == "__main__":
    main()
