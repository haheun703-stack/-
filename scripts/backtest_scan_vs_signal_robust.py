"""B-18 적대적 검증 — 시총 매칭 기저선 + SCAN 등급별 분해.

정본(backtest_scan_vs_signal_top.py) 결론에 대한 반론 검증용. 짝으로 함께 돌린다.

정본 결론: SCAN 실진입은 유니버스 기저선 대비 D+5 -3.99 / D+10 -5.24 / D+20 -9.32%p (전부 p<0.05).
남은 반론: 기저선이 유니버스 전체인데 SCAN은 시총 중앙값 2.33조 중소형주다.
          열위의 일부(혹은 전부)가 종목 선택이 아니라 순수 시총(사이즈) 효과일 수 있다.

→ 각 종목을 시총 5분위로 나누고, **같은 분위·같은 날 평균**을 기저선으로 재측정한다.
   이래도 SCAN이 음수로 남으면 사이즈로 설명되지 않는 고유 결함이다.
   universe.csv 시총은 현재(7/20) 기준 근사 — 분위 순위는 단기간 크게 변하지 않으므로 허용,
   단 매칭 실패 종목은 제외하고 n을 병기한다.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
DATA = QM / "data"
HORIZONS = [5, 10, 20]


def fwd_ret(s: pd.Series, day: pd.Timestamp, n: int):
    pos = s.index.searchsorted(day)
    if pos >= len(s.index) or s.index[pos] != day or pos + n >= len(s.index):
        return None
    p0, p1 = float(s.iloc[pos]), float(s.iloc[pos + n])
    return None if p0 <= 0 else (p1 / p0 - 1.0) * 100.0


def main() -> None:
    prices = {}
    for f in glob.glob(str(DATA / "raw" / "*.parquet")):
        t = os.path.basename(f)[:-8]
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
            s.index = pd.to_datetime(s.index)
            prices[t] = s.sort_index()
        except Exception:
            continue

    uni = pd.read_csv(DATA / "universe.csv", dtype={"ticker": str})
    uni = uni[uni["ticker"].isin(prices)].copy()
    uni["q"] = pd.qcut(uni["market_cap"], 5, labels=[1, 2, 3, 4, 5])
    q_of = dict(zip(uni["ticker"], uni["q"]))
    by_q = defaultdict(list)
    for t, q in q_of.items():
        by_q[q].append(t)
    print("[시총 5분위] " + " / ".join(
        f"Q{q}: {len(by_q[q])}종목 중앙 {uni[uni['q'] == q]['market_cap'].median() / 1e12:.2f}조"
        for q in [1, 2, 3, 4, 5]))

    pf = json.loads((DATA / "paper_portfolio.json").read_text(encoding="utf-8"))
    entries = [(t["ticker"], t["entry_date"], t.get("grade", "?"))
               for t in pf.get("closed_trades", [])
               if t.get("strategy") == "SCAN" and t.get("entry_date")]
    entries += [(tk, p["entry_date"], p.get("grade", "?"))
                for tk, p in pf.get("positions", {}).items()
                if p.get("strategy") == "SCAN" and p.get("entry_date")]
    scan_by_date = defaultdict(list)
    for tk, d, _g in entries:
        scan_by_date[pd.Timestamp(d)].append(tk)

    signals = {}
    for f in glob.glob(str(DATA / "signals" / "signals_*.json")):
        signals[pd.Timestamp(os.path.basename(f)[8:16])] = json.loads(
            Path(f).read_text(encoding="utf-8")).get("picks", [])

    days = sorted(set(scan_by_date) & set(signals))

    # SCAN / TOP5 분위 분포
    a_tk = [t for d in days for t in scan_by_date[d]]
    b_tk = [p["ticker"] for d in days for p in signals[d]]
    for lab, tks in (("A SCAN", a_tk), ("B TOP5", b_tk)):
        c = pd.Series([q_of.get(t) for t in tks if t in q_of]).value_counts().sort_index()
        print(f"[분위분포] {lab}: " + " ".join(f"Q{q}={v}" for q, v in c.items())
              + f"  (매칭 {sum(c)}/{len(tks)})")
    print()

    res = {g: {n: [] for n in HORIZONS} for g in ("A_SCAN", "B_TOP5")}
    for d in days:
        for n in HORIZONS:
            # 분위별 기저선
            qbase = {}
            for q, tks in by_q.items():
                vals = [r for t in tks if (r := fwd_ret(prices[t], d, n)) is not None]
                if len(vals) >= 30:
                    qbase[q] = float(np.mean(vals))
            if not qbase:
                continue
            groups = {"A_SCAN": scan_by_date[d],
                      "B_TOP5": [p["ticker"] for p in signals[d]]}
            for g, tks in groups.items():
                for t in tks:
                    q = q_of.get(t)
                    if q is None or q not in qbase or t not in prices:
                        continue
                    r = fwd_ret(prices[t], d, n)
                    if r is not None:
                        res[g][n].append(r - qbase[q])

    print("=" * 88)
    print("★ 시총 분위 매칭 기저선 대비 초과수익 (사이즈 효과 제거)")
    print("=" * 88)
    for n in HORIZONS:
        print(f"\n── D+{n} ──")
        for g, lab in (("A_SCAN", "A  메인A 실진입 SCAN"), ("B_TOP5", "B  시그널 TOP5 전체")):
            v = res[g][n]
            if len(v) < 2:
                print(f"  {lab:22s} n={len(v)} 표본부족")
                continue
            a = np.array(v, float)
            t, p = stats.ttest_1samp(a, 0.0)
            print(f"  {lab:22s} 평균 {a.mean():+7.2f}%p  중앙값 {np.median(a):+7.2f}%p  "
                  f"양전 {100 * (a > 0).mean():5.1f}%  n={len(a):3d}  t={t:+5.2f} p={p:.4f}")

    # ── SCAN 등급별 분해 ──
    # 7/20 밤 부검은 KOSPI 알파로 "A등급 -0.12%p 중립 → 차단 이득 없음"이라 판정하고
    # B-17을 철회했다. 그 판정이 기저선 기준으로도 재현되는지 확인한다.
    print("\n" + "=" * 88)
    print("SCAN 등급별 — 유니버스 기저선 대비 초과수익 (7/20 'A등급 중립' 판정 재검증)")
    print("=" * 88)
    by_grade = defaultdict(lambda: {n: [] for n in HORIZONS})
    for tk, ds, g in entries:
        d = pd.Timestamp(ds)
        for n in HORIZONS:
            base = [r for t in prices if (r := fwd_ret(prices[t], d, n)) is not None]
            if len(base) < 100 or tk not in prices:
                continue
            r = fwd_ret(prices[tk], d, n)
            if r is not None:
                by_grade[g][n].append(r - float(np.mean(base)))
    for g in sorted(by_grade):
        print(f"\n[{g}]")
        for n in HORIZONS:
            v = by_grade[g][n]
            if len(v) < 2:
                print(f"   D+{n:<2d} n={len(v)} 표본부족")
                continue
            a = np.array(v, float)
            t, p = stats.ttest_1samp(a, 0.0)
            print(f"   D+{n:<2d} 평균 {a.mean():+7.2f}%p  중앙값 {np.median(a):+7.2f}%p  "
                  f"양전 {100 * (a > 0).mean():5.1f}%  n={len(a):3d}  t={t:+5.2f} p={p:.4f}")
    print("\n※ 등급별은 n=12~49로 작고 다중검정(2등급×3지평) 미보정 — 확정 근거로 쓰지 말 것")


if __name__ == "__main__":
    main()
