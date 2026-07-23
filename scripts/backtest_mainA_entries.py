"""메인A 진입 전체(라벨 무관) vs 기저선 — "SCAN이 나쁜가, 진입 자체가 나쁜가".

배경(7/23): SCAN 진입 중단을 실제로 적용해봤더니 후보가 232→230건(-2)에 그쳤다.
  현행 SCAN 10종목 중 8종목이 ALPHA(5)·SWING(3) 라벨로 그대로 생환했기 때문이다.
  SCAN·ALPHA·SWING은 같은 tomorrow_picks 풀에서 나온 형제이고, 라벨은 어느 섹션이
  먼저 집었느냐에 불과하다.

→ 그렇다면 "SCAN 전략이 나쁘다"는 결론 자체가 라벨 아티팩트일 수 있다.
   전략 라벨을 지우고 **메인A가 진입한 모든 거래**를 같은 잣대로 다시 잰다.

판정 기준: 같은 날 유니버스 동일가중 평균(기저선) 대비 초과수익.
  (KOSPI 차감은 이 국면에서 판별력 0 — 유니버스 동일가중조차 KOSPI 대비 D+20 -9.9%p)

읽는 법:
  · 전 전략이 고르게 음수  → 문제는 전략이 아니라 **진입 행위/종목 풀**.
    처방은 "어느 라벨을 끌까"가 아니라 진입 수 축소 또는 풀 교체.
  · SCAN만 음수·나머지 중립 → 라벨이 실제 의미를 가지므로 SCAN 차단이 유효.
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


def fwd(s: pd.Series, d: pd.Timestamp, n: int):
    p = s.index.searchsorted(d)
    if p >= len(s.index) or s.index[p] != d or p + n >= len(s.index):
        return None
    a, b = float(s.iloc[p]), float(s.iloc[p + n])
    return None if a <= 0 else (b / a - 1.0) * 100.0


def line(vals: list[float]) -> str:
    if len(vals) < 2:
        return f"n={len(vals)} (표본부족)"
    a = np.array(vals, float)
    t, p = stats.ttest_1samp(a, 0.0)
    star = " ★" if p < 0.05 else ""
    return (f"평균 {a.mean():+7.2f}%p  중앙값 {np.median(a):+7.2f}%p  "
            f"양전 {100 * (a > 0).mean():5.1f}%  n={len(a):3d}  t={t:+5.2f} p={p:.4f}{star}")


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
    print(f"[load] 종목 {len(prices)}개")

    pf = json.loads((DATA / "paper_portfolio.json").read_text(encoding="utf-8"))
    rows = [(t["ticker"], t["entry_date"], t.get("strategy", "?"), t.get("grade", "?"))
            for t in pf.get("closed_trades", []) if t.get("entry_date")]
    rows += [(tk, p["entry_date"], p.get("strategy", "?"), p.get("grade", "?"))
             for tk, p in pf.get("positions", {}).items() if p.get("entry_date")]
    print(f"[표본] 메인A 진입 {len(rows)}건 "
          f"({', '.join(f'{k}={v}' for k, v in sorted(pd.Series([r[2] for r in rows]).value_counts().items()))})")

    # 기저선 캐시 (날짜×지평)
    base_cache: dict[tuple, float] = {}
    def baseline(d: pd.Timestamp, n: int):
        key = (d, n)
        if key not in base_cache:
            vals = [r for t in prices if (r := fwd(prices[t], d, n)) is not None]
            base_cache[key] = float(np.mean(vals)) if len(vals) >= 100 else None
        return base_cache[key]

    by_strategy = defaultdict(lambda: {n: [] for n in HORIZONS})
    overall = {n: [] for n in HORIZONS}
    for tk, ds, strat, _g in rows:
        if tk not in prices:
            continue
        d = pd.Timestamp(ds)
        for n in HORIZONS:
            b = baseline(d, n)
            r = fwd(prices[tk], d, n)
            if b is None or r is None:
                continue
            by_strategy[strat][n].append(r - b)
            overall[n].append(r - b)

    print("\n" + "=" * 92)
    print("★ 메인A 진입 전체 (라벨 무관) — 기저선 대비 초과수익")
    print("=" * 92)
    for n in HORIZONS:
        print(f"  D+{n:<2d} {line(overall[n])}")

    print("\n" + "=" * 92)
    print("전략 라벨별 분해 — 라벨이 실제 의미를 갖는지 확인")
    print("=" * 92)
    for strat in sorted(by_strategy, key=lambda s: -len(by_strategy[s][5])):
        print(f"\n[{strat}]")
        for n in HORIZONS:
            print(f"   D+{n:<2d} {line(by_strategy[strat][n])}")

    # 라벨 간 차이가 유의한가 (SCAN vs 나머지)
    print("\n" + "=" * 92)
    print("SCAN vs 비-SCAN (Welch) — 유의하지 않으면 라벨은 아티팩트")
    print("=" * 92)
    for n in HORIZONS:
        a = by_strategy.get("SCAN", {}).get(n, [])
        b = [v for s, d in by_strategy.items() if s != "SCAN" for v in d[n]]
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            verdict = "라벨 유의" if p < 0.05 else "차이 없음 → 라벨 아티팩트"
            print(f"  D+{n:<2d} SCAN {np.mean(a):+6.2f}%p (n={len(a)}) vs 비SCAN {np.mean(b):+6.2f}%p "
                  f"(n={len(b)})  t={t:+5.2f} p={p:.4f}  → {verdict}")


if __name__ == "__main__":
    main()
