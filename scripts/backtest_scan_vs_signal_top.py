"""B-18 처방 검증 (정본) — 메인A SCAN 실진입 종목군 vs 같은 날 시그널 엔진 TOP5.

질문: "SCAN 진입 필터를 외국인 유입 대형주 쪽으로 좁히면 상승장 미참여가 줄어드는가?"
적대적 검증(시총 매칭·등급 분해)은 backtest_scan_vs_signal_robust.py 참조.

★초안(KOSPI 차감 알파 기준)에서 드러난 문제 — 같은 실수를 반복하지 않기 위해 남긴다:
  ① KOSPI 차감 알파가 이 국면에선 부적절 — 유니버스 동일가중 전체조차 KOSPI 대비 D+20 -9.9%p.
     시총 최상위 주도장이라 개별종목은 구조적으로 지수에 미달한다. 지수를 잣대로 쓰면
     "모든 개별주 전략이 음수"라는 무의미한 결론만 나온다.
     → ★판정 기준을 **같은 날 유니버스 동일가중 평균(기저선) 대비 초과수익**으로 교체.
  ② FOREIGN_SURGE_PB 태그는 6/2부터만 존재(14일) → 5월 구간 부재. 기간 편향 명시.
  ③ BF 표본 n=4~11 → 단독 결론 금지, 표본수 병기.

추가:
  · 날짜 페어링 — 같은 날 A(SCAN 실진입)와 B(시그널 TOP5)를 짝지어 격차를 재면
    시장 국면 효과가 상쇄된다(풀링 비교의 교란 제거).
  · Welch t검정 + 페어드 t검정.
  · A/B 종목 중복 여부 점검.
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


def load_prices() -> dict[str, pd.Series]:
    prices = {}
    for f in glob.glob(str(DATA / "raw" / "*.parquet")):
        ticker = os.path.basename(f)[:-8]
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
            s.index = pd.to_datetime(s.index)
            prices[ticker] = s.sort_index()
        except Exception:
            continue
    return prices


def fwd_ret(s: pd.Series, day: pd.Timestamp, n: int):
    pos = s.index.searchsorted(day)
    if pos >= len(s.index) or s.index[pos] != day:
        return None
    if pos + n >= len(s.index):
        return None
    p0, p1 = float(s.iloc[pos]), float(s.iloc[pos + n])
    return None if p0 <= 0 else (p1 / p0 - 1.0) * 100.0


def stat_line(vals: list[float]) -> str:
    if len(vals) < 2:
        return f"n={len(vals)} (표본부족)"
    a = np.array(vals, float)
    t, p = stats.ttest_1samp(a, 0.0)
    return (f"평균 {a.mean():+7.2f}%p  중앙값 {np.median(a):+7.2f}%p  "
            f"양전 {100 * (a > 0).mean():5.1f}%  n={len(a):4d}  t={t:+5.2f} p={p:.3f}")


def main() -> None:
    prices = load_prices()
    kospi = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()
    print(f"[load] 종목 {len(prices)}개 · KOSPI {len(kospi)}행\n")

    pf = json.loads((DATA / "paper_portfolio.json").read_text(encoding="utf-8"))
    entries = [{"ticker": t["ticker"], "entry_date": t["entry_date"]}
               for t in pf.get("closed_trades", [])
               if t.get("strategy") == "SCAN" and t.get("entry_date")]
    for tk, p in pf.get("positions", {}).items():
        if p.get("strategy") == "SCAN" and p.get("entry_date"):
            entries.append({"ticker": tk, "entry_date": p["entry_date"]})

    scan_by_date = defaultdict(list)
    for e in entries:
        scan_by_date[pd.Timestamp(e["entry_date"])].append(e["ticker"])

    signals = {}
    for f in glob.glob(str(DATA / "signals" / "signals_*.json")):
        d = pd.Timestamp(os.path.basename(f)[8:16])
        signals[d] = json.loads(Path(f).read_text(encoding="utf-8")).get("picks", [])

    days = sorted(set(scan_by_date) & set(signals))
    print(f"[표본] 교집합 {len(days)}일, SCAN 진입 {sum(len(scan_by_date[d]) for d in days)}건")

    # A/B 종목 중복 점검
    a_all = {t for d in days for t in scan_by_date[d]}
    b_all = {p["ticker"] for d in days for p in signals[d]}
    print(f"[중복] A 고유 {len(a_all)}종목 / B 고유 {len(b_all)}종목 / 교집합 {len(a_all & b_all)}종목")
    print(f"[기간편향] FOREIGN_SURGE_PB 최초 등장 20260602 — 5월 구간 부재\n")

    universe = list(prices)
    # 기저선 대비 초과수익 수집: group -> n -> list, 그리고 날짜별 평균(페어링용)
    exc = {g: {n: {"all": [], "up": [], "dn": []} for n in HORIZONS}
           for g in ("A_SCAN", "B_TOP5", "BF_FOREIGN")}
    paired = {g: {n: [] for n in HORIZONS} for g in ("A_SCAN", "B_TOP5", "BF_FOREIGN")}
    base_by_day = {}

    for d in days:
        for n in HORIZONS:
            base_vals = [r for tk in universe
                         if (r := fwd_ret(prices[tk], d, n)) is not None]
            if len(base_vals) < 100:
                continue
            base = float(np.mean(base_vals))
            base_by_day[(d, n)] = base
            k = fwd_ret(kospi, d, n)
            is_up = (k is not None and k > 0)

            groups = {
                "A_SCAN": scan_by_date[d],
                "B_TOP5": [p["ticker"] for p in signals[d]],
                "BF_FOREIGN": [p["ticker"] for p in signals[d]
                               if "FOREIGN_SURGE_PB" in (p.get("signals") or [])],
            }
            for g, tks in groups.items():
                vals = [r - base for tk in tks
                        if (s := prices.get(tk)) is not None
                        and (r := fwd_ret(s, d, n)) is not None]
                if not vals:
                    continue
                exc[g][n]["all"].extend(vals)
                (exc[g][n]["up"] if is_up else exc[g][n]["dn"]).extend(vals)
                paired[g][n].append((d, float(np.mean(vals))))

    labels = {"A_SCAN": "A  메인A 실진입 SCAN",
              "B_TOP5": "B  시그널 TOP5 전체",
              "BF_FOREIGN": "BF TOP5 중 외국인급증"}

    print("=" * 92)
    print("★ 기저선(같은 날 유니버스 동일가중 평균) 대비 초과수익 — 교정된 판정 기준")
    print("=" * 92)
    for n in HORIZONS:
        print(f"\n── D+{n} ──")
        for g, lab in labels.items():
            print(f"  {lab:24s} {stat_line(exc[g][n]['all'])}")

    print("\n" + "=" * 92)
    print("KOSPI 상승/하락 구간 분해 (기저선 대비 초과)")
    print("=" * 92)
    for n in HORIZONS:
        print(f"\n── D+{n} ──")
        for g, lab in labels.items():
            print(f"  {lab:24s}")
            print(f"      상승구간  {stat_line(exc[g][n]['up'])}")
            print(f"      하락구간  {stat_line(exc[g][n]['dn'])}")

    print("\n" + "=" * 92)
    print("★★ 날짜 페어드 격차 — 같은 날 B(또는 BF) − A. 시장 국면 효과 상쇄")
    print("=" * 92)
    for n in HORIZONS:
        a_map = dict(paired["A_SCAN"][n])
        for g in ("B_TOP5", "BF_FOREIGN"):
            g_map = dict(paired[g][n])
            common = sorted(set(a_map) & set(g_map))
            diffs = [g_map[d] - a_map[d] for d in common]
            if len(diffs) < 2:
                print(f"  D+{n:<2d} {g:11s} − A : 페어 {len(diffs)}일 (표본부족)")
                continue
            a = np.array(diffs, float)
            t, p = stats.ttest_1samp(a, 0.0)
            print(f"  D+{n:<2d} {g:11s} − A : 평균 {a.mean():+7.2f}%p  중앙값 {np.median(a):+7.2f}%p  "
                  f"승 {100 * (a > 0).mean():5.1f}%  페어 {len(a):2d}일  t={t:+5.2f} p={p:.3f}")

    print("\n" + "=" * 92)
    print("독립표본 Welch 검정 (B vs A, 종목 단위)")
    print("=" * 92)
    for n in HORIZONS:
        a, b = exc["A_SCAN"][n]["all"], exc["B_TOP5"][n]["all"]
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(b, a, equal_var=False)
            print(f"  D+{n:<2d}  B−A = {np.mean(b) - np.mean(a):+6.2f}%p   "
                  f"t={t:+5.2f}  p={p:.4f}   (nA={len(a)}, nB={len(b)})")


if __name__ == "__main__":
    main()
