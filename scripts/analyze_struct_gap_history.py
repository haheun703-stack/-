"""구조격차(KOSPI − 유니버스 동일가중)의 역사적 재현성 — 7/23 발견의 적대적 검증.

7/23 발견: 메인A 구간(3/24~7/23)에 KOSPI +27.8% vs 유니버스 동일가중 -13.8%
  = 구조격차 **+41.6%p**. 이 탓에 계좌 알파가 -38.2%p로 찍히지만 동일가중 대비로는
  +3.4%p(이기는 중)였다.

★검증해야 할 질문 — 이 결론은 관측 1구간에 기반한다:
  Q1. 구조격차는 이 국면 특유인가, 늘 있는가?
  Q2. 지금 값(+41.6%p)은 역사적으로 얼마나 극단인가?
  Q3. 구조격차는 평균회귀하는가? (회귀한다면 지금 개별주를 버리는 건 최악의 타이밍)

측정: 롤링 윈도우(60·120일)로 KOSPI 수익률 − 동일가중 수익률을 전 구간 산출.

★생존편향 경고(중요): data/raw는 **현재 살아있는 종목**만 담는다. 과거로 갈수록
  그때 상장폐지된 종목이 빠져 동일가중 수익률이 실제보다 **좋게** 나온다
  = 구조격차가 실제보다 **작게** 측정된다. 따라서 "지금이 역사적 극단"이라는
  결론은 이 편향으로 인해 **과대평가**될 수 있다. 반대로 "구조격차가 늘 있다"는
  결론은 편향에도 불구하고 나온 것이므로 보수적으로 안전하다.
  또 2019년 이전 데이터가 없어 2020 코로나·2008 금융위기는 검증 못 한다.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
DATA = QM / "data"


def main() -> None:
    kospi = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()

    # 종목 종가 → 일별 수익률 행렬 (동일가중 = 횡단면 평균)
    rets = {}
    for f in glob.glob(str(DATA / "raw" / "*.parquet")):
        t = os.path.basename(f)[:-8]
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s = s[~s.index.duplicated(keep="last")]
            rets[t] = s.pct_change()
        except Exception:
            continue
    R = pd.DataFrame(rets).sort_index()
    print(f"[load] 종목 {R.shape[1]}개 · 거래일 {R.shape[0]}행 "
          f"({R.index[0].date()} ~ {R.index[-1].date()})")

    # 일별 동일가중 수익률(매일 리밸런스 가정) — 그날 데이터가 있는 종목만
    n_alive = R.notna().sum(axis=1)
    ew_daily = R.mean(axis=1, skipna=True)
    ew_daily = ew_daily[n_alive >= 100]
    print(f"[동일가중] 유효 거래일 {len(ew_daily)}행 "
          f"(종목 100개 이상인 날만) 최소생존 {int(n_alive.min())} 최대 {int(n_alive.max())}")

    kd = kospi.pct_change().reindex(ew_daily.index).dropna()
    ew_daily = ew_daily.reindex(kd.index)

    print("\n" + "=" * 88)
    print("구조격차 = KOSPI 누적 − 동일가중 누적 (롤링 윈도우, %p)")
    print("=" * 88)
    for win in (60, 120):
        k_cum = (1 + kd).rolling(win).apply(np.prod, raw=True) - 1
        e_cum = (1 + ew_daily).rolling(win).apply(np.prod, raw=True) - 1
        gap = (k_cum - e_cum) * 100
        gap = gap.dropna()
        cur = gap.iloc[-1]
        pct_rank = (gap < cur).mean() * 100
        print(f"\n── {win}거래일 롤링 (n={len(gap)}) ──")
        print(f"  평균 {gap.mean():+7.2f}%p   중앙값 {gap.median():+7.2f}%p   표준편차 {gap.std():6.2f}")
        print(f"  최소 {gap.min():+7.2f}%p   최대 {gap.max():+7.2f}%p")
        print(f"  양수 비율(KOSPI가 개별주 평균을 이긴 비율) {100 * (gap > 0).mean():5.1f}%")
        print(f"  ★현재값 {cur:+7.2f}%p → 역사적 백분위 {pct_rank:.1f}%")
        for q in (50, 75, 90, 95, 99):
            print(f"     {q}분위 {np.percentile(gap, q):+7.2f}%p")

    # 연도별
    print("\n" + "=" * 88)
    print("연도별 구조격차 (KOSPI 연수익 − 동일가중 연수익, %p)")
    print("=" * 88)
    for y, idx in kd.groupby(kd.index.year).groups.items():
        k = (1 + kd.loc[idx]).prod() - 1
        e = (1 + ew_daily.loc[idx]).prod() - 1
        print(f"  {y}: KOSPI {k * 100:+7.1f}%  동일가중 {e * 100:+7.1f}%  "
              f"→ 격차 {(k - e) * 100:+7.1f}%p  (거래일 {len(idx)})")

    # 평균회귀 검정: 구조격차가 클수록 이후 격차가 축소되는가
    print("\n" + "=" * 88)
    print("★평균회귀 검정 — 구조격차 확대 후 이후 60일 격차는?")
    print("=" * 88)
    win = 60
    k_cum = (1 + kd).rolling(win).apply(np.prod, raw=True) - 1
    e_cum = (1 + ew_daily).rolling(win).apply(np.prod, raw=True) - 1
    gap = ((k_cum - e_cum) * 100).dropna()
    fwd = gap.shift(-win)  # 다음 60일 구간의 격차
    df = pd.concat([gap.rename("now"), fwd.rename("next")], axis=1).dropna()
    if len(df) > 30:
        hi = df[df["now"] >= df["now"].quantile(0.8)]
        lo = df[df["now"] <= df["now"].quantile(0.2)]
        from scipy import stats
        r, p = stats.pearsonr(df["now"], df["next"])
        print(f"  전체 상관 now vs next: r={r:+.3f} (p={p:.4f}, n={len(df)})")
        print(f"  격차 상위20%일 때 → 다음 60일 격차 평균 {hi['next'].mean():+7.2f}%p (n={len(hi)})")
        print(f"  격차 하위20%일 때 → 다음 60일 격차 평균 {lo['next'].mean():+7.2f}%p (n={len(lo)})")
        print(f"  ※ 음의 상관·상위20% 이후 격차 축소 = 평균회귀 시사")
        print(f"     (겹치는 윈도우라 유의성은 과대평가됨 — 방향만 참고)")


if __name__ == "__main__":
    main()
