"""SHIELD 킬스위치 해제 조건은 발동 가능한가 — 4개월 RED 고착의 원인 규명.

배경(7/24 발견):
  · SHIELD가 6/23~7/23 전 로그 구간 내내 RED. 원인은 시장이 아니라 **계좌 자신의 MDD**
    (peak 40,863,823 @ 2026-03-11 → current 29,560,759 = -27.66% → LEVEL_3 고착)
  · RED → SHIELD_MAX_POSITIONS=3 → 자본의 ~70%가 4개월간 유휴
  · MDD는 peak 대비이고 peak는 절대 내려가지 않으므로, 해제하려면 +38% 회복 필요.
    그런데 3종목·28%만 굴려서는 그 회복이 사실상 불가능 = **자기강화 고착(ratchet)**
  · 유일한 탈출구가 `_check_recovery()`인데 조건이 2개 **동시** 충족:
        ① KOSPI 3일 누적 >= +3%      ② VIX 전일비 <= -10%
  · 7/23 실측 로그: `KOSPI 3일 +8.9% (OK), VIX -2.4% (미달)` → 해제 실패.
    **전환은 감지했는데 VIX 조건이 막았다.**

측정:
  Q1. 두 조건이 역사적으로 동시 충족되는 빈도는? (거의 0이면 해제 불가능한 게이트)
  Q2. 각 조건 단독 빈도는? 어느 쪽이 병목인가?
  Q3. VIX 조건은 값을 더하는가? — 조건① 단독 발동 후 forward 수익 vs ①+② 발동 후 수익.
      VIX를 걸어서 더 좋은 시점만 걸러낸다면 정당하지만, 그냥 막기만 한다면 결함이다.
  Q4. 임계를 완화하면(-10% → -5%/-3%/0%) 발동 빈도와 성과는?

★기저선 필수: 발동일 forward 수익은 비이벤트(전체 평균) 대비로 판정한다(7/7 교훈).
데이터: us_daily.parquet(vix_close, 2023-05~) + kospi_index.csv. 교집합 구간만.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
DATA = QM / "data"

HOLD = [10, 20, 60]


def main() -> None:
    us = pd.read_parquet(DATA / "us_market" / "us_daily.parquet")
    us.index = pd.to_datetime(us.index)
    vix = us["vix_close"].sort_index().dropna()

    k = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()

    # 교집합 구간
    idx = k.index.intersection(vix.index)
    k = k.reindex(idx).ffill()
    v = vix.reindex(idx).ffill()
    print(f"[구간] {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    kospi_3d = (k / k.shift(3) - 1) * 100
    vix_chg = (v / v.shift(1) - 1) * 100

    c1 = kospi_3d >= 3.0
    c2 = vix_chg <= -10.0
    both = c1 & c2

    n = len(idx)
    print("\n" + "=" * 84)
    print("Q1/Q2 — 조건별 발동 빈도")
    print("=" * 84)
    print(f"  ① KOSPI 3일 >= +3%      : {c1.sum():4d}일 ({100 * c1.mean():5.2f}%)")
    print(f"  ② VIX 전일비 <= -10%     : {c2.sum():4d}일 ({100 * c2.mean():5.2f}%)")
    print(f"  ★①AND② (현행 해제조건)   : {both.sum():4d}일 ({100 * both.mean():5.2f}%)")
    if both.sum():
        print(f"     발동일: {[str(d.date()) for d in idx[both]][:12]}")
        # 평균 간격
        pos = np.where(both.values)[0]
        if len(pos) > 1:
            print(f"     평균 발동 간격 {np.mean(np.diff(pos)):.0f}거래일 "
                  f"(≈ {np.mean(np.diff(pos)) / 21:.1f}개월)")
    else:
        print("     ★한 번도 발동하지 않음 = 해제 불가능한 게이트")

    # 독립 가정 기대치와 비교 (동시성이 우연보다 드문가)
    exp = c1.mean() * c2.mean() * n
    print(f"\n  독립 가정 기대 발동일 {exp:.1f}일 vs 실제 {both.sum()}일")
    print("  ※ KOSPI 급등일과 VIX 급락일은 양의 상관이 있어야 자연스러운데,")
    print("     실제가 기대보다 크게 낮으면 두 조건이 서로 다른 국면을 가리킨다는 뜻")

    # Q3: VIX 조건이 값을 더하는가
    print("\n" + "=" * 84)
    print("Q3 — VIX 조건은 더 좋은 시점을 걸러내는가? (기저선 대비 초과, %p)")
    print("=" * 84)

    def fwd(i, h):
        if i + h >= len(k):
            return None
        a, b = float(k.iloc[i]), float(k.iloc[i + h])
        return None if a <= 0 else (b / a - 1) * 100

    base = {h: np.array([r for i in range(60, len(k)) if (r := fwd(i, h)) is not None], float)
            for h in HOLD}
    for h in HOLD:
        print(f"\n── 보유 {h}일 (기저선 평균 {base[h].mean():+.2f}%) ──")
        for label, mask in (("① KOSPI만", c1), ("①AND② 현행", both)):
            ii = [k.index.get_loc(d) for d in idx[mask.fillna(False)]]
            vals = [r for i in ii if i >= 60 and (r := fwd(i, h)) is not None]
            if len(vals) < 5:
                print(f"   {label:14s} n={len(vals)} 표본부족 → 판정 불가")
                continue
            a = np.array(vals, float)
            t, p = stats.ttest_ind(a, base[h], equal_var=False)
            mark = " ★" if p < 0.05 and a.mean() > base[h].mean() else ""
            print(f"   {label:14s} 평균 {a.mean():+6.2f}%  승률 {100 * (a > 0).mean():5.1f}%  "
                  f"n={len(a):3d}  기저선대비 {a.mean() - base[h].mean():+6.2f}%p  p={p:.4f}{mark}")

    # Q4: 완화 시나리오
    print("\n" + "=" * 84)
    print("Q4 — VIX 임계 완화 시 발동 빈도 / D+20 성과")
    print("=" * 84)
    print(f"{'VIX임계':>10s} {'발동일':>7s} {'빈도%':>7s} {'D+20평균':>9s} {'기저선대비':>10s} {'p':>8s}")
    for thr in (-10.0, -7.0, -5.0, -3.0, 0.0, 100.0):
        m = c1 & (vix_chg <= thr)
        ii = [k.index.get_loc(d) for d in idx[m.fillna(False)]]
        vals = [r for i in ii if i >= 60 and (r := fwd(i, 20)) is not None]
        lab = "무조건(①만)" if thr == 100.0 else f"<= {thr:+.0f}%"
        if len(vals) < 5:
            print(f"{lab:>10s} {m.sum():>7d} {100 * m.mean():>6.2f}% {'표본부족':>9s}")
            continue
        a = np.array(vals, float)
        t, p = stats.ttest_ind(a, base[20], equal_var=False)
        print(f"{lab:>10s} {m.sum():>7d} {100 * m.mean():>6.2f}% {a.mean():>+8.2f}% "
              f"{a.mean() - base[20].mean():>+9.2f}%p {p:>8.4f}")

    print("\n※ 한계: VIX 데이터가 2023-05~라 구간이 3년뿐이고 코로나·금융위기 미포함.")
    print("  KOSPI 지수 수익률로만 판정했으며 실제 계좌 성과와는 다르다.")


if __name__ == "__main__":
    main()
