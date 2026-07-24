"""전환 '예측'이 아니라 '반응'은 되는가 — 7번째 타이밍 축 검증.

★배경: 지수 타이밍 축은 6번 기각됐다(레짐게이트·인버스·로테이션·약세장매집·
  drv진정·지수수급). 전부 "전환을 미리 맞힌다"는 예측형이었다.

이번 질문은 다르다: **이미 확인된 전환에 뒤늦게 붙어도 수익이 남는가?**
  예측이 불가능해도 반응이 충분히 빠르면 "하락장 쉬고 상승장 진입"이 성립한다.
  반대로 전환 확인 시점에 이미 상승분이 소진돼 있으면 그 전략은 원리적으로 불가능하다.

전환 정의(사후적 확인 규칙 — 미래 정보 미사용):
  · MA20 상향돌파: 종가가 20일선을 아래→위로 교차한 날 (D0)
  · MA60 상향돌파
  · 20일 신고가 갱신
  각 규칙은 그날 종가까지의 정보만 쓴다. D0 종가에 산다고 가정(가장 빠른 반응).
  이후 D+1·D+3·D+5·D+10에 지연 진입했을 때 남은 수익을 비교한다.

★핵심 대조: 비이벤트 기저선(무작위 날 진입) 대비 초과가 있어야 의미가 있다.
  (7/7 교훈: 이벤트 스터디는 비이벤트 기저선 없이 판정 금지)

측정 대상: KOSPI 지수 자체(지수를 살 것인지 묻는 것이므로).
기간: kospi_index.csv 전구간(1996~) — 종목 데이터와 달리 생존편향 없음.
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

HOLD = [10, 20, 60]      # 보유기간
DELAYS = [0, 1, 3, 5, 10]  # 전환 확인 후 진입 지연


def fwd(s: pd.Series, i: int, n: int):
    if i + n >= len(s):
        return None
    a, b = float(s.iloc[i]), float(s.iloc[i + n])
    return None if a <= 0 else (b / a - 1) * 100


def summarize(vals, label):
    if len(vals) < 5:
        return f"  {label:16s} n={len(vals)} 표본부족"
    a = np.array(vals, float)
    return (f"  {label:16s} 평균 {a.mean():+6.2f}%  중앙 {np.median(a):+6.2f}%  "
            f"승률 {100 * (a > 0).mean():5.1f}%  n={len(a):4d}")


def main() -> None:
    k = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()
    print(f"[load] KOSPI {len(k)}행 ({k.index[0].date()} ~ {k.index[-1].date()}) — 생존편향 없음")

    ma20 = k.rolling(20).mean()
    ma60 = k.rolling(60).mean()
    hi20 = k.rolling(20).max()

    events = {
        "MA20 상향돌파": (k > ma20) & (k.shift(1) <= ma20.shift(1)),
        "MA60 상향돌파": (k > ma60) & (k.shift(1) <= ma60.shift(1)),
        "20일 신고가": (k >= hi20) & (k.shift(1) < hi20.shift(1)),
    }

    # 비이벤트 기저선: 전 거래일 (충분한 워밍업 이후)
    base_idx = list(range(60, len(k)))
    print("\n" + "=" * 84)
    print("비이벤트 기저선 (모든 날 진입 — 이게 넘어야 할 기준선)")
    print("=" * 84)
    base = {}
    for h in HOLD:
        vals = [r for i in base_idx if (r := fwd(k, i, h)) is not None]
        base[h] = np.array(vals, float)
        print(summarize(vals, f"D+{h}"))

    for name, mask in events.items():
        idx = [k.index.get_loc(d) for d in k.index[mask.fillna(False)] if k.index.get_loc(d) >= 60]
        print("\n" + "=" * 84)
        print(f"★ {name} — 전환 확인 후 지연 진입별 수익 (전환 {len(idx)}회)")
        print("=" * 84)
        for h in HOLD:
            print(f"\n── 보유 {h}일 ──   (기저선 평균 {base[h].mean():+.2f}%)")
            for dly in DELAYS:
                vals = [r for i in idx if (r := fwd(k, i + dly, h)) is not None]
                if len(vals) < 5:
                    print(f"    지연 D+{dly:<2d}  표본부족")
                    continue
                a = np.array(vals, float)
                edge = a.mean() - base[h].mean()
                t, p = stats.ttest_ind(a, base[h], equal_var=False)
                mark = " ★" if p < 0.05 and edge > 0 else (" ✗역" if p < 0.05 and edge < 0 else "")
                print(f"    지연 D+{dly:<2d}  평균 {a.mean():+6.2f}%  승률 {100 * (a > 0).mean():5.1f}%  "
                      f"n={len(a):4d}  기저선대비 {edge:+6.2f}%p  t={t:+5.2f} p={p:.4f}{mark}")

    print("\n" + "=" * 84)
    print("판정 기준")
    print("=" * 84)
    print("  · 지연 D+0에도 기저선 대비 초과가 없으면 → 반응조차 무의미(7번째 기각)")
    print("  · D+0엔 있는데 지연될수록 소멸하면 → 반응 속도가 관건(구현 가치 있음)")
    print("  · 지연과 무관하게 초과가 유지되면 → 타이밍이 아니라 레짐 자체가 정보")


if __name__ == "__main__":
    main()
