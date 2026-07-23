"""스코어보드 알파 분해 — 구조적 격차 vs 진짜 초과손실.

배경(7/23): 메인A 진입 거래를 **같은 날 유니버스 동일가중 평균 대비**로 재면
  D+5 -1.53 ~ D+20 -3.65%p로 **전부 무의**(p=.13~.21)인데, A-0 스코어보드의
  계좌 알파는 **-34.1%p**다. 이 격차가 설명되지 않는다.

가설: 스코어보드가 KOSPI(시총가중) 대비로 재기 때문이다. 이 국면은 시총 최상위
  주도장이라 개별종목을 들고 있다는 사실만으로 지수에 크게 뒤진다
  (유니버스 동일가중 평균조차 KOSPI 대비 D+20 -9.9%p).

측정: 각 계좌의 실제 가동구간(daily_equity 첫날~마지막날)에 대해
  ① 계좌 누적수익률
  ② KOSPI 수익률            → 현행 알파 = ① - ②
  ③ 유니버스 동일가중 B&H   → 신규 알파 = ① - ③   ★종목 선택의 순수 성과
  ④ 구조적 격차 = ② - ③     ★"개별주를 들고 있다"는 사실만으로 생기는 손실

읽는 법: ④가 크고 ③이 0 근처면 진입 로직은 멀쩡하고 배분(개별주 vs 지수)이 문제.
        ③이 여전히 크게 음수면 종목 선택이 진짜 문제.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
DATA = QM / "data"

ACCOUNTS = [
    ("메인A", "paper_portfolio.json"),
    ("B안", "paper_portfolio_b.json"),
    ("블루칩V3", "paper_bluechip.json"),
    ("파도VF", "paper_portfolio_vf.json"),
    ("현금방어NAV", "paper_portfolio_holdnav.json"),
]


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
    kospi = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()
    print(f"[load] 종목 {len(prices)}개 · KOSPI {len(kospi)}행\n")

    def at_or_before(s: pd.Series, d: pd.Timestamp):
        idx = s.index.searchsorted(d, side="right") - 1
        return float(s.iloc[idx]) if idx >= 0 else None

    def equal_weight_bh(d0: pd.Timestamp, d1: pd.Timestamp):
        """d0 종가에 전 종목 동일금액 매수 → d1 종가 보유 시 평균 수익률(%)."""
        rets = []
        for t, s in prices.items():
            p0, p1 = at_or_before(s, d0), at_or_before(s, d1)
            if p0 and p1 and p0 > 0:
                # d0 이전에 데이터가 끝난 종목 제외(상폐)
                if s.index[-1] < d0:
                    continue
                rets.append((p1 / p0 - 1) * 100)
        return float(np.mean(rets)), len(rets)

    print("=" * 100)
    print(f"{'계좌':12s} {'구간':23s} {'①계좌':>8s} {'②KOSPI':>8s} {'③동일가중':>9s} "
          f"{'현행α(①-②)':>11s} {'신규α(①-③)':>11s} {'구조격차(②-③)':>13s}")
    print("=" * 100)

    rows = []
    for label, fname in ACCOUNTS:
        p = DATA / fname
        if not p.exists():
            continue
        de = json.loads(p.read_text(encoding="utf-8")).get("daily_equity", [])
        if len(de) < 2:
            continue
        d0, d1 = pd.Timestamp(de[0]["date"]), pd.Timestamp(de[-1]["date"])
        e0, e1 = de[0]["equity"], de[-1]["equity"]
        acct = (e1 / e0 - 1) * 100

        k0, k1 = at_or_before(kospi, d0), at_or_before(kospi, d1)
        kospi_ret = (k1 / k0 - 1) * 100 if k0 and k1 else float("nan")
        ew_ret, n = equal_weight_bh(d0, d1)

        a_old = acct - kospi_ret
        a_new = acct - ew_ret
        gap = kospi_ret - ew_ret
        rows.append((label, acct, kospi_ret, ew_ret, a_old, a_new, gap, n))
        print(f"{label:12s} {d0.date()}~{d1.date()} {acct:+8.1f}% {kospi_ret:+8.1f}% "
              f"{ew_ret:+9.1f}% {a_old:+11.1f} {a_new:+11.1f} {gap:+13.1f}")

    print("=" * 100)
    print(f"\n※ 동일가중 B&H는 각 구간 시작일 종가에 전 종목 동일금액 매수 가정 (표본 n은 계좌별 상이)")

    if rows:
        main_row = rows[0]
        print(f"\n★ 메인A 분해: 현행 알파 {main_row[4]:+.1f}%p")
        print(f"   = 구조적 격차 {main_row[6]:+.1f}%p (개별주를 들고 있다는 사실)")
        print(f"   + 종목선택 성과 {main_row[5]:+.1f}%p (동일가중 대비)")
        share = abs(main_row[6]) / abs(main_row[4]) * 100 if main_row[4] else 0
        print(f"   → 현행 알파의 {share:.0f}%가 구조적 격차")


if __name__ == "__main__":
    main()
