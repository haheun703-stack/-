"""메인A에 지수 익스포저를 섞었다면? — 사이즈/자산배분 축 1차 측정 (B-18 재정의 후속).

배경(7/23~24):
  · 메인A는 종목 선택으로는 시장 평균을 이기고 있다(αEW +3.4%p)
  · 그러나 2026년 구조격차 +62.8%p — 개별주 바스켓 자체가 지수에 크게 뒤진다
  · 그 격차는 평균회귀가 아니라 **추세 지속**(상위20% 이후 다음 60일 +5.78%p 추가 확대)
  → "좋은 종목 고르기"만으로는 이 국면을 이길 수 없다. 배분을 봐야 한다.

★핵심 사실: 메인A는 이미 **자본의 ~70%가 현금**이다(stock_ratio 27.8%).
  따라서 이 시뮬은 "주식을 팔아 지수를 산다"가 아니라
  **"놀고 있는 현금의 일부를 지수에 넣었다면"**을 묻는다. 기존 종목 선택은 무손상.

방법:
  · 실제 NAV 시계열(daily_equity)에서 일별 수익률을 추출
  · 현금 부분 중 w를 KOSPI에 투입한 합성 NAV를 재구성
    combined_ret = actual_ret + w_idx * kospi_ret
    (actual_ret은 이미 현금 비중이 녹아든 계좌 전체 수익률이므로,
     유휴현금 w를 지수에 넣으면 그만큼 지수 수익률이 가산된다)
  · w는 '전체 자본 대비 지수 배분 비율'이며 유휴현금 한도를 넘지 않게 캡

한계(먼저 명시):
  · 관측 1구간(4개월)·KOSPI 상승장 단일 국면 → "지수를 샀으면 좋았다"는 사후적으로 자명.
    ★따라서 이 시뮬의 목적은 '지수를 사자'는 결론이 아니라 **민감도와 리스크 측정**이다.
  · 거래비용·슬리피지·ETF 추적오차 미반영
  · KOSPI 지수를 ETF 프록시로 사용(KODEX200 CSV는 6/18부터라 구간 부족)
  · ★가장 중요: 이 국면에서 지수가 오른 것은 **결과론**이다. 하락 국면 재현은 별도 필요
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
DATA = QM / "data"


def mdd(nav: pd.Series) -> float:
    return float(((nav / nav.cummax()) - 1).min() * 100)


def stats_of(nav: pd.Series, label: str) -> dict:
    r = nav.pct_change().dropna()
    total = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return {"label": label, "total": total, "mdd": mdd(nav), "vol": vol, "sharpe": sharpe}


def main() -> None:
    pf = json.loads((DATA / "paper_portfolio.json").read_text(encoding="utf-8"))
    de = pf["daily_equity"]
    nav = pd.Series({pd.Timestamp(x["date"]): float(x["equity"]) for x in de}).sort_index()
    ratios = pd.Series({pd.Timestamp(x["date"]): float(x.get("stock_ratio") or np.nan) for x in de}).sort_index()

    kospi = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()
    k = kospi.reindex(nav.index).ffill()

    actual_ret = nav.pct_change()
    kospi_ret = k.pct_change()
    df = pd.concat([actual_ret.rename("a"), kospi_ret.rename("k")], axis=1).dropna()

    print(f"[구간] {nav.index[0].date()} ~ {nav.index[-1].date()} ({len(nav)}거래일)")
    print(f"[주식비중] 최근 {ratios.dropna().iloc[-1]:.1f}% / 평균 {ratios.dropna().mean():.1f}% "
          f"→ 유휴현금 평균 {100 - ratios.dropna().mean():.1f}%")
    print(f"[실제] 계좌 {(nav.iloc[-1] / nav.iloc[0] - 1) * 100:+.2f}%  "
          f"KOSPI {(k.iloc[-1] / k.iloc[0] - 1) * 100:+.2f}%")

    print("\n" + "=" * 96)
    print("유휴현금 중 w를 KOSPI에 배분했다면 (기존 종목 선택 무손상)")
    print("=" * 96)
    print(f"{'지수배분':>8s} {'최종수익':>10s} {'MDD':>9s} {'변동성':>9s} {'Sharpe':>8s} {'αK':>9s} {'αEW':>9s}")

    # 동일가중 벤치마크(같은 구간) — αEW 계산용
    import glob, os
    ew_rets = []
    for f in glob.glob(str(DATA / "raw" / "*.parquet")):
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s = s[~s.index.duplicated(keep="last")]
            if s.index[-1] < nav.index[0]:
                continue
            p0 = s.reindex([nav.index[0]], method="ffill").iloc[0]
            p1 = s.reindex([nav.index[-1]], method="ffill").iloc[0]
            if p0 and p0 > 0:
                ew_rets.append((p1 / p0 - 1) * 100)
        except Exception:
            continue
    ew_total = float(np.mean(ew_rets)) if ew_rets else float("nan")
    k_total = (k.iloc[-1] / k.iloc[0] - 1) * 100

    rows = []
    for w in (0.0, 0.15, 0.30, 0.50, 0.70):
        combo = df["a"] + w * df["k"]
        synth = (1 + combo).cumprod() * float(nav.iloc[0])
        synth.loc[nav.index[0]] = float(nav.iloc[0])
        synth = synth.sort_index()
        st = stats_of(synth, f"w={w:.0%}")
        st["aK"] = st["total"] - k_total
        st["aEW"] = st["total"] - ew_total
        rows.append(st)
        print(f"{w:>7.0%} {st['total']:>+9.2f}% {st['mdd']:>+8.2f}% {st['vol']:>8.1f}% "
              f"{st['sharpe']:>8.2f} {st['aK']:>+8.1f} {st['aEW']:>+8.1f}")

    print(f"\n(참고) 같은 구간 KOSPI {k_total:+.2f}% / 유니버스 동일가중 {ew_total:+.2f}% "
          f"→ 구조격차 {k_total - ew_total:+.1f}%p")

    # ★적대적 검증: 하락 국면에서는?
    print("\n" + "=" * 96)
    print("★적대적 검증 — KOSPI 하락 구간만 잘라서 같은 배분을 적용하면?")
    print("=" * 96)
    dn = df[df["k"] < 0]
    up = df[df["k"] >= 0]
    print(f"  하락일 {len(dn)}일 / 상승일 {len(up)}일")
    for w in (0.0, 0.30, 0.50, 0.70):
        d_cum = ((1 + (dn["a"] + w * dn["k"])).prod() - 1) * 100
        u_cum = ((1 + (up["a"] + w * up["k"])).prod() - 1) * 100
        print(f"  w={w:>4.0%}  하락일 누적 {d_cum:>+8.2f}%   상승일 누적 {u_cum:>+8.2f}%")
    print("  ※ 하락일 누적이 w에 따라 급격히 악화되면, 이 시뮬의 이득은 상승장 결과론이다")

    # 최악 구간 스트레스
    print("\n" + "=" * 96)
    print("★스트레스 — 구간 내 최악 20거래일(KOSPI 기준) 누적")
    print("=" * 96)
    k20 = df["k"].rolling(20).sum()
    if k20.notna().any():
        end = k20.idxmin()
        pos = df.index.get_loc(end)
        seg = df.iloc[max(0, pos - 19): pos + 1]
        print(f"  구간 {seg.index[0].date()} ~ {seg.index[-1].date()} "
              f"(KOSPI {((1 + seg['k']).prod() - 1) * 100:+.2f}%)")
        for w in (0.0, 0.30, 0.50, 0.70):
            c = ((1 + (seg["a"] + w * seg["k"])).prod() - 1) * 100
            print(f"    w={w:>4.0%} → {c:+.2f}%")


if __name__ == "__main__":
    main()
