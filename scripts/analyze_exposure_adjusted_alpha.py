"""노출도 보정 알파 — 어제(7/23) 도입한 αEW의 편향 검증.

★의심: αEW = 계좌누적 − 동일가중누적 인데, 메인A는 주식비중 평균 30%이고
  동일가중 벤치마크는 100% 투자다. **노출도가 다른 둘을 그냥 뺐다.**
  하락장에서는 현금 비중이 높을수록 αEW가 좋게 나온다 = 종목 선택 능력이 아니라
  단지 "덜 투자했다"는 사실이 알파로 둔갑한다.

올바른 비교: 같은 노출도의 동일가중을 벤치마크로 삼는다.
  exposure_matched_ret(t) = stock_ratio(t) × ew_ret(t)
  → 이걸 누적한 것과 실제 계좌 누적을 비교해야 **종목 선택의 순수 성과**가 나온다.

이게 중요한 이유: 지금 SHIELD 고착(3종목·현금 70%)을 풀지 말지 판단해야 하는데,
  · 종목 선택이 실제로 좋다면 → 자본을 더 굴려야 하고 (해제가 이득)
  · 종목 선택이 나쁘다면 → 풀면 손실만 증폭된다 (해제가 손해)
  αEW가 편향돼 있으면 이 판단을 정반대로 이끈다.
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


def main() -> None:
    pf = json.loads((DATA / "paper_portfolio.json").read_text(encoding="utf-8"))
    de = pf["daily_equity"]
    nav = pd.Series({pd.Timestamp(x["date"]): float(x["equity"]) for x in de}).sort_index()
    ratio = pd.Series({pd.Timestamp(x["date"]): x.get("stock_ratio") for x in de}).sort_index()
    ratio = pd.to_numeric(ratio, errors="coerce") / 100.0

    print(f"[구간] {nav.index[0].date()} ~ {nav.index[-1].date()} ({len(nav)}일)")
    print(f"[주식비중] 평균 {ratio.mean() * 100:.1f}%  최소 {ratio.min() * 100:.1f}%  "
          f"최대 {ratio.max() * 100:.1f}%  결측 {ratio.isna().sum()}일")

    # 동일가중 일별 수익률
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
    ew_daily = R.mean(axis=1, skipna=True).reindex(nav.index)

    kospi = pd.read_csv(DATA / "kospi_index.csv", parse_dates=["Date"]).set_index("Date")["close"].sort_index()
    k_daily = kospi.reindex(nav.index).ffill().pct_change()

    acct_daily = nav.pct_change()

    df = pd.DataFrame({
        "acct": acct_daily, "ew": ew_daily, "k": k_daily, "ratio": ratio,
    }).dropna(subset=["acct", "ew"])
    df["ratio"] = df["ratio"].ffill().fillna(df["ratio"].mean())

    # 노출도 매칭 벤치마크: 그날 주식비중만큼만 동일가중에 투자, 나머지 현금(0%)
    df["ew_matched"] = df["ratio"] * df["ew"]

    def cum(s):
        return ((1 + s).prod() - 1) * 100

    a_cum = cum(df["acct"])
    ew_cum = cum(df["ew"])
    ewm_cum = cum(df["ew_matched"])
    k_cum = cum(df["k"])

    print("\n" + "=" * 78)
    print("누적 수익률 비교")
    print("=" * 78)
    print(f"  실제 계좌                       {a_cum:+8.2f}%")
    print(f"  KOSPI (100% 투자)               {k_cum:+8.2f}%")
    print(f"  동일가중 (100% 투자)             {ew_cum:+8.2f}%")
    print(f"  ★동일가중 (노출도 매칭 {df['ratio'].mean() * 100:.0f}%)   {ewm_cum:+8.2f}%")

    print("\n" + "=" * 78)
    print("알파 비교 — 어느 게 진짜 종목 선택 성과인가")
    print("=" * 78)
    print(f"  αK   (KOSPI 대비)                {a_cum - k_cum:+8.2f}%p")
    print(f"  αEW  (동일가중 대비, 현행 지표)    {a_cum - ew_cum:+8.2f}%p")
    print(f"  ★αEW-adj (노출도 매칭 대비)       {a_cum - ewm_cum:+8.2f}%p   ← 종목선택 순수성과")

    print("\n" + "=" * 78)
    print("★해석 — SHIELD 해제(자본 투입 확대) 판단의 근거")
    print("=" * 78)
    adj = a_cum - ewm_cum
    if adj > 0:
        print(f"  종목 선택이 노출도 보정 후에도 +{adj:.2f}%p 우위")
        print("  → 자본을 더 굴리면 이득이 커진다. SHIELD 해제가 정당화된다.")
    else:
        print(f"  종목 선택이 노출도 보정 후 {adj:.2f}%p 열위")
        print("  → ★자본을 더 굴리면 손실이 증폭된다. SHIELD 해제는 위험하다.")
        print("     현금 70%가 '놀고 있던 것'이 아니라 '손실을 막고 있던 것'일 수 있다.")

    # 어제 αEW가 얼마나 부풀려졌나
    print(f"\n  ※ 현행 αEW({a_cum - ew_cum:+.2f}%p) vs 보정 후({adj:+.2f}%p) "
          f"= 편향 {(a_cum - ew_cum) - adj:+.2f}%p")
    print("     이 차이는 순전히 '덜 투자했다'는 사실에서 나온 것이며 능력이 아니다.")

    # 월별로도
    print("\n" + "=" * 78)
    print("월별 (계좌 vs 노출도매칭 동일가중)")
    print("=" * 78)
    g = df.groupby(df.index.to_period("M"))
    for per, sub in g:
        a, m, e = cum(sub["acct"]), cum(sub["ew_matched"]), cum(sub["ew"])
        print(f"  {per}  계좌 {a:+7.2f}%  매칭EW {m:+7.2f}%  차이 {a - m:+7.2f}%p "
              f"(참고 100%EW {e:+7.2f}%, 평균비중 {sub['ratio'].mean() * 100:.0f}%)")

    print("\n※ 한계: stock_ratio는 일별 종가 기준 스냅샷이라 장중 변동 미반영.")
    print("  현금 수익률 0% 가정(실제 예수금 이자 무시). 거래비용은 계좌에만 반영됨.")


if __name__ == "__main__":
    main()
