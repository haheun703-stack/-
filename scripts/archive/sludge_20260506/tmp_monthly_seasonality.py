#!/usr/bin/env python3
"""KOSPI 10년 월별 수익률 계절성 분석"""
from pykrx import stock
import pandas as pd
from datetime import datetime

# 10년 데이터 (2016~2025)
start = "20160101"
end = "20251231"

print("KOSPI 지수 10년 데이터 로딩 중...")
df = stock.get_index_ohlcv(start, end, "1001")  # 1001 = KOSPI
print(f"데이터: {len(df)}건 ({df.index[0]} ~ {df.index[-1]})")

# 월말 종가 기준 월별 수익률
df["YM"] = df.index.to_period("M")
monthly = df.groupby("YM")["종가"].last()
monthly_ret = monthly.pct_change() * 100
monthly_ret = monthly_ret.dropna()

# 월 추출
monthly_ret_df = pd.DataFrame({
    "year": [p.year for p in monthly_ret.index],
    "month": [p.month for p in monthly_ret.index],
    "ret": monthly_ret.values
})

print()
print("=" * 90)
print("  KOSPI 월별 수익률 계절성 분석 (2016~2025, 10년)")
print("=" * 90)

# 월별 통계
print()
print("{:>4s} | {:>7s} {:>7s} {:>7s} | {:>5s} {:>5s} | {:>6s} | {}".format(
    "월", "평균", "중위수", "표준편차", "상승", "하락", "승률", "막대"))
print("-" * 80)

for m in range(1, 13):
    subset = monthly_ret_df[monthly_ret_df["month"] == m]["ret"]
    avg = subset.mean()
    med = subset.median()
    std = subset.std()
    up = (subset > 0).sum()
    down = (subset <= 0).sum()
    total = len(subset)
    wr = up / total * 100 if total > 0 else 0

    # 막대 그래프
    bar_len = int(abs(avg) * 3)
    if avg >= 0:
        bar = "+" * min(bar_len, 20)
        color_tag = "▲" if avg >= 1 else "△"
    else:
        bar = "-" * min(bar_len, 20)
        color_tag = "▼" if avg <= -1 else "▽"

    marker = ""
    if m in (5, 6):
        marker = " ◀◀◀"

    print("{:>4d}월 | {:>+6.2f}% {:>+6.2f}% {:>6.2f}% | {:>3d}회 {:>3d}회 | {:>5.1f}% | {} {}{}".format(
        m, avg, med, std, up, down, wr, color_tag, bar, marker))

# 상반기 vs 하반기
print()
h1 = monthly_ret_df[monthly_ret_df["month"] <= 6]["ret"]
h2 = monthly_ret_df[monthly_ret_df["month"] > 6]["ret"]
print("상반기(1~6월) 평균: {:+.2f}%  |  하반기(7~12월) 평균: {:+.2f}%".format(h1.mean(), h2.mean()))

# Sell in May 검증
may_oct = monthly_ret_df[monthly_ret_df["month"].isin([5,6,7,8,9])]["ret"]
nov_apr = monthly_ret_df[monthly_ret_df["month"].isin([11,12,1,2,3,4])]["ret"]
print("Sell in May 기간(5~9월) 평균: {:+.2f}%  |  Buy 기간(11~4월) 평균: {:+.2f}%".format(
    may_oct.mean(), nov_apr.mean()))

# 연도별 5월 수익률
print()
print("=" * 60)
print("  연도별 5월 수익률")
print("=" * 60)
may_data = monthly_ret_df[monthly_ret_df["month"] == 5].sort_values("year")
for _, r in may_data.iterrows():
    yr = int(r["year"])
    ret = r["ret"]
    bar = "+" * int(abs(ret) * 2) if ret >= 0 else "-" * int(abs(ret) * 2)
    print("  {}년 5월: {:>+6.2f}%  {}".format(yr, ret, bar))

# 연도별 6월 수익률
print()
print("=" * 60)
print("  연도별 6월 수익률")
print("=" * 60)
jun_data = monthly_ret_df[monthly_ret_df["month"] == 6].sort_values("year")
for _, r in jun_data.iterrows():
    yr = int(r["year"])
    ret = r["ret"]
    bar = "+" * int(abs(ret) * 2) if ret >= 0 else "-" * int(abs(ret) * 2)
    print("  {}년 6월: {:>+6.2f}%  {}".format(yr, ret, bar))

# 최악/최고의 달
print()
print("=" * 60)
print("  월별 순위 (평균 수익률 기준)")
print("=" * 60)
rank = []
for m in range(1, 13):
    subset = monthly_ret_df[monthly_ret_df["month"] == m]["ret"]
    rank.append((m, subset.mean(), subset.median(), (subset > 0).sum() / len(subset) * 100))

rank.sort(key=lambda x: x[1])
for i, (m, avg, med, wr) in enumerate(rank):
    tag = ""
    if i < 3:
        tag = " ◀ 약세"
    elif i >= 9:
        tag = " ◀ 강세"
    print("  {:>2d}위: {:>2d}월 | 평균 {:>+6.2f}% | 중위수 {:>+6.2f}% | 승률 {:>5.1f}%{}".format(
        i + 1, m, avg, med, wr, tag))
