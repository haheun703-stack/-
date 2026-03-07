"""Daily Flow Sentinel 백테스트

KOSPI 투자자별(외국인/기관/개인) 일별 순매수와
다음 거래일 KOSPI 수익률의 상관관계 분석.

질문:
  1. 외국인 대량 순매도 다음날 코스피는 빠지는가?
  2. 외국인+기관 동시 순매수 다음날 코스피는 오르는가?
  3. 유의미한 임계치(백분위)가 존재하는가?

Usage:
    python scripts/backtest_flow_sentinel.py
    python scripts/backtest_flow_sentinel.py --days 500
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KOSPI_CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
FLOW_CACHE = PROJECT_ROOT / "data" / "kospi_investor_flow.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "flow_sentinel_backtest.json"


def fetch_kospi_investor_flow(days: int = 750) -> pd.DataFrame:
    """네이버 금융에서 KOSPI 투자자별 순매수 수집 (캐시 활용).

    네이버 금융 투자자별 매매동향 페이지를 스크래핑.
    데이터: 날짜, 개인, 외국인, 기관계 (억원 단위).
    """
    from datetime import datetime, timedelta
    import requests
    from bs4 import BeautifulSoup

    # 캐시 확인
    if FLOW_CACHE.exists():
        cached = pd.read_csv(FLOW_CACHE, index_col="Date", parse_dates=True)
        cache_days = (pd.Timestamp.now() - cached.index[-1]).days
        if cache_days <= 1 and len(cached) >= days * 0.8:
            print(f"  [캐시] {len(cached)}일 데이터 로드 ({FLOW_CACHE.name})")
            return cached

    cutoff = datetime.today() - timedelta(days=days)
    target_rows = int(days * 0.7)  # 거래일 ≈ 캘린더일 × 0.7
    max_pages = target_rows // 10 + 20  # 10행/페이지 + 안전 마진

    url = "https://finance.naver.com/sise/investorDealTrendDay.naver"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://finance.naver.com/sise/sise_deal.naver",
    }
    bizdate = datetime.today().strftime("%Y%m%d")

    all_rows = []
    print(f"  네이버 금융 스크래핑 시작 (목표: ~{target_rows}거래일)...")

    for page_num in range(1, max_pages + 1):
        params = {"bizdate": bizdate, "sosok": "01", "page": page_num}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.encoding = "euc-kr"
        except Exception as e:
            print(f"  페이지 {page_num} 요청 실패: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "type_1"})
        if not table:
            break

        rows = table.find_all("tr")
        page_added = 0
        oldest_date = None

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            date_text = cells[0].get_text(strip=True)
            if not date_text or "." not in date_text:
                continue

            # 날짜 파싱 (YY.MM.DD → 20YY.MM.DD)
            try:
                dt = datetime.strptime(f"20{date_text}", "%Y.%m.%d")
            except ValueError:
                continue

            # 순매수 파싱 (쉼표 제거)
            def parse_amount(text: str) -> float:
                text = text.replace(",", "").replace(" ", "")
                try:
                    return float(text)
                except ValueError:
                    return 0.0

            retail = parse_amount(cells[1].get_text(strip=True))    # 개인
            foreign = parse_amount(cells[2].get_text(strip=True))   # 외국인
            inst = parse_amount(cells[3].get_text(strip=True))      # 기관계

            all_rows.append({
                "Date": dt,
                "foreign_net": foreign,  # 억원
                "inst_net": inst,
                "retail_net": retail,
            })
            page_added += 1
            oldest_date = dt

        if page_num % 20 == 0:
            print(f"    ... {page_num}페이지, {len(all_rows)}행 수집"
                  f" (최고일: {oldest_date.strftime('%Y-%m-%d') if oldest_date else '?'})")

        # 목표 기간 도달 체크
        if oldest_date and oldest_date < cutoff:
            break

        time.sleep(0.15)  # rate limit

    if not all_rows:
        raise ValueError("네이버 금융 스크래핑 실패: 데이터 없음")

    df = pd.DataFrame(all_rows)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 캐시 저장
    df.to_csv(FLOW_CACHE)
    print(f"  수집 완료: {len(df)}일"
          f" ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  캐시 저장 → {FLOW_CACHE.name}")
    return df


def load_kospi_returns() -> pd.Series:
    """KOSPI 일별 수익률."""
    df = pd.read_csv(KOSPI_CSV, index_col="Date", parse_dates=True)
    df = df.sort_index()
    returns = df["close"].pct_change() * 100  # %
    returns.name = "kospi_ret"
    return returns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=750,
                        help="수집 기간 (기본: 750일 ≈ 3년)")
    args = parser.parse_args()

    print("[Daily Flow Sentinel 백테스트]")
    print("=" * 75)

    # ── 1단계: 데이터 수집 ──
    flow_df = fetch_kospi_investor_flow(args.days)
    kospi_ret = load_kospi_returns()

    # 합치기
    merged = flow_df.copy()
    merged["kospi_ret"] = kospi_ret
    merged["next_ret"] = merged["kospi_ret"].shift(-1)  # 다음날 수익률
    merged = merged.dropna(subset=["next_ret"])

    # 네이버 금융 데이터는 이미 억원 단위 — 변환 불필요

    print(f"\n  분석 기간: {merged.index[0].strftime('%Y-%m-%d')} ~ "
          f"{merged.index[-1].strftime('%Y-%m-%d')} ({len(merged)}일)")

    if "foreign_net" in merged.columns:
        print(f"  외국인 순매수: 평균 {merged['foreign_net'].mean():+,.0f}억, "
              f"중앙값 {merged['foreign_net'].median():+,.0f}억")
    if "inst_net" in merged.columns:
        print(f"  기관 순매수:   평균 {merged['inst_net'].mean():+,.0f}억, "
              f"중앙값 {merged['inst_net'].median():+,.0f}억")

    # ── 2단계: 상관관계 분석 ──
    print(f"\n{'═' * 75}")
    print(f"  상관관계 분석 (순매수 → 다음날 코스피)")
    print(f"{'═' * 75}")

    for col in ["foreign_net", "inst_net", "retail_net"]:
        if col not in merged.columns:
            continue
        corr = merged[col].corr(merged["next_ret"])
        label = {"foreign_net": "외국인", "inst_net": "기관", "retail_net": "개인"}[col]
        print(f"  {label} 순매수 → 다음날 코스피: 상관계수 {corr:+.4f}")

    # 외국인+기관 합산
    if "foreign_net" in merged.columns and "inst_net" in merged.columns:
        merged["smart_money"] = merged["foreign_net"] + merged["inst_net"]
        corr = merged["smart_money"].corr(merged["next_ret"])
        print(f"  외국인+기관 합산 → 다음날 코스피: 상관계수 {corr:+.4f}")

    # ── 3단계: 백분위별 분석 (핵심) ──
    print(f"\n{'═' * 75}")
    print(f"  외국인 순매수 백분위별 다음날 코스피 수익률")
    print(f"{'═' * 75}")

    if "foreign_net" not in merged.columns:
        print("  외국인 데이터 없음!")
        return

    # 10분위 분석
    merged["foreign_decile"] = pd.qcut(merged["foreign_net"], 10,
                                        labels=False, duplicates="drop")

    print(f"  {'분위':>6} {'구간(억)':>20} {'건수':>5} │"
          f" {'다음날평균':>9} {'다음날중앙':>9} {'승률':>6} │ {'당일평균':>9}")
    print(f"  {'─' * 6} {'─' * 20} {'─' * 5} ┼"
          f" {'─' * 9} {'─' * 9} {'─' * 6} ┼ {'─' * 9}")

    decile_stats = []
    for d in range(10):
        sub = merged[merged["foreign_decile"] == d]
        if len(sub) < 5:
            continue
        f_min = sub["foreign_net"].min()
        f_max = sub["foreign_net"].max()
        n = len(sub)
        next_avg = sub["next_ret"].mean()
        next_med = sub["next_ret"].median()
        win_rate = (sub["next_ret"] > 0).sum() / n * 100
        today_avg = sub["kospi_ret"].mean()

        label = f"D{d}" if d < 9 else "D9(최대매수)"
        if d == 0:
            label = "D0(최대매도)"

        print(f"  {label:>10} {f_min:>+8,.0f}~{f_max:>+8,.0f} {n:>5} │"
              f" {next_avg:>+8.2f}% {next_med:>+8.2f}% {win_rate:>5.1f}% │"
              f" {today_avg:>+8.2f}%")

        decile_stats.append({
            "decile": d, "n": n,
            "foreign_min": round(f_min), "foreign_max": round(f_max),
            "next_avg": round(next_avg, 3), "next_med": round(next_med, 3),
            "win_rate": round(win_rate, 1), "today_avg": round(today_avg, 3),
        })

    # ── 4단계: 임계치 검증 (외국인 -5000억, -1조 등) ──
    print(f"\n{'═' * 75}")
    print(f"  주요 임계치별 다음날 코스피 수익률")
    print(f"{'═' * 75}")

    thresholds = [
        ("외국인 < -1조", merged["foreign_net"] < -10000),
        ("외국인 < -5000억", merged["foreign_net"] < -5000),
        ("외국인 < -3000억", merged["foreign_net"] < -3000),
        ("외국인 < -1000억", merged["foreign_net"] < -1000),
        ("외국인 > +1000억", merged["foreign_net"] > 1000),
        ("외국인 > +3000억", merged["foreign_net"] > 3000),
        ("외국인 > +5000억", merged["foreign_net"] > 5000),
        ("외국인 > +1조", merged["foreign_net"] > 10000),
    ]

    threshold_stats = []
    print(f"  {'조건':>18} {'건수':>5} │ {'다음날평균':>9} {'다음날중앙':>9}"
          f" {'승률':>6} │ {'당일평균':>9}")
    print(f"  {'─' * 18} {'─' * 5} ┼ {'─' * 9} {'─' * 9}"
          f" {'─' * 6} ┼ {'─' * 9}")

    for label, mask in thresholds:
        sub = merged[mask]
        n = len(sub)
        if n < 3:
            print(f"  {label:>18} {n:>5} │ 데이터 부족")
            continue
        next_avg = sub["next_ret"].mean()
        next_med = sub["next_ret"].median()
        wr = (sub["next_ret"] > 0).sum() / n * 100
        today_avg = sub["kospi_ret"].mean()
        print(f"  {label:>18} {n:>5} │ {next_avg:>+8.2f}% {next_med:>+8.2f}%"
              f" {wr:>5.1f}% │ {today_avg:>+8.2f}%")
        threshold_stats.append({
            "label": label, "n": n,
            "next_avg": round(next_avg, 3), "next_med": round(next_med, 3),
            "win_rate": round(wr, 1), "today_avg": round(today_avg, 3),
        })

    # ── 5단계: 외국인+기관 동시 매수/매도 ──
    if "smart_money" in merged.columns:
        print(f"\n{'═' * 75}")
        print(f"  외국인+기관 스마트머니 복합 시그널")
        print(f"{'═' * 75}")

        combo_tests = [
            ("외+기 동시매도 (-3000억↓)",
             (merged["foreign_net"] < 0) & (merged["inst_net"] < 0) &
             (merged["smart_money"] < -3000)),
            ("외+기 동시매도 (-1000억↓)",
             (merged["foreign_net"] < 0) & (merged["inst_net"] < 0) &
             (merged["smart_money"] < -1000)),
            ("외+기 동시매수 (+1000억↑)",
             (merged["foreign_net"] > 0) & (merged["inst_net"] > 0) &
             (merged["smart_money"] > 1000)),
            ("외+기 동시매수 (+3000억↑)",
             (merged["foreign_net"] > 0) & (merged["inst_net"] > 0) &
             (merged["smart_money"] > 3000)),
            ("외국인매도+기관매수",
             (merged["foreign_net"] < -1000) & (merged["inst_net"] > 1000)),
            ("외국인매수+기관매도",
             (merged["foreign_net"] > 1000) & (merged["inst_net"] < -1000)),
        ]

        print(f"  {'조건':>25} {'건수':>5} │ {'다음날평균':>9} {'승률':>6}"
              f" │ {'2일후평균':>9}")
        print(f"  {'─' * 25} {'─' * 5} ┼ {'─' * 9} {'─' * 6}"
              f" ┼ {'─' * 9}")

        merged["next2_ret"] = merged["kospi_ret"].shift(-2)

        for label, mask in combo_tests:
            sub = merged[mask].dropna(subset=["next2_ret"])
            n = len(sub)
            if n < 3:
                continue
            next_avg = sub["next_ret"].mean()
            wr = (sub["next_ret"] > 0).sum() / n * 100
            next2_avg = sub["next2_ret"].mean()
            print(f"  {label:>25} {n:>5} │ {next_avg:>+8.2f}% {wr:>5.1f}%"
                  f" │ {next2_avg:>+8.2f}%")

    # ── 6단계: 연속 매매 패턴 ──
    print(f"\n{'═' * 75}")
    print(f"  외국인 연속 매매 패턴")
    print(f"{'═' * 75}")

    # 연속 매도일 계산
    merged["foreign_sell"] = (merged["foreign_net"] < 0).astype(int)
    merged["sell_streak"] = 0
    streak = 0
    for i in range(len(merged)):
        if merged["foreign_sell"].iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        merged.iloc[i, merged.columns.get_loc("sell_streak")] = streak

    # 연속 매수일 계산
    merged["foreign_buy"] = (merged["foreign_net"] > 0).astype(int)
    merged["buy_streak"] = 0
    streak = 0
    for i in range(len(merged)):
        if merged["foreign_buy"].iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        merged.iloc[i, merged.columns.get_loc("buy_streak")] = streak

    streak_tests = [
        ("3일+ 연속 매도", merged["sell_streak"] >= 3),
        ("5일+ 연속 매도", merged["sell_streak"] >= 5),
        ("3일+ 연속 매수", merged["buy_streak"] >= 3),
        ("5일+ 연속 매수", merged["buy_streak"] >= 5),
    ]

    print(f"  {'조건':>18} {'건수':>5} │ {'다음날평균':>9} {'승률':>6}")
    print(f"  {'─' * 18} {'─' * 5} ┼ {'─' * 9} {'─' * 6}")

    for label, mask in streak_tests:
        sub = merged[mask]
        n = len(sub)
        if n < 3:
            continue
        next_avg = sub["next_ret"].mean()
        wr = (sub["next_ret"] > 0).sum() / n * 100
        print(f"  {label:>18} {n:>5} │ {next_avg:>+8.2f}% {wr:>5.1f}%")

    # ── 7단계: 종합 판정 ──
    print(f"\n{'═' * 75}")
    print(f"  기준선 (전체 평균)")
    print(f"{'═' * 75}")
    base_wr = (merged["next_ret"] > 0).sum() / len(merged) * 100
    base_avg = merged["next_ret"].mean()
    print(f"  전체: {len(merged)}일, 다음날 평균 {base_avg:+.3f}%, 승률 {base_wr:.1f}%")

    # ── 8단계: 저장 ──
    output = {
        "period": {
            "start": merged.index[0].strftime("%Y-%m-%d"),
            "end": merged.index[-1].strftime("%Y-%m-%d"),
            "trading_days": len(merged),
        },
        "baseline": {
            "avg_next_ret": round(base_avg, 3),
            "win_rate": round(base_wr, 1),
        },
        "decile_analysis": decile_stats,
        "threshold_analysis": threshold_stats,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  [저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
