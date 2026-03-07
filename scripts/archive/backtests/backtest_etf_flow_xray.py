"""ETF Flow X-Ray v2: ETF 매수가 기초종목 수급에 어떻게 반영되는지 분석

가설: 개인이 섹터 ETF를 대량 매수하면,
  AP(지정참가회사=증권사)가 ETF 설정 과정에서 기초자산을 매수.
  이때 개별종목에서는 "기관(금융투자)" 매수로 기록됨.

v1 결함 (동일일 상관만 봄):
  - ETF 설정은 T+1~T+2에 반영 → 시차(lag) 분석 필요
  - 기관 전체에 노이즈 과다 → 대량 유입일만 필터링 필요
  - 이벤트 스터디: 대량 유입 전후 5일 누적 패턴

v2 추가 분석:
  1. 시차 상관 (T+0 ~ T+3)
  2. 규모 필터 (ETF 개인매수 상위 20% 날만)
  3. 이벤트 스터디 (전후 5일 기초종목 기관 누적)

Usage:
    python scripts/backtest_etf_flow_xray.py
    python scripts/backtest_etf_flow_xray.py --pages 30
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "etf_flow_xray.json"

# ── ETF ↔ 기초종목 매핑 ──
# 각 ETF와 대표 기초종목 2~3개
ETF_STOCK_PAIRS = {
    "반도체": {
        "etfs": [
            ("091160", "KODEX 반도체"),
            ("395160", "TIGER 반도체TOP10"),
        ],
        "stocks": [
            ("000660", "SK하이닉스"),
            ("005930", "삼성전자"),
        ],
    },
    "자동차": {
        "etfs": [
            ("091180", "KODEX 자동차"),
        ],
        "stocks": [
            ("005380", "현대차"),
            ("000270", "기아"),
        ],
    },
    "조선": {
        "etfs": [
            ("266370", "KODEX 조선"),
        ],
        "stocks": [
            ("010140", "삼성중공업"),
            ("009540", "HD한국조선해양"),
        ],
    },
    "건설": {
        "etfs": [
            ("117700", "KODEX 건설"),
        ],
        "stocks": [
            ("000720", "현대건설"),
            ("047040", "대우건설"),
        ],
    },
    "은행": {
        "etfs": [
            ("091170", "KODEX 은행"),
        ],
        "stocks": [
            ("105560", "KB금융"),
            ("055550", "신한지주"),
        ],
    },
}


def scrape_investor_data(code: str, pages: int = 10) -> pd.DataFrame:
    """네이버 금융에서 종목/ETF 투자자별 매매동향 스크래핑.

    Returns:
        DataFrame with columns: close, inst_net, frgn_net, frgn_hold, frgn_pct
        Index: Date (datetime)
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://finance.naver.com",
    }

    all_rows = []

    for page_num in range(1, pages + 1):
        url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={page_num}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.encoding = "euc-kr"
        except Exception:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table", {"class": "type2"})
        if len(tables) < 2:
            break

        rows = tables[1].find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 7:
                continue
            date_text = cells[0].get_text(strip=True)
            if not date_text or "." not in date_text:
                continue

            def parse_num(text: str) -> float:
                text = text.replace(",", "").replace(" ", "")
                # 상승/하락 텍스트 제거
                for prefix in ["상승", "하락", "보합"]:
                    text = text.replace(prefix, "")
                try:
                    return float(text)
                except ValueError:
                    return 0.0

            try:
                dt = datetime.strptime(date_text, "%Y.%m.%d")
            except ValueError:
                continue

            close = parse_num(cells[1].get_text(strip=True))
            inst_net = parse_num(cells[5].get_text(strip=True))
            frgn_net = parse_num(cells[6].get_text(strip=True))

            all_rows.append({
                "Date": dt,
                "close": close,
                "inst_net": inst_net,
                "frgn_net": frgn_net,
            })

        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def analyze_sector(sector_name: str, config: dict, pages: int) -> dict:
    """한 섹터의 ETF↔기초종목 교차분석 (v2: 시차+규모+이벤트스터디)."""

    print(f"\n{'═' * 75}")
    print(f"  섹터: {sector_name}")
    print(f"{'═' * 75}")

    # ETF 데이터 수집
    etf_dfs = {}
    for code, name in config["etfs"]:
        print(f"  ETF [{name}] 수집 중...", end=" ")
        df = scrape_investor_data(code, pages)
        print(f"{len(df)}일")
        if not df.empty:
            df["retail_net_est"] = -(df["inst_net"] + df["frgn_net"])
            etf_dfs[code] = df

    # 기초종목 데이터 수집
    stock_dfs = {}
    for code, name in config["stocks"]:
        print(f"  주식 [{name}] 수집 중...", end=" ")
        df = scrape_investor_data(code, pages)
        print(f"{len(df)}일")
        if not df.empty:
            stock_dfs[code] = df

    if not etf_dfs or not stock_dfs:
        print("  데이터 부족!")
        return {}

    results = []

    for etf_code, etf_name in config["etfs"]:
        if etf_code not in etf_dfs:
            continue
        etf_df = etf_dfs[etf_code]

        for stk_code, stk_name in config["stocks"]:
            if stk_code not in stock_dfs:
                continue
            stk_df = stock_dfs[stk_code]

            # 날짜 기준 inner join
            merged = etf_df[["inst_net", "frgn_net", "retail_net_est"]].join(
                stk_df[["inst_net", "frgn_net"]],
                lsuffix="_etf", rsuffix="_stk",
                how="inner",
            )

            if len(merged) < 20:
                continue

            print(f"\n  [{etf_name}] ↔ [{stk_name}] ({len(merged)}일)")

            # ═══════════════════════════════════════════════
            # 분석 1: 시차별 상관 (T+0 ~ T+3)
            # ═══════════════════════════════════════════════
            print(f"    ── 시차별 상관 (ETF개인매수 T → 주식기관매수 T+lag) ──")
            lag_corrs = {}
            for lag in range(4):
                if lag == 0:
                    corr = merged["retail_net_est"].corr(merged["inst_net_stk"])
                else:
                    # T일 ETF 개인매수 vs T+lag일 주식 기관매수
                    shifted = merged["inst_net_stk"].shift(-lag)
                    valid = merged["retail_net_est"].to_frame().join(
                        shifted.rename("stk_inst_lag")
                    ).dropna()
                    corr = valid["retail_net_est"].corr(valid["stk_inst_lag"])
                lag_corrs[f"T+{lag}"] = round(corr, 4)
                marker = " ★" if abs(corr) > 0.1 else ""
                print(f"      T+{lag}: {corr:+.4f}{marker}")

            # ═══════════════════════════════════════════════
            # 분석 2: 규모 필터 (상위 20% 대량 유입일만)
            # ═══════════════════════════════════════════════
            q80 = merged["retail_net_est"].quantile(0.8)
            q20 = merged["retail_net_est"].quantile(0.2)
            big_buy = merged[merged["retail_net_est"] >= q80]
            big_sell = merged[merged["retail_net_est"] <= q20]

            print(f"    ── 규모 필터 (ETF개인매수 상위20% vs 하위20%) ──")
            big_buy_corrs = {}
            big_sell_corrs = {}
            for lag in range(4):
                # 대량 매수일 기준, T+lag일 주식 기관매수 평균
                buy_vals = []
                sell_vals = []
                for dt in big_buy.index:
                    pos = merged.index.get_loc(dt)
                    if pos + lag < len(merged):
                        buy_vals.append(merged["inst_net_stk"].iloc[pos + lag])
                for dt in big_sell.index:
                    pos = merged.index.get_loc(dt)
                    if pos + lag < len(merged):
                        sell_vals.append(merged["inst_net_stk"].iloc[pos + lag])

                buy_avg = np.mean(buy_vals) if buy_vals else 0
                sell_avg = np.mean(sell_vals) if sell_vals else 0
                all_avg = merged["inst_net_stk"].mean()
                big_buy_corrs[f"T+{lag}"] = round(buy_avg)
                big_sell_corrs[f"T+{lag}"] = round(sell_avg)

                marker = " ★" if buy_avg > all_avg * 1.3 else ""
                print(f"      T+{lag}: 대량매수일→기관 {buy_avg:>+12,.0f}"
                      f"  |  대량매도일→기관 {sell_avg:>+12,.0f}"
                      f"  |  전체평균 {all_avg:>+12,.0f}{marker}")

            # ═══════════════════════════════════════════════
            # 분석 3: 이벤트 스터디 (전후 5일 누적)
            # ═══════════════════════════════════════════════
            print(f"    ── 이벤트 스터디 (ETF개인 대량매수일 전후 5일) ──")

            # 상위 10% 대량 유입 이벤트
            q90 = merged["retail_net_est"].quantile(0.9)
            events = merged[merged["retail_net_est"] >= q90].index

            window = range(-3, 6)  # T-3 ~ T+5
            event_profile = {d: [] for d in window}

            for evt_dt in events:
                evt_pos = merged.index.get_loc(evt_dt)
                for offset in window:
                    target_pos = evt_pos + offset
                    if 0 <= target_pos < len(merged):
                        event_profile[offset].append(
                            merged["inst_net_stk"].iloc[target_pos]
                        )

            print(f"      이벤트 수: {len(events)}건 (상위 10%)")
            print(f"      {'날짜':>8} {'주식기관순매수평균':>15} {'vs전체평균':>10}")
            all_avg = merged["inst_net_stk"].mean()
            event_avgs = {}
            for offset in window:
                vals = event_profile[offset]
                if vals:
                    avg = np.mean(vals)
                    diff = avg - all_avg
                    label = f"T{offset:+d}" if offset != 0 else "T+0(당일)"
                    marker = " ★" if diff > 0 and offset > 0 else ""
                    print(f"      {label:>10} {avg:>+15,.0f} {diff:>+10,.0f}{marker}")
                    event_avgs[f"T{offset:+d}"] = round(avg)

            # ═══════════════════════════════════════════════
            # 분석 4: 누적 효과 (T+1~T+3 합산)
            # ═══════════════════════════════════════════════
            cum_after = []
            for evt_dt in events:
                evt_pos = merged.index.get_loc(evt_dt)
                cum = 0
                for d in range(1, 4):  # T+1, T+2, T+3
                    if evt_pos + d < len(merged):
                        cum += merged["inst_net_stk"].iloc[evt_pos + d]
                cum_after.append(cum)

            # 비이벤트일 대조군
            non_events = merged[merged["retail_net_est"] < q90].index
            cum_control = []
            for dt in non_events:
                pos = merged.index.get_loc(dt)
                cum = 0
                for d in range(1, 4):
                    if pos + d < len(merged):
                        cum += merged["inst_net_stk"].iloc[pos + d]
                cum_control.append(cum)

            avg_cum_event = np.mean(cum_after) if cum_after else 0
            avg_cum_control = np.mean(cum_control) if cum_control else 0

            print(f"    ── T+1~T+3 기관 누적매수 ──")
            print(f"      ETF 대량유입 후:  {avg_cum_event:>+15,.0f}")
            print(f"      일반일 후:        {avg_cum_control:>+15,.0f}")
            diff_pct = ((avg_cum_event / avg_cum_control - 1) * 100
                        if avg_cum_control != 0 else 0)
            print(f"      차이:             {diff_pct:>+14.1f}%"
                  f"{'  ★ 유의미' if abs(diff_pct) > 30 else ''}")

            results.append({
                "etf": etf_name,
                "stock": stk_name,
                "days": len(merged),
                "lag_corrs": lag_corrs,
                "event_count": len(events),
                "event_profile": event_avgs,
                "cum_t1t3_event": round(avg_cum_event),
                "cum_t1t3_control": round(avg_cum_control),
                "cum_diff_pct": round(diff_pct, 1),
            })

    return {
        "sector": sector_name,
        "pairs": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=15,
                        help="스크래핑 페이지 수 (기본 15 ≈ 300일)")
    args = parser.parse_args()

    print("[ETF Flow X-Ray] ETF 매수 → 기초종목 수급 전달 분석")
    print("=" * 75)
    print(f"가설: ETF 개인매수 → AP 설정 → 기초종목 기관(금융투자) 매수")
    print(f"검증: ETF 개인순매수 vs 기초종목 기관순매수 상관관계")
    print(f"데이터: 네이버 금융 {args.pages}페이지 × 20행 ≈ {args.pages * 20}거래일")

    all_results = []
    for sector, config in ETF_STOCK_PAIRS.items():
        result = analyze_sector(sector, config, args.pages)
        if result:
            all_results.append(result)

    # ── 종합 요약 ──
    print(f"\n{'═' * 75}")
    print(f"  종합 요약 (v2: 시차+규모+이벤트스터디)")
    print(f"{'═' * 75}")

    all_pairs = []
    for r in all_results:
        all_pairs.extend(r.get("pairs", []))

    if all_pairs:
        # 시차별 상관 요약
        print(f"\n  ── 시차별 상관계수 평균 ──")
        for lag in range(4):
            key = f"T+{lag}"
            vals = [p["lag_corrs"].get(key, 0) for p in all_pairs
                    if "lag_corrs" in p]
            if vals:
                avg = np.mean(vals)
                best = max(vals)
                marker = " ★ BEST" if lag > 0 and avg > np.mean(
                    [p["lag_corrs"].get("T+0", 0) for p in all_pairs
                     if "lag_corrs" in p]
                ) else ""
                print(f"    {key}: 평균 {avg:+.4f}  (최대 {best:+.4f}){marker}")

        # T+1~T+3 누적 효과 요약
        print(f"\n  ── T+1~T+3 기관 누적매수 비교 ──")
        print(f"  {'ETF':>20} {'주식':>12} │"
              f" {'대량유입후':>12} {'일반일후':>12} {'차이':>8}")
        print(f"  {'─' * 20} {'─' * 12} ┼"
              f" {'─' * 12} {'─' * 12} {'─' * 8}")

        diffs = []
        for p in all_pairs:
            evt = p.get("cum_t1t3_event", 0)
            ctrl = p.get("cum_t1t3_control", 0)
            diff = p.get("cum_diff_pct", 0)
            marker = " ★" if diff > 30 else ""
            print(f"  {p['etf']:>20} {p['stock']:>12} │"
                  f" {evt:>+12,} {ctrl:>+12,} {diff:>+7.1f}%{marker}")
            diffs.append(diff)

        avg_diff = np.mean(diffs) if diffs else 0
        print(f"\n  ★ 평균 차이: {avg_diff:+.1f}%")
        if avg_diff > 30:
            print("  → ETF 대량유입 후 T+1~T+3 기관매수 유의미하게 증가!")
        elif avg_diff > 10:
            print("  → 약한 효과 존재: ETF 설정이 기초종목 기관매수에 일부 기여")
        else:
            print("  → 효과 미미 또는 없음")

    # 저장
    output = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": "v2",
        "pages_per_item": args.pages,
        "sectors": all_results,
        "summary": {
            "pairs_count": len(all_pairs),
            "avg_cum_diff_pct": round(avg_diff, 1) if all_pairs else None,
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  [저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
