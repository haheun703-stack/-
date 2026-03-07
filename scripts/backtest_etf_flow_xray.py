"""ETF Flow X-Ray: ETF 매수가 기초종목 수급에 어떻게 반영되는지 분석

가설: 개인이 섹터 ETF를 대량 매수하면,
  AP(지정참가회사=증권사)가 ETF 설정 과정에서 기초자산을 매수.
  이때 개별종목에서는 "기관(금융투자)" 매수로 기록됨.

검증: ETF 개인순매수(=-(기관+외국인)) vs 기초종목 기관순매수 상관관계.

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
    """한 섹터의 ETF↔기초종목 교차분석."""

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
            # 개인 순매수 추정 = -(기관 + 외국인)
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

    # 교차 상관분석
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

            if len(merged) < 10:
                continue

            # ── 핵심 상관관계 ──
            # 가설: ETF 개인매수(retail_net_est) ↔ 기초종목 기관매수(inst_net_stk)
            corr_retail_inst = merged["retail_net_est"].corr(merged["inst_net_stk"])

            # 대조군: ETF 기관매수 ↔ 기초종목 기관매수
            corr_inst_inst = merged["inst_net_etf"].corr(merged["inst_net_stk"])

            # 대조군: ETF 외국인매수 ↔ 기초종목 외국인매수
            corr_frgn_frgn = merged["frgn_net_etf"].corr(merged["frgn_net_stk"])

            # 추가: ETF 개인매수 ↔ 기초종목 외국인매수
            corr_retail_frgn = merged["retail_net_est"].corr(merged["frgn_net_stk"])

            print(f"\n  [{etf_name}] ↔ [{stk_name}] ({len(merged)}일)")
            print(f"    ★ ETF 개인매수  → 주식 기관매수: {corr_retail_inst:+.4f}")
            print(f"      ETF 기관매수  → 주식 기관매수: {corr_inst_inst:+.4f}")
            print(f"      ETF 외국인    → 주식 외국인:   {corr_frgn_frgn:+.4f}")
            print(f"      ETF 개인매수  → 주식 외국인:   {corr_retail_frgn:+.4f}")

            # 상위/하위 10% 분석
            q90 = merged["retail_net_est"].quantile(0.9)
            q10 = merged["retail_net_est"].quantile(0.1)

            top = merged[merged["retail_net_est"] >= q90]
            bot = merged[merged["retail_net_est"] <= q10]

            if len(top) >= 3 and len(bot) >= 3:
                top_stk_inst = top["inst_net_stk"].mean()
                bot_stk_inst = bot["inst_net_stk"].mean()
                avg_stk_inst = merged["inst_net_stk"].mean()

                print(f"    ETF 개인 상위10%일때 → 주식 기관 평균: {top_stk_inst:+,.0f}")
                print(f"    ETF 개인 하위10%일때 → 주식 기관 평균: {bot_stk_inst:+,.0f}")
                print(f"    전체 평균                           : {avg_stk_inst:+,.0f}")

            results.append({
                "etf": etf_name,
                "stock": stk_name,
                "days": len(merged),
                "corr_etf_retail_vs_stock_inst": round(corr_retail_inst, 4),
                "corr_etf_inst_vs_stock_inst": round(corr_inst_inst, 4),
                "corr_etf_frgn_vs_stock_frgn": round(corr_frgn_frgn, 4),
                "corr_etf_retail_vs_stock_frgn": round(corr_retail_frgn, 4),
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
    print(f"  종합 요약")
    print(f"{'═' * 75}")

    all_pairs = []
    for r in all_results:
        all_pairs.extend(r.get("pairs", []))

    if all_pairs:
        print(f"\n  {'ETF':>20} {'주식':>12} {'일수':>5} │"
              f" {'★ETF개인→주식기관':>18} {'ETF기관→주식기관':>16}"
              f" {'ETF외인→주식외인':>16}")
        print(f"  {'─' * 20} {'─' * 12} {'─' * 5} ┼"
              f" {'─' * 18} {'─' * 16} {'─' * 16}")

        corrs = []
        for p in all_pairs:
            print(f"  {p['etf']:>20} {p['stock']:>12} {p['days']:>5} │"
                  f" {p['corr_etf_retail_vs_stock_inst']:>+17.4f}"
                  f" {p['corr_etf_inst_vs_stock_inst']:>+15.4f}"
                  f" {p['corr_etf_frgn_vs_stock_frgn']:>+15.4f}")
            corrs.append(p["corr_etf_retail_vs_stock_inst"])

        avg_corr = np.mean(corrs)
        print(f"\n  ★ 핵심 상관계수 평균 (ETF개인→주식기관): {avg_corr:+.4f}")

        if avg_corr > 0.3:
            print("  → 강한 양의 상관: ETF 설정 메커니즘 확인!")
        elif avg_corr > 0.1:
            print("  → 약한 양의 상관: ETF 설정 효과 존재하나 다른 요인도 큼")
        elif avg_corr > -0.1:
            print("  → 상관 없음: ETF 설정이 기초종목 수급에 미치는 영향 미미")
        else:
            print("  → 음의 상관: 예상과 반대 (ETF와 개별종목 수급 분리)")

    # 저장
    output = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pages_per_item": args.pages,
        "sectors": all_results,
        "summary": {
            "pairs_count": len(all_pairs),
            "avg_corr_retail_inst": round(avg_corr, 4) if all_pairs else None,
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  [저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
