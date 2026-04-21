#!/usr/bin/env python3
"""연기금+금융투자 수급 → 단타봇 공유 JSON 내보내기

퀀트봇 BAT-D에서 collect_investor_bulk 후 실행.
단타봇 data_store에 JSON 저장 → 단타봇이 바로 읽기 가능.

출력:
  /home/ubuntu/bodyhunter/scalper-agent/data_store/quant_investor_extra.json

Usage:
    python scripts/export_investor_for_scalper.py
    python scripts/export_investor_for_scalper.py --days 20  # 최근 20거래일
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
SCALPER_OUT = Path("/home/ubuntu/bodyhunter/scalper-agent/data_store/quant_investor_extra.json")


def main():
    parser = argparse.ArgumentParser(description="연기금+금융투자 → 단타봇 JSON")
    parser.add_argument("--days", type=int, default=10, help="최근 N거래일 (기본 10)")
    args = parser.parse_args()

    if not DB_PATH.exists():
        logger.error("DB 없음: %s", DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    cur = conn.cursor()

    # 최근 N거래일 날짜 목록
    cur.execute(
        "SELECT DISTINCT date FROM investor_daily WHERE investor='연기금' ORDER BY date DESC LIMIT ?",
        (args.days,),
    )
    dates = sorted(r[0] for r in cur.fetchall())

    if not dates:
        logger.error("연기금 데이터 없음")
        sys.exit(1)

    min_date, max_date = dates[0], dates[-1]

    # 연기금 + 금융투자 데이터 추출
    cur.execute(
        """SELECT date, ticker, name, investor, net_val
           FROM investor_daily
           WHERE investor IN ('연기금', '금융투자')
             AND date >= ?
           ORDER BY date, ticker""",
        (min_date,),
    )
    rows = cur.fetchall()

    # {ticker: {date: {pension_net, finance_net}}} 구조
    data = {}
    names = {}
    for date, ticker, name, investor, net_val in rows:
        if ticker not in data:
            data[ticker] = {}
            names[ticker] = name
        if date not in data[ticker]:
            data[ticker][date] = {"pension_net": 0, "finance_net": 0}

        key = "pension_net" if investor == "연기금" else "finance_net"
        data[ticker][date][key] = round(net_val / 1e8, 1)  # 억원 단위

    # 누적 순매수 TOP (전체 기간)
    cur.execute(
        """SELECT ticker, name, investor, SUM(net_val) as total
           FROM investor_daily
           WHERE investor IN ('연기금', '금융투자')
           GROUP BY ticker, investor
           ORDER BY total DESC""",
    )
    cumulative = {}
    for ticker, name, investor, total in cur.fetchall():
        if ticker not in cumulative:
            cumulative[ticker] = {"name": name, "pension_total": 0, "finance_total": 0}
        key = "pension_total" if investor == "연기금" else "finance_total"
        cumulative[ticker][key] = round(total / 1e8, 1)

    conn.close()

    # JSON 구성
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "quantum-master/investor_daily.db",
        "period": {"from": min_date, "to": max_date, "days": len(dates)},
        "investors": ["연기금", "금융투자"],
        "unit": "억원",
        "daily": {},
        "cumulative_top30": {},
    }

    # daily: {ticker: {name, dates: {date: {pension_net, finance_net}}}}
    for ticker in data:
        output["daily"][ticker] = {
            "name": names.get(ticker, ""),
            "dates": data[ticker],
        }

    # cumulative_top30: 연기금+금융투자 합산 TOP 30
    combined = []
    for ticker, v in cumulative.items():
        combined.append({
            "ticker": ticker,
            "name": v["name"],
            "pension_total": v["pension_total"],
            "finance_total": v["finance_total"],
            "combined_total": round(v["pension_total"] + v["finance_total"], 1),
        })
    combined.sort(key=lambda x: x["combined_total"], reverse=True)
    output["cumulative_top30"] = combined[:30]

    # 저장
    out_path = SCALPER_OUT if SCALPER_OUT.parent.exists() else PROJECT_ROOT / "data" / "quant_investor_extra.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("내보내기 완료: %s", out_path)
    logger.info("  기간: %s ~ %s (%d일)", min_date, max_date, len(dates))
    logger.info("  종목: %d개 / 레코드: %d건", len(data), len(rows))


if __name__ == "__main__":
    main()
