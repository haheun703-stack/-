#!/usr/bin/env python3
"""연기금 포트폴리오 분석 — DB 기반 누적 순매수 TOP"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "investor_flow" / "investor_daily.db"

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

# 1) 전체 기간 누적 순매수 TOP 30
cur.execute("""
SELECT ticker, name,
       SUM(net_val) as total_net,
       COUNT(*) as days,
       SUM(CASE WHEN net_val > 0 THEN 1 ELSE 0 END) as buy_days
FROM investor_daily
WHERE investor = '연기금'
GROUP BY ticker
ORDER BY total_net DESC
LIMIT 30
""")
print("=== 연기금 누적 순매수 TOP 30 (243거래일) ===")
print(f"{'#':>3} | {'종목명':<14} | {'코드':<8} | {'누적(억)':>10} | {'매수일':>6} | {'총일':>4}")
print("-" * 62)
for i, (tk, nm, net, days, bd) in enumerate(cur.fetchall(), 1):
    print(f"{i:>3} | {nm:<14} | {tk:<8} | {net/1e8:>+10.0f} | {bd:>3}/{days:<3} | {days:>4}")

print()

# 2) 누적 순매도 TOP 15
cur.execute("""
SELECT ticker, name,
       SUM(net_val) as total_net,
       COUNT(*) as days,
       SUM(CASE WHEN net_val < 0 THEN 1 ELSE 0 END) as sell_days
FROM investor_daily
WHERE investor = '연기금'
GROUP BY ticker
ORDER BY total_net ASC
LIMIT 15
""")
print("=== 연기금 누적 순매도 TOP 15 ===")
print(f"{'#':>3} | {'종목명':<14} | {'코드':<8} | {'누적(억)':>10} | {'매도일':>6} | {'총일':>4}")
print("-" * 62)
for i, (tk, nm, net, days, sd) in enumerate(cur.fetchall(), 1):
    print(f"{i:>3} | {nm:<14} | {tk:<8} | {net/1e8:>+10.0f} | {sd:>3}/{days:<3} | {days:>4}")

print()

# 3) 최근 10일 매수 TOP 15
cur.execute("SELECT DISTINCT date FROM investor_daily WHERE investor='연기금' ORDER BY date DESC LIMIT 10")
dates = [r[0] for r in cur.fetchall()]
min_date, max_date = dates[-1], dates[0]

cur.execute("""
SELECT ticker, name,
       SUM(net_val) as total_net,
       SUM(CASE WHEN net_val > 0 THEN 1 ELSE 0 END) as buy_days
FROM investor_daily
WHERE investor = '연기금' AND date >= ?
GROUP BY ticker
ORDER BY total_net DESC
LIMIT 15
""", (min_date,))
print(f"=== 연기금 최근 10일 매수 TOP 15 ({min_date}~{max_date}) ===")
print(f"{'#':>3} | {'종목명':<14} | {'코드':<8} | {'순매수(억)':>10} | {'매수일':>6}")
print("-" * 55)
for i, (tk, nm, net, bd) in enumerate(cur.fetchall(), 1):
    print(f"{i:>3} | {nm:<14} | {tk:<8} | {net/1e8:>+10.0f} | {bd:>3}/10")

conn.close()
