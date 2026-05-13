"""FLOWX Supabase 테이블 데이터 신선도 확인"""
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/ubuntu/quantum-master")

try:
    from supabase import create_client
except ImportError:
    print("supabase not installed")
    sys.exit(1)

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_KEY", "")

if not url or not key:
    from dotenv import load_dotenv
    load_dotenv("/home/ubuntu/quantum-master/.env")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

if not url or not key:
    print("SUPABASE credentials not found")
    sys.exit(1)

sb = create_client(url, key)
today = datetime.now().strftime("%Y-%m-%d")
sep = "=" * 70

print(sep)
print("FLOWX DATA FRESHNESS CHECK -- " + today)
print(sep)

# Actual table names from flowx_uploader.py and scan scripts
quant_system_tables = [
    ("quant_jarvis", "date", "Jarvis Main (Row1,3~7)"),
    ("quant_scenario_dashboard", "date", "Scenario Dashboard"),
    ("quant_sector_momentum", "date", "Sector Momentum"),
]

scan_tables = [
    ("quant_fib_scanner", "date", "Fibonacci Scanner"),
    ("quant_alpha_scanner", "date", "Alpha Scanner (Nugget)"),
    ("quant_market_ranking", "date", "Market Ranking"),
    ("quant_bluechip_checkup", "date", "Bluechip Checkup"),
]

other_tables = [
    ("dashboard_crash_bounce", "date", "Crash Bounce"),
    ("sector_rotation", "date", "Sector Rotation"),
]

signal_tables = [
    ("etf_signals", "date", "ETF Signals"),
    ("china_flow", "date", "China Flow"),
    ("short_signals", "date", "Short Signals"),
    ("morning_briefings", "date", "Morning Briefing"),
    ("signals", "created_at", "Signals (tracker)"),
    ("scoreboard", "updated_at", "Scoreboard"),
]

trading_tables = [
    ("paper_trades", "created_at", "Paper Trades"),
]

results = []

def check_table(table_name, date_col, label):
    try:
        resp = sb.table(table_name).select(date_col).order(date_col, desc=True).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            latest = str(resp.data[0][date_col])[:10]
            fresh = "OK" if latest == today else "STALE"
            count_resp = sb.table(table_name).select("*", count="exact").limit(0).execute()
            cnt = count_resp.count if count_resp.count else "?"
            return (label, table_name, latest, fresh, cnt)
        else:
            return (label, table_name, "EMPTY", "FAIL", 0)
    except Exception as e:
        err = str(e)[:50]
        return (label, table_name, err, "ERROR", 0)

def print_section(title, tables):
    print("\n  --- " + title + " ---")
    fmt = "  {:<26} {:<28} {:<12} {:<6} {}"
    print(fmt.format("Label", "Table", "Latest", "Status", "Rows"))
    print("  " + "-" * 85)
    for tbl, col, lbl in tables:
        r = check_table(tbl, col, lbl)
        results.append(r)
        mark = "  " if r[3] == "OK" else ">>"
        print(mark + fmt.format(r[0], r[1], r[2], r[3], r[4]).strip())

print_section("Quant System (upload_flowx G4)", quant_system_tables)
print_section("Scan Scripts (G4/G4.5)", scan_tables)
print_section("Dashboard / Sector", other_tables)
print_section("Signal Tables", signal_tables)
print_section("Trading", trading_tables)

# Summary
ok = sum(1 for r in results if r[3] == "OK")
stale = sum(1 for r in results if r[3] == "STALE")
fail = sum(1 for r in results if r[3] in ("FAIL", "ERROR"))
total = len(results)

grade = "A" if ok == total else ("B" if stale <= 2 and fail == 0 else ("C" if fail <= 2 else "F"))

print("\n" + sep)
print("SUMMARY: {}/{} OK | {} STALE | {} FAIL | Grade: {}".format(ok, total, stale, fail, grade))
if stale > 0:
    print("  STALE:")
    for r in results:
        if r[3] == "STALE":
            print("    - {} ({}): last={}".format(r[0], r[1], r[2]))
if fail > 0:
    print("  FAIL/ERROR:")
    for r in results:
        if r[3] in ("FAIL", "ERROR"):
            print("    - {} ({}): {}".format(r[0], r[1], r[2]))
print(sep)
