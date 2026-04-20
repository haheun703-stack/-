"""NXT 수급 축적 후보 빠른 스캔 — DB 직접 조회.

DB 스키마: date, ticker, name, investor, sell_vol, buy_vol, net_vol, sell_val, buy_val, net_val
investor 컬럼: '외국인', '기관합계', '기타법인' 등 (long format)
"""
import sqlite3
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "investor_flow" / "investor_daily.db"
NXT_MASTER = PROJECT_ROOT / "data" / "nxt" / "nxt_master.json"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"


def main():
    if not DB_PATH.exists():
        print(f"[ERROR] DB 없음: {DB_PATH}")
        return

    db = sqlite3.connect(str(DB_PATH))

    # 최근 거래일
    dates = [r[0] for r in db.execute(
        "SELECT DISTINCT date FROM investor_daily ORDER BY date DESC LIMIT 10"
    ).fetchall()]
    last5 = dates[:5]
    print(f"기준 거래일: {last5}")

    # NXT 마스터
    nxt_tickers = set()
    if NXT_MASTER.exists():
        nxt = json.load(open(NXT_MASTER, encoding="utf-8"))
        nxt_tickers = set(nxt.get("tickers", []))
    print(f"NXT: {len(nxt_tickers)}종목")

    # 외국인 + 기관 net_val 로드 (최근 5거래일)
    placeholders = ",".join(["?"] * len(last5))
    rows_fgn = db.execute(f"""
        SELECT ticker, name, date, net_val
        FROM investor_daily
        WHERE investor = '외국인' AND date IN ({placeholders})
        ORDER BY ticker, date DESC
    """, last5).fetchall()

    rows_inst = db.execute(f"""
        SELECT ticker, date, net_val
        FROM investor_daily
        WHERE investor = '기관합계' AND date IN ({placeholders})
        ORDER BY ticker, date DESC
    """, last5).fetchall()

    # 종목별 그룹핑
    fgn_by_ticker = defaultdict(list)
    inst_by_ticker = defaultdict(list)
    names = {}

    for r in rows_fgn:
        fgn_by_ticker[r[0]].append(r[3])  # net_val, 최신 순
        names[r[0]] = r[1]

    for r in rows_inst:
        inst_by_ticker[r[0]].append(r[2])  # net_val, 최신 순

    # 수급 축적 스캔
    results = []
    for ticker in fgn_by_ticker:
        if nxt_tickers and ticker not in nxt_tickers:
            continue

        fgn_vals = fgn_by_ticker[ticker]
        inst_vals = inst_by_ticker.get(ticker, [])

        if len(fgn_vals) < 3 and len(inst_vals) < 3:
            continue

        # 외국인 연속 순매수 (억 단위)
        f_streak, f_cum = 0, 0.0
        for v in fgn_vals:
            if v and v > 0:
                f_streak += 1
                f_cum += v
            else:
                break

        # 기관 연속 순매수
        i_streak, i_cum = 0, 0.0
        for v in inst_vals:
            if v and v > 0:
                i_streak += 1
                i_cum += v
            else:
                break

        # 쌍끌이 (외인+기관 동시 양수)
        d_streak = 0
        for fv, iv in zip(fgn_vals, inst_vals):
            if fv and fv > 0 and iv and iv > 0:
                d_streak += 1
            else:
                break

        best = max(f_streak, i_streak)
        if best >= 3:
            score = 0
            if f_streak >= 3:
                score += f_streak * 6 + f_cum / 1e8 / 10
            if i_streak >= 3:
                score += i_streak * 6 + i_cum / 1e8 / 10
            if d_streak >= 2:
                score += d_streak * 8

            results.append({
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "f_streak": f_streak,
                "i_streak": i_streak,
                "d_streak": d_streak,
                "f_cum": round(f_cum / 1e8, 1),
                "i_cum": round(i_cum / 1e8, 1),
                "score": round(score, 1),
            })

    results.sort(key=lambda x: -x["score"])

    # 4/20 가격 정보 추가 (CSV)
    try:
        import pandas as pd
        for r in results[:30]:
            csv_candidates = list(CSV_DIR.glob(f"*_{r['ticker']}.csv"))
            if csv_candidates:
                try:
                    df = pd.read_csv(csv_candidates[0], encoding="utf-8-sig")
                    if "Close" in df.columns and len(df) >= 2:
                        close = int(df["Close"].iloc[-1])
                        prev = float(df["Close"].iloc[-2])
                        ret = round((close - prev) / prev * 100, 2) if prev > 0 else 0
                        r["close"] = close
                        r["ret_d0"] = ret
                except Exception:
                    pass
    except ImportError:
        pass

    # 출력
    print(f"\n{'='*90}")
    print(f"  NXT 수급 축적 후보 (3일+ 연속매수, ~{last5[0]} 기준)")
    print(f"  후보: {len(results)}종목")
    print(f"{'='*90}")

    for i, r in enumerate(results[:30], 1):
        s = ""
        if r["f_streak"] >= 3:
            s += f"외{r['f_streak']}일({r['f_cum']:+.0f}억) "
        if r["i_streak"] >= 3:
            s += f"기{r['i_streak']}일({r['i_cum']:+.0f}억) "
        if r["d_streak"] >= 2:
            s += f"쌍{r['d_streak']}일"

        close_str = f"{r.get('close', 0):>8,}원" if r.get("close") else "        -"
        ret_str = f"{r.get('ret_d0', 0):+.1f}%" if r.get("close") else "     -"
        print(f" {i:2d}. {r['name']:14s} ({r['ticker']}) {close_str} {ret_str:>7s}  {s:38s} 점수={r['score']:.0f}")

    if not results:
        print("  (후보 없음)")

    db.close()


if __name__ == "__main__":
    main()
