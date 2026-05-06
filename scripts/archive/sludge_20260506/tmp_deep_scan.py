#!/usr/bin/env python3
"""방산/바이오 섹터 종목별 심층 분석 — 매수 후보 스크리닝"""
import pandas as pd
import sqlite3, glob, json, os

db = sqlite3.connect("data/investor_flow/investor_daily.db")

all_dates = [r[0] for r in db.execute("SELECT DISTINCT date FROM investor_daily ORDER BY date DESC LIMIT 20").fetchall()]
dates_5 = all_dates[:5]
dates_10 = all_dates[:10]

DEFENSE = ["012450","079550","272210","064350","047810","102260",
           "071970","042660","009540"]

BIO = ["207940","196170","298380","068760","326030","145020",
       "087010","226950","476830","328130","141080","314930",
       "086900","003060","128940","006280"]


def analyze_stock(ticker, sector_name):
    result = {"ticker": ticker, "sector": sector_name}

    nm_r = db.execute("SELECT DISTINCT name FROM investor_daily WHERE ticker=? LIMIT 1", (ticker,)).fetchone()
    if not nm_r:
        return None
    result["name"] = nm_r[0]

    matches = glob.glob("stock_data_daily/*_%s.csv" % ticker)
    if not matches:
        return None

    try:
        df = pd.read_csv(matches[0], encoding="utf-8-sig")
        if len(df) < 21:
            return None
        last = df.iloc[-1]
        if str(last.get("Date",""))[:10] != "2026-04-24":
            return None

        close = float(last["Close"])
        prev = float(df.iloc[-2]["Close"])
        result["close"] = close
        result["chg_1d"] = (close - prev) / prev * 100

        if len(df) >= 6:
            result["chg_5d"] = (close - float(df.iloc[-6]["Close"])) / float(df.iloc[-6]["Close"]) * 100
        else:
            result["chg_5d"] = 0
        if len(df) >= 21:
            result["chg_20d"] = (close - float(df.iloc[-21]["Close"])) / float(df.iloc[-21]["Close"]) * 100
        else:
            result["chg_20d"] = 0

        ma20 = float(last.get("MA20",0)) if pd.notna(last.get("MA20")) else 0
        ma60 = float(last.get("MA60",0)) if pd.notna(last.get("MA60")) else 0
        rsi = float(last.get("RSI",50)) if pd.notna(last.get("RSI")) else 50
        result["ma20_dev"] = (close-ma20)/ma20*100 if ma20>0 else 0
        result["ma60_dev"] = (close-ma60)/ma60*100 if ma60>0 else 0
        result["rsi"] = rsi

        vol = float(last.get("Volume",0))
        vol_5 = df["Volume"].iloc[-6:-1].mean() if len(df)>=6 else vol
        result["vol_ratio"] = vol/vol_5 if vol_5>0 else 1

        ub = float(last.get("Upper_Band",0)) if pd.notna(last.get("Upper_Band")) else 0
        lb = float(last.get("Lower_Band",0)) if pd.notna(last.get("Lower_Band")) else 0
        result["bb_pos"] = (close - lb) / (ub - lb) * 100 if ub > lb > 0 else 50

        lookback = min(250, len(df))
        high_52w = df["High"].iloc[-lookback:].max()
        result["from_high"] = (close - high_52w) / high_52w * 100

        bullish_count = sum(1 for i in range(1, min(4, len(df))) if float(df.iloc[-i]["Close"]) > float(df.iloc[-i]["Open"]))
        result["bullish_3d"] = bullish_count

    except Exception:
        return None

    # 수급 분석
    investors = ["외국인", "기관합계", "연기금", "금융투자", "개인"]
    daily_flow = {}
    for inv in investors:
        date_str = ",".join(["'%s'" % d for d in dates_10])
        q = "SELECT date, net_val/1e8 FROM investor_daily WHERE ticker='%s' AND investor='%s' AND date IN (%s) ORDER BY date" % (ticker, inv, date_str)
        rows = db.execute(q).fetchall()
        daily_flow[inv] = {r[0]: r[1] for r in rows}

    for inv in investors:
        result[inv+"_5d"] = sum(daily_flow.get(inv, {}).get(d, 0) for d in dates_5)
        result[inv+"_10d"] = sum(daily_flow.get(inv, {}).get(d, 0) for d in dates_10)
        result[inv+"_1d"] = daily_flow.get(inv, {}).get(dates_5[0], 0)

    # 수급 반전: 최근 2일 vs 이전 3일
    for inv in ["외국인", "기관합계", "연기금"]:
        recent_2d = sum(daily_flow.get(inv, {}).get(d, 0) for d in dates_5[:2])
        prev_3d = sum(daily_flow.get(inv, {}).get(d, 0) for d in dates_5[2:5])
        result[inv+"_reversal"] = recent_2d - prev_3d
        result[inv+"_recent2d"] = recent_2d
        result[inv+"_prev3d"] = prev_3d

    # 연속 매수일
    for inv in ["외국인", "기관합계", "연기금"]:
        streak = 0
        for d in dates_10:
            val = daily_flow.get(inv, {}).get(d, 0)
            if val > 0:
                streak += 1
            else:
                break
        result[inv+"_streak"] = streak

    # 일별 추이
    lines = []
    for d in dates_5:
        fgn = daily_flow.get("외국인", {}).get(d, 0)
        inst = daily_flow.get("기관합계", {}).get(d, 0)
        pen = daily_flow.get("연기금", {}).get(d, 0)
        lines.append("%s: 외%+6.0f 기%+6.0f 연%+5.0f" % (d[-4:], fgn, inst, pen))
    result["flow_detail"] = "\n".join(lines)

    # supply_surge 포착 여부
    try:
        with open("data/supply_surge_20260424.json") as f:
            ss = json.load(f)
        for c in ss.get("buy_candidates", []):
            if c["ticker"] == ticker:
                result["surge_type"] = c["type"]
                result["surge_score"] = c["final_score"]
                break
    except Exception:
        pass

    # 종합 매수 스코어
    score = 0
    reasons = []

    if -5 <= result.get("ma20_dev",0) <= 5:
        score += 15
        reasons.append("MA20눌림%+.1f%%" % result["ma20_dev"])
    if 40 <= result.get("rsi",50) <= 60:
        score += 10
        reasons.append("RSI적정%.0f" % result["rsi"])

    fgn_rev = result.get("외국인_reversal", 0)
    inst_rev = result.get("기관합계_reversal", 0)
    if fgn_rev > 30:
        score += 15
        reasons.append("외인반전%+.0f억" % fgn_rev)
    elif fgn_rev > 0:
        score += 5
        reasons.append("외인소폭반전")
    if inst_rev > 30:
        score += 12
        reasons.append("기관반전%+.0f억" % inst_rev)
    elif inst_rev > 0:
        score += 4
        reasons.append("기관소폭반전")

    pen_5d = result.get("연기금_5d", 0)
    if pen_5d > 20:
        score += 10
        reasons.append("연기금매수%+.0f억" % pen_5d)
    elif pen_5d > 0:
        score += 3

    if result.get("외국인_1d", 0) > 0 and result.get("기관합계_1d", 0) > 0:
        score += 10
        reasons.append("당일쌍끌이")

    if result.get("vol_ratio", 1) >= 1.5:
        score += 5
        reasons.append("거래량%.1fx" % result["vol_ratio"])

    if "surge_type" in result:
        score += 8
        reasons.append("수급급변%s" % result.get("surge_type",""))

    fgn_str = result.get("외국인_streak", 0)
    if fgn_str >= 3:
        score += 8
        reasons.append("외인%d연속" % fgn_str)

    if result.get("chg_20d", 0) > 5:
        score += 5
        reasons.append("20일상승%+.0f%%" % result["chg_20d"])

    if result.get("ma20_dev", 0) > 15:
        score -= 10
        reasons.append("!!과열MA20%+.0f%%" % result["ma20_dev"])
    if result.get("rsi", 50) > 70:
        score -= 5
        reasons.append("!!RSI과매수%.0f" % result["rsi"])
    if result.get("개인_5d", 0) > 100:
        score -= 5
        reasons.append("!!개인과다%+.0f억" % result["개인_5d"])

    result["buy_score"] = score
    result["buy_reasons"] = ", ".join(reasons)
    return result


# === 실행 ===
print("=" * 90)
print("방산/바이오 종목별 심층 분석 — 매수 후보 스크리닝")
print("=" * 90)

all_results = []
for sector, tickers in [("방산", DEFENSE), ("바이오", BIO)]:
    for t in tickers:
        r = analyze_stock(t, sector)
        if r:
            all_results.append(r)

rdf = pd.DataFrame(all_results)

for sector in ["방산", "바이오"]:
    sdf = rdf[rdf["sector"] == sector].sort_values("buy_score", ascending=False)

    print("\n" + "=" * 90)
    print("[%s] 매수스코어 순위 (%d종목)" % (sector, len(sdf)))
    print("=" * 90)

    for rank, (_, r) in enumerate(sdf.iterrows(), 1):
        star = ""
        if r["buy_score"] >= 50:
            star = " >>> STRONG BUY"
        elif r["buy_score"] >= 35:
            star = " >> BUY"
        elif r["buy_score"] >= 20:
            star = " > WATCH"

        print("\n%d. %s %s (SCORE=%d)%s" % (rank, r["ticker"], r["name"], r["buy_score"], star))
        print("   기술: %s원 | MA20 %+.1f%% | MA60 %+.1f%% | RSI %.0f | BB %.0f%% | 고점 %.1f%%" % (
            "{:,.0f}".format(r["close"]), r["ma20_dev"], r["ma60_dev"], r["rsi"], r["bb_pos"], r["from_high"]))
        print("   모멘텀: 1일 %+.1f%% | 5일 %+.1f%% | 20일 %+.1f%% | 거래량 %.1fx | 양봉 %d/3일" % (
            r["chg_1d"], r.get("chg_5d",0), r.get("chg_20d",0), r["vol_ratio"], r["bullish_3d"]))
        print("   수급5일: 외%+6.0f 기%+6.0f 연%+5.0f 금투%+5.0f 개%+6.0f" % (
            r["외국인_5d"], r["기관합계_5d"], r["연기금_5d"], r["금융투자_5d"], r["개인_5d"]))
        print("   반전(2일-3일): 외%+.0f 기%+.0f 연%+.0f" % (
            r["외국인_reversal"], r["기관합계_reversal"], r.get("연기금_reversal",0)))
        print("   연속매수: 외 %d일 | 기 %d일 | 연 %d일" % (
            r["외국인_streak"], r["기관합계_streak"], r["연기금_streak"]))

        if "surge_type" in r and pd.notna(r.get("surge_type")):
            print("   *** supply_surge: %s %d점" % (r["surge_type"], r["surge_score"]))

        print("   근거: %s" % r["buy_reasons"])
        print("   --- 일별 수급(억) ---")
        for line in r["flow_detail"].split("\n"):
            print("   %s" % line)

db.close()
