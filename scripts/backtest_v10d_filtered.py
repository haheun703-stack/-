#!/usr/bin/env python3
"""v10d-filtered: 248종목에 잡주 필터 적용
필터 기준:
A) 최소 주가 (5천/1만/3만)
B) 최소 평균 거래대금 (1억/5억/10억)
C) 상한가(30%) 비율 제한 — 급등 중 30% 비율이 높으면 잡주
D) 복합필터: 주가+거래대금+상한가비율
"""
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = "data/raw"
START = "2025-11-01"
END = "2026-05-08"
FEE_RATE = 0.00315
CAPITAL = 50_000_000

# ═══ v10d 섹터맵 (248종목) — 그대로 사용 ═══
SECTOR_MAP = {
    "반도체": [
        "005930","000660","042700","058470","039030","403870","036930",
        "240810","095340","079370","357780","080580","080220","046890",
        "200710","330860","033640","432720","036540","084370","319660",
        "254490","122640","032940","399720","089030","067310","053610",
        "074600","104830","000990","089010","031980","030530","110990",
        "003160","126730","084850","440110","092190","007810","327260",
        "078600","353200","007660","095610","222800","356860","195870",
        "161580","101490","036810","322310","317330","064290","093370",
        "059090","092870","086390","090460","394280","218410","108320",
        "323350","429270","396270","076610","320000","321260","265520",
    ],
    "건설": [
        "000720","047040","006360","375500","028050","038500","002780",
        "452280","267270","045100","013580","005960","003070","294870","010780",
    ],
    "로봇": [
        "454910","277810","090360","058610","108860","138360","090710",
        "098460","348340","056080","232680","459510","117730","475400",
        "108490","466100","484810","388720","079900","319400","048770",
    ],
    "조선": [
        "009540","042660","010140","329180","010620","322000","097230",
        "443060","082740",
    ],
    "태양광": ["009830","010060","456040","336260"],
    "풍력": ["112610","018000"],
    "석유화학": [
        "096770","051910","011170","011790","006650","024060","005950","011780",
    ],
    "원전가스": ["052690","034020","083650","130660"],
    "방산": [
        "012450","079550","047810","064350","103140","274090","010820",
        "214430","073490","005810","272210","032820","347700","488900",
        "289930","474610","218410","037460","361390","484590","484870",
        "474170","065450","095270","448710","005870","065170","215090",
        "221840","024740","014970","020760",
    ],
    "2차전지": [
        "373220","006400","247540","086520","003670","066970","365340",
        "005070","121600","348370","336370","383310","278280","243840",
        "137400","307180","126340","006110","089980","259630","047310",
        "382900","025900","452200",
    ],
    "통신": ["017670","010170","025770"],
    "AI데이터센터": ["035420","440110","307950"],
    "항공": ["003490","274090","221840"],
    "우주스페이스": [
        "012450","047810","099320","304100","478340","462350","451760",
        "189300","098120","211270","065680",
    ],
    "전기전선": [
        "006260","010120","267260","298040","033100","001440","006340",
        "000500","103590","012200","417200","322180","009470","062040",
        "060370","017510",
    ],
    "광통신": ["010170"],
    "변압기": ["267260","298040","033100","103590"],
    "액침냉각": ["067170","004710"],
    "철강": ["005490","004020","010130","092790","001430"],
    "증권": ["006800","016360","039490","005940","100790","001510","003530","003470"],
    "은행": ["105560","055550","086790","316140"],
    "사이버보안": ["053800","215090"],
    "화장품": ["090430","051900","192820","222040"],
    "게임": ["036570","251270","259960","293490","263750"],
}

ALL_SECTOR_TICKERS = set()
TICKER_TO_SECTORS = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        ALL_SECTOR_TICKERS.add(t)
        if t not in TICKER_TO_SECTORS:
            TICKER_TO_SECTORS[t] = []
        TICKER_TO_SECTORS[t].append(sector)

# 종목명
try:
    from pykrx import stock as pykrx_stock
    def get_name(code):
        try: return pykrx_stock.get_market_ticker_name(code)
        except: return code
except:
    def get_name(code): return code

# 데이터 로딩
print("=== 데이터 로딩 ===")
stock_data = {}
for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".parquet"):
        continue
    ticker = fname.replace(".parquet", "")
    try:
        df = pd.read_parquet(f"{RAW_DIR}/{fname}")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[df["close"] > 0].copy()
        df = df[df["volume"] > 0].copy()
        if len(df) < 30:
            continue
        if "price_change" in df.columns:
            df["pct"] = df["price_change"]
        else:
            df["pct"] = df["close"].pct_change() * 100
        stock_data[ticker] = df
    except Exception:
        pass

print(f"총 {len(stock_data)}종목 로딩")

# ═══ 종목별 필터 메트릭 계산 ═══
print("\n=== 필터 메트릭 계산 ===")
ticker_metrics = {}
for ticker in ALL_SECTOR_TICKERS:
    if ticker not in stock_data:
        continue
    df = stock_data[ticker]
    mask = (df.index >= START) & (df.index <= END)
    dfp = df[mask]
    if len(dfp) < 10:
        continue

    avg_price = dfp["close"].mean()
    last_price = dfp.iloc[-1]["close"]
    avg_volume = dfp["volume"].mean()

    # 거래대금 = close * volume (trading_value가 있으면 사용)
    if "trading_value" in dfp.columns and dfp["trading_value"].sum() > 0:
        avg_tv = dfp["trading_value"].mean()
    else:
        avg_tv = (dfp["close"] * dfp["volume"]).mean()

    # 급등 중 상한가(29.5%+) 비율
    surges = dfp[dfp["pct"] >= 15]
    limit_ups = dfp[dfp["pct"] >= 29.5]
    surge_count = len(surges)
    limit_ratio = len(limit_ups) / max(surge_count, 1)

    ticker_metrics[ticker] = {
        "avg_price": avg_price,
        "last_price": last_price,
        "avg_volume": avg_volume,
        "avg_tv": avg_tv,
        "surge_count": surge_count,
        "limit_ratio": limit_ratio,
        "name": get_name(ticker),
    }

print(f"메트릭 계산: {len(ticker_metrics)}종목")

# 메트릭 분포 출력
prices = sorted([m["last_price"] for m in ticker_metrics.values()])
tvs = sorted([m["avg_tv"] for m in ticker_metrics.values()])
lrs = sorted([m["limit_ratio"] for m in ticker_metrics.values()])
print(f"\n  주가 분포: min={prices[0]:,.0f} / 25%={prices[len(prices)//4]:,.0f} / "
      f"50%={prices[len(prices)//2]:,.0f} / 75%={prices[len(prices)*3//4]:,.0f} / max={prices[-1]:,.0f}")
print(f"  거래대금 분포(원): min={tvs[0]:,.0f} / 50%={tvs[len(tvs)//2]:,.0f} / max={tvs[-1]:,.0f}")
print(f"  상한가비율 분포: min={lrs[0]:.2f} / 50%={lrs[len(lrs)//2]:.2f} / max={lrs[-1]:.2f}")


def run_sim(signal_pct=10.0, pullback_from_peak=0.07,
            watch_window=15, take_profit=0.10,
            add_threshold=-0.10, max_adds=3,
            max_hold=20, position_pct=0.20, max_positions=5,
            min_price=0, min_avg_tv=0, max_limit_ratio=1.0):
    """필터 적용 눌림대기 시뮬"""

    # 필터 적용
    universe = set()
    for t in ALL_SECTOR_TICKERS:
        if t not in ticker_metrics:
            continue
        m = ticker_metrics[t]
        if m["last_price"] < min_price:
            continue
        if m["avg_tv"] < min_avg_tv:
            continue
        if m["limit_ratio"] > max_limit_ratio:
            continue
        universe.add(t)

    all_dates = set()
    for t in universe:
        if t in stock_data:
            all_dates.update(stock_data[t].index.tolist())
    all_dates = sorted(d for d in all_dates
                       if pd.Timestamp(START) <= d <= pd.Timestamp(END))
    if not all_dates:
        return None

    event_map = {}
    ev_count = 0
    for ticker in universe:
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        mask = (df.index >= START) & (df.index <= END)
        dfp = df[mask]
        for date, row in dfp.iterrows():
            if row["pct"] >= signal_pct:
                if date not in event_map:
                    event_map[date] = []
                event_map[date].append({
                    "ticker": ticker, "date": date,
                    "close": row["close"], "pct": row["pct"],
                })
                ev_count += 1

    cash = CAPITAL
    positions = {}
    watchlist = {}
    trades = []
    equity = []

    for today in all_dates:
        watch_done = []
        for ticker, w in list(watchlist.items()):
            if ticker in positions:
                watch_done.append(ticker)
                continue
            dfp = stock_data.get(ticker)
            if dfp is None or today not in dfp.index:
                w["watch_days"] += 1
                if w["watch_days"] >= watch_window:
                    watch_done.append(ticker)
                continue
            row = dfp.loc[today]
            w["watch_days"] += 1
            if row["high"] > w["peak"]:
                w["peak"] = row["high"]
            pullback_price = w["peak"] * (1 - pullback_from_peak)
            if row["low"] <= pullback_price and len(positions) < max_positions:
                ep = pullback_price
                amt = CAPITAL * position_pct
                sh = int(amt / ep)
                if sh > 0 and cash >= ep * sh:
                    cash -= ep * sh
                    positions[ticker] = {
                        "ed": today, "ep": ep, "sh": sh,
                        "cb": ep, "adds": 0, "iamt": amt,
                    }
                    watch_done.append(ticker)
                    continue
            if w["watch_days"] >= watch_window:
                watch_done.append(ticker)
        for t in watch_done:
            watchlist.pop(t, None)

        closed = []
        for ticker, pos in list(positions.items()):
            dfp = stock_data.get(ticker)
            if dfp is None or today not in dfp.index:
                continue
            row = dfp.loc[today]
            if row["close"] <= 0:
                continue
            avg = pos["cb"]
            if avg <= 0:
                closed.append(ticker)
                continue
            hd = (today - pos["ed"]).days
            if (row["high"] / avg - 1) >= take_profit:
                sp = avg * (1 + take_profit)
                gross = (sp - avg) * pos["sh"]
                fees = sp * pos["sh"] * FEE_RATE
                cash += sp * pos["sh"] - fees
                trades.append({
                    "date": today, "ticker": ticker, "name": get_name(ticker),
                    "result": "WIN", "profit": round(gross - fees), "hd": hd,
                    "sectors": ", ".join(TICKER_TO_SECTORS.get(ticker, [])),
                })
                closed.append(ticker)
                continue
            if pos["adds"] < max_adds and (row["low"] / avg - 1) <= add_threshold:
                ap = avg * (1 + add_threshold)
                if ap > 0:
                    ash = int(pos["iamt"] / ap)
                    if ash > 0 and cash >= ap * ash:
                        cash -= ap * ash
                        tc = pos["cb"] * pos["sh"] + ap * ash
                        pos["sh"] += ash
                        pos["cb"] = tc / pos["sh"]
                        pos["adds"] += 1
            if hd >= max_hold:
                sp = row["close"]
                gross = (sp - avg) * pos["sh"]
                fees = sp * pos["sh"] * FEE_RATE
                cash += sp * pos["sh"] - fees
                trades.append({
                    "date": today, "ticker": ticker, "name": get_name(ticker),
                    "result": "EXPIRE", "profit": round(gross - fees), "hd": hd,
                    "sectors": ", ".join(TICKER_TO_SECTORS.get(ticker, [])),
                })
                closed.append(ticker)
        for t in closed:
            positions.pop(t, None)

        if today in event_map:
            for ev in event_map[today]:
                ticker = ev["ticker"]
                if ticker in positions or ticker in watchlist:
                    continue
                watchlist[ticker] = {
                    "signal_date": today, "signal_close": ev["close"],
                    "peak": ev["close"], "watch_days": 0,
                }

        pv = cash
        for ticker, pos in positions.items():
            dfp = stock_data.get(ticker)
            if dfp is not None and today in dfp.index:
                pv += dfp.loc[today, "close"] * pos["sh"]
            else:
                pv += pos["cb"] * pos["sh"]
        equity.append({"date": today, "eq": pv})

    sells = [t for t in trades if t["result"] in ("WIN", "EXPIRE")]
    wins = [t for t in sells if t["profit"] > 0]
    total = sum(t["profit"] for t in sells)
    final = equity[-1]["eq"] if equity else CAPITAL
    peak_eq = 0
    mdd = 0
    for e in equity:
        if e["eq"] > peak_eq:
            peak_eq = e["eq"]
        dd = (e["eq"] / peak_eq - 1) * 100
        if dd < mdd:
            mdd = dd

    return {
        "events": ev_count, "trades": len(sells), "universe": len(universe),
        "wins": len(wins), "loses": len(sells) - len(wins),
        "total_profit": total, "final": final, "mdd": mdd, "sells": sells,
    }


def pr(label, r):
    if r is None:
        print(f"{label:<50s} N/A")
        return
    wr = f"{r['wins']/r['trades']*100:.1f}%" if r["trades"] > 0 else "N/A"
    ret = (r['final']/CAPITAL-1)*100
    uni = r.get('universe', '?')
    print(f"{label:<50s} [{uni:3d}종목] {r['events']:4d}ev {r['trades']:3d}건 {wr:>6s} "
          f"{r['total_profit']:>+12,}원 ({ret:>+6.1f}%) MDD{r['mdd']:>+5.1f}%")


print("\n" + "=" * 90)
print(f"v10d-filtered: 248종목 잡주 필터 테스트 — {START} ~ {END}")
print("=" * 90)

# ═══ 필터 A: 최소 주가 ═══
print("\n--- 필터A: 최소 주가 (15%급등->10%눌림) ---")
for mp in [0, 3000, 5000, 10000, 20000, 30000, 50000]:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10, min_price=mp)
    pr(f"주가 >= {mp:,}원", r)

# ═══ 필터 B: 최소 평균 거래대금 ═══
print("\n--- 필터B: 최소 평균 거래대금 (15%급등->10%눌림) ---")
for mtv in [0, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10]:
    lbl = f"{mtv/1e8:.0f}억" if mtv >= 1e8 else "0"
    r = run_sim(signal_pct=15, pullback_from_peak=0.10, min_avg_tv=mtv)
    pr(f"거래대금 >= {lbl}", r)

# ═══ 필터 C: 상한가 비율 제한 ═══
print("\n--- 필터C: 상한가비율 제한 (15%급등->10%눌림) ---")
for mlr in [1.0, 0.8, 0.6, 0.5, 0.3, 0.0]:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10, max_limit_ratio=mlr)
    pr(f"상한가비율 <= {mlr:.0%}", r)

# ═══ 필터 D: 복합 필터 그리드 ═══
print("\n--- 필터D: 복합 필터 (15%급등->10%눌림) ---")
combos = [
    ("주가5천+거래1억", 5000, 1e8, 1.0),
    ("주가5천+거래5억", 5000, 5e8, 1.0),
    ("주가1만+거래1억", 10000, 1e8, 1.0),
    ("주가1만+거래5억", 10000, 5e8, 1.0),
    ("주가1만+거래10억", 10000, 1e9, 1.0),
    ("주가1만+상한가50%이하", 10000, 0, 0.5),
    ("주가1만+거래5억+상한가50%", 10000, 5e8, 0.5),
    ("주가1만+거래5억+상한가30%", 10000, 5e8, 0.3),
    ("주가1만+거래10억+상한가50%", 10000, 1e9, 0.5),
    ("주가2만+거래5억", 20000, 5e8, 1.0),
    ("주가2만+거래10억", 20000, 1e9, 1.0),
    ("주가3만+거래10억", 30000, 1e9, 1.0),
    ("주가3만+거래10억+상한가30%", 30000, 1e9, 0.3),
    ("주가5만+거래10억", 50000, 1e9, 1.0),
]
for label, mp, mtv, mlr in combos:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10,
                min_price=mp, min_avg_tv=mtv, max_limit_ratio=mlr)
    pr(label, r)

# ═══ 필터 E: 최적 필터로 매매 파라미터 재테스트 ═══
print("\n--- 필터E: 최적 필터 + 매매 파라미터 ---")
# 위 결과에서 가장 좋은 필터를 여러개 시도
for flabel, mp, mtv, mlr in [
    ("F1(주가1만+거래5억)", 10000, 5e8, 1.0),
    ("F2(주가1만+거래10억)", 10000, 1e9, 1.0),
    ("F3(주가2만+거래10억)", 20000, 1e9, 1.0),
]:
    print(f"\n  -- {flabel} --")
    for plabel, sig, pb, tp, at, ma, mh in [
        ("15%급등->10%눌림 기본", 15, 0.10, 0.10, -0.10, 3, 20),
        ("감시3일", 15, 0.10, 0.10, -0.10, 3, 3),
        ("감시5일", 15, 0.10, 0.10, -0.10, 3, 5),
        ("TP5%", 15, 0.10, 0.05, -0.10, 3, 20),
        ("물없TP10%", 15, 0.10, 0.10, -0.99, 0, 20),
        ("물없TP5%", 15, 0.10, 0.05, -0.99, 0, 20),
        ("20%급등->10%눌림", 20, 0.10, 0.10, -0.10, 3, 20),
        ("보유30일", 15, 0.10, 0.10, -0.10, 3, 30),
    ]:
        r = run_sim(signal_pct=sig, pullback_from_peak=pb,
                    take_profit=tp, add_threshold=at, max_adds=ma,
                    max_hold=mh if "감시" not in plabel else 20,
                    watch_window=mh if "감시" in plabel else 15,
                    min_price=mp, min_avg_tv=mtv, max_limit_ratio=mlr)
        pr(f"  {plabel}", r)

# ═══ Phase 최종: 최적 필터 상세 ═══
print("\n" + "=" * 90)
print("Phase 최종: 최적 필터 상세 (주가1만+거래5억)")
print("=" * 90)

r = run_sim(signal_pct=15, pullback_from_peak=0.10,
            min_price=10000, min_avg_tv=5e8)
if r:
    sells = r["sells"]
    wins = [t for t in sells if t["profit"] > 0]
    loses = [t for t in sells if t["profit"] <= 0]
    wr = f"{len(wins)/len(sells)*100:.1f}%" if sells else "N/A"

    print(f"  유니버스: {r['universe']}종목")
    print(f"  거래 {len(sells)}건, 승률 {wr}, "
          f"손익 {r['total_profit']:+,}원 ({(r['final']/CAPITAL-1)*100:+.1f}%), "
          f"MDD {r['mdd']:.1f}%")

    # 포함/제외 종목 리스트
    included = set()
    for t in ALL_SECTOR_TICKERS:
        if t in ticker_metrics:
            m = ticker_metrics[t]
            if m["last_price"] >= 10000 and m["avg_tv"] >= 5e8:
                included.add(t)

    excluded = ALL_SECTOR_TICKERS - included
    print(f"\n  포함: {len(included)}종목, 제외: {len(excluded)}종목")

    print(f"\n  === 제외된 종목 TOP20 ===")
    exc_list = []
    for t in excluded:
        if t in ticker_metrics:
            m = ticker_metrics[t]
            exc_list.append((t, m["name"], m["last_price"], m["avg_tv"]))
    exc_list.sort(key=lambda x: x[2], reverse=True)
    for t, n, p, tv in exc_list[:20]:
        nm = n[:12] if len(n) > 12 else n
        print(f"    {t} {nm:<14s} 주가{p:>10,.0f} 거래대금{tv/1e8:>8.1f}억")

    # 섹터별
    sector_stats = {}
    for t in sells:
        for s in t.get("sectors", "").split(", "):
            s = s.strip()
            if not s: continue
            if s not in sector_stats:
                sector_stats[s] = {"w": 0, "n": 0, "p": 0}
            sector_stats[s]["n"] += 1
            sector_stats[s]["p"] += t["profit"]
            if t["profit"] > 0:
                sector_stats[s]["w"] += 1

    print(f"\n  === 섹터별 성과 ===")
    for s in sorted(sector_stats.keys(), key=lambda x: sector_stats[x]["p"], reverse=True):
        d = sector_stats[s]
        wr2 = f"{d['w']/d['n']*100:.0f}%" if d["n"] > 0 else "N/A"
        print(f"    {s:<14s} {d['n']:3d}건 승률{wr2:>4s} {d['p']:>+12,}원")

    # 전체 거래 시간순
    print(f"\n  === 전체 거래 (시간순) ===")
    for t in sorted(sells, key=lambda x: x["date"]):
        mark = "+" if t["profit"] > 0 else "-"
        nm = t['name'][:10]
        print(f"    {mark} {t['date'].strftime('%m/%d')} {t['ticker']} "
              f"{nm:<12s} {t['result']:6s} {t['profit']:>+10,}원 {t['hd']:2d}일 "
              f"({t['sectors'][:30]})")

print("\n=== v10d-filtered 완료 ===")
