#!/usr/bin/env python3
"""v10d: 전수 확장 섹터맵 — 26개 섹터 200+ 종목
find_sector_stocks.py 스캔 + 사용자 직접 지정 종목 반영
데이터 보강 22종목 포함 (1,182개 parquet)
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

# ═══════════════════════════════════════════════════════════
# 전수 확장 섹터맵: 26개 섹터, 200+ 종목
# ═══════════════════════════════════════════════════════════
SECTOR_MAP = {
    "반도체": [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "042700",  # 한미반도체
        "058470",  # 리노공업
        "039030",  # 이오테크닉스
        "403870",  # HPSP
        "036930",  # 주성엔지니어링
        "240810",  # 원익IPS
        "095340",  # ISC
        "079370",  # 제우스
        "357780",  # 솔브레인
        "080580",  # 오킨스전자
        "080220",  # 제주반도체
        "046890",  # 서울반도체
        "200710",  # 에이디테크놀로지
        "330860",  # 네패스아크
        "033640",  # 네패스
        "432720",  # 퀄리타스반도체
        "036540",  # SFA반도체
        "084370",  # 유진테크
        "319660",  # 피에스케이
        "254490",  # 미래반도체
        "122640",  # 예스티
        "032940",  # 원익
        "399720",  # 가온칩스
        "089030",  # 테크윙
        "067310",  # 하나마이크론
        "053610",  # 프로텍
        "074600",  # 원익QnC
        "104830",  # 원익머트리얼즈
        "000990",  # DB하이텍
        "089010",  # 켐트로닉스
        "031980",  # 피에스케이홀딩스
        "030530",  # 원익홀딩스
        "110990",  # 디아이티
        "003160",  # 디아이
        "126730",  # 코칩
        "084850",  # 아이티엠반도체
        # 전수 스캔 추가
        "440110",  # 파두 (NPU)
        "092190",  # 서울바이오시스 (LED)
        "007810",  # 코리아써키트 (PCB)
        "327260",  # RF머트리얼즈
        "078600",  # 대주전자재료
        "353200",  # 대덕전자
        "007660",  # 이수페타시스
        "095610",  # 테스
        "222800",  # 심텍
        "356860",  # 티엘비
        "195870",  # 해성디에스
        "161580",  # 필옵틱스
        "101490",  # 에스앤에스텍
        "036810",  # 에프에스티
        "322310",  # 오로스테크놀로지
        "317330",  # 덕산테코피아
        "064290",  # 인텍플러스
        "093370",  # 후성
        "059090",  # 미코
        "092870",  # 엑시콘
        "086390",  # 유니테스트
        "090460",  # 비에이치
        "394280",  # 오픈엣지테크놀로지
        "218410",  # RFHIC
        "108320",  # LX세미콘
        # 데이터 보강
        "323350",  # 다원넥스뷰
        "429270",  # 시지트로닉스
        "396270",  # 넥스트칩
        "076610",  # 해성옵틱스
        "320000",  # 한울반도체
        "321260",  # 프로이천
        "265520",  # AP시스템
    ],
    "건설": [
        "000720",  # 현대건설
        "047040",  # 대우건설
        "006360",  # GS건설
        "375500",  # DL이앤씨
        "028050",  # 삼성E&A
        "038500",  # 삼표시멘트
        "002780",  # 진흥기업
        "452280",  # 한선엔지니어링
        "267270",  # HD건설기계
        "045100",  # 한양이엔지
        "013580",  # 계룡건설
        "005960",  # 동부건설
        "003070",  # 코오롱글로벌
        "294870",  # HDC현대산업개발
        "010780",  # 아이에스동서
    ],
    "로봇": [
        "454910",  # 두산로보틱스
        "277810",  # 레인보우로보틱스
        "090360",  # 로보스타
        "058610",  # 에스피지
        "108860",  # 셀바스AI/로보티즈
        "138360",  # 앤로보틱스
        "090710",  # 휴림로봇
        "098460",  # 고영
        "348340",  # 뉴로메카
        "056080",  # 유진로봇
        "232680",  # 라온로보틱스
        "459510",  # 나우로보틱스
        "117730",  # 티로보틱스
        "475400",  # 씨메스로보틱스
        "108490",  # 로보티즈
        "466100",  # 클로봇
        "484810",  # 티엑스알로보틱스
        "388720",  # 유일로보틱스
        "079900",  # 전진건설로봇
        "319400",  # 현대무벡스 (물류로봇)
        "048770",  # TPC로보틱스
    ],
    "조선": [
        "009540",  # HD한국조선해양
        "042660",  # 한화오션
        "010140",  # 삼성중공업
        "329180",  # HD현대중공업
        "010620",  # HD현대미포
        "322000",  # HD현대에너지솔루션
        "097230",  # HJ중공업
        "443060",  # HD현대마린솔루션
        "082740",  # 한화엔진
    ],
    "태양광": [
        "009830",  # 한화솔루션
        "010060",  # OCI홀딩스
        "456040",  # OCI
        "336260",  # 두산퓨얼셀
    ],
    "풍력": [
        "112610",  # 씨에스윈드
        "018000",  # 유니슨
    ],
    "석유화학": [
        "096770",  # SK이노베이션
        "051910",  # LG화학
        "011170",  # 롯데케미칼
        "011790",  # SKC
        "006650",  # 대한유화
        "024060",  # 흥구석유
        "005950",  # 이수화학
        "011780",  # 금호석유화학
    ],
    "원전가스": [
        "052690",  # 한전기술
        "034020",  # 두산에너빌리티
        "083650",  # 비에이치아이
        "130660",  # 한전산업
    ],
    "방산": [
        "012450",  # 한화에어로스페이스
        "079550",  # LIG디펜스앤에어로스페이스
        "047810",  # 한국항공우주
        "064350",  # 현대로템
        "103140",  # 풍산
        "274090",  # 켄코아에어로스페이스
        "010820",  # 퍼스텍
        "214430",  # 아이쓰리시스템
        "073490",  # LIG아큐버
        "005810",  # 풍산홀딩스
        "272210",  # 한화시스템
        "032820",  # 우리기술
        "347700",  # 스피어
        "488900",  # 비츠로넥스텍
        "289930",  # 웨이비스
        "474610",  # RF시스템즈
        "218410",  # RFHIC
        "037460",  # 삼지전자
        "361390",  # 제노코
        "484590",  # 삼양컴텍
        "484870",  # 엠앤씨솔루션
        "474170",  # 루미르
        "065450",  # 빅텍
        "095270",  # 웨이브일렉트로
        "448710",  # 코츠테크놀로지
        "005870",  # 휴니드
        "065170",  # 비엘팜텍
        "215090",  # 솔디펜스
        "221840",  # 하이즈항공
        "024740",  # 한일단조
        "014970",  # 삼륭물산
        "020760",  # 일진디스플
    ],
    "2차전지": [
        "373220",  # LG에너지솔루션
        "006400",  # 삼성SDI
        "247540",  # 에코프로비엠
        "086520",  # 에코프로
        "003670",  # 포스코퓨처엠
        "066970",  # 엘앤에프
        "365340",  # 성일하이텍
        "005070",  # 코스모신소재
        "121600",  # 나노신소재
        "348370",  # 엔켐
        "336370",  # 솔루스첨단소재
        "383310",  # 에코프로에이치엔
        "278280",  # 천보
        "243840",  # 신흥에스이씨
        "137400",  # 피엔티
        "307180",  # 아이엘
        "126340",  # 비나텍
        "006110",  # 삼아알미늄
        "089980",  # 상아프론테크
        "259630",  # 엠플러스
        "047310",  # 파워로직스
        "382900",  # 범한퓨얼셀
        "025900",  # 동화기업
        "452200",  # 민테크
    ],
    "통신": [
        "017670",  # SK텔레콤
        "010170",  # 대한광통신
        "025770",  # 한국정보통신
    ],
    "AI데이터센터": [
        "035420",  # 네이버
        "440110",  # 파두
        "307950",  # 현대오토에버
    ],
    "항공": [
        "003490",  # 대한항공
        "274090",  # 켄코아에어로스페이스
        "221840",  # 하이즈항공
    ],
    "우주스페이스": [
        "012450",  # 한화에어로스페이스
        "047810",  # 한국항공우주
        "099320",  # 쎄트렉아이
        "304100",  # 솔탑
        "478340",  # 나라스페이스테크놀로지
        "462350",  # 이노스페이스
        "451760",  # 컨텍
        "189300",  # 인텔리안테크
        "098120",  # 마이크로컨텍솔
        "211270",  # AP위성
        "065680",  # 우주일렉트로
    ],
    "전기전선": [
        "006260",  # LS
        "010120",  # LS ELECTRIC
        "267260",  # HD현대일렉트릭
        "298040",  # 효성중공업
        "033100",  # 제룡전기
        "001440",  # 대한전선
        "006340",  # 대원전선
        "000500",  # 가온전선
        "103590",  # 일진전기
        "012200",  # 계양전기
        "417200",  # LS머트리얼즈
        "322180",  # LS티라유텍
        "009470",  # 삼화전기
        "062040",  # 산일전기
        "060370",  # LS마린솔루션
        "017510",  # 세명전기
    ],
    "광통신": [
        "010170",  # 대한광통신
    ],
    "변압기": [
        "267260",  # HD현대일렉트릭
        "298040",  # 효성중공업
        "033100",  # 제룡전기
        "103590",  # 일진전기
    ],
    "액침냉각": [
        "067170",  # 오텍
        "004710",  # 한솔테크닉스
    ],
    "철강": [
        "005490",  # 포스코홀딩스
        "004020",  # 현대제철
        "010130",  # 고려아연
        "092790",  # 넥스틸
        "001430",  # 세아베스틸지주
    ],
    "증권": [
        "006800",  # 미래에셋증권
        "016360",  # 삼성증권
        "039490",  # 키움증권
        "005940",  # NH투자증권
        "100790",  # 미래에셋벤처투자
        "001510",  # SK증권
        "003530",  # 한화투자증권
        "003470",  # 유안타증권
    ],
    "은행": [
        "105560",  # KB금융
        "055550",  # 신한지주
        "086790",  # 하나금융지주
        "316140",  # 우리금융지주
    ],
    "사이버보안": [
        "053800",  # 안랩
        "215090",  # 솔디펜스
    ],
    "화장품": [
        "090430",  # 아모레퍼시픽
        "051900",  # LG생활건강
        "192820",  # 코스맥스
        "222040",  # 코스맥스엔비티
    ],
    "게임": [
        "036570",  # 엔씨소프트
        "251270",  # 넷마블
        "259960",  # 크래프톤
        "293490",  # 카카오게임즈
        "263750",  # 펄어비스
    ],
}

ALL_SECTOR_TICKERS = set()
TICKER_TO_SECTORS = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        ALL_SECTOR_TICKERS.add(t)
        if t not in TICKER_TO_SECTORS:
            TICKER_TO_SECTORS[t] = []
        TICKER_TO_SECTORS[t].append(sector)

print(f"=== v10d 전수 확장 섹터 유니버스: {len(ALL_SECTOR_TICKERS)}개 종목 ===")
for sector, tickers in SECTOR_MAP.items():
    print(f"  {sector:<10s}: {len(tickers)}종목")

# 종목명
try:
    from pykrx import stock as pykrx_stock
    def get_name(code):
        try: return pykrx_stock.get_market_ticker_name(code)
        except: return code
except:
    def get_name(code): return code

# 데이터 로딩
print("\n=== 데이터 로딩 ===")
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

loaded = sum(1 for t in ALL_SECTOR_TICKERS if t in stock_data)
print(f"총 {len(stock_data)}종목, 섹터 {loaded}/{len(ALL_SECTOR_TICKERS)}개 매칭")
missing = ALL_SECTOR_TICKERS - set(stock_data.keys())
if missing:
    print(f"  미매칭: {missing}")


def run_sim(signal_pct=10.0, pullback_from_peak=0.07,
            watch_window=15, take_profit=0.10,
            add_threshold=-0.10, max_adds=3,
            max_hold=20, position_pct=0.20, max_positions=5,
            exclude_sectors=None):

    universe = set()
    for t in ALL_SECTOR_TICKERS:
        if exclude_sectors:
            secs = TICKER_TO_SECTORS.get(t, [])
            if all(s in exclude_sectors for s in secs):
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
                    "adds": pos["adds"],
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
                    "adds": pos["adds"],
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
                    "signal_date": today,
                    "signal_close": ev["close"],
                    "peak": ev["close"],
                    "watch_days": 0,
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
        "events": ev_count, "trades": len(sells),
        "wins": len(wins), "loses": len(sells) - len(wins),
        "total_profit": total, "final": final, "mdd": mdd,
        "sells": sells,
    }


def pr(label, r):
    if r is None:
        print(f"{label:<45s} N/A")
        return
    wr = f"{r['wins']/r['trades']*100:.1f}%" if r["trades"] > 0 else "N/A"
    ret = (r['final']/CAPITAL-1)*100
    print(f"{label:<45s} {r['events']:5d}ev {r['trades']:4d}건 {wr:>6s} "
          f"{r['total_profit']:>+14,}원 ({ret:>+6.1f}%) "
          f"MDD {r['mdd']:>+5.1f}%")


print("\n" + "=" * 80)
print(f"v10d: 전수 확장 섹터맵 — {START} ~ {END}")
print(f"      26개 섹터, {len(ALL_SECTOR_TICKERS)}개 종목")
print("=" * 80)

# ── Phase 1: 핵심 그리드 ──
print("\n--- Phase 1: 시그널 x 눌림 그리드 ---")
for sig in [10, 15, 20]:
    for pb in [0.05, 0.07, 0.10, 0.15]:
        r = run_sim(signal_pct=sig, pullback_from_peak=pb)
        pr(f"{sig}%급등 -> {int(pb*100)}%눌림", r)
    print()

# ── Phase 2: 감시기간 (15%급등→10%눌림) ──
print("\n--- Phase 2: 감시기간별 (15%급등->10%눌림) ---")
for ww in [3, 5, 7, 10, 15, 20]:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10, watch_window=ww)
    pr(f"감시 {ww}일", r)

# ── Phase 3: 매매 파라미터 ──
print("\n--- Phase 3: 매매 파라미터 (15%급등->10%눌림) ---")
params = [
    ("기본 TP10% 물3회 20일", 0.10, -0.10, 3, 20),
    ("TP 5%", 0.05, -0.10, 3, 20),
    ("TP 15%", 0.15, -0.10, 3, 20),
    ("물타기없음 TP5%", 0.05, -0.99, 0, 20),
    ("물타기없음 TP10%", 0.10, -0.99, 0, 20),
    ("물타기없음 TP15%", 0.15, -0.99, 0, 20),
    ("물타기1회 TP10%", 0.10, -0.10, 1, 20),
    ("보유 10일", 0.10, -0.10, 3, 10),
    ("보유 30일", 0.10, -0.10, 3, 30),
]
for label, tp, at, ma, mh in params:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10,
                take_profit=tp, add_threshold=at, max_adds=ma, max_hold=mh)
    pr(label, r)

# ── Phase 4: 포지션 사이즈 ──
print("\n--- Phase 4: 포지션 사이즈 ---")
for label, pp, mp in [("10%x5종목", 0.10, 5), ("10%x10종목", 0.10, 10),
                       ("15%x5종목", 0.15, 5), ("20%x5종목", 0.20, 5),
                       ("20%x3종목", 0.20, 3), ("5%x10종목", 0.05, 10),
                       ("5%x20종목", 0.05, 20)]:
    r = run_sim(signal_pct=15, pullback_from_peak=0.10,
                take_profit=0.10, add_threshold=-0.10, max_adds=3,
                position_pct=pp, max_positions=mp)
    pr(label, r)

# ── Phase 5: v10c 최고 조합 vs v10d ──
print("\n--- Phase 5: v10c 최고 조합 (확장 유니버스) ---")
best_combos = [
    ("15%급등->10%눌림 기본", 15, 0.10, 0.10, -0.10, 3, 20, 0.20, 5),
    ("20%급등->10%눌림", 20, 0.10, 0.10, -0.10, 3, 20, 0.20, 5),
    ("감시3일 15%->10%", 15, 0.10, 0.10, -0.10, 3, 3, 0.20, 5),
    ("TP5% 15%->10%", 15, 0.10, 0.05, -0.10, 3, 20, 0.20, 5),
    ("10%x10 분산", 15, 0.10, 0.10, -0.10, 3, 20, 0.10, 10),
]
for label, sig, pb, tp, at, ma, ww_or_mh, pp, mp in best_combos:
    if "감시" in label:
        r = run_sim(signal_pct=sig, pullback_from_peak=pb,
                    take_profit=tp, add_threshold=at, max_adds=ma,
                    watch_window=ww_or_mh, position_pct=pp, max_positions=mp)
    else:
        r = run_sim(signal_pct=sig, pullback_from_peak=pb,
                    take_profit=tp, add_threshold=at, max_adds=ma,
                    max_hold=ww_or_mh, position_pct=pp, max_positions=mp)
    pr(label, r)

# ── Phase 6: TOP 조합 상세 ──
print("\n" + "=" * 80)
print("Phase 6: TOP 조합 상세 거래내역 (15%급등->10%눌림 기본)")
print("=" * 80)

r = run_sim(signal_pct=15, pullback_from_peak=0.10)
if r:
    sells = r["sells"]
    wins = [t for t in sells if t["profit"] > 0]
    loses = [t for t in sells if t["profit"] <= 0]
    wr = f"{len(wins)/len(sells)*100:.1f}%" if sells else "N/A"
    ret = (r['final']/CAPITAL-1)*100

    print(f"  거래 {len(sells)}건, 승률 {wr}, "
          f"손익 {r['total_profit']:+,}원 ({ret:+.1f}%), "
          f"MDD {r['mdd']:.1f}%")

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

    # TOP 수익 종목
    ticker_stats = {}
    for t in sells:
        tk = t["ticker"]
        if tk not in ticker_stats:
            ticker_stats[tk] = {"name": t["name"], "n": 0, "w": 0, "p": 0,
                                "sectors": t["sectors"]}
        ticker_stats[tk]["n"] += 1
        ticker_stats[tk]["p"] += t["profit"]
        if t["profit"] > 0:
            ticker_stats[tk]["w"] += 1

    print(f"\n  === TOP 15 수익 종목 ===")
    top15 = sorted(ticker_stats.items(), key=lambda x: x[1]["p"], reverse=True)[:15]
    for tk, d in top15:
        wr3 = f"{d['w']/d['n']*100:.0f}%" if d["n"] > 0 else "N/A"
        nm = d["name"][:10] if len(d["name"]) > 10 else d["name"]
        print(f"    {tk} {nm:<12s} {d['n']:2d}건 {wr3:>4s} {d['p']:>+10,}원 ({d['sectors'][:25]})")

    print(f"\n  === WORST 10 손실 종목 ===")
    worst10 = sorted(ticker_stats.items(), key=lambda x: x[1]["p"])[:10]
    for tk, d in worst10:
        wr3 = f"{d['w']/d['n']*100:.0f}%" if d["n"] > 0 else "N/A"
        nm = d["name"][:10] if len(d["name"]) > 10 else d["name"]
        print(f"    {tk} {nm:<12s} {d['n']:2d}건 {wr3:>4s} {d['p']:>+10,}원 ({d['sectors'][:25]})")

    if wins:
        print(f"\n  승리 평균: +{sum(t['profit'] for t in wins)/len(wins):,.0f}원")
    if loses:
        print(f"  패배 평균: {sum(t['profit'] for t in loses)/len(loses):,.0f}원")


# ── Phase 7: v10c 대비 신규 종목 기여도 ──
print("\n" + "=" * 80)
print("Phase 7: v10c → v10d 신규 종목 기여도")
print("=" * 80)

# v10c에 있었던 종목들
v10c_tickers = {
    "005930", "000660", "042700", "058470", "039030", "403870", "036930",
    "240810", "095340", "047050", "079370", "357780", "141080",
    "000720", "047040", "006360", "375500", "028050",
    "454910", "277810", "090360", "012450", "058610", "108860",
    "009540", "042660", "010140", "329180", "010620",
    "009830", "010060", "112610", "018000",
    "096770", "010950", "051910", "011170", "011790",
    "052690", "034020", "083650",
    "079550", "047810", "064350", "103140", "272110",
    "373220", "006400", "247540", "086520", "003670", "066570",
    "365340", "005070", "121600",
    "017670", "030200", "032640", "035420", "035720",
    "003490", "099190", "304100",
    "006260", "010120", "267260", "298040",
    "033100", "001440", "006340", "000500", "103590",
    "010170",
    "005490", "004020", "001230",
    "006800", "016360", "039490", "005940",
    "105560", "055550", "086790", "316140",
    "053800",
    "090430", "051900", "192820",
    "036570", "251270", "259960", "293490", "263750",
}

r = run_sim(signal_pct=15, pullback_from_peak=0.10)
if r:
    v10c_trades = [t for t in r["sells"] if t["ticker"] in v10c_tickers]
    new_trades = [t for t in r["sells"] if t["ticker"] not in v10c_tickers]

    v10c_profit = sum(t["profit"] for t in v10c_trades)
    new_profit = sum(t["profit"] for t in new_trades)

    print(f"\n  v10c 종목: {len(v10c_trades)}건, 손익 {v10c_profit:+,}원")
    print(f"  v10d 신규: {len(new_trades)}건, 손익 {new_profit:+,}원")
    print(f"  합계:      {len(r['sells'])}건, 손익 {r['total_profit']:+,}원")

    if new_trades:
        print(f"\n  === 신규 종목 TOP 거래 (수익순) ===")
        new_sorted = sorted(new_trades, key=lambda x: x["profit"], reverse=True)
        for t in new_sorted[:20]:
            mark = "+" if t["profit"] > 0 else "-"
            nm = t['name'][:10]
            print(f"    {mark} {t['date'].strftime('%m/%d')} {t['ticker']} "
                  f"{nm:<12s} {t['profit']:>+10,}원 {t['hd']:2d}일 "
                  f"({t['sectors'][:30]})")

print("\n=== v10d 완료 ===")
