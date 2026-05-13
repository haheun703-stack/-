#!/usr/bin/env python3
"""전체 1,157종목에서 26개 섹터 테마주 전수 탐색
- pykrx 종목명 조회
- 키워드 매칭으로 섹터 분류
- 최근 6개월 15%+ 급등 이력 체크
"""
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = "data/raw"
START = "2025-11-01"
END = "2026-05-08"

# 종목명 조회
try:
    from pykrx import stock as pykrx_stock
    def get_name(code):
        try:
            return pykrx_stock.get_market_ticker_name(code)
        except:
            return ""
except:
    def get_name(code):
        return ""

# 섹터 키워드 매칭 (종목명 기반)
SECTOR_KEYWORDS = {
    "반도체": ["반도체", "실리콘", "웨이퍼", "파운드리", "패키징", "칩", "메모리",
              "하이닉스", "테크윙", "한미반도체", "리노공업", "이오테크", "주성엔지",
              "원익", "ISC", "HPSP", "제우스", "솔브레인", "동진쎄미", "켐트로닉스",
              "피에스케이", "하나마이크론", "네패스", "에이디테크", "코미코",
              "유진테크", "프로텍", "SFA", "에스에프에이", "디아이", "피엔에이치",
              "엘비세미콘", "DB하이텍", "예스티", "오킨스전자"],
    "건설": ["건설", "건영", "이앤씨", "E&C", "엔지니어링", "시공", "건축",
            "현대건설", "대우건설", "삼성E&A", "GS건설", "DL이앤씨"],
    "로봇": ["로봇", "로보", "자동화", "모션", "서보", "액추에이터",
            "에스피지", "셀바스", "뉴로메카", "로보스타", "레인보우", "두산로보",
            "고영", "TPC", "유진로봇"],
    "조선": ["조선", "해양", "선박", "중공업", "미포", "오션",
            "삼성중공업", "HD현대", "한화오션"],
    "태양광": ["태양", "솔라", "셀", "모듈", "신재생", "OCI",
             "한화솔루션", "현대에너지"],
    "풍력": ["풍력", "윈드", "터빈", "유니슨", "씨에스윈드"],
    "석유화학": ["석유", "화학", "케미칼", "정유", "에탄", "프로필렌", "나프타",
              "LG화학", "롯데케미", "SK이노", "S-OIL", "금호석유", "한화케미",
              "SKC", "대한유화", "효성화학"],
    "원전가스": ["원전", "원자력", "가스", "터빈", "발전기", "한전",
              "두산에너빌", "비에이치아이", "한전기술", "보일러"],
    "방산": ["방산", "국방", "방위", "무기", "미사일", "레이더", "군수",
           "LIG", "넥스원", "한화에어로", "현대로템", "풍산", "한국항공우주",
           "켄코아", "빅텍", "퍼스텍", "아이쓰리시스템"],
    "2차전지": ["2차전지", "배터리", "리튬", "양극재", "음극재", "전해질", "분리막",
              "에코프로", "포스코퓨처엠", "엘앤에프", "삼성SDI", "LG에너지",
              "성일하이텍", "코스모신소재", "나노신소재", "천보",
              "엔켐", "솔루스", "피엔티", "신흥에스이씨"],
    "통신": ["통신", "텔레콤", "SKT", "KT ", "LG유플", "5G", "네트워크"],
    "AI데이터센터": ["AI", "인공지능", "데이터센터", "서버", "GPU", "클라우드",
                 "네이버", "카카오", "NHN", "쿨링", "냉각"],
    "항공": ["항공", "에어", "아시아나", "대한항공", "티웨이", "진에어", "제주항공"],
    "우주스페이스": ["우주", "위성", "로켓", "스페이스", "쎄트렉", "솔탑",
                "이노스페이스", "AP위성", "컨텍", "인텔리안"],
    "전기전선": ["전기", "전선", "케이블", "전력", "배전", "송전",
              "LS", "일진전기", "제룡전기", "대한전선", "대원전선",
              "가온전선", "극동전선"],
    "광통신": ["광통신", "광케이블", "광섬유", "광모듈",
             "대한광통신", "옵티시스"],
    "변압기": ["변압기", "트랜스", "전력기기",
             "HD현대일렉", "효성중공업", "제룡전기", "일진전기"],
    "액침냉각": ["냉각", "쿨링", "액침", "히트파이프", "방열",
              "이건산업"],
    "철강": ["철강", "스틸", "제철", "POSCO", "포스코", "현대제철",
           "동국제강", "세아", "고려아연"],
    "증권": ["증권", "투자증권", "자산운용", "금융투자",
           "미래에셋", "삼성증권", "키움", "NH투자", "한국투자"],
    "은행": ["은행", "금융지주", "KB", "신한", "하나금융", "우리금융",
           "기업은행", "BNK"],
    "사이버보안": ["보안", "시큐리티", "사이버", "정보보호",
               "안랩", "이글루", "시큐아이", "파이오링크", "지니언스"],
    "화장품": ["화장품", "뷰티", "코스메", "아모레", "LG생활건강",
             "코스맥스", "한국콜마", "클리오", "아이패밀리"],
    "게임": ["게임", "엔터", "넥슨", "엔씨", "넷마블", "크래프톤",
           "카카오게임", "펄어비스", "컴투스", "위메이드", "스마일게이트"],
}

# 전체 종목 스캔
print("=== 전체 종목 스캔 ===")
all_tickers = []
for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".parquet"):
        continue
    ticker = fname.replace(".parquet", "")
    all_tickers.append(ticker)

print(f"총 {len(all_tickers)}종목")

# 종목명 조회 + 섹터 매칭
print("\n종목명 조회 중...")
results = {}  # sector -> [(ticker, name, surge_count, max_pct)]

ticker_names = {}
for i, ticker in enumerate(all_tickers):
    name = get_name(ticker)
    ticker_names[ticker] = name
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(all_tickers)} 완료")

print(f"종목명 조회 완료: {sum(1 for v in ticker_names.values() if v)}개 매칭")

# 급등 이력 체크
print("\n급등 이력 체크 중...")
surge_data = {}  # ticker -> {count, max_pct, dates}
for ticker in all_tickers:
    try:
        df = pd.read_parquet(f"{RAW_DIR}/{ticker}.parquet")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[df["close"] > 0]
        df = df[df["volume"] > 0]
        if len(df) < 10:
            continue
        if "price_change" in df.columns:
            df["pct"] = df["price_change"]
        else:
            df["pct"] = df["close"].pct_change() * 100

        mask = (df.index >= START) & (df.index <= END)
        dfp = df[mask]
        surges = dfp[dfp["pct"] >= 10]
        if len(surges) > 0:
            surge_data[ticker] = {
                "count_10": len(surges),
                "count_15": len(dfp[dfp["pct"] >= 15]),
                "max_pct": round(surges["pct"].max(), 1),
                "last_close": int(dfp.iloc[-1]["close"]) if len(dfp) > 0 else 0,
            }
    except Exception:
        pass

print(f"10%+ 급등 종목: {len(surge_data)}개")

# 섹터 분류
print("\n" + "=" * 80)
print("섹터별 종목 전수 리스트 (★=15%+급등, ●=10%+급등)")
print("=" * 80)

already_mapped = set()  # 이미 매핑된 종목 추적

for sector, keywords in SECTOR_KEYWORDS.items():
    matches = []
    for ticker, name in ticker_names.items():
        if not name:
            continue
        combined = name + ticker
        matched = False
        for kw in keywords:
            if kw in name or kw in combined:
                matched = True
                break
        if matched:
            sd = surge_data.get(ticker, {})
            c10 = sd.get("count_10", 0)
            c15 = sd.get("count_15", 0)
            mp = sd.get("max_pct", 0)
            lc = sd.get("last_close", 0)
            matches.append((ticker, name, c10, c15, mp, lc))
            already_mapped.add(ticker)

    if matches:
        # 급등 횟수 기준 정렬
        matches.sort(key=lambda x: (-x[3], -x[2], x[1]))
        print(f"\n{'─'*70}")
        print(f"  {sector} ({len(matches)}종목)")
        print(f"{'─'*70}")
        print(f"  {'코드':<8s} {'종목명':<16s} {'현재가':>10s} {'10%+':>4s} {'15%+':>4s} {'최대':>6s} 상태")
        for ticker, name, c10, c15, mp, lc in matches:
            mark = ""
            if c15 > 0:
                mark = f"★×{c15}"
            elif c10 > 0:
                mark = f"●×{c10}"
            print(f"  {ticker:<8s} {name:<16s} {lc:>10,} {c10:>4d} {c15:>4d} {mp:>+5.1f}% {mark}")


# 미분류 급등주
print(f"\n{'─'*70}")
print(f"  미분류 급등주 (10%+ 급등했으나 섹터 미매칭)")
print(f"{'─'*70}")
unmatched = []
for ticker, sd in surge_data.items():
    if ticker not in already_mapped and sd.get("count_15", 0) > 0:
        name = ticker_names.get(ticker, "")
        unmatched.append((ticker, name, sd["count_10"], sd["count_15"],
                         sd["max_pct"], sd.get("last_close", 0)))

unmatched.sort(key=lambda x: (-x[3], -x[2]))
print(f"  {'코드':<8s} {'종목명':<16s} {'현재가':>10s} {'10%+':>4s} {'15%+':>4s} {'최대':>6s}")
for ticker, name, c10, c15, mp, lc in unmatched[:50]:
    print(f"  {ticker:<8s} {name:<16s} {lc:>10,} {c10:>4d} {c15:>4d} {mp:>+5.1f}%")

# 통계 요약
print(f"\n{'='*80}")
print("요약")
print(f"{'='*80}")
total_sector = len(already_mapped)
total_surge = len(surge_data)
total_15 = sum(1 for sd in surge_data.values() if sd.get("count_15", 0) > 0)
print(f"  전체 종목: {len(all_tickers)}개")
print(f"  섹터 매칭: {total_sector}개")
print(f"  6개월 10%+ 급등: {total_surge}개")
print(f"  6개월 15%+ 급등: {total_15}개")
print(f"  섹터+15%급등: {sum(1 for t in already_mapped if surge_data.get(t, {}).get('count_15', 0) > 0)}개")

print("\n=== 스캔 완료 ===")
