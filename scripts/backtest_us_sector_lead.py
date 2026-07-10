# -*- coding: utf-8 -*-
"""규칙 C 백테스트 — "전일 미국 섹터 강약 → 당일 국내 대응 섹터" 선행성 심판.

배경 (2026-07-09, TRADING_PRINCIPLES_퀀트봇.md 규칙 C):
- 성공기 인물 루틴: "새벽 미국장 섹터 강약 보고 국내 시나리오 → 7시 전 예약주문".
- 아침 리포트를 배선하기 전에 이 선행성이 실재하는지 과거 데이터로 먼저 심판한다.

사전 명시 설계 (그리드 탐색 금지):
- 신호: KR 거래일 T의 직전 미국 거래일 U의 섹터 ETF 1일 수익률.
  강세 = 상위 3개 ETF, 약세 = 하위 3개 ETF (랭크 기반 — 임계값 튜닝 없음).
- 매핑: 아래 US_KR_MAP 고정 (KRX 29업종 대응, 작성 후 수정 금지).
- 실행: 국내 T일 시가 진입 → 종가 평가 (아침 예약주문 실행 관점 = 실행가능 시점).
  보조 지표로 종가→종가도 기록.
- 기저선: 같은 날 유니버스(보유 parquet 전 종목) 동일가중 시가→종가 평균.
  초과수익 = 대응 섹터 평균 − 유니버스 평균 (레짐 베타 제거).
- 통계: 날짜 단위 1관측(그날 예상 섹터들의 평균 초과수익) → t (중첩 없음).
- 판정 기준: ①강세측 평균 초과 > 0 & t ≥ 2 ②약세측 < 0 (방향 대칭)
             ③연도별 부호 일관 — 셋 다 충족 시에만 아침 리포트 '검증됨' 표기.

한계 (정직 고지):
- KR 섹터 = krx_full_sectors.json (현재 시점 구성) → 생존·소속 이동 편향 존재.
- 미국 ETF는 yfinance 수정종가 기준 1일 수익률.

★판정 (2026-07-09 실행 결과, n=1,919일): 기각.
- 강세측(핵심 가설: 미국 강세 섹터 → 국내 대응 섹터 롱): +0.008%p/일, t=+0.97,
  양수비율 51%, 연도별 부호 혼재 → 예측력 없음. 아침 예약주문 시나리오 근거 불가.
- 부산물: 약세측은 유의 (시가→종가 -0.025%p/일 t=-3.04, 종가→종가 -0.037%p t=-2.67,
  2020~2026 7년 연속 음수) — 단 크기가 슬리피지 이하라 단독 매매 근거는 못 되고
  신규진입 감점 태그 후보로만 기록.
- 결론: 규칙 C 아침 리포트 신규 cron 배선 보류. 약세 회피축만 기존 리포트 편입 검토.
- 7/10 검수 주석(사정거리): 이 기각은 "KRX 29업종 동일가중 + 이 매핑 테이블 + 시가진입"
  설계에 대한 것. 실무 가설의 원형(미국 섹터 강세→국내 '대형 주도주')은 동일가중 평균에
  희석돼 검정되지 않았음 — 대형주 가중 설계는 별도 사전등록 후보로 남김. 약세측 유의성은
  NW t-3.13·2026 제외 t-2.88로 견고 재확인됨.

실행:
    python -u -X utf8 scripts/backtest_us_sector_lead.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SECTORS_JSON = PROJECT_ROOT / "data" / "sector_rotation" / "krx_full_sectors.json"
OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"

US_ETFS = ["SMH", "XLK", "XLE", "XLF", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU", "XLC", "XTN"]

# 미국 섹터 ETF → KRX 업종 대응 (사전 고정 — 작성 후 수정 금지)
US_KR_MAP = {
    "SMH": ["전기·전자"],
    "XLK": ["IT 서비스", "전기·전자"],
    "XLE": ["화학"],
    "XLF": ["금융", "증권", "은행", "보험"],
    "XLV": ["제약", "의료·정밀기기"],
    "XLI": ["기계·장비", "운송장비·부품", "건설"],
    "XLB": ["금속", "화학", "비금속"],
    "XLY": ["유통", "오락·문화", "섬유·의류"],
    "XLP": ["음식료·담배"],
    "XLU": ["전기·가스"],
    "XLC": ["통신", "IT 서비스"],
    "XTN": ["운송·창고"],
}

TOP_N = 3  # 강세/약세 각 상·하위 ETF 수 (랭크 기반)


def load_us() -> pd.DataFrame:
    """미국 섹터 ETF 일별 수익률 (yfinance 수정종가).

    ★로컬 실행 주의: 프로젝트 경로에 한글이 있어 libcurl이 venv 내 CA파일을 못 읽음.
      ASCII 경로로 cacert.pem 복사 후 CURL_CA_BUNDLE 환경변수 지정 필요 (7/9 실측).
      (stooq는 JS 검증벽 도입으로 사용 불가 확인.)
    """
    import yfinance as yf

    px = yf.download(US_ETFS, start="2018-12-01", auto_adjust=True, progress=False)["Close"]
    ret = px.pct_change() * 100.0
    return ret.dropna(how="all")


def load_kr_sectors() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """섹터별 구성종목 + 전 종목 시가→종가/종가→종가 수익률 행렬."""
    sectors = json.load(open(SECTORS_JSON, encoding="utf-8"))
    sector_members = {name: [s["code"] for s in v["stocks"]] for name, v in sectors.items()}

    o2c, c2c = {}, {}
    for f in glob.glob(str(RAW_DIR / "*.parquet")):
        code = Path(f).stem
        try:
            df = pd.read_parquet(f, columns=["open", "close"])
        except Exception:
            continue
        df = df[(df.index >= START) & (df["open"] > 0) & (df["close"] > 0)]
        if len(df) < 100:
            continue
        o2c[code] = (df["close"] / df["open"] - 1) * 100.0
        c2c[code] = df["close"].pct_change() * 100.0
    o2c_df = pd.DataFrame(o2c)
    c2c_df = pd.DataFrame(c2c)
    return sector_members, o2c_df, c2c_df


def main() -> None:
    print("[1/4] 미국 섹터 ETF 다운로드")
    us = load_us()
    print(f"  {us.index[0].date()} ~ {us.index[-1].date()} ({len(us)}일, {us.shape[1]}ETF)")

    print("[2/4] 국내 섹터 수익률 행렬 구축")
    members, o2c, c2c = load_kr_sectors()
    print(f"  종목 {o2c.shape[1]}개, 거래일 {o2c.shape[0]}일")

    # 섹터별 동일가중 일일 수익률 (보유 parquet 교집합)
    sec_o2c, sec_c2c = {}, {}
    for name, codes in members.items():
        cols = [c for c in codes if c in o2c.columns]
        if len(cols) >= 3:
            sec_o2c[name] = o2c[cols].mean(axis=1)
            sec_c2c[name] = c2c[cols].mean(axis=1)
    sec_o2c = pd.DataFrame(sec_o2c)
    sec_c2c = pd.DataFrame(sec_c2c)
    mkt_o2c = o2c.mean(axis=1)  # 유니버스 기저선
    mkt_c2c = c2c.mean(axis=1)
    print(f"  섹터 {sec_o2c.shape[1]}개 (구성 3종목 미만 제외)")

    print("[3/4] 신호 정렬 (KR일 T ← 직전 미국일 U, lookahead 없음)")
    us_dates = us.index.normalize()
    rows = []
    for t in sec_o2c.index:
        pos = us_dates.searchsorted(t)  # t보다 앞선 마지막 미국일
        if pos == 0:
            continue
        u = us_dates[pos - 1]
        if (t - u).days > 5:
            continue  # 연휴 등 과도한 간격 제외
        day = us.loc[u].dropna()
        if len(day) < 8:
            continue
        ranked = day.sort_values(ascending=False)
        strong, weak = list(ranked.index[:TOP_N]), list(ranked.index[-TOP_N:])

        def kr_excess(etfs, ret_df, mkt):
            secs = sorted({s for e in etfs for s in US_KR_MAP.get(e, []) if s in ret_df.columns})
            if not secs or t not in ret_df.index:
                return None, []
            vals = ret_df.loc[t, secs].dropna()
            if vals.empty or pd.isna(mkt.get(t)):
                return None, []
            return float(vals.mean() - mkt[t]), secs

        s_o2c, s_secs = kr_excess(strong, sec_o2c, mkt_o2c)
        w_o2c, _ = kr_excess(weak, sec_o2c, mkt_o2c)
        s_c2c, _ = kr_excess(strong, sec_c2c, mkt_c2c)
        w_c2c, _ = kr_excess(weak, sec_c2c, mkt_c2c)
        if s_o2c is None:
            continue
        rows.append({
            "date": t, "year": t.year, "us_date": u,
            "strong_etfs": ",".join(strong), "strong_secs": ",".join(s_secs),
            "strong_o2c": s_o2c, "weak_o2c": w_o2c,
            "strong_c2c": s_c2c, "weak_c2c": w_c2c,
            "us_top_ret": float(ranked.iloc[0]),
        })

    df = pd.DataFrame(rows).set_index("date")
    df.to_parquet(OUT_DIR / "us_sector_lead_daily.parquet")
    print(f"  정렬된 관측일 {len(df)}일")

    print("[4/4] 판정")

    def t_stat(s: pd.Series) -> tuple[float, float, int]:
        s = s.dropna()
        n = len(s)
        if n < 10:
            return np.nan, np.nan, n
        return float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(n))), n

    pd.set_option("display.width", 200)
    print("\n===== ① 전 기간 (일 단위 초과수익 %p, 기저선=유니버스 평균) =====")
    for label, col in [("강세측 시가→종가", "strong_o2c"), ("약세측 시가→종가", "weak_o2c"),
                       ("강세측 종가→종가", "strong_c2c"), ("약세측 종가→종가", "weak_c2c")]:
        m, t, n = t_stat(df[col])
        wr = (df[col].dropna() > 0).mean()
        print(f"  {label:14s}: 평균 {m:+.3f}%p | t {t:+.2f} | 양수비율 {wr:.2f} | n={n}")

    print("\n===== ② 연도별 강세측 시가→종가 초과 (%p) — 견고성 =====")
    yearly = df.groupby("year")[["strong_o2c", "weak_o2c"]].agg(["mean", "count"])
    print(yearly.round(3).to_string())

    print("\n===== ③ 미국 강세폭 조건부 (참고 관측 — 판정엔 미사용) =====")
    big = df[df["us_top_ret"] >= 1.0]
    m, t, n = t_stat(big["strong_o2c"])
    print(f"  미국 1위 ETF +1%↑인 날만: 평균 {m:+.3f}%p | t {t:+.2f} | n={n}")

    print("\n판정 기준: 강세측 o2c 평균>0 & t≥2, 약세측<0, 연도별 부호 일관 — 셋 다 충족 시 채택")


if __name__ == "__main__":
    main()
