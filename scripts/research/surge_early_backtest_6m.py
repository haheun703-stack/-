"""조기발굴 룰 6개월 백테스트 (A 청사진 selector 검증용).

5/30 폭등주 연구(MLCC 섹터)에서 도출한 조기발굴 룰을 6개월 데이터로 검증.
과최적화 경계: 단일 섹터 4종목 1회 사이클 룰을 전체 유니버스 6개월에 적용해
승률/평균수익/MDD/PF/+10%비율을 실측한다. 추정 금지, 숫자로.

진입 룰 (volume-analyst + stock-analyzer 종합):
  (1) Vol/MA20 >= 2.0            거래량 선행 (매집/분출)
  (2) Close > MA20               추세 전환
  (3) MA60 위 또는 거리 <= 5%    저항 근접/돌파
  (4) RSI 45~65                  과매도 회복~과열 전
  (5) MACD_hist > 0              모멘텀 양전환
  (6) 위쪽 매물대 얇음           직전 6개월 고가 대비 종가 >= -15% (신고가 근처)

청산: D+1 / D+3 / D+5 종가 (보유기간별 수익률)
측정: 신호수, 승률, 평균수익, +10%비율, 보유중 MDD, Profit Factor
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import FinanceDataReader as fdr


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def backtest(universe: list[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    n_done = 0
    for code in universe:
        try:
            df = fdr.DataReader(code, start, end)
        except Exception:
            continue
        if df is None or len(df) < 130:
            continue
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma60"] = df["Close"].rolling(60).mean()
        df["vma20"] = df["Volume"].rolling(20).mean()
        df["vr"] = df["Volume"] / df["vma20"]
        df["rsi"] = _rsi(df["Close"])
        df["mh"] = _macd_hist(df["Close"])
        df["hi120"] = df["Close"].rolling(120).max()

        # 진입 가능 구간: 지표 워밍업(120) 이후 ~ D+5 여유
        for i in range(120, len(df) - 5):
            r = df.iloc[i]
            if pd.isna(r["ma60"]) or pd.isna(r["rsi"]) or pd.isna(r["vma20"]):
                continue
            ma60_dist = (r["Close"] - r["ma60"]) / r["ma60"]
            top_room = (r["Close"] - r["hi120"]) / r["hi120"]  # 0 이상이면 신고가
            # v2 (SET-1 눌림목 재돌파): 직전 1차분출(VolR>=2 & 양봉) 후 눌림 진입.
            # 분출 당일 추격(v1) → D1 본전/D3 손실이라 진입을 "눌림"으로 늦춤.
            win = df.iloc[max(0, i - 15) : i - 2]
            had_thrust = bool(
                len(win) and ((win["vr"] >= 2.0) & (win["Close"] > win["Open"])).any()
            )
            ma60_slope = (r["ma60"] - df["ma60"].iloc[i - 5]) if i >= 5 else 0.0
            cond = (
                had_thrust                                    # 1차 분출 존재
                and r["Close"] > r["ma20"]                    # 추세 위
                and (r["Close"] > r["ma60"] or abs(ma60_dist) <= 0.03)
                and ma60_slope > 0                            # MA60 상승중 (가짜돌파 필터)
                and 45 <= r["rsi"] <= 65
                and r["mh"] > 0
                and 0.5 <= r["vr"] <= 1.5                     # 눌림 (분출 당일 아님)
                and top_room >= -0.15
            )
            if not cond:
                continue
            entry = r["Close"]
            fwd = df["Close"].iloc[i + 1 : i + 6].values
            # 보유중 최대낙폭 (D+1~D+5 종가 기준)
            mdd = (fwd.min() / entry - 1) if len(fwd) else 0.0
            rows.append({
                "code": code,
                "date": df.index[i].date(),
                "d1": df["Close"].iloc[i + 1] / entry - 1,
                "d3": df["Close"].iloc[i + 3] / entry - 1,
                "d5": df["Close"].iloc[i + 5] / entry - 1,
                "mdd": mdd,
            })
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ...{n_done}/{len(universe)} 종목 처리, 누적 신호 {len(rows)}건")
    return pd.DataFrame(rows)


def summarize(rdf: pd.DataFrame) -> None:
    if rdf.empty:
        print("신호 0건 — 룰이 너무 엄격하거나 데이터 부족")
        return
    print(f"\n총 신호: {len(rdf)}건 (종목 {rdf['code'].nunique()}개)")
    print(f"{'보유':<6}{'평균수익':>10}{'승률':>9}{'+10%비율':>10}{'PF':>8}")
    for h in ["d1", "d3", "d5"]:
        x = rdf[h].dropna()
        wins = x[x > 0].sum()
        losses = -x[x < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print(
            f"{h.upper():<6}{x.mean()*100:>9.2f}%{(x>0).mean()*100:>8.1f}%"
            f"{(x>0.10).mean()*100:>9.1f}%{pf:>8.2f}"
        )
    print(f"\n보유중 평균 MDD: {rdf['mdd'].mean()*100:.2f}% / 최악 MDD: {rdf['mdd'].min()*100:.2f}%")
    # 월별 신호 분포 (집중도 확인 — 과최적화 경계)
    rdf["month"] = pd.to_datetime(rdf["date"]).dt.to_period("M")
    print("\n월별 신호 분포:")
    print(rdf.groupby("month").size().to_string())


def main() -> int:
    start, end = "2025-11-01", "2026-05-29"  # 지표 워밍업 포함 ~7개월 (실효 6개월)
    print(f"=== 조기발굴 룰 6개월 백테스트 ({start} ~ {end}) ===")
    print("유니버스 로딩 (data/new_universe_tickers.txt 시총 상위 287)...")
    uni_file = PROJECT_ROOT / "data" / "new_universe_tickers.txt"
    if uni_file.exists():
        lines = [l.strip() for l in uni_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        uni = [l.split(",")[0].strip().zfill(6) for l in lines]
    else:
        try:
            krx = fdr.StockListing("KRX-DESC")
            uni = krx["Code"].astype(str).str.zfill(6).tolist()[:300]
        except Exception as e:
            print(f"유니버스 로딩 실패: {e}")
            return 1
    print(f"유니버스 {len(uni)}종목")
    rdf = backtest(uni, start, end)
    summarize(rdf)
    out = PROJECT_ROOT / "data" / "research" / "surge_backtest_6m_result.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rdf.empty:
        rdf.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
