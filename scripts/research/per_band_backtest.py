"""역사 PER 밴드 팩터 백테스트 — '자기 역사 대비 저PER'의 forward 예측력.

검증 질문 2개:
 Q1) 자기 5년 PER 밴드 하단(pct_rank 낮음) 종목이 forward 초과수익을 내는가?
 Q2) 그것이 7/4의 '횡단면 저PER'(같은날 유니버스 내 상대 저PER) 위에 신호를 더하는가?
     → 두 팩터의 이중정렬(double sort)로 독립성 확인.

방법 (point-in-time, lookahead 차단):
 - 유니버스 = processed parquet ∩ 4분기+ EPS 보유 종목
 - 월초 리밸런스(2021~2025). 각 시점 각 종목:
     · 밴드 = 그 시점까지의 트레일링 5년 PER (settle+60d 공시지연 반영)
     · pct_rank = 밴드 내 현재 PER 위치(0=역대최저평가)
 - forward = 종가 기준 D+20 수익률 − 동기간 KOSPI (초과수익)
 - pct_rank 5분위 & 횡단면 PER 5분위 통계

사용: ./venv/bin/python3.11 scripts/research/per_band_backtest.py
"""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.use_cases.valuation_band_history import (  # noqa: E402
    BAND_WINDOW_YEARS, MIN_BAND_DAYS, _load_close, _per_series, _ttm_eps_series,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("per_band_backtest")

FWD_DAYS = 20
START, END = "2021-01-01", "2025-12-01"


def _index_close(name: str) -> pd.Series:
    df = pd.read_csv(PROJECT_ROOT / "data" / f"{name}.csv",
                     parse_dates=["Date"]).sort_values("Date").set_index("Date")
    col = "close" if "close" in df.columns else "Close"
    return df[col]


def main() -> int:
    import json
    kospi = _index_close("kospi_index")
    try:
        kosdaq = _index_close("kosdaq_index")
    except Exception:
        kosdaq = None
    try:
        market_map = json.load(open(PROJECT_ROOT / "data" / "market_map.json", encoding="utf-8"))
    except Exception:
        market_map = {}

    def bench_for(ticker: str) -> pd.Series:
        # 종목 자기시장 벤치마크 (KOSDAQ 종목은 KOSDAQ, 그 외 KOSPI) — 7/5 v1-3 정확성
        if kosdaq is not None and market_map.get(ticker) == "KOSDAQ":
            return kosdaq
        return kospi

    tickers = [Path(p).stem for p in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))]
    logger.info("유니버스 후보 %d종 — PER 시계열 구축 중...", len(tickers))

    per_all: dict[str, pd.Series] = {}
    close_all: dict[str, pd.Series] = {}
    for i, tk in enumerate(tickers):
        per = _per_series(tk)
        if per is None or len(per) < MIN_BAND_DAYS:
            continue
        close = _load_close(tk)
        if close is None or len(close) < FWD_DAYS + 5:
            continue
        per_all[tk], close_all[tk] = per, close
        if (i + 1) % 300 == 0:
            logger.info("  %d/%d 처리 (유효 %d)", i + 1, len(tickers), len(per_all))
    logger.info("PER 밴드 유효 종목 %d종", len(per_all))

    rebal_dates = pd.date_range(START, END, freq="MS")
    window = pd.Timedelta(days=int(BAND_WINDOW_YEARS * 365.25))

    rows = []
    for d in rebal_dates:
        for tk, per in per_all.items():
            band = per[(per.index > d - window) & (per.index <= d)]
            if len(band) < MIN_BAND_DAYS:
                continue
            # 현재 PER = d 시점 최근값(10거래일 내 존재해야 = 상장·거래 중)
            if (d - band.index[-1]).days > 10:
                continue
            current = float(band.iloc[-1])
            pct_rank = float((band < current).mean())

            close = close_all[tk]
            ci = close.index.searchsorted(d, side="right") - 1
            if ci < 0 or ci + FWD_DAYS >= len(close):
                continue
            base = close.iloc[ci]
            if base <= 0:
                continue
            fwd = (close.iloc[ci + FWD_DAYS] / base - 1) * 100
            bench = bench_for(tk)  # 자기시장 지수 (KOSDAQ 종목=KOSDAQ, 그 외=KOSPI)
            ki = bench.index.searchsorted(d, side="right") - 1
            if ki < 0 or ki + FWD_DAYS >= len(bench):
                continue
            kfwd = (bench.iloc[ki + FWD_DAYS] / bench.iloc[ki] - 1) * 100
            # 흑자 안정성(point-in-time): d까지 공시된 TTM EPS 양수 비율
            ttm = _ttm_eps_series(tk)
            pos_ratio = 0.0
            if ttm is not None:
                ttm_d = ttm[ttm.index <= d]
                if len(ttm_d):
                    pos_ratio = float((ttm_d > 0).mean())
            rows.append({"date": d, "ticker": tk, "pct_rank": pct_rank,
                         "current_per": current, "excess": fwd - kfwd,
                         "pos_ratio": pos_ratio})

    df = pd.DataFrame(rows)
    logger.info("관측 %d (종목-월). 분석 시작.\n", len(df))
    print(f"══ PER 밴드 백테스트 ({START[:7]}~{END[:7]}, D+{FWD_DAYS} KOSPI초과, "
          f"{df['ticker'].nunique()}종·{len(df)}관측) ══")

    # Q1: 자기역사 밴드 pct_rank 5분위 (각 리밸런스 날짜 내 상대분위 = 시점중립)
    df["band_q"] = df.groupby("date")["pct_rank"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]))
    print("\n[Q1] 자기역사 PER 밴드 분위 (1=자기최저평가 ~ 5=자기최고평가)")
    for q in [1, 2, 3, 4, 5]:
        g = df[df["band_q"] == q]["excess"]
        print(f"  band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}% 중앙 {g.median():+.2f}")
    q1, q5 = df[df.band_q == 1]["excess"], df[df.band_q == 5]["excess"]
    print(f"  → 저평가(Q1)−고평가(Q5) 스프레드: {q1.mean()-q5.mean():+.2f}%p")

    # Q2: 횡단면 PER 5분위 (같은날 유니버스 내 절대 PER 순위)
    df["xs_q"] = df.groupby("date")["current_per"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]))
    print("\n[Q2] 횡단면 PER 분위 (1=유니버스 최저PER)")
    for q in [1, 2, 3, 4, 5]:
        g = df[df["xs_q"] == q]["excess"]
        print(f"  xs Q{q}  n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")

    # 이중정렬: 횡단면 저PER(xs Q1~2) 내에서 자기밴드가 추가 신호를 주는가
    print("\n[이중정렬] 횡단면 저PER(xs Q1~2) 서브셋 내 자기밴드 분위별 초과수익")
    sub = df[df["xs_q"].isin([1, 2])]
    for q in [1, 2, 3, 4, 5]:
        g = sub[sub["band_q"] == q]["excess"]
        if len(g) >= 30:
            print(f"  저PER∩band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")

    # Q3: 안정 흑자주(reliable, 흑자비율 80%+) 서브셋에서 밴드 신호가 강해지는가
    rel = df[df["pos_ratio"] >= 0.8].copy()
    print(f"\n[Q3] 안정흑자주(흑자비율 80%+, n={len(rel)}={len(rel)/len(df)*100:.0f}%) 내 밴드 분위")
    if len(rel) >= 500:
        rel["band_q"] = rel.groupby("date")["pct_rank"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            if s.nunique() >= 5 else np.nan)
        for q in [1, 2, 3, 4, 5]:
            g = rel[rel["band_q"] == q]["excess"]
            if len(g) >= 30:
                print(f"  안정∩band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")
        rq1, rq5 = rel[rel.band_q == 1]["excess"], rel[rel.band_q == 5]["excess"]
        print(f"  → 안정흑자주 저평가−고평가 스프레드: {rq1.mean()-rq5.mean():+.2f}%p")

    # 상관: 두 신호 독립성
    corr = df[["pct_rank", "current_per"]].corr().iloc[0, 1]
    print(f"\n[독립성] pct_rank vs current_per 상관 = {corr:+.2f} (낮을수록 독립 신호)")

    out = PROJECT_ROOT / "data" / "research" / "per_band_backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "period": f"{START}~{END}", "fwd_days": FWD_DAYS,
        "n_obs": len(df), "n_tickers": int(df["ticker"].nunique()),
        "band_q1_excess": round(float(q1.mean()), 2),
        "band_q5_excess": round(float(q5.mean()), 2),
        "band_spread": round(float(q1.mean() - q5.mean()), 2),
        "signal_corr": round(float(corr), 3),
    }
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
