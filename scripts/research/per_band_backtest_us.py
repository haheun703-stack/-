"""US 역사 PER 밴드 팩터 백테스트 — '자기 역사 대비 저PER'의 forward 예측력 (미국판).

한국 per_band_backtest.py의 이식. 검증 질문 동일:
 Q1) 자기 5년 PER 밴드 하단(pct_rank 낮음) 종목이 forward 초과수익을 내는가?
 Q2) 그것이 횡단면 저PER(같은날 유니버스 내 상대) 위에 신호를 더하는가? (이중정렬)

방법 (point-in-time, lookahead 차단):
 - 유니버스 = data/us_market/per_band/ 캐시(basis_ok 종목만 — ADR 통화오염 TSM·NVO·TM 제외)
 - 월초 리밸런스. 각 시점 밴드=그 시점까지 트레일링 5년 PER(EPS는 결산일+60d 가용).
 - forward = 종가 D+20 − 동기간 SPY(자기시장 벤치마크) 초과수익.

★7/5 교훈 반영:
 - 벤치마크 커버리지=백테스트 기간: SPY 6년치를 yfinance로 확보(us_daily 3.2년 그대로 쓰면
   조용한 드롭·구성편향). 드롭 카운터로 커버리지 부족 노출.
 - 생존편향: 유니버스=현재 대형주 → 편향 존재. 완화=시점내 상대분위(같은 생존풀 내 버킷 비교).
   절대수익엔 편향 있으나 버킷 스프레드(Q1-Q5)는 상대적으로 완화됨 — 정직 표기.
 - 데이터 제약: yfinance 연간 EPS 깊이 ~4년 → 밴드 유효구간 ~2024~2026·관측수 한국 대비 적음.
   종가 기준(익일시가 아님) — PER밴드=느린 월간 가치팩터라 close-to-close 표준(이벤트팝 아님).

사용: ./venv/bin/python3.11 scripts/research/per_band_backtest_us.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.use_cases.valuation_band_history_us import (  # noqa: E402
    BAND_WINDOW_YEARS, CACHE_DIR, MIN_BAND_DAYS, _load_close, _load_meta,
    _load_ttm, _per_series,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("per_band_backtest_us")

FWD_DAYS = 20
START, END = "2023-06-01", "2026-06-01"  # 실제 유효구간은 드롭 카운터가 노출


def _spy_close() -> pd.Series:
    """SPY 6년 종가(벤치마크) — 커버리지=백테스트 기간 보장 위해 yfinance 직접."""
    import yfinance as yf
    h = yf.Ticker("SPY").history(period="6y")
    return pd.Series(h["Close"].values,
                     index=pd.to_datetime(h.index).tz_localize(None)).sort_index()


def main() -> int:
    spy = _spy_close()
    logger.info("SPY 벤치마크 커버리지: %s ~ %s (%d행)",
                spy.index[0].date(), spy.index[-1].date(), len(spy))

    tickers = sorted({p.stem.replace("_close", "")
                      for p in CACHE_DIR.glob("*_close.parquet")})
    per_all: dict[str, pd.Series] = {}
    close_all: dict[str, pd.Series] = {}
    excluded_basis = 0
    for tk in tickers:
        meta = _load_meta(tk)
        if meta is not None and not meta.get("basis_ok", True):
            excluded_basis += 1
            continue
        per = _per_series(tk)
        if per is None or len(per) < MIN_BAND_DAYS:
            continue
        close = _load_close(tk)
        if close is None or len(close) < FWD_DAYS + 5:
            continue
        per_all[tk], close_all[tk] = per, close
    logger.info("유니버스 %d종 → PER밴드 유효 %d종 (basis제외 %d)",
                len(tickers), len(per_all), excluded_basis)

    rebal_dates = pd.date_range(START, END, freq="MS")
    window = pd.Timedelta(days=int(BAND_WINDOW_YEARS * 365.25))
    dropped_bench = 0
    rows = []
    for d in rebal_dates:
        for tk, per in per_all.items():
            band = per[(per.index > d - window) & (per.index <= d)]
            if len(band) < MIN_BAND_DAYS:
                continue
            if (d - band.index[-1]).days > 10:   # 상장·거래 중(10거래일 내 PER 존재)
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
            si = spy.index.searchsorted(d, side="right") - 1
            if si < 0 or si + FWD_DAYS >= len(spy):
                dropped_bench += 1
                continue
            sfwd = (spy.iloc[si + FWD_DAYS] / spy.iloc[si] - 1) * 100

            ttm = _load_ttm(tk)
            pos_ratio = 0.0
            if ttm is not None:
                ttm_d = ttm[ttm.index <= d]
                if len(ttm_d):
                    pos_ratio = float((ttm_d > 0).mean())
            cov_years = (band.index[-1] - band.index[0]).days / 365.25
            rows.append({"date": d, "ticker": tk, "pct_rank": pct_rank,
                         "current_per": current, "excess": fwd - sfwd,
                         "pos_ratio": pos_ratio, "coverage_years": round(cov_years, 1)})

    df = pd.DataFrame(rows)
    logger.info("관측 %d (종목-월) | 벤치마크 커버리지 부족 드롭 %d건 (%.0f%%)",
                len(df), dropped_bench, dropped_bench / max(len(df) + dropped_bench, 1) * 100)
    if not len(df):
        logger.error("관측 0 — 캐시(--fetch) 또는 기간 확인")
        return 1
    logger.info("실제 관측 기간: %s ~ %s", df["date"].min().date(), df["date"].max().date())
    print(f"\n══ US PER 밴드 백테스트 ({df['date'].min().date()}~{df['date'].max().date()}, "
          f"D+{FWD_DAYS} SPY초과, {df['ticker'].nunique()}종·{len(df)}관측) ══")
    print("★한계(적대검수 반영): ①생존편향=저밴드 프리미엄 방향으로 잔존(회복 생존자만 남음)→절대초과 해석불가 ")
    print("  ②벤치 SPY β=1 미조정(고베타 성장↔방어 혼재 스타일 confound) ③밴드 대부분 1~1.5년 얕은밴드")
    print("  (yfinance 연간EPS ~4년) → 유의성(t검정)·프로덕션정합(reliable coverage≥3) 위주로만 해석")

    # Q1: 자기역사 밴드 pct_rank 5분위 (시점내 상대 = 시점중립)
    def _q5(s):
        return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) if s.nunique() >= 5 else np.nan
    df["band_q"] = df.groupby("date")["pct_rank"].transform(_q5)
    print("\n[Q1] 자기역사 PER 밴드 분위 (1=자기최저평가 ~ 5=자기최고평가)")
    for q in [1, 2, 3, 4, 5]:
        g = df[df["band_q"] == q]["excess"]
        if len(g):
            print(f"  band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}% 중앙 {g.median():+.2f}")
    q1 = df[df.band_q == 1]["excess"]
    q5 = df[df.band_q == 5]["excess"]
    spread = (q1.mean() - q5.mean()) if len(q1) and len(q5) else float("nan")
    print(f"  → 저평가(Q1)−고평가(Q5) 스프레드: {spread:+.2f}%p")

    # ★유의성(렌즈2 #1): 독립 관측 ≈ 리밸런스 월 수(같은날 44종은 강한 횡단면 상관 → 클러스터링).
    #   월간 롱숏(Q1-Q5) 시계열의 t값으로 노이즈 구분. 승률 48%·비단조면 t가 낮게 나와야 정직.
    monthly = []
    for _d, g in df.groupby("date"):
        gq = g.dropna(subset=["band_q"])
        q1m, q5m = gq[gq.band_q == 1]["excess"], gq[gq.band_q == 5]["excess"]
        if len(q1m) and len(q5m):
            monthly.append(q1m.mean() - q5m.mean())
    ms = pd.Series(monthly)
    t_ls = float("nan")
    if len(ms) >= 3 and ms.std(ddof=1) > 0:
        t_ls = ms.mean() / (ms.std(ddof=1) / np.sqrt(len(ms)))
    verdict = "유의(t≥2)" if abs(t_ls) >= 2 else "유의성 없음 — 노이즈와 구분 불가"
    print(f"  [유의성] 월간 롱숏 {len(ms)}개월 평균 {ms.mean():+.2f}%p · t={t_ls:.2f} → {verdict}")

    # ★프로덕션 정합(렌즈2 #2): 엔진은 reliable(coverage≥3·흑자80%+)만 가점. 헤드라인 Q1은
    #   얕은밴드(len≥250만) 지배 → 프로덕션이 실제 신뢰하는 깊은밴드 서브셋을 따로 검증.
    rel3 = df[(df["coverage_years"] >= 3.0) & (df["pos_ratio"] >= 0.8)].copy()
    print(f"\n[프로덕션정합] reliable(coverage≥3·흑자80%+) n={len(rel3)} ({len(rel3)/len(df)*100:.0f}%)")
    if len(rel3) >= 100:
        rel3["band_q"] = rel3.groupby("date")["pct_rank"].transform(_q5)
        for q in [1, 2, 3, 4, 5]:
            g = rel3[rel3["band_q"] == q]["excess"]
            if len(g) >= 15:
                print(f"  rel band Q{q} n={len(g):<4} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")
        r1, r5 = rel3[rel3.band_q == 1]["excess"], rel3[rel3.band_q == 5]["excess"]
        if len(r1) and len(r5):
            print(f"  → reliable 저-고 스프레드: {r1.mean()-r5.mean():+.2f}%p")
    else:
        print("  (표본 부족 — yfinance EPS 깊이상 coverage≥3은 후기 관측에만 성립)")

    # Q2: 횡단면 PER 5분위
    df["xs_q"] = df.groupby("date")["current_per"].transform(_q5)
    print("\n[Q2] 횡단면 PER 분위 (1=유니버스 최저PER)")
    for q in [1, 2, 3, 4, 5]:
        g = df[df["xs_q"] == q]["excess"]
        if len(g):
            print(f"  xs Q{q}  n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")

    # 이중정렬: 횡단면 저PER 서브셋 내 자기밴드 추가신호
    print("\n[이중정렬] 횡단면 저PER(xs Q1~2) 서브셋 내 자기밴드 분위별 초과수익")
    sub = df[df["xs_q"].isin([1, 2])]
    for q in [1, 2, 3, 4, 5]:
        g = sub[sub["band_q"] == q]["excess"]
        if len(g) >= 20:
            print(f"  저PER∩band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")

    # Q3: 안정흑자주(흑자비율 80%+)
    rel = df[df["pos_ratio"] >= 0.8].copy()
    print(f"\n[Q3] 안정흑자주(흑자비율 80%+, n={len(rel)}={len(rel)/len(df)*100:.0f}%) 내 밴드 분위")
    if len(rel) >= 200:
        rel["band_q"] = rel.groupby("date")["pct_rank"].transform(_q5)
        for q in [1, 2, 3, 4, 5]:
            g = rel[rel["band_q"] == q]["excess"]
            if len(g) >= 20:
                print(f"  안정∩band Q{q} n={len(g):<5} 초과 {g.mean():+.2f}%p 승률 {(g>0).mean()*100:.0f}%")
        rq1, rq5 = rel[rel.band_q == 1]["excess"], rel[rel.band_q == 5]["excess"]
        if len(rq1) and len(rq5):
            print(f"  → 안정흑자주 저평가−고평가 스프레드: {rq1.mean()-rq5.mean():+.2f}%p")

    corr = df[["pct_rank", "current_per"]].corr().iloc[0, 1]
    print(f"\n[독립성] pct_rank vs current_per 상관 = {corr:+.2f} (낮을수록 독립 신호)")

    out = PROJECT_ROOT / "data" / "research" / "per_band_backtest_us.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "period": f"{df['date'].min().date()}~{df['date'].max().date()}", "fwd_days": FWD_DAYS,
        "n_obs": len(df), "n_tickers": int(df["ticker"].nunique()),
        "n_months": len(ms), "dropped_bench": dropped_bench,
        "band_spread": round(float(spread), 2) if spread == spread else None,
        "band_spread_tstat": round(float(t_ls), 2) if t_ls == t_ls else None,
        "significant": bool(abs(t_ls) >= 2) if t_ls == t_ls else False,
        "signal_corr": round(float(corr), 3),
        "verdict": ("US 밴드 예측력 탐지 안 됨(유의성 없음)" if abs(t_ls) < 2
                    else "US 밴드 유의"),
        "caveats": ("survivorship(저밴드 프리미엄 방향 잔존)·SPY β=1 미조정·"
                    "얕은밴드(yfinance EPS ~4y)·절대초과 해석불가 → 관측 전용"),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
