# -*- coding: utf-8 -*-
"""Phase 12: 인버스 시그널 확장 백테스트 (2023-02 ~ 2026-07, 3.4년) — 판정: 기각

Phase 9(D+3 부적합)·Phase 11(strict n=3 "활용가능") 재검증.
- 기간 3.4배 확장 + 2026 6/22~7/6 폭락장 포함 (Phase 11 대비 out-of-sample)
- 진입 = 시그널 익일 시가 (실행가능 기준 — 수급 데이터는 T일 16:35 확보)
- 청산 = D+1/D+2/D+5 종가 + -3% 종가 손절, non-overlap 에피소드
- t검정 필수 (US FV 교훈: 헤드라인 수익률 ≠ 유의성)

★결론 (상세: data/backtest/phase12_report.md):
1. 인버스 매수(전 변형) 기각 — Phase 11 strict는 n=14로 늘리자 D+2 -5.98%·t=-3.11 (역방향 유의).
   2025 +6.1%(n=3, Phase 11이 본 구간) vs 2026 -6.1%(n=16) = 소표본 착시.
2. 역가설(폭락일 익일 레버리지 롱)도 유의성 미달 (최고 t=+1.56) + 2026 반등 집중 과최적 위험.
3. 252670 현재가 67~84원 → 1틱=1.2~1.5% 슬리피지, 단기 트레이드 집행 자체가 비경제적.
→ 인버스 페이퍼 신설 안 함. bear_market_alpha의 -3조 경보는 '방어(리스크 축소) 경보'로만 유효.

데이터:
- data/kospi_investor_flow.csv (시장 외인 일별, 억원, 2023-02~)
- data/investor_flow/investor_daily.db (현행 라이브 시그널 원천, 2025-04~ 교차검증: corr 0.990·일치율 99%)
- data/kospi_index.csv / FDR: 252670·114800·122630·069500
"""
import io
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd

DATA = PROJECT_ROOT / "data"
OUT_DIR = DATA / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
START = "2023-02-01"
COST_RT = 0.10  # 왕복 비용 가정(%, 저가 ETF 스프레드 감안 시 실제는 이보다 나쁨)


def load_data() -> pd.DataFrame:
    idx = pd.read_csv(DATA / "kospi_index.csv")
    idx["Date"] = pd.to_datetime(idx["Date"])
    idx = idx[idx["Date"] >= "2022-10-01"].sort_values("Date").reset_index(drop=True)
    idx["kospi_ret1d"] = idx["close"].pct_change() * 100
    idx["ma20"] = idx["close"].rolling(20).mean()
    idx = idx.rename(columns={"close": "kospi_close"})[
        ["Date", "kospi_close", "kospi_ret1d", "ma20"]]

    flow = pd.read_csv(DATA / "kospi_investor_flow.csv", encoding="utf-8-sig")
    flow["Date"] = pd.to_datetime(flow["Date"])
    flow = flow.sort_values("Date").reset_index(drop=True)
    flow["f1d"] = flow["foreign_net"]  # 억원
    flow["f5d"] = flow["foreign_net"].rolling(5, min_periods=5).sum()

    db_path = DATA / "investor_flow" / "investor_daily.db"
    db = pd.DataFrame(columns=["Date", "foreign_db_eok", "db_f5d"])
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT date, SUM(net_val)/1e8 FROM investor_daily "
            "WHERE investor='외국인' GROUP BY date ORDER BY date").fetchall()
        conn.close()
        db = pd.DataFrame(rows, columns=["date", "foreign_db_eok"])
        db["Date"] = pd.to_datetime(db["date"], format="%Y%m%d")
        db["db_f5d"] = db["foreign_db_eok"].rolling(5, min_periods=5).sum()

    import FinanceDataReader as fdr
    etfs = {}
    for tk in ["252670", "114800", "122630", "069500"]:
        cache = OUT_DIR / f"phase12_etf_{tk}.parquet"
        if cache.exists():
            e = pd.read_parquet(cache)
        else:
            e = fdr.DataReader(tk, "2022-12-01").reset_index()
            e.to_parquet(cache)
        e = e.rename(columns={"Open": f"open_{tk}", "Close": f"close_{tk}"})
        etfs[tk] = e[["Date", f"open_{tk}", f"close_{tk}"]]

    df = etfs["252670"]
    for tk in ["114800", "122630", "069500"]:
        df = df.merge(etfs[tk], on="Date", how="inner")
    df = df.merge(idx, on="Date", how="left")
    df = df.merge(flow[["Date", "f1d", "f5d"]], on="Date", how="left")
    if len(db):
        df = df.merge(db[["Date", "foreign_db_eok", "db_f5d"]], on="Date", how="left")
    return df[df["Date"] >= START].reset_index(drop=True)


def cross_validate(df: pd.DataFrame) -> None:
    """CSV(시장) vs DB(유니버스 합산) 외인 시계열 정합 — 겹치는 구간."""
    if "foreign_db_eok" not in df.columns:
        return
    o = df.dropna(subset=["f1d", "foreign_db_eok"])
    if len(o) < 30:
        print("[교차검증] 겹침 표본 부족")
        return
    corr = o["f1d"].corr(o["foreign_db_eok"])
    agree = ((o["f5d"] <= -30000) == (o["db_f5d"] <= -30000)).mean()
    print(f"[교차검증] 일별 corr={corr:.3f} | -3조 시그널 일치율={agree * 100:.0f}% (n={len(o)})")


def episode_returns(df, sig_mask, tk, hold, stop_pct=-3.0, dedup=True) -> pd.DataFrame:
    """시그널일 T → T+1 시가 진입, T+hold 종가 청산(중도 -3% 종가 손절)."""
    op, cl = df[f"open_{tk}"].values, df[f"close_{tk}"].values
    n = len(df)
    out, busy_until = [], -1
    for i in np.where(sig_mask.values)[0]:
        if i + 1 >= n or i + hold >= n:
            continue
        if dedup and i <= busy_until:
            continue
        entry = op[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        exit_px, exit_j = None, None
        for j in range(i + 1, i + hold + 1):
            r = (cl[j] / entry - 1) * 100
            if r <= stop_pct or j == i + hold:
                exit_px, exit_j = cl[j], j
                break
        out.append({"i": i, "date": df["Date"].iloc[i],
                    "ret": (exit_px / entry - 1) * 100, "exit_day": exit_j - i})
        if dedup:
            busy_until = exit_j
    return pd.DataFrame(out)


def stats_line(name: str, r: pd.DataFrame) -> str:
    if len(r) == 0:
        return f"{name:34s} n=0"
    m, s = r["ret"].mean(), r["ret"].std(ddof=1)
    t = m / (s / np.sqrt(len(r))) if len(r) > 1 and s > 0 else np.nan
    return (f"{name:34s} n={len(r):3d} 평균{m:+6.2f}% net{m - COST_RT:+6.2f}% "
            f"승률{(r['ret'] > 0).mean() * 100:4.0f}% 중앙{r['ret'].median():+6.2f}% "
            f"t={t:+.2f} min{r['ret'].min():+.1f} max{r['ret'].max():+.1f}")


def main() -> int:
    df = load_data()
    print("=== Phase 12: 인버스 확장 백테스트 ===")
    print(f"기간: {df['Date'].min().date()} ~ {df['Date'].max().date()} ({len(df)}거래일)")
    cross_validate(df)

    sigs = {
        "CONTROL(전체일)": pd.Series(True, index=df.index),
        "S1 라이브: f5d<=-3조": df["f5d"] <= -30000,
        "S1b f5d<=-1.5조": df["f5d"] <= -15000,
        "S2 KOSPI1d<=-2%": df["kospi_ret1d"] <= -2,
        "S3 KOSPI-2% AND f5d-3조": (df["kospi_ret1d"] <= -2) & (df["f5d"] <= -30000),
        "S4 P11strict(-2.5/-5조/-3천억)": (df["kospi_ret1d"] <= -2.5) & (df["f5d"] <= -50000) & (df["f1d"] <= -3000),
        "S4b strictCSV(-2.5/-2.5조/-1.5천)": (df["kospi_ret1d"] <= -2.5) & (df["f5d"] <= -25000) & (df["f1d"] <= -1500),
        "S5 f5d-3조 AND KOSPI<MA20": (df["f5d"] <= -30000) & (df["kospi_close"] < df["ma20"]),
    }
    valid = df["f5d"].notna() & df["kospi_ret1d"].notna()

    for tk, tkname in [("252670", "인버스2X"), ("114800", "인버스1X")]:
        print(f"\n───── {tkname}({tk}) · 인버스 매수 · 에피소드 · -3%손절 ─────")
        for hold in [1, 2, 5]:
            print(f"  [보유 D+{hold} 종가 청산]")
            for name, mask in sigs.items():
                print("   ", stats_line(name, episode_returns(df, mask & valid, tk, hold)))

    print("\n───── 역가설: 폭락일 익일 시가 롱 · 에피소드 · -3%손절 ─────")
    for tk, tkname in [("122630", "레버리지2X"), ("069500", "KODEX200")]:
        for hold in [1, 2, 5]:
            print(f"  [{tkname} · 보유 D+{hold}]")
            for name, mask in sigs.items():
                if name.startswith("CONTROL"):
                    continue
                r = episode_returns(df, mask & valid, tk, hold)
                print("   ", stats_line(name, r))
                if len(r) >= 5:
                    r2 = r.copy()
                    r2["yr"] = pd.to_datetime(r2["date"]).dt.year
                    yrs = " ".join(f"{y}:{g['ret'].mean():+.1f}%(n={len(g)})" for y, g in r2.groupby("yr"))
                    print(f"      └연도별: {yrs}")

    print("\n───── 연도별 분해 · S3 · 252670 D+2 (Phase 11 소표본 착시 검증) ─────")
    r = episode_returns(df, sigs["S3 KOSPI-2% AND f5d-3조"] & valid, "252670", 2)
    if len(r):
        r["yr"] = pd.to_datetime(r["date"]).dt.year
        for yr, g in r.groupby("yr"):
            print("   ", stats_line(str(yr), g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
