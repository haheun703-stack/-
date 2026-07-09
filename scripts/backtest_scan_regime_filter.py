# -*- coding: utf-8 -*-
"""SCAN 눌림목 x KOSPI 레짐 필터 백테스트 — "CAUTION 차단" 가설 심판.

배경 (2026-07-09):
- 페이퍼 청산 106건 분해 결과 CAUTION 진입 66건 평균 -2.16% vs BEAR 40건 +0.57%.
- 단일 에피소드(6월 꼭지)일 수 있어 7.5년 전 구간으로 일반성 검증.

사전 명시 가설 (그리드 탐색 금지 — 단일 가설):
- H: "KOSPI 레짐이 CAUTION(고변동 상승)인 날의 SCAN 눌림목 진입은
     같은 레짐 기저선(비이벤트) 대비 엣지가 없거나 음(-)이다."
- 채택 조건: ①CAUTION 엣지 ≤ 0 (date-cluster t ≥ 2 수준으로 열위)
             ②BULL/BEAR 엣지는 양(+) 유지 (차단해도 잃을 것 없음)
             ③연도별 견고 (특정 1년 아님)

설계:
- 진입 시그널 = signal_engine.py A급 가격축 그대로 (계수 튜닝 금지)
  · PB15_BB:          ma20_dev ∈ [-15, -8] & bb_position ≤ 0.25 & RSI14 ≤ 40
  · PULLBACK_20MA_15: ma20_dev ∈ [-18, -10] & RSI14 ≤ 45
- 청산 엔진 = paper_trading_unified.py 그대로 (수급 동적보정 제외):
  손절 -7% → T2 +20% → T1 +10%(절반) → 트레일링(+8% 활성, 고점 -4%) → MAX_HOLD 5
  (페이퍼와 동일하게 '종가 평가·종가 체결')
- 진입가 = 시그널 당일(D0) 종가 (페이퍼 운영과 동일 — BAT-D 16:30 이후 판정)
- 레짐 = index_regime._regime_from_close와 동일 규칙의 일별 시계열 (D0 종가 기준, lookahead 없음)
- 기저선(비이벤트) = 종목별 21거래일 격자 진입(시그널 무관)에 동일 청산 엔진 적용
  → 이벤트 스터디 기저선 필수 교훈(7/7) 반영: 레짐 베타와 시그널 알파를 분리
- 통계 = 진입일 클러스터(같은 날 진입 평균 후 날짜 단위 t) — 중첩 √N 과대 교훈 반영
- 유동성 게이트 = 진입일 5일 평균 거래대금 ≥ 10억원

한계 (정직 고지):
- 유니버스 = 현재 관리 중인 data/raw parquet → 생존편향 존재 (레짐 간 '비교'에는 공통 영향)
- 페이퍼의 top-5 픽 선별·수급 게이트는 재현 불가 (가격축 시그널 전수로 근사)

★판정 (2026-07-09 실행 결과): 기각.
- 2019~2025 CAUTION 진입 980건 평균 -0.07% (자체 t +0.40) — 역사적 CAUTION 페널티 없음.
- 2026 상반기 CAUTION -3.05%는 기저선도 -2.42% → 엣지 -0.62%p(t -0.59) = 레짐 베타
  (시장 자체가 빠진 것)이지 시그널이 나쁜 종목을 고른 게 아님.
- 전체 헤드라인(-2.16%)은 2026 에피소드(2,305/3,285건=70%)가 지배한 착시.
- 레짐 필터 채택 조건 ①(t≥2 열위) ②는 무의미해짐 ③(연도 견고) 전부 미충족.
- 부차 관측: SCAN 가격축 시그널의 對기저선 엣지는 전 레짐·전 기간 -0.2~-0.5%p(전부 ns)
  → 가격축 단독으론 무작위 진입 대비 부가가치 없음 (top-5 선별·수급 게이트 효과는 미검증).
- 교훈 재확인: 라이브 106건 분해가 아무리 그럴듯해도(-2.16% vs +0.57%)
  비이벤트 기저선 없이 규칙화하면 안 된다 — 단일 에피소드 + 레짐 베타 오귀속이었다.

실행:
    python -u -X utf8 scripts/backtest_scan_regime_filter.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
KOSPI_CSV = PROJECT_ROOT / "data" / "kospi_index.csv"
OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"

# ── 청산 엔진 파라미터 (paper_trading_unified.py 원본과 동일) ──
STOP_LOSS_PCT = -0.07
TAKE_PROFIT_T1_PCT = 0.10
TAKE_PROFIT_T2_PCT = 0.20
TRAILING_ACTIVATE_PCT = 0.08
TRAILING_STOP_PCT = -0.04
MAX_HOLDING_DAYS = 5

MIN_TRADING_VALUE = 1_000_000_000  # 5일 평균 거래대금 10억
BASELINE_GRID = 21  # 기저선: 21거래일 격자 진입


# ──────────────────────────────────────────────
# KOSPI 일별 레짐 시계열 (index_regime._regime_from_close 규칙 동일)
# ──────────────────────────────────────────────
def build_regime_series() -> pd.Series:
    k = pd.read_csv(KOSPI_CSV)
    datecol = [c for c in k.columns if c.lower() in ("date", "날짜")][0]
    closecol = [c for c in k.columns if c.lower() in ("close", "종가")][0]
    k[datecol] = pd.to_datetime(k[datecol])
    k = k.sort_values(datecol).reset_index(drop=True)
    s = pd.to_numeric(k[closecol], errors="coerce")
    ma20 = s.rolling(20).mean()
    ma60 = s.rolling(60).mean()
    lr = np.log(s / s.shift(1))
    rv20 = lr.rolling(20).std() * np.sqrt(252) * 100
    rvp = rv20.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    lab = np.full(len(k), "CAUTION", dtype=object)
    for i in range(len(k)):
        if np.isnan(ma20[i]) or np.isnan(ma60[i]):
            continue
        if s[i] > ma20[i]:
            lab[i] = "BULL" if (not np.isnan(rvp[i]) and rvp[i] < 0.5) else "CAUTION"
        elif s[i] > ma60[i]:
            lab[i] = "BEAR"
        else:
            lab[i] = "CRISIS"
    return pd.Series(lab, index=k[datecol].dt.normalize())


# ──────────────────────────────────────────────
# 종목별 지표 (signal_engine.calc_indicators와 동일 정의, 벡터화)
# ──────────────────────────────────────────────
def calc_signal_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(진입시그널 마스크, 유동성 마스크) 반환."""
    close = df["close"]
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()  # ddof=1 — 원본 .iloc[-20:].std()와 동일
    ma20_dev = (close - ma20) / ma20 * 100

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    bb_lower = ma20 - 2 * std20
    bb_upper = ma20 + 2 * std20
    width = bb_upper - bb_lower
    bb_position = ((close - bb_lower) / width).where(width > 0, 0.5)

    pb15_bb = (ma20_dev <= -8) & (ma20_dev >= -15) & (bb_position <= 0.25) & (rsi <= 40)
    pullback = (ma20_dev <= -10) & (ma20_dev >= -18) & (rsi <= 45)
    sig = pb15_bb | pullback

    tval5 = df["trading_value"].rolling(5).mean() if "trading_value" in df.columns \
        else (close * df["volume"]).rolling(5).mean()
    liquid = tval5 >= MIN_TRADING_VALUE
    return sig, liquid


# ──────────────────────────────────────────────
# 청산 엔진 (paper_trading_unified 매도 elif 체인 미러, 종가 평가)
# ──────────────────────────────────────────────
def simulate_exit(closes: np.ndarray, i: int) -> tuple[float, int, str]:
    """진입 idx=i(D0 종가 체결) → (최종 pnl%, 청산 idx, 사유)."""
    entry = closes[i]
    t1_sold = False
    t1_pnl = 0.0
    trailing = False
    peak = entry
    n = len(closes)
    j = i
    for k in range(1, n - i):
        j = i + k
        c = closes[j]
        pnl = c / entry - 1
        if pnl <= STOP_LOSS_PCT:
            reason = "STOP_LOSS"
            break
        if pnl >= TAKE_PROFIT_T2_PCT:
            reason = "TAKE_PROFIT_T2"
            break
        if (not t1_sold) and pnl >= TAKE_PROFIT_T1_PCT:
            t1_sold = True
            t1_pnl = pnl
            # T1 체결 후 같은 날 트레일링 활성 조건도 충족 (+10% ≥ +8%)
            trailing = True
            peak = c
            if k >= MAX_HOLDING_DAYS:
                reason = "MAX_HOLD"
                break
            continue
        if trailing:
            if c <= peak * (1 + TRAILING_STOP_PCT):
                reason = "TRAILING_STOP"
                break
            peak = max(peak, c)
        elif pnl >= TRAILING_ACTIVATE_PCT:
            trailing = True
            peak = c
        if k >= MAX_HOLDING_DAYS:
            reason = "MAX_HOLD"
            break
    else:
        reason = "EOD_DATA"  # 데이터 끝 — 마지막 종가 청산
    final_pnl = closes[j] / entry - 1
    total = (0.5 * t1_pnl + 0.5 * final_pnl) if t1_sold else final_pnl
    return total * 100, j, reason


def run_track(df: pd.DataFrame, entries_idx: np.ndarray, sequential: bool) -> list[dict]:
    """entries_idx 진입 후보들에 청산 엔진 적용. sequential=True면 보유 중 재진입 금지."""
    closes = df["close"].to_numpy(dtype=float)
    dates = df.index
    out = []
    last_exit = -1
    for i in entries_idx:
        if sequential and i <= last_exit:
            continue
        if i >= len(closes) - 1:
            continue  # 익일 데이터 없음
        pnl, j, reason = simulate_exit(closes, i)
        out.append({
            "entry_date": dates[i], "exit_date": dates[j],
            "pnl": pnl, "days": j - i, "reason": reason,
        })
        last_exit = j
    return out


# ──────────────────────────────────────────────
# 통계: 날짜 클러스터 t
# ──────────────────────────────────────────────
def date_cluster_stats(t: pd.DataFrame) -> dict:
    """같은 진입일 평균 → 날짜 단위 시리즈로 요약."""
    by_date = t.groupby("entry_date")["pnl"].mean()
    n = len(by_date)
    if n < 3:
        return {"n_trades": len(t), "n_dates": n, "mean": t["pnl"].mean(),
                "t_stat": np.nan, "wr": (t["pnl"] > 0).mean()}
    m = by_date.mean()
    se = by_date.std(ddof=1) / np.sqrt(n)
    return {"n_trades": len(t), "n_dates": n, "mean": t["pnl"].mean(),
            "date_mean": m, "t_stat": m / se if se > 0 else np.nan,
            "wr": (t["pnl"] > 0).mean()}


def welch_t(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    if va + vb <= 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(va + vb)


# ──────────────────────────────────────────────
def main() -> None:
    regime = build_regime_series()
    print(f"[레짐] KOSPI 시계열 {regime.index[0].date()} ~ {regime.index[-1].date()} ({len(regime)}일)")
    print(regime.value_counts().to_string())

    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    print(f"\n[유니버스] parquet {len(files)}개 로드 시작")

    sig_trades, base_trades = [], []
    n_used = 0
    for fi, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["close", "volume", "trading_value"])
        except Exception:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
        if "close" not in df.columns or len(df) < 80:
            continue
        df = df[df.index >= START_DATE]
        if len(df) < 80:
            continue
        df = df[df["close"] > 0]
        ticker = Path(f).stem
        sig, liquid = calc_signal_mask(df)
        ok = (sig & liquid).to_numpy()
        sig_idx = np.flatnonzero(ok)
        # 기저선: 21거래일 격자 (종목별 위상 분산 — 티커 해시, 결정적)
        offset = sum(ord(c) for c in ticker) % BASELINE_GRID
        grid = np.arange(offset, len(df), BASELINE_GRID)
        grid = grid[liquid.to_numpy()[grid]]

        for tr in run_track(df, sig_idx, sequential=True):
            tr["ticker"] = ticker
            sig_trades.append(tr)
        for tr in run_track(df, grid, sequential=False):
            tr["ticker"] = ticker
            base_trades.append(tr)
        n_used += 1
        if (fi + 1) % 300 == 0:
            print(f"  ...{fi + 1}/{len(files)} (시그널 {len(sig_trades)}건)")

    print(f"[완료] 종목 {n_used}개 | 시그널 진입 {len(sig_trades)}건 | 기저선 {len(base_trades)}건")

    sig_df = pd.DataFrame(sig_trades)
    base_df = pd.DataFrame(base_trades)
    for d in (sig_df, base_df):
        d["entry_date"] = pd.to_datetime(d["entry_date"]).dt.normalize()
        d["regime"] = d["entry_date"].map(regime)
        d["year"] = d["entry_date"].dt.year
    sig_df = sig_df.dropna(subset=["regime"])
    base_df = base_df.dropna(subset=["regime"])

    # 저장 (재현용)
    sig_df.to_parquet(OUT_DIR / "scan_regime_filter_signal_trades.parquet")
    base_df.to_parquet(OUT_DIR / "scan_regime_filter_baseline_trades.parquet")

    pd.set_option("display.width", 200)

    print("\n===== ① 레짐별 시그널 성과 vs 기저선 (전 기간) =====")
    rows = []
    for rg in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
        s = sig_df[sig_df.regime == rg]
        b = base_df[base_df.regime == rg]
        if len(s) == 0:
            continue
        ss = date_cluster_stats(s)
        sb = b["pnl"].mean() if len(b) else np.nan
        s_dates = s.groupby("entry_date")["pnl"].mean()
        b_dates = b.groupby("entry_date")["pnl"].mean()
        rows.append({
            "레짐": rg, "건수": ss["n_trades"], "진입일수": ss["n_dates"],
            "시그널평균%": round(ss["mean"], 2), "승률": round(ss["wr"], 2),
            "시그널t(날짜)": round(ss["t_stat"], 2),
            "기저선평균%": round(sb, 2) if sb == sb else np.nan,
            "엣지%p": round(ss["mean"] - sb, 2) if sb == sb else np.nan,
            "엣지t(Welch,날짜)": round(welch_t(s_dates, b_dates), 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== ② 연도 x 레짐 시그널 평균 pnl% (건수) — 견고성 =====")
    pv = sig_df.pivot_table(index="year", columns="regime", values="pnl", aggfunc=["mean", "size"])
    print(pv.round(2).to_string())

    print("\n===== ③ 연도 x 레짐 엣지(시그널-기저선 %p) =====")
    e_s = sig_df.pivot_table(index="year", columns="regime", values="pnl", aggfunc="mean")
    e_b = base_df.pivot_table(index="year", columns="regime", values="pnl", aggfunc="mean")
    print((e_s - e_b).round(2).to_string())

    print("\n===== ④ 반사실: CAUTION 진입 차단 시 =====")
    all_mean = sig_df["pnl"].mean()
    keep = sig_df[sig_df.regime != "CAUTION"]
    print(f"전체 유지   : {len(sig_df):6d}건, 평균 {all_mean:+.2f}%")
    print(f"CAUTION차단 : {len(keep):6d}건, 평균 {keep['pnl'].mean():+.2f}%  (거래 {len(sig_df) - len(keep)}건 제거)")
    for rg in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
        sub = sig_df[sig_df.regime == rg]
        if len(sub):
            print(f"  {rg:8s}: {len(sub):6d}건 평균 {sub['pnl'].mean():+.2f}% 합계기여 {sub['pnl'].sum():+.0f}%p")

    print("\n===== ⑤ 청산 사유 분포 (시그널) =====")
    print(sig_df.groupby("reason")["pnl"].agg(["size", "mean"]).round(2).to_string())

    # 요약 JSON
    summary = {"generated": "2026-07-09", "hypothesis": "CAUTION 레짐 SCAN 진입 차단",
               "regime_table": rows}
    (OUT_DIR / "scan_regime_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {OUT_DIR / 'scan_regime_filter_summary.json'}")


if __name__ == "__main__":
    main()
