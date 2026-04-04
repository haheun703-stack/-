"""복합 시그널 백테스트 — PULLBACK + 수급 강화 조건

기존 STRONG_ALPHA인 PULLBACK_20MA_15pct를 기반으로,
수급 조건을 강화하면 알파가 더 높아지는지 검증.

조건 변형:
  1. PB15_FOR3d     : 이격-15% + 외인 3일 연속 순매수
  2. PB15_INST3d    : 이격-15% + 기관 3일 연속 순매수
  3. PB15_DUAL      : 이격-15% + 기관+외인 동시 순매수 (당일)
  4. PB15_FOR_REV   : 이격-15% + 외인 5일 매도→매수 반전
  5. PB15_VOL3x     : 이격-15% + 거래량 3배 폭발 + 수급
  6. PB10_FOR3d     : 이격-10% + 외인 3일 연속 (느슨한 PB + 엄격한 수급)
  7. PB15_BB        : 이격-15% + 볼린저밴드 하단 이탈 + 수급
  8. PB15_FOR3d_BEAR: 이격-15% + 외인 3일 + BEAR 레짐 (역발상)

비교 기준: 기존 PULLBACK_20MA_15pct (D+5 +2.25%, WR 55.5%, PF 2.02)

Usage (VPS):
    python -u scripts/backtest_combo.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULT_PATH = DATA_DIR / "alpha_backtest" / "combo_result.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_parquet(ticker: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def future_returns(df, signal_dates, horizons=None):
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]
    results = []
    for d in signal_dates:
        if d not in df.index:
            after = df.index[df.index >= d]
            if after.empty:
                continue
            d = after[0]
        entry = df.loc[d, "close"]
        if entry <= 0:
            continue
        row = {"signal_date": d.strftime("%Y-%m-%d"), "entry_price": entry}
        for h in horizons:
            future = df.index[df.index > d]
            if len(future) < h:
                row[f"D+{h}"] = np.nan
            else:
                row[f"D+{h}"] = round((df.loc[future[h-1], "close"] - entry) / entry * 100, 2)
        results.append(row)
    return pd.DataFrame(results)


def summarize(df, name):
    horizons = [c for c in df.columns if c.startswith("D+")]
    n = len(df)
    s = {"signal": name, "n": n}
    for h in horizons:
        col = df[h].dropna()
        if col.empty:
            continue
        wins = (col > 0).sum()
        s[h] = {
            "mean": round(col.mean(), 2),
            "median": round(col.median(), 2),
            "wr": round(wins / len(col) * 100, 1),
            "pf": round(abs(col[col > 0].sum() / col[col <= 0].sum()), 2) if col[col <= 0].sum() != 0 else 99,
            "n": len(col),
        }
    d5 = s.get("D+5", {})
    if d5:
        if d5["mean"] > 1.5 and d5["wr"] > 55:
            s["verdict"] = "STRONG_ALPHA"
        elif d5["mean"] > 0.8 and d5["wr"] > 52:
            s["verdict"] = "WEAK_ALPHA"
        elif d5["mean"] > 0 and d5["wr"] > 48:
            s["verdict"] = "MARGINAL"
        else:
            s["verdict"] = "NO_ALPHA"
    return s


def consec_buy(series: pd.Series) -> pd.Series:
    """연속 순매수 일수 계산."""
    is_buy = (series > 0).astype(int)
    result = is_buy.copy()
    for i in range(1, len(result)):
        if result.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
        else:
            result.iloc[i] = 0
    return result


def flow_reversal(series: pd.Series, sell_days: int = 5) -> pd.Series:
    """N일 매도 후 매수 전환 시그널."""
    is_sell = (series < 0).astype(int)
    is_buy = (series > 0)

    # 직전 sell_days일 중 (sell_days-1)일 이상 매도
    sell_count = is_sell.rolling(sell_days).sum().shift(1)  # 오늘 제외, 직전 N일
    reversal = (sell_count >= sell_days - 1) & is_buy

    return reversal


def run_backtest():
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)
    logger.info("총 %d 종목", total)

    # 시그널별 수집 버퍼
    SIGNALS = [
        "PB15_BASE",       # 기준선: 기존 PULLBACK_15%
        "PB15_FOR3d",      # + 외인 3일 연속
        "PB15_INST3d",     # + 기관 3일 연속
        "PB15_DUAL",       # + 기관+외인 동시 당일
        "PB15_FOR_REV",    # + 외인 반전 (5일 매도→매수)
        "PB15_VOL3x",      # + 거래량 3배 + 수급
        "PB10_FOR3d",      # 이격-10% + 외인 3일
        "PB15_BB",         # + 볼린저밴드 하단 + 수급
    ]
    buffers = {s: [] for s in SIGNALS}

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        df = load_parquet(ticker)
        if df is None or len(df) < 60:
            continue
        if "기관합계" not in df.columns or "외국인합계" not in df.columns:
            continue

        close = df["close"]
        inst = df["기관합계"].fillna(0)
        foreign = df["외국인합계"].fillna(0)
        volume = df["volume"].fillna(0)

        # 공통: 이격도
        ma20 = close.rolling(20).mean()
        gap = (close - ma20) / ma20 * 100
        pb15 = (gap <= -15) & (ma20 > 0)  # 20MA 이격 -15%
        pb10 = (gap <= -10) & (ma20 > 0)  # 20MA 이격 -10%

        # 수급 파생
        supply_any = (inst > 0) | (foreign > 0)
        inst_consec = consec_buy(inst)
        for_consec = consec_buy(foreign)
        dual_today = (inst > 0) & (foreign > 0)
        for_rev = flow_reversal(foreign, sell_days=5)

        # 거래량
        vol_ma20 = volume.rolling(20).mean()
        vol_3x = volume > (vol_ma20 * 3)

        # 볼린저밴드 하단
        bb_std = close.rolling(20).std()
        bb_lower = ma20 - 2 * bb_std
        below_bb = close < bb_lower

        # ── 시그널 생성 ──

        # 1. 기준선 (기존 PULLBACK_15%)
        mask = pb15 & supply_any
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_BASE"].append(r)

        # 2. PB15 + 외인 3일 연속
        mask = pb15 & (for_consec >= 3)
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_FOR3d"].append(r)

        # 3. PB15 + 기관 3일 연속
        mask = pb15 & (inst_consec >= 3)
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_INST3d"].append(r)

        # 4. PB15 + 쌍끌이 당일
        mask = pb15 & dual_today
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_DUAL"].append(r)

        # 5. PB15 + 외인 반전
        mask = pb15 & for_rev
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_FOR_REV"].append(r)

        # 6. PB15 + 거래량 3배 + 수급
        mask = pb15 & vol_3x & supply_any
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_VOL3x"].append(r)

        # 7. PB10 + 외인 3일 연속
        mask = pb10 & (for_consec >= 3)
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB10_FOR3d"].append(r)

        # 8. PB15 + 볼린저하단 + 수급
        mask = pb15 & below_bb & supply_any
        dates = df.index[mask].tolist()
        if dates:
            r = future_returns(df, dates)
            if not r.empty:
                r["ticker"] = ticker
                buffers["PB15_BB"].append(r)

        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d", i + 1, total)

    # ── 결과 요약 ──
    final = {}
    for sig in SIGNALS:
        dfs = buffers[sig]
        if not dfs:
            final[sig] = {"signal": sig, "n": 0, "verdict": "NO_DATA"}
            logger.info("  %s: 0건", sig)
            continue

        combined = pd.concat(dfs, ignore_index=True)
        s = summarize(combined, sig)
        final[sig] = s

        d5 = s.get("D+5", {})
        logger.info("  %-18s n=%5d  D+5 avg=%+6.2f%%  WR=%5.1f%%  PF=%5.2f  → %s",
                    sig, s["n"],
                    d5.get("mean", 0), d5.get("wr", 0), d5.get("pf", 0),
                    s.get("verdict", "?"))

    # 저장
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # ── 리포트 ──
    print()
    print("=" * 85)
    print("  복합 시그널 백테스트 — PULLBACK + 수급 강화")
    print("  비교 기준: PULLBACK_20MA_15pct (D+5 +2.25%, WR 55.5%, PF 2.02)")
    print("=" * 85)
    print()
    print("  %-18s %6s %8s %8s %6s %6s %10s" % ("시그널", "건수", "D+5 avg", "D+5 med", "WR", "PF", "판정"))
    print("  " + "-" * 70)
    for sig in SIGNALS:
        s = final[sig]
        d5 = s.get("D+5", {})
        d5_med = d5.get("median", 0)
        print("  %-18s %6d %7.2f%% %7.2f%% %5.1f%% %6.2f %10s" % (
            sig, s["n"],
            d5.get("mean", 0), d5_med, d5.get("wr", 0), d5.get("pf", 0),
            s.get("verdict", "NO_DATA"),
        ))

    # D+1, D+3, D+10 도 표시
    print()
    print("  ── D+1/3/10/20 상세 ──")
    print("  %-18s %8s %8s %8s %8s" % ("시그널", "D+1", "D+3", "D+10", "D+20"))
    print("  " + "-" * 50)
    for sig in SIGNALS:
        s = final[sig]
        vals = []
        for h in ["D+1", "D+3", "D+10", "D+20"]:
            d = s.get(h, {})
            vals.append("%+.2f%%" % d.get("mean", 0) if d else "  N/A  ")
        print("  %-18s %8s %8s %8s %8s" % (sig, *vals))

    # 승자 강조
    print()
    strong = [k for k, v in final.items() if v.get("verdict") == "STRONG_ALPHA"]
    weak = [k for k, v in final.items() if v.get("verdict") == "WEAK_ALPHA"]
    if strong:
        print("  ★ STRONG_ALPHA: %s" % ", ".join(strong))
    if weak:
        print("  ○ WEAK_ALPHA: %s" % ", ".join(weak))
    if not strong and not weak:
        print("  ✗ STRONG/WEAK 알파 없음 — 기존 PULLBACK_15%가 최적")

    # 기존 대비 개선 여부
    print()
    base = final.get("PB15_BASE", {}).get("D+5", {})
    if base:
        base_mean = base.get("mean", 0)
        for sig in SIGNALS[1:]:
            s5 = final.get(sig, {}).get("D+5", {})
            if s5 and s5.get("mean", 0) > base_mean:
                improvement = s5["mean"] - base_mean
                print("  ↑ %s: 기존 대비 +%.2f%%p 개선!" % (sig, improvement))


if __name__ == "__main__":
    run_backtest()
