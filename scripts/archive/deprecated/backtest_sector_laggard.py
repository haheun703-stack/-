"""섹터 래거드 추격 백테스트

가설: 섹터 리더가 D+0에 급등(+3%+)하면, 래거드(2군)가 D+1~3에 캐치업 상승.
검증: 2019~2026 전구간, 리더 급등일 → 래거드 D+1/D+3/D+5 수익률 측정.

Usage:
    python scripts/backtest_sector_laggard.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ──────────────────────────────────────────
# 1. 섹터 정의 로드
# ──────────────────────────────────────────

def load_sectors() -> list[dict]:
    """relay_sectors.yaml에서 kr_leaders + kr_secondaries 추출."""
    cfg_path = PROJECT_ROOT / "config" / "relay_sectors.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    sectors = []
    engine = cfg.get("relay_engine", cfg)
    sector_defs = engine.get("sectors", {})

    for key, sec in sector_defs.items():
        leaders = [t["ticker"] for t in sec.get("kr_leaders", [])]
        secondaries = [t["ticker"] for t in sec.get("kr_secondaries", [])]
        if leaders and secondaries:
            sectors.append({
                "name": sec.get("name", key),
                "leaders": leaders,
                "secondaries": secondaries,
            })

    return sectors


# ──────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────

def load_ticker(ticker: str) -> pd.DataFrame | None:
    """parquet 로드, close 일별 수익률 계산."""
    p = RAW_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if len(df) < 60 or "close" not in df.columns:
            return None
        df["ret"] = df["close"].pct_change()
        return df
    except Exception:
        return None


# ──────────────────────────────────────────
# 3. 백테스트 로직
# ──────────────────────────────────────────

def backtest_sector_laggard(
    sectors: list[dict],
    leader_threshold: float = 0.03,    # 리더 D+0 수익률 기준 (3%)
    min_leaders_firing: int = 1,        # 최소 N명의 리더가 급등
) -> dict:
    """섹터 래거드 추격 백테스트."""

    all_trades = []

    for sec in sectors:
        print(f"\n{'='*60}")
        print(f"  섹터: {sec['name']}")
        print(f"  리더: {sec['leaders']}")
        print(f"  래거드: {sec['secondaries']}")
        print(f"{'='*60}")

        # 리더/래거드 데이터 로드
        leader_data = {}
        for t in sec["leaders"]:
            df = load_ticker(t)
            if df is not None:
                leader_data[t] = df

        secondary_data = {}
        for t in sec["secondaries"]:
            df = load_ticker(t)
            if df is not None:
                secondary_data[t] = df

        if not leader_data or not secondary_data:
            print(f"  데이터 부족 — 스킵")
            continue

        # 공통 날짜 범위
        all_dates = None
        for df in list(leader_data.values()) + list(secondary_data.values()):
            dates = set(df.index)
            all_dates = dates if all_dates is None else all_dates & dates
        all_dates = sorted(all_dates)

        if len(all_dates) < 60:
            print(f"  공통 날짜 부족 ({len(all_dates)}일) — 스킵")
            continue

        # 시그널 스캔: 리더 급등일 찾기
        signal_count = 0
        for i, date in enumerate(all_dates):
            if i < 5 or i >= len(all_dates) - 6:
                continue

            # 리더 수익률 체크
            leaders_up = 0
            leader_avg_ret = 0
            for t, df in leader_data.items():
                if date in df.index:
                    ret = df.loc[date, "ret"]
                    if not np.isnan(ret) and ret >= leader_threshold:
                        leaders_up += 1
                    if not np.isnan(ret):
                        leader_avg_ret += ret
            leader_avg_ret /= max(len(leader_data), 1)

            if leaders_up < min_leaders_firing:
                continue

            signal_count += 1

            # 래거드 D+1~D+5 수익률 측정
            for t, df in secondary_data.items():
                if date not in df.index:
                    continue
                idx = list(df.index).index(date)
                if idx + 6 > len(df):
                    continue

                entry_price = df.iloc[idx]["close"]  # D+0 종가 매수
                d0_ret = df.iloc[idx]["ret"]

                # D+0에 래거드도 이미 급등했으면 패스 (이미 추격 완료)
                already_up = (not np.isnan(d0_ret) and d0_ret >= leader_threshold)

                d1 = df.iloc[idx + 1]["close"] / entry_price - 1
                d3 = df.iloc[idx + 3]["close"] / entry_price - 1 if idx + 3 < len(df) else None
                d5 = df.iloc[idx + 5]["close"] / entry_price - 1 if idx + 5 < len(df) else None

                all_trades.append({
                    "sector": sec["name"],
                    "date": str(date.date()),
                    "leader_ticker": ",".join(leader_data.keys()),
                    "leader_avg_ret": round(leader_avg_ret * 100, 2),
                    "laggard": t,
                    "already_up": already_up,
                    "d0_ret": round(d0_ret * 100, 2) if not np.isnan(d0_ret) else 0,
                    "d1": round(d1 * 100, 2),
                    "d3": round(d3 * 100, 2) if d3 is not None else None,
                    "d5": round(d5 * 100, 2) if d5 is not None else None,
                })

        print(f"  시그널: {signal_count}일")

    return analyze_results(all_trades)


def analyze_results(trades: list[dict]) -> dict:
    """결과 분석."""
    if not trades:
        print("\n결과 없음!")
        return {"trades": 0}

    df = pd.DataFrame(trades)

    print("\n" + "=" * 70)
    print("  섹터 래거드 추격 백테스트 결과")
    print("=" * 70)

    # 전체
    print(f"\n  전체 트레이드: {len(df)}건")

    for label, subset in [
        ("전체", df),
        ("래거드 미동(D+0 < 3%)", df[~df["already_up"]]),
        ("래거드도 급등(D+0 >= 3%)", df[df["already_up"]]),
    ]:
        if len(subset) == 0:
            continue
        print(f"\n  --- {label} (n={len(subset)}) ---")
        for col, day in [("d1", "D+1"), ("d3", "D+3"), ("d5", "D+5")]:
            valid = subset[col].dropna()
            if len(valid) == 0:
                continue
            avg = valid.mean()
            wr = (valid > 0).mean() * 100
            wins = valid[valid > 0]
            losses = valid[valid <= 0]
            pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
            print(f"    {day}: avg={avg:+.2f}%, WR={wr:.1f}%, PF={pf:.2f} (n={len(valid)})")

    # 섹터별
    print(f"\n  --- 섹터별 D+3 ---")
    not_up = df[~df["already_up"]]
    for sector in not_up["sector"].unique():
        sec_df = not_up[not_up["sector"] == sector]
        d3 = sec_df["d3"].dropna()
        if len(d3) >= 5:
            print(f"    {sector:12s}: avg={d3.mean():+.2f}%, WR={((d3>0).mean()*100):.0f}%, n={len(d3)}")

    # 리더 급등 강도별
    print(f"\n  --- 리더 급등 강도별 D+3 (래거드 미동만) ---")
    not_up = df[~df["already_up"]].copy()
    for lo, hi, label in [(3, 5, "3~5%"), (5, 8, "5~8%"), (8, 100, "8%+")]:
        subset = not_up[(not_up["leader_avg_ret"] >= lo) & (not_up["leader_avg_ret"] < hi)]
        d3 = subset["d3"].dropna()
        if len(d3) >= 5:
            print(f"    리더 {label:6s}: avg={d3.mean():+.2f}%, WR={((d3>0).mean()*100):.0f}%, n={len(d3)}")

    results = {
        "trades": len(df),
        "not_already_up": len(df[~df["already_up"]]),
        "d1_avg": round(df[~df["already_up"]]["d1"].mean(), 2),
        "d3_avg": round(df[~df["already_up"]]["d3"].dropna().mean(), 2),
        "d5_avg": round(df[~df["already_up"]]["d5"].dropna().mean(), 2),
    }
    return results


if __name__ == "__main__":
    sectors = load_sectors()
    print(f"섹터 {len(sectors)}개 로드")
    for s in sectors:
        print(f"  {s['name']}: 리더 {len(s['leaders'])}명, 래거드 {len(s['secondaries'])}명")

    # 기준 3% 급등
    backtest_sector_laggard(sectors, leader_threshold=0.03)

    # 기준 5% 급등 (더 강한 시그널)
    print("\n\n" + "#" * 70)
    print("  리더 5%+ 급등 기준")
    print("#" * 70)
    backtest_sector_laggard(sectors, leader_threshold=0.05)
