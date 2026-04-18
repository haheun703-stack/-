#!/usr/bin/env python3
"""4일 릴레이 가설 백테스트 — 퐝가님 관찰 패턴 검증

퐝가님 가설:
  D+0 (시드) : +1~4% 빨간 스타트, 거래량 1.5~2배, 수급 첫 진입
  D+1 (확장) : +4~7%, 쌍끌이 전환, 거래량 계속 증가
  D+2 (피크) : +8~12% 급등, 거래량 폭발
  D+3 (소강) : ±3% 횡보, 차익실현 시작

검증 목표:
  D+0 시드 조건 충족 → D+1/D+2/D+3/D+4 수익률 분포
  STRONG_ALPHA 기준: D+1~D+2 avg +5%+, WR 60%+

Usage:
    python scripts/backtest_4day_relay.py                  # 기본 1년 백테스트
    python scripts/backtest_4day_relay.py --days 500        # 500거래일
    python scripts/backtest_4day_relay.py --loose            # 완화된 조건
    python scripts/backtest_4day_relay.py --strict           # 엄격 조건
    python scripts/backtest_4day_relay.py --no-supply        # 수급 필터 제외
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "stock_data_daily"

# ─── 시드 조건 프리셋 ───

PRESETS = {
    "default": {
        "ret_min": 1.0,       # D+0 수익률 하한 (%)
        "ret_max": 5.0,       # D+0 수익률 상한 (%)
        "vol_ratio_min": 1.5, # 거래량/MA20 하한
        "vol_ratio_max": 3.0, # 거래량/MA20 상한
        "ma20_dev_min": -3.0, # 20MA 이격 하한 (%)
        "ma20_dev_max": 5.0,  # 20MA 이격 상한 (%)
        "vol5d_max": 5.0,     # 5일 변동성 상한 (%)
        "ret60_max": 40.0,    # 60일 수익률 상한 (%)
        "tv_min": 50.0,       # 거래대금 하한 (억원)
        "tv_max": 5000.0,     # 거래대금 상한 (억원)
        "supply_min": 30.0,   # 수급 임계값 (억원) — 외인 or 기관
    },
    "loose": {
        "ret_min": 0.5,
        "ret_max": 7.0,
        "vol_ratio_min": 1.2,
        "vol_ratio_max": 5.0,
        "ma20_dev_min": -5.0,
        "ma20_dev_max": 10.0,
        "vol5d_max": 8.0,
        "ret60_max": 60.0,
        "tv_min": 30.0,
        "tv_max": 10000.0,
        "supply_min": 20.0,
    },
    "strict": {
        "ret_min": 1.0,
        "ret_max": 4.0,
        "vol_ratio_min": 1.5,
        "vol_ratio_max": 3.0,
        "ma20_dev_min": -3.0,
        "ma20_dev_max": 3.0,
        "vol5d_max": 4.0,
        "ret60_max": 30.0,
        "tv_min": 50.0,
        "tv_max": 3000.0,
        "supply_min": 50.0,
    },
}


@dataclass
class SeedHit:
    """시드 조건 충족 이벤트."""
    ticker: str
    name: str
    date: str          # D+0 날짜
    close: float
    ret_d0: float      # D+0 수익률 (%)
    vol_ratio: float   # 거래량/MA20
    ma20_dev: float    # 20MA 이격 (%)
    vol5d: float       # 5일 변동성 (%)
    ret60: float       # 60일 누적 수익률 (%)
    tv: float          # 거래대금 (억원)
    foreign_net: float # 외인순매수 (억원)
    inst_net: float    # 기관순매수 (억원)
    # D+1~D+4 수익률 (D+0 종가 기준)
    ret_d1: float = np.nan
    ret_d2: float = np.nan
    ret_d3: float = np.nan
    ret_d4: float = np.nan
    # D+1~D+4 개별 일간 수익률 (전일 대비)
    daily_d1: float = np.nan
    daily_d2: float = np.nan
    daily_d3: float = np.nan
    daily_d4: float = np.nan


def load_stock_csv(path: Path, lookback: int = 400) -> pd.DataFrame | None:
    """CSV 로드 후 최근 lookback행만 반환."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if len(df) < 30:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        # 최근 lookback행만 (MA20 계산 버퍼 포함)
        if len(df) > lookback:
            df = df.iloc[-lookback:].reset_index(drop=True)
        return df
    except Exception:
        return None


def compute_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """시드 판별에 필요한 메트릭 계산."""
    df = df.copy()

    # 일간 수익률
    df["daily_ret"] = df["Close"].pct_change() * 100

    # 거래량 20일 이동평균
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"]

    # 20MA 이격률 (CSV에 MA20 있지만 NaN일 수 있으므로 재계산)
    ma20 = df["Close"].rolling(20).mean()
    df["ma20_dev"] = (df["Close"] / ma20 - 1) * 100

    # 5일 변동성 (일간 수익률의 표준편차)
    df["vol5d"] = df["daily_ret"].rolling(5).std()

    # 60일 누적 수익률
    df["ret60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100

    # 거래대금 (억원)
    df["trading_value"] = df["Close"] * df["Volume"] / 1e8

    return df


def find_seeds(df: pd.DataFrame, params: dict, use_supply: bool,
               lookback_days: int) -> list[dict]:
    """DataFrame에서 시드 조건 충족 행을 찾아 D+1~D+4 수익률 추적."""
    df = compute_seed_metrics(df)
    n = len(df)

    # 백테스트 범위: 최근 lookback_days 거래일 (D+4 여유 위해 -4)
    start_idx = max(25, n - lookback_days)  # MA20 최소 25행 필요
    end_idx = n - 4  # D+4까지 추적 가능해야 함

    if start_idx >= end_idx:
        return []

    results = []
    for i in range(start_idx, end_idx):
        row = df.iloc[i]

        # 기본 NaN 체크
        if pd.isna(row.get("daily_ret")) or pd.isna(row.get("vol_ratio")):
            continue
        if pd.isna(row.get("ma20_dev")) or pd.isna(row.get("vol5d")):
            continue

        ret_d0 = row["daily_ret"]
        vol_ratio = row["vol_ratio"]
        ma20_dev = row["ma20_dev"]
        vol5d = row["vol5d"]
        ret60 = row.get("ret60", 0)
        tv = row.get("trading_value", 0)
        foreign_net = row.get("Foreign_Net", 0) or 0
        inst_net = row.get("Inst_Net", 0) or 0

        if pd.isna(ret60):
            ret60 = 0
        if pd.isna(tv):
            continue

        # ─── 시드 조건 검증 ───
        if not (params["ret_min"] <= ret_d0 <= params["ret_max"]):
            continue
        if not (params["vol_ratio_min"] <= vol_ratio <= params["vol_ratio_max"]):
            continue
        if not (params["ma20_dev_min"] <= ma20_dev <= params["ma20_dev_max"]):
            continue
        if vol5d > params["vol5d_max"]:
            continue
        if ret60 > params["ret60_max"]:
            continue
        if not (params["tv_min"] <= tv <= params["tv_max"]):
            continue

        # 수급 필터 (선택)
        if use_supply:
            has_supply = (foreign_net >= params["supply_min"] or
                          inst_net >= params["supply_min"])
            if not has_supply:
                continue

        # ─── D+1~D+4 수익률 계산 ───
        close_d0 = row["Close"]
        close_d1 = df.iloc[i + 1]["Close"]
        close_d2 = df.iloc[i + 2]["Close"]
        close_d3 = df.iloc[i + 3]["Close"]
        close_d4 = df.iloc[i + 4]["Close"]

        ret_d1 = (close_d1 / close_d0 - 1) * 100
        ret_d2 = (close_d2 / close_d0 - 1) * 100
        ret_d3 = (close_d3 / close_d0 - 1) * 100
        ret_d4 = (close_d4 / close_d0 - 1) * 100

        daily_d1 = (close_d1 / close_d0 - 1) * 100
        daily_d2 = (close_d2 / close_d1 - 1) * 100
        daily_d3 = (close_d3 / close_d2 - 1) * 100
        daily_d4 = (close_d4 / close_d3 - 1) * 100

        results.append({
            "date": str(row["Date"].date()),
            "close": close_d0,
            "ret_d0": round(ret_d0, 2),
            "vol_ratio": round(vol_ratio, 2),
            "ma20_dev": round(ma20_dev, 2),
            "vol5d": round(vol5d, 2),
            "ret60": round(ret60, 2),
            "tv": round(tv, 1),
            "foreign_net": round(foreign_net, 1),
            "inst_net": round(inst_net, 1),
            "ret_d1": round(ret_d1, 2),
            "ret_d2": round(ret_d2, 2),
            "ret_d3": round(ret_d3, 2),
            "ret_d4": round(ret_d4, 2),
            "daily_d1": round(daily_d1, 2),
            "daily_d2": round(daily_d2, 2),
            "daily_d3": round(daily_d3, 2),
            "daily_d4": round(daily_d4, 2),
        })

    return results


def calc_stats(values: list[float]) -> dict:
    """수익률 리스트 통계."""
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"n": 0, "avg": 0, "med": 0, "wr": 0, "pf": 0, "std": 0,
                "q25": 0, "q75": 0, "max": 0, "min": 0}

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    total_win = wins.sum() if len(wins) > 0 else 0
    total_loss = abs(losses.sum()) if len(losses) > 0 else 0.001

    return {
        "n": len(arr),
        "avg": round(float(arr.mean()), 2),
        "med": round(float(np.median(arr)), 2),
        "wr": round(float(len(wins) / len(arr) * 100), 1),
        "pf": round(float(total_win / total_loss), 2),
        "std": round(float(arr.std()), 2),
        "q25": round(float(np.percentile(arr, 25)), 2),
        "q75": round(float(np.percentile(arr, 75)), 2),
        "max": round(float(arr.max()), 2),
        "min": round(float(arr.min()), 2),
    }


def classify_alpha(stats_d2: dict) -> str:
    """알파 등급 분류."""
    if stats_d2["n"] < 30:
        return "INSUFFICIENT_DATA"
    if stats_d2["avg"] >= 5.0 and stats_d2["wr"] >= 60.0 and stats_d2["pf"] >= 2.0:
        return "STRONG_ALPHA"
    if stats_d2["avg"] >= 3.0 and stats_d2["wr"] >= 55.0 and stats_d2["pf"] >= 1.5:
        return "MODERATE_ALPHA"
    if stats_d2["avg"] >= 1.5 and stats_d2["wr"] >= 50.0 and stats_d2["pf"] >= 1.2:
        return "WEAK_ALPHA"
    if stats_d2["avg"] >= 0.5:
        return "MARGINAL"
    return "NO_ALPHA"


def run_backtest(args) -> list[dict]:
    """전체 백테스트 실행."""
    # 프리셋 선택
    if args.strict:
        params = PRESETS["strict"]
        preset_name = "strict"
    elif args.loose:
        params = PRESETS["loose"]
        preset_name = "loose"
    else:
        params = PRESETS["default"]
        preset_name = "default"

    use_supply = not args.no_supply
    lookback = args.days + 70  # MA60/ret60 계산 버퍼

    logger.info("=== 4일 릴레이 백테스트 시작 ===")
    logger.info("프리셋: %s | 기간: %d거래일 | 수급필터: %s",
                preset_name, args.days, "ON" if use_supply else "OFF")

    # CSV 파일 수집
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    logger.info("CSV 파일: %d개", len(csv_files))

    all_hits = []
    processed = 0
    t0 = time.time()

    for path in csv_files:
        # 파일명에서 종목명/티커 추출
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, ticker = parts

        df = load_stock_csv(path, lookback=lookback)
        if df is None:
            continue

        hits = find_seeds(df, params, use_supply, args.days)
        for h in hits:
            h["ticker"] = ticker
            h["name"] = name
        all_hits.extend(hits)

        processed += 1
        if processed % 500 == 0:
            logger.info("  %d/%d 처리... (시드 %d건)", processed, len(csv_files), len(all_hits))

    elapsed = time.time() - t0
    logger.info("스캔 완료: %d종목 / %.1f초 / 시드 %d건", processed, elapsed, len(all_hits))

    return all_hits


def print_results(hits: list[dict], preset_name: str, use_supply: bool):
    """결과 출력."""
    if not hits:
        print("\n시드 조건 충족 건수: 0 → 조건 완화 필요")
        return

    # 누적 수익률 (D+0 종가 대비)
    ret_d1 = [h["ret_d1"] for h in hits]
    ret_d2 = [h["ret_d2"] for h in hits]
    ret_d3 = [h["ret_d3"] for h in hits]
    ret_d4 = [h["ret_d4"] for h in hits]

    # 일간 수익률 (전일 대비)
    daily_d1 = [h["daily_d1"] for h in hits]
    daily_d2 = [h["daily_d2"] for h in hits]
    daily_d3 = [h["daily_d3"] for h in hits]
    daily_d4 = [h["daily_d4"] for h in hits]

    stats_d1 = calc_stats(ret_d1)
    stats_d2 = calc_stats(ret_d2)
    stats_d3 = calc_stats(ret_d3)
    stats_d4 = calc_stats(ret_d4)

    daily_stats_d1 = calc_stats(daily_d1)
    daily_stats_d2 = calc_stats(daily_d2)
    daily_stats_d3 = calc_stats(daily_d3)
    daily_stats_d4 = calc_stats(daily_d4)

    alpha = classify_alpha(stats_d2)

    print()
    print("=" * 72)
    print(f"  4일 릴레이 백테스트 결과 — 프리셋: {preset_name} | 수급: {'ON' if use_supply else 'OFF'}")
    print("=" * 72)
    print(f"\n  시드 건수: {len(hits)}건")
    print(f"  알파 등급: {alpha}")

    # ─── 누적 수익률 (D+0 종가 대비) ───
    print(f"\n  {'─' * 68}")
    print(f"  {'누적 수익률 (D+0 종가 대비)':^60}")
    print(f"  {'─' * 68}")
    print(f"  {'':>8} {'n':>5} {'avg':>7} {'med':>7} {'WR%':>6} {'PF':>6} {'std':>6} {'Q25':>7} {'Q75':>7}")
    print(f"  {'D+1':>8} {stats_d1['n']:>5} {stats_d1['avg']:>+7.2f} {stats_d1['med']:>+7.2f} "
          f"{stats_d1['wr']:>5.1f}% {stats_d1['pf']:>6.2f} {stats_d1['std']:>6.2f} "
          f"{stats_d1['q25']:>+7.2f} {stats_d1['q75']:>+7.2f}")
    print(f"  {'D+2':>8} {stats_d2['n']:>5} {stats_d2['avg']:>+7.2f} {stats_d2['med']:>+7.2f} "
          f"{stats_d2['wr']:>5.1f}% {stats_d2['pf']:>6.2f} {stats_d2['std']:>6.2f} "
          f"{stats_d2['q25']:>+7.2f} {stats_d2['q75']:>+7.2f}")
    print(f"  {'D+3':>8} {stats_d3['n']:>5} {stats_d3['avg']:>+7.2f} {stats_d3['med']:>+7.2f} "
          f"{stats_d3['wr']:>5.1f}% {stats_d3['pf']:>6.2f} {stats_d3['std']:>6.2f} "
          f"{stats_d3['q25']:>+7.2f} {stats_d3['q75']:>+7.2f}")
    print(f"  {'D+4':>8} {stats_d4['n']:>5} {stats_d4['avg']:>+7.2f} {stats_d4['med']:>+7.2f} "
          f"{stats_d4['wr']:>5.1f}% {stats_d4['pf']:>6.2f} {stats_d4['std']:>6.2f} "
          f"{stats_d4['q25']:>+7.2f} {stats_d4['q75']:>+7.2f}")

    # ─── 일간 수익률 (전일 대비) — 릴레이 패턴 검증 ───
    print(f"\n  {'─' * 68}")
    print(f"  {'일간 수익률 (전일 대비) — 릴레이 가속 패턴 검증':^60}")
    print(f"  {'─' * 68}")
    print(f"  {'':>8} {'n':>5} {'avg':>7} {'med':>7} {'WR%':>6} {'PF':>6}")
    print(f"  {'D+1':>8} {daily_stats_d1['n']:>5} {daily_stats_d1['avg']:>+7.2f} "
          f"{daily_stats_d1['med']:>+7.2f} {daily_stats_d1['wr']:>5.1f}% {daily_stats_d1['pf']:>6.2f}")
    print(f"  {'D+2':>8} {daily_stats_d2['n']:>5} {daily_stats_d2['avg']:>+7.2f} "
          f"{daily_stats_d2['med']:>+7.2f} {daily_stats_d2['wr']:>5.1f}% {daily_stats_d2['pf']:>6.2f}")
    print(f"  {'D+3':>8} {daily_stats_d3['n']:>5} {daily_stats_d3['avg']:>+7.2f} "
          f"{daily_stats_d3['med']:>+7.2f} {daily_stats_d3['wr']:>5.1f}% {daily_stats_d3['pf']:>6.2f}")
    print(f"  {'D+4':>8} {daily_stats_d4['n']:>5} {daily_stats_d4['avg']:>+7.2f} "
          f"{daily_stats_d4['med']:>+7.2f} {daily_stats_d4['wr']:>5.1f}% {daily_stats_d4['pf']:>6.2f}")

    # ─── 릴레이 가속 패턴 빈도 분석 ───
    print(f"\n  {'─' * 68}")
    print(f"  {'릴레이 가속 패턴 빈도':^60}")
    print(f"  {'─' * 68}")

    # 패턴 A: D+1 > D+0 가속 (확장)
    accel_d1 = sum(1 for h in hits if h["daily_d1"] > h["ret_d0"])
    # 패턴 B: D+1 > D+0 AND D+2 > D+1 (연속 가속)
    accel_d12 = sum(1 for h in hits
                    if h["daily_d1"] > h["ret_d0"] and h["daily_d2"] > h["daily_d1"])
    # 패턴 C: 3일 연속 양봉 (D+1,D+2,D+3 모두 양수)
    triple_green = sum(1 for h in hits
                       if h["daily_d1"] > 0 and h["daily_d2"] > 0 and h["daily_d3"] > 0)
    # 패턴 D: 4일 릴레이 완성 (D+1,D+2 양수 + D+3 둔화)
    relay_complete = sum(1 for h in hits
                         if h["daily_d1"] > 0 and h["daily_d2"] > 0
                         and abs(h["daily_d3"]) < abs(h["daily_d2"]))
    # 패턴 E: D+2에 +5% 이상 폭등
    d2_surge = sum(1 for h in hits if h["ret_d2"] >= 5.0)

    n = len(hits)
    print(f"  D+1 가속 (daily_d1 > ret_d0)       : {accel_d1:>5}건 ({accel_d1/n*100:.1f}%)")
    print(f"  D+1→D+2 연속 가속                   : {accel_d12:>5}건 ({accel_d12/n*100:.1f}%)")
    print(f"  3일 연속 양봉 (D+1,D+2,D+3)         : {triple_green:>5}건 ({triple_green/n*100:.1f}%)")
    print(f"  릴레이 완성 (D+1,D+2↑ + D+3 둔화)   : {relay_complete:>5}건 ({relay_complete/n*100:.1f}%)")
    print(f"  D+2 누적 +5% 이상                   : {d2_surge:>5}건 ({d2_surge/n*100:.1f}%)")

    # ─── 릴레이 완성 건만 수익률 ───
    if relay_complete > 0:
        relay_hits = [h for h in hits
                      if h["daily_d1"] > 0 and h["daily_d2"] > 0
                      and abs(h["daily_d3"]) < abs(h["daily_d2"])]
        relay_d2 = calc_stats([h["ret_d2"] for h in relay_hits])
        relay_d3 = calc_stats([h["ret_d3"] for h in relay_hits])
        relay_alpha = classify_alpha(relay_d2)
        print(f"\n  {'─' * 68}")
        print(f"  {'릴레이 완성 건 수익률':^60}")
        print(f"  {'─' * 68}")
        print(f"  건수: {relay_d2['n']} | 알파: {relay_alpha}")
        print(f"  D+2 누적: avg {relay_d2['avg']:+.2f}% | WR {relay_d2['wr']:.1f}% | PF {relay_d2['pf']:.2f}")
        print(f"  D+3 누적: avg {relay_d3['avg']:+.2f}% | WR {relay_d3['wr']:.1f}% | PF {relay_d3['pf']:.2f}")

    # ─── D+2 +5% 이상 건만 ───
    if d2_surge > 0:
        surge_hits = [h for h in hits if h["ret_d2"] >= 5.0]
        surge_d3 = calc_stats([h["ret_d3"] for h in surge_hits])
        surge_d4 = calc_stats([h["ret_d4"] for h in surge_hits])
        print(f"\n  {'─' * 68}")
        print(f"  {'D+2 +5% 이상 폭등 건 후속 수익률':^60}")
        print(f"  {'─' * 68}")
        print(f"  건수: {len(surge_hits)}")
        print(f"  D+3 누적: avg {surge_d3['avg']:+.2f}% | WR {surge_d3['wr']:.1f}% | PF {surge_d3['pf']:.2f}")
        print(f"  D+4 누적: avg {surge_d4['avg']:+.2f}% | WR {surge_d4['wr']:.1f}% | PF {surge_d4['pf']:.2f}")

    # ─── TOP 사례 (D+2 수익률 상위 10건) ───
    print(f"\n  {'─' * 68}")
    print(f"  {'TOP 10 시드 사례 (D+2 수익률 순)':^60}")
    print(f"  {'─' * 68}")
    top10 = sorted(hits, key=lambda h: h["ret_d2"], reverse=True)[:10]
    print(f"  {'날짜':>12} {'종목':>12} {'D+0':>6} {'D+1':>6} {'D+2':>6} {'D+3':>6} {'D+4':>6} {'거래대금':>8}")
    for h in top10:
        print(f"  {h['date']:>12} {h['name'][:10]:>12} "
              f"{h['ret_d0']:>+5.1f}% {h['ret_d1']:>+5.1f}% {h['ret_d2']:>+5.1f}% "
              f"{h['ret_d3']:>+5.1f}% {h['ret_d4']:>+5.1f}% {h['tv']:>7.0f}억")

    # ─── WORST 사례 (D+2 수익률 하위 10건) ───
    print(f"\n  {'─' * 68}")
    print(f"  {'WORST 10 시드 사례 (D+2 수익률 순)':^60}")
    print(f"  {'─' * 68}")
    worst10 = sorted(hits, key=lambda h: h["ret_d2"])[:10]
    print(f"  {'날짜':>12} {'종목':>12} {'D+0':>6} {'D+1':>6} {'D+2':>6} {'D+3':>6} {'D+4':>6} {'거래대금':>8}")
    for h in worst10:
        print(f"  {h['date']:>12} {h['name'][:10]:>12} "
              f"{h['ret_d0']:>+5.1f}% {h['ret_d1']:>+5.1f}% {h['ret_d2']:>+5.1f}% "
              f"{h['ret_d3']:>+5.1f}% {h['ret_d4']:>+5.1f}% {h['tv']:>7.0f}억")

    print("=" * 72)

    return {
        "n": len(hits),
        "alpha": alpha,
        "d1": stats_d1,
        "d2": stats_d2,
        "d3": stats_d3,
        "d4": stats_d4,
        "relay_complete": relay_complete,
        "d2_surge": d2_surge,
    }


def save_csv(hits: list[dict], output_path: Path):
    """시드 전체 결과 CSV 저장."""
    if not hits:
        return
    df = pd.DataFrame(hits)
    cols = ["date", "ticker", "name", "close", "ret_d0", "vol_ratio", "ma20_dev",
            "vol5d", "ret60", "tv", "foreign_net", "inst_net",
            "ret_d1", "ret_d2", "ret_d3", "ret_d4",
            "daily_d1", "daily_d2", "daily_d3", "daily_d4"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("결과 저장: %s (%d건)", output_path, len(df))


def main():
    parser = argparse.ArgumentParser(description="4일 릴레이 가설 백테스트")
    parser.add_argument("--days", type=int, default=260, help="백테스트 거래일수 (기본 260)")
    parser.add_argument("--loose", action="store_true", help="완화된 조건")
    parser.add_argument("--strict", action="store_true", help="엄격 조건")
    parser.add_argument("--no-supply", action="store_true", help="수급 필터 제외")
    parser.add_argument("--output", type=str, default=None, help="결과 CSV 경로")
    args = parser.parse_args()

    hits = run_backtest(args)

    preset_name = "strict" if args.strict else ("loose" if args.loose else "default")
    use_supply = not args.no_supply

    result = print_results(hits, preset_name, use_supply)

    # CSV 저장
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "data" / f"relay_backtest_{preset_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(hits, out_path)


if __name__ == "__main__":
    main()
