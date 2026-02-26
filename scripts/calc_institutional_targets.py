"""
기관 추정 목표가 역산 엔진 — 662종목 전체

4축 가중평균:
  VPOC_60일     × 40%  (가장 많이 거래된 가격대)
  외인매수 VWAP × 30%  (외인이 산 평균 가격)
  피보나치 클러스터 × 20% (기술적 수렴점)
  MA120         × 10%  (장기 기관 기준가)

D존 분류:
  D-3: 목표가 대비 -10% 이하 (깊은 저평가, 분할매수 1차)
  D-2: -5% ~ -10% (저평가, 분할매수 2차)
  D-1: 0% ~ -5% (근접, 분할매수 3차)
  도달: 0% ~ +5% (관망)
  초과: +5% 이상 (차익실현 압력)

출력: data/institutional_targets.json
BAT-D 19.5단계 (내일 추천 직전)

Usage:
    python scripts/calc_institutional_targets.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = PROJECT_ROOT / "data" / "institutional_targets.json"
DB_PATH = PROJECT_ROOT / "data" / "jarvis_archive.db"

# ── 파라미터 ──
LOOKBACK = 60
VPOC_BINS = 50
MIN_FOREIGN_BUY_DAYS = 10

W_VPOC = 0.40
W_FWAP = 0.30
W_FIB = 0.20
W_MA120 = 0.10


def build_name_map() -> dict[str, str]:
    """종목코드 → 종목명 매핑."""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def calc_vpoc_60d(df: pd.DataFrame) -> float | None:
    """최근 60일 가격대별 거래량 분포에서 최대 거래량 가격 (VPOC).

    일봉 [low, high]를 VPOC_BINS등분하여 각 bin에 volume/구간수 할당.
    """
    recent = df.tail(LOOKBACK)
    if len(recent) < 30:
        return None

    lows = recent["low"].values
    highs = recent["high"].values
    vols = recent["volume"].values

    price_min = float(np.nanmin(lows))
    price_max = float(np.nanmax(highs))

    if price_max <= price_min or price_min <= 0:
        return None

    bin_size = (price_max - price_min) / VPOC_BINS
    volume_profile = np.zeros(VPOC_BINS)

    for i in range(len(recent)):
        low_i = float(lows[i])
        high_i = float(highs[i])
        vol_i = float(vols[i])

        if vol_i <= 0 or np.isnan(vol_i):
            continue

        bin_lo = max(0, int((low_i - price_min) / bin_size))
        bin_hi = min(VPOC_BINS - 1, int((high_i - price_min) / bin_size))
        n_bins = bin_hi - bin_lo + 1

        if n_bins > 0:
            vol_per_bin = vol_i / n_bins
            volume_profile[bin_lo : bin_hi + 1] += vol_per_bin

    max_bin = int(np.argmax(volume_profile))
    vpoc = price_min + (max_bin + 0.5) * bin_size

    return round(vpoc, 0)


def calc_foreign_vwap(df: pd.DataFrame) -> float | None:
    """최근 60일 중 외인 순매수일의 가중평균매수단가 추정."""
    recent = df.tail(LOOKBACK)
    if len(recent) < 30:
        return None

    # 외국인합계 컬럼 (일별 순매수 금액)
    if "외국인합계" in recent.columns:
        foreign_col = recent["외국인합계"].fillna(0)
        buy_mask = foreign_col > 0
    else:
        # fallback: foreign_net_5d (정확도 낮지만 대안)
        fn5 = recent.get("foreign_net_5d", pd.Series(0, index=recent.index))
        buy_mask = fn5.fillna(0) > 0

    buy_days = recent[buy_mask]

    if len(buy_days) < MIN_FOREIGN_BUY_DAYS:
        buy_days = recent  # 순매수일 부족 → 전체 VWAP

    closes = buy_days["close"].values
    volumes = buy_days["volume"].values

    total_cv = float(np.nansum(closes * volumes))
    total_v = float(np.nansum(volumes))

    if total_v <= 0:
        return None

    return round(total_cv / total_v, 0)


def calc_fib_cluster(df: pd.DataFrame, vpoc: float | None) -> float | None:
    """60일 고저 피보나치 되돌림 중 VPOC에 가장 가까운 수준."""
    recent = df.tail(LOOKBACK)
    if len(recent) < 30:
        return None

    high_60 = float(recent["high"].max())
    low_60 = float(recent["low"].min())
    diff = high_60 - low_60

    if diff <= 0:
        return None

    fib_382 = low_60 + diff * 0.382
    fib_500 = low_60 + diff * 0.500
    fib_618 = low_60 + diff * 0.618
    fibs = [fib_382, fib_500, fib_618]

    reference = vpoc if vpoc is not None else float(recent["close"].iloc[-1])
    closest = min(fibs, key=lambda x: abs(x - reference))

    return round(closest, 0)


def calc_confidence(
    vpoc: float | None,
    fwap: float | None,
    fib: float | None,
    ma120: float | None,
    current_close: float,
    n_days: int,
) -> float:
    """목표가 신뢰도 (0~1)."""
    # 거래일수 (0~0.30)
    if n_days >= 50:
        day_score = 0.30
    elif n_days >= 30:
        day_score = 0.15
    else:
        day_score = 0.05

    # 구성요소 존재 (0~0.30)
    components = [vpoc, fwap, fib, ma120]
    valid_count = sum(1 for c in components if c is not None)
    component_score = valid_count / 4 * 0.30

    # VPOC-FWAP 수렴도 (0~0.40)
    convergence_score = 0.0
    if vpoc is not None and fwap is not None and current_close > 0:
        gap_pct = abs(vpoc - fwap) / current_close * 100
        if gap_pct < 3:
            convergence_score = 0.40
        elif gap_pct < 5:
            convergence_score = 0.30
        elif gap_pct < 10:
            convergence_score = 0.15
        else:
            convergence_score = 0.05
    elif valid_count >= 2:
        convergence_score = 0.10

    return round(min(day_score + component_score + convergence_score, 1.0), 2)


def classify_zone(gap_pct: float) -> str:
    """현재가 vs 목표가 갭 기준 D존 분류.

    gap_pct = (current - target) / target * 100
    음수 = 목표가 대비 할인(저평가)
    """
    if gap_pct <= -10:
        return "D-3"
    elif gap_pct <= -5:
        return "D-2"
    elif gap_pct <= 0:
        return "D-1"
    elif gap_pct <= 5:
        return "도달"
    else:
        return "초과"


def init_vpoc_table():
    """vpoc_history 테이블 생성 (없으면)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vpoc_history (
        date   TEXT NOT NULL,
        ticker TEXT NOT NULL,
        vpoc   REAL,
        fwap   REAL,
        target REAL,
        PRIMARY KEY (date, ticker)
    )""")
    conn.commit()
    conn.close()


def save_vpoc_history(date_str: str, targets: dict):
    """오늘자 VPOC/FWAP/목표가 히스토리를 SQLite에 저장."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    rows = [
        (date_str, ticker, t.get("vpoc_60d"), t.get("foreign_vwap"), t.get("estimated_target"))
        for ticker, t in targets.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO vpoc_history (date, ticker, vpoc, fwap, target) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    logger.info("[VPOC 히스토리] %d건 저장 (날짜: %s)", len(rows), date_str)


def calc_velocity(targets: dict) -> dict[str, dict]:
    """과거 히스토리 대비 목표가 이동 속도(velocity) 계산.

    Returns: {ticker: {"delta_5d": float|None, "delta_10d": float|None, "direction": str}}
    direction: RISING(+2%↑) / STABLE / FALLING(-2%↓) / NEW(히스토리 부족)
    """
    conn = sqlite3.connect(str(DB_PATH))
    velocity = {}

    for ticker, t in targets.items():
        rows = conn.execute(
            "SELECT date, target FROM vpoc_history WHERE ticker=? ORDER BY date DESC LIMIT 21",
            (ticker,),
        ).fetchall()

        cur_target = t["estimated_target"]

        if len(rows) < 2:
            velocity[ticker] = {"delta_5d": None, "delta_10d": None, "direction": "NEW"}
            continue

        # 5일 전 target (가용한 만큼)
        idx_5 = min(5, len(rows) - 1)
        d5_target = rows[idx_5][1]
        delta_5 = round((cur_target - d5_target) / d5_target * 100, 2) if d5_target and d5_target > 0 else None

        # 10일 전 target
        delta_10 = None
        if len(rows) > 5:
            idx_10 = min(10, len(rows) - 1)
            d10_target = rows[idx_10][1]
            delta_10 = round((cur_target - d10_target) / d10_target * 100, 2) if d10_target and d10_target > 0 else None

        # 방향 분류 (5일 우선, 없으면 10일)
        ref = delta_5 if delta_5 is not None else delta_10
        if ref is None:
            direction = "NEW"
        elif ref > 2:
            direction = "RISING"
        elif ref < -2:
            direction = "FALLING"
        else:
            direction = "STABLE"

        velocity[ticker] = {"delta_5d": delta_5, "delta_10d": delta_10, "direction": direction}

    conn.close()
    return velocity


def calc_single_target(ticker: str) -> dict | None:
    """단일 종목의 추정 기관 목표가 계산."""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None

    try:
        df = pd.read_parquet(pq_path)
        if len(df) < 30:
            return None

        current_close = float(df["close"].iloc[-1])
        if current_close <= 0 or np.isnan(current_close):
            return None

        n_days = min(len(df), LOOKBACK)

        vpoc = calc_vpoc_60d(df)
        fwap = calc_foreign_vwap(df)
        fib = calc_fib_cluster(df, vpoc)

        ma120 = None
        if "sma_120" in df.columns:
            raw = df["sma_120"].iloc[-1]
            if not (pd.isna(raw) or float(raw) <= 0):
                ma120 = round(float(raw), 0)

        # 4축 가중평균 (None 축 제외 후 재정규화)
        weighted_sum = 0.0
        weight_sum = 0.0
        for val, w in [(vpoc, W_VPOC), (fwap, W_FWAP), (fib, W_FIB), (ma120, W_MA120)]:
            if val is not None and val > 0:
                weighted_sum += val * w
                weight_sum += w

        if weight_sum <= 0:
            return None

        estimated_target = round(weighted_sum / weight_sum, 0)
        gap_pct = round((current_close - estimated_target) / estimated_target * 100, 2)
        zone = classify_zone(gap_pct)
        confidence = calc_confidence(vpoc, fwap, fib, ma120, current_close, n_days)

        return {
            "vpoc_60d": int(vpoc) if vpoc else None,
            "foreign_vwap": int(fwap) if fwap else None,
            "fib_cluster": int(fib) if fib else None,
            "ma120": int(ma120) if ma120 else None,
            "estimated_target": int(estimated_target),
            "current_close": int(current_close),
            "gap_pct": gap_pct,
            "zone": zone,
            "confidence": confidence,
        }
    except Exception as e:
        logger.debug("목표가 계산 실패 %s: %s", ticker, e)
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # SQLite 테이블 초기화
    init_vpoc_table()

    name_map = build_name_map()
    parquet_files = sorted(PROCESSED_DIR.glob("*.parquet"))
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[기관목표가] 계산 대상: {len(parquet_files)}종목")

    targets = {}
    zone_stats = {"D-3": 0, "D-2": 0, "D-1": 0, "도달": 0, "초과": 0}

    for pf in parquet_files:
        ticker = pf.stem
        result = calc_single_target(ticker)
        if result:
            result["name"] = name_map.get(ticker, ticker)
            targets[ticker] = result
            zone_stats[result["zone"]] += 1

    # VPOC 히스토리 SQLite 저장 (velocity 계산 전에 저장해야 오늘 데이터 포함)
    save_vpoc_history(today_str, targets)

    # Velocity 계산 (과거 히스토리 대비 목표가 이동 속도)
    velocity = calc_velocity(targets)
    dir_stats = {"RISING": 0, "STABLE": 0, "FALLING": 0, "NEW": 0}
    for ticker, v in velocity.items():
        targets[ticker]["target_delta_5d"] = v["delta_5d"]
        targets[ticker]["target_delta_10d"] = v["delta_10d"]
        targets[ticker]["target_direction"] = v["direction"]
        dir_stats[v["direction"]] += 1

    output = {
        "date": today_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_stocks": len(parquet_files),
        "calculated": len(targets),
        "zone_distribution": zone_stats,
        "velocity_distribution": dir_stats,
        "targets": targets,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    # 요약 출력
    print(f"\n{'='*55}")
    print(f"[기관목표가] {len(targets)}/{len(parquet_files)}종목 계산 완료")
    print(f"{'='*55}")
    for zone in ["D-3", "D-2", "D-1", "도달", "초과"]:
        cnt = zone_stats.get(zone, 0)
        pct = cnt / len(targets) * 100 if targets else 0
        print(f"  {zone}: {cnt}종목 ({pct:.1f}%)")

    print(f"\n[Velocity 분포]")
    for d in ["RISING", "STABLE", "FALLING", "NEW"]:
        print(f"  {d}: {dir_stats[d]}종목")

    # D-3 (깊은 저평가) 종목 TOP 10 출력
    d3_stocks = [
        (t, v) for t, v in targets.items() if v["zone"] == "D-3"
    ]
    d3_stocks.sort(key=lambda x: x[1]["gap_pct"])

    if d3_stocks:
        print(f"\n{'─'*55}")
        print(f"  D-3 깊은 저평가 TOP 10 (분할매수 1차 대상)")
        print(f"{'─'*55}")
        for ticker, info in d3_stocks[:10]:
            dir_icon = {"RISING": "▲", "FALLING": "▼", "STABLE": "─", "NEW": "★"}.get(info.get("target_direction", ""), "")
            print(
                f"  {info['name']}({ticker}) "
                f"현재:{info['current_close']:,} → 목표:{info['estimated_target']:,} "
                f"({info['gap_pct']:+.1f}%) 신뢰:{info['confidence']:.0%} {dir_icon}"
            )

    print(f"\n[저장] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
