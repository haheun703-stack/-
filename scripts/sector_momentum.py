"""섹터 순환매 엔진 — Phase 1-2: 섹터 모멘텀 일별 계산.

각 섹터 ETF의 모멘텀 지표를 계산하여 섹터 순위를 매긴다.

지표:
  - ret_5/ret_20/ret_60: 5/20/60일 수익률
  - vol_ratio: 거래량 vs 20일 평균 비율
  - ma20_pos: 종가 vs MA20 위치 (1=위, -1=아래)
  - rsi_14: 14일 RSI
  - rel_strength: KRX300 대비 상대강도 (20일)
  - momentum_score: 종합 모멘텀 점수 (0~100)

사용법:
  python scripts/sector_momentum.py            # 최신 날짜 기준 계산
  python scripts/sector_momentum.py --history   # 전체 기간 히스토리 생성
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = DATA_DIR / "etf_daily"
OUT_DIR = DATA_DIR / "momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# ETF 유니버스 로드
# ─────────────────────────────────────────────

def load_etf_universe() -> dict:
    """etf_universe.json 로드."""
    path = DATA_DIR / "etf_universe.json"
    if not path.exists():
        logger.error("etf_universe.json 없음 — sector_etf_builder.py --init 먼저 실행")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_etf_ohlcv(etf_code: str) -> pd.DataFrame | None:
    """ETF 일별 OHLCV parquet 로드."""
    path = DAILY_DIR / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────
# RSI 계산
# ─────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────
# 섹터 모멘텀 계산
# ─────────────────────────────────────────────

def compute_sector_momentum(universe: dict) -> pd.DataFrame:
    """모든 섹터 ETF의 모멘텀 지표를 계산.

    Returns: MultiIndex DataFrame (date, sector) → 지표 컬럼들
    """
    # KRX300 벤치마크 로드
    krx300_info = universe.get("KRX300")
    krx300_df = load_etf_ohlcv(krx300_info["etf_code"]) if krx300_info else None
    if krx300_df is not None:
        krx300_ret20 = krx300_df["close"].pct_change(20) * 100
    else:
        krx300_ret20 = None

    all_rows = []

    for sector_name, info in universe.items():
        etf_code = info["etf_code"]
        category = info["category"]

        # market 카테고리(KRX300, 코리아TOP10)는 벤치마크이므로 스킵
        if category == "market":
            continue

        df = load_etf_ohlcv(etf_code)
        if df is None or len(df) < 61:
            logger.warning("%s (%s): 데이터 부족 (최소 61일 필요)", sector_name, etf_code)
            continue

        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0, index=df.index)

        # 수익률
        ret_5 = close.pct_change(5) * 100
        ret_20 = close.pct_change(20) * 100
        ret_60 = close.pct_change(60) * 100

        # 거래량 비율
        vol_ma20 = volume.rolling(20, min_periods=10).mean()
        vol_ratio = volume / vol_ma20.replace(0, np.nan)

        # MA20 위치
        ma20 = close.rolling(20, min_periods=20).mean()
        ma20_pos = (close > ma20).astype(int) * 2 - 1  # 1=위, -1=아래

        # RSI
        rsi = calc_rsi(close, 14)

        # 상대강도 (vs KRX300)
        if krx300_ret20 is not None:
            common_idx = ret_20.index.intersection(krx300_ret20.index)
            rel_strength = ret_20.reindex(common_idx) - krx300_ret20.reindex(common_idx)
        else:
            rel_strength = pd.Series(0, index=df.index)

        # 일별 DataFrame
        sector_df = pd.DataFrame({
            "sector": sector_name,
            "etf_code": etf_code,
            "category": category,
            "close": close,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "ret_60": ret_60,
            "vol_ratio": vol_ratio,
            "ma20_pos": ma20_pos,
            "rsi_14": rsi,
            "rel_strength": rel_strength,
        })
        sector_df.index.name = "date"
        all_rows.append(sector_df)

    if not all_rows:
        logger.error("모멘텀 계산할 섹터가 없음")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=False)
    result = result.dropna(subset=["ret_20"])

    return result


# ─────────────────────────────────────────────
# 모멘텀 스코어 + 순위
# ─────────────────────────────────────────────

def score_and_rank(momentum_df: pd.DataFrame) -> pd.DataFrame:
    """날짜별로 섹터 모멘텀 스코어를 계산하고 순위를 매긴다.

    점수 배분 (100점):
      - ret_20 순위: 30점 (중기 모멘텀)
      - ret_5 순위: 20점 (단기 모멘텀)
      - rel_strength 순위: 20점 (상대강도)
      - vol_ratio 순위: 15점 (거래량 급등)
      - rsi_14 역순위: 15점 (과매수 회피, 40~60 최적)
    """
    scored = []

    for date, group in momentum_df.groupby(level=0):
        if len(group) < 3:
            continue

        n = len(group)
        g = group.copy()

        # 순위 계산 (높을수록 좋은 것 = ascending=True → 높은 값이 높은 순위)
        g["rank_ret20"] = g["ret_20"].rank(ascending=True)
        g["rank_ret5"] = g["ret_5"].rank(ascending=True)
        g["rank_rel"] = g["rel_strength"].rank(ascending=True)
        g["rank_vol"] = g["vol_ratio"].rank(ascending=True)

        # RSI: 50에 가까울수록 좋음 (과매수/과매도 회피)
        g["rsi_penalty"] = (g["rsi_14"] - 50).abs()
        g["rank_rsi"] = g["rsi_penalty"].rank(ascending=True)  # 작을수록 좋음

        # 정규화 (1~n → 0~1)
        for col in ["rank_ret20", "rank_ret5", "rank_rel", "rank_vol", "rank_rsi"]:
            g[col] = (g[col] - 1) / max(n - 1, 1)

        # 가중 합산
        g["momentum_score"] = (
            g["rank_ret20"] * 30
            + g["rank_ret5"] * 20
            + g["rank_rel"] * 20
            + g["rank_vol"] * 15
            + g["rank_rsi"] * 15
        )

        # 최종 순위
        g["rank"] = g["momentum_score"].rank(ascending=False).astype(int)

        scored.append(g)

    if not scored:
        return pd.DataFrame()

    return pd.concat(scored)


# ─────────────────────────────────────────────
# 최신 날짜 리포트
# ─────────────────────────────────────────────

def load_prev_momentum() -> dict[str, dict]:
    """전일 모멘텀 데이터 로드 → {섹터명: {rank, vol_ratio, ...}}."""
    path = DATA_DIR / "sector_momentum_prev.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["sector"]: s for s in data.get("sectors", [])}


def save_as_prev(report: dict):
    """현재 모멘텀을 전일 데이터로 저장 (다음 날 가속도 비교용)."""
    out_path = DATA_DIR / "sector_momentum_prev.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("전일 모멘텀 저장 → %s", out_path)


def print_latest_report(scored_df: pd.DataFrame) -> dict:
    """최신 날짜의 섹터 순위를 출력하고 JSON으로 반환."""
    latest_date = scored_df.index.max()
    latest = scored_df.loc[latest_date].sort_values("rank")

    # 전일 데이터 로드 (가속도 비교용)
    prev_data = load_prev_momentum()

    print(f"\n{'=' * 80}")
    print(f"  섹터 모멘텀 순위 — {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'=' * 80}")
    print(f"  {'순위':>4} {'섹터':<10} {'점수':>6} {'5일%':>7} {'20일%':>7} {'60일%':>7} {'상대강도':>8} {'거래비':>6} {'RSI':>5} {'Δ순위':>5}")
    print(f"  {'─' * 76}")

    report_data = []
    accel_count = 0
    for _, row in latest.iterrows():
        rank = int(row["rank"])
        sector = row["sector"]
        score = row["momentum_score"]
        vol_ratio = float(row["vol_ratio"])

        # 가속도 계산
        prev = prev_data.get(sector, {})
        prev_rank = prev.get("rank", rank)
        prev_vol = prev.get("vol_ratio", vol_ratio)
        rank_change = prev_rank - rank  # 양수 = 순위 상승
        vol_change_pct = ((vol_ratio - prev_vol) / prev_vol * 100) if prev_vol > 0 else 0
        accel = rank_change >= 3 and vol_change_pct >= 30
        if accel:
            accel_count += 1

        # 표시
        marker = " ★" if rank <= 3 else "  " if rank <= 7 else " ▽"
        rank_str = f"{rank_change:+d}" if prev_data else "—"
        accel_str = " ⚡" if accel else ""

        print(
            f"  {rank:>4} {sector:<10} {score:>6.1f} "
            f"{row['ret_5']:>+7.2f} {row['ret_20']:>+7.2f} {row['ret_60']:>+7.2f} "
            f"{row['rel_strength']:>+8.2f} {row['vol_ratio']:>6.2f} {row['rsi_14']:>5.1f}"
            f" {rank_str:>5}{marker}{accel_str}"
        )

        report_data.append({
            "rank": rank,
            "sector": sector,
            "etf_code": row["etf_code"],
            "category": row["category"],
            "momentum_score": round(float(score), 1),
            "ret_5": round(float(row["ret_5"]), 2),
            "ret_20": round(float(row["ret_20"]), 2),
            "ret_60": round(float(row["ret_60"]), 2),
            "rel_strength": round(float(row["rel_strength"]), 2),
            "vol_ratio": round(float(row["vol_ratio"]), 2),
            "rsi_14": round(float(row["rsi_14"]), 1),
            "rank_prev": prev_rank,
            "rank_change": rank_change,
            "vol_change_pct": round(vol_change_pct, 1),
            "acceleration": accel,
        })

    print(f"\n  ★ = Top 3   ▽ = Bottom 3   ⚡ = 가속 (순위 +3↑ & 거래량 +30%↑)")
    if accel_count:
        print(f"  가속 감지: {accel_count}개 섹터")

    # JSON 저장
    report = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "sectors": report_data,
    }
    out_path = DATA_DIR / "sector_momentum.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("최신 모멘텀 → %s", out_path)

    # 현재 데이터를 전일로 저장 (다음 날 가속도 비교용)
    save_as_prev(report)

    return report


# ─────────────────────────────────────────────
# 히스토리 저장
# ─────────────────────────────────────────────

def save_history(scored_df: pd.DataFrame):
    """전체 히스토리를 parquet으로 저장."""
    out_path = OUT_DIR / "momentum_history.parquet"
    scored_df.to_parquet(out_path)
    logger.info(
        "모멘텀 히스토리 → %s (%d행, %s ~ %s)",
        out_path, len(scored_df),
        scored_df.index.min().strftime("%Y-%m-%d"),
        scored_df.index.max().strftime("%Y-%m-%d"),
    )


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 모멘텀 일별 계산")
    parser.add_argument("--history", action="store_true",
                        help="전체 기간 히스토리 생성")
    args = parser.parse_args()

    universe = load_etf_universe()
    logger.info("유니버스: %d개 섹터 ETF", len(universe))

    # 모멘텀 계산
    momentum_df = compute_sector_momentum(universe)
    if momentum_df.empty:
        logger.error("모멘텀 데이터가 비어있음")
        return

    logger.info("모멘텀 계산: %d행 (%d섹터)",
                len(momentum_df), momentum_df["sector"].nunique())

    # 스코어 + 순위
    scored = score_and_rank(momentum_df)
    if scored.empty:
        logger.error("스코어 계산 실패")
        return

    # 최신 리포트
    print_latest_report(scored)

    # 히스토리 저장
    if args.history:
        save_history(scored)


if __name__ == "__main__":
    main()
