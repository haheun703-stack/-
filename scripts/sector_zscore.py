"""섹터 순환매 엔진 — Phase 1-3: 섹터 내 z-score 계산.

강세 섹터 내에서 ETF 대비 저평가(래깅) 종목을 찾는다.

z-score = (종목 수익률 - ETF 수익률 - 그룹 평균) / 그룹 표준편차

신호:
  - z_20 < -0.8: 섹터 대비 래깅 → 매수 후보 (catch-up 기대)
  - z_20 >= 0.0: 섹터 평균 수렴 → 익절 시점

사용법:
  python scripts/sector_zscore.py                    # 최신 날짜 기준
  python scripts/sector_zscore.py --top 5            # Top 5 섹터만
  python scripts/sector_zscore.py --sector 증권       # 특정 섹터
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COMP_DIR = DATA_DIR / "etf_compositions"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_sector_map() -> dict:
    """sector_map.json 로드 → {섹터명: {etf_code, stocks: [{code, name, weight}]}}"""
    path = DATA_DIR / "sector_map.json"
    if not path.exists():
        logger.error("sector_map.json 없음 — sector_etf_builder.py --init 먼저")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_momentum_ranking() -> list[dict]:
    """sector_momentum.json에서 최신 모멘텀 순위 로드."""
    path = DATA_DIR / "sector_momentum.json"
    if not path.exists():
        logger.warning("sector_momentum.json 없음 — 전체 섹터 대상으로 계산")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sectors", [])


def load_etf_close(etf_code: str) -> pd.Series | None:
    """ETF 종가 시리즈."""
    path = DAILY_DIR / f"{etf_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df["close"].astype(float)


def load_stock_close(ticker: str) -> pd.Series | None:
    """종목 종가 시리즈 (processed parquet)."""
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    if "close" in df.columns:
        return df["close"].astype(float)
    return None


# ─────────────────────────────────────────────
# z-score 계산
# ─────────────────────────────────────────────

def compute_zscore_for_sector(
    sector_name: str,
    etf_code: str,
    stocks: list[dict],
    lookback: int = 20,
) -> pd.DataFrame:
    """단일 섹터 내 종목별 z-score 계산.

    z = (stock_ret - etf_ret - mean) / std

    Returns: DataFrame with columns [sector, ticker, name, z_20, z_5, stock_ret_20, etf_ret_20]
    """
    etf_close = load_etf_close(etf_code)
    if etf_close is None:
        return pd.DataFrame()

    etf_ret = etf_close.pct_change(lookback) * 100

    # 종목별 수익률 수집
    stock_rets = {}
    stock_names = {}
    for s in stocks:
        ticker = s["code"]
        name = s["name"]
        sc = load_stock_close(ticker)
        if sc is None:
            continue
        sr = sc.pct_change(lookback) * 100
        stock_rets[ticker] = sr
        stock_names[ticker] = name

    if len(stock_rets) < 2:
        return pd.DataFrame()

    # 공통 날짜에서 계산
    latest_date = etf_ret.dropna().index.max()

    # 최신 날짜 기준 excess returns
    excess = {}
    for ticker, sr in stock_rets.items():
        common = sr.index.intersection(etf_ret.index)
        if latest_date not in common:
            continue
        excess[ticker] = float(sr.loc[latest_date]) - float(etf_ret.loc[latest_date])

    if len(excess) < 2:
        return pd.DataFrame()

    vals = list(excess.values())
    mean_excess = np.mean(vals)
    std_excess = np.std(vals, ddof=1)
    if std_excess < 0.01:
        std_excess = 0.01

    # z-score 계산 + 5일 z-score도 추가
    etf_ret5 = etf_close.pct_change(5) * 100

    rows = []
    for ticker, ex_val in excess.items():
        z_20 = (ex_val - mean_excess) / std_excess

        # 5일 z-score
        sc = load_stock_close(ticker)
        if sc is not None and latest_date in sc.index:
            sr5 = sc.pct_change(5) * 100
            if latest_date in sr5.index and latest_date in etf_ret5.index:
                excess_5 = float(sr5.loc[latest_date]) - float(etf_ret5.loc[latest_date])
            else:
                excess_5 = np.nan
        else:
            excess_5 = np.nan

        stock_ret_val = float(stock_rets[ticker].loc[latest_date]) if latest_date in stock_rets[ticker].index else np.nan

        rows.append({
            "sector": sector_name,
            "ticker": ticker,
            "name": stock_names[ticker],
            "z_20": round(z_20, 3),
            "excess_20": round(ex_val, 2),
            "excess_5": round(excess_5, 2) if not np.isnan(excess_5) else None,
            "stock_ret_20": round(stock_ret_val, 2),
            "etf_ret_20": round(float(etf_ret.loc[latest_date]), 2),
            "date": latest_date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)

    # 선행주 감지: z_5 순위 + z_5 > z_20 반전 체크
    if "excess_5" in df.columns and df["excess_5"].notna().any():
        # excess_5 기준 z_5 계산 (섹터 내 상대 5일 초과수익률)
        valid_e5 = df["excess_5"].dropna()
        if len(valid_e5) >= 2:
            mean_e5 = valid_e5.mean()
            std_e5 = valid_e5.std(ddof=1)
            if std_e5 < 0.01:
                std_e5 = 0.01
            df["z_5"] = df["excess_5"].apply(
                lambda x: round((x - mean_e5) / std_e5, 3) if pd.notna(x) else None
            )
        else:
            df["z_5"] = None

        # z_5 순위 (높을수록 먼저 반등)
        z5_valid = df[df["z_5"].notna()].copy()
        if not z5_valid.empty:
            z5_valid["z5_rank"] = z5_valid["z_5"].rank(ascending=False).astype(int)
            df = df.merge(
                z5_valid[["ticker", "z5_rank"]],
                on="ticker", how="left",
            )
        else:
            df["z5_rank"] = None

        # z_5 > z_20 반전 = 단기 회복 시작
        df["z5_reversal"] = df.apply(
            lambda r: bool(
                pd.notna(r.get("z_5")) and pd.notna(r.get("z_20"))
                and r["z_5"] > r["z_20"]
            ),
            axis=1,
        )

        # 선행주 후보: z5_rank == 1 AND z5_reversal
        df["leader_candidate"] = df.apply(
            lambda r: bool(r.get("z5_rank") == 1 and r.get("z5_reversal")),
            axis=1,
        )
    else:
        df["z_5"] = None
        df["z5_rank"] = None
        df["z5_reversal"] = False
        df["leader_candidate"] = False

    df.sort_values("z_20", inplace=True)
    return df


# ─────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────

def run_zscore_analysis(
    target_sectors: list[str] | None = None,
    top_n: int = 0,
    z_threshold: float = -0.8,
) -> dict:
    """섹터 내 z-score 분석 실행.

    Args:
        target_sectors: 특정 섹터만 분석 (None=전체)
        top_n: 모멘텀 상위 N개 섹터만 (0=전체)
        z_threshold: 매수 후보 z-score 기준 (기본 -0.8)
    """
    sector_map = load_sector_map()
    momentum_ranking = load_momentum_ranking()

    # 대상 섹터 결정
    if target_sectors:
        sectors = {k: v for k, v in sector_map.items() if k in target_sectors}
    elif top_n > 0 and momentum_ranking:
        top_sector_names = [s["sector"] for s in momentum_ranking[:top_n]]
        sectors = {k: v for k, v in sector_map.items() if k in top_sector_names}
    else:
        sectors = sector_map

    all_candidates = []
    all_results = {}

    for sector_name, info in sectors.items():
        etf_code = info["etf_code"]
        stocks = info["stocks"]

        if len(stocks) < 3:
            continue

        df = compute_zscore_for_sector(sector_name, etf_code, stocks)
        if df.empty:
            continue

        all_results[sector_name] = df

        # 매수 후보 (z < threshold)
        candidates = df[df["z_20"] <= z_threshold]
        all_candidates.append(candidates)

    # 결과 출력
    print(f"\n{'=' * 80}")
    print(f"  섹터 내 z-score 분석 (기준: z < {z_threshold})")
    print(f"{'=' * 80}")

    for sector_name, df in all_results.items():
        # 모멘텀 순위 찾기
        rank_str = ""
        for m in momentum_ranking:
            if m["sector"] == sector_name:
                rank_str = f" [모멘텀 #{m['rank']}]"
                break

        candidates = df[df["z_20"] <= z_threshold]
        header = f"  [{sector_name}]{rank_str} — ETF 20일: {df.iloc[0]['etf_ret_20']:+.2f}%"

        if len(candidates) == 0:
            print(f"\n{header}  → 래깅 종목 없음")
            continue

        print(f"\n{header}")
        print(f"  {'종목':<12} {'z_20':>6} {'초과수익':>8} {'종목20일%':>9} {'상태'}")
        print(f"  {'─' * 50}")

        for _, row in candidates.iterrows():
            z = row["z_20"]
            status = "◆ 강한래깅" if z < -1.5 else "● 래깅" if z < -1.0 else "○ 약래깅"
            print(
                f"  {row['name']:<12} {z:>+6.2f} {row['excess_20']:>+8.2f}%p "
                f"{row['stock_ret_20']:>+9.2f}%  {status}"
            )

    # 종합 후보 리스트
    if all_candidates:
        combined = pd.concat(all_candidates, ignore_index=True)
        combined.sort_values("z_20", inplace=True)

        print(f"\n{'=' * 80}")
        print(f"  종합 매수 후보 (z_20 < {z_threshold}, {len(combined)}종목)")
        print(f"{'=' * 80}")
        print(f"  {'섹터':<8} {'종목':<12} {'z_20':>6} {'초과수익':>8} {'종목20일%':>9}")
        print(f"  {'─' * 55}")

        for _, row in combined.head(20).iterrows():
            print(
                f"  {row['sector']:<8} {row['name']:<12} {row['z_20']:>+6.2f} "
                f"{row['excess_20']:>+8.2f}%p {row['stock_ret_20']:>+9.2f}%"
            )

        if len(combined) > 20:
            print(f"  ... 외 {len(combined) - 20}종목")

    else:
        combined = pd.DataFrame()

    # JSON 저장
    report = {
        "date": list(all_results.values())[0].iloc[0]["date"] if all_results else "",
        "z_threshold": z_threshold,
        "total_candidates": len(combined) if not combined.empty else 0,
        "sectors": {},
    }
    for sector_name, df in all_results.items():
        report["sectors"][sector_name] = df.to_dict(orient="records")

    out_path = DATA_DIR / "sector_zscore.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("z-score 결과 → %s", out_path)

    return report


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="섹터 내 z-score 계산")
    parser.add_argument("--top", type=int, default=0,
                        help="모멘텀 상위 N개 섹터만 (0=전체)")
    parser.add_argument("--sector", type=str, default="",
                        help="특정 섹터명 (쉼표 구분)")
    parser.add_argument("--threshold", type=float, default=-0.8,
                        help="매수 후보 z-score 기준 (기본 -0.8)")
    args = parser.parse_args()

    target_sectors = [s.strip() for s in args.sector.split(",") if s.strip()] if args.sector else None

    run_zscore_analysis(
        target_sectors=target_sectors,
        top_n=args.top,
        z_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
