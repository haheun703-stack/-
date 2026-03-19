"""Alpha Engine V2 — STEP 4 밸류 팩터 독립 백테스트

V1 EBITDA/EV, V2 FCF Yield, V통합의 예측력을 quintile 분석으로 검증.

사용법:
  python -u -X utf8 scripts/v2_value_factor_backtest.py

출력:
  data/v2_migration/value_factor_backtest.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_monthly_returns(lookback_months: int = 36) -> pd.DataFrame:
    """data/processed/*.parquet → 월말 수익률 DataFrame."""
    parquet_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(parquet_dir.glob("*.parquet"))

    logger.info("parquet 로드: %d개 파일...", len(files))

    all_close: dict[str, pd.Series] = {}
    for f in files:
        ticker = f.stem
        if not ticker[-1].isdigit() or ticker[-1] == "5":
            continue
        try:
            df = pd.read_parquet(f, columns=["close"])
            if df.empty or len(df) < 60:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            all_close[ticker] = df["close"]
        except Exception:
            continue

    logger.info("  %d종목 close 로드 완료", len(all_close))

    close_df = pd.DataFrame(all_close)
    monthly = close_df.resample("ME").last()
    monthly = monthly.tail(lookback_months + 1)
    returns = monthly.pct_change().dropna(how="all")

    logger.info("  월간 수익률: %d개월 × %d종목", len(returns), len(returns.columns))
    return returns


def quintile_analysis(
    scores: dict[str, float],
    monthly_returns: pd.DataFrame,
) -> dict:
    """quintile별 성과 분석."""
    common = sorted(set(scores.keys()) & set(monthly_returns.columns))
    if len(common) < 50:
        return {"error": f"공통 종목 부족: {len(common)}"}

    sorted_tickers = sorted(common, key=lambda t: scores.get(t, 0.5))
    n = len(sorted_tickers)
    quintile_size = n // 5

    quintiles = []
    for q in range(5):
        start = q * quintile_size
        end = (q + 1) * quintile_size if q < 4 else n
        q_tickers = sorted_tickers[start:end]

        q_returns = monthly_returns[q_tickers].mean(axis=1).dropna()

        gains = q_returns[q_returns > 0].sum()
        losses = abs(q_returns[q_returns < 0].sum())
        pf = gains / losses if losses > 0 else float("inf")

        cum_return = (1 + q_returns).prod() - 1
        avg_monthly = q_returns.mean()

        quintiles.append({
            "quintile": q + 1,
            "n_stocks": len(q_tickers),
            "avg_monthly_return": round(float(avg_monthly), 6),
            "cumulative_return": round(float(cum_return), 4),
            "pf": round(float(pf), 4),
            "sharpe": round(float(avg_monthly / q_returns.std() * np.sqrt(12)), 4)
            if q_returns.std() > 0
            else 0.0,
        })

    top_pf = quintiles[4]["pf"]
    top_minus_bottom = quintiles[4]["avg_monthly_return"] - quintiles[0]["avg_monthly_return"]
    avg_returns = [q["avg_monthly_return"] for q in quintiles]
    monotonic = all(avg_returns[i] <= avg_returns[i + 1] for i in range(4))

    return {
        "quintiles": quintiles,
        "top_pf": round(float(top_pf), 4),
        "top_minus_bottom": round(float(top_minus_bottom), 6),
        "monotonic": monotonic,
        "n_common": len(common),
    }


def main():
    from src.alpha.factors.value_ebitda_ev import ValueEbitdaEV, ValueFCFYield
    from src.alpha.factors.value_composite import ValueComposite

    # 1. 월간 수익률 로드
    monthly_returns = load_monthly_returns(lookback_months=36)

    # 2. 독립 팩터 테스트
    factors = {
        "V1_EBITDA_EV": ValueEbitdaEV(),
        "V2_FCF_Yield": ValueFCFYield(),
    }

    results = {}
    logger.info("\n" + "=" * 70)
    logger.info("V1/V2 독립 팩터 Quintile 분석")
    logger.info("=" * 70)

    for name, factor in factors.items():
        scores = factor.score_universe()
        analysis = quintile_analysis(scores, monthly_returns)

        if "error" in analysis:
            logger.warning("%s: %s", name, analysis["error"])
            results[name] = analysis
            continue

        results[name] = analysis

        logger.info("\n--- %s (N=%d) ---", name, analysis["n_common"])
        logger.info(
            "  %-8s  %6s  %10s  %10s  %8s  %8s",
            "Quintile", "N", "Avg Monthly", "Cumulative", "PF", "Sharpe",
        )
        for q in analysis["quintiles"]:
            logger.info(
                "  Q%-7d  %6d  %10.4f%%  %10.2f%%  %8.2f  %8.2f",
                q["quintile"],
                q["n_stocks"],
                q["avg_monthly_return"] * 100,
                q["cumulative_return"] * 100,
                q["pf"],
                q["sharpe"],
            )
        logger.info(
            "  Top PF: %.2f | Top-Bottom: %.4f%% | Monotonic: %s",
            analysis["top_pf"],
            analysis["top_minus_bottom"] * 100,
            "✓" if analysis["monotonic"] else "✗",
        )

        passed = analysis["top_pf"] >= 1.1
        logger.info("  판정: %s (PF %.2f %s 1.1)", "PASS ✓" if passed else "FAIL ✗",
                    analysis["top_pf"], ">=" if passed else "<")

    # 3. V 통합 스코어 (레짐별)
    logger.info("\n" + "=" * 70)
    logger.info("V 통합 스코어 Quintile 분석 (레짐별)")
    logger.info("=" * 70)

    composite = ValueComposite()

    for regime in ["BULL", "CAUTION", "BEAR", "CRISIS"]:
        composite_scores = composite.score_universe(regime)
        analysis = quintile_analysis(composite_scores, monthly_returns)

        if "error" in analysis:
            logger.warning("V_Composite_%s: %s", regime, analysis["error"])
            results[f"V_Composite_{regime}"] = analysis
            continue

        results[f"V_Composite_{regime}"] = analysis

        logger.info("\n--- V_Composite_%s (N=%d) ---", regime, analysis["n_common"])
        logger.info(
            "  %-8s  %6s  %10s  %10s  %8s  %8s",
            "Quintile", "N", "Avg Monthly", "Cumulative", "PF", "Sharpe",
        )
        for q in analysis["quintiles"]:
            logger.info(
                "  Q%-7d  %6d  %10.4f%%  %10.2f%%  %8.2f  %8.2f",
                q["quintile"],
                q["n_stocks"],
                q["avg_monthly_return"] * 100,
                q["cumulative_return"] * 100,
                q["pf"],
                q["sharpe"],
            )
        logger.info(
            "  Top PF: %.2f | Top-Bottom: %.4f%% | Monotonic: %s",
            analysis["top_pf"],
            analysis["top_minus_bottom"] * 100,
            "✓" if analysis["monotonic"] else "✗",
        )

        top_sharpe = analysis["quintiles"][4]["sharpe"]
        logger.info(
            "  판정: PF %s (%.2f) | Sharpe %s (%.2f)",
            "PASS ✓" if analysis["top_pf"] >= 1.1 else "FAIL ✗", analysis["top_pf"],
            "PASS ✓" if top_sharpe >= 0.5 else "FAIL ✗", top_sharpe,
        )

    # 4. 기존 S2 단독 vs V 통합 비교 요약
    logger.info("\n" + "=" * 70)
    logger.info("기존 S2(PF 1.01) vs V 통합 비교")
    logger.info("=" * 70)
    caution = results.get("V_Composite_CAUTION", {})
    if "quintiles" in caution:
        logger.info("  V 통합 CAUTION: Top PF %.2f, Top Sharpe %.2f",
                    caution["top_pf"], caution["quintiles"][4]["sharpe"])
        logger.info("  기존 S2: Top PF 1.01 (STEP 1 결과)")
        if caution["top_pf"] > 1.01:
            logger.info("  → V 통합이 기존 S2보다 개선됨 ✓")

    # 5. 결과 저장
    output_path = PROJECT_ROOT / "data" / "v2_migration" / "value_factor_backtest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("\n저장: %s", output_path)


if __name__ == "__main__":
    main()
