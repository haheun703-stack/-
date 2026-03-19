"""STEP 6-1 검증: UnifiedV2Scorer 동작 확인

사용법:
  python -u -X utf8 scripts/v2_unified_scorer_test.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

import pandas as pd
import yaml

from src.alpha.factors.unified_scorer import UnifiedV2Scorer
from src.alpha.factors.regime_weighted_scorer import RegimeWeightedScorer
from src.alpha.models import AlphaRegimeLevel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # 1. 설정 로드
    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("UnifiedV2Scorer (4팩터) 검증 시작")

    # 2. 스코어러 초기화
    v2_scorer = UnifiedV2Scorer(config)
    v1_scorer = RegimeWeightedScorer(config)

    logger.info("  V2 scorer 초기화 완료")

    # 3. 샘플 종목 로드
    parquet_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(parquet_dir.glob("*.parquet"))[:20]  # 20종목 샘플

    logger.info("  %d종목 로드 시작...", len(files))

    results = []
    for f in files:
        ticker = f.stem
        if not ticker[-1].isdigit() or ticker[-1] == "5":
            continue

        try:
            df = pd.read_parquet(f)
            if len(df) < 60:
                continue

            row = df.iloc[-1]  # 최신 행

            for regime in [AlphaRegimeLevel.BULL, AlphaRegimeLevel.CAUTION, AlphaRegimeLevel.BEAR]:
                # V2 (4팩터)
                v2_result = v2_scorer.score(row, ticker, regime)

                # V1 (5축 재가중치)
                v1_result = v1_scorer.score(row, regime)

                results.append({
                    "ticker": ticker,
                    "regime": regime.value,
                    "v1_score": v1_result.total_score,
                    "v1_grade": v1_result.grade,
                    "v2_score": v2_result.total_score,
                    "v2_grade": v2_result.grade,
                    "v2_factors": {
                        s.name: round(s.weighted, 4) for s in v2_result.scores
                    },
                })
        except Exception as e:
            logger.warning("  %s 실패: %s", ticker, e)
            continue

    if not results:
        logger.error("결과 없음!")
        return

    # 4. 결과 출력
    logger.info("\n" + "=" * 80)
    logger.info("V1 (5축 재가중치) vs V2 (4팩터 통합) 비교")
    logger.info("=" * 80)

    logger.info(
        "\n  %-10s %-8s  %8s %-3s  %8s %-3s  %s",
        "Ticker", "Regime", "V1", "Grd", "V2", "Grd", "V2 팩터 (weighted)",
    )
    logger.info("  " + "-" * 75)

    for r in results:
        factors_str = " | ".join(
            f"{k}:{v:.3f}" for k, v in r["v2_factors"].items()
        )
        logger.info(
            "  %-10s %-8s  %8.4f %-3s  %8.4f %-3s  %s",
            r["ticker"], r["regime"],
            r["v1_score"], r["v1_grade"],
            r["v2_score"], r["v2_grade"],
            factors_str,
        )

    # 5. 등급 분포 요약
    logger.info("\n" + "=" * 80)
    logger.info("등급 분포 비교")
    logger.info("=" * 80)

    for regime in ["BULL", "CAUTION", "BEAR"]:
        r_items = [r for r in results if r["regime"] == regime]

        v1_grades = {}
        v2_grades = {}
        for r in r_items:
            v1_grades[r["v1_grade"]] = v1_grades.get(r["v1_grade"], 0) + 1
            v2_grades[r["v2_grade"]] = v2_grades.get(r["v2_grade"], 0) + 1

        logger.info(
            "\n  [%s] V1: %s  |  V2: %s",
            regime,
            dict(sorted(v1_grades.items())),
            dict(sorted(v2_grades.items())),
        )

    # 6. 4팩터 가중치 표시
    logger.info("\n" + "=" * 80)
    logger.info("4팩터 레짐별 가중치")
    logger.info("=" * 80)

    for regime in [AlphaRegimeLevel.BULL, AlphaRegimeLevel.CAUTION, AlphaRegimeLevel.BEAR, AlphaRegimeLevel.CRISIS]:
        w = v2_scorer.get_weights(regime)
        logger.info("  [%s] SD=%.2f  M=%.2f  V=%.2f  Q=%.2f", regime.value, w["sd"], w["momentum"], w["value"], w["quality"])

    logger.info("\n검증 완료!")


if __name__ == "__main__":
    main()
