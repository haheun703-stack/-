#!/usr/bin/env python3
"""
"기대가 식은 자리" v2.1 -- 듀얼 트리거 백테스트 메인 파이프라인

v2.1 핵심 변경:
  - 3단계 분할 매수: Impulse(40%) -> Confirm(40%) -> Breakout(20%)
  - 모드별 손절: Impulse -3% (타이트), Confirm -5% (여유)
  - 트리거별 성과 분리 통계 (승률/손익비/보유기간)

사용법:
    python main.py                    # 전체 파이프라인 실행
    python main.py --step collect     # 데이터 수집만
    python main.py --step indicators  # 지표 계산만
    python main.py --step backtest    # 백테스트만
    python main.py --step report      # 리포트만
    python main.py --sample           # 샘플 데이터로 테스트
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def step_collect(use_sample: bool = False):
    """Step 1: 데이터 수집"""
    from src.data_collector import DataCollector, generate_sample_data, PYKRX_AVAILABLE

    if use_sample or not PYKRX_AVAILABLE:
        logger.info("[Step 1] 샘플 데이터 생성 모드")
        generate_sample_data(n_stocks=30)
    else:
        logger.info("[Step 1] KRX 실제 데이터 수집 시작 (3~4시간 소요)")
        collector = DataCollector()
        collector.collect_all()


def step_indicators():
    """Step 2: 기술적 지표 계산 (v2.1 트리거 지표 포함)"""
    from src.indicators import IndicatorEngine

    engine = IndicatorEngine()
    count = engine.process_all()
    logger.info(f"[Step 2] 지표 계산 완료: {count}종목 (ATR/RSI/SMA + Trigger 지표)")


def step_backtest():
    """Step 3~7: 백테스트 실행 (스크리닝 + BES + 듀얼 트리거 매매)"""
    from src.backtest_engine import BacktestEngine

    engine = BacktestEngine()
    data = engine.load_data()

    if not data:
        logger.error("[Step 3] processed 디렉토리에 데이터 없음. 먼저 collect + indicators 실행 필요")
        return None

    results = engine.run(data)

    # 백테스트 요약 출력
    stats = results.get("stats", {})
    _print_summary(stats)

    return results


def _print_summary(stats: dict):
    """백테스트 핵심 성과 요약 출력"""
    if not stats:
        return

    logger.info("")
    logger.info("=" * 55)
    logger.info("  [백테스트 결과 요약]")
    logger.info("=" * 55)
    logger.info(f"  총 수익률:    {stats.get('total_return_pct', 0):+.1f}%")
    logger.info(f"  CAGR:         {stats.get('cagr_pct', 0):.1f}%")
    logger.info(f"  MDD:          {stats.get('max_drawdown_pct', 0):.1f}%")
    logger.info(f"  샤프 비율:    {stats.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  승률:         {stats.get('win_rate', 0):.1f}%")
    logger.info(f"  손익비:       1:{stats.get('avg_rr_ratio', 0):.1f}")
    logger.info(f"  총 거래:      {stats.get('total_trades', 0)}회")
    logger.info(f"  Profit Factor:{stats.get('profit_factor', 0):.2f}")
    logger.info("-" * 55)

    # 등급별 분석
    grade_bd = stats.get("grade_breakdown", {})
    if grade_bd:
        logger.info("  [등급별 성과]")
        for grade in ["A", "B", "C"]:
            g = grade_bd.get(grade, {})
            if g.get("count", 0) > 0:
                logger.info(
                    f"    {grade}등급: {g['count']}건, "
                    f"승률 {g.get('win_rate', 0):.1f}%, "
                    f"평균 {g.get('avg_pnl_pct', 0):+.2f}%"
                )
        logger.info("-" * 55)

    # 트리거별 분석 (v2.1)
    trigger_bd = stats.get("trigger_breakdown", {})
    if trigger_bd:
        logger.info("  [트리거별 성과] (v2.1)")
        labels = {
            "impulse": "Impulse(시동)",
            "confirm": "Confirm(확인)",
            "breakout": "Breakout(돌파)",
        }
        for key, label in labels.items():
            t = trigger_bd.get(key, {})
            if t.get("count", 0) > 0:
                logger.info(
                    f"    {label}: {t['count']}건, "
                    f"승률 {t.get('win_rate', 0):.1f}%, "
                    f"평균 {t.get('avg_pnl_pct', 0):+.2f}%, "
                    f"총손익 {t.get('total_pnl', 0):+,}원"
                )

    logger.info("=" * 55)


def step_report(results: dict = None):
    """Step 8: HTML 리포트 생성 (트리거별 성과 포함)"""
    from src.report_generator import ReportGenerator
    import pandas as pd

    generator = ReportGenerator()

    if results:
        path = generator.generate(
            stats=results["stats"],
            trades_df=results["trades_df"],
            equity_df=results["equity_df"],
            signals_df=results["signals_df"],
        )
    else:
        # 저장된 CSV에서 로드
        results_dir = Path("results")
        trades_df = pd.read_csv(results_dir / "trades_log.csv") if (results_dir / "trades_log.csv").exists() else pd.DataFrame()
        equity_df = pd.read_csv(results_dir / "daily_equity.csv") if (results_dir / "daily_equity.csv").exists() else pd.DataFrame()
        signals_df = pd.read_csv(results_dir / "signals_log.csv") if (results_dir / "signals_log.csv").exists() else pd.DataFrame()

        if equity_df.empty:
            logger.error("[Step 8] 백테스트 결과 없음. 먼저 backtest 실행 필요")
            return

        # stats 재계산
        from src.backtest_engine import BacktestEngine
        engine = BacktestEngine()
        stats = engine._calc_stats(trades_df, equity_df)
        path = generator.generate(stats, trades_df, equity_df, signals_df)

    logger.info(f"[Step 8] 리포트 생성 완료: {path}")


def run_full_pipeline(use_sample: bool = False):
    """전체 파이프라인 실행"""
    start_time = time.time()

    print("""
    ======================================================
      "기대가 식은 자리" v2.1 -- 듀얼 트리거 백테스트
    ======================================================
      Step 1: 데이터 수집 (KRX / 샘플)
      Step 2: 기술적 지표 계산 (19개 + 트리거 지표)
      Step 3: Pre-screening (매출/거래대금/영업이익)
      Step 4: Gate 필터 (추세 + 분배 리스크)
      Step 5: BES 스코어링 + 등급 (A/B/C)
      Step 6: 듀얼 트리거 매매 (Impulse 40% + Confirm 40%)
      Step 7: 포지션 관리 (모드별 손절/익절/트레일링)
      Step 8: HTML 리포트 생성
    ======================================================
    """)

    # Step 1: 데이터 수집
    logger.info("=" * 50)
    logger.info("[Step 1/4] 데이터 수집")
    logger.info("=" * 50)
    step_collect(use_sample)

    # Step 2: 지표 계산
    logger.info("=" * 50)
    logger.info("[Step 2/4] 기술적 지표 계산")
    logger.info("=" * 50)
    step_indicators()

    # Step 3~7: 백테스트
    logger.info("=" * 50)
    logger.info("[Step 3/4] 백테스트 실행 (스크리닝 + BES + 듀얼 트리거)")
    logger.info("=" * 50)
    results = step_backtest()

    # Step 8: 리포트
    if results:
        logger.info("=" * 50)
        logger.info("[Step 4/4] HTML 리포트 생성")
        logger.info("=" * 50)
        step_report(results)

    elapsed = time.time() - start_time
    logger.info(f"\n  전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='"기대가 식은 자리" v2.1 듀얼 트리거 백테스트'
    )
    parser.add_argument(
        "--step",
        choices=["collect", "indicators", "backtest", "report"],
        help="특정 단계만 실행",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="샘플 데이터로 테스트 (pykrx 없어도 가능)",
    )

    args = parser.parse_args()

    if args.step:
        if args.step == "collect":
            step_collect(args.sample)
        elif args.step == "indicators":
            step_indicators()
        elif args.step == "backtest":
            step_backtest()
        elif args.step == "report":
            step_report()
    else:
        run_full_pipeline(args.sample)
