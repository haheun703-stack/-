#!/usr/bin/env python3
"""
"Quantum Master" v7.0 -- 7-Layer Pipeline + Master Controller + 라이브 트레이딩

v4.0 신규 (라이브 트레이딩):
  - 한투 API 주문 모듈 (매수/매도/정정/취소)
  - 포지션 트래커 (보유종목 + PnL 관리)
  - 실시간 매매 루프 (시그널→주문→청산 감시)
  - 일일 스케줄러 (장전/장중/장후 자동 실행)
  - 안전장치 (STOP.signal, 손실 한도, 긴급 전량 청산)

v3.1 (분석 강화):
  - L-1 News Gate + Smart Money v2 + Grok API + TRIX/볼린저/MACD

v3.0 핵심:
  - 6-Layer Signal Pipeline + 4단계 부분청산 + HMM + OU

사용법 (기존 v3.1 호환):
    python main.py                    # 전체 파이프라인 실행
    python main.py --step collect     # 데이터 수집만
    python main.py --step indicators  # 지표 계산만
    python main.py --step backtest    # 백테스트만
    python main.py --step report      # 리포트만
    python main.py --sample           # 샘플 데이터로 테스트
    python main.py --telegram         # 텔레그램 메시지 송출
    python main.py --news-grade A:"뉴스내용" --sample   # 뉴스 등급 수동 지정
    python main.py --news-scan        # Grok API 뉴스 자동 스캔
    python main.py --smart-scan       # 전종목 매집 단계 스캔

사용법 (v4.0 라이브 트레이딩):
    python main.py stock_buy          # 매수 실행 (시그널 기반)
    python main.py stock_sell         # 매도 실행 (청산 조건 기반)
    python main.py monitor            # 장중 실시간 모니터링
    python main.py scheduler          # 일일 스케줄러 시작
    python main.py positions          # 현재 보유 포지션 조회
    python main.py emergency-stop     # 긴급 전량 청산
    python main.py balance            # 잔고 조회
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
    """Step 2: 기술적 지표 계산 (v3.1: 35개 지표 + OU/SmartMoney/TRIX/볼린저/MACD)"""
    from src.indicators import IndicatorEngine

    engine = IndicatorEngine()
    count = engine.process_all()
    logger.info(f"[Step 2] 지표 계산 완료: {count}종목 (35개 지표 + OU/SmartMoney/TRIX/볼린저/MACD)")


def step_backtest(
    use_sample: bool = False,
    use_v9: bool = False,
    use_parabola: bool = False,
    bt_start: str | None = None,
    bt_end: str | None = None,
):
    """Step 3~7: v3.0 6-Layer Pipeline 백테스트"""
    from src.backtest_engine import BacktestEngine

    config_path = "config/settings.yaml"

    # --sample 모드: OU/Regime 파라미터 자동 완화
    if use_sample:
        _relax_sample_params(config_path)

    engine = BacktestEngine(
        config_path, use_v9=use_v9, use_parabola=use_parabola,
        bt_start=bt_start, bt_end=bt_end,
    )
    data = engine.load_data()

    if not data:
        logger.error("[Step 3] processed 디렉토리에 데이터 없음. 먼저 collect + indicators 실행 필요")
        return None

    # HMM 레짐 감지 비활성화 (v10.0: 성과 영향 0% 확인 → 제거)
    # L1_regime 레이어는 P_Accum=NaN → 자동 통과

    if use_v9:
        logger.info("[Backtest] v9.0 C+E Kill 필터 활성화")
    if use_parabola:
        logger.info("[Backtest] Mode B 포물선 탐지 활성화")

    results = engine.run(data)

    # v3.0 성과지표는 backtest_engine 내부에서 출력됨 (quant_metrics)
    return results


def _relax_sample_params(config_path: str):
    """합성 데이터용 파라미터 완화 (임시 설정 파일 생성)"""
    import yaml
    import shutil

    # 원본 백업
    backup_path = config_path + ".bak"
    shutil.copy2(config_path, backup_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    qe = config.setdefault("quant_engine", {})

    # OU: 합성 데이터는 정상 평균회귀 아님 → 대폭 완화
    ou = qe.setdefault("ou", {})
    ou["z_entry"] = 2.0          # 사실상 비활성화
    ou["half_life_min"] = 1
    ou["half_life_max"] = 500
    ou["snr_min"] = 0.0

    # Regime: 완화
    regime = qe.setdefault("regime", {})
    regime["p_accum_entry"] = 0.20

    # Momentum: 완화
    mom = qe.setdefault("momentum", {})
    mom["vol_surge_min"] = 0.5
    mom["slope_ma60_min"] = -5.0

    # SmartMoney: 비활성화 (합성 데이터에 수급 없음)
    sm = qe.setdefault("smart_money", {})
    sm["min_smart_z"] = -999.0

    # 최대 보유일 완화
    exit_cfg = qe.setdefault("exit", {})
    exit_cfg["max_hold_days"] = 30

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("[Sample] 합성 데이터용 파라미터 완화 적용")

    # 프로세스 종료 시 원본 복원을 위해 등록
    import atexit
    atexit.register(shutil.move, backup_path, config_path)


def _parse_news_grade(grade_str: str):
    """
    '--news-grade A:뉴스내용' 파싱.

    Returns:
        NewsGateResult or None
    """
    from src.use_cases.news_gate import NewsGateUseCase

    parts = grade_str.split(":", 1)
    grade = parts[0].strip().upper()
    news_text = parts[1].strip() if len(parts) > 1 else ""

    if grade not in ("A", "B", "C"):
        logger.error(f"[News Gate] 잘못된 등급: {grade} (A/B/C 중 선택)")
        return None

    # 종목은 sample 모드에서 첫 번째 종목 사용 (실전에서는 ticker 별도 지정)
    ticker = "SAMPLE"

    gate = NewsGateUseCase()
    result = gate.evaluate(ticker=ticker, news_grade=grade, news_text=news_text)
    logger.info(f"[News Gate] {grade}등급 → {result.action.value} | {result.reason}")
    return result


def step_news_scan(send_telegram: bool = False):
    """v3.1 Grok API 뉴스 자동 스캔"""
    import asyncio
    from src.adapters.grok_news_adapter import GrokNewsAdapter
    from src.use_cases.news_gate import NewsGateUseCase

    logger.info("[News Scan] Grok API 뉴스 스캔 시작")

    adapter = GrokNewsAdapter()
    gate = NewsGateUseCase()

    # 시장 전체 뉴스 조회 (async 메서드)
    try:
        market_data = asyncio.run(adapter.search_market_overview())
        if not market_data:
            logger.warning("[News Scan] 뉴스 데이터 없음 (API 키 확인 필요)")
            return

        # market_data는 dict (시장 요약) — 개별 종목 뉴스는 search_stock_news 사용
        logger.info(f"[News Scan] 시장 동향 수집 완료")

        # 주요 종목 뉴스 스캔 (상위 종목)
        from src.entities.news_models import NewsItem
        hot_tickers = market_data.get("hot_tickers", [])
        for ticker_info in hot_tickers[:20]:
            ticker = ticker_info if isinstance(ticker_info, str) else str(ticker_info)
            try:
                raw = asyncio.run(adapter.search_stock_news(ticker))
                if not raw:
                    continue
                # dict → NewsItem 리스트 변환
                news_list = raw.get("news", [])
                news_items = [
                    NewsItem(
                        title=n.get("title", ""),
                        summary=n.get("summary", ""),
                        category=n.get("category", "theme"),
                        source=n.get("source", ""),
                        sentiment=n.get("sentiment", "neutral"),
                        is_confirmed=n.get("is_confirmed", False),
                        has_specific_amount=n.get("has_specific_amount", False),
                        has_definitive_language=n.get("has_definitive_language", False),
                        cross_verified=n.get("cross_verified", False),
                    )
                    for n in news_list if isinstance(n, dict)
                ]
                if not news_items:
                    continue
                result = gate.evaluate(ticker=ticker, news_items=news_items)
                if result.grade.value in ("A", "B"):
                    logger.info(
                        f"  {result.grade.value}등급: {ticker} — {result.reason}"
                    )
                    if send_telegram:
                        from src.telegram_sender import send_news_alert
                        send_news_alert(
                            ticker=ticker,
                            grade=result.grade.value,
                            action=result.action.value,
                            reason=result.reason,
                            param_overrides=result.param_overrides,
                        )
            except Exception as e:
                logger.debug(f"  {ticker} 뉴스 스캔 실패: {e}")

    except Exception as e:
        logger.error(f"[News Scan] 뉴스 스캔 실패: {e}")


def step_smart_scan(send_telegram: bool = False):
    """v3.1 전종목 매집 단계 스캔"""
    from pathlib import Path
    import pandas as pd
    from src.accumulation_detector import AccumulationDetector

    logger.info("[Smart Scan] 전종목 매집 단계 스캔 시작")
    detector = AccumulationDetector()

    processed_dir = Path("data/processed")
    files = sorted(processed_dir.glob("*.parquet"))

    if not files:
        logger.error("[Smart Scan] data/processed에 파일 없음. indicators 먼저 실행 필요")
        return

    results = []
    for fpath in files:
        ticker = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if len(df) < 60:
                continue
            signal = detector.detect(df)
            if signal.phase in ("phase1", "phase2", "phase3"):
                results.append((ticker, signal))
                logger.info(
                    f"  {signal.phase_name}: {ticker} "
                    f"(신뢰도={signal.confidence:.0f}%, 점수={signal.score_modifier:+d})"
                )
        except Exception as e:
            logger.debug(f"  {ticker} 스캔 실패: {e}")

    logger.info(f"[Smart Scan] 완료: {len(results)}종목 매집 감지")

    if send_telegram and results:
        from src.telegram_sender import send_accumulation_alert
        # phase 매핑: phase3 > phase2 > phase1
        phase_order = {"phase3": 3, "phase2": 2, "phase1": 1}
        sorted_results = sorted(
            results, key=lambda x: phase_order.get(x[1].phase, 0), reverse=True
        )
        for ticker, sig in sorted_results[:10]:  # 상위 10종목만 알림
            phase_num = phase_order.get(sig.phase, 0)
            send_accumulation_alert(
                ticker=ticker,
                phase=phase_num,
                confidence=sig.confidence,
                bonus_score=sig.score_modifier,
                inst_streak=sig.inst_streak,
                foreign_streak=sig.foreign_streak,
                description=", ".join(sig.reasons) if sig.reasons else "",
            )


def _fit_regime(data_dict: dict):
    """전종목에 HMM 레짐 확률 추가"""
    from src.regime_detector import RegimeDetector

    detector = RegimeDetector()
    for ticker, df in data_dict.items():
        try:
            regime_proba = detector.fit_predict(df)
            for col in ["P_Advance", "P_Distrib", "P_Accum"]:
                data_dict[ticker][col] = regime_proba[col]
        except Exception as e:
            logger.debug(f"{ticker} 레짐 감지 실패: {e}")
            for col in ["P_Advance", "P_Distrib", "P_Accum"]:
                data_dict[ticker][col] = 1 / 3


def step_telegram(results: dict = None):
    """Step: 텔레그램 메시지 송출"""
    from src.telegram_sender import send_backtest_report

    if results:
        success = send_backtest_report(results)
        if success:
            logger.info("[Telegram] 텔레그램 메시지 송출 완료")
        else:
            logger.error("[Telegram] 텔레그램 메시지 송출 실패")
    else:
        # 저장된 CSV에서 결과 로드 후 전송
        import pandas as pd
        results_dir = Path("results")
        trades_df = pd.read_csv(results_dir / "trades_log.csv") if (results_dir / "trades_log.csv").exists() else pd.DataFrame()
        equity_df = pd.read_csv(results_dir / "daily_equity.csv") if (results_dir / "daily_equity.csv").exists() else pd.DataFrame()
        signals_df = pd.read_csv(results_dir / "signals_log.csv") if (results_dir / "signals_log.csv").exists() else pd.DataFrame()

        if equity_df.empty:
            logger.error("[Telegram] 백테스트 결과 없음. 먼저 backtest 실행 필요")
            return

        from src.backtest_engine import BacktestEngine
        engine = BacktestEngine()
        stats = engine._calc_stats(trades_df, equity_df)

        rebuilt = {
            "stats": stats,
            "trades_df": trades_df,
            "equity_df": equity_df,
            "signals_df": signals_df,
            "diagnostic": {},
        }
        success = send_backtest_report(rebuilt)
        if success:
            logger.info("[Telegram] 텔레그램 메시지 송출 완료")


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


def run_full_pipeline(
    use_sample: bool = False,
    send_telegram: bool = False,
    news_grade_str: str | None = None,
):
    """v3.1 전체 파이프라인 실행"""
    start_time = time.time()

    print("""
    ======================================================
      "Quantum Master" v7.0 -- 7-Layer Pipeline + Master Controller
    ======================================================
      Step 1: 데이터 수집 (KRX / 샘플)
      Step 2: 기술적 지표 계산 (35개 + OU/SmartMoney/TRIX/MACD)
      Step 3: HMM 레짐 감지 (Advance/Distrib/Accum)
      Step 4: 6-Layer Pipeline 백테스트
             L-1 News Gate → L0 Pre-Gate → L0 Grade
             → L1 Regime → L2 OU → L3 Momentum
             → L4 SmartMoney v2 → L5 Risk → L6 Trigger
      Step 5: 4단계 부분청산 (2R/4R/8R/10R)
      Step 6: Quant Metrics + 6-Layer 진단 리포트
    ======================================================
    """)

    # v3.1 News Gate 설정
    news_gate_result = None
    if news_grade_str:
        news_gate_result = _parse_news_grade(news_grade_str)

    # Step 1: 데이터 수집
    logger.info("=" * 50)
    logger.info("[Step 1/4] 데이터 수집")
    logger.info("=" * 50)
    step_collect(use_sample)

    # Step 2: 지표 계산
    logger.info("=" * 50)
    logger.info("[Step 2/4] 기술적 지표 계산 (35개)")
    logger.info("=" * 50)
    step_indicators()

    # Step 3~5: 백테스트
    logger.info("=" * 50)
    logger.info("[Step 3/4] v3.1 백테스트 (6-Layer Pipeline + News Gate + 4단계 부분청산)")
    logger.info("=" * 50)
    results = step_backtest(use_sample)

    # Step 6: 리포트
    if results:
        logger.info("=" * 50)
        logger.info("[Step 4/4] HTML 리포트 생성")
        logger.info("=" * 50)
        step_report(results)

    # v3.1 뉴스 알림 (A급 뉴스가 지정된 경우)
    if news_gate_result and send_telegram:
        from src.telegram_sender import send_news_alert
        send_news_alert(
            ticker=news_gate_result.ticker,
            grade=news_gate_result.grade.value,
            action=news_gate_result.action.value,
            reason=news_gate_result.reason,
            param_overrides=news_gate_result.param_overrides,
        )

    # Step 7: 텔레그램 송출 (--telegram 옵션 시)
    if send_telegram and results:
        logger.info("=" * 50)
        logger.info("[Telegram] 텔레그램 메시지 송출")
        logger.info("=" * 50)
        step_telegram(results)

    elapsed = time.time() - start_time
    logger.info(f"\n  전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")


def cmd_stock_buy(args):
    """v4.0 매수 실행 (시그널 기반)"""
    from src.use_cases.live_trading import create_live_engine
    import pandas as pd

    logger.info("[stock_buy] 매수 실행 시작")

    engine = create_live_engine()

    # 시그널 로드
    signals_path = Path("results/signals_log.csv")
    if not signals_path.exists():
        logger.error("[stock_buy] signals_log.csv 없음 — 먼저 backtest 실행 필요")
        return

    df = pd.read_csv(signals_path)
    if "signal" in df.columns:
        df = df[df["signal"] == True]
    if "date" in df.columns:
        latest_date = df["date"].max()
        df = df[df["date"] == latest_date]

    signals = df.to_dict("records")
    if not signals:
        logger.info("[stock_buy] 매수 시그널 없음")
        return

    results = engine.execute_buy_signals(signals)
    for r in results:
        status = "성공" if r.get("success") else f"실패 ({r.get('reason', '')})"
        logger.info("  %s: %s", r.get("ticker", "?"), status)


def cmd_stock_sell(args):
    """v4.0 매도 실행 (청산 조건 기반)"""
    from src.use_cases.live_trading import create_live_engine

    logger.info("[stock_sell] 매도 실행 시작")
    engine = create_live_engine()
    results = engine.execute_sell_signals()

    if not results:
        logger.info("[stock_sell] 청산 대상 없음")
    else:
        for r in results:
            status = "성공" if r.get("success") else "실패"
            logger.info(
                "  %s: %d주 매도 (%s) — %s",
                r.get("ticker", "?"), r.get("shares", 0), r.get("reason", ""), status,
            )


def cmd_monitor(args):
    """v4.0 장중 실시간 모니터링"""
    from src.use_cases.live_trading import create_live_engine

    duration = getattr(args, "duration", 0) or 0
    logger.info("[monitor] 장중 모니터링 시작 (duration=%s)", duration or "15:20까지")

    engine = create_live_engine()
    engine.monitor_loop(duration_sec=duration)


def cmd_scheduler(args):
    """v4.0 일일 스케줄러"""
    from scripts.daily_scheduler import DailyScheduler, setup_logging

    setup_logging()
    scheduler = DailyScheduler()

    if getattr(args, "dry_run", False):
        scheduler.print_schedule()
    else:
        scheduler.run()


def cmd_positions(args):
    """v4.0 보유 포지션 조회"""
    import yaml
    from src.use_cases.position_tracker import PositionTracker

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracker = PositionTracker(config)
    summary = tracker.get_summary()

    print()
    print("=" * 55)
    print(f"  보유 포지션: {summary['count']}종목")
    print(f"  총 투입: {summary['total_investment']:>15,}원")
    print(f"  총 평가: {summary['total_eval']:>15,}원")
    print(f"  수익률:  {summary['total_pnl_pct']:>+14.1f}%")
    print("=" * 55)

    for p in summary.get("positions", []):
        print(
            f"  {p['ticker']} {p['name']:<10} "
            f"{p['shares']:>5}주 "
            f"@ {p['entry_price']:>8,.0f} → {p['current_price']:>8,.0f} "
            f"({p['pnl_pct']:>+6.1f}%) "
            f"[{p['grade']}/{p['trigger']}]"
        )

    if not summary.get("positions"):
        print("  (보유 종목 없음)")
    print()

    if getattr(args, "telegram", False):
        from src.telegram_sender import send_message
        lines = [
            f"보유 {summary['count']}종목 | "
            f"투입 {summary['total_investment']:,}원 | "
            f"평가 {summary['total_eval']:,}원 | "
            f"수익률 {summary['total_pnl_pct']:+.1f}%",
        ]
        for p in summary.get("positions", []):
            lines.append(
                f"  {p['ticker']} {p['name']}: "
                f"{p['shares']}주 ({p['pnl_pct']:+.1f}%)"
            )
        send_message("\n".join(lines))


def cmd_emergency_stop(args):
    """v4.0 긴급 전량 청산"""
    import yaml
    from src.adapters.kis_order_adapter import KisOrderAdapter
    from src.use_cases.position_tracker import PositionTracker
    from src.use_cases.safety_guard import SafetyGuard

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracker = PositionTracker(config)
    guard = SafetyGuard(config)
    adapter = KisOrderAdapter()

    if not tracker.positions:
        logger.info("[emergency-stop] 보유 포지션 없음")
        return

    logger.critical(
        "[emergency-stop] 긴급 전량 청산: %d종목", len(tracker.positions)
    )
    results = guard.emergency_liquidate(tracker, adapter)
    for r in results:
        if "error" in r:
            logger.error("  %s: 실패 — %s", r["ticker"], r["error"])
        else:
            logger.info("  %s: %d주 청산 접수", r["ticker"], r["shares"])


def cmd_balance(args):
    """v4.0 잔고 조회"""
    from src.adapters.kis_order_adapter import KisOrderAdapter

    adapter = KisOrderAdapter()
    balance = adapter.fetch_balance()

    print()
    print("=" * 55)
    print("  한국투자증권 잔고 조회")
    print("=" * 55)
    print(f"  예수금:    {balance.get('available_cash', 0):>15,}원")
    print(f"  총 평가:   {balance.get('total_eval', 0):>15,}원")
    print(f"  평가 손익: {balance.get('total_pnl', 0):>+15,}원")
    print("-" * 55)

    for h in balance.get("holdings", []):
        print(
            f"  {h['ticker']} {h['name']:<10} "
            f"{h['quantity']:>5}주 "
            f"@ {h['avg_price']:>8,.0f} → {h['current_price']:>8,} "
            f"({h['pnl_pct']:>+6.1f}%)"
        )

    if not balance.get("holdings"):
        print("  (보유 종목 없음)")
    print("=" * 55)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='"Quantum Master" v7.0 7-Layer Pipeline + Master Controller + 라이브 트레이딩'
    )

    # v4.0 subcommand
    subparsers = parser.add_subparsers(dest="command", help="v4.0 라이브 트레이딩 커맨드")

    # stock_buy
    sp_buy = subparsers.add_parser("stock_buy", help="매수 실행 (시그널 기반)")

    # stock_sell
    sp_sell = subparsers.add_parser("stock_sell", help="매도 실행 (청산 조건 기반)")

    # monitor
    sp_monitor = subparsers.add_parser("monitor", help="장중 실시간 모니터링")
    sp_monitor.add_argument("--duration", type=int, default=0, help="모니터링 시간 (초, 0=15:20까지)")

    # scheduler
    sp_sched = subparsers.add_parser("scheduler", help="일일 스케줄러 시작")
    sp_sched.add_argument("--dry-run", action="store_true", help="스케줄 확인만")

    # positions
    sp_pos = subparsers.add_parser("positions", help="현재 보유 포지션 조회")
    sp_pos.add_argument("--telegram", action="store_true", help="텔레그램 발송")

    # emergency-stop
    sp_emg = subparsers.add_parser("emergency-stop", help="긴급 전량 청산")

    # balance
    sp_bal = subparsers.add_parser("balance", help="한투 잔고 조회")

    # v3.1 호환 옵션 (--step, --sample 등)
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
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="텔레그램으로 결과 메시지 송출",
    )
    parser.add_argument(
        "--news-grade",
        type=str,
        default=None,
        help='뉴스 등급 수동 지정 (예: A:"SK이터닉스 매각 확정")',
    )
    parser.add_argument(
        "--news-scan",
        action="store_true",
        help="Grok API 뉴스 자동 스캔 (XAI_API_KEY 필요)",
    )
    parser.add_argument(
        "--smart-scan",
        action="store_true",
        help="전종목 매집 단계 스캔 (Smart Money v2)",
    )
    parser.add_argument(
        "--v9",
        action="store_true",
        help="v9.0 C+E Kill 필터 적용 (백테스트/스캔)",
    )
    parser.add_argument(
        "--parabola",
        action="store_true",
        help="Mode B 포물선 탐지 활성화 (백테스트)",
    )
    parser.add_argument(
        "--bt-start",
        type=str,
        default=None,
        help="백테스트 시작일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bt-end",
        type=str,
        default=None,
        help="백테스트 종료일 (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    # v4.0 subcommand 분기
    if args.command == "stock_buy":
        cmd_stock_buy(args)
    elif args.command == "stock_sell":
        cmd_stock_sell(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "scheduler":
        cmd_scheduler(args)
    elif args.command == "positions":
        cmd_positions(args)
    elif args.command == "emergency-stop":
        cmd_emergency_stop(args)
    elif args.command == "balance":
        cmd_balance(args)
    # v3.1 호환 모드
    elif args.news_scan:
        step_news_scan(send_telegram=args.telegram)
    elif args.smart_scan:
        step_smart_scan(send_telegram=args.telegram)
    elif args.step:
        if args.step == "collect":
            step_collect(args.sample)
        elif args.step == "indicators":
            step_indicators()
        elif args.step == "backtest":
            step_backtest(
                use_v9=args.v9, use_parabola=args.parabola,
                bt_start=args.bt_start, bt_end=args.bt_end,
            )
        elif args.step == "report":
            step_report()
        if args.telegram:
            step_telegram()
    else:
        run_full_pipeline(
            args.sample,
            send_telegram=args.telegram,
            news_grade_str=args.news_grade,
        )
