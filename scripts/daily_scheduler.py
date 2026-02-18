#!/usr/bin/env python3
"""
v4.1 일일 스케줄러 — 사용자 스케줄표 기반 자동 실행

일일 스케줄:
  00:00  Phase 1 — 일일 리셋 (STOP.signal 삭제 + 로그 로테이션)
  07:00  Phase 2 — 매크로 수집 + US Overnight Signal
  07:10  Phase 3 — 뉴스/리포트 (Grok API)
  08:20  Phase 4 — 매매 준비 (토큰 갱신 → 공휴일 체크 → 시그널 스캔)
  08:25  Phase 4.5 — 장전 포트폴리오 5D/6D 분석 리포트
  09:02  Phase 5 — 개별종목 매수 실행
  09:10  Phase 6 — 장중 모니터링 시작 (1분 간격, 15:20까지)
  15:25  Phase 7 — 매도 실행 (장마감 전)
  15:35  Phase 8 — 장마감 파이프라인 (9단계)
  16:30  Phase 9 — 장마감 업무일지 생성

안전장치:
  STOP.signal — 매수/매도/모니터링 중단
  reboot.trigger — 스케줄러 graceful 재시작
  공휴일 — Phase 5~7 자동 스킵

사용법:
  python scripts/daily_scheduler.py               # 스케줄러 시작
  python scripts/daily_scheduler.py --dry-run      # 스케줄 확인만 (실행 안함)
  python scripts/daily_scheduler.py --run-now 9    # 특정 Phase 즉시 실행 (1~9)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import yaml

logger = logging.getLogger("scheduler")


class DailyScheduler:
    """사용자 스케줄표 기반 일일 자동 실행"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        live_cfg = self.config.get("live_trading", {})
        self.schedule = live_cfg.get("schedule", {})
        self.enabled = live_cfg.get("enabled", False)
        self.mode = live_cfg.get("mode", "paper")

        # 오늘 휴일 여부 (Phase 4에서 갱신)
        self._is_holiday = False

        # 시그널 결과 캐시 (Phase 4 → Phase 5 전달)
        self._buy_signals: list[dict] = []

        logger.info(
            "DailyScheduler 초기화 (enabled=%s, mode=%s)",
            self.enabled, self.mode,
        )

    # ──────────────────────────────────────────
    # Phase 1: 일일 리셋 (00:00)
    # ──────────────────────────────────────────

    def phase_daily_reset(self) -> None:
        """STOP.signal 삭제 + 로그 로테이션 + 일일 초기화"""
        logger.info("=" * 50)
        logger.info("[Phase 1] 일일 리셋 시작 — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
        logger.info("=" * 50)

        from src.use_cases.safety_guard import SafetyGuard
        guard = SafetyGuard(self.config)
        guard.clear_stop_signal()

        self._is_holiday = False
        self._buy_signals = []

        logger.info("[Phase 1] 일일 리셋 완료")
        self._notify("Phase 1 완료: 일일 리셋")

    # ──────────────────────────────────────────
    # Phase 2: 매크로 수집 (07:00)
    # ──────────────────────────────────────────

    def phase_macro_collect(self) -> None:
        """KOSPI/KOSDAQ/환율/금리 등 매크로 데이터 수집 + US Overnight Signal"""
        logger.info("[Phase 2] 매크로 수집 시작")
        try:
            from scripts.update_daily_data import update_all
            update_all()
            logger.info("[Phase 2] 매크로 수집 완료")
        except Exception as e:
            logger.error("[Phase 2] 매크로 수집 실패: %s", e)

        # US Overnight Signal (yfinance 최신 → 신호 생성)
        try:
            from scripts.us_overnight_signal import update_latest, generate_signal
            df = update_latest()
            signal = generate_signal(df)
            logger.info("[Phase 2] US Overnight: %s (%.2f)", signal.get("composite"), signal.get("score", 0))
        except Exception as e:
            logger.error("[Phase 2] US Overnight 실패: %s", e)

        self._notify("Phase 2 완료: 매크로 + US Overnight")

    # ──────────────────────────────────────────
    # Phase 3: 뉴스/리포트 (07:10)
    # ──────────────────────────────────────────

    def phase_news_briefing(self) -> None:
        """Grok API 뉴스 스캔 + 등급 판정"""
        logger.info("[Phase 3] 뉴스/리포트 수집 시작")
        try:
            # main.py의 step_news_scan 직접 호출
            sys.path.insert(0, str(PROJECT_ROOT))
            from main import step_news_scan
            step_news_scan(send_telegram=True)
            logger.info("[Phase 3] 뉴스 스캔 완료")
        except Exception as e:
            logger.error("[Phase 3] 뉴스 스캔 실패: %s", e)
        self._notify("Phase 3 완료: 뉴스/리포트")

    # ──────────────────────────────────────────
    # Phase 4: 매매 준비 (08:20)
    # ──────────────────────────────────────────

    def phase_trade_prep(self) -> None:
        """토큰 갱신 → 공휴일 체크 → 시그널 스캔 → 매수 후보 확정"""
        logger.info("[Phase 4] 매매 준비 시작")

        # 1. 공휴일 체크
        from src.use_cases.safety_guard import SafetyGuard
        guard = SafetyGuard(self.config)
        self._is_holiday = guard.check_holiday()

        if self._is_holiday:
            logger.info("[Phase 4] 오늘은 공휴일/주말 — Phase 5~7 스킵")
            self._notify("Phase 4: 공휴일 감지 — 매매 스킵")
            return

        # 2. STOP.signal 체크
        if guard.check_stop_signal():
            logger.warning("[Phase 4] STOP.signal 활성 — 매매 스킵")
            self._notify("Phase 4: STOP.signal 활성 — 매매 스킵")
            return

        # 3. 한투 API 토큰 갱신 (어댑터 초기화 시 자동)
        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            balance = adapter.get_available_cash()
            logger.info("[Phase 4] 한투 API 토큰 갱신 완료 (예수금: %s원)", f"{balance:,.0f}")
        except Exception as e:
            logger.error("[Phase 4] 한투 API 연결 실패: %s", e)
            self._notify(f"Phase 4 경고: 한투 API 연결 실패 — {e}")
            return

        # 4. 시그널 스캔 (step_backtest 결과 로드)
        try:
            self._load_signals()
            logger.info("[Phase 4] 매수 후보 %d종목 확정", len(self._buy_signals))
        except Exception as e:
            logger.error("[Phase 4] 시그널 스캔 실패: %s", e)

        self._notify(f"Phase 4 완료: 매매 준비 (후보 {len(self._buy_signals)}종목)")

    def _load_signals(self) -> None:
        """저장된 시그널 결과 로드"""
        signals_path = Path("results/signals_log.csv")
        if not signals_path.exists():
            logger.warning("[Phase 4] signals_log.csv 없음 — 시그널 0건")
            self._buy_signals = []
            return

        import pandas as pd
        df = pd.read_csv(signals_path)
        # signal=True인 최신 데이터만
        if "signal" in df.columns:
            df = df[df["signal"] == True]
        if "date" in df.columns:
            latest_date = df["date"].max()
            df = df[df["date"] == latest_date]

        self._buy_signals = df.to_dict("records")

    # ──────────────────────────────────────────
    # Phase 4.5: 장전 포트폴리오 분석 리포트 (08:25)
    # ──────────────────────────────────────────

    def phase_morning_report(self) -> None:
        """5D/6D 포트폴리오 분석 HTML 리포트 생성 + 텔레그램 장시작 분석 보고서"""
        logger.info("[Phase 4.5] 장전 리포트 시작")

        # 기존 HTML 리포트 (유지)
        try:
            from src.use_cases.portfolio_reporter import PortfolioReporter
            reporter = PortfolioReporter(self.config)
            save_path = reporter.generate()

            if save_path:
                logger.info("[Phase 4.5] HTML 리포트 생성 완료: %s", save_path)
            else:
                logger.info("[Phase 4.5] 보유 포지션 없음 — HTML 리포트 생략")
        except Exception as e:
            logger.error("[Phase 4.5] HTML 리포트 생성 실패: %s", e)

        # 텔레그램 장시작 분석 보고서 (신규)
        try:
            from src.use_cases.market_analysis_reporter import MarketAnalysisReporter
            from src.telegram_sender import send_market_analysis

            ma_reporter = MarketAnalysisReporter(self.config)
            data = ma_reporter.generate(report_type="morning")
            send_market_analysis(data)
            logger.info("[Phase 4.5] 텔레그램 장시작 보고서 전송 완료")
        except Exception as e:
            logger.error("[Phase 4.5] 텔레그램 보고서 전송 실패: %s", e)

        self._notify("Phase 4.5 완료: 장전 리포트")

    # ──────────────────────────────────────────
    # Phase 5: 매수 실행 (09:02)
    # ──────────────────────────────────────────

    def phase_buy_execution(self) -> None:
        """v4 결과 기반 매수 실행"""
        if self._is_holiday:
            logger.info("[Phase 5] 공휴일 — 스킵")
            return

        logger.info("[Phase 5] 매수 실행 시작 (후보 %d종목)", len(self._buy_signals))

        if not self._buy_signals:
            logger.info("[Phase 5] 매수 후보 없음")
            self._notify("Phase 5: 매수 후보 없음")
            return

        if not self.enabled:
            logger.info("[Phase 5] live_trading.enabled=false — 모의 모드 (주문 안함)")
            self._notify("Phase 5: 모의 모드 — 실주문 안함")
            return

        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()

            # 시그널 최대 5분 대기 (이미 로드됨)
            results = engine.execute_buy_signals(self._buy_signals)

            success_count = sum(1 for r in results if r.get("success"))
            logger.info("[Phase 5] 매수 완료: %d/%d 성공", success_count, len(results))
            self._notify(f"Phase 5 완료: {success_count}종목 매수 성공")
        except Exception as e:
            logger.error("[Phase 5] 매수 실행 오류: %s", e)
            self._notify(f"Phase 5 오류: {e}")

    # ──────────────────────────────────────────
    # Phase 6: 장중 모니터링 (09:10 ~ 15:20)
    # ──────────────────────────────────────────

    def phase_intraday_monitor(self) -> None:
        """장중 실시간 모니터링 (1분 간격)"""
        if self._is_holiday:
            logger.info("[Phase 6] 공휴일 — 스킵")
            return

        if not self.enabled:
            logger.info("[Phase 6] live_trading.enabled=false — 모니터링 스킵")
            return

        logger.info("[Phase 6] 장중 모니터링 시작 (09:10 ~ 15:20)")
        self._notify("Phase 6 시작: 장중 모니터링")

        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()
            engine.monitor_loop()  # 15:20 자동 종료
            logger.info("[Phase 6] 장중 모니터링 종료")
        except Exception as e:
            logger.error("[Phase 6] 모니터링 오류: %s", e)
            self._notify(f"Phase 6 오류: {e}")

    # ──────────────────────────────────────────
    # Phase 7: 매도 실행 (15:25)
    # ──────────────────────────────────────────

    def phase_sell_execution(self) -> None:
        """장마감 전 청산 대상 매도"""
        if self._is_holiday:
            logger.info("[Phase 7] 공휴일 — 스킵")
            return

        if not self.enabled:
            logger.info("[Phase 7] live_trading.enabled=false — 스킵")
            return

        logger.info("[Phase 7] 매도 실행 시작")
        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()
            results = engine.execute_sell_signals()

            sell_count = sum(1 for r in results if r.get("success"))
            logger.info("[Phase 7] 매도 완료: %d건 실행", sell_count)
            self._notify(f"Phase 7 완료: {sell_count}건 매도")
        except Exception as e:
            logger.error("[Phase 7] 매도 오류: %s", e)
            self._notify(f"Phase 7 오류: {e}")

    # ──────────────────────────────────────────
    # Phase 8: 장마감 파이프라인 (15:35)
    # ──────────────────────────────────────────

    def phase_close_pipeline(self) -> None:
        """
        장마감 9단계 파이프라인:
        1. 일일 데이터 수집
        2. 종목 필터 (지표 계산)
        3. 매크로 데이터 수집 + 정제
        4. 매크로 NaN 검증
        5. 전일 추천 결과 업데이트
        6. 다음날 매수 후보 스캔
        7. 오늘 추천 기록 저장
        8. ML 재학습 (향후 확장)
        9. 일일 성과 리포트
        """
        logger.info("=" * 50)
        logger.info("[Phase 8] 장마감 파이프라인 시작")
        logger.info("=" * 50)

        # Step 1: 일일 데이터 수집
        self._run_step("8-1", "일일 데이터 수집", self._close_step_1_collect)

        # Step 2: 종목 필터 (지표 계산)
        self._run_step("8-2", "지표 계산", self._close_step_2_indicators)

        # Step 3: 매크로 수집
        self._run_step("8-3", "매크로 수집", self._close_step_3_macro)

        # Step 4: NaN 검증
        self._run_step("8-4", "데이터 검증", self._close_step_4_verify)

        # Step 5: 전일 추천 결과 업데이트 (향후)
        logger.info("[Phase 8-5] 전일 추천 결과 업데이트 — 향후 구현 예정")

        # Step 6: 다음날 매수 후보 스캔
        self._run_step("8-6", "매수 후보 스캔", self._close_step_6_scan)

        # Step 7: 오늘 추천 기록 저장
        self._run_step("8-7", "추천 기록 저장", self._close_step_7_save)

        # Step 8: US-KR 패턴DB 학습 루프 (일일 누적)
        self._run_step("8-8", "US-KR 패턴DB 업데이트", self._close_step_8_uskr)

        # Step 9: 일일 성과 리포트
        self._run_step("8-9", "성과 리포트", self._close_step_9_report)

        logger.info("[Phase 8] 장마감 파이프라인 완료")
        self._notify("Phase 8 완료: 장마감 파이프라인 (9단계)")

    def _close_step_1_collect(self) -> None:
        """8-1: CSV 업데이트 + parquet 증분 업데이트"""
        from scripts.update_daily_data import update_all
        update_all()

        # P2 fix: raw parquet 증분 업데이트 (pykrx)
        # 이 단계가 없으면 raw가 구버전 → 8-2 지표 계산도 구버전
        try:
            from scripts.extend_parquet_data import main as extend_main
            extend_main()
        except Exception as e:
            logger.error("[Phase 8-1] parquet 증분 업데이트 실패: %s", e)

    def _close_step_2_indicators(self) -> None:
        from main import step_indicators
        step_indicators()

    def _close_step_3_macro(self) -> None:
        # 매크로 수집 재실행 (장마감 데이터 반영)
        self.phase_macro_collect()

    def _close_step_4_verify(self) -> None:
        from scripts.update_daily_data import verify_all_data
        try:
            verify_all_data()
        except (ImportError, AttributeError):
            logger.info("[Phase 8-4] verify_all_data 없음 — 스킵")

    def _close_step_6_scan(self) -> None:
        from main import step_backtest
        step_backtest(use_sample=False)

    def _close_step_7_save(self) -> None:
        """오늘 시그널 결과를 날짜별로 저장"""
        import shutil
        signals_path = Path("results/signals_log.csv")
        if signals_path.exists():
            today = datetime.now().strftime("%Y%m%d")
            archive_dir = Path("results/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(signals_path, archive_dir / f"signals_{today}.csv")

    def _close_step_8_uskr(self) -> None:
        """8-8: US-KR 패턴DB 일일 누적 (학습 루프)"""
        from scripts.update_us_kr_daily import main as update_uskr_main
        update_uskr_main()

    def _close_step_9_report(self) -> None:
        """장마감 분석 보고서 생성 + 텔레그램 발송"""
        try:
            from src.use_cases.market_analysis_reporter import MarketAnalysisReporter
            from src.telegram_sender import send_market_analysis

            reporter = MarketAnalysisReporter(self.config)
            data = reporter.generate(report_type="closing")
            send_market_analysis(data)
            logger.info("[Phase 8-9] 텔레그램 장마감 보고서 전송 완료")
        except Exception as e:
            logger.error("[Phase 8-9] 장마감 보고서 실패: %s", e)

    # ──────────────────────────────────────────
    # Phase 9: 장마감 업무일지 (16:30)
    # ──────────────────────────────────────────

    def phase_eod_journal(self) -> None:
        """일일 업무일지 HTML 생성 + 텔레그램 발송"""
        logger.info("[Phase 9] 업무일지 생성 시작")
        try:
            from src.use_cases.daily_journal import DailyJournalWriter
            writer = DailyJournalWriter(self.config)
            save_path = writer.generate()

            if save_path:
                logger.info("[Phase 9] 업무일지 저장: %s", save_path)
                self._notify(f"Phase 9 완료: 일일 업무일지 생성\n{save_path}")
            else:
                logger.info("[Phase 9] 업무일지 생성 실패")
        except Exception as e:
            logger.error("[Phase 9] 업무일지 생성 실패: %s", e)
            self._notify(f"Phase 9 오류: {e}")

    # ──────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────

    def _run_step(self, step_id: str, name: str, func) -> None:
        """개별 스텝 실행 (에러 격리)"""
        logger.info("[Phase %s] %s 시작", step_id, name)
        try:
            func()
            logger.info("[Phase %s] %s 완료", step_id, name)
        except Exception as e:
            logger.error("[Phase %s] %s 실패: %s", step_id, name, e)

    def _notify(self, message: str) -> None:
        """텔레그램 알림 (실패해도 무시)"""
        try:
            from src.telegram_sender import send_message
            send_message(f"[스케줄러] {message}")
        except Exception:
            pass

    # ──────────────────────────────────────────
    # 메인 루프
    # ──────────────────────────────────────────

    def run(self) -> None:
        """스케줄러 메인 루프 (schedule 라이브러리 사용)"""
        import schedule

        logger.info("=" * 60)
        logger.info("  v4.1 일일 스케줄러 시작")
        logger.info("  모드: %s | 실주문: %s", self.mode, "ON" if self.enabled else "OFF")
        logger.info("=" * 60)

        # 스케줄 등록
        schedule.every().day.at(self.schedule.get("daily_reset", "00:00")).do(
            self._safe_run, self.phase_daily_reset
        )
        schedule.every().day.at(self.schedule.get("macro_collect", "07:00")).do(
            self._safe_run, self.phase_macro_collect
        )
        schedule.every().day.at(self.schedule.get("news_briefing", "07:10")).do(
            self._safe_run, self.phase_news_briefing
        )
        schedule.every().day.at(self.schedule.get("trade_prep", "08:20")).do(
            self._safe_run, self.phase_trade_prep
        )
        schedule.every().day.at(self.schedule.get("morning_report", "08:25")).do(
            self._safe_run, self.phase_morning_report
        )
        schedule.every().day.at(self.schedule.get("buy_execution", "09:02")).do(
            self._safe_run, self.phase_buy_execution
        )
        schedule.every().day.at(self.schedule.get("monitor_start", "09:10")).do(
            self._safe_run, self.phase_intraday_monitor
        )
        schedule.every().day.at(self.schedule.get("sell_execution", "15:25")).do(
            self._safe_run, self.phase_sell_execution
        )
        schedule.every().day.at(self.schedule.get("close_pipeline", "15:35")).do(
            self._safe_run, self.phase_close_pipeline
        )
        schedule.every().day.at(self.schedule.get("eod_journal", "16:30")).do(
            self._safe_run, self.phase_eod_journal
        )

        logger.info("등록된 스케줄:")
        for job in schedule.get_jobs():
            logger.info("  %s", job)

        self._notify("스케줄러 시작됨")

        # 무한 루프
        while True:
            try:
                schedule.run_pending()

                # reboot.trigger 체크
                from src.use_cases.safety_guard import SafetyGuard
                guard = SafetyGuard(self.config)
                if guard.check_reboot_trigger():
                    logger.info("[스케줄러] reboot.trigger 감지 — 10초 후 재시작")
                    self._notify("스케줄러 재시작 중...")
                    time.sleep(10)
                    # 재초기화
                    self.__init__()
                    continue

            except KeyboardInterrupt:
                logger.info("[스케줄러] Ctrl+C — 종료")
                self._notify("스케줄러 종료됨")
                break
            except Exception as e:
                logger.error("[스케줄러] 오류: %s", e)
                self._notify(f"스케줄러 오류: {e}")

            time.sleep(1)

    def _safe_run(self, func) -> None:
        """예외 격리 실행"""
        try:
            func()
        except Exception as e:
            logger.error("[스케줄러] %s 실행 오류: %s", func.__name__, e)
            self._notify(f"스케줄러 오류: {func.__name__} — {e}")

    def print_schedule(self) -> None:
        """스케줄 표 출력 (--dry-run)"""
        print()
        print("=" * 60)
        print("  v4.1 일일 스케줄 (dry-run)")
        print("=" * 60)
        print(f"  모드: {self.mode} | 실주문: {'ON' if self.enabled else 'OFF'}")
        print()
        entries = [
            (self.schedule.get("daily_reset", "00:00"), "Phase 1", "일일 리셋 (STOP.signal 삭제 + 로그 로테이션)"),
            (self.schedule.get("macro_collect", "07:00"), "Phase 2", "매크로 수집 + US Overnight Signal"),
            (self.schedule.get("news_briefing", "07:10"), "Phase 3", "뉴스/리포트 (Grok API 뉴스 스캔)"),
            (self.schedule.get("trade_prep", "08:20"), "Phase 4", "매매 준비 (토큰→공휴일→시그널 스캔)"),
            (self.schedule.get("morning_report", "08:25"), "Phase 4.5", "장전 5D/6D 포트폴리오 분석 리포트"),
            (self.schedule.get("buy_execution", "09:02"), "Phase 5", "개별종목 매수 실행"),
            (self.schedule.get("monitor_start", "09:10"), "Phase 6", "장중 모니터링 (1분간격 → 15:20)"),
            (self.schedule.get("sell_execution", "15:25"), "Phase 7", "매도 실행 (장마감 전)"),
            (self.schedule.get("close_pipeline", "15:35"), "Phase 8", "장마감 파이프라인 (9단계)"),
            (self.schedule.get("eod_journal", "16:30"), "Phase 9", "장마감 일일 업무일지"),
        ]
        for t, name, desc in entries:
            print(f"  {t:>5}  {name:<10}  {desc}")
        print()
        print("  장마감 파이프라인 상세:")
        print("    8-1. 일일 데이터 수집 (update_daily_data)")
        print("    8-2. 지표 계산 (35개 + OU/SmartMoney/TRIX)")
        print("    8-3. 매크로 수집 + 정제")
        print("    8-4. 데이터 NaN 검증")
        print("    8-5. 전일 추천 결과 D+1~D+5 업데이트")
        print("    8-6. 다음날 매수 후보 스캔 (6-Layer Pipeline)")
        print("    8-7. 오늘 추천 기록 저장")
        print("    8-8. US-KR 패턴DB 일일 누적 (학습 루프)")
        print("    8-9. 일일 성과 리포트 (텔레그램)")
        print()
        print("  안전장치:")
        print(f"    STOP.signal:     {self.config.get('live_trading', {}).get('safety', {}).get('stop_signal_file', 'STOP.signal')}")
        print(f"    reboot.trigger:  {self.config.get('live_trading', {}).get('safety', {}).get('reboot_trigger_file', 'reboot.trigger')}")
        print(f"    일일 손실 한도:  {self.config.get('live_trading', {}).get('safety', {}).get('max_daily_loss_pct', -0.03) * 100:.0f}%")
        print(f"    총 손실 한도:    {self.config.get('live_trading', {}).get('safety', {}).get('max_total_loss_pct', -0.10) * 100:.0f}%")
        print("=" * 60)


def setup_logging():
    """로깅 설정 (콘솔 + 파일)"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 콘솔
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console)

    # 파일 (로테이션: 10MB, 5개 백업)
    file_handler = RotatingFileHandler(
        log_dir / "scheduler.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root_logger.addHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v4.0 일일 스케줄러")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="스케줄 확인만 (실행 안함)",
    )
    parser.add_argument(
        "--run-now", type=int, choices=range(1, 10), metavar="N",
        help="특정 Phase를 즉시 실행 (1~9)",
    )
    args = parser.parse_args()

    setup_logging()

    scheduler = DailyScheduler()

    if args.dry_run:
        scheduler.print_schedule()
    elif args.run_now:
        phases = {
            1: scheduler.phase_daily_reset,
            2: scheduler.phase_macro_collect,
            3: scheduler.phase_news_briefing,
            4: scheduler.phase_trade_prep,
            5: scheduler.phase_buy_execution,
            6: scheduler.phase_intraday_monitor,
            7: scheduler.phase_sell_execution,
            8: scheduler.phase_close_pipeline,
            9: scheduler.phase_eod_journal,
        }
        func = phases.get(args.run_now)
        if func:
            func()
    else:
        scheduler.run()
