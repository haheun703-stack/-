#!/usr/bin/env python3
"""
v5.1 일일 스케줄러 — 한국장 준비 ~ 미장 마감 전체 사이클

일일 스케줄 (KST):
  === 미장 마감 + 한국장 준비 ===
  00:00  Phase 0    일일 리셋 (STOP.signal 삭제 + 로그 로테이션)
  06:10  Phase 1    미장 마감 데이터 + US Overnight Signal
  07:00  Phase 2    한국 매크로 수집
  07:20  Phase 3    뉴스 스캔 (Grok API)
  07:30  Phase 3B   [TG] 장전 마켓 브리핑
  08:00  Phase 3C   [TG] ETF 매매 시그널
  08:20  Phase 4    매매 준비 (토큰+공휴일+확정)

  === 한국장 운영 ===
  09:02  Phase 5    매수 실행
  09:10  Phase 6    장중 모니터링 (~15:20)
  09:30  수급 1차   개장 30분
  11:00  수급 2차   오전장
  13:30  수급 3차   오후장
  15:00  수급 4차   마감 직전
  15:25  Phase 7    매도 실행

  === 장마감 + 데이터 업데이트 ===
  15:32  Phase 8-0B 전종목 체결 스냅샷
  15:35  Phase 8-0A 전종목 분봉 아카이브
  15:40  Phase 8-1  종가 수집 + CSV 업데이트 (통합)
  16:10  Phase 8-3  parquet 증분
  16:20  Phase 8-4  지표 재계산 (35개)
  16:30  Phase 8-5  데이터 검증
  16:35  Phase 8-6  ETF 시그널 생성
  16:40  Phase 8-7  KOSPI 인덱스 업데이트

  === 수급 확정 + 스캔 + 리포트 ===
  18:20  Phase 9    수급 최종 확정
  18:30  Phase 9.5  릴레이 포지션 체크
  18:40  Phase 10   scan_all() 매수 후보 스캔
  19:00  Phase 10B  [TG] 통합 데일리 리포트
  19:30  Phase 11   업무일지

안전장치:
  STOP.signal — 매수/매도/모니터링 중단
  reboot.trigger — 스케줄러 graceful 재시작
  공휴일 — Phase 5~7 자동 스킵

사용법:
  python scripts/daily_scheduler.py               # 스케줄러 시작
  python scripts/daily_scheduler.py --dry-run      # 스케줄 확인만
  python scripts/daily_scheduler.py --run-now 10   # 특정 Phase 즉시 실행 (0~11, 8-7, 9.5)
"""

from __future__ import annotations

import argparse
import json
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

# 수급 스냅샷 저장 디렉토리
SUPPLY_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "supply_snapshots"
SUPPLY_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


class DailyScheduler:
    """v5.1 일일 스케줄러 — 한국장 준비 ~ 미장 마감"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        live_cfg = self.config.get("live_trading", {})
        self.schedule = live_cfg.get("schedule", {})
        self.enabled = live_cfg.get("enabled", False)
        self.mode = live_cfg.get("mode", "paper")
        self.supply_cfg = live_cfg.get("supply_monitor", {})

        # 상태
        self._is_holiday = False
        self._buy_signals: list[dict] = []
        self._supply_snapshots: list[dict] = []  # 장중 수급 스냅샷 누적
        self._cmd_bot = None  # 텔레그램 명령 봇

        logger.info(
            "DailyScheduler v5.0 초기화 (enabled=%s, mode=%s)",
            self.enabled, self.mode,
        )

    # ══════════════════════════════════════════
    # Phase 0: 일일 리셋 (00:00)
    # ══════════════════════════════════════════

    def phase_daily_reset(self) -> None:
        """STOP.signal 삭제 + 로그 로테이션 + 일일 초기화"""
        logger.info("=" * 50)
        logger.info("[Phase 0] 일일 리셋 — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
        logger.info("=" * 50)

        from src.use_cases.safety_guard import SafetyGuard
        guard = SafetyGuard(self.config)
        guard.clear_stop_signal()

        self._is_holiday = False
        self._buy_signals = []
        self._supply_snapshots = []

        logger.info("[Phase 0] 일일 리셋 완료")
        self._notify("Phase 0 완료: 일일 리셋")

    # ══════════════════════════════════════════
    # Phase 1: 미장 마감 데이터 수집 (06:10)
    # ══════════════════════════════════════════

    def phase_us_close_collect(self) -> None:
        """미장 마감 후 yfinance 데이터 수집 + US Overnight Signal 생성"""
        logger.info("[Phase 1] 미장 마감 데이터 수집 시작")
        try:
            from scripts.us_overnight_signal import update_latest, generate_signal
            df = update_latest()
            signal = generate_signal(df)
            grade = signal.get("grade", "NEUTRAL")
            score = signal.get("combined_score_100", 0)
            logger.info("[Phase 1] US Overnight: %s (%+.1f)", grade, score)
            self._notify(f"Phase 1 완료: US {grade} ({score:+.1f})")
        except Exception as e:
            logger.error("[Phase 1] 미장 데이터 수집 실패: %s", e)
            self._notify(f"Phase 1 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 2: 한국 매크로 수집 (07:00)
    # ══════════════════════════════════════════

    def phase_macro_collect(self) -> None:
        """KOSPI/KOSDAQ/환율/금리 매크로 데이터 수집"""
        logger.info("[Phase 2] 한국 매크로 수집 시작")
        try:
            from scripts.update_daily_data import update_all
            update_all()
            logger.info("[Phase 2] 매크로 수집 완료")
        except Exception as e:
            logger.error("[Phase 2] 매크로 수집 실패: %s", e)
        self._notify("Phase 2 완료: 한국 매크로 수집")

    # ══════════════════════════════════════════
    # Phase 3: 뉴스 스캔 (07:20)
    # ══════════════════════════════════════════

    def phase_news_briefing(self) -> None:
        """Grok API 뉴스 스캔 + 감정 분석"""
        logger.info("[Phase 3] 뉴스/리포트 수집 시작")
        try:
            from main import step_news_scan
            step_news_scan(send_telegram=False)  # 별도 브리핑에서 통합 발송
            logger.info("[Phase 3] 뉴스 스캔 완료")
        except Exception as e:
            logger.error("[Phase 3] 뉴스 스캔 실패: %s", e)
        self._notify("Phase 3 완료: 뉴스 스캔")

    # ══════════════════════════════════════════
    # Phase 3B: 📱 1발 장전 마켓 브리핑 (07:30)
    # ══════════════════════════════════════════

    def phase_morning_briefing(self) -> None:
        """장전 마켓 브리핑 텔레그램 발송 (상승/하락 확률 + S/A/B/C)"""
        logger.info("[Phase 3B] 📱 장전 마켓 브리핑 시작")
        try:
            from scripts.send_market_briefing import build_briefing_message
            from src.telegram_sender import send_message

            msg = build_briefing_message()
            ok = send_message(msg)
            if ok:
                logger.info("[Phase 3B] 📱 1발 장전 브리핑 전송 완료 (%d자)", len(msg))
            else:
                logger.error("[Phase 3B] 📱 1발 전송 실패")
        except Exception as e:
            logger.error("[Phase 3B] 장전 브리핑 실패: %s", e)
            self._notify(f"Phase 3B 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 3C: 📱 ETF 시그널 텔레그램 발송 (08:00)
    # ══════════════════════════════════════════

    def phase_etf_briefing(self) -> None:
        """전일 생성된 ETF 매매 시그널 텔레그램 발송 (장전)"""
        logger.info("[Phase 3C] 📱 ETF 시그널 텔레그램 발송 시작")
        try:
            from scripts.etf_trading_signal import build_telegram_message, OUT_PATH
            from src.telegram_sender import send_message

            if not OUT_PATH.exists():
                logger.warning("[Phase 3C] etf_trading_signal.json 없음 — 스킵")
                return

            with open(OUT_PATH, "r", encoding="utf-8") as f:
                signals = json.load(f)

            msg = build_telegram_message(signals)
            ok = send_message(msg)
            if ok:
                logger.info("[Phase 3C] 📱 ETF 시그널 전송 완료 (%d자)", len(msg))
            else:
                logger.error("[Phase 3C] 📱 ETF 시그널 전송 실패")
        except Exception as e:
            logger.error("[Phase 3C] ETF 시그널 발송 실패: %s", e)
            self._notify(f"Phase 3C 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 4: 매매 준비 (08:20)
    # ══════════════════════════════════════════

    def phase_trade_prep(self) -> None:
        """토큰 갱신 → 공휴일 체크 → 매수 후보 확정"""
        logger.info("[Phase 4] 매매 준비 시작")

        from src.use_cases.safety_guard import SafetyGuard
        guard = SafetyGuard(self.config)
        self._is_holiday = guard.check_holiday()

        if self._is_holiday:
            logger.info("[Phase 4] 오늘은 공휴일/주말 — Phase 5~7 스킵")
            self._notify("Phase 4: 공휴일 감지 — 매매 스킵")
            return

        if guard.check_stop_signal():
            logger.warning("[Phase 4] STOP.signal 활성 — 매매 스킵")
            self._notify("Phase 4: STOP.signal 활성 — 매매 스킵")
            return

        try:
            from src.adapters.kis_order_adapter import KisOrderAdapter
            adapter = KisOrderAdapter()
            balance = adapter.get_available_cash()
            logger.info("[Phase 4] 한투 API 토큰 OK (예수금: %s원)", f"{balance:,.0f}")
        except Exception as e:
            logger.error("[Phase 4] 한투 API 연결 실패: %s", e)
            self._notify(f"Phase 4 경고: 한투 API 실패 — {e}")
            return

        try:
            self._load_signals()
            logger.info("[Phase 4] 매수 후보 %d종목 확정", len(self._buy_signals))
        except Exception as e:
            logger.error("[Phase 4] 시그널 로드 실패: %s", e)

        self._notify(f"Phase 4 완료: 매수 후보 {len(self._buy_signals)}종목")

    def _load_signals(self) -> None:
        """scan_cache.json에서 매수 후보 로드 (Phase 10에서 저장)."""
        scan_path = Path("data/scan_cache.json")
        if not scan_path.exists():
            self._buy_signals = []
            return
        try:
            data = json.loads(scan_path.read_text(encoding="utf-8"))
            self._buy_signals = data.get("candidates", [])
        except Exception:
            self._buy_signals = []

    # ══════════════════════════════════════════
    # Phase 5: 매수 실행 (09:02)
    # ══════════════════════════════════════════

    def phase_buy_execution(self) -> None:
        """매수 후보 기반 주문 실행"""
        if self._is_holiday:
            return
        logger.info("[Phase 5] 매수 실행 (후보 %d종목)", len(self._buy_signals))
        if not self._buy_signals:
            self._notify("Phase 5: 매수 후보 없음")
            return
        if not self.enabled:
            self._notify("Phase 5: 모의 모드 — 실주문 안함")
            return
        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()
            results = engine.execute_buy_signals(self._buy_signals)
            ok = sum(1 for r in results if r.get("success"))
            self._notify(f"Phase 5 완료: {ok}종목 매수 성공")
        except Exception as e:
            logger.error("[Phase 5] 매수 오류: %s", e)
            self._notify(f"Phase 5 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 6: 장중 모니터링 (09:10 ~ 15:20)
    # ══════════════════════════════════════════

    def phase_intraday_monitor(self) -> None:
        """장중 실시간 모니터링 (1분 간격)"""
        if self._is_holiday or not self.enabled:
            return
        logger.info("[Phase 6] 장중 모니터링 시작 (09:10 ~ 15:20)")
        self._notify("Phase 6 시작: 장중 모니터링")
        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()
            engine.monitor_loop()
        except Exception as e:
            logger.error("[Phase 6] 모니터링 오류: %s", e)
            self._notify(f"Phase 6 오류: {e}")

    # ══════════════════════════════════════════
    # 📸 수급 스냅샷 (09:30, 11:00, 13:30, 15:00)
    # ══════════════════════════════════════════

    def phase_supply_snapshot(self, snapshot_num: int = 0) -> None:
        """장중 수급 스냅샷 수집 + 급변 알림"""
        if self._is_holiday:
            return

        labels = {1: "개장30분", 2: "오전장", 3: "오후장전환", 4: "마감직전"}
        label = labels.get(snapshot_num, f"#{snapshot_num}")
        logger.info("[📸 수급 %d차] %s 스냅샷 수집 시작", snapshot_num, label)

        try:
            from src.adapters.kis_intraday_adapter import KisIntradayAdapter
            adapter = KisIntradayAdapter(self.config)

            # 수집 대상: 보유종목 + 관심종목 (전일 스캔 결과)
            tickers = self._get_supply_tickers()
            if not tickers:
                logger.info("[📸 수급 %d차] 수집 대상 없음 — 스킵", snapshot_num)
                return

            now_str = datetime.now().strftime("%H:%M")
            snapshot = {
                "snapshot_num": snapshot_num,
                "label": label,
                "time": now_str,
                "timestamp": datetime.now().isoformat(),
                "stocks": {},
            }

            for ticker in tickers:
                try:
                    flow = adapter.fetch_investor_flow(ticker)
                    snapshot["stocks"][ticker] = {
                        "foreign_net": flow.get("foreign_net_buy", 0),
                        "inst_net": flow.get("inst_net_buy", 0),
                        "individual_net": flow.get("individual_net_buy", 0),
                    }
                except Exception as e:
                    logger.warning("[📸] %s 수급 조회 실패: %s", ticker, e)

            # 저장
            self._supply_snapshots.append(snapshot)
            today = datetime.now().strftime("%Y%m%d")
            snap_path = SUPPLY_SNAPSHOT_DIR / f"{today}_snap{snapshot_num}.json"
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

            logger.info(
                "[📸 수급 %d차] %d종목 수집 완료 → %s",
                snapshot_num, len(snapshot["stocks"]), snap_path.name,
            )

            # 급변 알림 체크 (2차부터)
            if snapshot_num >= 2:
                self._check_supply_alert(snapshot)

        except Exception as e:
            logger.error("[📸 수급 %d차] 오류: %s", snapshot_num, e)

    def _get_supply_tickers(self) -> list[str]:
        """수급 수집 대상 종목 리스트"""
        tickers = set()

        # 1. 보유 종목
        try:
            pos_path = Path("data/positions.json")
            if pos_path.exists():
                with open(pos_path, encoding="utf-8") as f:
                    pos_data = json.load(f)
                for p in pos_data.get("positions", []):
                    tickers.add(p["ticker"])
        except Exception:
            pass

        # 2. 전일 스캔 후보 (watchlist) — scan_cache.json
        try:
            scan_path = Path("data/scan_cache.json")
            if scan_path.exists():
                scan_data = json.loads(scan_path.read_text(encoding="utf-8"))
                for item in scan_data.get("candidates", []):
                    tickers.add(str(item["ticker"]).zfill(6))
        except Exception:
            pass

        return list(tickers)[:20]  # 최대 20종목 (API 부하 방지)

    def _check_supply_alert(self, current: dict) -> None:
        """이전 스냅샷 대비 수급 급변 알림"""
        if len(self._supply_snapshots) < 2:
            return

        prev = self._supply_snapshots[-2]
        alerts = []

        for ticker, cur_data in current.get("stocks", {}).items():
            prev_data = prev.get("stocks", {}).get(ticker)
            if not prev_data:
                continue

            cur_f = cur_data["foreign_net"]
            prev_f = prev_data["foreign_net"]
            cur_i = cur_data["inst_net"]
            prev_i = prev_data["inst_net"]

            # 외국인 방향 전환 (매수→매도 or 매도→매수)
            if self.supply_cfg.get("alert_conditions", {}).get("direction_flip"):
                if prev_f > 0 and cur_f < 0:
                    alerts.append(f"🔴 {ticker}: 외국인 매수→매도 전환!")
                elif prev_f < 0 and cur_f > 0:
                    alerts.append(f"🟢 {ticker}: 외국인 매도→매수 전환!")

            # 외국인+기관 동시 순매도
            threshold = self.supply_cfg.get("alert_conditions", {}).get("dual_sell_threshold", 100000)
            if cur_f < -threshold and cur_i < -threshold:
                alerts.append(
                    f"🚨 {ticker}: 외국인({cur_f:+,})+기관({cur_i:+,}) 동시 대량매도!"
                )

        if alerts:
            snap_num = current["snapshot_num"]
            msg = f"⚡ 수급 급변 알림 ({snap_num}차 스냅샷)\n" + "\n".join(alerts)
            logger.warning(msg)
            try:
                from src.telegram_sender import send_message
                send_message(msg)
            except Exception:
                pass

    # ══════════════════════════════════════════
    # Phase 7: 매도 실행 (15:25)
    # ══════════════════════════════════════════

    def phase_sell_execution(self) -> None:
        """장마감 전 청산 대상 매도"""
        if self._is_holiday or not self.enabled:
            return
        logger.info("[Phase 7] 매도 실행 시작")
        try:
            from src.use_cases.live_trading import create_live_engine
            engine = create_live_engine()
            results = engine.execute_sell_signals()
            sell_count = sum(1 for r in results if r.get("success"))
            self._notify(f"Phase 7 완료: {sell_count}건 매도")
        except Exception as e:
            logger.error("[Phase 7] 매도 오류: %s", e)
            self._notify(f"Phase 7 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 8: 장마감 데이터 업데이트 (15:40~16:30, 5단계)
    # ══════════════════════════════════════════

    # ══════════════════════════════════════════
    # Phase 8-0A: 분봉(5분/15분) 아카이브 (15:35)
    # ══════════════════════════════════════════

    def phase_candle_archive(self) -> None:
        """8-0A: 전종목 5분봉+15분봉 → parquet 아카이브"""
        logger.info("[Phase 8-0A] 분봉 아카이브 시작")
        try:
            from scripts.collect_intraday_candles import (
                load_universe, collect_one_ticker, KisIntradayAdapter,
            )
            from datetime import datetime as dt
            tickers = load_universe()
            adapter = KisIntradayAdapter()
            date_str = dt.now().strftime("%Y-%m-%d")
            ok, fail = 0, 0
            for ticker in tickers:
                try:
                    result = collect_one_ticker(adapter, ticker, date_str)
                    if result["status"] == "ok":
                        ok += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
            logger.info("[Phase 8-0A] 분봉 아카이브 완료 (%d성공, %d실패)", ok, fail)
            self._notify(f"Phase 8-0A: 분봉 아카이브 {ok}종목 완료")
        except Exception as e:
            logger.error("[Phase 8-0A] 분봉 아카이브 실패: %s", e)
            self._notify(f"Phase 8-0A 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 8-0B: 전종목 체결 스냅샷 (15:32)
    # ══════════════════════════════════════════

    def phase_tick_snapshot(self) -> None:
        """8-0B: 전종목 체결정보 스냅샷 → parquet"""
        logger.info("[Phase 8-0B] 체결 스냅샷 시작")
        try:
            from scripts.collect_tick_snapshot import (
                load_universe, collect_all_ticks, KisIntradayAdapter,
            )
            tickers = load_universe()
            adapter = KisIntradayAdapter()
            result = collect_all_ticks(adapter, tickers)
            logger.info(
                "[Phase 8-0B] 체결 스냅샷 완료 (%d종목, %d건)",
                result["stocks"], result["rows"],
            )
            self._notify(f"Phase 8-0B: 체결 스냅샷 {result['stocks']}종목")
        except Exception as e:
            logger.error("[Phase 8-0B] 체결 스냅샷 실패: %s", e)
            self._notify(f"Phase 8-0B 오류: {e}")

    def phase_close_data_collect(self) -> None:
        """8-1: 종가 수집 + CSV 업데이트 (통합)."""
        logger.info("[Phase 8-1] 종가 수집 + CSV 업데이트 시작")
        try:
            from scripts.update_daily_data import update_all
            update_all()
            logger.info("[Phase 8-1] 종가 수집 + CSV 업데이트 완료")
        except Exception as e:
            logger.error("[Phase 8-1] 종가 수집 실패: %s", e)

    def phase_parquet_update(self) -> None:
        """8-3: parquet 증분 업데이트"""
        logger.info("[Phase 8-3] parquet 증분 업데이트 시작")
        try:
            from scripts.extend_parquet_data import main as extend_main
            extend_main()
            logger.info("[Phase 8-3] parquet 증분 완료")
        except Exception as e:
            logger.error("[Phase 8-3] parquet 증분 실패: %s", e)

    def phase_indicator_calc(self) -> None:
        """8-4: 기술적 지표 재계산 (35개)"""
        logger.info("[Phase 8-4] 지표 재계산 시작")
        try:
            from main import step_indicators
            step_indicators()
            logger.info("[Phase 8-4] 지표 재계산 완료")
        except Exception as e:
            logger.error("[Phase 8-4] 지표 계산 실패: %s", e)

    def phase_data_verify(self) -> None:
        """8-5: 데이터 검증 (NaN 체크)"""
        logger.info("[Phase 8-5] 데이터 검증 시작")
        try:
            from scripts.update_daily_data import verify_all_data
            verify_all_data()
            logger.info("[Phase 8-5] 데이터 검증 완료")
        except (ImportError, AttributeError):
            logger.info("[Phase 8-5] verify_all_data 없음 — 스킵")
        except Exception as e:
            logger.error("[Phase 8-5] 검증 실패: %s", e)

    # ══════════════════════════════════════════
    # Phase 8-6: ETF 매매 시그널 생성 (16:35)
    # ══════════════════════════════════════════

    def phase_etf_signal(self) -> None:
        """8-6: 섹터 ETF 매매 시그널 생성 (JSON 저장만, 텔레그램은 08:00에 발송)"""
        logger.info("[Phase 8-6] ETF 매매 시그널 생성 시작")
        try:
            from scripts.etf_trading_signal import generate_etf_signals, save_signals

            signals = generate_etf_signals()
            save_signals(signals)

            s = signals.get("summary", {})
            logger.info(
                "[Phase 8-6] ETF 시그널 저장: SMART %d개, THEME %d개, 관찰 %d개",
                s.get("smart_buy", 0), s.get("theme_buy", 0), s.get("watch", 0),
            )
            self._notify(f"Phase 8-6: ETF 시그널 저장 (SMART {s.get('smart_buy', 0)}, THEME {s.get('theme_buy', 0)})")
        except Exception as e:
            logger.error("[Phase 8-6] ETF 시그널 실패: %s", e)
            self._notify(f"Phase 8-6 오류: {e}")

    # ══════════════════════════════════════════
    # Phase 8-7: KOSPI 인덱스 업데이트 (16:40)
    # ══════════════════════════════════════════

    def phase_kospi_update(self) -> None:
        """8-7: KOSPI 인덱스(^KS11) yfinance 업데이트 → kospi_index.csv"""
        logger.info("[Phase 8-7] KOSPI 인덱스 업데이트 시작")
        try:
            import yfinance as yf
            import pandas as pd
            kospi_path = PROJECT_ROOT / "data" / "kospi_index.csv"
            df_old = pd.read_csv(kospi_path, index_col="Date", parse_dates=True)
            last_date = df_old.index[-1].strftime("%Y-%m-%d")
            df_new = yf.download("^KS11", start=last_date, progress=False)
            if not df_new.empty:
                # MultiIndex 열 flatten (yfinance 신버전 대응)
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = [c[0].lower() for c in df_new.columns]
                else:
                    df_new.columns = [c.lower() for c in df_new.columns]
                df_new.index.name = "Date"
                combined = pd.concat([df_old, df_new[~df_new.index.isin(df_old.index)]])
                combined.to_csv(kospi_path)
                logger.info("[Phase 8-7] KOSPI 업데이트 완료 (%d행)", len(combined))
            else:
                logger.info("[Phase 8-7] KOSPI 신규 데이터 없음")
        except Exception as e:
            logger.error("[Phase 8-7] KOSPI 업데이트 실패: %s", e)

    # ══════════════════════════════════════════
    # Phase 9: 수급 최종 확정 수집 (18:20)
    # ══════════════════════════════════════════

    def phase_supply_final(self) -> None:
        """18:10 이후 확정된 외국인/기관/공매도 수급 수집"""
        logger.info("[Phase 9] 수급 최종 확정 수집 시작")
        try:
            from scripts.collect_supply_data import main as collect_supply
            collect_supply()
            logger.info("[Phase 9] 수급 최종 확정 수집 완료")
        except Exception as e:
            logger.error("[Phase 9] 수급 확정 수집 실패: %s", e)

        # US-KR 패턴DB 업데이트
        try:
            from scripts.update_us_kr_daily import main as update_uskr
            update_uskr()
            logger.info("[Phase 9] US-KR 패턴DB 업데이트 완료")
        except Exception as e:
            logger.error("[Phase 9] US-KR 패턴DB 실패: %s", e)

        self._notify("Phase 9 완료: 수급 확정 수집")

    # ══════════════════════════════════════════
    # Phase 9.5: 릴레이 포지션 체크 (18:30)
    # ══════════════════════════════════════════

    def phase_relay_check(self) -> None:
        """9.5: 릴레이 포지션 청산 조건 체크."""
        logger.info("[Phase 9.5] 릴레이 포지션 체크 시작")
        try:
            from scripts.relay_positions import check_all_positions
            results = check_all_positions()
            exits = [r for r in results if r.get("exit")]
            if exits:
                names = ", ".join(r["name"] for r in exits)
                self._notify(f"Phase 9.5: 릴레이 청산 {len(exits)}건 ({names})")
                logger.info("[Phase 9.5] 릴레이 청산 대상: %s", names)
            else:
                logger.info("[Phase 9.5] 릴레이 포지션 %d건 전부 HOLD", len(results))
        except Exception as e:
            logger.error("[Phase 9.5] 릴레이 체크 실패: %s", e)

    # ══════════════════════════════════════════
    # Phase 10: 내일 매수 후보 스캔 (18:40)
    # ══════════════════════════════════════════

    def phase_evening_scan(self) -> None:
        """수급 반영된 최신 데이터로 내일 매수 후보 스캔 (scan_all)."""
        logger.info("[Phase 10] 내일 매수 후보 스캔 시작")
        try:
            import importlib
            mod = importlib.import_module("scan_buy_candidates")
            candidates, stats = mod.scan_all(grade_filter="AB", use_news=False)
            # 캐시 저장 (Phase 10B 통합 리포트에서 사용)
            from scripts.daily_integrated_report import _save_scan_cache
            _save_scan_cache(candidates, stats)
            logger.info("[Phase 10] scan_all 완료: %d종목 통과", len(candidates))
        except Exception as e:
            logger.error("[Phase 10] 스캔 실패: %s", e)
        self._notify("Phase 10 완료: 스캔")

    # ══════════════════════════════════════════
    # Phase 10B: 📱 2발 장마감 리포트 (19:00)
    # ══════════════════════════════════════════

    def phase_evening_briefing(self) -> None:
        """장마감 통합 데일리 리포트 텔레그램 발송."""
        logger.info("[Phase 10B] 통합 데일리 리포트 시작")
        try:
            from scripts.daily_integrated_report import run_report
            run_report(send=True, run_scan=False, use_news=False)
            logger.info("[Phase 10B] 통합 데일리 리포트 완료")
        except Exception as e:
            logger.error("[Phase 10B] 통합 리포트 실패, 폴백: %s", e)
            try:
                msg = self._build_evening_message()
                from src.telegram_sender import send_message
                send_message(msg)
            except Exception as e2:
                logger.error("[Phase 10B] 폴백도 실패: %s", e2)
                self._notify(f"Phase 10B 오류: {e}")

    def _build_evening_message(self) -> str:
        """장마감 리포트 메시지 생성 (📱2발)"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = []

        lines.append(f"\U0001f319 Quantum Master v10.3 | {now}")
        lines.append("\u2501" * 28)
        lines.append("")

        # ── 오늘 매매 결과 ──
        lines.append("\U0001f4b0 오늘 매매 결과")
        lines.append("\u2500" * 28)
        try:
            pos_path = Path("data/positions.json")
            if pos_path.exists():
                with open(pos_path, encoding="utf-8") as f:
                    pos_data = json.load(f)
                positions = pos_data.get("positions", [])
                if positions:
                    for p in positions:
                        name = p.get("name", p.get("ticker", "?"))
                        pnl = p.get("unrealized_pnl_pct", 0)
                        pnl_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
                        lines.append(
                            f"  {pnl_emoji} {name} | {pnl:+.1f}%"
                        )
                    lines.append(f"  \U0001f4cb 보유: {len(positions)}종목")
                else:
                    lines.append("  보유 종목 없음")
            else:
                lines.append("  보유 종목 없음")
        except Exception:
            lines.append("  포지션 데이터 로드 실패")
        lines.append("")

        # ── MDD 모니터 (잼블랙 인사이트) ──
        try:
            from src.mdd_monitor import MDDMonitor
            mdd_mon = MDDMonitor()
            # 현재 자산 계산 (현금 + 보유 평가)
            try:
                pos_path_mdd = Path("data/positions.json")
                if pos_path_mdd.exists():
                    with open(pos_path_mdd, encoding="utf-8") as f:
                        pd_mdd = json.load(f)
                    capital = pd_mdd.get("capital", 100_000_000)
                    pos_list = pd_mdd.get("positions", [])
                    eval_total = sum(
                        p.get("current_price", p.get("entry_price", 0)) * p.get("shares", 0)
                        for p in pos_list
                    )
                    invested = sum(
                        p.get("entry_price", 0) * p.get("shares", 0)
                        for p in pos_list
                    )
                    current_equity = capital - invested + eval_total
                else:
                    current_equity = 100_000_000
            except Exception:
                current_equity = 100_000_000

            mdd_result = mdd_mon.update(current_equity)
            lines.append(f"  {mdd_mon.format_status_line()}")
            lines.append("")

            # MDD 알림이 있으면 별도 발송
            alert = mdd_mon.get_alert()
            if alert:
                from src.telegram_sender import send_mdd_alert
                send_mdd_alert(alert)
        except Exception as e:
            logger.debug("[Phase 10B] MDD 모니터 오류: %s", e)

        # ── 수급 히스토리 (4회 스냅샷) ──
        lines.append("\U0001f4ca 장중 수급 흐름 (4회 스냅샷)")
        lines.append("\u2500" * 28)
        if self._supply_snapshots:
            for snap in self._supply_snapshots:
                label = snap["label"]
                t = snap["time"]
                stocks = snap.get("stocks", {})
                if not stocks:
                    continue
                total_f = sum(s["foreign_net"] for s in stocks.values())
                total_i = sum(s["inst_net"] for s in stocks.values())
                f_emoji = "\U0001f7e2" if total_f > 0 else "\U0001f534"
                i_emoji = "\U0001f7e2" if total_i > 0 else "\U0001f534"
                lines.append(
                    f"  {t} {label}: "
                    f"{f_emoji}외 {total_f:+,} | "
                    f"{i_emoji}기 {total_i:+,}"
                )
            # 수급 트렌드 요약
            if len(self._supply_snapshots) >= 2:
                first = self._supply_snapshots[0]
                last = self._supply_snapshots[-1]
                f_first = sum(s["foreign_net"] for s in first.get("stocks", {}).values())
                f_last = sum(s["foreign_net"] for s in last.get("stocks", {}).values())
                if f_first > 0 and f_last > 0:
                    lines.append("  \u2514 \U0001f7e2 외국인 종일 매수 우위")
                elif f_first < 0 and f_last < 0:
                    lines.append("  \u2514 \U0001f534 외국인 종일 매도 우위")
                elif f_first > 0 and f_last < 0:
                    lines.append("  \u2514 \u26a0\ufe0f 외국인 매수\u2192매도 전환")
                elif f_first < 0 and f_last > 0:
                    lines.append("  \u2514 \U0001f7e2 외국인 매도\u2192매수 전환")
        else:
            lines.append("  스냅샷 데이터 없음 (장중 수집 안됨)")
        lines.append("")

        # ── 내일 매수 후보 (섹터 로테이션 기반 + S/A/B 등급) ──
        lines.append("\U0001f525 내일 매수 후보")
        lines.append("\u2501" * 28)
        try:
            scan_path = Path("data/sector_rotation/krx_sector_scan.json")
            if scan_path.exists():
                with open(scan_path, encoding="utf-8") as f:
                    scan_data = json.load(f)
                smart = scan_data.get("smart_money", [])
                theme = scan_data.get("theme_money", [])
                if smart or theme:
                    if smart:
                        lines.append("\U0001f48e Smart Money (외인+기관)")
                        for s in smart[:3]:
                            g, ge = self._grade_stock(s, "SMART")
                            name = s.get("name", str(s.get("ticker", "?")).zfill(6))
                            ticker = str(s.get("ticker", "")).zfill(6)
                            bb = s.get("bb_pct", 0)
                            rsi = s.get("rsi", 0)
                            stop = s.get("stop_pct", -7)
                            sizing = s.get("sizing", "FULL")
                            sector = s.get("etf_sector", "")
                            lines.append(f"{ge} {g}급 {name} ({ticker}) — {sector}")
                            lines.append(
                                f"  BB {bb:.0f}% | RSI {rsi:.0f} | "
                                f"손절 {stop}% | {sizing}"
                            )
                            lines.append("")
                    if theme:
                        lines.append("\U0001f525 Theme Money (모멘텀)")
                        for t in theme[:3]:
                            g, ge = self._grade_stock(t, "THEME")
                            name = t.get("name", str(t.get("ticker", "?")).zfill(6))
                            ticker = str(t.get("ticker", "")).zfill(6)
                            bb = t.get("bb_pct", 0)
                            rsi = t.get("rsi", 0)
                            adx = t.get("adx", 0)
                            sector = t.get("etf_sector", "")
                            lines.append(f"{ge} {g}급 {name} ({ticker}) — {sector}")
                            lines.append(f"  BB {bb:.0f}% | RSI {rsi:.0f} | ADX {adx:.0f}")
                            lines.append("")
                else:
                    lines.append("  스캔 통과 종목 없음")
                    lines.append("")
            else:
                lines.append("  스캔 결과 없음")
                lines.append("")
        except Exception:
            lines.append("  스캔 결과 로드 실패")
            lines.append("")

        # ── 내일 주의사항 ──
        lines.append("\U0001f4cb 내일 체크리스트")
        lines.append("\u2500" * 28)
        # US overnight signal 읽기
        try:
            sig_path = Path("data/us_market/overnight_signal.json")
            if sig_path.exists():
                with open(sig_path, encoding="utf-8") as f:
                    us_sig = json.load(f)
                grade = us_sig.get("grade", "NEUTRAL")
                combined = us_sig.get("combined_score_100", 0)
                lines.append(f"  \u251c US Signal: {grade} ({combined:+.1f})")
            kills = us_sig.get("sector_kills", {}) if sig_path.exists() else {}
            killed = [s for s, v in kills.items() if v.get("killed")]
            if killed:
                lines.append(f"  \u251c \U0001f6a8 섹터Kill: {', '.join(killed)}")
            specials = us_sig.get("special_rules", []) if sig_path.exists() else []
            if specials:
                for rule in specials[:2]:
                    lines.append(f"  \u251c \u26a0\ufe0f {rule}")
        except Exception:
            pass
        lines.append(f"  \u2514 07:30 장전 브리핑에서 최종 확률 확인")
        lines.append("")

        lines.append("\u26a0\ufe0f 투자 판단은 본인 책임 | Quantum Master v10.3")
        return "\n".join(lines)

    # ══════════════════════════════════════════
    # Phase 11: 업무일지 (19:30)
    # ══════════════════════════════════════════

    def phase_eod_journal(self) -> None:
        """일일 업무일지 HTML 생성"""
        logger.info("[Phase 11] 업무일지 생성 시작")
        try:
            from src.use_cases.daily_journal import DailyJournalWriter
            writer = DailyJournalWriter(self.config)
            save_path = writer.generate()
            if save_path:
                logger.info("[Phase 11] 업무일지 저장: %s", save_path)
            self._notify("Phase 11 완료: 업무일지")
        except Exception as e:
            logger.error("[Phase 11] 업무일지 실패: %s", e)
            self._notify(f"Phase 11 오류: {e}")

    # ══════════════════════════════════════════
    # 헬퍼
    # ══════════════════════════════════════════

    @staticmethod
    def _grade_stock(item: dict, money_type: str) -> tuple[str, str]:
        """종목 등급 판정. Returns (등급, 이모지)."""
        bb = item.get("bb_pct", 50)
        rsi = item.get("rsi", 50)
        adx = item.get("adx", 0)
        gx = item.get("stoch_golden_recent", False)
        if money_type == "SMART":
            if bb < 30 and rsi < 45:
                return "S", "\U0001f525"
            elif bb < 50 and rsi < 55:
                return "A", "\u2b50"
            else:
                return "B", "\U0001f539"
        else:
            if adx > 50:
                return "S", "\U0001f525"
            elif adx > 40 or gx:
                return "A", "\u2b50"
            else:
                return "B", "\U0001f539"

    def _notify(self, message: str) -> None:
        """텔레그램 상태 알림 (실패 무시)"""
        try:
            from src.telegram_sender import send_message
            send_message(f"[\uc2a4\ucf00\uc904\ub7ec] {message}")
        except Exception:
            pass

    def _safe_run(self, func, *args) -> None:
        """예외 격리 실행"""
        try:
            func(*args)
        except Exception as e:
            logger.error("[스케줄러] %s 오류: %s", func.__name__, e)
            self._notify(f"오류: {func.__name__} — {e}")

    # ══════════════════════════════════════════
    # 메인 루프
    # ══════════════════════════════════════════

    def run(self) -> None:
        """v5.0 스케줄러 메인 루프"""
        import schedule as sched

        logger.info("=" * 60)
        logger.info("  v5.0 일일 스케줄러 시작")
        logger.info("  모드: %s | 실주문: %s", self.mode, "ON" if self.enabled else "OFF")
        logger.info("=" * 60)

        S = self.schedule  # shorthand

        # === 미장 마감 + 한국장 준비 ===
        sched.every().day.at(S.get("daily_reset", "00:00")).do(
            self._safe_run, self.phase_daily_reset)
        sched.every().day.at(S.get("us_close_collect", "06:10")).do(
            self._safe_run, self.phase_us_close_collect)
        sched.every().day.at(S.get("macro_collect", "07:00")).do(
            self._safe_run, self.phase_macro_collect)
        sched.every().day.at(S.get("news_briefing", "07:20")).do(
            self._safe_run, self.phase_news_briefing)
        sched.every().day.at(S.get("morning_briefing", "07:30")).do(
            self._safe_run, self.phase_morning_briefing)
        sched.every().day.at(S.get("etf_briefing", "08:00")).do(
            self._safe_run, self.phase_etf_briefing)
        sched.every().day.at(S.get("trade_prep", "08:20")).do(
            self._safe_run, self.phase_trade_prep)

        # === 한국장 운영 ===
        sched.every().day.at(S.get("buy_execution", "09:02")).do(
            self._safe_run, self.phase_buy_execution)
        sched.every().day.at(S.get("monitor_start", "09:10")).do(
            self._safe_run, self.phase_intraday_monitor)

        # === 장중 수급 스냅샷 4회 ===
        sched.every().day.at(S.get("supply_snapshot_1", "09:30")).do(
            self._safe_run, self.phase_supply_snapshot, 1)
        sched.every().day.at(S.get("supply_snapshot_2", "11:00")).do(
            self._safe_run, self.phase_supply_snapshot, 2)
        sched.every().day.at(S.get("supply_snapshot_3", "13:30")).do(
            self._safe_run, self.phase_supply_snapshot, 3)
        sched.every().day.at(S.get("supply_snapshot_4", "15:00")).do(
            self._safe_run, self.phase_supply_snapshot, 4)

        # === 매도 + 장마감 데이터 ===
        sched.every().day.at(S.get("sell_execution", "15:25")).do(
            self._safe_run, self.phase_sell_execution)
        sched.every().day.at(S.get("tick_snapshot", "15:32")).do(
            self._safe_run, self.phase_tick_snapshot)
        sched.every().day.at(S.get("candle_archive", "15:35")).do(
            self._safe_run, self.phase_candle_archive)
        sched.every().day.at(S.get("close_data_collect", "15:40")).do(
            self._safe_run, self.phase_close_data_collect)
        # Phase 8-2 제거 (8-1에서 update_all 통합 호출)
        sched.every().day.at(S.get("parquet_update", "16:10")).do(
            self._safe_run, self.phase_parquet_update)
        sched.every().day.at(S.get("indicator_calc", "16:20")).do(
            self._safe_run, self.phase_indicator_calc)
        sched.every().day.at(S.get("data_verify", "16:30")).do(
            self._safe_run, self.phase_data_verify)
        sched.every().day.at(S.get("etf_signal", "16:35")).do(
            self._safe_run, self.phase_etf_signal)
        sched.every().day.at(S.get("kospi_update", "16:40")).do(
            self._safe_run, self.phase_kospi_update)

        # === 수급 확정 + 스캔 + 리포트 ===
        sched.every().day.at(S.get("supply_final", "18:20")).do(
            self._safe_run, self.phase_supply_final)
        sched.every().day.at(S.get("relay_check", "18:30")).do(
            self._safe_run, self.phase_relay_check)
        sched.every().day.at(S.get("evening_scan", "18:40")).do(
            self._safe_run, self.phase_evening_scan)
        sched.every().day.at(S.get("evening_briefing", "19:00")).do(
            self._safe_run, self.phase_evening_briefing)
        sched.every().day.at(S.get("eod_journal", "19:30")).do(
            self._safe_run, self.phase_eod_journal)

        logger.info("등록된 스케줄 (%d건):", len(sched.get_jobs()))
        for job in sched.get_jobs():
            logger.info("  %s", job)

        # 텔레그램 명령 봇 시작 (백그라운드 스레드)
        try:
            from src.telegram_command_handler import TelegramCommandBot
            self._cmd_bot = TelegramCommandBot(scheduler=self)
            self._cmd_bot.start()
        except Exception as e:
            logger.warning("[스케줄러] 텔레그램 명령 봇 시작 실패: %s", e)

        self._notify("v5.0 스케줄러 시작됨 (명령봇 활성)")

        while True:
            try:
                sched.run_pending()
                from src.use_cases.safety_guard import SafetyGuard
                guard = SafetyGuard(self.config)
                if guard.check_reboot_trigger():
                    logger.info("[스케줄러] reboot.trigger 감지 — 재시작")
                    self._notify("스케줄러 재시작 중...")
                    if self._cmd_bot:
                        self._cmd_bot.stop()
                    time.sleep(10)
                    self.__init__()
                    continue
            except KeyboardInterrupt:
                logger.info("[스케줄러] Ctrl+C — 종료")
                if self._cmd_bot:
                    self._cmd_bot.stop()
                self._notify("스케줄러 종료됨")
                break
            except Exception as e:
                logger.error("[스케줄러] 오류: %s", e)
                self._notify(f"스케줄러 오류: {e}")
            time.sleep(1)

    # ══════════════════════════════════════════
    # dry-run 출력
    # ══════════════════════════════════════════

    def print_schedule(self) -> None:
        """v5.1 스케줄 표 출력."""
        S = self.schedule
        print()
        print("=" * 65)
        print("  v5.1 일일 스케줄 (한국장 준비 ~ 미장 마감)")
        print("=" * 65)
        print(f"  모드: {self.mode} | 실주문: {'ON' if self.enabled else 'OFF'}")
        print()

        sections = [
            ("\U0001f1fa\U0001f1f8 미장 마감 + \U0001f1f0\U0001f1f7 한국장 준비", [
                (S.get("daily_reset", "00:00"), "Phase 0", "일일 리셋"),
                (S.get("us_close_collect", "06:10"), "Phase 1", "미장 마감 데이터 + US Overnight Signal"),
                (S.get("macro_collect", "07:00"), "Phase 2", "한국 매크로 수집"),
                (S.get("news_briefing", "07:20"), "Phase 3", "뉴스 스캔 (Grok API)"),
                (S.get("morning_briefing", "07:30"), "Phase 3B", "[TG] 장전 마켓 브리핑"),
                (S.get("etf_briefing", "08:00"), "Phase 3C", "[TG] ETF 매매 시그널"),
                (S.get("trade_prep", "08:20"), "Phase 4", "매매 준비 (토큰+공휴일+확정)"),
            ]),
            ("\U0001f1f0\U0001f1f7 한국장 운영", [
                (S.get("buy_execution", "09:02"), "Phase 5", "매수 실행"),
                (S.get("monitor_start", "09:10"), "Phase 6", "장중 모니터링 (~15:20)"),
                (S.get("supply_snapshot_1", "09:30"), "\U0001f4f8 1차", "개장 30분"),
                (S.get("supply_snapshot_2", "11:00"), "\U0001f4f8 2차", "오전장"),
                (S.get("supply_snapshot_3", "13:30"), "\U0001f4f8 3차", "오후장"),
                (S.get("supply_snapshot_4", "15:00"), "\U0001f4f8 4차", "마감 직전"),
                (S.get("sell_execution", "15:25"), "Phase 7", "매도 실행"),
            ]),
            ("\U0001f1f0\U0001f1f7 장마감 + 데이터 업데이트", [
                (S.get("tick_snapshot", "15:32"), "Phase 8-0B", "전종목 체결 스냅샷"),
                (S.get("candle_archive", "15:35"), "Phase 8-0A", "전종목 분봉 아카이브"),
                (S.get("close_data_collect", "15:40"), "Phase 8-1", "종가 수집 + CSV 업데이트 (통합)"),
                (S.get("parquet_update", "16:10"), "Phase 8-3", "parquet 증분"),
                (S.get("indicator_calc", "16:20"), "Phase 8-4", "지표 재계산 (35개)"),
                (S.get("data_verify", "16:30"), "Phase 8-5", "데이터 검증"),
                (S.get("etf_signal", "16:35"), "Phase 8-6", "ETF 시그널 생성"),
                (S.get("kospi_update", "16:40"), "Phase 8-7", "KOSPI 인덱스 업데이트"),
            ]),
            ("\U0001f319 수급 확정 + 스캔 + 리포트", [
                (S.get("supply_final", "18:20"), "Phase 9", "수급 최종 확정"),
                (S.get("relay_check", "18:30"), "Phase 9.5", "릴레이 포지션 체크"),
                (S.get("evening_scan", "18:40"), "Phase 10", "scan_all() 매수 후보 스캔"),
                (S.get("evening_briefing", "19:00"), "Phase 10B", "[TG] 통합 데일리 리포트"),
                (S.get("eod_journal", "19:30"), "Phase 11", "업무일지"),
            ]),
        ]

        for title, entries in sections:
            print(f"  --- {title} ---")
            for t, name, desc in entries:
                print(f"  {t:>5}  {name:<10}  {desc}")
            print()

        # 안전장치
        safety = self.config.get("live_trading", {}).get("safety", {})
        print("  안전장치:")
        print(f"    STOP.signal:    {safety.get('stop_signal_file', 'STOP.signal')}")
        print(f"    reboot.trigger: {safety.get('reboot_trigger_file', 'reboot.trigger')}")
        print(f"    일일 손실 한도: {safety.get('max_daily_loss_pct', -0.03) * 100:.0f}%")
        print(f"    총 손실 한도:   {safety.get('max_total_loss_pct', -0.10) * 100:.0f}%")
        print()

        # 수급 모니터링
        sup = self.config.get("live_trading", {}).get("supply_monitor", {})
        print("  수급 모니터링:")
        print(f"    활성: {sup.get('enabled', False)}")
        print(f"    대상: {sup.get('snapshot_tickers', 'watchlist')}")
        alert = sup.get("alert_conditions", {})
        print(f"    방향전환 알림: {alert.get('direction_flip', False)}")
        print(f"    동시매도 임계: {alert.get('dual_sell_threshold', 100000):,}주")
        print("=" * 65)


def setup_logging():
    """로깅 설정 (콘솔 + 파일)"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console)

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
    parser = argparse.ArgumentParser(description="v5.0 일일 스케줄러")
    parser.add_argument("--dry-run", action="store_true", help="스케줄 확인만")
    parser.add_argument(
        "--run-now", type=str, metavar="PHASE",
        help="Phase 즉시 실행 (0~11, 3b, snap1~4, 10b)",
    )
    args = parser.parse_args()

    setup_logging()
    scheduler = DailyScheduler()

    if args.dry_run:
        scheduler.print_schedule()
    elif args.run_now is not None:
        phases = {
            "0": scheduler.phase_daily_reset,
            "1": scheduler.phase_us_close_collect,
            "2": scheduler.phase_macro_collect,
            "3": scheduler.phase_news_briefing,
            "3b": scheduler.phase_morning_briefing,
            "3c": scheduler.phase_etf_briefing,
            "4": scheduler.phase_trade_prep,
            "5": scheduler.phase_buy_execution,
            "6": scheduler.phase_intraday_monitor,
            "snap1": lambda: scheduler.phase_supply_snapshot(1),
            "snap2": lambda: scheduler.phase_supply_snapshot(2),
            "snap3": lambda: scheduler.phase_supply_snapshot(3),
            "snap4": lambda: scheduler.phase_supply_snapshot(4),
            "7": scheduler.phase_sell_execution,
            "8-0a": scheduler.phase_candle_archive,
            "8-0b": scheduler.phase_tick_snapshot,
            "8": scheduler.phase_close_data_collect,
            "8-1": scheduler.phase_close_data_collect,
            "8-3": scheduler.phase_parquet_update,
            "8-4": scheduler.phase_indicator_calc,
            "8-5": scheduler.phase_data_verify,
            "8-6": scheduler.phase_etf_signal,
            "8-7": scheduler.phase_kospi_update,
            "9": scheduler.phase_supply_final,
            "9.5": scheduler.phase_relay_check,
            "10": scheduler.phase_evening_scan,
            "10b": scheduler.phase_evening_briefing,
            "11": scheduler.phase_eod_journal,
        }
        key = args.run_now.lower()
        func = phases.get(key)
        if func:
            func()
        else:
            print(f"알 수 없는 Phase: {key}")
            print(f"사용 가능: {', '.join(sorted(phases.keys()))}")
    else:
        scheduler.run()
