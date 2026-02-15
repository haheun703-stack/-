"""
Phase 1: 장중 데이터 수집 오케스트레이터

3계층 스케줄링:
  - 1분: 보유 종목 현재가 틱 수집 (fetch_tick)
  - 5분: 5분봉 캔들 + 시장 컨텍스트 (KOSPI/KOSDAQ)
  - 10분: 투자자별 수급 + 업종별 시세

장 운영 시간: 08:30 ~ 15:30 KST
  - 08:30: 프리마켓 시작 (시장 지수 + 수급만)
  - 09:00: 정규장 시작 → 전체 수집 가동
  - 15:20: 마감 직전 경고 수집
  - 15:30: 정규장 마감 → 수집 종료
  - 15:35: 종가 확정 데이터 최종 수집 1회

사용법:
  collector = IntradayCollector(config, holdings=["005930", "000660"])
  collector.start()   # 스케줄러 시작
  collector.stop()    # 스케줄러 중지
"""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.use_cases.ports import IntradayDataPort, IntradayStorePort

logger = logging.getLogger(__name__)


class IntradayCollector:
    """장중 데이터 수집 스케줄러 + 오케스트레이터"""

    def __init__(
        self,
        config: dict,
        data_port: IntradayDataPort,
        store_port: IntradayStorePort,
        holdings: list[str] | None = None,
    ):
        self.config = config.get("intraday_monitor", {})
        self.data_port = data_port
        self.store_port = store_port
        self.holdings = holdings or []

        # 설정값
        self.tick_interval_sec = self.config.get("tick_interval_sec", 60)
        self.candle_interval_min = self.config.get("candle_interval_min", 5)
        self.flow_interval_min = self.config.get("flow_interval_min", 10)
        self.cleanup_days = self.config.get("cleanup_days", 30)

        # 스케줄러
        self.scheduler = BackgroundScheduler(
            timezone="Asia/Seoul",
            job_defaults={"coalesce": True, "max_instances": 1},
        )

        # 상태
        self._running = False
        self._collect_count = {"tick": 0, "candle": 0, "flow": 0, "market": 0, "sector": 0}

    # ──────────────────────────────────────────
    # 보유종목 관리
    # ──────────────────────────────────────────

    def update_holdings(self, tickers: list[str]) -> None:
        """보유종목 목록 갱신"""
        old = set(self.holdings)
        new = set(tickers)
        added = new - old
        removed = old - new
        self.holdings = tickers
        if added:
            logger.info("[수집] 종목 추가: %s", added)
        if removed:
            logger.info("[수집] 종목 제거: %s", removed)

    # ──────────────────────────────────────────
    # 스케줄러 시작/중지
    # ──────────────────────────────────────────

    def start(self) -> None:
        """스케줄러 시작 — 장 운영시간 동안 자동 수집"""
        if self._running:
            logger.warning("[수집] 이미 실행 중")
            return

        # 1분 단위: 보유종목 틱 수집 (09:00~15:30)
        self.scheduler.add_job(
            self._collect_ticks,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-14",
                minute="*",
                second="0",
            ),
            id="tick_collector",
            name="1분 틱 수집",
        )
        # 15시대는 15:00~15:30
        self.scheduler.add_job(
            self._collect_ticks,
            CronTrigger(
                day_of_week="mon-fri",
                hour="15",
                minute="0-30",
                second="0",
            ),
            id="tick_collector_close",
            name="1분 틱 수집 (마감)",
        )

        # 5분 단위: 5분봉 + 시장 컨텍스트
        self.scheduler.add_job(
            self._collect_candles_and_market,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-14",
                minute="*/5",
                second="10",
            ),
            id="candle_collector",
            name="5분봉 + 시장지수",
        )
        self.scheduler.add_job(
            self._collect_candles_and_market,
            CronTrigger(
                day_of_week="mon-fri",
                hour="15",
                minute="0,5,10,15,20,25,30",
                second="10",
            ),
            id="candle_collector_close",
            name="5분봉 + 시장지수 (마감)",
        )

        # 10분 단위: 투자자 수급 + 업종 시세
        self.scheduler.add_job(
            self._collect_flow_and_sector,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-14",
                minute="*/10",
                second="20",
            ),
            id="flow_collector",
            name="투자자수급 + 업종시세",
        )
        self.scheduler.add_job(
            self._collect_flow_and_sector,
            CronTrigger(
                day_of_week="mon-fri",
                hour="15",
                minute="0,10,20,30",
                second="20",
            ),
            id="flow_collector_close",
            name="투자자수급 + 업종시세 (마감)",
        )

        # 08:30 프리마켓: 시장 지수만
        self.scheduler.add_job(
            self._collect_premarket,
            CronTrigger(
                day_of_week="mon-fri",
                hour="8",
                minute="30",
            ),
            id="premarket",
            name="프리마켓 지수",
        )

        # 15:35 마감 확정 데이터
        self.scheduler.add_job(
            self._collect_closing,
            CronTrigger(
                day_of_week="mon-fri",
                hour="15",
                minute="35",
            ),
            id="closing",
            name="마감 확정 데이터",
        )

        # 매일 자정: 오래된 데이터 정리
        self.scheduler.add_job(
            self._cleanup_old,
            CronTrigger(
                day_of_week="mon-fri",
                hour="0",
                minute="5",
            ),
            id="cleanup",
            name="데이터 정리",
        )

        self.scheduler.start()
        self._running = True
        logger.info(
            "[수집] 스케줄러 시작 — 보유종목 %d개: %s",
            len(self.holdings), self.holdings,
        )

    def stop(self) -> None:
        """스케줄러 중지"""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[수집] 스케줄러 중지 — 수집 통계: %s", self._collect_count)

    @property
    def is_running(self) -> bool:
        return self._running

    # ──────────────────────────────────────────
    # 수집 작업 (스케줄러에서 호출)
    # ──────────────────────────────────────────

    def _collect_ticks(self) -> None:
        """1분 단위: 보유종목 현재가 틱 수집"""
        if not self.holdings:
            return

        ticks = self.data_port.fetch_ticks_batch(self.holdings)
        if ticks:
            self.store_port.save_ticks_batch(ticks)
            self._collect_count["tick"] += len(ticks)
            logger.debug("[수집] 틱 %d건 저장", len(ticks))

    def _collect_candles_and_market(self) -> None:
        """5분 단위: 5분봉 + 시장 컨텍스트"""
        # 5분봉 — 각 보유종목
        for ticker in self.holdings:
            candles = self.data_port.fetch_minute_candles(ticker, period=5)
            if candles:
                # 최신 1개만 저장 (이미 있으면 REPLACE)
                self.store_port.save_candle(candles[-1])
                self._collect_count["candle"] += 1

        # 시장 컨텍스트
        market = self.data_port.fetch_market_index()
        if market.get("kospi", 0) > 0:
            self.store_port.save_market_context(market)
            self._collect_count["market"] += 1
            logger.debug(
                "[수집] 시장 KOSPI=%.1f (%+.2f%%), KOSDAQ=%.1f (%+.2f%%)",
                market["kospi"], market["kospi_change_pct"],
                market["kosdaq"], market["kosdaq_change_pct"],
            )

    def _collect_flow_and_sector(self) -> None:
        """10분 단위: 투자자 수급 + 업종 시세"""
        # 투자자별 수급
        for ticker in self.holdings:
            flow = self.data_port.fetch_investor_flow(ticker)
            if flow.get("foreign_net_buy", 0) != 0 or flow.get("inst_net_buy", 0) != 0:
                self.store_port.save_investor_flow(flow)
                self._collect_count["flow"] += 1

        # 업종 시세
        sectors = self.data_port.fetch_sector_prices()
        for s in sectors:
            self.store_port.save_sector_price(s)
            self._collect_count["sector"] += 1

    def _collect_premarket(self) -> None:
        """08:30 프리마켓: 시장 지수만 수집"""
        logger.info("[수집] 프리마켓 시작")
        market = self.data_port.fetch_market_index()
        if market:
            self.store_port.save_market_context(market)

    def _collect_closing(self) -> None:
        """15:35 마감 확정: 종목별 최종 데이터"""
        logger.info("[수집] 마감 확정 데이터 수집")
        self._collect_ticks()
        self._collect_candles_and_market()
        self._collect_flow_and_sector()
        logger.info("[수집] 오늘 수집 통계: %s", self._collect_count)

    def _cleanup_old(self) -> None:
        """자정: 오래된 데이터 정리"""
        deleted = self.store_port.cleanup_old_data(self.cleanup_days)
        if deleted > 0:
            logger.info("[수집] %d건 오래된 데이터 정리 완료", deleted)

    # ──────────────────────────────────────────
    # 수동 실행 / 테스트용
    # ──────────────────────────────────────────

    def collect_once(self) -> dict:
        """
        1회성 전체 수집 (테스트/디버깅용).
        스케줄러 없이 즉시 실행.

        Returns:
            수집 결과 요약 dict
        """
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "holdings": self.holdings,
            "ticks": 0,
            "candles": 0,
            "flows": 0,
            "market": False,
            "sectors": 0,
        }

        if not self.holdings:
            logger.warning("[수집] 보유종목이 없습니다")
            return results

        # 틱 수집
        ticks = self.data_port.fetch_ticks_batch(self.holdings)
        if ticks:
            self.store_port.save_ticks_batch(ticks)
            results["ticks"] = len(ticks)

        # 5분봉
        for ticker in self.holdings:
            candles = self.data_port.fetch_minute_candles(ticker, period=5)
            if candles:
                self.store_port.save_candle(candles[-1])
                results["candles"] += 1

        # 투자자 수급
        for ticker in self.holdings:
            flow = self.data_port.fetch_investor_flow(ticker)
            self.store_port.save_investor_flow(flow)
            results["flows"] += 1

        # 시장 컨텍스트
        market = self.data_port.fetch_market_index()
        if market.get("kospi", 0) > 0:
            self.store_port.save_market_context(market)
            results["market"] = True

        # 업종 시세
        sectors = self.data_port.fetch_sector_prices()
        for s in sectors:
            self.store_port.save_sector_price(s)
        results["sectors"] = len(sectors)

        logger.info("[수집] 1회 수집 완료: %s", results)
        return results

    def get_status(self) -> dict:
        """현재 수집기 상태"""
        db_stats = {}
        if hasattr(self.store_port, "get_db_stats"):
            db_stats = self.store_port.get_db_stats()

        return {
            "running": self._running,
            "holdings": self.holdings,
            "holdings_count": len(self.holdings),
            "collect_count": dict(self._collect_count),
            "db_stats": db_stats,
            "schedule": {
                "tick": f"매 {self.tick_interval_sec}초",
                "candle": f"매 {self.candle_interval_min}분",
                "flow": f"매 {self.flow_interval_min}분",
            },
        }
