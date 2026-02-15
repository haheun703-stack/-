"""
Phase 2: 상황보고서 생성기 (Situation Report Generator)

Phase 1에서 수집한 장중 데이터를 집계하여 Claude AI 입력용 구조화된 보고서를 생성한다.

생성 트리거:
  - 정기: 30분 간격 (설정 가능)
  - 긴급: 보유종목 3%+ 급변 시 즉시
  - 마감: 15:35 종가 확정 후

보고서 구성:
  1. 핵심 경고 / 기회 요약
  2. 시장 환경 (KOSPI/KOSDAQ/업종)
  3. 종목별 상세 현황:
     - 가격 + 장중 추세
     - 수급 (외인/기관/개인)
     - v8.1 기술지표 요약
     - v8.1 Gate/Score/Trigger 상태
     - 경고 플래그
  4. 포지션 정보 (보유 중인 경우)

의존성:
  - IntradayStorePort: Phase 1 수집 데이터 조회
  - data/positions.json: 보유종목 포지션 정보
  - data/parquet/*.parquet: v8.1 일봉 지표 (선택적)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.entities.intraday_models import (
    SituationReport,
    StockSituation,
)
from src.use_cases.ports import IntradayStorePort

logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("data/positions.json")
PARQUET_DIR = Path("data/parquet")


class SituationReporter:
    """Phase 2: 상황보고서 생성 엔진"""

    def __init__(
        self,
        config: dict,
        store_port: IntradayStorePort,
        positions_file: Path | str | None = None,
        parquet_dir: Path | str | None = None,
    ):
        self.config = config.get("intraday_monitor", {})
        self.report_cfg = self.config.get("situation_report", {})
        self.store_port = store_port

        self.positions_file = Path(positions_file) if positions_file else POSITIONS_FILE
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PARQUET_DIR

        # 설정
        self.emergency_threshold = self.report_cfg.get("emergency_threshold", 0.03)
        self.interval_min = self.report_cfg.get("interval_min", 30)

        # 이전 보고서 캐시 (긴급 판정용)
        self._last_prices: dict[str, int] = {}
        self._last_report_time: float = 0.0

        # 종목명 캐시
        self._name_cache: dict[str, str] = {}

    # ──────────────────────────────────────────
    # 메인 생성 메서드
    # ──────────────────────────────────────────

    def generate(
        self,
        holdings: list[str] | None = None,
        report_type: str = "regular",
    ) -> SituationReport:
        """
        상황보고서 생성.

        Args:
            holdings: 보유종목 리스트 (None이면 positions.json에서 로드)
            report_type: "regular" / "emergency" / "closing"

        Returns:
            SituationReport 엔티티
        """
        start_ms = time.time()

        if holdings is None:
            holdings = self._load_holdings()

        # 포지션 정보 로드
        positions = self._load_positions()

        # 시장 환경 조회
        market = self.store_port.get_latest_market_context()

        # 업종 시세 조회
        sectors = self.store_port.get_today_sector_prices()

        # 종목별 상황 생성
        stock_situations = []
        for ticker in holdings:
            sit = self._build_stock_situation(ticker, positions)
            stock_situations.append(sit)

        # 보고서 조립
        report = SituationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type=report_type,
            holdings_count=len(holdings),
        )

        # 시장 환경
        if market:
            report.kospi = market.get("kospi", 0.0)
            report.kospi_change_pct = market.get("kospi_change_pct", 0.0)
            report.kosdaq = market.get("kosdaq", 0.0)
            report.kosdaq_change_pct = market.get("kosdaq_change_pct", 0.0)
            report.market_regime = self._classify_market_regime(market)

        # 업종 분석
        if sectors:
            sorted_sectors = sorted(sectors, key=lambda s: s.get("change_pct", 0), reverse=True)
            report.top_sectors = [
                {"name": s["sector_name"], "change_pct": s.get("change_pct", 0)}
                for s in sorted_sectors[:3]
            ]
            report.bottom_sectors = [
                {"name": s["sector_name"], "change_pct": s.get("change_pct", 0)}
                for s in sorted_sectors[-3:]
            ]

        # 종목 상황 (dict 리스트로 변환)
        report.stocks = [s.to_dict() for s in stock_situations]

        # 경고 및 기회 요약 생성
        report.summary_alerts = self._generate_alerts(stock_situations, market)
        report.opportunities = self._generate_opportunities(stock_situations)

        # 데이터 신선도 계산
        report.data_freshness_sec = self._calc_data_freshness(holdings)

        # 생성 시간
        report.generation_ms = int((time.time() - start_ms) * 1000)

        # 가격 캐시 업데이트 (긴급 보고 판정용)
        for s in stock_situations:
            if s.current_price > 0:
                self._last_prices[s.ticker] = s.current_price
        self._last_report_time = time.time()

        logger.info(
            "[상황보고] %s 보고서 생성 — 종목 %d개, 경고 %d건, 기회 %d건 (%dms)",
            report_type,
            len(holdings),
            len(report.summary_alerts),
            len(report.opportunities),
            report.generation_ms,
        )

        return report

    def check_emergency(self, holdings: list[str] | None = None) -> str | None:
        """
        긴급 보고 필요 여부 판정.

        Returns:
            긴급 사유 문자열 또는 None (정상)
        """
        if not self._last_prices:
            return None

        if holdings is None:
            holdings = self._load_holdings()

        for ticker in holdings:
            ticks = self.store_port.get_recent_ticks(ticker, minutes=2)
            if not ticks:
                continue

            latest = ticks[-1]
            current_price = latest.get("current_price", 0)
            prev_price = self._last_prices.get(ticker, 0)

            if prev_price <= 0 or current_price <= 0:
                continue

            change = abs(current_price - prev_price) / prev_price
            if change >= self.emergency_threshold:
                direction = "급등" if current_price > prev_price else "급락"
                return (
                    f"{ticker} {direction}: "
                    f"{prev_price:,}→{current_price:,} "
                    f"({change * 100:+.1f}%)"
                )

        return None

    # ──────────────────────────────────────────
    # 종목별 상황 생성
    # ──────────────────────────────────────────

    def _build_stock_situation(
        self,
        ticker: str,
        positions: dict,
    ) -> StockSituation:
        """단일 종목의 상황 스냅샷 생성"""
        sit = StockSituation(ticker=ticker)
        sit.name = self._get_stock_name(ticker)

        # 1. 최신 틱 데이터
        ticks = self.store_port.get_recent_ticks(ticker, minutes=5)
        if ticks:
            latest = ticks[-1]
            sit.current_price = latest.get("current_price", 0)
            sit.change_pct = latest.get("change_pct", 0.0)
            sit.open_price = latest.get("open_price", 0)
            sit.high_price = latest.get("high_price", 0)
            sit.low_price = latest.get("low_price", 0)
            sit.volume = latest.get("cum_volume", latest.get("volume", 0))

        # 2. 5분봉 → 장중 추세 분석
        candles = self.store_port.get_today_candles(ticker)
        if candles:
            sit.intraday_trend = self._analyze_intraday_trend(candles)
            sit.intraday_high = max(c.get("high", 0) for c in candles)
            sit.intraday_low = min(c.get("low", 0) for c in candles if c.get("low", 0) > 0)
            if sit.open_price > 0:
                sit.price_from_open_pct = (
                    (sit.current_price - sit.open_price) / sit.open_price * 100
                )

        # 3. 투자자 수급
        flows = self.store_port.get_today_investor_flow(ticker)
        if flows:
            latest_flow = flows[-1]
            sit.foreign_net_buy = latest_flow.get("foreign_net_buy", 0)
            sit.inst_net_buy = latest_flow.get("inst_net_buy", 0)
            sit.individual_net_buy = latest_flow.get("individual_net_buy", 0)
            sit.flow_direction = self._classify_flow(
                sit.foreign_net_buy, sit.inst_net_buy
            )

        # 4. 포지션 정보
        pos = positions.get(ticker)
        if pos:
            sit.entry_price = pos.get("entry_price", 0)
            sit.shares = pos.get("shares", 0)
            sit.hold_days = pos.get("hold_days", 0)
            sit.stop_loss = pos.get("stop_loss", 0)
            sit.target_price = pos.get("target_price", 0)
            if sit.entry_price > 0 and sit.current_price > 0:
                sit.pnl_pct = (
                    (sit.current_price - sit.entry_price) / sit.entry_price * 100
                )

        # 5. v8.1 일봉 지표 (parquet에서 로드)
        self._fill_daily_indicators(sit, ticker)

        # 6. 경고 생성
        sit.alerts = self._detect_stock_alerts(sit)

        return sit

    # ──────────────────────────────────────────
    # 장중 추세 분석
    # ──────────────────────────────────────────

    @staticmethod
    def _analyze_intraday_trend(candles: list[dict]) -> str:
        """5분봉 기반 장중 추세 판별"""
        if len(candles) < 3:
            return "데이터부족"

        closes = [c.get("close", 0) for c in candles if c.get("close", 0) > 0]
        if len(closes) < 3:
            return "데이터부족"

        n = len(closes)
        first_third = closes[: n // 3]
        mid_third = closes[n // 3: 2 * n // 3]
        last_third = closes[2 * n // 3:]

        avg_first = sum(first_third) / len(first_third) if first_third else 0
        avg_mid = sum(mid_third) / len(mid_third) if mid_third else 0
        avg_last = sum(last_third) / len(last_third) if last_third else 0

        if avg_first == 0:
            return "데이터부족"

        # 전체 변화율
        total_change = (avg_last - avg_first) / avg_first

        if avg_mid < avg_first and avg_last > avg_mid:
            return "V반등"
        elif avg_mid > avg_first and avg_last < avg_mid:
            return "역V"
        elif total_change > 0.005:
            return "상승"
        elif total_change < -0.005:
            return "하락"
        else:
            return "횡보"

    # ──────────────────────────────────────────
    # 수급 분류
    # ──────────────────────────────────────────

    @staticmethod
    def _classify_flow(foreign_net: int, inst_net: int) -> str:
        """외인/기관 수급 방향 텍스트"""
        f_dir = "매수" if foreign_net > 0 else "매도"
        i_dir = "매수" if inst_net > 0 else "매도"

        if foreign_net > 0 and inst_net > 0:
            return "외인+기관 동반매수"
        elif foreign_net < 0 and inst_net < 0:
            return "외인+기관 동반매도"
        else:
            return f"외인 {f_dir}, 기관 {i_dir}"

    # ──────────────────────────────────────────
    # v8.1 일봉 지표 채우기
    # ──────────────────────────────────────────

    def _fill_daily_indicators(self, sit: StockSituation, ticker: str) -> None:
        """parquet 파일에서 최신 일봉 기술지표 로드"""
        parquet_path = self.parquet_dir / f"{ticker}.parquet"
        if not parquet_path.exists():
            return

        try:
            df = pd.read_parquet(parquet_path)
            if df.empty:
                return

            row = df.iloc[-1]

            # 기본 지표
            sit.rsi_14 = float(row.get("rsi_14", 0) or 0)
            sit.adx_14 = float(row.get("adx_14", 0) or 0)
            sit.macd_histogram = float(row.get("macd_histogram", 0) or 0)
            sit.smart_z = float(row.get("smart_z", 0) or 0)
            sit.ou_z = float(row.get("ou_z", 0) or 0)

            # 볼린저 밴드 위치 (0=하단, 1=상단)
            bb_upper = float(row.get("bb_upper", 0) or 0)
            bb_lower = float(row.get("bb_lower", 0) or 0)
            close = float(row.get("close", 0) or 0)
            if bb_upper > bb_lower > 0 and close > 0:
                sit.bb_position = round(
                    (close - bb_lower) / (bb_upper - bb_lower), 3
                )

            # 거래량 비율
            vol = float(row.get("volume", 0) or 0)
            vol_ma20 = float(row.get("volume_ma20", 0) or 0)
            if vol_ma20 > 0:
                sit.volume_ratio = round(vol / vol_ma20, 2)

            # TRIX 시그널 상태
            trix_val = float(row.get("trix", 0) or 0)
            trix_sig = float(row.get("trix_signal", 0) or 0)
            trix_prev = float(row.get("trix_prev", trix_val) or trix_val)
            trix_sig_prev = float(row.get("trix_signal_prev", trix_sig) or trix_sig)

            if trix_prev <= trix_sig_prev and trix_val > trix_sig:
                sit.trix_signal = "골든크로스"
            elif trix_prev >= trix_sig_prev and trix_val < trix_sig:
                sit.trix_signal = "데드크로스"
            elif trix_val > trix_sig:
                sit.trix_signal = "매수세"
            elif trix_val < trix_sig:
                sit.trix_signal = "매도세"
            else:
                sit.trix_signal = "중립"

            # v8 Gate/Score 결과 (parquet에 없으면 실시간 계산)
            self._fill_v8_summary(sit, row)

        except Exception as e:
            logger.warning("[상황보고] %s 일봉 지표 로드 실패: %s", ticker, e)

    def _fill_v8_summary(self, sit: StockSituation, row: pd.Series) -> None:
        """v8.1 Gate/Score/Trigger 요약을 채우기"""
        try:
            # v8 결과가 parquet에 기록되어 있을 수 있음
            if "v8_grade" in row.index:
                sit.v8_grade = str(row.get("v8_grade", ""))
                sit.v8_total_score = float(row.get("v8_total_score", 0) or 0)
                sit.v8_gate_passed = bool(row.get("v8_gate_passed", False))
                sit.v8_active_triggers = str(row.get("v8_active_triggers", "") or "")
                return

            # parquet에 v8 결과가 없으면 지표 기반 간이 판정
            import yaml

            from src.v8_gates import GateEngine
            from src.v8_scorers import ScoringEngine
            from src.v8_triggers import TriggerEngine
            cfg_path = Path("config/settings.yaml")
            with open(cfg_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            gate_engine = GateEngine(config)
            score_engine = ScoringEngine(config)
            trigger_engine = TriggerEngine(config)

            gate_passed, gate_reasons = gate_engine.run_all_gates(row)
            sit.v8_gate_passed = gate_passed

            if gate_passed:
                grade_result = score_engine.score_all(row)
                sit.v8_grade = grade_result.grade
                sit.v8_total_score = grade_result.total_score

                if grade_result.grade in ("A", "B"):
                    trigger_results = trigger_engine.check_all_triggers(row)
                    fired = [t["name"] for t in trigger_results if t["fired"]]
                    sit.v8_active_triggers = ", ".join(fired)

        except Exception as e:
            logger.debug("[상황보고] %s v8 요약 생성 실패 (선택적): %s", sit.ticker, e)

    # ──────────────────────────────────────────
    # 경고 / 기회 생성
    # ──────────────────────────────────────────

    def _detect_stock_alerts(self, sit: StockSituation) -> list[str]:
        """개별 종목 경고 플래그 생성"""
        alerts = []

        # 급등/급락 (3%+)
        if abs(sit.change_pct) >= 3.0:
            direction = "급등" if sit.change_pct > 0 else "급락"
            alerts.append(f"{direction} {sit.change_pct:+.1f}%")

        # 손절선 접근 (포지션 보유 중)
        if sit.stop_loss > 0 and sit.current_price > 0:
            buffer = (sit.current_price - sit.stop_loss) / sit.current_price
            if buffer < 0:
                alerts.append(f"손절선 이탈! ({sit.current_price:,} < {sit.stop_loss:,})")
            elif buffer < 0.02:
                alerts.append(f"손절선 임박 (여유 {buffer * 100:.1f}%)")

        # 목표가 접근
        if sit.target_price > 0 and sit.current_price > 0:
            distance = (sit.target_price - sit.current_price) / sit.current_price
            if distance < 0:
                alerts.append(f"목표가 돌파! ({sit.current_price:,} > {sit.target_price:,})")
            elif distance < 0.03:
                alerts.append(f"목표가 접근 ({distance * 100:.1f}% 남음)")

        # 거래량 폭발
        if sit.volume_ratio >= 3.0:
            alerts.append(f"거래량 폭발 ({sit.volume_ratio:.1f}x)")
        elif sit.volume_ratio >= 2.0:
            alerts.append(f"거래량 급증 ({sit.volume_ratio:.1f}x)")

        # RSI 극단
        if sit.rsi_14 >= 70:
            alerts.append(f"RSI 과매수 ({sit.rsi_14:.0f})")
        elif sit.rsi_14 > 0 and sit.rsi_14 <= 30:
            alerts.append(f"RSI 과매도 ({sit.rsi_14:.0f})")

        # 외인+기관 동반 매도
        if sit.foreign_net_buy < -1000 and sit.inst_net_buy < -1000:
            alerts.append("외인+기관 동반매도 주의")

        # OU 극단
        if sit.ou_z <= -2.0:
            alerts.append(f"OU 극과매도 (z={sit.ou_z:.2f})")
        elif sit.ou_z >= 2.0:
            alerts.append(f"OU 과매수 (z={sit.ou_z:.2f})")

        return alerts

    def _generate_alerts(
        self,
        stocks: list[StockSituation],
        market: dict | None,
    ) -> list[str]:
        """전체 포트폴리오 레벨 핵심 경고"""
        alerts = []

        # 시장 급변
        if market:
            kospi_chg = market.get("kospi_change_pct", 0)
            kosdaq_chg = market.get("kosdaq_change_pct", 0)
            if abs(kospi_chg) >= 2.0:
                direction = "급등" if kospi_chg > 0 else "급락"
                alerts.append(f"KOSPI {direction} {kospi_chg:+.2f}%")
            if abs(kosdaq_chg) >= 2.0:
                direction = "급등" if kosdaq_chg > 0 else "급락"
                alerts.append(f"KOSDAQ {direction} {kosdaq_chg:+.2f}%")

        # 종목별 심각 경고
        for s in stocks:
            if abs(s.change_pct) >= 3.0:
                direction = "급등" if s.change_pct > 0 else "급락"
                alerts.append(f"{s.name}({s.ticker}) {direction} {s.change_pct:+.1f}%")

            # 손절선 이탈
            if s.stop_loss > 0 and s.current_price > 0 and s.current_price < s.stop_loss:
                alerts.append(
                    f"{s.name} 손절선 이탈 "
                    f"({s.current_price:,} < {s.stop_loss:,})"
                )

        return alerts

    def _generate_opportunities(
        self,
        stocks: list[StockSituation],
    ) -> list[str]:
        """기회 포착 요약"""
        opps = []

        for s in stocks:
            # OU 반등 신호
            if s.ou_z <= -1.5 and s.rsi_14 < 45:
                opps.append(
                    f"{s.name}({s.ticker}) OU 반등 신호 "
                    f"(z={s.ou_z:.2f}, RSI={s.rsi_14:.0f})"
                )

            # 외인+기관 동반 매수 + 기술적 매수 구간
            if s.foreign_net_buy > 1000 and s.inst_net_buy > 1000 and s.rsi_14 < 55:
                opps.append(
                    f"{s.name}({s.ticker}) 외인+기관 동반매수 "
                    f"(외인 +{s.foreign_net_buy:,}, 기관 +{s.inst_net_buy:,})"
                )

            # TRIX 골든크로스
            if s.trix_signal == "골든크로스":
                opps.append(f"{s.name}({s.ticker}) TRIX 골든크로스 발생")

            # v8 A등급 트리거 발동
            if s.v8_grade == "A" and s.v8_active_triggers:
                opps.append(
                    f"{s.name}({s.ticker}) v8 A등급 트리거={s.v8_active_triggers}"
                )

        return opps

    # ──────────────────────────────────────────
    # 시장 레짐 분류
    # ──────────────────────────────────────────

    @staticmethod
    def _classify_market_regime(market: dict) -> str:
        """시장 분위기 간이 분류"""
        kospi_chg = market.get("kospi_change_pct", 0)
        kosdaq_chg = market.get("kosdaq_change_pct", 0)
        avg_chg = (kospi_chg + kosdaq_chg) / 2

        if avg_chg >= 1.5:
            return "강세"
        elif avg_chg >= 0.3:
            return "소폭상승"
        elif avg_chg >= -0.3:
            return "횡보"
        elif avg_chg >= -1.5:
            return "소폭하락"
        else:
            return "약세"

    # ──────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────

    def _load_holdings(self) -> list[str]:
        """positions.json에서 보유종목 목록 로드"""
        if not self.positions_file.exists():
            return []
        try:
            data = json.loads(self.positions_file.read_text(encoding="utf-8"))
            return [p["ticker"] for p in data if p.get("shares", 0) > 0]
        except Exception as e:
            logger.warning("[상황보고] positions.json 로드 실패: %s", e)
            return []

    def _load_positions(self) -> dict:
        """positions.json에서 포지션 상세 정보 로드 (ticker → dict)"""
        if not self.positions_file.exists():
            return {}
        try:
            data = json.loads(self.positions_file.read_text(encoding="utf-8"))
            result = {}
            for p in data:
                if p.get("shares", 0) > 0:
                    result[p["ticker"]] = p
            return result
        except Exception as e:
            logger.warning("[상황보고] positions.json 로드 실패: %s", e)
            return {}

    def _get_stock_name(self, ticker: str) -> str:
        """종목 코드 → 종목명 변환 (캐시 활용)"""
        if ticker in self._name_cache:
            return self._name_cache[ticker]

        # positions.json에서 이름 찾기
        if self.positions_file.exists():
            try:
                data = json.loads(self.positions_file.read_text(encoding="utf-8"))
                for p in data:
                    if p.get("ticker") == ticker:
                        name = p.get("name", p.get("stock_name", ticker))
                        self._name_cache[ticker] = name
                        return name
            except Exception:
                pass

        # parquet 파일명에 이름 없으므로 ticker 그대로 반환
        self._name_cache[ticker] = ticker
        return ticker

    def _calc_data_freshness(self, holdings: list[str]) -> int:
        """최신 틱 데이터의 나이 (초) — 0이면 매우 신선"""
        if not holdings:
            return -1

        now = datetime.now()
        min_age = float("inf")

        for ticker in holdings:
            ticks = self.store_port.get_recent_ticks(ticker, minutes=5)
            if ticks:
                last_ts = ticks[-1].get("timestamp", "")
                try:
                    tick_time = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        tick_time = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:00")
                    except ValueError:
                        continue
                age = (now - tick_time).total_seconds()
                min_age = min(min_age, age)

        return int(min_age) if min_age < float("inf") else -1
