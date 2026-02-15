"""
⑮ GeometryEngine — Phase 3 통합 엔진

3D~5D 분석 모듈을 통합하여 단일 인터페이스로 제공.
Phase 2 SituationReporter와 연동하여 Claude API 프롬프트에 기하학 분석을 주입.

Phase 3-A (활성):
  ⑦ Confluence Scorer   — N팩터 조합 적중률
  ⑧ Cycle Clock         — 3주파수 사이클 위치
  ⑨ Divergence Detector — 팩터간 발산 감지

Phase 3-B (활성):
  ⑩ Phase Transition    — 상전이 5대 전조 감지
  ⑫ Neglect Scorer      — 군중 무관심 프록시
  Genesis Detector       — 포물선 시작점 통합 감지기
  Kelly Sizer            — Kelly Criterion 포지션 사이저

향후 활성 (Phase 3-C):
  ⑬ Strategy Health, ⑭ Scenario Engine
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .confluence_scorer import ConfluenceScorer
from .cycle_clock import CycleClock
from .divergence_detector import DivergenceDetector
from .genesis_detector import GenesisDetector
from .phase_transition import PhaseTransitionDetector
from .neglect_score import NeglectScorer

logger = logging.getLogger(__name__)


class GeometryEngine:
    """Phase 3 통합 기하학 분석 엔진 (3-A + 3-B)"""

    def __init__(self, config: dict | None = None, parquet_dir: str | Path | None = None):
        self.config = config or {}
        self.enabled = self.config.get("geometry", {}).get("enabled", True)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else Path("data/processed")

        # Phase 3-A: 3D 엔진
        self.confluence = ConfluenceScorer(config)
        self.cycle = CycleClock(config)
        self.divergence = DivergenceDetector(config)

        # Phase 3-B: 시작점 엔진
        self.genesis = GenesisDetector(config)
        self.phase_transition = PhaseTransitionDetector(config)
        self.neglect = NeglectScorer(config)

        # 종목별 적중률 DB 캐시
        self._confluence_cache: dict[str, bool] = {}  # ticker → DB 구축 여부

    # ─── 종목 분석 ───────────────────────────────

    def analyze(self, ticker: str, row: dict | None = None) -> dict:
        """
        단일 종목에 대한 3D 기하학 분석 수행.

        Parameters:
            ticker: 종목 코드
            row: 현재 시점의 지표 딕셔너리 (없으면 parquet에서 로드)

        Returns:
            {
                "confluence": {...},  # Confluence Scorer 결과
                "cycle": {...},       # Cycle Clock 결과
                "divergence": {...},  # Divergence Detector 결과
                "prompt_text": "...", # Claude용 통합 텍스트
            }
        """
        if not self.enabled:
            return self._empty_result()

        result = {}

        # parquet 로드
        df = self._load_parquet(ticker)

        # ⑦ Confluence Scorer
        try:
            if df is not None and len(df) > 30:
                self._ensure_confluence_db(ticker, df)
                current_row = row or (df.iloc[-1].to_dict() if df is not None else {})
                result["confluence"] = self.confluence.score_current(current_row)
            else:
                result["confluence"] = {"active_factors": [], "active_triples": [], "best_hit_rate": 0, "triple_count": 0}
        except Exception as e:
            logger.debug("Confluence 분석 실패 [%s]: %s", ticker, e)
            result["confluence"] = {"active_factors": [], "active_triples": [], "best_hit_rate": 0, "triple_count": 0}

        # ⑧ Cycle Clock
        try:
            if df is not None and len(df) >= 120:
                prices = df["close"].values
                result["cycle"] = self.cycle.get_clock_position(prices)
            else:
                result["cycle"] = CycleClock._empty_result("데이터 부족")
        except Exception as e:
            logger.debug("Cycle Clock 분석 실패 [%s]: %s", ticker, e)
            result["cycle"] = CycleClock._empty_result(str(e))

        # ⑨ Divergence Detector
        try:
            current_row = row or (df.iloc[-1].to_dict() if df is not None else {})
            result["divergence"] = self.divergence.detect_from_row(current_row)
        except Exception as e:
            logger.debug("Divergence 분석 실패 [%s]: %s", ticker, e)
            result["divergence"] = {"directions": {}, "divergences": [], "risk_count": 0, "opportunity_count": 0, "confirm_count": 0, "net_signal": 0}

        # ⑩ Phase Transition + Genesis Detector
        try:
            if df is not None and len(df) >= 60:
                prices = df["close"].values
                volumes = df["volume"].values if "volume" in df.columns else None
                current_row = row or (df.iloc[-1].to_dict() if df is not None else {})
                result["genesis"] = self.genesis.detect(
                    prices=prices,
                    volumes=volumes,
                    row=current_row,
                    df=df,
                )
            else:
                result["genesis"] = GenesisDetector._empty_result("데이터 부족")
        except Exception as e:
            logger.debug("Genesis 분석 실패 [%s]: %s", ticker, e)
            result["genesis"] = GenesisDetector._empty_result(str(e))

        # 통합 프롬프트 텍스트
        result["prompt_text"] = self._build_prompt_text(result)

        return result

    # ─── 포트폴리오 요약 ─────────────────────────

    def summarize_portfolio(self, stock_results: list[dict]) -> str:
        """
        전체 포트폴리오의 기하학 분석 요약.

        Parameters:
            stock_results: [{"ticker": "005930", "geometry": {...}}, ...]

        Returns:
            Claude 프롬프트용 요약 텍스트
        """
        lines = ["### 기하학 분석 요약 (3D+Genesis)"]

        # 최고 적중률 트리플
        best_triple = None
        best_rate = 0
        for sr in stock_results:
            geo = sr.get("geometry", {})
            conf = geo.get("confluence", {})
            if conf.get("best_hit_rate", 0) > best_rate:
                best_rate = conf["best_hit_rate"]
                best_triple = {
                    "ticker": sr.get("ticker", ""),
                    "name": sr.get("name", ""),
                    "rate": best_rate,
                    "triples": conf.get("active_triples", []),
                }

        if best_triple and best_rate >= 0.6:
            t = best_triple["triples"][0] if best_triple["triples"] else {}
            lines.append(
                f"  최고 교차 신호: [{best_triple['ticker']}] "
                f"{t.get('labels', '?')} → 적중률 {best_rate:.0%}"
            )

        # 사이클 요약
        buy_zone = []
        sell_zone = []
        for sr in stock_results:
            geo = sr.get("geometry", {})
            cycle_data = geo.get("cycle", {})
            mid = cycle_data.get("mid", {})
            mid_clock = mid.get("clock", 0)
            if 5 <= mid_clock <= 7:
                buy_zone.append(sr.get("ticker", ""))
            elif 10 <= mid_clock or mid_clock <= 1:
                sell_zone.append(sr.get("ticker", ""))

        if buy_zone:
            lines.append(f"  매수 구간 종목: {', '.join(buy_zone)}")
        if sell_zone:
            lines.append(f"  매도 구간 종목: {', '.join(sell_zone)}")

        # 발산 경고
        risk_stocks = []
        for sr in stock_results:
            geo = sr.get("geometry", {})
            div = geo.get("divergence", {})
            if div.get("risk_count", 0) >= 2:
                risk_stocks.append(sr.get("ticker", ""))

        if risk_stocks:
            lines.append(f"  다중 위험 발산: {', '.join(risk_stocks)}")

        # Genesis Alert (시작점 감지)
        genesis_alerts = []
        for sr in stock_results:
            geo = sr.get("geometry", {})
            gen = geo.get("genesis", {})
            if gen.get("genesis_alert"):
                genesis_alerts.append(
                    f"{sr.get('ticker', '')} (Class {gen.get('signal_class', '?')},"
                    f" 점수 {gen.get('composite_score', 0):.2f})"
                )
        if genesis_alerts:
            lines.append(f"  GENESIS ALERT: {', '.join(genesis_alerts)}")

        if len(lines) == 1:
            lines.append("  특이 사항 없음")

        return "\n".join(lines)

    # ─── 내부 헬퍼 ───────────────────────────────

    def _load_parquet(self, ticker: str) -> pd.DataFrame | None:
        """종목 parquet 파일 로드"""
        pq_path = self.parquet_dir / f"{ticker}.parquet"
        if not pq_path.exists():
            return None
        try:
            df = pd.read_parquet(pq_path)
            if "close" not in df.columns:
                return None
            return df
        except Exception as e:
            logger.debug("Parquet 로드 실패 [%s]: %s", ticker, e)
            return None

    def _ensure_confluence_db(self, ticker: str, df: pd.DataFrame):
        """종목별 Confluence 적중률 DB가 없으면 구축"""
        if ticker not in self._confluence_cache:
            # macd_histogram_prev 보장
            if "macd_histogram" in df.columns and "macd_histogram_prev" not in df.columns:
                df = df.copy()
                df["macd_histogram_prev"] = df["macd_histogram"].shift(1)
            self.confluence.build_hit_rate_db(df)
            self._confluence_cache[ticker] = True

    def _build_prompt_text(self, result: dict) -> str:
        """Phase 3-A/B 모듈 결과를 통합 프롬프트로"""
        parts = []
        # Phase 3-A
        parts.append(ConfluenceScorer.to_prompt_text(result.get("confluence", {})))
        parts.append(CycleClock.to_prompt_text(result.get("cycle", {})))
        parts.append(DivergenceDetector.to_prompt_text(result.get("divergence", {})))
        # Phase 3-B
        genesis = result.get("genesis", {})
        if genesis.get("genesis_alert") or genesis.get("signal_class", "NONE") != "NONE":
            parts.append(GenesisDetector.to_prompt_text(genesis))
        return "\n".join(parts)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "confluence": {"active_factors": [], "active_triples": [], "best_hit_rate": 0, "triple_count": 0},
            "cycle": CycleClock._empty_result("비활성"),
            "divergence": {"directions": {}, "divergences": [], "risk_count": 0, "opportunity_count": 0, "confirm_count": 0, "net_signal": 0},
            "genesis": GenesisDetector._empty_result("비활성"),
            "prompt_text": "[기하학 분석 비활성]",
        }
