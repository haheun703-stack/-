"""
장전 포트폴리오 5D/6D 분석 리포트 생성기

08:25 스케줄: 보유 포지션을 5D/6D 프레임워크로 분석하여 HTML 리포트 생성.
- 1D: 가격 위치 (52주 고점 대비)
- 2D: 밸류에이션 (목표가 괴리, PER)
- 3D: 멀티팩터 (Confluence Scorer)
- 4D: 타이밍 (Cycle Clock 나선시계)
- 5D: 메타게임 (Neglect Scorer 군중역발상)
- 6D: 게임 설계 (GameAnalyst 에이전트 — Claude API)

의존성:
  - data/positions.json (보유 포지션)
  - data/processed/{ticker}.parquet (기술적 지표)
  - src/geometry/engine.py (3D~5D 분석)
  - src/agents/game_analyst.py (6D 분석 — Claude API)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.entities.game_models import (
    DimensionScores,
    PortfolioReportData,
    StockReportData,
)
from src.geometry.engine import GeometryEngine
from src.use_cases.position_tracker import PositionTracker

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/daily")
TEMPLATES_DIR = Path("templates")


def _score_color(score: float) -> str:
    """점수에 따른 CSS 컬러 클래스"""
    if score >= 60:
        return "green"
    if score >= 50:
        return "yellow"
    if score >= 40:
        return "orange"
    return "red"


def _forecast_emoji(forecast_class: str) -> str:
    if forecast_class == "up":
        return "📈"
    if forecast_class == "down":
        return "📉"
    return "➡️"


def _action_emoji(color: str) -> str:
    return {"green": "✅", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(color, "⚪")


def _rank_emoji(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}️⃣")


class PortfolioReporter:
    """5D/6D 포트폴리오 분석 리포트 생성기"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.geometry = GeometryEngine(config)
        self.parquet_dir = Path("data/processed")

    def generate(self) -> Path | None:
        """
        포트폴리오 분석 리포트 생성 + HTML 저장.

        Returns:
            저장된 HTML 파일 경로 (포지션이 없으면 None)
        """
        # 1. 포지션 로드
        tracker = PositionTracker(self.config)
        if not tracker.positions:
            logger.info("[리포트] 보유 포지션 없음 — 리포트 생략")
            return None

        today = datetime.now().strftime("%Y-%m-%d")

        # 2. 종목별 1D~5D 분석
        stock_data_list = []
        for pos in tracker.positions:
            stock = self._analyze_stock(pos, today)
            stock_data_list.append(stock)

        # 3. 6D 게임 분석 (Claude API)
        self._run_6d_analysis(stock_data_list)

        # 4. 종합 점수 계산 + 정렬
        for stock in stock_data_list:
            if stock.dimensions:
                stock.dimensions.calc_totals()

        stock_data_list.sort(
            key=lambda s: s.dimensions.total_6d if s.dimensions else 0,
            reverse=True,
        )
        for i, stock in enumerate(stock_data_list, 1):
            stock.rank = i

        # 5. 행동 판단
        for stock in stock_data_list:
            self._determine_action(stock)

        # 6. 리포트 데이터 조립
        report = self._build_report_data(stock_data_list, today)

        # 7. HTML 렌더링
        html = self._render_html(report)

        # 8. 파일 저장
        save_dir = REPORTS_DIR / today
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "morning_report.html"
        save_path.write_text(html, encoding="utf-8")
        logger.info("[리포트] 장전 리포트 저장: %s", save_path)

        return save_path

    # ──────────────────────────────────────────
    # 1D~5D 분석
    # ──────────────────────────────────────────

    def _analyze_stock(self, pos, today: str) -> StockReportData:
        """개별 종목의 1D~5D 차원 점수 계산"""
        ticker = pos.ticker
        df = self._load_parquet(ticker)

        dims = DimensionScores(ticker=ticker, name=pos.name)

        if df is not None and not df.empty:
            row = df.iloc[-1]

            # 1D: 가격 위치
            high_252 = row.get("high_252", 0) or row.get("high_252d", 0)
            close = row.get("close", 0)
            if high_252 and high_252 > 0:
                dims.d1_price_pct = (close / high_252) * 100
            else:
                dims.d1_price_pct = 50.0

            # 2D: 밸류에이션 (목표가, PER — 외부 데이터 필요, 기본값 사용)
            dims.d2_target_gap_pct = 0.0
            dims.d2_per = 0.0

            # 3D: 멀티팩터 (Confluence Scorer)
            try:
                geo_result = self.geometry.analyze(ticker, row.to_dict())
                conf = geo_result.get("confluence", {})
                best_rate = conf.get("best_hit_rate", 0)
                dims.d3_multifactor = best_rate * 100

                # 4D: 타이밍 (Cycle Clock)
                cycle = geo_result.get("cycle", {})
                mid = cycle.get("mid", {})
                dims.d4_clock_hour = mid.get("clock", 6.0)
                # 시계 위치를 점수로 변환 (5~7시가 최적)
                clock = dims.d4_clock_hour
                if 5 <= clock <= 7:
                    dims.d4_timing_score = 85.0
                elif 7 < clock <= 9:
                    dims.d4_timing_score = 70.0
                elif 3 <= clock < 5:
                    dims.d4_timing_score = 60.0
                elif 9 < clock <= 10:
                    dims.d4_timing_score = 45.0
                else:
                    dims.d4_timing_score = 30.0

                # 5D: 메타게임 (Neglect Scorer)
                # neglect_score는 genesis 내에서 계산됨
                genesis = geo_result.get("genesis", {})
                neglect = genesis.get("neglect", {})
                neglect_composite = neglect.get("composite", 0.5)
                dims.d5_neglect_pct = (1 - neglect_composite) * 100  # 관심도
                dims.d5_metagame_score = neglect_composite * 100  # 높을수록 기회
            except Exception as e:
                logger.debug("[리포트] %s Geometry 분석 실패: %s", ticker, e)
                dims.d3_multifactor = 40.0
                dims.d4_clock_hour = 6.0
                dims.d4_timing_score = 50.0
                dims.d5_neglect_pct = 50.0
                dims.d5_metagame_score = 50.0

        # 보유일 계산
        try:
            hold_days = (
                datetime.strptime(today, "%Y-%m-%d")
                - datetime.strptime(pos.entry_date, "%Y-%m-%d")
            ).days
        except Exception:
            hold_days = 0

        return StockReportData(
            rank=0,
            ticker=ticker,
            name=pos.name,
            current_price=int(pos.current_price),
            return_pct=round(pos.unrealized_pnl_pct, 2),
            shares=pos.shares,
            hold_days=hold_days,
            investment=int(pos.entry_price * pos.shares),
            dimensions=dims,
        )

    # ──────────────────────────────────────────
    # 6D 분석 (Claude API)
    # ──────────────────────────────────────────

    def _run_6d_analysis(self, stocks: list[StockReportData]) -> None:
        """6D 게임 분석을 Claude API로 실행 (deprecated — archive 이동됨)"""
        try:
            from scripts.archive.deprecated_agents.game_analyst import GameAnalystAgent

            agent = GameAnalystAgent()
            contexts = []
            for s in stocks:
                ctx = {
                    "ticker": s.ticker,
                    "name": s.name,
                    "current_price": s.current_price,
                    "return_pct": s.return_pct,
                    "hold_days": s.hold_days,
                    "pct_52w_high": s.dimensions.d1_price_pct if s.dimensions else 50,
                    "multifactor_score": s.dimensions.d3_multifactor if s.dimensions else 40,
                    "clock_hour": s.dimensions.d4_clock_hour if s.dimensions else 6,
                    "timing_score": s.dimensions.d4_timing_score if s.dimensions else 50,
                    "crowd_pct": s.dimensions.d5_neglect_pct if s.dimensions else 50,
                    "neglect_score": s.dimensions.d5_metagame_score if s.dimensions else 50,
                }
                contexts.append(ctx)

            # asyncio 실행 (기존 이벤트루프 존재 시 안전 처리)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        asyncio.run, agent.batch_analyze(contexts)
                    ).result()
            else:
                results = asyncio.run(agent.batch_analyze(contexts))

            for stock, game in zip(stocks, results):
                stock.game_analysis = game
                if stock.dimensions:
                    stock.dimensions.d6_game_score = game.game_score
                    stock.dimensions.d6_trap_pct = game.trap_risk_pct
        except Exception as e:
            logger.error("[리포트] 6D 분석 실패: %s", e)
            # 실패 시 기본값
            for s in stocks:
                if s.dimensions:
                    s.dimensions.d6_game_score = 50.0
                    s.dimensions.d6_trap_pct = 50.0

    # ──────────────────────────────────────────
    # 행동 판단
    # ──────────────────────────────────────────

    def _determine_action(self, stock: StockReportData) -> None:
        """차원 점수 기반 행동 판단"""
        d = stock.dimensions
        if not d:
            stock.action = "분석 불가"
            stock.action_color = "orange"
            stock.forecast = "알 수 없음"
            stock.forecast_class = "flat"
            return

        score_6d = d.total_6d

        # 행동 판단
        if d.d6_trap_pct >= 50 or d.d4_timing_score < 35:
            stock.action = "축소 검토" if d.d6_trap_pct < 60 else "반익 실현"
            stock.action_color = "red" if d.d6_trap_pct >= 60 else "orange"
            stock.action_reason = self._build_action_reason(d, "caution")
        elif score_6d >= 55:
            stock.action = "보유"
            stock.action_color = "green"
            stock.action_reason = self._build_action_reason(d, "hold")
        elif score_6d >= 45:
            stock.action = "보유/관망"
            stock.action_color = "yellow"
            stock.action_reason = self._build_action_reason(d, "watch")
        else:
            stock.action = "축소 검토"
            stock.action_color = "orange"
            stock.action_reason = self._build_action_reason(d, "reduce")

        # 전망 판단
        if d.d4_timing_score >= 65 and d.d5_metagame_score >= 50:
            stock.forecast = "상승 우세"
            stock.forecast_class = "up"
        elif d.d4_timing_score < 40 or d.d5_metagame_score < 35:
            stock.forecast = "조정 가능"
            stock.forecast_class = "down"
        else:
            stock.forecast = "횡보"
            stock.forecast_class = "flat"

    @staticmethod
    def _build_action_reason(d: DimensionScores, mode: str) -> str:
        """행동 근거 한 줄 생성"""
        parts = []
        if mode == "hold":
            if d.d6_trap_pct < 30:
                parts.append(f"함정 {d.d6_trap_pct:.0f}% 안전")
            if d.d4_timing_score >= 70:
                parts.append(f"나선 {d.d4_clock_hour:.1f}시 양호")
            if d.d5_metagame_score >= 60:
                parts.append("군중 무관심 = 기회")
        elif mode == "watch":
            parts.append(f"6D {d.total_6d:.1f}점 중립 구간")
        elif mode == "caution":
            if d.d6_trap_pct >= 50:
                parts.append(f"함정 {d.d6_trap_pct:.0f}% 주의")
            if d.d4_timing_score < 40:
                parts.append(f"나선 {d.d4_clock_hour:.1f}시 고점 근접")
            if d.d5_neglect_pct >= 60:
                parts.append(f"군중 과열 {d.d5_neglect_pct:.0f}%")
        elif mode == "reduce":
            parts.append(f"6D {d.total_6d:.1f}점 불리")

        return ". ".join(parts) if parts else "종합 판단 기반"

    # ──────────────────────────────────────────
    # 리포트 조립 + HTML 렌더링
    # ──────────────────────────────────────────

    def _build_report_data(
        self, stocks: list[StockReportData], today: str,
    ) -> PortfolioReportData:
        """리포트 데이터 조립"""
        total_eval = sum(s.current_price * s.shares for s in stocks)
        returns = [s.return_pct for s in stocks]
        avg_return = sum(returns) / len(returns) if returns else 0

        top_stock = max(stocks, key=lambda s: s.dimensions.total_6d if s.dimensions else 0)

        return PortfolioReportData(
            report_date=today,
            total_eval=total_eval,
            stock_count=len(stocks),
            avg_return_pct=round(avg_return, 2),
            top_6d_name=top_stock.name,
            top_6d_score=top_stock.dimensions.total_6d if top_stock.dimensions else 0,
            stocks=stocks,
        )

    def _render_html(self, report: PortfolioReportData) -> str:
        """Jinja2로 HTML 렌더링"""
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,
        )
        # 커스텀 필터/함수 등록
        env.globals["score_color"] = _score_color
        env.globals["forecast_emoji"] = _forecast_emoji
        env.globals["action_emoji"] = _action_emoji
        env.globals["rank_emoji"] = _rank_emoji
        env.globals["header_emoji"] = "🧠"

        template = env.get_template("portfolio_report.html")
        return template.render(report=report)

    # ──────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────

    def _load_parquet(self, ticker: str) -> pd.DataFrame | None:
        """종목 parquet 로드"""
        pq = self.parquet_dir / f"{ticker}.parquet"
        if not pq.exists():
            return None
        try:
            return pd.read_parquet(pq)
        except Exception:
            return None
