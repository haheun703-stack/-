"""Port 인터페이스 - 바깥 계층이 구현해야 할 계약"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.models import (
    AnalysisReport,
    AnalysisScore,
    ChartData,
    Condition,
    FlowPrediction,
    InvestorFlow,
    Stock,
    SupplyDemandZone,
    TechnicalIndicators,
    TechnicalPattern,
    VolumeAnalysis,
)
from src.entities.trading_models import Order


class StockDataPort(ABC):
    """주식 데이터 조회 포트"""

    @abstractmethod
    async def fetch(self, ticker: str, period_days: int = 120) -> tuple[Stock, ChartData]:
        """종목 데이터를 가져온다"""
        ...


class ChartAnalysisPort(ABC):
    """차트 기술적 분석 포트"""

    @abstractmethod
    async def analyze(self, stock: Stock, chart_data: ChartData) -> TechnicalPattern:
        """기술적 패턴/지표를 분석한다"""
        ...


class VolumeAnalysisPort(ABC):
    """거래량 및 매물대 분석 포트"""

    @abstractmethod
    async def analyze(self, stock: Stock, chart_data: ChartData) -> VolumeAnalysis:
        """거래량과 매물대를 분석한다"""
        ...


class FlowPredictionPort(ABC):
    """흐름 예측 포트"""

    @abstractmethod
    async def predict(
        self,
        stock: Stock,
        chart_data: ChartData,
        pattern: TechnicalPattern,
        volume: VolumeAnalysis,
    ) -> FlowPrediction:
        """내일 흐름을 예측한다"""
        ...


class ConditionJudgePort(ABC):
    """유지/대응 조건 판단 포트"""

    @abstractmethod
    async def judge(
        self,
        stock: Stock,
        chart_data: ChartData,
        pattern: TechnicalPattern,
        volume: VolumeAnalysis,
        prediction: FlowPrediction,
    ) -> list[Condition]:
        """유지조건과 대응조건을 생성한다"""
        ...


class ReportPresenterPort(ABC):
    """리포트 출력 포트"""

    @abstractmethod
    def present(self, report: AnalysisReport) -> str:
        """분석 리포트를 문자열로 변환한다"""
        ...


class InvestorFlowPort(ABC):
    """투자자별 수급 데이터 조회 포트"""

    @abstractmethod
    async def fetch(self, ticker: str) -> InvestorFlow:
        """외국인/기관/개인 수급 데이터를 가져온다"""
        ...


class TechnicalAnalysisPort(ABC):
    """순수 Python 기술지표 계산 포트"""

    @abstractmethod
    def analyze(self, chart_data: ChartData) -> TechnicalIndicators:
        """OHLCV 데이터에서 기술적 지표를 계산한다"""
        ...


class OutputPort(ABC):
    """출력 추상화 포트 (콘솔/파일/알림)"""

    @abstractmethod
    def display(self, message: str) -> None:
        """메시지를 출력한다"""
        ...


class AIAnalysisPort(ABC):
    """AI 100점 스코어링 분석 포트"""

    @abstractmethod
    async def analyze(
        self,
        stock: Stock,
        chart_data: ChartData,
        investor_flow: InvestorFlow | None = None,
    ) -> AnalysisScore:
        """AI가 종합 분석하여 100점 만점 스코어를 생성한다"""
        ...


class NewsSearchPort(ABC):
    """v3.1 뉴스 검색 포트 — 외부 뉴스 API 추상화"""

    @abstractmethod
    async def search_stock_news(self, stock_name: str, market: str = "korean") -> dict | None:
        """종목별 최신 뉴스 검색"""
        ...

    @abstractmethod
    async def search_market_overview(self) -> dict | None:
        """전체 시장 동향 요약"""
        ...

    @abstractmethod
    async def search_x_sentiment(self, stock_name: str, days: int = 3) -> dict | None:
        """X(트위터) 여론/감성 분석"""
        ...


# ─── v4.0 라이브 트레이딩 포트 ─────────────────────────

class OrderPort(ABC):
    """주문 실행 포트 — 매수/매도/정정/취소"""

    @abstractmethod
    def buy_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매수"""
        ...

    @abstractmethod
    def sell_limit(self, ticker: str, price: int, quantity: int) -> Order:
        """지정가 매도"""
        ...

    @abstractmethod
    def buy_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매수"""
        ...

    @abstractmethod
    def sell_market(self, ticker: str, quantity: int) -> Order:
        """시장가 매도"""
        ...

    @abstractmethod
    def cancel(self, order: Order) -> bool:
        """주문 취소"""
        ...

    @abstractmethod
    def modify(self, order: Order, new_price: int, new_quantity: int) -> Order:
        """주문 정정"""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """주문 상태 조회"""
        ...


class BalancePort(ABC):
    """잔고/포지션 조회 포트"""

    @abstractmethod
    def fetch_balance(self) -> dict:
        """전체 잔고 조회 (예수금 + 평가 + 보유종목)"""
        ...

    @abstractmethod
    def fetch_holdings(self) -> list[dict]:
        """보유종목 목록 조회"""
        ...

    @abstractmethod
    def get_available_cash(self) -> float:
        """주문 가능 예수금 조회"""
        ...


class CurrentPricePort(ABC):
    """실시간 현재가 조회 포트"""

    @abstractmethod
    def fetch_current_price(self, ticker: str) -> dict:
        """종목 현재가 + 호가 정보 조회"""
        ...


# ─── v5.0 Sci-CoE 합의 포트 ─────────────────────────

class ConsensusPort(ABC):
    """v5.0 합의 판정 포트"""

    @abstractmethod
    def verify(self, votes: list, geo_indicators: dict | None = None) -> dict:
        """3축 기하학적 합의 판정 수행"""
        ...


class AnchorPort(ABC):
    """v5.0 앵커 학습 포트"""

    @abstractmethod
    def learn_from_trades(self, trades: list) -> None:
        """백테스트 거래 결과에서 앵커 사례 추출"""
        ...

    @abstractmethod
    def load(self) -> dict:
        """앵커 DB 로드"""
        ...

    @abstractmethod
    def save(self) -> None:
        """앵커 DB 저장"""
        ...


# ─── Phase 1: 장중 데이터 수집 포트 ─────────────────────

class IntradayDataPort(ABC):
    """장중 실시간 데이터 수집 포트 — KIS API 등 외부 데이터 소스 추상화"""

    @abstractmethod
    def fetch_tick(self, ticker: str) -> dict:
        """1분 단위 현재가/체결 데이터 조회"""
        ...

    @abstractmethod
    def fetch_minute_candles(self, ticker: str, period: int = 5) -> list[dict]:
        """N분봉 캔들 데이터 조회"""
        ...

    @abstractmethod
    def fetch_investor_flow(self, ticker: str) -> dict:
        """투자자별 매매동향 조회"""
        ...

    @abstractmethod
    def fetch_market_index(self) -> dict:
        """시장 지수 (KOSPI/KOSDAQ) 조회"""
        ...

    @abstractmethod
    def fetch_sector_prices(self) -> list[dict]:
        """업종별 시세 조회"""
        ...


class IntradayStorePort(ABC):
    """장중 데이터 저장소 포트 — SQLite/PostgreSQL 등 영속화 추상화"""

    @abstractmethod
    def save_tick(self, tick: dict) -> None:
        """1분 틱 데이터 저장"""
        ...

    @abstractmethod
    def save_candle(self, candle: dict) -> None:
        """5분봉 캔들 저장"""
        ...

    @abstractmethod
    def save_investor_flow(self, flow: dict) -> None:
        """투자자별 매매동향 저장"""
        ...

    @abstractmethod
    def save_market_context(self, ctx: dict) -> None:
        """시장 컨텍스트 저장"""
        ...

    @abstractmethod
    def save_sector_price(self, sector: dict) -> None:
        """업종별 시세 저장"""
        ...

    @abstractmethod
    def get_recent_ticks(self, ticker: str, minutes: int = 60) -> list[dict]:
        """최근 N분간 틱 데이터 조회"""
        ...

    @abstractmethod
    def get_today_candles(self, ticker: str) -> list[dict]:
        """오늘 5분봉 전체 조회"""
        ...

    @abstractmethod
    def get_latest_market_context(self) -> dict | None:
        """최신 시장 컨텍스트 조회"""
        ...

    @abstractmethod
    def get_today_investor_flow(self, ticker: str) -> list[dict]:
        """오늘 투자자 수급 전체 조회"""
        ...

    @abstractmethod
    def get_today_sector_prices(self) -> list[dict]:
        """오늘 업종 시세 최신 스냅샷"""
        ...

    @abstractmethod
    def cleanup_old_data(self, days: int = 30) -> int:
        """N일 이전 데이터 정리"""
        ...


# ─── C-Suite Agent Ports ─────────────────────────


class CFOPort(ABC):
    """CFO 에이전트 포트 — 포트폴리오 리스크 관리 및 자본 배분"""

    @abstractmethod
    async def allocate_capital(
        self, signal: dict, portfolio: dict, risk_budget: dict
    ) -> dict:
        """신규 진입 시 자본 배분 결정"""
        ...

    @abstractmethod
    async def health_check(self, portfolio: dict, market_context: dict) -> dict:
        """포트폴리오 건강 진단"""
        ...

    @abstractmethod
    async def drawdown_analysis(
        self, equity_curve: list, current_positions: list
    ) -> dict:
        """낙폭 분석 및 대응 방안"""
        ...


class RiskSentinelPort(ABC):
    """Risk Sentinel 포트 — 꼬리 리스크 감시 및 스트레스 테스트"""

    @abstractmethod
    async def scan_tail_risk(self, market_data: dict, portfolio: dict) -> dict:
        """꼬리 리스크 스캔 (VKOSPI, 상관관계 급등, 외국인 이탈 등)"""
        ...

    @abstractmethod
    async def analyze_correlation(self, returns_matrix: dict) -> dict:
        """포트폴리오 상관관계 레짐 분석"""
        ...

    @abstractmethod
    async def stress_test(self, portfolio: dict, scenarios: list) -> list:
        """스트레스 시나리오별 포트폴리오 영향 분석"""
        ...


class MacroAnalystPort(ABC):
    """Macro Analyst 포트 — 시장 레짐, 섹터 로테이션, 시장 폭 분석"""

    @abstractmethod
    async def analyze_regime(self, market_data: dict) -> dict:
        """현재 시장 레짐 판별 (bull/recovery/sideways/correction/bear/crisis)"""
        ...

    @abstractmethod
    async def analyze_sector_rotation(self, sector_data: dict) -> dict:
        """14개 섹터 모멘텀 및 로테이션 시그널 분석"""
        ...

    @abstractmethod
    async def analyze_breadth(self, breadth_data: dict) -> dict:
        """시장 폭(Market Breadth) 내부 건전성 진단"""
        ...
