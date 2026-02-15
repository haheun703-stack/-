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
