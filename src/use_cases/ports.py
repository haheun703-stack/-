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
