"""주식 분석 유스케이스 - Port를 통해 서브에이전트를 조율하는 핵심 로직"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.entities.models import AnalysisReport
from src.use_cases.ports import (
    AIAnalysisPort,
    ChartAnalysisPort,
    ConditionJudgePort,
    FlowPredictionPort,
    InvestorFlowPort,
    ReportPresenterPort,
    StockDataPort,
    VolumeAnalysisPort,
)


@dataclass
class StockAnalysisInteractor:
    """주식 분석 오케스트레이터 - 전체 워크플로우를 조율한다"""

    stock_data: StockDataPort
    chart_analysis: ChartAnalysisPort
    volume_analysis: VolumeAnalysisPort
    flow_prediction: FlowPredictionPort
    condition_judge: ConditionJudgePort
    presenter: ReportPresenterPort
    investor_flow: InvestorFlowPort | None = None
    ai_scoring: AIAnalysisPort | None = None

    async def execute(self, ticker: str, period_days: int = 120) -> str:
        """
        분석 워크플로우 실행

        1. 데이터 수집
        2. 차트분석 + 거래량분석 (병렬)
        3. 흐름 예측 (위 결과 종합)
        4. 조건 판단
        5. 리포트 생성
        """
        # Step 1: 데이터 수집
        stock, chart_data = await self.stock_data.fetch(ticker, period_days)

        # Step 2: 차트분석 + 거래량분석 + 수급조회 (병렬 실행)
        parallel_tasks = [
            self.chart_analysis.analyze(stock, chart_data),
            self.volume_analysis.analyze(stock, chart_data),
        ]
        if self.investor_flow:
            parallel_tasks.append(self.investor_flow.fetch(ticker))

        results = await asyncio.gather(*parallel_tasks)
        pattern = results[0]
        volume = results[1]
        flow = results[2] if len(results) > 2 else None

        # Step 3: 흐름 예측 (차트 + 거래량 결과 종합)
        prediction = await self.flow_prediction.predict(
            stock, chart_data, pattern, volume,
        )

        # Step 4: 유지/대응 조건 판단
        conditions = await self.condition_judge.judge(
            stock, chart_data, pattern, volume, prediction,
        )

        # Step 5: AI 100점 스코어링 (선택적)
        score = None
        if self.ai_scoring:
            score = await self.ai_scoring.analyze(stock, chart_data, flow)

        # Step 6: 리포트 생성
        report = AnalysisReport(
            stock=stock,
            chart_data=chart_data,
            technical_pattern=pattern,
            volume_analysis=volume,
            flow_prediction=prediction,
            conditions=conditions,
            investor_flow=flow,
            score=score,
        )

        return self.presenter.present(report)
