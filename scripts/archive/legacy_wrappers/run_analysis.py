"""종합 분석 + 100점 스코어링 실행 스크립트"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.adapters.claude_scoring import ClaudeScoringAdapter
from src.adapters.csv_stock_data_adapter import CsvStockDataAdapter
from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
from src.adapters.markdown_presenter import MarkdownPresenter
from src.adapters.naver_finance import NaverFinanceAdapter
from src.agents.chart_analysis import ChartAnalysisAgent
from src.agents.condition_judge import ConditionJudgeAgent
from src.agents.flow_prediction import FlowPredictionAgent
from src.agents.volume_analysis import VolumeAnalysisAgent
from src.use_cases.stock_analysis import StockAnalysisInteractor


async def main(ticker: str = "005930", source: str = "kis") -> None:
    """종합 분석 + AI 스코어링 실행"""
    if source == "csv":
        stock_data = CsvStockDataAdapter()
    else:
        stock_data = KisStockDataAdapter()

    interactor = StockAnalysisInteractor(
        stock_data=stock_data,
        chart_analysis=ChartAnalysisAgent(),
        volume_analysis=VolumeAnalysisAgent(),
        flow_prediction=FlowPredictionAgent(),
        condition_judge=ConditionJudgeAgent(),
        presenter=MarkdownPresenter(),
        investor_flow=NaverFinanceAdapter(),
        ai_scoring=ClaudeScoringAdapter(),
    )

    src_label = "KIS API 실시간" if source == "kis" else "CSV 파일"
    print(f"  [종합분석+스코어링] {ticker} 분석 시작 (데이터: {src_label})\n")

    report = await interactor.execute(ticker)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 종합 분석 + 100점 스코어링")
    parser.add_argument(
        "--stock", type=str, default="005930",
        help="종목코드 (기본: 005930 삼성전자)",
    )
    parser.add_argument(
        "--source", type=str, default="kis", choices=["kis", "csv"],
        help="데이터 소스 (kis=한투API, csv=CSV파일, 기본: kis)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.stock, args.source))
