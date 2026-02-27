"""보유종목 동적 목표가 재판정 도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MonitorAction(Enum):
    """포지션 판정 액션."""

    ADD = "ADD"                    # 목표/현재 >= 1.08 (추가매수)
    HOLD = "HOLD"                  # 목표/현재 >= 1.03 (보유 유지)
    PARTIAL_SELL = "PARTIAL_SELL"  # 목표/현재 < 1.01 (부분매도 50%)
    FULL_SELL = "FULL_SELL"        # 목표/현재 < 0.97 (전량매도)


@dataclass
class AdjustmentBreakdown:
    """동적 목표가 7축 조정 내역."""

    report_adj: float = 0.0   # 1 증권사 리포트
    news_adj: float = 0.0     # 2 뉴스 임팩트
    supply_adj: float = 0.0   # 3 수급
    macd_adj: float = 0.0     # 4 MACD
    rsi_adj: float = 0.0      # 5 RSI
    bb_adj: float = 0.0       # 6 볼린저밴드
    dart_adj: float = 0.0     # 7 DART 이벤트

    @property
    def total(self) -> float:
        return (
            self.report_adj + self.news_adj + self.supply_adj
            + self.macd_adj + self.rsi_adj + self.bb_adj + self.dart_adj
        )


@dataclass
class PositionTarget:
    """단일 종목 재판정 결과."""

    date: str = ""
    ticker: str = ""
    name: str = ""
    # 포지션 정보
    quantity: int = 0
    avg_price: float = 0.0
    current_price: float = 0.0
    pnl_pct: float = 0.0
    # 목표가 산출
    base_target: float = 0.0
    adjustment: AdjustmentBreakdown = field(default_factory=AdjustmentBreakdown)
    final_target: float = 0.0
    # 판정
    action: MonitorAction = MonitorAction.HOLD
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # 참조
    ratio_to_current: float = 0.0  # final_target / current_price


@dataclass
class MonitorResult:
    """전체 재판정 실행 결과."""

    date: str = ""
    generated_at: str = ""
    total_holdings: int = 0
    processed: int = 0
    actions_summary: dict[str, int] = field(default_factory=dict)
    positions: list[PositionTarget] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
