"""6D Game Design 분석 모델 — 게임 설계자/역할/함정 분석"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameDesignAnalysis:
    """6D 게임 설계 분석 결과"""

    ticker: str
    name: str
    designer: str           # 설계자 (누가 이 게임을 만들었나)
    our_role: str           # 우리의 역할 (이 게임에서 우리는 누구인가)
    edge: str               # 우리의 엣지 (비대칭 정보 우위)
    trap_description: str   # 함정 설명
    trap_risk_pct: int      # 함정 위험도 (0-100)
    game_score: float       # 6D 종합 점수 (0-100)
    reasoning: str          # 종합 판단 근거

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "designer": self.designer,
            "our_role": self.our_role,
            "edge": self.edge,
            "trap_description": self.trap_description,
            "trap_risk_pct": self.trap_risk_pct,
            "game_score": self.game_score,
            "reasoning": self.reasoning,
        }


@dataclass
class DimensionScores:
    """5D/6D 차원별 점수 통합"""

    ticker: str
    name: str
    # 1D: 가격 위치
    d1_price_pct: float = 0.0           # 52주 고점 대비 % (0-100)
    # 2D: 밸류에이션
    d2_target_gap_pct: float = 0.0      # 목표가 괴리율 (%)
    d2_per: float = 0.0                 # PER
    # 3D: 멀티팩터
    d3_multifactor: float = 0.0         # Confluence 점수 (0-100)
    # 4D: 타이밍 (나선시계)
    d4_clock_hour: float = 0.0          # 시계 위치 (1-12시)
    d4_timing_score: float = 0.0        # 타이밍 점수 (0-100)
    # 5D: 메타게임
    d5_neglect_pct: float = 0.0         # 군중 관심도 (%) — 낮을수록 기회
    d5_metagame_score: float = 0.0      # 메타게임 점수 (0-100)
    # 6D: 게임 설계
    d6_game_score: float = 0.0          # 게임 점수 (0-100)
    d6_trap_pct: float = 0.0            # 함정 위험도 (%)
    # 종합
    total_5d: float = 0.0
    total_6d: float = 0.0

    def calc_totals(self) -> None:
        """5D/6D 종합 점수 계산"""
        # 5D 가중평균 (1D:15%, 2D:20%, 3D:25%, 4D:20%, 5D:20%)
        self.total_5d = (
            self.d1_price_pct * 0.15 * 0.01 * 100
            + min(self.d2_target_gap_pct, 100) * 0.20
            + self.d3_multifactor * 0.25
            + self.d4_timing_score * 0.20
            + self.d5_metagame_score * 0.20
        )
        # 6D = 5D 80% + 6D 20%
        self.total_6d = self.total_5d * 0.80 + self.d6_game_score * 0.20

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "d1_price_pct": self.d1_price_pct,
            "d2_target_gap_pct": self.d2_target_gap_pct,
            "d2_per": self.d2_per,
            "d3_multifactor": self.d3_multifactor,
            "d4_clock_hour": self.d4_clock_hour,
            "d4_timing_score": self.d4_timing_score,
            "d5_neglect_pct": self.d5_neglect_pct,
            "d5_metagame_score": self.d5_metagame_score,
            "d6_game_score": self.d6_game_score,
            "d6_trap_pct": self.d6_trap_pct,
            "total_5d": self.total_5d,
            "total_6d": self.total_6d,
        }


@dataclass
class PortfolioReportData:
    """포트폴리오 리포트 전체 데이터"""

    report_date: str
    total_eval: int = 0
    stock_count: int = 0
    avg_return_pct: float = 0.0
    top_6d_name: str = ""
    top_6d_score: float = 0.0
    stocks: list = None  # list[StockReportData]
    action_plan: list = None  # list[dict]

    def __post_init__(self):
        if self.stocks is None:
            self.stocks = []
        if self.action_plan is None:
            self.action_plan = []


@dataclass
class StockReportData:
    """종목별 리포트 데이터"""

    rank: int
    ticker: str
    name: str
    current_price: int
    return_pct: float
    shares: int
    hold_days: int
    investment: int
    # 차원 점수
    dimensions: DimensionScores | None = None
    # 6D 게임 분석
    game_analysis: GameDesignAnalysis | None = None
    # 행동 판단
    action: str = ""          # "보유", "축소검토", "반익실현" 등
    action_color: str = ""    # "green", "yellow", "orange", "red"
    action_reason: str = ""   # 행동 근거 한 줄
    # 전망
    forecast: str = ""        # "상승 우세", "횡보", "조정 가능"
    forecast_class: str = ""  # "up", "flat", "down"
    # 카탈리스트
    catalyst: str = ""
