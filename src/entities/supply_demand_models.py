"""수급 이면 데이터 레이어 모델 — 6개 레이어

Layer 1: 공매도 (Short Selling)
Layer 2: 허수호가/스푸핑 (Order Book Spoofing) — Phase 3
Layer 3: 체결강도 & 이상거래 (Execution Strength)
Layer 4: 프로그램매매 (Program Trading)
Layer 5: 기관/외인 수급 (Institutional Flow)
Layer 6: 옵션/선물 이상 신호 (Options Signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════
# Layer 1: 공매도 레이어 (pykrx — Phase 1)
# ═══════════════════════════════════════════════════
@dataclass
class ShortSellingData:
    """공매도 데이터"""

    ticker: str
    date: str
    short_volume: int = 0            # 공매도 거래량 (주)
    total_volume: int = 0            # 총 거래량 (주)
    short_ratio: float = 0.0         # 공매도 비중 (%)
    short_balance: int = 0           # 공매도 잔고 (주)
    short_balance_ratio: float = 0.0 # 잔고/상장주식수 (%)
    lending_balance: int = 0         # 대차잔고 (주)
    lending_change_5d: float = 0.0   # 대차잔고 5일 변화율 (%)
    is_overheated: bool = False      # 공매도 과열종목 지정 여부
    short_spike_ratio: float = 1.0   # 40일 평균 대비 비율 (1.0=정상)
    avg_short_ratio_40d: float = 0.0 # 40일 평균 공매도 비중

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "short_volume": self.short_volume,
            "short_ratio": self.short_ratio,
            "short_balance": self.short_balance,
            "short_balance_ratio": self.short_balance_ratio,
            "lending_balance": self.lending_balance,
            "lending_change_5d": self.lending_change_5d,
            "is_overheated": self.is_overheated,
            "short_spike_ratio": self.short_spike_ratio,
        }


# ═══════════════════════════════════════════════════
# Layer 2: 허수호가/스푸핑 (Phase 3 — KIS 실시간)
# ═══════════════════════════════════════════════════
@dataclass
class SpoofingData:
    """허수호가/스푸핑 탐지 데이터 — Phase 3 placeholder"""

    ticker: str
    date: str
    bid_wall_detected: bool = False    # 매수벽 감지
    ask_wall_detected: bool = False    # 매도벽 감지
    cancel_rate: float = 0.0           # 호가 취소율 (%)
    wall_disappeared: bool = False     # 벽 소멸 감지 (세력 매집 완료 신호)
    spoofing_score: float = 0.0        # 허수호가 의심도 (0-100)


# ═══════════════════════════════════════════════════
# Layer 3: 체결강도 & 이상거래 (Phase 1+2)
# ═══════════════════════════════════════════════════
@dataclass
class ExecutionStrengthData:
    """체결강도 및 이상거래 패턴"""

    ticker: str
    date: str
    execution_strength: float = 100.0  # 매수체결/매도체결 × 100
    large_trade_count: int = 0         # 대량 체결 건수 (평균 10배+)
    wash_trade_suspicious: bool = False  # 자전거래 의심
    time_pattern: str = ""             # "normal" / "opening_heavy" / "closing_heavy"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "execution_strength": self.execution_strength,
            "large_trade_count": self.large_trade_count,
            "wash_trade_suspicious": self.wash_trade_suspicious,
        }


# ═══════════════════════════════════════════════════
# Layer 4: 프로그램매매 (pykrx/KRX — Phase 1)
# ═══════════════════════════════════════════════════
@dataclass
class ProgramTradingData:
    """프로그램매매 & 선물 베이시스"""

    date: str
    arbitrage_buy: int = 0       # 차익거래 매수 (원)
    arbitrage_sell: int = 0      # 차익거래 매도 (원)
    non_arbitrage_buy: int = 0   # 비차익거래 매수 (원)
    non_arbitrage_sell: int = 0  # 비차익거래 매도 (원)
    basis: float = 0.0           # KOSPI200 선물-현물 베이시스

    @property
    def arbitrage_net(self) -> int:
        return self.arbitrage_buy - self.arbitrage_sell

    @property
    def non_arbitrage_net(self) -> int:
        return self.non_arbitrage_buy - self.non_arbitrage_sell

    @property
    def total_program_net(self) -> int:
        return self.arbitrage_net + self.non_arbitrage_net

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "arbitrage_net": self.arbitrage_net,
            "non_arbitrage_net": self.non_arbitrage_net,
            "total_program_net": self.total_program_net,
            "basis": self.basis,
        }


# ═══════════════════════════════════════════════════
# Layer 5: 기관/외인 수급 (pykrx — Phase 1)
# ═══════════════════════════════════════════════════
@dataclass
class InvestorFlowData:
    """투자자별 수급 데이터"""

    ticker: str
    date: str
    foreign_net: int = 0             # 외국인 순매수 (원)
    institution_net: int = 0         # 기관 순매수 (원)
    pension_net: int = 0             # 연기금 순매수 (원)
    individual_net: int = 0          # 개인 순매수 (원)
    foreign_consecutive_days: int = 0  # 외국인 연속 순매수 일수
    institution_cumulative_20d: int = 0  # 기관 20일 누적 순매수 (원)
    foreign_ownership_pct: float = 0.0  # 외국인 보유비율 (%)
    foreign_ownership_change: float = 0.0  # 보유비율 변화 (%p)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "foreign_net": self.foreign_net,
            "institution_net": self.institution_net,
            "pension_net": self.pension_net,
            "foreign_consecutive_days": self.foreign_consecutive_days,
            "institution_cumulative_20d": self.institution_cumulative_20d,
            "foreign_ownership_change": self.foreign_ownership_change,
        }


# ═══════════════════════════════════════════════════
# Layer 6: 옵션/선물 이상 신호 (Phase 2)
# ═══════════════════════════════════════════════════
@dataclass
class OptionsSignalData:
    """옵션/선물 이상 신호"""

    date: str
    put_call_ratio: float = 1.0      # Put/Call Ratio (> 1.2 공포, < 0.7 탐욕)
    implied_volatility: float = 0.0  # KOSPI200 내재변동성 (%)
    unusual_tickers: list = field(default_factory=list)  # 이상 옵션 거래 종목

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "put_call_ratio": self.put_call_ratio,
            "implied_volatility": self.implied_volatility,
            "unusual_tickers": self.unusual_tickers,
        }


# ═══════════════════════════════════════════════════
# 통합 수급 점수 (6D 프레임워크 연결)
# ═══════════════════════════════════════════════════
@dataclass
class SupplyDemandScore:
    """수급 이면 분석 통합 점수 → 6D 연결"""

    ticker: str
    date: str

    # 레이어별 점수 (0-100, 높을수록 위험)
    short_risk: float = 50.0       # L1: 공매도 위험도
    spoofing_risk: float = 50.0    # L2: 허수호가 위험도 (Phase 3)
    execution_score: float = 50.0  # L3: 체결강도 (높을수록 매수세)
    program_pressure: float = 50.0 # L4: 프로그램 매매 압력
    institutional: float = 50.0    # L5: 기관/외인 수급 (높을수록 유리)
    options_signal: float = 50.0   # L6: 옵션 이상 신호

    # 6D 연결값
    trap_adjustment: float = 0.0   # 함정률 보정 (-30 ~ +30)
    smart_money_boost: float = 0.0 # S5 스마트머니 부스트 (0.0 ~ 0.5)
    energy_adjustment: float = 0.0 # S1 에너지 보정 (-0.2 ~ +0.2)

    def calc_trap_v2(self, crowd_heat: float = 50.0) -> float:
        """함정률 v2.0 정량 공식

        함정률 = 0.25×군중과열 + 0.20×공매도위험 + 0.20×허수호가
                + 0.15×체결강도역전 + 0.10×자전거래의심 + 0.10×프로그램매도압력
        """
        execution_reverse = max(0, 100 - self.execution_score)
        return (
            crowd_heat * 0.25
            + self.short_risk * 0.20
            + self.spoofing_risk * 0.20
            + execution_reverse * 0.15
            + self.spoofing_risk * 0.10  # 자전거래 proxy
            + max(0, 100 - self.program_pressure) * 0.10
        )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "short_risk": self.short_risk,
            "spoofing_risk": self.spoofing_risk,
            "execution_score": self.execution_score,
            "program_pressure": self.program_pressure,
            "institutional": self.institutional,
            "options_signal": self.options_signal,
            "trap_adjustment": self.trap_adjustment,
            "smart_money_boost": self.smart_money_boost,
        }
