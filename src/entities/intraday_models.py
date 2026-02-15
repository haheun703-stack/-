"""
Phase 1~2: 장중 실시간 데이터 수집 + 상황보고서 엔티티

Phase 1 (7개 테이블):
  - TickData: 1분 단위 현재가/체결 데이터
  - Candle5Min: 5분봉 OHLCV
  - InvestorFlowIntraday: 10분 단위 투자자별 매매동향
  - MarketContext: 5분 단위 시장 컨텍스트 (KOSPI/KOSDAQ/VIX 등)
  - SectorPrice: 10분 단위 업종별 시세
  - NewsEvent: 뉴스/공시 이벤트 (Phase 5에서 확장)
  - AIJudgment: Claude AI 판단 결과 (Phase 3에서 확장)

Phase 2 (상황보고서):
  - StockSituation: 개별 종목 상황 (가격/수급/기술지표)
  - SituationReport: Claude API 입력용 종합 상황보고서
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ─── 알림 등급 (Phase 4 미리 정의) ────────────────────

class AlertLevel(Enum):
    GREEN = "GREEN"      # 정상 — 일일 1회 정기 보고
    YELLOW = "YELLOW"    # 주의 — 즉시 알림 + 판단 근거
    RED = "RED"          # 매도 — 즉시 알림 + 구체적 행동 지침
    BLUE = "BLUE"        # 추가매수 — 기회 포착 알림


# ─── 1분 Tick 데이터 ─────────────────────────────────

@dataclass
class TickData:
    """1분 단위 현재가/체결 데이터"""
    ticker: str = ""
    timestamp: str = ""          # "2026-02-15 09:01:00"
    current_price: int = 0
    open_price: int = 0
    high_price: int = 0
    low_price: int = 0
    volume: int = 0              # 해당 1분간 체결량
    cum_volume: int = 0          # 누적 거래량
    change_pct: float = 0.0      # 전일 대비 등락률 (%)
    bid_price: int = 0           # 매수1호가
    ask_price: int = 0           # 매도1호가
    strength: float = 0.0        # 체결강도 (매수/매도 비율)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "current_price": self.current_price,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "volume": self.volume,
            "cum_volume": self.cum_volume,
            "change_pct": self.change_pct,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "strength": self.strength,
        }


# ─── 5분봉 캔들 ──────────────────────────────────────

@dataclass
class Candle5Min:
    """5분봉 OHLCV 캔들"""
    ticker: str = ""
    timestamp: str = ""          # "2026-02-15 09:05:00" (5분봉 시작 시각)
    open: int = 0
    high: int = 0
    low: int = 0
    close: int = 0
    volume: int = 0
    vwap: float = 0.0            # 거래량 가중 평균가

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }


# ─── 투자자별 매매동향 (장중) ─────────────────────────

@dataclass
class InvestorFlowIntraday:
    """10분 단위 투자자별 매매동향"""
    ticker: str = ""
    timestamp: str = ""
    foreign_net_buy: int = 0     # 외국인 순매수 (주)
    inst_net_buy: int = 0        # 기관 순매수 (주)
    individual_net_buy: int = 0  # 개인 순매수 (주)
    foreign_cum_net: int = 0     # 외국인 누적 순매수
    inst_cum_net: int = 0        # 기관 누적 순매수
    program_net_buy: int = 0     # 프로그램 순매수 (주)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "foreign_net_buy": self.foreign_net_buy,
            "inst_net_buy": self.inst_net_buy,
            "individual_net_buy": self.individual_net_buy,
            "foreign_cum_net": self.foreign_cum_net,
            "inst_cum_net": self.inst_cum_net,
            "program_net_buy": self.program_net_buy,
        }


# ─── 시장 컨텍스트 ───────────────────────────────────

@dataclass
class MarketContext:
    """5분 단위 시장 지수/환경 데이터"""
    timestamp: str = ""
    kospi: float = 0.0
    kospi_change_pct: float = 0.0
    kosdaq: float = 0.0
    kosdaq_change_pct: float = 0.0
    usd_krw: float = 0.0        # 원/달러 환율
    us_futures: float = 0.0     # S&P500 선물
    vix: float = 0.0            # VIX 지수
    bond_yield_kr_3y: float = 0.0  # 국고채 3년물
    kospi_volume: int = 0       # KOSPI 거래대금 (억)
    kosdaq_volume: int = 0      # KOSDAQ 거래대금 (억)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "kospi": self.kospi,
            "kospi_change_pct": self.kospi_change_pct,
            "kosdaq": self.kosdaq,
            "kosdaq_change_pct": self.kosdaq_change_pct,
            "usd_krw": self.usd_krw,
            "us_futures": self.us_futures,
            "vix": self.vix,
            "bond_yield_kr_3y": self.bond_yield_kr_3y,
            "kospi_volume": self.kospi_volume,
            "kosdaq_volume": self.kosdaq_volume,
        }


# ─── 업종별 시세 ─────────────────────────────────────

@dataclass
class SectorPrice:
    """10분 단위 업종별 시세"""
    timestamp: str = ""
    sector_code: str = ""        # 업종 코드
    sector_name: str = ""        # 업종명
    index_value: float = 0.0     # 업종 지수
    change_pct: float = 0.0      # 등락률 (%)
    volume: int = 0              # 거래대금 (억)
    advance_count: int = 0       # 상승 종목 수
    decline_count: int = 0       # 하락 종목 수

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sector_code": self.sector_code,
            "sector_name": self.sector_name,
            "index_value": self.index_value,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "advance_count": self.advance_count,
            "decline_count": self.decline_count,
        }


# ─── 뉴스 이벤트 (Phase 5 스켈레톤) ─────────────────

@dataclass
class NewsEvent:
    """뉴스/공시 이벤트"""
    id: str = ""
    timestamp: str = ""
    ticker: str = ""             # 관련 종목 (없으면 "")
    source: str = ""             # DART / NEWS / RSS
    title: str = ""
    content: str = ""
    sentiment: str = ""          # positive / negative / neutral
    impact_score: float = 0.0    # 0.0 ~ 1.0
    category: str = ""           # earnings / disclosure / rumor / macro

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "sentiment": self.sentiment,
            "impact_score": self.impact_score,
            "category": self.category,
        }


# ─── AI 판단 결과 (Phase 3 스켈레톤) ─────────────────

@dataclass
class AIJudgment:
    """Claude AI 판단 결과"""
    id: str = ""
    timestamp: str = ""
    ticker: str = ""
    alert_level: str = "GREEN"   # AlertLevel.value
    action: str = ""             # hold / sell / buy_more / tighten_stop
    confidence: float = 0.0      # 0.0 ~ 1.0
    reasoning: str = ""          # AI의 판단 근거 (텍스트)
    target_price: float = 0.0    # 목표가
    stop_price: float = 0.0      # 손절가
    position_advice: str = ""    # 구체적 행동 지침
    context_summary: str = ""    # 입력된 상황 보고서 요약
    model: str = ""              # 사용된 모델명
    cost_usd: float = 0.0        # API 호출 비용

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "alert_level": self.alert_level,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "position_advice": self.position_advice,
            "context_summary": self.context_summary,
            "model": self.model,
            "cost_usd": self.cost_usd,
        }


# ═══════════════════════════════════════════════════
# Phase 2: 상황보고서 엔티티
# ═══════════════════════════════════════════════════

@dataclass
class StockSituation:
    """개별 종목 상황 스냅샷 — Phase 2 상황보고서의 종목별 섹션"""
    ticker: str = ""
    name: str = ""

    # 가격 현황
    current_price: int = 0
    change_pct: float = 0.0          # 전일 대비 등락률
    open_price: int = 0
    high_price: int = 0
    low_price: int = 0
    volume: int = 0

    # 포지션 정보 (보유 중인 경우)
    entry_price: int = 0
    shares: int = 0
    pnl_pct: float = 0.0            # 수익률
    hold_days: int = 0
    stop_loss: int = 0
    target_price: int = 0

    # 장중 추세 (5분봉 기반)
    intraday_trend: str = ""         # "상승", "하락", "횡보", "V반등", "역V"
    intraday_high: int = 0
    intraday_low: int = 0
    price_from_open_pct: float = 0.0  # 시가 대비 현재가 (%)

    # 수급
    foreign_net_buy: int = 0
    inst_net_buy: int = 0
    individual_net_buy: int = 0
    flow_direction: str = ""         # "외인+기관 매수", "외인 매도 기관 매수" 등

    # v8.1 기술지표 요약 (일봉 기반)
    rsi_14: float = 0.0
    adx_14: float = 0.0
    bb_position: float = 0.0        # 볼린저 밴드 위치 (0=하단, 1=상단)
    ou_z: float = 0.0               # OU z-score
    macd_histogram: float = 0.0
    trix_signal: str = ""            # "골든크로스", "데드크로스", "중립"
    volume_ratio: float = 0.0       # 오늘 거래량 / 20일 평균
    smart_z: float = 0.0            # 스마트머니 z-score

    # v8.1 Gate/Score/Trigger 요약
    v8_grade: str = ""               # A/B/C/F
    v8_total_score: float = 0.0
    v8_gate_passed: bool = False
    v8_active_triggers: str = ""     # "T1(TRIX), T3(곡률+OBV)"

    # 경고 플래그
    alerts: list = field(default_factory=list)  # ["52주 고점 접근", "거래량 급증"]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "current_price": self.current_price,
            "change_pct": self.change_pct,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "volume": self.volume,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "pnl_pct": self.pnl_pct,
            "hold_days": self.hold_days,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "intraday_trend": self.intraday_trend,
            "intraday_high": self.intraday_high,
            "intraday_low": self.intraday_low,
            "price_from_open_pct": self.price_from_open_pct,
            "foreign_net_buy": self.foreign_net_buy,
            "inst_net_buy": self.inst_net_buy,
            "individual_net_buy": self.individual_net_buy,
            "flow_direction": self.flow_direction,
            "rsi_14": self.rsi_14,
            "adx_14": self.adx_14,
            "bb_position": self.bb_position,
            "ou_z": self.ou_z,
            "macd_histogram": self.macd_histogram,
            "trix_signal": self.trix_signal,
            "volume_ratio": self.volume_ratio,
            "smart_z": self.smart_z,
            "v8_grade": self.v8_grade,
            "v8_total_score": self.v8_total_score,
            "v8_gate_passed": self.v8_gate_passed,
            "v8_active_triggers": self.v8_active_triggers,
            "alerts": self.alerts,
        }


@dataclass
class SituationReport:
    """
    Phase 2: 종합 상황보고서 — Claude AI 입력용 구조화된 보고서.

    구성:
      - 시장 환경 (KOSPI/KOSDAQ/업종)
      - 보유종목별 상세 현황
      - 핵심 경고/기회 요약
      - 추천 행동 (AI 판단 전 프리필)

    트리거:
      - 30분 정기 생성
      - 3%+ 급변 시 즉시 생성
    """
    timestamp: str = ""
    report_type: str = "regular"     # "regular" / "emergency" / "closing"

    # 시장 환경
    kospi: float = 0.0
    kospi_change_pct: float = 0.0
    kosdaq: float = 0.0
    kosdaq_change_pct: float = 0.0
    market_regime: str = ""          # "강세", "약세", "횡보", "변동성확대"
    top_sectors: list = field(default_factory=list)    # [{"name": "반도체", "change_pct": 2.1}]
    bottom_sectors: list = field(default_factory=list)  # [{"name": "건설", "change_pct": -1.5}]

    # 종목별 상황
    stocks: list = field(default_factory=list)  # list[StockSituation.to_dict()]

    # 핵심 요약 (Phase 3 Claude API 프롬프트의 최상단에 배치)
    summary_alerts: list = field(default_factory=list)  # ["삼성전자 -3.2% 급락", "외인 일괄 매도"]
    opportunities: list = field(default_factory=list)    # ["SK하이닉스 OU z=-1.8 반등 신호"]

    # 메타
    holdings_count: int = 0
    data_freshness_sec: int = 0      # 최신 틱 데이터 age (초)
    generation_ms: int = 0           # 보고서 생성 소요시간 (ms)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "report_type": self.report_type,
            "market": {
                "kospi": self.kospi,
                "kospi_change_pct": self.kospi_change_pct,
                "kosdaq": self.kosdaq,
                "kosdaq_change_pct": self.kosdaq_change_pct,
                "market_regime": self.market_regime,
                "top_sectors": self.top_sectors,
                "bottom_sectors": self.bottom_sectors,
            },
            "stocks": self.stocks,
            "summary_alerts": self.summary_alerts,
            "opportunities": self.opportunities,
            "holdings_count": self.holdings_count,
            "data_freshness_sec": self.data_freshness_sec,
            "generation_ms": self.generation_ms,
        }

    def to_prompt_text(self) -> str:
        """Claude API 입력용 구조화된 텍스트 생성"""
        lines = []
        lines.append(f"=== 상황보고서 ({self.timestamp}) ===")
        lines.append(f"유형: {self.report_type}")
        lines.append("")

        # 핵심 경고
        if self.summary_alerts:
            lines.append("### 핵심 경고")
            for a in self.summary_alerts:
                lines.append(f"  - {a}")
            lines.append("")

        # 기회 포착
        if self.opportunities:
            lines.append("### 기회 포착")
            for o in self.opportunities:
                lines.append(f"  - {o}")
            lines.append("")

        # 시장 환경
        lines.append("### 시장 환경")
        lines.append(f"  KOSPI: {self.kospi:,.1f} ({self.kospi_change_pct:+.2f}%)")
        lines.append(f"  KOSDAQ: {self.kosdaq:,.1f} ({self.kosdaq_change_pct:+.2f}%)")
        lines.append(f"  시장 분위기: {self.market_regime}")
        if self.top_sectors:
            tops = ", ".join(
                f"{s['name']}({s['change_pct']:+.1f}%)" for s in self.top_sectors[:3]
            )
            lines.append(f"  강세 업종: {tops}")
        if self.bottom_sectors:
            bots = ", ".join(
                f"{s['name']}({s['change_pct']:+.1f}%)" for s in self.bottom_sectors[:3]
            )
            lines.append(f"  약세 업종: {bots}")
        lines.append("")

        # 종목별 상황
        for s in self.stocks:
            lines.append(f"### [{s.get('ticker', '')}] {s.get('name', '')}")
            lines.append(
                f"  현재가: {s.get('current_price', 0):,}원 "
                f"({s.get('change_pct', 0):+.2f}%)"
            )

            # 포지션
            if s.get("shares", 0) > 0:
                lines.append(
                    f"  포지션: 매입가 {s.get('entry_price', 0):,}원, "
                    f"{s.get('shares', 0)}주, "
                    f"수익률 {s.get('pnl_pct', 0):+.2f}%, "
                    f"보유 {s.get('hold_days', 0)}일"
                )
                lines.append(
                    f"  손절: {s.get('stop_loss', 0):,}원 / "
                    f"목표: {s.get('target_price', 0):,}원"
                )

            # 장중 추세
            lines.append(
                f"  장중: {s.get('intraday_trend', '?')} "
                f"(시가대비 {s.get('price_from_open_pct', 0):+.2f}%)"
            )

            # 수급
            lines.append(
                f"  수급: 외인 {s.get('foreign_net_buy', 0):+,}주, "
                f"기관 {s.get('inst_net_buy', 0):+,}주 "
                f"→ {s.get('flow_direction', '?')}"
            )

            # 기술지표
            lines.append(
                f"  지표: RSI={s.get('rsi_14', 0):.1f}, "
                f"ADX={s.get('adx_14', 0):.1f}, "
                f"BB위치={s.get('bb_position', 0):.2f}, "
                f"OU_z={s.get('ou_z', 0):.2f}"
            )
            lines.append(
                f"  MACD히스토={s.get('macd_histogram', 0):.4f}, "
                f"TRIX={s.get('trix_signal', '?')}, "
                f"거래량비={s.get('volume_ratio', 0):.2f}x, "
                f"SmartZ={s.get('smart_z', 0):.2f}"
            )

            # v8 요약
            if s.get("v8_grade"):
                lines.append(
                    f"  v8: {s.get('v8_grade')}등급 "
                    f"(점수={s.get('v8_total_score', 0):.2f}, "
                    f"게이트={'통과' if s.get('v8_gate_passed') else '실패'}"
                    f"{', 트리거=' + s.get('v8_active_triggers') if s.get('v8_active_triggers') else ''})"
                )

            # 경고
            if s.get("alerts"):
                for a in s["alerts"]:
                    lines.append(f"  ⚠ {a}")

            lines.append("")

        return "\n".join(lines)
