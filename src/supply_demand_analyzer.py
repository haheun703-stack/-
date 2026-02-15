"""수급 이면 분석기 — 6개 레이어 → 6D 프레임워크 연결

Phase 1 (pykrx): L1 공매도 + L5 기관/외인 수급 → 점수화
Phase 2 (KIS):   L3 체결강도 + L4 프로그램매매 → 점수화
Phase 3 (KIS):   L2 허수호가 + L6 옵션/선물 → 점수화

최종 출력:
  - trap_adjustment: 6D 함정률 보정값 (-30 ~ +30)
  - smart_money_boost: v8 S5 스마트머니 부스트 (0.0 ~ 0.5)
  - energy_adjustment: v8 S1 에너지 보정 (-0.2 ~ +0.2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.entities.supply_demand_models import (
    InvestorFlowData,
    ShortSellingData,
    SupplyDemandScore,
)

logger = logging.getLogger(__name__)


@dataclass
class SupplyDemandConfig:
    """수급 분석 설정"""

    # L1: 공매도
    short_spike_danger: float = 2.0       # 40일 평균 대비 2배 이상 → 위험
    short_balance_danger: float = 3.0     # 상장주식수 3% 이상 → 숏스퀴즈 모니터링
    lending_increase_days: int = 5        # 대차잔고 연속 증가일 → 기관 하방 베팅

    # L5: 기관/외인
    foreign_consecutive_strong: int = 5   # 외국인 5일+ 순매수 → 추세 형성
    foreign_consecutive_weak: int = 3     # 3일 → 관심 시작
    institution_20d_positive: bool = True # 기관 20일 누적 양전환 → 매집 신호

    # 점수 가중치
    trap_weight_crowd: float = 0.25
    trap_weight_short: float = 0.20
    trap_weight_spoofing: float = 0.20
    trap_weight_execution: float = 0.15
    trap_weight_wash: float = 0.10
    trap_weight_program: float = 0.10


class SupplyDemandAnalyzer:
    """수급 이면 분석기"""

    def __init__(self, config: dict | None = None):
        sd_cfg = (config or {}).get("supply_demand", {})
        self.cfg = SupplyDemandConfig(
            short_spike_danger=sd_cfg.get("short_spike_danger", 2.0),
            short_balance_danger=sd_cfg.get("short_balance_danger", 3.0),
            lending_increase_days=sd_cfg.get("lending_increase_days", 5),
            foreign_consecutive_strong=sd_cfg.get("foreign_consecutive_strong", 5),
            foreign_consecutive_weak=sd_cfg.get("foreign_consecutive_weak", 3),
        )

    def analyze(
        self,
        ticker: str,
        date: str,
        short: ShortSellingData | None = None,
        flow: InvestorFlowData | None = None,
    ) -> SupplyDemandScore:
        """단일 종목 수급 분석 → 6D 연결 점수"""
        score = SupplyDemandScore(ticker=ticker, date=date)

        if short:
            score.short_risk = self._score_short_selling(short)
        if flow:
            score.institutional = self._score_investor_flow(flow)

        # 6D 연결값 계산
        score.trap_adjustment = self._calc_trap_adjustment(score, short, flow)
        score.smart_money_boost = self._calc_smart_money_boost(flow)
        score.energy_adjustment = self._calc_energy_adjustment(flow)

        return score

    # ─────────────────────────────────────────
    # L1: 공매도 위험도 (0-100, 높을수록 위험)
    # ─────────────────────────────────────────
    def _score_short_selling(self, s: ShortSellingData) -> float:
        risk = 50.0  # 기본 중립

        # 공매도 비중 스파이크
        if s.short_spike_ratio >= self.cfg.short_spike_danger:
            risk += 20  # 40일 평균 2배 이상 → 위험
        elif s.short_spike_ratio >= 1.5:
            risk += 10  # 1.5배 → 주의

        # 공매도 잔고 비중
        if s.short_balance_ratio >= self.cfg.short_balance_danger:
            risk += 15  # 상장주식수 3%+ → 숏스퀴즈 잠재 OR 하방 리스크
        elif s.short_balance_ratio >= 1.5:
            risk += 5

        # 대차잔고 급증 (5일 변화율)
        if s.lending_change_5d > 20:
            risk += 10  # 대차잔고 20% 이상 급증 → 기관 하방 준비
        elif s.lending_change_5d > 10:
            risk += 5

        # 과열종목 지정
        if s.is_overheated:
            risk += 15  # 자동매매 일시 중단 사유

        return min(100, max(0, risk))

    # ─────────────────────────────────────────
    # L5: 기관/외인 수급 (0-100, 높을수록 유리)
    # ─────────────────────────────────────────
    def _score_investor_flow(self, f: InvestorFlowData) -> float:
        score = 50.0  # 기본 중립

        # 외국인 연속 순매수
        if f.foreign_consecutive_days >= self.cfg.foreign_consecutive_strong:
            score += 25  # 5일+ 연속 → 추세 형성
        elif f.foreign_consecutive_days >= self.cfg.foreign_consecutive_weak:
            score += 10  # 3일+ → 관심

        # 외국인 연속 순매도 (역방향)
        if f.foreign_consecutive_days == 0 and f.foreign_net < 0:
            score -= 10

        # 기관 20일 누적 양전환
        if f.institution_cumulative_20d > 0:
            score += 15  # "기대가 식은 자리에 기관 진입" 신호
        elif f.institution_cumulative_20d < 0:
            score -= 5

        # 연기금 순매수
        if f.pension_net > 0:
            score += 5  # 가장 장기 관점 투자자
        elif f.pension_net < 0:
            score -= 3

        return min(100, max(0, score))

    # ─────────────────────────────────────────
    # 6D 연결: 함정률 보정 (-30 ~ +30)
    # ─────────────────────────────────────────
    def _calc_trap_adjustment(
        self,
        score: SupplyDemandScore,
        short: ShortSellingData | None,
        flow: InvestorFlowData | None,
    ) -> float:
        adj = 0.0

        # 공매도 위험 → 함정률 상승
        if short:
            if short.short_spike_ratio >= self.cfg.short_spike_danger:
                adj += 15  # 공매도 급증 → 함정률 +15p
            if short.is_overheated:
                adj += 10  # 과열종목 → +10p
            if short.lending_change_5d > 20:
                adj += 5   # 대차잔고 급증 → +5p

        # 기관/외인 매수세 → 함정률 감소
        if flow:
            if flow.foreign_consecutive_days >= self.cfg.foreign_consecutive_strong:
                adj -= 10  # 외국인 5일+ 순매수 → 함정률 -10p
            if flow.institution_cumulative_20d > 0:
                adj -= 5   # 기관 누적 양전환 → -5p

        return max(-30, min(30, adj))

    # ─────────────────────────────────────────
    # S5 스마트머니 부스트 (0.0 ~ 0.5)
    # ─────────────────────────────────────────
    def _calc_smart_money_boost(self, flow: InvestorFlowData | None) -> float:
        if flow is None:
            return 0.0

        boost = 0.0

        # 외국인 연속 순매수 → 스마트머니 존재
        if flow.foreign_consecutive_days >= 5:
            boost += 0.25
        elif flow.foreign_consecutive_days >= 3:
            boost += 0.10

        # 기관 20일 누적 양전환
        if flow.institution_cumulative_20d > 0:
            boost += 0.15

        # 연기금 순매수
        if flow.pension_net > 0:
            boost += 0.10

        return min(0.5, boost)

    # ─────────────────────────────────────────
    # S1 에너지 보정 (-0.2 ~ +0.2)
    # ─────────────────────────────────────────
    def _calc_energy_adjustment(self, flow: InvestorFlowData | None) -> float:
        if flow is None:
            return 0.0

        adj = 0.0

        # 외국인+기관 동시 순매수 → 에너지 부스트
        if flow.foreign_net > 0 and flow.institution_net > 0:
            adj += 0.10  # 쌍매수

        # 외국인+기관 동시 순매도 → 에너지 감소
        if flow.foreign_net < 0 and flow.institution_net < 0:
            adj -= 0.10  # 쌍매도

        return max(-0.2, min(0.2, adj))

    # ─────────────────────────────────────────
    # 일괄 분석
    # ─────────────────────────────────────────
    def analyze_batch(
        self, collected: dict, date: str
    ) -> dict[str, SupplyDemandScore]:
        """collect_all() 결과를 일괄 분석

        Args:
            collected: {ticker: {"short": ShortSellingData, "flow": InvestorFlowData}}
            date: 분석 기준일

        Returns:
            {ticker: SupplyDemandScore}
        """
        results = {}
        for ticker, data in collected.items():
            short = data.get("short") if data else None
            flow = data.get("flow") if data else None
            results[ticker] = self.analyze(ticker, date, short, flow)
        return results
