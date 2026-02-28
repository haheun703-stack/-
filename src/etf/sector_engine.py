"""
축1: 섹터 ETF 선정 엔진 (저격수)
====================================
모멘텀 Top N → Smart Money 필터 → 수급 확인 → 매수 후보
"""

from datetime import datetime
from dataclasses import dataclass

from src.etf.config import build_sector_universe, load_settings


@dataclass
class SectorETFCandidate:
    """섹터 ETF 매수 후보."""
    code: str
    name: str
    sector: str
    momentum_rank: int
    momentum_score: float
    momentum_5d: float
    momentum_20d: float
    momentum_60d: float
    smart_money_type: str       # "smart_money" / "theme_money" / "none"
    supply_score: float         # 0~100
    foreign_net_5d: float
    inst_net_5d: float
    composite_score: float = 0.0
    signal: str = "HOLD"
    reason: str = ""


@dataclass
class SectorETFPosition:
    """현재 보유 중인 섹터 ETF 포지션."""
    code: str
    name: str
    sector: str
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    pnl_pct: float = 0.0
    current_momentum_rank: int = 0


class SectorETFEngine:
    """섹터 ETF 선정 및 관리 엔진."""

    def __init__(self, settings: dict = None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.get("sector", {})
        self.universe = build_sector_universe()
        self.current_positions: list[SectorETFPosition] = []
        self.individual_stock_sectors: set[str] = set()

        # 설정값 로드
        self.max_holdings = self.cfg.get("max_holdings", 3)
        self.momentum_top_n = self.cfg.get("momentum_top_n", 5)
        self.smart_money_required = self.cfg.get("smart_money_required", True)
        self.min_supply_score = self.cfg.get("min_supply_score", 60)
        self.stop_loss_pct = self.cfg.get("stop_loss_pct", -5.0)
        self.rotation_rank_threshold = self.cfg.get("rotation_rank_threshold", 5)
        self.momentum_weights = self.cfg.get("momentum_weights", {"5d": 0.2, "20d": 0.5, "60d": 0.3})
        self.sector_overlap_block = self.settings.get("risk", {}).get("sector_overlap_block", True)

    def run(
        self,
        momentum_data: dict,
        smart_money_data: dict,
        supply_data: dict,
        individual_sectors: set[str] = None,
    ) -> dict:
        """
        섹터 ETF 엔진 메인 실행.

        Args:
            momentum_data: {sector: {"5d", "20d", "60d", "rank"}}
            smart_money_data: {code: {"type": "smart_money"/"theme_money"/"none"}}
            supply_data: {code: {"foreign_net_5d", "inst_net_5d", "score"}}
            individual_sectors: 개별주 보유 섹터 (중복 방지)
        """
        if individual_sectors:
            self.individual_stock_sectors = individual_sectors

        # 후보 스캔 및 필터링
        candidates = self._scan_candidates(momentum_data, smart_money_data, supply_data)
        buy_candidates = self._filter_and_rank(candidates)

        # 기존 포지션 점검
        sell_signals = self._check_existing_positions()

        # 결과 요약
        buy_names = [f"{c.name}({c.composite_score:.0f}점)" for c in buy_candidates]
        sell_names = [s["name"] for s in sell_signals if s["signal"] == "SELL"]

        summary = (
            f"[섹터 ETF] 매수후보 {len(buy_candidates)}종: "
            f"{', '.join(buy_names) or '없음'} | "
            f"매도시그널: {', '.join(sell_names) or '없음'}"
        )

        return {
            "buy_candidates": [self._candidate_to_dict(c) for c in buy_candidates],
            "sell_signals": sell_signals,
            "current_positions": [self._position_to_dict(p) for p in self.current_positions],
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    def _scan_candidates(
        self, momentum_data: dict, smart_money_data: dict, supply_data: dict
    ) -> list[SectorETFCandidate]:
        """전체 유니버스 스캔 → 후보 리스트."""
        candidates = []

        for code, info in self.universe.items():
            sector = info["sector"]

            # 개별주 섹터 중복 차단
            if self.sector_overlap_block and sector in self.individual_stock_sectors:
                continue

            # 모멘텀 데이터
            mom = momentum_data.get(sector, {})
            if not mom:
                continue

            mom_5d = mom.get("5d", 0)
            mom_20d = mom.get("20d", 0)
            mom_60d = mom.get("60d", 0)
            mom_rank = mom.get("rank", 99)

            # 가중 모멘텀 점수
            w = self.momentum_weights
            mom_score = mom_5d * w.get("5d", 0.2) + mom_20d * w.get("20d", 0.5) + mom_60d * w.get("60d", 0.3)

            # Smart Money
            sm = smart_money_data.get(code, {})
            sm_type = sm.get("type", "none")

            # 수급
            sd = supply_data.get(code, {})
            supply_score = sd.get("score", 0)
            foreign_net = sd.get("foreign_net_5d", 0)
            inst_net = sd.get("inst_net_5d", 0)

            candidates.append(SectorETFCandidate(
                code=code, name=info["name"], sector=sector,
                momentum_rank=mom_rank, momentum_score=mom_score,
                momentum_5d=mom_5d, momentum_20d=mom_20d, momentum_60d=mom_60d,
                smart_money_type=sm_type,
                supply_score=supply_score,
                foreign_net_5d=foreign_net, inst_net_5d=inst_net,
            ))

        return candidates

    def _filter_and_rank(self, candidates: list[SectorETFCandidate]) -> list[SectorETFCandidate]:
        """필터링 + 순위 매기기 → 최종 매수 후보."""
        # Filter 1: 모멘텀 Top N
        filtered = [c for c in candidates if c.momentum_rank <= self.momentum_top_n]

        # Filter 2: Smart Money 필수
        if self.smart_money_required:
            filtered = [c for c in filtered if c.smart_money_type != "none"]

        # Filter 3: 최소 수급 점수
        filtered = [c for c in filtered if c.supply_score >= self.min_supply_score]

        # 종합 점수 계산
        for c in filtered:
            c.composite_score = self._composite_score(c)

        # 정렬 + 제한
        filtered.sort(key=lambda x: x.composite_score, reverse=True)

        # 섹터 중복 제거 (동일 섹터에서 최고 점수만)
        seen_sectors = set()
        deduped = []
        for c in filtered:
            if c.sector not in seen_sectors:
                seen_sectors.add(c.sector)
                deduped.append(c)

        top = deduped[:self.max_holdings]
        for c in top:
            c.signal = "BUY"
            c.reason = self._reason(c)

        return top

    def _composite_score(self, c: SectorETFCandidate) -> float:
        """종합 점수 (0~100)."""
        # 모멘텀 정규화 (40%)
        mom_norm = min(max((c.momentum_score + 20) / 40 * 100, 0), 100)
        # Smart Money (20%)
        sm_map = {"smart_money": 100, "theme_money": 70, "none": 0}
        sm_norm = sm_map.get(c.smart_money_type, 0)
        # 수급 (30%)
        supply_norm = c.supply_score
        # 순위 보너스 (10%)
        rank_bonus = max(0, (6 - c.momentum_rank)) * 20

        return round(mom_norm * 0.40 + sm_norm * 0.20 + supply_norm * 0.30 + rank_bonus * 0.10, 2)

    def _reason(self, c: SectorETFCandidate) -> str:
        parts = [f"모멘텀 {c.momentum_rank}위"]
        if c.smart_money_type == "smart_money":
            parts.append("Smart Money 유입")
        elif c.smart_money_type == "theme_money":
            parts.append("Theme Money 감지")
        parts.append(f"수급 {c.supply_score:.0f}점")
        if c.foreign_net_5d > 0 and c.inst_net_5d > 0:
            parts.append("외인+기관 동시매수")
        return " | ".join(parts)

    def _check_existing_positions(self) -> list[dict]:
        """보유 포지션 점검 → 매도 시그널."""
        sell_signals = []
        for pos in self.current_positions:
            action = {"code": pos.code, "name": pos.name, "signal": "HOLD", "reason": ""}
            if pos.pnl_pct <= self.stop_loss_pct:
                action["signal"] = "SELL"
                action["reason"] = f"손절 ({pos.pnl_pct:.1f}% ≤ {self.stop_loss_pct}%)"
            elif pos.current_momentum_rank > self.rotation_rank_threshold:
                action["signal"] = "SELL"
                action["reason"] = f"모멘텀 순위 이탈 ({pos.current_momentum_rank}위 > {self.rotation_rank_threshold}위)"
            sell_signals.append(action)
        return sell_signals

    def _candidate_to_dict(self, c: SectorETFCandidate) -> dict:
        return {
            "code": c.code, "name": c.name, "sector": c.sector,
            "signal": c.signal, "composite_score": c.composite_score,
            "momentum_rank": c.momentum_rank, "momentum_score": c.momentum_score,
            "smart_money_type": c.smart_money_type, "supply_score": c.supply_score,
            "reason": c.reason,
        }

    def _position_to_dict(self, p: SectorETFPosition) -> dict:
        return {
            "code": p.code, "name": p.name, "sector": p.sector,
            "entry_price": p.entry_price, "entry_date": p.entry_date,
            "current_price": p.current_price, "pnl_pct": p.pnl_pct,
            "current_momentum_rank": p.current_momentum_rank,
        }
