"""
축3: 지수 ETF 엔진 (보급부대)
====================================
KODEX 200 + KODEX MSCI Korea TR
레짐에 따라 비중 조절, MA 보정
"""

from datetime import datetime

from src.etf.config import INDEX_ETF, load_settings, get_allocation


class IndexETFEngine:
    """지수 ETF 비중 관리 엔진."""

    def __init__(self, settings: dict = None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.get("index", {})
        self.etfs = INDEX_ETF
        self.rebalance_threshold = self.cfg.get("rebalance_threshold_pct", 2.0)
        self.current_holdings: dict = {}  # {code: {"weight_pct": float}}

    def set_current_holdings(self, holdings: dict):
        self.current_holdings = holdings

    def run(
        self,
        regime: str,
        ma_20_above: bool = True,
        ma_60_above: bool = True,
    ) -> dict:
        """
        지수 ETF 엔진 실행.

        Args:
            regime: KOSPI 레짐
            ma_20_above: KOSPI가 20일 이평선 위인지
            ma_60_above: KOSPI가 60일 이평선 위인지
        """
        regime = regime.upper()
        alloc = get_allocation(regime, self.settings)
        total_index_pct = alloc.get("index", 0)

        # CRISIS면 전량 매도
        if regime == "CRISIS":
            return self._close_all("CRISIS 레짐 - 지수 ETF 전량 매도")

        # MA 보정
        ma_adj = self._ma_adjustment(ma_20_above, ma_60_above)
        adjusted_pct = total_index_pct * ma_adj

        targets = []
        for key, etf_info in self.etfs.items():
            code = etf_info["code"]
            name = etf_info["name"]
            inner_weight = etf_info["weight"]

            target_pct = round(adjusted_pct * inner_weight, 2)
            current_pct = self.current_holdings.get(code, {}).get("weight_pct", 0)
            delta = round(target_pct - current_pct, 2)

            action, reason = self._determine_action(delta, target_pct, current_pct)

            targets.append({
                "code": code, "name": name,
                "target_weight_pct": target_pct,
                "current_weight_pct": current_pct,
                "action": action,
                "amount_delta_pct": delta,
                "reason": reason,
            })

        actions = [f"{t['name']}: {t['action']}" for t in targets if t["action"] != "HOLD"]
        if actions:
            summary = f"[지수 ETF] {regime} / 목표 {adjusted_pct:.1f}% | {' | '.join(actions)}"
        else:
            summary = f"[지수 ETF] {regime} / 목표 {adjusted_pct:.1f}% | 변동 없음"

        return {
            "targets": targets,
            "total_index_allocation_pct": adjusted_pct,
            "regime": regime,
            "ma_adjustment": ma_adj,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    def _ma_adjustment(self, ma_20_above: bool, ma_60_above: bool) -> float:
        if ma_20_above and ma_60_above:
            return 1.0
        elif ma_20_above:
            return 0.8
        elif ma_60_above:
            return 0.6
        return 0.4

    def _determine_action(self, delta: float, target: float, current: float) -> tuple[str, str]:
        if target == 0 and current > 0:
            return "SELL", "목표 비중 0% - 전량 매도"
        elif current == 0 and target > 0:
            return "BUY", f"신규 진입: 목표 {target:.1f}%"
        elif abs(delta) >= self.rebalance_threshold:
            direction = "추가매수" if delta > 0 else "일부매도"
            return "REBALANCE", f"{direction}: {current:.1f}% → {target:.1f}%"
        return "HOLD", f"비중 유지 (차이 {abs(delta):.1f}%p < {self.rebalance_threshold}%p)"

    def _close_all(self, reason: str) -> dict:
        targets = []
        for key, etf_info in self.etfs.items():
            code = etf_info["code"]
            current_pct = self.current_holdings.get(code, {}).get("weight_pct", 0)
            if current_pct > 0:
                targets.append({
                    "code": code, "name": etf_info["name"],
                    "target_weight_pct": 0, "current_weight_pct": current_pct,
                    "action": "SELL", "amount_delta_pct": -current_pct,
                    "reason": reason,
                })
        return {
            "targets": targets,
            "total_index_allocation_pct": 0,
            "regime": "CRISIS",
            "ma_adjustment": 0,
            "summary": f"[지수 ETF] 🚨 {reason}",
            "timestamp": datetime.now().isoformat(),
        }
