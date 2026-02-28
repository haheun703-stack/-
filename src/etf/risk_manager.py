"""
ETF 리스크 매니저
====================================
드로다운 킬스위치, 섹터 중복 방지, 총 노출도 관리
"""

from dataclasses import dataclass, field

from src.etf.config import load_settings


@dataclass
class RiskCheckResult:
    """리스크 체크 결과."""
    passed: bool
    level: str              # OK / WARNING / DANGER / KILLSWITCH
    violations: list = field(default_factory=list)
    adjustments: list = field(default_factory=list)
    summary: str = ""


class ETFRiskManager:
    """ETF 포트폴리오 리스크 관리."""

    def __init__(self, settings: dict = None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.get("risk", {})
        self.portfolio_peak: float = 0
        self.manual_mode: bool = False

    def run_checks(
        self,
        portfolio_value: float,
        sector_exposure: dict[str, float],
        leverage_exposure_pct: float,
        total_investment_pct: float,
        individual_stock_sectors: set[str],
        etf_sectors: set[str],
        regime: str,
        previous_regime: str = None,
    ) -> RiskCheckResult:
        """전체 리스크 체크 실행."""
        violations = []
        adjustments = []

        # 1. 섹터 중복
        v, a = self._check_sector_overlap(individual_stock_sectors, etf_sectors, sector_exposure)
        violations.extend(v); adjustments.extend(a)

        # 2. 레버리지 한도
        v, a = self._check_leverage_limit(leverage_exposure_pct)
        violations.extend(v); adjustments.extend(a)

        # 3. 총 투자 비중
        v, a = self._check_total_investment(total_investment_pct)
        violations.extend(v); adjustments.extend(a)

        # 4. 드로다운 킬스위치
        v, a = self._check_drawdown(portfolio_value)
        violations.extend(v); adjustments.extend(a)

        # 5. 레짐 급변
        v, a = self._check_regime_crash(regime, previous_regime)
        violations.extend(v); adjustments.extend(a)

        # 결과 종합
        if any(a.get("severity") == "KILLSWITCH" for a in adjustments):
            level, passed = "KILLSWITCH", False
        elif any(a.get("severity") == "DANGER" for a in adjustments):
            level, passed = "DANGER", False
        elif violations:
            level, passed = "WARNING", True
        else:
            level, passed = "OK", True

        summary = self._build_summary(level, adjustments)
        return RiskCheckResult(passed=passed, level=level, violations=violations, adjustments=adjustments, summary=summary)

    def _check_sector_overlap(self, ind_sectors: set, etf_sectors: set, exposure: dict) -> tuple[list, list]:
        violations, adjustments = [], []
        if not self.cfg.get("sector_overlap_block", True):
            return violations, adjustments

        overlap = ind_sectors & etf_sectors
        if overlap:
            violations.append({"type": "sector_overlap", "message": f"개별주-ETF 섹터 중복: {overlap}"})
            for sector in overlap:
                adjustments.append({
                    "type": "remove_etf_sector", "sector": sector, "severity": "WARNING",
                    "message": f"'{sector}' 섹터 ETF 매도 필요 (개별주 보유 중)",
                })

        max_exp = self.cfg.get("max_sector_exposure_pct", 25)
        for sector, pct in exposure.items():
            if pct > max_exp:
                violations.append({"type": "sector_overexposure", "message": f"'{sector}' 노출 {pct:.1f}% > {max_exp}%"})
                adjustments.append({
                    "type": "reduce_sector", "sector": sector, "severity": "WARNING",
                    "message": f"'{sector}' 비중 {pct:.1f}% → {max_exp}%로 축소",
                })
        return violations, adjustments

    def _check_leverage_limit(self, leverage_pct: float) -> tuple[list, list]:
        violations, adjustments = [], []
        max_lev = self.cfg.get("max_leverage_exposure_pct", 20)
        if leverage_pct > max_lev:
            violations.append({"type": "leverage_overlimit", "message": f"레버리지 비중 {leverage_pct:.1f}% > {max_lev}%"})
            adjustments.append({
                "type": "reduce_leverage", "severity": "DANGER",
                "message": f"레버리지 {leverage_pct:.1f}% → {max_lev}%로 축소 필요",
            })
        return violations, adjustments

    def _check_total_investment(self, total_pct: float) -> tuple[list, list]:
        violations, adjustments = [], []
        max_total = self.cfg.get("max_total_investment_pct", 90)
        min_cash = self.cfg.get("min_cash_pct", 10)
        if total_pct > max_total:
            violations.append({"type": "overinvested", "message": f"총 투자 {total_pct:.1f}% > {max_total}%"})
            adjustments.append({
                "type": "reduce_total", "severity": "WARNING",
                "message": f"총 비중 {total_pct:.1f}% → {max_total}%로 축소, 현금 확보",
            })
        return violations, adjustments

    def _check_drawdown(self, portfolio_value: float) -> tuple[list, list]:
        violations, adjustments = [], []
        if portfolio_value > self.portfolio_peak:
            self.portfolio_peak = portfolio_value
        if self.portfolio_peak == 0:
            return violations, adjustments

        dd_pct = ((portfolio_value - self.portfolio_peak) / self.portfolio_peak) * 100
        for level in self.cfg.get("killswitch_levels", []):
            threshold = level["drawdown_pct"]
            if dd_pct <= threshold:
                violations.append({"type": "drawdown_killswitch", "message": f"DD {dd_pct:.1f}% ≤ {threshold}%"})
                adjustments.append({
                    "type": level["action"], "severity": "KILLSWITCH",
                    "message": f"🚨 킬스위치: {level.get('desc', level['action'])} (DD {dd_pct:.1f}%)",
                })
        return violations, adjustments

    def _check_regime_crash(self, regime: str, prev_regime: str) -> tuple[list, list]:
        violations, adjustments = [], []
        crash_rule = self.cfg.get("regime_crash_rule", {})
        if not crash_rule or not prev_regime:
            return violations, adjustments
        if prev_regime.upper() == crash_rule.get("from", "").upper() and regime.upper() == crash_rule.get("to", "").upper():
            violations.append({"type": "regime_crash", "message": f"레짐 급변: {prev_regime} → {regime}"})
            adjustments.append({
                "type": crash_rule.get("action", "close_all"), "severity": "KILLSWITCH",
                "message": f"🚨 {crash_rule.get('desc', '레짐 급변 대응')}",
            })
        return violations, adjustments

    def _build_summary(self, level: str, adjustments: list) -> str:
        emoji = {"OK": "✅", "WARNING": "⚠️", "DANGER": "🔴", "KILLSWITCH": "🚨"}
        prefix = emoji.get(level, "❓")
        if level == "OK":
            return f"{prefix} [리스크] 정상 - 모든 체크 통과"
        msgs = [a["message"] for a in adjustments]
        return f"{prefix} [리스크] {level} - {' | '.join(msgs)}"
