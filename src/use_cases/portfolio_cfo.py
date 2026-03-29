"""CFO 순수 Python — 포트폴리오 리스크 관리 + 자본배분

Claude API 호출 없이 수학적 모델로 직접 계산:
  1. Quarter-Kelly 포지션 사이징 + 3가지 패널티
  2. 포트폴리오 건강 진단 (집중도, VaR, 상관노출)
  3. 낙폭 분석 + 대응 판단 (normal/reduce/halt/emergency)

사용:
  from src.use_cases.portfolio_cfo import PortfolioCFO
  cfo = PortfolioCFO()
  report = cfo.full_report(positions, equity_curve, regime)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ═══════════════════════════════════════════════════════
#  결과 데이터 클래스
# ═══════════════════════════════════════════════════════

@dataclass
class CapitalAllocation:
    """자본 배분 결정"""
    ticker: str = ""
    name: str = ""
    recommended_size_pct: float = 0.0
    recommended_amount: float = 0.0
    kelly_fraction: float = 0.0
    quarter_kelly: float = 0.0
    correlation_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    concentration_penalty: float = 0.0
    final_size_pct: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioHealthCheck:
    """포트폴리오 건강 진단"""
    overall_score: float = 0.0  # 0~100
    risk_level: str = "moderate"  # conservative/moderate/aggressive/emergency
    positions_count: int = 0
    sector_concentration: float = 0.0  # HHI 지수 (0~1)
    top_holding_pct: float = 0.0
    estimated_var_95: float = 0.0
    cash_ratio: float = 0.0
    max_sector_pct: float = 0.0
    max_sector_name: str = ""
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DrawdownAnalysis:
    """낙폭 분석"""
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0
    recovery_estimate_days: int = 0
    action: str = "normal"  # normal / reduce / halt / emergency
    action_label: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CFOReport:
    """CFO 통합 리포트"""
    generated_at: str = ""
    health: PortfolioHealthCheck = field(default_factory=PortfolioHealthCheck)
    drawdown: DrawdownAnalysis = field(default_factory=DrawdownAnalysis)
    allocation_budget: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "health": self.health.to_dict(),
            "drawdown": self.drawdown.to_dict(),
            "allocation_budget": self.allocation_budget,
        }


# ═══════════════════════════════════════════════════════
#  CFO 엔진
# ═══════════════════════════════════════════════════════

class PortfolioCFO:
    """순수 Python CFO — 포트폴리오 리스크 + 자본배분"""

    # 설정 상수
    MAX_SINGLE_POSITION_PCT = 0.20      # 단일종목 최대 20%
    MAX_SECTOR_PCT = 0.30               # 단일섹터 최대 30%
    MIN_CASH_PCT = 0.20                 # 최소 현금 비중
    MDD_REDUCE_THRESHOLD = -0.08        # 8% 낙폭 → reduce
    MDD_HALT_THRESHOLD = -0.12          # 12% 낙폭 → halt
    MDD_EMERGENCY_THRESHOLD = -0.18     # 18% 낙폭 → emergency
    KELLY_FRACTION = 0.25               # Quarter-Kelly

    def __init__(self, config: dict | None = None):
        if config:
            cfo_cfg = config.get("cfo", {})
            self.MAX_SINGLE_POSITION_PCT = cfo_cfg.get("max_single_position_pct", 0.20)
            self.MAX_SECTOR_PCT = cfo_cfg.get("max_sector_pct", 0.30)
            self.MIN_CASH_PCT = cfo_cfg.get("min_cash_pct", 0.20)
        self._accuracy_cache: dict | None = None

    # ─── 1. 자본배분 (Kelly) ────────────────────────────

    def allocate_capital(
        self,
        signal: dict,
        portfolio: dict,
        win_rate: float = 0.55,
        avg_win: float = 0.08,
        avg_loss: float = 0.04,
    ) -> CapitalAllocation:
        """Quarter-Kelly 기반 자본 배분 결정.

        Args:
            signal: {ticker, name, grade, sector, entry_price, stop_loss}
            portfolio: {total_capital, cash, positions: [{ticker,name,sector,weight_pct,pnl_pct}]}
            win_rate: 시그널 소스 적중률
            avg_win: 평균 수익률 (양수)
            avg_loss: 평균 손실률 (양수)
        """
        ticker = signal.get("ticker", "")
        sector = signal.get("sector", "기타")
        total_capital = portfolio.get("total_capital", 0)
        positions = portfolio.get("positions", [])

        # Kelly Criterion: f* = (p*b - q) / b  (b=avg_win/avg_loss)
        if avg_loss <= 0 or avg_win <= 0:
            kelly_f = 0.0
        else:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly_f = max((win_rate * b - q) / b, 0.0)

        quarter_kelly = kelly_f * self.KELLY_FRACTION

        # 패널티 계산
        corr_penalty = self._calc_correlation_penalty(sector, positions)
        dd_penalty = self._calc_drawdown_penalty(portfolio)
        conc_penalty = self._calc_concentration_penalty(ticker, positions)

        # 최종 사이즈
        raw_pct = min(quarter_kelly, self.MAX_SINGLE_POSITION_PCT)
        final_pct = raw_pct * (1 - corr_penalty) * (1 - dd_penalty) * (1 - conc_penalty)
        final_pct = max(final_pct, 0.0)

        # 현금 제한
        cash = portfolio.get("cash", 0)
        max_invest = cash - total_capital * self.MIN_CASH_PCT
        if max_invest <= 0:
            final_pct = 0.0
            reasoning = "현금 비중 부족으로 매수 불가"
        else:
            max_pct = max_invest / total_capital if total_capital > 0 else 0
            final_pct = min(final_pct, max_pct)
            reasoning = self._build_allocation_reasoning(
                kelly_f, quarter_kelly, corr_penalty, dd_penalty, conc_penalty, final_pct
            )

        return CapitalAllocation(
            ticker=ticker,
            name=signal.get("name", ""),
            recommended_size_pct=round(quarter_kelly * 100, 2),
            recommended_amount=round(final_pct * total_capital),
            kelly_fraction=round(kelly_f, 4),
            quarter_kelly=round(quarter_kelly, 4),
            correlation_penalty=round(corr_penalty, 3),
            drawdown_penalty=round(dd_penalty, 3),
            concentration_penalty=round(conc_penalty, 3),
            final_size_pct=round(final_pct * 100, 2),
            reasoning=reasoning,
        )

    def _calc_correlation_penalty(self, sector: str, positions: list) -> float:
        """동일 섹터 보유 시 패널티"""
        same_sector_weight = sum(
            p.get("weight_pct", 0) for p in positions
            if p.get("sector", "기타") == sector
        )
        if same_sector_weight >= self.MAX_SECTOR_PCT * 100:
            return 1.0  # 완전 차단
        elif same_sector_weight >= self.MAX_SECTOR_PCT * 100 * 0.7:
            return 0.5  # 50% 감축
        elif same_sector_weight > 0:
            return 0.2  # 20% 감축
        return 0.0

    def _calc_drawdown_penalty(self, portfolio: dict) -> float:
        """낙폭에 따른 패널티"""
        dd = portfolio.get("current_drawdown_pct", 0)
        if dd <= self.MDD_EMERGENCY_THRESHOLD:
            return 1.0
        elif dd <= self.MDD_HALT_THRESHOLD:
            return 0.8
        elif dd <= self.MDD_REDUCE_THRESHOLD:
            return 0.4
        elif dd <= -0.05:
            return 0.2
        return 0.0

    def _calc_concentration_penalty(self, ticker: str, positions: list) -> float:
        """포지션 수 기반 집중도 패널티"""
        count = len(positions)
        if count >= 8:
            return 0.5
        elif count >= 6:
            return 0.3
        elif count >= 4:
            return 0.1
        return 0.0

    def _build_allocation_reasoning(
        self, kelly_f, quarter_kelly, corr_p, dd_p, conc_p, final
    ) -> str:
        parts = [f"Kelly={kelly_f:.1%}, QK={quarter_kelly:.1%}"]
        if corr_p > 0:
            parts.append(f"섹터중복 -{corr_p:.0%}")
        if dd_p > 0:
            parts.append(f"낙폭감산 -{dd_p:.0%}")
        if conc_p > 0:
            parts.append(f"집중감산 -{conc_p:.0%}")
        parts.append(f"→ 최종 {final:.1%}")
        return " | ".join(parts)

    # ─── 2. 포트폴리오 건강 진단 ─────────────────────────

    def health_check(self, portfolio: dict, regime: str = "CAUTION") -> PortfolioHealthCheck:
        """포트폴리오 건강 점수 (0~100)"""
        positions = portfolio.get("positions", [])
        total_capital = portfolio.get("total_capital", 0)
        cash = portfolio.get("cash", 0)
        cash_ratio = cash / total_capital if total_capital > 0 else 1.0

        # 업종별 비중
        sector_weights: dict[str, float] = {}
        max_holding_pct = 0.0
        for p in positions:
            sec = p.get("sector", "기타")
            w = p.get("weight_pct", 0)
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            max_holding_pct = max(max_holding_pct, w)

        # HHI (Herfindahl-Hirschman Index)
        total_weight = sum(sector_weights.values())
        if total_weight > 0:
            hhi = sum((w / total_weight) ** 2 for w in sector_weights.values())
        else:
            hhi = 0.0

        max_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else ""
        max_sector_pct = sector_weights.get(max_sector_name, 0)

        # VaR 추정 (간이: 포지션 수 기반)
        n = len(positions)
        base_var = -0.02 * (1 + n * 0.3)  # 포지션 많을수록 VaR 증가
        if hhi > 0.5:
            base_var *= 1.3  # 집중도 높으면 VaR 증가

        # 점수 계산 (100점)
        score = 100.0
        warnings = []
        recommendations = []

        # 1. 현금 비중 (20점)
        if regime in ("BEAR", "PRE_BEAR", "CRISIS", "PRE_CRISIS"):
            min_cash = 0.50
        else:
            min_cash = self.MIN_CASH_PCT

        if cash_ratio < min_cash:
            deduction = min((min_cash - cash_ratio) / min_cash * 20, 20)
            score -= deduction
            warnings.append(f"현금 비중 {cash_ratio:.0%} < 권고 {min_cash:.0%}")
            recommendations.append(f"현금 비중을 {min_cash:.0%} 이상으로 확보 필요")

        # 2. 섹터 집중도 (20점)
        if max_sector_pct > self.MAX_SECTOR_PCT * 100:
            score -= 20
            warnings.append(f"{max_sector_name} 섹터 {max_sector_pct:.0f}% 초과 집중")
            recommendations.append(f"{max_sector_name} 비중 축소 또는 타 섹터 분산 필요")
        elif max_sector_pct > self.MAX_SECTOR_PCT * 100 * 0.7:
            score -= 10
            warnings.append(f"{max_sector_name} 섹터 {max_sector_pct:.0f}% 주의")

        # 3. 단일 종목 비중 (20점)
        if max_holding_pct > self.MAX_SINGLE_POSITION_PCT * 100:
            score -= 15
            warnings.append(f"단일종목 최대 {max_holding_pct:.1f}% → 20% 초과")
            recommendations.append("최대 비중 종목 일부 매도 권고")

        # 4. HHI (20점)
        if hhi > 0.5:
            score -= 15
            warnings.append(f"HHI {hhi:.2f} — 포트폴리오 극도 집중")
        elif hhi > 0.3:
            score -= 8
            warnings.append(f"HHI {hhi:.2f} — 집중도 높음")

        # 5. 포지션 수 (20점)
        if n == 0:
            score -= 5
            warnings.append("보유 종목 없음 (전액 현금)")
        elif n <= 2:
            score -= 10
            warnings.append(f"보유 {n}종목 — 분산 부족")
            recommendations.append("3~5종목으로 분산 권고")
        elif n > 8:
            score -= 5
            warnings.append(f"보유 {n}종목 — 과도한 분산으로 관리 어려움")

        # 리스크 레벨 결정
        if score >= 80:
            risk_level = "conservative"
        elif score >= 60:
            risk_level = "moderate"
        elif score >= 40:
            risk_level = "aggressive"
        else:
            risk_level = "emergency"

        return PortfolioHealthCheck(
            overall_score=round(max(score, 0), 1),
            risk_level=risk_level,
            positions_count=n,
            sector_concentration=round(hhi, 3),
            top_holding_pct=round(max_holding_pct, 1),
            estimated_var_95=round(base_var, 4),
            cash_ratio=round(cash_ratio, 3),
            max_sector_pct=round(max_sector_pct, 1),
            max_sector_name=max_sector_name,
            warnings=warnings,
            recommendations=recommendations,
        )

    # ─── 3. 낙폭 분석 ──────────────────────────────────

    def drawdown_analysis(
        self, equity_curve: list[dict], regime: str = "CAUTION"
    ) -> DrawdownAnalysis:
        """낙폭 분석 + 대응 판단.

        Args:
            equity_curve: [{date, equity}, ...] 일별 자산 곡선
            regime: 현재 KOSPI 레짐
        """
        if not equity_curve:
            return DrawdownAnalysis(action="normal", action_label="유지", reasoning="데이터 없음")

        equities = [p.get("equity", 0) for p in equity_curve]
        peak = max(equities) if equities else 0
        current = equities[-1] if equities else 0

        if peak <= 0:
            return DrawdownAnalysis(action="normal", action_label="유지", reasoning="유효한 자산 데이터 없음")

        current_dd = (current / peak) - 1
        max_dd = current_dd  # 현재 DD가 최대

        # 낙폭 지속 일수
        dd_start = len(equities) - 1
        for i in range(len(equities) - 1, -1, -1):
            if equities[i] >= peak:
                dd_start = i
                break
        duration = len(equities) - 1 - dd_start

        # 회복 예상 (단순 추정: 낙폭률 × 2 영업일)
        recovery_est = max(int(abs(current_dd) * 100 * 2), 0)

        # 대응 판단 (레짐 보정)
        regime_mult = 1.0
        if regime in ("BEAR", "PRE_BEAR"):
            regime_mult = 0.8  # 하락장에선 더 보수적
        elif regime in ("CRISIS", "PRE_CRISIS"):
            regime_mult = 0.6

        adjusted_dd = current_dd / regime_mult if regime_mult > 0 else current_dd

        if adjusted_dd <= self.MDD_EMERGENCY_THRESHOLD:
            action, label = "emergency", "긴급 청산"
            reasoning = f"낙폭 {current_dd:.1%}이 긴급 기준 {self.MDD_EMERGENCY_THRESHOLD:.0%} 초과. 레짐={regime}. 전량 청산 권고"
        elif adjusted_dd <= self.MDD_HALT_THRESHOLD:
            action, label = "halt", "매수 중단"
            reasoning = f"낙폭 {current_dd:.1%}이 중단 기준 {self.MDD_HALT_THRESHOLD:.0%} 초과. 신규 매수 전면 중단"
        elif adjusted_dd <= self.MDD_REDUCE_THRESHOLD:
            action, label = "reduce", "비중 축소"
            reasoning = f"낙폭 {current_dd:.1%}이 경고 기준 {self.MDD_REDUCE_THRESHOLD:.0%} 초과. 포지션 비중 축소 권고"
        else:
            action, label = "normal", "유지"
            reasoning = f"낙폭 {current_dd:.1%} — 정상 범위. 기존 전략 유지"

        return DrawdownAnalysis(
            current_drawdown_pct=round(current_dd, 4),
            max_drawdown_pct=round(max_dd, 4),
            drawdown_duration_days=duration,
            recovery_estimate_days=recovery_est,
            action=action,
            action_label=label,
            reasoning=reasoning,
        )

    # ─── 4. 통합 리포트 ─────────────────────────────────

    def full_report(
        self,
        portfolio: dict,
        equity_curve: list[dict] | None = None,
        regime: str = "CAUTION",
    ) -> CFOReport:
        """CFO 전체 리포트 생성.

        Args:
            portfolio: {total_capital, cash, current_drawdown_pct,
                        positions: [{ticker,name,sector,weight_pct,pnl_pct}]}
            equity_curve: [{date, equity}, ...]
            regime: KOSPI 레짐
        """
        health = self.health_check(portfolio, regime)
        drawdown = self.drawdown_analysis(equity_curve or [], regime)

        # 배분 예산 계산
        total = portfolio.get("total_capital", 0)
        cash = portfolio.get("cash", 0)
        investable = max(cash - total * self.MIN_CASH_PCT, 0)

        if drawdown.action == "emergency":
            max_new_invest = 0
        elif drawdown.action == "halt":
            max_new_invest = 0
        elif drawdown.action == "reduce":
            max_new_invest = investable * 0.3
        else:
            max_new_invest = investable

        budget = {
            "total_capital": total,
            "cash": cash,
            "investable": round(investable),
            "max_new_invest": round(max_new_invest),
            "action_override": drawdown.action,
            "regime": regime,
        }

        return CFOReport(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            health=health,
            drawdown=drawdown,
            allocation_budget=budget,
        )


# ═══════════════════════════════════════════════════════
#  CLI (독립 실행 + 텔레그램 발송)
# ═══════════════════════════════════════════════════════

def _load_portfolio_data() -> tuple[dict, list[dict]]:
    """paper_portfolio.json + equity_tracker.json → (portfolio_dict, equity_curve)"""
    # 페이퍼 포트폴리오
    pp_path = DATA_DIR / "paper_portfolio.json"
    pp = {}
    if pp_path.exists():
        with open(pp_path, encoding="utf-8") as f:
            pp = json.load(f)

    # 섹터맵 로드
    sector_map = _load_sector_map()

    total_capital = pp.get("initial_capital", 30_000_000)
    cash = pp.get("capital", total_capital)
    positions_raw = pp.get("positions", {})

    # 포지션 → 리스트 변환 + 비중 계산
    invested = sum(p.get("cost", 0) for p in positions_raw.values())
    portfolio_value = cash + invested
    positions = []
    for ticker, p in positions_raw.items():
        cost = p.get("cost", 0)
        weight = (cost / portfolio_value * 100) if portfolio_value > 0 else 0
        pnl = 0.0  # 실시간 가격 없이는 비용 기준
        positions.append({
            "ticker": ticker,
            "name": p.get("name", ""),
            "sector": sector_map.get(ticker, "기타"),
            "weight_pct": round(weight, 1),
            "pnl_pct": round(pnl, 1),
            "cost": cost,
        })

    portfolio = {
        "total_capital": portfolio_value,
        "cash": cash,
        "invested": invested,
        "positions": positions,
        "current_drawdown_pct": (cash + invested) / total_capital - 1 if total_capital > 0 else 0,
    }

    # 자산 곡선 (equity_tracker.json)
    eq_path = DATA_DIR / "equity_tracker.json"
    equity_curve = []
    if eq_path.exists():
        with open(eq_path, encoding="utf-8") as f:
            eq_data = json.load(f)
        if isinstance(eq_data, list):
            equity_curve = eq_data
        elif isinstance(eq_data, dict) and "history" in eq_data:
            equity_curve = eq_data["history"]

    return portfolio, equity_curve


def _load_sector_map() -> dict[str, str]:
    """종목→업종 매핑 로드"""
    csv_path = DATA_DIR / "universe" / "sector_map.csv"
    sector_map = {}
    if csv_path.exists():
        try:
            import csv
            with open(csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get("ticker", row.get("종목코드", "")).strip()
                    sector = row.get("sector", row.get("업종", "기타")).strip()
                    if ticker:
                        sector_map[ticker] = sector
        except Exception as e:
            logger.warning(f"sector_map 로드 실패: {e}")
    return sector_map


def _load_regime() -> str:
    """brain_decision.json에서 현재 레짐 로드"""
    brain_path = DATA_DIR / "brain_decision.json"
    if brain_path.exists():
        try:
            with open(brain_path, encoding="utf-8") as f:
                brain = json.load(f)
            return brain.get("regime", "CAUTION")
        except Exception:
            pass
    return "CAUTION"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CFO 포트폴리오 리스크 리포트")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    portfolio, equity_curve = _load_portfolio_data()
    regime = _load_regime()

    cfo = PortfolioCFO()
    report = cfo.full_report(portfolio, equity_curve, regime)

    # JSON 저장
    out_path = DATA_DIR / "cfo_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    h = report.health
    d = report.drawdown
    b = report.allocation_budget

    print(f"\n{'='*50}")
    print(f"  CFO 포트폴리오 리포트 — {report.generated_at}")
    print(f"{'='*50}")
    print(f"\n[건강 진단] 점수: {h.overall_score}/100 ({h.risk_level})")
    print(f"  보유 {h.positions_count}종목 | 현금 {h.cash_ratio:.0%} | HHI {h.sector_concentration:.2f}")
    if h.max_sector_name:
        print(f"  최대 섹터: {h.max_sector_name} {h.max_sector_pct:.0f}%")
    for w in h.warnings:
        print(f"  ⚠ {w}")
    for r in h.recommendations:
        print(f"  → {r}")

    print(f"\n[낙폭 분석] DD: {d.current_drawdown_pct:.1%} | 지속 {d.drawdown_duration_days}일")
    print(f"  판단: {d.action_label} ({d.action})")
    print(f"  근거: {d.reasoning}")

    print(f"\n[배분 예산]")
    print(f"  총자본: {b['total_capital']:,.0f}원 | 현금: {b['cash']:,.0f}원")
    print(f"  투자가능: {b['investable']:,.0f}원 | 최대신규: {b['max_new_invest']:,.0f}원")
    print(f"  레짐: {b['regime']} | 오버라이드: {b['action_override']}")

    # 텔레그램 발송
    if args.send:
        try:
            from src.adapters.telegram_adapter import send_message
            msg = _format_telegram(report)
            send_message(msg)
            print("\n[텔레그램] 발송 완료")
        except Exception as e:
            print(f"\n[텔레그램] 발송 실패: {e}")

    print(f"\n[저장] {out_path}")
    return report


def _format_telegram(report: CFOReport) -> str:
    """텔레그램 메시지 포매팅"""
    h = report.health
    d = report.drawdown
    b = report.allocation_budget

    icon = {"conservative": "🟢", "moderate": "🟡", "aggressive": "🟠", "emergency": "🔴"}
    lines = [
        f"{icon.get(h.risk_level, '⚪')} CFO 리포트 — {h.overall_score:.0f}/100",
        f"보유 {h.positions_count}종목 | 현금 {h.cash_ratio:.0%} | HHI {h.sector_concentration:.2f}",
    ]
    if h.warnings:
        lines.append("⚠ " + " | ".join(h.warnings[:3]))

    dd_icon = {"normal": "✅", "reduce": "⚠️", "halt": "🛑", "emergency": "🆘"}
    lines.append(f"\n{dd_icon.get(d.action, '❓')} 낙폭 {d.current_drawdown_pct:.1%} → {d.action_label}")

    lines.append(f"\n💰 투자가능: {b['max_new_invest']:,.0f}원 (레짐: {b['regime']})")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
