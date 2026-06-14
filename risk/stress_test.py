"""risk/stress_test.py — 스트레스 시나리오 라이브러리. RISK_ENGINE Phase 3 / 스펙 §4.1.

현재 포트폴리오에 역사적/가상 충격을 적용해 예상 손실을 산출한다. 매일 장전에 **최악 시나리오
1개와 예상 손실**을 본다 — "이 포트가 2008/2020/엔캐리 같은 날을 다시 만나면 얼마나 잃는가".

시나리오(스펙 §4.1):
  역사 재생  H1 글로벌 금융위기(2008-10-24) / H2 미 신용등급 강등(2011-08-09) /
            H3 코로나 폭락(2020-03-19) / H4 금리 쇼크(2022-09-26) / H5 엔캐리 청산(2024-08-05)
  가상       S1 USD/KRW +7% / S2 반도체 -15% / S3 KOSPI -8% 시초 갭 /
            S4 최대 비중 종목 하한가(-30%) / S5 S1+S2 복합

★순수 집계 엔진(사장님 승인 = freeze 일관). 시나리오별 종목 충격(역사 수익률·팩터 베타)은
  호출처가 주입한다:
  - S3(시장 갭)·S4(하한가)는 포트 비중만으로 자체 계산 → **항상 평가**(S3은 시장 베타 1 폴백).
  - S1·S2·S5는 종목별 팩터 베타 주입 시 평가. H1~H5는 해당 일자 종목 수익률 주입 시 평가.
    이들의 원천 factor_exposure(§3.1)는 아직 미구현이라 production에서는 대부분 휴면 = freeze 유지
    (상장 전 종목 팩터 근사는 §3.1 별 트랙 후 — 스펙 §4.1 "팩터 노출 × 당시 팩터 충격").
  미주입 시나리오는 '미평가'(crowding·vol_targeting과 동일 graceful, 과소평가 방지).

★게이트(G1~G8)가 아니라 L3 모니터. 순수 계산 — 실주문 접촉 0, write 0, risk/config 외 미import.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk.config import RISK_CONFIG, RiskConfig

# 역사 재생 일자(스펙 §4.1 표). 충격(종목 수익률)은 호출처가 historical_shocks로 주입.
_HISTORICAL = (
    ("H1", "글로벌 금융위기 최악일", "2008-10-24"),
    ("H2", "미국 신용등급 강등", "2011-08-09"),
    ("H3", "코로나 폭락", "2020-03-19"),
    ("H4", "금리 쇼크", "2022-09-26"),
    ("H5", "엔캐리 청산", "2024-08-05"),
)
# 가상 시나리오 충격 크기(스펙 §4.1 표)
_S1_FX_SHOCK = 0.07       # USD/KRW +7% 급등
_S2_SEMI_SHOCK = -0.15    # 반도체 지수 -15%
_S3_KOSPI_GAP = -0.08     # KOSPI -8% 시초 갭(장중 대응 불가 가정)
_S4_LIMIT_DOWN = -0.30    # 보유 최대 비중 종목 하한가


@dataclass(frozen=True)
class ScenarioResult:
    """단일 시나리오 결과. evaluable=False는 충격/베타 미주입('판단 불가', 손실 0 아님)."""

    scenario_id: str
    label: str
    evaluable: bool
    portfolio_pnl: float | None   # 포트폴리오 손익률(음수=손실)
    note: str = ""


@dataclass(frozen=True)
class StressReport:
    results: tuple[ScenarioResult, ...]

    @property
    def worst(self) -> ScenarioResult | None:
        """평가 가능한 시나리오 중 최악(최소 pnl). 평가 가능한 게 없으면 None."""
        ev = [r for r in self.results if r.evaluable and r.portfolio_pnl is not None]
        return min(ev, key=lambda r: r.portfolio_pnl) if ev else None


def _portfolio_impact(
    holdings: dict[str, float],
    per_ticker: dict[str, float] | None,
    multiplier: float,
) -> float | None:
    """Σ weight_i × per_ticker_i × multiplier.

    per_ticker=None(미주입) → None. 보유 종목 중 값 없는 게 하나라도 있으면 None(과소평가 방지 —
    일부만 충격 주면 손실이 작게 나온다). 빈 포트 → 0.0.
    """
    if per_ticker is None:
        return None
    if not holdings:
        return 0.0
    total = 0.0
    for t, w in holdings.items():
        if t not in per_ticker:
            return None
        total += w * per_ticker[t] * multiplier
    return total


def _virtual_result(
    sid: str, label: str, pnl: float | None, betas: dict | None, miss_note: str
) -> ScenarioResult:
    if pnl is not None:
        return ScenarioResult(sid, label, True, pnl, "")
    note = miss_note if betas is None else "보유 종목 베타 일부 누락"
    return ScenarioResult(sid, label, False, None, note)


def run_stress_test(
    holdings: dict[str, float] | None,
    *,
    historical_shocks: dict[str, dict[str, float]] | None = None,
    fx_betas: dict[str, float] | None = None,
    semi_betas: dict[str, float] | None = None,
    market_betas: dict[str, float] | None = None,
    cfg: RiskConfig = RISK_CONFIG,
) -> StressReport:
    """포트폴리오에 §4.1 시나리오 10종을 적용해 손실 리포트를 만든다.

    Args:
        holdings: {ticker: weight} 포트 비중(gross 대비). 빈/None → 손실 0(평가 가능 시나리오).
        historical_shocks: {scenario_id(H1..): {ticker: 당일 수익률}}. 역사 재생용. 미주입=H 미평가.
        fx_betas / semi_betas / market_betas: {ticker: beta}. S1/S2/S3용. 미주입 시
            S1/S2=미평가, S3=시장 베타 1 폴백(항상 평가).

    Returns:
        StressReport(results 10종). report.worst로 최악 시나리오 조회.
    """
    holdings = holdings or {}
    results: list[ScenarioResult] = []

    # 역사 재생 H1~H5
    hist = historical_shocks or {}
    for sid, label, day in _HISTORICAL:
        name = f"{label} ({day})"
        shocks = hist.get(sid)
        if shocks is None:
            results.append(ScenarioResult(sid, name, False, None, "역사 종목 수익률 미주입"))
            continue
        pnl = _portfolio_impact(holdings, shocks, 1.0)
        if pnl is None:
            results.append(ScenarioResult(sid, name, False, None, "보유 종목 충격 일부 누락"))
        else:
            results.append(ScenarioResult(sid, name, True, pnl, ""))

    # S1 USD/KRW +7%
    s1 = _portfolio_impact(holdings, fx_betas, _S1_FX_SHOCK)
    results.append(_virtual_result("S1", "USD/KRW +7% 급등", s1, fx_betas, "환율 베타 미주입"))

    # S2 반도체 -15%
    s2 = _portfolio_impact(holdings, semi_betas, _S2_SEMI_SHOCK)
    results.append(_virtual_result("S2", "반도체 지수 -15%", s2, semi_betas, "반도체 베타 미주입"))

    # S3 KOSPI -8% 시초 갭 — 시장 베타 미주입 시 베타 1 폴백(항상 평가)
    if market_betas is None:
        s3_pnl = _S3_KOSPI_GAP * sum(holdings.values())
        results.append(ScenarioResult("S3", "KOSPI -8% 시초 갭", True, s3_pnl,
                                      "시장 베타 1 가정(factor_exposure 미주입)"))
    else:
        s3_pnl = _portfolio_impact(holdings, market_betas, _S3_KOSPI_GAP)
        results.append(_virtual_result("S3", "KOSPI -8% 시초 갭", s3_pnl, market_betas, ""))

    # S4 최대 비중 종목 하한가(-30%)
    if holdings:
        max_t = max(holdings, key=lambda t: holdings[t])
        results.append(ScenarioResult("S4", "최대 비중 종목 하한가", True,
                                      holdings[max_t] * _S4_LIMIT_DOWN, f"{max_t} -30%"))
    else:
        results.append(ScenarioResult("S4", "최대 비중 종목 하한가", True, 0.0, "보유 없음"))

    # S5 S1+S2 복합 — 둘 다 평가 가능할 때만
    if s1 is not None and s2 is not None:
        results.append(ScenarioResult("S5", "USD/KRW +7% & 반도체 -15% 복합", True, s1 + s2, ""))
    else:
        results.append(ScenarioResult("S5", "USD/KRW +7% & 반도체 -15% 복합", False, None,
                                      "S1·S2 베타 미주입"))

    return StressReport(tuple(results))
