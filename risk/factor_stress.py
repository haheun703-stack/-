"""risk/factor_stress.py — 팩터 노출 × 스트레스 시나리오 오케스트레이터. §3.1 + §4.1 연결.

factor_exposure(§3.1)가 추정한 종목 팩터 베타를 stress_test(§4.1)에 주입해, 베타 미주입으로
휴면이던 가상 S1(FX)/S2(반도체)/S5(복합)와 역사 재생 H1~H5를 한 번에 실평가한다. 그동안
두 모듈은 데이터구조만 호환되게 설계돼 있었고(betas_for 출력 = run_stress_test 입력), 실제로
잇는 호출처가 없어 production에서 휴면이었다 — 이 모듈이 그 빠진 연결.

★스펙 §4.1 line 230 구현: "종목 데이터가 해당 일자에 없으면(상장 전) → 팩터 노출 × 당시 팩터
  충격으로 근사한다. 이것이 팩터 분해(3.1)의 실전 용도다."
  - 가상 S1/S2/S3/S5: betas_for(fx/semi/market)를 그대로 주입(stress_test가 충격 상수 보유).
  - 역사 H1~H5: 해당 일자의 팩터 충격 × 종목 베타 → 종목별 근사 수익률. 상장돼 그날 실제
    수익률이 있으면 그것을 우선(스펙 1순위 = "실제 종목 수익률 적용"), 없으면 팩터 근사(폴백).

★잔차화 정합성(핵심): factor_exposure는 semi·smallcap을 시장 잔차로 만들어 베타를 추정한다
  (공선성 제거). 따라서 H 시나리오 충격도 동일한 build_factor_panel(잔차화 + 센터링)을 거친
  그날 값을 써야 베타와 단위가 맞는다 — raw 충격을 잔차 베타에 곱하면 시장 성분이 이중 계산된다.
  같은 패널을 재사용(import)해 정합을 보장한다. 센터링은 회귀와 동일 변환이므로 기울기(베타)
  불변이고, 일별 수익률 평균≈0이라 충격 수준 왜곡도 무시 가능.

★게이트(G1~G8)가 아니라 L3 모니터 오케스트레이터. 순수 계산 — 실주문 경로 접촉 0, write 0,
  risk(config·factor_exposure·stress_test) 외 미import. factor_exposure·stress_test 자체는 무변경
  (이 모듈만 둘을 안다 = 격리 유지). 데이터 미주입(factor_returns 없음/짧아 H 일자 미포함)이면
  H1~H5 graceful 생략 = 기존 stress_test 동작 보존(과소평가 방지). production 미배선
  (factor_returns 데이터 파이프라인은 별 트랙 = freeze 유지).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig
from risk.factor_exposure import (
    FACTORS,
    FactorExposureReport,
    build_factor_panel,
    compute_factor_exposure,
)
# _HISTORICAL은 stress_test가 소유하는 역사 일자 표(단일 진실원천). 같은 risk 패키지 내부
# 모듈 간 재사용으로 일자 중복 정의(불일치 위험)를 피한다.
from risk.stress_test import _HISTORICAL, StressReport, run_stress_test


@dataclass(frozen=True)
class FactorStressResult:
    """팩터 노출 리포트 + 그 베타로 평가한 스트레스 리포트(쌍).

    exposure로 "포트가 무엇에 베팅 중인가"(노출)를, stress로 "그 베팅이 위기에 얼마 잃는가"
    (리스크)를 함께 본다 — §0-4 "노출이 아니라 리스크를 본다"의 입구→출구.
    """

    exposure: FactorExposureReport
    stress: StressReport


def _actual_return(
    returns_by_ticker: dict[str, pd.Series] | None, ticker: str, day_ts: pd.Timestamp
) -> float | None:
    """그 일자의 실제 종목 수익률(있으면). 스펙 §4.1 "실제 종목 수익률 적용" 1순위.

    returns_by_ticker는 일별 단순수익률 Series(close.pct_change). 그 일자가 인덱스에 없으면
    (상장 전/휴장/시계열이 짧음) None → 호출부가 팩터 근사로 폴백.
    """
    if not returns_by_ticker:
        return None
    s = returns_by_ticker.get(ticker)
    if s is None or len(s) == 0:
        return None
    try:
        if day_ts in s.index:
            v = float(s.loc[day_ts])
            return v if np.isfinite(v) else None
    except Exception:  # noqa: BLE001 — 인덱스 타입 불일치/중복 등은 '실측 없음'으로 흘려보낸다
        return None
    return None


def factor_historical_shocks(
    report: FactorExposureReport,
    factor_returns: dict[str, pd.Series],
    *,
    returns_by_ticker: dict[str, pd.Series] | None = None,
) -> dict[str, dict[str, float]]:
    """역사 H1~H5의 종목별 충격(수익률)을 구성 → run_stress_test(historical_shocks=)용.

    각 H 일자에 대해:
      1. build_factor_panel(잔차화 + 센터링)에서 그날 팩터 벡터를 실측(베타와 정합 — 모듈 주석 참조).
      2. 종목별: 실제 그날 수익률 우선(_actual_return), 없으면 Σ 베타 × 팩터충격 근사.
    factor_returns가 그 일자를 포함하지 않으면(시계열이 짧아 2008 등 미포함) 그 H는 생략(graceful).

    Returns:
        {scenario_id(H1..): {ticker: 근사/실측 수익률}}. 평가 가능한 종목 충격이 있는 H만 포함.
        (run_stress_test는 보유 종목 중 충격 없는 게 하나라도 있으면 그 H를 미평가 처리 = 과소평가 방지.)
    """
    panel = build_factor_panel(factor_returns)
    if panel is None or not report.stock_betas:
        return {}
    shocks: dict[str, dict[str, float]] = {}
    for sid, _label, day in _HISTORICAL:
        try:
            day_ts = pd.Timestamp(day)
        except Exception:  # noqa: BLE001 — 파싱 불가 일자는 건너뜀(상수표 오타 방어)
            continue
        if day_ts not in panel.index:
            continue  # 그 일자 팩터 데이터 없음 → 이 H는 graceful 생략(시계열이 거기까지 길지 않음)
        frow = panel.loc[day_ts]
        per_ticker: dict[str, float] = {}
        for sb in report.stock_betas:
            actual = _actual_return(returns_by_ticker, sb.ticker, day_ts)
            if actual is not None:
                per_ticker[sb.ticker] = actual  # 스펙 1순위: 상장돼 있으면 실제값
            else:
                # 팩터 근사: Σ_f β_f × (잔차화된 그날 팩터 충격). 상장 전 종목도 베타만 있으면 커버.
                per_ticker[sb.ticker] = float(
                    sum(sb.betas[f] * float(frow[f]) for f in FACTORS)
                )
        if per_ticker:
            shocks[sid] = per_ticker
    return shocks


def run_factor_stress_test(
    holdings: dict[str, float] | None,
    returns_by_ticker: dict[str, pd.Series],
    factor_returns: dict[str, pd.Series],
    *,
    cfg: RiskConfig = RISK_CONFIG,
) -> FactorStressResult:
    """factor_exposure → stress_test 완전체. 베타 자동 주입 + H1~H5 팩터 근사.

    Args:
        holdings: {ticker: weight} 포트 비중(gross 대비). 빈/None → 빈 노출 + S3/S4만 평가되는 stress.
        returns_by_ticker: {ticker: 일별수익률} — gate_wiring이 VaR(G1/G2)용으로 이미 로드한 것 재사용.
        factor_returns: {factor: 일별수익률/변화율} — market 필수. 역사 H는 이 시계열이 해당 일자까지
            길어야 평가(짧으면 graceful 생략). 미주입/market 없음 → 노출·베타 평가 불가 → S3/S4만.

    Returns:
        FactorStressResult(exposure, stress). stress.worst로 최악 시나리오 조회.
    """
    report = compute_factor_exposure(holdings, returns_by_ticker, factor_returns, cfg=cfg)
    hist = factor_historical_shocks(report, factor_returns, returns_by_ticker=returns_by_ticker)
    stress = run_stress_test(
        holdings,
        historical_shocks=hist or None,
        # 빈 dict → None으로(미주입 = 명시적 미평가). 종목 베타가 있으면 betas_for는 non-empty.
        fx_betas=report.betas_for("fx") or None,
        semi_betas=report.betas_for("semi") or None,
        market_betas=report.betas_for("market") or None,
        cfg=cfg,
    )
    return FactorStressResult(exposure=report, stress=stress)
