"""risk/factor_exposure.py — 팩터 노출(EWMA 가중 팩터 회귀). RISK_ENGINE 스펙 §3.1.

"노출(exposure)이 아니라 리스크(risk)를 본다"(§0-4)의 입구. 보유 종목을 5개 공통 팩터로
분해해 **포트가 사실상 무엇에 베팅하고 있는지**(예: "반도체 1배 레버리지 상태")를 드러낸다.

팩터 세트(§3.1 — 한국 개인 퀀트 스케일로 압축):
  market   KOSPI200 일별 수익률
  smallcap KOSDAQ 일별 수익률 (KOSPI 잔차 — 시장 성분 제거)
  fx       USD/KRW 일별 변화율
  semi     KRX 반도체 지수 (시장 잔차)
  rate     국고채 3년 금리 일별 변화

방법론(§3.1 — "단순 60일 회귀의 함정 보정"):
  1. EWMA 가중 회귀(halflife=60)로 베타 추정 — 최근성 반영.
  2. 252일 단순(균등가중) 베타도 병행 — 두 market 베타 괴리 ≥50%면 "베타 불안정" 경고.
  3. 종목별 베타 × 비중 합산 → 포트폴리오 팩터 노출 벡터 + 한 줄 해석.

★실전 용도 = stress_test(§4.1) 연료. betas_for("fx"/"semi"/"market")가
  run_stress_test의 fx_betas/semi_betas/market_betas로 그대로 들어가 S1/S2/S3/S5를
  실평가하게 한다(현재 미구현이라 그 시나리오들이 휴면). 상장 전 종목의 역사 재생 근사도
  이 팩터 노출 × 당시 팩터 충격으로 한다(§4.1).

★smallcap·semi를 시장 잔차로 만드는 이유: 시장과 강한 공선성을 제거해야 다변량 회귀의
  베타가 안정된다. 잔차화 후 각 베타는 "시장을 넘어선 추가 노출"을 뜻한다.

★게이트(G1~G8)가 아니라 L2/L3 분석 레이어. 순수 계산 — 실주문 경로 접촉 0, write 0,
  risk/config 외 프로젝트 모듈 미import(격리). 데이터 부족은 graceful로 흘려보낸다
  (종목 베타 추정 불가 → 해당 종목 생략, 팩터 데이터 미주입 → 빈 리포트). production 미배선.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig

# 팩터 순서 고정(회귀 설계행렬 컬럼 순서 = 출력 키 순서).
FACTORS: tuple[str, ...] = ("market", "smallcap", "fx", "semi", "rate")
# 시장 잔차로 만드는 팩터(시장과의 공선성 제거 — §3.1 "KOSPI 잔차"/"시장 잔차").
_RESIDUALIZED = ("smallcap", "semi")
# 불안정 판정 시 분모 floor(0 근처 베타의 % 괴리 폭주 방지).
_BETA_EPS = 1e-6


@dataclass(frozen=True)
class StockFactorBeta:
    """단일 종목의 팩터 베타. unstable=EWMA vs 252일 market 베타 괴리 ≥ 임계."""

    ticker: str
    betas: dict[str, float]   # {factor: beta} — FACTORS 순서
    unstable: bool
    n_obs: int


@dataclass(frozen=True)
class FactorExposureReport:
    """포트폴리오 팩터 노출 리포트."""

    stock_betas: tuple[StockFactorBeta, ...]
    portfolio: dict[str, float]          # {factor: Σ weight×beta} — 평가된 종목만
    covered_weight: float                # 베타가 추정된 종목들의 비중 합(커버리지)
    unstable_tickers: tuple[str, ...]
    interpretation: str

    def betas_for(self, factor: str) -> dict[str, float]:
        """stress_test 주입용 {ticker: beta} (단일 팩터). 미평가 팩터/종목은 제외."""
        if factor not in FACTORS:
            return {}
        return {sb.ticker: sb.betas[factor] for sb in self.stock_betas if factor in sb.betas}


# ── EWMA 가중치 ───────────────────────────────────────────────────────────────
def _halflife_lambda(halflife: int) -> float:
    """반감기(일) → EWMA 감쇠 λ. λ^halflife = 0.5."""
    if halflife <= 0:
        return 1.0
    return float(0.5 ** (1.0 / halflife))


def _ewma_weights(n: int, lam: float) -> np.ndarray:
    """최근 관측에 큰 가중을 주는 정규화 EWMA 가중치(합=1). w_t ∝ λ^(거리)."""
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


# ── 팩터 패널 ─────────────────────────────────────────────────────────────────
def _residualize(target: np.ndarray, market: np.ndarray) -> np.ndarray:
    """target에서 시장 성분 제거: resid = (target-μ) - β(market-μ). 시장 분산 0이면 그대로 센터링.

    β = cov(target,market)/var(market) (단순 OLS). 잔차는 시장과 무상관이 되어 공선성을 없앤다.
    """
    tc = target - float(np.mean(target))
    mc = market - float(np.mean(market))
    var_m = float(np.sum(mc * mc))
    if var_m <= 0.0:
        return tc
    beta = float(np.sum(tc * mc)) / var_m
    return tc - beta * mc


def build_factor_panel(factor_returns: dict[str, pd.Series]) -> pd.DataFrame | None:
    """팩터 수익률 dict → 공통 날짜로 정렬된 설계 패널(컬럼=FACTORS).

    smallcap·semi는 시장(market) 잔차로 변환한다(§3.1). market 결측이면 잔차화 불가 → None.
    누락 팩터 컬럼은 0으로 채운다(그 팩터 베타는 0으로 추정 = 노출 없음으로 안전 처리).

    Args:
        factor_returns: {factor_name: pd.Series(일별 수익률/변화율)}. market 필수 권장.

    Returns:
        DataFrame(index=공통 날짜, columns=FACTORS) 또는 데이터 부족 시 None.
    """
    present = {f: s for f, s in (factor_returns or {}).items()
              if f in FACTORS and s is not None and len(s) > 0}
    if "market" not in present:
        return None
    df = pd.DataFrame(present).dropna()
    if df.empty:
        return None
    market = df["market"].to_numpy(dtype=float)
    out = {}
    for f in FACTORS:
        if f not in df.columns:
            out[f] = np.zeros(len(df), dtype=float)
        elif f in _RESIDUALIZED:
            out[f] = _residualize(df[f].to_numpy(dtype=float), market)
        else:
            out[f] = df[f].to_numpy(dtype=float) - float(np.mean(df[f].to_numpy(dtype=float)))
    return pd.DataFrame(out, index=df.index)


# ── 회귀 ──────────────────────────────────────────────────────────────────────
def _wls_betas(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """가중 다변량 OLS β = argmin Σ w(y - Xβ)². 화이트닝 후 lstsq(SVD, 특이행렬 견고).

    x,y는 가중평균으로 이미 센터링됐다고 가정(절편 없음). 수치 실패 시 None.
    """
    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = y * sw
    try:
        beta, _res, rank, _sv = np.linalg.lstsq(xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta


def _wmean(a: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * a))


def estimate_stock_betas(
    ticker: str,
    stock_returns: pd.Series,
    panel: pd.DataFrame,
    *,
    cfg: RiskConfig = RISK_CONFIG,
) -> StockFactorBeta | None:
    """단일 종목의 팩터 베타(EWMA) + 불안정 플래그. 공통표본<min_obs/수치실패 → None.

    절차: 종목·패널 공통 날짜 → 최근 lookback일 → EWMA 가중평균 센터링 → 가중 다변량 OLS.
    252일 단순(균등) OLS도 계산해 market 베타 괴리 ≥ factor_beta_instability면 unstable=True.
    """
    if stock_returns is None or len(stock_returns) == 0 or panel is None or panel.empty:
        return None
    df = panel.join(stock_returns.rename("_y"), how="inner").dropna()
    if len(df) > cfg.factor_lookback:
        df = df.iloc[-cfg.factor_lookback:]
    if len(df) < cfg.factor_min_obs:
        return None

    x_raw = df[list(FACTORS)].to_numpy(dtype=float)
    y_raw = df["_y"].to_numpy(dtype=float)
    n = len(df)

    # EWMA 가중 회귀(센터링은 가중평균 기준 = 절편 흡수).
    w = _ewma_weights(n, _halflife_lambda(cfg.factor_ewma_halflife))
    xc = x_raw - np.array([_wmean(x_raw[:, j], w) for j in range(x_raw.shape[1])])
    yc = y_raw - _wmean(y_raw, w)
    b_ewma = _wls_betas(xc, yc, w)
    if b_ewma is None:
        return None

    # 252일 단순(균등) 회귀 — 불안정 판정용.
    w_eq = np.full(n, 1.0 / n)
    xc2 = x_raw - x_raw.mean(axis=0)
    yc2 = y_raw - y_raw.mean()
    b_simple = _wls_betas(xc2, yc2, w_eq)

    # 분산 0인 팩터(잔차/누락 컬럼)는 베타 0으로 고정 — lstsq 최소노름 잡음 제거.
    betas: dict[str, float] = {}
    for j, f in enumerate(FACTORS):
        col_var = float(np.sum((xc[:, j]) ** 2))
        betas[f] = 0.0 if col_var <= 0.0 else float(b_ewma[j])

    # 불안정 = market 베타 EWMA vs 단순 괴리 ≥ 임계(가장 지배적 팩터 기준 — 작은 베타 % 폭주 회피).
    unstable = False
    if b_simple is not None:
        mi = FACTORS.index("market")
        denom = max(abs(float(b_simple[mi])), _BETA_EPS)
        if abs(float(b_ewma[mi]) - float(b_simple[mi])) / denom >= cfg.factor_beta_instability:
            unstable = True

    return StockFactorBeta(ticker=ticker, betas=betas, unstable=unstable, n_obs=n)


# ── 포트폴리오 집계 ───────────────────────────────────────────────────────────
def _interpret(portfolio: dict[str, float], covered_weight: float) -> str:
    """노출 벡터 → 한 줄 해석. 지배 팩터(절대값 최대)를 자연어로."""
    if not portfolio or covered_weight <= 0.0:
        return "팩터 노출 평가 불가(베타 추정된 보유 종목 없음)"
    label = {"market": "KOSPI", "smallcap": "KOSDAQ", "fx": "FX",
             "semi": "반도체", "rate": "금리"}
    vec = " | ".join(f"{label[f]} {portfolio[f]:+.2f}" for f in FACTORS if f in portfolio)
    dom = max(portfolio, key=lambda f: abs(portfolio[f]))
    dv = portfolio[dom]
    if abs(dv) < 0.05:
        tail = "뚜렷한 단일 팩터 베팅 없음(분산됨)"
    else:
        tail = f"사실상 {label[dom]} {abs(dv):.1f}배 {'레버리지' if dv > 0 else '역(逆)'} 베팅 상태"
    return f"{vec} → {tail}"


def compute_factor_exposure(
    holdings: dict[str, float] | None,
    returns_by_ticker: dict[str, pd.Series],
    factor_returns: dict[str, pd.Series],
    *,
    cfg: RiskConfig = RISK_CONFIG,
) -> FactorExposureReport:
    """포트폴리오 팩터 노출 리포트.

    Args:
        holdings: {ticker: weight}. None/빈 포트 → 빈 노출 리포트.
        returns_by_ticker: {ticker: pd.Series(일별 수익률)} — gate_wiring이 VaR용으로 이미 로드.
        factor_returns: {factor: pd.Series} — FACTORS 중 가용한 것. market 없으면 평가 불가.

    Returns:
        FactorExposureReport. 베타 추정 가능한 종목만 portfolio에 합산(과소평가 방지 위해
        covered_weight로 커버리지 투명 노출 — 일부만 잡혀도 비중 합으로 신뢰도 판단).
    """
    holdings = holdings or {}
    panel = build_factor_panel(factor_returns)
    if panel is None or not holdings:
        return FactorExposureReport((), {}, 0.0, (), _interpret({}, 0.0))

    stock_betas: list[StockFactorBeta] = []
    portfolio = {f: 0.0 for f in FACTORS}
    covered_weight = 0.0
    unstable: list[str] = []
    for t, wgt in holdings.items():
        sb = estimate_stock_betas(t, returns_by_ticker.get(t), panel, cfg=cfg)
        if sb is None:
            continue
        stock_betas.append(sb)
        covered_weight += wgt
        for f in FACTORS:
            portfolio[f] += wgt * sb.betas[f]
        if sb.unstable:
            unstable.append(t)

    if not stock_betas:
        return FactorExposureReport((), {}, 0.0, (), _interpret({}, 0.0))
    portfolio = {f: round(v, 4) for f, v in portfolio.items()}
    return FactorExposureReport(
        stock_betas=tuple(stock_betas),
        portfolio=portfolio,
        covered_weight=round(covered_weight, 4),
        unstable_tickers=tuple(unstable),
        interpretation=_interpret(portfolio, covered_weight),
    )
