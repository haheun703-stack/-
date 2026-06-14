"""risk/crowding.py — 크라우딩/동질화 모니터. RISK_ENGINE Phase 4 / 스펙 §4.4.

우리 신호는 수급(외인·기관 흐름) 기반이다. 즉 **같은 신호를 보는 자금과 같이 들어가고
같이 나가게 된다.** 알라딘의 "리스크 동질화" 문제의 우리 버전 — 포트폴리오가 겉보기엔
분산돼 있어도 위기 때 한 덩어리로 무너지면 분산이 아니다.

모니터 항목(§4.4):
  C1  보유 종목 간 평균 쌍상관(60일) > crowding_corr(0.70)  → "사실상 단일 베팅" 경고
  C2  VKOSPI 20 초과 + 5일 내 30% 이상 급등                 → 레짐 전환 경고
  C3  외인 선물 순포지션이 2년 분위수 상/하위 5%            → 포지션 쏠림 경고
  경고 2개 이상 동시 발생 → 총노출 자동 -20%p (드로다운 사다리 §4.2와 *별도* 곱)

★게이트(G1~G8)가 아니라 L3 노출 조절 모니터다. pre_trade_gate의 PASS/RESIZE/REJECT가
  아니라 총노출 계수(gross_exposure_mult)를 낸다. 호출처(노출 관리 계층)가 사다리 계수·
  변동성 타겟팅(§4.3) 계수와 함께 곱한다.

★C1 vs G5의 차이(혼동 주의):
  - C1(여기)        = **평시 동질성** 스냅샷. 현재 보유들이 지금 같이 움직이는가(균등가중 60일).
  - G5(correlation) = **위기 보수화**. 신규 vs 보유의 ρ_stress(1방향 슈링크)로 군집 → 신규 거부.
  목적이 다르므로 C1은 슈링크를 적용하지 않는다(평시 측정에 위기 가정 주입은 이중 보수).

★순수 계산만 — 실주문 경로 접촉 0, 파일 write 0. risk/config 외 프로젝트 모듈 미import(격리).
  VKOSPI·외인선물 시계열은 호출처가 주입(drawdown_ladder가 DD를 받듯). 데이터 미가용은
  해당 항목을 '미평가'로 흘려보낸다 — 경고로 세지 않는다(데이터 없음 ≠ 위험, 과경고 차단).
  G8과 동일하게 production 배선(데이터 주입) 전까지 휴면 = freeze 유지.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.config import RISK_CONFIG, RiskConfig

# ── 스펙 §4.4 수치 박제(config 미등재 = §9 한도 테이블은 crowding_corr만 등재) ──
_C1_WINDOW = 60          # "평균 쌍상관(60일)" — 최근 60거래일 윈도우
_C1_MIN_OBS = 60         # 공통 표본 최소(correlation._MIN_PAIR_OBS와 동일 보수 기준)
_C2_VKOSPI_LEVEL = 20.0  # "VKOSPI 20 초과"
_C2_SURGE_PCT = 0.30     # "5일 내 30% 이상 급등"
_C2_SURGE_WINDOW = 5     # "5일"(거래일)
_C3_TAIL_PCT = 0.05      # "상/하위 5%"
_C3_MIN_OBS = 252        # 2년 분위수 — 최소 1년 표본 요구(그 미만은 분위수 신뢰 불가)
_WARN_THRESHOLD = 2      # "경고 2개 이상"
_HAIRCUT = 0.20          # "총노출 -20%p"


@dataclass(frozen=True)
class CrowdingState:
    """크라우딩 모니터 판정.

    각 신호는 triggered(경고 발동)와 evaluable(데이터로 평가 가능했는가)를 분리한다.
    evaluable=False는 '위험 없음'이 아니라 '판단 불가'(데이터 미주입) — warning_count에서 제외.
    """

    c1_triggered: bool
    c1_evaluable: bool
    c1_value: float | None      # 평균 쌍상관(평가 가능 시)
    c2_triggered: bool
    c2_evaluable: bool
    c2_value: float | None      # VKOSPI 5일 변화율(평가 가능 시)
    c3_triggered: bool
    c3_evaluable: bool
    c3_value: float | None      # 외인선물 현재값의 경험적 분위수 0~1(평가 가능 시)
    warning_count: int          # 발동 경고 수(0~3)
    exposure_haircut: float     # 노출 감축폭(0.0 또는 0.20)
    gross_exposure_mult: float  # 노출 계수(1.0 또는 0.80) — 사다리·변동성타겟팅과 별도로 곱


def _avg_pairwise_corr(
    holding_returns: dict[str, pd.Series] | None,
    cfg: RiskConfig,
) -> float | None:
    """보유 종목 간 평균 쌍상관(최근 60거래일, 균등가중 피어슨).

    절차: 종목별 수익률을 날짜 정렬 → 공통 날짜(dropna) → 최근 60일 → 쌍상관 행렬의
    상삼각(대각 제외) 평균. 보유 2종목 미만 / 공통 표본<60 / 모든 쌍 분산0 → None(미평가).
    """
    if not holding_returns or len(holding_returns) < 2:
        return None
    df = pd.DataFrame(holding_returns).dropna()
    if len(df) < _C1_MIN_OBS:
        return None
    df = df.iloc[-_C1_WINDOW:]
    corr = df.corr()                       # 피어슨; 상수열(분산0)은 NaN 행/열
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)           # 대각 제외 상삼각
    vals = corr.to_numpy()[iu]
    vals = vals[~np.isnan(vals)]           # 분산0 종목 쌍 제거
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _vkospi_surge(vkospi_series: pd.Series | None) -> float | None:
    """VKOSPI 5거래일 변화율(현재/5일전 - 1). 표본<6 / 5일전 값≤0 → None(미평가)."""
    if vkospi_series is None:
        return None
    s = vkospi_series.dropna()
    if len(s) < _C2_SURGE_WINDOW + 1:
        return None
    current = float(s.iloc[-1])
    past = float(s.iloc[-(_C2_SURGE_WINDOW + 1)])
    if past <= 0.0:
        return None
    return current / past - 1.0


def _empirical_quantile(foreign_futures_series: pd.Series | None) -> float | None:
    """외인 선물 순포지션 현재값의 경험적 분위수(현재값 이하 비율, 0~1).

    표본<252 → None(미평가). 상수(분산0)도 None — 분위수가 무의미한데 (s<=current).mean()이
    1.0을 반환해 거짓 경고가 되는 것을 차단(C1 상수→NaN, C2 상수→surge0과 동일 '미평가' 철학).
    """
    if foreign_futures_series is None:
        return None
    s = foreign_futures_series.dropna()
    if len(s) < _C3_MIN_OBS or s.nunique() <= 1:
        return None
    current = float(s.iloc[-1])
    return float((s <= current).mean())


def crowding_state(
    holding_returns: dict[str, pd.Series] | None = None,
    vkospi_series: pd.Series | None = None,
    foreign_futures_series: pd.Series | None = None,
    *,
    cfg: RiskConfig = RISK_CONFIG,
) -> CrowdingState:
    """크라우딩/동질화 3신호를 평가해 노출 조절 계수를 산출(§4.4).

    Args:
        holding_returns: {ticker: pd.Series(일별 수익률)} 보유 종목들. C1용.
            gate_wiring이 VaR용으로 로드한 returns_by_ticker를 그대로 넘길 수 있다.
        vkospi_series: VKOSPI 종가 시계열(최신이 마지막). C2용. 미주입=C2 미평가.
        foreign_futures_series: 외인 선물 순포지션 시계열(2년치 권장). C3용. 미주입=C3 미평가.
        cfg: crowding_corr(C1 임계) 단일 출처.

    Returns:
        CrowdingState. 평가 가능한 경고가 2개 이상이면 gross_exposure_mult=0.80.
    """
    # C1 — 평균 쌍상관
    c1_value = _avg_pairwise_corr(holding_returns, cfg)
    c1_evaluable = c1_value is not None
    c1_triggered = c1_evaluable and c1_value > cfg.crowding_corr

    # C2 — VKOSPI 레짐 전환(절대 수준 초과 AND 급등 동시)
    surge = _vkospi_surge(vkospi_series)
    c2_evaluable = surge is not None
    c2_triggered = False
    if c2_evaluable:
        current_vkospi = float(vkospi_series.dropna().iloc[-1])
        c2_triggered = current_vkospi > _C2_VKOSPI_LEVEL and surge >= _C2_SURGE_PCT

    # C3 — 외인 선물 순포지션 극단 분위수(쏠림)
    c3_value = _empirical_quantile(foreign_futures_series)
    c3_evaluable = c3_value is not None
    c3_triggered = c3_evaluable and (
        c3_value >= 1.0 - _C3_TAIL_PCT or c3_value <= _C3_TAIL_PCT
    )

    warning_count = int(c1_triggered) + int(c2_triggered) + int(c3_triggered)
    haircut = _HAIRCUT if warning_count >= _WARN_THRESHOLD else 0.0

    return CrowdingState(
        c1_triggered=c1_triggered,
        c1_evaluable=c1_evaluable,
        c1_value=c1_value,
        c2_triggered=c2_triggered,
        c2_evaluable=c2_evaluable,
        c2_value=surge,
        c3_triggered=c3_triggered,
        c3_evaluable=c3_evaluable,
        c3_value=c3_value,
        warning_count=warning_count,
        exposure_haircut=haircut,
        gross_exposure_mult=1.0 - haircut,
    )
