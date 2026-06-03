"""거시 조기신호 수집 + 경고 정의 (read-only / shadow).

사장님 ② (2026-06-03): 거시 신호가 self C60(가격)보다 먼저 2022형 약세전환
(시나리오 B)을 켰는가? 어제 regime_obs_adversarial 방법론(base_rate/precision/
honest_lead)을 거시로 확장하기 위한 데이터 레이어.

신호 (fdr FRED, 2021~ 가용 = 2022 약세장 커버):
  DGS10        10년물 국채금리
  DGS2         2년물
  T10Y2Y       10Y-2Y 스프레드 (음수=장단기 역전, 침체 선행)
  VIXCLS       VIX 변동성
  DTWEXBGS     달러인덱스(broad)
  BAMLH0A0HYM2 HY 스프레드 ★영상 핵심 — 단 이 환경 fdr/FRED는 2023-06~만 제공
               → 2022 선행성 검증 불가, 2023-06~ 현재 모니터링용으로만.

주문 어댑터 import/호출 없음 (실주문 0).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 2022 약세장을 덮는 신호 (선행성 검증 대상)
FRED_CORE = {
    "DGS10": "10년물",
    "DGS2": "2년물",
    "T10Y2Y": "장단기스프레드",
    "VIXCLS": "VIX",
    "DTWEXBGS": "달러인덱스",
}
# 2023-06~ 만 가용 (현재 모니터링용, 2022 검증 불가)
FRED_SHORT = {"BAMLH0A0HYM2": "HY스프레드"}

# 경고 임계 (초기값 — 적대검증 base_rate로 상시점등 여부 확인 후 조정)
VIX_WARN_LEVEL = 25.0
DXY_SURGE_PCT = 0.03      # 60거래일 달러 +3%
RATE_SURGE_BP = 0.5       # 60거래일 10Y +0.5%p
HY_WIDEN_BP = 1.0         # 120일 저점 대비 HY +1.0%p


def load_macro(start: str = "2021-01-01", end: str = "2026-06-02",
               include_hy: bool = True) -> pd.DataFrame:
    """fdr FRED로 거시 시계열 로드 후 일별 정렬·ffill."""
    import FinanceDataReader as fdr

    series = dict(FRED_CORE)
    if include_hy:
        series.update(FRED_SHORT)

    cols = {}
    for sid in series:
        try:
            df = fdr.DataReader(f"FRED:{sid}", start, end)
            if df is not None and not df.empty:
                cols[sid] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED load failed %s: %s", sid, exc)

    macro = pd.DataFrame(cols).sort_index()
    macro.index = pd.to_datetime(macro.index)
    macro = macro.ffill()
    return macro


def attach_warnings(macro: pd.DataFrame) -> pd.DataFrame:
    """거시 경고 bool 컬럼 추가 (rising edge로 선행성 측정 가능)."""
    m = macro.copy()

    if "VIXCLS" in m:
        m["vix_warn"] = m["VIXCLS"] > VIX_WARN_LEVEL
    if "T10Y2Y" in m:
        m["curve_invert_warn"] = m["T10Y2Y"] < 0.0  # 장단기 역전
    if "DTWEXBGS" in m:
        m["dxy_surge_warn"] = m["DTWEXBGS"].pct_change(60) > DXY_SURGE_PCT
    if "DGS10" in m:
        m["rate_surge_warn"] = m["DGS10"].diff(60) > RATE_SURGE_BP
    if "BAMLH0A0HYM2" in m:
        base = m["BAMLH0A0HYM2"].rolling(120, min_periods=20).min()
        m["hy_widen_warn"] = m["BAMLH0A0HYM2"] > base + HY_WIDEN_BP

    warn_cols = [c for c in m.columns if c.endswith("_warn")]
    for c in warn_cols:
        m[c] = m[c].fillna(False).astype(bool)
    return m


WARN_LABEL = {
    "vix_warn": "VIX>25",
    "curve_invert_warn": "장단기역전",
    "dxy_surge_warn": "달러급등",
    "rate_surge_warn": "금리급등(10Y)",
    "hy_widen_warn": "HY확대(2023-06~)",
}
