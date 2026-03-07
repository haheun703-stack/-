"""
NIGHTWATCH Engine — 글로벌 시장 야간 수호자

기존 US Overnight L1에 없는 고유 가치 레이어만 계산:
  L0: 선행지표 (HYG 크레딧 + VIX 기간구조)
  L1: 채권 자경단 (금리-주식 교차 분석 + 비토)
  L2: 레짐 전환 선행 (크레딧 스프레드 + MOVE + 수익률 커브)  ← 2D 신규
  L4: 환율 삼각형 (원/달러 + USD/JPY + CNH)

앙상블: 기존 US Overnight Score * 0.70 + NIGHTWATCH Score * 0.30
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NightwatchEngine:
    """NIGHTWATCH 4-Layer 고유 스코어 계산."""

    # 기본값 (settings.yaml로 오버라이드 가능)
    DEFAULT_LAYER_WEIGHTS = {
        "leading": 0.20,
        "regime_transition": 0.20,
        "bond_vigilante": 0.35,
        "fx_triangle": 0.25,
    }

    DEFAULT_BV_CFG = {
        "spy_threshold": -0.01,      # SPY -1%
        "tnx_threshold_bp": 5.0,     # 10Y +5bp
        "veto_score_cap": -0.50,
        "position_cap": 0.30,
    }

    DEFAULT_FX_CFG = {
        "yen_carry_threshold": -1.5,  # USD/JPY -1.5%
        "krw_threshold": 1.0,         # KRW/USD +1.0%
    }

    DEFAULT_HYG_CFG = {
        "warn_ret_5d": -0.02,         # -2%
        "critical_ret_5d": -0.04,     # -4%
    }

    DEFAULT_RT_CFG = {
        "credit_spread_z_warn": 1.5,
        "credit_spread_z_critical": 2.0,
        "move_z_warn": 1.5,
        "move_z_critical": 2.0,
        "curve_inversion_threshold": 0,  # 10Y-3M < 0 = 역전
    }

    def __init__(self, settings: dict | None = None):
        nw = (settings or {}).get("nightwatch", {})
        self.weights = nw.get("layer_weights", self.DEFAULT_LAYER_WEIGHTS)
        self.bv_cfg = nw.get("bond_vigilante", self.DEFAULT_BV_CFG)
        self.fx_cfg = nw.get("fx_triangle", self.DEFAULT_FX_CFG)
        self.hyg_cfg = nw.get("hyg", self.DEFAULT_HYG_CFG)
        self.rt_cfg = nw.get("regime_transition", self.DEFAULT_RT_CFG)

    def compute(self, df, latest, prev) -> dict:
        """NIGHTWATCH 스코어 계산.

        Args:
            df: 전체 parquet DataFrame
            latest: df.iloc[-1] (최근 행)
            prev: df.iloc[-2] (전일 행)

        Returns:
            {
                "nightwatch_score": float (-1.0 ~ +1.0),
                "bond_vigilante_veto": bool,
                "layers": {L0, L1, L2, L4 상세},
            }
        """
        l0 = self._compute_leading(latest, prev, df)
        l1 = self._compute_bond_vigilante(latest, prev)
        l2 = self._compute_regime_transition(latest, prev)
        l4 = self._compute_fx_triangle(latest, prev)

        nw_score = (
            l0["score"] * self.weights.get("leading", 0.20)
            + l1["score"] * self.weights.get("bond_vigilante", 0.35)
            + l2["score"] * self.weights.get("regime_transition", 0.20)
            + l4["score"] * self.weights.get("fx_triangle", 0.25)
        )
        nw_score = max(-1.0, min(1.0, nw_score))

        return {
            "version": "2.0",
            "nightwatch_score": round(nw_score, 4),
            "bond_vigilante_veto": l1["veto"],
            "layers": {
                "L0_leading": l0,
                "L1_bond_vigilante": l1,
                "L2_regime_transition": l2,
                "L4_fx_triangle": l4,
            },
        }

    # ──────────────────────────────────────────
    # L0: 선행지표 (HYG 크레딧 + VIX 기간구조)
    # ──────────────────────────────────────────
    def _compute_leading(self, latest, prev, df) -> dict:
        """PCR 대용: HYG 스프레드 + VIX 기간구조."""
        score = 0.0
        details: dict[str, Any] = {}

        # HYG 5일 수익률
        hyg_ret_5d = _safe_float(latest, "hyg_ret_5d")
        details["hyg_ret_5d"] = _pct_round(hyg_ret_5d)

        warn = self.hyg_cfg.get("warn_ret_5d", -0.02)
        critical = self.hyg_cfg.get("critical_ret_5d", -0.04)

        if hyg_ret_5d is not None:
            if hyg_ret_5d <= critical:
                score -= 0.8  # 크레딧 위기
                details["hyg_status"] = "CRITICAL"
            elif hyg_ret_5d <= warn:
                score -= 0.4  # 크레딧 경고
                details["hyg_status"] = "WARNING"
            elif hyg_ret_5d >= 0.01:
                score += 0.2  # 크레딧 안정
                details["hyg_status"] = "STABLE"
            else:
                details["hyg_status"] = "NORMAL"

        # HYG-SPY 괴리 (HYG가 SPY보다 약세 = 크레딧 선행 경고)
        hyg_spy_div = _safe_float(latest, "hyg_spy_div_5d")
        details["hyg_spy_divergence"] = _pct_round(hyg_spy_div)

        if hyg_spy_div is not None and hyg_spy_div <= -0.02:
            score -= 0.3
            details["credit_divergence"] = True
        else:
            details["credit_divergence"] = False

        # VIX 기간구조 (VIX / VIX3M)
        vix_term = _safe_float(latest, "vix_term_ratio")
        details["vix_term_ratio"] = round(vix_term, 4) if vix_term else None

        if vix_term is not None:
            if vix_term > 1.0:
                # Backwardation: 단기 공포 > 장기 → 극단 공포
                score -= 0.3
                details["vix_term_status"] = "BACKWARDATION"
            elif vix_term < 0.85:
                # Deep contango: 시장 안정
                score += 0.2
                details["vix_term_status"] = "CONTANGO"
            else:
                details["vix_term_status"] = "NORMAL"

        score = max(-1.0, min(1.0, score))
        details["score"] = round(score, 4)
        return details

    # ──────────────────────────────────────────
    # L1: 채권 자경단 (금리-주식 교차)
    # ──────────────────────────────────────────
    def _compute_bond_vigilante(self, latest, prev) -> dict:
        """채권 자경단 판정.

        핵심 교차 매트릭스:
        - 주식↓ + 금리↑ = VETO (최악: 자경단 발동)
        - 주식↓ + 금리↓ = 일반 risk-off
        - 주식↑ + 금리 안정 = 정상 랠리
        - 주식↑ + 금리↑ = 과열 경고
        """
        score = 0.0
        veto = False
        details: dict[str, Any] = {}

        spy_ret = _safe_float(latest, "spy_ret_1d")
        tnx_chg = _safe_float(latest, "tnx_change_bp")
        tyx_chg = _safe_float(latest, "tyx_change_bp")
        spread = _safe_float(latest, "yield_spread_10_30")

        details["spy_ret_1d"] = _pct_round(spy_ret)
        details["tnx_change_bp"] = round(tnx_chg, 3) if tnx_chg is not None else None
        details["tyx_change_bp"] = round(tyx_chg, 3) if tyx_chg is not None else None
        details["yield_spread_10_30"] = round(spread, 4) if spread is not None else None

        spy_th = self.bv_cfg.get("spy_threshold", -0.01)
        tnx_th = self.bv_cfg.get("tnx_threshold_bp", 5.0)

        if spy_ret is not None and tnx_chg is not None:
            # 교차 매트릭스 판정
            spy_down = spy_ret <= spy_th          # SPY -1% 이하
            spy_up = spy_ret >= abs(spy_th)       # SPY +1% 이상
            tnx_up = tnx_chg >= tnx_th            # 10Y +5bp 이상
            tnx_down = tnx_chg <= -tnx_th         # 10Y -5bp 이하
            tnx_surge = tnx_chg >= tnx_th * 2     # 10Y +10bp 이상

            if spy_down and tnx_up:
                # VETO: 주식↓ + 금리↑ = 채권 자경단 발동
                score = -0.8
                veto = True
                details["cross_regime"] = "VIGILANTE_VETO"
            elif spy_down and tnx_down:
                # 일반 risk-off
                score = -0.3
                details["cross_regime"] = "RISK_OFF"
            elif spy_down:
                # 단순 조정
                score = -0.2
                details["cross_regime"] = "CORRECTION"
            elif spy_up and not tnx_surge:
                # 정상 랠리
                score = 0.3
                details["cross_regime"] = "NORMAL_RALLY"
            elif spy_up and tnx_surge:
                # 과열 경고 (금리 급등)
                score = -0.1
                details["cross_regime"] = "OVERHEAT_WARNING"
            elif tnx_surge:
                # 주식 보합 + 금리 급등 = 경계
                score = -0.3
                details["cross_regime"] = "RATE_SURGE"
            else:
                score = 0.0
                details["cross_regime"] = "NEUTRAL"

            # 10Y-30Y 스프레드 역전 + 주식 약세 → 추가 비토
            if spread is not None and spread > 0 and spy_ret <= -0.005:
                veto = True
                score = min(score, -0.6)
                details["spread_inversion_veto"] = True
            else:
                details["spread_inversion_veto"] = False
        else:
            details["cross_regime"] = "NO_DATA"

        score = max(-1.0, min(1.0, score))
        details["score"] = round(score, 4)
        details["veto"] = veto
        return details

    # ──────────────────────────────────────────
    # L2: 레짐 전환 선행지표 (2D)
    # ──────────────────────────────────────────
    def _compute_regime_transition(self, latest, prev) -> dict:
        """레짐 전환 사전 감지.

        3가지 선행지표 교차 분석:
        - 크레딧 스프레드 z-score (HY-IG): 급등 = 위기 선행
        - MOVE 인덱스 z-score: 채권시장 공포
        - 10Y-3M 수익률 커브: 역전 = 침체 선행
        """
        score = 0.0
        details: dict[str, Any] = {}

        # ── 크레딧 스프레드 z-score ──
        credit_z = _safe_float(latest, "credit_spread_z")
        details["credit_spread_z"] = round(credit_z, 2) if credit_z is not None else None

        cs_warn = self.rt_cfg.get("credit_spread_z_warn", 1.5)
        cs_crit = self.rt_cfg.get("credit_spread_z_critical", 2.0)

        if credit_z is not None:
            if credit_z >= cs_crit:
                score -= 0.6
                details["credit_status"] = "CRITICAL"
            elif credit_z >= cs_warn:
                score -= 0.3
                details["credit_status"] = "WARNING"
            elif credit_z <= -1.0:
                score += 0.2
                details["credit_status"] = "STABLE"
            else:
                details["credit_status"] = "NORMAL"

        # ── MOVE 인덱스 z-score ──
        move_z = _safe_float(latest, "move_z")
        move_level = _safe_float(latest, "move_level")
        details["move_z"] = round(move_z, 2) if move_z is not None else None
        details["move_level"] = round(move_level, 1) if move_level is not None else None

        mv_warn = self.rt_cfg.get("move_z_warn", 1.5)
        mv_crit = self.rt_cfg.get("move_z_critical", 2.0)

        if move_z is not None:
            if move_z >= mv_crit:
                score -= 0.4
                details["move_status"] = "CRITICAL"
            elif move_z >= mv_warn:
                score -= 0.2
                details["move_status"] = "WARNING"
            elif move_z <= 0:
                score += 0.1
                details["move_status"] = "STABLE"
            else:
                details["move_status"] = "NORMAL"

        # ── 10Y-3M 수익률 커브 ──
        curve = _safe_float(latest, "yield_curve_10_3m")
        details["yield_curve_10_3m"] = round(curve, 3) if curve is not None else None

        inv_th = self.rt_cfg.get("curve_inversion_threshold", 0)

        if curve is not None:
            if curve < inv_th:
                score -= 0.3
                details["curve_status"] = "INVERTED"
            elif curve < 0.5:
                score -= 0.1
                details["curve_status"] = "FLAT"
            elif curve > 1.0:
                score += 0.1
                details["curve_status"] = "NORMAL"
            else:
                details["curve_status"] = "NEUTRAL"

        # ── 이중 경보: 크레딧 + MOVE 동시 경고 ──
        if (credit_z is not None and credit_z >= cs_warn and
                move_z is not None and move_z >= mv_warn):
            score -= 0.3
            details["dual_alarm"] = True
        else:
            details["dual_alarm"] = False

        score = max(-1.0, min(1.0, score))
        details["score"] = round(score, 4)
        return details

    # ──────────────────────────────────────────
    # L4: 환율 삼각형 (원/달러 + USD/JPY + CNH)
    # ──────────────────────────────────────────
    def _compute_fx_triangle(self, latest, prev) -> dict:
        """환율 삼각형: 엔캐리 청산 조기 감지.

        USD/JPY 급락 = 엔캐리 청산 → 글로벌 자금 회수
        원화 약세 + 엔화 강세 동시 = 아시아 risk-off
        """
        score = 0.0
        details: dict[str, Any] = {}

        jpyx_ret = _safe_float(latest, "jpyx_ret_1d")
        krwx_ret = _safe_float(latest, "krwx_ret_1d")

        details["usdjpy_change_pct"] = _pct_round(jpyx_ret)
        details["usdkrw_change_pct"] = _pct_round(krwx_ret)

        yen_th = self.fx_cfg.get("yen_carry_threshold", -1.5) / 100  # % → 분수
        krw_th = self.fx_cfg.get("krw_threshold", 1.0) / 100

        yen_carry_unwind = False

        if jpyx_ret is not None:
            # USD/JPY 급락 = 엔캐리 청산
            if jpyx_ret <= yen_th:
                score -= 0.5
                yen_carry_unwind = True
                details["yen_carry_status"] = "UNWIND"

                # 엔캐리 극단 (-3% 이상 급락)
                if jpyx_ret <= yen_th * 2:
                    score -= 0.3  # 추가 페널티
                    details["yen_carry_status"] = "EXTREME_UNWIND"
            elif jpyx_ret >= abs(yen_th):
                # 엔화 약세 = 캐리 유지 = 리스크온
                score += 0.2
                details["yen_carry_status"] = "CARRY_ON"
            else:
                details["yen_carry_status"] = "STABLE"

        if krwx_ret is not None:
            # 원화 약세 (USD/KRW 상승) = 외국인 이탈
            if krwx_ret >= krw_th:
                score -= 0.3
                details["krw_status"] = "WEAK"
            elif krwx_ret <= -krw_th:
                score += 0.2
                details["krw_status"] = "STRONG"
            else:
                details["krw_status"] = "STABLE"

        # 동시 발생: 엔캐리 청산 + 원화 약세 = 아시아 전체 risk-off
        if yen_carry_unwind and krwx_ret is not None and krwx_ret >= krw_th:
            score -= 0.2  # 추가 페널티
            details["asia_risk_off"] = True
        else:
            details["asia_risk_off"] = False

        details["yen_carry_unwind"] = yen_carry_unwind

        score = max(-1.0, min(1.0, score))
        details["score"] = round(score, 4)
        return details


# ──────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────

def _safe_float(row, col: str) -> float | None:
    """Series/dict에서 안전하게 float 추출. NaN/None → None."""
    import math

    try:
        v = row.get(col) if hasattr(row, "get") else row[col]
    except (KeyError, IndexError):
        return None

    if v is None:
        return None

    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _pct_round(v: float | None, decimals: int = 4) -> float | None:
    """퍼센트 표시용 반올림. None-safe."""
    if v is None:
        return None
    return round(v * 100, decimals)
