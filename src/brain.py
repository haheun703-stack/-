"""BRAIN — 자본배분 중앙 두뇌 (Phase 1)

기존 3개 ARM(스윙/ETF로테이션/인버스+현금)의 자본배분을
중앙에서 조율하는 의사결정 엔진.

핵심 원칙:
  1. 의사결정 중앙화 — 실행은 기존 ARM이 그대로 수행
  2. 동적 배분 — 정적 regime_allocation을 NIGHTWATCH+VIX로 실시간 보정
  3. 충돌 방지 — 섹터 중복, 과집중 감지
  4. 안전장치 — 2일 확인, ±20%p 제한, 잔고 확인

데이터 흐름:
  kospi_regime.json ─┐
  overnight_signal.json ─┤
  regime_macro_signal.json ─┼→ BRAIN.compute()
  positions.json ─┤           → BrainDecision(배분 비율 + 이유)
  portfolio_allocation.json ─┘
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# 출력 경로
BRAIN_OUTPUT_PATH = DATA_DIR / "brain_decision.json"
BRAIN_HISTORY_PATH = DATA_DIR / "brain_history.json"


# ================================================================
# 데이터 모델
# ================================================================

@dataclass
class ArmAllocation:
    """개별 ARM에 대한 배분 지시."""
    name: str           # "swing" | "etf_sector" | "etf_leverage" | "etf_index" | "etf_gold" | "etf_small_cap" | "etf_bonds" | "etf_dollar" | "cash"
    base_pct: float     # settings.yaml 기본 비중
    adjusted_pct: float  # BRAIN 보정 후 비중
    reason: str = ""    # 보정 이유
    frozen: bool = False  # True면 이번 사이클 변경 금지

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_pct": round(self.base_pct, 1),
            "adjusted_pct": round(self.adjusted_pct, 1),
            "delta_pct": round(self.adjusted_pct - self.base_pct, 1),
            "reason": self.reason,
            "frozen": self.frozen,
        }


@dataclass
class BrainDecision:
    """BRAIN의 최종 자본배분 결정."""
    timestamp: str
    effective_regime: str       # 최종 적용 레짐
    kospi_regime: str           # KOSPI 원본 레짐
    nightwatch_score: float     # NIGHTWATCH 점수
    vix_level: float            # VIX
    confidence: float           # 결정 신뢰도 (0~1)
    arms: list[ArmAllocation]   # ARM별 배분
    adjustments: list[str]      # 적용된 보정 목록
    warnings: list[str]         # 경고 사항
    briefing: str               # 텔레그램 브리핑 텍스트

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "effective_regime": self.effective_regime,
            "kospi_regime": self.kospi_regime,
            "nightwatch_score": round(self.nightwatch_score, 4),
            "vix_level": round(self.vix_level, 1),
            "confidence": round(self.confidence, 2),
            "arms": [a.to_dict() for a in self.arms],
            "total_invest_pct": round(sum(a.adjusted_pct for a in self.arms if a.name != "cash"), 1),
            "cash_pct": round(next((a.adjusted_pct for a in self.arms if a.name == "cash"), 0), 1),
            "adjustments": self.adjustments,
            "warnings": self.warnings,
            "briefing": self.briefing,
        }


# ================================================================
# BRAIN 엔진
# ================================================================

class Brain:
    """자본배분 중앙 두뇌.

    9개 ARM:
      - swing: 개별종목 스윙 (v10.3 기반)
      - etf_sector: 섹터 ETF 로테이션 (축1)
      - etf_leverage: 레버리지/인버스 ETF (축2)
      - etf_index: 지수 ETF (축3)
      - etf_gold: 금 ETF (안전자산 헤지)
      - etf_small_cap: 소형주 ETF (BULL 초과수익)
      - etf_bonds: 채권 ETF (금리하락 방어)
      - etf_dollar: 달러 ETF (환율 헤지)
      - cash: 현금
    """

    # 투자 ARM 리스트 (현금 제외)
    INVEST_ARMS = [
        "swing", "etf_sector", "etf_leverage", "etf_index",
        "etf_gold", "etf_small_cap", "etf_bonds", "etf_dollar",
    ]

    # VIX 구간별 보정 계수 (투자비중 축소/확대)
    VIX_BUCKETS = [
        # (vix_max, invest_mult, label)
        (15.0, 1.10, "LOW_VOL"),      # 안정기: 투자 10% 확대
        (20.0, 1.00, "NORMAL"),       # 정상: 기본값
        (25.0, 0.90, "ELEVATED"),     # 경계: 10% 축소
        (30.0, 0.75, "HIGH"),         # 공포: 25% 축소
        (40.0, 0.55, "EXTREME"),      # 극단: 45% 축소
        (999., 0.30, "PANIC"),        # 패닉: 70% 축소
    ]

    # NIGHTWATCH 점수 구간 → 레짐 강도 조정
    NW_THRESHOLDS = {
        "strong_negative": -0.40,  # 강한 부정 → 레짐 1단계 하향
        "negative": -0.20,         # 부정 → 투자비중 추가 축소
        "positive": 0.20,          # 긍정 → 투자비중 소폭 확대
        "strong_positive": 0.40,   # 강한 긍정 → 레짐 내 최대 투자
    }

    # 충격 유형별 ARM 보정
    SHOCK_ARM_ADJUSTMENTS = {
        "GEOPOLITICAL": {"etf_leverage": -10, "etf_gold": +5, "cash": +5},
        "RATE": {"etf_index": -5, "etf_sector": -5, "etf_bonds": +5, "cash": +5},
        "LIQUIDITY": {"etf_leverage": -15, "etf_sector": -5, "etf_dollar": +5, "cash": +15},
        "EARNINGS": {},  # 실적 → 개별 ARM이 판단
        "COMPOUND": {"etf_leverage": -15, "etf_sector": -10, "etf_gold": +5, "etf_dollar": +5, "cash": +15},
    }

    def __init__(self, settings: dict | None = None):
        if settings is None:
            settings = self._load_settings()
        self.settings = settings
        brain_cfg = settings.get("brain", {})
        self.enabled = brain_cfg.get("enabled", True)
        self.max_daily_change = brain_cfg.get("max_daily_change_pct", 20.0)
        self.confirmation_days = brain_cfg.get("confirmation_days", 2)
        self.swing_base_pct = brain_cfg.get("swing_base_pct", 30.0)

    @staticmethod
    def _load_settings() -> dict:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ────────────────────────────────────────
    # 메인 엔트리포인트
    # ────────────────────────────────────────
    def compute(self) -> BrainDecision:
        """모든 데이터를 읽고 자본배분 결정을 생성.

        Returns:
            BrainDecision — ARM별 배분 + 이유 + 브리핑
        """
        adjustments = []
        warnings = []

        # ── 1. 입력 데이터 수집 ──
        kospi = self._load_json(DATA_DIR / "kospi_regime.json")
        us_signal = self._load_json(DATA_DIR / "us_market" / "overnight_signal.json")
        macro = self._load_json(DATA_DIR / "regime_macro_signal.json")
        positions = self._load_json(DATA_DIR / "positions.json")
        prev_decision = self._load_json(BRAIN_OUTPUT_PATH)

        kospi_regime = kospi.get("regime", "CAUTION")
        nw = us_signal.get("nightwatch", {})
        nw_score = nw.get("nightwatch_score", 0.0) if isinstance(nw, dict) else 0.0
        bv_veto = nw.get("bond_vigilante_veto", False) if isinstance(nw, dict) else False
        vix_data = us_signal.get("vix", {})
        vix_level = float(vix_data.get("level", 20.0))
        us_grade = us_signal.get("grade", "NEUTRAL")
        shock = us_signal.get("shock_type", {})
        shock_type = shock.get("shock_type", "NONE") if isinstance(shock, dict) else "NONE"
        ensemble_score = float(us_signal.get("ensemble_score", 0.0))
        macro_score = macro.get("macro_score", 50)

        logger.info("BRAIN 입력: 레짐=%s, NW=%.3f, VIX=%.1f, US=%s, 충격=%s",
                     kospi_regime, nw_score, vix_level, us_grade, shock_type)

        # ── 2. 유효 레짐 결정 (선행 보정) ──
        effective_regime = self._determine_effective_regime(
            kospi_regime, us_grade, nw_score, bv_veto, ensemble_score
        )
        if effective_regime != kospi_regime:
            adjustments.append(f"레짐 보정: {kospi_regime}→{effective_regime} (US={us_grade}, NW={nw_score:+.3f})")

        # ── 3. 기본 배분 로드 (settings.yaml) ──
        etf_cfg = self.settings.get("etf_rotation", {})
        regime_alloc = etf_cfg.get("regime_allocation", {}).get(effective_regime, {})
        if not regime_alloc:
            regime_alloc = etf_cfg.get("regime_allocation", {}).get("CAUTION", {})
            warnings.append(f"레짐 '{effective_regime}' 배분 없음 → CAUTION 기본값 사용")

        # ARM별 기본 비중
        arms = {
            "swing": self.swing_base_pct,
            "etf_sector": float(regime_alloc.get("sector", 0)),
            "etf_leverage": float(regime_alloc.get("leverage", 0)),
            "etf_index": float(regime_alloc.get("index", 0)),
            "etf_gold": float(regime_alloc.get("gold", 0)),
            "etf_small_cap": float(regime_alloc.get("small_cap", 0)),
            "etf_bonds": float(regime_alloc.get("bonds", 0)),
            "etf_dollar": float(regime_alloc.get("dollar", 0)),
            "cash": float(regime_alloc.get("cash", 40)),
        }

        # ── 4. VIX 기반 동적 보정 ──
        vix_mult, vix_label = self._get_vix_multiplier(vix_level)
        if vix_mult != 1.0:
            arms = self._apply_vix_adjustment(arms, vix_mult, vix_label)
            adjustments.append(f"VIX 보정: {vix_level:.1f} ({vix_label}) → 투자비중 ×{vix_mult:.2f}")

        # ── 5. NIGHTWATCH 점수 보정 ──
        nw_adj = self._apply_nightwatch_adjustment(arms, nw_score, bv_veto)
        if nw_adj:
            arms = nw_adj["arms"]
            adjustments.extend(nw_adj["adjustments"])
            warnings.extend(nw_adj.get("warnings", []))

        # ── 5.5. 2D 레짐 전환 선제 방어 ──
        nw_layers = nw.get("layers", {}) if isinstance(nw, dict) else {}
        l2_data = nw_layers.get("L2_regime_transition", {})
        credit_z = l2_data.get("credit_spread_z") or 0
        move_z = l2_data.get("move_z") or 0

        if credit_z >= 2.0 or move_z >= 2.0:
            # 크레딧/채권 위기 선행 → 레버리지 제거, 방어자산 확대
            arms["etf_leverage"] = 0
            arms["etf_gold"] = arms.get("etf_gold", 0) + 5
            arms["etf_bonds"] = arms.get("etf_bonds", 0) + 5
            arms["cash"] = arms.get("cash", 0) + 10
            adjustments.append(f"2D 선제방어: credit_z={credit_z:.1f}, MOVE_z={move_z:.1f}")
        elif credit_z >= 1.5 and move_z >= 1.0:
            # 경고 구간 → 소폭 방어
            arms["etf_leverage"] = max(0, arms.get("etf_leverage", 0) - 5)
            arms["etf_gold"] = arms.get("etf_gold", 0) + 3
            arms["cash"] = arms.get("cash", 0) + 2
            adjustments.append(f"2D 경고: credit_z={credit_z:.1f}, MOVE_z={move_z:.1f}")

        # ── 6. 충격 유형별 보정 ──
        if shock_type != "NONE" and shock_type in self.SHOCK_ARM_ADJUSTMENTS:
            shock_adj = self.SHOCK_ARM_ADJUSTMENTS[shock_type]
            if shock_adj:
                arms = self._apply_shock_adjustment(arms, shock_adj, shock_type)
                shock_conf = shock.get("confidence", 0) if isinstance(shock, dict) else 0
                adjustments.append(f"충격 보정: {shock_type} (확신도 {shock_conf:.0%})")

        # ── 6.5. SHIELD 방어 보정 ──
        shield_report = self._load_json(DATA_DIR / "shield_report.json")
        shield_overrides = shield_report.get("brain_overrides", {})
        shield_level = shield_overrides.get("shield_level", "GREEN")

        if shield_level in ("ORANGE", "RED"):
            arm_adj = shield_overrides.get("arm_adjustments", {})
            for arm_name, delta in arm_adj.items():
                if arm_name in arms:
                    arms[arm_name] = max(0, arms[arm_name] + delta)
            for msg in shield_overrides.get("messages", []):
                adjustments.append(f"SHIELD: {msg}")
            warnings.append(f"SHIELD 방어 등급: {shield_level}")
        elif shield_level == "YELLOW":
            for msg in shield_overrides.get("messages", []):
                warnings.append(f"SHIELD: {msg}")

        # ── 7. 비중 정규화 (합계 = 100%) ──
        arms = self._normalize_allocations(arms)

        # ── 8. 안전장치: 일일 변경 제한 ──
        if prev_decision:
            arms, clamped = self._clamp_daily_change(arms, prev_decision)
            if clamped:
                adjustments.append(f"일일 변경 ±{self.max_daily_change}%p 제한 적용")

        # ── 9. 안전장치: 2일 확인 ──
        confirmation = self._check_regime_confirmation(
            effective_regime, prev_decision, macro
        )
        if not confirmation["confirmed"]:
            warnings.append(confirmation["message"])

        # ── 10. 신뢰도 계산 ──
        confidence = self._calc_confidence(
            kospi_regime, effective_regime, nw_score, vix_level,
            macro_score, shock_type, confirmation["confirmed"]
        )

        # ── 11. ArmAllocation 객체 생성 ──
        arm_objects = []
        base_alloc = {
            "swing": self.swing_base_pct,
            "etf_sector": float(regime_alloc.get("sector", 0)),
            "etf_leverage": float(regime_alloc.get("leverage", 0)),
            "etf_index": float(regime_alloc.get("index", 0)),
            "etf_gold": float(regime_alloc.get("gold", 0)),
            "etf_small_cap": float(regime_alloc.get("small_cap", 0)),
            "etf_bonds": float(regime_alloc.get("bonds", 0)),
            "etf_dollar": float(regime_alloc.get("dollar", 0)),
            "cash": float(regime_alloc.get("cash", 40)),
        }
        for name in self.INVEST_ARMS + ["cash"]:
            arm_objects.append(ArmAllocation(
                name=name,
                base_pct=base_alloc[name],
                adjusted_pct=arms[name],
                reason=self._arm_reason(name, base_alloc[name], arms[name]),
                frozen=not confirmation["confirmed"] and name != "cash",
            ))

        # ── 12. 브리핑 생성 ──
        briefing = self._build_briefing(
            effective_regime, kospi_regime, nw_score, vix_level,
            vix_label, us_grade, shock_type, arms, adjustments, warnings
        )

        decision = BrainDecision(
            timestamp=datetime.now().isoformat(),
            effective_regime=effective_regime,
            kospi_regime=kospi_regime,
            nightwatch_score=nw_score,
            vix_level=vix_level,
            confidence=confidence,
            arms=arm_objects,
            adjustments=adjustments,
            warnings=warnings,
            briefing=briefing,
        )

        # ── 13. 저장 ──
        self._save_decision(decision)

        return decision

    # ────────────────────────────────────────
    # 2. 유효 레짐 결정
    # ────────────────────────────────────────
    def _determine_effective_regime(
        self,
        kospi_regime: str,
        us_grade: str,
        nw_score: float,
        bv_veto: bool,
        ensemble_score: float,
    ) -> str:
        """KOSPI 레짐 + US/NW 신호 → 유효 레짐.

        기존 run_etf_rotation.py의 calc_leading_regime 로직을 확장.
        """
        regime = kospi_regime

        # 채권 자경단 비토 → 즉시 1단계 하향
        if bv_veto:
            regime = self._downgrade_regime(regime)
            return regime

        # 강한 부정 NIGHTWATCH → 1단계 하향
        if nw_score <= self.NW_THRESHOLDS["strong_negative"]:
            regime = self._downgrade_regime(regime)
            return regime

        # US STRONG_BEAR + KOSPI CAUTION 이상 → PRE_BEAR
        if us_grade == "STRONG_BEAR" and kospi_regime in ("BULL", "CAUTION"):
            return "PRE_BEAR"

        # US MILD_BEAR + ensemble 강한 부정 → PRE_BEAR
        if us_grade == "MILD_BEAR" and ensemble_score <= -0.40:
            if kospi_regime in ("BULL", "CAUTION"):
                return "PRE_BEAR"

        # US STRONG_BULL + 긍정적 NW → PRE_BULL
        if us_grade == "STRONG_BULL" and nw_score >= self.NW_THRESHOLDS["positive"]:
            if kospi_regime in ("CAUTION", "BEAR"):
                return "PRE_BULL"

        return regime

    @staticmethod
    def _downgrade_regime(regime: str) -> str:
        """레짐 1단계 하향."""
        order = {"BULL": "CAUTION", "CAUTION": "BEAR", "BEAR": "CRISIS",
                 "PRE_BULL": "CAUTION", "PRE_BEAR": "CRISIS", "PRE_CRISIS": "CRISIS",
                 "CRISIS": "CRISIS"}
        return order.get(regime, regime)

    # ────────────────────────────────────────
    # 4. VIX 보정
    # ────────────────────────────────────────
    def _get_vix_multiplier(self, vix: float) -> tuple[float, str]:
        """VIX 레벨 → 투자비중 배수."""
        for max_vix, mult, label in self.VIX_BUCKETS:
            if vix < max_vix:
                return mult, label
        return 0.30, "PANIC"

    def _apply_vix_adjustment(self, arms: dict, mult: float, label: str) -> dict:
        """VIX 배수로 투자 ARM 비중 축소/확대, 나머지 현금으로."""
        invest_arms = self.INVEST_ARMS
        total_invest = sum(arms.get(a, 0) for a in invest_arms)

        if total_invest <= 0:
            return arms

        adjusted_invest = total_invest * mult
        scale = adjusted_invest / total_invest if total_invest > 0 else 1.0

        result = dict(arms)
        for a in invest_arms:
            result[a] = arms.get(a, 0) * scale
        result["cash"] = 100.0 - sum(result[a] for a in invest_arms)
        return result

    # ────────────────────────────────────────
    # 5. NIGHTWATCH 보정
    # ────────────────────────────────────────
    def _apply_nightwatch_adjustment(
        self, arms: dict, nw_score: float, bv_veto: bool
    ) -> dict | None:
        """NIGHTWATCH 점수 기반 미세 보정."""
        result_arms = dict(arms)
        adj_list = []
        warn_list = []

        # 채권 자경단 비토 → 레버리지 즉시 0, 섹터 반토막
        if bv_veto:
            if result_arms["etf_leverage"] > 0:
                freed = result_arms["etf_leverage"]
                result_arms["etf_leverage"] = 0
                result_arms["cash"] += freed
                adj_list.append(f"채권 비토: 레버리지 {freed:.0f}%→0%, 현금으로 이동")
            sector_cut = result_arms["etf_sector"] * 0.5
            result_arms["etf_sector"] -= sector_cut
            result_arms["cash"] += sector_cut
            adj_list.append(f"채권 비토: 섹터 50% 축소 (-{sector_cut:.0f}%p)")
            warn_list.append("채권 자경단 비토 발동 — 방어 모드")
            return {"arms": result_arms, "adjustments": adj_list, "warnings": warn_list}

        # 강한 부정 (-0.4 이하) → 투자 20% 추가 축소
        if nw_score <= self.NW_THRESHOLDS["strong_negative"]:
            scale = 0.80
            invest_arms = self.INVEST_ARMS
            freed = 0
            for a in invest_arms:
                cut = result_arms[a] * (1 - scale)
                result_arms[a] -= cut
                freed += cut
            result_arms["cash"] += freed
            adj_list.append(f"NW 강한 부정({nw_score:+.3f}): 투자 20% 축소")

        # 부정 (-0.2 ~ -0.4) → 레버리지만 축소
        elif nw_score <= self.NW_THRESHOLDS["negative"]:
            if result_arms["etf_leverage"] > 0:
                cut = result_arms["etf_leverage"] * 0.3
                result_arms["etf_leverage"] -= cut
                result_arms["cash"] += cut
                adj_list.append(f"NW 부정({nw_score:+.3f}): 레버리지 30% 축소")

        # 긍정 (+0.2 이상) → 소폭 확대 (현금에서 이동)
        elif nw_score >= self.NW_THRESHOLDS["positive"]:
            if result_arms["cash"] > 15:  # 최소 현금 15% 유지
                boost = min(5.0, result_arms["cash"] - 15)
                result_arms["etf_sector"] += boost * 0.6
                result_arms["swing"] += boost * 0.4
                result_arms["cash"] -= boost
                adj_list.append(f"NW 긍정({nw_score:+.3f}): 투자 +{boost:.0f}%p 확대")

        if not adj_list:
            return None

        return {"arms": result_arms, "adjustments": adj_list, "warnings": warn_list}

    # ────────────────────────────────────────
    # 6. 충격 유형 보정
    # ────────────────────────────────────────
    def _apply_shock_adjustment(
        self, arms: dict, adj_map: dict, shock_type: str
    ) -> dict:
        """충격 유형에 따른 ARM별 %p 조정."""
        result = dict(arms)
        for arm, delta in adj_map.items():
            if arm in result:
                result[arm] = max(0, result[arm] + delta)
        return result

    # ────────────────────────────────────────
    # 7. 정규화
    # ────────────────────────────────────────
    @classmethod
    def _normalize_allocations(cls, arms: dict) -> dict:
        """합계 100%로 정규화. 현금은 나머지."""
        invest_arms = cls.INVEST_ARMS
        total = sum(max(0, arms.get(a, 0)) for a in invest_arms)

        result = {}
        for a in invest_arms:
            result[a] = max(0, arms.get(a, 0))

        # 투자 합계가 100% 초과하면 비례 축소
        if total > 95:
            scale = 95.0 / total
            for a in invest_arms:
                result[a] *= scale
            total = sum(result[a] for a in invest_arms)

        result["cash"] = max(5.0, 100.0 - total)  # 최소 현금 5%
        return result

    # ────────────────────────────────────────
    # 8. 일일 변경 제한
    # ────────────────────────────────────────
    def _clamp_daily_change(
        self, arms: dict, prev_decision: dict
    ) -> tuple[dict, bool]:
        """이전 결정 대비 ±max_daily_change %p 제한."""
        prev_arms = {}
        for arm_data in prev_decision.get("arms", []):
            prev_arms[arm_data["name"]] = arm_data.get("adjusted_pct", 0)

        if not prev_arms:
            return arms, False

        clamped = False
        result = dict(arms)
        max_chg = self.max_daily_change

        for name in self.INVEST_ARMS:
            prev_val = prev_arms.get(name, 0)
            new_val = result.get(name, 0)
            delta = new_val - prev_val
            if abs(delta) > max_chg:
                result[name] = prev_val + max_chg * (1 if delta > 0 else -1)
                result[name] = max(0, result[name])
                clamped = True

        # 현금은 나머지
        invest = sum(result.get(a, 0) for a in self.INVEST_ARMS)
        result["cash"] = max(5.0, 100.0 - invest)

        return result, clamped

    # ────────────────────────────────────────
    # 9. 레짐 확인
    # ────────────────────────────────────────
    def _check_regime_confirmation(
        self, effective_regime: str, prev_decision: dict, macro: dict
    ) -> dict:
        """레짐이 N일 연속 동일한지 확인."""
        if not prev_decision:
            return {"confirmed": True, "message": "첫 실행 — 확인 불필요"}

        prev_regime = prev_decision.get("effective_regime", "")
        if prev_regime == effective_regime:
            return {"confirmed": True, "message": f"{effective_regime} 연속 유지"}

        # 레짐 변경 시 — history에서 연속일 확인
        history = macro.get("regime_history_5d", [])
        if len(history) >= self.confirmation_days:
            recent = history[-self.confirmation_days:]
            if all(r == effective_regime for r in recent):
                return {"confirmed": True,
                        "message": f"{effective_regime} {self.confirmation_days}일 연속 확인"}

        return {
            "confirmed": False,
            "message": (f"레짐 변경 감지: {prev_regime}→{effective_regime}. "
                        f"{self.confirmation_days}일 확인 대기 중. 배분 동결.")
        }

    # ────────────────────────────────────────
    # 10. 신뢰도
    # ────────────────────────────────────────
    @staticmethod
    def _calc_confidence(
        kospi_regime: str, effective_regime: str,
        nw_score: float, vix_level: float,
        macro_score: int, shock_type: str,
        confirmed: bool,
    ) -> float:
        """결정 신뢰도 0~1."""
        conf = 0.50  # 기본

        # 레짐 일치 시 +0.15
        if kospi_regime == effective_regime:
            conf += 0.15

        # 매크로 점수 높으면 +0.15
        if macro_score >= 65:
            conf += 0.15
        elif macro_score >= 45:
            conf += 0.05

        # VIX 안정 +0.10
        if vix_level < 20:
            conf += 0.10
        elif vix_level > 30:
            conf -= 0.10

        # NW 중립 근처 +0.05
        if abs(nw_score) < 0.15:
            conf += 0.05

        # 충격 없음 +0.05
        if shock_type == "NONE":
            conf += 0.05
        else:
            conf -= 0.10

        # 레짐 미확인 -0.15
        if not confirmed:
            conf -= 0.15

        return max(0.10, min(0.95, conf))

    # ────────────────────────────────────────
    # 보조 함수
    # ────────────────────────────────────────
    @staticmethod
    def _arm_reason(name: str, base: float, adjusted: float) -> str:
        """ARM 비중 변경 이유 요약."""
        delta = adjusted - base
        if abs(delta) < 0.5:
            return "변경 없음"
        direction = "확대" if delta > 0 else "축소"
        return f"{direction} {abs(delta):.1f}%p"

    @staticmethod
    def _load_json(path: Path) -> dict | list:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_decision(self, decision: BrainDecision):
        """결정 저장 (현재 + 히스토리)."""
        data = decision.to_dict()

        # 현재 결정
        BRAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 히스토리 누적 (최대 90일)
        history = self._load_json(BRAIN_HISTORY_PATH)
        if not isinstance(history, list):
            history = []
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "regime": data["effective_regime"],
            "kospi_regime": data["kospi_regime"],
            "nw_score": data["nightwatch_score"],
            "vix": data["vix_level"],
            "confidence": data["confidence"],
            "total_invest": data["total_invest_pct"],
            "cash": data["cash_pct"],
            "adjustments_count": len(data["adjustments"]),
        })
        history = history[-90:]
        with open(BRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        logger.info("BRAIN 결정 저장: %s", BRAIN_OUTPUT_PATH)

    def _build_briefing(
        self,
        effective_regime: str, kospi_regime: str,
        nw_score: float, vix_level: float, vix_label: str,
        us_grade: str, shock_type: str,
        arms: dict, adjustments: list, warnings: list,
    ) -> str:
        """텔레그램 브리핑 텍스트 생성."""
        lines = []
        lines.append("[ BRAIN 자본배분 브리핑 ]")
        lines.append("")

        # 레짐
        regime_str = effective_regime
        if effective_regime != kospi_regime:
            regime_str = f"{effective_regime} (KOSPI: {kospi_regime})"
        lines.append(f"레짐: {regime_str}")
        lines.append(f"NW: {nw_score:+.3f} | VIX: {vix_level:.1f} ({vix_label}) | US: {us_grade}")

        if shock_type != "NONE":
            lines.append(f"충격: {shock_type}")
        lines.append("")

        # 배분
        lines.append("자본 배분:")
        arm_labels = {
            "swing": "개별스윙",
            "etf_sector": "섹터ETF",
            "etf_leverage": "레버리지",
            "etf_index": "지수ETF",
            "etf_gold": "금ETF",
            "etf_small_cap": "소형주",
            "etf_bonds": "채권",
            "etf_dollar": "달러",
            "cash": "현금",
        }
        for name in self.INVEST_ARMS + ["cash"]:
            label = arm_labels[name]
            pct = arms.get(name, 0)
            bar = "█" * int(pct / 5) if pct > 0 else ""
            lines.append(f"  {label:>8s} {pct:5.1f}% {bar}")
        lines.append("")

        # 보정 사항
        if adjustments:
            lines.append("보정:")
            for adj in adjustments:
                lines.append(f"  · {adj}")
            lines.append("")

        # 경고
        if warnings:
            lines.append("경고:")
            for w in warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)


# ================================================================
# CLI 실행
# ================================================================
def main():
    """커맨드라인 실행."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("  BRAIN — 자본배분 중앙 두뇌 (Phase 1)")
    print("=" * 60)

    brain = Brain()
    decision = brain.compute()

    print(f"\n{decision.briefing}")
    print(f"\n신뢰도: {decision.confidence:.0%}")
    print(f"저장: {BRAIN_OUTPUT_PATH}")

    return decision


if __name__ == "__main__":
    main()
