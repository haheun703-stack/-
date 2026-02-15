"""
Genesis Detector — 포물선 시작점 통합 감지기

"포물선의 초점"이 이미 형성된 패턴에서 최적 진입점을 찾는다면,
"포물선의 시작점"은 패턴이 태어나기 전의 원인을 추적한다.

4대 조건:
  1. 에너지 축적 (충분한 하락 + 바닥 확인 + 변동성 수렴)
  2. 촉매 대기 (외부에서 주입 — 이 모듈에서는 스코어만 수신)
  3. 군중 무관심 (NeglectScorer)
  4. 선행 점화 (Lead-Lag — 외부에서 주입)

상전이 5대 전조 (PhaseTransitionDetector):
  임계 감속, Vol of Vol, 허스트, 깜빡임, 비대칭 요동

Class 분류: S(슈퍼) / A(대형) / B(일반) / C(미니)
포지션 사이징: Kelly Criterion (KellySizer)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .cycle_clock import CycleClock
from .kelly_sizer import KellySizer
from .neglect_score import NeglectScorer
from .phase_transition import PhaseTransitionDetector

logger = logging.getLogger(__name__)


class GenesisDetector:
    """포물선 시작점 통합 감지기"""

    # 에너지 축적 기본 파라미터
    DEFAULT_ENERGY_CFG = {
        "min_drawdown_pct": 0.10,       # 최소 조정폭 10%
        "support_test_count": 2,        # 지지선 테스트 최소 횟수
        "support_tolerance_pct": 0.02,  # 지지선 허용 오차 2%
        "min_consolidation_days": 15,   # 최소 횡보 기간
    }

    # Class 분류 기준
    CLASS_THRESHOLDS = {
        "S": {"min_composite": 0.80, "min_conditions": 4, "min_energy_days": 120},
        "A": {"min_composite": 0.65, "min_conditions": 3, "min_energy_days": 40},
        "B": {"min_composite": 0.50, "min_conditions": 2, "min_energy_days": 10},
        "C": {"min_composite": 0.35, "min_conditions": 1, "min_energy_days": 3},
    }

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("geometry", {}).get("genesis", {})
        self.enabled = cfg.get("enabled", True)
        self.energy_cfg = {**self.DEFAULT_ENERGY_CFG, **cfg.get("energy", {})}

        # 하위 모듈
        self.phase_transition = PhaseTransitionDetector(config)
        self.neglect = NeglectScorer(config)
        self.kelly = KellySizer(config)
        self.cycle = CycleClock(config)

    # ─── 에너지 축적 분석 ──────────────────────────

    def assess_energy(self, prices: np.ndarray, volumes: np.ndarray | None = None) -> dict:
        """
        에너지 축적 상태 분석.

        - 고점 대비 조정폭
        - 바닥 지지선 테스트 횟수
        - 횡보 기간
        - 변동성 수렴도

        Returns:
            {"drawdown_pct", "support_tests", "consolidation_days",
             "vol_compression", "score": 0.0~1.0, "sufficient": bool}
        """
        if len(prices) < 30:
            return self._empty_energy("데이터 부족")

        # 고점 대비 조정폭
        high_idx = np.argmax(prices)
        if high_idx == len(prices) - 1:
            drawdown = 0.0
        else:
            subsequent_low = np.min(prices[high_idx:])
            drawdown = 1.0 - subsequent_low / prices[high_idx]

        # 현재가가 바닥 근처인지 확인
        recent_prices = prices[-30:]
        current = prices[-1]
        recent_low = np.min(recent_prices)
        near_bottom = (current - recent_low) / (recent_low + 1e-10) < 0.05

        # 지지선 테스트 횟수 (최근 30일 최저가 근처 터치)
        tolerance = recent_low * self.energy_cfg["support_tolerance_pct"]
        support_tests = int(np.sum(recent_prices <= recent_low + tolerance))

        # 횡보 기간 (최근 가격 범위가 좁은 구간)
        if len(prices) >= 60:
            recent_60 = prices[-60:]
            price_range = (np.max(recent_60) - np.min(recent_60)) / np.mean(recent_60)
            # 좁은 범위로 횡보 중인 날 수
            ma = np.convolve(recent_60, np.ones(5) / 5, mode="valid")
            deviations = np.abs(recent_60[4:] - ma) / ma
            consolidation_days = int(np.sum(deviations < 0.02))
        else:
            price_range = 0.0
            consolidation_days = 0

        # 변동성 수렴: 최근 10일 변동성 / 과거 30일 변동성
        if len(prices) >= 40:
            recent_vol = np.std(np.diff(np.log(prices[-11:])))
            past_vol = np.std(np.diff(np.log(prices[-31:-10])))
            vol_compression = recent_vol / (past_vol + 1e-10)
        else:
            vol_compression = 1.0

        # 종합 스코어 계산
        score_components = []

        # 조정폭 스코어 (10~30% 조정이 이상적)
        if drawdown >= self.energy_cfg["min_drawdown_pct"]:
            dd_score = min(1.0, drawdown / 0.30)
        else:
            dd_score = drawdown / self.energy_cfg["min_drawdown_pct"]
        score_components.append(("drawdown", dd_score, 0.25))

        # 지지선 스코어
        st_score = min(1.0, support_tests / 3)
        score_components.append(("support", st_score, 0.20))

        # 횡보 기간 스코어
        consol_score = min(1.0, consolidation_days / self.energy_cfg["min_consolidation_days"])
        score_components.append(("consolidation", consol_score, 0.20))

        # 변동성 수렴 스코어 (0.5 이하면 만점)
        vc_score = max(0.0, min(1.0, 1.0 - vol_compression))
        score_components.append(("vol_compress", vc_score, 0.20))

        # 바닥 근처 스코어
        bottom_score = 1.0 if near_bottom else 0.3
        score_components.append(("near_bottom", bottom_score, 0.15))

        total_score = sum(s * w for _, s, w in score_components)
        # 변동성 수렴 임계값: 1.5 이하면 충분
        # (깜빡임 구간에서 저항선 터치가 단기 변동성을 올리므로 여유 있게 설정)
        sufficient = (
            drawdown >= self.energy_cfg["min_drawdown_pct"]
            and support_tests >= self.energy_cfg["support_test_count"]
            and vol_compression < 1.5
        )

        return {
            "drawdown_pct": round(drawdown, 4),
            "support_tests": support_tests,
            "consolidation_days": consolidation_days,
            "vol_compression": round(vol_compression, 3),
            "near_bottom": near_bottom,
            "score": round(total_score, 3),
            "sufficient": sufficient,
        }

    # ─── 사이클 바닥 확인 ──────────────────────────

    def assess_cycle_bottom(self, prices: np.ndarray) -> dict:
        """
        Cycle Clock에서 바닥 구간(5~7시) 여부 확인.

        Returns:
            {"long_clock", "mid_clock", "at_bottom": bool, "score": 0.0~1.0}
        """
        if len(prices) < 120:
            return {"long_clock": 0, "mid_clock": 0, "at_bottom": False, "score": 0.0}

        try:
            cycle_result = self.cycle.get_clock_position(prices)
        except Exception:
            return {"long_clock": 0, "mid_clock": 0, "at_bottom": False, "score": 0.0}

        long_clock = cycle_result["long"]["clock"]
        mid_clock = cycle_result["mid"]["clock"]

        # 바닥 구간: 4시~7시
        def bottom_score(clock: float) -> float:
            if 5 <= clock <= 7:
                return 1.0
            elif 4 <= clock < 5 or 7 < clock <= 8:
                return 0.6
            elif 3 <= clock < 4:
                return 0.3
            return 0.0

        long_s = bottom_score(long_clock)
        mid_s = bottom_score(mid_clock)

        # 이중 바닥 (장기+중기 모두 바닥) = Class S 조건
        combined = long_s * 0.6 + mid_s * 0.4
        at_bottom = long_s >= 0.6 and mid_s >= 0.6

        return {
            "long_clock": long_clock,
            "mid_clock": mid_clock,
            "at_bottom": at_bottom,
            "score": round(combined, 3),
        }

    # ─── Class 분류 ────────────────────────────────

    def classify(
        self,
        composite_score: float,
        conditions_met: int,
        energy_days: int,
    ) -> str:
        """
        Genesis 신호의 Class 결정.

        Parameters:
            composite_score: 종합 점수 (0~1)
            conditions_met: 충족된 조건 수 (0~5)
            energy_days: 에너지 축적 기간 (횡보 일수)
        """
        for cls in ["S", "A", "B", "C"]:
            thresh = self.CLASS_THRESHOLDS[cls]
            if (
                composite_score >= thresh["min_composite"]
                and conditions_met >= thresh["min_conditions"]
                and energy_days >= thresh["min_energy_days"]
            ):
                return cls
        return "NONE"

    # ─── 메인 분석 ─────────────────────────────────

    def detect(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        row: dict | None = None,
        df: pd.DataFrame | None = None,
        catalyst_score: float = 0.0,
        lead_ignition_score: float = 0.0,
    ) -> dict:
        """
        포물선 시작점 종합 분석.

        Parameters:
            prices: 종가 배열 (최소 60일)
            volumes: 거래량 배열
            row: 현재 시점 지표 딕셔너리 (neglect 계산용)
            df: 전체 DataFrame (neglect 정밀 계산용)
            catalyst_score: 촉매 대기 점수 (0~1, 외부 주입)
            lead_ignition_score: 선행 점화 점수 (0~1, 외부 주입)

        Returns:
            {
                "genesis_alert": bool,
                "signal_class": "S"/"A"/"B"/"C"/"NONE",
                "composite_score": float,
                "conditions": {
                    "energy": {...},
                    "phase_transition": {...},
                    "neglect": {...},
                    "cycle_bottom": {...},
                    "catalyst": float,
                    "lead_ignition": float,
                },
                "conditions_met": int,
                "position": {...},  # Kelly 사이징
                "timeout": {...},
                "exit_rules": {...},
                "prompt_text": str,
            }
        """
        if not self.enabled:
            return self._empty_result("비활성")

        prices = np.asarray(prices, dtype=float)
        if len(prices) < 60:
            return self._empty_result("데이터 부족")

        returns = np.diff(np.log(prices + 1e-10))
        conditions = {}
        condition_scores = []

        # ── 조건 1: 에너지 축적 ──
        try:
            energy = self.assess_energy(prices, volumes)
            conditions["energy"] = energy
            condition_scores.append(("energy", energy["score"], 0.25))
        except Exception as e:
            logger.debug("에너지 분석 실패: %s", e)
            conditions["energy"] = self._empty_energy(str(e))
            condition_scores.append(("energy", 0.0, 0.25))

        # ── 조건 2: 상전이 전조 ──
        try:
            pt = self.phase_transition.analyze(prices, returns)
            conditions["phase_transition"] = pt
            condition_scores.append(("phase_transition", pt["composite_score"], 0.30))
        except Exception as e:
            logger.debug("상전이 분석 실패: %s", e)
            conditions["phase_transition"] = PhaseTransitionDetector._empty_result(str(e))
            condition_scores.append(("phase_transition", 0.0, 0.30))

        # ── 조건 3: 군중 무관심 ──
        try:
            if row is not None:
                neglect = self.neglect.score(row, df)
            elif df is not None and len(df) > 0:
                neglect = self.neglect.score(df.iloc[-1].to_dict(), df)
            else:
                neglect = {"total_score": 0.3, "neglect_level": "보통"}
            conditions["neglect"] = neglect
            condition_scores.append(("neglect", neglect["total_score"], 0.20))
        except Exception as e:
            logger.debug("무관심 분석 실패: %s", e)
            conditions["neglect"] = {"total_score": 0.0, "neglect_level": "분석실패"}
            condition_scores.append(("neglect", 0.0, 0.20))

        # ── 조건 4: 사이클 바닥 ──
        try:
            cycle_bottom = self.assess_cycle_bottom(prices)
            conditions["cycle_bottom"] = cycle_bottom
            condition_scores.append(("cycle_bottom", cycle_bottom["score"], 0.10))
        except Exception as e:
            logger.debug("사이클 바닥 분석 실패: %s", e)
            conditions["cycle_bottom"] = {"long_clock": 0, "mid_clock": 0, "at_bottom": False, "score": 0.0}
            condition_scores.append(("cycle_bottom", 0.0, 0.10))

        # ── 조건 5: 촉매 + 선행 (외부 주입) ──
        conditions["catalyst"] = catalyst_score
        conditions["lead_ignition"] = lead_ignition_score
        condition_scores.append(("catalyst", catalyst_score, 0.08))
        condition_scores.append(("lead_ignition", lead_ignition_score, 0.07))

        # ── 종합 점수 ──
        composite = sum(s * w for _, s, w in condition_scores)

        # 충족 조건 수 계산 (임계값 초과한 것만)
        conditions_met = 0
        if conditions["energy"].get("sufficient", False):
            conditions_met += 1
        pt_result = conditions.get("phase_transition", {})
        if pt_result.get("phase_transition_imminent", False):
            conditions_met += 1
        if conditions.get("neglect", {}).get("total_score", 0) >= 0.6:
            conditions_met += 1
        if conditions.get("cycle_bottom", {}).get("at_bottom", False):
            conditions_met += 1
        if catalyst_score >= 0.5:
            conditions_met += 1

        # ── Class 분류 ──
        energy_days = conditions["energy"].get("consolidation_days", 0)
        signal_class = self.classify(composite, conditions_met, energy_days)

        # ── Genesis Alert ──
        genesis_alert = signal_class in ("S", "A", "B")

        # ── 포지션 사이징 (Kelly) ──
        if genesis_alert:
            # 과거 통계 기본값 (실전에서는 DB에서 조회)
            win_rates = {"S": 0.75, "A": 0.70, "B": 0.65, "C": 0.55}
            avg_wins = {"S": 0.30, "A": 0.20, "B": 0.10, "C": 0.05}
            avg_losses = {"S": 0.12, "A": 0.10, "B": 0.07, "C": 0.04}

            position = self.kelly.size_position(
                signal_class=signal_class,
                confidence=composite,
                win_rate=win_rates.get(signal_class, 0.60),
                avg_win=avg_wins.get(signal_class, 0.10),
                avg_loss=avg_losses.get(signal_class, 0.07),
            )
            timeout = self.kelly.timeout_rules(signal_class)
            exit_rules = self.kelly.exit_rules(signal_class)
        else:
            position = {"final_pct": 0.0, "signal_class": signal_class}
            timeout = {}
            exit_rules = {}

        result = {
            "genesis_alert": genesis_alert,
            "signal_class": signal_class,
            "composite_score": round(composite, 3),
            "conditions": conditions,
            "conditions_met": conditions_met,
            "position": position,
            "timeout": timeout,
            "exit_rules": exit_rules,
        }
        result["prompt_text"] = self.to_prompt_text(result)
        return result

    # ─── 프롬프트 텍스트 ───────────────────────────

    @staticmethod
    def to_prompt_text(result: dict) -> str:
        """Claude API용 Genesis Alert 텍스트"""
        cls = result.get("signal_class", "NONE")
        alert = result.get("genesis_alert", False)

        if not alert and cls == "NONE":
            return "[시작점 분석] 감지 없음"

        lines = []
        class_names = {"S": "슈퍼", "A": "대형", "B": "일반", "C": "미니", "NONE": "-"}

        if alert:
            lines.append(f"[GENESIS ALERT — Class {cls} ({class_names.get(cls, '?')}) 포물선 탄생 감지]")
        else:
            lines.append(f"[시작점 분석] Class {cls} — 조건 부분 충족")

        lines.append(f"  종합 점수: {result['composite_score']:.2f}")
        lines.append(f"  충족 조건: {result['conditions_met']}/5")

        # 세부 조건
        conds = result.get("conditions", {})

        energy = conds.get("energy", {})
        lines.append(f"  에너지 축적: {'충분' if energy.get('sufficient') else '부족'}"
                     f" (조정 {energy.get('drawdown_pct', 0):.1%},"
                     f" 지지 {energy.get('support_tests', 0)}회,"
                     f" 수렴 {energy.get('vol_compression', 1):.2f})")

        pt = conds.get("phase_transition", {})
        lines.append(f"  상전이 전조: {pt.get('precursor_count', 0)}/5개"
                     f" (composite: {pt.get('composite_score', 0):.2f})")

        neglect = conds.get("neglect", {})
        lines.append(f"  군중 무관심: {neglect.get('neglect_level', '?')}"
                     f" ({neglect.get('total_score', 0):.2f})")

        cycle = conds.get("cycle_bottom", {})
        lines.append(f"  사이클 바닥: {'예' if cycle.get('at_bottom') else '아니오'}"
                     f" (장기 {cycle.get('long_clock', 0):.0f}시,"
                     f" 중기 {cycle.get('mid_clock', 0):.0f}시)")

        # 포지션
        pos = result.get("position", {})
        if pos.get("final_pct", 0) > 0:
            lines.append(f"  포지션: {pos['final_pct']:.1%} (Kelly 기반)")
            timeout = result.get("timeout", {})
            if timeout:
                lines.append(f"  타임아웃: D+{timeout.get('half_reduce_days', '?')} 50% 축소,"
                             f" D+{timeout.get('full_exit_days', '?')} 전량 청산")

        return "\n".join(lines)

    # ─── 헬퍼 ──────────────────────────────────────

    @staticmethod
    def _empty_energy(reason: str) -> dict:
        return {
            "drawdown_pct": 0.0,
            "support_tests": 0,
            "consolidation_days": 0,
            "vol_compression": 1.0,
            "near_bottom": False,
            "score": 0.0,
            "sufficient": False,
        }

    @staticmethod
    def _empty_result(reason: str) -> dict:
        return {
            "genesis_alert": False,
            "signal_class": "NONE",
            "composite_score": 0.0,
            "conditions": {},
            "conditions_met": 0,
            "position": {"final_pct": 0.0, "signal_class": "NONE"},
            "timeout": {},
            "exit_rules": {},
            "prompt_text": f"[시작점 분석] {reason}",
        }
