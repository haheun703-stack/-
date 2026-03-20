"""LENS LAYER — BRAIN ↔ SignalEngine 컨텍스트 연결 (STEP 7)

brain_decision.json을 읽어 4개 렌즈로 분석한 뒤
data/lens_context.json을 생성한다.

실행 순서: BRAIN(09:15) → LENS(09:16) → scan_buy(09:20)

4개 렌즈:
  1. GAME BOARD  : 공격/방어 모드 판정
  2. FLOW MAP    : 섹터 자금흐름 가중치
  3. STRUCTURAL VALUE : 밸류트랩 필터
  4. ASYMMETRY   : 레짐별 R:R 동적 조정
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.alpha.lens import game_board, flow_map, structural_value, asymmetry, derivatives_lens

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_BRAIN_PATH = _DATA_DIR / "brain_decision.json"
_FLOW_PATH = _DATA_DIR / "sector_rotation" / "investor_flow.json"
_OUTPUT_PATH = _DATA_DIR / "lens_context.json"


class LensLayer:
    """4개 렌즈를 오케스트레이션하여 lens_context.json을 생성한다."""

    def __init__(self, settings: dict):
        alpha_cfg = settings.get("alpha_v2", {})
        self.enabled = alpha_cfg.get("lens_enabled", False)
        self.lens_cfg = alpha_cfg.get("lens", {})

    def compute(self) -> dict:
        """모든 렌즈를 실행하고 결과를 저장한다.

        Returns:
            lens_context dict (enabled=False면 빈 기본값)
        """
        if not self.enabled:
            ctx = self._default_context("lens_enabled=false")
            self._save(ctx)
            return ctx

        # 데이터 로드
        brain = self._load_json(_BRAIN_PATH)
        flow_data = self._load_json(_FLOW_PATH)

        if not brain:
            ctx = self._default_context("brain_decision.json 로드 실패")
            self._save(ctx)
            return ctx

        regime = brain.get("effective_regime", "CAUTION")

        # 5개 렌즈 실행
        lens_1 = game_board.compute(brain)
        lens_2 = flow_map.compute(flow_data, self.lens_cfg)
        lens_3 = structural_value.compute(regime, self.lens_cfg)
        lens_4 = asymmetry.compute(regime, self.lens_cfg)
        lens_5 = derivatives_lens.compute()

        ctx = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "confidence": brain.get("confidence", 0.5),
            "game_board": lens_1,
            "flow_map": lens_2,
            "structural_value": lens_3,
            "asymmetry": lens_4,
            "derivatives": lens_5,
        }

        self._save(ctx)
        logger.info(
            "LENS 완료: mode=%s, hot=%s, min_rr=%.1f, deriv=%s(%+.0f)",
            lens_1.get("mode"),
            lens_2.get("hot_sectors", []),
            lens_4.get("min_rr_ratio", 0),
            lens_5.get("composite_grade", "?"),
            lens_5.get("composite_score", 0),
        )
        return ctx

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("LENS: %s 로드 실패 — %s", path, e)
            return {}

    @staticmethod
    def _save(ctx: dict) -> None:
        _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _default_context(reason: str) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "regime": "CAUTION",
            "confidence": 0.5,
            "game_board": {"mode": "DEFENSIVE", "reason": reason},
            "flow_map": {
                "hot_sectors": [],
                "cold_sectors": [],
                "flow_direction": "비활성",
                "sector_weight_adjustments": {},
            },
            "structural_value": {
                "min_quality_score": 0.4,
                "valuation_mode": "NORMAL",
                "trap_filter": False,
                "short_selling": {
                    "available": False,
                    "surge_tickers": [],
                    "cover_tickers": [],
                    "extreme_tickers": [],
                    "market_pressure": "NORMAL",
                },
            },
            "asymmetry": {
                "min_rr_ratio": 1.5,
                "target_atr_mult": 3.0,
                "stop_atr_mult": 2.0,
            },
            "derivatives": {
                "available": False,
                "composite_score": 0,
                "composite_grade": "NEUTRAL",
                "basis_status": "FLAT",
                "put_call_status": "NEUTRAL",
                "put_call_reversal": "",
                "flow_direction": "중립",
                "program_signal": "없음",
                "short_market": {"available": False, "sh4_triggered": False, "surge_ratio_pct": 0, "avg_risk": 0},
            },
        }
