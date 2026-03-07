"""유동성 사이클 트래커 — FRED 5대 지표 기반 중기 유동성 레짐 분석.

NIGHTWATCH(일간 Fast Eye), COT(주간 Slow Eye)와 대비되는 "중기 Meta Eye".
Net Liquidity = WALCL - TGA - RRP 를 핵심으로,
M2 통화공급 추세와 은행 지준 건전성을 보조 지표로 사용.

BRAIN에서 Step 5.6에 사용:
  - STRESS: 레버리지 0, 섹터 축소, 금/현금 확대
  - TIGHTENING: 레버리지/섹터 소폭 축소
  - NEUTRAL/AMPLE: 보정 없음 (방어만 하는 레이어)

핵심 원칙: 유동성 레이어는 "공격하지 않는다" — 오직 방어만.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "liquidity_cycle"
PARQUET_PATH = DATA_DIR / "liquidity_daily.parquet"
SIGNAL_PATH = DATA_DIR / "liquidity_signal.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# 복합 가중치
DEFAULT_WEIGHTS = {
    "net_liquidity": 0.50,
    "m2_momentum": 0.30,
    "reserves": 0.20,
}

DEFAULT_THRESHOLDS = {
    "tightening_z": -1.0,
    "stress_z": -1.5,
    "ample_z": 1.0,
    "reserve_stress_z": -1.0,
}


class LiquidityTracker:
    """유동성 사이클 분석 엔진."""

    def __init__(self, settings: dict | None = None):
        if settings is None:
            settings = self._load_settings()
        liq_cfg = settings.get("liquidity_cycle", {})
        self.enabled = liq_cfg.get("enabled", True)
        self.zscore_window = liq_cfg.get("zscore_window", 60)
        self.stale_warn_days = liq_cfg.get("stale_warn_days", 5)
        self.stale_ignore_days = liq_cfg.get("stale_ignore_days", 14)

        ind_cfg = liq_cfg.get("indicators", {})
        self.weights = {
            "net_liquidity": ind_cfg.get("net_liquidity_weight", 0.50),
            "m2_momentum": ind_cfg.get("m2_momentum_weight", 0.30),
            "reserves": ind_cfg.get("reserves_weight", 0.20),
        }
        self.thresholds = liq_cfg.get("thresholds", DEFAULT_THRESHOLDS)

    @staticmethod
    def _load_settings() -> dict:
        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def compute(self) -> dict:
        """유동성 시그널 계산 메인.

        Returns:
            dict — indicators, regime, signals, composite_direction, composite_score
        """
        if not self.enabled:
            return self._empty_result("유동성 사이클 비활성화")

        df = self._load_parquet()
        if df.empty or len(df) < 30:
            return self._empty_result("데이터 부족")

        today = datetime.now().date()
        latest_date = df.index.max().date()
        stale_days = (today - latest_date).days

        # 각 지표 z-score 계산
        net_liq_z = self._calc_zscore(df["net_liquidity"]) if "net_liquidity" in df.columns else 0.0
        m2_yoy = float(df["m2_yoy_pct"].iloc[-1]) if "m2_yoy_pct" in df.columns else 0.0
        m2_z = self._calc_zscore(df["m2_yoy_pct"]) if "m2_yoy_pct" in df.columns else 0.0
        reserves_z = self._calc_zscore(df["totresns"]) if "totresns" in df.columns else 0.0

        # 20일 변화량
        def change_20d(col):
            if col in df.columns and len(df) >= 20:
                return float(df[col].iloc[-1] - df[col].iloc[-20])
            return 0.0

        indicators = {
            "net_liquidity": {
                "value": round(float(df["net_liquidity"].iloc[-1]), 1) if "net_liquidity" in df.columns else 0,
                "z": round(net_liq_z, 2),
                "change_20d": round(change_20d("net_liquidity"), 1),
            },
            "m2_yoy_pct": {
                "value": round(m2_yoy, 2),
                "z": round(m2_z, 2),
            },
            "reserves": {
                "value": round(float(df["totresns"].iloc[-1]), 1) if "totresns" in df.columns else 0,
                "z": round(reserves_z, 2),
                "change_20d": round(change_20d("totresns"), 1),
            },
            "rrp": {
                "value": round(float(df["rrp"].iloc[-1]), 1) if "rrp" in df.columns else 0,
                "change_20d": round(change_20d("rrp"), 1),
            },
            "tga": {
                "value": round(float(df["tga"].iloc[-1]), 1) if "tga" in df.columns else 0,
                "change_20d": round(change_20d("tga"), 1),
            },
            "walcl": {
                "value": round(float(df["walcl"].iloc[-1]), 1) if "walcl" in df.columns else 0,
                "change_20d": round(change_20d("walcl"), 1),
            },
        }

        # 레짐 판정
        regime = self._classify_regime(net_liq_z, m2_yoy, reserves_z)

        # 시그널 생성
        signals = self._generate_signals(net_liq_z, m2_z, reserves_z)

        # 복합 점수
        composite_dir, composite_score = self._calc_composite(net_liq_z, m2_z, reserves_z)

        result = {
            "date": today.isoformat(),
            "data_date": latest_date.isoformat(),
            "stale_days": stale_days,
            "indicators": indicators,
            "regime": regime,
            "signals": signals,
            "composite_direction": composite_dir,
            "composite_score": round(composite_score, 3),
        }

        self._save_signal(result)

        logger.info(
            "유동성 사이클: %s (net_liq_z=%.2f, score=%.3f, stale=%dd)",
            regime, net_liq_z, composite_score, stale_days,
        )
        return result

    # ────────────────────────────────────────
    # 계산 유틸리티
    # ────────────────────────────────────────

    def _load_parquet(self) -> pd.DataFrame:
        if not PARQUET_PATH.exists():
            logger.warning("유동성 parquet 없음: %s", PARQUET_PATH)
            return pd.DataFrame()
        return pd.read_parquet(PARQUET_PATH)

    def _calc_zscore(self, series: pd.Series) -> float:
        """롤링 z-score 계산."""
        series = series.dropna()
        window = min(self.zscore_window, len(series))
        if window < 20:
            return 0.0
        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()
        if std == 0 or pd.isna(std):
            return 0.0
        return float((series.iloc[-1] - mean) / std)

    def _classify_regime(self, net_liq_z: float, m2_yoy: float, reserves_z: float) -> str:
        """유동성 레짐 판정.

        STRESS > TIGHTENING > NEUTRAL > AMPLE
        """
        th = self.thresholds
        stress_z = th.get("stress_z", -1.5)
        tight_z = th.get("tightening_z", -1.0)
        ample_z = th.get("ample_z", 1.0)
        reserve_stress = th.get("reserve_stress_z", -1.0)

        if net_liq_z <= stress_z and reserves_z <= reserve_stress:
            return "STRESS"
        if net_liq_z <= tight_z or m2_yoy < -2.0:
            return "TIGHTENING"
        if net_liq_z >= ample_z and m2_yoy > 0:
            return "AMPLE"
        return "NEUTRAL"

    def _generate_signals(self, net_liq_z: float, m2_z: float, reserves_z: float) -> dict:
        """3대 불리언 시그널."""
        th = self.thresholds
        return {
            "liquidity_tightening": net_liq_z <= th.get("tightening_z", -1.0),
            "liquidity_ample": net_liq_z >= th.get("ample_z", 1.0) and m2_z > 0,
            "reserve_stress": reserves_z <= th.get("reserve_stress_z", -1.0) and net_liq_z < 0,
        }

    def _calc_composite(self, net_liq_z: float, m2_z: float, reserves_z: float) -> tuple[str, float]:
        """복합 방향 + 점수.

        각 z-score를 [-1, +1] 범위로 클램프 후 가중 합산.
        """
        def clamp(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        # net_liq_z > 0 = 유동성 풍부 = bullish
        # m2_z > 0 = 통화공급 확대 = bullish
        # reserves_z > 0 = 지준 풍부 = bullish
        w = self.weights
        score = (
            clamp(net_liq_z) * w.get("net_liquidity", 0.50)
            + clamp(m2_z) * w.get("m2_momentum", 0.30)
            + clamp(reserves_z) * w.get("reserves", 0.20)
        )

        if score <= -0.50:
            direction = "STRONG_TIGHTENING"
        elif score <= -0.20:
            direction = "MILD_TIGHTENING"
        elif score <= 0.20:
            direction = "NEUTRAL"
        elif score <= 0.50:
            direction = "MILD_AMPLE"
        else:
            direction = "STRONG_AMPLE"

        return direction, score

    # ────────────────────────────────────────
    # 저장 / 유틸리티
    # ────────────────────────────────────────

    def _save_signal(self, result: dict):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("유동성 시그널 저장: %s", SIGNAL_PATH)

    @staticmethod
    def _empty_result(reason: str) -> dict:
        return {
            "date": datetime.now().date().isoformat(),
            "data_date": None,
            "stale_days": 999,
            "indicators": {},
            "regime": "NEUTRAL",
            "signals": {
                "liquidity_tightening": False,
                "liquidity_ample": False,
                "reserve_stress": False,
            },
            "composite_direction": "NEUTRAL",
            "composite_score": 0.0,
            "reason": reason,
        }
