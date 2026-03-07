"""COT Tracker — CFTC Commitments of Traders "Slow Eye" 분석 엔진

NIGHTWATCH(일간 "Fast Eye")와 대비되는 주간 "Slow Eye".
4개 선물 계약의 Managed Money/Asset Manager 포지셔닝을 z-score로 변환,
자산별 방향 시그널과 복합 방향을 산출.

BRAIN에서 Step 5.8에 사용:
  - Fast(NW) + Slow(COT) 정렬 시: 배분 변경 100% 적용 (confidence UP)
  - Fast(NW) + Slow(COT) 충돌 시: 배분 변경 50%만 적용

핵심 시그널:
  - S&P net sell (z < -1) → 리스크 자산 비중 축소
  - Gold net buy surge (z > 1.5) → 안전 수요 신호
  - Treasury net buy surge (z > 1.0) → 경기 둔화 베팅
  - Oil net sell (z < -1) → 경기순환 섹터 하락 신호
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
COT_DIR = PROJECT_ROOT / "data" / "cot"
PARQUET_PATH = COT_DIR / "cot_weekly.parquet"
SIGNAL_PATH = COT_DIR / "cot_signal.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# 계약별 라벨
CONTRACT_LABELS = {
    "sp500": "S&P 500 E-mini",
    "gold": "Gold COMEX",
    "treasury10y": "10Y T-Note",
    "crude_oil": "WTI Crude Oil",
}

# 기본 임계값 (settings.yaml에서 오버라이드 가능)
DEFAULT_THRESHOLDS = {
    "sp500": {
        "bearish_z": -1.0,
        "strong_bearish_z": -1.5,
        "bullish_z": 1.0,
    },
    "gold": {
        "safety_surge_z": 1.5,
        "extreme_surge_z": 2.0,
        "complacent_z": -1.0,
    },
    "treasury10y": {
        "slowdown_bet_z": 1.0,
        "strong_slowdown_z": 1.5,
        "reflation_z": -1.0,
    },
    "crude_oil": {
        "cyclical_down_z": -1.0,
        "demand_collapse_z": -1.5,
        "demand_boom_z": 1.5,
    },
}

# composite 가중치 (시장 위험 관점)
COMPOSITE_WEIGHTS = {
    "sp500": 0.35,
    "treasury10y": 0.25,
    "gold": 0.20,
    "crude_oil": 0.20,
}


class CotTracker:
    """COT 주간 시그널 분석."""

    def __init__(self, settings: dict | None = None):
        if settings is None:
            settings = self._load_settings()
        cot_cfg = settings.get("cot_tracker", {})
        self.enabled = cot_cfg.get("enabled", True)
        self.zscore_window = cot_cfg.get("zscore_window", 52)
        self.stale_warn_days = cot_cfg.get("stale_warn_days", 10)
        self.stale_ignore_days = cot_cfg.get("stale_ignore_days", 14)
        self.thresholds = cot_cfg.get("thresholds", DEFAULT_THRESHOLDS)

    @staticmethod
    def _load_settings() -> dict:
        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def compute(self) -> dict:
        """COT 시그널 계산 메인.

        Returns:
            dict — contracts, signals, composite_direction, composite_score
        """
        if not self.enabled:
            return self._empty_result("COT 비활성화")

        df = self._load_parquet()
        if df.empty or len(df) < 10:
            return self._empty_result("데이터 부족")

        today = datetime.now().date()
        latest_date = df.index.max().date()
        stale_days = (today - latest_date).days

        # 계약별 분석
        contracts = {}
        for name in CONTRACT_LABELS:
            net_col = f"{name}_net"
            if net_col not in df.columns:
                contracts[name] = self._empty_contract(name)
                continue

            series = df[net_col].dropna()
            if len(series) < 10:
                contracts[name] = self._empty_contract(name)
                continue

            latest_net = int(series.iloc[-1])
            z = self._calc_zscore(series)
            percentile = self._calc_percentile(series)
            direction = self._classify_direction(name, z)

            # 1주/4주 변화
            net_change_1w = int(series.iloc[-1] - series.iloc[-2]) if len(series) >= 2 else 0
            net_change_4w = int(series.iloc[-1] - series.iloc[-5]) if len(series) >= 5 else 0

            # long/short 원본
            long_col = f"{name}_long"
            short_col = f"{name}_short"
            latest_long = int(df[long_col].iloc[-1]) if long_col in df.columns else 0
            latest_short = int(df[short_col].iloc[-1]) if short_col in df.columns else 0

            contracts[name] = {
                "label": CONTRACT_LABELS[name],
                "net": latest_net,
                "long": latest_long,
                "short": latest_short,
                "z": round(z, 2),
                "direction": direction,
                "net_change_1w": net_change_1w,
                "net_change_4w": net_change_4w,
                "percentile_52w": round(percentile, 1),
            }

        # 시그널 생성
        signals = self._generate_signals(contracts)

        # 복합 방향
        composite_direction, composite_score = self._calc_composite(contracts)

        result = {
            "date": today.isoformat(),
            "report_date": latest_date.isoformat(),
            "stale_days": stale_days,
            "contracts": contracts,
            "signals": signals,
            "composite_direction": composite_direction,
            "composite_score": round(composite_score, 3),
        }

        # 저장
        self._save_signal(result)

        logger.info("COT Slow Eye: %s (score=%.3f, stale=%dd)",
                     composite_direction, composite_score, stale_days)
        return result

    # ────────────────────────────────────────
    # 계산 유틸리티
    # ────────────────────────────────────────

    def _load_parquet(self) -> pd.DataFrame:
        if not PARQUET_PATH.exists():
            logger.warning("COT parquet 없음: %s", PARQUET_PATH)
            return pd.DataFrame()
        return pd.read_parquet(PARQUET_PATH)

    def _calc_zscore(self, series: pd.Series) -> float:
        """52주 롤링 z-score."""
        window = min(self.zscore_window, len(series))
        if window < 10:
            return 0.0
        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()
        if std == 0 or pd.isna(std):
            return 0.0
        return float((series.iloc[-1] - mean) / std)

    def _calc_percentile(self, series: pd.Series) -> float:
        """52주 백분위. 0=최저, 100=최고."""
        window = min(self.zscore_window, len(series))
        recent = series.tail(window)
        return float((recent < series.iloc[-1]).sum() / len(recent) * 100)

    def _classify_direction(self, contract_name: str, z: float) -> str:
        """z-score → 시장 위험 방향 판정.

        모든 계약을 "시장 위험" 관점으로 통일:
          STRONG_BEARISH = 시장에 가장 위험
          STRONG_BULLISH = 시장에 가장 안전
        """
        th = self.thresholds.get(contract_name, {})

        if contract_name == "sp500":
            # S&P 매도 = 시장 위험
            if z <= th.get("strong_bearish_z", -1.5):
                return "STRONG_BEARISH"
            elif z <= th.get("bearish_z", -1.0):
                return "BEARISH"
            elif z >= th.get("bullish_z", 1.0):
                return "BULLISH"
            return "NEUTRAL"

        elif contract_name == "gold":
            # 금 매수 급증 = 안전수요 = 시장 위험
            if z >= th.get("extreme_surge_z", 2.0):
                return "STRONG_BEARISH"
            elif z >= th.get("safety_surge_z", 1.5):
                return "BEARISH"
            elif z <= th.get("complacent_z", -1.0):
                return "BULLISH"
            return "NEUTRAL"

        elif contract_name == "treasury10y":
            # 국채 매수 = 경기둔화 베팅 = 시장 위험
            if z >= th.get("strong_slowdown_z", 1.5):
                return "STRONG_BEARISH"
            elif z >= th.get("slowdown_bet_z", 1.0):
                return "BEARISH"
            elif z <= th.get("reflation_z", -1.0):
                return "BULLISH"
            return "NEUTRAL"

        elif contract_name == "crude_oil":
            # 원유 매도 = 수요 약화 = 시장 위험
            if z <= th.get("demand_collapse_z", -1.5):
                return "STRONG_BEARISH"
            elif z <= th.get("cyclical_down_z", -1.0):
                return "BEARISH"
            elif z >= th.get("demand_boom_z", 1.5):
                return "BULLISH"
            return "NEUTRAL"

        return "NEUTRAL"

    def _generate_signals(self, contracts: dict) -> dict:
        """4대 불리언 시그널."""
        sp = contracts.get("sp500", {})
        gd = contracts.get("gold", {})
        tr = contracts.get("treasury10y", {})
        oil = contracts.get("crude_oil", {})

        th_sp = self.thresholds.get("sp500", {})
        th_gd = self.thresholds.get("gold", {})
        th_tr = self.thresholds.get("treasury10y", {})
        th_oil = self.thresholds.get("crude_oil", {})

        return {
            "risk_off": sp.get("z", 0) <= th_sp.get("bearish_z", -1.0),
            "safety_demand": gd.get("z", 0) >= th_gd.get("safety_surge_z", 1.5),
            "slowdown_bet": tr.get("z", 0) >= th_tr.get("slowdown_bet_z", 1.0),
            "cyclical_down": oil.get("z", 0) <= th_oil.get("cyclical_down_z", -1.0),
        }

    def _calc_composite(self, contracts: dict) -> tuple[str, float]:
        """복합 방향 + 점수 계산.

        각 계약의 direction을 점수로 변환 후 가중 합산.
        """
        direction_scores = {
            "STRONG_BEARISH": -1.0,
            "BEARISH": -0.5,
            "NEUTRAL": 0.0,
            "BULLISH": 0.5,
            "STRONG_BULLISH": 1.0,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for name, weight in COMPOSITE_WEIGHTS.items():
            contract = contracts.get(name, {})
            direction = contract.get("direction", "NEUTRAL")
            score = direction_scores.get(direction, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 복합 방향 판정
        if composite_score <= -0.50:
            composite_dir = "STRONG_BEARISH"
        elif composite_score <= -0.20:
            composite_dir = "MILD_BEARISH"
        elif composite_score <= 0.20:
            composite_dir = "NEUTRAL"
        elif composite_score <= 0.50:
            composite_dir = "MILD_BULLISH"
        else:
            composite_dir = "STRONG_BULLISH"

        return composite_dir, composite_score

    # ────────────────────────────────────────
    # 저장 / 유틸리티
    # ────────────────────────────────────────

    def _save_signal(self, result: dict):
        COT_DIR.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("COT 시그널 저장: %s", SIGNAL_PATH)

    @staticmethod
    def _empty_contract(name: str) -> dict:
        return {
            "label": CONTRACT_LABELS.get(name, name),
            "net": 0, "long": 0, "short": 0,
            "z": 0.0, "direction": "NEUTRAL",
            "net_change_1w": 0, "net_change_4w": 0,
            "percentile_52w": 50.0,
        }

    @staticmethod
    def _empty_result(reason: str) -> dict:
        return {
            "date": datetime.now().date().isoformat(),
            "report_date": None,
            "stale_days": 999,
            "contracts": {},
            "signals": {"risk_off": False, "safety_demand": False,
                        "slowdown_bet": False, "cyclical_down": False},
            "composite_direction": "NEUTRAL",
            "composite_score": 0.0,
            "reason": reason,
        }
