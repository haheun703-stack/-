"""
Step 5: signal_engine.py — BES v3.1 Pipeline 엔진

v3.0 → v3.1 변경:
  - L-1 News Gate 추가 (뉴스 등급별 파라미터 조정)
  - L4 Smart Money 강화 (매집 3단계 + OBV 다이버전스)
  - L5 Risk 조정 (이벤트 드리븐 rr_min=1.2, rsi_max=65)

v4.6: L7 Geometric 보조 레이어 추가
  - 하모닉 패턴 / 엘리어트 파동 / 추세 각도 보조 확인
  - 최종 시그널의 confidence 가감 (blocking 아님)

Pipeline:
  [v3.1] L-1_news_gate → 뉴스 등급(A/B/C) → 파라미터 조정
  L0_pre_gate   → Pre-screening (매출/거래대금/수익성)
  L0_grade      → Zone Score + Grade (A/B/C/F)
  L1_regime     → HMM 레짐 (Accumulation만 통과)
  L2_ou         → OU 필터 (z-score, half-life, SNR)
  L3_momentum   → 모멘텀 (거래량 서지 + MA60 slope)
  L4_smart_money→ Smart Money Z-score + [v3.1] 매집/다이버전스
  L5_risk       → 손익비 (A급: 1.7→1.2, RSI: 70→65)
  L6_trigger    → Impulse/Confirm/Breakout
  [v4.6] L7_geometric → 하모닉/엘리어트/각도 보조 확인
"""

import logging
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .fundamental import FundamentalEngine
from .screener import Screener
from .regime_detector import RegimeDetector
from .ou_estimator import OUEstimator
from .smart_money import check_smart_money_gate
from .signal_diagnostic import SignalDiagnostic, LayerResult
from .accumulation_detector import AccumulationDetector
from .divergence_scanner import DivergenceScanner
from .probability_gate import ProbabilityGate
from .sector_classifier import SectorClassifier
from src.entities.news_models import NewsGateResult, NewsGrade, EventDrivenAction
from .geometric_engine import GeometricQuantEngine
from .martin_momentum import MartinMomentumEngine
from .extreme_volatility import ExtremeVolatilityDetector
from src.entities.consensus_models import LayerVote, ConsensusResult
from src.use_cases.consensus_engine import ConsensusVerifier
from .tgci_scorer import TGCIScorer
from .master_controller import MasterController

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """트리거 유형"""
    NONE = "none"
    IMPULSE = "impulse"       # 시동 (공격형)
    CONFIRM = "confirm"       # 확인 (보수형)
    BREAKOUT = "breakout"     # 전고점 돌파 (추가 매수)
    TREND_CONT = "trend_cont" # 추세 지속 (Grade F 우회)
    SETUP = "setup"            # v7.0 SETUP (게이트 통과 + 트리거 대기 + TGCI 확인)


@dataclass
class TriggerResult:
    """트리거 판정 결과"""
    trigger_type: TriggerType
    conditions_met: dict          # 개별 조건 충족 여부
    conditions_count: int         # 충족 조건 수
    stop_loss_pct: float          # 모드별 손절 비율
    entry_stage_pct: float        # 이 트리거의 비중(40%, 40%, 20%)
    confidence: float             # 트리거 신뢰도 (0~1)


class SignalEngine:
    """BES v3.0 6-Layer Pipeline 스코어링 + 시그널 생성"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.strategy = self.config["strategy"]
        self.triggers_cfg = self.strategy.get("triggers", {})
        self.fundamental = FundamentalEngine(config_path)
        self.screener = Screener(self.config, self.fundamental)

        # v3.0 퀀트 레이어 설정
        quant_cfg = self.config.get("quant_engine", {})
        self.regime_cfg = quant_cfg.get("regime", {})
        self.ou_cfg = quant_cfg.get("ou", {})
        self.momentum_cfg = quant_cfg.get("momentum", {})
        self.smart_money_cfg = quant_cfg.get("smart_money", {})

        # 진단 시스템
        self.diagnostic = SignalDiagnostic()

        # v4.5: 확률 게이트
        prob_cfg = quant_cfg.get("probability_model", {})
        self.prob_gate_enabled = prob_cfg.get("enabled", False)
        if self.prob_gate_enabled:
            self.prob_gate = ProbabilityGate(
                rolling_window=prob_cfg.get("rolling_window", 200),
                threshold=prob_cfg.get("threshold", 0.65),
                lookahead_bars=prob_cfg.get("lookahead_bars", 12),
                min_avg_return=prob_cfg.get("min_avg_return", 0.01),
            )
        else:
            self.prob_gate = None

        # v4.5: 섹터 분류기
        self.sector_classifier = SectorClassifier(config_path)

        # v4.6: Geometric 보조 레이어
        geo_cfg = self.config.get("geometric_engine", {})
        self.geo_enabled = geo_cfg.get("enabled", False)
        if self.geo_enabled:
            self.geo_engine = GeometricQuantEngine(config=geo_cfg)
            self.geo_confidence_boost = geo_cfg.get("confidence_boost", 0.10)
            self.geo_confidence_penalty = geo_cfg.get("confidence_penalty", 0.05)
            self.geo_min_confidence = geo_cfg.get("min_confidence", 30)
        else:
            self.geo_engine = None

        # v5.0: Sci-CoE 합의 엔진
        consensus_cfg = self.config.get("consensus_engine", {})
        self.consensus_mode = consensus_cfg.get("enabled", False)
        if self.consensus_mode:
            self.consensus_verifier = ConsensusVerifier(
                tau=consensus_cfg.get("tau", 0.8),
                min_voters=consensus_cfg.get("min_voters", 4),
                grade_thresholds=consensus_cfg.get("grade_thresholds", None),
            )
        else:
            self.consensus_verifier = None

        # v6.0: 극한 변동성 탐지기
        evol_cfg = self.config.get("extreme_volatility", {})
        self.extreme_vol_enabled = evol_cfg.get("enabled", False)
        if self.extreme_vol_enabled:
            cap_cfg = evol_cfg.get("capitulation", {})
            self.extreme_vol_detector = ExtremeVolatilityDetector(
                atr_ratio_threshold=evol_cfg.get("atr_ratio_threshold", 3.0),
                vol_ratio_threshold=evol_cfg.get("vol_ratio_threshold", 5.0),
                daily_range_threshold=evol_cfg.get("daily_range_threshold", 10.0),
                min_down_days=cap_cfg.get("min_down_days", 3),
                volume_climax_mult=cap_cfg.get("volume_climax_mult", 3.0),
                rsi_extreme=cap_cfg.get("rsi_extreme", 20),
                min_capitulation_score=cap_cfg.get("min_score", 70),
                allow_ambiguous=evol_cfg.get("allow_ambiguous", False),
            )
        else:
            self.extreme_vol_detector = None

        # v6.0: Martin Momentum 엔진
        martin_cfg = self.config.get("martin_momentum", {})
        self.martin_enabled = martin_cfg.get("enabled", False)
        if self.martin_enabled:
            vol_cfg = martin_cfg.get("vol_normalization", {})
            self.martin_engine = MartinMomentumEngine(
                n_fast=martin_cfg.get("n_fast", 8),
                n_slow=martin_cfg.get("n_slow", 24),
                epsilon=martin_cfg.get("epsilon", 0.6),
                sigmoid_k=martin_cfg.get("sigmoid_k", 5.0),
                min_confidence=martin_cfg.get("min_confidence", 0.3),
                target_sigma=vol_cfg.get("target_sigma", 0.02),
                max_vol_weight=vol_cfg.get("max_weight", 2.0),
                min_vol_weight=vol_cfg.get("min_weight", 0.3),
            )
        else:
            self.martin_engine = None

        # v6.1: RTTP 뉴스 프로세서
        rttp_cfg = self.config.get("rttp_news", {})
        self.rttp_enabled = rttp_cfg.get("enabled", False)
        if self.rttp_enabled:
            from src.use_cases.rttp_news_processor import RttpNewsProcessor
            self.rttp_processor = RttpNewsProcessor(
                source_weights=rttp_cfg.get("source_weights"),
                engagement_weights=rttp_cfg.get("engagement_levels"),
                recall_threshold=rttp_cfg.get("recall_monitoring", {}).get("threshold", 0.5),
            )
        else:
            self.rttp_processor = None

        # v3.1 Smart Money v2 + News Gate
        self.accum_detector = AccumulationDetector(config_path)
        self.div_scanner = DivergenceScanner(config_path)
        self._news_gates: dict[str, NewsGateResult] = {}  # ticker → NewsGateResult

        # v6.2: Config 범위 검증
        from .config_validator import ConfigValidator
        config_warnings = ConfigValidator.validate(self.config)
        for w in config_warnings:
            logger.warning("Config: %s", w)

        # v8.0: Gate + Score Hybrid 파이프라인
        v8_cfg = self.config.get("v8_hybrid", {})
        self.v8_mode = v8_cfg.get("enabled", False)
        if self.v8_mode:
            from .v8_pipeline import QuantumPipelineV8
            self.v8_pipeline = QuantumPipelineV8(self.config)
        else:
            self.v8_pipeline = None

        logger.info(
            "SignalEngine v8.0: v8_mode=%s, martin=%s, rttp=%s, extreme_vol=%s, consensus=%s",
            self.v8_mode, self.martin_enabled, self.rttp_enabled,
            self.extreme_vol_enabled, self.consensus_mode,
        )

    # ══════════════════════════════════════════════
    #  v3.1 News Gate 설정
    # ══════════════════════════════════════════════

    def set_news_gate(self, ticker: str, gate_result: NewsGateResult) -> None:
        """특정 종목의 News Gate 결과를 설정"""
        self._news_gates[ticker] = gate_result

    def clear_news_gates(self) -> None:
        """모든 News Gate 결과 초기화"""
        self._news_gates.clear()

    # ══════════════════════════════════════════════
    #  ZONE SCORING (기존 BES와 동일)
    # ══════════════════════════════════════════════

    def score_atr_pullback(self, pullback_atr: float) -> float:
        """ATR 조정폭 -> 점수"""
        if pd.isna(pullback_atr) or pullback_atr < 0:
            return 0.0
        ranges = self.strategy["atr_pullback_ranges"]
        if pullback_atr < ranges["noise"][1]:
            return 0.0
        elif pullback_atr < ranges["shallow"][1]:
            return 0.4
        elif pullback_atr < ranges["healthy"][1]:
            return 0.7
        elif pullback_atr < ranges["sweet_spot"][1]:
            return 1.0
        elif pullback_atr < ranges["deep"][1]:
            return 0.7
        else:
            return 0.4

    def score_valuation(self, ticker: str, df: pd.DataFrame, idx: int) -> float:
        """밸류에이션 종합 점수"""
        current_per = df["fund_PER"].iloc[idx] if "fund_PER" in df.columns else np.nan
        sector_avg = self.fundamental.get_sector_avg_per(ticker)
        per_score = self.fundamental.calc_trailing_value_score(current_per, sector_avg)
        eps_score = self.fundamental.calc_eps_revision_score(df, idx, lookback_days=60)
        return self.fundamental.calc_combined_value_score(per_score, eps_score)

    def score_supply_demand(self, df: pd.DataFrame, idx: int) -> float:
        """수급 종합 점수 (v4.5: Dynamic RSI 통합)"""
        row = df.iloc[idx]

        # RSI — Dynamic RSI 적응형 스코어링
        rsi = row.get("rsi_14", np.nan)
        dynamic_rsi_cfg = self.strategy.get("dynamic_rsi", {})
        use_dynamic = dynamic_rsi_cfg.get("enabled", False)

        if pd.isna(rsi):
            rsi_score = 0.5
        elif use_dynamic:
            # v4.5: 변동성 적응형 과매도 기준
            dyn_threshold = row.get("dynamic_rsi_oversold", 30)
            if pd.isna(dyn_threshold):
                dyn_threshold = 30
            rsi_rising = row.get("rsi_rising", 0)
            rsi_ema9 = row.get("rsi_ema9", np.nan)

            if rsi <= dyn_threshold and rsi_rising == 1 and (not pd.isna(rsi_ema9) and rsi > rsi_ema9):
                rsi_score = 1.0   # 과매도 반전 최적점
            elif rsi <= dyn_threshold:
                rsi_score = 0.8   # 과매도 구간 (반전 미확인)
            elif rsi <= dyn_threshold + 10:
                rsi_score = 0.6   # 과매도 근접
            elif rsi <= 55:
                rsi_score = 0.4   # 중립
            else:
                rsi_score = 0.2   # 과매수 방향
        else:
            # 기존 고정 RSI sweet spot
            rsi_range = self.strategy["rsi_sweet_spot"]
            if rsi_range[0] <= rsi <= rsi_range[1]:
                rsi_score = 1.0
            elif rsi < rsi_range[0]:
                rsi_score = 0.7
            elif rsi <= 50:
                rsi_score = 0.5
            else:
                rsi_score = 0.2

        # Stoch RSI
        stoch_k = row.get("stoch_rsi_k", np.nan)
        threshold = self.strategy["stoch_rsi_threshold"]
        if pd.isna(stoch_k):
            stoch_score = 0.5
        elif stoch_k <= threshold:
            stoch_score = 1.0
        elif stoch_k <= 40:
            stoch_score = 0.7
        elif stoch_k <= 60:
            stoch_score = 0.4
        else:
            stoch_score = 0.1

        # 거래량 수축
        vol_ma5 = row.get("volume_ma5", np.nan)
        vol_ma20 = row.get("volume_ma20", np.nan)
        if pd.isna(vol_ma5) or pd.isna(vol_ma20) or vol_ma20 == 0:
            vol_score = 0.5
        else:
            vol_ratio = vol_ma5 / vol_ma20
            if vol_ratio < 0.6:
                vol_score = 1.0
            elif vol_ratio < 0.8:
                vol_score = 0.8
            elif vol_ratio < 1.0:
                vol_score = 0.5
            elif vol_ratio < 1.5:
                vol_score = 0.3
            else:
                vol_score = 0.1

        return round(rsi_score * 0.30 + stoch_score * 0.30 + vol_score * 0.40, 3)

    def calc_trend_adjustment(self, df: pd.DataFrame, idx: int) -> float:
        """추세 보정 = MIN(ADX/30, 1.0)"""
        adx = df["adx_14"].iloc[idx] if "adx_14" in df.columns else np.nan
        if pd.isna(adx) or adx <= 0:
            return 0.5
        return min(adx / 30.0, 1.0)

    def calc_zone_score(self, ticker: str, df: pd.DataFrame, idx: int,
                        gate_result: dict) -> dict:
        """
        Zone Score (= 기존 BES). 위치의 매력도만 평가.
        v6.4: Consensus Bonus (팩터 수렴 보너스) 추가.
        트리거와 분리됨.
        """
        row = df.iloc[idx]
        pullback_atr = row.get("pullback_atr", np.nan)
        atr_score = self.score_atr_pullback(pullback_atr)
        value_score = self.score_valuation(ticker, df, idx)
        supply_score = self.score_supply_demand(df, idx)
        trend_adj = self.calc_trend_adjustment(df, idx)

        drs = gate_result["drs_value"]
        dist_adj = 1.0 - drs

        weights = self.strategy["weights"]
        raw_score = (weights["atr_pullback"] * atr_score +
                     weights["forward_value"] * value_score +
                     weights["supply_demand"] * supply_score)
        zone = round(min(max(raw_score * trend_adj * dist_adj, 0.0), 1.0), 3)

        # v6.4: Consensus Bonus — 3대 BES 팩터 동시 양호 시 가산
        consensus_cfg = self.strategy.get("consensus_bonus", {})
        consensus_bonus = 0.0
        consensus_tag = ""
        if consensus_cfg.get("enabled", False):
            # 각 팩터가 "양호" 기준(0.5 이상)인지 체크
            factor_scores = [atr_score, value_score, supply_score]
            positive_count = sum(1 for s in factor_scores if s >= 0.5)

            if positive_count == 3:
                # 3팩터 전부 양호 → FULL CONSENSUS
                bonus_pct = consensus_cfg.get("full_bonus_pct", 0.15)
                zone = min(zone * (1 + bonus_pct), 1.0)
                consensus_bonus = bonus_pct
                consensus_tag = "FULL_CONSENSUS"
            elif positive_count == 2:
                # 2팩터 양호 → STRONG
                bonus_pct = consensus_cfg.get("strong_bonus_pct", 0.08)
                zone = min(zone * (1 + bonus_pct), 1.0)
                consensus_bonus = bonus_pct
                consensus_tag = "STRONG"
            else:
                consensus_tag = "WEAK"

        return {
            "zone_score": zone,
            "components": {
                "atr_pullback_score": round(atr_score, 3),
                "forward_value_score": round(value_score, 3),
                "supply_demand_score": round(supply_score, 3),
                "raw_score": round(raw_score, 3),
                "trend_adjustment": round(trend_adj, 3),
                "distribution_decay": round(dist_adj, 3),
                "drs": round(drs, 3),
                "pullback_atr_mult": round(pullback_atr, 2) if not pd.isna(pullback_atr) else 0,
                "consensus_bonus": round(consensus_bonus, 3),
                "consensus_tag": consensus_tag,
            }
        }

    # ══════════════════════════════════════════════
    #  TRIGGER ENGINE (v2.1 신규)
    # ══════════════════════════════════════════════

    def _check_higher_low(self, df: pd.DataFrame, idx: int, lookback: int = 20) -> bool:
        """
        Higher Low 확인: 최근 20일 내 저점이 그 이전 저점보다 높은지.
        조정이 끝나가는 신호.
        """
        if idx < lookback * 2:
            return False
        recent_low = df["low"].iloc[idx - lookback:idx + 1].min()
        prev_low = df["low"].iloc[idx - lookback * 2:idx - lookback + 1].min()
        return recent_low > prev_low

    def _calc_swing_low(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        """최근 N일 스윙 저점 (Impulse 손절선)"""
        start = max(0, idx - lookback)
        return df["low"].iloc[start:idx + 1].min()

    def check_impulse_trigger(self, df: pd.DataFrame, idx: int) -> TriggerResult:
        """
        Trigger-1: 시동 트리거 (Impulse) - 급등 초입 선점

        조건 (3개 중 2개 이상 충족):
        1. 전일 고가 돌파 마감: close > prev_high
        2. 거래량 서지: volume >= volume_ma20 x 1.5
        3. 종가 > 5MA

        추가 확인: Higher Low (있으면 신뢰도 +0.1)
        """
        cfg = self.triggers_cfg.get("impulse", {})
        staged = self.triggers_cfg.get("staged_entry", {})
        min_conds = cfg.get("min_conditions", 2)
        vol_mult = cfg.get("volume_surge_mult", 1.5)

        if idx < 2:
            return TriggerResult(TriggerType.NONE, {}, 0, 0, 0, 0)

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        conditions = {}

        # 조건 1: 전일 고가 돌파 마감
        conditions["prev_high_breakout"] = bool(row["close"] > prev["high"])

        # 조건 2: 거래량 서지
        vol_ma20 = row.get("volume_ma20", np.nan)
        if pd.isna(vol_ma20) or vol_ma20 == 0:
            conditions["volume_surge"] = False
        else:
            conditions["volume_surge"] = bool(row["volume"] >= vol_ma20 * vol_mult)

        # 조건 3: 종가 > 5MA
        sma5 = df["close"].iloc[max(0, idx - 4):idx + 1].mean()
        conditions["close_above_ma5"] = bool(row["close"] > sma5)

        met_count = sum(conditions.values())

        if met_count < min_conds:
            return TriggerResult(TriggerType.NONE, conditions, met_count, 0, 0, 0)

        # Higher Low 보너스
        has_hl = self._check_higher_low(df, idx)
        conditions["higher_low"] = has_hl
        confidence = 0.6 + (met_count - min_conds) * 0.15
        if has_hl:
            confidence += 0.1

        # 아래꼬리 확인 (반전 캔들)
        body = abs(row["close"] - row["open"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        if body > 0 and lower_wick > body * 1.5:
            conditions["lower_wick"] = True
            confidence += 0.05
        else:
            conditions["lower_wick"] = False

        confidence = min(confidence, 1.0)

        return TriggerResult(
            trigger_type=TriggerType.IMPULSE,
            conditions_met=conditions,
            conditions_count=met_count,
            stop_loss_pct=cfg.get("stop_loss_pct", 0.03),
            entry_stage_pct=staged.get("stage1_impulse_pct", 0.40),
            confidence=round(confidence, 3),
        )

    def check_confirm_trigger(self, df: pd.DataFrame, idx: int) -> TriggerResult:
        """
        Trigger-2: 확인 트리거 (Confirm) - 안전 진입

        조건 (전부 충족):
        1. 20MA 위 복귀 + 2일 연속 유지
        2. 거래량 유지 (5일 평균 >= 20일 평균 x 0.8)
        3. 최근 10일간 저점 갱신 없음
        4. RSI > 50
        """
        cfg = self.triggers_cfg.get("confirm", {})
        staged = self.triggers_cfg.get("staged_entry", {})
        recovery_days = cfg.get("ma20_recovery_days", 2)
        rsi_threshold = cfg.get("rsi_above", 50)

        if idx < 20:
            return TriggerResult(TriggerType.NONE, {}, 0, 0, 0, 0)

        row = df.iloc[idx]
        conditions = {}

        # 조건 1: 20MA 위 복귀 + N일 연속 유지
        sma20 = row.get("sma_20", np.nan)
        if pd.isna(sma20):
            conditions["ma20_recovery"] = False
        else:
            above_count = 0
            for j in range(recovery_days):
                check_idx = idx - j
                if check_idx < 0:
                    break
                check_sma = df["sma_20"].iloc[check_idx]
                if not pd.isna(check_sma) and df["close"].iloc[check_idx] > check_sma:
                    above_count += 1
            conditions["ma20_recovery"] = (above_count >= recovery_days)

        # 조건 2: 거래량 유지
        vol_ma5 = row.get("volume_ma5", np.nan)
        vol_ma20 = row.get("volume_ma20", np.nan)
        if pd.isna(vol_ma5) or pd.isna(vol_ma20) or vol_ma20 == 0:
            conditions["volume_maintain"] = True
        else:
            conditions["volume_maintain"] = bool(vol_ma5 >= vol_ma20 * 0.8)

        # 조건 3: 최근 10일 저점 갱신 없음
        if idx >= 20:
            recent_low = df["low"].iloc[idx - 9:idx + 1].min()
            prev_low = df["low"].iloc[max(0, idx - 20):idx - 9].min()
            conditions["no_new_low"] = bool(recent_low >= prev_low * 0.99)
        else:
            conditions["no_new_low"] = False

        # 조건 4: RSI > 50
        rsi = row.get("rsi_14", np.nan)
        if pd.isna(rsi):
            conditions["rsi_recovery"] = False
        else:
            conditions["rsi_recovery"] = bool(rsi > rsi_threshold)

        all_met = all(conditions.values())
        met_count = sum(conditions.values())

        if not all_met:
            return TriggerResult(TriggerType.NONE, conditions, met_count, 0, 0, 0)

        confidence = 0.75 + met_count * 0.05
        confidence = min(confidence, 1.0)

        return TriggerResult(
            trigger_type=TriggerType.CONFIRM,
            conditions_met=conditions,
            conditions_count=met_count,
            stop_loss_pct=cfg.get("stop_loss_pct", 0.05),
            entry_stage_pct=staged.get("stage2_confirm_pct", 0.40),
            confidence=round(confidence, 3),
        )

    def check_breakout_trigger(self, df: pd.DataFrame, idx: int) -> TriggerResult:
        """
        Trigger-3: 돌파 트리거 (Breakout) - 보유 종목 추가 매수

        조건 (둘 다 충족):
        1. 종가가 60일 최고가 돌파
        2. 거래량 >= 20일 평균 x 1.5
        """
        staged = self.triggers_cfg.get("staged_entry", {})
        vol_mult = staged.get("breakout_volume_mult", 1.5)

        if idx < 60:
            return TriggerResult(TriggerType.NONE, {}, 0, 0, 0, 0)

        row = df.iloc[idx]
        conditions = {}

        # 조건 1: 60일 고점 돌파
        if idx >= 1:
            prev_high_60 = df["high"].iloc[max(0, idx - 60):idx].max()
            conditions["breakout_high60"] = bool(row["close"] > prev_high_60)
        else:
            conditions["breakout_high60"] = False

        # 조건 2: 거래량 서지
        vol_ma20 = row.get("volume_ma20", np.nan)
        if pd.isna(vol_ma20) or vol_ma20 == 0:
            conditions["volume_surge"] = False
        else:
            conditions["volume_surge"] = bool(row["volume"] >= vol_ma20 * vol_mult)

        all_met = all(conditions.values())

        if not all_met:
            return TriggerResult(TriggerType.NONE, conditions, sum(conditions.values()), 0, 0, 0)

        return TriggerResult(
            trigger_type=TriggerType.BREAKOUT,
            conditions_met=conditions,
            conditions_count=2,
            stop_loss_pct=0.03,
            entry_stage_pct=staged.get("stage3_breakout_pct", 0.20),
            confidence=0.85,
        )

    def check_trend_continuation(self, df: pd.DataFrame, idx: int) -> TriggerResult:
        """
        Trigger-4: 추세 지속 (Trend Continuation) - Grade F 우회

        MA20 위에서 강한 상승 추세가 확인될 때 보수적 진입 허용.
        7개 조건 중 min_conditions개 이상 충족 시 발동.

        조건:
        1. 종가 > MA20 AND MA60 (이중 이평선 위)
        2. ADX >= 25 (강한 추세)
        3. +DI > -DI (상승 방향성)
        4. RSI 50~72 (강세 + 과매수 아닌)
        5. MACD > Signal (모멘텀 유지)
        6. 거래량 >= 20일 평균 x 0.8 (참여도)
        7. OBV 5일 변화 > 0 (매집)
        """
        tc_cfg = self.triggers_cfg.get("trend_continuation", {})
        if not tc_cfg.get("enabled", False):
            return TriggerResult(TriggerType.NONE, {}, 0, 0, 0, 0)

        min_conds = tc_cfg.get("min_conditions", 5)

        if idx < 60:
            return TriggerResult(TriggerType.NONE, {}, 0, 0, 0, 0)

        row = df.iloc[idx]
        close = row["close"]
        conditions = {}

        # 조건 1: 종가 > MA20 AND MA60
        sma20 = row.get("sma_20", np.nan)
        sma60 = row.get("sma_60", np.nan)
        above_ma20 = not pd.isna(sma20) and close > sma20
        above_ma60 = not pd.isna(sma60) and close > sma60
        conditions["price_above_mas"] = bool(above_ma20 and above_ma60)

        # 조건 2: ADX >= 25
        adx = row.get("adx_14", np.nan)
        adx_min = tc_cfg.get("adx_min", 25)
        conditions["adx_strong"] = bool(not pd.isna(adx) and adx >= adx_min)

        # 조건 3: +DI > -DI
        plus_di = row.get("plus_di", np.nan)
        minus_di = row.get("minus_di", np.nan)
        conditions["plus_di_above"] = bool(
            not pd.isna(plus_di) and not pd.isna(minus_di)
            and plus_di > minus_di
        )

        # 조건 4: RSI 50~72
        rsi = row.get("rsi_14", np.nan)
        rsi_range = tc_cfg.get("rsi_range", [50, 72])
        conditions["rsi_strong"] = bool(
            not pd.isna(rsi) and rsi_range[0] <= rsi <= rsi_range[1]
        )

        # 조건 5: MACD > Signal
        macd = row.get("macd", np.nan)
        macd_sig = row.get("macd_signal", np.nan)
        conditions["macd_bullish"] = bool(
            not pd.isna(macd) and not pd.isna(macd_sig)
            and macd > macd_sig
        )

        # 조건 6: 거래량 >= 20일 평균 x 0.8
        vol = row.get("volume", 0)
        vol_ma20 = row.get("volume_ma20", np.nan)
        vol_ratio = tc_cfg.get("volume_min_ratio", 0.8)
        conditions["volume_adequate"] = bool(
            not pd.isna(vol_ma20) and vol_ma20 > 0
            and vol >= vol_ma20 * vol_ratio
        )

        # 조건 7: OBV 5일 변화 > 0
        obv_change = row.get("obv_change_5d", np.nan)
        if pd.isna(obv_change):
            # OBV 변화 데이터가 없으면 직접 계산
            if "obv" in df.columns and idx >= 5:
                obv_now = df["obv"].iloc[idx]
                obv_prev = df["obv"].iloc[idx - 5]
                if not pd.isna(obv_now) and not pd.isna(obv_prev):
                    obv_change = obv_now - obv_prev
        conditions["obv_rising"] = bool(
            not pd.isna(obv_change) and obv_change > 0
        )

        met_count = sum(conditions.values())

        if met_count < min_conds:
            return TriggerResult(TriggerType.NONE, conditions, met_count, 0, 0, 0)

        # 과매수 최종 차단 (RSI > max_rsi)
        max_rsi = tc_cfg.get("max_rsi", 72)
        if not pd.isna(rsi) and rsi > max_rsi:
            return TriggerResult(TriggerType.NONE, conditions, met_count, 0, 0, 0)

        # 신뢰도: 기본 0.5 + 조건 초과분 x 0.08
        confidence = 0.5 + (met_count - min_conds) * 0.08
        confidence = min(confidence, 0.85)  # 풀백 진입보다 낮은 최대 신뢰도

        return TriggerResult(
            trigger_type=TriggerType.TREND_CONT,
            conditions_met=conditions,
            conditions_count=met_count,
            stop_loss_pct=tc_cfg.get("stop_loss_pct", 0.03),
            entry_stage_pct=tc_cfg.get("position_pct", 0.50),
            confidence=round(confidence, 3),
        )

    # ══════════════════════════════════════════════
    #  MASTER: Zone + Trigger 통합
    # ══════════════════════════════════════════════

    def calculate_signal(self, ticker: str, df: pd.DataFrame, idx: int) -> dict:
        """
        v3.1 Pipeline 시그널 계산.
        v8.0: v8_hybrid.enabled=true이면 Gate+Score Hybrid 파이프라인으로 위임.

        [v3.1] L-1_news_gate → 뉴스 등급(A/B/C) → 파라미터 조정
        L0_pre_gate   → Pre-screening (매출/거래대금/수익성)
        L0_grade      → Zone Score + Grade (A/B/C/F)
        L1_regime     → HMM 레짐 (Accumulation 확률)
        L2_ou         → OU 필터 (z-score, half-life, SNR)
        L3_momentum   → 모멘텀 (거래량 서지 + MA60 slope)
        L4_smart_money→ Smart Money Z-score + [v3.1] 매집/다이버전스
        L5_risk       → 손익비 (A급: 1.7→1.2)
        L6_trigger    → Impulse/Confirm/Breakout
        """
        # v8.0: Gate + Score Hybrid 모드
        if self.v8_mode and self.v8_pipeline:
            row = df.iloc[idx]
            date_str = str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx])
            return self.v8_pipeline.scan_single(row, ticker=ticker, date=date_str)

        row = df.iloc[idx]
        date = df.index[idx] if hasattr(df.index[idx], "strftime") else str(df.index[idx])

        result = {
            "ticker": ticker,
            "date": date,
            "zone_score": 0.0,
            "bes_score": 0.0,
            "grade": "F",
            "trigger_type": "none",
            "trigger_confidence": 0.0,
            "trigger_conditions": {},
            "signal": False,
            "entry_stage_pct": 0.0,
            "stop_loss_pct": 0.0,
            "position_ratio": 0.0,
            "components": {},
            "entry_price": 0,
            "stop_loss": 0,
            "target_price": 0,
            "risk_reward_ratio": 0.0,
            "atr_value": 0.0,
            "gate_result": {},
            # v3.1
            "news_grade": "C",
            "news_action": "ignore",
            "accum_phase": "none",
            "divergence_type": "none",
        }

        # 진단 레코드
        diag = self.diagnostic.new_record(str(date), ticker)

        # v5.0: LayerVote 수집 리스트
        consensus_votes: list[LayerVote] = []

        # ── [v3.2] L-1_news_gate: 뉴스 등급 + 파라미터 오버라이드 + 스코어 부스트 ──
        news_gate = self._news_gates.get(ticker)
        param_overrides = {}
        news_score_boost = 0.0  # v3.2: Zone Score 가산분

        if news_gate and news_gate.grade != NewsGrade.C:
            param_overrides = news_gate.param_overrides or {}
            news_score_boost = getattr(news_gate, "score_boost", 0.0)
            result["news_grade"] = news_gate.grade.value
            result["news_action"] = news_gate.action.value
            result["news_score_boost"] = news_score_boost

            diag.add_layer(LayerResult(
                name="L-1_news_gate",
                passed=True,
                details={
                    "grade": news_gate.grade.value,
                    "action": news_gate.action.value,
                    "overrides": list(param_overrides.keys()),
                    "score_boost": news_score_boost,
                    "living_issues": len(getattr(news_gate, "living_issues", [])),
                    "has_earnings": getattr(news_gate, "earnings_estimate", None) is not None,
                },
            ))

            # B급 watchlist → C급과 동일하게 기본 파이프라인 진행 (파라미터 변경 없음)
            if news_gate.action == EventDrivenAction.IGNORE:
                param_overrides = {}
        elif news_gate and news_gate.grade == NewsGrade.C:
            # C급이지만 살아있는 이슈/실적 정보가 있으면 boost만 적용
            news_score_boost = getattr(news_gate, "score_boost", 0.0)
            result["news_score_boost"] = news_score_boost
            diag.add_layer(LayerResult(
                name="L-1_news_gate",
                passed=True,
                details={
                    "grade": "C", "action": "ignore",
                    "score_boost": news_score_boost,
                },
            ))
        else:
            # 뉴스 없음 → L-1 통과 (기본 파이프라인)
            diag.add_layer(LayerResult(
                name="L-1_news_gate",
                passed=True,
                details={"grade": "C", "action": "no_news"},
            ))

        # ── v6.1 RTTP 뉴스 강화: enhance_gate_result() 호출 ──
        if self.rttp_enabled and self.rttp_processor and news_gate:
            try:
                rttp_enhancement = self.rttp_processor.enhance_gate_result(news_gate, df, idx)
                news_gate.source_weighted_score = rttp_enhancement.source_weighted_score
                news_gate.engagement_depth = rttp_enhancement.engagement_depth

                # v6.1: RTTP 소스 권위에 의한 score boost 직접 가산
                src_score = rttp_enhancement.source_weighted_score
                if src_score >= 0.9:
                    news_score_boost += 0.05  # DART급
                elif src_score >= 0.7:
                    news_score_boost += 0.03  # 증권사급

                # RTTP engagement boost도 가산
                news_score_boost += rttp_enhancement.rttp_boost

                result["rttp"] = {
                    "source_weighted_score": rttp_enhancement.source_weighted_score,
                    "engagement_depth": rttp_enhancement.engagement_depth,
                    "rttp_boost": rttp_enhancement.rttp_boost,
                    "source_tier": rttp_enhancement.source_tier,
                }
            except Exception as e:
                logger.warning("RTTP enhancement failed for %s: %s", ticker, e)

        # ── v6.0 L-1_rttp: RTTP 소스가중 + 인게이지먼트 투표 ──
        if news_gate:
            rttp_score = getattr(news_gate, "source_weighted_score", 0.0)
            rttp_engage = getattr(news_gate, "engagement_depth", 0.0)
            rttp_conf = (rttp_score * 0.6 + min(rttp_engage / 5.0, 1.0) * 0.4)
            rttp_passed = rttp_conf > 0.3
            consensus_votes.append(LayerVote("L-1_rttp", rttp_passed, rttp_conf))

        # ── L0_pre_gate: Pre-screening + Trend + DRS ──
        gate = self.screener.check_all_gates(ticker, df, idx)
        result["gate_result"] = gate

        diag.add_layer(LayerResult(
            name="L0_pre_gate",
            passed=gate["passed"],
            block_reason=gate.get("fail_reason", "") or "",
        ))
        if not gate["passed"]:
            return result

        # ── L0_grade: Zone Score + Grade ──
        zone_info = self.calc_zone_score(ticker, df, idx, gate)
        raw_zone_score = zone_info["zone_score"]

        # v3.2: 뉴스 스코어 부스트 가산
        zone_score = min(raw_zone_score + news_score_boost, 1.0)
        result["zone_score"] = zone_score
        result["bes_score"] = zone_score
        result["components"] = zone_info["components"]
        if news_score_boost > 0:
            result["components"]["news_boost"] = news_score_boost
            result["components"]["raw_zone_score"] = raw_zone_score

        grades = self.strategy["grades"]
        if zone_score >= grades["A"]["min_bes"]:
            grade = "A"
        elif zone_score >= grades["B"]["min_bes"]:
            grade = "B"
        elif zone_score >= grades["C"]["min_bes"]:
            grade = "C"
        else:
            grade = "F"
        result["grade"] = grade

        # grade_blocked 추적 (v7.1: B등급 이상만 통과, C등급 차단)
        min_grade = self.strategy.get("min_entry_grade", "B")
        if min_grade == "B":
            grade_passed = grade in ("A", "B")
            block_reason = f"grade_{grade}" if not grade_passed else ""
        else:
            grade_passed = grade != "F"
            block_reason = "grade_F" if not grade_passed else ""
        diag.add_layer(LayerResult(
            name="L0_grade",
            passed=grade_passed,
            block_reason=block_reason,
            details={"zone_score": zone_score, "grade": grade, "min_grade": min_grade},
        ))
        consensus_votes.append(LayerVote("L0_grade", grade_passed, zone_score))
        if not grade_passed:
            # ── v3.2: Grade F 우회 — 추세 지속(Trend Continuation) 체크 ──
            # 뉴스 부스트가 높으면 min_conditions 완화 (5→4)
            if news_score_boost >= 0.08:
                tc_cfg = self.triggers_cfg.get("trend_continuation", {})
                original_min = tc_cfg.get("min_conditions", 5)
                tc_cfg["min_conditions"] = max(original_min - 1, 3)
            trend_cont = self.check_trend_continuation(df, idx)
            # 원복
            if news_score_boost >= 0.08:
                tc_cfg["min_conditions"] = original_min
            if trend_cont.trigger_type == TriggerType.TREND_CONT:
                diag.add_layer(LayerResult(
                    name="L0_trend_cont",
                    passed=True,
                    details={
                        "conditions": trend_cont.conditions_met,
                        "met_count": trend_cont.conditions_count,
                        "confidence": trend_cont.confidence,
                    },
                ))

                # 추세 지속 경로: L1~L4 스킵 (풀백 전용 레이어)
                # 직접 L5_risk 계산
                close = row["close"]
                atr_val = row.get("atr_14", 0)
                stop_pct = trend_cont.stop_loss_pct
                pct_stop = close * (1 - stop_pct)
                atr_stop = close - atr_val * 1.5  # 추세 지속용 ATR 1.5배 손절
                stop_price = max(pct_stop, atr_stop)

                tc_cfg = self.triggers_cfg.get("trend_continuation", {})
                target_mult = tc_cfg.get("target_atr_mult", 3.0)
                target_price = close + atr_val * target_mult

                risk = close - stop_price
                reward = target_price - close
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

                # 추세 지속은 최소 RR 1.2 (보수적이지만 추세 신뢰)
                # v3.2: 뉴스 부스트가 있으면 RR 기준 완화 (1.2→1.0)
                min_rr_trend = 1.0 if news_score_boost >= 0.10 else 1.2
                if rr_ratio >= min_rr_trend:
                    # 포지션 비중: 정상 C등급의 position_pct 배
                    # v3.2: 뉴스 부스트 시 비중 상향 (50%→70%)
                    base_ratio = grades["C"]["position_ratio"]
                    position_pct = trend_cont.entry_stage_pct
                    if news_score_boost >= 0.10:
                        position_pct = min(position_pct * 1.4, 0.80)
                    position_ratio = base_ratio * position_pct

                    # 뉴스 부스트로 confidence 상향
                    final_confidence = min(
                        trend_cont.confidence + news_score_boost * 0.5, 1.0
                    )

                    diag.add_layer(LayerResult(
                        name="L5_risk_trend",
                        passed=True,
                        details={"rr": rr_ratio, "stop": int(stop_price),
                                 "target": int(target_price),
                                 "news_boost": news_score_boost},
                    ))
                    diag.final_signal = True

                    # 등급 결정: 뉴스 있으면 "TN"(Trend+News), 없으면 "T"
                    trend_grade = "TN" if news_score_boost >= 0.08 else "T"

                    result.update({
                        "zone_score": zone_score,
                        "bes_score": zone_score,
                        "grade": trend_grade,
                        "trigger_type": TriggerType.TREND_CONT.value,
                        "trigger_confidence": final_confidence,
                        "trigger_conditions": trend_cont.conditions_met,
                        "signal": True,
                        "entry_stage_pct": position_pct,
                        "stop_loss_pct": stop_pct,
                        "position_ratio": round(position_ratio, 3),
                        "entry_price": int(close),
                        "stop_loss": int(stop_price),
                        "target_price": int(target_price),
                        "risk_reward_ratio": rr_ratio,
                        "atr_value": round(atr_val, 1),
                        "news_score_boost": news_score_boost,
                    })
                    return result
                else:
                    diag.add_layer(LayerResult(
                        name="L5_risk_trend",
                        passed=False,
                        block_reason=f"low_rr({rr_ratio:.1f})",
                    ))
            else:
                diag.add_layer(LayerResult(
                    name="L0_trend_cont",
                    passed=False,
                    block_reason=f"insufficient({trend_cont.conditions_count}/5)",
                    details={"conditions": trend_cont.conditions_met},
                ))
            return result

        # ── v6.0 극한 변동성 탐지 (L1 전 단계) ──
        extreme_vol_result = None
        extreme_vol_direction = None
        if self.extreme_vol_enabled and self.extreme_vol_detector:
            try:
                extreme_vol_result = self.extreme_vol_detector.detect(df, idx)
                if extreme_vol_result.is_extreme:
                    extreme_vol_direction = extreme_vol_result.direction
                    result["extreme_volatility"] = {
                        "is_extreme": True,
                        "atr_ratio": extreme_vol_result.atr_ratio,
                        "vol_ratio": extreme_vol_result.vol_ratio,
                        "daily_range_pct": extreme_vol_result.daily_range_pct,
                        "direction": extreme_vol_result.direction,
                        "is_capitulation": extreme_vol_result.is_capitulation,
                        "confidence": extreme_vol_result.confidence,
                    }
            except Exception as e:
                logger.warning("Extreme vol detection failed for %s: %s", ticker, e)

        # ── v6.0 L1_extreme_vol: 극한 변동성 투표 ──
        if extreme_vol_result and extreme_vol_result.is_extreme:
            evol_passed = extreme_vol_direction in ("capitulation", "bullish_breakout")
            consensus_votes.append(LayerVote(
                "L1_extreme_vol", evol_passed, extreme_vol_result.confidence,
            ))

        # ── L1_regime: HMM 레짐 체크 (+ 극한 변동성 방향 연동) ──
        p_accum_threshold = self.regime_cfg.get("p_accum_entry", 0.40)
        p_accum = row.get("P_Accum", np.nan)

        # v6.0: 극한 변동성 방향에 따른 레짐 판단
        if extreme_vol_direction in ("bearish_breakdown", "ambiguous"):
            regime_passed = False
            regime_reason = f"extreme_vol_{extreme_vol_direction}"
        elif extreme_vol_direction in ("capitulation", "bullish_breakout"):
            regime_passed = True
            regime_reason = ""
        elif pd.isna(p_accum):
            regime_passed = True  # 레짐 데이터 없으면 통과
            regime_reason = ""
        elif p_accum < p_accum_threshold:
            regime_passed = False
            regime_reason = "low_accum"
        else:
            regime_passed = True
            regime_reason = ""

        diag.add_layer(LayerResult(
            name="L1_regime",
            passed=regime_passed,
            block_reason=regime_reason,
            details={"P_Accum": float(p_accum) if not pd.isna(p_accum) else None},
        ))
        consensus_votes.append(LayerVote(
            "L1_regime", regime_passed,
            float(p_accum) if not pd.isna(p_accum) else 0.0,
        ))
        if not regime_passed:
            if self.consensus_mode:
                pass  # 합의 모드: 레이어 실패해도 계속 진행
            else:
                return result

        # ── L2_ou: OU 필터 ──
        ou_z_entry = self.ou_cfg.get("z_entry", -1.2)
        ou_hl_min = self.ou_cfg.get("half_life_min", 2)
        ou_hl_max = self.ou_cfg.get("half_life_max", 25)
        ou_snr_min = self.ou_cfg.get("snr_min", 0.15)

        ou_passed, ou_reason = OUEstimator.check_ou_gate(
            row,
            z_entry=ou_z_entry,
            hl_min=ou_hl_min,
            hl_max=ou_hl_max,
            snr_min=ou_snr_min,
        )

        diag.add_layer(LayerResult(
            name="L2_ou",
            passed=ou_passed,
            block_reason=ou_reason,
        ))
        ou_z = row.get("ou_zscore", np.nan)
        consensus_votes.append(LayerVote(
            "L2_ou", ou_passed,
            min(abs(float(ou_z)) / 3.0, 1.0) if not pd.isna(ou_z) else 0.0,
        ))
        if not ou_passed:
            if self.consensus_mode:
                pass
            else:
                return result

        # ── L3_momentum: Martin Momentum 또는 기존 거래량+slope ──
        mom_passed = True
        mom_reason = ""
        martin_result = None

        if self.martin_enabled and self.martin_engine:
            try:
                # v6.0: Martin Momentum 평가
                martin_result = self.martin_engine.evaluate(df, idx)
                result["martin_momentum"] = {
                    "ema2_norm": martin_result.ema2_normalized,
                    "signal_type": martin_result.signal_type,
                    "trend_strength": martin_result.trend_strength,
                    "confidence": martin_result.confidence,
                    "in_dead_zone": martin_result.in_dead_zone,
                    "vol_weight": martin_result.vol_normalized_weight,
                }

                if martin_result.in_dead_zone:
                    mom_passed = False
                    mom_reason = "martin_dead_zone"
                elif martin_result.confidence < self.martin_engine.min_confidence:
                    mom_passed = False
                    mom_reason = f"martin_low_conf({martin_result.confidence:.2f})"

                diag.add_layer(LayerResult(
                    name="L3_momentum",
                    passed=mom_passed,
                    block_reason=mom_reason,
                    details={
                        "engine": "martin",
                        "ema2_norm": martin_result.ema2_normalized,
                        "signal_type": martin_result.signal_type,
                        "confidence": martin_result.confidence,
                        "dead_zone": martin_result.in_dead_zone,
                    },
                ))
                consensus_votes.append(LayerVote(
                    "L3_momentum", mom_passed, martin_result.confidence,
                ))
                # v6.0: Martin 전용 투표 (합의 다양성 강화)
                martin_voted = martin_result.signal_type in ("trend", "reversal")
                consensus_votes.append(LayerVote(
                    "L3_martin", martin_voted, martin_result.trend_strength,
                ))
            except Exception as e:
                logger.warning("Martin evaluation failed for %s: %s", ticker, e)
                martin_result = None
        else:
            # 기존 로직: 거래량 서지 + MA60 slope
            vol_surge_min = self.momentum_cfg.get("vol_surge_min", 1.2)
            slope_min = self.momentum_cfg.get("slope_ma60_min", -0.5)

            vol_surge = row.get("volume_surge_ratio", np.nan)
            slope_60 = row.get("slope_ma60", np.nan)

            if not pd.isna(vol_surge) and not pd.isna(slope_60):
                if vol_surge < vol_surge_min and slope_60 < slope_min:
                    mom_passed = False
                    mom_reason = "weak_momentum"

            diag.add_layer(LayerResult(
                name="L3_momentum",
                passed=mom_passed,
                block_reason=mom_reason,
            ))
            vol_surge = row.get("volume_surge_ratio", np.nan)
            consensus_votes.append(LayerVote(
                "L3_momentum", mom_passed,
                min(float(vol_surge) / 2.0, 1.0) if not pd.isna(vol_surge) else 0.0,
            ))

        if not mom_passed:
            if self.consensus_mode:
                pass
            else:
                return result

        # ── L4_smart_money: Smart Money Z-score + [v3.1] 매집/다이버전스 ──
        sm_min = self.smart_money_cfg.get("min_smart_z", 0.0)
        sm_passed, sm_reason = check_smart_money_gate(row, min_smart_z=sm_min)

        # v3.1 매집 단계 감지
        try:
            accum_signal = self.accum_detector.detect(df, idx)
            result["accum_phase"] = accum_signal.phase
        except Exception:
            accum_signal = None

        # v3.1 OBV 다이버전스 감지
        try:
            div_signal = self.div_scanner.scan(df)
            result["divergence_type"] = div_signal.type
        except Exception:
            div_signal = None

        # 매집/다이버전스 보너스로 SmartZ 게이트 조정
        if accum_signal and accum_signal.phase == "dumping":
            sm_passed = False
            sm_reason = "smart_money_dumping"
        elif not sm_passed and accum_signal and accum_signal.score_modifier >= 10:
            # 매집 Phase2+ 이면 SmartZ 낮아도 통과 허용
            sm_passed = True
            sm_reason = ""

        diag.add_layer(LayerResult(
            name="L4_smart_money",
            passed=sm_passed,
            block_reason=sm_reason,
            details={
                "accum_phase": accum_signal.phase if accum_signal else "none",
                "accum_score": accum_signal.score_modifier if accum_signal else 0,
                "divergence": div_signal.type if div_signal else "none",
            },
        ))
        accum_score_val = accum_signal.score_modifier if accum_signal else 0
        consensus_votes.append(LayerVote(
            "L4_smart_money", sm_passed,
            min(accum_score_val / 100.0, 1.0),
        ))
        if not sm_passed:
            if self.consensus_mode:
                pass
            else:
                return result

        # ── v4.5 L4.5_probability: Rolling 확률 게이트 ──
        if self.prob_gate_enabled and self.prob_gate:
            prob_passed, prob_value = self.prob_gate.check_gate(df, idx)
            diag.add_layer(LayerResult(
                name="L4.5_probability",
                passed=prob_passed,
                block_reason=f"low_prob({prob_value:.0%})" if not prob_passed else "",
                details={"probability": prob_value, "threshold": self.prob_gate.threshold},
            ))
            consensus_votes.append(LayerVote(
                "L4.5_probability", prob_passed, prob_value,
            ))
            result["prob_success"] = prob_value
            if not prob_passed:
                if self.consensus_mode:
                    pass
                else:
                    return result

        # ── L6_trigger: Impulse/Confirm 판정 ──
        impulse = self.check_impulse_trigger(df, idx)
        confirm = self.check_confirm_trigger(df, idx)

        active_trigger = None
        if impulse.trigger_type != TriggerType.NONE:
            active_trigger = impulse
        elif confirm.trigger_type != TriggerType.NONE:
            active_trigger = confirm

        trigger_passed = active_trigger is not None
        diag.add_layer(LayerResult(
            name="L6_trigger",
            passed=trigger_passed,
            block_reason="no_trigger" if not trigger_passed else "",
        ))
        consensus_votes.append(LayerVote(
            "L6_trigger", trigger_passed,
            active_trigger.confidence if active_trigger else 0.0,
        ))
        if not trigger_passed:
            if self.consensus_mode:
                pass
            else:
                # ── v7.0 SETUP 시그널: 게이트 통과 + 트리거 대기 + TGCI 확인 ──
                tgci_cfg = self.config.get("tgci", {})
                tgci_result = TGCIScorer.score(row, config=tgci_cfg)
                setup_min = tgci_cfg.get("setup_min_score", 55)

                if tgci_result["score"] >= setup_min:
                    active_trigger = TriggerResult(
                        trigger_type=TriggerType.SETUP,
                        conditions_met={"tgci_score": tgci_result["score"]},
                        conditions_count=1,
                        stop_loss_pct=0.07,
                        entry_stage_pct=0.20,
                        confidence=tgci_result["score"] / 100.0,
                    )
                    result["tgci_score"] = tgci_result["score"]
                    result["tgci_grade"] = tgci_result["grade"]
                    result["tgci_details"] = tgci_result["details"]
                    logger.info(
                        "[SETUP] %s TGCI=%d(%s) — 공격적 진입 후보",
                        ticker, tgci_result["score"], tgci_result["grade"],
                    )
                else:
                    result["trigger_type"] = "waiting"
                    result["tgci_score"] = tgci_result["score"]
                    return result

        # ── L5_risk: 손익비 체크 ──
        atr_val = row.get("atr_14", 0)
        close = row["close"]
        stop_pct = active_trigger.stop_loss_pct

        if active_trigger.trigger_type == TriggerType.SETUP:
            # SETUP: ATR 기반 스톱 (넓게)
            atr_stop = close - atr_val * 2.0
            stop_price = max(atr_stop, close * (1 - stop_pct))
        elif active_trigger.trigger_type == TriggerType.IMPULSE:
            swing_low = self._calc_swing_low(df, idx, lookback=10)
            pct_stop = close * (1 - stop_pct)
            stop_price = max(swing_low * 0.995, pct_stop)
        else:
            pct_stop = close * (1 - stop_pct)
            atr_stop = close - atr_val * self.strategy["atr_stop_multiplier"]
            stop_price = min(pct_stop, atr_stop)

        target_price = close + atr_val * self.strategy["atr_target_multiplier"]

        risk = close - stop_price
        reward = target_price - close
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        # v3.1: 뉴스 A급이면 rr_min 완화
        default_min_rr = 1.5 if active_trigger.trigger_type == TriggerType.IMPULSE else 2.0
        min_rr = param_overrides.get("rr_min", default_min_rr)
        risk_passed = rr_ratio >= min_rr

        diag.add_layer(LayerResult(
            name="L5_risk",
            passed=risk_passed,
            block_reason=f"low_rr({rr_ratio:.1f})" if not risk_passed else "",
        ))
        consensus_votes.append(LayerVote(
            "L5_risk", risk_passed,
            min(rr_ratio / 10.0, 1.0) if rr_ratio > 0 else 0.0,
        ))
        if not risk_passed:
            if self.consensus_mode:
                pass
            else:
                return result

        # ── [v4.6] L7_geometric: 하모닉/엘리어트/각도 보조 확인 ──
        final_confidence = active_trigger.confidence
        geo_data = {}

        if self.geo_enabled and self.geo_engine:
            try:
                geo_result = self.geo_engine.generate_l7_result(df, ticker)
                geo_data = geo_result

                if geo_result["geo_confirms_buy"]:
                    # 기하학적 분석이 매수 확인 → confidence 부스트
                    final_confidence = min(
                        final_confidence + self.geo_confidence_boost, 1.0
                    )
                    diag.add_layer(LayerResult(
                        name="L7_geometric",
                        passed=True,
                        details={
                            "action": geo_result["geo_action"],
                            "confidence": geo_result["geo_confidence"],
                            "harmonic": geo_result["geo_harmonic"].get("pattern") if geo_result["geo_harmonic"] else None,
                            "elliott": geo_result["geo_elliott"].get("current_wave") if geo_result["geo_elliott"] else None,
                            "boost": self.geo_confidence_boost,
                        },
                    ))
                elif geo_result["geo_warns_sell"]:
                    # 기하학적 분석이 매도 경고 → confidence 감소
                    final_confidence = max(
                        final_confidence - self.geo_confidence_penalty, 0.0
                    )
                    diag.add_layer(LayerResult(
                        name="L7_geometric",
                        passed=True,
                        details={
                            "action": geo_result["geo_action"],
                            "confidence": geo_result["geo_confidence"],
                            "penalty": self.geo_confidence_penalty,
                            "warning": "sell_signal_detected",
                        },
                    ))
                else:
                    diag.add_layer(LayerResult(
                        name="L7_geometric",
                        passed=True,
                        details={
                            "action": geo_result["geo_action"],
                            "confidence": geo_result["geo_confidence"],
                            "effect": "neutral",
                        },
                    ))
            except Exception as e:
                logger.warning("L7 geometric failed for %s: %s", ticker, e)
                diag.add_layer(LayerResult(
                    name="L7_geometric",
                    passed=True,
                    details={"error": str(e)},
                ))

        # v5.0: L7 geometric LayerVote 추가
        if self.geo_enabled and geo_data:
            geo_conf = geo_data.get("geo_confidence", 0)
            geo_buy = geo_data.get("geo_confirms_buy", False)
            consensus_votes.append(LayerVote(
                "L7_geometric",
                geo_buy or (geo_conf >= self.geo_min_confidence),
                geo_conf / 100.0 if geo_conf > 0 else 0.0,
            ))

        # ── v5.0: 합의 판정 (consensus_mode일 때만) ──
        consensus_result = None
        if self.consensus_mode and self.consensus_verifier and consensus_votes:
            geo_indicators = geo_data.get("geo_indicators", {}) if geo_data else {}
            consensus_result = self.consensus_verifier.verify(
                consensus_votes, geo_indicators
            )

        # ── 최종 시그널 생성 ──
        grade_ratios = {
            "A": grades["A"]["position_ratio"],
            "B": grades["B"]["position_ratio"],
            "C": grades["C"]["position_ratio"],
        }
        position_ratio = grade_ratios.get(grade, 0)

        diag.final_signal = True

        result.update({
            "zone_score": zone_score,
            "bes_score": zone_score,
            "trigger_type": active_trigger.trigger_type.value,
            "trigger_confidence": final_confidence,
            "trigger_conditions": active_trigger.conditions_met,
            "signal": True,
            "entry_stage_pct": active_trigger.entry_stage_pct,
            "stop_loss_pct": stop_pct,
            "position_ratio": position_ratio,
            "entry_price": int(close),
            "stop_loss": int(stop_price),
            "target_price": int(target_price),
            "risk_reward_ratio": rr_ratio,
            "atr_value": round(atr_val, 1),
        })

        # v4.7: geometric 데이터 첨부 (10지표 + 프로파일)
        if geo_data:
            result["geo_action"] = geo_data.get("geo_action", "HOLD")
            result["geo_confidence"] = geo_data.get("geo_confidence", 0)
            result["geo_harmonic"] = geo_data.get("geo_harmonic")
            result["geo_elliott"] = geo_data.get("geo_elliott")
            result["geo_profile"] = geo_data.get("geo_profile", "default")
            result["geo_indicators"] = geo_data.get("geo_indicators", {})

        # v5.0: consensus 데이터 첨부
        if consensus_result:
            result["consensus"] = {
                "consistency": consensus_result.consistency,
                "reliability": consensus_result.reliability,
                "diversity": consensus_result.diversity,
                "geometric_reward": consensus_result.geometric_reward,
                "grade": consensus_result.consensus_grade,
                "passed_voters": consensus_result.passed_voters,
                "total_voters": consensus_result.total_voters,
            }

        # ── v7.0: Master Controller 종합 판정 ──
        try:
            # 파이프라인 통과 점수: 통과한 레이어 비율 * 100
            passed_layers = sum(1 for v in consensus_votes if v.passed)
            total_layers = max(len(consensus_votes), 1)
            pipeline_score = passed_layers / total_layers * 100

            # TGCI 점수 (이미 계산했으면 재사용)
            tgci_score = result.get("tgci_score", 0)
            if not tgci_score:
                tgci_cfg = self.config.get("tgci", {})
                tgci_res = TGCIScorer.score(row, config=tgci_cfg)
                tgci_score = tgci_res["score"]
                result["tgci_score"] = tgci_score
                result["tgci_grade"] = tgci_res["grade"]

            # 수급 점수
            smart_z = row.get("smart_z", 0)
            smart_money_score = min(100, max(0, smart_z * 25 + 50))

            # 레짐 점수
            regime_score = 80 if result.get("regime") == "accumulation" else (
                50 if result.get("regime") == "recovery" else 30
            )

            # 기하학 점수
            geometric_score = geo_data.get("geo_confidence", 0) if geo_data else 0

            mc = MasterController(self.config)
            master_result = mc.evaluate({
                "pipeline": pipeline_score,
                "tgci": tgci_score,
                "smart_money": smart_money_score,
                "regime": regime_score,
                "geometric": geometric_score,
            })
            result["master_score"] = master_result["master_score"]
            result["master_action"] = master_result["action"].value
            result["master_entry_mode"] = (
                master_result["entry_mode"].value if master_result["entry_mode"] else None
            )
            result["master_contributing"] = master_result["contributing_systems"]
        except Exception as e:
            logger.warning("Master Controller failed for %s: %s", ticker, e)

        return result

    # 하위호환
    def calculate_bes(self, ticker: str, df: pd.DataFrame, idx: int) -> dict:
        return self.calculate_signal(ticker, df, idx)

    # ══════════════════════════════════════════════
    #  전종목 스캔
    # ══════════════════════════════════════════════

    def scan_universe(self, data_dict: dict, idx: int,
                      held_positions: list | None = None) -> list:
        """전종목 스캔 -> 활성 시그널 (Zone Score 높은 순 + 섹터 제한)"""
        signals = []
        for ticker, df in data_dict.items():
            if idx >= len(df):
                continue
            try:
                result = self.calculate_signal(ticker, df, idx)
                if result["signal"]:
                    signals.append(result)
            except Exception as e:
                logger.debug(f"{ticker} 시그널 실패: {e}")

        signals.sort(key=lambda x: x["zone_score"], reverse=True)

        # v4.5: 섹터 제한 필터 (동일 섹터 최대 1종목)
        if self.sector_classifier.enabled:
            held_sectors = {}
            if held_positions:
                for pos in held_positions:
                    t = pos.ticker if hasattr(pos, "ticker") else pos.get("ticker", "")
                    held_sectors[t] = self.sector_classifier.classify(t)
            signals = self.sector_classifier.filter_by_sector_limit(
                signals, held_sectors
            )

        return signals

    def scan_breakout(self, data_dict: dict, idx: int,
                      held_tickers: set) -> list:
        """보유 종목 중 돌파 트리거 발동 탐색 (3차 매수용)"""
        breakouts = []
        for ticker in held_tickers:
            if ticker not in data_dict:
                continue
            df = data_dict[ticker]
            if idx >= len(df):
                continue
            try:
                bo = self.check_breakout_trigger(df, idx)
                if bo.trigger_type == TriggerType.BREAKOUT:
                    breakouts.append({
                        "ticker": ticker,
                        "trigger": bo,
                        "date": str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.index[idx]),
                    })
            except Exception:
                pass
        return breakouts

    # ══════════════════════════════════════════════
    #  v6.4 포물선 초점 탐지 (Focus Point Detection)
    # ══════════════════════════════════════════════

    def detect_focus_point(self, ticker: str, df: pd.DataFrame, idx: int,
                           lookback: int = 5) -> dict:
        """
        BES 구성 팩터의 기하학적 합의 탐지.
        최근 N일간 ATR/Value/Supply 점수가 동시에 상승 추세이면
        "포물선의 초점" = 에너지 수렴 타점으로 판정.

        Args:
            ticker: 종목코드
            df: 지표 계산된 DataFrame
            idx: 현재 인덱스
            lookback: 추세 분석 기간 (기본 5일)

        Returns:
            Dict with focus_detected, rising_factors, strength, signal
        """
        focus_cfg = self.strategy.get("focus_point", {})
        if not focus_cfg.get("enabled", False):
            return {"ticker": ticker, "focus_detected": False, "rising_factors": 0,
                    "strength": 0, "signal": "disabled"}

        if idx < lookback + 1:
            return {"ticker": ticker, "focus_detected": False, "rising_factors": 0,
                    "strength": 0, "signal": "insufficient_data"}

        # 최근 lookback일간 각 팩터 점수 계산
        atr_scores = []
        supply_scores = []
        for i in range(idx - lookback, idx + 1):
            if i < 0 or i >= len(df):
                continue
            row = df.iloc[i]
            pa = row.get("pullback_atr", np.nan)
            atr_scores.append(self.score_atr_pullback(pa) if not pd.isna(pa) else 0)
            supply_scores.append(self.score_supply_demand(df, i))

        # RSI Z-Score 추세 (Z-Score가 상승하면 과매도에서 회복 중)
        rsi_zscores = []
        for i in range(idx - lookback, idx + 1):
            if i < 0 or i >= len(df):
                continue
            rsi_z = df.iloc[i].get("rsi_zscore", np.nan)
            rsi_zscores.append(float(rsi_z) if not pd.isna(rsi_z) else 0)

        def calc_trend(series):
            """선형 기울기"""
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            return float(np.polyfit(x, series, 1)[0])

        trends = {
            "atr_pullback": calc_trend(atr_scores),
            "supply_demand": calc_trend(supply_scores),
            "rsi_recovery": calc_trend(rsi_zscores),
        }

        rising_count = sum(1 for t in trends.values() if t > 0)
        min_rising = focus_cfg.get("min_rising_factors", 3)
        all_rising = rising_count >= min_rising
        avg_strength = np.mean(list(trends.values()))

        result = {
            "ticker": ticker,
            "focus_detected": all_rising,
            "rising_factors": rising_count,
            "strength": round(avg_strength, 4),
            "trends": {k: round(v, 4) for k, v in trends.items()},
        }

        if all_rising:
            result["signal"] = f"focus_detected(strength={avg_strength:.4f})"
        elif rising_count >= 2:
            result["signal"] = f"focus_near({rising_count}/3)"
        else:
            result["signal"] = f"no_focus({rising_count}/3)"

        return result
