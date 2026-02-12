"""
Step 5: signal_engine.py — BES v2.1 듀얼 트리거 엔진

🔥 v2.0 → v2.1 핵심 변경:
  기존: Zone(위치) 점수만으로 진입 판단
  변경: Zone 점수 + 2단계 트리거(시동/확인)로 진입 판단

핵심 철학 (v2.1):
  1. 우리는 조정이 끝나는 자리(Zone)를 먼저 찾는다.
  2. 우리는 시동 신호(Impulse)에서 일부 선취한다.
  3. 우리는 확정 신호(Confirm)에서 비중을 늘린다.
  4. 우리는 가짜 시동을 손절로 비용 처리한다.
  5. 우리는 손익비와 승률을 '모드별'로 관리한다.

BES v2.1 = Zone Score (기존과 동일)
  + Trigger-1 (Impulse): 급등 초입 선점
  + Trigger-2 (Confirm): 안전 진입
  + Trigger-3 (Breakout): 전고점 돌파 추가 매수
"""

import logging
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .fundamental import FundamentalEngine
from .screener import Screener

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """트리거 유형"""
    NONE = "none"
    IMPULSE = "impulse"       # 시동 (공격형)
    CONFIRM = "confirm"       # 확인 (보수형)
    BREAKOUT = "breakout"     # 전고점 돌파 (추가 매수)


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
    """BES v2.1 듀얼 트리거 스코어링 + 시그널 생성"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.strategy = self.config["strategy"]
        self.triggers_cfg = self.strategy.get("triggers", {})
        self.fundamental = FundamentalEngine(config_path)
        self.screener = Screener(self.config, self.fundamental)

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
        """수급 종합 점수"""
        row = df.iloc[idx]

        # RSI
        rsi = row.get("rsi_14", np.nan)
        rsi_range = self.strategy["rsi_sweet_spot"]
        if pd.isna(rsi):
            rsi_score = 0.5
        elif rsi_range[0] <= rsi <= rsi_range[1]:
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

    # ══════════════════════════════════════════════
    #  MASTER: Zone + Trigger 통합
    # ══════════════════════════════════════════════

    def calculate_signal(self, ticker: str, df: pd.DataFrame, idx: int) -> dict:
        """
        v2.1 마스터 시그널 계산.

        Phase 1: Gate 체크
        Phase 2: Zone Score (위치 매력도)
        Phase 3: Trigger 판정 (시동/확인/돌파)
        Phase 4: 최종 시그널 생성

        Zone이 매력적이어도 Trigger가 없으면 "대기".
        Trigger가 있어도 Zone이 부족하면 "무시".
        """
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
        }

        # -- Phase 1: Gate --
        gate = self.screener.check_all_gates(ticker, df, idx)
        result["gate_result"] = gate
        if not gate["passed"]:
            return result

        # -- Phase 2: Zone Score --
        zone_info = self.calc_zone_score(ticker, df, idx, gate)
        zone_score = zone_info["zone_score"]
        result["zone_score"] = zone_score
        result["bes_score"] = zone_score  # 하위호환
        result["components"] = zone_info["components"]

        # 등급 판정
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

        if grade == "F":
            return result

        # -- Phase 3: Trigger 판정 --
        impulse = self.check_impulse_trigger(df, idx)
        confirm = self.check_confirm_trigger(df, idx)

        active_trigger = None
        if impulse.trigger_type != TriggerType.NONE:
            active_trigger = impulse
        elif confirm.trigger_type != TriggerType.NONE:
            active_trigger = confirm

        if active_trigger is None:
            result["trigger_type"] = "waiting"
            return result

        # -- Phase 4: 최종 시그널 --
        atr_val = row.get("atr_14", 0)
        close = row["close"]

        stop_pct = active_trigger.stop_loss_pct

        # Impulse: 스윙 저점 vs 퍼센트 중 타이트한 것
        if active_trigger.trigger_type == TriggerType.IMPULSE:
            swing_low = self._calc_swing_low(df, idx, lookback=10)
            pct_stop = close * (1 - stop_pct)
            stop_price = max(swing_low * 0.995, pct_stop)
        else:
            # Confirm: ATR 손절과 비교, 더 여유 있는 것
            pct_stop = close * (1 - stop_pct)
            atr_stop = close - atr_val * self.strategy["atr_stop_multiplier"]
            stop_price = min(pct_stop, atr_stop)

        target_price = close + atr_val * self.strategy["atr_target_multiplier"]

        risk = close - stop_price
        reward = target_price - close
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        grade_ratios = {
            "A": grades["A"]["position_ratio"],
            "B": grades["B"]["position_ratio"],
            "C": grades["C"]["position_ratio"],
        }
        position_ratio = grade_ratios.get(grade, 0)

        result.update({
            "zone_score": zone_score,
            "bes_score": zone_score,
            "trigger_type": active_trigger.trigger_type.value,
            "trigger_confidence": active_trigger.confidence,
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

        return result

    # 하위호환
    def calculate_bes(self, ticker: str, df: pd.DataFrame, idx: int) -> dict:
        return self.calculate_signal(ticker, df, idx)

    # ══════════════════════════════════════════════
    #  전종목 스캔
    # ══════════════════════════════════════════════

    def scan_universe(self, data_dict: dict, idx: int) -> list:
        """전종목 스캔 -> 활성 시그널 (Zone Score 높은 순)"""
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
