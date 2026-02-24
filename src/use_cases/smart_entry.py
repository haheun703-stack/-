"""
AI 스마트 진입 엔진 — 장중 실시간 판단 + 적응형 지정가 매수

핵심 원칙:
  1. 전일종가보다 싸게 산다 (지정가 우선)
  2. 갭업이라도 무조건 스킵 X → 호가창 + 5분봉 + 수급으로 AI 판단
  3. 적응형 주문 정정 (호가 따라가기)
  4. dry_run=True 시 실제 주문 안 나감 (로그만)

실행 흐름:
  Phase 1 (08:55): 전일종가 기준 지정가 접수
  Phase 2 (09:01): 시가 확인 → 갭업 분류
  Phase 3 (09:05~09:15): 5분봉 3개 형성 후 갭업 종목 AI 판단
  Phase 4 (09:15~10:30): 적응형 주문 관리 (2분 간격)
  Phase 5 (10:30): 미체결 전량 취소 + 결과 보고
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── 상수 ────────────────────────────────────────────

class GapType(Enum):
    GAP_DOWN = "gap_down"       # 시가 < 전일종가
    FLAT = "flat"               # 시가 ≈ 전일종가 (±0.5%)
    SMALL_GAP = "small_gap"     # +0.5% ~ +1.5%
    GAP_UP = "gap_up"           # +1.5% ~ +3.0%
    BIG_GAP = "big_gap"         # +3.0% 이상


class EntryDecision(Enum):
    BUY = "buy"                 # 매수 실행
    WAIT = "wait"               # 눌림 대기
    SKIP = "skip"               # 오늘 패스
    HOLDING = "holding"         # 주문 유지 (체결 대기)


# ─── 데이터 클래스 ────────────────────────────────────

@dataclass
class CandidateState:
    """추천 종목별 진입 상태 추적"""
    ticker: str = ""
    name: str = ""
    grade: str = ""
    prev_close: int = 0         # 전일종가 (진입 기준가)
    stop_loss: int = 0
    target_price: int = 0
    score: float = 0.0

    # 실시간 업데이트
    open_price: int = 0         # 당일 시가
    current_price: int = 0
    gap_pct: float = 0.0        # 갭업률 (%)
    gap_type: GapType = GapType.FLAT

    # AI 판단 결과
    decision: EntryDecision = EntryDecision.WAIT
    decision_reasons: list = field(default_factory=list)

    # 주문 상태
    order_id: str = ""
    order_price: int = 0
    order_qty: int = 0
    is_filled: bool = False
    filled_price: int = 0

    # 호가 분석
    bid_ask_ratio: float = 0.0  # 매수잔량/매도잔량
    orderbook_signal: str = ""  # "strong_buy", "neutral", "sell_pressure"

    # 5분봉 분석
    candle_pattern: str = ""    # "pullback_bounce", "trend_continue", "gap_fail"
    candle_count: int = 0

    # 수급 분석
    foreign_net: int = 0
    inst_net: int = 0
    flow_signal: str = ""       # "both_buy", "foreign_buy", "both_sell"

    # VWAP + 3중 확인 (60 EMA + VWAP + MACD Histogram)
    vwap: float = 0.0                # 당일 누적 VWAP
    vwap_position: str = ""          # "above" / "below"
    sma60: float = 0.0               # 일봉 SMA60 (추세 기준선)
    sma60_position: str = ""         # "above" / "below"
    macd_hist: float = 0.0           # MACD 히스토그램 (최신)
    macd_hist_prev: float = 0.0      # MACD 히스토그램 (전일)
    macd_hist_rising: bool = False   # 히스토그램 증가 중
    triple_confirm: int = 0          # 3중 확인 충족 개수 (0~3)
    triple_detail: str = ""          # "EMA✓ VWAP✓ MACD✓" 등


# ─── SmartEntryEngine ────────────────────────────────

class SmartEntryEngine:
    """AI 스마트 진입 엔진"""

    def __init__(
        self,
        intraday_adapter,       # KisIntradayAdapter
        order_adapter=None,     # KisOrderAdapter (dry_run이면 None 가능)
        dry_run: bool = True,
        config: dict | None = None,
    ):
        self.intraday = intraday_adapter
        self.order = order_adapter
        self.dry_run = dry_run
        self.config = config or {}

        # 설정값
        entry_cfg = self.config.get("smart_entry", {})
        self.initial_discount = entry_cfg.get("initial_discount_pct", 0.5)  # 전일종가 -0.5%
        self.gap_small_thresh = entry_cfg.get("gap_small_pct", 0.5)
        self.gap_medium_thresh = entry_cfg.get("gap_medium_pct", 1.5)
        self.gap_up_thresh = entry_cfg.get("gap_up_pct", 3.0)
        self.min_bid_ask_ratio = entry_cfg.get("min_bid_ask_ratio", 1.3)
        self.max_candle_wait = entry_cfg.get("max_candle_wait_min", 15)
        self.order_deadline_hhmm = entry_cfg.get("order_deadline", 1030)
        self.adapt_interval_sec = entry_cfg.get("adapt_interval_sec", 120)
        self.gap_position_scale = entry_cfg.get("gap_position_scale", 0.5)  # 갭업 시 포지션 50% 축소

        # 종목별 상태
        self.candidates: list[CandidateState] = []
        self.results: list[dict] = []

    # ──────────────────────────────────────────
    # Phase 1: 추천 종목 로드 + 초기 지정가
    # ──────────────────────────────────────────

    def load_picks(self, picks_path: str | Path | None = None) -> int:
        """tomorrow_picks.json 로드 → CandidateState 리스트 생성"""
        path = Path(picks_path) if picks_path else Path("data/tomorrow_picks.json")
        if not path.exists():
            logger.error("[로드] %s 없음", path)
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        picks = data.get("picks", [])
        # 강력매수 + 매수 + 관심매수만
        valid_grades = {"강력매수", "매수", "관심매수"}

        self.candidates = []
        for p in picks:
            if p.get("grade") not in valid_grades:
                continue
            c = CandidateState(
                ticker=p["ticker"],
                name=p["name"],
                grade=p["grade"],
                prev_close=int(p.get("close", 0) or p.get("entry_price", 0)),
                stop_loss=int(p.get("stop_loss", 0)),
                target_price=int(p.get("target_price", 0)),
                score=float(p.get("total_score", 0)),
            )
            self.candidates.append(c)

        logger.info("[로드] %d건 추천 종목 로드 완료", len(self.candidates))
        return len(self.candidates)

    def place_initial_orders(self) -> int:
        """전일종가 -0.5% 지정가 매수 접수 (Phase 1)"""
        placed = 0
        for c in self.candidates:
            if c.prev_close <= 0:
                continue

            # 지정가 = 전일종가 × (1 - discount)
            order_price = self._tick_round(
                int(c.prev_close * (1 - self.initial_discount / 100)),
                c.prev_close,
            )
            c.order_price = order_price
            c.decision = EntryDecision.WAIT

            if self.dry_run:
                logger.info(
                    "[DRY] 초기 지정가: %s(%s) %d원 (전일종가 %d원, -%.1f%%)",
                    c.name, c.ticker, order_price, c.prev_close, self.initial_discount,
                )
                placed += 1
            else:
                # TODO: 실제 주문 시 수량 계산 필요 (PositionSizer 연동)
                order = self.order.buy_limit(c.ticker, order_price, c.order_qty)
                if order.status.value != "failed":
                    c.order_id = order.order_id
                    placed += 1
                    logger.info(
                        "[주문] 초기 지정가: %s %d원 %d주 (주문번호=%s)",
                        c.name, order_price, c.order_qty, order.order_id,
                    )
                else:
                    logger.warning("[주문] 접수 실패: %s — %s", c.name, order.message)

        logger.info("[Phase1] %d/%d건 초기 지정가 접수", placed, len(self.candidates))
        return placed

    # ──────────────────────────────────────────
    # Phase 2: 시가 확인 → 갭 분류
    # ──────────────────────────────────────────

    def check_opening_prices(self) -> dict:
        """09:01 시가 확인 → 갭업 분류"""
        stats = {"gap_down": 0, "flat": 0, "small_gap": 0, "gap_up": 0, "big_gap": 0}

        for c in self.candidates:
            if c.prev_close <= 0:
                continue

            tick = self.intraday.fetch_tick(c.ticker)
            c.open_price = tick.get("open_price", 0) or tick.get("current_price", 0)
            c.current_price = tick.get("current_price", 0)

            if c.open_price <= 0:
                continue

            c.gap_pct = round((c.open_price / c.prev_close - 1) * 100, 2)

            # 갭 분류
            if c.gap_pct < 0:
                c.gap_type = GapType.GAP_DOWN
            elif c.gap_pct < self.gap_small_thresh:
                c.gap_type = GapType.FLAT
            elif c.gap_pct < self.gap_medium_thresh:
                c.gap_type = GapType.SMALL_GAP
            elif c.gap_pct < self.gap_up_thresh:
                c.gap_type = GapType.GAP_UP
            else:
                c.gap_type = GapType.BIG_GAP

            stats[c.gap_type.value] = stats.get(c.gap_type.value, 0) + 1

            logger.info(
                "[시가] %s(%s): 전일 %d → 시가 %d (%+.1f%%) [%s]",
                c.name, c.ticker, c.prev_close, c.open_price,
                c.gap_pct, c.gap_type.value,
            )

        return stats

    # ──────────────────────────────────────────
    # Phase 3: 5분봉 + 호가 + 수급 분석 → AI 판단
    # ──────────────────────────────────────────

    def analyze_and_decide(self) -> dict:
        """
        갭업 종목에 대해 5분봉 + 호가 + 수급 종합 분석.
        갭다운/플랫 종목은 기존 지정가 유지.
        """
        decisions = {"buy": 0, "wait": 0, "skip": 0}

        # VWAP 3중 확인 활성화 여부
        vwap_enabled = self.config.get("smart_entry", {}).get("vwap_enabled", True)

        for c in self.candidates:
            # ── 모든 종목에 대해 일봉 지표 + VWAP 사전 로드 ──
            if vwap_enabled:
                self._load_daily_indicators(c)
                self._calc_cumulative_vwap(c)

            # 갭다운/플랫: 초기 지정가 유지 (좋은 진입 기회)
            if c.gap_type in (GapType.GAP_DOWN, GapType.FLAT):
                c.decision = EntryDecision.HOLDING
                c.decision_reasons.append(f"갭{c.gap_pct:+.1f}% → 초기 지정가 유지")

                # 갭다운/플랫에서도 3중 확인 정보 기록 (참고용)
                if vwap_enabled and c.vwap > 0:
                    self._calc_triple_confirmation(c)
                    c.decision_reasons.append(
                        f"[3중확인] {c.triple_detail} ({c.triple_confirm}/3)"
                    )

                decisions["wait"] += 1
                continue

            # 소갭업/갭업/빅갭업: AI 분석
            ob_score = self._analyze_orderbook(c)
            candle_score = self._analyze_5min_candles(c)
            flow_score = self._analyze_investor_flow(c)

            base_total = ob_score + candle_score + flow_score

            # ── VWAP 3중 확인 보너스 ──
            vwap_bonus = 0
            if vwap_enabled and c.vwap > 0:
                vwap_bonus = self._calc_triple_confirmation(c)

            total = max(0, min(30, base_total + vwap_bonus))

            if vwap_bonus != 0:
                c.decision_reasons.append(
                    f"갭 +{c.gap_pct:.1f}% → 호가({ob_score}) + 캔들({candle_score}) "
                    f"+ 수급({flow_score}) = {base_total} "
                    f"+ 3중확인({vwap_bonus:+d}) = {total}"
                )
                c.decision_reasons.append(
                    f"[3중확인] {c.triple_detail} | "
                    f"VWAP={c.vwap:,.0f} SMA60={c.sma60:,.0f} "
                    f"MACD={c.macd_hist:+.2f}"
                )
            else:
                c.decision_reasons.append(
                    f"갭 +{c.gap_pct:.1f}% → 호가({ob_score}) + 캔들({candle_score}) + 수급({flow_score}) = {total}"
                )

            # 판단 기준: 3축 합산 + VWAP 보너스 (총 0~30)
            if total >= 18:
                c.decision = EntryDecision.BUY
                c.decision_reasons.append("→ 진입 결정 (강한 신호)")
                decisions["buy"] += 1
            elif total >= 12:
                c.decision = EntryDecision.WAIT
                c.decision_reasons.append("→ 눌림 대기 (보통 신호)")
                decisions["wait"] += 1
            else:
                c.decision = EntryDecision.SKIP
                c.decision_reasons.append("→ 오늘 스킵 (약한 신호)")
                decisions["skip"] += 1

            logger.info(
                "[판단] %s: %s (점수 %d/30, 3중확인 %d/3) — %s",
                c.name, c.decision.value, total, c.triple_confirm,
                " | ".join(c.decision_reasons[-2:]),
            )

        return decisions

    def _analyze_orderbook(self, c: CandidateState) -> int:
        """호가창 분석 → 0~10점"""
        try:
            ob = self.intraday.fetch_orderbook(c.ticker)
        except Exception as e:
            logger.warning("[호가] %s 조회 실패: %s", c.ticker, e)
            return 5  # 중립

        ratio = ob.get("bid_ask_ratio", 0)
        c.bid_ask_ratio = ratio
        asks = ob.get("asks", [])

        # 매도벽 감지: 상위 3호가에 대량 매물 (평균의 3배 이상)
        has_wall = False
        if asks:
            avg_ask_vol = sum(a["volume"] for a in asks) / len(asks) if asks else 0
            for a in asks[:3]:
                if avg_ask_vol > 0 and a["volume"] > avg_ask_vol * 3:
                    has_wall = True
                    c.decision_reasons.append(
                        f"⚠ 매도벽 {a['price']:,}원 ({a['volume']:,}주)"
                    )
                    break

        # 점수 산정
        score = 5  # 기본 중립
        if ratio >= 2.0:
            score = 10
            c.orderbook_signal = "strong_buy"
        elif ratio >= 1.5:
            score = 8
            c.orderbook_signal = "strong_buy"
        elif ratio >= 1.0:
            score = 6
            c.orderbook_signal = "neutral"
        elif ratio >= 0.7:
            score = 4
            c.orderbook_signal = "neutral"
        else:
            score = 2
            c.orderbook_signal = "sell_pressure"

        if has_wall:
            score = max(score - 2, 0)

        logger.info(
            "[호가] %s: 매수/매도잔량 비율 %.2f, 벽=%s → %d점",
            c.ticker, ratio, has_wall, score,
        )
        return score

    def _analyze_5min_candles(self, c: CandidateState) -> int:
        """5분봉 패턴 분석 → 0~10점"""
        try:
            candles = self.intraday.fetch_minute_candles(c.ticker, period=5)
        except Exception as e:
            logger.warning("[5분봉] %s 조회 실패: %s", c.ticker, e)
            return 5  # 중립

        # 09:05 이후 형성된 봉만 필터
        recent = [
            cd for cd in candles
            if cd.get("timestamp", "") >= datetime.now().strftime("%Y-%m-%d 09:0")
        ]
        c.candle_count = len(recent)

        if len(recent) < 2:
            c.candle_pattern = "insufficient"
            return 5  # 데이터 부족 → 중립

        score = 5  # 기본 중립

        # 패턴 분석
        first = recent[0]
        last = recent[-1]

        first_body = first.get("close", 0) - first.get("open", 0)
        last_body = last.get("close", 0) - last.get("open", 0)

        # 거래량 추세
        volumes = [cd.get("volume", 0) for cd in recent]
        vol_increasing = len(volumes) >= 2 and volumes[-1] > volumes[0]

        # 패턴 1: 눌림 후 반등 (1봉 음봉 → 2봉 이후 양봉, 저점 상승)
        if first_body < 0 and last_body > 0:
            lows = [cd.get("low", 0) for cd in recent]
            if len(lows) >= 2 and lows[-1] >= lows[0]:
                c.candle_pattern = "pullback_bounce"
                score = 9
                c.decision_reasons.append("5분봉: 눌림 후 반등 (저점 상승)")

        # 패턴 2: 연속 양봉 + 거래량 증가 → 추세 지속
        elif first_body > 0 and last_body > 0 and vol_increasing:
            c.candle_pattern = "trend_continue"
            score = 7
            c.decision_reasons.append("5분봉: 양봉 지속 + 거래량 증가")

        # 패턴 3: 연속 음봉 → 갭업 실패
        elif all(
            cd.get("close", 0) < cd.get("open", 0)
            for cd in recent[-2:]
        ):
            c.candle_pattern = "gap_fail"
            score = 2
            c.decision_reasons.append("5분봉: 연속 음봉 → 갭업 실패")

        # 패턴 4: VWAP 이하로 하락
        elif c.current_price > 0 and c.open_price > 0:
            # 간이 VWAP: 거래량가중평균
            total_val = sum(cd.get("close", 0) * cd.get("volume", 1) for cd in recent)
            total_vol = sum(cd.get("volume", 1) for cd in recent)
            vwap = total_val / total_vol if total_vol > 0 else c.open_price
            if c.current_price < vwap:
                c.candle_pattern = "below_vwap"
                score = 6  # VWAP 아래 = 저가 매수 기회
                c.decision_reasons.append(f"5분봉: VWAP({vwap:,.0f}) 아래 → 매수 기회")
            else:
                c.candle_pattern = "above_vwap"
                score = 4
                c.decision_reasons.append(f"5분봉: VWAP({vwap:,.0f}) 위 → 추격 주의")
        else:
            c.candle_pattern = "mixed"

        logger.info(
            "[5분봉] %s: %d개 봉, 패턴=%s → %d점",
            c.ticker, c.candle_count, c.candle_pattern, score,
        )
        return score

    def _analyze_investor_flow(self, c: CandidateState) -> int:
        """투자자별 수급 분석 → 0~10점"""
        try:
            flow = self.intraday.fetch_investor_flow(c.ticker)
        except Exception as e:
            logger.warning("[수급] %s 조회 실패: %s", c.ticker, e)
            return 5  # 중립

        foreign = flow.get("foreign_net_buy", 0)
        inst = flow.get("inst_net_buy", 0)
        c.foreign_net = foreign
        c.inst_net = inst

        score = 5  # 기본 중립

        if foreign > 0 and inst > 0:
            c.flow_signal = "both_buy"
            score = 10
            c.decision_reasons.append(f"수급: 외인({foreign:+,}) + 기관({inst:+,}) 동시매수")
        elif foreign > 0:
            c.flow_signal = "foreign_buy"
            score = 7
            c.decision_reasons.append(f"수급: 외인 순매수({foreign:+,})")
        elif inst > 0:
            c.flow_signal = "inst_buy"
            score = 7
            c.decision_reasons.append(f"수급: 기관 순매수({inst:+,})")
        elif foreign < 0 and inst < 0:
            c.flow_signal = "both_sell"
            score = 1
            c.decision_reasons.append(f"수급: 외인+기관 동시매도 → 위험")
        else:
            c.flow_signal = "mixed"
            c.decision_reasons.append("수급: 혼조")

        logger.info(
            "[수급] %s: 외인 %+d, 기관 %+d → %s (%d점)",
            c.ticker, foreign, inst, c.flow_signal, score,
        )
        return score

    # ──────────────────────────────────────────
    # VWAP + 3중 확인 (60 EMA + VWAP + MACD Histogram)
    # ──────────────────────────────────────────

    def _calc_cumulative_vwap(self, c: CandidateState) -> float:
        """
        당일 1분봉 기반 누적 VWAP 계산.
        VWAP = Σ(typical_price × volume) / Σ(volume)
        typical_price = (high + low + close) / 3

        기관 알고리즘의 70%+ 가 VWAP을 실행 기준으로 사용.
        VWAP 위 = 당일 매수자 수익 구간 (매도 압력 약함)
        VWAP 아래 = 당일 매수자 손실 구간 (매수 기회 or 위험)
        """
        try:
            candles = self.intraday.fetch_minute_candles(c.ticker, period=1)
        except Exception as e:
            logger.warning("[VWAP] %s 1분봉 조회 실패: %s", c.ticker, e)
            return 0.0

        if not candles:
            return 0.0

        cum_tp_vol = 0.0
        cum_vol = 0

        for cd in candles:
            h = cd.get("high", 0)
            l = cd.get("low", 0)
            cl = cd.get("close", 0)
            v = cd.get("volume", 0)
            if h > 0 and l > 0 and cl > 0 and v > 0:
                typical = (h + l + cl) / 3.0
                cum_tp_vol += typical * v
                cum_vol += v

        if cum_vol == 0:
            return 0.0

        vwap = cum_tp_vol / cum_vol
        c.vwap = round(vwap, 0)

        # 현재가 vs VWAP 위치 판단
        if c.current_price > 0:
            if c.current_price > vwap:
                c.vwap_position = "above"
            else:
                c.vwap_position = "below"

        logger.info(
            "[VWAP] %s: VWAP=%.0f, 현재가=%d → %s",
            c.ticker, vwap, c.current_price, c.vwap_position,
        )
        return vwap

    def _load_daily_indicators(self, c: CandidateState):
        """
        일봉 parquet에서 SMA60 + MACD 히스토그램 로드.
        60일 이동평균: 중기 추세 기준선 (EMA60 대용)
        MACD 히스토그램: 모멘텀의 '힘' (증가=강세 가속, 감소=약화)
        """
        try:
            import pandas as pd
            parquet_path = Path(f"data/processed/{c.ticker}.parquet")
            if not parquet_path.exists():
                logger.warning("[일봉] %s parquet 없음", c.ticker)
                return

            df = pd.read_parquet(parquet_path)
            if len(df) < 2:
                return

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # SMA60 (추세 기준선)
            c.sma60 = float(latest.get("sma_60", 0) or 0)
            if c.sma60 > 0 and c.current_price > 0:
                c.sma60_position = "above" if c.current_price > c.sma60 else "below"

            # MACD 히스토그램
            c.macd_hist = float(latest.get("macd_histogram", 0) or 0)
            c.macd_hist_prev = float(prev.get("macd_histogram", 0) or 0)
            c.macd_hist_rising = c.macd_hist > c.macd_hist_prev

            logger.info(
                "[일봉] %s: SMA60=%.0f(%s), MACD_Hist=%.2f(%s)",
                c.ticker, c.sma60, c.sma60_position,
                c.macd_hist, "상승" if c.macd_hist_rising else "하락",
            )

        except Exception as e:
            logger.warning("[일봉] %s 지표 로드 실패: %s", c.ticker, e)

    def _calc_triple_confirmation(self, c: CandidateState) -> int:
        """
        3중 확인 시스템 (60 EMA + VWAP + MACD Histogram).

        기관 알고리즘 + 트레이딩 논문 기반:
        - 60 EMA 위: 중기 상승 추세 확인 (방향)
        - VWAP 위: 당일 수급 우위 확인 (돈)
        - MACD Histogram 양수+상승: 모멘텀 가속 확인 (힘)

        3개 모두 충족 = 가짜 신호 극적 감소

        Returns:
            보너스 점수 (-3 ~ +3)
        """
        confirms = 0
        details = []

        # 확인 1: SMA60 위 = 중기 상승 추세
        if c.sma60_position == "above":
            confirms += 1
            details.append("EMA\u2713")
        else:
            details.append("EMA\u2717")

        # 확인 2: VWAP 위 = 당일 매수세 우위
        if c.vwap_position == "above":
            confirms += 1
            details.append("VWAP\u2713")
        else:
            details.append("VWAP\u2717")

        # 확인 3: MACD Histogram 양수 + 상승 = 모멘텀 가속
        if c.macd_hist > 0 and c.macd_hist_rising:
            confirms += 1
            details.append("MACD\u2713")
        elif c.macd_hist > 0:
            # 양수이지만 하락 중 → 0.5점 (반올림 안 함)
            details.append("MACD\u25b3")
        else:
            details.append("MACD\u2717")

        c.triple_confirm = confirms
        c.triple_detail = " ".join(details)

        # 보너스 점수: 3중 확인 충족도에 따라
        if confirms == 3:
            bonus = 3   # 완벽한 3중 확인 → 강한 보너스
        elif confirms == 2:
            bonus = 2   # 2개 확인 → 양호
        elif confirms == 1:
            bonus = 0   # 1개만 → 중립 (보너스 없음)
        else:
            bonus = -2  # 0개 → 역방향 → 페널티

        logger.info(
            "[3중확인] %s: %s (%d/3) → 보너스 %+d점",
            c.ticker, c.triple_detail, confirms, bonus,
        )
        return bonus

    # ──────────────────────────────────────────
    # Phase 4: 적응형 주문 관리
    # ──────────────────────────────────────────

    def update_orders(self) -> int:
        """
        적응형 주문 정정 (2분마다 호출).
        - BUY 판정 + 미체결 → 현재가/호가 추적하여 주문 정정
        - SKIP → 주문 취소
        - WAIT → 눌림 감지 시 새 지정가
        """
        updated = 0

        for c in self.candidates:
            if c.is_filled:
                continue

            # 현재가 갱신
            tick = self.intraday.fetch_tick(c.ticker)
            c.current_price = tick.get("current_price", 0)

            if c.decision == EntryDecision.SKIP:
                self._cancel_order(c)
                continue

            if c.decision in (EntryDecision.BUY, EntryDecision.WAIT, EntryDecision.HOLDING):
                new_price = self._calc_adaptive_price(c)
                if new_price != c.order_price and new_price > 0:
                    self._modify_order(c, new_price)
                    updated += 1

        return updated

    def _calc_adaptive_price(self, c: CandidateState) -> int:
        """적응형 지정가 계산"""
        if c.current_price <= 0 or c.prev_close <= 0:
            return c.order_price

        # 원칙: 전일종가보다 싸게 (또는 같게)
        max_price = c.prev_close

        # 갭다운/플랫: 현재가 -1틱 (전일종가 이하 보장)
        if c.gap_type in (GapType.GAP_DOWN, GapType.FLAT):
            target = min(c.current_price, max_price)
            return self._tick_round(target, c.prev_close)

        # 갭업 + BUY 판정: 5분봉 저점 기준
        if c.gap_type in (GapType.SMALL_GAP, GapType.GAP_UP, GapType.BIG_GAP):
            if c.decision == EntryDecision.BUY:
                # 호가 확인해서 bid1 (최우선 매수가) 부근
                try:
                    ob = self.intraday.fetch_orderbook(c.ticker)
                    bids = ob.get("bids", [])
                    if bids:
                        bid1 = bids[0]["price"]
                        # bid1이 전일종가 이하면 OK, 아니면 전일종가
                        if bid1 <= max_price:
                            return bid1
                        # 갭업이지만 수급 강하면 시가 -1% 수준까지 허용
                        if c.bid_ask_ratio >= self.min_bid_ask_ratio:
                            gap_limit = int(c.prev_close * 1.015)  # 최대 +1.5%
                            return min(bid1, gap_limit)
                except Exception:
                    pass

            # WAIT: 전일종가 이하 고수
            return self._tick_round(max_price, c.prev_close)

        return c.order_price

    def _modify_order(self, c: CandidateState, new_price: int):
        """주문 정정"""
        old = c.order_price
        c.order_price = new_price

        if self.dry_run:
            logger.info(
                "[DRY] 주문 정정: %s %d → %d원 (현재가 %d)",
                c.name, old, new_price, c.current_price,
            )
        else:
            if c.order_id and self.order:
                from src.entities.trading_models import Order
                order_obj = Order(order_id=c.order_id, ticker=c.ticker)
                result = self.order.modify(order_obj, new_price, c.order_qty)
                if result.status.value != "failed":
                    c.order_id = result.order_id or c.order_id
                    logger.info(
                        "[정정] %s: %d → %d원 (주문번호=%s)",
                        c.name, old, new_price, c.order_id,
                    )

    def _cancel_order(self, c: CandidateState):
        """주문 취소"""
        if self.dry_run:
            logger.info("[DRY] 주문 취소: %s (사유: %s)", c.name, c.decision.value)
        else:
            if c.order_id and self.order:
                from src.entities.trading_models import Order
                order_obj = Order(order_id=c.order_id, ticker=c.ticker, quantity=c.order_qty)
                self.order.cancel(order_obj)
                logger.info("[취소] %s (주문번호=%s)", c.name, c.order_id)
        c.order_id = ""

    # ──────────────────────────────────────────
    # Phase 5: 미체결 전량 취소 + 결과 보고
    # ──────────────────────────────────────────

    def cancel_all_unfilled(self) -> int:
        """미체결 전량 취소 (데드라인 도달)"""
        cancelled = 0
        for c in self.candidates:
            if not c.is_filled and c.order_id:
                self._cancel_order(c)
                cancelled += 1
        logger.info("[마감] 미체결 %d건 취소 완료", cancelled)
        return cancelled

    def generate_report(self) -> dict:
        """실행 결과 보고서 생성"""
        filled = [c for c in self.candidates if c.is_filled]
        skipped = [c for c in self.candidates if c.decision == EntryDecision.SKIP]
        unfilled = [c for c in self.candidates if not c.is_filled and c.decision != EntryDecision.SKIP]

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dry_run": self.dry_run,
            "total_candidates": len(self.candidates),
            "filled": len(filled),
            "skipped": len(skipped),
            "unfilled": len(unfilled),
            "details": [],
        }

        for c in self.candidates:
            detail = {
                "ticker": c.ticker,
                "name": c.name,
                "grade": c.grade,
                "prev_close": c.prev_close,
                "open_price": c.open_price,
                "gap_pct": c.gap_pct,
                "gap_type": c.gap_type.value,
                "decision": c.decision.value,
                "reasons": c.decision_reasons,
                "order_price": c.order_price,
                "is_filled": c.is_filled,
                "filled_price": c.filled_price,
                "bid_ask_ratio": c.bid_ask_ratio,
                "candle_pattern": c.candle_pattern,
                "flow_signal": c.flow_signal,
                "foreign_net": c.foreign_net,
                "inst_net": c.inst_net,
                # VWAP 3중 확인
                "vwap": c.vwap,
                "vwap_position": c.vwap_position,
                "sma60": c.sma60,
                "sma60_position": c.sma60_position,
                "macd_hist": c.macd_hist,
                "macd_hist_rising": c.macd_hist_rising,
                "triple_confirm": c.triple_confirm,
                "triple_detail": c.triple_detail,
            }
            report["details"].append(detail)

        return report

    def build_telegram_message(self, report: dict) -> str:
        """텔레그램 발송용 메시지 생성"""
        mode = "DRY-RUN" if report["dry_run"] else "LIVE"
        lines = [
            f"{'='*30}",
            f"AI 스마트 진입 결과 [{mode}]",
            f"{'='*30}",
            f"대상: {report['total_candidates']}종목",
            f"체결: {report['filled']}  |  스킵: {report['skipped']}  |  미체결: {report['unfilled']}",
            "",
        ]

        for d in report["details"]:
            emoji = {"buy": "✅", "skip": "❌", "wait": "⏳", "holding": "📋"}.get(
                d["decision"], "❓"
            )
            gap_str = f"{d['gap_pct']:+.1f}%" if d["gap_pct"] else ""
            lines.append(
                f"{emoji} {d['name']}({d['ticker']})"
                f" 전일{d['prev_close']:,} → 시가{d['open_price']:,}({gap_str})"
            )
            lines.append(f"   [{d['gap_type']}] {d['decision']}")
            if d["order_price"]:
                lines.append(f"   지정가: {d['order_price']:,}원")
            if d.get("bid_ask_ratio"):
                lines.append(f"   호가비: {d['bid_ask_ratio']:.2f} | 패턴: {d['candle_pattern']} | 수급: {d['flow_signal']}")
            if d.get("triple_detail"):
                vwap_str = f"VWAP={d['vwap']:,.0f}" if d.get("vwap") else ""
                lines.append(f"   3중확인: {d['triple_detail']} ({d['triple_confirm']}/3) {vwap_str}")
            for r in d.get("reasons", [])[:3]:
                lines.append(f"   • {r}")
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────
    # 메인 실행 루프
    # ──────────────────────────────────────────

    def run_full_session(self) -> dict:
        """
        전체 세션 실행 (08:55 ~ 10:30).
        스케줄러가 08:50에 시작하면 이 함수가 각 Phase를 시간에 맞춰 실행.
        """
        logger.info("=" * 50)
        logger.info("[SmartEntry] 세션 시작 (dry_run=%s)", self.dry_run)
        logger.info("=" * 50)

        # Phase 1: 종목 로드 + 초기 지정가
        count = self.load_picks()
        if count == 0:
            logger.warning("[SmartEntry] 추천 종목 없음 — 종료")
            return self.generate_report()

        self.place_initial_orders()

        # 09:01까지 대기
        self._wait_until(9, 1)

        # Phase 2: 시가 확인
        logger.info("[Phase2] 시가 확인 시작")
        gap_stats = self.check_opening_prices()
        logger.info("[Phase2] 갭 분포: %s", gap_stats)

        # 09:10까지 대기 (5분봉 2~3개 형성)
        self._wait_until(9, 10)

        # Phase 3: AI 판단
        logger.info("[Phase3] AI 분석 + 진입 판단 시작")
        decisions = self.analyze_and_decide()
        logger.info("[Phase3] 판단 결과: %s", decisions)

        # Phase 3.5: BUY 판정 종목 주문 갱신
        for c in self.candidates:
            if c.decision == EntryDecision.BUY and c.gap_type != GapType.GAP_DOWN:
                new_price = self._calc_adaptive_price(c)
                if new_price != c.order_price:
                    self._modify_order(c, new_price)

        # Phase 4: 적응형 주문 관리 루프 (10:30까지)
        logger.info("[Phase4] 적응형 주문 관리 시작 (마감 %02d:%02d)",
                    self.order_deadline_hhmm // 100, self.order_deadline_hhmm % 100)

        while True:
            now = datetime.now()
            hhmm = now.hour * 100 + now.minute
            if hhmm >= self.order_deadline_hhmm:
                break

            self.update_orders()
            self._check_fills()

            # 모든 종목 체결 완료 시 조기 종료
            if all(c.is_filled or c.decision == EntryDecision.SKIP for c in self.candidates):
                logger.info("[Phase4] 전 종목 체결/스킵 완료 → 조기 종료")
                break

            time.sleep(self.adapt_interval_sec)

        # Phase 5: 마감
        logger.info("[Phase5] 마감 처리")
        self.cancel_all_unfilled()
        report = self.generate_report()

        # 결과 저장
        report_path = Path("data/smart_entry_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("[저장] %s", report_path)

        # 텔레그램 발송
        try:
            from src.telegram_sender import send_message
            msg = self.build_telegram_message(report)
            send_message(msg)
            logger.info("[텔레그램] 결과 발송 완료")
        except Exception as e:
            logger.warning("[텔레그램] 발송 실패: %s", e)

        logger.info("=" * 50)
        logger.info("[SmartEntry] 세션 종료")
        logger.info("=" * 50)

        return report

    def run_analysis_only(self) -> dict:
        """
        분석만 수행 (주문 없음). 장중 아닐 때 테스트용.
        종목 로드 → 현재가 조회 → 호가/수급/캔들 분석 → 판단 → 보고서
        """
        logger.info("[AnalysisOnly] 분석 전용 모드 시작")

        count = self.load_picks()
        if count == 0:
            return self.generate_report()

        # 현재가 확인 (시가 대용)
        for c in self.candidates:
            tick = self.intraday.fetch_tick(c.ticker)
            c.open_price = tick.get("open_price", 0) or tick.get("current_price", 0)
            c.current_price = tick.get("current_price", 0)
            if c.prev_close > 0 and c.open_price > 0:
                c.gap_pct = round((c.open_price / c.prev_close - 1) * 100, 2)
            # 갭 분류
            if c.gap_pct < 0:
                c.gap_type = GapType.GAP_DOWN
            elif c.gap_pct < self.gap_small_thresh:
                c.gap_type = GapType.FLAT
            elif c.gap_pct < self.gap_medium_thresh:
                c.gap_type = GapType.SMALL_GAP
            elif c.gap_pct < self.gap_up_thresh:
                c.gap_type = GapType.GAP_UP
            else:
                c.gap_type = GapType.BIG_GAP

            # 초기 지정가 계산
            c.order_price = self._tick_round(
                int(c.prev_close * (1 - self.initial_discount / 100)),
                c.prev_close,
            )

        # AI 분석
        decisions = self.analyze_and_decide()
        report = self.generate_report()

        # 결과 저장
        report_path = Path("data/smart_entry_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _check_fills(self):
        """체결 여부 확인 (실제 모드)"""
        if self.dry_run:
            return
        if not self.order:
            return
        for c in self.candidates:
            if c.is_filled or not c.order_id:
                continue
            status = self.order.get_order_status(c.order_id)
            if status.status.value == "filled":
                c.is_filled = True
                c.filled_price = int(status.filled_price or c.order_price)
                logger.info(
                    "[체결] %s %d원 %d주", c.name, c.filled_price, c.order_qty,
                )

    def _wait_until(self, hour: int, minute: int):
        """특정 시각까지 대기 (dry_run이면 즉시 리턴)"""
        if self.dry_run:
            logger.info("[DRY] %02d:%02d 대기 스킵", hour, minute)
            return

        target = hour * 100 + minute
        while True:
            now = datetime.now()
            current = now.hour * 100 + now.minute
            if current >= target:
                break
            remain = (hour * 60 + minute) - (now.hour * 60 + now.minute)
            logger.info("[대기] %02d:%02d까지 %d분 남음", hour, minute, remain)
            time.sleep(min(remain * 60, 30))

    @staticmethod
    def _tick_round(price: int, reference: int) -> int:
        """호가 단위 맞춤 (KRX 규칙)"""
        if reference < 2000:
            tick = 1
        elif reference < 5000:
            tick = 5
        elif reference < 20000:
            tick = 10
        elif reference < 50000:
            tick = 50
        elif reference < 200000:
            tick = 100
        elif reference < 500000:
            tick = 500
        else:
            tick = 1000
        return (price // tick) * tick
