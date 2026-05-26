"""적응형 진입 게이트 통합 — H4 VWAP + H5 호가 + H6 매물대 + H7 ATR.

배경 (5/26 PDCA flexible-pullback-buy):
- 5/22 풀세트 11종 잔여 ❌ 4종 (VWAP/호가/매물대/ATR) 통합
- adaptive_buy_queue.execute_auto_buy() 직전 호출
- 모든 게이트 통과 시 → buy_limit 실행
- 차단 시 → stage 상태 BLOCKED + 차단 사유 기록

호출 흐름:
  큐 가격 도달 → check_all_entry_gates(ticker, target_price, broker)
                  → H4 VWAP (vwap_monitor.json)
                  → H5 호가 (KIS asking-price, optional)
                  → H6 매물대 (OHLCV 60일)
                  → 모두 allow=True면 buy_limit + H7 ATR로 stop/target 산출
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from src.use_cases.vwap_gate import check_vwap_buy_gate
from src.use_cases.orderbook_gate import check_orderbook_buy_gate
from src.use_cases.supply_zone_gate import (
    calc_supply_zones,
    check_supply_zone_buy_gate,
)
from src.use_cases.atr_dynamic_stop import (
    calc_atr_dynamic_stop,
    fetch_atr_via_kis,
    StopTarget,
)

logger = logging.getLogger(__name__)

# 환경변수 — 게이트별 ON/OFF
GATE_VWAP_ENABLED = os.getenv("GATE_VWAP_ENABLED", "1") == "1"
GATE_ORDERBOOK_ENABLED = os.getenv("GATE_ORDERBOOK_ENABLED", "1") == "1"
GATE_SUPPLY_ZONE_ENABLED = os.getenv("GATE_SUPPLY_ZONE_ENABLED", "1") == "1"
GATE_VOLUME_POWER_ENABLED = os.getenv("GATE_VOLUME_POWER_ENABLED", "1") == "1"
GATE_SUPPLY_FLOW_ENABLED = os.getenv("GATE_SUPPLY_FLOW_ENABLED", "1") == "1"
GATE_OPENING_CALL_ENABLED = os.getenv("GATE_OPENING_CALL_ENABLED", "1") == "1"
USE_ATR_STOPS = os.getenv("ATR_STOP_ENABLED", "0") == "1"

SUPPLY_ZONE_LOOKBACK = int(os.getenv("SUPPLY_ZONE_LOOKBACK_DAYS", "60"))
VOLUME_POWER_MIN = float(os.getenv("ADAPTIVE_VOLUME_POWER_MIN", "100"))  # 5/22 C2 임계 100+


@dataclass
class EntryGateResult:
    """진입 게이트 종합 결과."""
    allow: bool                              # 모두 통과 여부
    block_reason: str                        # 차단 시 첫 차단 게이트 사유
    vwap_dev_pct: float = 0.0
    vwap_reason: str = "DISABLED"
    is_vwap_dip: bool = False
    orderbook_reason: str = "DISABLED"
    is_strong_bid: bool = False
    supply_zone_reason: str = "DISABLED"
    supply_position: str = ""
    is_poc_breakout: bool = False
    volume_power: float = 0.0                # H8 체결강도 (tday_rltv) — 5/27 신규
    volume_power_reason: str = "DISABLED"
    foreign_net: int = 0                     # H9 4수급 외인 당일 누적 — 5/27 신규
    inst_net: int = 0                        # H9 기관 당일 누적
    supply_flow_reason: str = "DISABLED"
    is_dual_buy: bool = False                # 외인+기관 동시 매수 ★
    opening_call_reason: str = "DISABLED"    # 동시호가 강도 게이트 — 5/27 신규
    opening_call_ratio: float = 0.0
    is_strong_open: bool = False
    atr_stop: Optional[StopTarget] = None    # 매수 성공 시 stop/target 산출 결과
    blocked_gates: list[str] = field(default_factory=list)


def check_all_entry_gates(
    ticker: str,
    target_price: int,
    broker,
    intraday_adapter=None,
    regime: str = "NEUTRAL",
) -> EntryGateResult:
    """매수 직전 모든 진입 게이트 통합 검사.

    Args:
        ticker: 종목코드
        target_price: 매수 목표가 (큐의 stage["target_price"])
        broker: KisOrderAdapter (fetch_price/fetch_ohlcv 호출)
        intraday_adapter: KisIntradayAdapter (호가창 — Optional, None이면 H5 skip)
        regime: 시장 레짐 ('BULL' / 'NEUTRAL' / 'BEARISH') — H7 ATR 배수 선택

    Returns:
        EntryGateResult — allow=True이면 buy_limit 실행 가능.
    """
    result = EntryGateResult(allow=True, block_reason="")
    blocked = []

    # ─────────────────────────────────────────
    # H4: VWAP 게이트 (vwap_monitor.json 기반, broker 호출 없음)
    # ─────────────────────────────────────────
    if GATE_VWAP_ENABLED:
        vw = check_vwap_buy_gate(ticker)
        result.vwap_dev_pct = vw.get("vwap_dev_pct", 0.0)
        result.vwap_reason = vw.get("reason", "?")
        result.is_vwap_dip = vw.get("is_dip", False)
        if not vw.get("allow", True):
            blocked.append(f"VWAP:{result.vwap_reason}")

    # ─────────────────────────────────────────
    # H5: 호가 게이트 (KisIntradayAdapter 필요 — 없으면 skip)
    # ─────────────────────────────────────────
    if GATE_ORDERBOOK_ENABLED and intraday_adapter is not None:
        try:
            ob = intraday_adapter.fetch_orderbook(ticker)
            og = check_orderbook_buy_gate(ob, target_price)
            result.orderbook_reason = og.reason
            result.is_strong_bid = og.is_strong_bid
            if not og.allow:
                blocked.append(f"ORDERBOOK:{og.reason}")
        except Exception as e:
            logger.warning("[entry gates] %s 호가 조회 실패: %s — skip", ticker, e)
            result.orderbook_reason = "ERROR_SKIP"

    # ─────────────────────────────────────────
    # H8: 체결강도 (volume_power, tday_rltv) — 5/27 신규 통합
    # 5/22 C2 백테스트: 100+ 임계 D+1 +20.60% / 적중 84.2%
    # ─────────────────────────────────────────
    if GATE_VOLUME_POWER_ENABLED:
        try:
            from src.use_cases.entry_gates import _fetch_volume_power
            # broker는 KisOrderAdapter (mojito wrapping). _fetch_volume_power가
            # broker.access_token/api_key/api_secret 사용 → mojito raw 객체.
            # KisOrderAdapter는 self.broker = mojito 라서 access 가능.
            raw_broker = getattr(broker, "broker", broker)  # adapter면 raw, mojito면 그대로
            vp, source = _fetch_volume_power(raw_broker, ticker)
            result.volume_power = vp
            if vp <= 0:
                result.volume_power_reason = "FETCH_FAILED"
                # 체결강도 fetch 실패 → fail-open (장 초반 데이터 부족 케이스 보호)
                logger.warning("[entry gates] %s 체결강도 fetch 실패 — fail-open", ticker)
            elif vp < VOLUME_POWER_MIN:
                result.volume_power_reason = "WEAK_BUY"
                blocked.append(f"VOLUME_POWER:{vp:.0f}<{VOLUME_POWER_MIN:.0f}")
            else:
                result.volume_power_reason = "STRONG_BUY"
        except Exception as e:
            logger.warning("[entry gates] %s 체결강도 게이트 실패: %s — skip", ticker, e)
            result.volume_power_reason = "ERROR_SKIP"

    # ─────────────────────────────────────────
    # 동시호가 강도 게이트 — 5/27 신규 (08:30~09:00 매수 시도 시만 작동)
    # ─────────────────────────────────────────
    if GATE_OPENING_CALL_ENABLED and intraday_adapter is not None:
        try:
            from src.use_cases.opening_call_gate import check_opening_call_gate
            ob = intraday_adapter.fetch_orderbook(ticker) if hasattr(intraday_adapter, "fetch_orderbook") else None
            oc = check_opening_call_gate(ob)
            result.opening_call_reason = oc.reason
            result.opening_call_ratio = oc.ratio
            result.is_strong_open = oc.is_strong
            if not oc.allow:
                blocked.append(f"OPENING_CALL:{oc.reason}")
        except Exception as e:
            logger.warning("[entry gates] %s 동시호가 게이트 실패: %s — skip", ticker, e)
            result.opening_call_reason = "ERROR_SKIP"

    # ─────────────────────────────────────────
    # H9: 4수급 실시간 게이트 — 5/27 신규
    # 외인 + 기관 당일 누적으로 DUAL_BUY 우대 / DUAL_SELL 차단
    # ─────────────────────────────────────────
    if GATE_SUPPLY_FLOW_ENABLED and intraday_adapter is not None:
        try:
            from src.use_cases.supply_flow_gate import check_supply_flow_gate
            sf = check_supply_flow_gate(intraday_adapter, ticker)
            result.foreign_net = sf.foreign_net
            result.inst_net = sf.inst_net
            result.supply_flow_reason = sf.reason
            result.is_dual_buy = sf.is_dual_buy
            if not sf.allow:
                blocked.append(f"SUPPLY_FLOW:{sf.reason}")
        except Exception as e:
            logger.warning("[entry gates] %s 4수급 게이트 실패: %s — skip", ticker, e)
            result.supply_flow_reason = "ERROR_SKIP"

    # ─────────────────────────────────────────
    # H6: 매물대 게이트 (broker.fetch_ohlcv 60일)
    # ─────────────────────────────────────────
    if GATE_SUPPLY_ZONE_ENABLED:
        try:
            ohlcv = _fetch_ohlcv_safe(broker, ticker, SUPPLY_ZONE_LOOKBACK)
            profile = calc_supply_zones(ohlcv) if ohlcv else None
            sg = check_supply_zone_buy_gate(target_price, profile)
            result.supply_zone_reason = sg.reason
            result.supply_position = sg.position
            result.is_poc_breakout = sg.is_breakout
            if not sg.allow:
                blocked.append(f"SUPPLY:{sg.reason}")
        except Exception as e:
            logger.warning("[entry gates] %s 매물대 계산 실패: %s — skip", ticker, e)
            result.supply_zone_reason = "ERROR_SKIP"

    # ─────────────────────────────────────────
    # H7: ATR 동적 손익절 — 매수 결정에는 영향 X, stop/target 계산용
    # 차단 게이트 통과 후에만 계산 (불필요 KIS 호출 방지)
    # ─────────────────────────────────────────
    if not blocked and USE_ATR_STOPS:
        try:
            atr = fetch_atr_via_kis(broker, ticker)
            result.atr_stop = calc_atr_dynamic_stop(
                entry_price=target_price,
                atr_value=atr,
                regime=regime,
            )
        except Exception as e:
            logger.warning("[entry gates] %s ATR 계산 실패: %s — fallback", ticker, e)
            result.atr_stop = calc_atr_dynamic_stop(
                entry_price=target_price, atr_value=None, regime=regime,
            )

    # 최종 판정
    if blocked:
        result.allow = False
        result.block_reason = blocked[0]
        result.blocked_gates = blocked
        logger.info(
            "[entry gates] %s 차단: %s (target=%d)",
            ticker, ", ".join(blocked), target_price,
        )
    else:
        logger.info(
            "[entry gates] %s 통과 (VWAP=%s, OB=%s, SUPPLY=%s, target=%d)",
            ticker, result.vwap_reason, result.orderbook_reason,
            result.supply_zone_reason, target_price,
        )

    return result


def _fetch_ohlcv_safe(broker, ticker: str, lookback_days: int) -> list[dict]:
    """OHLCV 60일 조회 (정규화 + 안전 처리)."""
    if not hasattr(broker, "fetch_ohlcv"):
        return []
    try:
        from datetime import date, timedelta
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=lookback_days * 2)).strftime("%Y%m%d")  # 여유 2배
        raw = broker.fetch_ohlcv(ticker, timeframe="D",
                                   start_day=start_day, end_day=end_day)
        # mojito 반환 형식: {"output1": {...}, "output2": [{stck_bsop_date, stck_oprc, ...}]}
        out2 = raw.get("output2", []) if isinstance(raw, dict) else raw
        bars = []
        for r in out2[-lookback_days:]:
            try:
                bars.append({
                    "date": r.get("stck_bsop_date", ""),
                    "open": float(r.get("stck_oprc", 0) or 0),
                    "high": float(r.get("stck_hgpr", 0) or 0),
                    "low": float(r.get("stck_lwpr", 0) or 0),
                    "close": float(r.get("stck_clpr", 0) or 0),
                    "volume": int(r.get("acml_vol", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
        return bars
    except Exception as e:
        logger.warning("[entry gates] %s OHLCV 조회 실패: %s", ticker, e)
        return []
