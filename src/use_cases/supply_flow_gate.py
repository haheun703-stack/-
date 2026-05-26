"""4수급 실시간 매수 게이트 — 갭 #2 (5/26 19:30 신규).

배경:
- 5/22 풀세트 잔여 ❌ 4종 중 "4수급" = BAT-D 16:30 일별 갱신만 적용
- 매수 시점에 외인/기관 실시간 매매 동향 검증 X
- 5/14 메모리 V3_FULL_SWING: 3단계 (외인 + 기관 + 금투+연기금+기타) D+1 +2.02% / 적중 59.1%

KIS API: FHKST01010900 (주식현재가 투자자) — kis_intraday_adapter.fetch_investor_flow()
- foreign_net_buy: 외인 당일 누적 (주)
- inst_net_buy: 기관 당일 누적 (주)
- individual_net_buy: 개인 당일 누적 (주)

룰 (백테스트 검증 전 잠정치 — 5/30 회고 후 보정):
- DUAL_BUY: 외인 + 기관 모두 양수 → 매수 우대 (★ 5/14 3단계 패턴)
- NET_BUY: 외인 + 기관 합계 양수 (한쪽 만 양수) → 통과
- DUAL_SELL: 둘 다 음수 + 합계 < -SIG_THRESHOLD → 차단 (대량 동반 매도)
- WEAK_SELL: 한쪽 마이너스 + 다른 쪽 미미 → 통과 (안전 보호)
- DATA_MISSING: API 실패 → fail-open

환경변수:
- GATE_SUPPLY_FLOW_ENABLED (기본 1)
- SUPPLY_FLOW_DUAL_SELL_THRESHOLD (기본 -1000주, 동반 매도 차단 임계)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SUPPLY_FLOW_DUAL_SELL = int(os.getenv("SUPPLY_FLOW_DUAL_SELL_THRESHOLD", "-1000"))


@dataclass
class SupplyFlowGate:
    """4수급 게이트 결과."""
    allow: bool
    reason: str               # 'DUAL_BUY' / 'NET_BUY' / 'DUAL_SELL' / 'WEAK_SELL' / 'DATA_MISSING'
    foreign_net: int          # 외인 당일 누적
    inst_net: int             # 기관 당일 누적
    individual_net: int       # 개인 당일 누적
    total_net: int            # 외인 + 기관 합계
    is_dual_buy: bool         # 외인 + 기관 동시 매수 (★)


def check_supply_flow_gate(
    intraday_adapter,
    ticker: str,
    dual_sell_threshold: int = SUPPLY_FLOW_DUAL_SELL,
) -> SupplyFlowGate:
    """매수 직전 4수급 실시간 게이트.

    Args:
        intraday_adapter: KisIntradayAdapter (fetch_investor_flow 호출)
        ticker: 종목코드
        dual_sell_threshold: 동반 매도 차단 임계 (외인+기관 합계, 기본 -1000주)
    """
    if intraday_adapter is None or not hasattr(intraday_adapter, "fetch_investor_flow"):
        return SupplyFlowGate(
            allow=True, reason="DATA_MISSING",
            foreign_net=0, inst_net=0, individual_net=0,
            total_net=0, is_dual_buy=False,
        )

    try:
        flow = intraday_adapter.fetch_investor_flow(ticker)
    except Exception as e:
        logger.warning("[supply flow gate] %s fetch 실패 — fail-open: %s", ticker, e)
        return SupplyFlowGate(
            allow=True, reason="DATA_MISSING",
            foreign_net=0, inst_net=0, individual_net=0,
            total_net=0, is_dual_buy=False,
        )

    if not flow:
        return SupplyFlowGate(
            allow=True, reason="DATA_MISSING",
            foreign_net=0, inst_net=0, individual_net=0,
            total_net=0, is_dual_buy=False,
        )

    foreign = int(flow.get("foreign_net_buy", 0) or 0)
    inst = int(flow.get("inst_net_buy", 0) or 0)
    indiv = int(flow.get("individual_net_buy", 0) or 0)
    total = foreign + inst

    # 1. DUAL_BUY: 외인 + 기관 모두 양수 ★ 5/14 3단계 패턴
    if foreign > 0 and inst > 0:
        return SupplyFlowGate(
            allow=True, reason="DUAL_BUY",
            foreign_net=foreign, inst_net=inst, individual_net=indiv,
            total_net=total, is_dual_buy=True,
        )

    # 2. DUAL_SELL 차단: 둘 다 음수 + 합계 큰 매도
    if foreign < 0 and inst < 0 and total < dual_sell_threshold:
        return SupplyFlowGate(
            allow=False, reason="DUAL_SELL",
            foreign_net=foreign, inst_net=inst, individual_net=indiv,
            total_net=total, is_dual_buy=False,
        )

    # 3. NET_BUY: 합계 양수 (한쪽만 매수)
    if total > 0:
        return SupplyFlowGate(
            allow=True, reason="NET_BUY",
            foreign_net=foreign, inst_net=inst, individual_net=indiv,
            total_net=total, is_dual_buy=False,
        )

    # 4. WEAK_SELL: 약한 매도 — 통과 (지나친 보수 회피)
    return SupplyFlowGate(
        allow=True, reason="WEAK_SELL",
        foreign_net=foreign, inst_net=inst, individual_net=indiv,
        total_net=total, is_dual_buy=False,
    )
