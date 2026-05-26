"""갭 하락 + 변동성 폭증 가드 — P0-3 (5/26 23:45, ChatGPT 외부 검증 발견).

배경 (ChatGPT 외부 검증 5/26 23:30):
- 누락된 위험 시나리오 — 갭 하락 / 변동성 폭증 / 외인 대량 매도 대응 X
- Claude Code가 못 본 부분 = 시장 위급 상황 대응
- 5/27 가동 전 P0 즉시 추가

룰 (3가지 가드):

1. **GAP_DOWN_EMERGENCY**: 시초가 갭 하락 -5%↑ → 보유 종목 즉시 매도
   - 평단 대비 -5% 이미 도달 또는 시초가 가격 자체가 평단 -5% 미만
   - MVP-2.6 (-5% 손절)보다 빠른 발화 (09:00 시초가 바로)

2. **VOLATILITY_SURGE**: KOSPI 5분 변동성 +3% 이상 → 신규 매수 정지
   - 변동성 = (KOSPI 5분봉 high - low) / open × 100
   - 평소 0.5% 미만, 3%+는 패닉/급변
   - 신규 매수만 정지 (보유 자동 매도는 정상)

3. **FOREIGN_DUMP**: 외인 당일 누적 -10,000주↑ 또는 -10억원↑ → 신규 매수 정지
   - 외인 대량 매도 = 시장 충격 신호
   - 새로 매수 X (보유는 trailing 등 정상)

활용:
- run_adaptive_cycle.py 전체 cron 진입점에서 첫 체크
- 발화 시 MVP-2/5/6 매수 모두 차단
- 보유 종목 매도 (MVP-1/2.5/2.6/2.7/2.8)는 정상 작동
- 텔레그램 긴급 알림

환경변수:
- GAP_VOLATILITY_GUARD_ENABLED=1
- GAP_DOWN_EMERGENCY_PCT=-5.0
- VOLATILITY_5MIN_PCT=3.0
- FOREIGN_DUMP_THRESHOLD=-10000  (주)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENABLED = os.getenv("GAP_VOLATILITY_GUARD_ENABLED", "0") == "1"
GAP_DOWN_PCT = float(os.getenv("GAP_DOWN_EMERGENCY_PCT", "-5.0"))
VOLATILITY_5MIN = float(os.getenv("VOLATILITY_5MIN_PCT", "3.0"))
FOREIGN_DUMP = int(os.getenv("FOREIGN_DUMP_THRESHOLD", "-10000"))


@dataclass
class GuardResult:
    """가드 결과."""
    block_new_buy: bool = False        # 신규 매수 차단 여부
    force_sell_held: bool = False      # 보유 종목 즉시 매도 트리거
    reason: str = ""
    gap_down_ticker: str = ""          # 갭 하락 발화 종목 (있을 경우)
    gap_down_pct: float = 0.0
    volatility_5min_pct: float = 0.0
    foreign_net: int = 0


def check_kospi_volatility_surge(intraday_adapter,
                                  threshold_pct: float = VOLATILITY_5MIN) -> tuple[bool, float]:
    """KOSPI 5분 변동성 폭증 검사.

    Returns:
        (surge: bool, volatility_pct: float)
    """
    if intraday_adapter is None or not hasattr(intraday_adapter, "fetch_minute_candles"):
        return False, 0.0
    try:
        # ★ C4 fix (5/27 검수): KOSPI 종합지수는 "0001" 6자리로 종목 시세 API 호출 불가
        # → KODEX 200 (069500) 분봉으로 대체 (이미 외인 흐름도 069500 사용)
        candles = intraday_adapter.fetch_minute_candles("069500", period=5)
        if not candles:
            return False, 0.0
        last = candles[-1]
        open_p = float(last.get("open", 0) or 0)
        high = float(last.get("high", 0) or 0)
        low = float(last.get("low", 0) or 0)
        if open_p <= 0:
            return False, 0.0
        vol_pct = (high - low) / open_p * 100
        return vol_pct >= threshold_pct, vol_pct
    except Exception as e:
        logger.warning("[갭/변동성 가드] KOSPI 변동성 fetch 실패: %s", e)
        return False, 0.0


def check_foreign_dump(intraday_adapter,
                        threshold: int = FOREIGN_DUMP) -> tuple[bool, int]:
    """KOSPI 200 또는 코스피 인덱스 외인 당일 누적 매도 검사.

    실제로는 종목별 외인 매수/매도 합계가 더 정확하나 일단 KOSPI 인덱스 ETF로 대체.

    Returns:
        (dump: bool, foreign_net: int)
    """
    if intraday_adapter is None or not hasattr(intraday_adapter, "fetch_investor_flow"):
        return False, 0
    try:
        # KODEX 200 (069500) 외인 흐름 (시장 대표)
        flow = intraday_adapter.fetch_investor_flow("069500")
        foreign = int(flow.get("foreign_net_buy", 0) or 0) if flow else 0
        return foreign <= threshold, foreign
    except Exception as e:
        logger.warning("[갭/변동성 가드] KODEX 외인 흐름 fetch 실패: %s", e)
        return False, 0


def check_gap_down_held(broker, held_positions: list[dict],
                          gap_down_pct: float = GAP_DOWN_PCT) -> tuple[bool, str, float]:
    """보유 종목 중 시초가 갭 하락 -5%↑ 종목 검사.

    Returns:
        (has_gap_down: bool, ticker: str, gap_pct: float) — 가장 큰 갭 하락 종목
    """
    if not held_positions:
        return False, "", 0.0

    worst_ticker = ""
    worst_pct = 0.0
    for h in held_positions:
        ticker = str(h.get("ticker", "")).zfill(6)
        avg_price = float(h.get("avg_price", 0) or 0)
        if not ticker or avg_price <= 0:
            continue

        try:
            res = broker.fetch_price(ticker)
            out = res.get("output", {}) if res else {}
            open_p = int(out.get("stck_oprc", 0) or 0)
            if open_p <= 0:
                continue
            gap_pct = (open_p - avg_price) / avg_price * 100
            if gap_pct <= gap_down_pct and gap_pct < worst_pct:
                worst_ticker = ticker
                worst_pct = gap_pct
        except Exception as e:
            logger.warning("[갭/변동성 가드] %s 시초가 fetch 실패: %s", ticker, e)

    return bool(worst_ticker), worst_ticker, worst_pct


def evaluate_market_guard(
    broker,
    intraday_adapter,
    held_positions: list[dict] | None = None,
) -> GuardResult:
    """종합 시장 가드 평가 (cron 사이클 시작 시 호출).

    Args:
        broker: KisOrderAdapter
        intraday_adapter: KisIntradayAdapter
        held_positions: 보유 종목 list (broker.fetch_balance().get("holdings"))

    Returns:
        GuardResult — block_new_buy / force_sell_held / reason
    """
    result = GuardResult()
    if not ENABLED:
        return result

    held_positions = held_positions or []
    reasons = []

    # 1. 갭 하락 (보유 종목 보호 — 즉시 매도)
    has_gap, gap_ticker, gap_pct = check_gap_down_held(broker, held_positions)
    if has_gap:
        result.force_sell_held = True
        result.gap_down_ticker = gap_ticker
        result.gap_down_pct = gap_pct
        reasons.append(f"갭 하락 {gap_ticker} {gap_pct:+.2f}%")

    # 2. KOSPI 변동성 폭증 (신규 매수 정지)
    surge, vol_pct = check_kospi_volatility_surge(intraday_adapter)
    result.volatility_5min_pct = vol_pct
    if surge:
        result.block_new_buy = True
        reasons.append(f"KOSPI 5분 변동성 {vol_pct:.2f}% ≥ {VOLATILITY_5MIN}%")

    # 3. 외인 대량 매도 (신규 매수 정지)
    dump, foreign = check_foreign_dump(intraday_adapter)
    result.foreign_net = foreign
    if dump:
        result.block_new_buy = True
        reasons.append(f"외인 대량 매도 {foreign:,}주 ≤ {FOREIGN_DUMP:,}주")

    result.reason = " / ".join(reasons) if reasons else "정상"
    if reasons:
        logger.warning("[갭/변동성 가드] 발화: %s", result.reason)
    return result


def format_guard_for_telegram(result: GuardResult) -> str:
    """텔레그램 긴급 알림."""
    if not (result.block_new_buy or result.force_sell_held):
        return ""

    lines = ["🚨 [시장 긴급 가드 발화]"]
    if result.force_sell_held:
        lines.append(
            f"  ⚠️ 갭 하락 보유 종목: {result.gap_down_ticker} {result.gap_down_pct:+.2f}%"
        )
        lines.append("  → 즉시 매도 권장 (자동)")
    if result.block_new_buy:
        lines.append(f"  🛑 신규 매수 정지: {result.reason}")
        lines.append("  → MVP-2/5/6 매수 자동 차단")
    lines.append("")
    lines.append(f"  매크로 데이터:")
    lines.append(f"    KOSPI 5분 변동성: {result.volatility_5min_pct:.2f}%")
    lines.append(f"    KODEX 200 외인: {result.foreign_net:,}주")
    return "\n".join(lines)
