"""모멘텀 추격 매수 — MVP-6 (5/26 20:30 신규, 불장 추격 매매).

배경 (사용자 5/26 20:00 자기반성):
- 오늘 5/26 KOSPI 불장 (+10% 종목 62건) BUT 우리 매수 1건만 (적중 1.6%)
- 근본 원인: 큐 시스템 = 눌림목 매수 = 불장에서 발화 X
- 해결: 5분봉 + 거래량 폭발 + 양봉 모멘텀 추격 매수 추가

룰 (백테스트 X, 5/29 회고 시 보정 — 잠정치):
1. 5분봉 등락률 ≥ +3% (양봉)
2. 거래량 ≥ 평균(최근 20봉) × 3배 (거래량 폭발)
3. 종일 등락률 ≤ +20% (이미 과열된 종목 회피)
4. 종일 등락률 ≥ +1% (시초가 +α 강세 종목)
5. 외인 당일 누적 ≥ 0 (4수급 게이트 활용)

액션:
- 1주 매수 (지정가 = 현재가 +0.3% 슬리피지 허용)
- target = 매수가 +5% (단타 익절)
- stop = 매수가 -3% (빠른 손절)
- 만료 = 1거래일 (D+1까지만 보유)

후보 풀:
- intraday_eye 워치리스트 (settings.yaml)
- AI 동조 자동 워치리스트 (ai_chain_watchlist.json)
- sector_fire_map AI 6세부 섹터 35종

환경변수 (.env):
- MOMENTUM_CHASE_ENABLED=1
- MOMENTUM_CHASE_5MIN_PCT=3.0
- MOMENTUM_CHASE_VOL_RATIO=3.0
- MOMENTUM_CHASE_MAX_DAILY_PCT=20.0
- MOMENTUM_CHASE_MIN_DAILY_PCT=1.0
- MOMENTUM_CHASE_ALLOC_AMOUNT=300000  (1주 30만 한도)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENABLED = os.getenv("MOMENTUM_CHASE_ENABLED", "0") == "1"
MIN_5MIN_PCT = float(os.getenv("MOMENTUM_CHASE_5MIN_PCT", "3.0"))
MIN_VOL_RATIO = float(os.getenv("MOMENTUM_CHASE_VOL_RATIO", "3.0"))
MAX_DAILY_PCT = float(os.getenv("MOMENTUM_CHASE_MAX_DAILY_PCT", "20.0"))
MIN_DAILY_PCT = float(os.getenv("MOMENTUM_CHASE_MIN_DAILY_PCT", "1.0"))
ALLOC_AMOUNT = int(os.getenv("MOMENTUM_CHASE_ALLOC_AMOUNT", "300000"))


@dataclass
class MomentumSignal:
    """모멘텀 추격 시그널."""
    ticker: str
    name: str = ""
    fire: bool = False                  # 모멘텀 발화 여부
    five_min_pct: float = 0.0           # 5분봉 등락률
    vol_ratio: float = 0.0              # 거래량 평균 대비
    daily_pct: float = 0.0              # 종일 등락률
    current_price: int = 0
    target_price: int = 0               # 매수 지정가 (현재가 +0.3%)
    stop_price: int = 0                 # 손절 -3%
    profit_target: int = 0              # 익절 +5%
    reason: str = ""


def _adjust_to_tick(price: int) -> int:
    """KRX 호가 단위 보정."""
    if price <= 0:
        return price
    if price < 1_000: tick = 1
    elif price < 5_000: tick = 5
    elif price < 20_000: tick = 10
    elif price < 50_000: tick = 50
    elif price < 200_000: tick = 100
    elif price < 500_000: tick = 500
    else: tick = 1_000
    return (price // tick) * tick


def evaluate_momentum_signal(
    ticker: str,
    five_min_candles: list[dict],
    daily_pct: float,
    current_price: int,
    name: str = "",
    min_5min_pct: float = MIN_5MIN_PCT,
    min_vol_ratio: float = MIN_VOL_RATIO,
    max_daily_pct: float = MAX_DAILY_PCT,
    min_daily_pct: float = MIN_DAILY_PCT,
) -> MomentumSignal:
    """모멘텀 추격 시그널 평가.

    Args:
        five_min_candles: 5분봉 [{open, high, low, close, volume}, ...]
                          최신이 마지막. 평균 거래량 계산용 최소 21봉 필요.
        daily_pct: 종일 등락률 (전일 대비, %)
        current_price: 현재가
    """
    sig = MomentumSignal(ticker=ticker, name=name, current_price=current_price,
                         daily_pct=daily_pct)

    if not five_min_candles or len(five_min_candles) < 2:
        sig.reason = "데이터 부족"
        return sig

    if current_price <= 0:
        sig.reason = "현재가 0"
        return sig

    last = five_min_candles[-1]
    open_p = float(last.get("open", 0) or 0)
    close_p = float(last.get("close", 0) or 0)
    vol = int(last.get("volume", 0) or 0)

    if open_p <= 0 or close_p <= 0:
        sig.reason = "5분봉 가격 부적합"
        return sig

    # 1. 5분봉 등락률
    sig.five_min_pct = (close_p - open_p) / open_p * 100
    if sig.five_min_pct < min_5min_pct:
        sig.reason = f"5분봉 {sig.five_min_pct:+.2f}% < {min_5min_pct}%"
        return sig

    # 2. 양봉 확인
    if close_p <= open_p:
        sig.reason = "음봉"
        return sig

    # 3. 거래량 평균 대비 (최근 20봉 평균, 마지막 봉 제외)
    prev_candles = five_min_candles[-21:-1] if len(five_min_candles) >= 21 else five_min_candles[:-1]
    if prev_candles:
        avg_vol = sum(int(c.get("volume", 0) or 0) for c in prev_candles) / len(prev_candles)
        sig.vol_ratio = vol / max(avg_vol, 1)
    else:
        sig.vol_ratio = 0

    if sig.vol_ratio < min_vol_ratio:
        sig.reason = f"거래량 {sig.vol_ratio:.1f}x < {min_vol_ratio}x"
        return sig

    # 4. 종일 등락률 범위 (1~20%)
    if daily_pct > max_daily_pct:
        sig.reason = f"종일 {daily_pct:+.1f}% > {max_daily_pct}% (과열)"
        return sig
    if daily_pct < min_daily_pct:
        sig.reason = f"종일 {daily_pct:+.1f}% < {min_daily_pct}% (약세)"
        return sig

    # 5. 매수 가격 산출
    target = _adjust_to_tick(int(current_price * 1.003))   # +0.3% slippage
    stop = _adjust_to_tick(int(current_price * 0.97))      # -3%
    profit = _adjust_to_tick(int(current_price * 1.05))    # +5%

    sig.fire = True
    sig.target_price = target
    sig.stop_price = stop
    sig.profit_target = profit
    sig.reason = (
        f"5분봉 {sig.five_min_pct:+.2f}% + 거래량 {sig.vol_ratio:.1f}x "
        f"+ 종일 {daily_pct:+.1f}% (매수가 {target:,})"
    )
    return sig


def _verify_15min_trend(intraday_adapter, ticker: str) -> bool:
    """15분봉 추세 검증 (P0-2, ChatGPT 외부 검증 발견 추가).

    조건: 최근 3개 15분봉 중 2개 이상 양봉 + 평균 가격 우상향
    → 단기 루머/뉴스 가짜 신호 회피

    Returns:
        True = 15분봉 추세 확인 (매수 OK)
        False = 추세 약함 (가짜 신호 의심)
    """
    if not hasattr(intraday_adapter, "fetch_minute_candles"):
        return True  # fail-open (5분봉만 보고 매수)
    try:
        candles = intraday_adapter.fetch_minute_candles(ticker, period=15)
        if not candles or len(candles) < 3:
            return True  # fail-open
        recent_3 = candles[-3:]
        # 양봉 카운트
        bullish = sum(1 for c in recent_3 if c.get("close", 0) > c.get("open", 0))
        # 평균 가격 우상향 (첫 평균 < 마지막 close)
        first_avg = (recent_3[0].get("open", 0) + recent_3[0].get("close", 0)) / 2
        last_close = recent_3[-1].get("close", 0)
        return bullish >= 2 and last_close > first_avg
    except Exception as e:
        logger.warning("[모멘텀] %s 15분봉 검증 실패: %s — fail-open", ticker, e)
        return True  # fail-open


def scan_momentum_candidates(
    intraday_adapter,
    candidate_tickers: list[str],
    name_map: dict[str, str] | None = None,
    held_tickers: set[str] | None = None,
    protected_tickers: set[str] | None = None,
) -> list[MomentumSignal]:
    """후보 풀 일괄 평가 → 모멘텀 발화 시그널 반환.

    Args:
        intraday_adapter: KisIntradayAdapter (fetch_minute_candles + fetch_tick)
        candidate_tickers: 평가 대상
        name_map: ticker → name 매핑
        held_tickers: 보유 종목 (제외)
        protected_tickers: 보호 종목 (제외)
    """
    if not ENABLED:
        logger.debug("[모멘텀 추격] MOMENTUM_CHASE_ENABLED=0 — 비활성")
        return []

    held_tickers = held_tickers or set()
    protected_tickers = protected_tickers or set()
    name_map = name_map or {}

    fired = []
    for tk in candidate_tickers:
        tkey = str(tk).zfill(6)
        if tkey in held_tickers or tkey in protected_tickers:
            continue

        try:
            # 5분봉 (최근 30분 = 6봉, 평균은 20봉 권장)
            candles = intraday_adapter.fetch_minute_candles(tkey, period=5)
            if not candles or len(candles) < 2:
                continue

            # 현재가 + 종일 등락률 (fetch_tick 또는 fetch_price)
            try:
                tick = intraday_adapter.fetch_tick(tkey)
                cur = int(tick.get("current_price", 0) or 0)
                daily_pct = float(tick.get("change_pct", 0) or 0)
            except Exception:
                cur = int(candles[-1].get("close", 0) or 0)
                daily_pct = 0  # fallback 시 daily_pct=0 → 미발화

            nm = name_map.get(tkey, tkey)
            sig = evaluate_momentum_signal(
                ticker=tkey, name=nm, five_min_candles=candles,
                daily_pct=daily_pct, current_price=cur,
            )
            if sig.fire:
                # ★ P0-2 fix (ChatGPT 외부 검증): 15분봉 추세 검증 추가 — 가짜 신호 회피
                if not _verify_15min_trend(intraday_adapter, tkey):
                    sig.fire = False
                    sig.reason = f"{sig.reason} → 15분봉 추세 약함 (가짜 신호 의심)"
                    logger.warning("[모멘텀 추격] %s 15분봉 추세 검증 실패 — 매수 보류", tkey)
                else:
                    fired.append(sig)
                    logger.warning(
                        "[모멘텀 추격] %s 발화 + 15분봉 추세 확인 — %s", tkey, sig.reason,
                    )
        except Exception as e:
            logger.warning("[모멘텀 추격] %s fetch 실패: %s", tkey, e)

    return fired


def format_momentum_for_telegram(sig: MomentumSignal) -> str:
    """텔레그램 알림 포맷."""
    return (
        f"⚡ [모멘텀 추격] {sig.name}({sig.ticker})\n"
        f"  현재가 {sig.current_price:,}원 (종일 {sig.daily_pct:+.2f}%)\n"
        f"  5분봉 {sig.five_min_pct:+.2f}% + 거래량 {sig.vol_ratio:.1f}x\n"
        f"  매수 {sig.target_price:,}원 → 익절 {sig.profit_target:,} (+5%)\n"
        f"                         / 손절 {sig.stop_price:,} (-3%)"
    )
