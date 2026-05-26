"""ATR 기반 동적 손익절 — H7 (5/26 퐝가님 지시 11종 풀세트).

배경:
- 현행 MVP-2.6: 손절 -5% 고정 (5/25 백테스트 R2 채택)
- 5/22 풀세트 D 학습 11종 중 ATR은 "알파엔진에만, 자비스 매수 X" 상태
- 종목 변동성에 따라 -5%가 너무 좁거나(저변동주) 너무 넓을(고변동주) 수 있음
- ATR(14일) × 배수로 종목별 동적 손익절 계산

데이터 소스 우선순위:
1. KIS API 일봉 14일 (실시간, 신선) — `fetch_atr_via_kis()`
2. 정보봇 OHLCV CSV 39컬럼 'ATR' 컬럼 (VPS만, 일별 갱신) — `load_atr_from_jgis()`
3. fallback: -5% 고정 (현행 MVP-2.6 유지)

룰 (백테스트 검증 전 잠정치, 5/27 백테스트 후 보정):
- 강세장 (regime=BULL): stop = entry - ATR × 2.0,  target = entry + ATR × 3.0 (R:R = 1.5)
- 중립장 (regime=NEUTRAL): stop = entry - ATR × 1.5, target = entry + ATR × 2.5 (R:R = 1.67)
- 약세장 (regime=BEARISH): stop = entry - ATR × 1.0, target = entry + ATR × 2.0 (R:R = 2.0, 빠른 익절)

가드레일:
- ATR이 entry × 10% 초과 시 → -5% 고정 fallback (이상치 방어)
- ATR <= 0 또는 fetch 실패 → -5% 고정 fallback

활용 위치:
- src/use_cases/adaptive_buy_queue.py execute_auto_buy() 직후 stop/target 산출
- MVP-2.6 stop_loss_pct=-5.0 → atr_dynamic_stop으로 대체 (env: ATR_STOP_ENABLED=1)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 환경변수
ATR_STOP_ENABLED = os.getenv("ATR_STOP_ENABLED", "0") == "1"
FALLBACK_STOP_PCT = float(os.getenv("MVP_AUTO_STOP_LOSS_PCT", "-5.0"))
ATR_CAP_PCT = float(os.getenv("ATR_CAP_PCT", "10.0"))  # ATR이 entry × X% 초과 시 fallback


# 레짐별 ATR 배수 (잠정치 — 백테스트 후 보정)
REGIME_MULTIPLIERS = {
    "BULL":    {"stop_mult": 2.0, "target_mult": 3.0},
    "NEUTRAL": {"stop_mult": 1.5, "target_mult": 2.5},
    "BEARISH": {"stop_mult": 1.0, "target_mult": 2.0},
}


@dataclass
class StopTarget:
    """ATR 기반 동적 손익절 결과."""
    stop_price: int          # 손절가 (원)
    target_price: int        # 익절가 (원)
    stop_pct: float          # entry 대비 손절 % (음수)
    target_pct: float        # entry 대비 익절 % (양수)
    atr_value: float         # 사용된 ATR
    regime: str              # 적용 레짐
    source: str              # 'ATR' or 'FALLBACK'
    reason: str              # 결정 사유


def calc_atr_dynamic_stop(
    entry_price: int,
    atr_value: float | None,
    regime: str = "NEUTRAL",
    fallback_stop_pct: float = FALLBACK_STOP_PCT,
    ticker: str = "?",  # ★ C4 fix (5/26 검수): 로그 추적용
) -> StopTarget:
    """ATR 기반 동적 손익절 계산.

    Args:
        entry_price: 매수 단가 (원)
        atr_value: ATR(14일) 절대값 (원). None 또는 0이면 fallback.
        regime: 'BULL' / 'NEUTRAL' / 'BEARISH'
        fallback_stop_pct: ATR 미사용 시 손절 % (기본 -5.0)

    Returns:
        StopTarget dataclass.
    """
    if entry_price <= 0:
        return StopTarget(
            stop_price=0, target_price=0,
            stop_pct=0.0, target_pct=0.0,
            atr_value=0.0, regime=regime,
            source="FALLBACK",
            reason="invalid entry_price",
        )

    # fallback case 1: ATR 데이터 없음
    if not atr_value or atr_value <= 0:
        stop_p = int(entry_price * (1 + fallback_stop_pct / 100))
        target_p = int(entry_price * (1 + 7.0 / 100))  # MVP-2.5 +7% trailing 시작점
        return StopTarget(
            stop_price=stop_p,
            target_price=target_p,
            stop_pct=fallback_stop_pct,
            target_pct=7.0,
            atr_value=0.0,
            regime=regime,
            source="FALLBACK",
            reason="ATR 미수신 또는 0 — 고정 -5% / +7%",
        )

    # 가드레일: ATR 이상치 (변동성 폭주 — entry × 10% 초과)
    atr_pct = atr_value / entry_price * 100
    if atr_pct > ATR_CAP_PCT:
        stop_p = int(entry_price * (1 + fallback_stop_pct / 100))
        target_p = int(entry_price * (1 + 7.0 / 100))
        logger.warning(
            "[ATR stop] %s entry=%d 이상치 (ATR=%.0f, %.1f%% > %.0f%%) — fallback",
            ticker, entry_price, atr_value, atr_pct, ATR_CAP_PCT,
        )
        return StopTarget(
            stop_price=stop_p, target_price=target_p,
            stop_pct=fallback_stop_pct, target_pct=7.0,
            atr_value=atr_value, regime=regime,
            source="FALLBACK",
            reason=f"ATR 이상치 {atr_pct:.1f}% > {ATR_CAP_PCT:.0f}%",
        )

    # 정상 경로: ATR × 배수
    mult = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["NEUTRAL"])
    stop_p = int(entry_price - atr_value * mult["stop_mult"])
    target_p = int(entry_price + atr_value * mult["target_mult"])

    stop_pct = (stop_p - entry_price) / entry_price * 100
    target_pct = (target_p - entry_price) / entry_price * 100

    logger.info(
        "[ATR stop] %s entry=%d ATR=%.0f regime=%s → stop %d (%+.1f%%) target %d (%+.1f%%)",
        ticker, entry_price, atr_value, regime, stop_p, stop_pct, target_p, target_pct,
    )

    return StopTarget(
        stop_price=stop_p,
        target_price=target_p,
        stop_pct=stop_pct,
        target_pct=target_pct,
        atr_value=atr_value,
        regime=regime,
        source="ATR",
        reason=f"ATR × ({mult['stop_mult']}/{mult['target_mult']}) @ {regime}",
    )


def fetch_atr_via_kis(broker, ticker: str, period: int = 14) -> float | None:
    """KIS 일봉 14일 → ATR 계산. broker는 KisOrderAdapter 또는 mojito.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = simple moving average of True Range over period

    Returns:
        ATR 절대값 (원). 실패 시 None.
    """
    try:
        # KIS 일봉 조회 (mojito: fetch_ohlcv 또는 직접 fetch)
        # 어댑터 인터페이스에 따라 분기
        if hasattr(broker, "fetch_ohlcv_daily"):
            ohlcv = broker.fetch_ohlcv_daily(ticker, period + 5)  # 여유 5일
        elif hasattr(broker, "fetch_ohlcv"):
            ohlcv = broker.fetch_ohlcv(ticker, "D", period + 5)
        else:
            logger.warning("[ATR] broker에 fetch_ohlcv_daily/fetch_ohlcv 없음 — None")
            return None

        if not ohlcv or len(ohlcv) < period + 1:
            logger.warning("[ATR] %s 일봉 부족 (%d < %d)", ticker, len(ohlcv or []), period + 1)
            return None

        # True Range 계산
        trs = []
        for i in range(1, len(ohlcv)):
            high = float(ohlcv[i].get("high", 0))
            low = float(ohlcv[i].get("low", 0))
            prev_close = float(ohlcv[i-1].get("close", 0))
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        # 최근 period개의 평균 (SMA)
        recent_trs = trs[-period:]
        atr = sum(recent_trs) / len(recent_trs)
        return atr if atr > 0 else None

    except Exception as e:
        logger.warning("[ATR] %s fetch 실패: %s", ticker, e)
        return None


def load_atr_from_jgis(ticker: str, csv_dir: str = "data/external/jgis_ohlcv") -> float | None:
    """정보봇 OHLCV CSV의 'ATR' 컬럼에서 최근값 로드 (VPS 전용).

    Returns:
        ATR 최근값. 파일 없거나 컬럼 누락 시 None.
    """
    from pathlib import Path
    import csv

    p = Path(csv_dir) / f"{str(ticker).zfill(6)}.csv"
    if not p.exists():
        # 폴더에 종목명_ticker.csv 형식도 시도
        candidates = list(Path(csv_dir).glob(f"*_{str(ticker).zfill(6)}.csv"))
        if candidates:
            p = candidates[0]
        else:
            return None

    try:
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_atr = None
            for row in reader:
                atr_str = row.get("ATR") or row.get("atr")
                if atr_str:
                    try:
                        v = float(atr_str)
                        if v > 0:
                            last_atr = v
                    except (ValueError, TypeError):
                        continue
            return last_atr
    except Exception as e:
        logger.warning("[ATR] CSV 로드 실패 %s: %s", p, e)
        return None
