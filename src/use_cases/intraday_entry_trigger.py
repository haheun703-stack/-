"""5분봉 진입 트리거 (MVP-6, 2026-05-27 신규).

배경 (5/27 퐝가님 통찰):
  현재 MVP1~MVP5 = 일봉 기반 (천장 -3%/-10% 등)
  실전 매매법은 5분봉 진입 타이밍이 핵심
  → 일봉 (후보 자격) × 5분봉 (정확한 매수 시점) 결합

조건 (4개 중 3개 충족 시 trigger):
  1. 5분봉 양봉 (close > open)
  2. 거래량 급증 (직전 5분 평균 × 1.5+)
  3. VWAP 회복 (현재가 ≥ 일중 VWAP)
  4. 5분봉 RSI 30~70 (과매도 회복 + 과매수 회피)

데이터:
  - mojito.fetch_today_1m_ohlcv (1분봉 오늘만) → 5분 resample
  - 실시간 진입 트리거 전용 (백테스트는 자체 data/intraday/5min/ 사용)

통합:
  - run_adaptive_cycle.py MVP6로 추가
  - step5 통과 후보 풀 × intraday_entry_trigger → trigger 시 paper 매수
  - 보호 종목(protected_tickers.yaml) 자동 제외

5/27 협업 체제:
  - 메인 AI (클로드 코드) 작성
  - 코덱스 외부 검수
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IntradayEntryDecision:
    """5분봉 진입 트리거 평가 결과."""

    ticker: str
    name: str = ""
    # 조건별 결과 (4개)
    bullish_candle: bool = False        # 1. 양봉
    volume_surge: bool = False           # 2. 거래량 급증
    vwap_recovery: bool = False          # 3. VWAP 회복
    rsi_healthy: bool = False            # 4. RSI 30~70
    # 측정값
    current_price: int = 0
    five_min_open: int = 0
    five_min_close: int = 0
    five_min_volume: int = 0
    avg_volume_prev5: float = 0.0
    vwap: float = 0.0
    rsi: float = 0.0
    # 종합
    pass_count: int = 0
    trigger: bool = False                # 3+ 충족 시 True
    reasons_pass: list[str] = field(default_factory=list)
    reasons_fail: list[str] = field(default_factory=list)


def _resample_1m_to_5m(rows: list[dict]) -> list[dict]:
    """1분봉 → 5분봉 resample.

    Args:
        rows: KIS API 1분봉 응답 (output2). 최신순(역순).

    Returns:
        5분봉 리스트 (최신순). 각 dict: open/high/low/close/volume/timestamp.
    """
    if not rows:
        return []

    # 시간순 정렬 (오래된 것부터) — 5분 묶음 계산용
    sorted_rows = list(reversed(rows))
    bars = []
    bucket: list[dict] = []
    bucket_start_min = None

    for row in sorted_rows:
        try:
            t = row.get("stck_cntg_hour", "")  # "HHMMSS"
            if len(t) < 4:
                continue
            hh = int(t[:2])
            mm = int(t[2:4])
            slot_min = (hh * 60 + mm) // 5 * 5  # 5분 슬롯

            if bucket_start_min is None:
                bucket_start_min = slot_min

            if slot_min != bucket_start_min:
                # 새 5분 슬롯 시작 → 직전 bucket 마감
                if bucket:
                    bars.append(_aggregate_bucket(bucket, bucket_start_min))
                bucket = []
                bucket_start_min = slot_min

            bucket.append(row)
        except (ValueError, KeyError):
            continue

    # 마지막 bucket 마감
    if bucket and bucket_start_min is not None:
        bars.append(_aggregate_bucket(bucket, bucket_start_min))

    # 최신순으로 반환
    return list(reversed(bars))


def _aggregate_bucket(bucket: list[dict], slot_min: int) -> dict:
    """1분봉 5개를 5분봉 1개로 집계."""
    opens = [float(r.get("stck_oprc", 0)) for r in bucket]
    highs = [float(r.get("stck_hgpr", 0)) for r in bucket]
    lows = [float(r.get("stck_lwpr", 0)) for r in bucket]
    closes = [float(r.get("stck_prpr", 0)) for r in bucket]
    volumes = [float(r.get("cntg_vol", 0)) for r in bucket]
    hh = slot_min // 60
    mm = slot_min % 60
    return {
        "open": opens[0] if opens else 0,
        "high": max(highs) if highs else 0,
        "low": min(lows) if lows else 0,
        "close": closes[-1] if closes else 0,
        "volume": sum(volumes),
        "slot": f"{hh:02d}:{mm:02d}",
    }


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """간단 RSI 계산. 데이터 부족 시 50 반환."""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate_intraday_entry(
    broker,
    ticker: str,
    name: str = "",
    volume_surge_ratio: float = 1.5,
    rsi_min: float = 30.0,
    rsi_max: float = 70.0,
    min_pass_count: int = 3,
) -> IntradayEntryDecision:
    """5분봉 진입 트리거 평가.

    Args:
        broker: mojito.KoreaInvestment 또는 호환 어댑터
        ticker: 6자리 종목 코드
        name: 종목명 (로깅용)
        volume_surge_ratio: 거래량 급증 배수 (기본 1.5)
        rsi_min/rsi_max: RSI 정상 범위 (기본 30~70)
        min_pass_count: 트리거 발화 최소 조건 수 (기본 3 of 4)

    Returns:
        IntradayEntryDecision (trigger + 4 조건 결과 + 측정값)
    """
    dec = IntradayEntryDecision(ticker=ticker, name=name)

    # 1분봉 fetch (오늘만)
    try:
        res = broker.fetch_today_1m_ohlcv(ticker)
        rows = res.get("output2", []) if res else []
    except Exception as e:
        dec.reasons_fail.append(f"1분봉 fetch 실패: {e}")
        return dec

    if not rows:
        dec.reasons_fail.append("1분봉 데이터 없음")
        return dec

    # 5분봉 resample
    bars = _resample_1m_to_5m(rows)
    if len(bars) < 2:
        dec.reasons_fail.append(f"5분봉 부족 ({len(bars)} bars)")
        return dec

    # 현재 (최신) 5분봉
    cur = bars[0]
    dec.five_min_open = int(cur["open"])
    dec.five_min_close = int(cur["close"])
    dec.five_min_volume = int(cur["volume"])
    dec.current_price = dec.five_min_close

    # 조건 1: 양봉
    dec.bullish_candle = cur["close"] > cur["open"]
    if dec.bullish_candle:
        dec.reasons_pass.append(f"양봉 ({cur['open']:.0f} → {cur['close']:.0f})")
    else:
        dec.reasons_fail.append(f"음봉/도지 ({cur['open']:.0f} → {cur['close']:.0f})")

    # 조건 2: 거래량 급증 (직전 5개 5분봉 평균 대비)
    prev_bars = bars[1:6] if len(bars) >= 6 else bars[1:]
    if prev_bars:
        avg_vol = sum(b["volume"] for b in prev_bars) / len(prev_bars)
        dec.avg_volume_prev5 = avg_vol
        dec.volume_surge = cur["volume"] >= avg_vol * volume_surge_ratio if avg_vol > 0 else False
        ratio = cur["volume"] / avg_vol if avg_vol > 0 else 0
        if dec.volume_surge:
            dec.reasons_pass.append(f"거래량 급증 ({ratio:.2f}x ≥ {volume_surge_ratio}x)")
        else:
            dec.reasons_fail.append(f"거래량 부족 ({ratio:.2f}x < {volume_surge_ratio}x)")
    else:
        dec.reasons_fail.append("직전 5분봉 없음 (거래량 비교 불가)")

    # 조건 3: VWAP 회복 (일중 누적)
    total_pv = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in bars)
    total_v = sum(b["volume"] for b in bars)
    if total_v > 0:
        dec.vwap = total_pv / total_v
        dec.vwap_recovery = cur["close"] >= dec.vwap
        gap = cur["close"] - dec.vwap
        if dec.vwap_recovery:
            dec.reasons_pass.append(f"VWAP 위 (현재 {cur['close']:.0f} ≥ VWAP {dec.vwap:.0f}, +{gap:.0f})")
        else:
            dec.reasons_fail.append(f"VWAP 아래 (현재 {cur['close']:.0f} < VWAP {dec.vwap:.0f}, {gap:.0f})")
    else:
        dec.reasons_fail.append("거래량 0 (VWAP 계산 불가)")

    # 조건 4: RSI 30~70
    closes = [b["close"] for b in reversed(bars)]  # 오래된 순
    dec.rsi = _compute_rsi(closes)
    dec.rsi_healthy = rsi_min <= dec.rsi <= rsi_max
    if dec.rsi_healthy:
        dec.reasons_pass.append(f"RSI 정상 ({dec.rsi:.1f}, {rsi_min}~{rsi_max})")
    else:
        if dec.rsi < rsi_min:
            dec.reasons_fail.append(f"RSI 과매도 ({dec.rsi:.1f} < {rsi_min})")
        else:
            dec.reasons_fail.append(f"RSI 과매수 ({dec.rsi:.1f} > {rsi_max})")

    # 종합
    dec.pass_count = sum([
        dec.bullish_candle,
        dec.volume_surge,
        dec.vwap_recovery,
        dec.rsi_healthy,
    ])
    dec.trigger = dec.pass_count >= min_pass_count

    return dec


def format_for_telegram(dec: IntradayEntryDecision) -> str:
    """진입 트리거 결과를 텔레그램 메시지로 포맷."""
    if not dec.trigger:
        return ""  # 트리거 X면 알림 안 보냄

    emoji = "🟢"
    lines = [
        f"{emoji} [5분봉 진입 트리거] {dec.name}({dec.ticker})",
        f"  현재가: {dec.current_price:,} (5분봉 {dec.five_min_open:,} → {dec.five_min_close:,})",
        f"  통과: {dec.pass_count}/4",
    ]
    for r in dec.reasons_pass:
        lines.append(f"  ✅ {r}")
    if dec.reasons_fail:
        for r in dec.reasons_fail:
            lines.append(f"  ⚪ {r}")
    return "\n".join(lines)
