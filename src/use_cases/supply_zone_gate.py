"""매물대 지지/저항 게이트 — H6 (5/26 PDCA flexible-pullback-buy).

배경:
- 5/22 풀세트 D 학습 11종 중 매물대: "모델만 정의 (SupplyZone), 분석 함수 X"
- entities/models.py에 SupplyZone 클래스만 있고 실제 계산/게이트 없음
- 매물대 = 거래량 누적 가격대 = 지지/저항선
- 매수 시점에 매물대 위치 확인 → 진입 정밀도 향상

알고리즘 (Volume Profile / Market Profile 표준):
1. N일 (기본 60일) OHLCV 거래량 합계
2. 가격대를 K개 bin (기본 20)으로 분할
3. 각 bin에 거래량 누적 (해당 가격대를 거친 봉의 거래량)
4. POC (Point of Control) = 최대 거래량 bin의 가격 중앙값 = 주요 지지/저항
5. VAH / VAL = 70% 거래량 범위 (Value Area High/Low) = 매물 집중 구간

룰 (백테스트 검증 전 잠정치 — 5/27 백테스트 후 보정):
- POC_BREAKOUT: 현재가 > POC × 1.005 → 저항 돌파, 매수 우대 (★)
- POC_SUPPORT: POC × 0.995 ≤ 현재가 ≤ POC × 1.005 → POC 지지, 정상 통과
- VAH_OVERHEATED: 현재가 > VAH × 1.02 → 매물대 과열 차단
- VAL_BREAKDOWN: 현재가 < VAL × 0.98 → 매도세 강함 차단
- INSIDE_VA: VAL < 현재가 < VAH → 정상 (가치 구간)
- DATA_MISSING: OHLCV 부족 → fail-open

활용:
- src/use_cases/adaptive_buy_queue.py execute_auto_buy() 직전 호출
- OHLCV 소스: broker.fetch_ohlcv() 60일 또는 정보봇 OHLCV CSV
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 환경변수
DEFAULT_LOOKBACK_DAYS = int(os.getenv("SUPPLY_ZONE_LOOKBACK_DAYS", "60"))
DEFAULT_BINS = int(os.getenv("SUPPLY_ZONE_BINS", "20"))
DEFAULT_VA_PCT = float(os.getenv("SUPPLY_ZONE_VA_PCT", "70"))  # Value Area 70%
POC_BREAKOUT_PCT = float(os.getenv("SUPPLY_ZONE_BREAKOUT_PCT", "0.5"))  # POC × 1.005
VAH_OVERHEAT_PCT = float(os.getenv("SUPPLY_ZONE_OVERHEAT_PCT", "2.0"))  # VAH × 1.02
VAL_BREAKDOWN_PCT = float(os.getenv("SUPPLY_ZONE_BREAKDOWN_PCT", "2.0"))  # VAL × 0.98


@dataclass
class SupplyZoneProfile:
    """매물대 프로파일 계산 결과."""
    poc_price: float          # 최대 거래량 가격 (Point of Control)
    vah_price: float          # Value Area High
    val_price: float          # Value Area Low
    total_volume: int         # 전체 누적 거래량
    bin_count: int            # bin 개수
    zones: list[dict] = field(default_factory=list)  # [{price_lo, price_hi, volume, pct}]


@dataclass
class SupplyZoneGate:
    """매물대 게이트 결과."""
    allow: bool
    reason: str               # 'POC_BREAKOUT' / 'POC_SUPPORT' / 'INSIDE_VA' / 'VAH_OVERHEATED' / 'VAL_BREAKDOWN' / 'DATA_MISSING'
    position: str             # 현재가 위치 표기 (예: 'POC +0.8%')
    current_price: int
    poc_price: float
    vah_price: float
    val_price: float
    is_breakout: bool         # POC 돌파 우대 여부


def calc_supply_zones(
    ohlcv: list[dict],
    bins: int = DEFAULT_BINS,
    value_area_pct: float = DEFAULT_VA_PCT,
) -> Optional[SupplyZoneProfile]:
    """OHLCV로부터 매물대 프로파일 계산.

    Args:
        ohlcv: [{open, high, low, close, volume}, ...] (오름차순 정렬 권장)
        bins: 가격대 분할 수
        value_area_pct: VA 누적 거래량 % (기본 70%)

    Returns:
        SupplyZoneProfile or None (데이터 부족 시).
    """
    if not ohlcv or len(ohlcv) < 5:
        return None

    # 가격 범위 추출
    highs = []
    lows = []
    for bar in ohlcv:
        try:
            h = float(bar.get("high", 0))
            l = float(bar.get("low", 0))
            if h > 0 and l > 0:
                highs.append(h)
                lows.append(l)
        except (TypeError, ValueError):
            continue
    if not highs or not lows:
        return None

    price_max = max(highs)
    price_min = min(lows)
    if price_max <= price_min:
        return None

    bin_width = (price_max - price_min) / bins
    if bin_width <= 0:
        return None

    # bin 거래량 누적
    bin_volumes = [0] * bins
    for bar in ohlcv:
        try:
            h = float(bar.get("high", 0))
            l = float(bar.get("low", 0))
            v = int(bar.get("volume", 0))
            if h <= 0 or l <= 0 or v <= 0:
                continue
            # 봉의 가격 범위에 거래량 균등 분배
            lo_idx = max(0, int((l - price_min) / bin_width))
            hi_idx = min(bins - 1, int((h - price_min) / bin_width))
            n = hi_idx - lo_idx + 1
            vol_per_bin = v / n
            for i in range(lo_idx, hi_idx + 1):
                bin_volumes[i] += vol_per_bin
        except (TypeError, ValueError, ZeroDivisionError):
            continue

    total_vol = sum(bin_volumes)
    if total_vol <= 0:
        return None

    # POC: 최대 거래량 bin의 중앙 가격
    poc_idx = bin_volumes.index(max(bin_volumes))
    poc_price = price_min + (poc_idx + 0.5) * bin_width

    # Value Area: POC부터 양쪽으로 70% 거래량 도달까지 확장
    target_vol = total_vol * value_area_pct / 100
    lo_idx = poc_idx
    hi_idx = poc_idx
    accumulated = bin_volumes[poc_idx]
    while accumulated < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
        # 양쪽 중 큰 거래량 쪽으로 확장
        left_vol = bin_volumes[lo_idx - 1] if lo_idx > 0 else 0
        right_vol = bin_volumes[hi_idx + 1] if hi_idx < bins - 1 else 0
        if right_vol >= left_vol and hi_idx < bins - 1:
            hi_idx += 1
            accumulated += bin_volumes[hi_idx]
        elif lo_idx > 0:
            lo_idx -= 1
            accumulated += bin_volumes[lo_idx]
        else:
            break

    vah_price = price_min + (hi_idx + 1) * bin_width
    val_price = price_min + lo_idx * bin_width

    # zone 리스트 (상위 5개)
    indexed = sorted(enumerate(bin_volumes), key=lambda x: x[1], reverse=True)[:5]
    zones = [
        {
            "price_lo": price_min + i * bin_width,
            "price_hi": price_min + (i + 1) * bin_width,
            "volume": int(v),
            "pct": v / total_vol * 100,
        }
        for i, v in indexed
    ]

    return SupplyZoneProfile(
        poc_price=poc_price,
        vah_price=vah_price,
        val_price=val_price,
        total_volume=int(total_vol),
        bin_count=bins,
        zones=zones,
    )


def check_supply_zone_buy_gate(
    current_price: int,
    profile: Optional[SupplyZoneProfile],
    breakout_pct: float = POC_BREAKOUT_PCT,
    overheat_pct: float = VAH_OVERHEAT_PCT,
    breakdown_pct: float = VAL_BREAKDOWN_PCT,
) -> SupplyZoneGate:
    """매수 직전 매물대 게이트.

    우선순위: VAH 과열 차단 > VAL 이탈 차단 > POC 돌파 우대 > POC 지지 > 정상.
    """
    if profile is None or current_price <= 0:
        return SupplyZoneGate(
            allow=True, reason="DATA_MISSING", position="-",
            current_price=current_price, poc_price=0.0,
            vah_price=0.0, val_price=0.0, is_breakout=False,
        )

    poc, vah, val = profile.poc_price, profile.vah_price, profile.val_price
    pos_pct = (current_price - poc) / poc * 100 if poc else 0
    pos = f"POC {pos_pct:+.2f}%"

    # 차단 룰
    if current_price > vah * (1 + overheat_pct / 100):
        return SupplyZoneGate(
            allow=False, reason="VAH_OVERHEATED", position=pos,
            current_price=current_price, poc_price=poc,
            vah_price=vah, val_price=val, is_breakout=False,
        )
    if current_price < val * (1 - breakdown_pct / 100):
        return SupplyZoneGate(
            allow=False, reason="VAL_BREAKDOWN", position=pos,
            current_price=current_price, poc_price=poc,
            vah_price=vah, val_price=val, is_breakout=False,
        )

    # 우대 룰
    if current_price > poc * (1 + breakout_pct / 100):
        return SupplyZoneGate(
            allow=True, reason="POC_BREAKOUT", position=pos,
            current_price=current_price, poc_price=poc,
            vah_price=vah, val_price=val, is_breakout=True,
        )

    if abs(pos_pct) <= breakout_pct:
        return SupplyZoneGate(
            allow=True, reason="POC_SUPPORT", position=pos,
            current_price=current_price, poc_price=poc,
            vah_price=vah, val_price=val, is_breakout=False,
        )

    return SupplyZoneGate(
        allow=True, reason="INSIDE_VA", position=pos,
        current_price=current_price, poc_price=poc,
        vah_price=vah, val_price=val, is_breakout=False,
    )
