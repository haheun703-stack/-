"""장중 실시간 통합 점수 — 우리꺼 + 정보봇 시그널 (5/22 퐝가님 인사이트).

배경:
  사용자 핵심 질문 (5/22 저녁):
    "장중에 4번의 수급이 있는데 더 오를지 내릴지를 빨리 알 수 있는 방법은?
     일봉 지표 외에 어떤 시그널이 있나?"

  답: 우리꺼 (분봉/VWAP/체결강도/호가) + 정보봇 (smart_money/sniper/섹터/시간대별 수급)
  통합 8 시그널 가중 점수 시스템.

8 시그널 통합 (총 -12 ~ +22):
  [우리 시스템]
  1. 분봉 양봉비율 (entry_score 재활용)            0~+2
  2. VWAP 위치 상/중/하                            0~+3
  3. 거래량 서지                                  0~+2
  4. 호가 매수1/매도1                              0~+2
  5. 체결강도 90+ 추세 (volume_power_tracker)     -1~+3

  [정보봇 통합 — quant_supabase_reader]
  6. 정보봇 dashboard_smart_money (외인+기관 동시)  0~+3
  7. 정보봇 dashboard_sniper (반등임박/수급반전)    0~+3
  8. 섹터 sector_fire 모멘텀                       0~+2

  [매크로 — market_regime_guard]
  9. KOSPI regime                                -3~+2

진입/홀드/매도 결정:
  ≥ +15: ★STRONG_BUY (신규 진입 또는 추매)
  +8~+14: BUY (자비스 신규 진입 / 보유 종목 유지)
  +3~+7:  HOLD (관망)
  -2~+2:  WEAK_HOLD (일부 익절 검토)
  ≤ -3:   EXIT (선제 청산 — owner_rule 트리거 전)

5/22 HPSP 예시:
  정보봇 sniper 82 ('반등임박' 5/21) → 다음날 +10% 적중
  → smart_money +3 + sniper +3 + 섹터 +2 + 우리 entry_score +6
  = +14 BUY 추정

사용:
  from src.use_cases.realtime_score import calculate_realtime_score
  result = calculate_realtime_score(broker, ticker)
  if result['recommend'] == 'STRONG_BUY':
      적극 매수
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

from src.use_cases.entry_score import calculate_entry_score
from src.use_cases.market_regime_guard import get_kospi_regime

logger = logging.getLogger(__name__)

# 진입/홀드 결정 임계
SCORE_STRONG_BUY = 15
SCORE_BUY = 8
SCORE_HOLD = 3
SCORE_EXIT = -3


def _fetch_intel_smart_money(ticker: str) -> dict[str, Any]:
    """정보봇 dashboard_smart_money 조회 (외인+기관 동시 매수).

    score 구간 (정보봇 매핑):
        ≥ 100: DUAL_FLOW (외인+기관 동시 폭발) — +3
        ≥ 60:  연속 유입 — +2
        ≥ 30:  일부 매수 — +1
        그 외:  0
    """
    try:
        from src.adapters.quant_supabase_reader import _get_client
        client = _get_client()
        if not client:
            return {"score": 0, "raw": 0, "reason": "supabase 미연결"}
        res = (
            client.table("dashboard_smart_money")
            .select("ticker,score,signal,date")
            .eq("ticker", str(ticker).zfill(6))
            .order("date", desc=True)
            .limit(3)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {"score": 0, "raw": 0, "reason": "smart_money 시그널 없음"}
        latest = rows[0]
        raw = float(latest.get("score", 0) or 0)
        signal = latest.get("signal", "")

        if raw >= 100:
            return {"score": 3, "raw": raw, "signal": signal,
                    "reason": f"smart_money DUAL_FLOW {raw} (+3)"}
        elif raw >= 60:
            return {"score": 2, "raw": raw, "signal": signal,
                    "reason": f"smart_money 연속유입 {raw} (+2)"}
        elif raw >= 30:
            return {"score": 1, "raw": raw, "signal": signal,
                    "reason": f"smart_money 일부매수 {raw} (+1)"}
        return {"score": 0, "raw": raw, "signal": signal,
                "reason": f"smart_money 약함 {raw}"}
    except Exception as e:
        logger.debug("smart_money 조회 실패 %s: %s", ticker, e)
        return {"score": 0, "raw": 0, "reason": f"조회 예외 {e}"}


def _fetch_intel_sniper(ticker: str) -> dict[str, Any]:
    """정보봇 dashboard_sniper 조회 (반등임박/수급반전).

    score + signal 구간:
        score ≥ 80 + signal in (반등임박, 수급반전): +3 (5/21 HPSP +10% 적중 사례)
        score ≥ 70: +2
        score ≥ 50: +1
        그 외: 0
    """
    try:
        from src.adapters.quant_supabase_reader import _get_client
        client = _get_client()
        if not client:
            return {"score": 0, "raw": 0, "reason": "supabase 미연결"}
        res = (
            client.table("dashboard_sniper")
            .select("ticker,score,signal,date")
            .eq("ticker", str(ticker).zfill(6))
            .order("date", desc=True)
            .limit(3)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {"score": 0, "raw": 0, "reason": "sniper 시그널 없음"}
        latest = rows[0]
        raw = float(latest.get("score", 0) or 0)
        signal = latest.get("signal", "")
        strong_signals = {"반등임박", "수급반전", "수급 반전", "반등 임박"}

        if raw >= 80 and signal in strong_signals:
            return {"score": 3, "raw": raw, "signal": signal,
                    "reason": f"★sniper {signal} {raw} (+3, HPSP사례)"}
        elif raw >= 70:
            return {"score": 2, "raw": raw, "signal": signal,
                    "reason": f"sniper {signal} {raw} (+2)"}
        elif raw >= 50:
            return {"score": 1, "raw": raw, "signal": signal,
                    "reason": f"sniper {signal} {raw} (+1)"}
        return {"score": 0, "raw": raw, "signal": signal,
                "reason": f"sniper 약함 {raw}"}
    except Exception as e:
        logger.debug("sniper 조회 실패 %s: %s", ticker, e)
        return {"score": 0, "raw": 0, "reason": f"조회 예외 {e}"}


def _fetch_sector_momentum(ticker: str) -> dict[str, Any]:
    """정보봇 sector_fire 기반 섹터 모멘텀.

    종목 → 섹터 → sector_fire_score 매핑.
    fire_score 기준:
        ≥ 80: 섹터 강세 → +2
        ≥ 60: 섹터 보통 → +1
        그 외: 0
    """
    try:
        from src.adapters.quant_supabase_reader import get_sector_fire_today, get_company_card
        # 종목 → 섹터 (정보봇 company_card)
        card = get_company_card(str(ticker).zfill(6))
        if not card:
            return {"score": 0, "sector": "", "reason": "섹터 정보 없음"}
        sector = card.get("sector", "")
        if not sector:
            return {"score": 0, "sector": "", "reason": "섹터 미지정"}
        # 오늘 sector_fire
        today = datetime.now().strftime("%Y-%m-%d")
        sectors = get_sector_fire_today(today, min_score=0)
        match = next((s for s in sectors if s.get("sector") == sector), None)
        if not match:
            return {"score": 0, "sector": sector, "reason": f"섹터 fire 없음 ({sector})"}
        fire = float(match.get("fire_score", 0) or 0)
        if fire >= 80:
            return {"score": 2, "sector": sector, "fire": fire,
                    "reason": f"섹터({sector}) fire {fire:.0f} (+2)"}
        elif fire >= 60:
            return {"score": 1, "sector": sector, "fire": fire,
                    "reason": f"섹터({sector}) fire {fire:.0f} (+1)"}
        return {"score": 0, "sector": sector, "fire": fire,
                "reason": f"섹터({sector}) fire 약함 {fire:.0f}"}
    except Exception as e:
        logger.debug("sector_momentum 조회 실패 %s: %s", ticker, e)
        return {"score": 0, "sector": "", "reason": f"조회 예외 {e}"}


def _score_macro_regime() -> dict[str, Any]:
    """매크로 KOSPI regime 점수.

    STRONG_BULL: +2
    NEUTRAL:     +1
    CAUTION:     0
    BEARISH:    -3 (매크로 가드와 동일)
    UNKNOWN:    -2
    """
    info = get_kospi_regime()
    regime = info["regime"]
    chg = info.get("kospi_chg_pct", 0)
    mapping = {
        "STRONG_BULL": (2, f"매크로 STRONG_BULL (KOSPI {chg:+.2f}%) (+2)"),
        "NEUTRAL": (1, f"매크로 NEUTRAL (KOSPI {chg:+.2f}%) (+1)"),
        "CAUTION": (0, f"매크로 CAUTION (KOSPI {chg:+.2f}%) (0)"),
        "BEARISH": (-3, f"매크로 BEARISH (KOSPI {chg:+.2f}%) (-3)"),
        "UNKNOWN": (-2, "매크로 UNKNOWN (-2)"),
    }
    score, reason = mapping.get(regime, (0, f"매크로 {regime}"))
    return {"score": score, "regime": regime, "kospi_chg": chg, "reason": reason}


def calculate_realtime_score(
    broker,
    ticker: str,
    current_price: int | None = None,
) -> dict[str, Any]:
    """장중 실시간 통합 점수 (8 시그널 가중).

    호출 시점:
      - 자비스 cron (매 5분, 14:00~14:55)
      - counter_trade_monitor (보유 종목 매 30분)
      - owner_rule_monitor trailing 발동 전 (선제 검증)

    Args:
        broker: KIS broker
        ticker: 종목 코드
        current_price: 현재가 (None이면 broker.fetch_price)

    Returns:
        {
            "total": int,                       # 합산 점수 (-12 ~ +22)
            "recommend": str,                   # STRONG_BUY/BUY/HOLD/WEAK_HOLD/EXIT
            "breakdown": dict,                  # 시그널별 점수 + reason
            "entry_score": dict,                # entry_score 전체 결과 (Level 1 필수)
            "intel_smart_money": dict,
            "intel_sniper": dict,
            "sector_momentum": dict,
            "macro": dict,
            "reasoning": str,
        }
    """
    breakdown = {}

    # === 1. 우리 시스템 (entry_score 5 시그널: 분봉/VWAP/거래량/호가/체결강도 90+) ===
    es = calculate_entry_score(broker, ticker, current_price=current_price)
    es_score = es["score"]
    breakdown["우리_entry_score"] = {"score": es_score, "reason": es["reasoning"]}

    # === 2. 정보봇 dashboard_smart_money (외인+기관 동시) ===
    intel_sm = _fetch_intel_smart_money(ticker)
    breakdown["정보봇_smart_money"] = {"score": intel_sm["score"], "reason": intel_sm["reason"]}

    # === 3. 정보봇 dashboard_sniper (반등임박/수급반전) ===
    intel_sn = _fetch_intel_sniper(ticker)
    breakdown["정보봇_sniper"] = {"score": intel_sn["score"], "reason": intel_sn["reason"]}

    # === 4. 정보봇 섹터 sector_fire 모멘텀 ===
    sector = _fetch_sector_momentum(ticker)
    breakdown["섹터_모멘텀"] = {"score": sector["score"], "reason": sector["reason"]}

    # === 5. 매크로 KOSPI regime ===
    macro = _score_macro_regime()
    breakdown["매크로_KOSPI"] = {"score": macro["score"], "reason": macro["reason"]}

    # === 합산 ===
    total = es_score + intel_sm["score"] + intel_sn["score"] + sector["score"] + macro["score"]

    # === 진입/홀드/매도 결정 ===
    if total >= SCORE_STRONG_BUY:
        recommend = "STRONG_BUY"
    elif total >= SCORE_BUY:
        recommend = "BUY"
    elif total >= SCORE_HOLD:
        recommend = "HOLD"
    elif total > SCORE_EXIT:
        recommend = "WEAK_HOLD"
    else:
        recommend = "EXIT"

    reasoning = (
        f"실시간 점수 {total:+d} ({recommend}): "
        f"우리={es_score:+d} / smart_money={intel_sm['score']:+d} / "
        f"sniper={intel_sn['score']:+d} / 섹터={sector['score']:+d} / 매크로={macro['score']:+d}"
    )

    return {
        "total": total,
        "recommend": recommend,
        "breakdown": breakdown,
        "entry_score": es,
        "intel_smart_money": intel_sm,
        "intel_sniper": intel_sn,
        "sector_momentum": sector,
        "macro": macro,
        "reasoning": reasoning,
    }


def format_realtime_summary(result: dict[str, Any]) -> str:
    """텔레그램/로그용 요약 포맷."""
    bd = result.get("breakdown", {})
    lines = [
        f"📊 {result['recommend']} (점수 {result['total']:+d})",
    ]
    for label, val in bd.items():
        lines.append(f"  {label}: {val['score']:+d} | {val['reason']}")
    return "\n".join(lines)
