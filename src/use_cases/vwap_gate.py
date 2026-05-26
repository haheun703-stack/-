"""VWAP 매수 게이트 — H4 (5/26 퐝가님 지시 11종 풀세트).

배경:
- 5/22 풀세트 D 학습 11종 중 VWAP는 "분류만 사용, 매수 게이트 X" 상태였음
- 5/26 퐝가님 지시: "기준 금액 미도달이어도 눌림목이면 유연 매수, 4수급 확인"
- VWAP는 눌림목/오버슈팅 판별의 핵심 — 매수 직전 게이트로 통합

데이터 소스:
- data/vwap_monitor.json (매 분 갱신, vwap_monitor 시스템이 생성)
- 구조: {"stocks": {ticker: {"vwap": float, "current_price": int,
                              "vwap_dev_pct": float, "day_high": int,
                              "day_low": int, "dip_count": int, ...}}}

룰 (백테스트 검증 전 잠정치 — 5/27 백테스트 후 보정):
1. OVERHEAT_BUY_BLOCK: vwap_dev_pct >= +2.0% → 매수 차단 (오버슈팅 회피)
2. DIP_BUY_BONUS: vwap_dev_pct <= -1.5% → 매수 우대 (눌림목 진입)
3. NORMAL: -1.5% < dev < +2.0% → 게이트 통과 (정상)
4. DATA_MISSING: vwap_monitor.json에 종목 없음 → fail-open (게이트 통과, warning 로그)
   사유: 신규 큐 등록 종목은 vwap_monitor 풀에 없을 수 있음. 차단 시 정상 매수 막힘.

활용 위치 (5/26 코드 작성, 16:00 이후 VPS 배포):
- src/use_cases/adaptive_buy_queue.py execute_auto_buy() — 매수 직전 게이트
- src/strategies/chart_hero_executor.py — 차트영웅 매수 직전 게이트 (선택)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VWAP_STATE_PATH = PROJECT_ROOT / "data" / "vwap_monitor.json"

# 임계값 (환경변수로 오버라이드 가능)
DEFAULT_OVERHEAT_PCT = float(os.getenv("VWAP_OVERHEAT_BLOCK_PCT", "2.0"))
DEFAULT_DIP_PCT = float(os.getenv("VWAP_DIP_BONUS_PCT", "-1.5"))


def _load_vwap_state() -> dict:
    """vwap_monitor.json 로드. 없거나 파싱 실패 시 빈 dict."""
    if not VWAP_STATE_PATH.exists():
        logger.warning("vwap_monitor.json 없음 — fail-open (게이트 통과)")
        return {}
    try:
        return json.loads(VWAP_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("vwap_monitor.json 로드 실패: %s — fail-open", e)
        return {}


def check_vwap_buy_gate(
    ticker: str,
    overheat_threshold: float = DEFAULT_OVERHEAT_PCT,
    dip_threshold: float = DEFAULT_DIP_PCT,
) -> dict:
    """매수 직전 VWAP 게이트 검사.

    Args:
        ticker: 종목코드 (6자리)
        overheat_threshold: 매수 차단 임계 (VWAP 대비 %, 기본 +2.0%)
        dip_threshold: 매수 우대 임계 (VWAP 대비 %, 기본 -1.5%)

    Returns:
        {
            "allow": bool,           # True=매수 허용, False=차단
            "reason": str,           # 사유 ('OVERHEAT_BLOCK' / 'DIP_BONUS' / 'NORMAL' / 'DATA_MISSING')
            "vwap_dev_pct": float,   # VWAP 대비 편차 (%)
            "vwap": float | None,    # VWAP 절대값
            "current_price": int | None,
            "is_dip": bool,          # 눌림목 우대 대상 여부
        }
    """
    state = _load_vwap_state()
    stocks = state.get("stocks", {})

    # ticker 정규화 (6자리)
    tkey = str(ticker).zfill(6)
    info = stocks.get(tkey) or stocks.get(ticker)

    if not info:
        # 데이터 미수신 — fail-open + warning (큐 신규 등록 종목 보호)
        logger.warning(
            "[VWAP gate] %s vwap_monitor.json 미수신 — fail-open (allow=True)",
            ticker,
        )
        return {
            "allow": True,
            "reason": "DATA_MISSING",
            "vwap_dev_pct": 0.0,
            "vwap": None,
            "current_price": None,
            "is_dip": False,
        }

    dev = float(info.get("vwap_dev_pct", 0))
    vwap = info.get("vwap")
    cur = info.get("current_price")

    # OVERHEAT: VWAP 대비 +X% 이상 = 오버슈팅 = 매수 차단
    if dev >= overheat_threshold:
        logger.info(
            "[VWAP gate] %s 차단 (OVERHEAT): dev=%+.2f%% >= %+.2f%%",
            ticker, dev, overheat_threshold,
        )
        return {
            "allow": False,
            "reason": "OVERHEAT_BLOCK",
            "vwap_dev_pct": dev,
            "vwap": vwap,
            "current_price": cur,
            "is_dip": False,
        }

    # DIP: VWAP 대비 -X% 이하 = 눌림목 = 매수 우대
    if dev <= dip_threshold:
        logger.info(
            "[VWAP gate] %s 통과 + 눌림목 우대 (DIP_BONUS): dev=%+.2f%% <= %+.2f%%",
            ticker, dev, dip_threshold,
        )
        return {
            "allow": True,
            "reason": "DIP_BONUS",
            "vwap_dev_pct": dev,
            "vwap": vwap,
            "current_price": cur,
            "is_dip": True,
        }

    # NORMAL: 정상 범위
    return {
        "allow": True,
        "reason": "NORMAL",
        "vwap_dev_pct": dev,
        "vwap": vwap,
        "current_price": cur,
        "is_dip": False,
    }


def format_gate_result(result: dict, ticker: str) -> str:
    """텔레그램/로그용 게이트 결과 포맷."""
    r = result.get("reason", "?")
    dev = result.get("vwap_dev_pct", 0)
    vwap = result.get("vwap")
    cur = result.get("current_price")

    icon = {
        "OVERHEAT_BLOCK": "🛑",
        "DIP_BONUS": "💚",
        "NORMAL": "✓",
        "DATA_MISSING": "⚠️",
    }.get(r, "?")

    parts = [f"{icon} VWAP {r}", f"{ticker} dev {dev:+.2f}%"]
    if vwap and cur:
        parts.append(f"현재 {cur:,} / VWAP {vwap:,.0f}")
    return " | ".join(parts)
