"""EDGAR 미국 SEC 공시 → 한국 수혜주 점수 매핑.

배경 (5/23 옵션 D):
  정보봇이 미국 SEC EDGAR 공시 추적 → Form 4 (임원 내부자 거래) /
  10-Q (실적) / 8-K (중대 사건) 등. 그러나 미국 ticker (NVDA/TSLA 등)만
  있어 한국 종목 점수 산출 시 매칭 불가.
  config/edgar_us_kr_mapping.yaml 사전으로 자동 변환.

5/22 인사이트 (퐝가님):
  "AMD Lisa Su $55.7M 매도 같은 내부자 거래는
   한국 메모리주에도 영향을 줄 수 있어"
  → Form 4 임원 대량 매수/매도가 한국 매핑 종목에 시그널 전파.

시그널 종류:
  - INSIDER_SELL_LARGE (Form 4 매도 $10M+): 약한 악재 → -1
  - INSIDER_SELL_MEGA  (Form 4 매도 $50M+): 명백 악재 → -2
  - INSIDER_BUY_LARGE  (Form 4 매수 $10M+): 약한 호재 → +1
  - INSIDER_BUY_MEGA   (Form 4 매수 $50M+): 명백 호재 → +2
  - EARNING_BEAT       (10-Q 실적 비트):     호재 → +1
  - EARNING_MISS       (10-Q 실적 미스):     악재 → -1
  - GUIDANCE_UP        (가이던스 상향):       호재 → +2
  - GUIDANCE_DOWN      (가이던스 하향):       악재 → -2

매핑 적용:
  미국 종목 시그널 → 한국 매핑 종목 (primary 1.0 / secondary 0.6 / tertiary 0.3)
  최종 점수 = base × multiplier × 신선도 → 정수 라운드, -3 ~ +3 클램프

사용:
  from src.use_cases.edgar_signal_scorer import calculate_edgar_score
  result = calculate_edgar_score("000660")  # SK하이닉스
  # NVDA INSIDER_BUY_MEGA → primary 1.0 → +2점
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAPPING_FILE = PROJECT_ROOT / "config" / "edgar_us_kr_mapping.yaml"

# 시그널 base 점수
EDGAR_BASE_SCORES = {
    "INSIDER_SELL_LARGE": -1,
    "INSIDER_SELL_MEGA": -2,
    "INSIDER_BUY_LARGE": +1,
    "INSIDER_BUY_MEGA": +2,
    "EARNING_BEAT": +1,
    "EARNING_MISS": -1,
    "GUIDANCE_UP": +2,
    "GUIDANCE_DOWN": -2,
}

# 매핑 강도
MAP_PRIMARY = 1.0
MAP_SECONDARY = 0.6
MAP_TERTIARY = 0.3

# 신선도 (시간)
FRESH_H = 48
STALE_H = 168  # 7일

# 금액 임계 (USD)
INSIDER_MEGA_USD = 50_000_000
INSIDER_LARGE_USD = 10_000_000


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    """edgar_us_kr_mapping.yaml 로드 (캐시)."""
    try:
        with open(MAPPING_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # _metadata 제외
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning("edgar_us_kr_mapping.yaml 로드 실패: %s", e)
        return {}


def _build_reverse_index() -> dict[str, list[dict]]:
    """한국 ticker → [{us_ticker, multiplier, reason}, ...] 역인덱스."""
    mapping = _load_mapping()
    reverse: dict[str, list[dict]] = {}
    tier_map = {
        "primary": MAP_PRIMARY,
        "secondary": MAP_SECONDARY,
        "tertiary": MAP_TERTIARY,
    }
    for us_ticker, info in mapping.items():
        for tier_name, multiplier in tier_map.items():
            for item in info.get(tier_name, []) or []:
                kr_ticker = str(item.get("ticker", "")).zfill(6)
                if not kr_ticker:
                    continue
                reverse.setdefault(kr_ticker, []).append({
                    "us_ticker": us_ticker,
                    "us_name": info.get("name", us_ticker),
                    "tier": tier_name,
                    "multiplier": multiplier,
                    "reason": item.get("reason", ""),
                })
    return reverse


@lru_cache(maxsize=1)
def _reverse_index_cached() -> dict[str, list[dict]]:
    return _build_reverse_index()


def get_us_mapping_for_kr(ticker: str) -> list[dict]:
    """한국 ticker에 매핑된 미국 종목 리스트."""
    return _reverse_index_cached().get(str(ticker).zfill(6), [])


def classify_insider_signal(amount_usd: float, action: str) -> str:
    """Form 4 매수/매도 금액 → 시그널 분류."""
    is_buy = action.upper() in ("BUY", "P", "PURCHASE")
    if amount_usd >= INSIDER_MEGA_USD:
        return "INSIDER_BUY_MEGA" if is_buy else "INSIDER_SELL_MEGA"
    if amount_usd >= INSIDER_LARGE_USD:
        return "INSIDER_BUY_LARGE" if is_buy else "INSIDER_SELL_LARGE"
    return ""


def _freshness_weight(filed_at: Optional[str]) -> float:
    if not filed_at:
        return 1.0
    try:
        if isinstance(filed_at, str):
            ts = datetime.fromisoformat(filed_at.replace("Z", "+00:00"))
        else:
            ts = filed_at
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        elapsed_h = (datetime.now() - ts).total_seconds() / 3600
        if elapsed_h < 0:
            return 1.0
        if elapsed_h <= FRESH_H:
            return 1.0
        if elapsed_h <= STALE_H:
            return 0.5
        return 0.0
    except Exception:
        return 1.0


def _fetch_edgar_signals(us_tickers: list[str], days: int = 7) -> list[dict]:
    """정보봇 intelligence_edgar 또는 intelligence_disclosures(source='EDGAR') 조회.

    스키마 가정 (정보봇 표준):
        us_ticker / signal_type / amount_usd / action / filed_at / form_type /
        insider_name / insider_title / company_name / source
    """
    if not us_tickers:
        return []
    try:
        from src.adapters.quant_supabase_reader import _get_client
        client = _get_client()
        if not client:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        # intelligence_edgar 테이블 우선, 없으면 intelligence_disclosures source=EDGAR
        try:
            res = (
                client.table("intelligence_edgar")
                .select(
                    "us_ticker,signal_type,amount_usd,action,filed_at,form_type,"
                    "insider_name,insider_title,company_name"
                )
                .in_("us_ticker", us_tickers)
                .gte("filed_at", cutoff)
                .order("filed_at", desc=True)
                .limit(20)
                .execute()
            )
            return res.data or []
        except Exception:
            # fallback: intelligence_disclosures source=EDGAR
            res = (
                client.table("intelligence_disclosures")
                .select("ticker,signal_type,sentiment_score,tags,date,disclosed_at,severity")
                .in_("ticker", us_tickers)
                .gte("date", cutoff)
                .order("date", desc=True)
                .limit(20)
                .execute()
            )
            return res.data or []
    except Exception as e:
        logger.debug("EDGAR 조회 실패: %s", e)
        return []


def calculate_edgar_score(ticker: str) -> dict[str, Any]:
    """EDGAR 시그널 → 한국 종목 점수 산정.

    Returns:
        {
            "score": int,           # -3 ~ +3
            "n_signals": int,
            "matched_us": list[str],
            "breakdown": list[str],
            "reason": str,
        }
    """
    kr_ticker = str(ticker).zfill(6)
    mappings = get_us_mapping_for_kr(kr_ticker)
    if not mappings:
        return {
            "score": 0,
            "n_signals": 0,
            "matched_us": [],
            "breakdown": [],
            "reason": "EDGAR 매핑 없음 (미국 빅테크 무관 종목)",
        }

    us_tickers = list({m["us_ticker"] for m in mappings})
    signals = _fetch_edgar_signals(us_tickers)
    if not signals:
        return {
            "score": 0,
            "n_signals": 0,
            "matched_us": us_tickers,
            "breakdown": [],
            "reason": f"매핑 US ticker {len(us_tickers)}건 — EDGAR 시그널 없음 (7일)",
        }

    # multiplier lookup: us_ticker → 가장 강한 매칭
    mult_by_us = {}
    for m in mappings:
        prev = mult_by_us.get(m["us_ticker"], 0)
        if m["multiplier"] > prev:
            mult_by_us[m["us_ticker"]] = m["multiplier"]

    total = 0.0
    breakdown = []
    for sig in signals:
        # us_ticker / signal_type 추출 (테이블별 컬럼명 다를 수 있음)
        us_ticker = sig.get("us_ticker") or sig.get("ticker") or ""
        if us_ticker not in mult_by_us:
            continue
        signal_type = (sig.get("signal_type") or "").upper()
        # Form 4 자동 분류 (amount_usd + action 있으면)
        if not signal_type and sig.get("amount_usd") and sig.get("action"):
            signal_type = classify_insider_signal(
                float(sig["amount_usd"]), str(sig["action"])
            )
        base = EDGAR_BASE_SCORES.get(signal_type, 0)
        if base == 0:
            continue
        mult = mult_by_us[us_ticker]
        fresh = _freshness_weight(sig.get("filed_at") or sig.get("date"))
        contrib = base * mult * fresh
        total += contrib
        breakdown.append(
            f"{us_ticker} {signal_type} × {mult:.1f} × fresh {fresh:.1f} = {contrib:+.1f}"
        )

    final = max(-3, min(3, int(round(total))))
    reason = (
        f"EDGAR {len(breakdown)}/{len(signals)} 시그널 영향 → 합계 {total:+.1f}, 클램프 {final:+d}"
        if breakdown
        else f"매핑 US {len(us_tickers)}건 시그널 {len(signals)} 모두 중립"
    )

    return {
        "score": final,
        "n_signals": len(signals),
        "n_effective": len(breakdown),
        "matched_us": us_tickers,
        "breakdown": breakdown,
        "reason": reason,
    }
