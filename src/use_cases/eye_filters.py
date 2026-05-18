"""사장님 EYE 사전 필터 4종 (2026-05-18 신규)

배경: 5/18 사장님 통찰 "수익이 아니라 손실 종목 회피 EYE가 본질"
- 자비스 강력포착 9건 중 -6% 손실 3종목 (한화시스템/삼화콘덴서/인벤티지랩)
- 모두 추천 시점에 KIS API로 사전 감지 가능했던 위험 시그널 보유

필터 4종 (5/18 손실 3종목 100% 회피 검증 목표):
  ① is_long_term_weak() — 52주 고가 대비 -20% 이상 약세 (한화시스템 -50%, 인벤티지랩 -50%)
  ② is_program_selling() — 프로그램 순매수 < 0 (삼화콘덴서 -9,877주)
  ③ is_low_volume() — 거래량 비율 < 30% (거래 부진)
  ④ is_low_buy_ratio() — intraday_minute 매수비율 < 45% (매도 우세, 25 구독 한정)

종합 함수:
  should_skip(ticker) — 4종 중 하나라도 True → 추천 회피

사용:
  from src.use_cases.eye_filters import should_skip, evaluate_filters
  if should_skip(broker, '095270'):
      continue  # 추천 제외
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTRADAY_DB_PATTERN = "data/intraday/intraday_minute_{date}.db"

# 임계값 (5/18 학습 결과 기반, 5/22 캘리브레이션 후 정밀화 예정)
THRESHOLD_W52_DIST_PCT = -20.0       # 52주 고가 대비 -20% 이상 약세
THRESHOLD_PROGRAM_NTBY = 0            # 프로그램 순매수 0 미만 (매도)
THRESHOLD_VOL_RATIO_PCT = 30.0        # 거래량 비율 30% 미만
THRESHOLD_BUY_RATIO_PCT = 45.0        # 매수비율 45% 미만


def fetch_price_safe(broker, ticker: str) -> Optional[dict]:
    """KIS fetch_price 안전 호출."""
    try:
        resp = broker.fetch_price(ticker)
        return resp.get("output", {}) if resp else None
    except Exception as e:
        logger.warning("fetch_price 실패 %s: %s", ticker, e)
        return None


# ────────────────────────────────────────────────────────────
# 필터 ① 52주 고가 대비 장기 약세
# ────────────────────────────────────────────────────────────


def is_long_term_weak(broker, ticker: str, threshold_pct: float = THRESHOLD_W52_DIST_PCT) -> tuple[bool, dict]:
    """52주 고가 대비 임계% 이상 약세 = 장기 추세 약함 → 회피.

    Returns:
        (회피여부, 상세 dict)
    """
    out = fetch_price_safe(broker, ticker)
    if not out:
        return False, {"error": "fetch_price 실패"}

    w52_dist = float(out.get("w52_hgpr_vrss_prpr_ctrt", 0) or 0)
    is_weak = w52_dist <= threshold_pct  # -50% <= -20% (약세)
    return is_weak, {
        "filter": "long_term_weak",
        "w52_dist_pct": w52_dist,
        "threshold": threshold_pct,
        "verdict": "SKIP" if is_weak else "PASS",
    }


# ────────────────────────────────────────────────────────────
# 필터 ② 프로그램 순매수 음수 (매도 우세)
# ────────────────────────────────────────────────────────────


def is_program_selling(broker, ticker: str, threshold: int = THRESHOLD_PROGRAM_NTBY) -> tuple[bool, dict]:
    """프로그램 순매수 < 임계 AND 가격 음봉 → 회피.

    5/18 라이브 검증에서 발견 (사장님 통찰):
    - 프로그램 매도 + 가격 하락 = DB 반례 (진짜 위험) → 회피 ✅
    - 프로그램 매도 + 가격 상승 = 외인/기관/연기금 매수 압도 = 매수 가능 (삼성전자 +4.62% 케이스)

    따라서 단순 프로그램 매도 X, AND 음봉 조건으로 진짜 위험만 회피.
    """
    out = fetch_price_safe(broker, ticker)
    if not out:
        return False, {"error": "fetch_price 실패"}

    pgtr = int(out.get("pgtr_ntby_qty", 0) or 0)
    prdy_ctrt = float(out.get("prdy_ctrt", 0) or 0)

    is_program_sell = pgtr < threshold
    is_price_down = prdy_ctrt < 0
    is_dangerous = is_program_sell and is_price_down  # AND 조건

    return is_dangerous, {
        "filter": "program_selling",
        "pgtr_ntby_qty": pgtr,
        "prdy_ctrt": prdy_ctrt,
        "is_program_sell": is_program_sell,
        "is_price_down": is_price_down,
        "verdict": "SKIP" if is_dangerous else "PASS",
        "note": "프로그램 매도 + 음봉 동시 조건 (5/18 사장님 통찰 반영)",
    }


# ────────────────────────────────────────────────────────────
# 필터 ③ 거래량 비율 부진
# ────────────────────────────────────────────────────────────


def is_low_volume(broker, ticker: str, threshold_pct: float = THRESHOLD_VOL_RATIO_PCT) -> tuple[bool, dict]:
    """전일 대비 거래량 비율 < 임계% → 거래 부진 → 회피.

    Note: prdy_vrss_vol_rate는 KIS 응답에서 "전일 대비 거래량 비율 %" 형태.
    예: 42.88 = 전일 대비 42.88% (즉 절반 이하 거래량).
    """
    out = fetch_price_safe(broker, ticker)
    if not out:
        return False, {"error": "fetch_price 실패"}

    vol_rate = float(out.get("prdy_vrss_vol_rate", 0) or 0)
    is_low = vol_rate < threshold_pct
    return is_low, {
        "filter": "low_volume",
        "vol_ratio_pct": vol_rate,
        "threshold": threshold_pct,
        "verdict": "SKIP" if is_low else "PASS",
    }


# ────────────────────────────────────────────────────────────
# 필터 ④ 매수비율 부진 (intraday_minute DB 한정)
# ────────────────────────────────────────────────────────────


def is_low_buy_ratio(
    ticker: str,
    date: str,
    threshold_pct: float = THRESHOLD_BUY_RATIO_PCT,
) -> tuple[bool, dict]:
    """intraday_minute DB의 buy_count/(buy_count+sell_count) < 임계% → 매도 우세 → 회피.

    Note: 25 구독 종목 한정. 미구독 종목은 (False, no_data) 반환 (필터 통과로 간주).
    5/19 작업: 구독 25 → 50 확장 후 강력포착 종목도 적용 가능.
    """
    db_path = PROJECT_ROOT / INTRADAY_DB_PATTERN.format(date=date.replace("-", ""))
    if not db_path.exists():
        return False, {"filter": "low_buy_ratio", "error": "DB 없음", "verdict": "PASS_NO_DATA"}

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(
            "SELECT SUM(buy_count), SUM(sell_count) FROM intraday_minute WHERE code=?",
            (ticker,),
        )
        row = cur.fetchone()
        con.close()

        if not row or row[0] is None:
            return False, {
                "filter": "low_buy_ratio",
                "ticker": ticker,
                "verdict": "PASS_NO_DATA",
                "note": "intraday_learner 미구독 (5/19 구독 확장 후 적용)",
            }

        buy, sell = row[0], row[1]
        total = buy + sell
        if total == 0:
            return False, {"filter": "low_buy_ratio", "verdict": "PASS_NO_DATA"}

        buy_ratio = buy / total * 100
        is_low = buy_ratio < threshold_pct
        return is_low, {
            "filter": "low_buy_ratio",
            "ticker": ticker,
            "buy_count": buy,
            "sell_count": sell,
            "buy_ratio_pct": round(buy_ratio, 1),
            "threshold": threshold_pct,
            "verdict": "SKIP" if is_low else "PASS",
        }
    except Exception as e:
        logger.warning("is_low_buy_ratio 실패 %s: %s", ticker, e)
        return False, {"filter": "low_buy_ratio", "error": str(e), "verdict": "PASS_ERROR"}


# ────────────────────────────────────────────────────────────
# 종합 함수
# ────────────────────────────────────────────────────────────


def evaluate_filters(broker, ticker: str, date: str, include_dart: bool = True) -> dict:
    """4종 (또는 5종) 필터 모두 평가 → 종합 결과 dict 반환.

    Args:
        broker, ticker, date
        include_dart: True면 DART EYE ⑤ 포함 (5/18 막내 단축 후 가동)

    Returns:
        {
            "ticker": str,
            "should_skip": bool,
            "skip_reasons": list[str],
            "filters": {
                "long_term_weak": {...},
                "program_selling": {...},
                "low_volume": {...},
                "low_buy_ratio": {...},
                "dart_negative": {...}  # 5/18 추가
            }
        }
    """
    f1 = is_long_term_weak(broker, ticker)
    f2 = is_program_selling(broker, ticker)
    f3 = is_low_volume(broker, ticker)
    f4 = is_low_buy_ratio(ticker, date)

    # 필터 ⑤ DART (5/18 추가, 막내 단축 결과)
    f5 = (False, {"filter": "dart_negative", "verdict": "SKIPPED"})
    if include_dart:
        try:
            from src.use_cases.dart_eye_filter import has_dart_negative
            f5 = has_dart_negative(ticker)
        except Exception as e:
            logger.debug("DART 필터 실패: %s", e)

    skip_reasons = []
    if f1[0]:
        skip_reasons.append(f"장기약세(w52 {f1[1]['w52_dist_pct']:.1f}%)")
    if f2[0]:
        skip_reasons.append(f"프로그램매도({f2[1]['pgtr_ntby_qty']:+,})")
    if f3[0]:
        skip_reasons.append(f"거래부진({f3[1]['vol_ratio_pct']:.1f}%)")
    if f4[0]:
        skip_reasons.append(f"매도우세({f4[1].get('buy_ratio_pct', 'N/A')}%)")
    if f5[0]:
        skip_reasons.append(f"DART악재({f5[1].get('worst_score', 'N/A')}점)")

    should_skip = bool(skip_reasons)

    return {
        "ticker": ticker,
        "should_skip": should_skip,
        "skip_reasons": skip_reasons,
        "filters": {
            "long_term_weak": f1[1],
            "program_selling": f2[1],
            "low_volume": f3[1],
            "low_buy_ratio": f4[1],
            "dart_negative": f5[1],  # 5/18 추가
        },
    }


def should_skip(broker, ticker: str, date: str) -> bool:
    """간단 boolean 인터페이스 — 4종 중 하나라도 SKIP → True."""
    return evaluate_filters(broker, ticker, date)["should_skip"]
