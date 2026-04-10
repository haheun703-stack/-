#!/usr/bin/env python
"""노다지 리포트 — 장기 가치투자 종목 발굴 스크리너

6개월~1년 관점의 가치투자 종목을 발굴한다.
핵심 철학: "과하게 빠진 펀더멘털 건전 종목" = 노다지

스코어링 (100점, 5축):
  Value      (30): ValueComposite(PER+EBITDA/EV+FCF+내재가치) — 레짐별 가중치
  Quality    (25): QualityComposite(ROE+부채+이익품질+배당) — 레짐별 가중치
  Earnings   (20): 분기 실적 방향성 (ACCELERATING→DETERIORATING)
  Drawdown   (15): 52주 고점 대비 낙폭 (크게 빠질수록 고점수)
  PeerValue  (10): 동종업종 PER 대비 할인율

등급:
  GOLD   (75+): 장기 핵심 매수 후보
  SILVER (60+): 관심 + 분할매수 후보
  BRONZE (45+): 워치리스트 편입

필터 (사전):
  - 시가총액 2,000억+ (universe.csv 기준)
  - 매출 500억+ (DART 캐시)
  - 영업이익 흑자 (최근 기준)
  - PER > 0 (적자주 배제)

스케줄: 주 2회 (수/토) BAT-D 이후 실행
출력: data/nugget_report.json → FLOWX 업로드 + 텔레그램 알림

Usage:
    python -u -X utf8 scripts/scan_nugget.py              # 기본 모드
    python -u -X utf8 scripts/scan_nugget.py --telegram    # 텔레그램 알림
    python -u -X utf8 scripts/scan_nugget.py --dry-run     # 업로드 없이 출력만
    python -u -X utf8 scripts/scan_nugget.py --top 30      # 상위 N종목
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "nugget_report.json"
UNIVERSE_PATH = DATA_DIR / "universe.csv"

# ── 스코어링 가중치 ─────────────────────────────
WEIGHTS = {
    "value": 30,
    "quality": 25,
    "earnings": 20,
    "drawdown": 15,
    "peer_value": 10,
}

# ── 등급 커트라인 ─────────────────────────────────
GRADE_CUTOFFS = {
    "GOLD": 75,
    "SILVER": 60,
    "BRONZE": 45,
}

# ── 필터 기준 ─────────────────────────────────────
MIN_MARKET_CAP_억 = 2000      # 시총 2,000억+
MIN_REVENUE_억 = 500           # 매출 500억+
MIN_PER = 0                    # PER > 0 (적자 배제)
MAX_PER = 200                  # 극단 PER 배제

# ── Drawdown → Score 매핑 ─────────────────────────
# 낙폭이 클수록 높은 점수 (역전 사고)
# -50% 이상 → 15점, -30% → 12점, -15% → 8점, -5% → 3점, 0~→ 0점
def _drawdown_to_score(dd_pct: float) -> float:
    """52주 낙폭(%) → 0~15 점수. 음수가 클수록 높은 점수."""
    dd = abs(dd_pct)  # dd_pct는 음수 (-30 → 30)
    if dd >= 50:
        return 15.0
    elif dd >= 40:
        return 14.0
    elif dd >= 30:
        return 12.0
    elif dd >= 25:
        return 10.0
    elif dd >= 20:
        return 8.0
    elif dd >= 15:
        return 6.0
    elif dd >= 10:
        return 4.0
    elif dd >= 5:
        return 2.0
    else:
        return 0.0


# ── Peer Valuation Score ──────────────────────────
def _peer_discount_to_score(discount_pct: float) -> float:
    """동종 PER 대비 할인율(%) → 0~10 점수."""
    if discount_pct >= 60:
        return 10.0
    elif discount_pct >= 40:
        return 8.0
    elif discount_pct >= 25:
        return 6.0
    elif discount_pct >= 10:
        return 4.0
    elif discount_pct >= 0:
        return 2.0
    else:
        return 0.0   # 프리미엄(할인 음수) → 0점


# ═══════════════════════════════════════════════════
# 메인 스캔 로직
# ═══════════════════════════════════════════════════

def load_universe(min_cap_억: float = MIN_MARKET_CAP_억) -> pd.DataFrame:
    """universe.csv 로드 + 시총 필터."""
    if not UNIVERSE_PATH.exists():
        logger.error("universe.csv 없음: %s", UNIVERSE_PATH)
        return pd.DataFrame()

    df = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)
    # market_cap은 원 단위 → 억원 변환
    df["market_cap_억"] = df["market_cap"] / 1e8
    df = df[df["market_cap_억"] >= min_cap_억].copy()
    logger.info("유니버스 로드: %d종목 (시총 %.0f억+)", len(df), min_cap_억)
    return df


def fetch_pykrx_fundamentals() -> dict[str, dict]:
    """pykrx에서 당일 전종목 PER/PBR/배당 조회.

    Returns:
        {ticker: {"PER": float, "PBR": float, "DIV": float, "DPS": int}}
    """
    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        logger.warning("pykrx 미설치 — PER/PBR 조회 불가")
        return {}

    from datetime import timedelta

    # 최근 7일 이내 거래일 시도 (장마감 후/KRX API 불안정 대응)
    df = pd.DataFrame()
    for days_back in range(0, 8):
        date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            df = pykrx_stock.get_market_fundamental_by_ticker(date_str, market="ALL")
            if not df.empty and "PER" in df.columns:
                logger.info("pykrx 조회 성공: %s (%d종목)", date_str, len(df))
                break
        except Exception:
            continue

    if df.empty:
        logger.warning("pykrx PER/PBR 조회 실패 (7일 모두)")
        return {}

    result = {}
    for ticker, row in df.iterrows():
        result[ticker] = {
            "PER": float(row.get("PER", 0)),
            "PBR": float(row.get("PBR", 0)),
            "DIV": float(row.get("DIV", 0)),
            "DPS": int(row.get("DPS", 0)),
            "EPS": int(row.get("EPS", 0)),
            "BPS": int(row.get("BPS", 0)),
        }
    logger.info("pykrx 펀더멘탈 로드: %d종목", len(result))
    return result


def calc_per_from_dart_eps() -> dict[str, dict]:
    """DART EPS + parquet 종가로 PER 직접 계산 (pykrx 폴백용).

    PER = 현재 종가 / EPS (TTM)
    """
    # DART fundamentals_all.csv에서 EPS 로드
    dart_path = DATA_DIR / "dart_cache" / "fundamentals_all.csv"
    if not dart_path.exists():
        logger.warning("fundamentals_all.csv 없음 — PER 폴백 불가")
        return {}

    dart_df = pd.read_csv(dart_path, dtype={"ticker": str})
    dart_df["ticker"] = dart_df["ticker"].str.zfill(6)
    eps_map = {}
    for _, row in dart_df.iterrows():
        eps = row.get("eps")
        if pd.notna(eps) and float(eps) > 0:
            eps_map[row["ticker"]] = float(eps)

    # parquet에서 종가 로드 → PER 계산
    processed_dir = DATA_DIR / "processed"
    result = {}

    for ticker, eps in eps_map.items():
        pq_path = processed_dir / f"{ticker}.parquet"
        if not pq_path.exists():
            continue
        try:
            df = pd.read_parquet(pq_path, columns=["close"])
            if df.empty:
                continue
            close = float(df["close"].iloc[-1])
            if close <= 0 or eps <= 0:
                continue
            per = round(close / eps, 2)
            if per > 0:
                result[ticker] = {
                    "PER": per, "PBR": 0, "DIV": 0,
                    "DPS": 0, "EPS": int(eps), "BPS": 0,
                }
        except Exception:
            continue

    if result:
        logger.info("DART EPS 기반 PER 계산: %d종목", len(result))
    return result


def calc_drawdown(ticker: str) -> dict | None:
    """52주 고점 대비 낙폭 + 현재가 계산.

    processed parquet 또는 stock_data_daily CSV에서 종가 데이터를 읽는다.
    """
    # 1순위: processed parquet
    parquet_path = DATA_DIR / "processed" / f"{ticker}.parquet"
    csv_path = PROJECT_ROOT / "stock_data_daily" / f"{ticker}.csv"

    close_series = None

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if "close" in df.columns and len(df) >= 60:
                close_series = df["close"].tail(252)
        except Exception:
            pass

    if close_series is None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            col = "close" if "close" in df.columns else "종가"
            if col in df.columns and len(df) >= 60:
                close_series = pd.to_numeric(df[col], errors="coerce").tail(252)
        except Exception:
            pass

    if close_series is None or len(close_series) < 60:
        return None

    close_series = close_series.dropna()
    if len(close_series) < 60:
        return None

    current_close = float(close_series.iloc[-1])
    high_252 = float(close_series.max())
    low_252 = float(close_series.min())

    if high_252 <= 0:
        return None

    drawdown_pct = ((current_close / high_252) - 1) * 100  # 음수

    return {
        "close": current_close,
        "high_252": high_252,
        "low_252": low_252,
        "drawdown_pct": round(drawdown_pct, 1),
    }


PROCESSED_DIR = DATA_DIR / "processed"
_FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


def calc_smart_prices(
    ticker: str,
    close: float,
    high_252: float,
    low_252: float,
    eps: float,
    sector_avg_per: float,
    intrinsic_engine=None,
) -> dict:
    """역산 가격 엔진 — 하방 치밀 + 상방 열림.

    상방(Target): max(내재가치, 섹터PER, 52주고점, 수급부스트) — 열린 목표
    하방(Stop):   max(ATR×2, 52주 저점) — 치밀한 손절
    분할매수 구간: 피보나치 0.382 / 0.618 / 현재가

    Returns:
        entry_price, stop_loss, target_price, target_levels (분할매도용),
        methods, risk_pct, reward_pct, rr_ratio
    """
    methods = {}
    targets = []    # (가격, 라벨) 튜플로 분할매도 구간 기록
    stops = []

    # ── (1) DCF/RIM 내재가치 → Target ──
    if intrinsic_engine is not None:
        try:
            iv = intrinsic_engine.score_raw(ticker)
            if iv and iv.get("fair_value"):
                fair = iv["fair_value"]
                if fair > close:
                    targets.append((fair, "내재가치"))
                    methods["intrinsic"] = {
                        "fair_value": int(fair),
                        "dcf": int(iv.get("dcf_value") or 0),
                        "rim": int(iv.get("rim_value") or 0),
                        "upside_pct": round(iv["upside"] * 100, 1),
                    }
        except Exception:
            pass

    # ── (2) 섹터PER × EPS → Target ──
    if sector_avg_per > 0 and eps > 0:
        sector_fair = sector_avg_per * eps
        if sector_fair > close:
            targets.append((sector_fair, "섹터PER"))
            methods["sector_per"] = {
                "fair_value": int(sector_fair),
                "sector_avg_per": round(sector_avg_per, 1),
                "eps": int(eps),
                "upside_pct": round((sector_fair / close - 1) * 100, 1),
            }

    # ── (3) 52주 고점 회복 → Target ──
    if high_252 > close:
        targets.append((high_252, "52주고점"))
        methods["recovery_52w"] = {
            "high_252": int(high_252),
            "recovery_pct": round((high_252 / close - 1) * 100, 1),
        }

    # ── (4) Parquet: 피보나치 + ATR + 수급 부스트 ──
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            if len(df) >= 60:
                # 피보나치 0.618 / 0.382 지지 (60일) → 분할매수 구간
                recent_60 = df.tail(60)
                fib_high = float(recent_60["high"].max())
                fib_low = float(recent_60["low"].min())
                if fib_high > fib_low:
                    fib_618 = fib_high - (fib_high - fib_low) * 0.618
                    fib_382 = fib_high - (fib_high - fib_low) * 0.382
                    methods["fibonacci"] = {
                        "fib_618": int(round(fib_618)),
                        "fib_382": int(round(fib_382)),
                        "60d_high": int(fib_high),
                        "60d_low": int(fib_low),
                    }

                # ATR(20) → Stop Loss
                if len(df) >= 21:
                    recent = df.tail(21)
                    h, l, c = recent["high"], recent["low"], recent["close"]
                    tr = pd.concat([
                        h - l,
                        (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.iloc[1:].mean())
                    atr_stop = close - 2.0 * atr
                    if atr_stop > 0:
                        stops.append(atr_stop)
                        methods["atr"] = {
                            "atr_20": int(round(atr)),
                            "multiplier": 2.0,
                            "atr_stop": int(round(atr_stop)),
                        }

                # 수급 모멘텀 → Target 부스트 (20일 누적)
                last_row = df.iloc[-1]
                fgn_20d = float(last_row.get("foreign_net_20d", 0) or 0)
                inst_20d = float(last_row.get("inst_net_20d", 0) or 0)

                if fgn_20d > 0 and inst_20d > 0:
                    # 쌍끌이 매수 → 52주 고점 × 1.2 오버슈트
                    overshoot = high_252 * 1.20
                    targets.append((overshoot, "수급오버슈트"))
                    methods["supply_boost"] = {
                        "foreign_20d_억": round(fgn_20d / 1e8, 1),
                        "inst_20d_억": round(inst_20d / 1e8, 1),
                        "type": "DUAL_INFLOW",
                        "overshoot_target": int(overshoot),
                    }
                elif fgn_20d > 0 or inst_20d > 0:
                    # 단일 주체 매수 → 52주 고점 × 1.1
                    overshoot = high_252 * 1.10
                    targets.append((overshoot, "수급단일"))
                    buyer = "외인" if fgn_20d > inst_20d else "기관"
                    methods["supply_boost"] = {
                        "foreign_20d_억": round(fgn_20d / 1e8, 1),
                        "inst_20d_억": round(inst_20d / 1e8, 1),
                        "type": f"SINGLE_{buyer}",
                        "overshoot_target": int(overshoot),
                    }
        except Exception:
            pass

    # ── 52주 저점 → Stop 참조 ──
    if low_252 > 0:
        stops.append(low_252)

    # ══ 최종 가격 결정 ══

    # Target: 상방 열림 — max (폴백 +30%)
    if targets:
        targets_sorted = sorted(targets, key=lambda x: x[0])
        target_price = int(round(targets_sorted[-1][0], -1))  # max
    else:
        targets_sorted = []
        target_price = int(round(close * 1.30, -1))
        methods["target_fallback"] = "+30%"

    # 분할매도 구간 (target_levels): 각 방법론의 가격을 오름차순
    target_levels = []
    for price, label in targets_sorted:
        target_levels.append({"price": int(round(price, -1)), "label": label})

    # Entry: 현재 종가
    entry_price = int(close)

    # Stop: 하방 치밀 — max(타이트) (폴백 -15%)
    if stops:
        stop_loss = int(round(max(stops), -1))
    else:
        stop_loss = int(round(close * 0.85, -1))
        methods["stop_fallback"] = "-15%"

    # ── 안전장치 ──
    if stop_loss >= entry_price:
        stop_loss = int(round(entry_price * 0.85, -1))
    if target_price <= entry_price:
        target_price = int(round(entry_price * 1.20, -1))

    risk = round((entry_price - stop_loss) / entry_price * 100, 1) if entry_price > 0 else 0
    reward = round((target_price - entry_price) / entry_price * 100, 1) if entry_price > 0 else 0

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "target_levels": target_levels,
        "methods": methods,
        "risk_pct": risk,
        "reward_pct": reward,
        "rr_ratio": round(reward / risk, 2) if risk > 0 else 0,
    }


def get_regime() -> str:
    """현재 Brain 레짐 조회."""
    brain_path = DATA_DIR / "brain_decision.json"
    if brain_path.exists():
        try:
            with open(brain_path, encoding="utf-8") as f:
                data = json.load(f)
            regime = data.get("regime", "CAUTION")
            logger.info("현재 레짐: %s", regime)
            return regime
        except Exception:
            pass
    return "CAUTION"


def scan_nuggets(top_n: int = 20) -> list[dict]:
    """노다지 스크리닝 메인 로직."""
    start_time = time.time()

    # 1. 유니버스 로드
    universe = load_universe()
    if universe.empty:
        return []

    # 2. pykrx 펀더멘탈 로드 (실패 시 parquet 폴백)
    pykrx_data = fetch_pykrx_fundamentals()
    if not pykrx_data:
        logger.info("pykrx 실패 → DART EPS 기반 PER 계산 폴백")
        pykrx_data = calc_per_from_dart_eps()

    # 3. DART 재무 데이터 (FundamentalEngine)
    from src.fundamental import FundamentalEngine
    fund_engine = FundamentalEngine()

    # 4. Value / Quality Composite (레짐별)
    from src.alpha.factors.value_composite import ValueComposite
    from src.alpha.factors.quality_composite import QualityComposite

    regime = get_regime()
    value_comp = ValueComposite()
    quality_comp = QualityComposite()

    # 유니버스 스코어 사전 계산 (한 번만)
    try:
        v_scores = value_comp.score_universe(regime)
    except Exception as e:
        logger.warning("ValueComposite 스코어 실패: %s", e)
        v_scores = {}

    try:
        q_scores = quality_comp.score_universe(regime)
    except Exception as e:
        logger.warning("QualityComposite 스코어 실패: %s", e)
        q_scores = {}

    logger.info("Value 스코어: %d종목, Quality 스코어: %d종목", len(v_scores), len(q_scores))

    # 5. 내재가치 엔진 (역산 가격용)
    intrinsic_engine = None
    try:
        from src.alpha.factors.value_intrinsic import ValueIntrinsic
        intrinsic_engine = ValueIntrinsic()
        logger.info("ValueIntrinsic 엔진 초기화 완료")
    except Exception as e:
        logger.warning("ValueIntrinsic 초기화 실패 (폴백 사용): %s", e)

    # 5. 종목별 스캔
    candidates = []
    filtered_counts = {"per_filter": 0, "revenue_filter": 0, "profit_filter": 0,
                       "no_data": 0, "passed": 0}

    for _, uni_row in universe.iterrows():
        ticker = uni_row["ticker"]
        name = uni_row.get("name", ticker)
        market_cap_억 = uni_row["market_cap_억"]

        # ── PER/PBR 필터 ──
        pykrx = pykrx_data.get(ticker, {})
        per = pykrx.get("PER", 0)
        pbr = pykrx.get("PBR", 0)
        div_yield = pykrx.get("DIV", 0)

        if per <= MIN_PER or per > MAX_PER:
            filtered_counts["per_filter"] += 1
            continue

        # ── 매출/흑자 필터 (DART 캐시) ──
        financials = fund_engine.get_financials(ticker)
        revenue = financials.get("revenue")
        op_income = financials.get("operating_income")
        op_margin = financials.get("operating_margin")

        if revenue is not None and revenue < MIN_REVENUE_억:
            filtered_counts["revenue_filter"] += 1
            continue

        if op_income is not None and op_income <= 0:
            filtered_counts["profit_filter"] += 1
            continue

        # ── 52주 낙폭 ──
        dd_data = calc_drawdown(ticker)
        if dd_data is None:
            filtered_counts["no_data"] += 1
            continue

        close = dd_data["close"]
        drawdown_pct = dd_data["drawdown_pct"]

        # ══ 5축 스코어링 ══

        # (1) Value Score (0~1 → 0~30)
        v_raw = v_scores.get(ticker, 0.5)
        value_score = round(v_raw * WEIGHTS["value"], 1)

        # (2) Quality Score (0~1 → 0~25)
        q_raw = q_scores.get(ticker, 0.5)
        quality_score = round(q_raw * WEIGHTS["quality"], 1)

        # (3) Earnings Momentum (0~20)
        earnings = fund_engine.calc_earnings_momentum(ticker)
        earnings_score = float(earnings.get("score", 0))
        earnings_verdict = earnings.get("verdict", "NO_DATA")
        earnings_detail = earnings.get("detail", "")

        # (4) Drawdown Opportunity (0~15)
        drawdown_score = _drawdown_to_score(drawdown_pct)

        # (5) Peer Valuation (0~10)
        sector_avg_per = fund_engine.get_sector_avg_per(ticker)
        if sector_avg_per > 0 and per > 0:
            peer_discount = ((sector_avg_per - per) / sector_avg_per) * 100
        else:
            peer_discount = 0
        peer_score = _peer_discount_to_score(peer_discount)

        # ── 총점 ──
        total = value_score + quality_score + earnings_score + drawdown_score + peer_score

        # ── 등급 ──
        if total >= GRADE_CUTOFFS["GOLD"]:
            grade = "GOLD"
        elif total >= GRADE_CUTOFFS["SILVER"]:
            grade = "SILVER"
        elif total >= GRADE_CUTOFFS["BRONZE"]:
            grade = "BRONZE"
        else:
            grade = "WATCH"  # 45점 미만

        # ── 역산 가격 엔진 (하방치밀 + 상방열림 + 수급부스트) ──
        eps_val = pykrx.get("EPS", 0)
        smart = calc_smart_prices(
            ticker=ticker,
            close=close,
            high_252=dd_data["high_252"],
            low_252=dd_data["low_252"],
            eps=eps_val,
            sector_avg_per=sector_avg_per,
            intrinsic_engine=intrinsic_engine,
        )
        entry_price = smart["entry_price"]
        stop_loss = smart["stop_loss"]
        target_price = smart["target_price"]

        candidates.append({
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "total_score": round(total, 1),
            "market_cap_억": round(market_cap_억, 0),
            "close": int(close),
            "per": round(per, 1),
            "pbr": round(pbr, 2),
            "div_yield": round(div_yield, 2),
            "revenue_억": round(revenue, 0) if revenue else None,
            "op_margin_pct": round(op_margin, 1) if op_margin else None,
            "drawdown_pct": drawdown_pct,
            "high_252": int(dd_data["high_252"]),
            "low_252": int(dd_data["low_252"]),
            "earnings_verdict": earnings_verdict,
            "earnings_detail": earnings_detail,
            "sector": fund_engine.get_sector(ticker),
            "sector_avg_per": round(sector_avg_per, 1),
            "peer_discount_pct": round(peer_discount, 1),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "risk_pct": smart["risk_pct"],
            "reward_pct": smart["reward_pct"],
            "rr_ratio": smart["rr_ratio"],
            "target_levels": smart["target_levels"],
            "price_methods": smart["methods"],
            # 개별 축 점수 (디버깅/표시용)
            "scores": {
                "value": value_score,
                "quality": quality_score,
                "earnings": earnings_score,
                "drawdown": drawdown_score,
                "peer_value": peer_score,
            },
            "regime": regime,
        })
        filtered_counts["passed"] += 1

    # 총점 내림차순 정렬
    candidates.sort(key=lambda x: x["total_score"], reverse=True)

    elapsed = time.time() - start_time
    logger.info(
        "노다지 스캔 완료: %.1f초 | 통과 %d / 유니버스 %d | "
        "PER차단=%d 매출차단=%d 적자차단=%d 데이터없음=%d",
        elapsed, filtered_counts["passed"], len(universe),
        filtered_counts["per_filter"], filtered_counts["revenue_filter"],
        filtered_counts["profit_filter"], filtered_counts["no_data"],
    )

    return candidates[:top_n]


# ═══════════════════════════════════════════════════
# 출력 + 저장
# ═══════════════════════════════════════════════════

def save_report(nuggets: list[dict], date_str: str = ""):
    """노다지 리포트 JSON 저장."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "regime": nuggets[0]["regime"] if nuggets else "CAUTION",
        "total_scanned": len(nuggets),
        "grade_summary": {
            "GOLD": sum(1 for n in nuggets if n["grade"] == "GOLD"),
            "SILVER": sum(1 for n in nuggets if n["grade"] == "SILVER"),
            "BRONZE": sum(1 for n in nuggets if n["grade"] == "BRONZE"),
            "WATCH": sum(1 for n in nuggets if n["grade"] == "WATCH"),
        },
        "nuggets": nuggets,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("노다지 리포트 저장: %s (%d종목)", OUTPUT_PATH, len(nuggets))
    return report


def print_report(nuggets: list[dict]):
    """콘솔 출력."""
    if not nuggets:
        print("\n[노다지] 스크리닝 결과 없음")
        return

    regime = nuggets[0].get("regime", "?")
    gold = [n for n in nuggets if n["grade"] == "GOLD"]
    silver = [n for n in nuggets if n["grade"] == "SILVER"]
    bronze = [n for n in nuggets if n["grade"] == "BRONZE"]

    print(f"\n{'='*70}")
    print(f"  노다지 리포트 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  레짐: {regime} | GOLD {len(gold)} | SILVER {len(silver)} | BRONZE {len(bronze)}")
    print(f"{'='*70}")

    for n in nuggets:
        grade_icon = {"GOLD": "🥇", "SILVER": "🥈", "BRONZE": "🥉"}.get(n["grade"], "  ")
        dd = n["drawdown_pct"]
        scores = n["scores"]
        print(
            f"  {grade_icon} {n['grade']:6s} {n['total_score']:5.1f}점 | "
            f"{n['name']:12s} ({n['ticker']}) | "
            f"PER {n['per']:5.1f} PBR {n['pbr']:4.2f} 배당 {n['div_yield']:.1f}% | "
            f"낙폭 {dd:+.1f}% | "
            f"{n['earnings_verdict']}"
        )
        print(
            f"         V={scores['value']:.0f} Q={scores['quality']:.0f} "
            f"E={scores['earnings']:.0f} D={scores['drawdown']:.0f} "
            f"P={scores['peer_value']:.0f} | "
            f"매출 {n['revenue_억'] or '?'}억 OPM {n['op_margin_pct'] or '?'}%"
        )
        # 역산 가격 정보
        methods = n.get("price_methods", {})
        method_names = [k for k in methods if k not in ("target_fallback", "stop_fallback")]
        rr = n.get("rr_ratio", 0)
        print(
            f"         진입 {n['entry_price']:,} → 목표 {n['target_price']:,} "
            f"(손절 {n['stop_loss']:,}) | "
            f"R:R {rr:.1f} | {','.join(method_names) or 'fallback'}"
        )
        # 분할매도 구간
        levels = n.get("target_levels", [])
        if len(levels) >= 2:
            lvl_str = " → ".join(f"{lv['price']:,}({lv['label']})" for lv in levels)
            print(f"         분할매도: {lvl_str}")

    print(f"{'='*70}")


def send_telegram(nuggets: list[dict]):
    """텔레그램 노다지 알림."""
    gold = [n for n in nuggets if n["grade"] == "GOLD"]
    silver = [n for n in nuggets if n["grade"] == "SILVER"]

    if not gold and not silver:
        logger.info("텔레그램: GOLD/SILVER 없음 — 알림 생략")
        return

    regime = nuggets[0].get("regime", "?") if nuggets else "?"

    lines = [
        f"⛏️ 노다지 리포트 ({datetime.now().strftime('%m/%d %H:%M')})",
        f"레짐: {regime} | GOLD {len(gold)} SILVER {len(silver)}",
        "",
    ]

    for n in gold + silver:
        emoji = "🥇" if n["grade"] == "GOLD" else "🥈"
        methods = n.get("price_methods", {})
        method_tags = []
        if "intrinsic" in methods:
            method_tags.append("DCF/RIM")
        if "sector_per" in methods:
            method_tags.append("섹터PER")
        if "recovery_52w" in methods:
            method_tags.append("52W")
        if "supply_boost" in methods:
            boost_type = methods["supply_boost"].get("type", "")
            method_tags.append("쌍끌이" if "DUAL" in boost_type else "수급")
        if "atr" in methods:
            method_tags.append("ATR")
        if "fibonacci" in methods:
            method_tags.append("피보")
        method_str = "+".join(method_tags) if method_tags else "기본"
        rr = n.get("rr_ratio", 0)

        lines.append(
            f"{emoji} {n['name']} {n['total_score']:.0f}점"
        )
        lines.append(
            f"  PER {n['per']:.1f} | 낙폭 {n['drawdown_pct']:+.1f}% | "
            f"{n['earnings_verdict']}"
        )
        lines.append(
            f"  진입 {n['entry_price']:,} → 목표 {n['target_price']:,} "
            f"(손절 {n['stop_loss']:,})"
        )
        # 분할매도 구간
        levels = n.get("target_levels", [])
        if len(levels) >= 2:
            lvl_parts = [f"{lv['price']:,}({lv['label']})" for lv in levels]
            lines.append(f"  분할매도: {' → '.join(lvl_parts)}")
        lines.append(
            f"  R:R {rr:.1f} | {method_str}"
        )
        lines.append("")

    msg = "\n".join(lines)

    try:
        from src.telegram_sender import send_message
        send_message(msg)
        logger.info("텔레그램 노다지 알림 발송: GOLD %d + SILVER %d", len(gold), len(silver))
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


# ═══════════════════════════════════════════════════
# 알파 스캐너 업로드 (새 테이블)
# ═══════════════════════════════════════════════════

REGIME_KR = {"BULL": "상승장", "BULL_CAUTION": "상승 주의", "CAUTION": "주의",
             "BEAR": "하락장", "CRISIS": "위기", "NEUTRAL": "보합"}
SHIELD_KR = {"GREEN": "안전", "YELLOW": "주의", "RED": "경고"}
SHIELD_POSITIONS = {"RED": 3, "YELLOW": 5, "GREEN": 8}


def _load_context() -> dict:
    """brain + shield → 상단 컨텍스트 바 데이터."""
    ctx = {"regime": "CAUTION", "regime_kr": "주의",
           "shield_status": "YELLOW", "shield_kr": "주의",
           "max_drawdown": 0, "max_positions": 5}

    brain_path = DATA_DIR / "brain_decision.json"
    if brain_path.exists():
        try:
            with open(brain_path, encoding="utf-8") as f:
                brain = json.load(f)
            regime = brain.get("effective_regime", "CAUTION")
            ctx["regime"] = regime
            ctx["regime_kr"] = REGIME_KR.get(regime, regime)
        except Exception:
            pass

    shield_path = DATA_DIR / "shield_report.json"
    if shield_path.exists():
        try:
            with open(shield_path, encoding="utf-8") as f:
                shield = json.load(f)
            status = shield.get("overall_level", "YELLOW")
            ctx["shield_status"] = status
            ctx["shield_kr"] = SHIELD_KR.get(status, status)
            ctx["max_positions"] = SHIELD_POSITIONS.get(status, 5)
            mdd = shield.get("mdd_status", {})
            ctx["max_drawdown"] = round(mdd.get("current_mdd_pct", 0), 1)
        except Exception:
            pass

    return ctx


def _load_sector_heat() -> list[dict]:
    """sector_composite.json → 섹터 온도 히트맵."""
    path = DATA_DIR / "sector_rotation" / "sector_composite.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = []
        for s in data.get("sectors", []):
            ret = s.get("ret_5", 0)
            if ret >= 3:
                temp = "HOT"
            elif ret >= 0:
                temp = "WARM"
            elif ret >= -3:
                temp = "COOL"
            else:
                temp = "COLD"
            result.append({
                "sector": s.get("sector", "?"),
                "ret_5d": round(ret, 1),
                "temperature": temp,
            })
        result.sort(key=lambda x: x["ret_5d"], reverse=True)
        return result
    except Exception:
        return []


def _load_smart_money() -> dict:
    """tomorrow_picks.json → 스마트 머니 흐름."""
    path = DATA_DIR / "tomorrow_picks.json"
    empty = {"dual_buy": [], "inst_top": [], "fgn_top": []}
    if not path.exists():
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        picks = data.get("picks", [])

        def _to_item(p):
            return {
                "ticker": p.get("ticker", ""),
                "name": p.get("name", ""),
                "foreign_5d_억": round(p.get("foreign_5d", 0) / 1e8, 1),
                "inst_5d_억": round(p.get("inst_5d", 0) / 1e8, 1),
                "close": int(p.get("close", 0)),
                "change_pct": round(p.get("ret_5d", 0), 1),
            }

        dual = [_to_item(p) for p in picks
                if p.get("foreign_5d", 0) > 0 and p.get("inst_5d", 0) > 0]
        dual.sort(key=lambda x: x["foreign_5d_억"] + x["inst_5d_억"], reverse=True)

        inst = [_to_item(p) for p in picks if p.get("inst_5d", 0) > 0]
        inst.sort(key=lambda x: x["inst_5d_억"], reverse=True)

        fgn = [_to_item(p) for p in picks if p.get("foreign_5d", 0) > 0]
        fgn.sort(key=lambda x: x["foreign_5d_억"], reverse=True)

        return {"dual_buy": dual[:5], "inst_top": inst[:5], "fgn_top": fgn[:5]}
    except Exception:
        return empty


def _load_portfolio() -> dict:
    """brain_decision.json arms → 자산 배분."""
    default = {"defense_pct": 50, "offense_pct": 50,
               "allocation": {"주식": 50, "채권": 0, "금/달러": 0, "현금": 50}}
    brain_path = DATA_DIR / "brain_decision.json"
    if not brain_path.exists():
        return default
    try:
        with open(brain_path, encoding="utf-8") as f:
            brain = json.load(f)
        arms = {a["name"]: a.get("adjusted_pct", 0) for a in brain.get("arms", [])}

        stocks = (arms.get("swing", 0) + arms.get("etf_sector", 0) +
                  arms.get("etf_leverage", 0) + arms.get("etf_index", 0) +
                  arms.get("etf_small_cap", 0))
        bonds = arms.get("etf_bonds", 0)
        gold_dollar = arms.get("etf_gold", 0) + arms.get("etf_dollar", 0)
        cash = arms.get("cash", 0)

        offense = round(stocks)
        defense = round(bonds + gold_dollar + cash)

        return {
            "defense_pct": int(defense),
            "offense_pct": int(offense),
            "allocation": {
                "주식": round(stocks),
                "채권": round(bonds),
                "금/달러": round(gold_dollar),
                "현금": round(cash),
            },
        }
    except Exception:
        return default


def _load_company_profiles() -> dict:
    """company_profiles.json 캐시 로드."""
    path = DATA_DIR / "company_profiles.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# 5축 점수 한글 라벨 (주린이 친화)
AXIS_LABELS_KR = {
    "value": "싸게 살 수 있나",
    "quality": "기업 체질",
    "earnings": "돈 잘 버나",
    "drawdown": "얼마나 빠졌나",
    "peer_value": "같은 업종 내 순위",
}


def _enrich_candidates(nuggets: list[dict], profiles: dict) -> list[dict]:
    """종목 데이터에 프로파일 + 5축 한글 라벨 추가."""
    enriched = []
    for n in nuggets:
        item = dict(n)
        code = item.get("ticker", item.get("code", ""))
        prof = profiles.get(code, {})
        item["company_desc"] = prof.get("desc", "")
        item["drop_reason"] = prof.get("drop_reason", "")
        item["axis_labels"] = AXIS_LABELS_KR
        enriched.append(item)
    return enriched


def upload_alpha_scanner(nuggets: list[dict], date_str: str = ""):
    """알파 스캐너 데이터를 quant_alpha_scanner 테이블에 업로드.

    brain/shield/섹터/수급/포트폴리오 + 종목 전체를 하나의 JSONB로 통합.
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    context = _load_context()
    sector_heat = _load_sector_heat()
    smart_money = _load_smart_money()
    portfolio = _load_portfolio()
    profiles = _load_company_profiles()

    # 종목 데이터에 프로파일 + 5축 라벨 추가
    candidates = _enrich_candidates(nuggets, profiles)

    grade_summary = {
        "GOLD": sum(1 for n in nuggets if n["grade"] == "GOLD"),
        "SILVER": sum(1 for n in nuggets if n["grade"] == "SILVER"),
        "BRONZE": sum(1 for n in nuggets if n["grade"] == "BRONZE"),
    }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "context": context,
        "sector_heat": sector_heat,
        "grade_summary": grade_summary,
        "candidates": candidates,
        "axis_labels": AXIS_LABELS_KR,
        "smart_money": smart_money,
        "portfolio": portfolio,
    }

    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        row = {"date": date_str, "data": payload}
        uploader.client.table("quant_alpha_scanner").upsert(
            row, on_conflict="date"
        ).execute()
        logger.info("[알파 스캐너] 업로드 완료: %s (%d종목, 섹터 %d, 수급 %d)",
                    date_str, len(nuggets), len(sector_heat),
                    len(smart_money.get("dual_buy", [])))
        return True
    except Exception as e:
        logger.error("[알파 스캐너] 업로드 오류: %s", e)
        return False


def upload_flowx(nuggets: list[dict], date_str: str = ""):
    """노다지 결과를 FLOWX short_signals 테이블에 업로드.

    signal_type = "NUGGET" 으로 구분.
    grade: GOLD→AA, SILVER→A, BRONZE→B
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    grade_map = {"GOLD": "AA", "SILVER": "A", "BRONZE": "B", "WATCH": "B"}

    rows = []
    for n in nuggets:
        if n["grade"] == "WATCH":
            continue  # WATCH는 업로드 안 함

        rows.append({
            "date": date_str,
            "code": n["ticker"],
            "name": n["name"],
            "grade": grade_map.get(n["grade"], "B"),
            "total_score": n["total_score"],
            "foreign_detail": None,
            "inst_support": False,
            "entry_price": n["entry_price"],
            "stop_loss": n["stop_loss"],
            "target_price": n["target_price"],
            "holding_days": 120,  # 장기 (6개월)
            "signal_type": "NUGGET",
            "volume_ratio": 1.0,
            "momentum_regime": n.get("regime", "CAUTION"),
        })

    if not rows:
        logger.info("[FLOWX] 노다지 업로드 대상 없음")
        return

    try:
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        ok = uploader.upload_ai_picks(rows)
        if ok:
            logger.info("[FLOWX] 노다지 업로드 완료: %d건", len(rows))
        else:
            logger.warning("[FLOWX] 노다지 업로드 실패")
    except Exception as e:
        logger.error("[FLOWX] 노다지 업로드 오류: %s", e)


# ═══════════════════════════════════════════════════
# CLI 엔트리포인트
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="노다지 리포트 — 장기 가치투자 종목 발굴")
    parser.add_argument("--telegram", action="store_true", help="텔레그램 알림 발송")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 출력만")
    parser.add_argument("--top", type=int, default=20, help="상위 N종목 (기본 20)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n[노다지] 스크리닝 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 스캔
    nuggets = scan_nuggets(top_n=args.top)

    if not nuggets:
        print("[노다지] 조건 충족 종목 없음")
        return

    # 콘솔 출력
    print_report(nuggets)

    # JSON 저장
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_report(nuggets, date_str)

    # FLOWX 업로드
    if not args.dry_run:
        upload_flowx(nuggets, date_str)          # 기존 short_signals (하위호환)
        upload_alpha_scanner(nuggets, date_str)   # 새 quant_alpha_scanner

    # 텔레그램
    if args.telegram:
        send_telegram(nuggets)

    print(f"\n[노다지] 완료 — {len(nuggets)}종목 발굴")


if __name__ == "__main__":
    main()
