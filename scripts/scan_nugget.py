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

        # ── Entry/StopLoss/Target (장기 관점) ──
        entry_price = int(close)
        stop_loss = int(close * 0.85)     # -15% 손절 (장기)
        target_price = int(close * 1.30)  # +30% 목표 (장기)

        candidates.append({
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "total_score": round(total, 1),
            "market_cap_억": round(market_cap_억, 0),
            "close": entry_price,
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
        lines.append("")

    msg = "\n".join(lines)

    try:
        from src.adapters.telegram_adapter import send_message
        send_message(msg)
        logger.info("텔레그램 노다지 알림 발송: GOLD %d + SILVER %d", len(gold), len(silver))
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


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
        upload_flowx(nuggets, date_str)

    # 텔레그램
    if args.telegram:
        send_telegram(nuggets)

    print(f"\n[노다지] 완료 — {len(nuggets)}종목 발굴")


if __name__ == "__main__":
    main()
