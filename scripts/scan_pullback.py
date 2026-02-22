"""
눌림목(건강한 조정) 스캐너 — 84종목 유니버스 전체 스캔

상승추세 종목 중 건강한 조정 구간에 있는 종목을 찾아
반등 매수 타이밍을 포착합니다.

판정 기준:
  1. 상승추세: Close > MA60, MA20 > MA60
  2. 조정 구간: RSI 30~58, BB% 10~55
  3. 반등 시그널: TRIX/Stoch/MACD 골든크로스
  4. 수급: 외인/기관 5일 순매수

등급:
  반등임박 (>=65)  — 상승추세 + 조정 적정 + 반등시그널 + 수급
  매수대기 (>=45)  — 상승추세 + 조정 중 (아직 반등시그널 약함)
  조정진행 (>=25)  — 하락 중이나 MA60 지지
  추가하락주의 (<25) — MA60 이탈 or 수급 악화

Usage:
    python scripts/scan_pullback.py
    python scripts/scan_pullback.py --top 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sf(val, default=0):
    """NaN/Inf 안전 변환"""
    v = float(val)
    return default if (np.isnan(v) or np.isinf(v)) else v
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = PROJECT_ROOT / "data" / "pullback_scan.json"


# ──────────────────────────────────────────
# 종목명 매핑
# ──────────────────────────────────────────

def build_name_map() -> dict[str, str]:
    """CSV 파일명에서 종목코드 → 종목명 매핑"""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name, ticker = parts
            name_map[ticker] = name
    return name_map


# ──────────────────────────────────────────
# 수급 계산 (CSV 기반)
# ──────────────────────────────────────────

def get_flow_from_csv(ticker: str) -> dict:
    """CSV에서 5일 수급 + 연속매수 일수"""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}

    try:
        df = pd.read_csv(csvs[0], parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 5:
            return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}

        last5 = df.tail(5)
        f5 = i5 = 0.0

        if "Foreign_Net" in df.columns:
            aligned = pd.DataFrame({
                "net": last5["Foreign_Net"].values,
                "close": last5["Close"].values,
            }).dropna()
            if len(aligned) > 0:
                f5 = round(float((aligned["net"] * aligned["close"]).sum()) / 1e8, 1)

        if "Inst_Net" in df.columns:
            aligned = pd.DataFrame({
                "net": last5["Inst_Net"].values,
                "close": last5["Close"].values,
            }).dropna()
            if len(aligned) > 0:
                i5 = round(float((aligned["net"] * aligned["close"]).sum()) / 1e8, 1)

        # 연속매수 일수
        def streak(col):
            if col not in df.columns:
                return 0
            vals = df[col].dropna().values
            s = 0
            for v in reversed(vals):
                if v > 0:
                    s += 1
                else:
                    break
            return s

        return {
            "foreign_5d": f5,
            "inst_5d": i5,
            "f_streak": streak("Foreign_Net"),
            "i_streak": streak("Inst_Net"),
        }
    except Exception as e:
        logger.debug("수급 로드 실패 %s: %s", ticker, e)
        return {"foreign_5d": 0, "inst_5d": 0, "f_streak": 0, "i_streak": 0}


# ──────────────────────────────────────────
# 눌림목 점수 (100점)
# ──────────────────────────────────────────

# 점수 배분
S_TREND = 25      # 상승추세 강도
S_PULLBACK = 25   # 건강한 조정 구간
S_BOUNCE = 25     # 반등 시그널
S_FLOW = 25       # 수급


def calc_pullback_score(row: pd.Series, flow: dict) -> dict:
    """한 종목의 눌림목 점수 계산"""
    close = _sf(row.get("close", 0))
    ma20 = _sf(row.get("sma_20", 0))
    ma60 = _sf(row.get("sma_60", 0))
    rsi = _sf(row.get("rsi_14", 50), 50)
    bb = _sf(row.get("bb_position", 0.5), 0.5) * 100  # 0~1 → 0~100
    adx = _sf(row.get("adx_14", 0))
    trix = _sf(row.get("trix", 0))
    trix_sig = _sf(row.get("trix_signal", 0))
    stoch_k = _sf(row.get("stoch_slow_k", 50), 50)
    stoch_d = _sf(row.get("stoch_slow_d", 50), 50)
    stoch_gx = bool(row.get("stoch_slow_golden", False))
    macd_hist = _sf(row.get("macd_histogram", 0))
    macd_hist_prev = _sf(row.get("macd_histogram_prev", 0))
    slope20 = _sf(row.get("linreg_slope_20", 0))
    ret1 = _sf(row.get("ret1", 0)) * 100  # fraction → %
    vol_ratio = _sf(row.get("volume_surge_ratio", 1), 1)

    f5 = flow.get("foreign_5d", 0)
    i5 = flow.get("inst_5d", 0)

    score = 0.0
    reasons = []
    signals = []

    # ── 1. 상승추세 (25점) ──
    trend_score = 0
    if ma20 > 0 and ma60 > 0:
        if close > ma60 and ma20 > ma60:
            trend_score = S_TREND
            reasons.append("MA20>MA60 상승추세")
        elif close > ma60:
            trend_score = S_TREND * 0.6
            reasons.append("MA60 위(약한 추세)")
        elif close > ma60 * 0.97:  # MA60 근접 (3% 이내)
            trend_score = S_TREND * 0.3
            reasons.append("MA60 근접 지지")
    if slope20 > 0:
        trend_score = min(trend_score + 3, S_TREND)
    score += trend_score

    # 상승추세 아니면 눌림목 대상 아님
    is_uptrend = close > 0 and ma60 > 0 and close > ma60 * 0.97

    # ── 2. 건강한 조정 (25점) ──
    pull_score = 0
    if is_uptrend:
        # RSI 조정 구간 (30~58 이상적)
        if 30 <= rsi <= 45:
            pull_score += 12
            reasons.append(f"RSI{rsi:.0f}(깊은조정)")
        elif 45 < rsi <= 58:
            pull_score += 10
            reasons.append(f"RSI{rsi:.0f}(적정조정)")
        elif 25 <= rsi < 30:
            pull_score += 6
            reasons.append(f"RSI{rsi:.0f}(과매도)")
        elif 58 < rsi <= 65:
            pull_score += 4
            reasons.append(f"RSI{rsi:.0f}(소폭조정)")

        # BB% 조정 구간 (10~55 이상적)
        if 10 <= bb <= 35:
            pull_score += 10
            reasons.append(f"BB{bb:.0f}%(하단)")
        elif 35 < bb <= 55:
            pull_score += 8
            reasons.append(f"BB{bb:.0f}%(중하단)")
        elif 0 <= bb < 10:
            pull_score += 5
            reasons.append(f"BB{bb:.0f}%(극하단)")
        elif 55 < bb <= 70:
            pull_score += 3

        # MA20 근접 (조정 중 지지)
        if ma20 > 0 and abs(close - ma20) / ma20 < 0.02:
            pull_score += 3
            reasons.append("MA20 지지")
    score += min(pull_score, S_PULLBACK)

    # ── 3. 반등 시그널 (25점) ──
    bounce_score = 0
    if is_uptrend:
        # TRIX 골든크로스 (또는 임박)
        if trix > trix_sig:
            bounce_score += 8
            signals.append("TRIX↑")
        elif trix > trix_sig * 0.95 and trix_sig < 0:
            bounce_score += 4
            signals.append("TRIX↑임박")

        # Stoch 골든크로스
        if stoch_gx or (stoch_k > stoch_d and stoch_k < 50):
            bounce_score += 8
            signals.append("Stoch↑")
        elif stoch_k < 30:
            bounce_score += 3
            signals.append("Stoch과매도")

        # MACD 히스토그램 반전 (하락→상승)
        if macd_hist > macd_hist_prev and macd_hist_prev < 0:
            bounce_score += 6
            signals.append("MACD반전")
        elif macd_hist > macd_hist_prev:
            bounce_score += 3
            signals.append("MACD개선")

        # ADX (추세 강도)
        if adx >= 20:
            bounce_score += 3
            signals.append(f"ADX{adx:.0f}")
    score += min(bounce_score, S_BOUNCE)

    # ── 4. 수급 (25점) ──
    flow_score = 0
    dual = f5 > 0 and i5 > 0
    has_buyer = f5 > 0 or i5 > 0

    if dual:
        flow_score = S_FLOW
        reasons.append(f"외인{f5:+.0f}억+기관{i5:+.0f}억(동시)")
    elif f5 > 5:
        flow_score = S_FLOW * 0.7
        reasons.append(f"외인{f5:+.0f}억(5일)")
    elif i5 > 5:
        flow_score = S_FLOW * 0.7
        reasons.append(f"기관{i5:+.0f}억(5일)")
    elif has_buyer:
        flow_score = S_FLOW * 0.4
        buyer = f"외인{f5:+.0f}억" if f5 > 0 else f"기관{i5:+.0f}억"
        reasons.append(f"{buyer}(5일)")
    elif f5 < -5 and i5 < -5:
        reasons.append("수급악화")
    score += flow_score

    score = round(min(score, 100), 1)

    # 등급
    if score >= 65:
        grade = "반등임박"
    elif score >= 45:
        grade = "매수대기"
    elif score >= 25:
        grade = "조정진행"
    else:
        grade = "추가하락주의"

    return {
        "score": score,
        "grade": grade,
        "reasons": reasons,
        "signals": signals,
        "is_uptrend": is_uptrend,
        "trend_score": round(trend_score, 1),
        "pull_score": round(min(pull_score, S_PULLBACK), 1),
        "bounce_score": round(min(bounce_score, S_BOUNCE), 1),
        "flow_score": round(flow_score, 1),
    }


# ──────────────────────────────────────────
# 메인 스캔
# ──────────────────────────────────────────

def scan_pullback(top_n: int = 30) -> dict:
    """전종목 눌림목 스캔"""
    name_map = build_name_map()
    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    logger.info("parquet %d개 스캔 시작", len(parquets))

    results = []

    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 60:
                continue

            row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) >= 2 else row
            close = float(row.get("close", 0))
            if close <= 0:
                continue

            # 수급 조회
            flow = get_flow_from_csv(ticker)

            # 점수 계산
            result = calc_pullback_score(row, flow)

            # 상승추세 아닌 종목은 제외 (눌림목 대상 아님)
            if not result["is_uptrend"]:
                continue

            # 기본 정보
            ma20 = _sf(row.get("sma_20", 0))
            ma60 = _sf(row.get("sma_60", 0))

            results.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "close": int(close),
                "ret_1": round(_sf(row.get("ret1", 0)) * 100, 2),
                "rsi": round(_sf(row.get("rsi_14", 50), 50), 1),
                "bb_pct": round(_sf(row.get("bb_position", 0.5), 0.5) * 100, 1),
                "adx": round(_sf(row.get("adx_14", 0)), 1),
                "trix": round(_sf(row.get("trix", 0)), 4),
                "trix_signal": round(_sf(row.get("trix_signal", 0)), 4),
                "trix_bull": _sf(row.get("trix", 0)) > _sf(row.get("trix_signal", 0)),
                "stoch_k": round(_sf(row.get("stoch_slow_k", 50), 50), 1),
                "stoch_d": round(_sf(row.get("stoch_slow_d", 50), 50), 1),
                "stoch_gx": bool(row.get("stoch_slow_golden", False)),
                "macd_improving": _sf(row.get("macd_histogram", 0)) > _sf(row.get("macd_histogram_prev", 0)),
                "ma20_gap": round((close / ma20 - 1) * 100, 1) if ma20 > 0 else 0,
                "ma60_gap": round((close / ma60 - 1) * 100, 1) if ma60 > 0 else 0,
                "foreign_5d": flow["foreign_5d"],
                "inst_5d": flow["inst_5d"],
                "f_streak": flow["f_streak"],
                "i_streak": flow["i_streak"],
                "score": result["score"],
                "grade": result["grade"],
                "reasons": result["reasons"],
                "signals": result["signals"],
                "detail": {
                    "trend": result["trend_score"],
                    "pullback": result["pull_score"],
                    "bounce": result["bounce_score"],
                    "flow": result["flow_score"],
                },
            })
        except Exception as e:
            logger.debug("스캔 실패 %s: %s", ticker, e)
            continue

    # 점수 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    # 등급별 분류
    grade_counts = {}
    for r in results:
        g = r["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1

    report = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_scanned": len(parquets),
        "uptrend_count": len(results),
        "grade_counts": grade_counts,
        "candidates": results[:top_n],
        "all_uptrend": results,  # 전체 (대시보드 테이블용)
    }

    return report


def save_report(report: dict) -> Path:
    """JSON 저장"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s (%d개 종목)", OUTPUT_PATH, len(report["candidates"]))
    return OUTPUT_PATH


def print_report(report: dict) -> None:
    """콘솔 출력"""
    print("\n" + "=" * 70)
    print(f"  눌림목 스캔 결과 — {report['updated_at']}")
    print(f"  전체 {report['total_scanned']}개 → 상승추세 {report['uptrend_count']}개")
    print(f"  등급: {report['grade_counts']}")
    print("=" * 70)

    for r in report["candidates"]:
        sigs = " ".join(r["signals"]) if r["signals"] else ""
        f5 = r["foreign_5d"]
        i5 = r["inst_5d"]
        dual = "★" if f5 > 0 and i5 > 0 else ""
        print(f"  [{r['grade']:6s}] {r['name']:12s} ({r['ticker']}) "
              f"{r['score']:5.1f}점 | RSI {r['rsi']:4.1f} BB {r['bb_pct']:4.1f}% "
              f"| 외{f5:+6.0f} 기{i5:+6.0f}{dual} "
              f"| {sigs}")

    print()


def main():
    parser = argparse.ArgumentParser(description="눌림목 스캐너")
    parser.add_argument("--top", type=int, default=30, help="상위 N개 (기본: 30)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = scan_pullback(top_n=args.top)
    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
