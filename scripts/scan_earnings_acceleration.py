"""실적 가속도 분석기 — CFO 업그레이드 #1

DART 분기 재무제표(financial_quarterly.json)에서:
  1. 영업이익 QoQ/YoY 변화율 계산
  2. 영업이익 가속도 (이번 QoQ - 전번 QoQ)
  3. 영업이익률 추이
  4. 5가지 상태 분류:
     - TURNAROUND_STRONG: 적자→흑자 전환 완료
     - TURNAROUND_EARLY:  적자 축소 중
     - ACCELERATING:      흑자 + 성장 가속
     - DECELERATING:      흑자이지만 성장 둔화
     - DETERIORATING:     실적 악화 중

실행:
  python -u -X utf8 scripts/scan_earnings_acceleration.py
  python -u -X utf8 scripts/scan_earnings_acceleration.py --send
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EA] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("earnings_accel")

DATA_DIR = PROJECT_ROOT / "data"
FINANCIAL_PATH = DATA_DIR / "v2_migration" / "financial_quarterly.json"
OUTPUT_PATH = DATA_DIR / "earnings_acceleration.json"


def _load_name_map() -> dict[str, str]:
    """pykrx 종목명 캐시 (data/universe/name_map.json)"""
    cache_path = DATA_DIR / "universe" / "name_map.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_financials() -> dict:
    """financial_quarterly.json → {ticker: {bs: {quarter: {op_income, revenue}}, quality: {...}, name}} 변환

    원본 구조: {meta, bs_data: {ticker: {q: {op_income_cum, revenue_cum}}}, quality: {ticker: {debt_ratio}}}
    Q4는 연간 누적이므로 제외하고 Q1-Q3만 사용.
    """
    if not FINANCIAL_PATH.exists():
        logger.error(f"재무데이터 없음: {FINANCIAL_PATH}")
        return {}
    with open(FINANCIAL_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    bs_data = raw.get("bs_data", {})
    quality_data = raw.get("quality", {})
    name_map = _load_name_map()

    # 캐시 미스 시 pykrx로 보충
    missing = [t for t in bs_data if t not in name_map]
    if missing:
        try:
            from pykrx import stock as krx
            for t in missing:
                try:
                    name_map[t] = krx.get_market_ticker_name(t) or t
                except Exception:
                    name_map[t] = t
            # 캐시 저장
            cache_path = DATA_DIR / "universe" / "name_map.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(name_map, f, ensure_ascii=False, indent=2)
            logger.info(f"종목명 캐시 갱신: {len(missing)}개 추가")
        except ImportError:
            logger.warning("pykrx 미설치 — 종목명 ticker로 대체")

    result = {}
    for ticker, quarters in bs_data.items():
        bs = {}
        for q, vals in quarters.items():
            if q.endswith("Q4"):
                continue  # Q4는 연간 누적 → QoQ 비교 불가
            bs[q] = {
                "op_income": vals.get("op_income_cum"),
                "revenue": vals.get("revenue_cum"),
            }
        if not bs:
            continue
        q_info = quality_data.get(ticker, {})
        result[ticker] = {
            "bs": bs,
            "quality": q_info,
            "name": name_map.get(ticker, ticker),
        }
    logger.info(f"재무데이터 로드: {len(result)}종목")
    return result


def _safe_div(a: float, b: float) -> float | None:
    """안전한 나눗셈"""
    if b is None or b == 0:
        return None
    return a / abs(b)


def analyze_stock(ticker: str, data: dict) -> dict | None:
    """단일 종목 실적 가속도 분석"""
    bs = data.get("bs", {})
    quality = data.get("quality", {})
    name = data.get("name", ticker)

    # 분기별 영업이익 추출 (최근 8분기)
    quarters = sorted(bs.keys())[-8:]  # "2024Q1", "2024Q2", ...
    if len(quarters) < 3:
        return None

    op_incomes = []
    revenues = []
    for q in quarters:
        q_data = bs.get(q, {})
        oi = q_data.get("op_income")
        rev = q_data.get("revenue")
        if oi is not None:
            op_incomes.append({"quarter": q, "value": oi, "revenue": rev})

    if len(op_incomes) < 3:
        return None

    # QoQ 변화율 계산
    qoq_changes = []
    for i in range(1, len(op_incomes)):
        prev = op_incomes[i - 1]["value"]
        curr = op_incomes[i]["value"]
        change = _safe_div(curr - prev, prev)
        qoq_changes.append({
            "quarter": op_incomes[i]["quarter"],
            "op_income": curr,
            "prev_op_income": prev,
            "qoq_change": change,
        })

    # YoY 변화율 (같은 분기 비교)
    yoy_changes = []
    for i in range(4, len(op_incomes)):
        prev_yr = op_incomes[i - 4]["value"]
        curr = op_incomes[i]["value"]
        change = _safe_div(curr - prev_yr, prev_yr)
        yoy_changes.append({
            "quarter": op_incomes[i]["quarter"],
            "yoy_change": change,
        })

    # 가속도 (이번 QoQ - 전번 QoQ)
    acceleration = None
    if len(qoq_changes) >= 2:
        curr_qoq = qoq_changes[-1]["qoq_change"]
        prev_qoq = qoq_changes[-2]["qoq_change"]
        if curr_qoq is not None and prev_qoq is not None:
            acceleration = curr_qoq - prev_qoq

    # 영업이익률 추이
    opm_trend = []
    for item in op_incomes[-4:]:
        rev = item["revenue"]
        oi = item["value"]
        if rev and rev > 0:
            opm_trend.append(round(oi / rev * 100, 1))
        else:
            opm_trend.append(None)

    # 최근 2분기 영업이익
    latest = op_incomes[-1]["value"]
    prev = op_incomes[-2]["value"]
    prev2 = op_incomes[-3]["value"] if len(op_incomes) >= 3 else None

    # ─── 5가지 상태 분류 ───────────────────────
    status = "UNKNOWN"
    confidence = 0.0

    latest_qoq = qoq_changes[-1]["qoq_change"] if qoq_changes else None

    if prev < 0 and latest > 0:
        # 적자 → 흑자 전환
        status = "TURNAROUND_STRONG"
        confidence = 0.9
    elif prev < 0 and latest < 0 and prev2 is not None and prev2 < 0:
        # 연속 적자지만 적자폭 축소?
        if abs(latest) < abs(prev):
            status = "TURNAROUND_EARLY"
            confidence = 0.6 + min(abs(prev - latest) / abs(prev) * 0.3, 0.3)
        else:
            status = "DETERIORATING"
            confidence = 0.7
    elif latest > 0 and acceleration is not None:
        if acceleration > 0 and (latest_qoq is not None and latest_qoq > 0):
            status = "ACCELERATING"
            confidence = 0.5 + min(acceleration * 2, 0.4)
        elif latest_qoq is not None and latest_qoq > 0:
            status = "DECELERATING"
            confidence = 0.5
        else:
            status = "DETERIORATING"
            confidence = 0.6
    elif latest > 0:
        status = "ACCELERATING" if latest_qoq and latest_qoq > 0 else "DECELERATING"
        confidence = 0.5
    else:
        status = "DETERIORATING"
        confidence = 0.7

    # 스코어 (TURNAROUND_STRONG이 가장 높음)
    score_map = {
        "TURNAROUND_STRONG": 90,
        "TURNAROUND_EARLY": 70,
        "ACCELERATING": 60,
        "DECELERATING": 30,
        "DETERIORATING": 10,
        "UNKNOWN": 0,
    }
    score = score_map.get(status, 0) * confidence

    return {
        "ticker": ticker,
        "name": name,
        "status": status,
        "score": round(score, 1),
        "confidence": round(confidence, 2),
        "latest_op_income": latest,
        "prev_op_income": prev,
        "qoq_change": round(latest_qoq, 3) if latest_qoq is not None else None,
        "yoy_change": round(yoy_changes[-1]["yoy_change"], 3) if yoy_changes and yoy_changes[-1]["yoy_change"] is not None else None,
        "acceleration": round(acceleration, 3) if acceleration is not None else None,
        "opm_trend": opm_trend,
        "quarters_analyzed": len(op_incomes),
    }


def main():
    parser = argparse.ArgumentParser(description="실적 가속도 분석기")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--top", type=int, default=20, help="출력 종목 수")
    args = parser.parse_args()

    financials = load_financials()
    if not financials:
        print("재무 데이터 없음. 종료.")
        return

    results = []
    for ticker, data in financials.items():
        r = analyze_stock(ticker, data)
        if r:
            results.append(r)

    # 상태별 분류
    by_status = {}
    for r in results:
        s = r["status"]
        by_status.setdefault(s, []).append(r)

    # 스코어 내림차순
    results.sort(key=lambda x: x["score"], reverse=True)

    # 요약
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "status_counts": {s: len(v) for s, v in by_status.items()},
        "turnaround_strong": sorted(
            by_status.get("TURNAROUND_STRONG", []),
            key=lambda x: x["score"], reverse=True,
        ),
        "turnaround_early": sorted(
            by_status.get("TURNAROUND_EARLY", []),
            key=lambda x: x["score"], reverse=True,
        ),
        "accelerating": sorted(
            by_status.get("ACCELERATING", []),
            key=lambda x: x["score"], reverse=True,
        )[:args.top],
        "decelerating": sorted(
            by_status.get("DECELERATING", []),
            key=lambda x: x["score"], reverse=True,
        )[:10],
        "deteriorating_count": len(by_status.get("DETERIORATING", [])),
    }

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    print(f"\n{'='*60}")
    print(f"  실적 가속도 분석 — {summary['date']}")
    print(f"  분석 종목: {summary['total_analyzed']}개")
    print(f"{'='*60}")

    for status_key, label in [
        ("turnaround_strong", "🔥 적자→흑자 전환 (TURNAROUND_STRONG)"),
        ("turnaround_early", "📈 적자 축소 중 (TURNAROUND_EARLY)"),
        ("accelerating", "🚀 성장 가속 (ACCELERATING)"),
    ]:
        items = summary.get(status_key, [])
        if items:
            print(f"\n{label} — {len(items)}종목")
            for r in items[:10]:
                qoq_str = f"QoQ {r['qoq_change']:+.1%}" if r['qoq_change'] is not None else "QoQ N/A"
                accel_str = f"가속도 {r['acceleration']:+.3f}" if r['acceleration'] is not None else ""
                oi = r['latest_op_income']
                oi_str = f"{oi/1e8:.0f}억" if abs(oi) >= 1e8 else f"{oi/1e4:.0f}만"
                print(f"  {r['name']:12s} ({r['ticker']}) 영업이익 {oi_str:>8s} | {qoq_str} | {accel_str} | 점수 {r['score']:.0f}")

    counts = summary["status_counts"]
    print(f"\n[분포] STRONG:{counts.get('TURNAROUND_STRONG',0)} EARLY:{counts.get('TURNAROUND_EARLY',0)} "
          f"ACCEL:{counts.get('ACCELERATING',0)} DECEL:{counts.get('DECELERATING',0)} "
          f"DETER:{counts.get('DETERIORATING',0)}")
    print(f"[저장] {OUTPUT_PATH}")

    # 텔레그램
    if args.send:
        try:
            from src.adapters.telegram_adapter import send_message
            lines = [f"📊 실적 가속도 분석 — {summary['date']}"]
            lines.append(f"분석: {summary['total_analyzed']}종목")

            for s in summary.get("turnaround_strong", [])[:5]:
                lines.append(f"🔥 {s['name']} — 흑자전환! 영업이익 {s['latest_op_income']/1e8:.0f}억")
            for s in summary.get("turnaround_early", [])[:3]:
                lines.append(f"📈 {s['name']} — 적자축소 중")
            for s in summary.get("accelerating", [])[:5]:
                qoq = f"QoQ {s['qoq_change']:+.0%}" if s['qoq_change'] else ""
                lines.append(f"🚀 {s['name']} — 성장가속 {qoq}")

            send_message("\n".join(lines))
            print("[텔레그램] 발송 완료")
        except Exception as e:
            print(f"[텔레그램] 발송 실패: {e}")


if __name__ == "__main__":
    main()
