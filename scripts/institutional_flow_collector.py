"""TIER2 — 섹터별 기관/외인 수급 수집 + 기관 매집 감지

Phase 0-1: 섹터별 상위 종목의 기관/외인/개인 수급 30일 시계열 수집
Phase 0-2: 기관 매집 감지기 (연속 순매수 + 금액 임계치)

데이터 소스: KIS API (FHKST01010900) — 30일 투자자별 매매동향
수집 대상: sector_map.json의 섹터별 상위 N종목 (가중치 기준)

BAT-D Phase 3 이후 실행 권장.

Usage:
    python -u -X utf8 scripts/institutional_flow_collector.py
    python -u -X utf8 scripts/institutional_flow_collector.py --top-n 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.kis_investor_adapter import fetch_investor_by_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 경로
SECTOR_MAP_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "sector_map.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "institutional_flow"
OUTPUT_PATH = OUTPUT_DIR / "sector_institutional_flow.json"
ACCUMULATION_ALERT_PATH = OUTPUT_DIR / "accumulation_alert.json"

# 기관 매집 감지 기준
ACCUMULATION_CRITERIA = {
    "min_consecutive_days": 3,     # 최소 연속 순매수 일수
    "strong_consecutive_days": 5,  # 강한 매집 기준
    "min_cumulative_5d_억": 50,    # 5일 누적 50억+
    "min_cumulative_5d_억_foreign": 100,  # 외국인 5일 누적 100억+
}


def load_sector_map() -> dict:
    """sector_map.json 로드"""
    if not SECTOR_MAP_PATH.exists():
        logger.error(f"sector_map.json 없음: {SECTOR_MAP_PATH}")
        return {}
    with open(SECTOR_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_sector_top_stocks(sector_map: dict, top_n: int = 5) -> dict:
    """섹터별 상위 N종목 추출 (가중치 기준)"""
    result = {}
    for sector, info in sector_map.items():
        stocks = info.get("stocks", [])
        sorted_stocks = sorted(stocks, key=lambda x: x.get("weight", 0), reverse=True)
        result[sector] = sorted_stocks[:top_n]
    return result


def _analyze_investor_df(df) -> dict:
    """KIS 투자자 DataFrame → 분석 지표 추출"""
    if df is None or df.empty:
        return {
            "inst_net": 0, "foreign_net": 0, "individual_net": 0,
            "inst_consecutive": 0, "foreign_consecutive": 0,
            "inst_5d": 0, "inst_20d": 0,
            "foreign_5d": 0, "foreign_20d": 0,
            "dual_buying_5d": False,
        }

    # 오늘(장중) 제외하고 마지막 유효 행
    valid = df[df["기관합계"] != 0]
    if valid.empty:
        valid = df

    latest = valid.iloc[-1]

    # 기관 연속 순매수 일수
    inst_consec = 0
    for val in reversed(valid["기관합계"].values):
        if val > 0:
            inst_consec += 1
        else:
            break

    # 외국인 연속 순매수 일수
    foreign_consec = 0
    for val in reversed(valid["외국인합계"].values):
        if val > 0:
            foreign_consec += 1
        else:
            break

    # 누적 순매수
    inst_5d = int(valid["기관합계"].tail(5).sum()) if len(valid) >= 5 else int(valid["기관합계"].sum())
    inst_20d = int(valid["기관합계"].tail(20).sum()) if len(valid) >= 20 else int(valid["기관합계"].sum())
    foreign_5d = int(valid["외국인합계"].tail(5).sum()) if len(valid) >= 5 else int(valid["외국인합계"].sum())
    foreign_20d = int(valid["외국인합계"].tail(20).sum()) if len(valid) >= 20 else int(valid["외국인합계"].sum())

    # 쌍끌이 (기관+외국인 동시 5일 순매수)
    dual = inst_5d > 0 and foreign_5d > 0

    return {
        "inst_net": int(latest["기관합계"]),
        "foreign_net": int(latest["외국인합계"]),
        "individual_net": int(latest["개인"]),
        "inst_consecutive": inst_consec,
        "foreign_consecutive": foreign_consec,
        "inst_5d": inst_5d,
        "inst_20d": inst_20d,
        "foreign_5d": foreign_5d,
        "foreign_20d": foreign_20d,
        "dual_buying_5d": dual,
    }


def collect_sector_flows(sector_stocks: dict) -> dict:
    """섹터별 종목의 기관/외인 수급 수집 (KIS API)

    Returns:
        {섹터: {stocks: {code: 분석결과}, aggregate: 섹터합산}}
    """
    results = {}
    total_stocks = sum(len(stocks) for stocks in sector_stocks.values())
    collected = 0
    errors = 0

    logger.info(f"[섹터 수급] {len(sector_stocks)}개 섹터, {total_stocks}종목 수집 시작")
    t0 = time.time()

    for sector, stocks in sector_stocks.items():
        sector_data = {"stocks": {}, "aggregate": {}}

        agg_inst_5d = 0
        agg_inst_20d = 0
        agg_foreign_5d = 0
        agg_foreign_20d = 0
        agg_inst_net = 0
        agg_foreign_net = 0
        max_inst_consec = 0
        max_foreign_consec = 0
        dual_count = 0

        for stock_info in stocks:
            code = stock_info["code"]
            name = stock_info.get("name", code)

            try:
                df = fetch_investor_by_ticker(code)
                analysis = _analyze_investor_df(df)
                analysis["name"] = name
                sector_data["stocks"][code] = analysis

                agg_inst_net += analysis["inst_net"]
                agg_foreign_net += analysis["foreign_net"]
                agg_inst_5d += analysis["inst_5d"]
                agg_inst_20d += analysis["inst_20d"]
                agg_foreign_5d += analysis["foreign_5d"]
                agg_foreign_20d += analysis["foreign_20d"]
                if analysis["inst_consecutive"] > max_inst_consec:
                    max_inst_consec = analysis["inst_consecutive"]
                if analysis["foreign_consecutive"] > max_foreign_consec:
                    max_foreign_consec = analysis["foreign_consecutive"]
                if analysis["dual_buying_5d"]:
                    dual_count += 1

                collected += 1
            except Exception as e:
                logger.warning(f"  [{sector}] {name}({code}) 실패: {e}")
                errors += 1

            time.sleep(0.12)  # KIS rate limit 방지

        # 섹터 집계
        sector_data["aggregate"] = {
            "inst_net": agg_inst_net,
            "foreign_net": agg_foreign_net,
            "inst_5d": agg_inst_5d,
            "inst_20d": agg_inst_20d,
            "foreign_5d": agg_foreign_5d,
            "foreign_20d": agg_foreign_20d,
            "inst_5d_억": round(agg_inst_5d / 1e8, 1),
            "foreign_5d_억": round(agg_foreign_5d / 1e8, 1),
            "inst_20d_억": round(agg_inst_20d / 1e8, 1),
            "foreign_20d_억": round(agg_foreign_20d / 1e8, 1),
            "max_inst_consecutive": max_inst_consec,
            "max_foreign_consecutive": max_foreign_consec,
            "dual_buying_stocks": dual_count,
        }

        # 가중 수급 점수: 기관(40%) + 외국인(40%) + 쌍끌이 보너스(20%)
        total_abs = abs(agg_inst_5d) + abs(agg_foreign_5d)
        if total_abs > 0:
            inst_score = (agg_inst_5d / total_abs) * 40
            foreign_score = (agg_foreign_5d / total_abs) * 40
            dual_bonus = min(dual_count / len(stocks), 1.0) * 20 if stocks else 0
            raw = 50 + inst_score + foreign_score + dual_bonus
            sector_data["aggregate"]["weighted_score"] = round(
                max(0, min(100, raw)), 1
            )
        else:
            sector_data["aggregate"]["weighted_score"] = 50.0

        results[sector] = sector_data

        logger.info(
            f"  [{sector}] {len(stocks)}종목 | "
            f"기관5d {agg_inst_5d/1e8:+.0f}억 | "
            f"외인5d {agg_foreign_5d/1e8:+.0f}억 | "
            f"쌍끌이 {dual_count}종목"
        )

    elapsed = time.time() - t0
    logger.info(
        f"[섹터 수급] 완료: {collected}/{total_stocks}종목, "
        f"{elapsed:.0f}초 | 오류: {errors}"
    )
    return results


def detect_accumulation(sector_results: dict) -> list[dict]:
    """기관/외인 매집 감지 — 종목 레벨

    Returns:
        [{ticker, name, sector, inst_consecutive, foreign_consecutive, ...grade}]
    """
    alerts = []
    c = ACCUMULATION_CRITERIA

    for sector, data in sector_results.items():
        for code, s in data.get("stocks", {}).items():
            inst_consec = s.get("inst_consecutive", 0)
            foreign_consec = s.get("foreign_consecutive", 0)
            inst_5d_억 = s.get("inst_5d", 0) / 1e8
            foreign_5d_억 = s.get("foreign_5d", 0) / 1e8

            # 감지 조건
            inst_accum = inst_consec >= c["min_consecutive_days"] or inst_5d_억 >= c["min_cumulative_5d_억"]
            foreign_accum = foreign_consec >= c["min_consecutive_days"] or foreign_5d_억 >= c["min_cumulative_5d_억_foreign"]
            dual = s.get("dual_buying_5d", False)

            if not (inst_accum or foreign_accum):
                continue

            # 등급 판정
            if dual and (inst_consec >= c["strong_consecutive_days"] or foreign_consec >= c["strong_consecutive_days"]):
                grade = "STRONG"   # 쌍끌이 + 연속 5일+
            elif dual:
                grade = "MODERATE" # 쌍끌이
            elif inst_consec >= c["strong_consecutive_days"] or foreign_consec >= c["strong_consecutive_days"]:
                grade = "NOTABLE"  # 한쪽 강한 연속
            else:
                grade = "WATCH"    # 관심

            alerts.append({
                "ticker": code,
                "name": s.get("name", code),
                "sector": sector,
                "inst_consecutive": inst_consec,
                "foreign_consecutive": foreign_consec,
                "inst_5d_억": round(inst_5d_억, 1),
                "foreign_5d_억": round(foreign_5d_억, 1),
                "inst_20d_억": round(s.get("inst_20d", 0) / 1e8, 1),
                "foreign_20d_억": round(s.get("foreign_20d", 0) / 1e8, 1),
                "dual_buying": dual,
                "grade": grade,
            })

    grade_order = {"STRONG": 0, "MODERATE": 1, "NOTABLE": 2, "WATCH": 3}
    alerts.sort(key=lambda x: (grade_order.get(x["grade"], 9), -abs(x["inst_5d_억"]) - abs(x["foreign_5d_억"])))

    # 같은 종목이 여러 섹터에 중복 감지된 경우 가장 높은 등급만 유지
    seen = {}
    deduped = []
    for a in alerts:
        tk = a["ticker"]
        if tk not in seen:
            seen[tk] = True
            deduped.append(a)
    return deduped


def detect_sector_flow(sector_results: dict) -> list[dict]:
    """섹터 레벨 수급 플로우"""
    sector_flows = []

    for sector, data in sector_results.items():
        agg = data.get("aggregate", {})
        inst_5d = agg.get("inst_5d_억", 0)
        foreign_5d = agg.get("foreign_5d_억", 0)
        combined = inst_5d + foreign_5d

        if combined >= 200:
            grade = "STRONG_BUY"
        elif combined >= 50:
            grade = "BUY"
        elif combined <= -200:
            grade = "STRONG_SELL"
        elif combined <= -50:
            grade = "SELL"
        else:
            grade = "NEUTRAL"

        sector_flows.append({
            "sector": sector,
            "inst_5d_억": inst_5d,
            "foreign_5d_억": foreign_5d,
            "combined_5d_억": round(combined, 1),
            "inst_20d_억": agg.get("inst_20d_억", 0),
            "foreign_20d_억": agg.get("foreign_20d_억", 0),
            "weighted_score": agg.get("weighted_score", 50),
            "dual_buying_stocks": agg.get("dual_buying_stocks", 0),
            "grade": grade,
        })

    sector_flows.sort(key=lambda x: x["combined_5d_억"], reverse=True)
    return sector_flows


def format_alert_text(stock_alerts: list[dict], sector_flows: list[dict]) -> str:
    """텔레그램 알림용 텍스트"""
    lines = []
    lines.append("TIER2 섹터 수급 리포트")
    lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 매집 종목
    strong = [a for a in stock_alerts if a["grade"] in ("STRONG", "MODERATE")]
    if strong:
        lines.append("== 기관/외인 매집 감지 ==")
        for a in strong[:10]:
            tag = "[DUAL]" if a["dual_buying"] else "[INST]" if a["inst_consecutive"] >= 3 else "[FRGN]"
            lines.append(
                f"{tag} {a['name']}({a['ticker']}) [{a['sector']}]"
            )
            lines.append(
                f"   기관 {a['inst_consecutive']}일/{a['inst_5d_억']:+.0f}억(5d) | "
                f"외인 {a['foreign_consecutive']}일/{a['foreign_5d_억']:+.0f}억(5d)"
            )
        lines.append("")

    # 섹터 플로우
    top = [s for s in sector_flows if s["grade"] in ("STRONG_BUY", "BUY")]
    bot = [s for s in sector_flows if s["grade"] in ("STRONG_SELL", "SELL")]

    if top:
        lines.append("== 수급 유입 섹터 ==")
        for s in top[:5]:
            lines.append(
                f"  {s['sector']} | 기관5d {s['inst_5d_억']:+.0f}억 | "
                f"외인5d {s['foreign_5d_억']:+.0f}억 | "
                f"점수 {s['weighted_score']:.0f}"
            )
        lines.append("")

    if bot:
        lines.append("== 수급 유출 섹터 ==")
        for s in bot[:5]:
            lines.append(
                f"  {s['sector']} | 기관5d {s['inst_5d_억']:+.0f}억 | "
                f"외인5d {s['foreign_5d_억']:+.0f}억"
            )
        lines.append("")

    if not strong and not top and not bot:
        lines.append("특이 사항 없음 (수급 중립)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TIER2 섹터 수급 수집")
    parser.add_argument("--top-n", type=int, default=5, help="섹터별 수집 종목 수 (기본 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TIER2 섹터별 기관/외인 수급 수집 + 매집 감지")
    print("=" * 60)

    sector_map = load_sector_map()
    if not sector_map:
        print("sector_map.json 로드 실패")
        return

    sector_stocks = get_sector_top_stocks(sector_map, top_n=args.top_n)
    total = sum(len(v) for v in sector_stocks.values())
    print(f"\n수집 대상: {len(sector_stocks)}개 섹터 x 상위 {args.top_n}종목 = {total}종목")

    # 수집
    sector_results = collect_sector_flows(sector_stocks)

    # 감지
    stock_alerts = detect_accumulation(sector_results)
    sector_flows = detect_sector_flow(sector_results)

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "collected_at": datetime.now().isoformat(),
        "sector_count": len(sector_results),
        "top_n": args.top_n,
        "sectors": {
            sector: {"aggregate": data["aggregate"], "stocks": data["stocks"]}
            for sector, data in sector_results.items()
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUTPUT_PATH}")

    alert_output = {
        "detected_at": datetime.now().isoformat(),
        "stock_alerts": stock_alerts,
        "sector_flows": sector_flows,
        "alert_text": format_alert_text(stock_alerts, sector_flows),
    }
    with open(ACCUMULATION_ALERT_PATH, "w", encoding="utf-8") as f:
        json.dump(alert_output, f, ensure_ascii=False, indent=2)
    print(f"저장: {ACCUMULATION_ALERT_PATH}")

    # 출력
    print(f"\n{alert_output['alert_text']}")

    print("\n" + "=" * 60)
    print("  섹터별 수급 점수 (0=극약세, 100=극강세)")
    print("=" * 60)
    for sf in sector_flows:
        bar_len = max(0, min(20, int(sf["weighted_score"] / 5)))
        bar = "+" * bar_len + "-" * (20 - bar_len)
        print(f"  {sf['sector']:>8s} | [{bar}] {sf['weighted_score']:5.1f} | "
              f"기관 {sf['inst_5d_억']:+6.0f}억 | 외인 {sf['foreign_5d_억']:+6.0f}억")

    print(f"\n매집 감지: {len(stock_alerts)}건")
    strong_count = len([a for a in stock_alerts if a["grade"] in ("STRONG", "MODERATE")])
    if strong_count:
        print(f"  STRONG/MODERATE: {strong_count}건")


if __name__ == "__main__":
    main()
