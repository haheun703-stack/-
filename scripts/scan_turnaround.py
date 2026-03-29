"""적자→흑자 턴어라운드 스크리너 — CFO 업그레이드 #4

DART 분기 재무제표에서 턴어라운드 후보를 자동 발견:
  조건:
    1. 직전 2분기 연속 영업적자
    2. 최근 1분기 흑자전환 OR 적자폭 50%+ 축소
    3. 매출액 비감소 (구조조정 아닌 실질적 개선)
    4. 부채비율 200% 미만 (부실기업 제외)

출력: data/turnaround_candidates.json

실행:
  python -u -X utf8 scripts/scan_turnaround.py
  python -u -X utf8 scripts/scan_turnaround.py --send
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TA] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("turnaround")

DATA_DIR = PROJECT_ROOT / "data"
FINANCIAL_PATH = DATA_DIR / "v2_migration" / "financial_quarterly.json"
OUTPUT_PATH = DATA_DIR / "turnaround_candidates.json"


def _load_name_map() -> dict[str, str]:
    """종목명 캐시 (data/universe/name_map.json)"""
    cache_path = DATA_DIR / "universe" / "name_map.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_financials() -> dict:
    """financial_quarterly.json → {ticker: {bs: {q: {op_income, revenue}}, quality: {...}, name}} 변환

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
                continue
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


def screen_turnaround(ticker: str, data: dict) -> dict | None:
    """턴어라운드 스크리닝"""
    bs = data.get("bs", {})
    quality = data.get("quality", {})
    name = data.get("name", ticker)

    quarters = sorted(bs.keys())
    if len(quarters) < 3:
        return None

    # 최근 3분기 영업이익
    q_latest = quarters[-1]
    q_prev = quarters[-2]
    q_prev2 = quarters[-3]

    oi_latest = bs.get(q_latest, {}).get("op_income")
    oi_prev = bs.get(q_prev, {}).get("op_income")
    oi_prev2 = bs.get(q_prev2, {}).get("op_income")

    if oi_latest is None or oi_prev is None or oi_prev2 is None:
        return None

    # 조건 1: 직전 2분기 적자
    if not (oi_prev < 0 and oi_prev2 < 0):
        return None

    # 조건 2: 흑자전환 OR 적자폭 50%+ 축소
    is_turnaround = oi_latest > 0
    deficit_reduction = 0.0
    if not is_turnaround:
        if oi_prev < 0 and oi_latest < 0:
            deficit_reduction = 1 - abs(oi_latest) / abs(oi_prev) if abs(oi_prev) > 0 else 0
            if deficit_reduction < 0.5:
                return None  # 적자폭 50% 미만 축소
        else:
            return None

    # 조건 3: 매출액 비감소
    rev_latest = bs.get(q_latest, {}).get("revenue")
    rev_prev = bs.get(q_prev, {}).get("revenue")
    if rev_latest and rev_prev and rev_latest < rev_prev * 0.9:
        return None  # 매출 10%+ 감소 → 구조조정

    # 조건 4: 부채비율 200% 미만
    # quality는 종목 단위 (분기별 아님), debt_ratio는 소수점 비율 (0.21 = 21%)
    debt_ratio_raw = quality.get("debt_ratio")
    debt_ratio = debt_ratio_raw * 100 if debt_ratio_raw is not None else None
    if debt_ratio and debt_ratio > 200:
        return None

    # 흑자전환 예상 시기
    if is_turnaround:
        turnaround_type = "STRONG"
        est_turnaround = q_latest
    else:
        turnaround_type = "EARLY"
        # 적자 축소 속도 → 예상 흑자전환 시기
        if oi_prev < 0 and oi_latest < 0:
            improvement_per_q = abs(oi_prev) - abs(oi_latest)
            if improvement_per_q > 0:
                quarters_to_zero = abs(oi_latest) / improvement_per_q
                est_turnaround = f"약 {quarters_to_zero:.0f}분기 후"
            else:
                est_turnaround = "불확실"
        else:
            est_turnaround = "불확실"

    # 스코어
    if is_turnaround:
        score = 90
        if rev_latest and rev_prev and rev_latest > rev_prev:
            score += 5  # 매출도 증가
    else:
        score = 50 + deficit_reduction * 30

    return {
        "ticker": ticker,
        "name": name,
        "turnaround_type": turnaround_type,
        "score": round(score, 1),
        "op_income_q2": oi_prev2,
        "op_income_q1": oi_prev,
        "op_income_latest": oi_latest,
        "deficit_reduction_pct": round(deficit_reduction * 100, 1) if not is_turnaround else None,
        "revenue_latest": rev_latest,
        "revenue_prev": rev_prev,
        "debt_ratio": round(debt_ratio, 1) if debt_ratio else None,
        "est_turnaround": est_turnaround,
        "quarters": [q_prev2, q_prev, q_latest],
    }


def main():
    parser = argparse.ArgumentParser(description="턴어라운드 스크리너")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    financials = load_financials()
    if not financials:
        print("재무 데이터 없음. 종료.")
        return

    candidates = []
    for ticker, data in financials.items():
        r = screen_turnaround(ticker, data)
        if r:
            candidates.append(r)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    strong = [c for c in candidates if c["turnaround_type"] == "STRONG"]
    early = [c for c in candidates if c["turnaround_type"] == "EARLY"]

    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "total_screened": len(financials),
        "candidates_found": len(candidates),
        "strong_count": len(strong),
        "early_count": len(early),
        "strong": strong,
        "early": early,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    print(f"\n{'='*60}")
    print(f"  턴어라운드 스크리닝 — {output['date']}")
    print(f"  스크리닝: {output['total_screened']}종목 → 후보: {output['candidates_found']}종목")
    print(f"{'='*60}")

    if strong:
        print(f"\n🔥 적자→흑자 전환 완료 (STRONG) — {len(strong)}종목")
        for c in strong[:15]:
            oi = c["op_income_latest"]
            oi_str = f"{oi/1e8:.0f}억" if abs(oi) >= 1e8 else f"{oi/1e4:.0f}만"
            prev_oi = c["op_income_q1"]
            prev_str = f"{prev_oi/1e8:.0f}억" if abs(prev_oi) >= 1e8 else f"{prev_oi/1e4:.0f}만"
            debt_str = f"부채 {c['debt_ratio']:.0f}%" if c['debt_ratio'] else ""
            print(f"  {c['name']:12s} ({c['ticker']}) {prev_str}→{oi_str} | {debt_str} | 점수 {c['score']:.0f}")

    if early:
        print(f"\n📈 적자 축소 중 (EARLY) — {len(early)}종목")
        for c in early[:10]:
            oi = c["op_income_latest"]
            oi_str = f"{oi/1e8:.0f}억" if abs(oi) >= 1e8 else f"{oi/1e4:.0f}만"
            red_str = f"축소 {c['deficit_reduction_pct']:.0f}%" if c['deficit_reduction_pct'] else ""
            print(f"  {c['name']:12s} ({c['ticker']}) 영업이익 {oi_str} | {red_str} | 예상 {c['est_turnaround']}")

    print(f"\n[저장] {OUTPUT_PATH}")

    # 텔레그램
    if args.send:
        try:
            from src.adapters.telegram_adapter import send_message
            lines = [f"🔄 턴어라운드 스크리닝 — {output['date']}"]
            lines.append(f"후보: {output['candidates_found']}종목 (STRONG {len(strong)} / EARLY {len(early)})")
            for c in strong[:5]:
                lines.append(f"🔥 {c['name']} — 흑자전환! {c['op_income_latest']/1e8:.0f}억")
            for c in early[:3]:
                lines.append(f"📈 {c['name']} — 적자축소 {c['deficit_reduction_pct']:.0f}%")
            send_message("\n".join(lines))
            print("[텔레그램] 발송 완료")
        except Exception as e:
            print(f"[텔레그램] 발송 실패: {e}")


if __name__ == "__main__":
    main()
