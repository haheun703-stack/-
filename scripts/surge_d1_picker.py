"""차트영웅 매매법 5-Gate 종목 선정기 — 5/22 paper mirror 핵심.

매일 15:30 (장 마감 후) 호출 → 다음날 D+1 양봉 확인 후 진입할 후보 1~3종목 산출.

5-Gate 흐름:
  Gate 1: 매크로 4-시그널 3/4 GO       (four_signal_gate)
       ↓ 미통과 시 즉시 종료 (매수 후보 0)
  Gate 2: 어제 상한가/급등 종목 풀     (surge_pullback Supabase or KIS)
       ↓
  Gate 3: 저변동 + MA20 눌림 필터       (indicators.py)
       ↓
  Gate 4-A: catalyst 연속성 ≥ 60      (perplexity_catalyst 또는 정보봇 catalyst)
  Gate 4-B: 목표가 상승률 ≥ 30%        (analyst_target_collector 시드/Perplexity)
       ↓ (둘 중 하나라도 만족하면 통과)
  Gate 5: 종목 주봉 %K < 30            (kis_weekly_kit)
       ↓
  최종 후보 (1~3종목, buy_score 기준 정렬)

다음날 09:00~15:00:
  D+1 양봉 확인 → 종가 예약 매수 (1차 1.0%)
  (별도 모듈: scripts/d1_confirm_executor.py — 5/21 작성)
"""

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.macro.four_signal_gate import compute_four_signal_gate
from src.macro.chart_hero_advisory import build_chart_hero_advisory
from src.adapters.kis_weekly_kit import get_stock_weekly, compute_weekly_stoch_k
from src.intel.analyst_target_collector import load_seed_csv, find_high_upside_picks
from src.intel.perplexity_catalyst import analyze_catalyst, compute_continuity_score


# === 임계값 (긴장 타입) ===
MIN_CONTINUITY_SCORE = 60       # Gate 4-A
MIN_UPSIDE_PCT       = 30.0     # Gate 4-B
MAX_WEEKLY_K         = 30.0     # Gate 5
MAX_PICKS            = 3        # 일 최대 진입 종목 수


def gate_1_macro(today: str) -> dict:
    """Gate 1: 매크로 4-시그널."""
    g = compute_four_signal_gate(today)
    adv = build_chart_hero_advisory(today)
    return {
        "passed": g["gate_pass"],
        "score": g["gate_score"],
        "reason": g["reason"],
        "advisory_mode": adv["mode"],
        "advisory_summary": adv["summary"],
    }


def gate_2_surge_pool(today: str, min_surge_pct: float = 25.0) -> list[dict]:
    """Gate 2: 어제 급등 종목 풀 (상한가/급등).

    1차: Supabase quant_surge_pullback (정보봇 의존 X, 우리 자체)
    2차: 사장님 시드 데이터 (5/19 시점)

    TODO 5/21: Supabase 직접 조회 어댑터 작성
    """
    # PLACEHOLDER: 5/19 시드 32종목을 surge_pool로 사용 (백테스트용)
    # 5/22 실전에는 Supabase quant_surge_pullback에서 어제(D-1) 상한가 풀 가져옴
    seed = load_seed_csv()
    pool = []
    for s in seed:
        if not s.get("ticker"):
            continue
        pool.append({
            "ticker": s["ticker"],
            "name": s["name"],
            "current_price": s.get("current_price", 0),
            "source": "seed_5_19_target_uplift",  # 5/19 목표가 상향 = D-1 강한 종목 proxy
        })
    return pool


def gate_3_pullback_filter(pool: list[dict]) -> list[dict]:
    """Gate 3: 저변동 + MA20 눌림 (-3~0%).

    TODO 5/21: 일봉 데이터로 MA20 dev 계산 (정보봇 OHLCV 또는 KIS)
    현재는 PLACEHOLDER: 전부 통과 (실제 필터는 5/21 보강)
    """
    # PLACEHOLDER
    for p in pool:
        p["ma20_dev_pct"] = None
        p["passed_gate3"] = True
    return pool


def gate_4a_catalyst(pool: list[dict], analyze_all: bool = False) -> list[dict]:
    """Gate 4-A: catalyst 연속성 ≥ 60.

    Args:
        analyze_all: True면 Perplexity로 전체 분석 (비용 발생)
                     False면 시드 또는 정보봇 quant_surge_catalyst 활용 (기본)
    """
    out = []
    for p in pool:
        if analyze_all:
            c = analyze_catalyst(p["ticker"], p["name"])
            cs = compute_continuity_score(c)
            p["catalyst_category"] = c.get("catalyst_category", "기타")
            p["catalyst_summary"] = c.get("catalyst_summary", "")
            p["continuity_score"] = cs
            p["is_one_off_event"] = c.get("is_one_off_event", False)
        else:
            # PLACEHOLDER: 정보봇 quant_surge_catalyst 조회 자리
            p["catalyst_category"] = None
            p["catalyst_summary"] = None
            p["continuity_score"] = None  # 정보봇 응답 대기
            p["is_one_off_event"] = None

        # 통과 기준
        if p.get("continuity_score") is not None:
            p["passed_gate4a"] = p["continuity_score"] >= MIN_CONTINUITY_SCORE
        else:
            p["passed_gate4a"] = None  # 데이터 부재 시 None
        out.append(p)
    return out


def gate_4b_upside(pool: list[dict]) -> list[dict]:
    """Gate 4-B: 목표가 상승률 ≥ 30% (analyst_target 시드 또는 자동 수집)."""
    seed = load_seed_csv()
    target_map = {t["ticker"]: t for t in seed if t.get("ticker")}
    for p in pool:
        t = target_map.get(p["ticker"])
        if t:
            up = t.get("upside_pct", 0)
            p["upside_pct"] = up
            p["target_broker"] = t.get("broker", "")
            p["passed_gate4b"] = up >= MIN_UPSIDE_PCT
        else:
            p["upside_pct"] = None
            p["passed_gate4b"] = None
    return pool


def gate_5_weekly_stoch(pool: list[dict]) -> list[dict]:
    """Gate 5: 종목 주봉 %K < 30 (과매도)."""
    end_dt = dt.date.today()
    start_dt = end_dt - dt.timedelta(days=200)
    out = []
    for p in pool:
        weekly = get_stock_weekly(p["ticker"],
                                   start_dt.strftime("%Y%m%d"),
                                   end_dt.strftime("%Y%m%d"))
        k = compute_weekly_stoch_k(weekly) if weekly else None
        p["weekly_k"] = k
        p["passed_gate5"] = (k is not None and k < MAX_WEEKLY_K)
        out.append(p)
    return out


def compute_buy_score(pick: dict) -> float:
    """후보 점수 (정렬용). 5-Gate 통과 후 우선순위 결정.

    공식:
      + continuity_score (0~100) × 0.4
      + upside_pct (0~150 cap) × 0.3
      + (30 - weekly_k) × 1.0 (낮을수록 좋음)
      + (1 if not is_one_off_event else -50)
    """
    cs = pick.get("continuity_score") or 50  # 정보봇 미응답 시 중립값
    up = min(pick.get("upside_pct") or 0, 150)
    wk = pick.get("weekly_k") or 50
    one_off = pick.get("is_one_off_event") or False
    return round(cs * 0.4 + up * 0.3 + (30 - wk) * 1.0 + (-50 if one_off else 0), 2)


def run_picker(today: str | None = None,
                analyze_catalyst_live: bool = False) -> dict:
    """5-Gate 종목 선정 메인 함수.

    Args:
        today: 'YYYY-MM-DD'
        analyze_catalyst_live: True면 Perplexity catalyst 실시간 분석 (비용 발생)
    """
    today = today or dt.date.today().isoformat()

    # === Gate 1 ===
    g1 = gate_1_macro(today)
    if not g1["passed"]:
        return {
            "date": today,
            "status": "GATE1_BLOCKED",
            "gate_1": g1,
            "picks": [],
            "summary": f"매크로 게이트 차단 ({g1['score']}/4). 신규 진입 X.",
        }

    # === Gate 2 ===
    pool = gate_2_surge_pool(today)
    if not pool:
        return {"date": today, "status": "GATE2_EMPTY", "gate_1": g1, "picks": [],
                "summary": "어제 상한가 종목 풀 없음."}

    # === Gate 3 ===
    pool = gate_3_pullback_filter(pool)
    pool = [p for p in pool if p.get("passed_gate3")]

    # === Gate 4-B (먼저) — 빠른 필터로 후보 축소 ===
    pool = gate_4b_upside(pool)
    pool = [p for p in pool if p.get("passed_gate4b")]

    # === Gate 4-A (catalyst) ===
    pool = gate_4a_catalyst(pool, analyze_all=analyze_catalyst_live)
    # Gate 4-A는 정보봇 데이터 없으면 None — 일단 통과로 간주 (5/21 정보봇 통합 후 강제)

    # === Gate 5 (주봉 스토캐스틱) ===
    pool = gate_5_weekly_stoch(pool)
    pool = [p for p in pool if p.get("passed_gate5")]

    # 점수 계산 + 정렬
    for p in pool:
        p["buy_score"] = compute_buy_score(p)
    pool.sort(key=lambda x: -x["buy_score"])
    final = pool[:MAX_PICKS]

    return {
        "date": today,
        "status": "OK" if final else "NO_FINAL_PICKS",
        "gate_1": g1,
        "picks": final,
        "summary": f"최종 {len(final)}종목 선정 (4-시그널 {g1['score']}/4 GO)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--live-catalyst", action="store_true",
                        help="Perplexity catalyst 실시간 분석 ($)")
    args = parser.parse_args()

    result = run_picker(args.date, args.live_catalyst)

    print(f"=== 차트영웅 5-Gate 선정기 ({result['date']}) ===\n")
    print(f"상태: {result['status']}")
    print(f"요약: {result['summary']}\n")
    print(f"--- Gate 1 매크로 ---")
    print(f"  점수: {result['gate_1']['score']}/4, 통과: {result['gate_1']['passed']}")
    print(f"  Advisory: {result['gate_1']['advisory_summary']}\n")
    if result["picks"]:
        print(f"--- 최종 진입 후보 ({len(result['picks'])}종목) ---")
        for p in result["picks"]:
            print(f"  ⭐ {p['name']:14} ({p['ticker']}) "
                  f"score={p['buy_score']:>5.1f} "
                  f"upside={p.get('upside_pct')}% "
                  f"weekly_k={p.get('weekly_k')}")
    else:
        print("--- 후보 없음 (Gate 차단 또는 데이터 부족) ---")
