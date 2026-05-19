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
from src.adapters.kis_weekly_kit import get_stock_weekly, compute_weekly_stoch_k, compute_ma20_dev
from src.adapters.quant_supabase_reader import (
    get_yesterday_surge_pool, get_sector_picks_today, get_catalyst, get_company_card,
    get_chart_hero_d1_candidates, get_catalyst_batch,
)
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
    """Gate 2: 오늘(D0) 상한가/급등 종목 풀.

    시간 흐름 (정보봇 가이드 5/19 반영):
      D0 16:30  정보봇 quant_surge_catalyst 작성 (catalyst_filled=True)
      D0 17:00  본 picker 실행 → 다음날 진입 후보 저장
      D+1 14:55 d1_confirm 양봉 확인 → 진입

    1차: 정보봇 quant_surge_catalyst (D0 상한가+catalyst 분석 완료)
    2차: quant_surge_pullback (어제 D-1 데이터, 보강용)
    3차: 시드 CSV (fallback)
    """
    import datetime as dt
    yesterday = (dt.date.fromisoformat(today) - dt.timedelta(days=1)).isoformat()
    pool = []

    # 1차: 정보봇 D0 catalyst (limit_up/strong, 일회성 X, continuity ≥ 40)
    candidates = get_chart_hero_d1_candidates(today, min_continuity=40)
    for c in candidates:
        sp = c.get("surge_pct", 0)
        if sp and sp > 50:    # 데이터 이상치 필터
            continue
        pool.append({
            "ticker": c.get("ticker"),
            "name": c.get("name", ""),
            "current_price": c.get("surge_price", 0),
            "sector": c.get("sector", ""),
            "surge_pct": sp,
            "surge_type": c.get("surge_type"),
            "source": "jgis_chart_hero_catalyst",
            # 정보봇 catalyst 정보 미리 inject (Gate 4-A에서 재확인)
            "catalyst_summary": c.get("catalyst_summary"),
            "catalyst_category": c.get("catalyst_category"),
            "continuity_score": c.get("continuity_score"),
            "is_one_off_event": c.get("is_one_off_event"),
            "catalyst_source": "jgis_supabase",
        })

    # 2차: 어제 surge_pullback 보강 (정보봇 catalyst 미커버 종목)
    if len(pool) < 10:
        existing_tk = {p["ticker"] for p in pool}
        surges = get_yesterday_surge_pool(yesterday, min_surge_pct)
        for s in surges:
            tk = s.get("ticker")
            sp = s.get("surge_pct", 0)
            if not tk or tk in existing_tk or (sp and sp > 50):
                continue
            pool.append({
                "ticker": tk,
                "name": s.get("name", ""),
                "current_price": s.get("surge_close", 0) or s.get("latest_close", 0),
                "sector": s.get("sector", ""),
                "surge_pct": sp,
                "source": "quant_surge_pullback",
            })

    # 3차: fallback
    if not pool:
        seed = load_seed_csv()
        for s in seed:
            if not s.get("ticker"):
                continue
            pool.append({
                "ticker": s["ticker"], "name": s["name"],
                "current_price": s.get("current_price", 0),
                "source": "seed_5_19_target_uplift",
            })

    return pool


def gate_3_pullback_filter(pool: list[dict], today: str) -> list[dict]:
    """Gate 3: MA20 눌림 (-3~0%) + 수급 정보 enrichment.

    quant_sector_picks에 BAT-D가 매일 계산한 ma20_dev / rsi / 외인기관 수급 컬럼 있음.
    pool의 ticker로 매칭해서 enrich + 필터.

    차트영웅 룰: ma20_dev -3 ~ 0% (눌림 영역 = 이상적 진입)
    """
    picks_today = get_sector_picks_today(today)
    enrich_map = {p["ticker"]: p for p in picks_today if p.get("ticker")}

    out = []
    for p in pool:
        enrich = enrich_map.get(p["ticker"])
        if enrich:
            p["ma20_dev_pct"] = enrich.get("ma20_dev")
            p["rsi"] = enrich.get("rsi")
            p["vol_ratio"] = enrich.get("vol_ratio")
            p["fgn_5d"] = enrich.get("fgn_5d")
            p["inst_5d"] = enrich.get("inst_5d")
            p["pension_5d"] = enrich.get("pension_5d")
            p["buy_score_bat"] = enrich.get("buy_score")
        else:
            # sector_picks 미매칭 → KIS 일봉 직접 계산 (1 API 호출)
            try:
                p["ma20_dev_pct"] = compute_ma20_dev(p["ticker"])
            except Exception:
                p["ma20_dev_pct"] = None

        # 차트영웅 룰 완화 버전: -5 ≤ ma20_dev ≤ +5% (저변동 + 미세 눌림)
        dev = p["ma20_dev_pct"]
        if dev is not None and -5 <= dev <= 5:
            p["passed_gate3"] = True
        else:
            p["passed_gate3"] = False
        out.append(p)
    return out


def gate_4a_catalyst(pool: list[dict], today: str,
                     fallback_to_perplexity: bool = False) -> list[dict]:
    """Gate 4-A: 정보봇 quant_surge_catalyst → catalyst 연속성 점수.

    1차: 정보봇 Supabase (5/19~ 매일 16:30 갱신, get_catalyst_batch)
    2차: Perplexity 자체 분석 (fallback, 비용 발생)

    통과 기준: continuity_score ≥ 40 (정보봇 권장) OR ≥ MIN_CONTINUITY_SCORE(60, 우리 자체)
    """
    if not pool:
        return []
    tickers = [p["ticker"] for p in pool if p.get("ticker")]
    catalyst_map = get_catalyst_batch(today, tickers)

    out = []
    for p in pool:
        c = catalyst_map.get(p["ticker"])
        if c:
            p["catalyst_category"] = c.get("catalyst_category")
            p["catalyst_summary"] = c.get("catalyst_summary")
            p["continuity_score"] = c.get("continuity_score")
            p["is_one_off_event"] = c.get("is_one_off_event")
            p["surge_type"] = c.get("surge_type")
            p["smart_money_5d_pct"] = c.get("smart_money_5d_pct")
            p["news_count_5d"] = c.get("news_count_5d")
            p["catalyst_source"] = "jgis_supabase"
        elif fallback_to_perplexity:
            # Perplexity fallback
            pc = analyze_catalyst(p["ticker"], p["name"])
            p["catalyst_category"] = pc.get("catalyst_category")
            p["catalyst_summary"] = pc.get("catalyst_summary")
            p["continuity_score"] = compute_continuity_score(pc)
            p["is_one_off_event"] = pc.get("is_one_off_event")
            p["catalyst_source"] = "perplexity_fallback"
        else:
            p["catalyst_category"] = None
            p["continuity_score"] = None
            p["is_one_off_event"] = None
            p["catalyst_source"] = "none"

        # 통과 기준: 정보봇 40 ≥ (정보봇 권장) OR 우리 60 ≥ (Perplexity 자체)
        cs = p.get("continuity_score")
        one_off = p.get("is_one_off_event")
        if cs is None:
            p["passed_gate4a"] = None   # 데이터 부재
        elif one_off:
            p["passed_gate4a"] = False  # 일회성 이슈 제외
        else:
            p["passed_gate4a"] = cs >= 40   # 정보봇 권장 기준 채택
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
    pool = gate_3_pullback_filter(pool, today)
    pool = [p for p in pool if p.get("passed_gate3")]

    # === Gate 4-B (먼저) — 빠른 필터로 후보 축소 ===
    pool = gate_4b_upside(pool)
    pool = [p for p in pool if p.get("passed_gate4b")]

    # === Gate 4-A (catalyst, 정보봇 우선) ===
    pool = gate_4a_catalyst(pool, today, fallback_to_perplexity=analyze_catalyst_live)
    # passed_gate4a is False면 제외, None(데이터 부재)은 통과로 간주 (백업)
    pool = [p for p in pool if p.get("passed_gate4a") is not False]

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
