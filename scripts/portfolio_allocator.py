"""멀티전략 포트폴리오 배분 — Timefolio 벤치마크 전략 2

5개 서브전략(모멘텀/수급추종/이벤트/섹터순환/안전마진)에
KOSPI 레짐 기반 동적 가중치를 배분한다.

입력:
  - data/kospi_regime.json
  - data/regime_macro_signal.json (전략1 출력)
  - data/tomorrow_picks.json (최종 추천 종목)
  - data/scan_cache.json (퀀텀 시그널)
  - data/dual_buying_watch.json (동반매수)
  - data/force_hybrid.json (세력감지)
  - data/dart_event_signals.json (전략3 출력)
  - data/sector_rotation/relay_trading_signal.json
  - data/group_relay/group_relay_today.json
  - data/pullback_scan.json

출력:
  - data/portfolio_allocation.json

Usage:
    python scripts/portfolio_allocator.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "portfolio_allocation.json"

# 레짐별 배분 비율 (%)
REGIME_WEIGHTS = {
    "BULL":    {"모멘텀": 35, "수급추종": 25, "이벤트": 15, "섹터순환": 20, "안전마진": 5},
    "CAUTION": {"모멘텀": 25, "수급추종": 25, "이벤트": 15, "섹터순환": 20, "안전마진": 15},
    "BEAR":    {"모멘텀": 10, "수급추종": 20, "이벤트": 20, "섹터순환": 15, "안전마진": 35},
    "CRISIS":  {"모멘텀": 0,  "수급추종": 10, "이벤트": 15, "섹터순환": 10, "안전마진": 65},
}

MAX_SINGLE_STOCK_PCT = 15  # 단일 종목 최대 비중


def load_json(rel_path: str) -> dict | list:
    fp = DATA_DIR / rel_path
    if not fp.exists():
        return {}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def collect_momentum_stocks() -> list[dict]:
    """모멘텀 전략: 퀀텀 시그널 통과 종목"""
    q = load_json("scan_cache.json")
    results = []
    for c in q.get("candidates", []):
        ticker = c.get("ticker", "")
        if ticker:
            results.append({
                "ticker": ticker,
                "name": c.get("name", ""),
                "score": 90,
                "source": "퀀텀통과",
            })
    # kill 중 R:R >= 2.0도 추가
    for k in q.get("stats", {}).get("v9_killed_list", []):
        rr = k.get("risk_reward", 0)
        if rr >= 2.0:
            results.append({
                "ticker": k.get("ticker", ""),
                "name": k.get("name", ""),
                "score": 50 + min(rr * 10, 30),
                "source": f"Kill(R:R {rr:.1f})",
            })
    return sorted(results, key=lambda x: -x["score"])


def collect_flow_stocks() -> list[dict]:
    """수급추종 전략: 동반매수 + 세력감지"""
    results = []

    # 동반매수
    db = load_json("dual_buying_watch.json")
    for grade_key, label, base in [("s_grade", "S등급", 80), ("a_grade", "A등급", 65)]:
        for item in db.get(grade_key, []):
            ticker = item.get("ticker", "")
            if ticker:
                results.append({
                    "ticker": ticker,
                    "name": item.get("name", ""),
                    "score": base,
                    "source": f"동반매수_{label}",
                })

    # 세력감지
    fh = load_json("force_hybrid.json")
    for item in fh.get("detected", []):
        grade = item.get("grade", "")
        if grade in ("세력포착", "매집의심"):
            ticker = item.get("ticker", "")
            if ticker:
                results.append({
                    "ticker": ticker,
                    "name": item.get("name", ""),
                    "score": 70 if grade == "세력포착" else 55,
                    "source": f"세력감지_{grade}",
                })

    return sorted(results, key=lambda x: -x["score"])


def collect_event_stocks() -> list[dict]:
    """이벤트 전략: DART 이벤트 BUY 시그널"""
    de = load_json("dart_event_signals.json")
    results = []
    for sig in de.get("signals", []):
        if sig.get("action") == "BUY":
            results.append({
                "ticker": sig.get("ticker", ""),
                "name": sig.get("name", ""),
                "score": sig.get("event_score", 0),
                "source": f"DART_{sig.get('event', '')}",
            })
    return sorted(results, key=lambda x: -x["score"])


def collect_sector_stocks() -> list[dict]:
    """섹터순환 전략: 릴레이 + 그룹순환"""
    results = []

    # 섹터 릴레이
    relay = load_json("sector_rotation/relay_trading_signal.json")
    for sig in relay.get("signals", []):
        for p in sig.get("picks", []):
            ticker = p.get("ticker", "")
            if ticker:
                results.append({
                    "ticker": ticker,
                    "name": p.get("name", ""),
                    "score": p.get("score", 0),
                    "source": "릴레이",
                })

    # 그룹순환
    gr = load_json("group_relay/group_relay_today.json")
    for g in gr.get("fired_groups", []):
        for w in g.get("waiting_subsidiaries", []):
            ticker = w.get("ticker", "")
            if ticker:
                results.append({
                    "ticker": ticker,
                    "name": w.get("name", ""),
                    "score": w.get("score", 0) or w.get("composite_score", 0),
                    "source": f"그룹순환_{g.get('group_name','')}",
                })

    return sorted(results, key=lambda x: -x["score"])


def collect_safety_stocks() -> list[dict]:
    """안전마진 전략: 눌림목 반등임박/매수대기"""
    pb = load_json("pullback_scan.json")
    results = []
    for item in pb.get("candidates", pb.get("items", [])):
        grade = item.get("grade", "")
        if grade in ("반등임박", "매수대기"):
            ticker = item.get("ticker", "")
            if ticker:
                results.append({
                    "ticker": ticker,
                    "name": item.get("name", ""),
                    "score": item.get("score", 0),
                    "source": f"눌림목_{grade}",
                })
    return sorted(results, key=lambda x: -x["score"])


def allocate(regime: str, macro_grade: str, position_mult: float) -> dict:
    """레짐 기반 포트폴리오 배분"""
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["CAUTION"])

    # 각 전략별 종목 수집
    strategy_stocks = {
        "모멘텀": collect_momentum_stocks(),
        "수급추종": collect_flow_stocks(),
        "이벤트": collect_event_stocks(),
        "섹터순환": collect_sector_stocks(),
        "안전마진": collect_safety_stocks(),
    }

    # 비중 0%인 전략은 종목 배분 안 함
    allocations = {}
    total_positions = 0
    used_tickers = set()  # 중복 방지

    for strategy, weight in weights.items():
        if weight <= 0:
            allocations[strategy] = {"weight": 0, "count": 0, "stocks": []}
            continue

        stocks = strategy_stocks.get(strategy, [])
        # 중복 제거: 이미 다른 전략에 배정된 종목은 제외
        unique_stocks = [s for s in stocks if s["ticker"] not in used_tickers]

        # 비중에 비례하여 종목 수 배분 (최소 1, 최대 5)
        n_stocks = max(1, min(5, weight // 10))

        selected = unique_stocks[:n_stocks]
        for s in selected:
            used_tickers.add(s["ticker"])

        allocations[strategy] = {
            "weight": weight,
            "count": len(selected),
            "stocks": [
                {
                    "ticker": s["ticker"],
                    "name": s["name"],
                    "score": s["score"],
                    "source": s["source"],
                }
                for s in selected
            ],
        }
        total_positions += len(selected)

    # 현금 비율: CRISIS면 65%, 나머지는 0%
    cash_reserve = 0
    if regime == "CRISIS":
        cash_reserve = 65
    elif regime == "BEAR":
        cash_reserve = 20

    return {
        "allocations": allocations,
        "total_positions": total_positions,
        "cash_reserve_pct": cash_reserve,
    }


def main():
    logger.info("=" * 60)
    logger.info("  멀티전략 포트폴리오 배분 — 전략 2")
    logger.info("=" * 60)

    # 레짐 로드
    regime_data = load_json("kospi_regime.json")
    regime = regime_data.get("regime", "CAUTION")

    # 매크로 시그널 로드
    macro = load_json("regime_macro_signal.json")
    macro_grade = macro.get("macro_grade", "보통")
    position_mult = macro.get("position_multiplier", 1.0)

    logger.info("  레짐: %s, 매크로: %s (x%.1f)", regime, macro_grade, position_mult)

    # 배분 실행
    result = allocate(regime, macro_grade, position_mult)

    # 출력 구성
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "macro_grade": macro_grade,
        "position_multiplier": position_mult,
        "total_budget_pct": 100,
        "max_single_stock_pct": MAX_SINGLE_STOCK_PCT,
        "cash_reserve_pct": result["cash_reserve_pct"],
        "total_positions": result["total_positions"],
        "allocations": result["allocations"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 결과 출력
    logger.info("── 포트폴리오 배분 결과 ──")
    for strategy, alloc in result["allocations"].items():
        w = alloc["weight"]
        n = alloc["count"]
        tickers = ", ".join(s["ticker"] for s in alloc["stocks"][:3])
        if n > 0:
            logger.info("  %s (%d%%): %d종목 — %s", strategy, w, n, tickers)
        else:
            logger.info("  %s (%d%%): 배분 없음", strategy, w)

    logger.info("  총 포지션: %d종목, 현금비중: %d%%",
                result["total_positions"], result["cash_reserve_pct"])
    logger.info("  저장: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
