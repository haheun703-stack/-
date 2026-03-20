"""
NXT 추천 결과 추적 + 학습 — BAT-D 연동

매일 장마감 후 실행:
  1. 이전 NXT 추천(nxt_picks.json) 로드
  2. 당일 실제 시가/종가/고가 대비 성과 측정
  3. 히스토리 누적 (nxt_accuracy.json)
  4. 학습 가중치 업데이트 → nxt_recommend.py에 피드백

사용: python -u -X utf8 scripts/nxt_track_results.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
NXT_DIR = DATA_DIR / "nxt"
PICKS_PATH = NXT_DIR / "nxt_picks.json"
ACCURACY_PATH = NXT_DIR / "nxt_accuracy.json"
WEIGHTS_PATH = NXT_DIR / "nxt_learning_weights.json"
PROCESSED_DIR = DATA_DIR / "processed"


def _load_accuracy() -> dict:
    """누적 정확도 히스토리 로드."""
    if ACCURACY_PATH.exists():
        return json.loads(ACCURACY_PATH.read_text(encoding="utf-8"))
    return {"records": [], "stats": {}}


def _save_accuracy(data: dict):
    """정확도 히스토리 저장."""
    NXT_DIR.mkdir(parents=True, exist_ok=True)
    ACCURACY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_ohlcv(ticker: str, target_date: str) -> dict | None:
    """parquet에서 특정 날짜 OHLCV 가져오기."""
    pq_path = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        # 날짜 매칭 (DatetimeIndex)
        target_ts = pd.Timestamp(target_date)
        row = df.loc[df.index == target_ts]
        if row.empty:
            return None
        r = row.iloc[0]
        # 소문자/대문자 컬럼 모두 지원
        def _col(name):
            return float(r.get(name, r.get(name.lower(), 0)))
        return {
            "open": _col("Open"),
            "high": _col("High"),
            "low": _col("Low"),
            "close": _col("Close"),
            "volume": int(_col("Volume")),
        }
    except Exception:
        return None


def track_results(eval_date: str | None = None) -> dict:
    """
    NXT 추천 결과 추적.

    eval_date: 평가할 날짜 (당일). NXT 추천은 전일 생성 → 오늘 결과 확인.
    """
    today = eval_date or date.today().isoformat()

    # NXT 추천 로드
    if not PICKS_PATH.exists():
        logger.warning("nxt_picks.json 없음 → 추적 스킵")
        return {}

    picks_data = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
    picks = picks_data.get("picks", [])
    picks_date = picks_data.get("date", "")

    if not picks:
        logger.info("NXT 추천 0건 → 추적 스킵")
        return {}

    # 이미 추적된 날짜인지 확인
    accuracy = _load_accuracy()
    tracked_dates = {r["eval_date"] for r in accuracy["records"]}
    if today in tracked_dates:
        logger.info("이미 추적됨: %s → 스킵", today)
        return accuracy

    logger.info("=== NXT 추천 결과 추적 (%s 추천 → %s 평가) ===", picks_date, today)

    results = []
    for p in picks:
        ticker = p["ticker"]
        name = p.get("name", ticker)
        nxt_grade = p.get("nxt_grade", "")
        nxt_price = p.get("nxt_last_price", 0)
        prev_close = p.get("prev_close", 0)
        suggested = p.get("suggested_price", 0)

        # 당일 실제 시세 가져오기
        ohlcv = _get_ohlcv(ticker, today)
        if not ohlcv:
            logger.debug("  %s(%s): 시세 없음", name, ticker)
            continue

        actual_open = ohlcv["open"]
        actual_close = ohlcv["close"]
        actual_high = ohlcv["high"]

        # 수익률 계산
        base_price = suggested if suggested > 0 else (nxt_price if nxt_price > 0 else prev_close)
        if base_price <= 0:
            continue

        # 시가 대비 수익률 (NXT 매수 → 다음날 시가에 매도)
        open_return = round((actual_open / base_price - 1) * 100, 2)
        # 종가 대비 수익률 (NXT 매수 → 다음날 종가에 매도)
        close_return = round((actual_close / base_price - 1) * 100, 2)
        # 고가 대비 (최대 수익 가능)
        high_return = round((actual_high / base_price - 1) * 100, 2)

        # 성공 판정: 종가 기준 +0.5% 이상이면 성공
        is_win = close_return >= 0.5

        result = {
            "ticker": ticker,
            "name": name,
            "nxt_grade": nxt_grade,
            "total_score": p.get("total_score", 0),
            "base_price": int(base_price),
            "actual_open": int(actual_open),
            "actual_close": int(actual_close),
            "actual_high": int(actual_high),
            "open_return_pct": open_return,
            "close_return_pct": close_return,
            "high_return_pct": high_return,
            "is_win": is_win,
            "nxt_premium_pct": p.get("nxt_premium_pct", 0),
            "nxt_net_buy_ratio": p.get("nxt_net_buy_ratio", 0.5),
            "had_nxt_data": p.get("has_nxt_data", False),
        }
        results.append(result)

        icon = "✅" if is_win else "❌"
        logger.info(
            "  %s %s(%s) [%s] 매수%d→종가%d (%+.1f%%)",
            icon, name, ticker, nxt_grade,
            int(base_price), int(actual_close), close_return,
        )

    if not results:
        logger.info("  평가 가능 종목 0건")
        return accuracy

    # 당일 기록 추가
    record = {
        "picks_date": picks_date,
        "eval_date": today,
        "total": len(results),
        "wins": sum(1 for r in results if r["is_win"]),
        "avg_close_return": round(
            sum(r["close_return_pct"] for r in results) / len(results), 2
        ),
        "avg_high_return": round(
            sum(r["high_return_pct"] for r in results) / len(results), 2
        ),
        "details": results,
    }
    record["win_rate"] = round(record["wins"] / record["total"] * 100, 1)

    accuracy["records"].append(record)

    # 최근 20일 통계 재계산
    recent = accuracy["records"][-20:]
    all_results = [r for rec in recent for r in rec["details"]]
    if all_results:
        accuracy["stats"] = {
            "period_days": len(recent),
            "total_picks": len(all_results),
            "total_wins": sum(1 for r in all_results if r["is_win"]),
            "win_rate": round(
                sum(1 for r in all_results if r["is_win"]) / len(all_results) * 100, 1
            ),
            "avg_close_return": round(
                sum(r["close_return_pct"] for r in all_results) / len(all_results), 2
            ),
            "avg_high_return": round(
                sum(r["high_return_pct"] for r in all_results) / len(all_results), 2
            ),
            # 등급별 승률
            "by_grade": _calc_grade_stats(all_results),
            # NXT 데이터 유무별 승률
            "nxt_data_win_rate": _calc_nxt_data_win_rate(all_results),
            "updated_at": datetime.now().isoformat(),
        }

    _save_accuracy(accuracy)

    # 학습 가중치 업데이트
    _update_learning_weights(accuracy["stats"])

    # 결과 출력
    logger.info(
        "\n  당일: %d종목 중 %d승 (승률 %.0f%%, 평균수익 %+.1f%%)",
        record["total"], record["wins"],
        record["win_rate"], record["avg_close_return"],
    )
    if accuracy["stats"]:
        s = accuracy["stats"]
        logger.info(
            "  누적(%d일): %d종목 승률 %.0f%% 평균수익 %+.1f%%",
            s["period_days"], s["total_picks"],
            s["win_rate"], s["avg_close_return"],
        )

    return accuracy


def _calc_grade_stats(results: list[dict]) -> dict:
    """등급별 승률 계산."""
    grades = {}
    for r in results:
        g = r.get("nxt_grade", "기타")
        if g not in grades:
            grades[g] = {"total": 0, "wins": 0, "returns": []}
        grades[g]["total"] += 1
        if r["is_win"]:
            grades[g]["wins"] += 1
        grades[g]["returns"].append(r["close_return_pct"])

    stats = {}
    for g, d in grades.items():
        stats[g] = {
            "total": d["total"],
            "wins": d["wins"],
            "win_rate": round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0,
            "avg_return": round(sum(d["returns"]) / len(d["returns"]), 2) if d["returns"] else 0,
        }
    return stats


def _calc_nxt_data_win_rate(results: list[dict]) -> dict:
    """NXT 데이터 유무별 승률."""
    with_nxt = [r for r in results if r.get("had_nxt_data")]
    without_nxt = [r for r in results if not r.get("had_nxt_data")]

    return {
        "with_nxt": {
            "total": len(with_nxt),
            "win_rate": round(
                sum(1 for r in with_nxt if r["is_win"]) / len(with_nxt) * 100, 1
            ) if with_nxt else 0,
            "avg_return": round(
                sum(r["close_return_pct"] for r in with_nxt) / len(with_nxt), 2
            ) if with_nxt else 0,
        },
        "without_nxt": {
            "total": len(without_nxt),
            "win_rate": round(
                sum(1 for r in without_nxt if r["is_win"]) / len(without_nxt) * 100, 1
            ) if without_nxt else 0,
            "avg_return": round(
                sum(r["close_return_pct"] for r in without_nxt) / len(without_nxt), 2
            ) if without_nxt else 0,
        },
    }


def _update_learning_weights(stats: dict):
    """
    학습 가중치 업데이트 → nxt_recommend.py에 피드백.

    등급별 승률을 기반으로 스코어 보정 가중치 생성.
    승률 높은 등급 → 보너스, 낮은 등급 → 페널티.
    """
    if not stats:
        return

    by_grade = stats.get("by_grade", {})
    nxt_data = stats.get("nxt_data_win_rate", {})

    weights = {
        "updated_at": datetime.now().isoformat(),
        "data_points": stats.get("total_picks", 0),
        "overall_win_rate": stats.get("win_rate", 50),
        # 등급별 보정 승수 (기준: 50% 승률 = 1.0)
        "grade_multiplier": {},
        # NXT 데이터 보너스 보정
        "nxt_data_bonus": 0,
    }

    for grade, gs in by_grade.items():
        wr = gs.get("win_rate", 50)
        # 승률 50% 기준으로 보정: 70% → 1.4, 30% → 0.6
        weights["grade_multiplier"][grade] = round(wr / 50, 2)

    # NXT 데이터 있는 종목이 더 좋으면 보너스 증가
    with_wr = nxt_data.get("with_nxt", {}).get("win_rate", 50)
    without_wr = nxt_data.get("without_nxt", {}).get("win_rate", 50)
    if with_wr > without_wr:
        weights["nxt_data_bonus"] = min(20, round((with_wr - without_wr) * 0.5))
    else:
        weights["nxt_data_bonus"] = max(-10, round((with_wr - without_wr) * 0.3))

    WEIGHTS_PATH.write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("학습 가중치 업데이트: %s", WEIGHTS_PATH.name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NXT 추천 결과 추적")
    parser.add_argument("--date", default=None, help="평가 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    result = track_results(eval_date=args.date)

    if result and result.get("stats"):
        s = result["stats"]
        print(f"\n=== NXT 추천 누적 성적 ({s['period_days']}일) ===")
        print(f"  총 {s['total_picks']}종목, 승률 {s['win_rate']}%")
        print(f"  평균수익 {s['avg_close_return']:+.1f}%, 최대 {s['avg_high_return']:+.1f}%")

        bg = s.get("by_grade", {})
        if bg:
            print(f"\n  등급별:")
            for g, gs in bg.items():
                print(f"    {g}: {gs['total']}건 승률{gs['win_rate']}% 수익{gs['avg_return']:+.1f}%")


if __name__ == "__main__":
    main()
