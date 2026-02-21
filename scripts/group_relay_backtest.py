"""
그룹 릴레이 백테스트 — 대장주 발화 후 계열사 전파 검증

가설: 대장주(그룹 시총 1위)가 +N% 이상 급등하면,
      D+1~D+3 내에 하위 계열사도 따라 오른다.

Usage:
    python scripts/group_relay_backtest.py                  # 기본 (3%, 2년)
    python scripts/group_relay_backtest.py --threshold 5.0  # 발화 기준 5%
    python scripts/group_relay_backtest.py --days 756       # 3년
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

STOCK_DATA_DIR = PROJECT_ROOT / "stock_data_daily"
GROUP_YAML = PROJECT_ROOT / "config" / "group_structure.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "group_relay"


# ──────────────────────────────────────────
# CSV 데이터 로드
# ──────────────────────────────────────────

def find_csv_by_ticker(ticker: str) -> Path | None:
    """stock_data_daily/에서 티커에 해당하는 CSV 찾기.
    파일명 형식: '종목명_티커.csv'
    """
    pattern = f"*_{ticker}.csv"
    matches = list(STOCK_DATA_DIR.glob(pattern))
    if matches:
        return matches[0]
    return None


def load_daily_closes(ticker: str, min_rows: int = 60) -> pd.Series | None:
    """티커의 일봉 종가 Series 반환 (index=Date)"""
    csv_path = find_csv_by_ticker(ticker)
    if csv_path is None:
        logger.debug("CSV 없음: %s", ticker)
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if len(df) < min_rows:
            logger.debug("데이터 부족: %s (%d행)", ticker, len(df))
            return None
        df = df.sort_values("Date")
        df = df.set_index("Date")
        return df["Close"]
    except Exception as e:
        logger.debug("CSV 로드 실패: %s — %s", ticker, e)
        return None


def load_daily_returns(ticker: str, min_rows: int = 60) -> pd.Series | None:
    """티커의 일별 수익률(%) Series 반환"""
    closes = load_daily_closes(ticker, min_rows)
    if closes is None:
        return None
    return closes.pct_change() * 100


# ──────────────────────────────────────────
# 그룹 구조 로드
# ──────────────────────────────────────────

def load_group_structure() -> dict:
    """group_structure.yaml 로드"""
    with open(GROUP_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("groups", {})


# ──────────────────────────────────────────
# 백테스트 핵심 로직
# ──────────────────────────────────────────

def backtest_group(
    group_name: str,
    group_data: dict,
    fire_threshold: float = 3.0,
    lag_days: list[int] | None = None,
    lookback_days: int = 504,
) -> dict:
    """단일 그룹의 대장주→계열사 릴레이 백테스트.

    Args:
        group_name: 그룹명 (삼성, SK 등)
        group_data: {leader, tier1, tier2, tier3}
        fire_threshold: 대장주 발화 기준 등락률(%)
        lag_days: 측정할 래그 [1, 2, 3]
        lookback_days: 백테스트 기간 (거래일)

    Returns:
        그룹별 통계 dict
    """
    if lag_days is None:
        lag_days = [1, 2, 3]

    leader = group_data["leader"]
    leader_returns = load_daily_returns(leader["ticker"])
    if leader_returns is None:
        logger.warning("[%s] 대장주 %s 데이터 없음", group_name, leader["name"])
        return {"error": f"대장주 데이터 없음: {leader['name']}"}

    # lookback 기간만 사용
    leader_returns = leader_returns.iloc[-lookback_days:]

    # 발화일 추출: 대장주 수익률 >= fire_threshold
    fire_dates = leader_returns[leader_returns >= fire_threshold].index.tolist()
    logger.info("[%s] 대장주 %s: 발화일 %d건 (>= %.1f%%)",
                group_name, leader["name"], len(fire_dates), fire_threshold)

    if len(fire_dates) < 3:
        logger.warning("[%s] 발화일 %d건 — 통계 불충분", group_name, len(fire_dates))
        return {
            "leader": leader["name"],
            "leader_ticker": leader["ticker"],
            "fire_events": len(fire_dates),
            "error": "발화일 부족",
        }

    # 모든 계열사 수익률 미리 로드
    all_subs = {}
    for tier_name in ["tier1", "tier2", "tier3"]:
        for stock in group_data.get(tier_name, []):
            ret = load_daily_returns(stock["ticker"])
            if ret is not None:
                all_subs[stock["ticker"]] = {
                    "name": stock["name"],
                    "tier": tier_name,
                    "returns": ret,
                }

    # 티어별 통계 계산
    result = {
        "leader": leader["name"],
        "leader_ticker": leader["ticker"],
        "fire_events": len(fire_dates),
        "fire_dates_sample": [d.strftime("%Y-%m-%d") for d in fire_dates[:5]],
    }

    for tier_name in ["tier1", "tier2", "tier3"]:
        tier_stocks = [s for s in all_subs.values() if s["tier"] == tier_name]
        if not tier_stocks:
            result[tier_name] = {"stocks": 0, "error": "종목 없음"}
            continue

        tier_result = _calc_tier_stats(
            fire_dates, tier_stocks, lag_days, leader_returns
        )
        tier_result["stocks"] = len(tier_stocks)
        tier_result["stock_names"] = [s["name"] for s in tier_stocks]
        result[tier_name] = tier_result

    return result


def _calc_tier_stats(
    fire_dates: list,
    tier_stocks: list[dict],
    lag_days: list[int],
    leader_returns: pd.Series,
) -> dict:
    """티어별 래그 통계 계산"""
    lag_stats = {}

    for lag in lag_days:
        returns_all = []  # 모든 (발화일, 종목) 쌍의 후행 수익률
        propagation_hits = 0  # +1% 이상 오른 케이스 수
        total_cases = 0
        cases = []  # 상세 케이스

        for fire_date in fire_dates:
            for stock in tier_stocks:
                ret_series = stock["returns"]
                # 발화일 이후 lag거래일의 수익률
                try:
                    fire_idx = ret_series.index.get_loc(fire_date)
                except KeyError:
                    continue

                target_idx = fire_idx + lag
                if target_idx >= len(ret_series):
                    continue

                target_date = ret_series.index[target_idx]
                follow_ret = ret_series.iloc[target_idx]

                if pd.isna(follow_ret):
                    continue

                returns_all.append(follow_ret)
                total_cases += 1

                if follow_ret >= 1.0:
                    propagation_hits += 1

                # 누적 수익률 (D+1 ~ D+lag)
                cum_ret = ret_series.iloc[fire_idx + 1: target_idx + 1].sum()

                cases.append({
                    "fire_date": fire_date.strftime("%Y-%m-%d"),
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "stock": stock["name"],
                    "leader_ret": float(leader_returns.loc[fire_date]),
                    "follow_ret": float(follow_ret),
                    "cum_ret": float(cum_ret),
                })

        if not returns_all:
            lag_stats[f"lag{lag}"] = {"samples": 0, "error": "데이터 없음"}
            continue

        arr = np.array(returns_all)
        win_count = int(np.sum(arr > 0))

        lag_stats[f"lag{lag}"] = {
            "samples": total_cases,
            "propagation_rate": round(propagation_hits / total_cases, 3),
            "win_rate": round(win_count / total_cases, 3),
            "avg_return": round(float(np.mean(arr)), 3),
            "med_return": round(float(np.median(arr)), 3),
            "std_return": round(float(np.std(arr)), 3),
            "best_cases": sorted(cases, key=lambda x: x["follow_ret"], reverse=True)[:3],
            "worst_cases": sorted(cases, key=lambda x: x["follow_ret"])[:3],
        }

    # 최적 래그: 승률 × 평균수익률 최대화
    best_lag = None
    best_score = -999
    for lag in lag_days:
        key = f"lag{lag}"
        if key not in lag_stats or "error" in lag_stats[key]:
            continue
        s = lag_stats[key]
        score = s["win_rate"] * s["avg_return"]
        if score > best_score:
            best_score = score
            best_lag = lag

    # 최적 래그의 요약 통계를 최상위로
    summary = {}
    if best_lag and f"lag{best_lag}" in lag_stats:
        bl = lag_stats[f"lag{best_lag}"]
        summary = {
            "propagation_rate": bl.get("propagation_rate", 0),
            "win_rate": bl.get("win_rate", 0),
            "avg_return": bl.get("avg_return", 0),
            "best_lag": best_lag,
        }

    return {**summary, "lag_detail": lag_stats}


# ──────────────────────────────────────────
# 결과 저장 + 요약 출력
# ──────────────────────────────────────────

def save_results(results: dict, params: dict) -> Path:
    """결과를 JSON으로 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "group_relay_patterns.json"

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fire_threshold": params["fire_threshold"],
        "lookback_days": params["lookback_days"],
        "groups": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("결과 저장: %s", output_path)
    return output_path


def print_summary(results: dict, fire_threshold: float) -> None:
    """콘솔 요약 출력"""
    print("\n" + "=" * 70)
    print(f"그룹 릴레이 백테스트 결과 (발화 기준: >= {fire_threshold}%)")
    print("=" * 70)

    for group_name, data in results.items():
        if "error" in data and data.get("fire_events", 0) < 3:
            print(f"\n  [{group_name}] 대장주: {data.get('leader', '?')} "
                  f"— 발화 {data.get('fire_events', 0)}건 (데이터 부족)")
            continue

        print(f"\n  [{group_name}] 대장주: {data['leader']} — 발화 {data['fire_events']}건")

        for tier_name in ["tier1", "tier2", "tier3"]:
            tier = data.get(tier_name, {})
            if "error" in tier or tier.get("stocks", 0) == 0:
                print(f"    {tier_name}: 종목 없음")
                continue

            prop = tier.get("propagation_rate", 0)
            wr = tier.get("win_rate", 0)
            avg = tier.get("avg_return", 0)
            bl = tier.get("best_lag", "?")
            names = ", ".join(tier.get("stock_names", []))

            # 판정
            if prop >= 0.6 and wr >= 0.55:
                grade = "STRONG"
            elif prop >= 0.4 and wr >= 0.50:
                grade = "MODERATE"
            else:
                grade = "WEAK"

            print(f"    {tier_name} [{grade}]: "
                  f"전파율 {prop:.1%} | 승률 {wr:.1%} | "
                  f"평균 {avg:+.2f}% | 최적래그 D+{bl}")
            print(f"           종목: {names}")

    # 성공 기준 체크
    print("\n" + "-" * 70)
    print("성공 기준 체크:")
    pass_count = 0
    for group_name, data in results.items():
        tier1 = data.get("tier1", {})
        if tier1.get("propagation_rate", 0) >= 0.6:
            pass_count += 1
    print(f"  tier1 전파율 >= 60%: {pass_count}/{len(results)} 그룹 통과")

    pass_count = 0
    for group_name, data in results.items():
        tier1 = data.get("tier1", {})
        if tier1.get("win_rate", 0) >= 0.55:
            pass_count += 1
    print(f"  tier1 승률 >= 55%: {pass_count}/{len(results)} 그룹 통과")

    pass_count = 0
    for group_name, data in results.items():
        if data.get("fire_events", 0) >= 30:
            pass_count += 1
    print(f"  발화 이벤트 >= 30건: {pass_count}/{len(results)} 그룹 통과")

    print()


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="그룹 릴레이 백테스트")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="대장주 발화 기준 등락률(%%) (기본: 3.0)")
    parser.add_argument("--days", type=int, default=504,
                        help="백테스트 기간 거래일 (기본: 504 ≈ 2년)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 로그")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    groups = load_group_structure()
    if not groups:
        print("그룹 구조 로드 실패!")
        return

    print(f"로드된 그룹: {len(groups)}개 — {', '.join(groups.keys())}")

    results = {}
    for group_name, group_data in groups.items():
        results[group_name] = backtest_group(
            group_name=group_name,
            group_data=group_data,
            fire_threshold=args.threshold,
            lookback_days=args.days,
        )

    # 결과 저장
    save_results(results, {
        "fire_threshold": args.threshold,
        "lookback_days": args.days,
    })

    # 요약 출력
    print_summary(results, args.threshold)


if __name__ == "__main__":
    main()
