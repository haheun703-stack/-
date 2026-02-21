"""
그룹 릴레이 감지기 — 장마감 후 대장주 발화 감지 + 계열사 대기 점수화

백테스트 결과 단독 시그널로는 약함(전파율 25~44%).
→ 기존 시그널의 보조 정보(참고용)로 사용.

Usage:
    python scripts/group_relay_detector.py               # 기본 (3% 발화 기준)
    python scripts/group_relay_detector.py --threshold 5  # 5% 기준
    python scripts/group_relay_detector.py --json         # JSON 출력
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

from scripts.group_relay_backtest import find_csv_by_ticker, load_daily_closes

logger = logging.getLogger(__name__)

GROUP_YAML = PROJECT_ROOT / "config" / "group_structure.yaml"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = PROJECT_ROOT / "data" / "group_relay" / "group_relay_today.json"

# 점수 배분 (100점)
SCORE_WAITING = 30   # 대기 상태 (-2%~+3%)
SCORE_VOLUME = 20    # 거래량 이상 (5일 대비)
SCORE_RSI = 15       # RSI 위치 (30~50 이상적)
SCORE_TIER = 15      # 티어 가산 (tier1 > tier2 > tier3)
SCORE_FLOW = 20      # 수급 흐름 (외인/기관 연속매수)


def load_group_structure() -> dict:
    """group_structure.yaml 로드"""
    with open(GROUP_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("groups", {})


def load_stock_latest(ticker: str) -> dict | None:
    """종목의 최신 데이터 로드 (stock_data_daily CSV)"""
    csv_path = find_csv_by_ticker(ticker)
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.dropna(subset=["Close"]).sort_values("Date")
        if len(df) < 10:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 최근 5일 평균 거래량
        vol_5d = df["Volume"].tail(6).head(5).mean()

        # RSI (CSV에 있으면 사용)
        rsi = float(latest.get("RSI", 50)) if pd.notna(latest.get("RSI")) else 50

        # 외인/기관 연속매수 일수
        foreign_streak = _calc_consecutive_buy(df, "Foreign_Net")
        inst_streak = _calc_consecutive_buy(df, "Inst_Net")

        return {
            "ticker": ticker,
            "name": csv_path.stem.rsplit("_", 1)[0],
            "date": str(latest["Date"].date()),
            "close": float(latest["Close"]),
            "prev_close": float(prev["Close"]),
            "change_pct": round(
                (float(latest["Close"]) - float(prev["Close"]))
                / float(prev["Close"]) * 100, 2
            ) if float(prev["Close"]) > 0 else 0,
            "volume": float(latest["Volume"]),
            "vol_5d_avg": float(vol_5d) if vol_5d > 0 else 1,
            "vol_ratio": round(float(latest["Volume"]) / vol_5d, 2) if vol_5d > 0 else 1,
            "rsi": rsi,
            "foreign_streak": foreign_streak,
            "inst_streak": inst_streak,
        }
    except Exception as e:
        logger.debug("CSV 로드 실패: %s — %s", ticker, e)
        return None


def _calc_consecutive_buy(df: pd.DataFrame, col: str) -> int:
    """연속 순매수 일수 계산"""
    if col not in df.columns:
        return 0
    vals = df[col].dropna().values
    if len(vals) == 0:
        return 0
    streak = 0
    for v in reversed(vals):
        if v > 0:
            streak += 1
        else:
            break
    return streak


# ──────────────────────────────────────────
# 대장주 발화 감지
# ──────────────────────────────────────────

def detect_fired_leaders(fire_threshold: float = 3.0) -> list[dict]:
    """오늘 +fire_threshold% 이상 오른 대장주 감지"""
    groups = load_group_structure()
    fired = []

    for group_name, group_data in groups.items():
        leader = group_data["leader"]
        data = load_stock_latest(leader["ticker"])
        if data is None:
            logger.debug("[%s] 대장주 %s 데이터 없음", group_name, leader["name"])
            continue

        if data["change_pct"] >= fire_threshold:
            fired.append({
                "group": group_name,
                "leader": data,
                "group_data": group_data,
            })
            logger.info("[%s] 대장주 발화: %s %+.2f%%",
                        group_name, data["name"], data["change_pct"])

    return fired


# ──────────────────────────────────────────
# 계열사 점수화
# ──────────────────────────────────────────

def score_subsidiaries(
    group_name: str,
    group_data: dict,
    leader_data: dict,
    top_n: int = 3,
) -> list[dict]:
    """발화 그룹의 계열사 점수화 (대기 상태 우선)"""
    candidates = []

    for tier_name in ["tier1", "tier2", "tier3"]:
        tier_bonus = {"tier1": SCORE_TIER, "tier2": SCORE_TIER * 0.67, "tier3": SCORE_TIER * 0.33}
        for stock in group_data.get(tier_name, []):
            data = load_stock_latest(stock["ticker"])
            if data is None or data["close"] <= 0:
                continue

            score = 0
            reasons = []

            # 1. 대기 상태 (-2% ~ +3%) — 아직 안 움직인 게 핵심
            chg = data["change_pct"]
            if -2 <= chg <= 3:
                score += SCORE_WAITING
                reasons.append(f"대기중({chg:+.1f}%)")
            elif -5 <= chg < -2:
                score += SCORE_WAITING * 0.5
                reasons.append(f"소폭하락({chg:+.1f}%)")

            # 2. 거래량 이상 (5일 대비 120%+)
            vr = data["vol_ratio"]
            if vr >= 2.0:
                score += SCORE_VOLUME
                reasons.append(f"거래량{vr:.1f}x")
            elif vr >= 1.5:
                score += SCORE_VOLUME * 0.8
                reasons.append(f"거래량{vr:.1f}x")
            elif vr >= 1.2:
                score += SCORE_VOLUME * 0.5
                reasons.append(f"거래량{vr:.1f}x")

            # 3. RSI 위치 (30~50 이상적)
            rsi = data["rsi"]
            if 30 <= rsi <= 50:
                score += SCORE_RSI
                reasons.append(f"RSI{rsi:.0f}(적정)")
            elif 50 < rsi <= 60:
                score += SCORE_RSI * 0.7
                reasons.append(f"RSI{rsi:.0f}")
            elif 20 <= rsi < 30:
                score += SCORE_RSI * 0.5
                reasons.append(f"RSI{rsi:.0f}(과매도)")

            # 4. 티어 가산
            score += tier_bonus[tier_name]
            reasons.append(tier_name)

            # 5. 수급 흐름
            fs = data.get("foreign_streak", 0)
            is_ = data.get("inst_streak", 0)
            if fs >= 3 and is_ >= 3:
                score += SCORE_FLOW
                reasons.append(f"외인{fs}일+기관{is_}일")
            elif fs >= 3 or is_ >= 3:
                score += SCORE_FLOW * 0.7
                tag = f"외인{fs}일" if fs >= 3 else f"기관{is_}일"
                reasons.append(f"{tag}연속매수")
            elif fs >= 1 or is_ >= 1:
                score += SCORE_FLOW * 0.3

            candidates.append({
                "ticker": data["ticker"],
                "name": data["name"],
                "tier": tier_name,
                "change_pct": data["change_pct"],
                "volume_ratio": data["vol_ratio"],
                "rsi": data["rsi"],
                "score": round(min(score, 100.0), 1),
                "reasons": reasons,
            })

    # 점수 내림차순 정렬
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


# ──────────────────────────────────────────
# 통합 리포트용 함수
# ──────────────────────────────────────────

def generate_group_relay_report(fire_threshold: float = 3.0) -> dict:
    """통합 리포트용 그룹 릴레이 결과 생성"""
    fired = detect_fired_leaders(fire_threshold)

    report = {
        "scan_time": datetime.now().isoformat(),
        "fire_threshold": fire_threshold,
        "fired_groups": [],
        "no_fire": True,
    }

    if not fired:
        report["summary"] = "오늘 발화한 대장주 없음"
        return report

    report["no_fire"] = False
    for item in fired:
        group_name = item["group"]
        leader = item["leader"]
        group_data = item["group_data"]

        top_subs = score_subsidiaries(group_name, group_data, leader)
        report["fired_groups"].append({
            "group": group_name,
            "leader_name": leader["name"],
            "leader_change": leader["change_pct"],
            "leader_volume_ratio": leader["vol_ratio"],
            "waiting_subsidiaries": top_subs,
        })

    groups_str = ", ".join(
        f"{g['group']}({g['leader_name']} {g['leader_change']:+.1f}%)"
        for g in report["fired_groups"]
    )
    report["summary"] = f"그룹 발화: {groups_str}"
    return report


def save_report(report: dict) -> Path:
    """결과를 JSON으로 저장"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return OUTPUT_PATH


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def print_report(report: dict) -> None:
    """콘솔 출력"""
    print("\n" + "=" * 60)
    print("그룹 릴레이 감지 결과")
    print("=" * 60)

    if report["no_fire"]:
        print("  오늘 발화한 대장주 없음")
        print()
        return

    for fg in report["fired_groups"]:
        print(f"\n  [{fg['group']}] 대장주: {fg['leader_name']} "
              f"{fg['leader_change']:+.1f}% (vol {fg['leader_volume_ratio']:.1f}x)")
        print("  대기 계열사:")

        for sub in fg["waiting_subsidiaries"]:
            reasons_str = ", ".join(sub["reasons"])
            print(f"    {sub['tier']} {sub['name']}({sub['ticker']}) "
                  f"{sub['change_pct']:+.1f}% "
                  f"Score={sub['score']:.0f} [{reasons_str}]")

    print()


def main():
    parser = argparse.ArgumentParser(description="그룹 릴레이 감지기")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="대장주 발화 기준 등락률(%%) (기본: 3.0)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = generate_group_relay_report(args.threshold)
    save_report(report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
