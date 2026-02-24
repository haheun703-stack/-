"""[D] 릴레이 트레이딩 통합 리포트.

relay_backtest.py → relay_stock_picker.py → relay_sizer.py → relay_exit.py
의 결과를 통합하여 콘솔에 출력한다.

사용법:
  python scripts/relay_report.py                    # 전체 릴레이 리포트
  python scripts/relay_report.py --portfolio 50000000  # 포트폴리오 5천만
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"

from relay_stock_picker import pick_relay_stocks, load_stock_latest
from relay_sizer import calc_full_sizing, grade_fire_intensity
from relay_exit import check_exit_conditions, get_profit_target
from relay_positions import (load_positions, check_all_positions,
                             get_current_price, print_check_results)

# relay_backtest.py의 SUPER_SECTORS 정의 재사용
SUPER_SECTORS = {
    "금융": {
        "relay_pairs": [
            ("증권", "생명보험"), ("증권", "손해보험"), ("증권", "은행"),
            ("생명보험", "손해보험"), ("은행", "생명보험"), ("은행", "손해보험"),
        ],
    },
    "IT/반도체": {
        "relay_pairs": [
            ("반도체와반도체장비", "디스플레이장비및부품"),
            ("반도체와반도체장비", "전자장비와기기"),
            ("반도체와반도체장비", "IT서비스"),
        ],
    },
    "2차전지/에너지": {
        "relay_pairs": [
            ("전기제품", "화학"), ("전기제품", "에너지장비및서비스"),
            ("석유와가스", "화학"),
        ],
    },
    "방산/조선": {
        "relay_pairs": [
            ("우주항공과국방", "조선"), ("우주항공과국방", "기계"),
            ("조선", "기계"),
        ],
    },
    "소비/유통": {
        "relay_pairs": [("화장품", "식품")],
    },
    "바이오/헬스": {
        "relay_pairs": [
            ("제약", "생물공학"), ("생물공학", "건강관리장비와용품"),
        ],
    },
}


def load_patterns() -> dict:
    """relay_patterns.json 로드 → {(lead, follow): info}."""
    path = DATA_DIR / "relay_patterns.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = {}
    for super_name, pairs in data.get("super_sectors", {}).items():
        for pair_name, pair_data in pairs.items():
            parts = pair_name.split("→")
            if len(parts) != 2:
                continue
            lead, follow = parts[0].strip(), parts[1].strip()
            best_lag = pair_data.get("best_lag", 1)
            lag_data = pair_data.get(f"lag{best_lag}", {})
            patterns[(lead, follow)] = {
                "confidence": lag_data.get("confidence", "LOW"),
                "win_rate": lag_data.get("win_rate", 0),
                "avg_return": lag_data.get("avg_return", 0),
                "best_lag": best_lag,
                "samples": lag_data.get("samples", 0),
                "super_sector": super_name,
            }
    return patterns


def get_sector_today_stats(sector: str) -> dict:
    """섹터의 오늘 통계: avg_return, breadth, top_stocks."""
    naver_map = pd.read_csv(
        DATA_DIR / "naver_sector_map.csv", dtype={"ticker": str}
    )
    sector_stocks = naver_map[naver_map["sector"] == sector]

    returns_list = []
    stock_details = []

    for _, row in sector_stocks.iterrows():
        ticker = row["ticker"]
        matches = list(DAILY_DIR.glob(f"*_{ticker}.csv"))
        if not matches:
            continue
        try:
            df = pd.read_csv(matches[0], usecols=["Date", "Close", "Volume"])
            df = df.dropna().sort_values("Date")
            if len(df) < 2:
                continue
            ret = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
            returns_list.append(ret)
            stock_details.append({
                "name": row["name"],
                "ticker": ticker,
                "return": round(ret, 2),
            })
        except Exception:
            continue

    if not returns_list:
        return {"avg_return": 0, "breadth": 0, "count": 0, "top_stocks": []}

    stock_details.sort(key=lambda x: x["return"], reverse=True)

    return {
        "avg_return": round(float(np.mean(returns_list)), 2),
        "breadth": round(sum(1 for r in returns_list if r > 0) / len(returns_list), 2),
        "count": len(returns_list),
        "up_count": sum(1 for r in returns_list if r > 0),
        "top_stocks": stock_details[:3],
    }


def detect_fired_sectors(patterns: dict, fire_threshold: float = 3.5,
                         breadth_threshold: float = 0.6) -> list[dict]:
    """오늘 발화된 선행 섹터 감지."""
    # 모든 선행 섹터 수집
    lead_sectors = set()
    for super_name, cfg in SUPER_SECTORS.items():
        for lead, _ in cfg["relay_pairs"]:
            lead_sectors.add(lead)

    fired = []
    for sector in lead_sectors:
        stats = get_sector_today_stats(sector)
        if (stats["avg_return"] >= fire_threshold
                and stats["breadth"] >= breadth_threshold):
            grade, mult = grade_fire_intensity(stats["avg_return"])
            fired.append({
                "sector": sector,
                "avg_return": stats["avg_return"],
                "breadth": stats["breadth"],
                "count": stats["count"],
                "up_count": stats.get("up_count", 0),
                "top_stocks": stats["top_stocks"],
                "grade": grade,
                "multiplier": mult,
            })

    fired.sort(key=lambda x: x["avg_return"], reverse=True)
    return fired


def generate_relay_report(
    portfolio: int = 30_000_000,
    top_picks: int = 3,
) -> dict:
    """릴레이 통합 리포트 생성."""
    today = datetime.now().strftime("%Y-%m-%d")
    patterns = load_patterns()

    if not patterns:
        print("  relay_patterns.json 없음. relay_backtest.py 먼저 실행하세요.")
        return {}

    # 1. 발화 섹터 감지
    fired_sectors = detect_fired_sectors(patterns)

    # 2. 발화된 선행 섹터별 릴레이 분석
    relay_signals = []

    for fired in fired_sectors:
        lead_sector = fired["sector"]

        # 이 선행 섹터와 연결된 후행 섹터들
        for super_name, cfg in SUPER_SECTORS.items():
            for lead, follow in cfg["relay_pairs"]:
                if lead != lead_sector:
                    continue

                pattern = patterns.get((lead, follow))
                if not pattern or pattern["confidence"] == "LOW":
                    continue

                # 후행 섹터 현재 상태
                follow_stats = get_sector_today_stats(follow)

                # 종목 선정
                picks = pick_relay_stocks(follow, top_n=top_picks)

                # 비중 계산
                sizing = calc_full_sizing(
                    win_rate=pattern["win_rate"],
                    lead_return=fired["avg_return"],
                    confidence=pattern["confidence"],
                    total_portfolio=portfolio,
                )

                # 청산 조건 계산 (진입 전이므로 가상)
                profit_target = get_profit_target(pattern["win_rate"])

                relay_signals.append({
                    "super_sector": super_name,
                    "lead_sector": lead,
                    "follow_sector": follow,
                    "lead_return": fired["avg_return"],
                    "lead_grade": fired["grade"],
                    "lead_top_stocks": fired["top_stocks"],
                    "pattern": pattern,
                    "follow_stats": follow_stats,
                    "picks": picks,
                    "sizing": sizing,
                    "profit_target": profit_target,
                    "timeout_days": pattern["best_lag"] + 1,
                })

    return {
        "date": today,
        "portfolio": portfolio,
        "fired_sectors": fired_sectors,
        "relay_signals": relay_signals,
    }


def print_relay_report(report: dict):
    """통합 리포트 출력."""
    if not report:
        return

    today = report["date"]
    portfolio = report["portfolio"]
    fired = report["fired_sectors"]
    signals = report["relay_signals"]

    print(f"\n{'=' * 60}")
    print(f"  RELAY SIGNAL — {today}")
    print(f"{'=' * 60}")

    # 발화 섹터
    if not fired:
        print(f"\n  발화 섹터 없음 (기준: +3.5%, breadth 60%)")
        print(f"{'=' * 60}\n")
        # 발화 없음 상태도 JSON에 기록 (대시보드 날짜 갱신용)
        out_path = DATA_DIR / "relay_trading_signal.json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": today,
            "portfolio": portfolio,
            "fired_count": 0,
            "signal_count": 0,
            "signals": [],
            "status": "NO_FIRE",
            "message": "발화 섹터 없음 (기준: +3.5%, breadth 60%)",
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  저장: {out_path}")
        return

    for f in fired:
        grade_icon = {"EXTREME": "EXTREME", "STRONG": "STRONG",
                      "MODERATE": "MODERATE", "WEAK": "WEAK"}
        print(f"\n  발화: {f['sector']} {f['avg_return']:+.1f}% "
              f"{f['grade']} ({f['up_count']}/{f['count']} 상승)")
        if f["top_stocks"]:
            tops = " | ".join(
                f"{s['name']}({s['return']:+.1f}%)" for s in f["top_stocks"][:3]
            )
            print(f"    상위: {tops}")

    # 릴레이 신호별 상세
    if not signals:
        print(f"\n  MED 이상 릴레이 패턴 없음")
    else:
        for sig in signals:
            pattern = sig["pattern"]
            sizing = sig["sizing"]
            picks = sig["picks"]
            follow_stats = sig["follow_stats"]

            conf_icon = {"HIGH": "[HIGH]", "MED": "[MED]"}.get(
                pattern["confidence"], "[LOW]")

            print(f"\n{'─' * 60}")
            print(f"  [{sig['super_sector']}] {sig['lead_sector']} → {sig['follow_sector']}")
            print(f"  래그{pattern['best_lag']}일  "
                  f"승률{pattern['win_rate']:.0f}%  "
                  f"평균{pattern['avg_return']:+.1f}%  "
                  f"{conf_icon}  n={pattern['samples']}")

            # 후행 섹터 현재 상태
            fr = follow_stats["avg_return"]
            fb = follow_stats["breadth"]
            status = "대기중" if -2 < fr < 3 else "이미움직임" if fr >= 3 else "하락중"
            print(f"  후행 섹터 현재: {fr:+.1f}% (breadth {fb*100:.0f}%) → {status}")

            # 매수 후보
            if picks:
                print(f"\n  매수 대상 ({sig['follow_sector']} 상위 {len(picks)}종목)")
                for i, p in enumerate(picks, 1):
                    reason = " | ".join(p["reasons"][:3])
                    fs = p.get("foreign_streak", 0)
                    is_ = p.get("inst_streak", 0)
                    flow_str = ""
                    if fs > 0 or is_ > 0:
                        parts = []
                        if fs > 0:
                            parts.append(f"외인{fs}일")
                        if is_ > 0:
                            parts.append(f"기관{is_}일")
                        flow_str = f"  수급[{'+'.join(parts)}]"
                    print(f"    {i}위. {p['name']}({p['ticker']}) "
                          f"점수:{p['score']}  {p['change_pct']:+.1f}%  "
                          f"거래량{p['vol_ratio']:.1f}x{flow_str}  {reason}")

            # 비중
            print(f"\n  비중: {sizing['weight_pct']:.1f}% "
                  f"({sizing['grade']} x{sizing['multiplier']:.1f})")
            print(f"    → {portfolio:,}원 포트 기준 "
                  f"{sizing['invest_amount']:,}원 투입")

            # 청산 조건
            print(f"\n  청산 조건")
            print(f"    목표: +{sig['profit_target']}% 도달 시")
            print(f"    타임아웃: {sig['timeout_days']}일 무반응 시 손절")
            print(f"    릴레이완료: {sig['follow_sector']} 섹터 +3.5% 발화 시")
            print(f"    손절: -5% 이하")

    # 보유 포지션 현황
    positions = load_positions()
    if positions:
        print(f"\n{'─' * 60}")
        print(f"  보유 포지션 ({len(positions)}건)")
        total_invest = 0
        total_pnl = 0
        for pos in positions:
            cur = get_current_price(pos["ticker"])
            if cur <= 0:
                continue
            pnl = (cur - pos["entry_price"]) / pos["entry_price"] * 100
            pnl_amt = int((cur - pos["entry_price"]) * pos["quantity"])
            total_invest += pos["investment"]
            total_pnl += pnl_amt
            icon = "+" if pnl > 0 else ""
            print(f"    {pos['name']}({pos['ticker']}) "
                  f"{pos['entry_price']:,}→{cur:,} "
                  f"{icon}{pnl:.1f}% "
                  f"{pos.get('trading_days_held',0)}일 "
                  f"| {pos['fired_sector']}→{pos['sector']}")
        if total_invest > 0:
            print(f"    합계: {total_invest:,}원 투입 → "
                  f"{total_pnl:+,}원 ({total_pnl/total_invest*100:+.1f}%)")

    print(f"\n{'=' * 60}")

    # JSON 저장
    out_path = DATA_DIR / "relay_trading_signal.json"
    payload = {
        "date": today,
        "portfolio": portfolio,
        "fired_count": len(fired),
        "signal_count": len(signals),
        "signals": [
            {
                "super_sector": s["super_sector"],
                "lead": s["lead_sector"],
                "follow": s["follow_sector"],
                "lead_return": s["lead_return"],
                "confidence": s["pattern"]["confidence"],
                "win_rate": s["pattern"]["win_rate"],
                "best_lag": s["pattern"]["best_lag"],
                "weight_pct": s["sizing"]["weight_pct"],
                "invest_amount": s["sizing"]["invest_amount"],
                "profit_target": s["profit_target"],
                "timeout_days": s["timeout_days"],
                "picks": [
                    {"name": p["name"], "ticker": p["ticker"],
                     "score": p["score"], "change_pct": p["change_pct"]}
                    for p in s["picks"]
                ],
            }
            for s in signals
        ],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="릴레이 트레이딩 통합 리포트")
    parser.add_argument("--portfolio", type=int, default=30_000_000,
                        help="총 포트폴리오 (기본 3000만원)")
    parser.add_argument("--top", type=int, default=3,
                        help="섹터당 종목 수 (기본 3)")
    args = parser.parse_args()

    report = generate_relay_report(
        portfolio=args.portfolio,
        top_picks=args.top,
    )
    print_relay_report(report)


if __name__ == "__main__":
    main()
