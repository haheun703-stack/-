"""
일일 안전마진 스캔 — GREEN/YELLOW 종목 탐지 + 텔레그램 알림

컨센서스 기반만 사용 (DART 폴백 OFF).
컨센서스 풀(~140종목)을 대상으로 스캔.
관심종목/내일추천 중 풀 밖 종목만 wisereport fetch_one 시도.

Usage:
    python scripts/scan_safety_margin.py              # 기본
    python scripts/scan_safety_margin.py --no-tg      # 텔레그램 OFF
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.safety_margin import (
    SafetyMarginResult,
    _load_consensus_pool,
    calc_safety_margin,
    safety_margin_batch,
    save_consensus_snapshot,
)
from src.telegram_sender import send_message

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULT_PATH = DATA_DIR / "safety_margin_daily.json"

# 관심종목 (워치리스트 + 보유종목 등)
WATCHLIST = [
    "363250",  # 산일전기
    "272210",  # 한화시스템
    "103140",  # 풍산
    "071050",  # 한국금융지주
    "258790",  # 달바글로벌
    "090430",  # 아모레퍼시픽
    "051900",  # LG생활건강
    "042700",  # 한미반도체
    "000660",  # SK하이닉스
    "005930",  # 삼성전자
]


def _load_parquet_prices() -> dict[str, float]:
    """data/processed/*.parquet에서 종목별 최신 종가 로드."""
    prices = {}
    parquet_dir = DATA_DIR / "processed"
    if not parquet_dir.exists():
        return prices
    for f in parquet_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                prices[f.stem] = float(df["close"].iloc[-1])
        except Exception:
            pass
    return prices


def _load_stock_names() -> dict[str, str]:
    """stock_data_daily/*.csv에서 종목명 매핑."""
    names = {}
    csv_dir = DATA_DIR.parent / "stock_data_daily"
    if not csv_dir.exists():
        return names
    for f in csv_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f, nrows=1)
            if "Name" in df.columns:
                names[f.stem] = str(df["Name"].iloc[0])
        except Exception:
            pass
    return names


def _load_tomorrow_picks_tickers() -> list[str]:
    """tomorrow_picks.json에서 추천 종목 코드 로드."""
    tp_path = DATA_DIR / "tomorrow_picks.json"
    if not tp_path.exists():
        return []
    try:
        with open(tp_path, encoding="utf-8") as f:
            data = json.load(f)
        return [p.get("ticker", "") for p in data.get("picks", []) if p.get("ticker")]
    except Exception:
        return []


def run_scan() -> list[SafetyMarginResult]:
    """안전마진 스캔 실행.

    1단계: 컨센서스 풀 종목 배치 스캔 (빠름)
    2단계: 관심종목/내일추천 중 풀 밖 종목 wisereport 시도 (소량)
    """
    pool = _load_consensus_pool()
    prices = _load_parquet_prices()
    names = _load_stock_names()

    # ── 1단계: 컨센서스 풀 종목 (현재가 있는 것만) ──
    picks = []
    for ticker, cons in pool.items():
        close = prices.get(ticker, 0)
        if close <= 0:
            continue
        name = cons.get("name", "") or names.get(ticker, "")
        picks.append({"ticker": ticker, "name": name, "close": close})

    logger.info("[안전마진] 컨센서스 풀 %d종목 배치 스캔", len(picks))
    results = safety_margin_batch(picks, use_wisereport=False, use_dart=False)

    # ── 2단계: 워치리스트 중 풀 밖 → wisereport (소량만) ──
    pool_tickers = set(pool.keys())
    need_wisereport = set(WATCHLIST) - pool_tickers

    # 현재가 있는 것만
    ws_targets = []
    for t in need_wisereport:
        close = prices.get(t, 0)
        if close <= 0:
            continue
        name = names.get(t, "")
        ws_targets.append((t, name, close))

    if ws_targets:
        logger.info(
            "[안전마진] 관심종목 %d개 wisereport 크롤링", len(ws_targets)
        )
        for i, (t, name, close) in enumerate(ws_targets):
            r = calc_safety_margin(
                t, name, close,
                use_wisereport=True, use_dart=False,
            )
            results.append(r)
            if i < len(ws_targets) - 1:
                time.sleep(0.5)

    # GREEN + YELLOW만 필터
    good = [r for r in results if r.signal in ("GREEN", "YELLOW")]
    good.sort(
        key=lambda r: (0 if r.signal == "GREEN" else 1, -r.floor_margin_pct)
    )

    # 통계
    n = {s: sum(1 for r in results if r.signal == s)
         for s in ("GREEN", "YELLOW", "RED", "NO_DATA")}
    logger.info(
        "[안전마진] 결과: GREEN %d / YELLOW %d / RED %d / NO_DATA %d",
        n["GREEN"], n["YELLOW"], n["RED"], n["NO_DATA"],
    )

    return good


def save_results(results: list[SafetyMarginResult]) -> Path:
    """결과 JSON 저장."""
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary": {
            "green": sum(1 for r in results if r.signal == "GREEN"),
            "yellow": sum(1 for r in results if r.signal == "YELLOW"),
        },
        "stocks": [r.to_dict() for r in results],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("[안전마진] 결과 저장: %s", RESULT_PATH)
    return RESULT_PATH


def format_telegram(results: list[SafetyMarginResult]) -> str:
    """텔레그램 메시지 포맷."""
    greens = [r for r in results if r.signal == "GREEN"]
    yellows = [r for r in results if r.signal == "YELLOW"]

    lines = [f"[안전마진 스캔] {datetime.now():%m/%d %H:%M}"]
    lines.append(f"GREEN {len(greens)} / YELLOW {len(yellows)}")
    lines.append("")

    if greens:
        lines.append("=== GREEN (바닥가 이하) ===")
        for r in greens:
            lines.append(
                f"  {r.name} {r.close:,}원 "
                f"(바닥 {r.floor_price:,} {r.floor_margin_pct:+.1f}%)"
            )
        lines.append("")

    if yellows:
        lines.append(f"=== YELLOW TOP 15 ===")
        for r in yellows[:15]:
            lines.append(
                f"  {r.name} {r.close:,}원 "
                f"({r.floor_margin_pct:+.1f}%)"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="일일 안전마진 스캔")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 알림 OFF")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 컨센서스 히스토리 스냅샷
    save_consensus_snapshot()

    # 스캔 실행
    results = run_scan()

    # 결과 저장
    save_results(results)

    # 콘솔 출력
    greens = [r for r in results if r.signal == "GREEN"]
    yellows = [r for r in results if r.signal == "YELLOW"]
    print(f"\n{'='*60}")
    print(f"GREEN {len(greens)}종목 / YELLOW {len(yellows)}종목")
    print(f"{'='*60}")

    if greens:
        print(f"\n{'종목':16s} {'현재가':>10s} {'바닥가':>10s} {'마진':>8s}")
        print("-" * 50)
        for r in greens:
            print(f"{r.name:16s} {r.close:>10,} {r.floor_price:>10,} {r.floor_margin_pct:>+7.1f}%")

    if yellows:
        print(f"\n--- YELLOW TOP 15 ---")
        for r in yellows[:15]:
            print(f"{r.name:16s} {r.close:>10,} {r.floor_price:>10,} {r.floor_margin_pct:>+7.1f}%")

    # 텔레그램 알림
    if not args.no_tg and (greens or yellows):
        msg = format_telegram(results)
        send_message(msg)
        print(f"\n텔레그램 발송 완료")


if __name__ == "__main__":
    main()
