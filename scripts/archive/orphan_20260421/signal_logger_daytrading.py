"""FLOWX STEP 3 — 단타봇 시그널 로거.

3가지 단타 데이터 소스에서 시그널을 추출하여 Supabase signals 테이블에 기록.

데이터 소스:
  1. smallcap_explosion.json — 소형주 급등 (PRIMARY/SECONDARY 등급)
  2. volume_spike_watchlist.json — 거래량 급등 후 눌림목 (pullback)
  3. whale_detect.json — 세력 대량매집 탐지

실행 시점: 매일 09:05 KST (장 시작 직후)

Usage:
    python scripts/signal_logger_daytrading.py
    python scripts/signal_logger_daytrading.py --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(path: Path) -> dict | list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_settings() -> dict:
    """settings.yaml에서 flowx.daytrading 설정."""
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("flowx", {}).get("daytrading", {})
    except Exception:
        return {}


def build_daytrading_signals(date_str: str = "") -> list[dict]:
    """3가지 소스 → 단타 시그널 리스트 생성."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    signals = []
    seen_tickers: set[str] = set()

    # ── 소스 1: 소형주 급등 ──
    explosion = _load_json(DATA_DIR / "smallcap_explosion.json")
    for sig in explosion.get("signals", []):
        ticker = sig.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue

        grade_raw = sig.get("grade", "")
        supply = sig.get("supply_grade", "")
        change_pct = sig.get("change_pct", 0)
        close = sig.get("close", 0)

        if close <= 0:
            continue

        # 등급 매핑
        if grade_raw == "PRIMARY" and supply == "CONFIRMED":
            grade = "AA"
            score = 90
        elif grade_raw == "PRIMARY":
            grade = "A"
            score = 80
        elif grade_raw == "SECONDARY":
            grade = "B"
            score = 70
        else:
            continue  # C등급 이하 스킵

        seen_tickers.add(ticker)
        signals.append({
            "bot_type": "DAYTRADING",
            "ticker": ticker,
            "ticker_name": sig.get("name", ""),
            "signal_type": "PICK",
            "grade": grade,
            "score": score,
            "entry_price": int(close),
            "target_price": int(close * 1.05),   # 단타 목표 +5%
            "stop_price": int(close * 0.97),      # 단타 손절 -3%
            "current_price": int(close),
            "return_pct": 0,
            "max_return_pct": 0,
            "status": "OPEN",
            "signal_date": date_str,
            "multiplier": 1.0,
            "memo": f"급등 {change_pct:+.1f}% grade={grade_raw} supply={supply}",
        })

    # ── 소스 2: 거래량 급등 눌림목 ──
    vol_spike = _load_json(DATA_DIR / "volume_spike_watchlist.json")
    for sig in vol_spike.get("signals", []):
        ticker = sig.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue

        pullback_pct = sig.get("pullback_pct", 0)
        rsi = sig.get("rsi", 50)
        close = sig.get("current_close", 0)

        if close <= 0:
            continue

        # 눌림폭 -5%~-15% + RSI < 40 = 좋은 눌림목
        if pullback_pct > -3 or pullback_pct < -20:
            continue  # 너무 적거나 너무 큰 조정은 스킵

        if rsi < 35 and pullback_pct <= -8:
            grade, score = "A", 80
        elif rsi < 45 and pullback_pct <= -5:
            grade, score = "B", 70
        else:
            grade, score = "B", 65

        seen_tickers.add(ticker)
        signals.append({
            "bot_type": "DAYTRADING",
            "ticker": ticker,
            "ticker_name": sig.get("name", ""),
            "signal_type": "PICK",
            "grade": grade,
            "score": score,
            "entry_price": int(close),
            "target_price": int(close * 1.05),
            "stop_price": int(close * 0.97),
            "current_price": int(close),
            "return_pct": 0,
            "max_return_pct": 0,
            "status": "OPEN",
            "signal_date": date_str,
            "multiplier": 1.0,
            "memo": f"눌림목 pullback={pullback_pct:.1f}% RSI={rsi:.0f}",
        })

    # ── 소스 3: 세력 탐지 ──
    whale = _load_json(DATA_DIR / "whale_detect.json")
    for item in whale.get("items", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue

        close = item.get("close", 0)
        trading_value = item.get("trading_value_m", 0)
        adx = item.get("adx", 0)

        if close <= 0 or trading_value < 500:  # 거래대금 5억 이상
            continue

        # ADX 강함 + 대량거래
        if adx >= 25 and trading_value >= 1000:
            grade, score = "A", 78
        elif adx >= 20:
            grade, score = "B", 68
        else:
            continue

        seen_tickers.add(ticker)
        signals.append({
            "bot_type": "DAYTRADING",
            "ticker": ticker,
            "ticker_name": item.get("name", ""),
            "signal_type": "PICK",
            "grade": grade,
            "score": score,
            "entry_price": int(close),
            "target_price": int(close * 1.05),
            "stop_price": int(close * 0.97),
            "current_price": int(close),
            "return_pct": 0,
            "max_return_pct": 0,
            "status": "OPEN",
            "signal_date": date_str,
            "multiplier": 1.0,
            "memo": f"세력 거래대금={trading_value:.0f}M ADX={adx:.0f}",
        })

    # 점수 기준 상위 10개만 (너무 많은 시그널 방지)
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals[:10]


def log_daytrading_signals(dry_run: bool = False) -> int:
    """단타 시그널 생성 → Supabase INSERT."""
    settings = _load_settings()
    min_grade = settings.get("min_grade", "A")
    min_score = settings.get("min_score", 65)

    signals = build_daytrading_signals()

    # 등급/점수 필터 적용
    grade_order = {"AA": 4, "A": 3, "B": 2, "C": 1}
    min_grade_val = grade_order.get(min_grade, 3)
    signals = [
        s for s in signals
        if grade_order.get(s["grade"], 0) >= min_grade_val and s["score"] >= min_score
    ]

    if not signals:
        print("  기록할 단타 시그널 없음")
        return 0

    print(f"  단타 시그널 {len(signals)}건 생성")
    for s in signals:
        print(f"    {s['ticker_name']}({s['ticker']}) [{s['grade']}] {s['score']}점 "
              f"진입 {s['entry_price']:,} | {s['memo']}")

    if dry_run:
        print("  [DRY-RUN] 업로드 스킵")
        return len(signals)

    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()
    if not uploader.is_active:
        print("  [WARN] Supabase 미연결")
        return 0

    count = 0
    for sig in signals:
        ok = uploader.insert_signal(sig)
        if ok:
            count += 1

    print(f"  Supabase 기록: {count}/{len(signals)}건")
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 단타 시그널 로거")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*40}")
    print(f"  FLOWX 단타 시그널 로거 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")

    count = log_daytrading_signals(dry_run=args.dry_run)
    print(f"\n완료: {count}건 기록")


if __name__ == "__main__":
    main()
