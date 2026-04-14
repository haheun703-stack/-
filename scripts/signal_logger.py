"""FLOWX STEP 2 — 퀀트봇 시그널 로거.

tomorrow_picks.json에서 A등급 이상 종목을 Supabase signals 테이블에 기록.

Grade 매핑:
  강력 포착 → AA, 포착 → A, 관심 → B
  ai_largecap: confidence ≥ 0.85 → AA, ≥ 0.75 → A, else → B

실행 시점: 매일 08:20 (BAT-D 이후, scan_buy 완료 후)

Usage:
    python scripts/signal_logger.py
    python scripts/signal_logger.py --dry-run
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

# 등급 매핑: 한글 → 영문
GRADE_MAP = {"강력 포착": "AA", "포착": "A", "관심": "B", "보류": "C",
             "적극매수": "AA", "매수": "A", "관심매수": "B"}  # 하위호환

# settings.yaml 최소 등급/점수 기본값
DEFAULT_MIN_GRADE = "A"
DEFAULT_MIN_SCORE = 65


def _load_settings() -> dict:
    """settings.yaml에서 flowx.signal_logging 설정 로드."""
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("flowx", {}).get("signal_logging", {})
    except Exception:
        return {}


def _grade_passes(grade_kr: str, min_grade: str) -> bool:
    """등급 필터: min_grade 이상인지 확인."""
    order = {"AA": 4, "A": 3, "B": 2, "C": 1}
    eng = GRADE_MAP.get(grade_kr, grade_kr)  # 이미 영문이면 그대로
    return order.get(eng, 0) >= order.get(min_grade, 3)


def build_signals_from_picks(date_str: str = "", min_grade: str = "A", min_score: float = 65) -> list[dict]:
    """tomorrow_picks.json → signals 테이블 행 변환.

    Returns:
        list[dict] — Supabase signals 테이블에 INSERT할 행 목록
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    picks_path = DATA_DIR / "tomorrow_picks.json"
    if not picks_path.exists():
        logger.warning("tomorrow_picks.json 없음")
        return []

    with open(picks_path, encoding="utf-8") as f:
        data = json.load(f)

    signals = []
    seen_tickers: set[str] = set()

    # 1) picks → 퀀트봇 시그널
    for p in data.get("picks", []):
        ticker = p.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue

        grade_kr = p.get("grade", "")
        grade_eng = GRADE_MAP.get(grade_kr, grade_kr)
        score = p.get("total_score", 0)

        if not _grade_passes(grade_kr, min_grade):
            continue
        if score < min_score:
            continue

        seen_tickers.add(ticker)

        entry_price = p.get("entry_price", 0) or p.get("close", 0)
        if entry_price <= 0:
            continue

        # multiplier: 소스 수에 따른 확신 배율
        n_sources = p.get("n_sources", 1)
        if n_sources >= 4:
            multiplier = 1.5
        elif n_sources >= 3:
            multiplier = 1.2
        else:
            multiplier = 1.0

        signals.append({
            "bot_type": "QUANT",
            "ticker": ticker,
            "ticker_name": p.get("name", ""),
            "signal_type": "PICK",
            "grade": grade_eng,
            "score": min(int(round(score)), 100),
            "entry_price": int(entry_price),
            "target_price": int(p.get("target_price", 0)) or int(entry_price * 1.10),
            "stop_price": int(p.get("stop_loss", 0)) or int(entry_price * 0.92),
            "current_price": int(entry_price),
            "return_pct": 0,
            "max_return_pct": 0,
            "status": "OPEN",
            "signal_date": date_str,
            "multiplier": multiplier,
            "memo": f"sources: {', '.join(p.get('sources', [])[:3])}",
        })

    # 2) ai_largecap → AI 두뇌 시그널
    for item in data.get("ai_largecap", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen_tickers:
            continue

        confidence = float(item.get("confidence", 0))
        if confidence < 0.75:
            continue

        close = _get_close(ticker)
        if close <= 0:
            continue

        seen_tickers.add(ticker)

        if confidence >= 0.85:
            grade = "AA"
        elif confidence >= 0.75:
            grade = "A"
        else:
            grade = "B"

        impact_pct = float(item.get("expected_impact_pct", 5))

        signals.append({
            "bot_type": "QUANT",
            "ticker": ticker,
            "ticker_name": item.get("name", ""),
            "signal_type": "PICK",
            "grade": grade,
            "score": min(int(round(confidence * 100)), 100),
            "entry_price": int(close),
            "target_price": int(close * (1 + impact_pct / 100)),
            "stop_price": int(close * 0.92),
            "current_price": int(close),
            "return_pct": 0,
            "max_return_pct": 0,
            "status": "OPEN",
            "signal_date": date_str,
            "multiplier": 1.0,
            "memo": f"AI Brain confidence={confidence:.0%}",
        })

    return signals


def _get_close(ticker: str) -> int:
    """parquet에서 최신 종가."""
    pq = DATA_DIR / "processed" / f"{ticker}.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            if len(df) > 0:
                return int(df.iloc[-1]["close"])
        except Exception:
            pass
    return 0


def log_signals(dry_run: bool = False) -> int:
    """시그널 생성 → Supabase INSERT.

    Returns:
        기록된 시그널 수
    """
    settings = _load_settings()
    min_grade = settings.get("min_grade", DEFAULT_MIN_GRADE)
    min_score = settings.get("min_score", DEFAULT_MIN_SCORE)

    signals = build_signals_from_picks(min_grade=min_grade, min_score=min_score)

    if not signals:
        print("  기록할 시그널 없음")
        return 0

    print(f"  시그널 {len(signals)}건 생성 (min_grade={min_grade}, min_score={min_score})")
    for s in signals:
        print(f"    {s['ticker_name']}({s['ticker']}) [{s['grade']}] {s['score']}점 "
              f"진입 {s['entry_price']:,} → 목표 {s['target_price']:,} / 손절 {s['stop_price']:,}")

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
        else:
            print(f"  [FAIL] {sig['ticker_name']} INSERT 실패")

    print(f"  Supabase 기록: {count}/{len(signals)}건")
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLOWX 시그널 로거")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 생성만")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*40}")
    print(f"  FLOWX 시그널 로거 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")

    count = log_signals(dry_run=args.dry_run)
    print(f"\n완료: {count}건 기록")


if __name__ == "__main__":
    main()
