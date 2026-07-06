"""US 미래가치 엔진 일일 러너 (BAT-D G5.7) — 컨센서스 → 스냅샷 축적 → PER밴드캐시 → 엔진.

7/6 배선(퐝가님 결정). ★관측 전용·매매 미배선·shadow_unvalidated·전 단계 graceful(exit 0).

목적: ④ 백테스트가 PER밴드 무효 판정(t=0.61·reliable -0.61%p 역전)이라, 엔진 점수의 실제
  검증수단은 **컨센서스 목표가 괴리축의 20거래일 forward**뿐(역사 목표가 부재로 사후 백테스트 불가).
  → 매일 목표가 스냅샷을 `consensus_us_history.jsonl`에 축적(날짜별 dedup) → 20일 뒤 '고괴리 픽이
  실제 SPY 초과했나' 검증. 이 러너가 그 데이터 파이프라인.

단계:
  1. scan_consensus_us  → data/consensus_screening_us.json (최신)
  2. 일별 스냅샷 append → data/consensus_us_history.jsonl (forward 검증용·오늘 이미 있으면 스킵)
  3. PER밴드 캐시       → stale(>3일)일 때만 refetch (EPS 분기·yfinance rate-limit 절약)
  4. future_value_engine_us → data/shadow/future_value_us.json

사용: ./venv/bin/python3.11 scripts/run_future_value_us.py
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("run_future_value_us")

DATA_DIR = PROJECT_ROOT / "data"
HISTORY_PATH = DATA_DIR / "consensus_us_history.jsonl"
CACHE_STALE_DAYS = 3


def _append_snapshot(result: dict) -> None:
    """오늘 목표가 스냅샷을 forward 검증용 jsonl에 append(오늘 날짜 이미 있으면 스킵)."""
    picks = (result or {}).get("all_picks", [])
    if not picks:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    # dedup: 파일 마지막 줄이 오늘이면 스킵(러너 재실행 안전)
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4096))
                tail = f.read().decode("utf-8", "replace")
            if f'"date": "{today}"' in tail:
                logger.info("스냅샷 오늘(%s) 이미 존재 — append 스킵", today)
                return
        except Exception:  # noqa: BLE001
            pass
    n = 0
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        for p in picks:
            f.write(json.dumps({
                "date": today, "ticker": p["ticker"], "close": p.get("close"),
                "target": p.get("target_price"), "upside_pct": p.get("upside_pct"),
                "forward_per": p.get("forward_per"), "opinion_score": p.get("opinion_score"),
            }, ensure_ascii=False) + "\n")
            n += 1
    logger.info("forward 스냅샷 %d종 append → %s (%s)", n, HISTORY_PATH.name, today)


def _cache_stale(days: int = CACHE_STALE_DAYS) -> bool:
    try:
        from src.use_cases.valuation_band_history_us import CACHE_DIR
        files = list(CACHE_DIR.glob("*_close.parquet"))
        if not files:
            return True
        newest = max(f.stat().st_mtime for f in files)
        return (time.time() - newest) > days * 86400
    except Exception:  # noqa: BLE001
        return True


def main() -> int:
    # 1+2. 컨센서스 + forward 스냅샷 축적
    try:
        from scripts.scan_consensus_us import run_scan
        result = run_scan(top_n=20)
        _append_snapshot(result)
    except Exception as e:  # noqa: BLE001 — shadow, 죽지 않음
        logger.warning("[G5.7] 컨센서스 단계 실패(graceful): %s", e)

    # 3. PER밴드 캐시(stale>3일에만 — EPS 분기·rate-limit 절약)
    try:
        if _cache_stale():
            from src.use_cases.valuation_band import us_fv_universe
            from src.use_cases.valuation_band_history_us import fetch_and_cache
            logger.info("[G5.7] PER밴드 캐시 stale → refetch")
            fetch_and_cache(us_fv_universe())
        else:
            logger.info("[G5.7] PER밴드 캐시 최신(≤%d일) — fetch 스킵", CACHE_STALE_DAYS)
    except Exception as e:  # noqa: BLE001
        logger.warning("[G5.7] PER밴드 캐시 실패(graceful·기존 캐시 사용): %s", e)

    # 4. 엔진
    try:
        from src.use_cases.future_value_engine_us import run
        run()
    except Exception as e:  # noqa: BLE001
        logger.warning("[G5.7] 엔진 실패(graceful): %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
