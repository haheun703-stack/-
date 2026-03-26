"""수급 이면 데이터 일일 수집 스크립트

사용법:
  python scripts/collect_supply_data.py
  python scripts/collect_supply_data.py --tickers 039130 021240 006360

실행 시점: 매일 장마감 후 (16:00~18:00)
  - daily_scheduler.py Phase 8 이후 실행 권장
  - 공매도 데이터: 15:40 이후 확정
  - 투자자별 매매동향: 18:10 이후 전체 확정

출력: data/supply_demand/{date}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.adapters.pykrx_supply_adapter import PykrxSupplyAdapter
from src.supply_demand_analyzer import SupplyDemandAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tickers(config_path: str = "config/settings.yaml") -> list[str]:
    """모니터링 종목 동적 로드 (보유종목 + 추천후보 + config 폴백).

    기존 config만 의존하던 방식 → 동적 소스에서 자동 수집.
    사고 이력: config에 tickers 비어있어 21일간 수집 중단 (2026-03-05~26).
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add(ticker: str):
        if ticker and ticker not in seen and len(ticker) == 6 and ticker.isdigit():
            seen.add(ticker)
            result.append(ticker)

    # 1차: 보유 종목 (가장 중요)
    try:
        pos_path = project_root / "data" / "positions.json"
        if pos_path.exists():
            pos_data = json.loads(pos_path.read_text(encoding="utf-8"))
            positions = pos_data if isinstance(pos_data, list) else pos_data.get("positions", [])
            for p in positions:
                _add(p.get("ticker", ""))
    except Exception:
        pass

    # 2차: 내일 추천 종목
    for fname in ("tomorrow_picks.json", "tomorrow_picks_flowx.json"):
        try:
            path = project_root / "data" / fname
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                for pick in data.get("picks", []):
                    _add(pick.get("ticker", ""))
        except Exception:
            pass

    # 3차: 스캔 캐시 상위 종목
    try:
        cache_path = project_root / "data" / "scan_cache.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            for item in data.get("candidates", [])[:50]:
                _add(item.get("ticker", ""))
    except Exception:
        pass

    # 4차: config 폴백
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for t in cfg.get("supply_demand", {}).get("tickers", []):
            _add(t)
        for t in cfg.get("universe", {}).get("tickers", []):
            _add(t)
    except Exception:
        pass

    if result:
        logger.info(f"동적 종목 로드: {len(result)}종목 (보유+추천+스캔+config)")
    else:
        logger.warning("동적 종목 로드 실패 — 모든 소스 비어있음")

    return result


def save_results(results: dict, date: str):
    """결과를 JSON으로 저장"""
    out_dir = project_root / "data" / "supply_demand"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.json"

    serializable = {}
    for ticker, score in results.items():
        serializable[ticker] = score.to_dict()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    logger.info(f"수급 분석 결과 저장: {out_path} ({len(results)}종목)")


def main():
    parser = argparse.ArgumentParser(description="수급 이면 데이터 일일 수집")
    parser.add_argument(
        "--tickers", nargs="*", help="수집할 종목코드 (미지정 시 config에서 로드)"
    )
    parser.add_argument(
        "--date", type=str, default=None, help="수집 기준일 (YYYYMMDD)"
    )
    args = parser.parse_args()

    tickers = args.tickers or load_tickers()
    date = args.date or datetime.today().strftime("%Y%m%d")

    if not tickers:
        logger.error("수집할 종목이 없습니다. --tickers 또는 config 확인")
        return

    logger.info(f"수급 데이터 수집 시작: {len(tickers)}종목, 기준일={date}")

    from src.pipeline_alert import PipelineErrorTracker
    tracker = PipelineErrorTracker("collect_supply_data")

    # Phase 1: pykrx 수집
    adapter = PykrxSupplyAdapter(lookback_days=60)

    try:
        collected = adapter.collect_all(tickers)
    except ImportError:
        logger.error("pykrx 미설치: pip install pykrx")
        tracker.record("pykrx", "ImportError: pykrx 미설치")
        tracker.finalize(total=len(tickers))
        return
    except Exception as e:
        logger.error("pykrx 수집 실패: %s", e)
        tracker.record("pykrx_collect_all", e)
        tracker.finalize(total=len(tickers))
        return

    # 수집 실패 종목 추적
    failed_tickers = set(tickers) - set(collected.keys())
    for t in failed_tickers:
        tracker.record(t, "수급 데이터 수집 실패")

    # 분석
    analyzer = SupplyDemandAnalyzer()
    scores = analyzer.analyze_batch(collected, date)

    # 결과 출력
    logger.info(f"\n{'='*60}")
    logger.info("수급 이면 분석 결과")
    logger.info(f"{'='*60}")

    for ticker, score in scores.items():
        emoji = "🟢" if score.trap_adjustment <= 0 else "🔴"
        logger.info(
            f"  {emoji} {ticker}: "
            f"공매도위험={score.short_risk:.0f} | "
            f"기관수급={score.institutional:.0f} | "
            f"함정보정={score.trap_adjustment:+.0f}p | "
            f"S5부스트={score.smart_money_boost:.2f}"
        )

    # 저장
    save_results(scores, date)

    # 에러 집계 + 알림
    tracker.finalize(total=len(tickers))

    logger.info(f"\n총 {len(scores)}종목 수급 분석 완료")


if __name__ == "__main__":
    main()
