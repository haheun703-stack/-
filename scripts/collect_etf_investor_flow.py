"""일일 ETF 투자자별 매매동향 수집

KIS API로 23개 섹터 ETF의 개인/기관/외국인 순매수 데이터 수집.
BAT-D 9c.3단계에서 실행.

사용처:
  1. flow_distortion.py → 수급 왜곡 보정 (A)
  2. flow_distortion.py → 선행 매수 시그널 (B)
  3. flow_distortion.py → 섹터 센티먼트 (C)

Usage:
    python -u -X utf8 scripts/collect_etf_investor_flow.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.kis_investor_adapter import fetch_investor_by_ticker
from src.etf.config import build_sector_universe, _KODEX_ADDITIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "etf_investor_flow.json"


def build_sector_etf_list() -> dict[str, dict]:
    """섹터별 대표 ETF 1개씩 선정.

    TIGER + KODEX 통합 유니버스에서 섹터당 하나만 선택.
    KODEX가 있으면 KODEX 우선 (네이버 데이터와 일관성).
    """
    universe = build_sector_universe()

    # 섹터별 ETF 그룹핑
    sector_etfs: dict[str, list[tuple[str, str]]] = {}
    for code, info in universe.items():
        sector = info.get("sector", "")
        if not sector:
            continue
        if sector not in sector_etfs:
            sector_etfs[sector] = []
        sector_etfs[sector].append((code, info.get("name", "")))

    # 섹터별 대표 1개 선택 (KODEX 우선)
    result = {}
    for sector, etfs in sector_etfs.items():
        # KODEX 우선
        kodex = [e for e in etfs if "KODEX" in e[1]]
        if kodex:
            code, name = kodex[0]
        else:
            code, name = etfs[0]
        result[code] = {"name": name, "sector": sector}

    return result


def collect_all_etf_flows() -> dict:
    """전체 섹터 ETF 투자자 데이터 수집."""
    etf_list = build_sector_etf_list()
    logger.info("수집 대상: %d개 섹터 ETF", len(etf_list))

    results = {}
    success = 0

    for i, (etf_code, info) in enumerate(etf_list.items()):
        logger.info("[%d/%d] %s (%s) 수집 중...",
                     i + 1, len(etf_list), info["name"], etf_code)
        try:
            df = fetch_investor_by_ticker(etf_code)
            if df.empty:
                logger.warning("  → 데이터 없음")
                continue

            # DataFrame → list[dict] 변환
            days = []
            for dt, row in df.iterrows():
                days.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "individual_net": int(row.get("개인", 0)),
                    "inst_net": int(row.get("기관합계", 0)),
                    "foreign_net": int(row.get("외국인합계", 0)),
                    "close": int(row.get("close", 0)),
                })

            results[etf_code] = {
                "name": info["name"],
                "sector": info["sector"],
                "days": days,
            }
            success += 1
            logger.info("  → %d일 수집 완료", len(days))

        except Exception as e:
            logger.warning("  → 실패: %s", e)

        if i > 0:
            time.sleep(0.12)  # KIS rate limit 방지

    logger.info("수집 완료: %d/%d 성공", success, len(etf_list))
    return results


def main():
    print("=" * 60)
    print("  ETF 투자자별 매매동향 수집 (KIS API)")
    print("=" * 60)

    etf_data = collect_all_etf_flows()

    output = {
        "collected_at": datetime.now().isoformat(),
        "etf_count": len(etf_data),
        "etfs": etf_data,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {OUTPUT_PATH}")
    print(f"수집: {len(etf_data)}개 ETF")

    # 요약 출력
    for code, info in etf_data.items():
        days = info.get("days", [])
        if days:
            latest = days[-1]
            indiv = latest["individual_net"]
            print(f"  {info['sector']:>8s} | {info['name']:<25s} | "
                  f"개인: {indiv / 1e8:>+8.0f}억 | {len(days)}일")


if __name__ == "__main__":
    main()
