"""종목 시장구분(KOSPI/KOSDAQ) 맵 생성 — FDR StockListing 기반.

FV 엔진 v1-3번: 각 종목이 KOSPI/KOSDAQ 중 어디 상장인지 알아야 시장별 국면·벤치마크
적용 가능. universe.csv·consensus 모두 market 컬럼 없음 → 별도 맵 필요.

출력: data/market_map.json = {ticker: "KOSPI"|"KOSDAQ"|"KONEX"|...}. 주간 1회 갱신 충분
(상장시장은 거의 안 바뀜). graceful — FDR 실패 시 기존 맵 유지.

사용: python -u -X utf8 scripts/update_market_map.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "data" / "market_map.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        import FinanceDataReader as fdr

        df = fdr.StockListing("KRX")
        code_col = "Code" if "Code" in df.columns else df.columns[1]
        if "Market" not in df.columns:
            logger.error("StockListing에 Market 컬럼 없음")
            return 0
        mapping = {}
        for code, market in zip(df[code_col].astype(str), df["Market"].astype(str)):
            c = code.zfill(6)
            if c.isdigit() and len(c) == 6 and market and market != "nan":
                # "KOSDAQ GLOBAL" → "KOSDAQ" 정규화
                mapping[c] = "KOSDAQ" if market.startswith("KOSDAQ") else market
        if not mapping:
            logger.warning("맵 비어있음 — 기존 유지")
            return 0
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
        from collections import Counter
        logger.info("시장맵 %d종 저장 → %s | 분포=%s", len(mapping), OUT.name,
                    dict(Counter(mapping.values())))
    except Exception as e:  # noqa: BLE001 — graceful, 기존 맵 유지
        logger.warning("[market_map] 갱신 실패(기존 유지): %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
