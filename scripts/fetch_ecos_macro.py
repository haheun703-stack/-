"""한국은행 ECOS API로 한국 매크로 지표 수집 — BAT-D Level 0

기준금리, 원/달러 환율, M2 통화량, 소비자물가(CPI) 수집.
ECOS 무료: 제한 없음 (일 10만건).

사용: python -u -X utf8 scripts/fetch_ecos_macro.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "ecos_macro.json"


# ECOS 통계코드 (https://ecos.bok.or.kr/api)
INDICATORS = {
    "base_rate": {
        "stat_code": "722Y001",
        "item_code": "0101000",
        "freq": "M",
        "label": "기준금리(%)",
    },
    "usd_krw": {
        "stat_code": "731Y001",
        "item_code": "0000001",
        "freq": "D",
        "label": "원/달러 환율",
    },
    "cpi": {
        "stat_code": "901Y009",
        "item_code": "0",
        "freq": "M",
        "label": "소비자물가지수(2020=100)",
    },
}


def _fetch_ecos(api_key: str, name: str, cfg: dict) -> dict | None:
    """ECOS API에서 단일 지표 최신값 조회."""
    import requests

    freq = cfg["freq"]
    now = datetime.now()

    if freq == "D":
        # 일별: 최근 5일 범위
        start = (now - timedelta(days=7)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")
    else:
        # 월별: 최근 3개월 범위
        start = (now - timedelta(days=120)).strftime("%Y%m")
        end = now.strftime("%Y%m")

    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch"
        f"/{api_key}/json/kr/1/5"
        f"/{cfg['stat_code']}/{freq}/{start}/{end}/{cfg['item_code']}"
    )

    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        if "StatisticSearch" in data:
            rows = data["StatisticSearch"].get("row", [])
            if rows:
                latest = rows[-1]  # 최신이 마지막
                return {
                    "value": float(latest["DATA_VALUE"].replace(",", "")),
                    "period": latest["TIME"],
                    "unit": latest.get("UNIT_NAME", ""),
                    "label": cfg["label"],
                }
        # 에러 응답
        err = data.get("RESULT", {})
        if err:
            logger.warning(f"  {name}: {err.get('MESSAGE', 'unknown error')}")
        return None

    except Exception as e:
        logger.warning(f"  {name}: {e}")
        return None


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ECOS_API_KEY", "").strip()
    if not api_key:
        logger.error("ECOS_API_KEY 없음. .env 확인 필요.")
        return

    logger.info("=== 한국은행 ECOS 매크로 지표 수집 ===")

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "source": "ecos_bok",
        "indicators": {},
    }

    for name, cfg in INDICATORS.items():
        logger.info(f"  {cfg['label']}...")
        data = _fetch_ecos(api_key, name, cfg)
        if data:
            result["indicators"][name] = data
            logger.info(f"    → {data['value']} {data.get('unit', '')} ({data['period']})")
        else:
            logger.warning(f"    → 실패")
        time.sleep(0.5)

    # 저장
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"저장: {OUTPUT_PATH} ({len(result['indicators'])}개 지표)")


if __name__ == "__main__":
    main()
