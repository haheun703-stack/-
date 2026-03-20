"""Alpha Vantage API로 원자재 가격 수집 — BAT-D Level 0

WTI, Gold, Copper, DXY 일별 가격을 수집하여 commodity_prices.json 저장.
Alpha Vantage 무료: 25 req/day, 5 req/min.

사용: python -u -X utf8 scripts/fetch_commodity_prices.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
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
OUTPUT_PATH = DATA_DIR / "commodity_prices.json"


# Alpha Vantage 심볼 매핑
COMMODITIES = {
    "wti": {
        "function": "WTI",
        "interval": "daily",
        "label": "WTI Crude Oil",
    },
    "gold": {
        "function": "COPPER",  # placeholder, will use GLOBAL_QUOTE
        "symbol": "GLD",       # Gold ETF as proxy
        "label": "Gold (GLD ETF)",
    },
    "copper": {
        "function": "COPPER",
        "interval": "daily",
        "label": "Copper",
    },
    "dxy": {
        "symbol": "UUP",       # Dollar ETF as proxy
        "label": "DXY (UUP ETF)",
    },
}


def _fetch_commodity(api_key: str, name: str) -> dict | None:
    """Alpha Vantage에서 단일 원자재 가격 조회."""
    import requests

    cfg = COMMODITIES[name]

    # 원자재 전용 endpoint (WTI, Copper)
    if name in ("wti", "copper"):
        func_name = cfg["function"]
        url = (
            f"https://www.alphavantage.co/query"
            f"?function={func_name}"
            f"&interval=daily"
            f"&apikey={api_key}"
        )
        r = requests.get(url, timeout=15)
        data = r.json()

        # "data" 키에 시계열 배열
        series = data.get("data", [])
        if series:
            latest = series[0]  # 최신이 첫 번째
            return {
                "price": float(latest.get("value", 0)),
                "date": latest.get("date", ""),
                "source": "alpha_vantage",
            }
        # 에러 처리
        if "Note" in data or "Information" in data:
            logger.warning(f"  {name}: Rate limit — {data.get('Note', data.get('Information', ''))[:80]}")
        return None

    # ETF proxy (Gold=GLD, DXY=UUP)
    if "symbol" in cfg:
        symbol = cfg["symbol"]
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={api_key}"
        )
        r = requests.get(url, timeout=15)
        data = r.json()

        quote = data.get("Global Quote", {})
        if quote and quote.get("05. price"):
            return {
                "price": float(quote["05. price"]),
                "change_pct": quote.get("10. change percent", "0%").replace("%", ""),
                "date": quote.get("07. latest trading day", ""),
                "symbol": symbol,
                "source": "alpha_vantage",
            }
        if "Note" in data:
            logger.warning(f"  {name}: Rate limit")
        return None

    return None


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip().rstrip(".")
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY 없음. .env 확인 필요.")
        return

    logger.info("=== Alpha Vantage 원자재 가격 수집 ===")

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "source": "alpha_vantage",
        "commodities": {},
    }

    for name in COMMODITIES:
        logger.info(f"  {name} ({COMMODITIES[name]['label']})...")
        data = _fetch_commodity(api_key, name)
        if data:
            result["commodities"][name] = data
            logger.info(f"    → {data.get('price')} ({data.get('date', '')})")
        else:
            logger.warning(f"    → 실패")
        time.sleep(13)  # 5 req/min 제한 → 안전하게 13초 간격

    # 저장
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"저장: {OUTPUT_PATH} ({len(result['commodities'])}종)")


if __name__ == "__main__":
    main()
