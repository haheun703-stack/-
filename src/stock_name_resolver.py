"""종목명 <-> 종목코드 양방향 변환 유틸.

소스 우선순위: universe.csv → stock_data_daily/ CSV → pykrx 폴백.
한 번 로드 후 캐시.
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = PROJECT_ROOT / "stock_data_daily"
UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe.csv"

_name_to_ticker: dict[str, str] = {}  # "삼성전자" -> "005930"
_ticker_to_name: dict[str, str] = {}  # "005930" -> "삼성전자"
_loaded = False


def _load_mapping() -> None:
    global _loaded
    if _loaded:
        return
    # 1) universe.csv (최우선)
    if UNIVERSE_CSV.exists():
        try:
            with open(UNIVERSE_CSV, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get("ticker", row.get("code", "")).strip()
                    name = row.get("name", row.get("종목명", "")).strip()
                    if ticker and name:
                        _name_to_ticker[name] = ticker
                        _ticker_to_name[ticker] = name
        except Exception:
            pass
    # 2) stock_data_daily/ CSV 파일명 (폴백)
    if STOCK_DIR.exists():
        for csv_file in STOCK_DIR.glob("*.csv"):
            stem = csv_file.stem
            if stem.startswith("Stock_") or stem.startswith("_"):
                continue
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                name, ticker = parts
                if ticker not in _ticker_to_name:
                    _name_to_ticker[name] = ticker
                    _ticker_to_name[ticker] = name
    _loaded = True


def resolve_name(query: str) -> list[tuple[str, str]]:
    """종목명 검색 (부분 매칭).  Returns: [(name, ticker), ...]."""
    _load_mapping()
    q = query.strip()
    if not q:
        return []
    # 정확 매칭 우선
    exact = [(n, t) for n, t in _name_to_ticker.items() if n == q]
    if exact:
        return exact
    # 부분 매칭 (contains)
    return [(n, t) for n, t in _name_to_ticker.items() if q in n][:10]


def name_to_ticker(name: str) -> str | None:
    """종목명 -> 종목코드 (정확 매칭)."""
    _load_mapping()
    return _name_to_ticker.get(name.strip())


def ticker_to_name(ticker: str) -> str:
    """종목코드 -> 종목명.  없으면 코드 그대로 반환."""
    _load_mapping()
    return _ticker_to_name.get(ticker, ticker)
