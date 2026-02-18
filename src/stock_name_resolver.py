"""종목명 <-> 종목코드 양방향 변환 유틸.

stock_data_daily/ 디렉토리의 CSV 파일명(종목명_종목코드.csv)에서
매핑 테이블을 구축한다.  한 번 로드 후 캐시.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = PROJECT_ROOT / "stock_data_daily"

_name_to_ticker: dict[str, str] = {}  # "삼성전자" -> "005930"
_ticker_to_name: dict[str, str] = {}  # "005930" -> "삼성전자"
_loaded = False


def _load_mapping() -> None:
    global _loaded
    if _loaded:
        return
    if STOCK_DIR.exists():
        for csv_file in STOCK_DIR.glob("*.csv"):
            stem = csv_file.stem
            if stem.startswith("Stock_") or stem.startswith("_"):
                continue
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                name, ticker = parts
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
