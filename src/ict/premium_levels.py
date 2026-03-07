"""
Step 1: 프리미엄 유동성 레벨 — 전일/주간/월간 고저 자동 계산

일봉 데이터만 사용. 분봉 불필요.
매일 장 시작 전 또는 나이트워치 시점에 계산.

출력: data/premium_levels/{date}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "processed"
PREMIUM_DIR = DATA_DIR / "premium_levels"
PREMIUM_DIR.mkdir(parents=True, exist_ok=True)


class PremiumLevelCalculator:
    """전일/주간/월간 프리미엄 유동성 레벨 계산기"""

    def __init__(self, parquet_dir: Path | None = None):
        self.parquet_dir = parquet_dir or PARQUET_DIR

    def compute_all(self, date_str: str | None = None) -> list[dict]:
        """유니버스 전 종목의 프리미엄 레벨 계산.

        Args:
            date_str: 기준 날짜 (YYYY-MM-DD). None이면 오늘.

        Returns:
            종목별 프리미엄 레벨 리스트
        """
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        ref_date = pd.Timestamp(date_str)

        results = []
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        for pf in parquet_files:
            try:
                result = self._compute_one(pf, ref_date)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug("프리미엄 레벨 계산 실패 %s: %s", pf.stem, e)

        # 저장
        out_path = PREMIUM_DIR / f"{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"date": date_str, "count": len(results), "levels": results},
                f, ensure_ascii=False, indent=2,
            )
        logger.info("프리미엄 레벨 %d종목 계산 → %s", len(results), out_path)
        return results

    def _compute_one(self, parquet_path: Path, ref_date: pd.Timestamp) -> dict | None:
        """단일 종목 프리미엄 레벨 계산"""
        df = pd.read_parquet(parquet_path)
        if df.empty or "close" not in df.columns:
            return None

        # 날짜 인덱스 정리
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # ref_date 이전 데이터만
        df = df[df.index < ref_date]
        if len(df) < 5:
            return None

        symbol = parquet_path.stem
        current_price = int(df["close"].iloc[-1])

        # ── 전일 고저 ──
        prev_day = df.iloc[-1]
        prev_day_high = int(prev_day["high"])
        prev_day_low = int(prev_day["low"])

        # ── 주간 고저 (직전 완성된 주봉) ──
        prev_week_high, prev_week_low = self._calc_prev_week(df, ref_date)

        # ── 월간 고저 (직전 완성된 월봉) ──
        prev_month_high, prev_month_low = self._calc_prev_month(df, ref_date)

        # ── 거리 계산 ──
        levels = {
            "prev_day_high": prev_day_high,
            "prev_day_low": prev_day_low,
            "prev_week_high": prev_week_high,
            "prev_week_low": prev_week_low,
            "prev_month_high": prev_month_high,
            "prev_month_low": prev_month_low,
        }

        distances = {}
        for key, price in levels.items():
            if price > 0 and current_price > 0:
                pct = (price - current_price) / current_price * 100
                distances[f"to_{key}"] = round(pct, 2)

        # ── 가장 가까운 저항/지지 ──
        resistances = [
            (k, v) for k, v in levels.items() if v > current_price
        ]
        supports = [
            (k, v) for k, v in levels.items() if v < current_price
        ]

        nearest_resistance = None
        if resistances:
            nearest = min(resistances, key=lambda x: x[1] - current_price)
            dist_pct = round((nearest[1] - current_price) / current_price * 100, 2)
            nearest_resistance = {
                "level": nearest[0], "price": nearest[1], "distance_pct": dist_pct,
            }

        nearest_support = None
        if supports:
            nearest = max(supports, key=lambda x: x[1])
            dist_pct = round((nearest[1] - current_price) / current_price * 100, 2)
            nearest_support = {
                "level": nearest[0], "price": nearest[1], "distance_pct": dist_pct,
            }

        # ── 종목명 추출 ──
        name = self._get_stock_name(symbol)

        return {
            "symbol": symbol,
            "name": name,
            "current_price": current_price,
            "levels": levels,
            "distances": distances,
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "or_high": None,   # Step 2에서 장중 채움
            "or_low": None,
            "ir_high": None,
            "ir_low": None,
        }

    def _calc_prev_week(self, df: pd.DataFrame, ref_date: pd.Timestamp) -> tuple[int, int]:
        """직전 완성된 주봉의 고저"""
        # ref_date가 속한 주의 월요일
        ref_weekday = ref_date.weekday()  # 0=월, 6=일
        this_monday = ref_date - pd.Timedelta(days=ref_weekday)
        prev_monday = this_monday - pd.Timedelta(days=7)
        prev_friday = this_monday - pd.Timedelta(days=1)

        week_data = df[(df.index >= prev_monday) & (df.index <= prev_friday)]
        if week_data.empty:
            # 최근 5거래일 fallback
            week_data = df.tail(5)

        return int(week_data["high"].max()), int(week_data["low"].min())

    def _calc_prev_month(self, df: pd.DataFrame, ref_date: pd.Timestamp) -> tuple[int, int]:
        """직전 완성된 월봉의 고저"""
        # 전월
        if ref_date.month == 1:
            prev_year, prev_month = ref_date.year - 1, 12
        else:
            prev_year, prev_month = ref_date.year, ref_date.month - 1

        month_data = df[(df.index.year == prev_year) & (df.index.month == prev_month)]
        if month_data.empty:
            # 최근 20거래일 fallback
            month_data = df.tail(20)

        return int(month_data["high"].max()), int(month_data["low"].min())

    _name_cache: dict[str, str] = {}

    @classmethod
    def _get_stock_name(cls, symbol: str) -> str:
        """종목코드 → 종목명 (CSV에서 추출)"""
        if not cls._name_cache:
            csv_dir = Path("stock_data_daily")
            for csv_file in csv_dir.glob("*.csv"):
                # 파일명: "삼성전자_005930.csv" 형태
                parts = csv_file.stem.rsplit("_", 1)
                if len(parts) == 2:
                    cls._name_cache[parts[1]] = parts[0]
        return cls._name_cache.get(symbol, symbol)


def compute_premium_levels(
    date_str: str | None = None,
    symbols: list[str] | None = None,
) -> list[dict]:
    """편의 함수: 프리미엄 레벨 계산 + 저장

    Args:
        date_str: 기준 날짜
        symbols: 특정 종목만 계산 (None이면 전체)

    Returns:
        종목별 프리미엄 레벨
    """
    calc = PremiumLevelCalculator()
    all_levels = calc.compute_all(date_str)

    if symbols:
        all_levels = [lv for lv in all_levels if lv["symbol"] in symbols]

    return all_levels


def load_premium_levels(date_str: str, symbol: str | None = None) -> dict | list | None:
    """저장된 프리미엄 레벨 로드

    Args:
        date_str: 날짜 (YYYY-MM-DD)
        symbol: 특정 종목 (None이면 전체)

    Returns:
        dict(단일종목) 또는 list(전체) 또는 None
    """
    path = PREMIUM_DIR / f"{date_str}.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    levels = data.get("levels", [])
    if symbol:
        for lv in levels:
            if lv["symbol"] == symbol:
                return lv
        return None
    return levels


def format_premium_briefing(levels: list[dict], symbols: list[str] | None = None) -> str:
    """텔레그램 나이트워치 브리핑용 프리미엄 레벨 포맷

    Args:
        levels: 프리미엄 레벨 리스트
        symbols: 표시할 종목 필터 (None이면 전체)

    Returns:
        텔레그램 출력 문자열
    """
    if symbols:
        levels = [lv for lv in levels if lv["symbol"] in symbols]

    if not levels:
        return ""

    lines = ["프리미엄 레벨 (보유종목)"]
    lines.append("━" * 30)

    for lv in levels:
        name = lv.get("name", lv["symbol"])
        price = lv["current_price"]
        lines.append(f"{name} {price:,}")

        res = lv.get("nearest_resistance")
        if res:
            level_name = _level_display(res["level"])
            lines.append(
                f"  ↑ {level_name} {res['price']:,} ({res['distance_pct']:+.1f}%)"
            )

        sup = lv.get("nearest_support")
        if sup:
            level_name = _level_display(sup["level"])
            lines.append(
                f"  ↓ {level_name} {sup['price']:,} ({sup['distance_pct']:+.1f}%)"
            )

    lines.append("━" * 30)
    return "\n".join(lines)


def _level_display(level_key: str) -> str:
    """레벨 키 → 한글 표시"""
    mapping = {
        "prev_day_high": "전일고",
        "prev_day_low": "전일저",
        "prev_week_high": "주간고",
        "prev_week_low": "주간저",
        "prev_month_high": "월간고",
        "prev_month_low": "월간저",
    }
    return mapping.get(level_key, level_key)
