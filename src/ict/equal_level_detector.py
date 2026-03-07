"""
Phase 2: Equal High/Low 탐지 — 유동성 축적 포인트 식별

정의:
  Equal Low: 최근 N거래일(기본 60일) 중 일봉 저점이 ±0.3% 이내에
             2회 이상 형성된 가격대 → 손절 주문 축적 구간
  Equal High: 최근 N거래일 중 일봉 고점이 ±0.3% 이내에
              2회 이상 형성된 가격대 → 익절/물타기 주문 축적 구간

의미:
  Equal Low 이탈 후 반등 = 유동성 사냥 완료 → 강한 매수
  Equal High 돌파 = 숏커버 + 추격매수 → 추세 강화

출력: data/equal_levels/{date}.json
갱신: 매일 장마감 후 (BAT-D)
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
EQUAL_LEVELS_DIR = DATA_DIR / "equal_levels"
EQUAL_LEVELS_DIR.mkdir(parents=True, exist_ok=True)

# 기본 파라미터
DEFAULT_LOOKBACK = 60       # 분석 기간 (거래일)
DEFAULT_TOLERANCE = 0.003   # 허용 오차 ±0.3%
MIN_TOUCHES = 2             # 최소 터치 횟수


class EqualLevelDetector:
    """Equal High/Low 클러스터 탐지기"""

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        tolerance: float = DEFAULT_TOLERANCE,
    ):
        self.lookback = lookback
        self.tolerance = tolerance

    def compute_all(self, date_str: str | None = None) -> list[dict]:
        """유니버스 전 종목의 Equal Level 탐지.

        Args:
            date_str: 기준 날짜 (YYYY-MM-DD). None이면 오늘.

        Returns:
            종목별 Equal Level 리스트
        """
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        ref_date = pd.Timestamp(date_str)

        results = []
        parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

        for pf in parquet_files:
            try:
                result = self._detect_one(pf, ref_date)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug("Equal Level 탐지 실패 %s: %s", pf.stem, e)

        # 저장
        out_path = EQUAL_LEVELS_DIR / f"{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"date": date_str, "count": len(results), "levels": results},
                f, ensure_ascii=False, indent=2,
            )
        logger.info("Equal Levels %d종목 탐지 → %s", len(results), out_path)
        return results

    def _detect_one(self, parquet_path: Path, ref_date: pd.Timestamp) -> dict | None:
        """단일 종목 Equal High/Low 탐지"""
        df = pd.read_parquet(parquet_path)
        if df.empty or "high" not in df.columns or "low" not in df.columns:
            return None

        # 날짜 인덱스 정리
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # ref_date 이전 데이터만, lookback 기간
        df = df[df.index < ref_date]
        if len(df) < self.lookback:
            if len(df) < 20:
                return None
            # 데이터 부족하면 있는 만큼 사용
        df = df.tail(self.lookback)

        symbol = parquet_path.stem
        current_price = float(df["close"].iloc[-1])
        if current_price <= 0:
            return None

        # Equal Low 탐지
        lows = list(zip(df.index, df["low"].values))
        equal_lows = self._cluster_levels(lows, current_price, level_type="low")

        # Equal High 탐지
        highs = list(zip(df.index, df["high"].values))
        equal_highs = self._cluster_levels(highs, current_price, level_type="high")

        if not equal_lows and not equal_highs:
            return None

        # 종목명
        name = self._get_stock_name(symbol)

        return {
            "symbol": symbol,
            "name": name,
            "current_price": int(current_price),
            "analysis_period": len(df),
            "equal_lows": equal_lows,
            "equal_highs": equal_highs,
        }

    def _cluster_levels(
        self,
        price_points: list[tuple],
        current_price: float,
        level_type: str,
    ) -> list[dict]:
        """가격 포인트들을 클러스터링하여 Equal Level 탐지.

        Args:
            price_points: [(date, price), ...]
            current_price: 현재가
            level_type: "low" or "high"

        Returns:
            Equal Level 리스트
        """
        if not price_points:
            return []

        # 가격순 정렬 (low는 오름차순, high는 내림차순)
        sorted_points = sorted(price_points, key=lambda x: float(x[1]))
        if level_type == "high":
            sorted_points = list(reversed(sorted_points))

        used = set()
        clusters = []

        for i, (date_i, price_i) in enumerate(sorted_points):
            if i in used:
                continue

            price_i = float(price_i)
            if price_i <= 0:
                continue

            # 이 가격 기준 ±tolerance 범위 내 다른 포인트 찾기
            cluster_dates = [date_i]
            cluster_prices = [price_i]
            used.add(i)

            for j, (date_j, price_j) in enumerate(sorted_points):
                if j in used:
                    continue
                price_j = float(price_j)
                if price_j <= 0:
                    continue

                # 같은 날짜는 스킵
                if date_j == date_i:
                    used.add(j)
                    continue

                # ±tolerance 범위 체크
                if abs(price_j - price_i) / price_i <= self.tolerance:
                    cluster_dates.append(date_j)
                    cluster_prices.append(price_j)
                    used.add(j)

            # MIN_TOUCHES 이상이면 Equal Level
            if len(cluster_dates) >= MIN_TOUCHES:
                center = int(round(sum(cluster_prices) / len(cluster_prices)))
                price_min = int(min(cluster_prices))
                price_max = int(max(cluster_prices))
                distance_pct = round((center - current_price) / current_price * 100, 1)
                strength = "strong" if len(cluster_dates) >= 3 else "normal"

                # 날짜 정렬
                sorted_dates = sorted(cluster_dates)

                clusters.append({
                    "price_center": center,
                    "price_range": [price_min, price_max],
                    "touches": len(cluster_dates),
                    "dates": [d.strftime("%Y-%m-%d") for d in sorted_dates],
                    "strength": strength,
                    "distance_pct": distance_pct,
                })

        # 현재가에서 가까운 순 정렬
        clusters.sort(key=lambda x: abs(x["distance_pct"]))
        return clusters

    _name_cache: dict[str, str] = {}

    @classmethod
    def _get_stock_name(cls, symbol: str) -> str:
        """종목코드 → 종목명"""
        if not cls._name_cache:
            csv_dir = Path("stock_data_daily")
            if csv_dir.exists():
                for csv_file in csv_dir.glob("*.csv"):
                    parts = csv_file.stem.rsplit("_", 1)
                    if len(parts) == 2:
                        cls._name_cache[parts[1]] = parts[0]
        return cls._name_cache.get(symbol, symbol)


# ================================================================
# 편의 함수
# ================================================================

def compute_equal_levels(
    date_str: str | None = None,
    symbols: list[str] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict]:
    """Equal Level 탐지 + 저장."""
    detector = EqualLevelDetector(lookback=lookback, tolerance=tolerance)
    all_levels = detector.compute_all(date_str)

    if symbols:
        all_levels = [lv for lv in all_levels if lv["symbol"] in symbols]

    return all_levels


def load_equal_levels(date_str: str, symbol: str | None = None) -> dict | list | None:
    """저장된 Equal Level 로드."""
    path = EQUAL_LEVELS_DIR / f"{date_str}.json"
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


def format_equal_levels_briefing(
    levels: list[dict],
    symbols: list[str] | None = None,
) -> str:
    """텔레그램 나이트워치 브리핑용 Equal Level 포맷."""
    if symbols:
        levels = [lv for lv in levels if lv["symbol"] in symbols]

    # Equal Level이 있는 종목만
    levels = [lv for lv in levels if lv.get("equal_lows") or lv.get("equal_highs")]

    if not levels:
        return ""

    lines = ["Equal Levels (보유종목)"]
    lines.append("━" * 30)

    for lv in levels:
        name = lv.get("name", lv["symbol"])
        price = lv["current_price"]
        lines.append(f"{name} {price:,}")

        for eq_low in lv.get("equal_lows", [])[:2]:
            star = " ★" if eq_low["strength"] == "strong" else ""
            lines.append(
                f"  ▽ EqLow {eq_low['price_center']:,} "
                f"x{eq_low['touches']} ({eq_low['distance_pct']:+.1f}%){star}"
            )

        for eq_high in lv.get("equal_highs", [])[:2]:
            star = " ★" if eq_high["strength"] == "strong" else ""
            lines.append(
                f"  △ EqHigh {eq_high['price_center']:,} "
                f"x{eq_high['touches']} ({eq_high['distance_pct']:+.1f}%){star}"
            )

    lines.append("━" * 30)
    return "\n".join(lines)


# ================================================================
# 백테스트 유틸
# ================================================================

def backtest_equal_levels(
    lookback: int = DEFAULT_LOOKBACK,
    tolerance: float = DEFAULT_TOLERANCE,
    forward_days: list[int] | None = None,
    test_months: int = 6,
) -> dict:
    """Equal Level 유효성 백테스트.

    전 종목에서 Equal Low/High를 탐지하고,
    탐지 시점 이후 forward_days(5/10/20일) 수익률을 측정.

    Returns:
        {
            "eq_low_reversal": {반전율, 평균수익률, ...},
            "eq_low_sweep_reversal": {sweep 후 반등 적중률, ...},
            "eq_high_breakout": {돌파 후 추세 지속률, ...},
        }
    """
    if forward_days is None:
        forward_days = [5, 10, 20]

    detector = EqualLevelDetector(lookback=lookback, tolerance=tolerance)
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

    # 통계 수집
    eq_low_events = []       # Equal Low 터치 후 결과
    eq_low_sweep = []        # Equal Low 이탈 후 반등
    eq_high_breakout = []    # Equal High 돌파 후 결과

    test_start = pd.Timestamp.now() - pd.Timedelta(days=test_months * 30)
    max_forward = max(forward_days)

    processed = 0
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if df.empty or len(df) < lookback + max_forward + 20:
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                else:
                    df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            symbol = pf.stem

            # 테스트 기간 내 각 날짜에서 Equal Level 탐지
            test_dates = df[df.index >= test_start].index
            if len(test_dates) < max_forward + 1:
                continue

            processed += 1

            for i, ref_date in enumerate(test_dates):
                # 미래 데이터 확보 필요
                future_idx = df.index.get_loc(ref_date)
                if future_idx + max_forward >= len(df):
                    break

                # lookback 구간
                lookback_df = df.iloc[max(0, future_idx - lookback):future_idx]
                if len(lookback_df) < 20:
                    continue

                current_price = float(df.iloc[future_idx]["close"])
                if current_price <= 0:
                    continue

                # 매 5거래일마다만 체크 (속도)
                if i % 5 != 0:
                    continue

                # Equal Low 탐지
                lows = list(zip(lookback_df.index, lookback_df["low"].values))
                eq_lows = detector._cluster_levels(lows, current_price, "low")

                # Equal High 탐지
                highs = list(zip(lookback_df.index, lookback_df["high"].values))
                eq_highs = detector._cluster_levels(highs, current_price, "high")

                # 미래 수익률 계산
                future_returns = {}
                for fd in forward_days:
                    if future_idx + fd < len(df):
                        future_close = float(df.iloc[future_idx + fd]["close"])
                        future_returns[fd] = round(
                            (future_close / current_price - 1) * 100, 2
                        )

                if not future_returns:
                    continue

                prev_day_low = float(df.iloc[future_idx - 1]["low"]) if future_idx > 0 else 0

                # ── Equal Low 이벤트 ──
                for eq in eq_lows:
                    dist = eq["distance_pct"]
                    # 현재가가 Equal Low 위 2% 이내
                    if -2.0 <= dist <= 2.0:
                        eq_low_events.append({
                            "symbol": symbol,
                            "date": ref_date.strftime("%Y-%m-%d"),
                            "eq_price": eq["price_center"],
                            "current_price": int(current_price),
                            "touches": eq["touches"],
                            "strength": eq["strength"],
                            "distance_pct": dist,
                            "returns": future_returns,
                        })

                    # Equal Low Sweep Reversal: 전일 저점이 EqLow 아래 & 오늘 위
                    if prev_day_low > 0 and prev_day_low < eq["price_range"][0] and current_price > eq["price_center"]:
                        eq_low_sweep.append({
                            "symbol": symbol,
                            "date": ref_date.strftime("%Y-%m-%d"),
                            "eq_price": eq["price_center"],
                            "current_price": int(current_price),
                            "prev_low": int(prev_day_low),
                            "touches": eq["touches"],
                            "strength": eq["strength"],
                            "returns": future_returns,
                        })

                # ── Equal High Breakout ──
                for eq in eq_highs:
                    # 현재가가 Equal High 돌파 (위에 있음)
                    if current_price > eq["price_range"][1]:
                        eq_high_breakout.append({
                            "symbol": symbol,
                            "date": ref_date.strftime("%Y-%m-%d"),
                            "eq_price": eq["price_center"],
                            "current_price": int(current_price),
                            "touches": eq["touches"],
                            "strength": eq["strength"],
                            "returns": future_returns,
                        })

        except Exception as e:
            logger.debug("백테스트 실패 %s: %s", pf.stem, e)

    # ── 결과 집계 ──
    result = {
        "params": {
            "lookback": lookback,
            "tolerance": tolerance,
            "test_months": test_months,
            "forward_days": forward_days,
            "symbols_processed": processed,
        },
        "eq_low_reversal": _summarize_events(eq_low_events, forward_days),
        "eq_low_sweep_reversal": _summarize_events(eq_low_sweep, forward_days),
        "eq_high_breakout": _summarize_events(eq_high_breakout, forward_days),
    }

    return result


def _summarize_events(events: list[dict], forward_days: list[int]) -> dict:
    """이벤트 리스트 → 통계 요약."""
    if not events:
        return {"count": 0}

    summary = {"count": len(events)}

    for fd in forward_days:
        returns = [e["returns"].get(fd, 0) for e in events if fd in e.get("returns", {})]
        if not returns:
            continue

        positive = sum(1 for r in returns if r > 0)
        avg_ret = sum(returns) / len(returns)
        median_ret = sorted(returns)[len(returns) // 2]

        summary[f"{fd}d"] = {
            "count": len(returns),
            "win_rate": round(positive / len(returns) * 100, 1),
            "avg_return": round(avg_ret, 2),
            "median_return": round(median_ret, 2),
            "max_return": round(max(returns), 2),
            "min_return": round(min(returns), 2),
        }

    # strength별 분류
    strong = [e for e in events if e.get("strength") == "strong"]
    normal = [e for e in events if e.get("strength") == "normal"]
    summary["by_strength"] = {
        "strong": len(strong),
        "normal": len(normal),
    }

    return summary
