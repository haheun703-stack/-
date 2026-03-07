"""
Step 2: OR/IR 자동 기록 — Opening Range / Initial Range 계산

기존 5분봉 parquet(data/intraday/5min/{date}/)에서 OR/IR을 계산.
신규 분봉 수집 불필요 — BAT-D 장마감 후 아카이브된 데이터 사용.

실행 타이밍:
  - 장마감 후(15:35): 당일 OR/IR 확정 + daily_bias 판정
  - 나이트워치(17:00): OR/IR 리뷰 브리핑 출력

출력: data/daily/or_ir/{date}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INTRADAY_5MIN_DIR = DATA_DIR / "intraday" / "5min"
OR_IR_DIR = DATA_DIR / "daily" / "or_ir"
OR_IR_DIR.mkdir(parents=True, exist_ok=True)

# ATR 계산용 일봉 데이터
PARQUET_DIR = DATA_DIR / "processed"


class OpeningRangeCalculator:
    """OR(09:00~09:30) / IR(09:00~10:00) 자동 계산기"""

    def compute_all(self, date_str: str | None = None) -> list[dict]:
        """당일 전 종목의 OR/IR 계산.

        Args:
            date_str: 날짜 (YYYY-MM-DD). None이면 오늘.

        Returns:
            종목별 OR/IR 리스트
        """
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        candle_dir = INTRADAY_5MIN_DIR / date_str

        if not candle_dir.exists():
            logger.warning("5분봉 디렉토리 없음: %s", candle_dir)
            return []

        results = []
        for pf in sorted(candle_dir.glob("*.parquet")):
            try:
                result = self._compute_one(pf, date_str)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug("OR/IR 계산 실패 %s: %s", pf.stem, e)

        # 저장
        out_path = OR_IR_DIR / f"{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"date": date_str, "count": len(results), "records": results},
                f, ensure_ascii=False, indent=2,
            )
        logger.info("OR/IR %d종목 계산 → %s", len(results), out_path)
        return results

    def _compute_one(self, parquet_path: Path, date_str: str) -> dict | None:
        """단일 종목 OR/IR 계산"""
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return None

        # timestamp 컬럼 정리
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            return None

        symbol = parquet_path.stem

        # ── OR: 09:00~09:30 (5분봉 6개) ──
        or_mask = (
            (df["timestamp"] >= f"{date_str} 09:00:00")
            & (df["timestamp"] < f"{date_str} 09:30:00")
        )
        or_candles = df[or_mask]

        if or_candles.empty:
            return None

        or_high = int(or_candles["high"].max())
        or_low = int(or_candles["low"].min())
        or_range = or_high - or_low
        or_range_pct = round(or_range / or_low * 100, 2) if or_low > 0 else 0

        # ── IR: 09:00~10:00 (5분봉 12개) ──
        ir_mask = (
            (df["timestamp"] >= f"{date_str} 09:00:00")
            & (df["timestamp"] < f"{date_str} 10:00:00")
        )
        ir_candles = df[ir_mask]

        ir_high = int(ir_candles["high"].max())
        ir_low = int(ir_candles["low"].min())
        ir_range_pct = round((ir_high - ir_low) / ir_low * 100, 2) if ir_low > 0 else 0

        # ── Daily Bias: 10:00 시점 종가 기준 ──
        candle_10 = df[df["timestamp"] == f"{date_str} 10:00:00"]
        if not candle_10.empty:
            close_at_10 = int(candle_10["close"].iloc[0])
            bias_time = "10:00"
        else:
            # 10:00 봉이 없으면 IR 마지막 봉 종가
            close_at_10 = int(ir_candles["close"].iloc[-1])
            bias_time = ir_candles["timestamp"].iloc[-1].strftime("%H:%M")

        if close_at_10 > or_high:
            daily_bias = "bullish"
        elif close_at_10 < or_low:
            daily_bias = "bearish"
        else:
            daily_bias = "neutral"

        # ── OR vs ATR (축적 판정) ──
        or_vs_atr = self._calc_or_vs_atr(symbol, or_range)

        # ── 종가 vs OR (종일 결과) ──
        last_candle = df.iloc[-1]
        closing_price = int(last_candle["close"])

        if closing_price > or_high:
            close_vs_or = "above"
        elif closing_price < or_low:
            close_vs_or = "below"
        else:
            close_vs_or = "inside"

        # ── 데이터 품질 ──
        n_or_candles = len(or_candles)
        n_ir_candles = len(ir_candles)
        if n_or_candles >= 6 and n_ir_candles >= 12:
            data_quality = "full"
        elif n_or_candles >= 3:
            data_quality = "partial"
        else:
            data_quality = "insufficient"

        return {
            "date": date_str,
            "symbol": symbol,
            "or_high": or_high,
            "or_low": or_low,
            "or_range_pct": or_range_pct,
            "ir_high": ir_high,
            "ir_low": ir_low,
            "ir_range_pct": ir_range_pct,
            "daily_bias": daily_bias,
            "bias_trigger_time": bias_time,
            "close_at_10": close_at_10,
            "closing_price": closing_price,
            "close_vs_or": close_vs_or,
            "or_vs_atr": or_vs_atr,
            "data_quality": data_quality,
            "n_or_candles": n_or_candles,
            "n_ir_candles": n_ir_candles,
        }

    def _calc_or_vs_atr(self, symbol: str, or_range: int) -> float | None:
        """OR 폭 / 20일 ATR 비율"""
        parquet_path = PARQUET_DIR / f"{symbol}.parquet"
        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(parquet_path)
            if len(df) < 20 or "high" not in df.columns or "low" not in df.columns:
                return None

            # True Range 계산
            df = df.tail(21).copy()
            df["prev_close"] = df["close"].shift(1)
            df["tr"] = df.apply(
                lambda r: max(
                    r["high"] - r["low"],
                    abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                    abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                ),
                axis=1,
            )
            atr_20 = df["tr"].tail(20).mean()

            if atr_20 > 0:
                return round(or_range / atr_20, 2)
        except Exception:
            pass
        return None


def compute_or_ir(date_str: str | None = None) -> list[dict]:
    """편의 함수: OR/IR 계산 + 저장"""
    calc = OpeningRangeCalculator()
    return calc.compute_all(date_str)


def load_or_ir(date_str: str, symbol: str | None = None) -> dict | list | None:
    """저장된 OR/IR 로드"""
    path = OR_IR_DIR / f"{date_str}.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if symbol:
        for rec in records:
            if rec["symbol"] == symbol:
                return rec
        return None
    return records


def backfill_or_ir(days: int = 20) -> dict:
    """과거 N거래일 OR/IR 백필 (이미 수집된 5분봉 사용)

    Returns:
        {"dates_processed": N, "total_records": M}
    """
    calc = OpeningRangeCalculator()
    dates = sorted(d.name for d in INTRADAY_5MIN_DIR.iterdir() if d.is_dir())

    # 최근 N일만
    dates = dates[-days:]
    total = 0

    for date_str in dates:
        out_path = OR_IR_DIR / f"{date_str}.json"
        if out_path.exists():
            logger.debug("OR/IR 이미 존재: %s", date_str)
            # 재계산 (덮어쓰기)
        results = calc.compute_all(date_str)
        total += len(results)

    return {"dates_processed": len(dates), "total_records": total}


def measure_bias_accuracy(days: int = 20) -> dict:
    """daily_bias vs 실제 종가 방향 일치율 측정

    Returns:
        {"total": N, "correct": M, "accuracy_pct": X,
         "by_bias": {"bullish": {total, correct}, ...}}
    """
    dates = sorted(d.stem for d in OR_IR_DIR.glob("*.json") if d.stem != "_index")
    dates = dates[-days:]

    stats = {"total": 0, "correct": 0}
    by_bias = {
        "bullish": {"total": 0, "correct": 0},
        "bearish": {"total": 0, "correct": 0},
        "neutral": {"total": 0, "correct": 0},
    }

    for date_str in dates:
        records = load_or_ir(date_str)
        if not records:
            continue

        for rec in records:
            bias = rec.get("daily_bias", "unknown")
            if bias == "unknown" or rec.get("data_quality") == "insufficient":
                continue

            close_vs_or = rec.get("close_vs_or", "inside")
            stats["total"] += 1
            by_bias.setdefault(bias, {"total": 0, "correct": 0})
            by_bias[bias]["total"] += 1

            # 정확도 판정: bias 방향과 종가 위치가 일치하는가
            correct = False
            if bias == "bullish" and close_vs_or == "above":
                correct = True
            elif bias == "bearish" and close_vs_or == "below":
                correct = True
            elif bias == "neutral" and close_vs_or == "inside":
                correct = True

            if correct:
                stats["correct"] += 1
                by_bias[bias]["correct"] += 1

    stats["accuracy_pct"] = (
        round(stats["correct"] / stats["total"] * 100, 1)
        if stats["total"] > 0 else 0
    )
    stats["by_bias"] = by_bias

    return stats


def format_or_ir_briefing(records: list[dict], symbols: list[str] | None = None) -> str:
    """텔레그램 나이트워치 OR/IR 리뷰 포맷"""
    if symbols:
        records = [r for r in records if r["symbol"] in symbols]

    if not records:
        return ""

    # KOSPI 대표 ETF 먼저
    kospi_etf = [r for r in records if r["symbol"] == "069500"]
    others = [r for r in records if r["symbol"] != "069500"]

    lines = ["OR/IR 리뷰"]
    lines.append("━" * 30)

    for rec in kospi_etf + others:
        symbol = rec["symbol"]
        or_h = rec["or_high"]
        or_l = rec["or_low"]
        bias = rec["daily_bias"]
        bias_time = rec.get("bias_trigger_time", "?")

        label = "KOSPI" if symbol == "069500" else symbol
        bias_emoji = {"bullish": "↑", "bearish": "↓", "neutral": "−"}.get(bias, "?")

        lines.append(
            f"{label} OR: {or_l:,}~{or_h:,} | bias: {bias} {bias_emoji} ({bias_time})"
        )

    lines.append("━" * 30)
    return "\n".join(lines)
