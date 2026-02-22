"""
자비스 컨트롤 타워 — 대시보드 데이터 제공자

JSON 파일 기반으로 6대 시그널 데이터를 통합 로드.
FastAPI 어댑터에서 호출하여 브라우저에 전달.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class DashboardDataProvider:
    """JSON 파일 통합 로더 + TTL 캐시."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl  # 초
        self._cache: dict[str, tuple[float, dict | list]] = {}

    # ──────────────────────────────────────────
    # 메인 API
    # ──────────────────────────────────────────

    def get_full_dashboard(self) -> dict:
        """전체 대시보드 데이터 통합 반환."""
        market = self._get_market_data()
        relay = self._load_json("sector_rotation/relay_trading_signal.json")
        etf = self._load_json("sector_rotation/etf_trading_signal.json")
        group_relay = self._load_json("group_relay/group_relay_today.json")
        quantum = self._load_json("scan_cache.json")
        positions = self._load_json("positions.json")
        theme = self._load_json("theme_alert_history.json")
        summary = self._load_json("integrated_report.json")

        return {
            "date": date.today().isoformat(),
            "generated_at": summary.get("generated_at", ""),
            "market": market,
            "positions": positions,
            "quantum": quantum,
            "relay": relay,
            "group_relay": group_relay,
            "etf": etf,
            "etf_master": self._load_json("etf_master.json"),
            "theme": theme if isinstance(theme, list) else [],
            "dual_buying": self._load_json("dual_buying_watch.json"),
            "pullback": self._load_json("pullback_scan.json"),
            "whale_detect": self._load_json("whale_detect.json"),
            "tomorrow_picks": self._load_json("tomorrow_picks.json"),
            "picks_history": self._load_json("picks_history.json"),
            "market_news": self._load_json("market_news.json"),
            "summary": summary,
        }

    def get_relay(self) -> dict:
        return self._load_json("sector_rotation/relay_trading_signal.json")

    def get_etf(self) -> dict:
        return self._load_json("sector_rotation/etf_trading_signal.json")

    def get_quantum(self) -> dict:
        return self._load_json("scan_cache.json")

    def get_positions(self) -> dict:
        return self._load_json("positions.json")

    def get_us_overnight(self) -> dict:
        return self._load_json("us_market/overnight_signal.json")

    def clear_cache(self) -> None:
        """캐시 강제 초기화."""
        self._cache.clear()
        logger.info("대시보드 캐시 초기화")

    # ──────────────────────────────────────────
    # 시장 데이터 통합
    # ──────────────────────────────────────────

    def _get_market_data(self) -> dict:
        """US Overnight + KOSPI 레짐 + 스탠스 통합."""
        us = self._load_json("us_market/overnight_signal.json")
        kospi = self._calc_kospi_regime()
        summary = self._load_json("integrated_report.json")

        # L2 패턴매칭 기반 KOSPI 예측
        l2 = us.get("l2_pattern", {})
        l2_kospi = l2.get("kospi", {})
        l2_gap = l2.get("kospi_open_gap", {})
        mean_chg = l2_kospi.get("mean_chg", 0)
        median_chg = l2_kospi.get("median_chg", 0)
        std_chg = l2_kospi.get("std", 0)
        up_prob = l2_kospi.get("positive_rate", 50)

        kospi_forecast = {
            "up_prob": round(up_prob, 1),
            "down_prob": round(100 - up_prob, 1),
            "mean_chg": round(mean_chg, 2),
            "median_chg": round(median_chg, 2),
            "range_low": round(median_chg, 2),
            "range_high": round(mean_chg + std_chg * 0.5, 2),
            "gap_mean": round(l2_gap.get("mean_gap", 0), 2),
            "gap_median": round(l2_gap.get("median_gap", 0), 2),
            "sample_count": l2.get("sample_count", 0),
            "l2_sectors": l2.get("sectors", {}),
        }

        return {
            "stance": summary.get("market_stance", "관망"),
            "us_grade": us.get("grade", "NEUTRAL"),
            "us_score": us.get("combined_score_100", 0),
            "us_summary": us.get("summary", ""),
            "kospi_regime": kospi.get("regime", "CAUTION"),
            "kospi_slots": kospi.get("slots", 3),
            "kospi_close": kospi.get("close", 0),
            "kospi_change": kospi.get("change", 0),
            "vix": us.get("vix", {}),
            "index_direction": us.get("index_direction", {}),
            "special_rules": us.get("special_rules", []),
            "sector_kills": us.get("sector_kills", {}),
            "kospi_forecast": kospi_forecast,
        }

    def _calc_kospi_regime(self) -> dict:
        """KOSPI 레짐 계산 (캐시 적용)."""
        cache_key = "__kospi_regime__"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        kospi_path = DATA_DIR / "kospi_index.csv"
        if not kospi_path.exists():
            return {"regime": "CAUTION", "slots": 3, "close": 0, "change": 0}

        try:
            df = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma60"] = df["close"].rolling(60).mean()

            if len(df) < 60:
                return {"regime": "CAUTION", "slots": 3, "close": 0, "change": 0}

            row = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else row
            close = float(row["close"])
            ma20 = float(row["ma20"]) if not pd.isna(row["ma20"]) else 0
            ma60 = float(row["ma60"]) if not pd.isna(row["ma60"]) else 0

            log_ret = np.log(df["close"] / df["close"].shift(1))
            rv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
            rv20_pct = rv20.rolling(252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            rv_pct = float(rv20_pct.iloc[-1]) if not pd.isna(rv20_pct.iloc[-1]) else 0.5

            if ma20 == 0 or ma60 == 0:
                regime, slots = "CAUTION", 3
            elif close > ma20:
                regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
            elif close > ma60:
                regime, slots = "BEAR", 2
            else:
                regime, slots = "CRISIS", 0

            prev_close = float(prev["close"]) if not pd.isna(prev["close"]) else close
            change = round((close / prev_close - 1) * 100, 2) if prev_close > 0 else 0

            result = {
                "regime": regime,
                "slots": slots,
                "close": round(close, 2),
                "change": change,
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "rv_pct": round(rv_pct, 2),
            }
            self._cache[cache_key] = (time.time(), result)
            return result
        except Exception as e:
            logger.error("KOSPI 레짐 계산 실패: %s", e)
            return {"regime": "CAUTION", "slots": 3, "close": 0, "change": 0}

    # ──────────────────────────────────────────
    # JSON 파일 로더 + 캐시
    # ──────────────────────────────────────────

    def _load_json(self, rel_path: str) -> dict | list:
        """JSON 파일 로드 (TTL 캐시)."""
        cached = self._cache.get(rel_path)
        if cached and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        full_path = DATA_DIR / rel_path
        if not full_path.exists():
            logger.debug("JSON 파일 없음: %s", full_path)
            return {}

        try:
            with open(full_path, encoding="utf-8") as f:
                data = json.load(f)
            self._cache[rel_path] = (time.time(), data)
            return data
        except Exception as e:
            logger.warning("JSON 로드 실패 (%s): %s", rel_path, e)
            return {}
