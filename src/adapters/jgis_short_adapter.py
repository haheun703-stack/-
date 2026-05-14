"""정보봇 KIS API 공매도 3종 데이터 어댑터

데이터 소스 (2곳):
  1. D:/Global_Stock_Overview_Scripter_정보봇/data/supply_tracker/{ticker}.csv
     → 2,536 종목, 일별 누적 (공매도+신용+대차)
  2. D:/shared-bot-data/jgis_to_quant/daily_intelligence.json
     → short_selling_summary (TOP 30, 빠른 조회 fallback)

use_short_selling_filter (pykrx 기반)와 독립 — jgis_short_selling.enabled로 제어.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# OS별 기본 경로 (Windows 로컬 / Linux VPS 자동 분기)
_IS_POSIX = os.name == "posix"
_DEFAULT_CSV_DIR = (
    "/home/ubuntu/jgis/data/supply_tracker" if _IS_POSIX
    else "D:/Global_Stock_Overview_Scripter_정보봇/data/supply_tracker"
)
_DEFAULT_JGIS_JSON = (
    "/home/ubuntu/shared-bot-data/jgis_to_quant/daily_intelligence.json" if _IS_POSIX
    else "D:/shared-bot-data/jgis_to_quant/daily_intelligence.json"
)
_DEFAULT_FLOW_DIR = (
    "/home/ubuntu/jgis/data/supply_daily" if _IS_POSIX
    else "D:/Global_Stock_Overview_Scripter_정보봇/data/supply_daily"
)

JGIS_JSON_PATH = Path(_DEFAULT_JGIS_JSON)


def _safe_float(val, default: float = 0.0) -> float:
    """NaN/None/빈문자열 안전 변환."""
    try:
        v = float(val) if val not in (None, "", "nan", "NaN") else default
        return default if v != v else v  # NaN check
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """정수 안전 변환."""
    try:
        return int(float(val)) if val not in (None, "", "nan", "NaN") else default
    except (TypeError, ValueError):
        return default


class JgisShortAdapter:
    """정보봇 공매도 3종 데이터 통합 읽기 어댑터."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = self._load_config()

        jgis_cfg = config.get("jgis_short_selling", {})
        self.enabled = jgis_cfg.get("enabled", False)
        self.csv_dir = Path(jgis_cfg.get("csv_dir", _DEFAULT_CSV_DIR))
        self.lookback_days = jgis_cfg.get("lookback_days", 20)

    @staticmethod
    def _load_config() -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ──────────────────────────────────────────
    # CSV 로드 (종목별 supply_tracker)
    # ──────────────────────────────────────────
    def load_ticker_csv(self, ticker: str, lookback_days: int | None = None) -> list[dict]:
        """개별 종목 CSV 로드 → 최근 N일 dict 리스트 반환.

        2026-05-13 이전 행은 새 컬럼이 0/빈칸 → _safe_float/_safe_int 처리.
        """
        days = lookback_days or self.lookback_days
        path = self.csv_dir / f"{ticker}.csv"
        if not path.exists():
            return []

        try:
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            logger.warning("CSV 읽기 실패 %s: %s", ticker, e)
            return []

        if not rows:
            return []

        # 최근 N일만
        recent = rows[-days:] if len(rows) >= days else rows

        # 정규화
        result = []
        for row in recent:
            result.append({
                "date": row.get("date", ""),
                "short_selling_qty": _safe_int(row.get("short_selling_qty")),
                "loan_balance_rate": _safe_float(row.get("loan_balance_rate")),
                "short_overheat": _safe_int(row.get("short_overheat")),
                "price": _safe_float(row.get("price")),
                "change_pct": _safe_float(row.get("change_pct")),
                "credit_balance_qty": _safe_int(row.get("credit_balance_qty")),
                "credit_balance_rate": _safe_float(row.get("credit_balance_rate")),
                "loan_new_qty": _safe_int(row.get("loan_new_qty")),
                "loan_repay_qty": _safe_int(row.get("loan_repay_qty")),
                "loan_balance_qty": _safe_int(row.get("loan_balance_qty")),
                "exec_strength": _safe_float(row.get("exec_strength")),
                "foreign_net_amt": _safe_float(row.get("foreign_net_amt")),
                "inst_net_amt": _safe_float(row.get("inst_net_amt")),
            })
        return result

    def load_batch(
        self,
        tickers: list[str],
        lookback_days: int | None = None,
        max_workers: int = 8,
    ) -> dict[str, list[dict]]:
        """유니버스 배치 로드 (ThreadPoolExecutor 병렬)."""
        days = lookback_days or self.lookback_days
        result: dict[str, list[dict]] = {}

        def _load(t: str) -> tuple[str, list[dict]]:
            return t, self.load_ticker_csv(t, days)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load, t): t for t in tickers}
            for fut in as_completed(futures):
                try:
                    ticker, rows = fut.result()
                    if rows:
                        result[ticker] = rows
                except Exception:
                    pass

        return result

    # ──────────────────────────────────────────
    # JSON fallback (daily_intelligence.json TOP 30)
    # ──────────────────────────────────────────
    def load_jgis_top30(self) -> dict[str, dict]:
        """daily_intelligence.json → short_selling_summary.data (TOP 30).

        Returns: {ticker: {short_ratio, credit_balance_rate, loan_balance_qty, ...}}
        """
        if not JGIS_JSON_PATH.exists():
            return {}

        try:
            with open(JGIS_JSON_PATH, encoding="utf-8") as f:
                intel = json.load(f)
        except Exception as e:
            logger.warning("daily_intelligence.json 로드 실패: %s", e)
            return {}

        summary = intel.get("short_selling_summary", {})
        return summary.get("data", {})

    def load_jgis_signals(self) -> list[dict]:
        """daily_intelligence.json → short_selling_summary.signals."""
        if not JGIS_JSON_PATH.exists():
            return []

        try:
            with open(JGIS_JSON_PATH, encoding="utf-8") as f:
                intel = json.load(f)
        except Exception:
            return []

        summary = intel.get("short_selling_summary", {})
        return summary.get("signals", [])

    # ──────────────────────────────────────────
    # 유니버스 종목 리스트 조회
    # ──────────────────────────────────────────
    def list_available_tickers(self) -> list[str]:
        """CSV 디렉토리에서 사용 가능한 종목 코드 리스트."""
        if not self.csv_dir.exists():
            return []
        return sorted(p.stem for p in self.csv_dir.glob("*.csv"))

    # ──────────────────────────────────────────
    # 단일 종목 최신 데이터 요약
    # ──────────────────────────────────────────
    def get_latest(self, ticker: str) -> dict | None:
        """단일 종목 최신 행 반환."""
        rows = self.load_ticker_csv(ticker, lookback_days=5)
        if not rows:
            return None
        return rows[-1]


# ══════════════════════════════════════════════════════════
# 투자자 수급 통합본 (pykrx investor_flow) 로드 함수
# ══════════════════════════════════════════════════════════

def _load_inv_flow_cfg() -> dict:
    """settings.yaml에서 investor_flow_pykrx 블록 로드 (1회용 helper)."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("investor_flow_pykrx", {})
    except Exception:
        return {}


def _resolve_flow_dir(flow_dir: str | None = None) -> Path:
    """investor_flow JSON 디렉토리 결정."""
    if flow_dir:
        return Path(flow_dir)
    return Path(_load_inv_flow_cfg().get("flow_dir", _DEFAULT_FLOW_DIR))


def load_investor_flow(date_str: str | None = None, flow_dir: str | None = None) -> dict | None:
    """정보봇 supply_daily/{date}_investor_flow.json 로드.

    Args:
        date_str: "2026-05-13" 형식. None이면 오늘→어제→그제 자동 탐색.
        flow_dir: 디렉토리 경로. None이면 settings.yaml 참조.

    Returns:
        파싱된 JSON dict 또는 None.
    """
    from datetime import datetime, timedelta

    base = _resolve_flow_dir(flow_dir)
    if not base.exists():
        logger.warning("investor_flow 디렉토리 없음: %s", base)
        return None

    # 날짜 탐색: 지정 → 오늘 → 최근 N일 역순 (lookback_days yaml 연동, 연휴 대비 +2일 여유)
    if date_str:
        candidates = [date_str]
    else:
        today = datetime.now()
        _lookback = _load_inv_flow_cfg().get("lookback_days", 5)
        # 주말/공휴일 fallback 여유: lookback + 4일까지 탐색
        candidates = [(today - timedelta(days=d)).strftime("%Y-%m-%d")
                      for d in range(_lookback + 4)]

    for ds in candidates:
        path = base / f"{ds}_investor_flow.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                # top 리스트 존재 여부 체크
                has_top = bool(data.get("foreign_top_buy") or data.get("foreign_top_sell"))
                if not has_top:
                    logger.info("investor_flow %s: TOP 리스트 비어있음 (pykrx 미수집?)", ds)
                return data
            except Exception as e:
                logger.warning("investor_flow %s 로드 실패: %s", ds, e)
    return None


def load_investor_flow_history(days: int = 5, flow_dir: str | None = None) -> list[dict]:
    """최근 N일 investor_flow JSON 로드 (연속 매수/매도 판정용).

    Returns:
        날짜 오름차순 정렬된 dict 리스트.
    """
    base = _resolve_flow_dir(flow_dir)
    if not base.exists():
        return []

    files = sorted(base.glob("*_investor_flow.json"))
    if not files:
        return []

    # 최근 N개
    recent_files = files[-days:]
    result = []
    for path in recent_files:
        try:
            with open(path, encoding="utf-8") as f:
                result.append(json.load(f))
        except Exception:
            continue
    return result


def load_investor_flow_summary_from_intel() -> dict | None:
    """daily_intelligence.json → investor_flow_summary (향후 활성화).

    Returns:
        investor_flow_summary dict 또는 None (현재는 정보봇 미배포).
    """
    if not JGIS_JSON_PATH.exists():
        return None
    try:
        with open(JGIS_JSON_PATH, encoding="utf-8") as f:
            intel = json.load(f)
        return intel.get("investor_flow_summary")
    except Exception:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    adapter = JgisShortAdapter()
    print(f"Enabled: {adapter.enabled}")
    print(f"CSV dir: {adapter.csv_dir} (exists={adapter.csv_dir.exists()})")

    tickers = adapter.list_available_tickers()
    print(f"Available tickers: {len(tickers)}")

    # 삼성전자 테스트
    latest = adapter.get_latest("005930")
    if latest:
        print(f"\n005930 latest: {latest}")

    # TOP 30 테스트
    top30 = adapter.load_jgis_top30()
    print(f"\nJGIS TOP 30: {len(top30)} stocks")
    for t, d in list(top30.items())[:3]:
        print(f"  {t}: {d}")
