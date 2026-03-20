"""Execution Alpha — 매매 실행 품질 개선 (EX-1 ~ EX-5)

EX-1: VWAP 기반 동적 진입가 (전일 VWAP → 장중 VWAP)
EX-2: 호가창 스프레드 분석 (유동성 판단)
EX-3: 프로그램 매도 프록시 (3중 감지 + 소화 판정)
EX-5: 실행 품질 측정 (체결가 vs VWAP)

toggle: execution_alpha.enabled: false → 기존 동작 100% 동일
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_AUDIT_DB = _DATA_DIR / "order_audit.db"


class ExecutionAlpha:
    """EX-1~EX-3 진입 최적화 + EX-5 품질 측정"""

    def __init__(self, config: dict, intraday_adapter=None):
        ea_cfg = config.get("execution_alpha", {})
        self.enabled = ea_cfg.get("enabled", False)
        self.intraday = intraday_adapter
        self.dynamic_cfg = ea_cfg.get("dynamic_entry", {})
        self.spread_cfg = ea_cfg.get("spread_analysis", {})
        self.program_cfg = ea_cfg.get("program_proxy", {})
        self.quality_cfg = ea_cfg.get("quality_tracking", {})
        self._db_initialized = False

    # ── EX-1: VWAP 기반 동적 진입가 ──────────────────────

    def calc_dynamic_entry_price(
        self, prev_close: int, ticker: str, phase: str = "1"
    ) -> int:
        """VWAP 기반 동적 할인율로 진입가 계산.

        Phase 1: 전일 VWAP vs 전일종가 비교 → 할인율 조정
          VWAP > 종가 → 매수 우위 → 할인 축소 (-0.3%)
          VWAP < 종가 → 매도 압력 → 할인 확대 (-0.8%)

        Phase 3+: 장중 VWAP 대비 현재가 위치
          현재가 < VWAP → AGGRESSIVE (현재가 근처 지정가)
          현재가 ≈ VWAP → LIMIT_AT_VWAP
          현재가 > VWAP+0.5% → WAIT
          현재가 > VWAP+1% → SKIP
        """
        # STEP 3에서 구현
        base_discount = self.dynamic_cfg.get("base_discount_pct", 0.5)
        return int(prev_close * (1 - base_discount / 100))

    def calc_vwap_price_limit(self, vwap: float) -> int:
        """Phase 4: VWAP 기반 가격 상한 (이 이상이면 정정 보류)."""
        limit_pct = self.dynamic_cfg.get("vwap_premium_limit_pct", 0.5)
        return int(vwap * (1 + limit_pct / 100))

    # ── EX-2: 호가창 스프레드 분석 ──────────────────────

    def analyze_spread(self, orderbook: dict, reference_price: int) -> dict:
        """기존 fetch_orderbook() 결과를 재활용하여 스프레드 분석.

        Returns:
            {
                "spread_bps": float,        # 스프레드 (basis points)
                "depth_ratio": float,       # 매수/매도 잔량비
                "timing": str,              # "aggressive" / "normal" / "patient"
                "recommended_price": int,   # 추천 주문가
            }
        """
        # STEP 4에서 구현
        return {
            "spread_bps": 0.0,
            "depth_ratio": 1.0,
            "timing": "normal",
            "recommended_price": reference_price,
        }

    # ── EX-3: 프로그램 매도 프록시 ──────────────────────

    def detect_program_selling(
        self,
        candles_5m: list[dict],
        kospi_change_pct: float,
        investor_flow: dict,
    ) -> dict:
        """3중 프록시: 거래량폭발+음봉, KOSPI연동하락, 기관동시매도.

        Returns:
            {
                "selling_active": bool,     # 프로그램 매도 감지
                "volume_spike_sell": bool,   # 거래량폭발+음봉
                "kospi_correlated": bool,    # KOSPI 연동 하락
                "institutional_dump": bool,  # 기관 동시 매도
                "absorption_complete": bool, # 소화 완료 여부
            }
        """
        # STEP 5에서 구현
        return {
            "selling_active": False,
            "volume_spike_sell": False,
            "kospi_correlated": False,
            "institutional_dump": False,
            "absorption_complete": False,
        }

    # ── EX-5: 실행 품질 측정 ──────────────────────────

    def record_quality(
        self,
        side: str,
        ticker: str,
        filled_price: int,
        vwap: float,
        prev_close: int,
        order_price: int,
        spread_bps: float = 0.0,
        exit_rule: str = "",
        phase: str = "",
    ) -> dict | None:
        """체결 완료 후 품질 기록 → order_audit.db.

        Returns:
            품질 측정 dict 또는 None (disabled)
        """
        # STEP 2에서 구현
        return None

    def daily_quality_report(self, date_str: str) -> dict:
        """일간 실행 품질 리포트.

        Returns:
            {
                "date": str,
                "buy_count": int,
                "sell_count": int,
                "avg_buy_quality_bps": float,   # 양수=좋음
                "avg_sell_quality_bps": float,
                "total_saved_won": int,
                "best": dict | None,
                "worst": dict | None,
            }
        """
        # STEP 2에서 구현
        return {
            "date": date_str,
            "buy_count": 0,
            "sell_count": 0,
            "avg_buy_quality_bps": 0.0,
            "avg_sell_quality_bps": 0.0,
            "total_saved_won": 0,
            "best": None,
            "worst": None,
        }

    # ── 내부 유틸 ────────────────────────────────────

    def _ensure_db(self) -> None:
        """order_audit.db에 execution_quality 테이블 생성."""
        if self._db_initialized:
            return
        _AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(_AUDIT_DB)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                side TEXT NOT NULL,
                ticker TEXT NOT NULL,
                filled_price INTEGER,
                order_price INTEGER,
                vwap REAL,
                prev_close INTEGER,
                vs_vwap_bps REAL,
                spread_bps REAL,
                exit_rule TEXT,
                phase TEXT
            )
            """)
        self._db_initialized = True
