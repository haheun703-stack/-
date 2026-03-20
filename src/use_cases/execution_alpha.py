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
          VWAP > 종가 → 매수 우위 (장중 매수세 강) → 할인 축소
          VWAP < 종가 → 매도 압력 (후반 올린 것) → 할인 확대
        """
        if not self.dynamic_cfg.get("enabled", True):
            base_discount = self.dynamic_cfg.get("base_discount_pct", 0.5)
            return int(prev_close * (1 - base_discount / 100))

        base_discount = self.dynamic_cfg.get("base_discount_pct", 0.5)
        adj_range = self.dynamic_cfg.get("vwap_adj_range_pct", 0.3)

        # 전일 VWAP 조회 (intraday_adapter 없으면 기본값)
        prev_vwap = self._fetch_prev_day_vwap(ticker)

        if prev_vwap <= 0:
            # 데이터 없으면 기본 할인율
            discount = base_discount
        elif prev_vwap > prev_close:
            # VWAP > 종가: 장중 매수 우위 → 할인 축소 (base - adj)
            discount = base_discount - adj_range
        elif prev_vwap < prev_close * 0.995:
            # VWAP < 종가-0.5%: 후반 끌어올림 → 할인 확대 (base + adj)
            discount = base_discount + adj_range
        else:
            # VWAP ≈ 종가: 기본
            discount = base_discount

        # 할인율 클램핑 (0.1% ~ 1.0%)
        discount = max(0.1, min(1.0, discount))

        entry_price = int(prev_close * (1 - discount / 100))
        logger.info(
            "EX-1 동적진입: %s — 전일VWAP=%.0f, 종가=%d, 할인=%.1f%% → %d원",
            ticker, prev_vwap, prev_close, discount, entry_price,
        )
        return entry_price

    def _fetch_prev_day_vwap(self, ticker: str) -> float:
        """전일 1분봉으로 VWAP 계산."""
        if not self.intraday:
            return 0.0
        try:
            candles = self.intraday.fetch_full_day_1m_candles(ticker)
            if not candles:
                return 0.0
            cum_tp_vol = 0.0
            cum_vol = 0
            for cd in candles:
                h = cd.get("high", 0)
                low = cd.get("low", 0)
                cl = cd.get("close", 0)
                v = cd.get("volume", 0)
                if h > 0 and low > 0 and cl > 0 and v > 0:
                    typical = (h + low + cl) / 3.0
                    cum_tp_vol += typical * v
                    cum_vol += v
            return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0
        except Exception as e:
            logger.warning("EX-1 전일VWAP 조회 실패: %s — %s", ticker, e)
            return 0.0

    def calc_vwap_price_limit(self, vwap: float) -> int:
        """Phase 4: VWAP 기반 가격 상한 (이 이상이면 정정 보류)."""
        limit_pct = self.dynamic_cfg.get("vwap_premium_limit_pct", 0.5)
        return int(vwap * (1 + limit_pct / 100))

    # ── EX-2: 호가창 스프레드 분석 ──────────────────────

    def analyze_spread(self, orderbook: dict, reference_price: int) -> dict:
        """기존 fetch_orderbook() 결과를 재활용하여 스프레드 분석.

        추가 API 호출 0 — _analyze_orderbook()에서 이미 가져온 데이터 활용.

        Returns:
            {
                "spread_bps": float,        # 스프레드 (basis points)
                "depth_ratio": float,       # 매수/매도 잔량비
                "timing": str,              # "aggressive" / "normal" / "patient"
                "recommended_price": int,   # 추천 주문가
                "score_adj": int,           # 점수 가감 (-2 ~ +2)
            }
        """
        if not self.spread_cfg.get("enabled", True):
            return {
                "spread_bps": 0.0,
                "depth_ratio": 1.0,
                "timing": "normal",
                "recommended_price": reference_price,
                "score_adj": 0,
            }

        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])
        depth_levels = self.spread_cfg.get("depth_levels", 3)

        # 스프레드 계산: 최우선 매도호가 - 최우선 매수호가
        if asks and bids:
            best_ask = asks[0].get("price", 0)
            best_bid = bids[0].get("price", 0)
            mid_price = (best_ask + best_bid) / 2 if (best_ask + best_bid) > 0 else reference_price
            spread_bps = (best_ask - best_bid) / mid_price * 10000 if mid_price > 0 else 0
        else:
            spread_bps = 0.0
            mid_price = reference_price

        # 매수/매도 잔량비 (depth_levels 호가)
        ask_vol = sum(a.get("volume", 0) for a in asks[:depth_levels]) if asks else 0
        bid_vol = sum(b.get("volume", 0) for b in bids[:depth_levels]) if bids else 0
        depth_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

        # 타이밍 판정
        tight_bps = self.spread_cfg.get("tight_bps", 10)
        wide_bps = self.spread_cfg.get("wide_bps", 30)

        if spread_bps < tight_bps:
            timing = "aggressive"  # 유동성 풍부 → 즉시 진입 유리
            score_adj = 2
            recommended = int(mid_price) if bids else reference_price
        elif spread_bps > wide_bps:
            timing = "patient"     # 유동성 부족 → 눌림 대기
            score_adj = -2
            recommended = bids[0].get("price", reference_price) if bids else reference_price
        else:
            timing = "normal"
            score_adj = 0
            recommended = bids[0].get("price", reference_price) if bids else reference_price

        logger.info(
            "EX-2 스프레드: 스프레드=%.1fbps, 잔량비=%.2f → %s (%+d점)",
            spread_bps, depth_ratio, timing, score_adj,
        )

        return {
            "spread_bps": round(spread_bps, 1),
            "depth_ratio": round(depth_ratio, 2),
            "timing": timing,
            "recommended_price": recommended,
            "score_adj": score_adj,
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

        vs_vwap_bps 부호 규칙:
          BUY:  양수 = 좋음 (VWAP보다 싸게 매수)
          SELL: 양수 = 좋음 (VWAP보다 비싸게 매도)

        Returns:
            품질 측정 dict 또는 None (disabled/vwap=0)
        """
        if not self.quality_cfg.get("enabled", True):
            return None
        if vwap <= 0 or filled_price <= 0:
            return None

        self._ensure_db()

        # VWAP 대비 체결 품질 (bps)
        raw_bps = (filled_price - vwap) / vwap * 10000
        # BUY: 싸게 살수록 좋음 → 부호 반전 (음수→양수)
        # SELL: 비싸게 팔수록 좋음 → 그대로
        vs_vwap_bps = -raw_bps if side.upper() == "BUY" else raw_bps

        ts = datetime.now().isoformat()

        with sqlite3.connect(str(_AUDIT_DB)) as conn:
            conn.execute(
                """INSERT INTO execution_quality
                   (ts, side, ticker, filled_price, order_price,
                    vwap, prev_close, vs_vwap_bps, spread_bps, exit_rule, phase)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, side.upper(), ticker, filled_price, order_price,
                 vwap, prev_close, vs_vwap_bps, spread_bps, exit_rule, phase),
            )

        result = {
            "side": side.upper(),
            "ticker": ticker,
            "filled_price": filled_price,
            "vwap": round(vwap, 1),
            "vs_vwap_bps": round(vs_vwap_bps, 1),
            "spread_bps": round(spread_bps, 1),
        }

        # 경고 임계값 초과 시 로그
        threshold = self.quality_cfg.get("alert_threshold_bps", 50)
        if vs_vwap_bps < -threshold:
            logger.warning(
                "EX-5 슬리피지 경고: %s %s %s — vs_vwap=%+.1fbps (임계: -%dbps)",
                side, ticker, f"{filled_price:,}원", vs_vwap_bps, threshold,
            )
        else:
            logger.info(
                "EX-5 품질: %s %s %s — vs_vwap=%+.1fbps",
                side, ticker, f"{filled_price:,}원", vs_vwap_bps,
            )

        return result

    def daily_quality_report(self, date_str: str) -> dict:
        """일간 실행 품질 리포트 (JOURNAL/텔레그램 연동).

        Returns:
            {
                "date": str,
                "buy_count": int,
                "sell_count": int,
                "avg_buy_quality_bps": float,
                "avg_sell_quality_bps": float,
                "total_saved_won": int,
                "best": dict | None,
                "worst": dict | None,
            }
        """
        self._ensure_db()

        empty = {
            "date": date_str,
            "buy_count": 0,
            "sell_count": 0,
            "avg_buy_quality_bps": 0.0,
            "avg_sell_quality_bps": 0.0,
            "total_saved_won": 0,
            "best": None,
            "worst": None,
        }

        try:
            with sqlite3.connect(str(_AUDIT_DB)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT side, ticker, filled_price, vwap, vs_vwap_bps,
                              order_price, spread_bps, exit_rule, phase
                       FROM execution_quality
                       WHERE ts LIKE ? || '%'
                       ORDER BY vs_vwap_bps DESC""",
                    (date_str,),
                ).fetchall()
        except sqlite3.OperationalError:
            return empty

        if not rows:
            return empty

        buys = [r for r in rows if r["side"] == "BUY"]
        sells = [r for r in rows if r["side"] == "SELL"]

        avg_buy = sum(r["vs_vwap_bps"] for r in buys) / len(buys) if buys else 0.0
        avg_sell = sum(r["vs_vwap_bps"] for r in sells) / len(sells) if sells else 0.0

        # 절약 금액 추정: BUY는 (VWAP - filled) × qty(1주 가정), SELL은 반대
        total_saved = 0
        for r in rows:
            if r["side"] == "BUY":
                total_saved += int(r["vwap"] - r["filled_price"])
            else:
                total_saved += int(r["filled_price"] - r["vwap"])

        def _row_to_dict(r):
            return {
                "ticker": r["ticker"],
                "side": r["side"],
                "filled_price": r["filled_price"],
                "vwap": r["vwap"],
                "vs_vwap_bps": round(r["vs_vwap_bps"], 1),
            }

        all_rows = list(rows)
        best = _row_to_dict(all_rows[0]) if all_rows else None
        worst = _row_to_dict(all_rows[-1]) if all_rows else None

        return {
            "date": date_str,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "avg_buy_quality_bps": round(avg_buy, 1),
            "avg_sell_quality_bps": round(avg_sell, 1),
            "total_saved_won": total_saved,
            "best": best,
            "worst": worst,
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
