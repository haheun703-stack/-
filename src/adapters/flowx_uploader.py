"""FLOWX Supabase 업로드 어댑터.

퀀트봇 데이터를 FLOWX PRO 대시보드용 Supabase DB에 업로드.
담당 테이블: etf_signals, foreign_flow, paper_trades

Usage:
    from src.adapters.flowx_uploader import FlowxUploader
    uploader = FlowxUploader()
    uploader.upload_etf_signals(rows)
    uploader.upload_foreign_flow(rows)
    uploader.upload_paper_trade(trade)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class FlowxUploader:
    """FLOWX Supabase 업로드 클라이언트."""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

        if not url or not key:
            logger.warning("[FLOWX] SUPABASE_URL/KEY 미설정 — 업로드 비활성")
            self.client = None
            return

        try:
            from supabase import create_client
            self.client = create_client(url, key)
            logger.info("[FLOWX] Supabase 연결 완료")
        except Exception as e:
            logger.error("[FLOWX] Supabase 연결 실패: %s", e)
            self.client = None

    @property
    def is_active(self) -> bool:
        return self.client is not None

    # ── ETF 시그널 ─────────────────────────────────

    def upload_etf_signals(self, rows: list[dict]) -> bool:
        """ETF 시그널 업로드 (UPSERT on date+code)."""
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("etf_signals").upsert(
                rows, on_conflict="date,code"
            ).execute()
            logger.info("[FLOWX] ETF 시그널 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] ETF 시그널 업로드 실패: %s", e)
            return False

    # ── 외국인 자금 흐름 ─────────────────────────────

    def upload_foreign_flow(self, rows: list[dict]) -> bool:
        """외국인 자금 흐름 업로드 (UPSERT on date+code).

        Supabase 테이블명: china_flow (기존 테이블 재활용, 스키마 동일).
        """
        if not self.is_active or not rows:
            return False
        try:
            result = self.client.table("china_flow").upsert(
                rows, on_conflict="date,code"
            ).execute()
            logger.info("[FLOWX] 외국인 자금 업로드: %d건", len(rows))
            return True
        except Exception as e:
            logger.error("[FLOWX] 외국인 자금 업로드 실패: %s", e)
            return False

    # ── 페이퍼 트레이딩 ──────────────────────────────

    def upload_paper_trade(self, trade: dict) -> bool:
        """페이퍼 매매 기록 업로드 (INSERT, UPSERT 아님)."""
        if not self.is_active or not trade:
            return False
        try:
            result = self.client.table("paper_trades").insert(trade).execute()
            side = trade.get("side", "?")
            name = trade.get("name", "?")
            price = trade.get("price", 0)
            logger.info("[FLOWX] 매매기록: %s %s @ %s", side, name, f"{price:,}")
            return True
        except Exception as e:
            logger.error("[FLOWX] 매매기록 업로드 실패: %s", e)
            return False


# ── 데이터 변환 함수 ─────────────────────────────

def build_etf_signal_rows(date_str: str = "") -> list[dict]:
    """기존 JSON → etf_signals 테이블 포맷 변환.

    소스:
      - data/sector_rotation/sector_momentum.json (섹터 순위 + RSI)
      - data/etf_rotation_result.json (order_queue → BUY/SELL)
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    rows = []

    # 1) 섹터 모멘텀 → 20개 섹터 ETF
    sm_path = DATA_DIR / "sector_rotation" / "sector_momentum.json"
    if sm_path.exists():
        with open(sm_path, encoding="utf-8") as f:
            sm = json.load(f)

        for sec in sm.get("sectors", []):
            rows.append({
                "date": date_str,
                "code": sec.get("etf_code", ""),
                "name": sec.get("sector", ""),
                "signal": "HOLD",  # 기본값, 아래에서 order_queue로 오버라이드
                "score": round(sec.get("momentum_score", 0), 1),
                "change_1d": round(sec.get("ret_5", 0) / 5, 2) if sec.get("ret_5") else 0,
                "change_5d": round(sec.get("ret_5", 0), 2),
                "rsi": round(sec.get("rsi_14", 0), 1),
                "sector_rotation_rank": sec.get("rank", 0),
                "group_tag": None,
            })

    # 2) order_queue에서 BUY/SELL 오버라이드
    etf_path = DATA_DIR / "etf_rotation_result.json"
    if etf_path.exists():
        with open(etf_path, encoding="utf-8") as f:
            etf = json.load(f)

        code_signal_map = {}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            action = order.get("action", "").upper()
            if code and action in ("BUY", "SELL"):
                code_signal_map[code] = action

        for row in rows:
            if row["code"] in code_signal_map:
                row["signal"] = code_signal_map[row["code"]]

        # order_queue에 있지만 sector_momentum에 없는 ETF (인버스, 레버리지 등)
        existing_codes = {r["code"] for r in rows}
        for order in etf.get("order_queue", []):
            code = order.get("code", "")
            if code and code not in existing_codes:
                action = order.get("action", "").upper()
                if action in ("BUY", "SELL"):
                    rows.append({
                        "date": date_str,
                        "code": code,
                        "name": order.get("name", ""),
                        "signal": action,
                        "score": 0,
                        "change_1d": 0,
                        "change_5d": 0,
                        "rsi": 0,
                        "sector_rotation_rank": 0,
                        "group_tag": None,
                    })

    return rows


def build_foreign_flow_rows(date_str: str = "") -> list[dict]:
    """국적별 외국인 수급 → foreign_flow 테이블 포맷 변환.

    소스: data/krx_nationality/nationality_signal.json
    시그널 매핑:
      - foreign_direction=="BUY" → INFLOW
      - signal=="SELL" or foreign_direction=="SELL" → OUTFLOW
      - 나머지 → NEUTRAL
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    sig_path = DATA_DIR / "krx_nationality" / "nationality_signal.json"
    if not sig_path.exists():
        return []

    with open(sig_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for sig in data.get("signals", []):
        # 시그널 매핑: 외국인 순매수/순매도 방향 기준
        foreign_dir = sig.get("foreign_direction", "NEUTRAL")
        raw_signal = sig.get("signal", "NEUTRAL")

        if foreign_dir == "BUY":
            mapped = "INFLOW"
        elif foreign_dir == "SELL" or raw_signal == "SELL":
            mapped = "OUTFLOW"
        else:
            mapped = "NEUTRAL"

        rows.append({
            "date": date_str,
            "code": sig.get("ticker", ""),
            "name": sig.get("name", ""),
            "signal": mapped,
            "score": sig.get("score", 0),
            "z_score": round(sig.get("inst_zscore", 0), 2),
            "change_5d_pct": round(sig.get("price_change_pct", 0), 2),
        })

    return rows


def build_paper_trade(
    code: str,
    name: str,
    side: str,
    price: float,
    quantity: int,
    strategy: str,
    pnl_pct: float | None = None,
    memo: str = "",
    stats: dict | None = None,
) -> dict:
    """페이퍼 매매 기록 dict 생성.

    Args:
        stats: paper_portfolio.json의 stats dict (PF/MDD/승률).
    """
    s = stats or {}
    total = s.get("total_trades", 0)
    wins = s.get("wins", 0)

    return {
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "code": code,
        "name": name,
        "side": side.upper(),
        "price": price,
        "quantity": quantity,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "strategy": strategy,
        "cumulative_pf": round(s.get("pf", 0), 2) if s.get("pf") else None,
        "cumulative_mdd": round(s.get("mdd", 0), 2) if s.get("mdd") else None,
        "win_rate": round(wins / total * 100, 1) if total > 0 else None,
        "memo": memo,
    }
