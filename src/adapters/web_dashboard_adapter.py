"""
자비스 컨트롤 타워 — FastAPI 웹 어댑터

localhost:8000에서 대시보드 UI + REST API 제공.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def _sanitize(obj):
    """NaN/Inf → None 재귀 변환 (JSON 직렬화 안전)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.use_cases.dashboard_data_provider import DashboardDataProvider

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jarvis Control Tower",
    description="Quantum Master 6대 시그널 통합 대시보드",
    version="1.0",
)

provider = DashboardDataProvider(cache_ttl=300)

# 정적 파일 디렉토리
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ──────────────────────────────────────────
# 페이지 라우트
# ──────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """메인 페이지 → jarvis.html"""
    html_path = STATIC_DIR / "jarvis.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"error": "jarvis.html not found"}, status_code=404)


# ──────────────────────────────────────────
# REST API
# ──────────────────────────────────────────

@app.get("/api/dashboard")
async def get_dashboard():
    """전체 대시보드 데이터 통합."""
    return _sanitize(provider.get_full_dashboard())


@app.get("/api/relay")
async def get_relay():
    """섹터 릴레이 시그널."""
    return provider.get_relay()


@app.get("/api/etf")
async def get_etf():
    """ETF 매매 시그널."""
    return provider.get_etf()


@app.get("/api/quantum")
async def get_quantum():
    """Quantum 스캔 후보."""
    return provider.get_quantum()


@app.get("/api/positions")
async def get_positions():
    """보유 포지션 (파일 기반)."""
    return provider.get_positions()


# ──────────────────────────────────────────
# KIS 실시간 잔고
# ──────────────────────────────────────────

_kis_cache: dict[str, tuple[float, dict]] = {}
_KIS_CACHE_TTL = 120  # 2분 캐시


@app.get("/api/kis-balance")
async def get_kis_balance():
    """한국투자증권 실시간 보유종목 + 잔고 조회."""
    cached = _kis_cache.get("balance")
    if cached and (time.time() - cached[0]) < _KIS_CACHE_TTL:
        return cached[1]

    if not os.getenv("KIS_APP_KEY"):
        return JSONResponse(
            {"error": "KIS API 키 미설정 (.env 확인)", "holdings": []},
            status_code=503,
        )

    try:
        from src.adapters.kis_order_adapter import KisOrderAdapter

        kis = KisOrderAdapter()
        balance = kis.fetch_balance()
        result = {
            "status": "ok",
            "holdings": balance.get("holdings", []),
            "total_eval": balance.get("total_eval", 0),
            "total_pnl": balance.get("total_pnl", 0),
            "available_cash": balance.get("available_cash", 0),
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _kis_cache["balance"] = (time.time(), result)
        logger.info(
            "[KIS] 잔고 조회 성공: %d종목, 총평가 %s원",
            len(result["holdings"]),
            f"{result['total_eval']:,}",
        )
        return result
    except Exception as e:
        logger.error("[KIS] 잔고 조회 실패: %s", e)
        return JSONResponse(
            {"error": f"KIS API 오류: {e}", "holdings": []},
            status_code=500,
        )


@app.get("/api/us-overnight")
async def get_us_overnight():
    """US 오버나이트 시그널."""
    return provider.get_us_overnight()


# ──────────────────────────────────────────
# 매매일지
# ──────────────────────────────────────────

JOURNAL_DIR = PROJECT_ROOT / "data" / "trade_journal"


@app.get("/api/trade-journal")
async def get_trade_journal():
    """매매일지 데이터 (전체 이력 + 월별 요약)."""
    trades_file = JOURNAL_DIR / "trades.json"
    trades = []
    if trades_file.exists():
        trades = json.loads(trades_file.read_text(encoding="utf-8"))

    # 최신 월별 요약
    monthly_dir = JOURNAL_DIR / "monthly"
    monthly = None
    if monthly_dir.exists():
        files = sorted(monthly_dir.glob("*.json"), reverse=True)
        if files:
            monthly = json.loads(files[0].read_text(encoding="utf-8"))

    # 최신 스냅샷
    snap_dir = JOURNAL_DIR / "snapshots"
    latest_snap = None
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("*.json"), reverse=True)
        if snaps:
            latest_snap = json.loads(snaps[0].read_text(encoding="utf-8"))

    return {
        "trades": trades,
        "monthly": monthly,
        "latest_snapshot": latest_snap,
        "trade_count": len(trades),
    }


@app.post("/api/refresh")
async def refresh_cache():
    """캐시 강제 갱신."""
    provider.clear_cache()
    return {"status": "ok", "message": "캐시 초기화 완료"}


@app.get("/api/health")
async def health():
    """서버 상태 확인."""
    return {"status": "ok", "version": "1.0"}
