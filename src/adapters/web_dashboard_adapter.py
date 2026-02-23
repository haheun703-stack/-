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
from fastapi import FastAPI, Query, Request, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


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

# ──────────────────────────────────────────
# 인증 설정
# ──────────────────────────────────────────
JARVIS_USER = os.getenv("JARVIS_USER", "admin")
JARVIS_PASS = os.getenv("JARVIS_PASS", "quantum2026!")
JARVIS_SECRET = os.getenv("JARVIS_SECRET", "jarvis-default-secret")
_serializer = URLSafeTimedSerializer(JARVIS_SECRET)
_COOKIE_NAME = "jarvis_session"
_SESSION_MAX_AGE = 86400 * 7  # 7일


def _check_session(request: Request) -> bool:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return False
    try:
        _serializer.loads(token, max_age=_SESSION_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    OPEN_PATHS = {"/login", "/api/health", "/api/sync"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.OPEN_PATHS or path.startswith("/static/"):
            return await call_next(request)
        if not _check_session(request):
            if path.startswith("/api/"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)


app.add_middleware(AuthMiddleware)

# 정적 파일 디렉토리
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ──────────────────────────────────────────
# 로그인 / 로그아웃
# ──────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jarvis - Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{min-height:100vh;background:#0A0B0F;display:flex;justify-content:center;align-items:center;font-family:'Pretendard',-apple-system,sans-serif}
.login-box{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:40px;width:360px}
.login-box h1{color:#F9FAFB;font-size:20px;text-align:center;margin-bottom:6px}
.login-box .sub{color:#6B7280;font-size:12px;text-align:center;margin-bottom:24px}
.login-box input{width:100%;padding:12px 14px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);color:#E5E7EB;font-size:14px;margin-bottom:12px;outline:none}
.login-box input:focus{border-color:#6366F1}
.login-box button{width:100%;padding:12px;border-radius:8px;border:none;background:linear-gradient(135deg,#6366F1,#8B5CF6);color:#fff;font-size:14px;font-weight:700;cursor:pointer;transition:opacity 0.2s}
.login-box button:hover{opacity:0.9}
.error{color:#EF4444;font-size:12px;text-align:center;margin-bottom:12px}
</style></head><body>
<form class="login-box" method="POST" action="/login">
<h1>Jarvis Control Tower</h1>
<div class="sub">Quantum Master Dashboard</div>
{error}
<input type="text" name="username" placeholder="ID" autocomplete="username" required>
<input type="password" name="password" placeholder="Password" autocomplete="current-password" required>
<button type="submit">Login</button>
</form></body></html>"""


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    if _check_session(request):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(LOGIN_HTML.replace("{error}", ""))


@app.post("/login", include_in_schema=False)
async def login_submit(username: str = Form(...), password: str = Form(...)):
    if username == JARVIS_USER and password == JARVIS_PASS:
        token = _serializer.dumps({"user": username})
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(
            _COOKIE_NAME, token,
            max_age=_SESSION_MAX_AGE, httponly=True, samesite="lax",
        )
        return resp
    return HTMLResponse(
        LOGIN_HTML.replace("{error}", '<div class="error">ID 또는 비밀번호가 틀렸습니다</div>'),
        status_code=401,
    )


@app.get("/logout", include_in_schema=False)
async def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(_COOKIE_NAME)
    return resp


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
        # Railway용 fallback: 로컬에서 동기화된 캐시 파일 사용
        cache_path = PROJECT_ROOT / "data" / "kis_balance.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return {
                "status": "cached",
                "holdings": data.get("holdings", []),
                "total_eval": data.get("total_eval", 0),
                "total_pnl": data.get("total_pnl", 0),
                "available_cash": data.get("available_cash", 0),
                "fetched_at": data.get("fetched_at", ""),
            }
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


# ──────────────────────────────────────────
# 히스토리 API (SQLite 아카이브)
# ──────────────────────────────────────────

from src.daily_archive import (
    get_date_list, get_daily_summary, get_daily_picks, get_daily_etfs,
    get_weekly_reports, get_monthly_reports, get_performance_chart_data,
    get_stock_history, init_db,
)

# 서버 시작 시 DB 초기화
init_db()


@app.get("/api/history/dates")
async def history_dates(limit: int = Query(90, ge=1, le=365)):
    """아카이브된 날짜 목록."""
    return {"dates": get_date_list(limit)}


@app.get("/api/history/{date}")
async def history_detail(date: str):
    """특정 날짜 상세 (요약 + 추천종목 + ETF)."""
    summary = get_daily_summary(date)
    if not summary:
        return JSONResponse({"error": f"{date} 데이터 없음"}, status_code=404)
    return {
        "summary": summary,
        "picks": get_daily_picks(date),
        "etfs": get_daily_etfs(date),
    }


@app.get("/api/reports/weekly")
async def weekly_reports(limit: int = Query(12, ge=1, le=52)):
    """주간 보고서 목록."""
    return {"reports": get_weekly_reports(limit)}


@app.get("/api/reports/monthly")
async def monthly_reports(limit: int = Query(12, ge=1, le=24)):
    """월간 보고서 목록."""
    return {"reports": get_monthly_reports(limit)}


@app.get("/api/chart/performance")
async def chart_performance(days: int = Query(30, ge=7, le=365)):
    """Chart.js용 일별 성과 데이터."""
    return {"data": get_performance_chart_data(days)}


@app.get("/api/history/stock/{ticker}")
async def stock_history(ticker: str):
    """종목별 추천 이력."""
    return {"history": get_stock_history(ticker)}


@app.get("/api/health")
async def health():
    """서버 상태 확인."""
    return {"status": "ok", "version": "1.0"}


# ──────────────────────────────────────────
# 데이터 동기화 (로컬 → Railway 푸시)
# ──────────────────────────────────────────

SYNC_TOKEN = os.getenv("JARVIS_SECRET", "jarvis-default-secret")


@app.post("/api/sync")
async def sync_data(request: Request):
    """로컬 BAT-D 완료 후 JSON 데이터를 Railway로 푸시."""
    auth = request.headers.get("X-Sync-Token", "")
    if auth != SYNC_TOKEN:
        return JSONResponse({"error": "Invalid sync token"}, status_code=403)

    body = await request.json()
    target = body.get("file")  # 예: "tomorrow_picks.json"
    content = body.get("data")
    if not target or content is None:
        return JSONResponse({"error": "file and data required"}, status_code=400)

    # 안전한 파일명만 허용 (실제 스크립트 출력 파일명 기준)
    safe_names = {
        "tomorrow_picks.json", "etf_master.json", "picks_history.json",
        "us_market/overnight_signal.json", "sector_rotation/sector_momentum.json",
        "sector_rotation/etf_trading_signal.json",
        "sector_rotation/sector_zscore.json", "sector_rotation/investor_flow.json",
        "whale_detect.json", "dual_buying_watch.json", "pullback_scan.json",
        "group_relay/group_relay_today.json",
        "kis_balance.json", "kospi_regime.json",
    }
    if target not in safe_names:
        return JSONResponse({"error": f"File not allowed: {target}"}, status_code=400)

    dest = PROJECT_ROOT / "data" / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

    # 캐시 갱신
    provider.clear_cache()
    logger.info("[SYNC] %s 수신 완료", target)
    return {"status": "ok", "file": target}
