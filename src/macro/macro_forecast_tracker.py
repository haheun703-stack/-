"""매크로 예실(예상 vs 실제) 트래커 — 미국 지표 적재 어댑터.

웹봇 의뢰(`macro_forecast_actual` Supabase 테이블)에 미국 매크로 지표의
예상(consensus=나우캐스트/선물내재)과 실제(actual=발표) 서프라이즈를 적재한다.

철학(웹 면책과 동일): 방향 예측 아님. 이벤트 전 노출 결정(리스크관리)용.
매매신호·투자자문 금지. 퀀트봇 매매로직/실주문/scheduler/SAJANG/C60과 무관.

담당 분담: 미국 지표 = 퀀트봇(본 모듈) / 한국 지표(BOK 등) = 정보봇.

★score 공식 주의: 서규하 스펙 원본이 코드/문서에 없어, 아래 공식은 **투명한 임시
정의**다(의뢰서 요약 "코어차이 가중 등" 기반). 스펙 확정 시 compute_surprise_score /
META의 typical_sigma·hawkish_sign만 교체하면 된다. 웹은 surprise_score를 읽기만 한다.

쓰기 경로: psycopg2 + DATABASE_URL(Supabase). RLS 활성 테이블이라 service_role/owner
연결이 필요 → anon key(SUPABASE_KEY)가 아닌 DATABASE_URL 직결을 쓴다.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 지표 메타 (미국) ──────────────────────────────────────────────
# hawkish_sign: actual>consensus(서프라이즈 +)일 때 통화정책 함의.
#   +1 = 긴축(매파) 신호  /  -1 = 완화(비둘기) 신호  /  0 = 통화함의 맥락의존
# typical_sigma: 서프라이즈 정규화용 통상 1회 변동폭(임시값, 과거 변동성으로 보정 예정).
# impact_clear: market_impact(증시 악재/호재)를 단정해도 되는 지표인가.
#   인플레/금리는 명확(상회=긴축=악재), 경기/고용은 국면의존("good news bad news") → False.
US_INDICATOR_META: dict[str, dict[str, Any]] = {
    "FOMC":     {"name_ko": "미국 기준금리(FOMC)", "unit": "%",    "frequency": "per-meeting", "impact": 5, "hawkish_sign": 1,  "typical_sigma": 0.25, "impact_clear": True},
    "CPI_HEAD": {"name_ko": "미국 소비자물가(헤드라인)", "unit": "%", "frequency": "monthly",   "impact": 5, "hawkish_sign": 1,  "typical_sigma": 0.1,  "impact_clear": True},
    "CPI_CORE": {"name_ko": "미국 근원 소비자물가", "unit": "%",   "frequency": "monthly",     "impact": 5, "hawkish_sign": 1,  "typical_sigma": 0.1,  "impact_clear": True},
    "PCE":      {"name_ko": "미국 PCE 물가",       "unit": "%",   "frequency": "monthly",     "impact": 4, "hawkish_sign": 1,  "typical_sigma": 0.1,  "impact_clear": True},
    "PPI":      {"name_ko": "미국 생산자물가",     "unit": "%",   "frequency": "monthly",     "impact": 4, "hawkish_sign": 1,  "typical_sigma": 0.2,  "impact_clear": True},
    "NFP":      {"name_ko": "미국 비농업고용",     "unit": "천명", "frequency": "monthly",     "impact": 5, "hawkish_sign": 0,  "typical_sigma": 50.0, "impact_clear": False},
    "UNRATE":   {"name_ko": "미국 실업률",         "unit": "%",   "frequency": "monthly",     "impact": 4, "hawkish_sign": -1, "typical_sigma": 0.1,  "impact_clear": False},
    "ISM_MFG":  {"name_ko": "미국 ISM 제조업",     "unit": "index","frequency": "monthly",     "impact": 3, "hawkish_sign": 0,  "typical_sigma": 1.0,  "impact_clear": False},
    "RETAIL":   {"name_ko": "미국 소매판매",       "unit": "%",   "frequency": "monthly",     "impact": 3, "hawkish_sign": 0,  "typical_sigma": 0.3,  "impact_clear": False},
    "GDP":      {"name_ko": "미국 GDP 성장률",     "unit": "%",   "frequency": "quarterly",   "impact": 4, "hawkish_sign": 0,  "typical_sigma": 0.3,  "impact_clear": False},
    "JOLTS":    {"name_ko": "미국 구인건수(JOLTS)", "unit": "천명", "frequency": "monthly",     "impact": 3, "hawkish_sign": 0,  "typical_sigma": 200.0,"impact_clear": False},
}

STANCE_TIGHT = "긴축"
STANCE_EASE = "완화"
STANCE_NEUTRAL = "중립"
IMPACT_BAD = "악재"
IMPACT_GOOD = "호재"
IMPACT_NEUTRAL = "중립"

_STANCE_THRESHOLD = 0.5  # |hawkish 정규화 점수| 이 값 넘으면 긴축/완화 단정


def compute_surprise(actual: float | None, consensus: float | None) -> float | None:
    """서프라이즈 = 실제 − 예상. 둘 중 하나라도 없으면 None(발표 전)."""
    if actual is None or consensus is None:
        return None
    return round(float(actual) - float(consensus), 4)


def compute_surprise_score(code: str, surprise: float | None) -> float | None:
    """정규화 서프라이즈 점수 = surprise / typical_sigma. ★임시 공식(스펙 확정시 교체).

    +면 실제가 예상 상회(절대 부호, 통화함의 아님). 통화함의는 classify_stance가 본다.
    """
    if surprise is None:
        return None
    meta = US_INDICATOR_META.get(code)
    if not meta:
        return None
    sigma = meta.get("typical_sigma") or 1.0
    return round(surprise / sigma, 2)


def _hawkish_value(code: str, surprise_score: float | None) -> float | None:
    """통화정책 함의 점수 = surprise_score × hawkish_sign. +면 매파(긴축), −면 비둘기."""
    if surprise_score is None:
        return None
    meta = US_INDICATOR_META.get(code)
    if not meta:
        return None
    return surprise_score * meta.get("hawkish_sign", 0)


def classify_stance(code: str, surprise_score: float | None) -> str | None:
    """통화정책 스탠스. hawkish_sign=0(맥락의존 지표)은 중립으로 둔다."""
    hawk = _hawkish_value(code, surprise_score)
    if hawk is None:
        return None
    if hawk >= _STANCE_THRESHOLD:
        return STANCE_TIGHT
    if hawk <= -_STANCE_THRESHOLD:
        return STANCE_EASE
    return STANCE_NEUTRAL


def classify_market_impact(code: str, surprise_score: float | None) -> str | None:
    """증시 영향(악재/호재). ★긴축=악재 / 완화=호재.

    인플레·금리 지표(impact_clear=True)만 단정. 경기·고용 지표는 국면의존
    ("good news bad news")이라 중립으로 두고 note로 맥락 안내(임의 단정 금지).
    """
    meta = US_INDICATOR_META.get(code)
    if not meta:
        return None
    if surprise_score is None:
        return None
    if not meta.get("impact_clear"):
        return IMPACT_NEUTRAL  # 경기/고용 → 국면의존, 단정 안 함
    hawk = _hawkish_value(code, surprise_score)
    if hawk is None:
        return IMPACT_NEUTRAL
    if hawk >= _STANCE_THRESHOLD:
        return IMPACT_BAD   # 긴축 서프라이즈 = 증시 악재
    if hawk <= -_STANCE_THRESHOLD:
        return IMPACT_GOOD  # 완화 서프라이즈 = 증시 호재
    return IMPACT_NEUTRAL


def build_forecast_row(
    code: str,
    event_date: str,
    *,
    consensus: float | None = None,
    consensus_source: str | None = None,
    actual: float | None = None,
    prior: float | None = None,
    event_datetime_kst: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """단일 지표 → macro_forecast_actual 행(18컬럼). 순수 함수(외부호출 없음).

    발표 전: actual=None → surprise/score/stance/market_impact=None.
    발표 후: actual 채우면 서프라이즈·점수·스탠스·증시영향 자동 계산.
    """
    meta = US_INDICATOR_META.get(code, {})
    surprise = compute_surprise(actual, consensus)
    score = compute_surprise_score(code, surprise)
    return {
        "indicator_code": code,
        "event_date": event_date,
        "region": "US",
        "indicator_name_ko": meta.get("name_ko", code),
        "impact": meta.get("impact", 3),
        "frequency": meta.get("frequency"),
        "event_datetime_kst": event_datetime_kst,
        "consensus": consensus,
        "consensus_source": consensus_source,
        "actual": actual,
        "prior": prior,
        "surprise": surprise,
        "surprise_score": score,
        "stance": classify_stance(code, score),
        "market_impact": classify_market_impact(code, score),
        "unit": meta.get("unit"),
        "note": note,
    }


# ── Supabase 적재 (psycopg2 + DATABASE_URL, RLS 우회) ──────────────
_COLUMNS = [
    "indicator_code", "event_date", "region", "indicator_name_ko", "impact",
    "frequency", "event_datetime_kst", "consensus", "consensus_source", "actual",
    "prior", "surprise", "surprise_score", "stance", "market_impact", "unit", "note",
]


def upsert_forecast_actual(rows: list[dict[str, Any]], *, dry_run: bool = True) -> dict[str, Any]:
    """macro_forecast_actual upsert(PK indicator_code+event_date). 매매 무관·주문 0.

    ★dry_run=True(기본)면 실제 write 0 — 행 검증만 하고 SQL/건수만 돌려준다.
    실제 적재는 dry_run=False 명시 필요(외부 DB write 안전장치).
    """
    if not rows:
        return {"dry_run": dry_run, "rows": 0, "written": 0, "note": "no rows"}

    # 컬럼 누락 방어 + 안전 정렬
    norm = [{c: r.get(c) for c in _COLUMNS} for r in rows]

    if dry_run:
        return {
            "dry_run": True,
            "rows": len(norm),
            "written": 0,
            "sample": norm[0],
            "note": "dry_run — 실제 write 0. dry_run=False로 적재.",
        }

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        return {"dry_run": False, "rows": len(norm), "written": 0, "note": "DATABASE_URL 없음"}

    import psycopg2
    from psycopg2.extras import execute_values

    cols_sql = ", ".join(_COLUMNS)
    update_sql = ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c not in ("indicator_code", "event_date"))
    sql = (
        f"insert into macro_forecast_actual ({cols_sql}, updated_at) "
        f"values %s "
        f"on conflict (indicator_code, event_date) do update set {update_sql}, updated_at=now()"
    )
    template = "(" + ", ".join(["%s"] * len(_COLUMNS)) + ", now())"
    values = [[r[c] for c in _COLUMNS] for r in norm]

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template=template)
        written = len(values)
    finally:
        conn.close()
    return {"dry_run": False, "rows": len(norm), "written": written, "note": "upsert 완료"}


# ── 외부 소스 수집 ────────────────────────────────────────────────
_CLEVELAND_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_month.json"
)
# 클리블랜드 시리즈명 → 우리 indicator_code (MoM % 나우캐스트)
_CLEVELAND_SERIES = {
    "CPI Inflation": "CPI_HEAD",
    "Core CPI Inflation": "CPI_CORE",
    "PCE Inflation": "PCE",
}


def _cleveland_month_to_date(target: str | None) -> str | None:
    """'2026-6' → '2026-06-01'(대상월 귀속일). 파싱 실패 시 None."""
    if not target or "-" not in target:
        return None
    try:
        y, m = target.split("-")[:2]
        return f"{int(y):04d}-{int(m):02d}-01"
    except (ValueError, TypeError):
        return None


def fetch_cleveland_nowcast(timeout: int = 25) -> dict[str, Any]:
    """클리블랜드 연준 인플레이션 나우캐스트 → 최신 MoM consensus(읽기 전용).

    반환: {target_month, target_date, asof, source, values:{CPI_HEAD, CPI_CORE, PCE}}
    consensus_source='cleveland_nowcast', 단위=MoM %. 매매 무관·write 0.
    """
    import json as _json

    import requests

    r = requests.get(_CLEVELAND_URL, timeout=timeout)
    r.raise_for_status()
    data = _json.loads(r.content)
    if not isinstance(data, list) or not data:
        return {"target_month": None, "target_date": None, "asof": None, "source": "cleveland_nowcast", "values": {}}
    latest = data[-1]
    chart = latest.get("chart", {})
    target = chart.get("subcaption")
    values: dict[str, float] = {}
    for series in latest.get("dataset", []):
        code = _CLEVELAND_SERIES.get(series.get("seriesname"))
        if not code:
            continue
        nums = [x.get("value") for x in series.get("data", []) if x.get("value") not in (None, "")]
        if nums:
            try:
                values[code] = round(float(nums[-1]), 3)
            except (TypeError, ValueError):
                continue
    return {
        "target_month": target,
        "target_date": _cleveland_month_to_date(target),
        "asof": chart.get("_comment"),
        "source": "cleveland_nowcast",
        "values": values,
    }
