"""Quant Supabase Reader — 차트영웅 매매 5-Gate 데이터 조회.

조회 대상 테이블:
  - quant_surge_pullback   Gate 2 (어제 상한가/급등 종목 풀)
  - quant_sector_fire      Gate 3 보조 (섹터 fire_score)
  - quant_sector_picks     Gate 4 보조 (섹터별 종목 매수 후보)
  - quant_company_card     Gate 4-A (정보봇 펀더멘털, 정보봇 응답 후)
  - quant_surge_catalyst   Gate 4-A (정보봇 catalyst, 정보봇 응답 후)

flowx_uploader.py가 쓰기 전담 → 본 모듈은 읽기 전담 (격리).

5/22 paper mirror 진입 후보 산출 핵심.
"""

import os
from pathlib import Path
from typing import Optional


def _load_env():
    if os.getenv("SUPABASE_URL"):
        return
    p = Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ[k.strip()] = v.strip()
    except Exception:
        pass

_load_env()


_client = None


def _get_client():
    """Supabase 클라이언트 (lazy init)."""
    global _client
    if _client is not None:
        return _client
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return None
        _client = create_client(url, key)
        return _client
    except ImportError:
        return None
    except Exception:
        return None


def get_yesterday_surge_pool(date: str, min_surge_pct: float = 25.0) -> list[dict]:
    """Gate 2: 어제(date) 상한가/급등 종목 풀.

    Args:
        date: 'YYYY-MM-DD' (어제 = D0 = 상한가일)
        min_surge_pct: 최소 급등률 (%)

    Returns:
        [{ticker, name, sector, surge_pct, surge_close, fire_score, ...}]
    """
    client = _get_client()
    if not client:
        return []
    try:
        res = (client.table("quant_surge_pullback")
               .select("*")
               .eq("date", date)
               .gte("surge_pct", min_surge_pct)
               .order("surge_pct", desc=True)
               .execute())
        return res.data or []
    except Exception as e:
        print(f"[get_yesterday_surge_pool] error: {e}")
        return []


def get_sector_fire_today(date: str, min_score: float = 60.0) -> list[dict]:
    """Gate 4 보조: 오늘 섹터 fire_score ≥ min_score 섹터 목록."""
    client = _get_client()
    if not client:
        return []
    try:
        res = (client.table("quant_sector_fire")
               .select("sector, fire_score, composite_grade, "
                       "fire_grade, leverage_etf_code, leverage_etf_name")
               .eq("date", date)
               .gte("fire_score", min_score)
               .order("fire_score", desc=True)
               .execute())
        return res.data or []
    except Exception as e:
        print(f"[get_sector_fire_today] error: {e}")
        return []


def get_sector_picks_today(date: str) -> list[dict]:
    """Gate 2 대체/보조: quant_sector_picks (오늘 섹터 종목 매수 후보).

    BAT-D 결과로 매일 16:30 업데이트되는 종목 풀.
    """
    client = _get_client()
    if not client:
        return []
    try:
        res = (client.table("quant_sector_picks")
               .select("*")
               .eq("date", date)
               .order("buy_score", desc=True)
               .execute())
        return res.data or []
    except Exception as e:
        print(f"[get_sector_picks_today] error: {e}")
        return []


def get_company_card(ticker: str) -> Optional[dict]:
    """Gate 4-A: 정보봇 펀더멘털 카드 (정보봇 응답 후 가능).

    quant_company_card 테이블은 정보봇이 5/21~ 생성.
    """
    client = _get_client()
    if not client:
        return None
    try:
        res = (client.table("quant_company_card")
               .select("*")
               .eq("ticker", ticker)
               .single()
               .execute())
        return res.data
    except Exception:
        return None


def get_catalyst(date: str, ticker: str) -> Optional[dict]:
    """Gate 4-A: 정보봇 catalyst (정보봇 응답 후 가능)."""
    client = _get_client()
    if not client:
        return None
    try:
        res = (client.table("quant_surge_catalyst")
               .select("*")
               .eq("date", date)
               .eq("ticker", ticker)
               .single()
               .execute())
        return res.data
    except Exception:
        return None


def get_market_brain_today(date: str) -> Optional[dict]:
    """단타봇용 advisory가 아닌, 우리 자체 brain 데이터 확인용."""
    client = _get_client()
    if not client:
        return None
    try:
        res = (client.table("quant_market_brain")
               .select("*")
               .eq("date", date)
               .single()
               .execute())
        return res.data
    except Exception:
        return None


if __name__ == "__main__":
    import datetime as dt
    today = dt.date.today().isoformat()
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    print(f"=== Supabase Reader 자가 검증 ({today}) ===\n")

    # 1) 어제 상한가 풀
    print(f"[1] Gate 2: 어제({yesterday}) 상한가 풀 (surge_pct ≥ 25)")
    surge_pool = get_yesterday_surge_pool(yesterday, 25.0)
    print(f"   결과: {len(surge_pool)}건")
    for p in surge_pool[:5]:
        print(f"   - {p.get('ticker')} {p.get('name', '?'):14} "
              f"surge={p.get('surge_pct')}% close={p.get('surge_close')}")

    # 2) 오늘 섹터 fire ≥ 60
    print(f"\n[2] Gate 4 보조: 오늘 섹터 fire_score ≥ 60")
    sectors = get_sector_fire_today(today, 60.0)
    print(f"   결과: {len(sectors)}개 섹터")
    for s in sectors[:5]:
        print(f"   - {s.get('sector'):14} fire={s.get('fire_score'):>5.1f} "
              f"등급={s.get('composite_grade')}")

    # 3) 오늘 섹터 매수 후보
    print(f"\n[3] Gate 2 대체: 오늘 sector_picks")
    picks = get_sector_picks_today(today)
    print(f"   결과: {len(picks)}건")
    for p in picks[:5]:
        print(f"   - {p.get('ticker')} {p.get('name'):14} "
              f"buy_score={p.get('buy_score')} sector={p.get('sector')}")
