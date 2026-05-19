"""증권사 목표가 컨센서스 수집기 — 우리 자체 모듈 (정보봇 부담 X).

5/19 사장님 결단: 정보봇은 catalyst/펀더멘털 작업 중 → 우리는 목표가만 자체.
.env에 PERPLEXITY_API_KEY 보유 → Perplexity로 일별 컨센서스 수집 (~$0.002/일).

차트영웅 매매법 Gate 4-B (펀더멘털 강화):
  - 목표가 상승률 ≥ 30%   = 매수 가산점
  - 신규 커버리지          = 가산점
  - 목표가 하향            = 감점

데이터 흐름:
  Perplexity 일별 호출 → JSON 정제 → CSV/Supabase 저장 → 5-Gate 입력

장점:
  - $0.002/일 (월 약 $0.06) → 정보봇 부담 X
  - SPA 사이트 문제 우회 (Perplexity가 실시간 웹 검색)
  - 블로그봇이 Supabase에서 받아 자동 포스팅 가능
"""

import csv
import json
import os
import re
import requests
import datetime as dt
from pathlib import Path

# C4: 표준 python-dotenv 사용 (3중 복제 _load_env 제거)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


_PPLX_URL = "https://api.perplexity.ai/chat/completions"

# C5: 한국 주식 ticker 검증 (6자리 숫자, 외국주/오타 차단 — 실주문 안전망)
_KR_TICKER_RE = re.compile(r"^\d{6}$")

def _is_valid_kr_ticker(t) -> bool:
    """KOSPI/KOSDAQ ticker 형식 검증 (6자리 숫자만 허용)."""
    return isinstance(t, str) and bool(_KR_TICKER_RE.match(t))


def collect_target_consensus(target_date: str | None = None,
                              direction: str = "상향") -> list[dict]:
    """일별 증권사 목표가 컨센서스 수집.

    Args:
        target_date: 'YYYY-MM-DD' (None = 오늘)
        direction: '상향' | '하향' | '신규' | '전체'

    Returns:
        [{ticker, name, current_price, target_price, upside_pct, broker, report_type, date, source}]
    """
    target_date = target_date or dt.date.today().isoformat()
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return []

    sys_prompt = "JSON 배열만 출력. 마크다운/설명 없이."
    user_prompt = (
        f"{target_date} 한국 증권사가 목표가 {direction}한 종목 20개 알려줘. "
        f'JSON: [{{"name":"종목명","ticker":"6자리","current_price":정수,'
        f'"target_price":정수,"broker":"증권사"}}]'
    )

    try:
        r = requests.post(
            _PPLX_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 3000,
                "temperature": 0.0,
            },
            timeout=60,
        )
        if r.status_code != 200:
            return [{"error": f"HTTP {r.status_code}: {r.text[:200]}"}]
        content = r.json()["choices"][0]["message"]["content"]
        # JSON 추출 (Perplexity가 가끔 markdown 코드블럭 감쌈)
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if not m:
            return [{"error": "JSON 배열 추출 실패", "raw": content[:300]}]
        items = json.loads(m.group(0))

        # 메타데이터 + upside_pct 자동 계산 (Perplexity 미제공 시) + 중복 제거
        seen = set()
        out = []
        skipped_invalid = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tk = it.get("ticker")
            # C5: LLM 응답 ticker 검증 — 외국주(AAPL)/5자리/None 차단
            if not _is_valid_kr_ticker(tk):
                skipped_invalid.append(tk)
                continue
            if tk in seen:  # 같은 ticker 중복 제거 (첫 번째만)
                continue
            seen.add(tk)

            # upside_pct 자동 계산
            cp = it.get("current_price") or 0
            tp = it.get("target_price") or 0
            if not it.get("upside_pct") and cp > 0 and tp > 0:
                it["upside_pct"] = round((tp - cp) / cp * 100, 1)

            it["date"] = target_date
            it["source"] = "perplexity_sonar_pro"
            it["report_type"] = it.get("report_type") or direction
            out.append(it)
        # C5: 차단된 invalid ticker 가시성 (silent 차단 방지)
        if skipped_invalid:
            import sys
            print(f"[C5] LLM invalid ticker 차단: {skipped_invalid[:5]} "
                  f"(총 {len(skipped_invalid)}건)", file=sys.stderr)
        return out
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def load_seed_csv(seed_path: str = "docs/seed/analyst_target_20260519.csv") -> list[dict]:
    """시드 CSV 로드 (Perplexity 결과 0건 시 fallback).

    사장님이 5/19 캡처한 32종목 데이터로 5/22 가동 시점 baseline 확보.
    """
    p = Path(seed_path)
    if not p.exists():
        return []
    items = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["current_price"] = int(row["current_price"])
                row["target_price"] = int(row["target_price"])
                row["upside_pct"] = float(row["upside_pct"])
            except Exception:
                pass
            items.append(dict(row))
    return items


def collect_with_fallback(target_date: str | None = None,
                          direction: str = "상향",
                          min_results: int = 5) -> tuple[list[dict], str]:
    """Perplexity 1차 → 결과 부족 시 시드 CSV fallback.

    Returns:
        (items, source) — source: 'perplexity' | 'seed' | 'empty'
    """
    items = collect_target_consensus(target_date, direction)
    items_clean = [i for i in items if "error" not in i]
    if len(items_clean) >= min_results:
        return items_clean, "perplexity"
    # fallback: 시드 CSV (5/19 사장님 캡처)
    seed = load_seed_csv()
    seed_filtered = [s for s in seed if s.get("report_type") == direction]
    if seed_filtered:
        return seed_filtered, "seed_20260519"
    return [], "empty"


def save_consensus_csv(items: list[dict], csv_path: str | None = None) -> str:
    """수집 결과 CSV 저장 (append 모드, 일별 누적)."""
    if not items or "error" in (items[0] if items else {}):
        return ""
    csv_path = csv_path or "data/analyst_target_consensus.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    cols = ["date", "ticker", "name", "current_price", "target_price",
            "upside_pct", "broker", "report_type", "source"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for it in items:
            if "error" in it:
                continue
            w.writerow({k: it.get(k, "") for k in cols})
    return csv_path


def find_high_upside_picks(items: list[dict], min_upside: float = 30.0) -> list[dict]:
    """상승여력 ≥ 임계값 종목만 필터 (차트영웅 Gate 4-B 입력)."""
    return [it for it in items
            if "error" not in it and (it.get("upside_pct") or 0) >= min_upside]


if __name__ == "__main__":
    today = "2026-05-19"
    print(f"=== 증권사 목표가 컨센서스 수집 ({today}) ===\n")

    # Perplexity + 시드 fallback
    items, source = collect_with_fallback(today, "상향", min_results=5)
    print(f"📡 source: {source}")
    if not items:
        print(f"❌ 데이터 없음")
    else:
        print(f"✅ 총 {len(items)}건 수집\n")
        print(f"{'종목코드':>8} {'종목명':16} {'현재가':>10} {'목표가':>10} {'상승':>8} {'증권사':12}")
        print("-" * 80)
        for it in items[:15]:
            tk = (it.get("ticker") or "")[:6]
            nm = (it.get("name") or "")[:14]
            cp = it.get("current_price") or 0
            tp = it.get("target_price") or 0
            up = it.get("upside_pct") or 0
            br = (it.get("broker") or "")[:10]
            print(f"{tk:>8} {nm:16} {cp:>10,} {tp:>10,} {up:>7.1f}% {br:12}")

        # 30%+ 필터
        picks = find_high_upside_picks(items, 30.0)
        print(f"\n🎯 상승여력 30%+ : {len(picks)}건 (차트영웅 Gate 4-B 입력 후보)")
        for it in picks[:10]:
            print(f"   {it.get('name'):12} +{it.get('upside_pct'):.1f}% ({it.get('broker')})")

        # CSV 저장
        path = save_consensus_csv(items)
        if path:
            print(f"\n💾 저장: {path}")
