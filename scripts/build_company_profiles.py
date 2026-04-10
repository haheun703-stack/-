#!/usr/bin/env python
"""기업 프로파일 생성기 — Perplexity AI 배치 생성 + JSON 캐시

한 줄 소개 + 하락 원인을 Perplexity Sonar로 생성하여 캐시.
scan_nugget.py가 이 캐시를 읽어서 알파 스캐너 카드에 표시.

캐시 전략:
  - data/company_profiles.json에 {종목코드: {desc, drop_reason, updated}} 저장
  - 이미 캐시된 종목은 건너뜀 (--force로 전체 재생성)
  - 10종목씩 배치 호출 (API 비용 절감)

Usage:
    python -u -X utf8 scripts/build_company_profiles.py              # 노다지 후보만
    python -u -X utf8 scripts/build_company_profiles.py --all        # 유니버스 전체
    python -u -X utf8 scripts/build_company_profiles.py --force      # 캐시 무시 재생성
    python -u -X utf8 scripts/build_company_profiles.py --dry-run    # API 호출 없이 테스트
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "company_profiles.json"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

BATCH_SIZE = 10  # 1회 API 호출당 종목 수
RATE_LIMIT_SEC = 2.0  # API 호출 간 대기


# ═══════════════════════════════════════════════════
# 캐시 관리
# ═══════════════════════════════════════════════════

def load_cache() -> dict:
    """기존 캐시 로드."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache: dict):
    """캐시 저장."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════
# 후보 종목 로드
# ═══════════════════════════════════════════════════

def load_nugget_candidates() -> list[dict]:
    """노다지 리포트 결과에서 종목 목록 로드."""
    path = DATA_DIR / "nugget_report.json"
    if not path.exists():
        logger.warning("nugget_report.json 없음 — 유니버스 폴백")
        return load_universe_stocks()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stocks = []
    for item in data.get("nuggets", data if isinstance(data, list) else []):
        code = item.get("code", item.get("ticker", ""))
        name = item.get("name", "")
        sector = item.get("sector", "")
        if code and name:
            stocks.append({"code": code, "name": name, "sector": sector})
    return stocks


def load_universe_stocks() -> list[dict]:
    """유니버스 전체 로드."""
    import pandas as pd
    path = DATA_DIR / "universe.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"code": row.get("ticker", ""), "name": row.get("name", ""), "sector": row.get("sector", "")}
        for _, row in df.iterrows()
        if row.get("ticker")
    ]


# ═══════════════════════════════════════════════════
# Perplexity API 배치 호출
# ═══════════════════════════════════════════════════

def _call_perplexity(prompt: str, api_key: str) -> dict | None:
    """Perplexity Sonar API 호출."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 한국 주식시장 전문 애널리스트다. "
                    "기업의 핵심 사업을 간결하게 설명하고, "
                    "최근 주가 하락 원인을 분석한다. "
                    "반드시 요청된 JSON 형식으로만 응답한다. "
                    "다른 텍스트 없이 순수 JSON만 출력해."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    try:
        resp = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(content)
    except Exception as e:
        logger.error("Perplexity API 오류: %s", e)
        return None


def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 추출."""
    import re
    # ```json ... ``` 블록 추출
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    # { ... } 추출
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # [ ... ] 추출 (배열인 경우)
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            arr = json.loads(m.group(0))
            return {"stocks": arr}
        except json.JSONDecodeError:
            pass
    return None


def build_batch_prompt(stocks: list[dict]) -> str:
    """배치 프롬프트 생성 — 10종목씩."""
    stock_list = "\n".join(
        f"  {i+1}. {s['name']} ({s['code']}) — 섹터: {s.get('sector', '미분류')}"
        for i, s in enumerate(stocks)
    )
    return (
        f"다음 {len(stocks)}개 한국 상장 종목에 대해 각각:\n"
        "1) **한 줄 소개** (20자 이내): 핵심 사업/제품을 초보 투자자도 이해할 수 있게\n"
        "2) **하락 원인** (30자 이내): 최근 주가가 52주 고점 대비 왜 빠졌는지\n\n"
        f"종목 목록:\n{stock_list}\n\n"
        "반드시 아래 JSON 형식으로만 응답해:\n"
        '{"stocks": [\n'
        '  {"code": "종목코드", "desc": "한줄소개", "drop_reason": "하락원인"},\n'
        "  ...\n"
        "]}\n\n"
        "모든 종목을 빠짐없이 포함해. "
        "한줄소개는 '반도체 메모리 세계 1위 제조업체'처럼 사업 내용 중심으로. "
        "하락 원인은 '글로벌 반도체 업황 둔화 + 중국 경쟁 심화'처럼 구체적으로."
    )


# ═══════════════════════════════════════════════════
# 메인 로직
# ═══════════════════════════════════════════════════

def build_profiles(stocks: list[dict], cache: dict, api_key: str,
                   force: bool = False, dry_run: bool = False) -> dict:
    """프로파일 배치 생성."""
    # 캐시 미스 종목 필터
    if force:
        todo = stocks
    else:
        todo = [s for s in stocks if s["code"] not in cache]

    if not todo:
        print(f"  캐시 히트: {len(stocks)}종목 전부 이미 생성됨")
        return cache

    print(f"  생성 대상: {len(todo)}종목 ({len(stocks) - len(todo)}종목 캐시 히트)")

    if dry_run:
        print(f"  [dry-run] API 호출 건너뜀")
        return cache

    # 배치 처리
    total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    generated = 0

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"  배치 {batch_num}/{total_batches}: {', '.join(s['name'] for s in batch)}")

        prompt = build_batch_prompt(batch)
        result = _call_perplexity(prompt, api_key)

        if result and "stocks" in result:
            for item in result["stocks"]:
                code = item.get("code", "").strip()
                if not code:
                    continue
                cache[code] = {
                    "desc": item.get("desc", ""),
                    "drop_reason": item.get("drop_reason", ""),
                    "updated": datetime.now().strftime("%Y-%m-%d"),
                }
                generated += 1
        else:
            # 실패 시 개별 종목에 빈 값 저장 (재시도 방지)
            logger.warning("배치 %d 응답 파싱 실패", batch_num)
            for s in batch:
                if s["code"] not in cache:
                    cache[s["code"]] = {
                        "desc": "",
                        "drop_reason": "",
                        "updated": datetime.now().strftime("%Y-%m-%d"),
                    }

        # 진행 저장 (중간 실패 시 복구용)
        save_cache(cache)

        if i + BATCH_SIZE < len(todo):
            time.sleep(RATE_LIMIT_SEC)

    print(f"  생성 완료: {generated}종목 신규")
    return cache


def main():
    parser = argparse.ArgumentParser(description="기업 프로파일 생성기")
    parser.add_argument("--all", action="store_true", help="유니버스 전체 생성")
    parser.add_argument("--force", action="store_true", help="캐시 무시 재생성")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 테스트")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key and not args.dry_run:
        print("[오류] PERPLEXITY_API_KEY 미설정")
        sys.exit(1)

    print(f"\n[프로파일] 생성 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 종목 로드
    if args.all:
        stocks = load_universe_stocks()
        print(f"  모드: 유니버스 전체 ({len(stocks)}종목)")
    else:
        stocks = load_nugget_candidates()
        print(f"  모드: 노다지 후보 ({len(stocks)}종목)")

    if not stocks:
        print("  종목 없음")
        return

    # 캐시 로드
    cache = load_cache()
    print(f"  기존 캐시: {len(cache)}종목")

    # 프로파일 생성
    cache = build_profiles(stocks, cache, api_key, args.force, args.dry_run)

    # 최종 저장
    save_cache(cache)

    # 통계
    filled = sum(1 for v in cache.values() if v.get("desc"))
    empty = sum(1 for v in cache.values() if not v.get("desc"))
    print(f"\n[프로파일] 완료 — 총 {len(cache)}종목 (설명있음 {filled}, 빈값 {empty})")


if __name__ == "__main__":
    main()
