"""91건 시작점 뉴스/공시 크롤링 — 촉매 vs 소음 분류

signal_points.csv의 각 시작점 전후 5영업일 뉴스를 수집하고,
촉매(선행)/소음(후행)/무관으로 분류한 후 패턴을 분석한다.

사용법:
    python scripts/crawl_signal_news.py
    python scripts/crawl_signal_news.py --classify   # GPT 분류 포함
    python scripts/crawl_signal_news.py --analyze     # 분석만 (크롤링 스킵)
"""

import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ─── 네이버 금융 뉴스 크롤링 ───

NAVER_NEWS_URL = "https://finance.naver.com/item/news_news.naver"
NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}


def fetch_naver_news(stock_code: str, page: int = 1) -> list[dict]:
    """네이버 금융 종목 뉴스 1페이지 크롤링.

    Returns:
        [{"title": str, "date": "YYYY.MM.DD", "source": str, "url": str}, ...]
    """
    params = {
        "code": stock_code,
        "page": page,
        "sm": "title_entity_id.basic",
        "clusterId": "",
    }

    try:
        resp = requests.get(
            NAVER_NEWS_URL,
            params=params,
            headers=NAVER_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  요청 실패: {stock_code} page={page}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # 뉴스 테이블 파싱 — 첫 번째 table.type5만 (관련 뉴스 제외)
    news_list = []
    tables = soup.select("table.type5")
    if not tables:
        return news_list

    rows = tables[0].select("tr")
    for row in rows:
        tds = row.select("td")
        if len(tds) < 3:
            continue

        title_td = tds[0]
        info_td = tds[1]
        date_td = tds[2]

        title_tag = title_td.select_one("a")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        date_str = date_td.get_text(strip=True)
        source = info_td.get_text(strip=True)
        url = title_tag.get("href", "")

        if url and not url.startswith("http"):
            url = "https://finance.naver.com" + url

        news_list.append({
            "title": title,
            "date": date_str,
            "source": source,
            "url": url,
        })

    return news_list


def collect_news_for_signal(
    stock_code: str,
    signal_date: str,
    days_range: int = 5,
    max_pages: int = 20,
) -> list[dict]:
    """시작점 전후 N영업일 뉴스 수집.

    Args:
        stock_code: 종목코드 (6자리)
        signal_date: 시작점 날짜 (YYYY-MM-DD)
        days_range: 전후 조회 범위 (영업일 기준, 달력일로 환산 시 ×1.5)
        max_pages: 최대 페이지 수

    Returns:
        날짜 범위 내 뉴스 리스트 (각 뉴스에 timing 필드 추가)
    """
    sig_date = datetime.strptime(signal_date, "%Y-%m-%d")
    # 영업일 5일 ≈ 달력일 7~8일 → 여유 있게 10일
    start_dt = sig_date - timedelta(days=days_range * 2)
    end_dt = sig_date + timedelta(days=days_range * 2)

    all_news = []
    seen_titles = set()
    past_target = False

    for page in range(1, max_pages + 1):
        news = fetch_naver_news(stock_code, page)
        if not news:
            break

        for item in news:
            try:
                news_dt = datetime.strptime(item["date"], "%Y.%m.%d %H:%M")
            except ValueError:
                try:
                    news_dt = datetime.strptime(item["date"], "%Y.%m.%d")
                except ValueError:
                    continue

            # 목표 날짜를 완전히 지나쳤으면 중단
            if news_dt < start_dt:
                past_target = True
                break
            if news_dt > end_dt:
                continue

            title_key = item["title"][:50]  # 중복 제거용
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # timing 판정: 시작점 전/후
            if news_dt.date() < sig_date.date():
                timing = "before"
            elif news_dt.date() == sig_date.date():
                timing = "same_day"
            else:
                timing = "after"

            item["news_date"] = news_dt.strftime("%Y-%m-%d")
            item["timing"] = timing
            item["days_diff"] = (news_dt.date() - sig_date.date()).days
            all_news.append(item)

        if past_target:
            break

        time.sleep(0.5)  # rate limit

    return all_news


# ─── 촉매 분류 (규칙 기반) ───

CATALYST_PATTERNS = [
    # 실적
    (r"실적.*서프라이즈|흑자.*전환|영업이익.*증가|매출.*성장|분기.*최대|어닝.*서프", "catalyst", "실적"),
    (r"수주|계약.*체결|공급.*계약|납품|수출.*계약", "catalyst", "수주"),
    (r"자사주.*매입|자사주.*소각|자기주식", "catalyst", "자사주"),
    (r"인수.*합병|M&A|지분.*인수|경영권", "catalyst", "M&A"),
    (r"신사업|신제품|신약.*승인|FDA|허가", "catalyst", "신사업"),
    (r"정책.*수혜|보조금|지원.*사업|국책|정부.*지원", "catalyst", "정책"),
    (r"공매도.*해제|공매도.*과열.*지정", "catalyst", "공매도"),
    (r"대주주.*매입|최대주주.*변경|지분.*확대", "catalyst", "대주주"),
    (r"배당.*확대|특별.*배당|배당.*증가", "catalyst", "배당"),
    (r"투자.*확대|설비.*증설|증설|CAPEX", "catalyst", "투자"),

    # 후행 소음
    (r"급등|상한가|이유는|왜.*올랐|급상승", "noise", "급등해설"),
    (r"목표가.*상향|목표주가.*상향|투자의견.*상향", "noise", "목표가"),
    (r"수급.*동향|외국인.*매수|기관.*매수|거래량.*폭발", "noise", "수급보도"),
    (r"52주.*신고|역대.*최고|사상.*최고", "noise", "최고가보도"),
    (r"테마.*급등|관련주.*강세", "noise", "테마"),

    # 무관
    (r"감사보고서|분기보고서|반기보고서|사업보고서|주주총회", "irrelevant", "정기공시"),
    (r"IR.*일정|투자설명회|기업설명회", "irrelevant", "IR"),
    (r"임원.*선임|이사회|사외이사", "irrelevant", "경영"),
]


def classify_news_rule_based(title: str) -> tuple[str, str]:
    """규칙 기반 뉴스 분류.

    Returns:
        (category, sub_category)
        category: "catalyst" | "noise" | "irrelevant" | "unknown"
    """
    for pattern, category, sub_cat in CATALYST_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return category, sub_cat
    return "unknown", ""


# ─── GPT 기반 촉매 추론 (네이버 크롤링 실패/부족 시 보조) ───

def infer_catalyst_with_gpt(
    stock_name: str,
    stock_code: str,
    signal_date: str,
    notes: str,
) -> dict:
    """GPT-4o-mini로 시작점 촉매 추론.

    네이버 크롤링이 대형주 뉴스량 초과로 실패할 때 사용.
    GPT의 학습 데이터 기반으로 해당 시점의 촉매를 추론.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"status": "error", "reason": "openai not installed"}

    client = OpenAI()

    prompt = f"""한국 주식 종목의 특정 날짜 전후 촉매(catalyst)를 분석해주세요.

종목: {stock_name} ({stock_code})
시작점 날짜: {signal_date}
참고: {notes}

이 날짜 전후 5영업일에 해당 종목에 영향을 줄 수 있었던 뉴스/공시/이벤트를 분석해주세요.

다음 JSON 형식으로 답변해주세요:
{{
  "catalysts": [
    {{"type": "실적/수주/정책/M&A/신사업/자사주/배당/공매도/테마/기타", "title": "촉매 제목", "timing": "before/after/same_day", "confidence": "high/medium/low"}}
  ],
  "noise": [
    {{"type": "급등해설/목표가/수급보도/기타", "title": "소음 제목"}}
  ],
  "summary": "한줄 요약",
  "catalyst_timing": "before/after/none"
}}

주의:
- 확실하지 않은 정보는 confidence를 "low"로 표시
- 촉매가 없었다면 catalysts를 빈 리스트로
- 기술적 상승이었다면 summary에 "기술적 상승 (촉매 없음)" 기재"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.warning(f"  GPT 추론 실패: {e}")
        return {"status": "error", "reason": str(e)}


# ─── 메인 실행 ───

def run_crawling(signal_csv: str, output_dir: str, gpt_only: bool = False) -> dict:
    """91건 시작점 뉴스 수집 + 분류.

    Args:
        gpt_only: True면 네이버 크롤링 스킵, GPT 추론만 사용

    Returns:
        {"signal_news": [...], "summary": {...}}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # signal_points.csv 로드
    with open(signal_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        signals = list(reader)

    logger.info(f"시작점 {len(signals)}건 뉴스 수집 시작")

    all_results = []
    failed_count = 0

    for i, sig in enumerate(signals):
        stock_code = sig["stock_code"]
        stock_name = sig["stock_name"]
        signal_date = sig["signal_date"]

        logger.info(f"[{i+1}/{len(signals)}] {stock_name}({stock_code}) {signal_date}")

        # 네이버 크롤링 (gpt_only가 아닐 때만)
        news = []
        if not gpt_only:
            news = collect_news_for_signal(stock_code, signal_date)
            naver_count = len(news)
            if naver_count > 0:
                logger.info(f"  → 네이버 뉴스 {naver_count}건")

        # GPT 추론 (항상 실행 — 네이버가 부족해도 GPT로 보완)
        gpt_result = infer_catalyst_with_gpt(
            stock_name, stock_code, signal_date, sig.get("notes", "")
        )
        gpt_status = gpt_result.get("status", "ok")
        if gpt_status == "error":
            logger.warning(f"  → GPT 추론 실패: {gpt_result.get('reason', '')}")
        else:
            gpt_cat_count = len(gpt_result.get("catalysts", []))
            logger.info(f"  → GPT 촉매 {gpt_cat_count}건: {gpt_result.get('summary', '')[:50]}")

        if not news and gpt_status == "error":
            failed_count += 1

        # GPT 결과에서 촉매/소음 뉴스 아이템 생성
        gpt_catalysts = gpt_result.get("catalysts", [])
        gpt_noises = gpt_result.get("noise", [])
        gpt_summary = gpt_result.get("summary", "")
        gpt_timing = gpt_result.get("catalyst_timing", "none")

        # 규칙 기반 분류 (네이버 뉴스)
        catalysts = []
        noises = []
        irrelevants = []
        unknowns = []

        for n in news:
            cat, sub = classify_news_rule_based(n["title"])
            n["category"] = cat
            n["sub_category"] = sub

            if cat == "catalyst":
                catalysts.append(n)
            elif cat == "noise":
                noises.append(n)
            elif cat == "irrelevant":
                irrelevants.append(n)
            else:
                unknowns.append(n)

        # GPT 추론 촉매를 추가 (네이버에서 못 찾은 경우)
        for gc in gpt_catalysts:
            catalysts.append({
                "title": gc.get("title", ""),
                "category": "catalyst",
                "sub_category": gc.get("type", "기타"),
                "source": "GPT추론",
                "timing": gc.get("timing", "unknown"),
                "confidence": gc.get("confidence", "low"),
                "news_date": signal_date,
                "days_diff": 0,
            })

        # 선행 촉매 (시작점 전)
        pre_catalysts = [c for c in catalysts if c["timing"] in ("before", "same_day")]
        post_catalysts = [c for c in catalysts if c["timing"] == "after"]

        # GPT timing을 우선 사용 (네이버 뉴스가 없을 때)
        effective_timing = (
            "before" if pre_catalysts else
            "after" if post_catalysts else
            gpt_timing if gpt_timing != "none" else
            "none"
        )

        result = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "signal_date": signal_date,
            "notes": sig.get("notes", ""),
            "total_news": len(news),
            "catalyst_count": len(catalysts),
            "noise_count": len(noises),
            "irrelevant_count": len(irrelevants),
            "unknown_count": len(unknowns),
            "pre_catalyst_count": len(pre_catalysts),
            "post_catalyst_count": len(post_catalysts),
            "top_catalyst": pre_catalysts[0]["title"] if pre_catalysts else (
                post_catalysts[0]["title"] if post_catalysts else
                gpt_summary if gpt_summary else ""
            ),
            "catalyst_timing": effective_timing,
            "gpt_summary": gpt_summary,
            "gpt_catalysts": gpt_catalysts,
            "news_items": news,
        }
        all_results.append(result)

        # 중간 저장 (10건마다)
        if (i + 1) % 10 == 0:
            _save_intermediate(all_results, output_path)
            logger.info(f"  중간 저장 ({i+1}/{len(signals)})")

        time.sleep(0.5)  # 종목 간 간격

    # 최종 저장
    _save_intermediate(all_results, output_path)

    # 요약 CSV 생성
    _save_summary_csv(all_results, output_path)

    # 패턴 분석
    analysis = _analyze_patterns(all_results)

    logger.info(f"\n크롤링 완료: {len(signals)}건 중 실패 {failed_count}건")
    logger.info(f"결과 저장: {output_path}")

    return {"results": all_results, "analysis": analysis}


def _save_intermediate(results: list, output_path: Path):
    """중간 결과 JSON 저장."""
    with open(output_path / "signal_news_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _save_summary_csv(results: list, output_path: Path):
    """요약 CSV 저장 — signal_points.csv에 촉매 컬럼 추가."""
    fieldnames = [
        "stock_code", "stock_name", "signal_date", "notes",
        "total_news", "catalyst_count", "noise_count",
        "pre_catalyst_count", "top_catalyst", "catalyst_timing",
    ]
    with open(output_path / "signal_catalyst_summary.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    logger.info(f"요약 CSV 저장: {output_path / 'signal_catalyst_summary.csv'}")


def _analyze_patterns(results: list) -> dict:
    """3그룹 패턴 분석.

    그룹 A: 선행 촉매가 있었던 시작점
    그룹 B: 촉매 없이 기술적으로만 시작된 시작점
    그룹 C: 후행 뉴스만 있었던 시작점
    """
    group_a = []  # 선행 촉매
    group_b = []  # 촉매 없음 (기술적 시작)
    group_c = []  # 후행 뉴스만

    for r in results:
        if r["catalyst_timing"] == "before":
            group_a.append(r)
        elif r["catalyst_count"] == 0 and r["noise_count"] == 0:
            group_b.append(r)
        else:
            group_c.append(r)

    total = len(results)

    analysis = {
        "total_signals": total,
        "group_a": {
            "name": "선행 촉매",
            "count": len(group_a),
            "pct": f"{len(group_a)/total:.1%}" if total > 0 else "0%",
            "examples": [
                f"{r['stock_name']} {r['signal_date']}: {r['top_catalyst']}"
                for r in group_a[:5]
            ],
        },
        "group_b": {
            "name": "기술적 시작 (촉매 없음)",
            "count": len(group_b),
            "pct": f"{len(group_b)/total:.1%}" if total > 0 else "0%",
            "examples": [f"{r['stock_name']} {r['signal_date']}" for r in group_b[:5]],
        },
        "group_c": {
            "name": "후행 뉴스/소음만",
            "count": len(group_c),
            "pct": f"{len(group_c)/total:.1%}" if total > 0 else "0%",
            "examples": [
                f"{r['stock_name']} {r['signal_date']}: noise={r['noise_count']}"
                for r in group_c[:5]
            ],
        },
    }

    # 촉매 유형 분포
    catalyst_types = {}
    for r in results:
        for n in r.get("news_items", []):
            if n.get("category") == "catalyst":
                sub = n.get("sub_category", "기타")
                catalyst_types[sub] = catalyst_types.get(sub, 0) + 1

    analysis["catalyst_type_distribution"] = dict(
        sorted(catalyst_types.items(), key=lambda x: -x[1])
    )

    return analysis


def print_analysis(analysis: dict):
    """분석 결과 출력."""
    print("\n" + "=" * 60)
    print("91건 시작점 촉매 분석 결과")
    print("=" * 60)

    for group_key in ["group_a", "group_b", "group_c"]:
        g = analysis[group_key]
        print(f"\n{g['name']}: {g['count']}건 ({g['pct']})")
        for ex in g["examples"]:
            print(f"  - {ex}")

    print(f"\n촉매 유형 분포:")
    for cat_type, count in analysis.get("catalyst_type_distribution", {}).items():
        print(f"  {cat_type}: {count}건")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="91건 시작점 뉴스 크롤링")
    parser.add_argument("--analyze", action="store_true", help="분석만 실행 (크롤링 스킵)")
    parser.add_argument("--gpt-only", action="store_true", help="GPT 추론만 (네이버 크롤링 스킵)")
    parser.add_argument("--signal-csv", default="data/signal_points.csv")
    parser.add_argument("--output-dir", default="data/signal_news")
    args = parser.parse_args()

    if args.analyze:
        # 기존 크롤링 결과에서 분석만
        raw_path = Path(args.output_dir) / "signal_news_raw.json"
        if not raw_path.exists():
            print(f"크롤링 결과 없음: {raw_path}")
            sys.exit(1)
        with open(raw_path, encoding="utf-8") as f:
            results = json.load(f)
        analysis = _analyze_patterns(results)
        print_analysis(analysis)
    else:
        result = run_crawling(args.signal_csv, args.output_dir, gpt_only=args.gpt_only)
        print_analysis(result["analysis"])
