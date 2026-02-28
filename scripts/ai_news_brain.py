"""AI 두뇌 — 뉴스 종합 분석 → 종목별 BUY/WATCH/AVOID 판단

기존 5개 뉴스 소스(RSS, Perplexity, Grok, DART, 네이버)에서
50~70개 뉴스를 수집 → Claude Sonnet 4.6이 정성적 분석 → JSON 출력.

출력: data/ai_brain_judgment.json
연동: scan_tomorrow_picks.py (ai_bonus ±7점), send_evening_summary.py

Usage:
    python scripts/ai_news_brain.py           # 분석만
    python scripts/ai_news_brain.py --send    # 분석 + 텔레그램 요약 전송
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.agents.news_brain import NewsBrainAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
OUTPUT_PATH = DATA_DIR / "ai_brain_judgment.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ──────────────────────────────────────────
# 설정 로드
# ──────────────────────────────────────────

def load_settings() -> dict:
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f).get("ai_brain", {})


# ──────────────────────────────────────────
# 유니버스 빌드 (ticker → name)
# ──────────────────────────────────────────

def build_universe() -> dict[str, str]:
    """mechanical_universe_final.csv 기반 84종목 유니버스 (in_parquet=True)."""
    csv_path = DATA_DIR / "mechanical_universe_final.csv"
    if not csv_path.exists():
        # fallback: processed parquet 전체
        logger.warning("유니버스 CSV 없음 — processed/*.parquet 전체 사용")
        universe = {}
        for pq in PROCESSED_DIR.glob("*.parquet"):
            ticker = pq.stem
            name = ticker
            for c in CSV_DIR.glob(f"*_{ticker}.csv"):
                parts = c.stem.rsplit("_", 1)
                if len(parts) == 2:
                    name = parts[0]
                    break
            universe[ticker] = name
        return universe

    df = pd.read_csv(csv_path)
    df = df[df["in_parquet"] == True]  # noqa: E712
    universe = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        name = str(row["name"])
        universe[ticker] = name
    return universe


# ──────────────────────────────────────────
# 뉴스 수집 (기존 JSON 5개 소스 통합)
# ──────────────────────────────────────────

def _load_json(name: str) -> dict | list:
    path = DATA_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _collect_market_news() -> list[dict]:
    """Google RSS 뉴스 (market_news.json)."""
    data = _load_json("market_news.json")
    articles = data.get("articles", [])
    result = []
    for a in articles:
        if a.get("impact") in ("high", "medium"):
            result.append({
                "title": a.get("title", ""),
                "summary": "",
                "source": a.get("source", "Google RSS"),
                "category": a.get("category", "매크로"),
                "impact": a.get("impact", ""),
            })
    return result


def _collect_intelligence() -> list[dict]:
    """Perplexity 인텔리전스 (market_intelligence.json)."""
    data = _load_json("market_intelligence.json")
    result = []

    # key_events → 뉴스 항목
    for ev in data.get("key_events", []):
        result.append({
            "title": ev.get("event", ""),
            "summary": ev.get("detail", ""),
            "source": "Perplexity",
            "category": ev.get("category", "매크로"),
            "impact": ev.get("urgency", ""),
        })

    # us_market_summary → 요약 뉴스
    us_summary = data.get("us_market_summary", "")
    if us_summary:
        result.append({
            "title": f"미국장 요약: {data.get('us_market_mood', '')}",
            "summary": us_summary[:300],
            "source": "Perplexity",
            "category": "미국장",
            "impact": "IMPORTANT",
        })

    # sector_impacts → 섹터별 뉴스
    for sec in data.get("sector_impacts", []):
        result.append({
            "title": f"[{sec.get('sector', '')}] 섹터 영향",
            "summary": sec.get("reason", "")[:200],
            "source": "Perplexity",
            "category": f"섹터:{sec.get('sector', '')}",
            "impact": "medium",
        })

    return result


def _collect_dart() -> list[dict]:
    """DART 공시 (dart_disclosures.json) — tier1만."""
    data = _load_json("dart_disclosures.json")
    result = []
    for d in data.get("tier1", []):
        result.append({
            "title": f"[DART 공시] {d.get('corp_name', '')} — {d.get('report_nm', '')}",
            "summary": f"키워드: {d.get('keyword', '')} | 시장: {d.get('market', '')}",
            "source": "DART",
            "category": "공시",
            "impact": "high",
        })
    return result


def _collect_signal_news() -> list[dict]:
    """네이버 크롤링 뉴스 (signal_news.json)."""
    data = _load_json("signal_news.json")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("news", data.get("items", []))
    else:
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", item.get("description", "")),
            "source": item.get("source", "네이버"),
            "category": item.get("category", item.get("classification", "종목")),
            "impact": item.get("impact", "medium"),
        })
    return result


def _collect_grok_news() -> list[dict]:
    """Grok 뉴스 (data/grok_news_*.json)."""
    result = []
    for gf in sorted(DATA_DIR.glob("grok_news_*.json")):
        try:
            with open(gf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("news", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append({
                "title": item.get("title", item.get("headline", "")),
                "summary": item.get("summary", item.get("analysis", "")),
                "source": item.get("source", "Grok"),
                "category": item.get("category", "종목"),
                "impact": item.get("impact", "medium"),
            })
    return result


def _deduplicate(news: list[dict], threshold: float = 0.7) -> list[dict]:
    """제목 유사도 기반 중복 제거."""
    unique = []
    seen_titles = []
    for item in news:
        title = item.get("title", "")
        if not title:
            continue
        is_dup = False
        for seen in seen_titles:
            if SequenceMatcher(None, title, seen).ratio() > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
            seen_titles.append(title)
    return unique


def collect_all_news(max_items: int = 70) -> list[dict]:
    """5개 소스에서 뉴스 통합 수집."""
    all_news = []

    # 소스별 수집 (우선순위: DART > Perplexity > RSS > Grok > 네이버)
    dart_news = _collect_dart()
    intel_news = _collect_intelligence()
    market_news = _collect_market_news()
    grok_news = _collect_grok_news()
    signal_news = _collect_signal_news()

    logger.info(
        "뉴스 수집: DART=%d, Perplexity=%d, RSS=%d, Grok=%d, 네이버=%d",
        len(dart_news), len(intel_news), len(market_news),
        len(grok_news), len(signal_news),
    )

    # 우선순위 순 병합
    all_news.extend(dart_news)
    all_news.extend(intel_news)
    all_news.extend(market_news)
    all_news.extend(grok_news)
    all_news.extend(signal_news)

    # 중복 제거
    unique = _deduplicate(all_news)
    logger.info("중복 제거: %d → %d건", len(all_news), len(unique))

    # 최대 개수 제한
    if len(unique) > max_items:
        unique = unique[:max_items]

    return unique


# ──────────────────────────────────────────
# 채널 4: 피드백 루프 — 이전 판단 아카이브 + 수익률 추적
# ──────────────────────────────────────────

HISTORY_PATH = DATA_DIR / "ai_brain_history.json"


def _archive_previous_judgment():
    """이전 AI 판단을 히스토리에 아카이브하고 D+3/5/7 수익률 추적."""
    if not OUTPUT_PATH.exists():
        return

    try:
        prev = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    prev_date = prev.get("date", "")
    if not prev_date:
        return

    # 히스토리 로드/생성
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = []

    # 중복 방지
    existing_dates = {h.get("date") for h in history}
    if prev_date not in existing_dates:
        history.append(prev)
        logger.info("[아카이브] %s AI 판단 저장 (%d건)",
                     prev_date, len(prev.get("stock_judgments", [])))

    # D+3/5/7 수익률 추적
    _update_tracking_returns(history)

    # 최대 90일분 유지
    history = history[-90:]
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _update_tracking_returns(history: list):
    """과거 AI BUY 판단의 D+3/5/7 수익률 추적."""
    today = datetime.now().date()

    for entry in history:
        if entry.get("tracking_complete"):
            continue

        entry_date_str = entry.get("date", "")
        if not entry_date_str:
            continue
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        days_elapsed = (today - entry_date).days
        if days_elapsed < 3:
            continue

        for j in entry.get("stock_judgments", []):
            if j.get("action") != "BUY":
                continue
            ticker = j.get("ticker", "")
            if not ticker:
                continue

            pq_path = PROCESSED_DIR / f"{ticker}.parquet"
            if not pq_path.exists():
                continue

            try:
                df = pd.read_parquet(pq_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                mask = df.index.date >= entry_date
                if mask.sum() < 2:
                    continue
                sub = df[mask]
                base_price = float(sub.iloc[0]["close"])

                for d in [3, 5, 7]:
                    key = f"ret_d{d}"
                    if key in j:
                        continue
                    if len(sub) > d:
                        future_price = float(sub.iloc[d]["close"])
                        j[key] = round((future_price / base_price - 1) * 100, 2)
            except Exception:
                continue

        if days_elapsed >= 7:
            entry["tracking_complete"] = True


# ──────────────────────────────────────────
# 메인 분석 실행
# ──────────────────────────────────────────

async def run_daily_analysis(cfg: dict) -> dict:
    """뉴스 수집 → AI 분석 → JSON 저장."""
    if not cfg.get("enabled", True):
        logger.info("AI Brain 비활성화 — 스킵")
        return {}

    # 0. 이전 AI 판단 아카이브 + 수익률 추적
    _archive_previous_judgment()

    max_items = cfg.get("max_news_items", 70)
    max_judgments = cfg.get("max_stock_judgments", 50)
    model = cfg.get("model", "claude-sonnet-4-5-20250929")

    # 1. 뉴스 수집
    news = collect_all_news(max_items=max_items)
    if len(news) < 5:
        logger.warning("뉴스 %d건 — 분석 불충분, 스킵", len(news))
        return {}

    # 2. 유니버스 빌드
    universe = build_universe()
    logger.info("유니버스: %d종목, 최대 판단: %d개", len(universe), max_judgments)

    # 3. AI 분석
    agent = NewsBrainAgent(model=model)
    result = await agent.analyze_daily_news(news, universe, max_judgments=max_judgments)

    # 4. 메타데이터 추가
    result["date"] = datetime.now().strftime("%Y-%m-%d")
    result["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    result["model"] = model
    result["news_count"] = len(news)

    # 5. JSON 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("저장 완료: %s", OUTPUT_PATH)

    return result


def _print_summary(result: dict) -> None:
    """분석 결과 콘솔 요약."""
    if not result:
        return

    print("\n" + "=" * 60)
    print("🧠 AI 두뇌 분석 결과")
    print("=" * 60)
    print(f"날짜: {result.get('date', '?')}")
    print(f"시장 센티먼트: {result.get('market_sentiment', '?')}")
    themes = result.get("key_themes", [])
    if themes:
        print(f"핵심 테마: {', '.join(themes)}")

    judgments = result.get("stock_judgments", [])
    if judgments:
        print(f"\n종목 판단: {len(judgments)}개")
        print("-" * 50)
        for j in judgments:
            action = j.get("action", "?")
            icon = {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴"}.get(action, "⚪")
            conf = j.get("confidence", 0)
            print(
                f"  {icon} {j.get('name', '?'):10s} | {action:5s} "
                f"({conf:.0%}) | {j.get('reasoning', '')[:40]}"
            )

    sector = result.get("sector_outlook", {})
    if sector:
        print(f"\n섹터 전망: {len(sector)}개")
        for s, info in sector.items():
            d = info.get("direction", "?") if isinstance(info, dict) else "?"
            print(f"  {s}: {d}")

    print(f"\n분석 뉴스: {result.get('news_count', 0)}건")
    print("=" * 60)


def _send_telegram(result: dict) -> None:
    """핵심 요약을 텔레그램으로 전송."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        logger.warning("telegram_sender 임포트 실패 — 전송 스킵")
        return

    if not result or not result.get("stock_judgments"):
        return

    lines = ["🧠 AI 두뇌 일일 분석"]
    lines.append(f"센티먼트: {result.get('market_sentiment', '?')}")
    themes = result.get("key_themes", [])
    if themes:
        lines.append(f"테마: {', '.join(themes[:3])}")
    lines.append("")

    for j in result.get("stock_judgments", []):
        action = j.get("action", "?")
        icon = {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴"}.get(action, "⚪")
        conf = j.get("confidence", 0)
        lines.append(
            f"{icon} {j.get('name', '?')} {action}({conf:.0%})"
        )
        lines.append(f"  → {j.get('reasoning', '')[:50]}")

    msg = "\n".join(lines)
    send_message(msg)
    logger.info("텔레그램 전송 완료")


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI 두뇌 뉴스 분석")
    parser.add_argument("--send", action="store_true", help="텔레그램 전송")
    args = parser.parse_args()

    cfg = load_settings()
    result = asyncio.run(run_daily_analysis(cfg))

    _print_summary(result)

    if args.send and result:
        _send_telegram(result)


if __name__ == "__main__":
    main()
