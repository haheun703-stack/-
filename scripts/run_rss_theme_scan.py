"""RSS 테마 스캔 러너 (BAT-D용, graceful).

settings.yaml의 theme_scanner 설정을 로드해 RssThemeScanner를 돌리고,
당일 발화 테마 스냅샷을 data/theme_alerts_today.json에 기록한다.
scan_tomorrow_picks 전략 T(_load_news_theme_map)가 이 파일을 소비해
뉴스 발화 테마 관련주에 동적 가점을 준다.

★설계 원칙 (옛 theme_scan_runner 폐지 교훈, run_bat.sh:116):
  - archive 밖 신설 (LOCK 위반 방지)
  - **어떤 실패에도 exit 0** (RSS 장애·피드 오류가 BAT-D FAIL_COUNT를 올리지 않게)
  - 실패 시에도 빈 스냅샷(당일 날짜+빈 ticker_map)을 남겨 소비측 freshness 가드가 정상 동작

★dedup 비활성(dedup_hours=0) 이유:
  스냅샷은 "오늘 뉴스에 발화 중인 테마 전량"이 필요. 어댑터 기본 dedup(24h)이면
  지속 테마가 최초 발화 이튿날부터 알림에서 빠져 가점이 사라진다. 매일 현재 스냅샷을 원함.

사용: python scripts/run_rss_theme_scan.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # BAT PYTHONPATH 안전장치

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_rss_theme_scan")

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "theme_alerts_today.json"


def _write_snapshot(payload: dict) -> None:
    """스냅샷 기록 (실패해도 조용히)."""
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("스냅샷 저장: %s (테마 %d·티커 %d)",
                    OUTPUT_PATH.name, len(payload.get("themes", [])),
                    len(payload.get("ticker_map", {})))
    except Exception as e:
        logger.warning("스냅샷 저장 실패: %s", e)


def _load_excluded_themes() -> set:
    """theme_dictionary.yaml에서 news_track:false 테마 (뉴스 동적가점 제외).

    거시/상시 테마(금리·환율·밸류업·부동산 등)는 일반명사 키워드라 상시 발화 →
    대형주 광범위 오가점 유발. 이런 테마를 스냅샷에서 배제해 판별력 확보.
    (opt-out: 플래그 없으면 뉴스 추적 대상)
    """
    try:
        import yaml
        fp = PROJECT_ROOT / "config" / "theme_dictionary.yaml"
        with open(fp, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return {name for name, td in (d.get("themes", {}) or {}).items()
                if isinstance(td, dict) and td.get("news_track") is False}
    except Exception as e:
        logger.warning("news_track 제외목록 로드 실패: %s", e)
        return set()


def _empty_snapshot(reason: str) -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "alert_count": 0,
        "themes": [],
        "ticker_map": {},
        "alerts": [],
        "note": reason,
    }


def run() -> dict:
    """RSS 테마 스캔 실행 → 스냅샷 dict 반환 (예외는 상위에서 포착)."""
    import yaml

    # 1) 설정 로드
    if not SETTINGS_PATH.exists():
        logger.warning("settings.yaml 없음 — 빈 스냅샷")
        return _empty_snapshot("settings_missing")
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    cfg = (settings.get("theme_scanner") or {})

    if not cfg.get("enabled", False):
        logger.info("theme_scanner 비활성 — 빈 스냅샷")
        return _empty_snapshot("disabled")

    feeds = cfg.get("rss_feeds", []) or []
    if not feeds:
        logger.warning("rss_feeds 미설정 — 빈 스냅샷")
        return _empty_snapshot("no_feeds")

    # 2) 스캐너 실행 (dedup=0: 당일 발화 테마 전량 스냅샷)
    from src.adapters.rss_theme_scanner import RssThemeScanner

    scanner = RssThemeScanner(
        rss_feeds=feeds,
        dedup_hours=0,
        max_articles=int(cfg.get("max_articles", 50)),
        max_age_hours=int(cfg.get("max_age_hours", 48)),
        min_hits=int(cfg.get("min_hits", 3)),  # 단발 언급 노이즈 차단
    )
    alerts = scanner.scan()

    # 2.5) 거시/상시 테마(news_track:false) 배제 — 대형주 오가점 방지
    excluded = _load_excluded_themes()
    if excluded:
        before = len(alerts)
        alerts = [a for a in alerts if a.theme_name not in excluded]
        if before != len(alerts):
            logger.info("news_track 제외: %d테마 배제 (%s)", before - len(alerts),
                        ", ".join(sorted(excluded)))

    # 3) 스냅샷 구성: ticker → [(theme, order)]
    ticker_map: dict[str, list] = {}
    themes: list[str] = []
    alert_dump: list[dict] = []
    for a in alerts:
        themes.append(a.theme_name)
        stocks_dump = []
        for s in a.related_stocks:
            tk = str(getattr(s, "ticker", "") or "")
            order = int(getattr(s, "order", 1) or 1)
            if tk:
                ticker_map.setdefault(tk, []).append([a.theme_name, order])
                stocks_dump.append({"ticker": tk, "name": getattr(s, "name", ""), "order": order})
        alert_dump.append({
            "theme": a.theme_name,
            "keyword": a.matched_keyword,
            "hit_count": getattr(a, "hit_count", 0),
            "title": a.news_title,
            "url": a.news_url,
            "source": a.news_source,
            "stocks": stocks_dump,
        })

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "alert_count": len(alerts),
        "themes": sorted(set(themes)),
        "ticker_map": ticker_map,
        "alerts": alert_dump,
    }


def main() -> int:
    logger.info("=== RSS 테마 스캔 시작 ===")
    try:
        payload = run()
    except Exception as e:  # 어떤 실패도 BAT FAIL_COUNT로 잇지 않는다
        logger.warning("스캔 실패 (%s) — 빈 스냅샷으로 대체", e)
        payload = _empty_snapshot(f"error:{type(e).__name__}")
    _write_snapshot(payload)
    if payload.get("alert_count", 0):
        top = ", ".join(payload["themes"][:8])
        logger.info("발화 테마 %d종: %s", len(payload["themes"]), top)
    else:
        logger.info("발화 테마 없음")
    logger.info("=== RSS 테마 스캔 완료 ===")
    return 0  # ★항상 정상 종료


if __name__ == "__main__":
    sys.exit(main())
