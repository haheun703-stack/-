"""
RSS 테마 스캐너 어댑터

RSS 피드에서 뉴스를 수집하고 테마 키워드 매칭으로
관련 종목을 찾아내는 1차 감지 시스템.

theme_dictionary.yaml의 키워드와 매칭하여 ThemeAlert를 생성한다.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import yaml

from src.entities.news_models import ThemeAlert, ThemeStock

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
THEME_DICT_PATH = PROJECT_ROOT / "config" / "theme_dictionary.yaml"
ALERT_HISTORY_PATH = PROJECT_ROOT / "data" / "theme_alert_history.json"


class RssThemeScanner:
    """RSS 피드 테마 키워드 매칭 스캐너"""

    def __init__(
        self,
        theme_dict_path: str | Path | None = None,
        rss_feeds: list[dict] | None = None,
        dedup_hours: int = 24,
        max_articles: int = 50,
    ):
        self.theme_dict_path = Path(theme_dict_path) if theme_dict_path else THEME_DICT_PATH
        self.rss_feeds = rss_feeds or []
        self.dedup_hours = dedup_hours
        self.max_articles = max_articles
        self.theme_dict: dict = {}

    def scan(self) -> list[ThemeAlert]:
        """RSS 피드 스캔 → 테마 키워드 매칭 → ThemeAlert 리스트 반환"""
        self.theme_dict = self._load_theme_dict()
        if not self.theme_dict:
            logger.warning("테마 딕셔너리가 비어있음 — 스캔 건너뜀")
            return []

        if not self.rss_feeds:
            logger.warning("RSS 피드가 설정되지 않음 — 스캔 건너뜀")
            return []

        # RSS 기사 수집
        articles = self._fetch_all_feeds()
        logger.info("RSS 총 %d건 수집", len(articles))

        # 테마 키워드 매칭
        alerts: list[ThemeAlert] = []
        matched_themes: set[str] = set()

        for article in articles:
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            for theme_name, theme_data in self.theme_dict.items():
                if theme_name in matched_themes:
                    continue  # 이미 매칭된 테마는 건너뜀 (중복 방지)

                keywords = theme_data.get("keywords", [])
                matched_kw = self._match_keywords(text, keywords)
                if matched_kw:
                    # 중복 체크 (최근 dedup_hours 내 같은 테마)
                    if self._is_recent_duplicate(theme_name):
                        logger.debug("테마 '%s' 최근 %dh 내 중복 — 건너뜀",
                                     theme_name, self.dedup_hours)
                        matched_themes.add(theme_name)
                        continue

                    # 관련주 로드
                    stocks = self._load_theme_stocks(theme_data)

                    alert = ThemeAlert(
                        theme_name=theme_name,
                        matched_keyword=matched_kw,
                        news_title=article.get("title", ""),
                        news_url=article.get("link", ""),
                        news_source=article.get("source", ""),
                        published=article.get("published", ""),
                        related_stocks=stocks,
                        grok_expanded=False,
                        timestamp=datetime.now().isoformat(),
                    )
                    alerts.append(alert)
                    matched_themes.add(theme_name)
                    logger.info("테마 감지: '%s' (키워드: %s) — 관련주 %d개",
                                theme_name, matched_kw, len(stocks))

        # 히스토리 업데이트
        if alerts:
            self._update_alert_history(alerts)

        return alerts

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _load_theme_dict(self) -> dict:
        """theme_dictionary.yaml 로드"""
        if not self.theme_dict_path.exists():
            logger.error("테마 딕셔너리 파일 없음: %s", self.theme_dict_path)
            return {}
        try:
            with open(self.theme_dict_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("themes", {})
        except Exception as e:
            logger.error("테마 딕셔너리 로드 실패: %s", e)
            return {}

    def _fetch_all_feeds(self) -> list[dict]:
        """모든 RSS 피드에서 기사 수집"""
        articles = []
        for feed_cfg in self.rss_feeds:
            url = feed_cfg.get("url", "")
            name = feed_cfg.get("name", url)
            if not url:
                continue
            try:
                d = feedparser.parse(url)
                for entry in d.entries[: self.max_articles]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": _strip_html(entry.get("summary", "")),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": name,
                    })
                logger.info("RSS '%s': %d건 수집", name, min(len(d.entries), self.max_articles))
            except Exception as e:
                logger.warning("RSS '%s' 수집 실패: %s", name, e)
        return articles

    @staticmethod
    def _match_keywords(text: str, keywords: list[str]) -> str | None:
        """텍스트에서 키워드 매칭. 매칭된 키워드 반환, 없으면 None"""
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                return kw
        return None

    @staticmethod
    def _load_theme_stocks(theme_data: dict) -> list[ThemeStock]:
        """테마 딕셔너리에서 관련주 ThemeStock 리스트 생성"""
        stocks = []
        for s in theme_data.get("stocks", []):
            stocks.append(ThemeStock(
                ticker=s.get("ticker", ""),
                name=s.get("name", ""),
                order=s.get("order", 1),
                source="dictionary",
            ))
        return stocks

    def _is_recent_duplicate(self, theme_name: str) -> bool:
        """최근 dedup_hours 내 같은 테마 알림이 있었는지 체크"""
        history = self._load_alert_history()
        cutoff = (datetime.now() - timedelta(hours=self.dedup_hours)).isoformat()
        for entry in history:
            if entry.get("theme") == theme_name and entry.get("timestamp", "") > cutoff:
                return True
        return False

    def _update_alert_history(self, alerts: list[ThemeAlert]) -> None:
        """알림 히스토리 업데이트 (오래된 항목 자동 정리)"""
        history = self._load_alert_history()
        cutoff = (datetime.now() - timedelta(hours=self.dedup_hours * 3)).isoformat()
        # 오래된 항목 제거
        history = [h for h in history if h.get("timestamp", "") > cutoff]
        # 새 항목 추가
        for alert in alerts:
            history.append({
                "theme": alert.theme_name,
                "keyword": alert.matched_keyword,
                "title": alert.news_title,
                "timestamp": alert.timestamp,
            })
        try:
            ALERT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ALERT_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("알림 히스토리 저장 실패: %s", e)

    @staticmethod
    def _load_alert_history() -> list[dict]:
        """알림 히스토리 로드"""
        if not ALERT_HISTORY_PATH.exists():
            return []
        try:
            with open(ALERT_HISTORY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


def _strip_html(text: str) -> str:
    """HTML 태그 제거"""
    return re.sub(r"<[^>]+>", "", text)
