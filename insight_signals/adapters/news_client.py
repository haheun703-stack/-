# -*- coding: utf-8 -*-
"""뉴스 수집 어댑터.

소스 1) 한국경제 RSS (구조 안정적, 확인 완료)
소스 2) 네이버 금융 주요뉴스 (HTML — 구조 변경 가능성 있으므로 실패해도 전체는 계속)

의존성: requests + 표준 라이브러리만 사용 (bs4 불필요).
"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET

import requests

from insight_signals.entities import NewsArticle

log = logging.getLogger("insight_signals.news")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

DEFAULT_RSS_FEEDS = [
    ("한경증권", "https://www.hankyung.com/feed/finance"),
    ("한경경제", "https://www.hankyung.com/feed/economy"),
    ("한경IT", "https://www.hankyung.com/feed/it"),
]

NAVER_MAINNEWS_URL = "https://finance.naver.com/news/mainnews.naver"
# 네이버 금융 주요뉴스: <dd class="articleSubject"><a href="..." ...>제목</a>
_NAVER_ITEM_RE = re.compile(
    r'class="articleSubject"[^>]*>\s*<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.S,
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def fetch_rss(feeds=None, timeout: int = 15) -> list:
    """RSS 피드들에서 (제목, 링크, 날짜) 수집."""
    feeds = feeds or DEFAULT_RSS_FEEDS
    out = []
    for name, url in feeds:
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = _strip_tags(item.findtext("title") or "")
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if title:
                    out.append(NewsArticle(title=title, link=link, published=pub, source=name))
        except Exception as e:  # noqa: BLE001 — 소스 하나 죽어도 계속
            log.warning("RSS 수집 실패 [%s]: %s", name, e)
    log.info("RSS 기사 %d건 수집", len(out))
    return out


def fetch_naver_mainnews(pages: int = 2, timeout: int = 15) -> list:
    """네이버 금융 주요뉴스 (best-effort). 실패 시 빈 리스트."""
    out = []
    for page in range(1, pages + 1):
        try:
            r = requests.get(
                NAVER_MAINNEWS_URL, params={"page": page}, headers=UA, timeout=timeout
            )
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "euc-kr"
            for m in _NAVER_ITEM_RE.finditer(r.text):
                title = _strip_tags(m.group("title"))
                href = html.unescape(m.group("href"))
                if href.startswith("/"):
                    href = "https://finance.naver.com" + href
                if title:
                    out.append(
                        NewsArticle(title=title, link=href, published="", source="네이버금융")
                    )
        except Exception as e:  # noqa: BLE001
            log.warning("네이버 주요뉴스 수집 실패 (page=%s): %s", page, e)
            break
    log.info("네이버 주요뉴스 %d건 수집", len(out))
    return out


# ----------------------------------------------------------------------
# 키워드 매칭 + 종목 매핑
# ----------------------------------------------------------------------
def match_keywords(articles, keywords, negative_keywords=None) -> list:
    """제목에 관심 키워드가 포함된 기사만 남기고 matched_keywords 채움."""
    negative_keywords = negative_keywords or []
    out = []
    for a in articles:
        if any(nk in a.title for nk in negative_keywords):
            continue
        hits = [k for k in keywords if k in a.title]
        if hits:
            a.matched_keywords = hits
            out.append(a)
    return out


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ("가" <= ch <= "힣")


def _name_in_text(name: str, text: str) -> bool:
    """종목명 매칭. 3글자 이하 이름은 단어 경계 필수 (7/10 검수 픽스).

    구버전 부분일치는 '레이저→레이', '송호성→호성', '태양광→태양' 류의
    체계적 오탐을 만들었음 (7/9 실측 5건+).
    """
    if len(name) >= 4:
        return name in text
    start = 0
    while True:
        i = text.find(name, start)
        if i < 0:
            return False
        before = text[i - 1] if i > 0 else " "
        after_i = i + len(name)
        after = text[after_i] if after_i < len(text) else " "
        if not _is_word_char(before) and not _is_word_char(after):
            return True
        start = i + 1


def map_stocks(articles, name_to_code: dict, blacklist=None, min_name_len: int = 2) -> list:
    """기사 제목에 등장하는 상장사명을 찾아 (code, name) 매핑.

    - 긴 이름 우선 매칭 ("LG에너지솔루션"이 "LG"보다 먼저)
    - 겹치는 짧은 이름은 긴 이름에 포함되면 제외
    - blacklist: 일반명사와 겹치는 회사명 제외 목록
    - 매칭 전에 제목 꼬리의 매체명(" - 매체")을 절단 (구글뉴스 형식,
      매체명이 상장사명과 겹치는 오탐 방지 — 7/10 검수 픽스)
    """
    blacklist = set(blacklist or [])
    names = sorted(
        (n for n in name_to_code if len(n) >= min_name_len and n not in blacklist),
        key=len,
        reverse=True,
    )
    out = []
    for a in articles:
        body = a.title.rsplit(" - ", 1)[0] if " - " in a.title else a.title
        found, consumed = [], []
        for n in names:
            if _name_in_text(n, body) and not any(n in c for c in consumed):
                found.append((name_to_code[n], n))
                consumed.append(n)
            if len(found) >= 3:  # 제목 하나에서 과도한 매핑 방지
                break
        if found:
            a.matched_stocks = found
            out.append(a)
    return out
