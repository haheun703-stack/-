# -*- coding: utf-8 -*-
"""엔티티 계층 — 순수 데이터 구조. 어떤 바깥 계층도 import 하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


SOURCE_NEWS = "news_keyword"      # 뉴스 키워드 ("사상 최대", "게임 체인저" ...)
SOURCE_DART = "dart_insider"      # 임원·주요주주 자사주 매입
SOURCE_FLOW = "flow_contrarian"   # 역발상 수급 (기관/외인 vs 개인)


@dataclass
class NewsArticle:
    title: str
    link: str
    published: str          # 원문 표기 그대로 (RSS pubDate 등)
    source: str             # 피드/사이트 이름
    matched_keywords: list = field(default_factory=list)
    matched_stocks: list = field(default_factory=list)  # [(code, name), ...]


@dataclass
class InsiderFiling:
    rcept_no: str
    rcept_dt: str           # YYYYMMDD
    corp_code: str
    corp_name: str
    stock_code: str         # 6자리 (비상장이면 "")
    reporter: str           # 보고자 (임원명 등)
    position: str           # 직위/구분
    change_qty: int         # 특정증권 증감 수량 (+매수 / -매도)
    change_reason: str      # 증감 사유 (장내매수 등)


@dataclass
class FlowSnapshot:
    stock_code: str
    days: int               # 집계 일수
    person_net: int         # 개인 순매수 합 (주)
    foreign_net: int        # 외국인 순매수 합
    org_net: int            # 기관 순매수 합


@dataclass
class Signal:
    date: str               # YYYY-MM-DD (수집일)
    source: str             # SOURCE_* 중 하나
    stock_code: str
    stock_name: str
    score: float            # 소스 내부 점수 (0~1 정규화 권장)
    reason: str             # 사람이 읽는 근거 한 줄
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailyPick:
    """소스별 시그널을 합산한 '오늘의 관찰 픽'. 매매하지 않고 기록만 한다."""
    date: str
    stock_code: str
    stock_name: str
    combined_score: float
    sources: list           # 기여한 소스 이름 목록
    reasons: list           # 근거 문장 목록
    price_at_pick: Optional[float] = None   # 픽 시점 가격 (성과 추적용)

    def to_dict(self) -> dict:
        return asdict(self)
