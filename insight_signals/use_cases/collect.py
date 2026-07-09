# -*- coding: utf-8 -*-
"""시그널 산출·합산 로직.

바깥(어댑터)에서 가져온 원자료를 받아 Signal / DailyPick으로 변환한다.
이 모듈은 entities만 import 한다 (클린 아키텍처 준수).
"""
from __future__ import annotations

from collections import defaultdict

from insight_signals.entities import (
    SOURCE_DART,
    SOURCE_FLOW,
    SOURCE_NEWS,
    DailyPick,
    Signal,
)


# ----------------------------------------------------------------------
# 1) 뉴스 키워드 -> Signal
# ----------------------------------------------------------------------
def news_signals(date: str, matched_articles) -> list:
    """종목별로 기사 수·키워드 다양성을 점수화 (0~1)."""
    per_stock = defaultdict(lambda: {"articles": [], "keywords": set(), "name": ""})
    for a in matched_articles:
        for code, name in a.matched_stocks:
            per_stock[code]["articles"].append(a)
            per_stock[code]["keywords"].update(a.matched_keywords)
            per_stock[code]["name"] = name

    out = []
    for code, d in per_stock.items():
        n_art = len(d["articles"])
        n_kw = len(d["keywords"])
        # 기사 1건=0.5, 2건=0.75, 3건 이상=~1.0 + 키워드 다양성 보너스
        score = min(1.0, 0.5 + 0.25 * (n_art - 1) + 0.1 * (n_kw - 1))
        titles = [a.title for a in d["articles"][:3]]
        out.append(
            Signal(
                date=date,
                source=SOURCE_NEWS,
                stock_code=code,
                stock_name=d["name"],
                score=round(score, 3),
                reason=f"키워드 {sorted(d['keywords'])} 뉴스 {n_art}건: " + " / ".join(titles),
                meta={"articles": [{"title": a.title, "link": a.link} for a in d["articles"]]},
            )
        )
    return out


# ----------------------------------------------------------------------
# 2) DART 임원 매수 -> Signal
# ----------------------------------------------------------------------
def dart_signals(date: str, filings) -> list:
    """종목별 임원 순매수 집계.

    점수 로직 (인터뷰: "임원들이 사면 항상 올라" — 단 방향은 매수여야 함):
      - 순매수 보고 건수와 매수 보고자 수가 많을수록 가점
      - 순매도가 우세하면 시그널 제외
    """
    per_stock = defaultdict(lambda: {"buy": [], "sell": [], "name": ""})
    for f in filings:
        if not f.stock_code:  # 비상장 제외
            continue
        d = per_stock[f.stock_code]
        d["name"] = f.corp_name
        (d["buy"] if f.change_qty > 0 else d["sell"]).append(f)

    out = []
    for code, d in per_stock.items():
        buys, sells = d["buy"], d["sell"]
        net_reports = len(buys) - len(sells)
        if net_reports <= 0 or not buys:
            continue
        buyers = {f.reporter for f in buys}
        # 매수 보고 1건=0.5, 매수자 2인 이상이면 강한 신호
        score = min(1.0, 0.4 + 0.2 * len(buys) + 0.15 * (len(buyers) - 1))
        sample = buys[0]
        out.append(
            Signal(
                date=date,
                source=SOURCE_DART,
                stock_code=code,
                stock_name=d["name"],
                score=round(score, 3),
                reason=(
                    f"임원·주요주주 매수 보고 {len(buys)}건 (보고자 {len(buyers)}명, "
                    f"예: {sample.reporter} {sample.position} +{sample.change_qty:,}주)"
                ),
                meta={
                    "buy_filings": len(buys),
                    "sell_filings": len(sells),
                    "buyers": sorted(buyers),
                    "rcept_nos": [f.rcept_no for f in buys],
                },
            )
        )
    return out


# ----------------------------------------------------------------------
# 3) 역발상 수급 -> Signal (후보 종목에만 적용하는 필터형 시그널)
# ----------------------------------------------------------------------
def flow_signal(date: str, stock_code: str, stock_name: str, snap) -> Signal | None:
    """기관+외인이 사고 개인이 파는 종목 = 가점(소수 의견 추종).

    반대로 개인만 사는 종목은 음수 점수(감점)로 기록해
    합산 때 '불나방 쏠림' 경고 역할을 한다.
    """
    if snap is None:
        return None
    smart = snap.foreign_net + snap.org_net   # '힘 있는 소수'
    crowd = snap.person_net                    # 다수(개인)
    if smart == 0 and crowd == 0:
        return None

    scale = max(abs(smart), abs(crowd), 1)
    raw = (smart - crowd) / (2.0 * scale)      # -1.0 ~ +1.0
    score = round(max(-1.0, min(1.0, raw)), 3)

    direction = "기관/외인 매집 + 개인 이탈" if score > 0 else "개인 쏠림 경고"
    return Signal(
        date=date,
        source=SOURCE_FLOW,
        stock_code=stock_code,
        stock_name=stock_name,
        score=score,
        reason=(
            f"{direction} — 최근 {snap.days}일 순매수(주): "
            f"외인 {snap.foreign_net:+,} / 기관 {snap.org_net:+,} / 개인 {snap.person_net:+,}"
        ),
        meta={
            "person_net": snap.person_net,
            "foreign_net": snap.foreign_net,
            "org_net": snap.org_net,
        },
    )


# ----------------------------------------------------------------------
# 합산 -> DailyPick
# ----------------------------------------------------------------------
def combine(date: str, signals, weights: dict, top_n: int = 5, min_score: float = 0.3) -> list:
    """소스별 가중 합산. 뉴스+DART 동시 포착이면 시너지 보너스."""
    # 종목코드 기준으로 묶는다 (소스마다 회사명 표기가 달라도 같은 종목으로 합산)
    per_stock = defaultdict(list)
    for s in signals:
        per_stock[s.stock_code].append(s)

    picks = []
    for code, sigs in per_stock.items():
        name = next((s.stock_name for s in sigs if s.stock_name), code)
        by_src = {s.source: s for s in sigs}
        total = 0.0
        for src, s in by_src.items():
            total += weights.get(src, 0.0) * s.score
        # 시너지: 서로 다른 확증 소스 2개 이상 (양수 점수만)
        positive_srcs = [s.source for s in sigs if s.score > 0]
        if len(set(positive_srcs)) >= 2:
            total *= 1.25
        if total < min_score:
            continue
        picks.append(
            DailyPick(
                date=date,
                stock_code=code,
                stock_name=name,
                combined_score=round(total, 4),
                sources=sorted({s.source for s in sigs}),
                reasons=[s.reason for s in sigs],
            )
        )

    picks.sort(key=lambda p: p.combined_score, reverse=True)
    return picks[:top_n]


def candidate_stocks(signals) -> list:
    """수급 필터를 적용할 후보 (뉴스/DART가 잡은 종목의 합집합)."""
    seen, out = set(), []
    for s in signals:
        if s.stock_code and s.stock_code not in seen:
            seen.add(s.stock_code)
            out.append((s.stock_code, s.stock_name))
    return out
