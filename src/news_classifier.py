"""
v3.2 뉴스 분류기 — A/B/C 등급 판정 + 파라미터 오버라이드 + 스코어 부스트

v3.1 → v3.2 변경:
  - 뉴스 스코어 부스트 계산 (Zone Score 가산)
  - 살아있는 이슈(LivingIssue) 점수 반영
  - 실적 예상(EarningsEstimate) 점수 반영
  - Trend Continuation과 연동

엔티티(news_models)에만 의존 — 외부 API 없음.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.entities.news_models import (
    EarningsEstimate,
    EventDrivenAction,
    LivingIssue,
    NewsGateResult,
    NewsGrade,
    NewsItem,
)

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_news_gate_config(config_path: Path | str | None = None) -> dict:
    """news_gate 섹션 로드"""
    path = Path(config_path) if config_path else _CFG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("news_gate", {})


class NewsClassifier:
    """뉴스 등급 분류 + 파라미터 오버라이드 생성"""

    def __init__(self, config_path: Path | str | None = None):
        self.cfg = _load_news_gate_config(config_path)

    # ──────────────────────────────────────────
    # 등급 분류
    # ──────────────────────────────────────────

    def classify(self, ticker: str, news_items: list[NewsItem]) -> NewsGateResult:
        """
        뉴스 항목 리스트를 분석해서 등급 판정.

        A급 조건 (4/4 모두 충족):
          1. DART/거래소 공식 공시 확인 (is_confirmed)
          2. 구체적 금액/수치 언급 (has_specific_amount)
          3. 확정적 표현 사용 (has_definitive_language)
          4. 복수 언론 교차 확인 (cross_verified)

        B급 조건 (4가지 중 2개 이상):
          동일 4가지 체크리스트

        C급: 나머지
        """
        if not news_items:
            return NewsGateResult(
                grade=NewsGrade.C,
                action=EventDrivenAction.IGNORE,
                ticker=ticker,
                reason="뉴스 항목 없음",
            )

        # 가장 높은 등급의 뉴스를 기준으로 판정
        best_grade = NewsGrade.C
        best_score = 0
        best_item = news_items[0]

        for item in news_items:
            score = sum([
                item.is_confirmed,
                item.has_specific_amount,
                item.has_definitive_language,
                item.cross_verified,
            ])
            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= 4:
            best_grade = NewsGrade.A
        elif best_score >= 2:
            best_grade = NewsGrade.B

        return self._build_result(ticker, best_grade, news_items, best_item)

    def classify_from_grade_string(
        self, ticker: str, grade_str: str, news_text: str = ""
    ) -> NewsGateResult:
        """
        CLI에서 수동 등급 지정 시 사용.
        예: --news-grade A:"SK이터닉스 매각 확정"
        """
        grade_map = {"A": NewsGrade.A, "B": NewsGrade.B, "C": NewsGrade.C}
        grade = grade_map.get(grade_str.upper(), NewsGrade.C)

        items = []
        if news_text:
            items.append(NewsItem(
                title=news_text,
                summary=news_text,
                is_confirmed=(grade == NewsGrade.A),
                has_specific_amount=(grade == NewsGrade.A),
                has_definitive_language=(grade == NewsGrade.A),
                cross_verified=(grade == NewsGrade.A),
            ))

        return self._build_result(ticker, grade, items, items[0] if items else None)

    # ──────────────────────────────────────────
    # v3.2: Grok deep_analysis 결과 통합 분류
    # ──────────────────────────────────────────

    def classify_deep(self, ticker: str, grok_result: dict) -> NewsGateResult:
        """
        Grok search_deep_analysis() 결과를 통합 분류.
        최신 뉴스 + 살아있는 이슈 + 실적 예상을 모두 반영.
        """
        if not grok_result:
            return NewsGateResult(
                grade=NewsGrade.C,
                action=EventDrivenAction.IGNORE,
                ticker=ticker,
                reason="Grok 분석 결과 없음",
            )

        # 1. 최신 뉴스 → NewsItem 변환
        news_items = []
        for n in grok_result.get("latest_news", []):
            news_items.append(NewsItem(
                title=n.get("title", ""),
                summary=n.get("summary", ""),
                category=n.get("category", "theme"),
                source=n.get("source", ""),
                date=n.get("date", ""),
                impact_score=int(n.get("impact_score", 0)),
                sentiment=n.get("sentiment", "neutral"),
                is_confirmed=bool(n.get("is_confirmed", False)),
                has_specific_amount=bool(n.get("has_specific_amount", False)),
                has_definitive_language=bool(n.get("has_definitive_language", False)),
                cross_verified=bool(n.get("cross_verified", False)),
            ))

        # 2. 살아있는 이슈 → LivingIssue 변환
        living_issues = []
        for li in grok_result.get("living_issues", []):
            living_issues.append(LivingIssue(
                title=li.get("title", ""),
                category=li.get("category", "theme"),
                start_date=li.get("start_date", ""),
                status=li.get("status", "active"),
                impact_score=int(li.get("impact_score", 0)),
                sentiment=li.get("sentiment", "neutral"),
                description=li.get("description", ""),
                expected_resolution=li.get("expected_resolution", ""),
                source_count=int(li.get("source_count", 0)),
            ))

        # 3. 실적 예상 → EarningsEstimate 변환
        ee_raw = grok_result.get("earnings_estimate", {})
        earnings = EarningsEstimate(
            next_earnings_date=ee_raw.get("next_earnings_date", ""),
            days_until_earnings=int(ee_raw.get("days_until_earnings", -1)),
            consensus_revenue=float(ee_raw.get("consensus_revenue_억", 0)),
            consensus_op=float(ee_raw.get("consensus_op_억", 0)),
            consensus_eps=float(ee_raw.get("consensus_eps_원", 0)),
            surprise_direction=ee_raw.get("surprise_direction", "neutral"),
            yoy_growth_pct=float(ee_raw.get("yoy_growth_pct", 0)),
            analyst_count=int(ee_raw.get("analyst_count", 0)),
            description=ee_raw.get("description", ""),
        )

        # 4. 기존 A/B/C 등급 판정
        base_result = self.classify(ticker, news_items)

        # 5. 스코어 부스트 계산
        boost = self._calc_score_boost(
            base_result.grade, news_items, living_issues, earnings
        )

        # 6. 결과 조합
        base_result.score_boost = boost
        base_result.living_issues = living_issues
        base_result.earnings_estimate = earnings

        return base_result

    def _calc_score_boost(
        self,
        grade: NewsGrade,
        news_items: list[NewsItem],
        living_issues: list[LivingIssue],
        earnings: EarningsEstimate,
    ) -> float:
        """
        뉴스 스코어 부스트 계산.

        배점 체계 (최대 0.30 = Zone Score의 30% 가산):
        ┌─────────────────────────────────┬────────┐
        │ 요인                            │ 부스트  │
        ├─────────────────────────────────┼────────┤
        │ A급 뉴스 (확정 공시)              │ +0.15  │
        │ B급 뉴스 (신뢰 루머)              │ +0.08  │
        │ 살아있는 이슈 (개당)              │ +0.03  │
        │  - 고영향(8+) 이슈               │ +0.05  │
        │ 실적 서프라이즈 예상 (beat)        │ +0.08  │
        │ 실적 전 매집 구간 (30일 이내)      │ +0.05  │
        │ 긍정 sentiment 다수              │ +0.03  │
        ├─────────────────────────────────┼────────┤
        │ 최대 합계                        │ 0.30   │
        └─────────────────────────────────┴────────┘
        """
        boost_cfg = self.cfg.get("score_boost", {})
        boost = 0.0

        # 뉴스 등급 기본 부스트
        if grade == NewsGrade.A:
            boost += boost_cfg.get("grade_a", 0.15)
        elif grade == NewsGrade.B:
            boost += boost_cfg.get("grade_b", 0.08)

        # 살아있는 이슈 부스트
        for li in living_issues:
            if li.status != "active" or li.sentiment == "negative":
                continue
            if li.impact_score >= 8:
                boost += boost_cfg.get("living_issue_high", 0.05)
            elif li.impact_score >= 5:
                boost += boost_cfg.get("living_issue_normal", 0.03)

        # 실적 예상 부스트
        if earnings.surprise_direction == "beat":
            boost += boost_cfg.get("earnings_beat", 0.08)
        elif earnings.surprise_direction == "in_line" and earnings.yoy_growth_pct > 20:
            boost += boost_cfg.get("earnings_growth", 0.05)

        # 실적 전 매집 구간 (30일 이내)
        if 0 < earnings.days_until_earnings <= 30:
            boost += boost_cfg.get("pre_earnings_window", 0.05)

        # 전반적 긍정 뉴스가 많으면 보너스
        positive_count = sum(
            1 for n in news_items if n.sentiment == "positive"
        )
        if positive_count >= 3:
            boost += boost_cfg.get("positive_cluster", 0.03)

        max_boost = boost_cfg.get("max_boost", 0.30)
        return round(min(boost, max_boost), 3)

    def apply_rttp_source_boost(
        self,
        base_boost: float,
        source_avg: float,
    ) -> float:
        """
        v6.0 RTTP 소스 권위 가중 부스트 추가.

        source_avg >= 0.9 (DART급) → +0.05
        source_avg >= 0.7 (증권사급) → +0.03
        """
        extra = 0.0
        if source_avg >= 0.9:
            extra = 0.05
        elif source_avg >= 0.7:
            extra = 0.03

        max_boost = self.cfg.get("score_boost", {}).get("max_boost", 0.30)
        return round(min(base_boost + extra, max_boost), 3)

    # ──────────────────────────────────────────
    # 파라미터 오버라이드
    # ──────────────────────────────────────────

    def get_param_overrides(self, grade: NewsGrade) -> dict:
        """등급에 따른 v3.0 파라미터 조정값 반환"""
        if grade == NewsGrade.A:
            ga = self.cfg.get("grade_a", {})
            return {
                "rr_min": ga.get("rr_min", 1.2),
                "rsi_entry_max": ga.get("rsi_entry_max", 65),
                "position_size_pct": ga.get("position_size_pct", 3),
                "position_size_pct_small_cap": ga.get("position_size_pct_small_cap", 2),
                "max_gap_up_pct": ga.get("max_gap_up_pct", 15),
                "max_hold_days": ga.get("max_hold_days", 3),
                "settlement_strength_floor": ga.get("settlement_strength_floor", 95),
                "max_concurrent_positions": ga.get("max_concurrent_positions", 2),
            }
        # B/C 등급은 파라미터 변경 없음
        return {}

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _build_result(
        self,
        ticker: str,
        grade: NewsGrade,
        items: list[NewsItem],
        best_item: NewsItem | None,
    ) -> NewsGateResult:
        overrides = self.get_param_overrides(grade)

        if grade == NewsGrade.A:
            action = EventDrivenAction.ENTER
            reason = "A급 확정 공시 — 파라미터 조정 후 즉시 필터"
        elif grade == NewsGrade.B:
            action = EventDrivenAction.WATCHLIST
            gb = self.cfg.get("grade_b", {})
            reason = f"B급 루머 — {gb.get('monitor_days', 14)}일 관찰 리스트"
        else:
            action = EventDrivenAction.IGNORE
            reason = "C급 추측/테마 — 무시"

        from datetime import datetime

        return NewsGateResult(
            grade=grade,
            action=action,
            ticker=ticker,
            news_items=items,
            param_overrides=overrides,
            watchlist_days=self.cfg.get("grade_b", {}).get("monitor_days", 14) if grade == NewsGrade.B else 0,
            timestamp=datetime.now().isoformat(),
            reason=reason,
        )
