"""
v3.1 L-1 News Gate 유스케이스

뉴스 등급(A/B/C)을 판정하고, v3.0 파이프라인 파라미터를 조정.
핵심 원칙: 뉴스는 v3.0을 override 불가, parameter tuning만 가능.

Uses:
  - NewsClassifier (등급 분류)
  - EventPositionManager (진입 제약 검증)
  - NewsSearchPort (외부 뉴스 API — 어댑터가 구현)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.entities.news_models import (
    EventDrivenAction,
    NewsGateResult,
    NewsGrade,
    NewsItem,
)
from src.news_classifier import NewsClassifier
from src.event_position import EventPositionManager

logger = logging.getLogger(__name__)

_NEWS_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news_log.json"


class NewsGateUseCase:
    """L-1 News Gate — 뉴스 기반 파라미터 조정 유스케이스"""

    def __init__(self, config_path: str | Path | None = None):
        self.classifier = NewsClassifier(config_path)
        self.position_mgr = EventPositionManager(config_path)
        self._active_event_positions: list[str] = []  # 활성 이벤트 포지션 ticker 목록
        self._watchlist: dict[str, dict] = {}          # B급 관찰 리스트

    # ──────────────────────────────────────────
    # 메인 평가
    # ──────────────────────────────────────────

    def evaluate(
        self,
        ticker: str,
        news_grade: str | None = None,
        news_text: str = "",
        news_items: list[NewsItem] | None = None,
    ) -> NewsGateResult:
        """
        뉴스 등급 판정 + 파라미터 오버라이드 생성.

        Args:
            ticker: 종목코드
            news_grade: 수동 등급 지정 ("A"/"B"/"C") — CLI용
            news_text: 뉴스 요약 텍스트 — CLI용
            news_items: 자동 분류용 뉴스 항목 리스트

        Returns:
            NewsGateResult (grade, action, param_overrides 등)
        """
        # 수동 등급 지정 (CLI: --news-grade A:"뉴스내용")
        if news_grade:
            result = self.classifier.classify_from_grade_string(
                ticker, news_grade, news_text
            )
        # 자동 분류
        elif news_items:
            result = self.classifier.classify(ticker, news_items)
        # 뉴스 없음
        else:
            return NewsGateResult(
                grade=NewsGrade.C,
                action=EventDrivenAction.IGNORE,
                ticker=ticker,
                reason="뉴스 입력 없음 — v3.0 기본 파이프라인 작동",
            )

        # B급 관찰 리스트 관리
        if result.grade == NewsGrade.B:
            self._add_to_watchlist(ticker, result)

        # 뉴스 이력 기록
        self._log_news(result)

        return result

    # ──────────────────────────────────────────
    # 진입 제약 검증
    # ──────────────────────────────────────────

    def check_entry_constraints(
        self,
        result: NewsGateResult,
        current_price: float,
        prev_close: float,
        rsi: float,
        rr_ratio: float,
    ) -> tuple[bool, str]:
        """
        A급 뉴스 종목의 진입 가능 여부 검증.

        Returns:
            (can_enter, reason)
        """
        return self.position_mgr.check_entry(
            news_grade=result.grade,
            current_price=current_price,
            prev_close=prev_close,
            rsi=rsi,
            rr_ratio=rr_ratio,
            active_event_count=len(self._active_event_positions),
        )

    # ──────────────────────────────────────────
    # 포지션 관리
    # ──────────────────────────────────────────

    def register_event_position(self, ticker: str) -> None:
        """이벤트 드리븐 포지션 등록"""
        if ticker not in self._active_event_positions:
            self._active_event_positions.append(ticker)

    def release_event_position(self, ticker: str) -> None:
        """이벤트 드리븐 포지션 해제"""
        if ticker in self._active_event_positions:
            self._active_event_positions.remove(ticker)

    @property
    def active_event_count(self) -> int:
        return len(self._active_event_positions)

    # ──────────────────────────────────────────
    # 관찰 리스트
    # ──────────────────────────────────────────

    def _add_to_watchlist(self, ticker: str, result: NewsGateResult) -> None:
        self._watchlist[ticker] = {
            "added": datetime.now().isoformat(),
            "monitor_days": result.watchlist_days,
            "reason": result.reason,
        }

    def get_watchlist(self) -> dict[str, dict]:
        return dict(self._watchlist)

    # ──────────────────────────────────────────
    # 뉴스 이력 기록
    # ──────────────────────────────────────────

    @staticmethod
    def _log_news(result: NewsGateResult) -> None:
        """data/news_log.json에 이력 추가"""
        try:
            _NEWS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            logs = []
            if _NEWS_LOG_PATH.exists():
                logs = json.loads(_NEWS_LOG_PATH.read_text(encoding="utf-8"))

            logs.append({
                "timestamp": result.timestamp or datetime.now().isoformat(),
                "ticker": result.ticker,
                "grade": result.grade.value,
                "action": result.action.value,
                "reason": result.reason,
                "news_count": len(result.news_items),
            })

            # 최근 500건만 유지
            if len(logs) > 500:
                logs = logs[-500:]

            _NEWS_LOG_PATH.write_text(
                json.dumps(logs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("뉴스 로그 기록 실패: %s", e)
