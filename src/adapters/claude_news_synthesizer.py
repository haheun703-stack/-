"""
v6.0 Claude 뉴스 합성기 (기본 disabled)

RTTP 논문의 LLM 합성 신호 생성:
- 뉴스 텍스트 → Claude → 1/2/3차 수혜주 + 역수혜주 JSON
- 비용 관리: 일 100콜 제한
- 주의: LLM 환각 방지를 위한 종목 검증 필수

기본 enabled: false (Claude API 비용 고려)
"""

import logging

from src.entities.rttp_models import BeneficiarySignal

logger = logging.getLogger(__name__)


class ClaudeNewsSynthesizer:
    """Claude API를 이용한 뉴스 → 수혜주 합성"""

    def __init__(
        self,
        enabled: bool = False,
        model: str = "claude-sonnet-4-5-20250929",
        max_beneficiaries: int = 5,
        daily_call_limit: int = 100,
    ):
        self.enabled = enabled
        self.model = model
        self.max_beneficiaries = max_beneficiaries
        self.daily_call_limit = daily_call_limit
        self._daily_calls = 0

    async def synthesize_beneficiaries(
        self, news_text: str
    ) -> list[BeneficiarySignal]:
        """
        뉴스 → Claude → 1/2/3차 수혜주 + 역수혜주 JSON

        기본 disabled 상태. enabled=true 시에만 Claude API 호출.
        """
        if not self.enabled:
            logger.debug("Claude 합성기 비활성화 — 스킵")
            return []

        if self._daily_calls >= self.daily_call_limit:
            logger.warning("Claude 일일 호출 한도(%d) 도달", self.daily_call_limit)
            return []

        try:
            # API 호출 (향후 구현)
            # 현재는 플레이스홀더 — 실제 구현 시 anthropic SDK 사용
            self._daily_calls += 1
            logger.info("Claude 합성 호출 #%d (미구현 플레이스홀더)", self._daily_calls)
            return []
        except Exception as e:
            logger.error("Claude 합성 실패: %s", e)
            return []

    def reset_daily_counter(self) -> None:
        """일일 카운터 리셋 (스케줄러에서 매일 자정 호출)"""
        self._daily_calls = 0
