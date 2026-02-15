"""
v6.0 동종 업체 분석기 (기본 disabled)

WaveLSFormer 논문 기반:
- DTW(Dynamic Time Warping) 기반 종목 유사도
- Granger Causality 기반 선행/후행 관계
- 산업 내 관련 종목 그룹 다중 입력

연산 비용이 높아 기본 disabled.
"""

import logging

logger = logging.getLogger(__name__)


class PeerAnalyzer:
    """DTW + Granger 동종 업체 분석"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def find_peers(self, ticker: str, data_dict: dict, top_n: int = 5) -> list[str]:
        """
        DTW 기반 유사 종목 탐색.

        현재 플레이스홀더 — 실제 구현 시 dtw-python 패키지 사용.
        """
        if not self.enabled:
            return []

        logger.debug("PeerAnalyzer 비활성화 — 빈 리스트 반환")
        return []

    def check_granger_lead(
        self, ticker: str, peer: str, data_dict: dict, max_lag: int = 5
    ) -> dict:
        """
        Granger Causality 테스트.

        현재 플레이스홀더 — 실제 구현 시 statsmodels 사용.
        """
        if not self.enabled:
            return {"lead": False, "lag": 0, "p_value": 1.0}

        return {"lead": False, "lag": 0, "p_value": 1.0}
