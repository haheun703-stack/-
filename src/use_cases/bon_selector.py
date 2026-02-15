"""
v5.0 BoN Selector — Best-of-N 프로파일 자동 선택기

Sci-CoE 논문의 BoN(Best-of-N) 개념 적용:
  - 7개 가중치 프로파일 중 합의 점수가 가장 높은 프로파일 자동 선택
  - 캐시: cache_ttl_bars봉마다 재선택 (성능 최적화)

클린 아키텍처: entities만 import, 외부 엔진은 파라미터로 주입.
"""

import logging
from typing import Optional

from src.entities.consensus_models import ConsensusResult, LayerVote

logger = logging.getLogger(__name__)

# GeometricQuantEngine의 7개 프로파일
PROFILES = [
    "default", "reversal", "breakout", "capitulation",
    "bull", "bear", "sideways",
]


class BoNSelector:
    """Best-of-N 프로파일 자동 선택기"""

    def __init__(self, cache_ttl_bars: int = 20):
        self.cache_ttl_bars = cache_ttl_bars
        self._cache: dict = {}  # ticker → {"profile": str, "bar_count": int, "score": float}

    def select_best(
        self,
        df,
        ticker: str,
        geo_engine,
        verifier,
        base_votes: list,
    ) -> tuple:
        """
        7개 프로파일 중 합의 최고 프로파일 선택.

        Args:
            df: OHLCV DataFrame
            ticker: 종목코드
            geo_engine: GeometricQuantEngine 인스턴스
            verifier: ConsensusVerifier 인스턴스
            base_votes: 기본 레이어 투표 리스트 (L7 제외)

        Returns:
            (best_profile_name, best_geo_result, best_consensus_result)
        """
        # 캐시 확인
        cached = self._cache.get(ticker)
        if cached and cached["bar_count"] < self.cache_ttl_bars:
            cached["bar_count"] += 1
            return cached["profile"], cached["geo_result"], cached["consensus"]

        best_profile = "default"
        best_score = -1.0
        best_geo = None
        best_consensus = None

        for profile in PROFILES:
            try:
                geo_result = geo_engine.run_with_profile(df, ticker, profile)
            except Exception as e:
                logger.debug("BoN %s/%s 실패: %s", ticker, profile, e)
                continue

            # L7 투표 생성
            geo_conf = geo_result.get("geo_confidence", 0)
            geo_buy = geo_result.get("geo_confirms_buy", False)
            geo_vote = LayerVote(
                "L7_geometric",
                geo_buy or (geo_conf >= 30),
                geo_conf / 100.0 if geo_conf > 0 else 0.0,
            )

            # base_votes + L7 투표로 합의 판정
            all_votes = list(base_votes) + [geo_vote]
            geo_indicators = geo_result.get("geo_indicators", {})
            consensus = verifier.verify(all_votes, geo_indicators)

            if consensus.geometric_reward > best_score:
                best_score = consensus.geometric_reward
                best_profile = profile
                best_geo = geo_result
                best_consensus = consensus

        # 캐시 저장
        self._cache[ticker] = {
            "profile": best_profile,
            "bar_count": 0,
            "score": best_score,
            "geo_result": best_geo,
            "consensus": best_consensus,
        }

        logger.debug(
            "BoN %s: %s (score=%.3f)",
            ticker, best_profile, best_score,
        )

        return best_profile, best_geo, best_consensus

    def invalidate_cache(self, ticker: Optional[str] = None) -> None:
        """캐시 무효화"""
        if ticker:
            self._cache.pop(ticker, None)
        else:
            self._cache.clear()
