"""
v4.5 섹터 분류기

Adaptive Regime Sector Strategy 기반:
  - 15개 섹터로 종목 분류 (업종명 키워드 매칭)
  - 동일 섹터 최대 1종목 제한 (분산 투자)
  - 섹터별 비중 배분 (BEST 45% / RESERVE1 30% / RESERVE2 25%)
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SectorClassifier:
    """종목 섹터 분류 + 섹터 제한 관리"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        sc_cfg = config.get("sector_classification", {})
        self.enabled = sc_cfg.get("enabled", False)
        self.max_per_sector = sc_cfg.get("max_per_sector", 1)
        self.sectors = sc_cfg.get("sectors", {})
        self.allocation = sc_cfg.get("allocation", {})

        # 키워드 → 섹터명 매핑 (빠른 조회용)
        self._keyword_map: dict[str, str] = {}
        for sector_name, sector_info in self.sectors.items():
            for kw in sector_info.get("keywords", []):
                self._keyword_map[kw] = sector_name

        # 종목코드 → 섹터 캐시
        self._ticker_sector_cache: dict[str, str] = {}

    def classify(self, ticker: str, industry_name: str = "") -> str:
        """
        종목의 섹터를 판별.

        Args:
            ticker: 종목 코드
            industry_name: 업종명 (예: "반도체", "자동차부품")

        Returns:
            섹터명 (예: "반도체", "자동차", "기타")
        """
        if ticker in self._ticker_sector_cache:
            return self._ticker_sector_cache[ticker]

        sector = "기타"

        if industry_name:
            # 키워드 매칭 (가장 먼저 매칭되는 섹터)
            for kw, sector_name in self._keyword_map.items():
                if kw in industry_name:
                    sector = sector_name
                    break

        self._ticker_sector_cache[ticker] = sector
        return sector

    def set_sector(self, ticker: str, sector: str) -> None:
        """종목 섹터를 수동 설정 (캐시에 직접 입력)."""
        self._ticker_sector_cache[ticker] = sector

    def filter_by_sector_limit(
        self,
        signals: list[dict],
        held_sectors: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        섹터 제한에 따라 시그널 필터링.

        동일 섹터에서 max_per_sector 이상이면 추가 진입 차단.

        Args:
            signals: Zone Score 높은 순 정렬된 시그널 리스트
            held_sectors: 현재 보유 중인 {ticker: sector} 딕셔너리

        Returns:
            섹터 제한을 적용한 시그널 리스트
        """
        if not self.enabled:
            return signals

        # 현재 보유 종목의 섹터별 카운트
        sector_counts: dict[str, int] = {}
        if held_sectors:
            for _ticker, sector in held_sectors.items():
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        filtered = []
        for sig in signals:
            ticker = sig["ticker"]
            # 시그널에 industry가 있으면 사용, 없으면 캐시 조회
            industry = sig.get("industry", "")
            sector = self.classify(ticker, industry)
            sig["sector"] = sector

            current_count = sector_counts.get(sector, 0)
            if current_count >= self.max_per_sector:
                logger.debug(
                    f"  섹터 제한: {ticker} ({sector}) — "
                    f"이미 {current_count}종목 보유"
                )
                continue

            sector_counts[sector] = current_count + 1
            filtered.append(sig)

        return filtered

    def get_allocation_pct(self, rank: int) -> float:
        """순위에 따른 비중 배분.

        Args:
            rank: 0-based 순위 (0=BEST, 1=RESERVE1, 2=RESERVE2)
        """
        if rank == 0:
            return self.allocation.get("best", 0.45)
        elif rank == 1:
            return self.allocation.get("reserve1", 0.30)
        elif rank == 2:
            return self.allocation.get("reserve2", 0.25)
        return 0.0

    def get_sector_summary(self, positions: list[dict]) -> dict[str, list[str]]:
        """보유 포지션의 섹터별 요약.

        Returns:
            {섹터명: [종목코드, ...]}
        """
        summary: dict[str, list[str]] = {}
        for pos in positions:
            ticker = pos.get("ticker", "")
            sector = self._ticker_sector_cache.get(ticker, "기타")
            if sector not in summary:
                summary[sector] = []
            summary[sector].append(ticker)
        return summary
