"""공매도 체제 관리자 (Short-Selling Regime Manager)

캘린더 기반 공매도 체제(active/banned/reopened) 판별 + 프로파일 파라미터 제공.
backtest_engine, scan_tomorrow_picks, 라이브 파이프라인에서 공통 사용.

config/settings.yaml 의존:
  - use_short_selling_filter: true/false (마스터 스위치)
  - short_selling_calendar: [{start, end, status}, ...]
  - regime_profiles: {short_selling_active, short_selling_banned, short_selling_reopened}
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class ShortRegimeManager:
    """공매도 체제 판별 + 프로파일 제공."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = self._load_config()

        self.enabled = config.get("use_short_selling_filter", False)
        self._calendar: list[tuple[date, date, str]] = []
        self._profiles: dict[str, dict] = {}

        if self.enabled:
            self._calendar = self._parse_calendar(
                config.get("short_selling_calendar", [])
            )
            self._profiles = config.get("regime_profiles", {})

        # 캐시
        self._cache_date: date | None = None
        self._cache_status: str = ""
        self._cache_profile: dict = {}

    @staticmethod
    def _load_config() -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _parse_calendar(cal_list: list) -> list[tuple[date, date, str]]:
        parsed = []
        for entry in cal_list:
            start = datetime.strptime(str(entry["start"]), "%Y-%m-%d").date()
            end = datetime.strptime(str(entry["end"]), "%Y-%m-%d").date()
            parsed.append((start, end, entry["status"]))
        return parsed

    def get_status(self, target: date | str | None = None) -> str:
        """날짜 기준 공매도 상태 반환: 'active', 'banned', 'reopened'.

        비활성(use_short_selling_filter=false) → 항상 'banned' (기존 동작 유지).
        """
        if not self.enabled:
            return "banned"

        if isinstance(target, str):
            target = datetime.strptime(target, "%Y-%m-%d").date()
        if target is None:
            target = date.today()

        if target == self._cache_date:
            return self._cache_status

        status = "active"  # 캘린더에 없으면 기본 active
        for start, end, st in self._calendar:
            if start <= target <= end:
                status = st
                break

        self._cache_date = target
        self._cache_status = status
        return status

    def get_profile(self, target: date | str | None = None) -> dict:
        """현재 공매도 체제에 해당하는 프로파일 파라미터 반환.

        비활성 시 빈 dict (기존 동작 그대로).
        """
        if not self.enabled:
            return {}

        status = self.get_status(target)
        profile_key = f"short_selling_{status}"
        return dict(self._profiles.get(profile_key, {}))

    def is_short_active(self, target: date | str | None = None) -> bool:
        """공매도가 활발한 상태인지 (active 또는 reopened)."""
        return self.get_status(target) in ("active", "reopened")

    def get_sa_floor(self, target: date | str | None = None) -> float:
        """현재 체제의 SA Floor 값."""
        profile = self.get_profile(target)
        return profile.get("sa_floor", 0.55)

    def get_position_scale(self, target: date | str | None = None) -> float:
        """포지션 크기 승수 (position_scale_mult)."""
        profile = self.get_profile(target)
        return profile.get("position_scale_mult", 1.0)

    def get_max_positions_scale(self, target: date | str | None = None) -> float:
        """최대 보유 종목수 스케일."""
        profile = self.get_profile(target)
        return profile.get("max_positions_scale", 1.0)

    def get_stop_loss_scale(self, target: date | str | None = None) -> float:
        """손절 스케일."""
        profile = self.get_profile(target)
        return profile.get("stop_loss_scale", 1.0)

    def get_min_rr_ratio(self, target: date | str | None = None) -> float:
        """최소 손익비."""
        profile = self.get_profile(target)
        return profile.get("min_rr_ratio", 1.5)

    def summary(self, target: date | str | None = None) -> dict:
        """현재 체제 요약 (로깅/대시보드용)."""
        status = self.get_status(target)
        profile = self.get_profile(target)
        return {
            "enabled": self.enabled,
            "status": status,
            "is_short_active": status in ("active", "reopened"),
            "profile": profile,
            "label": {
                "active": "공매도 활발",
                "banned": "공매도 금지",
                "reopened": "공매도 재개",
            }.get(status, status),
        }


# ── 싱글턴 인스턴스 (모듈 레벨 캐싱) ──
_instance: ShortRegimeManager | None = None


def get_short_regime_manager() -> ShortRegimeManager:
    """싱글턴 ShortRegimeManager 반환."""
    global _instance
    if _instance is None:
        _instance = ShortRegimeManager()
    return _instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    mgr = ShortRegimeManager()
    s = mgr.summary()
    print(f"공매도 체제: {s['label']} (enabled={s['enabled']})")
    if s["profile"]:
        print(f"  프로파일: {s['profile']}")
    else:
        print("  프로파일: 비활성 (use_short_selling_filter=false)")

    # 히스토리 테스트
    test_dates = ["2020-01-15", "2020-06-15", "2023-12-01", "2025-04-15", "2026-03-27"]
    for d in test_dates:
        st = mgr.get_status(d)
        prof = mgr.get_profile(d)
        active = mgr.is_short_active(d)
        print(f"  {d}: {st} (active={active}) sa_floor={prof.get('sa_floor', '-')}")
