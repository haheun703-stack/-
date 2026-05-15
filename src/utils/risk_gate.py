"""정보봇 위험감지 SDK 래퍼 (P0-7 통합).

사용:
    from src.utils.risk_gate import get_risk_gate
    rg = get_risk_gate()
    multiplier = rg.get_position_multiplier()  # 0.2~1.0
    if rg.should_block_new_entry():  # CRISIS 차단
        return

데이터 출처: Supabase macro_risk_daily (정보봇 매일 16:49 갱신)
원본 SDK: src/utils/risk_gate_client.py (5/16 정보봇 → 퀀트봇)
가이드: jgis/docs/[정보봇 → 퀀트봇] 위험감지시스템 통합 가이드.md
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.utils.risk_gate_client import RiskGateClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Singleton (모듈 import 시 1회 초기화)
_INSTANCE: Optional[RiskGateClient] = None


def get_risk_gate() -> Optional[RiskGateClient]:
    """싱글톤 RiskGateClient 반환. Supabase 환경변수 누락 시 None.

    Fail-safe: 호출 측에서 None 체크 후 NORMAL 가정.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        logger.warning("[risk_gate] SUPABASE_URL/KEY 없음 → 위험감지 비활성 (NORMAL fallback)")
        return None

    try:
        _INSTANCE = RiskGateClient(
            supabase_url=url,
            supabase_key=key,
            bot_name="quant",
            cache_minutes=10,
        )
        logger.info("[risk_gate] 초기화 OK (bot_name=quant)")
        return _INSTANCE
    except Exception as e:
        logger.error(f"[risk_gate] 초기화 실패: {e}")
        return None


def get_position_multiplier_safe() -> float:
    """매수금액 곱하기 계수. SDK 실패 시 1.0 (NORMAL)."""
    rg = get_risk_gate()
    if rg is None:
        return 1.0
    try:
        return rg.get_position_multiplier()
    except Exception as e:
        logger.warning(f"[risk_gate] multiplier 조회 실패 → 1.0: {e}")
        return 1.0


def should_block_new_entry_safe() -> bool:
    """신규 진입 차단 여부. SDK 실패 시 False (정상 진입)."""
    rg = get_risk_gate()
    if rg is None:
        return False
    try:
        return rg.should_block_new_entry()
    except Exception:
        return False


def get_risk_status_safe() -> dict:
    """현재 위험 상태 전체 (로그/알림용). 실패 시 빈 dict."""
    rg = get_risk_gate()
    if rg is None:
        return {}
    try:
        return rg.get_full_status() or {}
    except Exception as e:
        logger.warning(f"[risk_gate] status 조회 실패: {e}")
        return {}


def get_exhaustion_threshold() -> Optional[float]:
    """현재 등급별 외인소진율 차단 임계값 (높을수록 위험).

    정상: None (차단 없음)
    주의: 80%+
    경고: 60%+
    위험: 50%+
    위기: 30%+
    """
    rg = get_risk_gate()
    if rg is None:
        return None
    try:
        level = rg.get_current_level()
        thresholds = {
            "NORMAL": None,
            "CAUTION": 80.0,
            "WARNING": 60.0,
            "DANGER": 50.0,
            "CRISIS": 30.0,
        }
        return thresholds.get(level)
    except Exception:
        return None
