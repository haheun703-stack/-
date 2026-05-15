# -*- coding: utf-8 -*-
"""
정보봇 위험감지 SDK Wrapper for quantum-master (퀀트봇)

목적:
    quantum-master 매수 승인 단계(TradeApprovalGateway.request_buy_approval)에서
    정보봇이 산출한 한국시장 위험점수(0~100) + MSCI 차단목록을 자동 조회하여
    텔레그램 승인 메시지에 첨부 + CRISIS 구간 자동 거부 옵션 제공.

작동 원리:
    1. /home/ubuntu/jgis 의 정보봇 SDK를 sys.path로 import
    2. Supabase macro_risk_daily / msci_blacklist 조회 (10분 캐시)
    3. 매수 승인 메시지에 위험 정보 자동 첨부
    4. CRISIS 또는 MSCI차단+DANGER 구간: 자동 거부 (옵션)

설치 (quantum-master 측):
    파일을 src/adapters/risk_gate_helper.py 로 배치
    SUPABASE_URL, SUPABASE_KEY 환경변수 필요 (.env 에 이미 있음)

사용 (trade_approval.py 수정):
    from src.adapters.risk_gate_helper import get_risk_info_for_buy, should_auto_reject

    def request_buy_approval(...):
        # 자동 거부 체크
        reject, reason = should_auto_reject(ticker)
        if reject:
            logger.warning(f"매수 자동 거부: {reason}")
            return False

        # 메시지 풍부화
        text = ... 기존 메시지 ...
        risk = get_risk_info_for_buy(ticker)
        if risk.get("available"):
            text += risk["message"]
        ...

신뢰 분리:
    정보봇 SDK 미가용/오류 시 graceful fallback (NORMAL 처리)
    운영 중단 절대 없음.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# 정보봇 SDK를 importlib으로 명시적 경로 로드
# (quantum-master의 src/ 와 정보봇의 src/ 패키지명 충돌 회피)
_JGIS_SDK_PATH = "/home/ubuntu/jgis/src/infrastructure/adapters/risk_gate_client.py"
RiskGateClient = None  # type: ignore
_SDK_AVAILABLE = False

try:
    if os.path.exists(_JGIS_SDK_PATH):
        _spec = importlib.util.spec_from_file_location(
            "jgis_risk_gate_client", _JGIS_SDK_PATH
        )
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        RiskGateClient = _module.RiskGateClient
        _SDK_AVAILABLE = True
    else:
        logger.warning("[risk_gate_helper] 정보봇 SDK 파일 없음: %s", _JGIS_SDK_PATH)
except Exception as e:
    logger.warning("[risk_gate_helper] 정보봇 SDK 로드 실패: %s — 위험감지 비활성", e)


_client_singleton: Optional["RiskGateClient"] = None


def _get_client() -> Optional["RiskGateClient"]:
    """RiskGateClient 싱글톤 — 캐시 활용 (10분)"""
    global _client_singleton
    if not _SDK_AVAILABLE:
        return None
    if _client_singleton is not None:
        return _client_singleton
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        logger.warning("[risk_gate_helper] SUPABASE_URL/KEY 환경변수 없음 — 위험감지 비활성")
        return None
    try:
        _client_singleton = RiskGateClient(url, key, bot_name="quant", cache_minutes=10)
        return _client_singleton
    except Exception as e:
        logger.warning("[risk_gate_helper] 초기화 실패: %s", e)
        return None


def get_risk_info_for_buy(ticker: str) -> dict:
    """매수 직전 호출 — 위험 정보 종합 + 텔레그램 메시지 첨부 텍스트

    Args:
        ticker: 6자리 종목 코드

    Returns:
        {
          'available': bool,         # 정보봇 SDK 가용 여부
          'level': str,              # NORMAL/CAUTION/WARNING/DANGER/CRISIS
          'level_kr': str,           # 정상/주의/경고/위험/위기
          'score': float,            # 0~100
          'multiplier': float,       # 매수금액 곱하기 계수 (0.2~1.0)
          'block_new_entry': bool,   # 신규 매수 차단 여부 (CRISIS만)
          'msci_blocked': bool,      # MSCI 차단 종목 여부
          'key_signals': list,       # 핵심 위험 시그널 (텍스트)
          'recommendation': str,     # 권장 행동
          'message': str,            # 텔레그램 메시지 첨부용 (이미 포맷됨)
        }
    """
    client = _get_client()
    if client is None:
        return {"available": False, "message": ""}

    try:
        level = client.get_current_level()
        level_kr = client.get_current_level_kr()
        multiplier = client.get_position_multiplier()
        block = client.should_block_new_entry()
        msci_blocked = client.is_msci_blacklisted(ticker)
        signals = client.get_key_signals()
        action = client.get_recommended_action()
        status = client.get_full_status()
        score = status.get("total_score", 0) if status else 0

        emoji = {
            "NORMAL": "🟢", "CAUTION": "🟡", "WARNING": "🟠",
            "DANGER": "🔴", "CRISIS": "⚠️",
        }.get(level, "🟢")

        lines = []
        lines.append("")
        lines.append("━" * 20)
        lines.append(f"  {emoji} 정보봇 위험점수: {score}점 ({level_kr})")
        if msci_blocked:
            lines.append(f"  🚫 MSCI 차단 종목 — 매수 강력 비권장")
        if block:
            lines.append(f"  🚨 신규 진입 자동 차단 권장")
        elif multiplier < 1.0:
            lines.append(f"  ⚠️ 매수금액 ×{multiplier} 축소 권장")

        if signals:
            lines.append(f"  🚨 핵심 시그널 ({len(signals)}건):")
            for s in signals[:4]:
                lines.append(f"    • {s}")
            if len(signals) > 4:
                lines.append(f"    ... 외 {len(signals) - 4}건")

        lines.append(f"  👉 {action}")
        lines.append("")  # 메시지 끝 줄바꿈 (다음 콘텐츠와 분리)

        return {
            "available": True,
            "level": level,
            "level_kr": level_kr,
            "score": score,
            "multiplier": multiplier,
            "block_new_entry": block,
            "msci_blocked": msci_blocked,
            "key_signals": signals,
            "recommendation": action,
            "message": "\n".join(lines),
        }
    except Exception as e:
        logger.warning("[risk_gate_helper] 조회 실패: %s", e)
        return {"available": False, "message": ""}


def should_auto_reject(ticker: str) -> tuple[bool, str]:
    """매수 자동 거부 여부 (보수적 안전장치)

    설계 원칙 (P0-7 결정):
      - 정보봇은 참고 의견(Reference Opinion). 자동 거부는 최소화.
      - 사용자(텔레그램 승인 단계)가 정보 보고 결정권 보유.
      - 단, 사용자가 직접 등록한 MSCI 차단 종목만 자동 거부
        (manual_official 신뢰도 100% — false positive 위험 없음)

    거부 조건 (단 1가지):
      - MSCI 차단목록(msci_blacklist)에 등록된 종목
        (자동 추정 auto_estimated + 사용자 수동 manual_official)

    Note:
      - CRISIS 자동 거부 제외 (정보봇 산출 오류 시 false positive 우려)
      - CRISIS는 텔레그램 메시지에 강조 표시되어 사용자가 결정

    Args:
        ticker: 종목 코드

    Returns:
        (reject_or_not, reason)
    """
    info = get_risk_info_for_buy(ticker)
    if not info.get("available"):
        return False, ""  # SDK 미가용 시 차단 안 함 (운영 지속)

    if info.get("msci_blocked"):
        return True, f"MSCI 차단목록 종목 (5/29 강제매도 대상)"

    return False, ""


def get_position_multiplier(default: float = 1.0) -> float:
    """매수금액 자동 조절 계수 (DANGER 시 ×0.4 등)

    quantum-master에서 매수 시:
        actual_amount = base_amount * get_position_multiplier()
    """
    client = _get_client()
    if client is None:
        return default
    try:
        return client.get_position_multiplier()
    except Exception:
        return default
