"""KILL_SWITCH 자동 활성화 매니저 — Layer 7 (5/19 사장님 1년 패턴 돌파)

5명 검수팀 중 1명이라도 FAIL 검출 시 호출 → KILL_SWITCH 자동 활성화
→ auto_buy_executor 다음 cron부터 즉시 종료 (사장님 개입 0건)

사용:
    from src.agents.kill_switch_manager import activate_kill_switch
    activate_kill_switch(reason="AUTO_TRADE_5_20=false", source="EnvChecker")

환경변수 가드 (5/19 사장님 결단 C — 카톡 도배 방지):
- AGENT_TELEGRAM_ENABLED=false (디폴트): 4명 워커 + Reporter 카톡 OFF (logger.info만)
- AGENT_TELEGRAM_ENABLED=true: 4명 워커 + Reporter 카톡 ON (5/27+ 웹 대시보드 통합 검토 후 결정)
- KILL_SWITCH RED 발동 카톡: 항상 ON (kill_switch_manager.activate_kill_switch send_tg=True가 디폴트)

배경 (1년 패턴 + 5/19 단타봇 사고 통합):
- 단타봇 사고: 카톡 누락 → 사장님이 매매 0건을 알아채지 못함
- 퀀트봇 함정: 카톡 도배 (5명 워커 × 4슬롯 + 매매 cron) → 사장님이 무시 → 진짜 RED 묻힘
- 결단 C: 평상시 카톡 OFF, 오직 KILL_SWITCH RED만 단일 채널로 알림
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH = PROJECT_ROOT / "data" / "KILL_SWITCH"
KILL_SWITCH_LOG = PROJECT_ROOT / "data" / "kill_switch_history.jsonl"
REPORTS_DIR = PROJECT_ROOT / "data" / "agent_reports"


def activate_kill_switch(reason: str, source: str, send_tg: bool = True) -> bool:
    """KILL_SWITCH 자동 활성화. 5명 워커 어디서든 FAIL 시 호출.

    Args:
        reason: 차단 사유 (사장님 카톡에 노출됨)
        source: 호출 워커 이름 (EnvChecker / CodeAuditor / FlowMonitor / DataIntegrity / Reporter)
        send_tg: 텔레그램 알람 발송 여부

    Returns:
        True if newly activated, False if already exists
    """
    # 이미 존재하면 중복 알람 방지
    if KILL_SWITCH.exists():
        existing = KILL_SWITCH.read_text(encoding="utf-8")
        if "AUTO" in existing:
            logger.info("[KILL_SWITCH] 이미 자동 활성화됨 — 중복 알람 스킵")
            _append_history(reason, source, "DUPLICATE")
            return False

    # 활성화
    content = (
        f"AUTO ACTIVATED\n"
        f"timestamp={datetime.now().isoformat()}\n"
        f"source={source}\n"
        f"reason={reason}\n"
    )
    KILL_SWITCH.write_text(content, encoding="utf-8")
    _append_history(reason, source, "ACTIVATED")
    logger.warning("[KILL_SWITCH] 자동 활성화 by %s: %s", source, reason)

    if send_tg:
        try:
            from src.telegram_sender import send_message
            send_message(
                f"🚨 [자동 차단 발동]\n"
                f"검수팀 [{source}]가 FAIL 검출\n"
                f"━━━━━━━━━━━━━━━━━\n"
                f"사유: {reason}\n"
                f"data/KILL_SWITCH 자동 활성화됨\n"
                f"→ 자동매매 cron 즉시 중단\n"
                f"━━━━━━━━━━━━━━━━━\n"
                f"사장님 확인 후 수동 해제:\n"
                f"  rm ~/quantum-master/data/KILL_SWITCH"
            )
        except Exception as e:
            logger.error("KILL_SWITCH 텔레그램 발송 실패: %s", e)
    return True


def is_kill_switch_active() -> bool:
    return KILL_SWITCH.exists()


def get_kill_switch_info() -> dict | None:
    if not KILL_SWITCH.exists():
        return None
    content = KILL_SWITCH.read_text(encoding="utf-8")
    info = {"raw": content, "auto": "AUTO" in content}
    for line in content.split("\n"):
        if "=" in line:
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
    return info


def manual_deactivate() -> bool:
    """수동 해제 — 사장님이 명시적으로 호출하거나 cron 정규 복구 시"""
    if not KILL_SWITCH.exists():
        return False
    KILL_SWITCH.unlink()
    _append_history("manual", "user", "DEACTIVATED")
    return True


def _append_history(reason: str, source: str, action: str) -> None:
    KILL_SWITCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(),
        "action": action,
        "source": source,
        "reason": reason,
    }
    with KILL_SWITCH_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_worker_report(name: str, data: dict) -> Path:
    """워커 결과를 data/agent_reports/{name}_latest.json에 저장 — Reporter가 읽음

    Args:
        name: env_checker / code_auditor / flow_monitor / data_integrity / reporter
        data: 워커 결과 dict (timestamp는 자동 추가)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {**data, "saved_at": datetime.now().isoformat()}
    path = REPORTS_DIR / f"{name}_latest.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
