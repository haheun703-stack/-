"""일일 위험감지 상태 텔레그램 알림 (P0-7).

운영:
- 매일 17:00 (정보봇 갱신 16:49 + 10분 여유) 실행
- DANGER/CRISIS 진입 시 강조 알림
- NORMAL/CAUTION/WARNING은 일반 알림
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("risk_status_notify")

from src.utils.risk_gate import get_risk_status_safe, get_risk_gate


def main():
    status = get_risk_status_safe()
    if not status:
        logger.warning("위험 상태 조회 실패 (Supabase 또는 데이터 없음)")
        return

    level = status.get("level", "NORMAL")
    level_kr = status.get("level_kr", "정상")
    score = status.get("total_score", 0)
    date = status.get("date", "")

    rg = get_risk_gate()
    mult = rg.get_position_multiplier() if rg else 1.0

    # 강조 알림 (DANGER/CRISIS)
    if level in ("DANGER", "CRISIS"):
        prefix = f"⚠️⚠️⚠️ {level_kr} 구간 진입 ⚠️⚠️⚠️\n\n"
    else:
        prefix = ""

    msg_lines = [
        prefix + f"📊 {date} 한국시장 위험점수 {score}점 ({level_kr})",
        f"매수금액 배수: ×{mult}",
        "",
        f"외부:{status.get('external_score', '-')} "
        f"외인:{status.get('foreign_flow_score', '-')} "
        f"이벤트:{status.get('event_score', '-')} "
        f"디커플:{status.get('decoupling_score', '-')}",
        "",
        "🚨 핵심 시그널:",
    ]
    for sig in status.get("key_signals", [])[:6]:
        msg_lines.append(f"  • {sig}")

    action = status.get("recommended_action", "")
    if action:
        msg_lines.append("")
        msg_lines.append(f"👉 권장: {action}")

    message = "\n".join(msg_lines)

    try:
        from src.telegram_sender import send_message
        send_message(message)
        logger.info(f"[notify] 전송 완료 — {level_kr} ({score}점)")
    except Exception as e:
        logger.warning(f"[notify] 텔레그램 전송 실패: {e}")
        print(message)


if __name__ == "__main__":
    main()
