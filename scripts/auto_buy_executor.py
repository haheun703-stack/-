"""자동 매수 실행 — 5/20 자비스 자율 가동 (2026-05-18 신규)

배경: 5/18 사장님 자동매매 ON 결단 — should_auto_buy + kis_order_adapter 통합 진입점.

흐름 (14:00~14:55 매 5분 cron):
  1. 가드: AUTO_TRADE_5_20=true + KILL_SWITCH 없음
  2. 후보 풀: tomorrow_picks 강력포착 9건 + sector_fire 상위 (옵션)
  3. 각 종목:
     a. evaluate_filters(broker, ticker, date) → EYE 결과
     b. fetch_price → 현재가 + 가격 시그널
     c. calculate_integrated_score() → 종합 점수 0~100
     d. should_auto_buy() → 안전선 8건 평가 (kis_adapter ⑨ 별도)
     e. action == 'BUY':
        - kis_order_adapter.buy_limit(ticker, current_price, 1) ★ 지정가 (사장님 5/18 결단)
        - 안전선 ⑨ (지정가 현재가±5%) 자동 발동
        - owner_rule_positions.json INSERT
        - 텔레그램 발송
        - break (일일 1건 한도)

가동:
  cron: */5 14 * * 1-5 (평일 14:00~14:55, 12회 기회)
  환경변수: AUTO_TRADE_5_20=true 시만 작동
  안전망: KILL_SWITCH 존재 시 즉시 종료

사용:
  python scripts/auto_buy_executor.py             # 1회 실행
  python scripts/auto_buy_executor.py --dry-run   # 매수 결정만, 실제 주문 X
  python scripts/auto_buy_executor.py --no-tg     # 텔레그램 OFF
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
POSITIONS_STATE_PATH = PROJECT_ROOT / "data" / "owner_rule_positions.json"
KILL_SWITCH = PROJECT_ROOT / "data" / "KILL_SWITCH"
DEFAULT_TOP_N = 9
DEFAULT_GRADE = "강력 포착"


def load_candidates(top_n: int = DEFAULT_TOP_N, grade: str = DEFAULT_GRADE) -> list[tuple[str, str]]:
    """후보 종목 풀 — tomorrow_picks 강력포착 TOP N."""
    if not TOMORROW_PICKS.exists():
        logger.warning("tomorrow_picks.json 없음 — 후보 0건")
        return []
    try:
        data = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
        return [
            (p.get("ticker", ""), p.get("name", p.get("ticker", "")))
            for p in data.get("picks", [])
            if p.get("grade") == grade
        ][:top_n]
    except Exception as e:
        logger.error("tomorrow_picks 로드 실패: %s", e)
        return []


def load_market_regime() -> str:
    """advisory regime 로드 (기본 NEUTRAL).

    5/18 잠정: snapshot_session.py 결과 파일 확인.
    5/19+ 정밀화 예정.
    """
    snap_path = PROJECT_ROOT / "data" / "snapshot_latest.json"
    if not snap_path.exists():
        return "NEUTRAL"
    try:
        data = json.loads(snap_path.read_text(encoding="utf-8"))
        return data.get("regime", "NEUTRAL")
    except Exception:
        return "NEUTRAL"


def load_positions_state() -> dict:
    if not POSITIONS_STATE_PATH.exists():
        return {"positions": {}, "updated_at": None}
    try:
        return json.loads(POSITIONS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "updated_at": None}


def save_positions_state(state: dict) -> None:
    POSITIONS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def send_telegram(msg: str) -> None:
    try:
        from src.telegram_sender import send_message
        send_message(msg)
    except Exception as e:
        logger.warning("텔레그램 발송 실패: %s", e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="매수 결정만, 실제 주문 X")
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 OFF")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ① KILL_SWITCH 가드 (최우선) — dry-run 시 WARN만
    if KILL_SWITCH.exists():
        if args.dry_run:
            logger.warning("[DRY-RUN] KILL_SWITCH 존재 — 실주문이라면 차단됐을 것")
        else:
            logger.info("KILL_SWITCH 존재 — 자동매매 OFF 스킵")
            return 0

    # ② AUTO_TRADE_5_20 가드 (날짜 가드) — dry-run 시 WARN만
    if os.environ.get("AUTO_TRADE_5_20", "false").lower() != "true":
        if args.dry_run:
            logger.warning("[DRY-RUN] AUTO_TRADE_5_20 != true — 실주문이라면 차단됐을 것")
        else:
            logger.info("AUTO_TRADE_5_20 != true — 스킵")
            return 0

    # ③ 후보 풀
    candidates = load_candidates()
    if not candidates:
        logger.warning("후보 0건 — tomorrow_picks 확인 필요")
        if not args.no_tg:
            send_telegram("⛔ [자동매수] 후보 0건 — tomorrow_picks.json 확인 필요")
        return 1

    # ④ KIS broker 초기화
    from src.adapters.kis_stock_data_adapter import KisStockDataAdapter
    from src.adapters.kis_order_adapter import KisOrderAdapter
    from src.use_cases.eye_filters import evaluate_filters
    from src.use_cases.integrated_score import calculate_integrated_score
    from src.use_cases.auto_buy_decider import should_auto_buy, format_decision_for_telegram

    data_adp = KisStockDataAdapter()
    broker = data_adp.broker

    # 시장 regime
    regime = load_market_regime()

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    now_hhmm = now.strftime("%H:%M")

    print("=" * 60)
    print(f"  자동매수 평가 ({now_hhmm} KST, regime={regime})")
    print(f"  후보 {len(candidates)}건, dry-run={args.dry_run}")
    print("=" * 60)

    state = load_positions_state()
    positions = state.get("positions", {})

    buy_executed = False
    decisions_log = []

    for tk, nm in candidates:
        if not tk:
            continue

        # EYE 필터 + 가격
        try:
            eye_res = evaluate_filters(broker, tk, today)
        except Exception as e:
            logger.warning("EYE 평가 실패 %s: %s", tk, e)
            continue

        try:
            px_resp = broker.fetch_price(tk).get("output", {})
            current_price = int(px_resp.get("stck_prpr", 0))
        except Exception as e:
            logger.warning("fetch_price 실패 %s: %s", tk, e)
            continue

        if current_price <= 0:
            logger.warning("현재가 0 (거래정지 추정) %s — 스킵", tk)
            continue

        # 종합 점수
        try:
            sc = calculate_integrated_score(broker, tk, nm, eye_res, regime)
        except Exception as e:
            logger.warning("integrated_score 실패 %s: %s", tk, e)
            continue

        # 안전선 9건 평가 (8건 함수 + ⑨ kis_adapter)
        decision = should_auto_buy(
            ticker=tk,
            name=nm,
            integrated_score=sc.score,
            eye_should_skip=eye_res["should_skip"],
            eye_skip_reasons=eye_res["skip_reasons"],
            market_regime=regime,
            current_price=current_price,
            blocked_by_nega=False,  # 5/20만 SKIP (안전선 ⑧)
            now_str=now_hhmm,
            today=today,
        )

        decisions_log.append((tk, nm, sc.score, decision.action, decision.reason))
        print(f"  [{decision.action}] {nm}({tk}) 점수 {sc.score} → {decision.reason}")

        if decision.action != "BUY":
            continue

        # BUY 실행
        if args.dry_run:
            print(f"  ★ DRY-RUN: 매수 SKIP (실주문 X) — {nm} {current_price:,}원 1주")
            buy_executed = True
            # dry-run에서도 일일 1건 한도 시뮬레이션
            break

        # 실제 주문 — 지정가 (사장님 5/18 결단, 안전선 ⑨ 발동)
        try:
            kis_order = KisOrderAdapter()
            order = kis_order.buy_limit(tk, current_price, 1)
            ok = order.status.value == "PENDING" if hasattr(order.status, "value") else str(order.status) == "OrderStatus.PENDING"

            if not ok:
                tg_msg = (
                    f"❌ [자동매수 실패] {nm}({tk})\n"
                    f"  지정가 {current_price:,}원 1주\n"
                    f"  사유: {order.message or 'unknown'}\n"
                    f"  → 다음 5분 cron 재시도"
                )
                logger.error("주문 실패 %s: %s", tk, order.message)
                if not args.no_tg:
                    send_telegram(tg_msg)
                continue

            # 성공 — positions.json 갱신
            positions[tk] = {
                "entry_price": current_price,
                "entry_date": today,
                "name": nm,
                "qty": 1,
                "peak_price": current_price,
                "trailing_active": False,
                "order_id": order.order_id,
                "integrated_score": sc.score,
                "created_at": datetime.now().isoformat(),
            }

            tg_msg = (
                f"✅ [자동매수 성공] {nm}({tk})\n"
                f"  지정가 {current_price:,}원 × 1주 = {current_price:,}원\n"
                f"  점수: {sc.score} (STRONG)\n"
                f"  주문번호: {order.order_id}\n"
                f"  ─────────────\n"
                f"  사장님 룰 모니터: 14:35부터 매 5분\n"
                f"  - 룰 ① -3% 절대 손절\n"
                f"  - 룰 ② peak -3% 트레일링\n"
                f"  - 룰 ③ 15:20 강제 청산\n"
                f"  - 룰 ④ 수급 지속 시 5/21 이월"
            )
            logger.info("자동매수 성공 %s @ %s원 (주문번호 %s)", tk, current_price, order.order_id)
            print(tg_msg)
            if not args.no_tg:
                send_telegram(tg_msg)
            buy_executed = True
            break  # 일일 1건 한도

        except Exception as e:
            logger.error("주문 실행 예외 %s: %s", tk, e)
            if not args.no_tg:
                send_telegram(f"❌ [자동매수 예외] {nm}({tk}): {e}")
            continue

    # ⑤ 상태 저장
    state["positions"] = positions
    state["updated_at"] = datetime.now().isoformat()
    save_positions_state(state)

    # ⑥ 요약
    print("─" * 60)
    if buy_executed:
        print(f"  ✅ 매수 1건 완료 (일일 한도 도달)")
    else:
        n_skip = sum(1 for _, _, _, action, _ in decisions_log if action == "SKIP")
        print(f"  ⏭️  매수 0건 (전체 {n_skip}건 SKIP)")
        # 14:55 마지막 시도였는데도 0건이면 알림
        if now_hhmm >= "14:55" and not args.no_tg and not args.dry_run:
            top3 = sorted(decisions_log, key=lambda x: x[2], reverse=True)[:3]
            top3_str = "\n".join(f"  {nm} 점수 {sc:.0f} ({reason[:30]})" for _, nm, sc, _, reason in top3)
            send_telegram(
                f"⏭️ [자동매수 0건] 14:00~14:55 통과 후보 없음\n"
                f"TOP 3 (참고):\n{top3_str}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
