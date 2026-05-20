"""차트영웅 매매 — 매일 09:30 KST 보유 포지션 모니터링.

흐름:
  1. executor.monitor_positions()  보유 종목 현재가 조회
  2. decide_action() 자동 판정    추매/익절/손절/HOLD
  3. KIS 주문 실행 (real 모드)    또는 시뮬 (paper 모드)
  4. 결과 보고 + 텔레그램 알림

cron 등록 (VPS):
  30 9 * * 1-5  cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_morning_monitor.py --paper >> /tmp/chart_hero_morning.log 2>&1
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.chart_hero_executor import ChartHeroExecutor
from src.agents.kill_switch_manager import is_kill_switch_active


def main():
    parser = argparse.ArgumentParser()
    # C3: --paper / --real 상호배타 (둘 중 하나 반드시 지정 — 5/27 실전 전환 시 실수 방지)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="paper 시뮬 모드")
    mode_group.add_argument("--real", action="store_true", help="실전 모드 (KIS 주문)")
    parser.add_argument("--capital", type=float, default=25_000_000)
    parser.add_argument("--max-qty", type=int, default=None,
                        help="종목당 최대 수량 (1주차 워밍업: 1)")
    args = parser.parse_args()

    # C3: 명시적 paper/real 판별 (default 의존 제거)
    is_paper = args.paper
    today = dt.date.today().isoformat()

    print("=" * 70)
    print(f"🔍 차트영웅 보유 포지션 모니터 ({today} {dt.datetime.now():%H:%M:%S})")
    print(f"   모드: {'PAPER' if is_paper else '🔴 REAL'}")
    print("=" * 70)

    # ★ P0 (5/20 추가): KILL_SWITCH 경고 (매도 액션은 손실 회피 차원에서 진행)
    # morning_monitor는 매수(ADD_BUY)와 매도(STOPLOSS/PARTIAL_SELL) 모두 — 매도는 허용
    # 매수 차단은 executor 내부 (executor가 KILL_SWITCH 체크하여 ADD_BUY 막음)
    kill_switch_active = is_kill_switch_active()
    if kill_switch_active:
        print("\n🛑 KILL_SWITCH 활성 — 추매(ADD_BUY) 자동 차단, 매도(STOPLOSS/PARTIAL_SELL)는 진행")

    executor = ChartHeroExecutor(paper=is_paper, total_capital=args.capital,
                                  max_qty_per_ticker=args.max_qty,
                                  kill_switch_active=kill_switch_active)

    if not executor.positions:
        print("\n📭 보유 포지션 없음.")
        return

    print(f"\n[1] 보유 포지션: {len(executor.positions)}건")
    for tk, p in executor.positions.items():
        if p.get("is_closed"):
            continue
        held_days = (dt.date.today() - dt.date.fromisoformat(p["entry_date"])).days
        print(f"   {tk} {p['name']:14} 평단={p['avg_price']:>10} "
              f"수량={p['total_qty']:>4} 단계={p['stage']} D+{held_days}")

    print(f"\n[2] 액션 실행")
    results = executor.monitor_positions()
    for r in results:
        action = r.get("action", "?")
        flag = {"HOLD": "  ", "ADD_BUY": "📥", "PARTIAL_SELL": "💰",
                "STOPLOSS": "🛑", "FORCE_CLOSE": "🏁", "PRICE_FAIL": "❌"}.get(action, "?")
        print(f"   {flag} {r}")

    print(f"\n[3] 누적 요약")
    summary = executor.get_summary()
    for k, v in summary.items():
        print(f"   {k}: {v}")

    # 로그 저장
    log_dir = Path("data/logs/chart_hero")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"morning_monitor_{today}.json"
    log_file.write_text(json.dumps({
        "date": today, "mode": "paper" if is_paper else "real",
        "actions": results, "summary": summary,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n💾 로그: {log_file}")

    # 텔레그램 알림 (액션 있을 때만)
    try:
        actionable = [r for r in results if r.get("action") not in ("HOLD", "PRICE_FAIL")]
        if actionable:
            from src.telegram_sender import send_message
            mode_kr = "PAPER" if is_paper else "🔴 REAL"
            lines = [
                f"🔍 차트영웅 보유 모니터 ({today})",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"모드: {mode_kr} | 활성: {summary['active_count']}건 | PnL {summary['pnl_pct']}%",
                f"",
            ]
            for r in actionable:
                action_kr = {"ADD_BUY": "📥 추매", "PARTIAL_SELL": "💰 부분익절",
                             "STOPLOSS": "🛑 손절", "FORCE_CLOSE": "🏁 강제청산"}.get(
                    r.get("action"), r.get("action"))
                lines.append(f"  {action_kr} {r.get('ticker')} "
                             f"{r.get('qty', 0)}주 @ {r.get('price', 0):,}원")
                lines.append(f"     이유: {r.get('reason', '')}")
            send_message("\n".join(lines))
    except Exception as e:
        print(f"⚠️ 텔레그램 알림 실패: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
