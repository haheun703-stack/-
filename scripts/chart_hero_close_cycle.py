"""차트영웅 매매 — 매일 14:55 KST 마감 사이클.

흐름:
  1. surge_d1_picker.run_picker()    5-Gate 후보 산출 (전날 결과 활용)
  2. d1_confirm.monitor_picks()      D+1 양봉 확인
  3. executor.execute_d1_entry()     양봉 확인 종목 1차 진입 (paper/real)
  4. 결과 보고 (CSV + 텔레그램)

실행 방법:
  # paper 모드 (5/22~5/26)
  python -u -X utf8 scripts/chart_hero_close_cycle.py --paper

  # real 모드 (5/27~, 사장님 GO 후)
  python -u -X utf8 scripts/chart_hero_close_cycle.py --real --capital 25000000

cron 등록 (VPS):
  55 14 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_close_cycle.py --paper >> /tmp/chart_hero_close.log 2>&1
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.surge_d1_picker import run_picker
from src.strategies.d1_confirm import monitor_picks, is_check_window
from src.strategies.chart_hero_executor import ChartHeroExecutor
from src.strategies.chart_hero_entry_filter import filter_picks_for_entry


def main():
    parser = argparse.ArgumentParser()
    # C3: --paper / --real 상호배타 (둘 중 하나 반드시 지정 — 5/27 실전 전환 시 실수 방지)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="paper 시뮬 모드")
    mode_group.add_argument("--real", action="store_true", help="실전 모드 (KIS 주문)")
    parser.add_argument("--capital", type=float, default=25_000_000, help="잔고 (원)")
    parser.add_argument("--max-qty", type=int, default=None,
                        help="종목당 최대 수량 (1주차 워밍업: 1)")
    parser.add_argument("--no-entry-filter", action="store_true",
                        help="VWAP+수급 진입 필터 비활성 (테스트용)")
    parser.add_argument("--force", action="store_true", help="시각 윈도우 무시 (테스트용)")
    args = parser.parse_args()

    # C3: 명시적 paper/real 판별 (default 의존 제거)
    is_paper = args.paper
    today = dt.date.today().isoformat()

    print("=" * 70)
    print(f"📊 차트영웅 매매 마감 사이클 ({today} {dt.datetime.now():%H:%M:%S})")
    print(f"   모드: {'PAPER 시뮬' if is_paper else '🔴 REAL 실전'}")
    print(f"   잔고: {args.capital:,.0f}원")
    print("=" * 70)

    # 시각 윈도우 체크 (force 옵션 없으면)
    if not args.force and not is_check_window():
        print(f"\n⚠️  현재 시각이 14:30~15:00 KST 윈도우 밖. --force 옵션으로 강제 실행 가능.")
        # paper 모드면 계속 진행, real이면 차단
        if not is_paper:
            print("REAL 모드 차단. 종료.")
            return

    # 1) 5-Gate 종목 선정
    print("\n[1] 5-Gate 종목 선정")
    result = run_picker(today)
    print(f"   상태: {result['status']}")
    print(f"   매크로: {result['gate_1']['score']}/4")
    print(f"   요약: {result['summary']}")

    if not result["picks"]:
        print("\n→ 진입 후보 없음. 사이클 종료.")
        return

    # 2) D+1 양봉 확인
    print(f"\n[2] D+1 양봉 확인 ({len(result['picks'])}종목)")
    confirmed = monitor_picks(result["picks"])
    will_enter = [c for c in confirmed if c.get("will_enter")]
    for c in confirmed:
        flag = "⭐" if c.get("will_enter") else "🛑"
        chk = c.get("d1_check", {})
        print(f"   {flag} {c['name']:14} 양봉={chk.get('is_bullish_candle')} "
              f"회복={chk.get('recovery_pct')}% → {'ENTER' if c.get('will_enter') else 'SKIP'}")

    if not will_enter:
        print("\n→ D+1 양봉 확인 통과 종목 없음. 사이클 종료.")
        return

    # 2-B) 진입 필터 (VWAP + 수급, Phase 1 — 5/20 추가)
    if not args.no_entry_filter:
        print(f"\n[2-B] 진입 필터 (VWAP + 수급) — {len(will_enter)}종목")
        will_enter = filter_picks_for_entry(will_enter)
        for p in will_enter:
            fr = p.get("filter_result", {})
            ok = p.get("will_enter_final", False)
            mark = "✅" if ok else "🛑"
            api_fail = fr.get("api_failures", [])
            api_warn = f" ⚠️API실패={api_fail}" if api_fail else ""
            print(f"   {mark} {p['name']:14} ({p['ticker']}) "
                  f"VWAP아래={fr.get('vwap_below')} 수급+={fr.get('supply_positive')} "
                  f"→ {fr.get('pass_count')}/{fr.get('pass_threshold')}{api_warn}")
        # 필터 통과 종목만 추출
        will_enter = [p for p in will_enter if p.get("will_enter_final")]
        if not will_enter:
            print("\n→ 진입 필터 통과 종목 없음. 사이클 종료.")
            return

    # 3) 진입 실행
    print(f"\n[3] 진입 실행 ({len(will_enter)}종목, max_qty={args.max_qty or '무제한'})")
    executor = ChartHeroExecutor(paper=is_paper, total_capital=args.capital,
                                  max_qty_per_ticker=args.max_qty)
    results = executor.execute_d1_entry(will_enter)
    for r in results:
        print(f"   {r}")

    # 4) 요약
    print("\n[4] 포지션 요약")
    summary = executor.get_summary()
    for k, v in summary.items():
        print(f"   {k}: {v}")

    # 5) 로그 저장
    log_dir = Path("data/logs/chart_hero")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"close_cycle_{today}.json"
    log_file.write_text(json.dumps({
        "date": today, "mode": "paper" if is_paper else "real",
        "gate_result": result, "confirmed": confirmed,
        "entry_results": results, "summary": summary,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n💾 로그: {log_file}")

    # 6) 텔레그램 알림
    try:
        from src.telegram_sender import send_message
        mode_kr = "PAPER 시뮬" if is_paper else "🔴 REAL 실전"
        lines = [
            f"📊 차트영웅 마감 사이클 ({today})",
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"모드: {mode_kr}",
            f"매크로 게이트: {result['gate_1']['score']}/4 "
            f"{'✓ GO' if result['gate_1']['passed'] else '🛑 차단'}",
            f"진입: {summary['active_count']}건 "
            f"(누적 PnL {summary['pnl_pct']}%)",
        ]
        for r in results:
            if r.get("action") in ("PAPER_BUY", "REAL_BUY"):
                lines.append(f"  ⭐ {r['ticker']} {r.get('qty')}주 @ {r.get('price'):,}원")
            elif r.get("action") == "SKIP_TOO_EXPENSIVE":
                lines.append(f"  ⏭️ {r['ticker']} 고가주 제외")
            elif r.get("action") == "BLOCKED":
                lines.append(f"  🚫 {r.get('reason', '')}")
        send_message("\n".join(lines))
    except Exception as e:
        print(f"⚠️ 텔레그램 알림 실패: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
