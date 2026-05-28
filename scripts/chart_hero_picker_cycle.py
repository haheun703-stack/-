"""차트영웅 매매 — D0 17:30 후보 선정 사이클 (정보봇 가이드 5/19 반영).

시간 흐름 (정보봇 통합 가이드 채택):
  D0 16:30  정보봇 quant_surge_catalyst 갱신 완료
  D0 17:30  본 사이클 → 5-Gate 통과 종목을 picks_d1.json 에 저장
  D+1 14:55 chart_hero_close_cycle 양봉 확인 + 진입 (D+1 종가 직전)

저장 위치:
  data/chart_hero/picks_d1_{D+1_date}.json
    → 다음날 14:55 close_cycle이 읽음 (D+1 양봉 확인 후 종가 직전 진입)

Trading Factory v1 (5/28 Phase 1 row 3):
  picker는 selector 역할 — 매매 호출 X.
  paper 모드 시 picks → register_intent(mode='paper', executor_bot='quant', side='BUY') 등록.
  real 모드 시 intent 등록 금지 (코덱스 5차: live 호출처 마이그레이션 금지).
  → D+1 14:55 close_cycle이 PaperOrderAdapter.buy_limit(mode='paper', executor_bot='quant') 호출 시 통과.

cron 등록:
  30 17 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_picker_cycle.py --paper >> /tmp/chart_hero_picker.log 2>&1
"""

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.surge_d1_picker import run_picker

logger = logging.getLogger(__name__)


def get_next_trading_date(today: dt.date) -> dt.date:
    """다음 영업일 (주말 제외)."""
    d = today + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def main():
    parser = argparse.ArgumentParser()
    # C3: --paper / --real 상호배타 (둘 중 하나 반드시 지정 — 실수 방지)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="paper 시뮬 모드")
    mode_group.add_argument("--real", action="store_true", help="실전 모드 (KIS 주문)")
    parser.add_argument("--date", default=None, help="D0 (YYYY-MM-DD)")
    parser.add_argument("--live-catalyst", action="store_true",
                        help="Perplexity catalyst 실시간 fallback")
    args = parser.parse_args()

    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    next_day = get_next_trading_date(today)

    print("=" * 70)
    print(f"📋 차트영웅 D0 후보 선정 ({today} {dt.datetime.now():%H:%M:%S})")
    print(f"   → 진입 예정일: {next_day} D+1 14:55")
    print(f"   모드: {'PAPER' if args.paper else '🔴 REAL'}")
    print("=" * 70)

    # 5-Gate 실행
    result = run_picker(today.isoformat(), args.live_catalyst)
    print(f"\n[Gate 1] 매크로: {result['gate_1']['score']}/4 "
          f"{'GO' if result['gate_1']['passed'] else 'NO-GO'}")
    print(f"   Advisory: {result['gate_1']['advisory_summary']}")
    print(f"\n[종합] 상태: {result['status']}")
    print(f"       요약: {result['summary']}")

    # 후보 저장
    picks_dir = Path("data/chart_hero")
    picks_dir.mkdir(parents=True, exist_ok=True)
    picks_file = picks_dir / f"picks_d1_{next_day.isoformat()}.json"
    picks_file.write_text(json.dumps({
        "d0_date": today.isoformat(),
        "d1_date": next_day.isoformat(),
        "gate_1": result["gate_1"],
        "status": result["status"],
        "picks": result["picks"],
        "summary": result["summary"],
        "generated_at": dt.datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n💾 후보 저장: {picks_file}")

    if result["picks"]:
        print(f"\n[진입 예정 후보] ({len(result['picks'])}종목)")
        for p in result["picks"]:
            print(f"  ⭐ {p['name']:14} ({p['ticker']}) "
                  f"score={p['buy_score']:>5.1f} "
                  f"cs={p.get('continuity_score')} "
                  f"upside={p.get('upside_pct')}% "
                  f"wk={p.get('weekly_k')}")
    else:
        print(f"\n📭 진입 후보 없음 — {next_day}에는 신규 진입 X")

    # Trading Factory v1 Phase 1 row 3 (5/28):
    # paper 모드 시 picks → order_intents 등록 (D+1 14:55 close_cycle 매매 시 가드 통과).
    # real 모드는 intent 등록 금지 (코덱스 5차: live 호출처 마이그레이션 금지).
    n_intents_registered = 0
    n_intents_failed = 0
    if args.paper and result["picks"]:
        try:
            from src.use_cases.order_intents_gate import register_intent
            from datetime import timezone, timedelta as _td
            seoul = timezone(_td(hours=9))
            now_kst = dt.datetime.now(tz=seoul)
            # D+1 14:55 + 30분 (마감 close까지 여유)
            d1_close_kst = dt.datetime.combine(
                next_day, dt.time(15, 30), tzinfo=seoul,
            )
            for p in result["picks"]:
                tk = p["ticker"]
                intent_id = f"q_{tk}_chart_hero_d1_{next_day.isoformat()}"
                intent = {
                    "intent_id": intent_id, "bot": "quant",
                    "engine": "chart_hero_5gate",
                    "ticker": tk, "name": p.get("name", tk),
                    "side": "BUY", "mode": "paper",
                    "score": float(p.get("buy_score", 0.0)),
                    "confidence": "strong" if p.get("buy_score", 0) >= 80 else "medium",
                    "created_at": now_kst.isoformat(),
                    "expires_at": d1_close_kst.isoformat(),
                    "d0_date": today.isoformat(),
                    "d1_date": next_day.isoformat(),
                    "continuity_score": p.get("continuity_score"),
                    "upside_pct": p.get("upside_pct"),
                    "weekly_k": p.get("weekly_k"),
                }
                try:
                    register_intent(intent, bot="quant")
                    n_intents_registered += 1
                except Exception as e:
                    logger.warning("[picker] register_intent 실패 %s: %s", tk, e)
                    n_intents_failed += 1
            print(f"\n📝 order_intents 등록: {n_intents_registered}건 "
                  f"(실패 {n_intents_failed}건)")
        except Exception as e:
            logger.warning("[picker] order_intents_gate import 실패: %s", e)
            print(f"\n⚠️ order_intents 등록 스킵: {e}")
    elif args.real:
        print("\n⛔ real 모드 — order_intents 등록 금지 (코덱스 5차 지시)")

    # 텔레그램 알림
    try:
        from src.telegram_sender import send_message
        lines = [
            f"📋 차트영웅 D0 후보 선정 ({today})",
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"진입 예정: {next_day}",
            f"매크로: {result['gate_1']['score']}/4",
            f"후보: {len(result['picks'])}건",
        ]
        for p in result["picks"]:
            lines.append(f"  ⭐ {p['name']} cs={p.get('continuity_score')} "
                         f"upside={p.get('upside_pct')}%")
        if not result["picks"]:
            lines.append("  (조건 만족 종목 없음 — 시장 대기)")
        if args.paper and n_intents_registered > 0:
            lines.append(f"📝 paper intents: {n_intents_registered}건 등록")
        send_message("\n".join(lines))
    except Exception as e:
        print(f"⚠️ 텔레그램: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
