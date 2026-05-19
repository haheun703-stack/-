"""차트영웅 매매 — D0 17:30 후보 선정 사이클 (정보봇 가이드 5/19 반영).

시간 흐름 (정보봇 통합 가이드 채택):
  D0 16:30  정보봇 quant_surge_catalyst 갱신 완료
  D0 17:30  본 사이클 → 5-Gate 통과 종목을 picks_d1.json 에 저장
  D+1 14:55 chart_hero_entry_cycle 양봉 확인 + 진입

저장 위치:
  data/chart_hero_picks_{D+1_date}.json
    → 다음날 14:55 entry_cycle이 읽음

cron 등록:
  30 17 * * 1-5 cd ~/quantum-master && ./venv/bin/python3.11 scripts/chart_hero_picker_cycle.py --paper >> /tmp/chart_hero_picker.log 2>&1
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.surge_d1_picker import run_picker


def get_next_trading_date(today: dt.date) -> dt.date:
    """다음 영업일 (주말 제외)."""
    d = today + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--date", default=None, help="D0 (YYYY-MM-DD)")
    parser.add_argument("--live-catalyst", action="store_true",
                        help="Perplexity catalyst 실시간 fallback")
    args = parser.parse_args()

    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    next_day = get_next_trading_date(today)

    print("=" * 70)
    print(f"📋 차트영웅 D0 후보 선정 ({today} {dt.datetime.now():%H:%M:%S})")
    print(f"   → 진입 예정일: {next_day} D+1 14:55")
    print(f"   모드: {'PAPER' if not args.real else '🔴 REAL'}")
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
        send_message("\n".join(lines))
    except Exception as e:
        print(f"⚠️ 텔레그램: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
