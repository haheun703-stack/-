#!/usr/bin/env python3
"""
상한가 눌림목 분할매수 스캐너 v1.0
═══════════════════════════════════════════════

2층 구조 일일 스캔:
  Layer 1: v10c 큐레이션 93종목 중 오늘 급등 감시
  Layer 2: 전 종목 스캔 → 품질 필터 → 자동 편입

Usage:
  # 오늘 기준 스캔
  python scripts/scan_surge_pullback.py

  # 특정일 스캔 (백필)
  python scripts/scan_surge_pullback.py --date 20260508

  # 여러 날 백필 (시작~종료)
  python scripts/scan_surge_pullback.py --backfill 20260505 20260508

  # 현재 워치리스트 상태 확인
  python scripts/scan_surge_pullback.py --status

  # 텔레그램 알림 포함
  python scripts/scan_surge_pullback.py --telegram

BAT-D 연동:
  schedule_D_after_close.bat 에 추가:
  python -u -X utf8 scripts/scan_surge_pullback.py --telegram
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pykrx_quiet import silence_pykrx_logging

silence_pykrx_logging()  # pykrx 로그인/로깅 노이즈 억제 (진입부 1회)

from src.use_cases.surge_pullback_engine import SurgePullbackEngine


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def send_telegram(result: dict):
    """텔레그램 알림 전송 (기존 어댑터 활용)"""
    try:
        from src.adapters.telegram_adapter import send_message
    except ImportError:
        try:
            from src.adapters.flowx_uploader import _send_telegram as send_message
        except ImportError:
            logging.warning("텔레그램 어댑터 없음 — 알림 스킵")
            return

    lines = []
    date = result["date"]

    # 헤더
    lines.append(f"[눌림목 엔진] {date}")
    lines.append("")

    # 매수 시그널 (최우선)
    if result["signals"]:
        lines.append("★ 매수 시그널 ★")
        for s in result["signals"]:
            sectors = ",".join(s.get("sectors", [])[:2])
            lines.append(
                f"  {s['ticker']} {s['name']} "
                f"L{s['layer']} [{sectors}]"
            )
            lines.append(
                f"  급등{s['surge_pct']:+.1f}% → "
                f"피크{s['peak_price']:,} → "
                f"현재{s['entry_price']:,} ({s['pullback_pct']:+.1f}%)"
            )
        lines.append("")

    # 신규 발굴
    new_l1 = result.get("new_layer1", [])
    new_l2 = result.get("new_layer2", [])
    if new_l1 or new_l2:
        lines.append(f"신규 편입: L1={len(new_l1)} L2={len(new_l2)}")
        for e in new_l1:
            lines.append(f"  L1 {e['ticker']} {e['name']} +{e['surge_pct']:.1f}%")
        for e in new_l2[:10]:  # 최대 10개
            sectors = ",".join(e.get("sectors", [])[:2])
            lines.append(
                f"  L2 {e['ticker']} {e['name']} "
                f"+{e['surge_pct']:.1f}% [{sectors}]"
            )
        if len(new_l2) > 10:
            lines.append(f"  ... 외 {len(new_l2)-10}건")
        lines.append("")

    # 감시 현황
    active = result.get("active_watchlist", [])
    lines.append(f"감시중: {len(active)}건")

    # 위험 종목 (눌림 -8% 이상)
    danger = [e for e in active if e.get("pullback_from_peak", 0) <= -8]
    if danger:
        lines.append("주의 (눌림 -8% 이상):")
        for e in sorted(danger, key=lambda x: x.get("pullback_from_peak", 0)):
            pb = e.get("pullback_from_peak", 0)
            lines.append(
                f"  {e['ticker']} {e['name']} "
                f"피크→{pb:+.1f}% D{e['watch_day']}"
            )

    msg = "\n".join(lines)
    try:
        send_message(msg)
        logging.info("텔레그램 전송 완료")
    except Exception as e:
        logging.error("텔레그램 전송 실패: %s", e)
        print(msg)


def print_result(result: dict):
    """결과 콘솔 출력"""
    date = result["date"]
    print(f"\n{'='*70}")
    print(f" 상한가 눌림목 엔진 결과: {date}")
    print(f"{'='*70}")

    # 매수 시그널
    signals = result.get("signals", [])
    if signals:
        print(f"\n ★★★ 매수 시그널: {len(signals)}건 ★★★")
        print(f"{'─'*70}")
        for s in signals:
            sectors = ",".join(s.get("sectors", [])[:2])
            print(f"  L{s['layer']} {s['ticker']} {s['name']:12s} "
                  f"[{sectors}]")
            print(f"     급등: {s['surge_date']} +{s['surge_pct']:.1f}% | "
                  f"피크: {s['peak_price']:,}원 | "
                  f"진입: {s['entry_price']:,}원 ({s['pullback_pct']:+.1f}%) | "
                  f"D{s['watch_day']}")
            # v1.1: 수급 확인 정보
            buyers = s.get("supply_buyers", [])
            detail = s.get("supply_detail", {})
            if buyers or detail:
                parts = [f"{k}:{v:+.1f}억" for k, v in detail.items()]
                print(f"     수급: {' | '.join(parts)}  ← {s.get('supply_reason', '')}")
            # v1.2: 보유비중/탄력성 정보
            e_grade = s.get("elasticity_grade", "")
            e_score = s.get("elasticity_score", 0)
            f_pct = s.get("frgn_pct", 0)
            if e_grade and e_grade != "N/A":
                total_shares = s.get("total_shares", 0)
                ts_str = f"{total_shares/1e6:.0f}백만주" if total_shares else "?"
                print(f"     보유: 외국인{f_pct:.1f}% | "
                      f"발행{ts_str} | "
                      f"탄력성 {e_grade}등급({e_score:.0f}점)")
                e_reason = s.get("elasticity_reason", "")
                if e_reason:
                    print(f"     근거: {e_reason}")
                # 투자자별 상세
                inv_data = s.get("ownership_investors", {})
                if inv_data:
                    inv_parts = []
                    for inv_name, inv_info in inv_data.items():
                        if inv_name == "개인":
                            continue  # 개인은 생략
                        cum = inv_info.get("cum_net_val_억", 0)
                        r5 = inv_info.get("r5d", {}).get("net_val_억", 0)
                        trend = inv_info.get("trend", "")
                        inv_parts.append(f"{inv_name}:누적{cum:+.0f}억/5일{r5:+.0f}억({trend})")
                    if inv_parts:
                        print(f"     누적: {' | '.join(inv_parts)}")
    else:
        print(f"\n 매수 시그널: 없음")

    # 신규 편입
    new_l1 = result.get("new_layer1", [])
    new_l2 = result.get("new_layer2", [])
    if new_l1 or new_l2:
        print(f"\n ■ 신규 편입: Layer1={len(new_l1)}건, Layer2={len(new_l2)}건")
        print(f"{'─'*70}")
        for e in new_l1:
            sectors = ",".join(e.get("sectors", [])[:2])
            print(f"  [L1] {e['ticker']} {e['name']:12s} "
                  f"+{e['surge_pct']:.1f}% "
                  f"종가{e['surge_close']:,}원 "
                  f"[{sectors}]")
        for e in new_l2:
            sectors = ",".join(e.get("sectors", [])[:2])
            tv = e.get("trading_value", 0)
            print(f"  [L2] {e['ticker']} {e['name']:12s} "
                  f"+{e['surge_pct']:.1f}% "
                  f"종가{e['surge_close']:,}원 "
                  f"거래{tv/1e8:.0f}억 "
                  f"[{sectors}]")

    # 감시 현황
    active = result.get("active_watchlist", [])
    print(f"\n ■ 감시 중: {len(active)}건")
    print(f"{'─'*70}")

    if active:
        l1_active = [e for e in active if e["layer"] == 1]
        l2_active = [e for e in active if e["layer"] == 2]

        for label, items in [("Layer1", l1_active), ("Layer2", l2_active)]:
            if not items:
                continue
            print(f"  [{label}] {len(items)}건")
            for e in sorted(items, key=lambda x: x.get("pullback_from_peak", 0)):
                pb = e.get("pullback_from_peak", 0)
                icon = "🔴" if pb <= -8 else "🟡" if pb <= -5 else "🟢"
                sectors = ",".join(e.get("sectors", [])[:2])
                print(f"    {icon} {e['ticker']} {e['name']:12s} "
                      f"급등+{e['surge_pct']:.1f}% "
                      f"피크→{pb:+.1f}% "
                      f"D{e['watch_day']} "
                      f"[{sectors}]")

    # 만료
    expired = result.get("expired", [])
    if expired:
        print(f"\n ■ 만료: {len(expired)}건")
        for e in expired:
            print(f"  {e['ticker']} {e['name']:12s} "
                  f"급등+{e['surge_pct']:.1f}% → 눌림 미발생")

    # 요약
    print(f"\n{'─'*70}")
    print(f" {result['summary']}")
    print(f"{'='*70}\n")


def run_backfill(engine: SurgePullbackEngine, start: str, end: str):
    """여러 날짜 백필 실행"""
    from pykrx import stock as pykrx_stock
    import time

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")

        # 주말 스킵
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        print(f"\n{'━'*70}")
        print(f" 백필: {date_str}")
        print(f"{'━'*70}")

        try:
            result = engine.run_daily(date_str)
            print_result(result)
        except Exception as e:
            logging.error("백필 실패 %s: %s", date_str, e)

        time.sleep(1)  # pykrx 부하 방지
        current += timedelta(days=1)


def run_enrich_signals(engine: SurgePullbackEngine):
    """시그널 파일에서 탄력성/수급이 null인 항목을 재계산하여 채움."""
    import json
    sig_path = PROJECT_ROOT / "data" / "surge_pullback_signals.json"
    if not sig_path.exists():
        print("시그널 파일 없음")
        return

    with open(sig_path, encoding="utf-8") as f:
        signals = json.load(f)

    updated = 0
    today_str = datetime.now().strftime("%Y%m%d")

    for s in signals:
        ticker = s.get("ticker", "")
        name = s.get("name", "")

        # elasticity/수급/FIRE 중 하나라도 null이면 재계산
        needs_update = (
            s.get("elasticity_score") is None
            or s.get("frgn_pct") is None
            or s.get("sectors") == ["미분류"]
            or s.get("fire_sector") is None
            or s.get("supply_buyers") is None
        )
        if not needs_update:
            continue

        sig_date = (s.get("signal_date") or today_str).replace("-", "")
        print(f"  보강: {ticker} {name} (시그널 {sig_date})")

        # 탄력성 재계산
        try:
            profile = engine.calc_ownership_profile(ticker, sig_date)
            if profile.get("elasticity_grade") not in (None, "N/A"):
                s["elasticity_score"] = profile["elasticity_score"]
                s["elasticity_grade"] = profile["elasticity_grade"]
                s["elasticity_reason"] = profile.get("elasticity_reason", "")
                s["frgn_pct"] = profile.get("frgn_pct", 0)
                s["total_shares"] = profile.get("total_shares", 0)
                s["ownership_investors"] = profile.get("investors", {})
                print(f"    → 탄력성 {profile['elasticity_grade']} {profile['elasticity_score']:.0f}점, "
                      f"외국인 {profile.get('frgn_pct', 0):.1f}%")
        except Exception as e:
            print(f"    → 탄력성 실패: {e}")

        # 수급 확인 재계산
        if s.get("supply_buyers") is None:
            try:
                supply = engine.check_supply_demand(ticker, sig_date)
                s["supply_buyers"] = supply.get("buyers", [])
                s["supply_detail"] = supply.get("detail", {})
                s["supply_reason"] = supply.get("reason", "")
                if supply.get("buyers"):
                    print(f"    → 수급: {supply['buyers']}")
            except Exception as e:
                print(f"    → 수급 실패: {e}")

        # 섹터 재분류
        if s.get("sectors") == ["미분류"] or not s.get("sectors"):
            try:
                sectors = engine.tag_sector(ticker, name)
                if sectors and sectors != ["미분류"]:
                    s["sectors"] = sectors
                    print(f"    → 섹터: {sectors}")
            except Exception:
                pass

        # FIRE 점수 계산
        if s.get("fire_sector") is None:
            try:
                fire_info = engine.get_sector_fire_for_stock(ticker, s.get("sectors", []))
                s["fire_sector"] = fire_info.get("fire_sector", "")
                s["fire_grade"] = fire_info.get("fire_grade", "")
                s["fire_score"] = fire_info.get("fire_score", 0)
                if fire_info.get("fire_sector"):
                    print(f"    → FIRE: {fire_info['fire_sector']} {fire_info['fire_grade']} {fire_info['fire_score']}점")
            except Exception as e:
                print(f"    → FIRE 실패: {e}")

        updated += 1

    if updated > 0:
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(signals, f, ensure_ascii=False, indent=2)
        print(f"\n총 {updated}건 보강 완료 → {sig_path}")
    else:
        print("보강 필요한 시그널 없음")


def main():
    parser = argparse.ArgumentParser(
        description="상한가 눌림목 분할매수 스캐너"
    )
    parser.add_argument("--date", type=str, default=None,
                       help="스캔 날짜 (YYYYMMDD, 기본=오늘)")
    parser.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                       help="백필 기간 (YYYYMMDD YYYYMMDD)")
    parser.add_argument("--enrich", action="store_true",
                       help="시그널 탄력성/수급 null 항목 보강")
    parser.add_argument("--status", action="store_true",
                       help="현재 워치리스트 상태만 출력")
    parser.add_argument("--telegram", action="store_true",
                       help="텔레그램 알림 전송")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 로그")
    # 전략 파라미터 오버라이드
    parser.add_argument("--surge", type=float, default=15.0,
                       help="급등 기준 %% (기본=15)")
    parser.add_argument("--pullback", type=float, default=10.0,
                       help="눌림 기준 %% (기본=10)")
    parser.add_argument("--watch-days", type=int, default=3,
                       help="감시 기간 (기본=3일)")
    parser.add_argument("--min-price", type=int, default=10000,
                       help="최소 주가 (기본=10000원)")
    parser.add_argument("--min-tv", type=float, default=10,
                       help="최소 거래대금 (억, 기본=10)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = {
        "surge_threshold": args.surge,
        "pullback_pct": args.pullback,
        "watch_days": args.watch_days,
        "min_price": args.min_price,
        "min_trading_value": int(args.min_tv * 1e8),
    }

    engine = SurgePullbackEngine(config=config)

    # 상태 확인 모드
    if args.status:
        engine.print_status()
        return

    # 탄력성/수급 보강 모드
    if args.enrich:
        run_enrich_signals(engine)
        return

    # 백필 모드
    if args.backfill:
        run_backfill(engine, args.backfill[0], args.backfill[1])
        return

    # 단일 날짜 스캔
    date = args.date or datetime.now().strftime("%Y%m%d")
    result = engine.run_daily(date)
    print_result(result)

    if args.telegram:
        send_telegram(result)


if __name__ == "__main__":
    main()
