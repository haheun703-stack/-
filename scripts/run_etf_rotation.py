#!/usr/bin/env python3
"""
ETF 3축 로테이션 일일 실행 스크립트
=============================================
사용법:
    python -u -X utf8 scripts/run_etf_rotation.py [--dry-run] [--no-telegram]
    python -u -X utf8 scripts/run_etf_rotation.py --blind-test
    python -u -X utf8 scripts/run_etf_rotation.py --blind-test --no-ai

    --dry-run       : 결과 저장만 (텔레그램 발송 안 함)
    --no-telegram   : 텔레그램 발송 안 함
    --blind-test    : 블라인드 테스트 모드 (일별 시그널 로그 누적)
    --no-ai         : AI 필터 비활성화 (룰베이스만 실행)

데이터 브릿지(src/etf/data_bridge.py)가 방탄 fallback 처리:
  - parquet → JSON → 직접계산 → 안전 기본값
  - 5축 없으면 레짐 추정, Smart Money 없으면 수급에서 분류

AI 필터 레이어 (v11):
  - 룰베이스 주문 → AI가 PASS/KILL/HOLD 판단
  - AI는 "사지 마라"만 할 수 있음 (공격은 룰, 방어만 AI)
  - settings.yaml의 etf_rotation.ai_filter.enabled로 제어
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

# BAT 실행 대비 PYTHONPATH 안전장치
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.etf.orchestrator import ETFOrchestrator
from src.etf.data_bridge import load_all

logger = logging.getLogger(__name__)
OUTPUT_PATH = PROJECT_ROOT / "data" / "etf_rotation_result.json"
BLIND_DIR = PROJECT_ROOT / "data" / "etf_rotation_blind"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _save_blind_log(result: dict, data: dict):
    """블라인드 테스트 일별 시그널 로그 저장.

    data/etf_rotation_blind/YYYY-MM-DD.json 형태로 누적 기록.
    향후 백테스트 결과와 대조하여 시그널 일관성 검증용.
    """
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    log_entry = {
        "date": today,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 시장 상태
        "regime": data["regime"]["regime"],
        "kospi_close": data["regime"]["close"],
        "kospi_ma20_above": data["regime"]["ma20_above"],
        "kospi_ma60_above": data["regime"]["ma60_above"],
        "us_overnight_grade": data["us_overnight"]["grade"],
        "us_overnight_signal": data["us_overnight"]["signal"],
        # 입력 데이터 요약
        "momentum_sectors": len(data["momentum"]),
        "smart_money_count": len(data["smart_money"]),
        "supply_count": len(data["supply"]),
        "five_axis_score": data["five_axis_score"],
        # 오케스트레이터 결과
        "allocation": result.get("allocation"),
        "order_queue": result.get("order_queue", []),
        "sector_candidates": result.get("sector_candidates", []),
        "leverage_action": result.get("leverage_action"),
        "index_action": result.get("index_action"),
        "risk_level": result.get("risk_level"),
        "risk_summary": result.get("risk_summary"),
    }

    log_path = BLIND_DIR / f"{today}.json"
    log_path.write_text(
        json.dumps(log_entry, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"🔬 블라인드 로그 저장: {log_path}")

    # 누적 인덱스 업데이트 (간단한 날짜 목록)
    index_path = BLIND_DIR / "_index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_data = {"start_date": today, "logs": []}
    if today not in index_data["logs"]:
        index_data["logs"].append(today)
    index_data["total_days"] = len(index_data["logs"])
    index_data["last_updated"] = today
    index_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_ai_filter_config() -> dict:
    """settings.yaml에서 AI 필터 설정 로드."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("etf_rotation", {}).get("ai_filter", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="ETF 3축 로테이션 일일 실행")
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 발송 안 함")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 안 함")
    parser.add_argument("--blind-test", action="store_true",
                        help="블라인드 테스트 모드 (일별 시그널 로그 누적)")
    parser.add_argument("--no-ai", action="store_true",
                        help="AI 필터 비활성화 (룰베이스만 실행)")
    args = parser.parse_args()

    send_telegram = not (args.dry_run or args.no_telegram)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    mode_tag = " [🔬 블라인드 테스트]" if args.blind_test else ""
    print(f"\n🚀 ETF 3축 로테이션 시작{mode_tag} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 1. 데이터 로드 (방탄 브릿지) ----
    print("\n📂 데이터 로드 중 (방탄 fallback 적용)...")
    data = load_all()

    kospi = data["regime"]
    us = data["us_overnight"]
    print(f"  📊 KOSPI 레짐: {kospi['regime']} (종가 {kospi['close']:,.0f})")
    print(f"  📊 모멘텀 섹터: {len(data['momentum'])}개")
    print(f"  📊 Smart Money: {len(data['smart_money'])}개")
    print(f"  📊 수급 데이터: {len(data['supply'])}개")
    print(f"  📊 US Overnight: {us['grade']}등급 ({us['signal']})")
    print(f"  📊 레버리지 5축: {data['five_axis_score']:.0f}점")
    if data["individual_sectors"]:
        print(f"  📊 개별주 섹터: {data['individual_sectors']}")

    # ---- 2. 오케스트레이터 실행 ----
    orchestrator = ETFOrchestrator()
    result = orchestrator.run(
        regime=kospi["regime"],
        kospi_ma20_above=kospi["ma20_above"],
        kospi_ma60_above=kospi["ma60_above"],
        momentum_data=data["momentum"],
        smart_money_data=data["smart_money"],
        supply_data=data["supply"],
        us_overnight=data["us_overnight"],
        five_axis_score=data["five_axis_score"],
        individual_stock_sectors=data["individual_sectors"],
    )

    # ---- 2.5. AI 필터 레이어 ----
    ai_filter_result = None
    ai_cfg = _load_ai_filter_config()
    ai_enabled = ai_cfg.get("enabled", False) and not args.no_ai
    order_queue = result.get("order_queue", [])

    if ai_enabled and order_queue:
        print("\n🧠 AI 필터 레이어 실행 중...")
        try:
            from src.etf.ai_filter import apply_ai_filter, build_ai_telegram_section

            ai_filter_result = apply_ai_filter(
                order_queue=order_queue,
                regime=kospi["regime"],
                allocation=result.get("allocation", {}),
                regime_data=kospi,
                us_data=us,
                model=ai_cfg.get("model", "claude-sonnet-4-5-20250929"),
            )

            stats = ai_filter_result["stats"]
            print(f"  🧠 AI 판정: PASS {stats['pass']} / KILL {stats['kill']} / HOLD {stats['hold']}")

            # KILL된 주문 출력
            for k in ai_filter_result.get("killed_orders", []):
                print(f"  🚫 KILL: {k['name']} — {k.get('ai_reason', '')}")
            for h in ai_filter_result.get("held_orders", []):
                print(f"  ⏸️ HOLD: {h['name']} — {h.get('ai_reason', '')}")
            for w in ai_filter_result.get("warnings", []):
                print(f"  ⚠️ WARNING: {w}")

            # AI 필터 통과한 주문으로 교체
            result["order_queue_original"] = order_queue
            result["order_queue"] = ai_filter_result["filtered_queue"]
            result["ai_filter"] = {
                "stats": stats,
                "killed": ai_filter_result["killed_orders"],
                "held": ai_filter_result["held_orders"],
                "warnings": ai_filter_result.get("warnings", []),
                "market_assessment": ai_filter_result["ai_result"].get("market_assessment", ""),
                "risk_level": ai_filter_result["ai_result"].get("risk_level", ""),
            }

        except Exception as e:
            print(f"  ⚠️ AI 필터 오류 — 전체 PASS 처리: {e}")
            logger.exception("AI 필터 실행 실패")
    elif not ai_enabled:
        print("\n🧠 AI 필터: 비활성화 (settings.yaml 또는 --no-ai)")

    # ---- 3. 결과 JSON 저장 ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "telegram_report"}
    OUTPUT_PATH.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n💾 결과 저장: {OUTPUT_PATH}")

    # ---- 3.5. 블라인드 테스트 시그널 로그 ----
    if args.blind_test:
        _save_blind_log(result, data)

    # ---- 4. 텔레그램 발송 ----
    if send_telegram:
        try:
            from src.telegram_sender import send_message
            report = result.get("telegram_report", "")
            if report:
                # AI 필터 섹션 추가
                if ai_filter_result:
                    from src.etf.ai_filter import build_ai_telegram_section
                    ai_section = build_ai_telegram_section(ai_filter_result)
                    report += "\n" + ai_section

                # 블라인드 테스트 모드면 태그 추가
                if args.blind_test:
                    report = "🔬 [블라인드 테스트 — 관찰 전용]\n\n" + report
                ok = send_message(report)
                if ok:
                    print("📨 텔레그램 발송 완료")
                else:
                    print("⚠️ 텔레그램 발송 실패")
            else:
                print("⚠️ 리포트 내용 없음 — 텔레그램 스킵")
        except Exception as e:
            print(f"⚠️ 텔레그램 발송 오류: {e}")
    else:
        print("📭 텔레그램 발송 스킵 (--dry-run / --no-telegram)")

    print(f"\n✅ ETF 3축 로테이션 완료{mode_tag} — {datetime.now().strftime('%H:%M:%S')}")
    return result


if __name__ == "__main__":
    main()
