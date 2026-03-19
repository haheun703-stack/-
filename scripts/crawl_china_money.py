"""
KRX 외국인 수급 수집 + 차이나머니 시그널 판정
==============================================
Phase 1: KIS API 외국인 합계 + EWY 프록시 분석
Phase 2: KRX SMILE 국적별 보유 (로그인 자동화 후)

사용법:
  python -u -X utf8 scripts/crawl_china_money.py           # 전체 유니버스
  python -u -X utf8 scripts/crawl_china_money.py --test     # 대형주 10종목만
  python -u -X utf8 scripts/crawl_china_money.py --no-telegram  # 텔레그램 OFF
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# PYTHONPATH 안전장치
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.krx_foreign_adapter import (
    ChinaMoneyAnalyzer,
    KRXForeignAdapter,
    load_universe_tickers,
    save_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_telegram_message(data: dict) -> str:
    """텔레그램 알림 메시지 생성."""
    summary = data.get("summary", {})
    signals = data.get("signals", [])
    top_buyers = data.get("top_foreign_buyers", [])

    lines = []
    lines.append("🇨🇳 [차이나머니 수급 감지]")
    lines.append("━━━━━━━━━━━━━━━━━━")

    # SURGE
    surge = [s for s in signals if s["signal"] == "SURGE"]
    if surge:
        lines.append("")
        lines.append("🔥 SURGE (대량 유입):")
        for s in surge:
            reasons = " | ".join(s.get("reasons", [])[:2])
            lines.append(f"  {s['name']}({s['ticker']}) — 점수 {s['score']}")
            lines.append(f"    외국인 5일: {s['foreign_net_5d']:+,}주 | z={s['foreign_zscore']:.1f}")
            if reasons:
                lines.append(f"    📌 {reasons}")

    # INFLOW
    inflow = [s for s in signals if s["signal"] == "INFLOW"]
    if inflow:
        lines.append("")
        lines.append("📊 INFLOW (유입 추세):")
        for s in inflow[:5]:
            lines.append(f"  {s['name']}({s['ticker']}) — 점수 {s['score']} | 5일 {s['foreign_net_5d']:+,}주")

    # SECTOR_FOCUS / WATCH
    focus = [s for s in signals if s["signal"] in ("SECTOR_FOCUS", "WATCH")]
    if focus:
        lines.append("")
        lines.append(f"👀 관심 종목: {len(focus)}개")
        for s in focus[:5]:
            lines.append(f"  {s['name']} — {s['signal']} (점수 {s['score']})")

    # 요약
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append(f"📈 스캔: {data['total_stocks']}종목")
    lines.append(f"  SURGE {summary.get('SURGE', 0)} | INFLOW {summary.get('INFLOW', 0)} | FOCUS {summary.get('SECTOR_FOCUS', 0)} | WATCH {summary.get('WATCH', 0)}")

    # TOP 외국인 순매수 종목
    if top_buyers:
        lines.append("")
        lines.append("📋 외국인 순매수 TOP 5 (5일):")
        for s in top_buyers[:5]:
            lines.append(f"  {s['name']}: {s['foreign_net_5d']:+,}주")

    # EWY 참고
    try:
        ewy_path = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
        if ewy_path.exists():
            ewy_data = json.loads(ewy_path.read_text(encoding="utf-8"))
            ewy = ewy_data.get("index_direction", {}).get("EWY", {})
            if ewy:
                lines.append("")
                lines.append(f"🌏 EWY: 1일 {ewy.get('ret_1d', 0):+.1f}% | 5일 {ewy.get('ret_5d', 0):+.1f}%")
    except Exception:
        pass

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="차이나머니 수급 수집 + 시그널")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (대형주 10종목)")
    parser.add_argument("--full", action="store_true", help="전체 1000+ 종목 (매우 느림)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 OFF")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("차이나머니 수급 감지 시작 (Phase 1: KIS API + EWY 프록시)")
    logger.info("=" * 60)

    # 1. 유니버스 로드
    if args.test:
        # 테스트: 대형주 10종목만
        tickers = dict(list(KRXForeignAdapter.LARGE_CAP_TICKERS.items())[:10])
        logger.info("테스트 모드: %d종목", len(tickers))
    elif args.full:
        tickers = load_universe_tickers(mode="full")
        logger.info("전체 모드: %d종목 (소요 ~2시간)", len(tickers))
    else:
        tickers = load_universe_tickers(mode="large")
        logger.info("대형주 모드: %d종목 (시총 TOP50)", len(tickers))

    # 2. 외국인 수급 수집
    adapter = KRXForeignAdapter(delay=0.12)
    snapshots = adapter.fetch_universe_flow(tickers)
    logger.info("수집 완료: %d종목", len(snapshots))

    if not snapshots:
        logger.error("수집된 데이터 없음 — 종료")
        return

    # 3. 시그널 분석
    analyzer = ChinaMoneyAnalyzer()
    signals = analyzer.analyze(snapshots)

    # 4. 저장
    data = save_signals(signals)

    # 5. 결과 출력
    summary = data.get("summary", {})
    logger.info("━━━━━━━━ 결과 ━━━━━━━━")
    logger.info("SURGE: %d | INFLOW: %d | SECTOR_FOCUS: %d | WATCH: %d | NORMAL: %d",
                summary.get("SURGE", 0), summary.get("INFLOW", 0),
                summary.get("SECTOR_FOCUS", 0), summary.get("WATCH", 0),
                summary.get("NORMAL", 0))

    non_normal = [s for s in signals if s.signal != "NORMAL"]
    for s in non_normal[:10]:
        logger.info("  [%s] %s(%s) — 점수 %d | 5일 %+d주 | z=%.1f | %s",
                     s.signal, s.name, s.ticker, s.score,
                     s.foreign_net_5d, s.foreign_zscore,
                     ", ".join(s.reasons[:2]))

    # 6. 텔레그램 알림
    if not args.no_telegram and (summary.get("SURGE", 0) > 0 or summary.get("INFLOW", 0) > 0):
        try:
            from src.telegram_sender import send_message
            msg = build_telegram_message(data)
            send_message(msg)
            logger.info("텔레그램 알림 전송 완료")
        except Exception as e:
            logger.warning("텔레그램 전송 실패: %s", e)
    # NOTE(2026-03-20): "특이 시그널 없음" 노이즈 메시지 제거.
    # SURGE/INFLOW가 있을 때만 위 블록에서 발송됨.

    logger.info("차이나머니 수급 감지 완료")


if __name__ == "__main__":
    main()
