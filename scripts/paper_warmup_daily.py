"""§13 워밍업 v2 — 자비스 추천 종목 일일 실전 추적 (2026-05-18 신규)

배경: 5/18 1차 클 결과 — 자비스 강력포착 TOP 9 시초가→09:15 평균 -3.24% / 양봉 1/9
원인: 시장 레짐 매크로 보정 부재 (5/15 -6.12% 폭락 흡수 첫날을 시그널이 못 봄)
대책: 5/18~5/26 9일치 매일 추적 데이터 누적 → 5/27 출격 전 §13-3 임계값 + 시장 보정 보강

흐름:
  09:05 (--open): tomorrow_picks 강력포착 TOP N → 시초가 기록 + reflection JSON 생성
  15:30 (--close): 종가 기록 + 당일 적중률 (양봉 여부) 계산
  D+1 16:30 (--label-d1): D+1 종가 → 익일 적중률 라벨링
  D+3, D+5도 동일 (BAT-D 이후 수동 또는 cron)

산출:
  data/reflection/warmup_{YYYYMMDD}.json
  텔레그램 [WARMUP] 알림 (open + close 시점)

Usage:
  python scripts/paper_warmup_daily.py --open --top 9
  python scripts/paper_warmup_daily.py --close
  python scripts/paper_warmup_daily.py --label-d1 --date 2026-05-18
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter  # noqa: E402

TOMORROW_PICKS_PATH = PROJECT_ROOT / "data" / "tomorrow_picks.json"
REFLECTION_DIR = PROJECT_ROOT / "data" / "reflection"
DEFAULT_GRADE_FILTER = "강력 포착"

logger = logging.getLogger(__name__)


def fetch_picks(top_n: int = 9, grade_filter: str = DEFAULT_GRADE_FILTER) -> list[dict]:
    """tomorrow_picks.json에서 grade 필터 → 상위 top_n 종목 추출."""
    if not TOMORROW_PICKS_PATH.exists():
        logger.error("tomorrow_picks.json 없음: %s", TOMORROW_PICKS_PATH)
        return []

    data = json.loads(TOMORROW_PICKS_PATH.read_text(encoding="utf-8"))
    picks = data.get("picks", [])
    filtered = [p for p in picks if p.get("grade") == grade_filter][:top_n]
    return filtered


def fetch_ohlcv(broker, ticker: str) -> dict | None:
    """KIS broker.fetch_price 호출 → OHLCV + 변동률 dict."""
    try:
        resp = broker.fetch_price(ticker)
        out = resp.get("output", {}) if resp else {}
        return {
            "ticker": ticker,
            "open": int(out.get("stck_oprc", 0) or 0),
            "high": int(out.get("stck_hgpr", 0) or 0),
            "low": int(out.get("stck_lwpr", 0) or 0),
            "current": int(out.get("stck_prpr", 0) or 0),
            "prdy_ctrt": float(out.get("prdy_ctrt", 0) or 0),
            "acml_vol": int(out.get("acml_vol", 0) or 0),
            "fetched_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning("fetch_price 실패 %s: %s", ticker, e)
        return None


def reflection_path(date: str) -> Path:
    REFLECTION_DIR.mkdir(parents=True, exist_ok=True)
    return REFLECTION_DIR / f"warmup_{date.replace('-', '')}.json"


def cmd_open(top_n: int, send_tg: bool) -> int:
    """09:05 시초가 기록 + reflection 신규 생성."""
    today = datetime.now().strftime("%Y-%m-%d")
    picks = fetch_picks(top_n=top_n)
    if not picks:
        print("강력 포착 후보 0건 — tomorrow_picks.json 확인 필요")
        return 1

    adp = KisStockDataAdapter()
    broker = adp.broker

    records = []
    for p in picks:
        tk = p.get("ticker", "")
        nm = p.get("name", tk)
        px = fetch_ohlcv(broker, tk)
        if px is None:
            continue
        records.append(
            {
                "ticker": tk,
                "name": nm,
                "grade": p.get("grade", ""),
                "rank": picks.index(p) + 1,
                "open_session": px,  # 09:05 측정 (시초가 + 5분 진행)
            }
        )

    reflection = {
        "date": today,
        "mode": "open",
        "top_n": top_n,
        "grade_filter": DEFAULT_GRADE_FILTER,
        "n_records": len(records),
        "records": records,
        "created_at": datetime.now().isoformat(),
    }
    out_path = reflection_path(today)
    out_path.write_text(json.dumps(reflection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OPEN] {today} 시초가 기록 — {len(records)}건 → {out_path.name}")

    if send_tg:
        try:
            from src.telegram_sender import send_message

            msg_lines = [f"[WARMUP] {today} 시초가 기록 ({len(records)}건)"]
            for r in records[:10]:
                op = r["open_session"]["open"]
                cur = r["open_session"]["current"]
                ch = ((cur - op) / op * 100) if op > 0 else 0
                msg_lines.append(f"  {r['rank']}. {r['name']} {op:,}→{cur:,} ({ch:+.2f}%)")
            send_message("\n".join(msg_lines))
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)

    return 0


def cmd_close(send_tg: bool) -> int:
    """15:30 종가 기록 + 당일 적중률 계산 + reflection 업데이트."""
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = reflection_path(today)
    if not out_path.exists():
        print(f"[CLOSE] {today} reflection 파일 없음 — --open 먼저 실행 필요")
        return 1

    reflection = json.loads(out_path.read_text(encoding="utf-8"))
    adp = KisStockDataAdapter()
    broker = adp.broker

    n_hits = 0
    for rec in reflection.get("records", []):
        px_close = fetch_ohlcv(broker, rec["ticker"])
        if px_close is None:
            continue
        rec["close_session"] = px_close
        op = rec["open_session"]["open"]
        cl = px_close["current"]
        rec["intraday_chg_pct"] = ((cl - op) / op * 100) if op > 0 else 0
        rec["intraday_hit"] = rec["intraday_chg_pct"] > 0
        if rec["intraday_hit"]:
            n_hits += 1

    n = len(reflection.get("records", []))
    reflection["mode"] = "close"
    reflection["intraday_accuracy"] = (n_hits / n) if n > 0 else 0.0
    reflection["intraday_avg_chg_pct"] = (
        sum(r.get("intraday_chg_pct", 0) for r in reflection["records"]) / n if n > 0 else 0.0
    )
    reflection["closed_at"] = datetime.now().isoformat()

    out_path.write_text(json.dumps(reflection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[CLOSE] {today} — 적중률 {reflection['intraday_accuracy']:.1%} ({n_hits}/{n}), "
        f"평균 {reflection['intraday_avg_chg_pct']:+.2f}%"
    )

    if send_tg:
        try:
            from src.telegram_sender import send_message

            msg = (
                f"[WARMUP CLOSE] {today}\n"
                f"적중률 {reflection['intraday_accuracy']:.1%} ({n_hits}/{n})\n"
                f"평균 변동 {reflection['intraday_avg_chg_pct']:+.2f}%"
            )
            send_message(msg)
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)

    return 0


def cmd_label_d1(date: str) -> int:
    """D+1 종가 라벨링 (다음 거래일 16:30 이후 실행)."""
    src_path = reflection_path(date)
    if not src_path.exists():
        print(f"reflection 없음: {src_path.name}")
        return 1

    reflection = json.loads(src_path.read_text(encoding="utf-8"))
    adp = KisStockDataAdapter()
    broker = adp.broker

    n_hits = 0
    for rec in reflection.get("records", []):
        px = fetch_ohlcv(broker, rec["ticker"])
        if px is None:
            continue
        rec["d1_close"] = px["current"]
        op = rec["open_session"]["open"]
        rec["d1_chg_pct"] = ((px["current"] - op) / op * 100) if op > 0 else 0
        rec["d1_hit"] = rec["d1_chg_pct"] > 0
        if rec["d1_hit"]:
            n_hits += 1

    n = len(reflection.get("records", []))
    reflection["d1_accuracy"] = (n_hits / n) if n > 0 else 0.0
    reflection["d1_labeled_at"] = datetime.now().isoformat()
    src_path.write_text(json.dumps(reflection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[D+1 LABEL] {date} — D+1 적중률 {reflection['d1_accuracy']:.1%} ({n_hits}/{n})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="§13 워밍업 일일 추적")
    parser.add_argument("--open", action="store_true", help="09:05 시초가 기록")
    parser.add_argument("--close", action="store_true", help="15:30 종가 + 당일 적중률")
    parser.add_argument("--label-d1", action="store_true", help="D+1 라벨링 (다음 거래일)")
    parser.add_argument("--date", help="--label-d1 대상 날짜 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=9, help="강력포착 TOP N (기본 9)")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 안 함")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    send_tg = not args.no_telegram

    if args.open:
        return cmd_open(args.top, send_tg)
    if args.close:
        return cmd_close(send_tg)
    if args.label_d1:
        if not args.date:
            print("--label-d1 에는 --date 필수")
            return 1
        return cmd_label_d1(args.date)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
