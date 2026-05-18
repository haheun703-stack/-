"""§13 워밍업 v2 — 장중 스냅샷 학습 (2026-05-18 신규)

배경: 5/18 09:34 KIS 실시간 클 결과 학습 시작점
- 인버스ETF 강도 폭발 = 시장 약세 베팅
- HPSP 강세 = 1월/4월 2단 폭발 후 5/18 반등 시그널
- 자비스 TOP 9 적중률 11% = 시장 매크로 보정 부재

스냅샷 내용 (매 호출 시):
  1. 자비스 tomorrow_picks 강력포착 TOP N 현재가 + 프로그램순매수 + 외인보유율
  2. intraday_minute DB 25 종목 강도 분포 (mean/median/TOP 5)
  3. KOSPI 거시 (sector 추이는 별도)
  4. 주봉/월봉 컨텍스트 (TOP 종목)

저장: data/snapshots/{YYYYMMDD}/{HHMM}_session.json
텔레그램: [SNAPSHOT] 알림 (옵션, 핵심 변화 시점만)

Usage:
  python scripts/snapshot_session.py             # 1회 즉시 실행
  python scripts/snapshot_session.py --no-tg     # 텔레그램 OFF
  python scripts/snapshot_session.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.kis_stock_data_adapter import KisStockDataAdapter  # noqa: E402

TOMORROW_PICKS = PROJECT_ROOT / "data" / "tomorrow_picks.json"
SNAPSHOT_BASE = PROJECT_ROOT / "data" / "snapshots"
INTRADAY_DB_PATTERN = "data/intraday/intraday_minute_{date}.db"
DEFAULT_TOP_N = 9
DEFAULT_GRADE = "강력 포착"

logger = logging.getLogger(__name__)


def fetch_top_picks(top_n: int = DEFAULT_TOP_N, grade: str = DEFAULT_GRADE) -> list[dict]:
    if not TOMORROW_PICKS.exists():
        return []
    data = json.loads(TOMORROW_PICKS.read_text(encoding="utf-8"))
    return [p for p in data.get("picks", []) if p.get("grade") == grade][:top_n]


def fetch_price_extended(broker, ticker: str) -> dict | None:
    """가격 + 프로그램 순매수 + 외인 보유율 + 거래량 회전율."""
    try:
        resp = broker.fetch_price(ticker)
        out = resp.get("output", {}) if resp else {}
        return {
            "ticker": ticker,
            "current": int(out.get("stck_prpr", 0) or 0),
            "open": int(out.get("stck_oprc", 0) or 0),
            "high": int(out.get("stck_hgpr", 0) or 0),
            "low": int(out.get("stck_lwpr", 0) or 0),
            "prdy_ctrt": float(out.get("prdy_ctrt", 0) or 0),
            "vol_ratio_pct": float(out.get("prdy_vrss_vol_rate", 0) or 0),
            "program_ntby": int(out.get("pgtr_ntby_qty", 0) or 0),
            "foreign_ehrt_pct": float(out.get("hts_frgn_ehrt", 0) or 0),
            "acml_vol": int(out.get("acml_vol", 0) or 0),
            "vol_tnrt": float(out.get("vol_tnrt", 0) or 0),
            "w52_hgpr_dist_pct": float(out.get("w52_hgpr_vrss_prpr_ctrt", 0) or 0),
            "d250_hgpr_dist_pct": float(out.get("d250_hgpr_vrss_prpr_rate", 0) or 0),
        }
    except Exception as e:
        logger.warning("fetch_price 실패 %s: %s", ticker, e)
        return None


def fetch_weekly_recent(broker, ticker: str, n: int = 8) -> list[dict]:
    """최근 N주봉."""
    try:
        resp = broker.fetch_ohlcv(ticker, timeframe="W", end_day="", adj_price=True)
        out = resp.get("output2", [])[:n]
        bars = []
        for r in out:
            o = int(r.get("stck_oprc", 0) or 0)
            c = int(r.get("stck_clpr", 0) or 0)
            bars.append(
                {
                    "date": r.get("stck_bsop_date", ""),
                    "open": o,
                    "high": int(r.get("stck_hgpr", 0) or 0),
                    "low": int(r.get("stck_lwpr", 0) or 0),
                    "close": c,
                    "volume": int(r.get("acml_vol", 0) or 0),
                    "chg_pct": ((c - o) / o * 100) if o > 0 else 0,
                }
            )
        return bars
    except Exception as e:
        logger.warning("fetch_weekly 실패 %s: %s", ticker, e)
        return []


def fetch_monthly_recent(broker, ticker: str, n: int = 6) -> list[dict]:
    """최근 N월봉."""
    try:
        resp = broker.fetch_ohlcv(ticker, timeframe="M", end_day="", adj_price=True)
        out = resp.get("output2", [])[:n]
        bars = []
        for r in out:
            o = int(r.get("stck_oprc", 0) or 0)
            c = int(r.get("stck_clpr", 0) or 0)
            bars.append(
                {
                    "date": r.get("stck_bsop_date", ""),
                    "open": o,
                    "close": c,
                    "volume": int(r.get("acml_vol", 0) or 0),
                    "chg_pct": ((c - o) / o * 100) if o > 0 else 0,
                }
            )
        return bars
    except Exception as e:
        logger.warning("fetch_monthly 실패 %s: %s", ticker, e)
        return []


def collect_intraday_strength(today: str) -> dict:
    """intraday_minute DB에서 25 종목 강도 분포 집계."""
    db_path = PROJECT_ROOT / INTRADAY_DB_PATTERN.format(date=today.replace("-", ""))
    if not db_path.exists():
        return {"db_exists": False, "path": str(db_path)}

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # 종목별 강도 평균 + 매수/매도 체결 + 거래량
    cur.execute(
        """
        SELECT code, AVG(strength_avg) as avg_str, SUM(buy_count) as buy,
               SUM(sell_count) as sell, SUM(volume) as v, MAX(minute) as last_min
        FROM intraday_minute GROUP BY code ORDER BY avg_str DESC
        """
    )
    rows = cur.fetchall()

    ranks = []
    for code, avg, buy, sell, vol, last_min in rows:
        total = (buy or 0) + (sell or 0)
        buy_ratio = (buy / total * 100) if total > 0 else 0
        ranks.append(
            {
                "code": code,
                "avg_strength": round(avg or 0, 2),
                "buy_count": buy or 0,
                "sell_count": sell or 0,
                "buy_ratio_pct": round(buy_ratio, 1),
                "volume": vol or 0,
                "last_minute": last_min,
            }
        )

    strengths = [r["avg_strength"] for r in ranks if r["avg_strength"] > 0]
    summary = {
        "db_exists": True,
        "n_codes": len(ranks),
        "strength_min": min(strengths) if strengths else None,
        "strength_max": max(strengths) if strengths else None,
        "strength_median": round(statistics.median(strengths), 2) if strengths else None,
        "strength_mean": round(statistics.mean(strengths), 2) if strengths else None,
        "top5": ranks[:5],
        "bottom5_buy_ratio": sorted(ranks, key=lambda x: x["buy_ratio_pct"])[:5],
    }
    con.close()
    return summary


def take_snapshot(top_n: int = DEFAULT_TOP_N) -> dict:
    """1회 스냅샷 생성."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    hhmm = now.strftime("%H:%M")

    adp = KisStockDataAdapter()
    broker = adp.broker

    picks = fetch_top_picks(top_n=top_n)
    pick_records = []
    for i, p in enumerate(picks, 1):
        tk = p.get("ticker", "")
        nm = p.get("name", tk)
        px = fetch_price_extended(broker, tk)
        if px:
            px["name"] = nm
            px["rank"] = i
            px["grade"] = p.get("grade", "")
            pick_records.append(px)

    # TOP 1 종목만 주봉/월봉 추가 (API 부하 절감)
    weekly = fetch_weekly_recent(broker, pick_records[0]["ticker"], n=8) if pick_records else []
    monthly = fetch_monthly_recent(broker, pick_records[0]["ticker"], n=6) if pick_records else []

    intraday = collect_intraday_strength(today)

    snapshot = {
        "snapshot_at": now.isoformat(),
        "date": today,
        "time": hhmm,
        "tomorrow_picks_top9": pick_records,
        "top1_weekly_8": weekly,
        "top1_monthly_6": monthly,
        "intraday_strength_summary": intraday,
    }
    return snapshot


def save_snapshot(snap: dict) -> Path:
    date = snap["date"].replace("-", "")
    time_tag = snap["time"].replace(":", "")
    out_dir = SNAPSHOT_BASE / date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{time_tag}_session.json"
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def format_telegram(snap: dict) -> str:
    picks = snap.get("tomorrow_picks_top9", [])
    intra = snap.get("intraday_strength_summary", {})

    lines = [f"[SNAPSHOT] {snap['time']} 강력포착 TOP {len(picks)}"]
    for p in picks[:5]:
        ch = ((p["current"] - p["open"]) / p["open"] * 100) if p["open"] > 0 else 0
        sign = "🟢" if ch > 0 else "🔴"
        prog = "+" if p["program_ntby"] > 0 else ""
        lines.append(
            f"  {sign} {p['name'][:12]} {p['current']:,} ({ch:+.1f}%) "
            f"프로그램 {prog}{p['program_ntby']:,}"
        )
    if intra.get("db_exists"):
        lines.append(
            f"📊 시장 강도: 평균 {intra.get('strength_mean')}, "
            f"중앙 {intra.get('strength_median')} "
            f"(TOP {intra.get('top5', [{}])[0].get('code')} "
            f"강도 {intra.get('top5', [{}])[0].get('avg_strength')})"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    snap = take_snapshot(top_n=args.top)
    out_path = save_snapshot(snap)
    print(f"[SAVED] {out_path.name} ({out_path.stat().st_size:,} bytes)")
    print(format_telegram(snap))

    if not args.no_tg:
        try:
            from src.telegram_sender import send_message

            send_message(format_telegram(snap))
            print("[TG SENT]")
        except Exception as e:
            print(f"[TG FAIL] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
