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
import os
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

    # supply_surge TOP 5 (5/18 3번 작업)
    supply_surges = []
    try:
        from src.use_cases.supply_surge_advisor import fetch_recent_supply_surge

        supply_surges = fetch_recent_supply_surge(top_n=5)
    except Exception as e:
        logger.warning("supply_surge SELECT 실패: %s", e)

    # vwap_monitor + intraday_eye (5/18 4번 작업)
    vwap_state = {"dips": [], "overheats": []}
    eye_events = []
    eye_counts = {}
    try:
        from src.use_cases.vwap_eye_advisor import (
            get_vwap_state, get_recent_eye_events, count_eye_per_ticker,
        )

        vwap_state = get_vwap_state()
        eye_events = get_recent_eye_events(minutes=120)
        eye_counts = count_eye_per_ticker(eye_events)
    except Exception as e:
        logger.warning("vwap+eye SELECT 실패: %s", e)

    # 막내 intraday_signals (5/18 5번 작업, 5/19 막내 가동 후 자동 활성)
    intraday_signals = []
    blocked_tickers = set()
    try:
        from src.use_cases.intraday_signals_advisor import (
            fetch_intraday_signals, get_nega_blocked_tickers,
        )

        intraday_signals = fetch_intraday_signals(min_impact=70)
        blocked_tickers = get_nega_blocked_tickers(intraday_signals)
    except Exception as e:
        logger.warning("intraday_signals SELECT 실패: %s", e)

    # ETF 추천 (5/18 1번 작업 — 사장님 순차 진행)
    etf_rec = None
    try:
        from src.use_cases.etf_advisor import suggest_etf_position

        # regime 미리 판정 (snap 완성 전이라 임시)
        market_str = intraday.get("strength_mean", 100) if intraday.get("db_exists") else 100
        inv_str = 100
        for r in intraday.get("top5", []):
            if r.get("code") == "252670":
                inv_str = r.get("avg_strength", 100)
                break
        # 임시 regime (정식은 determine_regime() 호출 시점)
        tmp_regime = "MILD_BULL" if market_str >= 100 else ("CAUTION" if inv_str > 120 else "NEUTRAL")
        etf_rec = suggest_etf_position(tmp_regime, inv_str, market_str)
    except Exception as e:
        logger.warning("ETF advisor 실패: %s", e)

    snapshot = {
        "snapshot_at": now.isoformat(),
        "date": today,
        "time": hhmm,
        "tomorrow_picks_top9": pick_records,
        "top1_weekly_8": weekly,
        "top1_monthly_6": monthly,
        "intraday_strength_summary": intraday,
        "etf_recommendation": {
            "action": etf_rec.action if etf_rec else None,
            "ticker": etf_rec.ticker if etf_rec else None,
            "name": etf_rec.name if etf_rec else None,
            "grade": etf_rec.grade if etf_rec else None,
            "reasoning": etf_rec.reasoning if etf_rec else None,
            "size_won": etf_rec.suggested_size_won if etf_rec else 0,
        } if etf_rec else None,
        "supply_surge_top5": supply_surges,  # 3번 작업: 외인+기관 동반 매수
        "vwap_dips_top3": vwap_state["dips"][:3],  # 4번 작업: VWAP 눌림
        "vwap_overheats_top3": vwap_state["overheats"][:3],
        "eye_event_counts": eye_counts,  # 종목별 EYE 알림 횟수 (HPSP 4번 = 황금 표준)
        "eye_top_ticker": max(eye_counts, key=eye_counts.get) if eye_counts else None,
        "eye_top_count": max(eye_counts.values()) if eye_counts else 0,
        "intraday_signals": intraday_signals,  # 5번 작업: 막내 시그널
        "blocked_tickers": list(blocked_tickers),  # 5번 작업: NEGA 차단 종목
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


def determine_regime(snap: dict) -> tuple[str, str]:
    """시장 강도 + 인버스 강도 + KOSPI 변동으로 regime + risk 판정."""
    intra = snap.get("intraday_strength_summary", {})
    if not intra.get("db_exists"):
        return "UNKNOWN", "MED"

    market_str = intra.get("strength_mean") or 100
    # KODEX 200선물인버스2X(252670) 강도 찾기 (top5에서)
    inverse_str = 0
    for r in intra.get("top5", []):
        if r.get("code") == "252670":
            inverse_str = r.get("avg_strength", 0)
            break

    # 간이 규칙 (5/18 학습 결과 반영, 5/22 캘리브레이션 후 보강 예정)
    if inverse_str >= 150 and market_str < 90:
        return "BEAR", "HIGH"
    if inverse_str >= 120 and market_str < 95:
        return "CAUTION", "MED"
    if market_str >= 110:
        return "MILD_BULL", "LOW"
    if market_str >= 120:
        return "STRONG_BULL", "LOW"
    return "NEUTRAL", "MED"


def insert_advisory_to_supabase(snap: dict) -> int | None:
    """quant_bot_advisory INSERT — 동생 단타봇이 SELECT하는 형의 시장 진단."""
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        logger.warning("psycopg2 미설치 — advisory INSERT 스킵")
        return None

    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.warning("DATABASE_URL 미설정 — advisory INSERT 스킵")
        return None

    intra = snap.get("intraday_strength_summary", {})
    picks = snap.get("tomorrow_picks_top9", [])
    regime, risk_level = determine_regime(snap)

    market_str = intra.get("strength_mean")
    inverse_str = None
    inverse_buy_ratio = None
    for r in intra.get("top5", []):
        if r.get("code") == "252670":
            inverse_str = r.get("avg_strength")
            inverse_buy_ratio = r.get("buy_ratio_pct")
            break

    # KOSPI200(069500) 변동률 = picks에 없으므로 별도 측정 필요. 일단 None
    kospi_chg = None

    # 양봉 / 음봉 종목 카운트
    n_pos = sum(1 for p in picks if p["current"] > p["open"])
    n_total = len(picks)
    avg_chg = (
        sum(((p["current"] - p["open"]) / p["open"] * 100) for p in picks if p["open"] > 0) / n_total
        if n_total > 0
        else 0
    )

    top_pos_tickers = [
        p["ticker"] for p in sorted(picks, key=lambda x: x["current"] - x["open"], reverse=True)[:5]
    ]

    title = f"[자동 advisory] {snap['time']} 시장 강도 {market_str} / 인버스 {inverse_str} ({regime})"
    body = (
        f"5/18 자동 스냅샷 #{snap['time']}. "
        f"강력포착 TOP {n_total}: 평균 {avg_chg:+.2f}% (양봉 {n_pos}/{n_total}). "
        f"시장 매크로: 강도 평균 {market_str}, 중앙 {intra.get('strength_median')}, "
        f"인버스ETF 252670 강도 {inverse_str} (매수비율 {inverse_buy_ratio}%). "
        f"동생 단타봇: regime={regime} risk={risk_level} 참고하여 진입 결정."
    )

    reasoning = {
        "market_strength_mean": market_str,
        "market_strength_median": intra.get("strength_median"),
        "inverse_etf_strength": inverse_str,
        "inverse_etf_buy_ratio": inverse_buy_ratio,
        "etf_recommendation": snap.get("etf_recommendation"),  # 1번 작업: ETF 추천 통합
        "supply_surge_top5": snap.get("supply_surge_top5", []),  # 3번 작업: 외인+기관 동반 매수
        "vwap_dips_top3": snap.get("vwap_dips_top3", []),  # 4번 작업: VWAP 눌림
        "vwap_overheats_top3": snap.get("vwap_overheats_top3", []),  # 4번 작업: VWAP 과열
        "eye_event_counts": snap.get("eye_event_counts", {}),  # 4번 작업: EYE 알림 횟수
        "eye_top_ticker": snap.get("eye_top_ticker"),  # 4번 작업: EYE 황금 표준
        "intraday_signals_count": len(snap.get("intraday_signals", [])),  # 5번: 막내 시그널 수
        "blocked_tickers": snap.get("blocked_tickers", []),  # 5번: NEGA 차단 종목
        "top9_avg_chg_pct": round(avg_chg, 2),
        "top9_positive_count": n_pos,
        "top9_total": n_total,
        "top9_records": [
            {
                "ticker": p["ticker"],
                "name": p["name"],
                "chg_pct": round((p["current"] - p["open"]) / p["open"] * 100, 2) if p["open"] > 0 else 0,
                "program_ntby": p["program_ntby"],
                "vol_ratio_pct": p["vol_ratio_pct"],
            }
            for p in picks
        ],
    }

    try:
        con = psycopg2.connect(url, connect_timeout=10)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO quant_bot_advisory
              (advisory_date, advisory_time, msg_type, severity, target_bot,
               market_regime, market_strength_avg, inverse_etf_strength,
               inverse_etf_buy_ratio, kospi_chg_pct, risk_level,
               title, body, related_tickers, alert_codes, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                snap["date"],
                snap["time"],
                "SNAPSHOT" if "11:00" not in snap["time"] and "13:30" not in snap["time"] else "ADVICE",
                "INFO" if regime in ("MILD_BULL", "NEUTRAL", "STRONG_BULL") else "WARN",
                "scalper",
                regime,
                market_str,
                inverse_str,
                inverse_buy_ratio,
                kospi_chg,
                risk_level,
                title,
                body,
                top_pos_tickers,
                ["SNAPSHOT-AUTO"],
                Json(reasoning),
            ),
        )
        new_id = cur.fetchone()[0]
        con.commit()
        con.close()
        return new_id
    except Exception as e:
        logger.error("advisory INSERT 실패: %s", e)
        return None


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

    # ETF 추천 (1번 작업, 5/18 신규)
    etf = snap.get("etf_recommendation")
    if etf and etf.get("action"):
        if etf["action"] in ("HOLD", "CASH_UP"):
            lines.append(f"📦 ETF: {etf['action']} — {etf['reasoning']}")
        else:
            grade_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "WATCH": "🟠"}.get(etf.get("grade"), "⚪")
            lines.append(
                f"📦 ETF {grade_emoji} {etf['grade']} {etf['name']}({etf['ticker']}) — {etf['reasoning'][:40]}"
            )

    # 수급 폭발 (3번 작업, 5/18 신규)
    surges = snap.get("supply_surge_top5", [])
    if surges:
        top = surges[0]
        lines.append(
            f"💎 수급폭발 TOP: {top['name']}({top['ticker']}) "
            f"[{top['supply_type']} {top['final_score']:.0f}점]"
        )

    # VWAP 눌림 / 과열 (4번 작업, 5/18 신규)
    dips = snap.get("vwap_dips_top3", [])
    overheats = snap.get("vwap_overheats_top3", [])
    if dips:
        top_dip = dips[0]
        lines.append(f"⬇️ VWAP 눌림 TOP: {top_dip['name']}({top_dip['ticker']}) {top_dip['vwap_dev_pct']:+.1f}%")
    if overheats:
        top_oh = overheats[0]
        lines.append(f"⬆️ VWAP 과열 TOP: {top_oh['name']}({top_oh['ticker']}) {top_oh['vwap_dev_pct']:+.1f}%")

    # EYE 황금 표준 (4번 작업, 5/18 신규)
    eye_top = snap.get("eye_top_ticker")
    eye_n = snap.get("eye_top_count", 0)
    if eye_top and eye_n >= 2:
        lines.append(f"✨ EYE 황금 표준: {eye_top} {eye_n}번 알림 (최근 2h)")

    # 막내 intraday_signals (5번 작업, 5/18 신규)
    sigs = snap.get("intraday_signals", [])
    blocked = snap.get("blocked_tickers", [])
    if sigs:
        top = sigs[0]
        emoji = {"NEGA": "🔴", "POSI": "🟢", "NEUT": "⚪"}.get(top["sentiment"], "⚪")
        lines.append(f"🤖 막내 {emoji} {top['sentiment']} {top['impact_score']}: {top['title'][:40]}")
    if blocked:
        lines.append(f"🚫 NEGA 차단: {len(blocked)}건 ({', '.join(blocked[:3])})")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--no-tg", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--no-advisory", action="store_true", help="Supabase advisory INSERT 안 함")
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

    # Supabase advisory INSERT (5/18 추가, 동생 단타봇이 SELECT)
    if not args.no_advisory:
        advisory_id = insert_advisory_to_supabase(snap)
        if advisory_id:
            print(f"[ADVISORY] id={advisory_id} INSERT 성공 — 동생 SELECT 가능")
        else:
            print("[ADVISORY] INSERT 스킵 또는 실패")

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
