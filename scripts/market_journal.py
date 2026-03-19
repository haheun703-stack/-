"""
Market Journal — 장중 기록 + 학습 + 주간/월간 보고

사용법:
  python scripts/market_journal.py              # 일간 기록 + 학습
  python scripts/market_journal.py --weekly     # 주간 보고서 강제 생성
  python scripts/market_journal.py --monthly    # 월간 보고서 강제 생성
  python scripts/market_journal.py --dry-run    # 텔레그램 비발송

실행 시점: BAT-D 24단계 직전 (~17:20)
  금요일 → 주간 보고서 자동 발송
  월말 영업일 → 월간 보고서 자동 발송
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

DATA_DIR = PROJECT_ROOT / "data"
JOURNAL_DIR = DATA_DIR / "market_journal"
DAILY_DIR = JOURNAL_DIR / "daily"
INSIGHTS_DIR = JOURNAL_DIR / "insights"
WEEKLY_DIR = JOURNAL_DIR / "weekly"
MONTHLY_DIR = JOURNAL_DIR / "monthly"
DB_PATH = JOURNAL_DIR / "journal.db"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
PARQUET_DIR = DATA_DIR / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("journal")


# ═══════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════

def _load_json(name: str) -> dict | list:
    path = DATA_DIR / name if not Path(name).is_absolute() else Path(name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_yaml() -> dict:
    p = PROJECT_ROOT / "config" / "settings.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_name_map() -> dict[str, str]:
    """stock_data_daily 폴더에서 ticker→name 맵 생성."""
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def _ensure_dirs():
    for d in [JOURNAL_DIR, DAILY_DIR, INSIGHTS_DIR, WEEKLY_DIR, MONTHLY_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _is_last_business_day(d: date) -> bool:
    """월말 영업일 판별 (주말 제외, 공휴일은 무시)."""
    next_day = d + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day.month != d.month


# ═══════════════════════════════════════════
#  SQLite 스키마
# ═══════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_summary (
    date TEXT PRIMARY KEY,
    kospi_close REAL, kospi_change_pct REAL,
    kosdaq_close REAL, kosdaq_change_pct REAL,
    regime TEXT,
    usd_krw REAL, vix REAL,
    foreign_net_bil REAL,
    institution_net_bil REAL,
    portfolio_return_pct REAL,
    eye_event_count INTEGER DEFAULT 0,
    buy_count INTEGER DEFAULT 0,
    sell_count INTEGER DEFAULT 0,
    realized_pnl INTEGER DEFAULT 0,
    market_event TEXT DEFAULT '',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_movers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, category TEXT,
    rank INTEGER,
    ticker TEXT, name TEXT,
    change_pct REAL, volume_ratio REAL,
    foreign_net_5d REAL, inst_net_5d REAL,
    sector TEXT DEFAULT '',
    reason_tags TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS daily_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, ticker TEXT, name TEXT,
    entry_date TEXT, entry_price REAL,
    current_price REAL, unrealized_pct REAL,
    today_change_pct REAL,
    days_held INTEGER,
    status TEXT DEFAULT 'HEALTHY'
);

CREATE TABLE IF NOT EXISTS learning_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, insight_type TEXT,
    content TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_movers_date ON daily_movers(date);
CREATE INDEX IF NOT EXISTS idx_holdings_date ON daily_holdings(date);
CREATE INDEX IF NOT EXISTS idx_insights_date ON learning_insights(date);
"""


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_SCHEMA)
    return conn


# ═══════════════════════════════════════════
#  9-1. 일간 기록 엔진
# ═══════════════════════════════════════════

def _collect_market_overview(today_str: str) -> dict:
    """시장 개요 수집: KOSPI/KOSDAQ + 레짐 + 환율 + VIX + 수급."""
    overview = {
        "date": today_str,
        "kospi_close": 0, "kospi_change_pct": 0,
        "kosdaq_close": 0, "kosdaq_change_pct": 0,
        "regime": "UNKNOWN",
        "usd_krw": 0, "vix": 0,
        "foreign_net_bil": 0, "institution_net_bil": 0,
        "market_event": "",
    }

    # KOSPI from kospi_index.csv
    kospi_csv = DATA_DIR / "kospi_index.csv"
    if kospi_csv.exists():
        try:
            df = pd.read_csv(kospi_csv, index_col=0, parse_dates=True)
            if len(df) >= 2:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                overview["kospi_close"] = round(float(last["close"]), 1)
                if prev["close"] > 0:
                    overview["kospi_change_pct"] = round(
                        (last["close"] / prev["close"] - 1) * 100, 2
                    )
        except Exception as e:
            log.warning("KOSPI CSV 로드 실패: %s", e)

    # KOSDAQ from pykrx (1 API call)
    try:
        from pykrx import stock as krx
        kq = krx.get_index_ohlcv_by_date(today_str.replace("-", ""),
                                          today_str.replace("-", ""), "2001")
        if not kq.empty:
            row = kq.iloc[-1]
            overview["kosdaq_close"] = round(float(row["종가"]), 1)
            overview["kosdaq_change_pct"] = round(float(row["등락률"]), 2)
    except Exception as e:
        log.warning("KOSDAQ pykrx 실패: %s", e)

    # 레짐 + 매크로
    regime = _load_json("regime_macro_signal.json")
    if regime:
        overview["regime"] = regime.get("current_regime", "UNKNOWN")
        overview["vix"] = regime.get("signals", {}).get("vix_level", 0)
        overview["usd_krw"] = regime.get("signals", {}).get("usdkrw_close", 0)

    # VIX fallback: parquet 마지막 행에서
    if not overview["vix"]:
        try:
            sample = next(PARQUET_DIR.glob("*.parquet"))
            df = pd.read_parquet(sample, columns=["vix_close"])
            if not df.empty:
                overview["vix"] = round(float(df.iloc[-1]["vix_close"]), 1)
        except Exception:
            pass

    # USD/KRW fallback
    if not overview["usd_krw"]:
        try:
            sample = next(PARQUET_DIR.glob("*.parquet"))
            df = pd.read_parquet(sample, columns=["usdkrw_close"])
            if not df.empty:
                overview["usd_krw"] = round(float(df.iloc[-1]["usdkrw_close"]), 1)
        except Exception:
            pass

    # 시장 외국인/기관 순매수 (pykrx)
    try:
        from pykrx import stock as krx
        dt = today_str.replace("-", "")
        inv = krx.get_market_trading_value_by_date(dt, dt, "KOSPI")
        if not inv.empty:
            row = inv.iloc[-1]
            # pykrx 열: 기관합계, 외국인합계, 개인 등 (단위: 원)
            for col in inv.columns:
                if "외국인" in col:
                    overview["foreign_net_bil"] = round(float(row[col]) / 1e8, 0)
                elif "기관" in col and "합계" in col:
                    overview["institution_net_bil"] = round(float(row[col]) / 1e8, 0)
    except Exception as e:
        log.warning("시장 수급 pykrx 실패: %s", e)

    # market_event from brain_decision or market_intelligence
    intel = _load_json("market_intelligence.json")
    if intel:
        mood = intel.get("mood", "")
        themes = intel.get("hot_themes", [])
        if mood or themes:
            overview["market_event"] = f"{mood} | {', '.join(themes[:3])}" if themes else mood

    return overview


def _collect_daily_movers(today_str: str, name_map: dict) -> dict:
    """급등/급락/거래량폭발/수급쏠림/이탈 TOP5 추출."""
    stocks = []

    for pq in PARQUET_DIR.glob("*.parquet"):
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 2:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2]

            change_pct = float(last.get("price_change", 0))
            if pd.isna(change_pct) or abs(change_pct) > 50:
                # NaN이거나 비정상 수치 → 직접 계산
                if prev["close"] > 0:
                    change_pct = (last["close"] / prev["close"] - 1) * 100
                else:
                    change_pct = 0
            # 여전히 비정상이면 스킵
            if abs(change_pct) > 50:
                continue

            vol = float(last.get("volume", 0))
            vol_ma20 = float(last.get("volume_ma20", vol)) if "volume_ma20" in df.columns else vol
            vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else 1.0

            foreign_5d = float(last.get("foreign_net_5d", 0)) if "foreign_net_5d" in df.columns else 0
            inst_5d = float(last.get("inst_net_5d", 0)) if "inst_net_5d" in df.columns else 0

            stocks.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "change_pct": round(change_pct, 2),
                "volume_ratio": round(vol_ratio, 2),
                "foreign_net_5d": round(foreign_5d, 0),
                "inst_net_5d": round(inst_5d, 0),
                "close": float(last.get("close", 0)),
            })
        except Exception:
            continue

    if not stocks:
        return {}

    # reason_tags 자동 판별
    for s in stocks:
        tags = []
        if s["foreign_net_5d"] > 0 and s["inst_net_5d"] > 0:
            tags.append("수급_쌍끌이")
        elif s["foreign_net_5d"] > 0:
            tags.append("외인_매수")
        elif s["inst_net_5d"] > 0:
            tags.append("기관_매수")
        if s["foreign_net_5d"] < 0 and s["inst_net_5d"] < 0:
            tags.append("수급_쌍매도")
        if s["volume_ratio"] >= 3.0:
            tags.append("비정상_거래량")
        if abs(s["change_pct"]) >= 5:
            tags.append("대폭등락")
        s["reason_tags"] = tags

    # 카테고리별 TOP5
    result = {}
    by_change = sorted(stocks, key=lambda x: x["change_pct"], reverse=True)
    result["급등_top5"] = by_change[:5]
    result["급락_top5"] = by_change[-5:][::-1]  # 가장 큰 하락부터

    by_vol = sorted(stocks, key=lambda x: x["volume_ratio"], reverse=True)
    result["거래량_폭발_top5"] = by_vol[:5]

    # 수급 쏠림: foreign + inst 합산
    for s in stocks:
        s["_flow_sum"] = s["foreign_net_5d"] + s["inst_net_5d"]
    by_flow = sorted(stocks, key=lambda x: x["_flow_sum"], reverse=True)
    result["수급_쏠림_top5"] = by_flow[:5]
    result["수급_이탈_top5"] = by_flow[-5:][::-1]

    # _flow_sum 제거
    for s in stocks:
        s.pop("_flow_sum", None)

    return result


def _collect_holdings(today_str: str, name_map: dict) -> dict:
    """보유 종목 일지 수집."""
    equity = _load_json("equity_tracker.json")
    portfolio_return = 0.0
    if equity:
        daily_log = equity.get("daily_log", [])
        if len(daily_log) >= 2:
            today_eq = daily_log[-1].get("equity", 0)
            prev_eq = daily_log[-2].get("equity", 0)
            if prev_eq > 0:
                portfolio_return = round((today_eq / prev_eq - 1) * 100, 2)

    holdings = []
    # portfolio_allocation.json에서 현재 보유 종목 확인
    alloc = _load_json("portfolio_allocation.json")
    if alloc:
        for strategy, info in alloc.get("allocations", {}).items():
            for s in info.get("stocks", []):
                ticker = s.get("ticker", "")
                entry_price = s.get("entry_price", s.get("close_at_pick", 0))

                # parquet에서 현재가 조회
                pq_path = PARQUET_DIR / f"{ticker}.parquet"
                current_price = 0
                today_change = 0
                if pq_path.exists():
                    try:
                        df = pd.read_parquet(pq_path, columns=["close", "price_change"])
                        if not df.empty:
                            current_price = float(df.iloc[-1]["close"])
                            today_change = float(df.iloc[-1].get("price_change", 0))
                    except Exception:
                        pass

                unrealized = 0
                if entry_price > 0 and current_price > 0:
                    unrealized = round((current_price / entry_price - 1) * 100, 2)

                entry_date = s.get("entry_date", "")
                days_held = 0
                if entry_date:
                    try:
                        ed = datetime.strptime(entry_date, "%Y-%m-%d").date()
                        days_held = (date.today() - ed).days
                    except Exception:
                        pass

                # 상태 판정
                status = "HEALTHY"
                if unrealized <= -5:
                    status = "DANGER"
                elif unrealized <= -2:
                    status = "WARNING"

                holdings.append({
                    "ticker": ticker,
                    "name": s.get("name", name_map.get(ticker, ticker)),
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pct": unrealized,
                    "today_change_pct": round(today_change, 2),
                    "days_held": days_held,
                    "status": status,
                    "strategy": strategy,
                })

    # EYE 이벤트
    eye_path = DATA_DIR / "eye_events" / f"{today_str}.json"
    eye_events = []
    if eye_path.exists():
        try:
            ed = json.loads(eye_path.read_text(encoding="utf-8"))
            eye_events = ed.get("events", [])
        except Exception:
            pass

    return {
        "portfolio_return": portfolio_return,
        "holdings": holdings,
        "eye_alerts_today": eye_events,
    }


def _collect_trades(today_str: str) -> dict:
    """당일 매매 실적 수집 (picks_history에서)."""
    hist = _load_json("picks_history.json")
    if not hist:
        return {"buys": [], "sells": [], "total_realized_pnl": 0}

    records = hist if isinstance(hist, list) else hist.get("records", [])
    buys = []
    sells = []
    total_pnl = 0

    for rec in records:
        pick_date = rec.get("pick_date", "")
        settled_date = rec.get("settled_date", "")

        if pick_date == today_str:
            buys.append({
                "ticker": rec.get("ticker", ""),
                "name": rec.get("name", ""),
                "price": rec.get("entry_price", 0),
                "reason": rec.get("grade", ""),
            })
        if settled_date == today_str and rec.get("status") in ("settled", "stopped"):
            pnl_pct = rec.get("settled_return", 0)
            sells.append({
                "ticker": rec.get("ticker", ""),
                "name": rec.get("name", ""),
                "price": rec.get("settled_price", 0),
                "reason": rec.get("status", ""),
                "pnl_pct": pnl_pct,
            })
            entry_p = rec.get("entry_price", 0)
            if entry_p:
                total_pnl += int(entry_p * pnl_pct / 100)

    return {"buys": buys, "sells": sells, "total_realized_pnl": total_pnl}


def record_daily(today_str: str, conn: sqlite3.Connection) -> dict:
    """일간 기록: 시장 개요 + 무버 + 보유 + 매매 → JSON + SQLite."""
    name_map = _build_name_map()

    overview = _collect_market_overview(today_str)
    movers = _collect_daily_movers(today_str, name_map)
    holdings_data = _collect_holdings(today_str, name_map)
    trades = _collect_trades(today_str)

    daily = {
        "date": today_str,
        "market": overview,
        "movers": movers,
        "holdings": holdings_data,
        "trades": trades,
        "created_at": datetime.now().isoformat(),
    }

    # JSON 저장
    out_path = DAILY_DIR / f"{today_str}.json"
    out_path.write_text(json.dumps(daily, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("일간 기록 저장: %s", out_path.name)

    # SQLite 저장 — daily_summary
    eye_count = len(holdings_data.get("eye_alerts_today", []))
    conn.execute(
        """INSERT OR REPLACE INTO daily_summary
           (date, kospi_close, kospi_change_pct, kosdaq_close, kosdaq_change_pct,
            regime, usd_krw, vix, foreign_net_bil, institution_net_bil,
            portfolio_return_pct, eye_event_count, buy_count, sell_count,
            realized_pnl, market_event, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (today_str, overview["kospi_close"], overview["kospi_change_pct"],
         overview["kosdaq_close"], overview["kosdaq_change_pct"],
         overview["regime"], overview["usd_krw"], overview["vix"],
         overview["foreign_net_bil"], overview["institution_net_bil"],
         holdings_data["portfolio_return"],
         eye_count, len(trades["buys"]), len(trades["sells"]),
         trades["total_realized_pnl"], overview["market_event"],
         datetime.now().isoformat()),
    )

    # SQLite — daily_movers
    conn.execute("DELETE FROM daily_movers WHERE date = ?", (today_str,))
    for category, items in movers.items():
        for rank, item in enumerate(items, 1):
            conn.execute(
                """INSERT INTO daily_movers
                   (date, category, rank, ticker, name, change_pct,
                    volume_ratio, foreign_net_5d, inst_net_5d, reason_tags)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (today_str, category, rank, item["ticker"], item["name"],
                 item["change_pct"], item["volume_ratio"],
                 item["foreign_net_5d"], item["inst_net_5d"],
                 json.dumps(item["reason_tags"], ensure_ascii=False)),
            )

    # SQLite — daily_holdings
    conn.execute("DELETE FROM daily_holdings WHERE date = ?", (today_str,))
    for h in holdings_data["holdings"]:
        conn.execute(
            """INSERT INTO daily_holdings
               (date, ticker, name, entry_date, entry_price, current_price,
                unrealized_pct, today_change_pct, days_held, status)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (today_str, h["ticker"], h["name"], h["entry_date"],
             h["entry_price"], h["current_price"], h["unrealized_pct"],
             h["today_change_pct"], h["days_held"], h["status"]),
        )

    conn.commit()
    log.info("SQLite 저장 완료: summary + %d movers + %d holdings",
             sum(len(v) for v in movers.values()), len(holdings_data["holdings"]))

    return daily


# ═══════════════════════════════════════════
#  9-2. 일간 학습 엔진
# ═══════════════════════════════════════════

def _learn_pattern_match(today_str: str, conn: sqlite3.Connection) -> dict | None:
    """학습 A: 오늘과 유사한 과거 조건 → 다음날 통계."""
    rows = conn.execute(
        """SELECT date, regime, kospi_change_pct, vix, foreign_net_bil
           FROM daily_summary WHERE date < ? ORDER BY date""",
        (today_str,),
    ).fetchall()

    if len(rows) < 10:
        return None

    today_row = conn.execute(
        "SELECT regime, kospi_change_pct, vix, foreign_net_bil FROM daily_summary WHERE date = ?",
        (today_str,),
    ).fetchone()
    if not today_row:
        return None

    t_regime, t_kospi_chg, t_vix, t_foreign = today_row

    # 유사 조건: 같은 레짐 + VIX ±5 + 외인방향 동일 + KOSPI 변동 ±2%
    similar = []
    for r in rows:
        r_date, r_regime, r_kospi, r_vix, r_foreign = r
        if r_regime != t_regime:
            continue
        if t_vix and r_vix and abs(r_vix - t_vix) > 5:
            continue
        if (t_foreign > 0) != (r_foreign > 0) and t_foreign != 0:
            continue
        if abs(r_kospi - t_kospi_chg) > 2:
            continue
        similar.append(r_date)

    if len(similar) < 3:
        return None

    # 유사일의 다음 거래일 KOSPI 변동
    next_day_changes = []
    for sd in similar:
        nxt = conn.execute(
            "SELECT kospi_change_pct FROM daily_summary WHERE date > ? ORDER BY date LIMIT 1",
            (sd,),
        ).fetchone()
        if nxt:
            next_day_changes.append(nxt[0])

    if not next_day_changes:
        return None

    up_count = sum(1 for c in next_day_changes if c > 0)
    avg_change = sum(next_day_changes) / len(next_day_changes)

    insight = {
        "similar_count": len(similar),
        "next_day_up_pct": round(up_count / len(next_day_changes) * 100, 1),
        "next_day_avg_change": round(avg_change, 2),
        "condition": f"레짐={t_regime}, VIX≈{t_vix:.0f}, 외인={'매수' if t_foreign > 0 else '매도'}",
    }
    log.info("패턴매칭: %d건 유사일 → 다음날 상승 %.0f%%", len(similar), insight["next_day_up_pct"])
    return insight


def _learn_mover_tracking(today_str: str, conn: sqlite3.Connection) -> dict | None:
    """학습 B: N일 전 급등 종목의 오늘 결과."""
    lookback = 3
    target_date = conn.execute(
        """SELECT date FROM daily_summary WHERE date < ?
           ORDER BY date DESC LIMIT 1 OFFSET ?""",
        (today_str, lookback - 1),
    ).fetchone()

    if not target_date:
        return None

    past_date = target_date[0]
    past_gainers = conn.execute(
        """SELECT ticker, name, change_pct FROM daily_movers
           WHERE date = ? AND category = '급등_top5' ORDER BY rank""",
        (past_date,),
    ).fetchall()

    if not past_gainers:
        return None

    results = []
    for ticker, name, orig_chg in past_gainers:
        today_row = conn.execute(
            "SELECT change_pct FROM daily_movers WHERE date = ? AND ticker = ?",
            (today_str, ticker),
        ).fetchone()
        if today_row:
            results.append({
                "ticker": ticker, "name": name,
                "original_change": orig_chg,
                "today_change": today_row[0],
            })

    if not results:
        return None

    continued = sum(1 for r in results if r["today_change"] > 0)
    avg_ret = sum(r["today_change"] for r in results) / len(results)

    insight = {
        "lookback_days": lookback,
        "past_date": past_date,
        "tracked": len(results),
        "continued_up": continued,
        "avg_return": round(avg_ret, 2),
    }
    log.info("급등추적: %s 급등 %d종목 → %d일 후 %d건 추가상승 (평균 %+.1f%%)",
             past_date, len(results), lookback, continued, avg_ret)
    return insight


def _learn_signal_accuracy(today_str: str) -> dict | None:
    """학습 C: 기존 signal_accuracy.json 요약."""
    acc = _load_json("market_learning/signal_accuracy.json")
    if not acc or not acc.get("signals"):
        return None

    summary = {}
    for sig_name, info in acc["signals"].items():
        if info.get("total", 0) >= 5:
            summary[sig_name] = {
                "hit_rate": round(info.get("hit_rate", 0), 1),
                "total": info["total"],
                "avg_ret": round(info.get("avg_ret", 0), 2),
            }

    return summary if summary else None


def _learn_flow_effectiveness(today_str: str, conn: sqlite3.Connection) -> dict | None:
    """학습 D: 수급 신호별 이후 수익률."""
    rows = conn.execute(
        """SELECT m.ticker, m.reason_tags, m.change_pct, m.date
           FROM daily_movers m
           WHERE m.category = '수급_쏠림_top5'
             AND m.date < ?
           ORDER BY m.date DESC LIMIT 100""",
        (today_str,),
    ).fetchall()

    if len(rows) < 10:
        return None

    dual_buy_rets = []
    foreign_only_rets = []
    inst_only_rets = []

    for ticker, tags_json, chg, mdate in rows:
        tags = json.loads(tags_json) if tags_json else []
        # 이후 5일 평균 수익률 (간이: 해당 종목의 movers 데이터로 추정)
        future = conn.execute(
            """SELECT change_pct FROM daily_movers
               WHERE ticker = ? AND date > ? ORDER BY date LIMIT 5""",
            (ticker, mdate),
        ).fetchall()
        if not future:
            continue
        avg_5d = sum(r[0] for r in future) / len(future)

        if "수급_쌍끌이" in tags:
            dual_buy_rets.append(avg_5d)
        elif "외인_매수" in tags:
            foreign_only_rets.append(avg_5d)
        elif "기관_매수" in tags:
            inst_only_rets.append(avg_5d)

    insight = {}
    if dual_buy_rets:
        insight["dual_buy"] = {"avg_5d": round(sum(dual_buy_rets) / len(dual_buy_rets), 2),
                               "count": len(dual_buy_rets)}
    if foreign_only_rets:
        insight["foreign_only"] = {"avg_5d": round(sum(foreign_only_rets) / len(foreign_only_rets), 2),
                                   "count": len(foreign_only_rets)}
    if inst_only_rets:
        insight["inst_only"] = {"avg_5d": round(sum(inst_only_rets) / len(inst_only_rets), 2),
                                "count": len(inst_only_rets)}

    return insight if insight else None


def learn_daily(today_str: str, conn: sqlite3.Connection) -> dict:
    """일간 학습: 4가지 학습 실행."""
    insights = {}

    pattern = _learn_pattern_match(today_str, conn)
    if pattern:
        insights["pattern_match"] = pattern

    mover = _learn_mover_tracking(today_str, conn)
    if mover:
        insights["mover_tracking"] = mover

    accuracy = _learn_signal_accuracy(today_str)
    if accuracy:
        insights["signal_accuracy"] = accuracy

    flow = _learn_flow_effectiveness(today_str, conn)
    if flow:
        insights["flow_effectiveness"] = flow

    if insights:
        # SQLite 저장
        conn.execute(
            "INSERT INTO learning_insights (date, insight_type, content, created_at) VALUES (?,?,?,?)",
            (today_str, "daily", json.dumps(insights, ensure_ascii=False),
             datetime.now().isoformat()),
        )
        conn.commit()

        # JSON 저장
        out = INSIGHTS_DIR / f"{today_str}.json"
        out.write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("학습 인사이트 저장: %d개 카테고리", len(insights))

    return insights


# ═══════════════════════════════════════════
#  9-3. 일간 텔레그램 저널
# ═══════════════════════════════════════════

def format_daily_journal(daily: dict, insights: dict) -> str:
    """일간 저널 텔레그램 메시지 포맷."""
    m = daily["market"]
    lines = [f"📊 [일간 저널] {daily['date']}"]
    lines.append("")

    # 시장 개요
    regime_emoji = {"BULL": "🟢", "CAUTION": "🟡", "BEAR": "🟠", "CRISIS": "🔴"}.get(
        m["regime"], "⚪")
    lines.append(
        f"시장: KOSPI {m['kospi_change_pct']:+.1f}% ({m['kospi_close']:,.0f})"
        f" | KOSDAQ {m['kosdaq_change_pct']:+.1f}%"
        f" | {regime_emoji}{m['regime']}"
    )
    parts = []
    if m["foreign_net_bil"]:
        parts.append(f"외인 {m['foreign_net_bil']:+,.0f}억")
    if m["institution_net_bil"]:
        parts.append(f"기관 {m['institution_net_bil']:+,.0f}억")
    if m["usd_krw"]:
        parts.append(f"환율 {m['usd_krw']:,.0f}원")
    if parts:
        lines.append(" | ".join(parts))

    # 급등/급락 TOP3
    movers = daily.get("movers", {})
    if movers.get("급등_top5"):
        top3 = movers["급등_top5"][:3]
        gain_strs = [f"{s['name']}{s['change_pct']:+.1f}%" for s in top3]
        lines.append(f"\n🔥 급등: {' | '.join(gain_strs)}")
    if movers.get("급락_top5"):
        bot3 = movers["급락_top5"][:3]
        loss_strs = [f"{s['name']}{s['change_pct']:+.1f}%" for s in bot3]
        lines.append(f"💧 급락: {' | '.join(loss_strs)}")

    # 보유 종목
    hd = daily.get("holdings", {})
    holdings = hd.get("holdings", [])
    if holdings:
        lines.append(f"\n📦 보유 ({len(holdings)}종목) 수익률 {hd['portfolio_return']:+.1f}%")
        STATUS_EMOJI = {"HEALTHY": "🟢", "WARNING": "🟡", "DANGER": "🔴"}
        for h in holdings:
            se = STATUS_EMOJI.get(h["status"], "⚪")
            lines.append(
                f"  {se} {h['name']} {h['unrealized_pct']:+.1f}%"
                f" ({h['days_held']}일차, {h['status']})"
            )

    # EYE 이벤트
    eye = hd.get("eye_alerts_today", [])
    if eye:
        lines.append(f"\n👁 장중 EYE: {len(eye)}건")
        for e in eye[:3]:
            lines.append(f"  {e.get('time', '?')} {e.get('eye_id', '')} {e.get('name', '')} {e.get('reason', '')[:25]}")

    # 학습 인사이트
    if insights.get("pattern_match"):
        pm = insights["pattern_match"]
        lines.append(
            f"\n📈 학습: 유사조건 {pm['similar_count']}회"
            f" → 다음날 상승 {pm['next_day_up_pct']:.0f}%"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════
#  9-4. 주간 보고서
# ═══════════════════════════════════════════

def generate_weekly_report(today_str: str, conn: sqlite3.Connection, send: bool = False) -> str | None:
    """주간 보고서 생성 + 텔레그램 발송."""
    today_d = datetime.strptime(today_str, "%Y-%m-%d").date()
    week_start = today_d - timedelta(days=today_d.weekday())
    week_start_str = week_start.strftime("%Y-%m-%d")

    rows = conn.execute(
        """SELECT date, kospi_close, kospi_change_pct, regime,
                  foreign_net_bil, institution_net_bil,
                  portfolio_return_pct, eye_event_count,
                  buy_count, sell_count, realized_pnl
           FROM daily_summary
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        (week_start_str, today_str),
    ).fetchall()

    if not rows:
        log.warning("주간 데이터 없음: %s ~ %s", week_start_str, today_str)
        return None

    # 집계
    kospi_start = rows[0][1]
    kospi_end = rows[-1][1]
    kospi_weekly = round((kospi_end / kospi_start - 1) * 100, 2) if kospi_start else 0
    weekly_foreign = sum(r[4] or 0 for r in rows)
    regimes = [r[3] for r in rows]
    regime_main = max(set(regimes), key=regimes.count)
    port_return = sum(r[6] or 0 for r in rows)
    total_buys = sum(r[8] or 0 for r in rows)
    total_sells = sum(r[9] or 0 for r in rows)
    total_pnl = sum(r[10] or 0 for r in rows)
    total_eye = sum(r[7] or 0 for r in rows)

    # 주간 무버 MVP/WORST
    mvp = conn.execute(
        """SELECT name, change_pct FROM daily_movers
           WHERE date >= ? AND date <= ? AND category = '급등_top5' AND rank = 1
           ORDER BY change_pct DESC LIMIT 1""",
        (week_start_str, today_str),
    ).fetchone()
    worst = conn.execute(
        """SELECT name, change_pct FROM daily_movers
           WHERE date >= ? AND date <= ? AND category = '급락_top5' AND rank = 1
           ORDER BY change_pct ASC LIMIT 1""",
        (week_start_str, today_str),
    ).fetchone()

    # 섹터 흐름 (급등 종목의 reason_tags에서 섹터 추출은 복잡 → 간단 요약)
    week_num = today_d.isocalendar()[1]

    lines = [
        f"📋 [주간 보고서] {week_start_str} ~ {today_str}",
        "",
        "■ 시장 요약",
        f"  KOSPI: {kospi_start:,.0f}→{kospi_end:,.0f} ({kospi_weekly:+.1f}%)"
        f" | 주간 외인 {weekly_foreign:+,.0f}억",
        f"  레짐: {regime_main} 주도 ({len(rows)}일)",
        "",
        "■ 주간 성과",
        f"  포트폴리오: {port_return:+.1f}%"
        f" (KOSPI 대비 {port_return - kospi_weekly:+.1f}%p)",
        f"  매수 {total_buys}건 / 매도 {total_sells}건"
        f" / 실현손익 {total_pnl:+,}원",
    ]

    if mvp:
        lines.append(f"\n■ MVP: {mvp[0]} {mvp[1]:+.1f}%")
    if worst:
        lines.append(f"■ WORST: {worst[0]} {worst[1]:+.1f}%")

    if total_eye:
        lines.append(f"\n■ EYE 감지: 총 {total_eye}건")

    # 학습 인사이트 (최신)
    insight = conn.execute(
        """SELECT content FROM learning_insights
           WHERE date >= ? AND date <= ? ORDER BY date DESC LIMIT 1""",
        (week_start_str, today_str),
    ).fetchone()
    if insight:
        try:
            ins = json.loads(insight[0])
            pm = ins.get("pattern_match")
            if pm:
                lines.append(
                    f"\n■ 학습 인사이트"
                    f"\n  \"{pm['condition']}\" → 다음날 상승 {pm['next_day_up_pct']:.0f}%"
                )
        except Exception:
            pass

    report = "\n".join(lines)

    # JSON 저장
    out = WEEKLY_DIR / f"{today_d.year}-W{week_num:02d}.json"
    out.write_text(json.dumps({
        "week": f"{today_d.year}-W{week_num:02d}",
        "start": week_start_str,
        "end": today_str,
        "kospi_weekly_pct": kospi_weekly,
        "portfolio_return_pct": port_return,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "realized_pnl": total_pnl,
        "regime_main": regime_main,
        "report_text": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("주간 보고서 저장: %s", out.name)

    if send:
        _send_telegram(report, "주간 보고서")

    return report


# ═══════════════════════════════════════════
#  9-5. 월간 보고서
# ═══════════════════════════════════════════

def generate_monthly_report(today_str: str, conn: sqlite3.Connection, send: bool = False) -> str | None:
    """월간 보고서 생성 + 텔레그램 발송."""
    today_d = datetime.strptime(today_str, "%Y-%m-%d").date()
    month_start = today_d.replace(day=1)
    month_start_str = month_start.strftime("%Y-%m-%d")

    rows = conn.execute(
        """SELECT date, kospi_close, kospi_change_pct, regime,
                  foreign_net_bil, institution_net_bil,
                  portfolio_return_pct, buy_count, sell_count, realized_pnl
           FROM daily_summary
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        (month_start_str, today_str),
    ).fetchall()

    if not rows:
        log.warning("월간 데이터 없음: %s", month_start_str)
        return None

    kospi_start = rows[0][1]
    kospi_end = rows[-1][1]
    kospi_monthly = round((kospi_end / kospi_start - 1) * 100, 2) if kospi_start else 0
    port_return = sum(r[6] or 0 for r in rows)
    total_buys = sum(r[7] or 0 for r in rows)
    total_sells = sum(r[8] or 0 for r in rows)
    total_pnl = sum(r[9] or 0 for r in rows)

    # 승률
    sell_wins = 0
    if total_sells > 0:
        sell_data = conn.execute(
            """SELECT realized_pnl FROM daily_summary
               WHERE date >= ? AND date <= ? AND sell_count > 0 AND realized_pnl > 0""",
            (month_start_str, today_str),
        ).fetchall()
        sell_wins = len(sell_data)

    # 레짐 분포
    regimes = [r[3] for r in rows]
    regime_dist = {}
    for r in regimes:
        regime_dist[r] = regime_dist.get(r, 0) + 1

    month_label = today_d.strftime("%Y년 %m월")
    lines = [
        f"📊 [월간 보고서] {month_label}",
        "",
        "■ 월간 성과",
        f"  포트폴리오: {port_return:+.1f}%"
        f" (KOSPI {kospi_monthly:+.1f}% 대비 {port_return - kospi_monthly:+.1f}%p)",
        f"  매수 {total_buys}건 / 매도 {total_sells}건",
        f"  실현손익: {total_pnl:+,}원",
    ]
    if total_sells > 0:
        lines.append(f"  승률: {sell_wins}/{total_sells} ({sell_wins/total_sells*100:.0f}%)")

    lines.append(f"\n■ 레짐 변화 ({len(rows)}일)")
    for regime, cnt in sorted(regime_dist.items(), key=lambda x: -x[1]):
        emoji = {"BULL": "🟢", "CAUTION": "🟡", "BEAR": "🟠", "CRISIS": "🔴"}.get(regime, "⚪")
        lines.append(f"  {emoji} {regime}: {cnt}일 ({cnt/len(rows)*100:.0f}%)")

    # 시그널 정확도 (기존 signal_accuracy.json에서)
    acc = _learn_signal_accuracy(today_str)
    if acc:
        lines.append("\n■ 시그널 적중률")
        for sig, info in sorted(acc.items(), key=lambda x: -x[1]["hit_rate"])[:5]:
            lines.append(f"  {sig}: {info['hit_rate']:.0f}% (N={info['total']})")

    # 학습 통계
    flow = _learn_flow_effectiveness(today_str, conn)
    if flow:
        lines.append("\n■ 수급 패턴 통계")
        if "dual_buy" in flow:
            lines.append(
                f"  쌍끌이 5일후: {flow['dual_buy']['avg_5d']:+.1f}%"
                f" (N={flow['dual_buy']['count']})"
            )
        if "foreign_only" in flow:
            lines.append(
                f"  외인단독 5일후: {flow['foreign_only']['avg_5d']:+.1f}%"
                f" (N={flow['foreign_only']['count']})"
            )

    lines.append(f"\n■ 개선 제안")
    lines.append("  → 데이터 3개월 누적 후 자동 파라미터 조정 예정")

    report = "\n".join(lines)

    # JSON 저장
    month_key = today_d.strftime("%Y-%m")
    out = MONTHLY_DIR / f"{month_key}.json"
    out.write_text(json.dumps({
        "month": month_key,
        "start": month_start_str,
        "end": today_str,
        "kospi_monthly_pct": kospi_monthly,
        "portfolio_return_pct": port_return,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "realized_pnl": total_pnl,
        "regime_distribution": regime_dist,
        "report_text": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("월간 보고서 저장: %s", out.name)

    if send:
        _send_telegram(report, "월간 보고서")

    return report


# ═══════════════════════════════════════════
#  텔레그램 발송
# ═══════════════════════════════════════════

def _send_telegram(msg: str, label: str = "저널"):
    """텔레그램 발송 (4096자 분할)."""
    try:
        from src.telegram_sender import send_message
        MAX = 4000
        if len(msg) <= MAX:
            ok = send_message(msg)
            log.info("[텔레그램] %s 발송 %s (%d자)", label, "성공" if ok else "실패", len(msg))
        else:
            parts = []
            current = ""
            for line in msg.split("\n"):
                if len(current) + len(line) + 1 > MAX:
                    parts.append(current)
                    current = line
                else:
                    current += ("\n" + line) if current else line
            if current:
                parts.append(current)

            for i, part in enumerate(parts, 1):
                ok = send_message(f"[{i}/{len(parts)}]\n{part}")
                log.info("[텔레그램] %s 파트 %d/%d 발송 %s",
                         label, i, len(parts), "성공" if ok else "실패")
    except Exception as e:
        log.error("[텔레그램] %s 발송 실패: %s", label, e)


# ═══════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Market Journal")
    parser.add_argument("--weekly", action="store_true", help="주간 보고서 강제 생성")
    parser.add_argument("--monthly", action="store_true", help="월간 보고서 강제 생성")
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 비발송")
    parser.add_argument("--date", default=None, help="날짜 지정 (YYYY-MM-DD)")
    args = parser.parse_args()

    _ensure_dirs()
    today_str = args.date or date.today().strftime("%Y-%m-%d")
    today_d = datetime.strptime(today_str, "%Y-%m-%d").date()
    send = not args.dry_run

    conn = _init_db()

    try:
        # 1) 일간 기록
        log.info("═══ Market Journal [%s] ═══", today_str)
        daily = record_daily(today_str, conn)

        # 2) 일간 학습
        insights = learn_daily(today_str, conn)

        # 3) 일간 텔레그램 저널
        journal_msg = format_daily_journal(daily, insights)
        if send:
            _send_telegram(journal_msg, "일간 저널")
        else:
            print("[미리보기 — 일간 저널]")
            print(journal_msg)
            print(f"({len(journal_msg)}자)")

        # 4) 주간 보고서 (금요일 또는 --weekly)
        if args.weekly or today_d.weekday() == 4:
            log.info("═══ 주간 보고서 생성 ═══")
            weekly = generate_weekly_report(today_str, conn, send=send)
            if weekly and not send:
                print("\n[미리보기 — 주간 보고서]")
                print(weekly)

        # 5) 월간 보고서 (월말 영업일 또는 --monthly)
        if args.monthly or _is_last_business_day(today_d):
            log.info("═══ 월간 보고서 생성 ═══")
            monthly = generate_monthly_report(today_str, conn, send=send)
            if monthly and not send:
                print("\n[미리보기 — 월간 보고서]")
                print(monthly)

    finally:
        conn.close()

    log.info("═══ Market Journal 완료 ═══")


if __name__ == "__main__":
    main()
