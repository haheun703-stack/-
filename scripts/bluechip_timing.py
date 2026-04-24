#!/usr/bin/env python3
"""우량주 TOP 100 매매타이밍 — 별도 포트폴리오 운영.

백테스트: 665건, WR 52%, 평균 +7.58%, PF 3.35, 보유 14.9일
핵심: 수급 급변(D+0) → D+1 양봉 확인 → 진입, 수급이탈3일 → 매도

Entry:
  - scan_supply_surge 전일 BUY 후보 중 시총 TOP 100 필터
  - D+1 양봉 확인 (close > open)
  - 균등 사이징 (1억 / MAX_POSITIONS)

Exit:
  - 외인+기관 3일 연속 순매도 (합산 < -10억/일)
  - 손절 -10%
  - 만기 없음 (수급 이탈 시까지 보유)

BAT-D G4.3 (scan_supply_surge 직후)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════

DATA_DIR = PROJECT_ROOT / "data"
PORTFOLIO_PATH = DATA_DIR / "paper_bluechip.json"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"
DB_PATH = DATA_DIR / "investor_flow" / "investor_daily.db"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
SURGE_DIR = DATA_DIR

INITIAL_CAPITAL = 100_000_000   # 1억원 가상 자본
MAX_POSITIONS = 10
MAX_NEW_PER_DAY = 3
STOP_LOSS_PCT = -0.10           # -10% 손절
SLIPPAGE_PCT = 0.001
COMMISSION_PCT = 0.00015
TAX_PCT = 0.0018
EXIT_FLOW_THRESHOLD = -10       # 외인+기관 합산 -10억/일
EXIT_FLOW_CONSECUTIVE = 3       # 3일 연속


# ═══════════════════════════════════════════════
# 1. 데이터 로더
# ═══════════════════════════════════════════════

def load_top100() -> set[str]:
    """universe.csv에서 market_cap 상위 100종목 티커 set."""
    if not UNIVERSE_PATH.exists():
        logger.error("universe.csv 없음: %s", UNIVERSE_PATH)
        return set()
    df = pd.read_csv(UNIVERSE_PATH)
    if "market_cap" not in df.columns or "ticker" not in df.columns:
        logger.error("universe.csv에 market_cap/ticker 컬럼 없음")
        return set()
    df = df.dropna(subset=["market_cap"])
    df = df.nlargest(100, "market_cap")
    tickers = set(df["ticker"].astype(str).str.zfill(6))
    logger.info("TOP 100 우량주: %d종목 로드", len(tickers))
    return tickers


def load_yesterday_surge(today_str: str) -> list[dict]:
    """전일(오늘 제외) supply_surge JSON의 buy_candidates 반환."""
    files = sorted(SURGE_DIR.glob("supply_surge_*.json"), reverse=True)
    for f in files:
        date_part = f.stem.replace("supply_surge_", "")
        if date_part != today_str:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                candidates = data.get("buy_candidates", [])
                logger.info(
                    "전일 수급급변: %s (%d건 BUY)",
                    f.name, len(candidates),
                )
                return candidates
            except Exception as e:
                logger.warning("JSON 로드 실패: %s — %s", f.name, e)
    logger.warning("전일 supply_surge JSON 없음")
    return []


def load_price_data(ticker: str) -> pd.DataFrame | None:
    """stock_data_daily CSV → OHLCV DataFrame (date index)."""
    csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if not csvs:
        return None
    try:
        df = pd.read_csv(csvs[0], header=0)
        if len(df.columns) < 6 or len(df) < 30:
            return None
        df = df.iloc[:, :6]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = df["date"].astype(str).str.replace("-", "")
        df = df.sort_values("date").reset_index(drop=True)
        return df.set_index("date")
    except Exception:
        return None


def get_latest_ohlcv(ticker: str) -> dict | None:
    """최신일 OHLCV dict 반환."""
    pdf = load_price_data(ticker)
    if pdf is None or pdf.empty:
        return None
    last = pdf.iloc[-1]
    return {
        "date": pdf.index[-1],
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
    }


# ═══════════════════════════════════════════════
# 2. Entry 로직
# ═══════════════════════════════════════════════

def find_new_candidates(
    surge_candidates: list[dict],
    top100: set[str],
    existing_tickers: set[str],
) -> list[dict]:
    """전일 surge BUY 중 TOP100 + 당일 양봉 확인된 종목."""
    results = []
    top100_match = 0
    for cand in surge_candidates:
        ticker = str(cand.get("ticker", "")).zfill(6)
        if ticker not in top100:
            continue
        top100_match += 1
        if ticker in existing_tickers:
            continue
        ohlcv = get_latest_ohlcv(ticker)
        if ohlcv is None:
            continue
        # D+1 양봉 확인: close > open
        if ohlcv["close"] <= ohlcv["open"]:
            continue
        results.append({
            "ticker": ticker,
            "name": cand.get("name", ticker),
            "close": ohlcv["close"],
            "entry_type": cand.get("type", ""),
            "entry_score": cand.get("final_score", 0),
            "fgn": cand.get("fgn", 0),
            "inst": cand.get("inst", 0),
            "pension": cand.get("pension", 0),
        })
    logger.info(
        "후보: surge→TOP100 %d건 → 양봉확인 %d건",
        top100_match, len(results),
    )
    return results


def enter_new_positions(
    pf: dict, candidates: list[dict], today_str: str,
) -> list[dict]:
    """후보 종목 가상 매수."""
    entries = []
    current_count = len(pf["positions"])
    slots = MAX_POSITIONS - current_count
    if slots <= 0:
        logger.info("포지션 한도 %d/%d — 신규 진입 불가", current_count, MAX_POSITIONS)
        return entries

    size_per = INITIAL_CAPITAL / MAX_POSITIONS
    new_today = 0

    for cand in candidates:
        if new_today >= MAX_NEW_PER_DAY:
            break
        if len(pf["positions"]) >= MAX_POSITIONS:
            break

        ticker = cand["ticker"]
        price = cand["close"]
        buy_price = price * (1 + SLIPPAGE_PCT)
        qty = int(size_per / buy_price)
        if qty < 1:
            continue
        cost = buy_price * qty * (1 + COMMISSION_PCT)
        pf["capital"] = pf.get("capital", INITIAL_CAPITAL) - cost

        pf["positions"][ticker] = {
            "name": cand["name"],
            "ticker": ticker,
            "entry_date": today_str,
            "entry_price": round(buy_price),
            "qty": qty,
            "cost": round(cost),
            "peak_price": round(buy_price),
            "entry_type": cand["entry_type"],
            "entry_score": cand["entry_score"],
        }

        entries.append({
            "ticker": ticker,
            "name": cand["name"],
            "price": round(buy_price),
            "qty": qty,
            "type": cand["entry_type"],
            "score": cand["entry_score"],
        })
        new_today += 1
        logger.info(
            "  진입: %s %s  %s원 x%d  [%s] %d점",
            ticker, cand["name"], f"{round(buy_price):,}",
            qty, cand["entry_type"], cand["entry_score"],
        )

    return entries


# ═══════════════════════════════════════════════
# 3. Exit 로직
# ═══════════════════════════════════════════════

def check_supply_exit(ticker: str, entry_date: str = "") -> tuple[bool, str]:
    """외인+기관 3일 연속 순매도 확인. entry_date 이후만 판정."""
    if not DB_PATH.exists():
        return False, ""
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        query = """
            SELECT date,
                   SUM(CASE WHEN investor = '외국인'   THEN net_val ELSE 0 END) / 1e8 AS fgn,
                   SUM(CASE WHEN investor = '기관합계' THEN net_val ELSE 0 END) / 1e8 AS inst
            FROM investor_daily
            WHERE ticker = ?
        """
        params: list = [ticker]
        if entry_date:
            query += " AND date >= ?"
            params.append(entry_date)
        query += " GROUP BY date ORDER BY date DESC LIMIT ?"
        params.append(EXIT_FLOW_CONSECUTIVE + 2)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception as e:
        logger.warning("수급 DB 쿼리 실패 (%s): %s", ticker, e)
        return False, ""

    if len(df) < EXIT_FLOW_CONSECUTIVE:
        return False, ""

    # 최근 3일 외인+기관 합산
    df = df.head(EXIT_FLOW_CONSECUTIVE)
    df["flow"] = df["fgn"] + df["inst"]
    consecutive = all(df["flow"] < EXIT_FLOW_THRESHOLD)

    if consecutive:
        flows = [f"{r['flow']:+.0f}" for _, r in df.iterrows()]
        detail = f"외+기 3일: {', '.join(flows)}억"
        return True, detail
    return False, ""


def check_exits(pf: dict, today_str: str) -> list[dict]:
    """보유 종목 매도 조건 체크."""
    exits = []
    to_remove = []

    for ticker, pos in pf["positions"].items():
        ohlcv = get_latest_ohlcv(ticker)
        if ohlcv is None:
            continue

        price = ohlcv["close"]
        entry_price = pos["entry_price"]
        pnl_pct = price / entry_price - 1 if entry_price > 0 else 0

        # 고점 갱신
        if price > pos.get("peak_price", 0):
            pos["peak_price"] = round(price)

        # 보유일
        try:
            entry_dt = datetime.strptime(pos["entry_date"], "%Y%m%d")
            today_dt = datetime.strptime(today_str, "%Y%m%d")
            days_held = (today_dt - entry_dt).days
        except ValueError:
            days_held = 0

        exit_reason = None
        detail = ""

        # 1. 손절 -10%
        if pnl_pct <= STOP_LOSS_PCT:
            exit_reason = "STOP_LOSS"
            detail = f"PnL {pnl_pct*100:+.1f}%"

        # 2. 수급이탈 3일 (입장일 이후만 판정)
        if not exit_reason:
            is_exit, flow_detail = check_supply_exit(ticker, pos.get("entry_date", ""))
            if is_exit:
                exit_reason = "SUPPLY_EXIT"
                detail = flow_detail

        if exit_reason:
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * pos["qty"] * (1 - TAX_PCT)
            final_pnl = sell_price / entry_price - 1

            pf["closed_trades"].append({
                "ticker": ticker,
                "name": pos["name"],
                "entry_type": pos.get("entry_type", ""),
                "entry_date": pos["entry_date"],
                "exit_date": today_str,
                "entry_price": entry_price,
                "exit_price": round(sell_price),
                "qty": pos["qty"],
                "pnl_pct": round(final_pnl * 100, 2),
                "exit_reason": exit_reason,
                "days_held": days_held,
            })

            pf["stats"]["total_trades"] = pf["stats"].get("total_trades", 0) + 1
            if final_pnl > 0:
                pf["stats"]["wins"] = pf["stats"].get("wins", 0) + 1
            else:
                pf["stats"]["losses"] = pf["stats"].get("losses", 0) + 1

            pf["capital"] = pf.get("capital", INITIAL_CAPITAL) + proceeds

            exits.append({
                "ticker": ticker,
                "name": pos["name"],
                "reason": exit_reason,
                "pnl_pct": round(final_pnl * 100, 2),
                "days_held": days_held,
                "detail": detail,
            })
            to_remove.append(ticker)

            logger.info(
                "  매도: %s %s  %s  PnL %+.1f%%  %d일 보유",
                ticker, pos["name"], exit_reason,
                final_pnl * 100, days_held,
            )

    for t in to_remove:
        del pf["positions"][t]

    return exits


# ═══════════════════════════════════════════════
# 4. 포트폴리오 관리
# ═══════════════════════════════════════════════

def _default_portfolio() -> dict:
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "initial_capital": INITIAL_CAPITAL,
        "capital": INITIAL_CAPITAL,
        "positions": {},
        "closed_trades": [],
        "daily_equity": [],
        "stats": {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "max_equity": INITIAL_CAPITAL,
            "mdd": 0.0,
        },
        "updated": "",
    }


def load_portfolio() -> dict:
    if PORTFOLIO_PATH.exists():
        try:
            return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _default_portfolio()


def save_portfolio(pf: dict) -> None:
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    PORTFOLIO_PATH.write_text(
        json.dumps(pf, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("포트폴리오 저장: %s", PORTFOLIO_PATH.name)


def update_equity(pf: dict, today_str: str) -> None:
    """일일 자산 기록 + MDD 갱신."""
    total_value = pf.get("capital", INITIAL_CAPITAL)
    for ticker, pos in pf["positions"].items():
        ohlcv = get_latest_ohlcv(ticker)
        if ohlcv:
            total_value += ohlcv["close"] * pos["qty"]
        else:
            total_value += pos["entry_price"] * pos["qty"]

    pf["daily_equity"].append({
        "date": today_str,
        "equity": round(total_value),
    })

    max_eq = pf["stats"].get("max_equity", INITIAL_CAPITAL)
    if total_value > max_eq:
        pf["stats"]["max_equity"] = round(total_value)
        max_eq = total_value

    dd = (total_value / max_eq - 1) * 100 if max_eq > 0 else 0
    if dd < pf["stats"].get("mdd", 0):
        pf["stats"]["mdd"] = round(dd, 2)


# ═══════════════════════════════════════════════
# 5. 리포트
# ═══════════════════════════════════════════════

def print_report(
    today_str: str,
    entries: list[dict],
    exits: list[dict],
    pf: dict,
    surge_total: int,
    top100_match: int,
    candidate_count: int = 0,
) -> str:
    """콘솔 리포트 출력 + 텔레그램 메시지 문자열 반환."""
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"  [BLUECHIP] 우량주 TOP100 매매타이밍  {today_str}")
    lines.append(sep)

    # 매도
    if exits:
        lines.append(f"\n  매도 {len(exits)}건:")
        for e in exits:
            lines.append(
                f"    {e['ticker']} {e['name']}  {e['reason']}  "
                f"PnL {e['pnl_pct']:+.1f}%  {e['days_held']}일"
            )

    # 매수
    if entries:
        lines.append(f"\n  신규 진입 {len(entries)}건:")
        for e in entries:
            lines.append(
                f"    {e['ticker']} {e['name']}  "
                f"{e['price']:,}원 x{e['qty']}  [{e['type']}] {e['score']}점"
            )

    # 현재 보유
    lines.append(f"\n  보유 {len(pf['positions'])}/{MAX_POSITIONS}:")
    for ticker, pos in pf["positions"].items():
        ohlcv = get_latest_ohlcv(ticker)
        cur = ohlcv["close"] if ohlcv else pos["entry_price"]
        pnl = (cur / pos["entry_price"] - 1) * 100 if pos["entry_price"] > 0 else 0
        lines.append(
            f"    {ticker} {pos['name']}  {cur:,.0f}원  "
            f"PnL {pnl:+.1f}%  [{pos.get('entry_type', '')}]"
        )

    # 통계
    stats = pf["stats"]
    total = stats.get("total_trades", 0)
    wins = stats.get("wins", 0)
    wr = wins / total * 100 if total > 0 else 0
    equity = pf["daily_equity"][-1]["equity"] if pf["daily_equity"] else INITIAL_CAPITAL
    lines.append(f"\n  총 거래: {total}건  승률: {wr:.1f}%  MDD: {stats.get('mdd', 0):.1f}%")
    lines.append(f"  자산: {equity:,.0f}원  수익률: {(equity/INITIAL_CAPITAL-1)*100:+.2f}%")
    bullish = candidate_count if candidate_count else len(entries)
    lines.append(f"  후보: surge {surge_total} → TOP100 {top100_match} → 양봉 {bullish} → 진입 {len(entries)}")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)

    # 텔레그램 메시지
    tg_lines = [f"[BLUECHIP] 우량주 TOP100  {today_str}"]
    if exits:
        for e in exits:
            tg_lines.append(f"  매도 {e['name']} {e['reason']} {e['pnl_pct']:+.1f}%")
    if entries:
        for e in entries:
            tg_lines.append(f"  매수 {e['name']} [{e['type']}] {e['price']:,}원")
    tg_lines.append(f"보유 {len(pf['positions'])}/{MAX_POSITIONS}  자산 {equity:,.0f}원")
    return "\n".join(tg_lines)


def send_telegram(msg: str) -> None:
    """텔레그램 전송 (실패해도 계속)."""
    try:
        from src.telegram_sender import send_message
        send_message(msg)
        logger.info("텔레그램 전송 완료")
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


# ═══════════════════════════════════════════════
# 6. 메인
# ═══════════════════════════════════════════════

def run_daily(dry_run: bool = False) -> dict:
    """일일 메인 루틴."""
    now = datetime.now()
    today_str = now.strftime("%Y%m%d")

    # 주말 가드
    if now.weekday() >= 5:
        logger.info("주말 — 스킵")
        return {}

    logger.info("=" * 50)
    logger.info("[BLUECHIP] 우량주 TOP100 매매타이밍  %s", today_str)
    logger.info("=" * 50)

    # 1. 포트폴리오 로드
    pf = load_portfolio()

    # 2. TOP 100 로드
    top100 = load_top100()
    if not top100:
        logger.error("TOP 100 로드 실패 — 종료")
        return {}

    # 3. 매도 체크
    exits = check_exits(pf, today_str)

    # 4. 전일 surge + TOP100 + 양봉 → 신규 후보
    surge_candidates = load_yesterday_surge(today_str)
    surge_total = len(surge_candidates)
    existing = set(pf["positions"].keys())
    candidates = find_new_candidates(surge_candidates, top100, existing)

    # 5. 신규 진입
    if dry_run:
        entries = []
        logger.info("[DRY-RUN] 진입 스킵 (%d건 후보)", len(candidates))
        for c in candidates:
            logger.info(
                "  후보: %s %s  %s원  [%s] %d점",
                c["ticker"], c["name"], f"{c['close']:,.0f}",
                c["entry_type"], c["entry_score"],
            )
    else:
        entries = enter_new_positions(pf, candidates, today_str)

    # 6. 자산 갱신
    update_equity(pf, today_str)

    # 7. 저장
    if not dry_run:
        save_portfolio(pf)

    # 8. 리포트
    top100_match = sum(
        1 for c in surge_candidates
        if str(c.get("ticker", "")).zfill(6) in top100
    )
    tg_msg = print_report(
        today_str, entries, exits, pf, surge_total, top100_match,
        candidate_count=len(candidates),
    )

    # 9. 텔레그램 (매도 or 매수 있을 때만)
    if (entries or exits) and not dry_run:
        send_telegram(tg_msg)

    return {
        "entries": len(entries),
        "exits": len(exits),
        "positions": len(pf["positions"]),
        "candidates": len(candidates),
    }


def main():
    parser = argparse.ArgumentParser(description="우량주 TOP100 매매타이밍")
    parser.add_argument("--dry-run", action="store_true", help="진입/저장 없이 시뮬레이션")
    parser.add_argument("--reset", action="store_true", help="포트폴리오 초기화")
    parser.add_argument("--status", action="store_true", help="현재 보유 상태 출력")
    args = parser.parse_args()

    if args.reset:
        pf = _default_portfolio()
        save_portfolio(pf)
        print("포트폴리오 초기화 완료")
        return

    if args.status:
        pf = load_portfolio()
        print(json.dumps(pf, ensure_ascii=False, indent=2))
        return

    run_daily(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
