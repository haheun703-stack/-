"""통합 Paper Trading 엔진 — scan_buy 추천 자동 가상 매매.

매일 장마감 후 BAT-D에서 실행:
  python -u -X utf8 scripts/paper_trading_unified.py

동작:
  1. tomorrow_picks.json에서 추천 종목 수집 (AI대형주 + 전략종합)
  2. 등급별 가상 포트폴리오 진입 (max 3개/일, max 8개 보유)
  3. 보유 종목 일별 현재가 업데이트 + 매도 조건 체크
  4. 텔레그램 일일 리포트 ([PAPER] 태그)
  5. FLOWX Supabase paper_trades 업로드
  6. 금요일 주간 리포트

데이터:
  - 입력: data/tomorrow_picks.json, data/processed/*.parquet
  - 출력: data/paper_portfolio.json (포지션 + 성적)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
PORTFOLIO_PATH = DATA_DIR / "paper_portfolio.json"
PICKS_PATH = DATA_DIR / "tomorrow_picks.json"
PROCESSED_DIR = DATA_DIR / "processed"

# 포지션 규칙
INITIAL_CAPITAL = 30_000_000   # 3,000만원 가상 자본
MAX_POSITIONS = 8              # 최대 동시 보유
MAX_NEW_PER_DAY = 3            # 하루 최대 신규 진입
SLIPPAGE_PCT = 0.001           # 슬리피지 0.1%
COMMISSION_PCT = 0.00015       # 수수료 0.015%
TAX_PCT = 0.0018               # 매도세 0.18%

# 등급별 사이징 (자본 대비 %)
SIZING = {
    "AA": 0.15,    # 적극매수/confidence>=0.85: 자본의 15%
    "A": 0.12,     # 매수/confidence>=0.75: 12%
    "B": 0.10,     # 관심매수/기타: 10%
}

# 매도 규칙
STOP_LOSS_PCT = -0.07          # -7% 손절
TAKE_PROFIT_T1_PCT = 0.10      # +10% 1차 익절 (50% 매도)
TAKE_PROFIT_T2_PCT = 0.20      # +20% 2차 익절 (전량 매도)
TRAILING_ACTIVATE_PCT = 0.08   # +8% 이후 트레일링 활성화
TRAILING_STOP_PCT = -0.04      # 고점 대비 -4% 하락 시 매도
MAX_HOLDING_DAYS = 15          # 최대 보유일 (거래일 기준)


# ═══════════════════════════════════════════════
# 포트폴리오 관리
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
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 마이그레이션: 이전 포맷 호환
        if data.get("initial_capital", 0) != INITIAL_CAPITAL and not data.get("positions"):
            logger.info("포트폴리오 자본금 갱신: %s → %s", data.get("initial_capital"), INITIAL_CAPITAL)
            data["initial_capital"] = INITIAL_CAPITAL
            data["capital"] = INITIAL_CAPITAL
        return data
    pf = _default_portfolio()
    save_portfolio(pf)
    return pf


def save_portfolio(pf: dict) -> None:
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 종가 조회
# ═══════════════════════════════════════════════

def get_latest_price(ticker: str) -> tuple[float, str]:
    """processed parquet에서 최신 종가 + 날짜 반환."""
    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        # raw fallback
        pq = DATA_DIR / "raw" / f"{ticker}.parquet"
    if not pq.exists():
        return 0.0, ""
    try:
        df = pd.read_parquet(pq)
        if len(df) == 0:
            return 0.0, ""
        last = df.iloc[-1]
        return float(last["close"]), df.index[-1].strftime("%Y-%m-%d")
    except Exception:
        return 0.0, ""


# ═══════════════════════════════════════════════
# 추천 종목 수집
# ═══════════════════════════════════════════════

def collect_candidates() -> list[dict]:
    """tomorrow_picks.json에서 paper trading 후보 수집.

    Returns:
        [{"ticker", "name", "grade", "score", "price", "strategy", "reason"}]
        score 내림차순 정렬.
    """
    if not PICKS_PATH.exists():
        logger.warning("tomorrow_picks.json 없음")
        return []

    with open(PICKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    seen = set()

    # 1) AI 대형주 (confidence >= 0.7)
    for item in data.get("ai_largecap", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        conf = float(item.get("confidence", 0))
        if conf < 0.7:
            continue

        price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        if conf >= 0.85:
            grade = "AA"
        elif conf >= 0.75:
            grade = "A"
        else:
            grade = "B"

        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "name": item.get("name", ticker),
            "grade": grade,
            "score": round(conf * 100, 1),
            "price": price,
            "strategy": "AI_BRAIN",
            "reason": item.get("reasoning", "")[:80],
        })

    # 2) 전략 종합 picks (score >= 40, 상위 5개)
    picks_sorted = sorted(
        data.get("picks", []),
        key=lambda x: x.get("total_score", 0),
        reverse=True,
    )
    for pick in picks_sorted[:5]:
        ticker = pick.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        score = pick.get("total_score", 0)
        if score < 40:
            continue

        price = pick.get("close", 0)
        if price <= 0:
            price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        grade_kr = pick.get("grade", "")
        if grade_kr == "적극매수":
            grade = "AA"
        elif grade_kr == "매수":
            grade = "A"
        else:
            grade = "B"

        seen.add(ticker)
        reasons = pick.get("reasons", [])
        reason_str = ", ".join(reasons[:3]) if reasons else ""

        candidates.append({
            "ticker": ticker,
            "name": pick.get("name", ticker),
            "grade": grade,
            "score": round(score, 1),
            "price": float(price),
            "strategy": "SCAN",
            "reason": reason_str[:80],
        })

    # 3) top5_swing (스윙 전용, 있으면)
    for item in data.get("top5_swing", []):
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        price, _ = get_latest_price(ticker)
        if price <= 0:
            continue

        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "name": item.get("name", ticker),
            "grade": "A",
            "score": round(item.get("total_score", 50), 1),
            "price": price,
            "strategy": "SWING",
            "reason": "스윙 전략 추천",
        })

    # score 내림차순 정렬
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════
# 매도 체크
# ═══════════════════════════════════════════════

def check_exits(pf: dict, today_str: str) -> list[dict]:
    """보유 종목 매도 조건 체크. 매도된 포지션 리스트 반환."""
    exits = []
    codes_to_remove = []

    for ticker, pos in list(pf["positions"].items()):
        price, price_date = get_latest_price(ticker)
        if price <= 0:
            continue

        avg_price = pos["avg_price"]
        if avg_price <= 0:
            logger.warning("[check_exits] %s avg_price=0, 스킵", ticker)
            continue
        peak_price = pos.get("peak_price", avg_price)
        pnl_pct = price / avg_price - 1

        # 최고가 갱신
        if price > peak_price:
            pos["peak_price"] = price
            peak_price = price

        # 보유일수 계산
        entry_date = pos.get("entry_date", today_str)
        try:
            days_held = (pd.Timestamp(today_str) - pd.Timestamp(entry_date)).days
        except Exception:
            days_held = 0

        exit_reason = None
        exit_qty = pos["qty"]  # 기본: 전량 매도

        # 1. 손절
        if pnl_pct <= STOP_LOSS_PCT:
            exit_reason = "STOP_LOSS"

        # 2. 2차 익절 (+20%)
        elif pnl_pct >= TAKE_PROFIT_T2_PCT:
            exit_reason = "TAKE_PROFIT"

        # 3. 1차 익절 (+10%, 50% 매도)
        elif pnl_pct >= TAKE_PROFIT_T1_PCT and not pos.get("t1_sold"):
            exit_qty = max(1, pos["qty"] // 2)
            pos["t1_sold"] = True
            pos["qty"] -= exit_qty
            # 부분 매도 — 포지션 유지
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * exit_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds
            exits.append({
                "ticker": ticker,
                "name": pos["name"],
                "reason": "TAKE_PROFIT_T1",
                "exit_price": round(sell_price),
                "pnl_pct": round(pnl_pct * 100, 2),
                "qty": exit_qty,
                "partial": True,
            })
            # 트레일링 활성화
            pos["trailing_active"] = True
            continue

        # 4. 트레일링 스탑
        elif pos.get("trailing_active"):
            drop_from_peak = price / peak_price - 1
            if drop_from_peak <= TRAILING_STOP_PCT:
                exit_reason = "TRAILING_STOP"

        # 5. 트레일링 활성화 (고점 갱신 중)
        elif pnl_pct >= TRAILING_ACTIVATE_PCT:
            pos["trailing_active"] = True

        # 6. 최대 보유일 초과
        elif days_held >= MAX_HOLDING_DAYS:
            exit_reason = "MAX_HOLD"

        # 전량 매도 처리
        if exit_reason:
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * exit_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds

            final_pnl = sell_price / avg_price - 1

            pf["closed_trades"].append({
                "ticker": ticker,
                "name": pos["name"],
                "strategy": pos.get("strategy", ""),
                "grade": pos.get("grade", ""),
                "entry_date": pos.get("entry_date", ""),
                "exit_date": today_str,
                "avg_price": round(avg_price),
                "exit_price": round(sell_price),
                "qty": exit_qty,
                "pnl_pct": round(final_pnl * 100, 2),
                "exit_reason": exit_reason,
                "days_held": days_held,
            })

            pf["stats"]["total_trades"] += 1
            if final_pnl > 0:
                pf["stats"]["wins"] += 1
            else:
                pf["stats"]["losses"] += 1

            exits.append({
                "ticker": ticker,
                "name": pos["name"],
                "reason": exit_reason,
                "exit_price": round(sell_price),
                "pnl_pct": round(final_pnl * 100, 2),
                "qty": exit_qty,
                "partial": False,
            })
            codes_to_remove.append(ticker)

    for ticker in codes_to_remove:
        if ticker in pf["positions"]:
            del pf["positions"][ticker]

    return exits


# ═══════════════════════════════════════════════
# 신규 진입
# ═══════════════════════════════════════════════

def enter_new_positions(pf: dict, candidates: list[dict], today_str: str) -> list[dict]:
    """후보 종목 가상 매수. 진입한 종목 리스트 반환."""
    entries = []
    slots_available = MAX_POSITIONS - len(pf["positions"])
    new_today = 0

    for cand in candidates:
        if new_today >= MAX_NEW_PER_DAY:
            break
        if slots_available <= 0:
            break
        if cand["ticker"] in pf["positions"]:
            continue

        # 사이징
        grade = cand["grade"]
        size_pct = SIZING.get(grade, SIZING["B"])
        buy_amount = min(
            pf["initial_capital"] * size_pct,
            pf["capital"] * 0.90,  # 현금의 90%까지만
        )

        price = cand["price"]
        buy_price = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
        qty = int(buy_amount / buy_price)
        if qty <= 0:
            continue

        cost = buy_price * qty
        if cost > pf["capital"]:
            continue

        pf["capital"] -= cost
        pf["positions"][cand["ticker"]] = {
            "name": cand["name"],
            "ticker": cand["ticker"],
            "entry_date": today_str,
            "avg_price": round(buy_price),
            "qty": qty,
            "cost": round(cost),
            "peak_price": price,
            "strategy": cand["strategy"],
            "grade": grade,
            "reason": cand["reason"],
            "trailing_active": False,
            "t1_sold": False,
        }

        entries.append({
            "ticker": cand["ticker"],
            "name": cand["name"],
            "grade": grade,
            "price": round(price),
            "qty": qty,
            "cost": round(cost),
            "strategy": cand["strategy"],
        })

        new_today += 1
        slots_available -= 1

    return entries


# ═══════════════════════════════════════════════
# 일일 자산 기록 + 통계
# ═══════════════════════════════════════════════

def update_equity(pf: dict, today_str: str) -> float:
    """일일 자산 평가 + MDD 업데이트. 현재 equity 반환."""
    equity = pf["capital"]

    for ticker, pos in pf["positions"].items():
        price, _ = get_latest_price(ticker)
        if price > 0:
            equity += price * pos["qty"]
        else:
            equity += pos["avg_price"] * pos["qty"]

    # 중복 날짜 방지
    pf["daily_equity"] = [e for e in pf["daily_equity"] if e["date"] != today_str]
    pf["daily_equity"].append({
        "date": today_str,
        "equity": round(equity),
        "capital": round(pf["capital"]),
        "positions": len(pf["positions"]),
    })

    # MDD
    if pf["daily_equity"]:
        max_eq = max(e["equity"] for e in pf["daily_equity"])
        pf["stats"]["max_equity"] = max_eq
        if max_eq > 0:
            current_dd = (equity / max_eq - 1) * 100
            pf["stats"]["mdd"] = round(min(pf["stats"].get("mdd", 0), current_dd), 2)

    return equity


def calc_stats(pf: dict) -> dict:
    """성과 통계 계산."""
    initial = pf["initial_capital"]
    equity_list = pf.get("daily_equity", [])
    equity = equity_list[-1]["equity"] if equity_list else initial

    total = pf["stats"]["total_trades"]
    wins = pf["stats"]["wins"]
    losses = pf["stats"]["losses"]

    # Profit Factor
    gross_profit = sum(
        t["pnl_pct"] for t in pf["closed_trades"] if t["pnl_pct"] > 0
    )
    gross_loss = abs(sum(
        t["pnl_pct"] for t in pf["closed_trades"] if t["pnl_pct"] < 0
    ))
    pf_ratio = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

    return {
        "equity": round(equity),
        "total_return_pct": round((equity / initial - 1) * 100, 2),
        "pf": pf_ratio,
        "mdd": pf["stats"]["mdd"],
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "open_positions": len(pf["positions"]),
        "cash": round(pf["capital"]),
    }


# ═══════════════════════════════════════════════
# FLOWX 업로드
# ═══════════════════════════════════════════════

def upload_to_flowx(entries: list[dict], exits: list[dict], stats: dict) -> None:
    """FLOWX Supabase에 매매 기록 업로드."""
    try:
        from src.adapters.flowx_uploader import FlowxUploader, build_paper_trade
        uploader = FlowxUploader()
        if not uploader.is_active:
            return

        for e in entries:
            trade = build_paper_trade(
                code=e["ticker"], name=e["name"], side="BUY",
                price=e["price"], quantity=e["qty"],
                strategy=e["strategy"], memo=f"등급:{e['grade']}",
                stats=stats,
            )
            uploader.upload_paper_trade(trade)

        for x in exits:
            trade = build_paper_trade(
                code=x["ticker"], name=x["name"], side="SELL",
                price=x["exit_price"], quantity=x["qty"],
                strategy=x["reason"], pnl_pct=x["pnl_pct"],
                memo=f"{'부분' if x.get('partial') else '전량'}",
                stats=stats,
            )
            uploader.upload_paper_trade(trade)

        logger.info("[FLOWX] Paper 업로드: BUY %d건, SELL %d건", len(entries), len(exits))
    except Exception as e:
        logger.warning("[FLOWX] Paper 업로드 실패: %s", e)


# ═══════════════════════════════════════════════
# 텔레그램 리포트
# ═══════════════════════════════════════════════

def send_daily_report(
    today_str: str,
    entries: list[dict],
    exits: list[dict],
    pf: dict,
    stats: dict,
    candidates_count: int,
) -> None:
    """일일 Paper Trading 텔레그램 리포트."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        return

    lines = [
        f"📋 [PAPER] 일일 리포트 ({today_str})",
        f"자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)",
        f"PF: {stats['pf']} | MDD: {stats['mdd']:.1f}% | "
        f"승률: {stats['win_rate']:.0f}% ({stats['wins']}W/{stats['losses']}L)",
        "",
    ]

    if entries:
        lines.append(f"-- 신규 진입 ({len(entries)}건) --")
        for e in entries:
            lines.append(
                f"  [{e['grade']}] {e['name']} {e['price']:,}원 "
                f"x{e['qty']}주 ({e['strategy']})"
            )
        lines.append("")

    if exits:
        lines.append(f"-- 매도 ({len(exits)}건) --")
        for x in exits:
            emoji = "🟢" if x["pnl_pct"] > 0 else "🔴"
            partial = " (부분)" if x.get("partial") else ""
            lines.append(
                f"  {emoji} {x['name']} {x['pnl_pct']:+.1f}% "
                f"[{x['reason']}]{partial}"
            )
        lines.append("")

    if not entries and not exits:
        lines.append(f"스캔: {candidates_count}후보 | 변동 없음")
        lines.append("")

    # 보유 현황
    if pf["positions"]:
        lines.append(f"-- 보유 ({len(pf['positions'])}종목) --")
        for ticker, pos in pf["positions"].items():
            cur_price, _ = get_latest_price(ticker)
            if cur_price > 0:
                pnl = (cur_price / pos["avg_price"] - 1) * 100
            else:
                pnl = 0
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            days = 0
            try:
                days = (pd.Timestamp(today_str) - pd.Timestamp(pos["entry_date"])).days
            except Exception:
                pass
            lines.append(
                f"  {emoji} {pos['name']} {pnl:+.1f}% "
                f"({days}일, {pos.get('strategy', '')})"
            )

    msg = "\n".join(lines)
    try:
        send_message(msg)
    except Exception as e:
        logger.warning("텔레그램 전송 실패: %s", e)


def send_weekly_report(pf: dict, stats: dict) -> None:
    """금요일 주간 리포트."""
    try:
        from src.telegram_sender import send_message
    except ImportError:
        return

    equity_list = pf.get("daily_equity", [])
    if len(equity_list) < 2:
        return

    recent5 = equity_list[-5:]
    week_start_eq = recent5[0]["equity"]
    current_eq = equity_list[-1]["equity"]
    week_return = (current_eq / week_start_eq - 1) * 100 if week_start_eq > 0 else 0

    # 이번 주 거래
    week_start_date = recent5[0]["date"]
    week_trades = [
        t for t in pf["closed_trades"]
        if t.get("exit_date", "") >= week_start_date
    ]

    lines = [
        "📊 [PAPER] 주간 리포트",
        f"기간: {week_start_date} ~ {equity_list[-1]['date']}",
        "",
        f"총 자산: {stats['equity']:,}원 (누적 {stats['total_return_pct']:+.1f}%)",
        f"주간 수익: {week_return:+.1f}%",
        f"PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%",
        f"누적: {stats['total_trades']}건 ({stats['wins']}W/{stats['losses']}L)",
        "",
    ]

    if week_trades:
        lines.append(f"-- 금주 거래 ({len(week_trades)}건) --")
        for t in week_trades:
            emoji = "🟢" if t["pnl_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} {t['name']} {t['pnl_pct']:+.1f}% "
                f"[{t['exit_reason']}] {t.get('days_held', '?')}일"
            )
        lines.append("")

    if pf["positions"]:
        lines.append(f"-- 보유 ({len(pf['positions'])}종목) --")
        for ticker, pos in pf["positions"].items():
            cur_price, _ = get_latest_price(ticker)
            pnl = (cur_price / pos["avg_price"] - 1) * 100 if cur_price > 0 else 0
            lines.append(f"  {pos['name']} {pnl:+.1f}% ({pos.get('strategy', '')})")

    msg = "\n".join(lines)
    try:
        send_message(msg)
    except Exception as e:
        logger.warning("주간 리포트 전송 실패: %s", e)


# ═══════════════════════════════════════════════
# 메인 일일 실행
# ═══════════════════════════════════════════════

def run_daily() -> dict:
    """일일 Paper Trading 메인 루틴."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"  [PAPER] 통합 Paper Trading — {today_str}")
    print("=" * 60)

    # 1. 포트폴리오 로드
    pf = load_portfolio()
    print(f"  자본금: {pf['initial_capital']:,}원 | 현금: {pf['capital']:,}원")
    print(f"  보유: {len(pf['positions'])}종목")

    # 2. 매도 체크 (기존 포지션)
    exits = check_exits(pf, today_str)
    if exits:
        print(f"\n  매도 시그널: {len(exits)}건")
        for x in exits:
            print(f"    {x['name']} {x['pnl_pct']:+.1f}% [{x['reason']}]")

    # 3. 추천 종목 수집
    candidates = collect_candidates()
    print(f"\n  추천 후보: {len(candidates)}종목")
    for c in candidates[:5]:
        print(f"    [{c['grade']}] {c['name']} score={c['score']} "
              f"{c['price']:,}원 ({c['strategy']})")

    # 4. 신규 진입
    entries = enter_new_positions(pf, candidates, today_str)
    if entries:
        print(f"\n  신규 진입: {len(entries)}건")
        for e in entries:
            print(f"    [{e['grade']}] {e['name']} {e['price']:,}원 "
                  f"x{e['qty']}주 = {e['cost']:,}원")

    # 5. 일일 자산 기록
    equity = update_equity(pf, today_str)
    stats = calc_stats(pf)

    # 6. 저장
    save_portfolio(pf)

    # 7. 텔레그램 리포트
    send_daily_report(today_str, entries, exits, pf, stats, len(candidates))

    # 8. FLOWX 업로드
    upload_to_flowx(entries, exits, stats)

    # 9. 콘솔 요약
    print(f"\n  {'='*40}")
    print(f"  자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)")
    print(f"  PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%")
    print(f"  거래: {stats['total_trades']}건 "
          f"({stats['wins']}W/{stats['losses']}L) "
          f"승률 {stats['win_rate']:.0f}%")
    print(f"  보유: {stats['open_positions']}종목 | 현금: {stats['cash']:,}원")
    print("=" * 60)

    return {
        "status": "ok",
        "date": today_str,
        "entries": len(entries),
        "exits": len(exits),
        "candidates": len(candidates),
        **stats,
    }


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="통합 Paper Trading 엔진")
    parser.add_argument("--reset", action="store_true", help="포트폴리오 초기화 (3000만원)")
    parser.add_argument("--weekly", action="store_true", help="주간 리포트")
    parser.add_argument("--status", action="store_true", help="현재 상태")
    parser.add_argument("--dry-run", action="store_true", help="매매 없이 후보만 출력")
    args = parser.parse_args()

    if args.reset:
        pf = _default_portfolio()
        save_portfolio(pf)
        print("  [PAPER] 포트폴리오 초기화 완료 (3,000만원)")
        return

    if args.status:
        pf = load_portfolio()
        stats = calc_stats(pf)
        print(f"  자산: {stats['equity']:,}원 ({stats['total_return_pct']:+.1f}%)")
        print(f"  PF: {stats['pf']} | MDD: {stats['mdd']:.1f}%")
        print(f"  거래: {stats['total_trades']}건 "
              f"({stats['wins']}W/{stats['losses']}L) 승률 {stats['win_rate']:.0f}%")
        print(f"  보유: {stats['open_positions']}종목 | 현금: {stats['cash']:,}원")
        if pf["positions"]:
            print("  -- 보유 종목 --")
            for ticker, pos in pf["positions"].items():
                cur_price, _ = get_latest_price(ticker)
                pnl = (cur_price / pos["avg_price"] - 1) * 100 if cur_price > 0 else 0
                print(f"    {pos['name']} {pnl:+.1f}% | "
                      f"진입 {pos['avg_price']:,}원 | {pos.get('strategy', '')}")
        return

    if args.weekly:
        pf = load_portfolio()
        stats = calc_stats(pf)
        send_weekly_report(pf, stats)
        print("  주간 리포트 전송 완료")
        return

    if args.dry_run:
        candidates = collect_candidates()
        print(f"\n  [DRY-RUN] 후보 {len(candidates)}종목:")
        for i, c in enumerate(candidates, 1):
            print(f"    {i}. [{c['grade']}] {c['name']} score={c['score']} "
                  f"{c['price']:,}원 ({c['strategy']}) — {c['reason']}")
        return

    result = run_daily()
    print(f"\n  결과: {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
