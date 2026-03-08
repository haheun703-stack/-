"""역발상 저점매집 전략 — 페이퍼 트레이딩 엔진.

매일 장마감 후 실행:
  python -u -X utf8 scripts/paper_trader.py

기능:
  1. 시총 1,000억+ 유니버스 스캔
  2. 매크로 공포 점수 → 종목 과매도 스크리닝
  3. BRAIN 레짐 연동 (brain_decision.json)
  4. 가상 포트폴리오 관리 (paper_portfolio.json)
  5. 텔레그램 알림 ([PAPER] 태그)
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.contrarian.config import (
    COMMISSION_PCT,
    COOLDOWN_DAYS,
    DUAL_STOPLOSS_PAUSE,
    MACRO_ENTRY_THRESHOLD,
    MACRO_ENTRY_THRESHOLD_BEAR,
    MAX_PER_STOCK_PCT,
    MAX_STOCKS,
    MONTHLY_LOSS_LIMIT,
    REGIME_CAP,
    SLIPPAGE_PCT,
    SLOT_CAPITAL_PCT,
    SLOT_MDD_LIMIT,
    SPLIT_BUY,
    STOCK_ENTRY_THRESHOLD,
    STOP_LOSS_PCT,
    TAX_PCT,
    TAKE_PROFIT_TIERS,
    TRAILING_ACTIVATE_PCT,
    TRAILING_STOP_PCT,
)

# 백테스트의 스크리닝 함수 재사용
from scripts.backtest_contrarian import (
    calc_macro_fear_score,
    load_all_data,
    screen_stocks,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 경로
# ═══════════════════════════════════════════════
DATA_DIR = PROJECT_ROOT / "data"
PORTFOLIO_PATH = DATA_DIR / "paper_portfolio.json"
BRAIN_PATH = DATA_DIR / "brain_decision.json"


# ═══════════════════════════════════════════════
# 가상 포트폴리오 관리
# ═══════════════════════════════════════════════

def _default_portfolio() -> dict:
    """빈 포트폴리오 초기화."""
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "initial_capital": 15_000_000,  # 1,500만원
        "capital": 15_000_000,
        "positions": {},
        "closed_trades": [],
        "daily_equity": [],
        "cooldown_until": "",
        "monthly_loss": {},
        "stats": {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_pct": 0.0,
            "max_equity": 15_000_000,
            "mdd": 0.0,
        },
    }


def load_portfolio() -> dict:
    """paper_portfolio.json 로드. 없으면 초기화."""
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    pf = _default_portfolio()
    save_portfolio(pf)
    return pf


def save_portfolio(pf: dict) -> None:
    """paper_portfolio.json 저장."""
    pf["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# BRAIN 레짐 읽기
# ═══════════════════════════════════════════════

def get_brain_regime() -> tuple[str, float]:
    """brain_decision.json에서 현재 레짐 + VIX 읽기.

    Returns:
        (regime, vix_level) — 파일 없으면 ("CAUTION", 0)
    """
    if not BRAIN_PATH.exists():
        return "CAUTION", 0.0

    try:
        with open(BRAIN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        regime = data.get("effective_regime", "CAUTION")
        vix = data.get("vix_level", 0.0)
        return regime, vix
    except Exception:
        return "CAUTION", 0.0


# ═══════════════════════════════════════════════
# 매도 처리
# ═══════════════════════════════════════════════

def check_sells(pf: dict, stocks: dict, today: pd.Timestamp) -> list[dict]:
    """보유 종목의 매도 조건 체크. 매도 시그널 리스트 반환."""
    today_str = today.strftime("%Y-%m-%d")
    sell_signals = []

    codes_to_close = []
    for code, pos in list(pf["positions"].items()):
        if code not in stocks or today not in stocks[code].index:
            continue

        idx = stocks[code].index.get_loc(today)
        if isinstance(idx, slice):
            idx = idx.start
        price = float(stocks[code]["close"].iloc[idx])

        avg_price = pos["avg_price"]
        peak_price = pos.get("peak_price", avg_price)

        # 최고가 갱신
        if price > peak_price:
            pos["peak_price"] = price
            peak_price = price

        pnl_pct = price / avg_price - 1

        # 1. 손절 -7%
        if pnl_pct <= STOP_LOSS_PCT:
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * pos["total_qty"] * (1 - TAX_PCT)
            sell_signals.append({
                "code": code,
                "name": pos["name"],
                "reason": "STOP_LOSS",
                "price": price,
                "sell_price": sell_price,
                "pnl_pct": pnl_pct,
                "proceeds": proceeds,
                "qty": pos["total_qty"],
            })
            _close_position(pf, code, today_str, sell_price, pnl_pct, "STOP_LOSS")
            codes_to_close.append(code)
            continue

        # 2. 트레일링 스탑
        if pos.get("trailing_active", False):
            trail_pnl = price / peak_price - 1
            if trail_pnl <= TRAILING_STOP_PCT:
                sell_price = price * (1 - SLIPPAGE_PCT)
                proceeds = sell_price * pos["total_qty"] * (1 - TAX_PCT)
                sell_signals.append({
                    "code": code,
                    "name": pos["name"],
                    "reason": "TRAILING_STOP",
                    "price": price,
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct,
                    "proceeds": proceeds,
                    "qty": pos["total_qty"],
                })
                _close_position(pf, code, today_str, sell_price, pnl_pct, "TRAILING_STOP")
                codes_to_close.append(code)
                continue

        # 3. 구간 익절
        if not pos.get("tier1_sold") and pnl_pct >= TAKE_PROFIT_TIERS[0]["from"]:
            sell_qty = max(1, int(pos["total_qty"] * TAKE_PROFIT_TIERS[0]["sell_pct"]))
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * sell_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds
            pos["total_qty"] -= sell_qty
            pos["tier1_sold"] = True
            sell_signals.append({
                "code": code,
                "name": pos["name"],
                "reason": "TAKE_PROFIT_T1",
                "price": price,
                "sell_price": sell_price,
                "pnl_pct": pnl_pct,
                "proceeds": proceeds,
                "qty": sell_qty,
            })
            if pos["total_qty"] <= 0:
                _close_position(pf, code, today_str, sell_price, pnl_pct, "TAKE_PROFIT_T1")
                codes_to_close.append(code)
            continue

        if not pos.get("tier2_sold") and pnl_pct >= TAKE_PROFIT_TIERS[1]["from"]:
            sell_qty = max(1, int(pos["total_qty"] * TAKE_PROFIT_TIERS[1]["sell_pct"]))
            sell_price = price * (1 - SLIPPAGE_PCT)
            proceeds = sell_price * sell_qty * (1 - TAX_PCT)
            pf["capital"] += proceeds
            pos["total_qty"] -= sell_qty
            pos["tier2_sold"] = True
            sell_signals.append({
                "code": code,
                "name": pos["name"],
                "reason": "TAKE_PROFIT_T2",
                "price": price,
                "sell_price": sell_price,
                "pnl_pct": pnl_pct,
                "proceeds": proceeds,
                "qty": sell_qty,
            })
            if pos["total_qty"] <= 0:
                _close_position(pf, code, today_str, sell_price, pnl_pct, "TAKE_PROFIT_T2")
                codes_to_close.append(code)
            continue

        # 4. 트레일링 활성화
        if pnl_pct >= TRAILING_ACTIVATE_PCT:
            pos["trailing_active"] = True

    for code in codes_to_close:
        if code in pf["positions"]:
            del pf["positions"][code]

    return sell_signals


def _close_position(
    pf: dict, code: str, exit_date: str,
    exit_price: float, pnl_pct: float, reason: str,
) -> None:
    """포지션 청산 기록."""
    pos = pf["positions"].get(code)
    if not pos:
        return

    proceeds = exit_price * pos["total_qty"] * (1 - TAX_PCT)
    pf["capital"] += proceeds

    pf["closed_trades"].append({
        "code": code,
        "name": pos["name"],
        "entry_date": pos["entry_date"],
        "exit_date": exit_date,
        "avg_price": pos["avg_price"],
        "exit_price": exit_price,
        "qty": pos["total_qty"],
        "pnl_pct": round(pnl_pct, 4),
        "exit_reason": reason,
        "phases": pos.get("current_phase", 1),
    })

    # 통계 업데이트
    pf["stats"]["total_trades"] += 1
    if pnl_pct > 0:
        pf["stats"]["wins"] += 1
    else:
        pf["stats"]["losses"] += 1


# ═══════════════════════════════════════════════
# 분할매수 2/3차
# ═══════════════════════════════════════════════

def check_split_buys(
    pf: dict, stocks: dict, today: pd.Timestamp,
    effective_capital: float,
) -> list[dict]:
    """기존 포지션 추가매수 체크."""
    buy_signals = []

    for code, pos in list(pf["positions"].items()):
        if code not in stocks or today not in stocks[code].index:
            continue

        current_phase = pos.get("current_phase", 1)
        if current_phase >= 3:
            continue

        idx = stocks[code].index.get_loc(today)
        if isinstance(idx, slice):
            idx = idx.start
        price = float(stocks[code]["close"].iloc[idx])

        next_phase = current_phase + 1
        should_buy = False

        if next_phase == 2:
            entry1_price = pos["entries"][0]["price"]
            entry1_date = pos["entries"][0]["date"]
            days_since = (today - pd.Timestamp(entry1_date)).days
            if price <= entry1_price * 0.97:
                should_buy = True
            elif days_since >= 3 and price > pos["avg_price"]:
                should_buy = True

        elif next_phase == 3:
            if price <= pos["avg_price"] * 0.95:
                should_buy = True
            elif (not pd.isna(stocks[code]["volume_surge_ratio"].iloc[idx])
                  and stocks[code]["volume_surge_ratio"].iloc[idx] >= 1.5
                  and stocks[code].get("is_bullish", pd.Series()).iloc[idx]
                  if "is_bullish" in stocks[code].columns else False):
                should_buy = True

        if should_buy:
            invested = pos["avg_price"] * pos["total_qty"]
            buy_pct = SPLIT_BUY[next_phase]["pct"]
            buy_amount = min(
                effective_capital * buy_pct,
                effective_capital * MAX_PER_STOCK_PCT - invested,
                pf["capital"] * 0.95,
            )
            if buy_amount > price * 1.01:
                qty = int(buy_amount / (price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)))
                if qty > 0:
                    cost = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT) * qty
                    pf["capital"] -= cost

                    # 엔트리 추가
                    entry_price = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
                    pos["entries"].append({
                        "date": today.strftime("%Y-%m-%d"),
                        "price": entry_price,
                        "qty": qty,
                        "phase": next_phase,
                    })
                    # 평균단가 재계산
                    total_cost = sum(e["price"] * e["qty"] for e in pos["entries"])
                    total_qty = sum(e["qty"] for e in pos["entries"])
                    pos["avg_price"] = total_cost / total_qty
                    pos["total_qty"] = total_qty
                    pos["current_phase"] = next_phase

                    buy_signals.append({
                        "code": code,
                        "name": pos["name"],
                        "phase": next_phase,
                        "price": price,
                        "qty": qty,
                        "cost": cost,
                    })

    return buy_signals


# ═══════════════════════════════════════════════
# 메인 일일 실행
# ═══════════════════════════════════════════════

def run_daily(force: bool = False) -> dict:
    """일일 페이퍼 트레이딩 실행.

    Args:
        force: True이면 주말/공휴일에도 강제 실행 (최근 거래일 기준)

    Returns:
        결과 요약 dict
    """
    from src.telegram_sender import send_message

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    # 주말/공휴일 가드
    if today.weekday() >= 5 and not force:
        print(f"  [PAPER] {today_str} 주말 — 스킵")
        return {"status": "skip", "reason": "weekend"}

    print("=" * 60)
    print(f"  [PAPER] 역발상 페이퍼 트레이딩 — {today_str}")
    print("=" * 60)

    # 1. 포트폴리오 로드
    pf = load_portfolio()
    initial_capital = pf["initial_capital"]

    # 2. BRAIN 레짐 읽기
    regime, vix = get_brain_regime()
    regime_mult = REGIME_CAP.get(regime, 1.0)
    effective_capital = initial_capital * regime_mult
    print(f"  BRAIN 레짐: {regime} (×{regime_mult}) | VIX: {vix}")

    # 3. 킬스위치/냉각기 체크
    if pf.get("cooldown_until") and today_str < pf["cooldown_until"]:
        msg = (
            f"[PAPER] {today_str} 냉각기 중 "
            f"(~{pf['cooldown_until']}) — 스캔만 수행"
        )
        print(f"  {msg}")
        send_message(msg)
        return {"status": "cooldown"}

    # CRISIS/PANIC → 신규 진입 불가
    if regime_mult == 0:
        msg = (
            f"[PAPER] 역발상 스캔 완료 ({today_str})\n"
            f"BRAIN 레짐: {regime} — 신규 진입 차단\n"
            f"VIX: {vix:.1f}"
        )
        print(f"  {msg}")
        send_message(msg)
        # 보유 종목 매도 체크는 계속
        # (아래에서 데이터 로드 후 처리)

    # 4. 데이터 로드 (최근 120일)
    end_str = today_str
    start_dt = today - timedelta(days=180)
    start_str = start_dt.strftime("%Y-%m-%d")

    print(f"  데이터 로드: {start_str} ~ {end_str}")
    stocks, stock_names, kospi, us = load_all_data(
        start_str, end_str, min_market_cap=1000,
    )
    print(f"  종목: {len(stocks)}개 로드")

    # 오늘 날짜가 거래일인지 확인
    # parquet 데이터는 전일까지만 있을 수 있으므로 가장 최근 거래일 사용
    trading_days = sorted(kospi.index)
    if not trading_days:
        print("  [PAPER] 거래일 데이터 없음 — 스킵")
        return {"status": "skip", "reason": "no_trading_days"}

    scan_day = trading_days[-1]  # 가장 최근 거래일
    scan_day_str = scan_day.strftime("%Y-%m-%d")
    print(f"  스캔 기준일: {scan_day_str}")

    # 5. 보유 종목 매도 체크
    sell_signals = check_sells(pf, stocks, scan_day)
    stoploss_count = sum(1 for s in sell_signals if s["reason"] == "STOP_LOSS")

    # 동시 2종목 손절 킬스위치
    if DUAL_STOPLOSS_PAUSE and stoploss_count >= 2:
        cool_idx = trading_days.index(scan_day)
        cool_target = cool_idx + COOLDOWN_DAYS
        if cool_target < len(trading_days):
            pf["cooldown_until"] = trading_days[cool_target].strftime("%Y-%m-%d")

    # 6. 분할매수 체크
    split_buys = []
    if regime_mult > 0:
        split_buys = check_split_buys(pf, stocks, scan_day, effective_capital)

    # 7. 매크로 공포 점수 + 신규 스크리닝
    fear_score, fear_details = calc_macro_fear_score(scan_day, kospi, us, stocks)

    entry_threshold = (
        MACRO_ENTRY_THRESHOLD_BEAR if regime == "BEAR"
        else MACRO_ENTRY_THRESHOLD
    )

    new_candidates = []
    if (regime_mult > 0
        and len(pf["positions"]) < MAX_STOCKS
        and fear_score >= entry_threshold
        and pf["capital"] > effective_capital * 0.10):

        candidates = screen_stocks(scan_day, stocks, stock_names)
        # 이미 보유 중 제외
        candidates = [c for c in candidates if c["code"] not in pf["positions"]]

        slots_available = MAX_STOCKS - len(pf["positions"])
        for cand in candidates[:slots_available]:
            buy_pct = SPLIT_BUY[1]["pct"]
            buy_amount = min(
                effective_capital * buy_pct,
                pf["capital"] * 0.95,
            )
            price = cand["price"]
            if buy_amount > price * 1.01:
                qty = int(buy_amount / (price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)))
                if qty > 0:
                    cost = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT) * qty
                    entry_price = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
                    pf["capital"] -= cost

                    pf["positions"][cand["code"]] = {
                        "name": cand["name"],
                        "entry_date": scan_day_str,
                        "entries": [{
                            "date": scan_day_str,
                            "price": entry_price,
                            "qty": qty,
                            "phase": 1,
                        }],
                        "avg_price": entry_price,
                        "total_qty": qty,
                        "peak_price": price,
                        "current_phase": 1,
                        "trailing_active": False,
                        "tier1_sold": False,
                        "tier2_sold": False,
                    }

                    new_candidates.append({
                        "code": cand["code"],
                        "name": cand["name"],
                        "price": price,
                        "score": cand["score"],
                        "reasons": cand["reasons"],
                        "qty": qty,
                        "cost": cost,
                    })

    # 7.5. 시그널 발견 시 tomorrow_picks.json에 paper_mode로 추가
    _inject_to_tomorrow_picks(new_candidates, scan_day_str, fear_score)

    # 8. 일일 자산 기록
    prices = {}
    for code in pf["positions"]:
        if code in stocks and scan_day in stocks[code].index:
            idx = stocks[code].index.get_loc(scan_day)
            if isinstance(idx, slice):
                idx = idx.start
            prices[code] = float(stocks[code]["close"].iloc[idx])

    equity = pf["capital"]
    for code, pos in pf["positions"].items():
        if code in prices:
            equity += prices[code] * pos["total_qty"]

    pf["daily_equity"].append({
        "date": scan_day_str,
        "equity": round(equity, 0),
        "capital": round(pf["capital"], 0),
        "positions": len(pf["positions"]),
    })

    # MDD 업데이트
    max_eq = max(e["equity"] for e in pf["daily_equity"])
    pf["stats"]["max_equity"] = max_eq
    current_mdd = (equity / max_eq - 1) * 100 if max_eq > 0 else 0
    pf["stats"]["mdd"] = round(min(pf["stats"].get("mdd", 0), current_mdd), 2)

    # MDD 킬스위치
    if current_mdd <= SLOT_MDD_LIMIT * 100:  # -10%
        # 전량 매도
        for code, pos in list(pf["positions"].items()):
            if code in prices:
                sell_price = prices[code] * (1 - SLIPPAGE_PCT)
                pnl_pct = sell_price / pos["avg_price"] - 1
                _close_position(pf, code, scan_day_str, sell_price, pnl_pct, "KILLSWITCH_MDD")
                sell_signals.append({
                    "code": code,
                    "name": pos["name"],
                    "reason": "KILLSWITCH_MDD",
                    "price": prices[code],
                    "sell_price": sell_price,
                    "pnl_pct": pnl_pct,
                    "proceeds": sell_price * pos["total_qty"] * (1 - TAX_PCT),
                    "qty": pos["total_qty"],
                })
        pf["positions"].clear()
        cool_idx = trading_days.index(scan_day)
        cool_target = cool_idx + COOLDOWN_DAYS
        if cool_target < len(trading_days):
            pf["cooldown_until"] = trading_days[cool_target].strftime("%Y-%m-%d")

    # 9. 저장
    save_portfolio(pf)

    # 10. 텔레그램 알림
    total_return = (equity / initial_capital - 1) * 100
    _send_telegram_report(
        scan_day_str, regime, vix, fear_score, fear_details,
        new_candidates, sell_signals, split_buys,
        pf, equity, total_return,
        len(stocks),
    )

    # 11. 콘솔 요약
    print()
    print(f"  공포점수: {fear_score}/60 (임계치: {entry_threshold})")
    print(f"  신규 시그널: {len(new_candidates)}건")
    print(f"  매도 시그널: {len(sell_signals)}건")
    print(f"  분할매수: {len(split_buys)}건")
    print(f"  보유종목: {len(pf['positions'])}개")
    print(f"  자산: {equity:,.0f}원 ({total_return:+.1f}%)")
    print(f"  MDD: {pf['stats']['mdd']:.1f}%")
    print(f"  거래: {pf['stats']['total_trades']}건 "
          f"({pf['stats']['wins']}W/{pf['stats']['losses']}L)")
    print("=" * 60)

    return {
        "status": "ok",
        "date": scan_day_str,
        "regime": regime,
        "fear_score": fear_score,
        "new_entries": len(new_candidates),
        "sells": len(sell_signals),
        "split_buys": len(split_buys),
        "equity": equity,
        "return_pct": round(total_return, 2),
    }


# ═══════════════════════════════════════════════
# tomorrow_picks.json 연동
# ═══════════════════════════════════════════════

TOMORROW_PICKS_PATH = DATA_DIR / "tomorrow_picks.json"


def _inject_to_tomorrow_picks(
    candidates: list[dict],
    scan_date: str,
    fear_score: int,
) -> None:
    """역발상 시그널을 tomorrow_picks.json에 paper_mode로 병합.

    기존 picks를 유지하면서 역발상 후보를 추가한다.
    이미 있는 ticker는 중복 추가하지 않는다.
    시그널이 없어도 이전 역발상 picks를 정리한다.
    """
    # 기존 파일 로드
    if TOMORROW_PICKS_PATH.exists():
        with open(TOMORROW_PICKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"generated_at": "", "picks": []}

    # 기존 역발상(paper_mode) picks 제거 — 매일 갱신
    data["picks"] = [p for p in data["picks"] if not p.get("paper_mode")]

    existing_tickers = {p["ticker"] for p in data["picks"]}

    added = 0
    for cand in candidates:
        if cand["code"] in existing_tickers:
            continue

        pick = {
            "ticker": cand["code"],
            "name": cand["name"],
            "grade": "관심매수",
            "total_score": cand["score"],
            "close": int(cand["price"]),
            "entry_price": int(cand["price"]),
            "stop_loss": int(cand["price"] * (1 + STOP_LOSS_PCT)),
            "target_price": int(cand["price"] * (1 + TRAILING_ACTIVATE_PCT)),
            "reasons": cand["reasons"],
            "sources": ["역발상"],
            "paper_mode": True,
            "contrarian": {
                "fear_score": fear_score,
                "stock_score": cand["score"],
                "scan_date": scan_date,
            },
        }
        data["picks"].append(pick)
        existing_tickers.add(cand["code"])
        added += 1

    with open(TOMORROW_PICKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if added:
        print(f"  [PAPER] tomorrow_picks에 {added}건 역발상 후보 추가 (paper_mode)")
    else:
        print("  [PAPER] tomorrow_picks: 역발상 후보 0건 (이전 paper picks 정리)")


# ═══════════════════════════════════════════════
# 텔레그램 알림
# ═══════════════════════════════════════════════

def _send_telegram_report(
    date: str, regime: str, vix: float,
    fear_score: int, fear_details: dict,
    new_entries: list, sells: list, split_buys: list,
    pf: dict, equity: float, total_return: float,
    universe_count: int,
) -> None:
    """텔레그램 알림 전송."""
    from src.telegram_sender import send_message

    lines = []

    # 헤더
    lines.append(f"[PAPER] 역발상 일일 리포트 ({date})")
    lines.append(f"BRAIN: {regime} | VIX: {vix:.1f} | 공포: {fear_score}/60")
    lines.append("")

    # 신규 진입
    if new_entries:
        lines.append("-- 신규 시그널 --")
        for e in new_entries:
            reasons_str = ", ".join(e["reasons"])
            lines.append(
                f"  {e['name']}({e['code']}) "
                f"{e['price']:,.0f}원 | 점수 {e['score']} | "
                f"{e['qty']}주 ({e['cost']:,.0f}원)"
            )
            lines.append(f"    사유: {reasons_str}")
        lines.append("")

    # 매도
    if sells:
        lines.append("-- 매도 시그널 --")
        for s in sells:
            lines.append(
                f"  {s['name']}({s['code']}) "
                f"{s['pnl_pct']*100:+.1f}% [{s['reason']}] "
                f"{s['qty']}주"
            )
        lines.append("")

    # 분할매수
    if split_buys:
        lines.append("-- 분할매수 --")
        for b in split_buys:
            lines.append(
                f"  {b['name']}({b['code']}) "
                f"P{b['phase']} {b['price']:,.0f}원 {b['qty']}주"
            )
        lines.append("")

    # 시그널 없음
    if not new_entries and not sells and not split_buys:
        lines.append(f"스캔 완료: {universe_count:,}종목 중 후보 0건")
        if fear_details:
            details_str = ", ".join(f"{k}={v}" for k, v in fear_details.items())
            lines.append(f"  공포 세부: {details_str}")
        lines.append("")

    # 포트폴리오 현황
    lines.append("-- 포트폴리오 --")
    lines.append(f"  자산: {equity:,.0f}원 ({total_return:+.1f}%)")
    lines.append(f"  현금: {pf['capital']:,.0f}원")
    lines.append(f"  MDD: {pf['stats']['mdd']:.1f}%")
    lines.append(f"  거래: {pf['stats']['total_trades']}건 "
                 f"({pf['stats']['wins']}W/{pf['stats']['losses']}L)")

    if pf["positions"]:
        lines.append("  보유:")
        for code, pos in pf["positions"].items():
            lines.append(
                f"    {pos['name']} P{pos.get('current_phase', 1)} "
                f"평균 {pos['avg_price']:,.0f}원 {pos['total_qty']}주"
            )

    msg = "\n".join(lines)
    send_message(msg)


# ═══════════════════════════════════════════════
# 주간 리포트
# ═══════════════════════════════════════════════

def weekly_report() -> None:
    """금요일 주간 요약 리포트."""
    from src.telegram_sender import send_message

    pf = load_portfolio()

    if not pf["daily_equity"]:
        print("  [PAPER] 데이터 없음 — 주간 리포트 스킵")
        return

    initial = pf["initial_capital"]
    latest = pf["daily_equity"][-1]
    equity = latest["equity"]
    total_return = (equity / initial - 1) * 100

    # 최근 5거래일
    recent = pf["daily_equity"][-5:]
    if len(recent) >= 2:
        week_start = recent[0]["equity"]
        week_return = (equity / week_start - 1) * 100
    else:
        week_return = 0.0

    # 최근 5거래일 거래
    recent_trades = [
        t for t in pf["closed_trades"]
        if t["exit_date"] >= recent[0]["date"]
    ] if recent else []

    lines = [
        "[PAPER] 역발상 주간 리포트",
        f"기간: {recent[0]['date']} ~ {latest['date']}" if recent else "",
        "",
        f"총 자산: {equity:,.0f}원 (누적 {total_return:+.1f}%)",
        f"주간 수익: {week_return:+.1f}%",
        f"MDD: {pf['stats']['mdd']:.1f}%",
        f"누적 거래: {pf['stats']['total_trades']}건 "
        f"({pf['stats']['wins']}W/{pf['stats']['losses']}L)",
        "",
    ]

    if recent_trades:
        lines.append("-- 금주 거래 --")
        for t in recent_trades:
            lines.append(
                f"  {t['name']} {t['pnl_pct']*100:+.1f}% [{t['exit_reason']}]"
            )
        lines.append("")

    if pf["positions"]:
        lines.append("-- 보유 종목 --")
        for code, pos in pf["positions"].items():
            lines.append(
                f"  {pos['name']} P{pos.get('current_phase', 1)} "
                f"평균 {pos['avg_price']:,.0f}원"
            )

    msg = "\n".join(lines)
    send_message(msg)
    print("  [PAPER] 주간 리포트 전송 완료")


# ═══════════════════════════════════════════════
# 포트폴리오 초기화
# ═══════════════════════════════════════════════

def reset_portfolio() -> None:
    """포트폴리오 완전 초기화."""
    pf = _default_portfolio()
    save_portfolio(pf)
    print("  [PAPER] 포트폴리오 초기화 완료")


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="역발상 페이퍼 트레이딩")
    parser.add_argument("--reset", action="store_true", help="포트폴리오 초기화")
    parser.add_argument("--weekly", action="store_true", help="주간 리포트 전송")
    parser.add_argument("--status", action="store_true", help="현재 상태 출력")
    parser.add_argument("--force", action="store_true", help="주말에도 강제 실행")
    args = parser.parse_args()

    if args.reset:
        reset_portfolio()
    elif args.weekly:
        weekly_report()
    elif args.status:
        pf = load_portfolio()
        initial = pf["initial_capital"]
        eq_list = pf["daily_equity"]
        equity = eq_list[-1]["equity"] if eq_list else initial
        ret = (equity / initial - 1) * 100

        print(f"  자산: {equity:,.0f}원 ({ret:+.1f}%)")
        print(f"  현금: {pf['capital']:,.0f}원")
        print(f"  보유: {len(pf['positions'])}종목")
        print(f"  거래: {pf['stats']['total_trades']}건 "
              f"({pf['stats']['wins']}W/{pf['stats']['losses']}L)")
        print(f"  MDD: {pf['stats']['mdd']:.1f}%")
        if pf["positions"]:
            print("  -- 보유 종목 --")
            for code, pos in pf["positions"].items():
                print(f"    {pos['name']} P{pos.get('current_phase', 1)} "
                      f"{pos['avg_price']:,.0f}원 {pos['total_qty']}주")
    else:
        result = run_daily(force=args.force)
        print(f"\n  결과: {result}")
