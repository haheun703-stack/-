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
    """15:30 종가 기록 + 당일 적중률 계산 + paper P&L 시뮬 + reflection 업데이트.

    5/28 P0 추가: paper trading P&L 시뮬레이션
      - 종목당 가상 매수 PAPER_BUDGET_PER_TICKER(기본 100,000원)
      - 시초가 매수 → 종가 매도 (시뮬)
      - 수수료 0.015% × 2 + 거래세 0.18% (매도 시)
      - 누적 P&L data/paper_trader_history.json 추적
    """
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = reflection_path(today)
    if not out_path.exists():
        print(f"[CLOSE] {today} reflection 파일 없음 — --open 먼저 실행 필요")
        return 1

    reflection = json.loads(out_path.read_text(encoding="utf-8"))
    adp = KisStockDataAdapter()
    broker = adp.broker

    # paper trading 파라미터 (PaperOrderAdapter 가정 동일)
    import os
    budget_per_ticker = int(os.getenv("PAPER_BUDGET_PER_TICKER", "100000"))
    fee_pct = 0.015 / 100   # 매수+매도 각 0.015%
    tax_pct = 0.18 / 100    # 매도 시 (코스피 디폴트, 보수)

    n_hits = 0
    total_buy_cost = 0
    total_sell_proceed = 0
    total_pnl = 0
    n_paper_records = 0
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

        # paper P&L 시뮬: 시초가 매수 → 종가 매도
        if op > 0 and cl > 0:
            qty = budget_per_ticker // op
            if qty > 0:
                buy_cost = qty * op * (1 + fee_pct)
                sell_proceed = qty * cl * (1 - fee_pct - tax_pct)
                pnl = int(sell_proceed - buy_cost)
                pnl_pct = (pnl / buy_cost * 100) if buy_cost > 0 else 0
                rec["paper_qty"] = qty
                rec["paper_buy_cost"] = int(buy_cost)
                rec["paper_sell_proceed"] = int(sell_proceed)
                rec["paper_pnl"] = pnl
                rec["paper_pnl_pct"] = round(pnl_pct, 2)
                total_buy_cost += buy_cost
                total_sell_proceed += sell_proceed
                total_pnl += pnl
                n_paper_records += 1

    n = len(reflection.get("records", []))
    reflection["mode"] = "close"
    reflection["intraday_accuracy"] = (n_hits / n) if n > 0 else 0.0
    reflection["intraday_avg_chg_pct"] = (
        sum(r.get("intraday_chg_pct", 0) for r in reflection["records"]) / n if n > 0 else 0.0
    )
    # paper P&L 요약
    paper_pnl_pct = (total_pnl / total_buy_cost * 100) if total_buy_cost > 0 else 0
    reflection["paper_total_buy_cost"] = int(total_buy_cost)
    reflection["paper_total_sell_proceed"] = int(total_sell_proceed)
    reflection["paper_total_pnl"] = int(total_pnl)
    reflection["paper_total_pnl_pct"] = round(paper_pnl_pct, 2)
    reflection["paper_n_records"] = n_paper_records
    reflection["paper_budget_per_ticker"] = budget_per_ticker
    reflection["closed_at"] = datetime.now().isoformat()

    out_path.write_text(json.dumps(reflection, indent=2, ensure_ascii=False), encoding="utf-8")

    # 누적 P&L history 갱신
    history_path = PROJECT_ROOT / "data" / "paper_trader_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = {"sessions": [], "cumulative_pnl": 0}
    else:
        history = {"sessions": [], "cumulative_pnl": 0}
    history["sessions"].append({
        "date": today,
        "n_records": n_paper_records,
        "buy_cost": int(total_buy_cost),
        "sell_proceed": int(total_sell_proceed),
        "pnl": int(total_pnl),
        "pnl_pct": round(paper_pnl_pct, 2),
        "intraday_accuracy": reflection["intraday_accuracy"],
    })
    history["cumulative_pnl"] = sum(s["pnl"] for s in history["sessions"])
    history["n_sessions"] = len(history["sessions"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[CLOSE] {today} — 적중률 {reflection['intraday_accuracy']:.1%} ({n_hits}/{n}), "
        f"평균 {reflection['intraday_avg_chg_pct']:+.2f}%"
    )
    print(
        f"[PAPER P&L] 당일 {total_pnl:+,}원 ({paper_pnl_pct:+.2f}%) | "
        f"누적 {history['cumulative_pnl']:+,}원 ({history['n_sessions']}일)"
    )

    if send_tg:
        try:
            from src.telegram_sender import send_message

            # paper P&L 종목별 TOP 3
            sorted_by_pnl = sorted(
                [r for r in reflection["records"] if "paper_pnl" in r],
                key=lambda x: x["paper_pnl"], reverse=True,
            )
            top_lines = []
            for r in sorted_by_pnl[:3]:
                top_lines.append(
                    f"  {r['name']} {r['paper_pnl']:+,}원 ({r['paper_pnl_pct']:+.2f}%)"
                )
            bot_lines = []
            for r in sorted_by_pnl[-3:]:
                if r["paper_pnl"] < 0:
                    bot_lines.append(
                        f"  {r['name']} {r['paper_pnl']:+,}원 ({r['paper_pnl_pct']:+.2f}%)"
                    )

            msg_parts = [
                f"[PAPER TRADING] {today}",
                f"적중률 {reflection['intraday_accuracy']:.1%} ({n_hits}/{n})",
                f"평균 변동 {reflection['intraday_avg_chg_pct']:+.2f}%",
                "",
                f"💰 당일 P&L: {total_pnl:+,}원 ({paper_pnl_pct:+.2f}%)",
                f"📊 누적 P&L: {history['cumulative_pnl']:+,}원 ({history['n_sessions']}일)",
                f"종목당 시드: {budget_per_ticker:,}원 × {n_paper_records}건",
            ]
            if top_lines:
                msg_parts.append("\n🏆 TOP 3:")
                msg_parts.extend(top_lines)
            if bot_lines:
                msg_parts.append("\n📉 손실:")
                msg_parts.extend(bot_lines)

            send_message("\n".join(msg_parts))
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)

    return 0


def cmd_paper_trade_open(top_n: int, send_tg: bool) -> int:
    """Phase 1 paper 호출처 마이그레이션 (5/28 코덱스 5차 PASS).

    Trading Factory v1: register_intent → PaperOrderAdapter.buy_limit
    - tomorrow_picks 강력포착 TOP N → mode='paper' + executor_bot='quant' intent 등록
    - PaperOrderAdapter.buy_limit 호출 → assert_order_intent_exists 통과 검증
    - 결과를 data/phase1_paper_trades/{date}_open.json에 기록

    추적표: docs/02-design/phase-1-migration-tracking.md
    """
    from datetime import timedelta, timezone
    from src.use_cases.order_intents_gate import register_intent, NoIntentError
    from src.adapters.paper_order_adapter import PaperOrderAdapter
    from src.entities.trading_models import OrderStatus

    today = datetime.now().strftime("%Y-%m-%d")
    picks = fetch_picks(top_n=top_n)
    if not picks:
        print(f"[PAPER TRADE OPEN] {today} 강력포착 후보 0건")
        return 1

    adp = KisStockDataAdapter()
    broker = adp.broker
    paper_adapter = PaperOrderAdapter()

    seoul = timezone(timedelta(hours=9))
    now_kst = datetime.now(tz=seoul)
    expires = (now_kst + timedelta(hours=8)).isoformat()  # 당일 17:00까지

    budget = int(os.getenv("PAPER_BUDGET_PER_TICKER", "100000"))
    records = []
    n_registered = 0
    n_filled = 0
    n_blocked = 0

    for p in picks:
        tk = p.get("ticker", "")
        nm = p.get("name", tk)
        rank = picks.index(p) + 1
        px = fetch_ohlcv(broker, tk)
        if px is None or px["current"] <= 0:
            records.append({"ticker": tk, "name": nm, "rank": rank,
                            "status": "fetch_failed"})
            continue
        entry = px["current"]
        qty = budget // entry
        if qty <= 0:
            records.append({"ticker": tk, "name": nm, "rank": rank,
                            "status": "qty_zero", "entry": entry})
            continue

        intent_id = f"q_{tk}_{now_kst.strftime('%Y%m%d%H%M%S')}"
        intent = {
            "intent_id": intent_id, "bot": "quant",
            "engine": "phase1_paper_warmup", "ticker": tk, "name": nm,
            "side": "BUY", "mode": "paper",
            "score": float(p.get("score", 0.0)),
            "confidence": p.get("grade", ""),
            "created_at": now_kst.isoformat(), "expires_at": expires,
            "budget": budget, "target_qty": qty, "entry_price": entry,
        }
        try:
            register_intent(intent, bot="quant")
            n_registered += 1
        except Exception as e:
            logger.warning("[paper_trade_open] register 실패 %s: %s", tk, e)
            records.append({"ticker": tk, "name": nm, "rank": rank,
                            "status": "register_failed", "error": str(e)})
            continue

        # PaperOrderAdapter — assert_order_intent_exists 자동 호출 (P0-2 검증)
        try:
            order = paper_adapter.buy_limit(
                ticker=tk, price=entry, quantity=qty,
                mode="paper", executor_bot="quant",
            )
            if order.status == OrderStatus.FILLED:
                n_filled += 1
                records.append({
                    "ticker": tk, "name": nm, "rank": rank, "status": "filled",
                    "intent_id": intent_id, "entry": entry,
                    "filled_price": int(order.filled_price), "qty": qty,
                    "order_id": order.order_id,
                })
            else:
                records.append({"ticker": tk, "name": nm, "rank": rank,
                                "status": "order_not_filled", "intent_id": intent_id})
        except NoIntentError as e:
            n_blocked += 1
            records.append({"ticker": tk, "name": nm, "rank": rank,
                            "status": "intent_blocked", "intent_id": intent_id,
                            "error": str(e)})

    # 결과 저장
    out_dir = PROJECT_ROOT / "data" / "phase1_paper_trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today.replace('-', '')}_open.json"
    out_data = {
        "date": today, "mode": "open", "top_n": top_n,
        "n_registered": n_registered, "n_filled": n_filled, "n_blocked": n_blocked,
        "records": records, "created_at": now_kst.isoformat(),
    }
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[PAPER TRADE OPEN] {today} — registered={n_registered}, "
        f"filled={n_filled}, blocked={n_blocked} → {out_path.name}"
    )

    if send_tg:
        try:
            from src.telegram_sender import send_message
            msg = (
                f"[PHASE 1 PAPER TRADE OPEN] {today}\n"
                f"register: {n_registered}건 / fill: {n_filled}건 / blocked: {n_blocked}건\n"
                f"종목당 시드: {budget:,}원"
            )
            send_message(msg)
        except Exception as e:
            logger.warning("텔레그램 발송 실패: %s", e)

    return 0


def cmd_paper_trade_close(send_tg: bool) -> int:
    """Phase 1 row 2: BUY→SELL paper 페어 종결 + P&L 계산 (5/28).

    Trading Factory v1: register_intent(SELL) → PaperOrderAdapter.sell_limit
    - data/phase1_paper_trades/{date}_open.json 로드 (open 결과)
    - 각 filled 종목별:
      - mode='paper' + executor_bot='quant' SELL intent 등록
      - PaperOrderAdapter.sell_limit 호출 (P0-2 + L10 통과)
      - filled_price 차이 + 수수료 + 거래세 = paper_pnl
    - 결과를 data/phase1_paper_trades/{date}_close.json에 기록
    - data/paper_trader_history.json 누적 P&L 갱신 (cmd_close와 별개 — phase1 전용)

    추적표: docs/02-design/phase-1-migration-tracking.md row 2
    """
    from datetime import timedelta, timezone
    from src.use_cases.order_intents_gate import register_intent, NoIntentError
    from src.adapters.paper_order_adapter import PaperOrderAdapter
    from src.entities.trading_models import OrderStatus

    today = datetime.now().strftime("%Y-%m-%d")
    today_yyyymmdd = today.replace("-", "")
    open_path = PROJECT_ROOT / "data" / "phase1_paper_trades" / f"{today_yyyymmdd}_open.json"
    if not open_path.exists():
        print(f"[PAPER TRADE CLOSE] {today} open 파일 없음 — --paper-trade-open 먼저 실행")
        return 1

    open_data = json.loads(open_path.read_text(encoding="utf-8"))
    filled_records = [r for r in open_data.get("records", []) if r.get("status") == "filled"]
    if not filled_records:
        print(f"[PAPER TRADE CLOSE] {today} filled 종목 0건 — close 스킵")
        return 0

    adp = KisStockDataAdapter()
    broker = adp.broker
    paper_adapter = PaperOrderAdapter()

    seoul = timezone(timedelta(hours=9))
    now_kst = datetime.now(tz=seoul)
    expires = (now_kst + timedelta(hours=2)).isoformat()  # 종가 직후 2시간

    close_records = []
    n_registered = 0
    n_filled = 0
    n_blocked = 0
    total_pnl = 0
    total_buy_cost = 0
    total_sell_proceed = 0

    for rec in filled_records:
        tk = rec["ticker"]
        nm = rec.get("name", tk)
        qty = rec["qty"]
        buy_filled = rec["filled_price"]

        # 현재가 (종가) 조회
        px = fetch_ohlcv(broker, tk)
        if px is None or px["current"] <= 0:
            close_records.append({**rec, "close_status": "fetch_failed"})
            continue
        current = px["current"]

        # SELL intent 등록
        intent_id = f"q_{tk}_SELL_{now_kst.strftime('%Y%m%d%H%M%S')}"
        intent = {
            "intent_id": intent_id, "bot": "quant",
            "engine": "phase1_paper_warmup", "ticker": tk, "name": nm,
            "side": "SELL", "mode": "paper",
            "score": float(rec.get("score", 0.0)),
            "created_at": now_kst.isoformat(), "expires_at": expires,
            "target_qty": qty, "exit_price": current,
            "parent_buy_intent_id": rec.get("intent_id", ""),
        }
        try:
            register_intent(intent, bot="quant")
            n_registered += 1
        except Exception as e:
            logger.warning("[paper_trade_close] SELL register 실패 %s: %s", tk, e)
            close_records.append({**rec, "close_status": "register_failed",
                                  "close_error": str(e)})
            continue

        # PaperOrderAdapter.sell_limit (mode='paper' + executor_bot='quant')
        try:
            order = paper_adapter.sell_limit(
                ticker=tk, price=current, quantity=qty,
                mode="paper", executor_bot="quant",
            )
            if order.status == OrderStatus.FILLED:
                n_filled += 1
                sell_filled = int(order.filled_price)
                # P&L 계산 (수수료/세금은 PaperOrderAdapter 내부에서 이미 반영됐으나
                # 단순 buy/sell 차익 기록은 추적표용 — 정확한 P&L은 sell_limit message 참고)
                pnl_raw = (sell_filled - buy_filled) * qty
                # 수수료 0.015% × 2 + 거래세 0.18% (cmd_close 동일 가정 — 추적성)
                fee = int(buy_filled * qty * 0.015 / 100) + int(sell_filled * qty * 0.015 / 100)
                tax = int(sell_filled * qty * 0.18 / 100)
                pnl = pnl_raw - fee - tax
                buy_cost = buy_filled * qty + int(buy_filled * qty * 0.015 / 100)
                sell_proceed = sell_filled * qty - int(sell_filled * qty * 0.015 / 100) - tax
                pnl_pct = (pnl / buy_cost * 100) if buy_cost > 0 else 0
                total_pnl += pnl
                total_buy_cost += buy_cost
                total_sell_proceed += sell_proceed

                close_records.append({
                    **rec, "close_status": "filled",
                    "sell_intent_id": intent_id,
                    "sell_filled_price": sell_filled,
                    "sell_order_id": order.order_id,
                    "paper_pnl": pnl, "paper_pnl_pct": round(pnl_pct, 2),
                    "paper_buy_cost": buy_cost, "paper_sell_proceed": sell_proceed,
                })
            else:
                close_records.append({**rec, "close_status": "order_not_filled",
                                      "sell_intent_id": intent_id})
        except NoIntentError as e:
            n_blocked += 1
            close_records.append({**rec, "close_status": "intent_blocked",
                                  "sell_intent_id": intent_id, "close_error": str(e)})

    # 결과 저장
    out_path = PROJECT_ROOT / "data" / "phase1_paper_trades" / f"{today_yyyymmdd}_close.json"
    total_pnl_pct = (total_pnl / total_buy_cost * 100) if total_buy_cost > 0 else 0
    out_data = {
        "date": today, "mode": "close",
        "n_open_filled": len(filled_records),
        "n_sell_registered": n_registered,
        "n_sell_filled": n_filled,
        "n_sell_blocked": n_blocked,
        "total_buy_cost": total_buy_cost,
        "total_sell_proceed": total_sell_proceed,
        "total_pnl": total_pnl,
        "total_pnl_pct": round(total_pnl_pct, 2),
        "records": close_records,
        "closed_at": now_kst.isoformat(),
    }
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 누적 history 갱신 (phase1 전용)
    history_path = PROJECT_ROOT / "data" / "phase1_paper_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = {"sessions": [], "cumulative_pnl": 0}
    else:
        history = {"sessions": [], "cumulative_pnl": 0}
    # 같은 날짜 중복 제거
    history["sessions"] = [s for s in history["sessions"] if s.get("date") != today]
    history["sessions"].append({
        "date": today,
        "n_filled": n_filled, "n_blocked": n_blocked,
        "buy_cost": total_buy_cost, "sell_proceed": total_sell_proceed,
        "pnl": total_pnl, "pnl_pct": round(total_pnl_pct, 2),
    })
    history["cumulative_pnl"] = sum(s["pnl"] for s in history["sessions"])
    history["n_sessions"] = len(history["sessions"])
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[PAPER TRADE CLOSE] {today} — SELL registered={n_registered}, "
        f"filled={n_filled}, blocked={n_blocked}"
    )
    print(
        f"[PHASE 1 P&L] 당일 {total_pnl:+,}원 ({total_pnl_pct:+.2f}%) | "
        f"누적 {history['cumulative_pnl']:+,}원 ({history['n_sessions']}일)"
    )

    if send_tg:
        try:
            from src.telegram_sender import send_message
            msg_parts = [
                f"[PHASE 1 PAPER TRADE CLOSE] {today}",
                f"SELL register: {n_registered} / fill: {n_filled} / blocked: {n_blocked}",
                "",
                f"💰 당일 P&L: {total_pnl:+,}원 ({total_pnl_pct:+.2f}%)",
                f"📊 누적 P&L: {history['cumulative_pnl']:+,}원 ({history['n_sessions']}일)",
            ]
            # 종목별 TOP 3 / 손실
            sorted_pnl = sorted(
                [r for r in close_records if r.get("close_status") == "filled"],
                key=lambda x: x.get("paper_pnl", 0), reverse=True,
            )
            if sorted_pnl[:3]:
                msg_parts.append("\n🏆 TOP 3:")
                for r in sorted_pnl[:3]:
                    msg_parts.append(
                        f"  {r['name']} {r.get('paper_pnl', 0):+,}원 ({r.get('paper_pnl_pct', 0):+.2f}%)"
                    )
            losses = [r for r in sorted_pnl[-3:] if r.get("paper_pnl", 0) < 0]
            if losses:
                msg_parts.append("\n📉 손실:")
                for r in losses:
                    msg_parts.append(
                        f"  {r['name']} {r.get('paper_pnl', 0):+,}원 ({r.get('paper_pnl_pct', 0):+.2f}%)"
                    )
            send_message("\n".join(msg_parts))
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
    parser.add_argument("--paper-trade-open", action="store_true",
                        help="Phase 1: register_intent + PaperOrderAdapter.buy_limit (5/28)")
    parser.add_argument("--paper-trade-close", action="store_true",
                        help="Phase 1 row 2: register_intent(SELL) + PaperOrderAdapter.sell_limit + P&L (5/28)")
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
    if args.paper_trade_open:
        return cmd_paper_trade_open(args.top, send_tg)
    if args.paper_trade_close:
        return cmd_paper_trade_close(send_tg)
    if args.label_d1:
        if not args.date:
            print("--label-d1 에는 --date 필수")
            return 1
        return cmd_label_d1(args.date)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
