"""차트영웅 매매 실행기 — 5/22 paper mirror + 5/27 실전.

구성:
  surge_d1_picker (5-Gate 선정)
    ↓ 14:30~14:55
  d1_confirm (D+1 양봉 체크)
    ↓ 14:55
  ChartHeroExecutor.execute_d1_entry()  ⭐ 본 모듈
    ↓ 15:00 종가 체결
  포지션 등록 → 매일 monitor_positions() 추매/익절/손절

격리 원칙 (project_chart_hero_independence.md):
  - 단타봇 advisory 일체 참조 X
  - 우리 자체 brain.py PANIC 모드 참조 X
  - VIX/regime BEARISH 차단 X
  - chart_hero_advisory + four_signal_gate만 본다
"""

import datetime as dt
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.adapters.kis_order_adapter import KisOrderAdapter
from src.strategies.chart_hero_tension_rule import (
    Position, TradeStage, decide_action, check_entry_gate,
    INIT_WEIGHT_PCT, ADD_WEIGHT_PCT,
)
from src.strategies.d1_confirm import check_d1_candle


# 포지션 상태 영구 저장 경로
POSITIONS_FILE = "data/chart_hero_positions.json"
PNL_LOG_FILE   = "data/chart_hero_pnl_log.csv"


class ChartHeroExecutor:
    """차트영웅 매매 실행기 (paper/real 모드).

    paper=True → KIS 주문 X, 시뮬레이션만 (5/22~5/26 paper mirror)
    paper=False → KIS 실제 주문 (5/27 사장님 GO 후)
    """

    def __init__(self, paper: bool = True, total_capital: float = 25_000_000):
        self.paper = paper
        self.total_capital = total_capital  # 5/22 잔고 2,500만 시뮬
        if not paper:
            self.order = KisOrderAdapter()
        else:
            self.order = None
        self.positions: dict[str, dict] = self._load_positions()
        self.weekly_loss_pct  = 0.0  # 실제 운영 시 PnL 로그에서 계산
        self.monthly_loss_pct = 0.0

    def _load_positions(self) -> dict:
        p = Path(POSITIONS_FILE)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_positions(self):
        p = Path(POSITIONS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        # dataclass 직렬화 시 TradeStage Enum 처리
        out = {}
        for k, v in self.positions.items():
            out[k] = {**v, "stage": v["stage"] if isinstance(v.get("stage"), str) else v["stage"].name}
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def _compute_qty(self, capital_pct: float, price: int) -> int:
        amount = self.total_capital * (capital_pct / 100)
        return max(int(amount // price), 1)

    # 고가주 필터: 1주 가격이 잔고의 max_single_share_pct% 초과면 제외
    MAX_SINGLE_SHARE_PCT = 3.0   # 1주 가격 > 잔고 3% = 제외

    def is_affordable(self, price: int) -> tuple[bool, str]:
        """1주 가격이 잔고 대비 적정한지 판정.

        2,500만 잔고 기준: 1주 75만 초과 = 제외.
        한화에어로 128만, SK하이닉스 174만 등 제외.
        """
        share_pct = price / self.total_capital * 100
        if share_pct > self.MAX_SINGLE_SHARE_PCT:
            return False, f"1주 비중 {share_pct:.1f}% > {self.MAX_SINGLE_SHARE_PCT}% (고가주 제외)"
        return True, f"1주 비중 {share_pct:.2f}% OK"

    def execute_d1_entry(self, picks_with_confirm: list[dict]) -> list[dict]:
        """D+1 양봉 확인된 종목 1차 진입 (긴장 타입 1.0%).

        Args:
            picks_with_confirm: d1_confirm.monitor_picks() 출력
        Returns:
            [{ticker, action, qty, price, result, ...}, ...]
        """
        results = []
        # 게이트 체크 (주/월 누적 손실)
        gate = check_entry_gate(self.weekly_loss_pct, self.monthly_loss_pct)
        if not gate["allow_new_entry"]:
            return [{"action": "BLOCKED", "reason": gate["reason"]}]

        for p in picks_with_confirm:
            if not p.get("will_enter"):
                continue
            ticker = p["ticker"]
            if ticker in self.positions:
                results.append({"ticker": ticker, "action": "ALREADY_HELD"})
                continue

            price = p.get("entry_price_estimate") or p.get("current_price")
            # 고가주 필터 (1주 비중 > 3% 제외)
            affordable, msg = self.is_affordable(price)
            if not affordable:
                results.append({"ticker": ticker, "action": "SKIP_TOO_EXPENSIVE",
                                "price": price, "reason": msg})
                continue
            qty = self._compute_qty(INIT_WEIGHT_PCT, price)

            if self.paper:
                # 시뮬레이션: 포지션만 등록
                self.positions[ticker] = {
                    "ticker": ticker,
                    "name": p.get("name", ""),
                    "entry_date": dt.date.today().isoformat(),
                    "avg_price": price,
                    "total_qty": qty,
                    "total_cost": price * qty,
                    "stage": TradeStage.INIT.name,
                    "realized_pl": 0,
                    "is_closed": False,
                    "entries": [{"stage": "INIT", "price": price, "qty": qty,
                                 "date": dt.date.today().isoformat()}],
                }
                results.append({"ticker": ticker, "action": "PAPER_BUY",
                                "qty": qty, "price": price,
                                "amount": price * qty})
            else:
                # 실제 KIS 주문
                try:
                    order = self.order.buy_limit(ticker, price, qty)
                    self.positions[ticker] = {
                        "ticker": ticker, "name": p.get("name", ""),
                        "entry_date": dt.date.today().isoformat(),
                        "avg_price": price, "total_qty": qty,
                        "total_cost": price * qty,
                        "stage": TradeStage.INIT.name,
                        "realized_pl": 0, "is_closed": False,
                        "order_id": getattr(order, "order_id", None),
                        "entries": [{"stage": "INIT", "price": price, "qty": qty,
                                     "date": dt.date.today().isoformat()}],
                    }
                    results.append({"ticker": ticker, "action": "REAL_BUY",
                                    "qty": qty, "price": price, "order_id": order})
                except Exception as e:
                    results.append({"ticker": ticker, "action": "ERROR", "error": str(e)})

        self._save_positions()
        return results

    def _to_position_obj(self, d: dict) -> Position:
        return Position(
            ticker=d["ticker"], name=d["name"],
            entry_date=dt.date.fromisoformat(d["entry_date"]),
            avg_price=d["avg_price"], total_qty=d["total_qty"],
            total_cost=d["total_cost"],
            stage=TradeStage[d["stage"]] if isinstance(d["stage"], str) else d["stage"],
            realized_pl=d.get("realized_pl", 0),
            is_closed=d.get("is_closed", False),
        )

    def monitor_positions(self) -> list[dict]:
        """기존 포지션 일일 모니터링 → 긴장 룰에 따라 액션 실행."""
        from src.adapters.kis_nxt_kit import get_nx_price
        today = dt.date.today()
        results = []

        for ticker, d in list(self.positions.items()):
            if d.get("is_closed"):
                continue
            # 현재가 조회
            p = get_nx_price(ticker)
            if not p:
                results.append({"ticker": ticker, "action": "PRICE_FAIL"})
                continue
            current_price = p.get("price") or p.get("current") or 0

            pos = self._to_position_obj(d)
            action = decide_action(pos, current_price, today)

            if action["action"] == "HOLD":
                results.append({"ticker": ticker, "action": "HOLD",
                                "current": current_price, "reason": action["reason"]})
                continue

            # 액션 실행 (paper or real)
            if action["action"] == "ADD_BUY":
                add_qty = self._compute_qty(action["qty_pct"], current_price)
                self._execute_add_buy(d, current_price, add_qty)
                results.append({"ticker": ticker, "action": "ADD_BUY",
                                "qty": add_qty, "price": current_price,
                                "reason": action["reason"]})
            elif action["action"] in ("PARTIAL_SELL", "STOPLOSS", "FORCE_CLOSE"):
                sell_qty = int(d["total_qty"] * action["qty_pct"] / 100)
                self._execute_sell(d, current_price, sell_qty, action)
                results.append({"ticker": ticker, "action": action["action"],
                                "qty": sell_qty, "price": current_price,
                                "reason": action["reason"]})

            # 단계 업데이트
            d["stage"] = action["new_stage"].name

        self._save_positions()
        return results

    def _execute_add_buy(self, d: dict, price: int, qty: int):
        """추매 실행 (paper or real)."""
        if not self.paper:
            try:
                self.order.buy_limit(d["ticker"], price, qty)
            except Exception:
                return
        # 평단가/누적수량 업데이트
        new_total_qty = d["total_qty"] + qty
        new_total_cost = d["total_cost"] + price * qty
        d["avg_price"] = round(new_total_cost / new_total_qty, 2)
        d["total_qty"] = new_total_qty
        d["total_cost"] = new_total_cost
        d.setdefault("entries", []).append({
            "stage": "ADD", "price": price, "qty": qty,
            "date": dt.date.today().isoformat(),
        })

    def _execute_sell(self, d: dict, price: int, qty: int, action: dict):
        """매도 실행 (paper or real)."""
        if not self.paper:
            try:
                self.order.sell_limit(d["ticker"], price, qty)
            except Exception:
                return
        realized = (price - d["avg_price"]) * qty
        d["realized_pl"] = d.get("realized_pl", 0) + int(realized)
        d["total_qty"] -= qty
        if d["total_qty"] <= 0 or action["action"] in ("STOPLOSS", "FORCE_CLOSE"):
            d["is_closed"] = True
            d["close_reason"] = action["reason"]

    def get_summary(self) -> dict:
        """현재 포지션 요약."""
        active = [p for p in self.positions.values() if not p.get("is_closed")]
        closed = [p for p in self.positions.values() if p.get("is_closed")]
        total_pl = sum(p.get("realized_pl", 0) for p in self.positions.values())
        return {
            "active_count": len(active),
            "closed_count": len(closed),
            "total_realized_pl": total_pl,
            "total_capital": self.total_capital,
            "pnl_pct": round(total_pl / self.total_capital * 100, 2),
        }


if __name__ == "__main__":
    print("=== ChartHeroExecutor 시뮬레이션 ===\n")
    # paper 모드 + 가상 picks
    ex = ChartHeroExecutor(paper=True, total_capital=25_000_000)

    fake_picks = [
        {"ticker": "005930", "name": "삼성전자",     "entry_price_estimate": 275500,
         "will_enter": True},
        {"ticker": "012450", "name": "한화에어로",   "entry_price_estimate": 1286000,
         "will_enter": True},
        {"ticker": "000660", "name": "SK하이닉스",   "entry_price_estimate": 1745000,
         "will_enter": False},  # 음봉이라 skip
    ]
    results = ex.execute_d1_entry(fake_picks)
    print("진입 결과:")
    for r in results:
        print(f"  {r}")

    print(f"\n현재 포지션 요약:")
    summary = ex.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
