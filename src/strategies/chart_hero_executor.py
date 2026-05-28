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

import csv
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


# 포지션 상태 영구 저장 경로 (#10: cron cwd 누락 사고 방지 — 절대경로화)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POSITIONS_FILE = str(PROJECT_ROOT / "data/chart_hero_positions.json")
PNL_LOG_FILE   = str(PROJECT_ROOT / "data/chart_hero_pnl_log.csv")


class ChartHeroExecutor:
    """차트영웅 매매 실행기 (paper/real 모드).

    paper=True → PaperOrderAdapter (시뮬, 5/28 Phase 1 row 4: order_intents_gate L10 강제)
    paper=False → KisOrderAdapter (실제 주문, 단 quant + live 조합은 OrderIntentError 차단됨)

    Trading Factory v1 (5/28):
      - paper 모드: PaperOrderAdapter.buy_limit(mode='paper', executor_bot='quant')
        → row 3 picker가 D0 17:30에 등록한 intent와 페어 검증
      - real 모드: KisOrderAdapter.buy_limit(mode='live', executor_bot='quant')
        → quant + live 조합은 register 단계에서 거부됨 (Note 1) → 자동 차단
        → 실거래는 Phase 2 (6/9+) day bot으로 이전 후만 가능
    """

    def __init__(self, paper: bool = True, total_capital: float = 25_000_000,
                 max_qty_per_ticker: int | None = None,
                 kill_switch_active: bool = False):
        self.paper = paper
        self.total_capital = total_capital
        self.max_qty_per_ticker = max_qty_per_ticker
        self.kill_switch_active = kill_switch_active
        self.hold_list = self._load_hold_list()
        # row 4 (5/28): paper도 PaperOrderAdapter 사용 (intent 강제)
        if paper:
            from src.adapters.paper_order_adapter import PaperOrderAdapter
            self.order = PaperOrderAdapter()
        else:
            self.order = KisOrderAdapter()
        self.positions: dict[str, dict] = self._load_positions()
        # C2: PnL 로그에서 누적 손실률 계산 (긴장 안전망)
        self.weekly_loss_pct  = self._compute_loss_pct(days=7)
        self.monthly_loss_pct = self._compute_loss_pct(days=30)

    # ─────────────────────────────────────────────────
    # ★ P3 (5/20 추가): 보호 종목 (Hold List) 가드
    # 단타봇 200만원 손실 사건: "7월 실적 발표 기대 종목 자동 매도"
    # → 퀀트봇에서 같은 사고 재발 방지
    # ─────────────────────────────────────────────────
    def _load_hold_list(self) -> dict:
        """config/hold_list.json 로드. 형식: {ticker: {name, reason, until}}

        Minor 보강 (5/20 검수): `_` prefix 키 필터링 (메타/예시 오염 방지).
        예: `_meta`, `_example_엘앤에프` 등은 실제 보호 항목이 아니므로 제외.
        """
        # 우선 config/, 폴백으로 data/ (마이그레이션 호환)
        for path_str in ("config/hold_list.json", "data/hold_list.json"):
            p = Path(path_str)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    # `_` prefix 키 제외 (메타/예시 노이즈 차단)
                    return {k: v for k, v in data.items() if not k.startswith("_")}
                except Exception:
                    continue
        return {}

    def _is_protected(self, ticker: str) -> tuple[bool, str]:
        """ticker가 보호 종목인지 + 사유.

        Returns:
            (보호중 여부, 사유 메시지)
        """
        item = self.hold_list.get(ticker)
        if not item:
            return False, ""
        until = item.get("until", "")
        if until:
            try:
                until_date = dt.date.fromisoformat(until)
                if dt.date.today() > until_date:
                    return False, f"보호 만료 ({until})"
            except Exception:
                pass
        return True, f"보호 종목: {item.get('name', '')} — {item.get('reason', '')} (until {until})"

    def _load_positions(self) -> dict:
        p = Path(POSITIONS_FILE)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ─────────────────────────────────────────────────
    # C2: PnL 누적 추적 (긴장 안전망 — 주-3%/월-5% 게이트)
    # ─────────────────────────────────────────────────
    def _load_pnl_log(self) -> list[dict]:
        """PnL CSV 로그 읽기. 없으면 빈 리스트."""
        p = Path(PNL_LOG_FILE)
        if not p.exists():
            return []
        try:
            with p.open(encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def _append_pnl_log(self, ticker: str, name: str, qty: int, price: int,
                       avg_price: float, realized_pl: int, reason: str):
        """매도 시 PnL 로그 1줄 append (#2: atomic 헤더 — race condition 방지)."""
        p = Path(PNL_LOG_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 헤더 atomic write (1회만 — 동시 실행 중복 헤더 방지)
        if not p.exists():
            with p.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["date", "ticker", "name", "qty", "price",
                                       "avg_price", "realized_pl", "reason"])
        with p.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([dt.date.today().isoformat(), ticker, name, qty,
                                   price, avg_price, realized_pl, reason])

    def _compute_loss_pct(self, days: int) -> float:
        """최근 N일간 실현 손익률 (%). 음수면 손실 — 게이트 비교용."""
        cutoff = dt.date.today() - dt.timedelta(days=days)
        total = 0
        for r in self._load_pnl_log():
            try:
                if dt.date.fromisoformat(r["date"]) >= cutoff:
                    total += int(float(r["realized_pl"]))
            except Exception:
                continue
        if self.total_capital <= 0:
            return 0.0
        return round(total / self.total_capital * 100, 2)

    def _refresh_loss_pct(self):
        """다회 실행 대비 매번 갱신 (entry/monitor 진입 직전 호출)."""
        self.weekly_loss_pct  = self._compute_loss_pct(days=7)
        self.monthly_loss_pct = self._compute_loss_pct(days=30)

    def _save_positions(self):
        p = Path(POSITIONS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        # dataclass 직렬화 시 TradeStage Enum 처리
        out = {}
        for k, v in self.positions.items():
            out[k] = {**v, "stage": v["stage"] if isinstance(v.get("stage"), str) else v["stage"].name}
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def _compute_qty(self, capital_pct: float, price: int) -> int:
        # C1: price=0/None/음수 가드 (거래정지/시간외/응답 이상)
        if price is None or price <= 0:
            return 0
        amount = self.total_capital * (capital_pct / 100)
        qty = max(int(amount // price), 1)
        # 5/20 추가: 1주차 워밍업 max_qty 클램프 (paper/real 모두 적용 — 학습 일관성)
        if self.max_qty_per_ticker is not None:
            qty = min(qty, self.max_qty_per_ticker)
        return qty

    # 고가주 필터: 1주 가격이 잔고의 max_single_share_pct% 초과면 제외
    MAX_SINGLE_SHARE_PCT = 3.0   # 1주 가격 > 잔고 3% = 제외

    def is_affordable(self, price: int) -> tuple[bool, str]:
        """1주 가격이 잔고 대비 적정한지 판정.

        2,500만 잔고 기준: 1주 75만 초과 = 제외.
        한화에어로 128만, SK하이닉스 174만 등 제외.
        """
        # C1: price 검증 (None/0/음수 = 거래정지·시간외·응답 이상)
        if price is None or price <= 0:
            return False, f"price 무효({price}) — 거래정지/시간외 의심"
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
        # ★ P0 (5/20 추가): KILL_SWITCH 이중 차단 (close_cycle에서 1차 차단했지만 직접 호출 대비)
        if self.kill_switch_active:
            return [{"action": "BLOCKED", "reason": "KILL_SWITCH 활성 — 시장 패닉 매수 차단"}]
        # C2: 게이트 직전 PnL 갱신 (장중 다회 실행 대비)
        self._refresh_loss_pct()
        gate = check_entry_gate(self.weekly_loss_pct, self.monthly_loss_pct)
        if not gate["allow_new_entry"]:
            return [{"action": "BLOCKED", "reason": gate["reason"],
                     "weekly_loss_pct": self.weekly_loss_pct,
                     "monthly_loss_pct": self.monthly_loss_pct}]

        for p in picks_with_confirm:
            if not p.get("will_enter"):
                continue
            ticker = p["ticker"]
            if ticker in self.positions:
                results.append({"ticker": ticker, "action": "ALREADY_HELD"})
                continue

            price = p.get("entry_price_estimate") or p.get("current_price")
            # C1: price 무효 시 진입 차단 (거래정지/시간외/응답 이상)
            if price is None or price <= 0:
                results.append({"ticker": ticker, "action": "SKIP_INVALID_PRICE",
                                "price": price, "reason": f"price={price} 무효"})
                continue
            # 고가주 필터 (1주 비중 > 3% 제외)
            affordable, msg = self.is_affordable(price)
            if not affordable:
                results.append({"ticker": ticker, "action": "SKIP_TOO_EXPENSIVE",
                                "price": price, "reason": msg})
                continue
            qty = self._compute_qty(INIT_WEIGHT_PCT, price)
            # C1: qty=0이면 진입 불가
            if qty <= 0:
                results.append({"ticker": ticker, "action": "SKIP_INVALID_QTY",
                                "price": price, "reason": f"qty={qty} 산출 불가"})
                continue

            if self.paper:
                # row 4 (5/28): PaperOrderAdapter.buy_limit + row 3 picker intent 페어 검증
                try:
                    order = self.order.buy_limit(
                        ticker, price, qty,
                        mode="paper", executor_bot="quant",
                    )
                    filled_price = int(getattr(order, "filled_price", price) or price)
                    self.positions[ticker] = {
                        "ticker": ticker,
                        "name": p.get("name", ""),
                        "entry_date": dt.date.today().isoformat(),
                        "avg_price": filled_price,
                        "total_qty": qty,
                        "total_cost": filled_price * qty,
                        "stage": TradeStage.INIT.name,
                        "realized_pl": 0,
                        "is_closed": False,
                        "order_id": getattr(order, "order_id", None),
                        "entries": [{"stage": "INIT", "price": filled_price, "qty": qty,
                                     "date": dt.date.today().isoformat()}],
                    }
                    results.append({"ticker": ticker, "action": "PAPER_BUY",
                                    "qty": qty, "price": filled_price,
                                    "amount": filled_price * qty,
                                    "order_id": getattr(order, "order_id", None)})
                except Exception as e:
                    # NoIntentError = picker가 D0 17:30 intent 등록 안 했음
                    results.append({"ticker": ticker, "action": "BLOCKED_NO_INTENT",
                                    "price": price, "reason": str(e)})
            else:
                # 실제 KIS 주문 — quant + live 조합은 register 거부됨 (Note 1) → 자동 차단
                try:
                    order = self.order.buy_limit(
                        ticker, price, qty,
                        mode="live", executor_bot="quant",
                    )
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
                    # quant + live 조합 차단 (Trading Factory v1 정합)
                    results.append({"ticker": ticker, "action": "BLOCKED_QUANT_LIVE",
                                    "reason": str(e)})

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
        # #4: 추매(ADD_BUY)도 신규 자금 투입 → 게이트 적용
        self._refresh_loss_pct()
        gate = check_entry_gate(self.weekly_loss_pct, self.monthly_loss_pct)

        for ticker, d in list(self.positions.items()):
            if d.get("is_closed"):
                continue
            # 현재가 조회
            p = get_nx_price(ticker)
            if not p:
                results.append({"ticker": ticker, "action": "PRICE_FAIL"})
                continue
            # C1: price=0 폴백 제거 — 거래정지/시간외 시 가짜 -100% STOPLOSS 방지
            current_price = p.get("price") or p.get("current")
            if not current_price or current_price <= 0:
                results.append({"ticker": ticker, "action": "PRICE_FAIL",
                                "reason": f"current_price={current_price} 무효 (거래정지 의심)"})
                continue

            pos = self._to_position_obj(d)
            action = decide_action(pos, current_price, today)

            if action["action"] == "HOLD":
                results.append({"ticker": ticker, "action": "HOLD",
                                "current": current_price, "reason": action["reason"]})
                continue

            # 액션 실행 (paper or real)
            if action["action"] == "ADD_BUY":
                # ★ P0 (5/20): KILL_SWITCH 활성 시 추매 차단 (시장 패닉 매수 금지)
                if self.kill_switch_active:
                    results.append({"ticker": ticker, "action": "ADD_BUY_BLOCKED",
                                    "reason": "KILL_SWITCH 활성 — 시장 패닉 매수 차단"})
                    continue
                # #4: 추매도 신규 자금 투입 → 주/월 게이트 적용 (매도는 항상 허용)
                if not gate["allow_new_entry"]:
                    results.append({"ticker": ticker, "action": "ADD_BUY_BLOCKED",
                                    "reason": f"긴장 안전망: {gate['reason']}"})
                    continue
                add_qty = self._compute_qty(action["qty_pct"], current_price)
                if add_qty <= 0:
                    results.append({"ticker": ticker, "action": "ADD_BUY_SKIP",
                                    "reason": f"qty={add_qty} 산출 불가"})
                    continue
                self._execute_add_buy(d, current_price, add_qty)
                results.append({"ticker": ticker, "action": "ADD_BUY",
                                "qty": add_qty, "price": current_price,
                                "reason": action["reason"]})
            elif action["action"] in ("PARTIAL_SELL", "STOPLOSS", "FORCE_CLOSE"):
                # ★ P3 (5/20): 보호 종목(Hold List) 매도 차단 — 단타봇 200만원 사건 재발 방지
                protected, protect_msg = self._is_protected(ticker)
                if protected:
                    results.append({"ticker": ticker, "action": "SELL_BLOCKED_HOLD_LIST",
                                    "reason": protect_msg, "original_action": action["action"]})
                    # 텔레그램 알림 (실제 발송은 외부에서)
                    print(f"   🛡️ {ticker} 매도 차단 (보호 종목): {protect_msg}")
                    continue
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
        """추매 실행 (paper or real).

        row 4 (5/28): paper도 PaperOrderAdapter.buy_limit + intent 페어 강제.
        추매 intent는 D+1 14:55 (초기) 또는 추매 시점 별도 등록 필요.
        intent 미등록 시 NoIntentError → 추매 실패 (평단/수량 미변경).
        """
        try:
            self.order.buy_limit(
                d["ticker"], price, qty,
                mode="paper" if self.paper else "live",
                executor_bot="quant",
            )
        except Exception:
            return  # 추매 차단 — 평단/수량 변경 X
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
        """매도 실행 (paper or real).

        row 4 (5/28): paper도 PaperOrderAdapter.sell_limit + SELL intent 페어 강제.
        매도 intent는 매도 시점 별도 등록 필요 (close_cycle 외부 또는 별도 selector).
        intent 미등록 시 NoIntentError → 매도 실패.
        """
        try:
            self.order.sell_limit(
                d["ticker"], price, qty,
                mode="paper" if self.paper else "live",
                executor_bot="quant",
            )
        except Exception:
            return  # 매도 차단
        realized = (price - d["avg_price"]) * qty
        d["realized_pl"] = d.get("realized_pl", 0) + int(realized)
        d["total_qty"] -= qty
        # C2: PnL 로그 CSV append (주/월 누적 손실 추적용)
        self._append_pnl_log(
            ticker=d["ticker"], name=d.get("name", ""),
            qty=qty, price=price, avg_price=d["avg_price"],
            realized_pl=int(realized), reason=action.get("reason", ""),
        )
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
