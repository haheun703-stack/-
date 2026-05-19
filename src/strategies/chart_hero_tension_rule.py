"""차트영웅 매매법 — 긴장 타입 룰 엔진 (5/19 사장님 결단).

차트영웅 원본 (옵션 A) vs 긴장 타입:
  원본: 1.5% × 4단계 누적 6.0% / 손절 X / 익절 +5%/+10% 50/50
  긴장: 1.0% × 4단계 누적 4.0% / 손절 -40% / 익절 +3%/+5%/+8% 30/30/40

긴장 타입 안전망 5가지:
  1) 1차 비중 1.0% (원본 1.5% 대비 -33%)
  2) -40% 도달 시 강제 손절 (원본은 손절 X)
  3) +3% 빠른 부분 익절 (원본은 +5%부터)
  4) D+3 +3% 미달 → 50% 청산 (원본은 D+5까지 보유)
  5) 주 -3% / 월 -5% 누적 손실 시 신규 진입 자동 중지

목적: 차트영웅 73연승 알파를 유지하면서 폭락장 안전판 강화.
백테스트 비교 (5/21 예정): 원본 vs 긴장 → WR/PF/MDD 모두 측정 후 결단.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional


class TradeStage(Enum):
    """진입 단계."""
    NONE = 0          # 미진입
    INIT = 1          # 1차 진입 (1.0%)
    ADD_10 = 2        # -10% 추매 (1.0%)
    ADD_20 = 3        # -20% 추매 (1.0%)
    ADD_30 = 4        # -30% 추매 (1.0%)
    STOPLOSS = 5      # -40% 손절
    PARTIAL_3 = 6     # +3% 30% 익절
    PARTIAL_5 = 7     # +5% 30% 익절
    FINAL_8 = 8       # +8% 나머지 익절
    D3_CUT = 9        # D+3 +3% 미달 50% 청산
    D5_FORCE = 10     # D+5 강제 청산


@dataclass
class Position:
    """1종목 포지션 상태."""
    ticker: str
    name: str
    entry_date: date         # D+1 양봉 확인 후 진입한 날
    avg_price: float         # 평균 단가
    total_qty: int           # 누적 수량
    total_cost: int          # 누적 매수 금액 (원)
    stage: TradeStage = TradeStage.INIT
    realized_pl: int = 0     # 부분 익절 실현 손익
    is_closed: bool = False
    close_reason: Optional[str] = None

    # 단계별 진입 기록
    entries: list[dict] = field(default_factory=list)


# === 비중/임계 설정 (긴장 타입) ===
INIT_WEIGHT_PCT     = 1.0       # 1차 1.0%
ADD_WEIGHT_PCT      = 1.0       # 추매 매번 1.0%
STOPLOSS_PCT        = -40.0     # 손절선 (1종목 -40%)
ADD_TRIGGERS        = [-10.0, -20.0, -30.0]   # 추매 트리거
TAKE_PROFIT_3       = 3.0       # +3% 30% 익절
TAKE_PROFIT_5       = 5.0       # +5% 30% 익절
TAKE_PROFIT_8       = 8.0       # +8% 나머지 40% 익절
D3_HOLDING_MIN_PROFIT = 3.0     # D+3까지 +3% 미달 시 50% 청산
D5_FORCE_CLOSE      = True      # D+5 종가 강제 청산
WEEKLY_LOSS_LIMIT_PCT  = -3.0   # 주 -3% 누적 손실 시 신규 중지
MONTHLY_LOSS_LIMIT_PCT = -5.0   # 월 -5% 누적 손실 시 사장님 알림 + 중지


def compute_pnl_pct(current_price: float, avg_price: float) -> float:
    """현재가 대비 평단가 손익률 (%)."""
    if avg_price <= 0:
        return 0.0
    return round((current_price - avg_price) / avg_price * 100, 2)


def days_held(entry_date: date, today: date) -> int:
    """진입 후 경과 영업일 (단순 캘린더 일수, 추후 영업일 계산 가능)."""
    return (today - entry_date).days


def decide_action(pos: Position, current_price: float, today: date) -> dict:
    """포지션 1개에 대해 오늘 취해야 할 액션 결단.

    Returns:
        {
          action: 'HOLD' | 'ADD_BUY' | 'PARTIAL_SELL' | 'STOPLOSS' | 'FORCE_CLOSE',
          stage: TradeStage,
          qty_pct: float,         # 매수/매도 비중 (%)
          reason: str,
          new_stage: TradeStage,  # 액션 후 단계
        }
    """
    if pos.is_closed:
        return {"action": "HOLD", "stage": pos.stage, "qty_pct": 0,
                "reason": "이미 청산됨", "new_stage": pos.stage}

    pnl = compute_pnl_pct(current_price, pos.avg_price)
    held = days_held(pos.entry_date, today)

    # 1) 손절 (최우선)
    if pnl <= STOPLOSS_PCT:
        return {"action": "STOPLOSS", "stage": pos.stage, "qty_pct": 100.0,
                "reason": f"손절선 도달 PnL={pnl}%", "new_stage": TradeStage.STOPLOSS}

    # 2) D+5 강제 청산
    if D5_FORCE_CLOSE and held >= 5:
        return {"action": "FORCE_CLOSE", "stage": pos.stage, "qty_pct": 100.0,
                "reason": f"D+5 강제 청산 PnL={pnl}% (메모리 D+5 -0.35% 꺾임 룰)",
                "new_stage": TradeStage.D5_FORCE}

    # 3) D+3 +3% 미달 → 50% 청산
    if held >= 3 and pnl < D3_HOLDING_MIN_PROFIT and pos.stage.value < TradeStage.D3_CUT.value:
        return {"action": "PARTIAL_SELL", "stage": pos.stage, "qty_pct": 50.0,
                "reason": f"D+3 +3% 미달 (PnL={pnl}%) → 50% 청산", "new_stage": TradeStage.D3_CUT}

    # 4) 익절 (단계적)
    if pnl >= TAKE_PROFIT_8 and pos.stage.value < TradeStage.FINAL_8.value:
        return {"action": "PARTIAL_SELL", "stage": pos.stage, "qty_pct": 40.0,
                "reason": f"+8% 도달, 나머지 40% 익절", "new_stage": TradeStage.FINAL_8}
    if pnl >= TAKE_PROFIT_5 and pos.stage.value < TradeStage.PARTIAL_5.value:
        return {"action": "PARTIAL_SELL", "stage": pos.stage, "qty_pct": 30.0,
                "reason": f"+5% 도달, 30% 익절", "new_stage": TradeStage.PARTIAL_5}
    if pnl >= TAKE_PROFIT_3 and pos.stage.value < TradeStage.PARTIAL_3.value:
        return {"action": "PARTIAL_SELL", "stage": pos.stage, "qty_pct": 30.0,
                "reason": f"+3% 도달, 30% 부분 익절 (빠른 안전판)", "new_stage": TradeStage.PARTIAL_3}

    # 5) 추매 트리거
    if pnl <= ADD_TRIGGERS[2] and pos.stage.value < TradeStage.ADD_30.value:
        return {"action": "ADD_BUY", "stage": pos.stage, "qty_pct": ADD_WEIGHT_PCT,
                "reason": f"-30% 도달, 4차 추매 1.0%", "new_stage": TradeStage.ADD_30}
    if pnl <= ADD_TRIGGERS[1] and pos.stage.value < TradeStage.ADD_20.value:
        return {"action": "ADD_BUY", "stage": pos.stage, "qty_pct": ADD_WEIGHT_PCT,
                "reason": f"-20% 도달, 3차 추매 1.0%", "new_stage": TradeStage.ADD_20}
    if pnl <= ADD_TRIGGERS[0] and pos.stage.value < TradeStage.ADD_10.value:
        return {"action": "ADD_BUY", "stage": pos.stage, "qty_pct": ADD_WEIGHT_PCT,
                "reason": f"-10% 도달, 2차 추매 1.0%", "new_stage": TradeStage.ADD_10}

    return {"action": "HOLD", "stage": pos.stage, "qty_pct": 0,
            "reason": f"HOLD PnL={pnl}% Day+{held}", "new_stage": pos.stage}


def check_entry_gate(weekly_loss_pct: float, monthly_loss_pct: float) -> dict:
    """신규 진입 게이트 (주/월 누적 손실).

    Returns:
        { allow_new_entry: bool, reason: str }
    """
    if monthly_loss_pct <= MONTHLY_LOSS_LIMIT_PCT:
        return {"allow_new_entry": False,
                "reason": f"⚠️ 월 누적 손실 {monthly_loss_pct}% ≤ {MONTHLY_LOSS_LIMIT_PCT}% — 시스템 점검 + 사장님 알림"}
    if weekly_loss_pct <= WEEKLY_LOSS_LIMIT_PCT:
        return {"allow_new_entry": False,
                "reason": f"⚠️ 주 누적 손실 {weekly_loss_pct}% ≤ {WEEKLY_LOSS_LIMIT_PCT}% — 신규 진입 일시 중지"}
    return {"allow_new_entry": True,
            "reason": f"OK (주 {weekly_loss_pct}%, 월 {monthly_loss_pct}%)"}


if __name__ == "__main__":
    print("=== 긴장 타입 룰 엔진 — 시나리오 시뮬레이션 ===\n")

    # 시나리오 1: D+1 진입 → 다음날 -10% 갭다운 → 추매
    pos = Position(
        ticker="123456", name="테스트종목",
        entry_date=date(2026, 5, 22),
        avg_price=10000, total_qty=4, total_cost=40000,
        stage=TradeStage.INIT,
    )
    today = date(2026, 5, 23)
    for scenario, price in [
        ("진입 다음날 -3%", 9700),
        ("그 다음날 -10%", 9000),
        ("그 다음날 -20%", 8000),
        ("폭락 -35%",      6500),
        ("폭락 -42% (손절)", 5800),
    ]:
        r = decide_action(pos, price, today)
        print(f"  [{scenario:20}] price={price:>6} | action={r['action']:14} | {r['reason']}")
        today = today + timedelta(days=1)

    print()
    print("=== 시나리오 2: 진입 후 빠른 +5% 익절 ===")
    pos2 = Position(
        ticker="654321", name="테스트2",
        entry_date=date(2026, 5, 22), avg_price=10000,
        total_qty=4, total_cost=40000, stage=TradeStage.INIT,
    )
    for scenario, price, today_offset in [
        ("D+1 +2%",   10200, 1),
        ("D+1 +3% 익절", 10300, 1),
        ("D+2 +5% 익절", 10500, 2),
        ("D+3 +8% 최종 익절", 10800, 3),
    ]:
        r = decide_action(pos2, price, date(2026, 5, 22) + timedelta(days=today_offset))
        # 단계 진행 시뮬레이트
        if r["action"] == "PARTIAL_SELL":
            pos2.stage = r["new_stage"]
        print(f"  [{scenario:18}] price={price:>6} | action={r['action']:14} | {r['reason']}")

    print()
    print("=== 시나리오 3: 주/월 손실 게이트 ===")
    for w, m in [(-1.5, -2.0), (-3.5, -2.0), (-3.0, -5.5)]:
        r = check_entry_gate(w, m)
        flag = "✓ 진입 OK" if r["allow_new_entry"] else "🛑 진입 중지"
        print(f"  주={w:>5}%, 월={m:>5}% → {flag}  {r['reason']}")
