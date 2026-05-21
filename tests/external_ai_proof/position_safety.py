"""position_safety.py — 외부 AI 제안 모듈 (2026-05-21 검증용)

배경 (퐝가님 5/21 19:40):
  다른 AI가 8개 유닛 테스트 패턴 제안. 우리 시스템 기존 모듈
  (owner_rule_monitor, chart_hero_executor)에 유사 기능 있으나,
  유닛 테스트 패턴 학습 목적으로 별도 구현.

함수 명세 (test_position_safety.py 추론):
  sync_positions(self):
    - self.kis.fetch_balance() 호출
    - None 리턴 시 기존 self._positions 보존 (API 장애 안전)
    - KIS 잔고 ↔ self._positions 양방향 동기화
    - KIS에만 있음 → 메모리 추가 + 기본 SL (-3%) 자동 세팅
    - 메모리에만 있음 → 제거 (매도 완료)
    - 수량 불일치 → KIS 기준 덮어쓰기
    - 마지막에 self._save_positions() 호출

  enforce_sl(self, default_sl_pct=0.03, default_tp_pct=0.05):
    - self._positions 순회
    - SL=None인 종목에 기본 SL 자동 세팅 (buy_price * (1-default_sl_pct))
    - 이미 SL 있는 종목은 건드리지 않음

  hard_kill_check(self, kill_pct=0.05, dry_run=False):
    - self.kis.fetch_balance() → 보유 종목
    - 각 종목 self.kis.fetch_price() → 현재가
    - 손실률 ≤ -kill_pct면 매도 트리거
    - dry_run=True면 sell_market 호출 X, 로깅만
    - dry_run=False면 self.kis.sell_market(code, qty) 호출
    - killed 리스트 반환

추후 통합:
  5/23~25 주말 PDCA로 owner_rule_monitor.py / chart_hero_executor.py에 통합 검토.
"""

from __future__ import annotations


# ─────────────────────────────────────────────
# 1. sync_positions — KIS ↔ 메모리 동기화
# ─────────────────────────────────────────────

def sync_positions(self) -> None:
    """KIS fetch_balance ↔ self._positions 양방향 동기화.

    안전 원칙:
      1. KIS API 실패(None) 시 기존 메모리 보존 (절대 데이터 X)
      2. KIS에만 있는 종목 → 메모리 추가 + 기본 SL 자동 세팅
      3. 메모리에만 있는 종목 → 제거 (매도 완료 추정)
      4. 수량 불일치 → KIS 기준 덮어쓰기 (KIS가 진실)
    """
    try:
        kis_balance = self.kis.fetch_balance()
    except Exception:
        kis_balance = None

    # API 실패 시 보존
    if kis_balance is None:
        return

    # KIS 잔고 dict 변환
    kis_dict = {item["code"]: item for item in kis_balance}

    # 1. 메모리 → KIS 비교 (제거 또는 수량 조정)
    for code in list(self._positions.keys()):
        if code not in kis_dict:
            # 메모리에는 있는데 KIS에는 없음 → 매도 완료 추정 → 제거
            del self._positions[code]
        else:
            # 수량 불일치 시 KIS 기준 덮어쓰기
            kis_item = kis_dict[code]
            if self._positions[code].get("qty") != kis_item["qty"]:
                self._positions[code]["qty"] = kis_item["qty"]

    # 2. KIS → 메모리 비교 (자동 등록)
    for code, kis_item in kis_dict.items():
        if code not in self._positions:
            # KIS에만 있음 → 메모리 자동 등록
            buy_price = kis_item["buy_price"]
            default_sl = round(buy_price * 0.97)  # -3%
            default_tp = round(buy_price * 1.05)  # +5%
            self._positions[code] = {
                "name": kis_item.get("name", code),
                "qty": kis_item["qty"],
                "buy_price": buy_price,
                "stop_loss": default_sl,
                "take_profit": default_tp,
            }

    # 3. 저장
    self._save_positions()


# ─────────────────────────────────────────────
# 2. enforce_sl — SL=None 자동 세팅
# ─────────────────────────────────────────────

def enforce_sl(self, default_sl_pct: float = 0.03, default_tp_pct: float = 0.05) -> None:
    """self._positions 순회. SL=None인 종목에 기본 SL 자동 세팅.

    이미 SL 있는 종목은 건드리지 않음 (덮어쓰기 안 함).
    """
    for code, pos in self._positions.items():
        if pos.get("stop_loss") is None:
            buy_price = pos["buy_price"]
            pos["stop_loss"] = round(buy_price * (1 - default_sl_pct))
        if pos.get("take_profit") is None:
            buy_price = pos["buy_price"]
            pos["take_profit"] = round(buy_price * (1 + default_tp_pct))


# ─────────────────────────────────────────────
# 3. hard_kill_check — 최후 방어선
# ─────────────────────────────────────────────

def hard_kill_check(self, kill_pct: float = 0.05, dry_run: bool = False) -> list:
    """KIS 잔고 → 각 종목 현재가 → 손실률 ≤ -kill_pct면 시장가 매도.

    Args:
        kill_pct: 킬라인 (0.05 = -5% 초과 시 매도)
        dry_run: True면 sell_market 호출 X, 로깅만

    Returns:
        killed: [{code, name, qty, buy_price, current_price, loss_pct, action}, ...]
            action ∈ {"KILLED", "dry_run"}
    """
    killed = []
    try:
        kis_balance = self.kis.fetch_balance()
    except Exception:
        return killed

    if not kis_balance:
        return killed

    for item in kis_balance:
        code = item["code"]
        qty = item["qty"]
        buy_price = item["buy_price"]

        try:
            current_price = self.kis.fetch_price(code)
        except Exception:
            continue

        if current_price is None or current_price <= 0:
            continue

        loss_pct = (current_price - buy_price) / buy_price

        if loss_pct <= -kill_pct:
            entry = {
                "code": code,
                "name": item.get("name", code),
                "qty": qty,
                "buy_price": buy_price,
                "current_price": current_price,
                "loss_pct": round(loss_pct * 100, 2),
            }

            if dry_run:
                entry["action"] = "dry_run"
            else:
                try:
                    self.kis.sell_market(code, qty)
                    entry["action"] = "KILLED"
                except Exception as e:
                    entry["action"] = "FAILED"
                    entry["error"] = str(e)

            killed.append(entry)

    return killed
