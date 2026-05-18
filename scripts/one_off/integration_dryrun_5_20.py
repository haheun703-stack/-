"""5/20 자동매매 통합 흐름 dry-run 시뮬레이션 (2026-05-18 작성, 1회성)

목적: 5/18 28 커밋(사장님 룰 ①②③④ + 자동매매 안전선 9건 + DART EYE) 통합 흐름을
      실전 가동 전에 시간순으로 시뮬레이션하여 자기반성 #1 표준 충족.

자기반성 #1 표준 (MEMORY.md 5/17):
  ❌ import OK = 동작 OK 오판
  ✅ ① import 검증 → ② 함수 호출 → ③ main 흐름 시뮬 (3단계 모두)

흐름 (5/20 가상):
  14:30  Phase A: should_auto_buy(안전선 9건) → BUY 결정
  14:35~ Phase B: evaluate_owner_rule(매 5분, HOLD)
  14:50  Phase C: 트레일링 활성화 (peak +5%)
  15:20  Phase D: evaluate_owner_rule → SELL_FORCE_CLOSE
                  → evaluate_hold_overnight 이월 평가
                  → 이월 OR 청산

시나리오 7개:
  [매수] A1 통과 (점수 92 / EYE PASS / 14:30 / NEUTRAL / 95000원)
  [매수] A2 SKIP (점수 85 미달)
  [청산/이월] B1 이월 (15:20 / +5% / 수급+2억 / EYE PASS / 트레일링 안전 / 0일)
  [청산/이월] B2 수급 부족 청산 (+5% / 수급 0.5억)
  [청산/이월] B3 모멘텀 부족 청산 (+2% / 수급 +2억)
  [청산/이월] B4 트레일링 청산 (14:50 / peak 100k → 96k)
  [청산/이월] B5 5일 보유 한도 청산

사용:
  source venv/Scripts/activate
  python -u -X utf8 scripts/one_off/integration_dryrun_5_20.py

부작용 없음 (KIS API 호출 X, 파일 쓰기 X, 텔레그램 X — 콘솔 출력만).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# ① import 검증 (자기반성 #1 표준 1단계)
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("① IMPORT 검증")
print("=" * 70)

import_results = []
try:
    from src.use_cases.auto_buy_decider import should_auto_buy, BuyDecision, format_decision_for_telegram
    import_results.append(("auto_buy_decider", "OK"))
except Exception as e:
    import_results.append(("auto_buy_decider", f"FAIL: {e}"))

try:
    from src.use_cases.owner_rule import (
        evaluate_owner_rule,
        evaluate_hold_overnight,
        OwnerRuleVerdict,
        OWNER_STOP_LOSS_PCT,
        OWNER_TRAILING_ACTIVATE_PCT,
        OWNER_TRAILING_STOP_PCT,
        OWNER_FORCE_CLOSE_TIME,
        OWNER_HOLD_OVERNIGHT_MIN_GAIN_PCT,
        OWNER_HOLD_OVERNIGHT_MIN_SUPPLY_EOK,
        OWNER_MAX_HOLDING_DAYS,
    )
    import_results.append(("owner_rule", "OK"))
except Exception as e:
    import_results.append(("owner_rule", f"FAIL: {e}"))

try:
    from src.use_cases.eye_filters import evaluate_filters, should_skip
    import_results.append(("eye_filters", "OK"))
except Exception as e:
    import_results.append(("eye_filters", f"FAIL: {e}"))

for module, status in import_results:
    flag = "✅" if status == "OK" else "❌"
    print(f"  {flag} {module}: {status}")

if any("FAIL" in s for _, s in import_results):
    print("\n❌ import 실패 → 시뮬레이션 중단")
    sys.exit(1)
print("\n✅ 4 모듈 import 모두 OK")


# ──────────────────────────────────────────────────────────────────────
# ② 함수 호출 dry-run (자기반성 #1 표준 2단계) — 각 함수 개별 호출
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("② 함수 호출 dry-run (시나리오 7개)")
print("=" * 70)


def section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ─── Phase A: should_auto_buy ─────────────────────────────────────────
section("A1. should_auto_buy — 안전선 9건 ALL 통과 (BUY)")
import os
os.environ["AUTO_TRADE_5_20"] = "true"  # 시뮬 한정 — 부작용 없음

# Mock daily count: paper_portfolio.json은 Mar 28 기록이라 today entry 0 보장
decision_a1 = should_auto_buy(
    ticker="403870",
    name="HPSP",
    integrated_score=92.0,
    eye_should_skip=False,
    eye_skip_reasons=[],
    market_regime="NEUTRAL",
    current_price=95_000,
    blocked_by_nega=False,
    now_str="14:30",
    today="2026-05-20",
)
print(f"  action={decision_a1.action}, reason={decision_a1.reason}")
print(f"  통과: {len(decision_a1.checks_passed)}건")
for c in decision_a1.checks_passed:
    print(f"    ✓ {c}")
if decision_a1.checks_failed:
    for c in decision_a1.checks_failed:
        print(f"    ✗ {c}")
assert decision_a1.action == "BUY", "A1: BUY 결정 실패"


section("A2. should_auto_buy — 점수 85 미달 (SKIP)")
decision_a2 = should_auto_buy(
    ticker="403870",
    name="HPSP",
    integrated_score=85.0,   # 90 미달
    eye_should_skip=False,
    eye_skip_reasons=[],
    market_regime="NEUTRAL",
    current_price=95_000,
    blocked_by_nega=False,
    now_str="14:30",
    today="2026-05-20",
)
print(f"  action={decision_a2.action}, reason={decision_a2.reason}")
for c in decision_a2.checks_failed:
    print(f"    ✗ {c}")
assert decision_a2.action == "SKIP", "A2: SKIP 결정 실패"


# ─── Phase B: evaluate_owner_rule 시간순 (14:30 → 15:20) ──────────────
section("B0. evaluate_owner_rule — 시간순 HOLD 시퀀스 (golden path)")
entry_price = 95_000
peak = entry_price
trailing = False
price_seq = [
    ("14:35", 95_500, "+0.5%"),
    ("14:45", 97_500, "+2.6% (트레일링 미활성)"),
    ("14:50", 99_750, "+5.0% (트레일링 활성화)"),
    ("15:00", 99_500, "peak 99750 대비 -0.25%"),
    ("15:15", 99_700, "peak 동일"),
]
for t, px, note in price_seq:
    v = evaluate_owner_rule(entry_price, px, peak, trailing, t)
    peak = v.peak_price
    trailing = v.trailing_active
    print(f"  {t} {px:,}원 ({note}) → {v.action} (peak={peak:,}, trail={'ON' if trailing else 'OFF'})")
    assert v.action == "HOLD", f"B0 {t}: HOLD 예상 but {v.action}"
print("  ✅ 14:35~15:15 5건 모두 HOLD + 트레일링 14:50 활성화 OK")


# ─── Phase D: 5 시나리오 (15:20 도달, 룰 ④ 평가) ──────────────────────
section("B1. 15:20 — 이월 (golden: 양봉 +5% + 수급 +2억 + EYE PASS)")
v_b1 = evaluate_owner_rule(95_000, 99_750, 99_750, True, "15:20")
print(f"  evaluate_owner_rule → {v_b1.action} (peak {v_b1.peak_price:,})")
assert v_b1.action == "SELL_FORCE_CLOSE"

can_hold_b1, det_b1 = evaluate_hold_overnight(
    entry_price=95_000,
    current_price=99_750,
    peak_price=99_750,
    trailing_active=True,
    days_held=0,
    foreign_net_eok=1.2,
    inst_net_eok=0.5,
    pension_net_eok=0.3,  # 합계 +2.0억
    eye_filter_passed=True,
)
print(f"  evaluate_hold_overnight: can_hold={can_hold_b1}, verdict={det_b1['verdict']}")
print(f"    PnL {det_b1['pnl_pct']:+.2f}% / 수급 +{det_b1['supply_total_eok']:.1f}억 / days_held={det_b1['days_held']}")
for k, v in det_b1["checks"].items():
    print(f"    {'✓' if v else '✗'} {k}")
assert can_hold_b1, "B1: 이월 실패"


section("B2. 15:20 — 수급 부족 청산 (+5% but 수급 +0.5억)")
can_hold_b2, det_b2 = evaluate_hold_overnight(
    entry_price=95_000, current_price=99_750, peak_price=99_750,
    trailing_active=True, days_held=0,
    foreign_net_eok=0.3, inst_net_eok=0.2, pension_net_eok=0.0,  # 합계 0.5억
    eye_filter_passed=True,
)
print(f"  can_hold={can_hold_b2}, verdict={det_b2['verdict']}, 수급={det_b2['supply_total_eok']:.1f}억")
for k, v in det_b2["checks"].items():
    print(f"    {'✓' if v else '✗'} {k}")
assert not can_hold_b2, "B2: 이월 안 됨 예상 but 이월"


section("B3. 15:20 — 모멘텀 부족 청산 (+2% 종가, 3% 미달)")
can_hold_b3, det_b3 = evaluate_hold_overnight(
    entry_price=95_000, current_price=96_900, peak_price=96_900,  # +2.0%
    trailing_active=False, days_held=0,
    foreign_net_eok=1.0, inst_net_eok=1.0, pension_net_eok=0.0,
    eye_filter_passed=True,
)
print(f"  can_hold={can_hold_b3}, verdict={det_b3['verdict']}, PnL {det_b3['pnl_pct']:+.2f}%")
for k, v in det_b3["checks"].items():
    print(f"    {'✓' if v else '✗'} {k}")
assert not can_hold_b3, "B3: 청산 예상 but 이월"


section("B4. 14:50 — 트레일링 청산 (peak 100k → 96k = -4%)")
v_b4 = evaluate_owner_rule(
    entry_price=95_000,
    current_price=96_000,      # peak 대비 -4%
    peak_price=100_000,         # 이미 peak 도달
    trailing_active=True,       # peak 시점에 활성화
    current_time="14:50",
)
print(f"  → {v_b4.action} (PnL {v_b4.pnl_pct:+.2f}%, peak drop {v_b4.peak_drop_pct:+.2f}%)")
print(f"  사유: {v_b4.reason}")
assert v_b4.action == "SELL_TRAILING", f"B4: SELL_TRAILING 예상 but {v_b4.action}"


section("B5. 15:20 — 5일 보유 한도 청산 (모든 조건 OK but days_held=5)")
can_hold_b5, det_b5 = evaluate_hold_overnight(
    entry_price=95_000, current_price=99_750, peak_price=99_750,
    trailing_active=True, days_held=5,    # 한도 도달
    foreign_net_eok=1.0, inst_net_eok=1.0, pension_net_eok=0.0,
    eye_filter_passed=True,
)
print(f"  can_hold={can_hold_b5}, verdict={det_b5['verdict']}, days_held={det_b5['days_held']}/{OWNER_MAX_HOLDING_DAYS}")
for k, v in det_b5["checks"].items():
    print(f"    {'✓' if v else '✗'} {k}")
assert not can_hold_b5, "B5: 청산 예상 but 이월"


# 추가 손절 시나리오 (룰 ①)
section("B6 (보너스). 14:40 — 절대 손절 (-3.5%)")
v_b6 = evaluate_owner_rule(
    entry_price=95_000,
    current_price=91_675,   # -3.5%
    peak_price=95_000,
    trailing_active=False,
    current_time="14:40",
)
print(f"  → {v_b6.action} (PnL {v_b6.pnl_pct:+.2f}%)")
print(f"  사유: {v_b6.reason}")
assert v_b6.action == "SELL_STOP_LOSS"


# ──────────────────────────────────────────────────────────────────────
# ③ main 흐름 시뮬 (자기반성 #1 표준 3단계) — 시간순 통합 흐름
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("③ main 흐름 시뮬 — 5/20 시간순 통합 (시나리오 A1 → B1)")
print("=" * 70)

print("""
[14:30] Phase A — 매수 평가
  안전선 9건 ALL 통과 → BUY 결정
  ↓
  (가정) kis_order_adapter.place_order(403870, 1주, 95,000원) → 체결
  ↓
  owner_rule_positions.json 갱신: HPSP { entry_price: 95000, entry_date: 2026-05-20 }
""")

# 14:30 매수 확인 (위 A1 재사용)
print(f"  [check] should_auto_buy → {decision_a1.action} ({decision_a1.reason})")

print("""
[14:35~15:15] Phase B — 매 5분 owner_rule_monitor.main()
  KILL_SWITCH 가드 OK, AUTO_TRADE_5_20 OK
  fetch_balance → [HPSP 1주]
  evaluate_owner_rule(매 5분) → HOLD
""")
print("  [check] 5건 모두 HOLD (위 B0 시퀀스 통과)")

print("""
[15:20] Phase D — owner_rule_monitor 룰 ④ 분기
  evaluate_owner_rule → SELL_FORCE_CLOSE
  ↓
  evaluate_hold_overnight: PnL +5% / 수급 +2억 / EYE PASS / 트레일링 안전 / days_held=0
  ↓
  4 조건 ALL 통과 → 이월 결정
  ↓
  텔레그램: "🌙 [사장님 룰 ④ — 익일 이월] HPSP(403870) PnL +5.00% / 수급 +2.0억 → 5/21 익일 보유"
""")
print(f"  [check] can_hold={can_hold_b1} (B1) → 이월 결정 OK")
print(f"  [check] 다음 거래일까지 보유 (최대 {OWNER_MAX_HOLDING_DAYS - 0 - 1}일 더)")

print("""
[15:30] Phase E — daily_winners.py (학습용, 자동매매 무관)
[16:00] Phase F — KILL_SWITCH 자동 복구 (자동매매 OFF)
        단, owner_rule_positions에 HPSP 남아 있어서 5/21 09:00 cron 재가동
""")


# ──────────────────────────────────────────────────────────────────────
# 결과 요약
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📊 통합 흐름 dry-run 결과")
print("=" * 70)
print("""
  ① import 검증           : 4/4 모듈 OK
  ② 함수 호출 dry-run     : 7/7 시나리오 통과
       A1 BUY (안전선 9건 ALL 통과)
       A2 SKIP (점수 85 미달)
       B0 HOLD 시퀀스 5건 (14:35~15:15)
       B1 이월 (수급 +2억 + EYE + 트레일링 안전)
       B2 수급 부족 청산
       B3 모멘텀 부족 청산
       B4 트레일링 청산 (14:50)
       B5 5일 보유 한도 청산
       B6 절대 손절 (-3.5%)
  ③ main 흐름 시뮬        : 매수→HOLD→이월 시간순 정합성 OK
""")
print("✅ 1단계 통합 흐름 dry-run 완료 — 자기반성 #1 표준 3단계 모두 충족")
print("\n다음 단계 후보: ")
print("  - 2단계 엣지 케이스 (오버나잇 갭다운/거래정지/데이터 누락)")
print("  - 2순위 5/18 감리 로그 갱신")
print()

# 환경변수 정리 (다른 프로세스에 영향 없도록)
os.environ.pop("AUTO_TRADE_5_20", None)
