# -*- coding: utf-8 -*-
"""정의 검산 게이트 자기 실패 모드 검증 — 오탐/미탐 양쪽"""
import sys; sys.path.insert(0,"/home/ubuntu/quantum-master")
import pandas as pd
from scripts.fill_short_from_jgis import check_definition, _violates_shift, _violates_constant

ok=fail=0
def case(name, got, want):
    global ok,fail
    if got==want: ok+=1; print(f"  ✅ {name}")
    else: fail+=1; print(f"  ❌ {name} — 기대 {want}, 실제 {got}")

print("=== 정의 검산 게이트 selftest ===")
# 1) 실제 관측 형태: sbal[t] == selling[t-1]  → 차단돼야
shift = pd.DataFrame({"short_balance_qty":[0,100,200,300,400,500],
                      "short_selling_qty":[100,200,300,400,500,600]})
case("shift 일치 → 차단", check_definition("short_balance", shift) is not None, True)

# 2) 진짜 잔고처럼 독립적으로 움직임 → 통과해야 (★오탐 검사)
real = pd.DataFrame({"short_balance_qty":[5000,5120,4980,5300,5210,5050],
                     "short_selling_qty":[100,200,300,400,500,600]})
case("독립적 잔고 → 통과", check_definition("short_balance", real) is None, True)

# 3) 전 기간 상수 → 차단
const = pd.DataFrame({"short_balance_qty":[777]*6,
                      "short_selling_qty":[1,2,3,4,5,6]})
case("상수 컬럼 → 차단", check_definition("short_balance", const) is not None, True)

# 4) 표본 부족(값 2개) → 판정하지 않음 (섣부른 차단 방지)
few = pd.DataFrame({"short_balance_qty":[0,0,0,0,100,200],
                    "short_selling_qty":[9,9,9,100,200,300]})
case("표본 부족 → 판정 보류", check_definition("short_balance", few) is None, True)

# 5) 원천에 컬럼 자체가 없음 → 통과(다른 단계가 거른다)
none = pd.DataFrame({"short_selling_qty":[1,2,3,4,5,6]})
case("컬럼 부재 → 게이트 무관", check_definition("short_balance", none) is None, True)

# 6) 게이트 미등록 컬럼은 항상 통과
case("미등록 컬럼 통과", check_definition("short_volume", shift) is None, True)

# 7) 부분 일치(50%)는 차단하지 않음 — 임계 0.7
half = pd.DataFrame({"short_balance_qty":[0,100,999,300,888,500],
                     "short_selling_qty":[100,200,300,400,500,600]})
case("부분일치 50% → 통과", check_definition("short_balance", half) is None, True)

print(f"\n{ok} 통과 · {fail} 실패")
sys.exit(1 if fail else 0)
