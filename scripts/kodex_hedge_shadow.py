"""KODEX 레버리지+인버스 국면별 헤지 shadow 관측 CLI (실매매 0).

6/10~6/12 **장마감(15:30 KST) 후** 매일 --write 실행 → 6/12 밤 다음 주 실전 전환 판정.
사용:
  python -u -X utf8 scripts/kodex_hedge_shadow.py            # dry 조회(기본·기록 안 함)
  python -u -X utf8 scripts/kodex_hedge_shadow.py --write    # ledger 기록(장마감 후에만)

★dry-run이 기본(macro 패턴): 인자 없이 = 조회만, 쓰기는 --write 명시할 때만.
  점검 도구가 건드려도 ledger 오염 0. 장중 --write 시 is_final=false(provisional)로 남아
  저녁 종가 재실행 전까지 소비자가 미확정값을 안 읽도록 자기상태를 선언.
★실매매/주문/브로커/SAJANG/scheduler 무접촉. 추천·실전 신호 아님. read-only + shadow ledger.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.etf.kodex_hedge_regime_shadow import (  # noqa: E402
    load_ledger,
    run_snapshot,
    stale_provisional_warning,
)


def main():
    write = "--write" in sys.argv  # dry 기본, --write로만 기록
    rec = run_snapshot(write=write)
    mode = "기록(--write)" if write else "조회(dry·기본)"
    final = "확정(is_final)" if rec.get("is_final") else "미확정(provisional·장중)"
    print(f"=== KODEX 국면별 헤지 shadow — {rec['date']} [{mode}] {final} ===")
    print(f"regime={rec['regime']} | policy={rec['hedge_policy']} | hedge_ratio={rec['hedge_ratio']}")
    print(f"  사유: {rec['hedge_reason']}")
    print(f"KODEX200 {rec['kodex200_close']} ({rec['kodex200_ret_1d']:+.2f}%) | "
          f"레버 {rec['leverage_close']} ({rec['leverage_ret_1d']:+.2f}%) | "
          f"인버스 {rec['inverse_close']} ({rec['inverse_ret_1d']:+.2f}%)")
    print(f"regime-adaptive: 당일 {rec['portfolio_ret_1d']:+.2f}% · 누적 {rec['portfolio_cum_ret']:+.2f}% · "
          f"MDD {rec['mdd']:.2f}% · 실현변동성 {rec['realized_vol']:.1f}% · whipsaw={rec['whipsaw_flag']}")
    print("  7종 shadow 비교(당일%):")
    for k, v in rec["portfolios_ret_1d"].items():
        print(f"    {k:<30} {v:+.2f}%")
    print(f"안전: real_order=False · broker=None · SAJANG·scheduler 무변경 · 추천 아님 "
          f"(확정 관측 누적 {len(load_ledger())}일)")
    warn = stale_provisional_warning()
    if warn:
        print(f"⚠️  {warn}")
    if not write and rec.get("is_final"):
        print("ℹ️  dry 조회 — 기록하려면 --write (장마감 후)")


if __name__ == "__main__":
    main()
