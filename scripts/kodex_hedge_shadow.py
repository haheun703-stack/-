"""KODEX 레버리지+인버스 국면별 헤지 shadow 관측 CLI (실매매 0).

6/10~6/12 장마감 후 매일 실행 → 6/12 밤 다음 주 실전 전환 판정.
사용:
  python -u -X utf8 scripts/kodex_hedge_shadow.py            # 오늘 snapshot + ledger 기록
  python -u -X utf8 scripts/kodex_hedge_shadow.py --no-write # 조회만(기록 안 함)

★실매매/주문/브로커/SAJANG/scheduler 무접촉. 추천·실전 신호 아님. read-only + shadow ledger.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.etf.kodex_hedge_regime_shadow import load_ledger, run_snapshot  # noqa: E402


def main():
    write = "--no-write" not in sys.argv
    rec = run_snapshot(write=write)
    print(f"=== KODEX 국면별 헤지 shadow — {rec['date']} ===")
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
          f"(관측 누적 {len(load_ledger())}일)")


if __name__ == "__main__":
    main()
