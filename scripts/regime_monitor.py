"""Regime monitor CLI — C60 단독 국면 판단 + 보조 관측 로그 (실주문 0 / shadow).

사용:
    python -u -X utf8 scripts/regime_monitor.py
    python -u -X utf8 scripts/regime_monitor.py --days 1300 --no-remote
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # PYTHONPATH 안전장치

from src.etf.regime_monitor import run_all  # noqa: E402


def _fmt(value, suffix=""):
    return "—" if value is None else f"{value}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="C60 regime monitor (shadow)")
    parser.add_argument("--days", type=int, default=1300)
    parser.add_argument("--no-remote", action="store_true", help="pykrx 최신 강제 끄기")
    args = parser.parse_args()

    reports = run_all(days=args.days, prefer_remote=not args.no_remote)

    print("=" * 64)
    print("국면 판단기 (C60 단독 hard gate / 보조 관측 로그만)")
    print("=" * 64)
    for ticker, rep in reports.items():
        if rep.get("rows", 0) == 0:
            print(f"\n[{ticker}] 데이터 없음 ({rep.get('error','')})")
            continue
        regime_kr = "강세장(BULL)" if rep["current_regime"] == "BULL" else "약세전환(BEAR_TRANSITION)"
        print(f"\n[{ticker}] {rep['name']}  ({rep['first_date']} ~ {rep['last_date']}, {rep['rows']}일)")
        print(f"  현재 국면 : {regime_kr}  (지속 {rep['days_in_current_regime']}일)")
        print(f"  종가/60선 : {rep['current_close']} / {rep['current_ma60']}")
        obs = rep["current_observations"]
        print(f"  [관측] 변동성20: {_fmt(obs['realized_vol_20'])}  클러스터경고: {obs['vol_cluster_warn']}")
        print(f"  [관측] KOSPI 60선상: {_fmt(obs['kospi_above_ma60'])}  120선상: {_fmt(obs['kospi_above_ma120'])}  경고: {obs['kospi_warn']}")
        print(f"  [관측] 외국인순매수: {_fmt(obs['foreign_net'])}  경고(순매도): {obs['foreign_warn']}")
        print(f"  약세전환 횟수: {rep['bear_switch_count']}회")
        ll = rep["raw_lead_unadjusted"]
        print(f"  [RAW lead(미보정·착시주의)] 변동성:{_fmt(ll['vol_cluster_raw_lead_unadjusted'])} "
              f"KOSPI:{_fmt(ll['kospi_raw_lead_unadjusted'])} 외국인:{_fmt(ll['foreign_raw_lead_unadjusted'])}")
        print(f"  → ⚠️ base-rate 미보정 RAW값. 정직판정은 regime_obs_adversarial.py")
        print(f"  → 외국인=NOISE(gate제외) / 변동성=WEAK(로그) / KOSPI=후보(추적, gate미승격)")

    print("\n" + "=" * 64)
    print("실주문 0 / 봇 OFF / hard gate=C60 단독 / 보조관측=로그만(gate 미사용)")
    print("=" * 64)


if __name__ == "__main__":
    main()
