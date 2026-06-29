"""주도주 사이클 진단 — 백테스트 캘리브레이션 (지시서 §8).

목적: 알려진 주도주 사이클의 길이가 엔진으로 재현되는지 검증하고,
      역추적 tolerance_weeks(잔파동 허용)를 캘리브레이션한다.

방법: 각 케이스 종목의 전체 주봉에서 정배열 사이클을 추출 → "대표 사이클"
      (최대 상승=ret_to_peak 최대)의 길이를 정답과 비교. tolerance를 sweep.

정답(expect_months)은 시장 통념 기반 추정치이며 ±수개월 오차 가정(지시서 §7).
US 케이스(엔비디아 24m·메타 18m)는 로컬 데이터 없음 → 이번 범위 제외(yfinance 별도 수집 필요).

사용:
    python -u -X utf8 scripts/backtest_leader_cycle.py
    python -u -X utf8 scripts/backtest_leader_cycle.py --detail   # 사이클 구간 상세
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.use_cases.leader_cycle_diagnosis import extract_all_cycles  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 지시서 §8 검증 케이스 (KR). expect=정답 사이클 길이(개월, 통념 추정).
CASES = [
    {"code": "086520", "name": "에코프로",         "expect": 7,  "note": "정상 사이클 — 정배열 기준 잘 재현"},
    {"code": "247540", "name": "에코프로비엠",      "expect": 7,  "note": "★V자 급반등 → MA60 지연으로 2023 폭등 정배열 미포착(방법론 한계)"},
    {"code": "329180", "name": "HD현대중공업",     "expect": 24, "note": "라벨=조선 업종 내러티브. 종목 정배열 최대는 2021~22(+419%, 11m)"},
    {"code": "012450", "name": "한화에어로스페이스", "expect": 24, "note": "방산 — tol6에서 26m(+532%) 정확 재현"},
    {"code": "034020", "name": "두산에너빌리티",    "expect": 18, "note": "라벨=원전 업종. 종목은 2024(+57%)/2025(+105%) 2개 사이클로 분리(추세붕괴=종료 정의)"},
]
SWEEP = [3, 4, 5, 6, 8]


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    return d.resample("W-FRI").agg(agg).dropna(subset=["close"])


def representative_cycle(cycles: list[dict]) -> dict | None:
    """대표 사이클 = 그 종목의 역사적 신고가(peak_close 최대)를 만든 구간.

    ret_to_peak(시작 종가 대비)는 시작점에 의존해 분해된 옛 조각을 오선택하므로,
    "주도 사이클 = 신고가 구간"이라는 정의에 맞게 절대 정점가 기준으로 robust화.
    """
    if not cycles:
        return None
    return max(cycles, key=lambda c: (c.get("peak_close") or -1e9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="사이클 구간 상세 출력")
    args = ap.parse_args()

    # 종목별 주봉 캐시
    weeklies: dict[str, pd.DataFrame] = {}
    for c in CASES:
        path = RAW_DIR / f"{c['code']}.parquet"
        if path.exists():
            weeklies[c["code"]] = resample_weekly(pd.read_parquet(path))

    print("=== 주도주 사이클 백테스트 캘리브레이션 (대표 사이클 = 최대상승 구간) ===")
    print(f"케이스 {len(CASES)}종목 (KR). US(엔비디아/메타)는 로컬 데이터 없음 → 제외.\n")

    sweep_mae: dict[int, list] = {t: [] for t in SWEEP}

    for c in CASES:
        w = weeklies.get(c["code"])
        if w is None:
            print(f"  [parquet없음] {c['code']} {c['name']}")
            continue
        print(f"■ {c['code']} {c['name']}  (정답≈{c['expect']}개월 · {c['note']})")
        for t in SWEEP:
            cycles = extract_all_cycles(w, params={"tolerance_weeks": t})
            rep = representative_cycle(cycles)
            if rep is None:
                print(f"    tol={t}: 사이클 없음")
                continue
            measured = rep["duration_months"]
            err = abs(measured - c["expect"])
            sweep_mae[t].append(err)
            flag = "✓" if err <= 4 else ("~" if err <= 7 else "✗")
            print(f"    tol={t}: 대표사이클 {rep['start']}~{rep['end']} "
                  f"= {measured:>5.1f}m (정점 {rep['peak_date']}, +{rep['ret_to_peak_pct']:.0f}%) "
                  f"| 오차 {err:>4.1f}m {flag}  [사이클 {len(cycles)}개]")
            if args.detail and t == 4:
                for cc in cycles:
                    print(f"          · {cc['start']}~{cc['end']} {cc['duration_months']:>5.1f}m "
                          f"정점{cc['peak_date']} +{cc['ret_to_peak_pct']:.0f}%")
        print()

    print("=== tolerance별 평균절대오차(MAE) — 작을수록 정답 재현 ===")
    best_t, best_mae = None, 1e9
    for t in SWEEP:
        errs = sweep_mae[t]
        mae = sum(errs) / len(errs) if errs else float("nan")
        mark = ""
        if errs and mae < best_mae:
            best_mae, best_t = mae, t
        print(f"  tol={t}: MAE={mae:5.2f}m  (n={len(errs)})")
    if best_t is not None:
        print(f"\n  ▶ 권장 tolerance_weeks = {best_t} (MAE {best_mae:.2f}m)")
    print("\n  ※ 정답은 통념 추정(±수개월). MAE 최소가 곧 진실은 아니며 분포가 넓다(지시서 §7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
