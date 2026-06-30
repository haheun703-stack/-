"""주도주 사이클 진단 — US 백테스트 캘리브레이션 (KR 백테스트의 US 짝).

목적:
  1) US 주도주 대표 사이클 길이가 엔진으로 재현되는지 검증(미국 ~2.5년 가설).
  2) US 메가캡은 리딩 도중 깊은 조정(예: NVDA 2025 딥시크·관세 -37%)이 잦아
     정배열이 6주 tolerance를 넘겨 깨지면 사이클이 파편화됨 → US 적정 tolerance_weeks 탐색.

데이터: data/us_market/leader_cycle/{ticker}.parquet (주봉, fetch_us_leader_data.py 산출).
        로컬에 없으면 VPS에서 먼저:
          ssh ... 'cd ~/quantum-master && ./venv/bin/python3.11 -' < scripts/fetch_us_leader_data.py
        후 scp로 가져온다.

정답(expect_months)은 시장 통념 기반 추정(±수개월, 지시서 §7). MAE 최소가 곧 진실은 아님.

사용:
    python -u -X utf8 scripts/backtest_leader_cycle_us.py
    python -u -X utf8 scripts/backtest_leader_cycle_us.py --detail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.use_cases.leader_cycle_diagnosis import extract_all_cycles  # noqa: E402

US_DIR = PROJECT_ROOT / "data" / "us_market" / "leader_cycle"

# US 검증 케이스. expect=대표(최대상승) 사이클 길이(개월, 통념 추정).
US_CASES = [
    {"code": "NVDA", "expect": 30, "note": "AI반도체 2023~ 최장 리더. 2025 딥시크/관세 깊은조정→파편화 위험"},
    {"code": "META", "expect": 24, "note": "2022-11 바닥→2023~ 효율화 리커버리(+400%대)"},
    {"code": "AVGO", "expect": 24, "note": "AI 인프라(맞춤형 ASIC) 2023~"},
    {"code": "TSLA", "expect": 24, "note": "2019~2021 메가사이클(+1000%대) — 역사적 대표"},
    {"code": "SMCI", "expect": 16, "note": "AI서버 2023~2024 폭등 후 회계이슈 급락"},
    {"code": "PLTR", "expect": 20, "note": "AI 내러티브 2023~ (상장 2020-09)"},
    {"code": "AMD",  "expect": 18, "note": "AI가속기 2023~ NVDA 후행"},
    {"code": "AAPL", "expect": 12, "note": "메가캡 대조군 — 추세 완만/잦은 조정(짧게 분리 예상)"},
]
# US는 메가캡 깊은조정 때문에 KR(6)보다 큰 tolerance가 필요할 수 있어 상단 확장 sweep.
SWEEP = [6, 8, 10, 12, 16]


def representative_cycle(cycles: list[dict]) -> dict | None:
    """대표 사이클 = 그 종목 역사적 정점가(peak_close 최대)를 만든 구간."""
    if not cycles:
        return None
    return max(cycles, key=lambda c: (c.get("peak_close") or -1e9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="tol=8 사이클 구간 상세")
    args = ap.parse_args()

    weeklies: dict[str, pd.DataFrame] = {}
    for c in US_CASES:
        path = US_DIR / f"{c['code']}.parquet"
        if path.exists():
            weeklies[c["code"]] = pd.read_parquet(path)   # 이미 주봉

    if not weeklies:
        print(f"✗ US 주봉 데이터 없음: {US_DIR}")
        print("  먼저 VPS에서 fetch_us_leader_data.py 실행 후 scp로 가져오세요.")
        return 1

    print("=== 주도주 사이클 US 백테스트 (대표 사이클 = 최대상승 구간, market=US) ===")
    print(f"케이스 {len(weeklies)}/{len(US_CASES)}종목. us_stretch=1.25는 '시계'에만 적용(사이클 추출은 공통).\n")

    sweep_mae: dict[int, list] = {t: [] for t in SWEEP}

    for c in US_CASES:
        w = weeklies.get(c["code"])
        if w is None:
            print(f"  [데이터없음] {c['code']}")
            continue
        print(f"■ {c['code']}  (정답≈{c['expect']}개월 · {c['note']})")
        for t in SWEEP:
            cycles = extract_all_cycles(w, params={"tolerance_weeks": t})
            rep = representative_cycle(cycles)
            if rep is None:
                print(f"    tol={t:>2}: 사이클 없음")
                continue
            measured = rep["duration_months"]
            err = abs(measured - c["expect"])
            sweep_mae[t].append(err)
            flag = "✓" if err <= 4 else ("~" if err <= 7 else "✗")
            print(f"    tol={t:>2}: 대표 {rep['start']}~{rep['end']} "
                  f"= {measured:>5.1f}m (정점 {rep['peak_date']}, +{rep['ret_to_peak_pct']:.0f}%) "
                  f"| 오차 {err:>4.1f}m {flag}  [총 {len(cycles)}개 사이클]")
            if args.detail and t == 8:
                for cc in cycles:
                    print(f"          · {cc['start']}~{cc['end']} {cc['duration_months']:>5.1f}m "
                          f"정점{cc['peak_date']} +{cc['ret_to_peak_pct']:.0f}%")
        print()

    print("=== tolerance별 평균절대오차(MAE) — 작을수록 통념 재현 ===")
    best_t, best_mae = None, 1e9
    for t in SWEEP:
        errs = sweep_mae[t]
        mae = sum(errs) / len(errs) if errs else float("nan")
        if errs and mae < best_mae:
            best_mae, best_t = mae, t
        print(f"  tol={t:>2}: MAE={mae:5.2f}m  (n={len(errs)})")
    if best_t is not None:
        print(f"\n  ▶ MAE 최소 = tol {best_t} ({best_mae:.2f}m) — 단, 이는 과병합 효과 포함이라 그대로 채택 금지.")
    print("  ▶ 실무 권장 US tolerance_weeks = 10 (diagnose CLI 기본값).")
    print("     · tol 16은 MAE 최소지만 SMCI를 42m로 과병합(2020·2023 별개 런 합침)·NVDA만 39m로 정확.")
    print("     · tol 10이 파편화(NVDA/AVGO가 6주선 2m로 끊김) vs 과병합(SMCI)의 절충점.")
    print("  ★ 발견: US 메가캡은 리딩 중 깊은조정(NVDA 2025 딥시크·관세 -37%)이 6주 tolerance를")
    print("     넘겨 정배열을 깨므로 KR(6)보다 큰 tolerance 필수. us_stretch는 '시계'만 늘릴 뿐")
    print("     '사이클 추출 tolerance'는 별개 → US는 tolerance도 상향해야 함(이 백테스트의 결론).")
    print("  ※ 정답(expect)은 통념 추정(±수개월)·분포 매우 넓음 → MAE는 절대지표 아닌 참고용(§7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
