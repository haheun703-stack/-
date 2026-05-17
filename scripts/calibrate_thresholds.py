"""§13-3 임계값 캘리브레이션 CLI (P2 §13-7 #11)

설계: docs/02-design/threshold-calibration-design.md
로직: src/use_cases/threshold_calibration.py

Usage:
    # 데이터 가용성만 확인 (시그널 평가 0건)
    python scripts/calibrate_thresholds.py --check-only

    # 시총 상위 100종목 1년치 라벨링 (sweep 함수 미구현 → 라벨만 산출)
    python scripts/calibrate_thresholds.py --tickers-top 100 --label-only

    # 전체 sweep 실행 (5/20 이후 evaluate_signal_* 구현 후 활성화)
    python scripts/calibrate_thresholds.py --target d1_accuracy --tickers-top 500
    python scripts/calibrate_thresholds.py --target entry_guard --tickers-top 500
    python scripts/calibrate_thresholds.py --target volume_5min --tickers-top 100
    python scripts/calibrate_thresholds.py --target strength --tickers-top 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.use_cases.threshold_calibration import (  # noqa: E402
    SweepResult,
    label_forward_returns,
    list_available_tickers,
    load_jgis_daily,
    save_sweep_result,
)


def cmd_check_only() -> int:
    """데이터 가용성만 확인 (대표 종목 + 데이터 분포)."""
    tickers = list_available_tickers()
    print(f"[정보봇 일봉] 종목 수: {len(tickers)}")

    if not tickers:
        print("❌ 정보봇 CSV 없음 — 심볼릭 링크 확인 필요")
        return 1

    # 대표 종목 우선 시도 (삼성전자/SK하이닉스/NAVER), 없으면 데이터 충분한 첫 종목
    preferred = ["005930", "000660", "035420"]
    sample = None
    df = None
    for cand in preferred:
        if cand in tickers:
            df = load_jgis_daily(cand)
            if df is not None:
                sample = cand
                break

    if df is None:
        # fallback: 앞 50종목 중 처음 성공하는 것
        for cand in tickers[:50]:
            df = load_jgis_daily(cand)
            if df is not None:
                sample = cand
                break

    if df is None:
        print("❌ 대표 종목 + 앞 50종목 모두 min_rows=60 미달")
        return 1

    print(f"[샘플] {sample} — {len(df)}행, 컬럼 {len(df.columns)}개")
    print(f"       기간: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"       컬럼: {list(df.columns[:10])}... (총 {len(df.columns)}개)")

    # 데이터 분포 (행 수)
    row_counts = []
    for t in tickers[:200]:  # 샘플 200종목 분포
        d = load_jgis_daily(t, min_rows=1)
        if d is not None:
            row_counts.append(len(d))

    if row_counts:
        import statistics

        print(
            f"[분포 (앞 200종목 샘플)] "
            f"min={min(row_counts)}, median={int(statistics.median(row_counts))}, "
            f"max={max(row_counts)}, ≥60행 비율={sum(r >= 60 for r in row_counts) / len(row_counts):.1%}"
        )

    return 0


def cmd_label_only(tickers_top: int) -> int:
    """라벨링만 실행 (sweep 함수 미구현 단계 dry-run)."""
    tickers = list_available_tickers()[:tickers_top]
    print(f"[라벨링] 종목 {len(tickers)}개 처리 중...")

    total_rows = 0
    skipped = 0
    sample_summary = None

    for i, t in enumerate(tickers):
        df = load_jgis_daily(t)
        if df is None:
            skipped += 1
            continue
        labeled = label_forward_returns(df)
        total_rows += len(labeled)

        if i == 0:
            # 첫 종목 요약 출력
            n_valid_d1 = labeled["ret_d1"].notna().sum()
            mean_ret = labeled["ret_d1"].mean()
            sample_summary = (t, len(labeled), n_valid_d1, mean_ret)

    if sample_summary:
        t, n_rows, n_valid, mean_ret = sample_summary
        print(f"[샘플 첫 종목] {t}: {n_rows}행, ret_d1 valid={n_valid}, mean={mean_ret:+.4%}")

    print(f"[완료] 처리 {len(tickers) - skipped}/{len(tickers)}, 총 라벨 {total_rows:,}행 (skip {skipped})")
    return 0


def cmd_sweep(target: str, tickers_top: int) -> int:
    """본 sweep 실행 (5/20 이후 evaluate_signal_* 구현 후 활성화)."""
    print(f"[sweep] target={target}, tickers={tickers_top}")
    print("⚠️  evaluate_signal_* 함수 미구현 — 5/20~5/21 일정")
    print("    설계: docs/02-design/threshold-calibration-design.md §4 산식")
    print("    placeholder 모두 빈 시그널 반환 → 0건 산출 후 종료")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep = SweepResult(run_id=run_id, target=target)
    sweep.finished_at = datetime.now()
    save_sweep_result(sweep)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="§13-3 임계값 캘리브레이션")
    parser.add_argument("--check-only", action="store_true", help="데이터 가용성만 확인")
    parser.add_argument("--label-only", action="store_true", help="라벨링만 실행")
    parser.add_argument(
        "--target",
        choices=["d1_accuracy", "entry_guard", "volume_5min", "strength"],
        help="sweep 대상",
    )
    parser.add_argument("--tickers-top", type=int, default=100, help="처리할 종목 수 (시총 무관 단순 N개)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.check_only:
        return cmd_check_only()
    if args.label_only:
        return cmd_label_only(args.tickers_top)
    if args.target:
        return cmd_sweep(args.target, args.tickers_top)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
