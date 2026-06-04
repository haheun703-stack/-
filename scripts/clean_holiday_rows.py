"""휴장일 유령행 청소 도구 (6/3 선거 휴장 등)

배경: pykrx 우회 수집이 휴장일을 거래일로 잡고 OHLCV=0 빈 행을 삽입.
close=0 행은 MA/수익률/C60 지표를 파괴하므로 제거해야 함.

안전장치:
  - 지정한 날짜의 행만 대상
  - 해당 행의 OHLCV가 모두 0(진짜 유령)일 때만 제거 → 실데이터는 절대 손대지 않음
  - 기본 dry-run, --apply 플래그로만 실제 저장
  - 제거 전후 행수 로깅

사용:
  python -u -X utf8 scripts/clean_holiday_rows.py 2026-06-03            # dry-run
  python -u -X utf8 scripts/clean_holiday_rows.py 2026-06-03 --apply    # 실제 제거
  python -u -X utf8 scripts/clean_holiday_rows.py 2026-06-03 --apply --dir data/processed
"""
import sys
import glob
import os

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("ERROR: 날짜 인자 필요 (예: 2026-06-03)")
        sys.exit(1)
    target = pd.Timestamp(args[0])
    apply = "--apply" in sys.argv
    # --dir 옵션
    data_dir = "data/processed"
    if "--dir" in sys.argv:
        data_dir = sys.argv[sys.argv.index("--dir") + 1]

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    removed = 0       # OHLCV=0 → 제거 대상
    kept_real = 0     # 해당 날짜 있으나 실데이터 → 보존
    no_date = 0       # 해당 날짜 없음
    errors = 0
    sample = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            idx_name = df.index.name
            df.index = pd.to_datetime(df.index)
            if target not in df.index:
                no_date += 1
                continue
            row = df.loc[target]
            cols = [c for c in OHLCV if c in df.columns]
            is_ghost = bool(cols) and all(float(row[c]) == 0 for c in cols)
            if not is_ghost:
                kept_real += 1
                continue
            removed += 1
            if len(sample) < 3:
                sample.append(os.path.basename(f))
            if apply:
                before = len(df)
                df2 = df.drop(target)
                df2.index.name = idx_name
                df2.to_parquet(f)
                assert len(df2) == before - 1
        except Exception as e:  # noqa
            errors += 1
            if errors <= 3:
                print(f"  ! {os.path.basename(f)}: {e}")

    mode = "APPLY(실제 제거)" if apply else "DRY-RUN(미적용)"
    print(f"[{mode}] dir={data_dir} target={target.date()} files={len(files)}")
    print(f"  유령행(OHLCV=0) 제거대상: {removed}")
    print(f"  실데이터 보존(손대지 않음): {kept_real}")
    print(f"  해당 날짜 없음: {no_date} | 오류 skip: {errors}")
    print(f"  예시: {sample}")
    if not apply and removed:
        print("  → 확인 후 --apply 로 실제 제거")


if __name__ == "__main__":
    main()
