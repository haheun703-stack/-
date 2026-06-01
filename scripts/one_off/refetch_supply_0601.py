"""6/1 수급 확정 재수집 — 잠정값 강제 덮어쓰기 (일회성, 6/1 밤 예약).

배경: extend_parquet는 'last_date>=end면 스킵'이라 6/1 잠정 수급을 못 덮어씀.
이 헬퍼는 processed 6/1 행의 수급 4컬럼(외국인합계/기관합계/개인/기타법인)만
pykrx 확정값으로 강제 덮어쓴다. 종가·OHLCV·지표는 손대지 않음.
KRX 종목별 투자자 수급이 밤늦게~익일 확정되므로 23시·익일06시 2회 실행 예약.

Usage:
  python scripts/one_off/refetch_supply_0601.py --sample 5   # 5종목 테스트
  python scripts/one_off/refetch_supply_0601.py              # 전종목
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd

# VPS/로컬 모두 대응
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TARGET = "20260601"
TDATE = pd.Timestamp("2026-06-01")
SUPPLY = ["기관합계", "외국인합계", "개인", "기타법인"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    from pykrx import stock

    files = sorted(glob.glob(str(ROOT / "data" / "processed" / "*.parquet")))
    if args.sample:
        files = files[: args.sample]

    upd = skip = err = 0
    changed = []
    flip = []  # 부호 반전 종목 (잠정→확정 차이 큰 것)
    for f in files:
        code = Path(f).stem
        try:
            df = pd.read_parquet(f)
            if TDATE not in df.index:
                skip += 1
                continue
            inv = stock.get_market_trading_value_by_date(TARGET, TARGET, code)
            if inv is None or inv.empty:
                err += 1
                continue
            row = inv.iloc[-1]
            old_f = float(df.at[TDATE, "외국인합계"]) if "외국인합계" in df.columns else 0.0
            for c in SUPPLY:
                if c in inv.columns:
                    df.at[TDATE, c] = int(row[c])
            new_f = float(df.at[TDATE, "외국인합계"])
            df.to_parquet(f)
            upd += 1
            if abs(new_f - old_f) > 1e9:  # 10억 이상 차이
                changed.append((code, old_f / 1e8, new_f / 1e8))
            if (old_f > 0) != (new_f > 0) and abs(new_f - old_f) > 1e9:
                flip.append(code)
            time.sleep(args.sleep)
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"  [{code}] ERR {type(e).__name__}: {e}")

    print(f"\n[6/1 수급 재수집] 갱신 {upd} / 스킵(6/1없음) {skip} / 에러 {err}")
    print(f"잠정→확정 10억+ 변동: {len(changed)}종목 / 부호반전: {len(flip)}종목")
    for code, o, n in changed[:25]:
        mark = " ★반전" if code in flip else ""
        print(f"  {code}: 외국인 {o:+.0f}억 → {n:+.0f}억{mark}")
    if upd and err > upd:
        print("⚠️ 에러>갱신 = KRX 아직 미확정 가능. 다음 회차 재시도 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
