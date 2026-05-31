"""전환 신호 수익성 검증 (사장님 5/31 "검증 해야지").

scan_turning_up 신호(당일 급등+7% + 기관 매수 전환 + 외인흡수>0 + 거래대금≥100억)가
뜬 다음날 진입 시, D+N 후 실제로 올랐는가. 벤치마크(거래대금100억+ 전종목 평균) 대비.
생존편향 보정(상폐 190 포함). lookahead 없음(신호 i일 종가 확정 → i+1 진입).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

COST = 0.005
MIN_TV = 1e10  # 100억


def main() -> int:
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    files += glob.glob(str(PROJECT_ROOT / "data" / "delisted" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
            if len(df) >= 40 and all(c in df.columns for c in ["trading_value", "외국인합계", "기관합계"]):
                dfs.append(df)
        except Exception:
            pass
    print(f"종목 {len(dfs)}개 (생존+상폐) / 전환신호=급등7%+기관전환+흡수>0+거래대금100억\n")

    def mm(x):
        return float(np.mean(x)) * 100 if len(x) else 0.0

    periods = [
        ("2023-06-01", "2024-12-31", "23~24"),
        ("2025-01-01", "2025-12-31", "2025"),
        ("2026-01-01", "2026-05-29", "2026"),
        ("2023-06-01", "2026-05-29", "전체"),
    ]
    for H in [10, 20]:
        print(f"=== D+{H} 보유 (vs 거래대금100억+ 전종목 평균) ===")
        print(f'{"구간":<8}{"베이스":>8}{"전환신호":>9}{"초과":>9}{"승률":>6}{"신호수":>7}')
        for s, e, lbl in periods:
            base, sig = [], []
            for df in dfs:
                ret = df["close"].pct_change() * 100
                inst = df["기관합계"]
                inst5 = inst.rolling(5).sum()
                instp = inst.shift(5).rolling(5).sum()
                sret = ret.where(df["외국인합계"] < 0)
                absorb = sret.rolling(10).mean()
                surge = ret >= 7
                no = df["open"].shift(-1)
                ec = df["close"].shift(-(H + 1))
                fwd = ec / no - 1 - COST
                valid = (no > 0) & (ec > 0) & fwd.notna() & (df["trading_value"] >= MIN_TV)
                inr = (df.index >= pd.Timestamp(s)) & (df.index <= pd.Timestamp(e))
                base.extend(fwd[valid & inr].tolist())
                sigm = (surge & (inst5 > 0) & (inst5 > instp) & (absorb > 0)).fillna(False)
                wins = fwd[valid & inr & sigm]
                sig.extend(wins.tolist())
            b = mm(base)
            wr = (np.array(sig) > 0).mean() * 100 if sig else 0
            print(f'{lbl:<8}{b:>+7.2f}%{mm(sig):>+8.2f}%{mm(sig)-b:>+8.2f}%p{wr:>5.0f}%{len(sig):>7}')
        print()
    print("★ 전환신호 초과(+)면 '고개 들 때 진입→추가상승' 유효. 0~음수면 추격=상투.")
    print("  생존편향 보정(상폐 포함). PASS 전 가설.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
