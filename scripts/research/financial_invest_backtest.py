"""금융투자 신호 검증 — 삼성전기 패턴이 전체에서 통하는가 (사장님 5/31).

금융투자가 외국인 매도를 받으며 선행 매집(삼성전기 +1.7조)한 게 폭등 엔진이었음.
사후선택인지 진짜 알파인지 검증.
  A. 금융투자 매집강도(20일순매수/60일평균거래대금) 상위
  B. 금투 매수 + 외인 매도 다이버전스(삼성전기 패턴)
가 D+20 수익·폭등% 높이는지.

★ 검증 구간은 1년/6개월만 (사장님 지시): 통짜로 길게 섞으면 레짐 혼재 → 최근 시장 기준.
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
H = 20


def main() -> int:
    files = glob.glob(str(PROJECT_ROOT / "data" / "financial_invest" / "*.parquet"))
    if not files:
        print("data/financial_invest/ 비어있음 — 수집 먼저")
        return 1

    def mm(x):
        return np.mean(x) * 100 if len(x) else 0.0

    print(f"금융투자 신호 검증 (종목 {len(files)} / D+{H} / 1년·6개월 단위)\n")
    # 1년 / 6개월 단위 (레짐 혼재 방지)
    periods = [
        ("2025-06-01", "2026-05-29", "최근 1년"),
        ("2025-12-01", "2026-05-29", "최근 6개월"),
    ]
    for s, e, lbl in periods:
        base, A, B = [], [], []
        for f in files:
            try:
                df = pd.read_parquet(f).sort_index()
                df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) < H + 25:
                continue
            fin = df["금융투자"]; fr = df["외국인"]
            fin20 = fin.rolling(20).sum()
            fr20 = fr.rolling(20).sum()
            tvma = df["trading_value"].rolling(60).mean().replace(0, np.nan)
            finstr = fin20 / tvma
            no = df["open"].shift(-1); ec = df["close"].shift(-(H + 1))
            fwd = ec / no - 1 - COST
            v = fwd.notna() & (no > 0) & (ec > 0) & np.isfinite(fwd)
            inr = (df.index >= pd.Timestamp(s)) & (df.index <= pd.Timestamp(e))
            base.extend(fwd[v & inr].tolist())
            A.extend(fwd[v & inr & (finstr > 0.3)].tolist())
            B.extend(fwd[v & inr & (fin20 > 0) & (fr20 < 0)].tolist())
        b = mm(base)
        boom_b = (np.array(base) >= 0.5).mean() * 100 if base else 0
        boom_A = (np.array(A) >= 0.5).mean() * 100 if A else 0
        boom_B = (np.array(B) >= 0.5).mean() * 100 if B else 0
        print(f"[{lbl}] 베이스 {b:+.2f}%(폭등{boom_b:.1f}%, n{len(base)})")
        print(f"   A 금투매집강도>0.3   : 초과{mm(A)-b:+.2f}%p / 폭등{boom_A:.1f}% (n{len(A)})")
        print(f"   B 금투매수+외인매도  : 초과{mm(B)-b:+.2f}%p / 폭등{boom_B:.1f}% (n{len(B)})")
        print()
    print("★ A/B 초과 +면 금융투자 진짜 알파(담아야 할 신호). 베이스 수준이면 삼성전기는 사후선택.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
