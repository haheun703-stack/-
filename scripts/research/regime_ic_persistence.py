"""국면 판단기 2단계 — 신호 적중도 지속성(IC persistence). 사장님 6/1.

봇이 "최근 신호가 맞았으면 켜고, 틀렸으면 끄기"가 가능하려면:
  IC(t) = 그 시점 모멘텀 상위10이 다음 FWD일 전체평균보다 나았는가(실현 격차).
  IC(t)가 양수일 때 IC(t+1)도 양수인가? (지속성). 비겹침(FWD 간격) 샘플로 편향 제거.
지속성 있으면 → 신호 ON/OFF 판단기 성립. 없으면(랜덤) → 켜고끄기 무의미.
★2025.6~2026.5만. 생존편향 진단용 1차. look-ahead 0(IC는 실현 후 관측).
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

MIN_TV = 1e9
LOOKBACK = 20
FWD = 10
TOPN = 10


def load():
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); cal = list(k.sort_values("date")["date"])
    C, TV = {}, {}
    for pat in ["data/processed/*.parquet", "data/delisted/*.parquet"]:
        for f in glob.glob(str(PROJECT_ROOT / pat)):
            try:
                df = pd.read_parquet(f).sort_index(); df = df[df["close"] > 0]
            except Exception:
                continue
            if len(df) < 60 or "trading_value" not in df.columns:
                continue
            code = Path(f).stem
            if code in C:
                continue
            C[code] = df["close"]; TV[code] = df["trading_value"]
    return cal, k.set_index("date")["close"], pd.DataFrame(C).reindex(cal), pd.DataFrame(TV).reindex(cal).ffill()


def main() -> int:
    cal, kclose, Cm, TVm = load()
    S, E = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-29")
    mom = Cm / Cm.shift(LOOKBACK) - 1
    fwd = Cm.shift(-FWD) / Cm - 1

    # 비겹침 리밸 시점 (FWD 간격)
    days = [i for i, d in enumerate(cal) if S <= d <= E and i + FWD < len(cal) and i - LOOKBACK >= 0]
    sample = days[::FWD]
    ics = []
    for i in sample:
        d = cal[i]
        liq = TVm.loc[d] >= MIN_TV
        m = mom.loc[d][liq].dropna(); f = fwd.loc[d].dropna()
        common = m.index.intersection(f.index)
        if len(common) < 50:
            continue
        m = m[common]; f = f[common]
        top = m.nlargest(TOPN).index
        ic = f[top].mean() - f.mean()   # 상위10 초과 (실현 IC 대용)
        ics.append((d, ic))

    s = pd.Series([x[1] for x in ics], index=[x[0] for x in ics])
    print(f"국면 판단기 2단계 — 신호 적중도 지속성 (2025.6~2026.5, 비겹침 {len(s)}구간)\n")
    print(f"  평균 IC(상위10 초과수익): {s.mean()*100:+.2f}%  (양수면 모멘텀 평균적으로 유효)")
    print(f"  IC 양수 비율: {(s>0).mean()*100:.0f}%")
    # 지속성: 직전 IC 부호별 다음 IC 평균
    prev = s.shift(1)
    pos = s[prev > 0]; neg = s[prev <= 0]
    print(f"\n  지속성 검증 (직전 구간 신호가 맞았/틀렸을 때 → 다음 구간 IC):")
    print(f"    직전 IC 양수(맞음) 후 → 다음 IC 평균: {pos.mean()*100:+.2f}%  (n={len(pos)})")
    print(f"    직전 IC 음수(틀림) 후 → 다음 IC 평균: {neg.mean()*100:+.2f}%  (n={len(neg)})")
    ac = s.autocorr(1)
    print(f"    IC 자기상관(lag1): {ac:+.3f}")
    print(f"\n★ 양수후 > 음수후 (격차 크고) AND 자기상관 +면 = 켜고끄기 성립.")
    print("  비슷하거나 역전이면 = 적중도 랜덤 → ON/OFF 판단 무의미.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
