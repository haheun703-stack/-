"""엔진 앞/뒤 변동 지문 (사장님 5/31 — "각 엔진이 잡은 경우 앞/뒤 변동").

parquet에 시계열이 있는 25개 엔진 신호 각각에 대해, 신호 발생일 기준:
  앞(D-10→D0): 신호 전 10일 수익 — 음수면 '바닥에서 잡음', 양수면 '오른 뒤 잡음'
  뒤(D0→D+5, D+20): 신호 후 수익 — 양수면 '먹는 신호'
성격: 앞-/뒤+ = 바닥반등 선행(좋음) / 앞+/뒤- = 추격 상투(위험) / 앞+/뒤+ = 추세지속 / ~ = 무의미.
생존편향 보정(상폐 포함), 거래대금≥10억.
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
PRE, P5, P20 = 10, 5, 20

SIGNALS = {
    "supply_div": lambda d: d["supply_divergence"] > 0,
    "inst_consec3": lambda d: d["inst_consecutive_buy"] >= 3,
    "foreign_consec3": lambda d: d["foreign_consecutive_buy"] >= 3,
    "inst_streak3": lambda d: d["inst_net_streak"] >= 3,
    "foreign_streak3": lambda d: d["foreign_net_streak"] >= 3,
    "accum_eff>0": lambda d: d["accumulation_efficiency"] > 0,
    "inst_volcfm": lambda d: d["inst_vol_confirm"] > 0,
    "foreign_volcfm": lambda d: d["foreign_vol_confirm"] > 0,
    "pension_top": lambda d: d["pension_top_buyer"] > 0,
    "trix_golden": lambda d: d["trix_golden_cross"] > 0,
    "sar_reversal": lambda d: d["sar_reversal_up"] > 0,
    "stoch_golden": lambda d: d["stoch_slow_golden"] > 0,
    "dyn_rsi": lambda d: d["dynamic_rsi_signal"] > 0,
    "rsi_rising": lambda d: d["rsi_rising"] > 0,
    "is_bullish": lambda d: d["is_bullish"] > 0,
    "vol_surge2": lambda d: d["volume_surge_ratio"] >= 2,
    "short_cover": lambda d: d["short_cover_signal"] > 0,
    "smart_z1": lambda d: d["smart_z"] > 1,
    "sentiment_extreme": lambda d: d["sentiment_extreme"] > 0,
    "macro_fav": lambda d: d["macro_favorable"] > 0,
}


def main() -> int:
    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    files += glob.glob(str(PROJECT_ROOT / "data" / "delisted" / "*.parquet"))
    acc = {k: {"pre": [], "p5": [], "p20": []} for k in SIGNALS}
    nfiles = 0
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < PRE + P20 + 5 or "trading_value" not in df.columns:
            continue
        nfiles += 1
        c = df["close"].values
        n = len(c)
        tvok = (df["trading_value"].values >= MIN_TV)
        for k, fn in SIGNALS.items():
            try:
                mask = fn(df).fillna(False).values & tvok
            except Exception:
                continue
            for i in np.where(mask)[0]:
                if i - PRE < 0 or i + P20 >= n or c[i] <= 0:
                    continue
                acc[k]["pre"].append(c[i] / c[i - PRE] - 1)
                acc[k]["p5"].append(c[i + P5] / c[i] - 1)
                acc[k]["p20"].append(c[i + P20] / c[i] - 1)

    print(f"종목 {nfiles}개 (생존+상폐) / 거래대금≥10억 / 앞=신호전10일, 뒤=신호후5·20일\n")
    print(f'{"엔진":<16}{"신호수":>8}{"앞(전10일)":>11}{"뒤(D+5)":>9}{"뒤(D+20)":>9}{"성격":>14}')
    rows = []
    for k in SIGNALS:
        a = acc[k]
        if len(a["p20"]) < 30:
            continue
        pre = np.mean(a["pre"]) * 100
        p5 = np.mean(a["p5"]) * 100
        p20 = np.mean(a["p20"]) * 100
        rows.append((k, len(a["p20"]), pre, p5, p20))
    rows.sort(key=lambda x: -x[4])  # 뒤 D+20 순
    for k, n, pre, p5, p20 in rows:
        if pre < -1 and p20 > 1:
            char = "바닥반등 선행★"
        elif pre > 3 and p20 < 0:
            char = "추격 상투⚠"
        elif pre > 1 and p20 > 1:
            char = "추세지속"
        elif abs(p20) < 1:
            char = "무의미"
        else:
            char = "혼조"
        print(f'{k:<16}{n:>8}{pre:>+10.1f}%{p5:>+8.1f}%{p20:>+8.1f}%{char:>14}')
    print("\n★ 앞- 뒤+ = 바닥에서 잡아 오르는 선행(진짜 알파) / 앞+ 뒤- = 오른 뒤 잡는 추격(상투).")
    print("  생존편향 보정(상폐 포함). 전기간 평균이라 레짐 혼재 — 구간별은 후속.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
