"""엔진 신호 통합 백테스트 — 보유 엔진 조합 탐색 (사장님 5/31 지시).

핵심: parquet에 이미 박힌 엔진 신호 25개를 종목×날짜로 모아,
①단일 신호 raw edge 스크리닝 → ②살아있는 신호 2~3개 조합 → '어떤 통합이 수익 좋은지' 랭킹.
6조건: signal≠entry(다음날 시가) / 유동성(거래대금≥10억) / 비용 0.5% / 청산 D+N 고정 /
구간 분리(워크포워드) / 리포트(PF·승률·평균·거래수). 사후 최적화 금지(신호 사전 고정).

벡터화(analyze 없음) → 전체 1160종목 빠름. 실매수 HOLD. PASS 전까지 가설.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

COST = 0.005
MIN_TV = 1e9  # 거래대금 하한 10억

# 엔진 신호 (parquet 컬럼 → boolean). 사전 고정, 사후 최적화 금지.
SIGNALS = {
    "foreign_consec3": lambda d: d["foreign_consecutive_buy"] >= 3,
    "inst_consec3": lambda d: d["inst_consecutive_buy"] >= 3,
    "both_consec3": lambda d: (d["foreign_consecutive_buy"] >= 3) & (d["inst_consecutive_buy"] >= 3),
    "supply_div": lambda d: d["supply_divergence"] > 0,
    "accum_eff": lambda d: d["accumulation_efficiency"] > 0,
    "foreign_streak3": lambda d: d["foreign_net_streak"] >= 3,
    "inst_streak3": lambda d: d["inst_net_streak"] >= 3,
    "foreign_volcfm": lambda d: d["foreign_vol_confirm"] > 0,
    "inst_volcfm": lambda d: d["inst_vol_confirm"] > 0,
    "pension_top": lambda d: d["pension_top_buyer"] > 0,
    "trix_golden": lambda d: d["trix_golden_cross"] > 0,
    "sar_rev": lambda d: d["sar_reversal_up"] > 0,
    "stoch_golden": lambda d: d["stoch_slow_golden"] > 0,
    "dyn_rsi": lambda d: d["dynamic_rsi_signal"] > 0,
    "rsi_rising": lambda d: d["rsi_rising"] > 0,
    "is_bullish": lambda d: d["is_bullish"] > 0,
    "vol_surge2": lambda d: d["volume_surge_ratio"] >= 2,
    "macro_fav": lambda d: d["macro_favorable"] > 0,
    "short_cover": lambda d: d["short_cover_signal"] > 0,
    "smart_z1": lambda d: d["smart_z"] > 1,
}


def collect(dfs, s, e, keys, hold, combine="single"):
    """keys 조합의 진입 신호 → D+hold 수익 리스트.
    combine: 'single'=keys[0], 'AND'=모두 만족, 'OR'=하나 이상, 'majority'=과반.
    """
    out = []
    for _code, df in dfs:
        d = df[(df.index >= s) & (df.index <= e)]
        if len(d) < hold + 2:
            continue
        nextopen = d["open"].shift(-1)
        exitclose = d["close"].shift(-hold)
        fwd = exitclose / nextopen - 1 - COST
        fwd = fwd.where((nextopen > 0) & (exitclose > 0))  # 거래정지/0가 제거
        fwd = fwd.replace([np.inf, -np.inf], np.nan)
        tvok = d["trading_value"] >= MIN_TV if "trading_value" in d.columns else pd.Series(True, index=d.index)
        masks = []
        ok = True
        for k in keys:
            try:
                masks.append(SIGNALS[k](d).fillna(False))
            except Exception:
                ok = False
                break
        if not ok or not masks:
            continue
        if combine == "AND":
            sig = masks[0]
            for m in masks[1:]:
                sig = sig & m
        elif combine == "OR":
            sig = masks[0]
            for m in masks[1:]:
                sig = sig | m
        elif combine == "majority":
            sig = sum(m.astype(int) for m in masks) >= (len(masks) // 2 + 1)
        else:  # single
            sig = masks[0]
        sig = sig & tvok & fwd.notna()
        out.extend(fwd[sig].tolist())
    return out


def stats(rets):
    a = np.array(rets)
    n = len(a)
    if n == 0:
        return (0, 0.0, 0.0, 0.0)
    win = a[a > 0]
    loss = a[a <= 0]
    pf = win.sum() / (-loss.sum()) if loss.sum() < 0 else 99.0
    return (n, len(win) / n * 100, a.mean() * 100, pf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--stage", choices=["single", "pairs", "triple", "live"], default="single")
    args = ap.parse_args()

    files = glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f).sort_index()
            if len(df) >= args.hold + 5:
                dfs.append((Path(f).stem, df))
        except Exception:
            pass
    print(f"종목 {len(dfs)}개 / D+{args.hold} 보유 / 거래대금≥10억 / 비용 0.5% / entry=다음날시가\n")

    S, E = "2023-06-01", "2026-05-29"
    if args.stage == "single":
        rows = []
        for k in SIGNALS:
            n, wr, avg, pf = stats(collect(dfs, S, E, [k], args.hold, "single"))
            rows.append((k, n, wr, avg, pf))
        rows.sort(key=lambda x: -x[4])
        print(f"=== 단일 엔진 신호 raw edge (전체 23.6~26.5, PF순) ===")
        print(f'{"신호":<18}{"거래":>8}{"승률":>7}{"평균":>9}{"PF":>7}')
        for k, n, wr, avg, pf in rows:
            print(f"{k:<18}{n:>8}{wr:>6.0f}%{avg:>+8.2f}%{pf:>7.2f}")
        print("\n→ PF>1 + 거래수 충분한 신호 = 조합 재료. 다음: --stage pairs")
    elif args.stage == "pairs":
        # 단일 PF 상위 8개로 2개 AND 조합
        base = []
        for k in SIGNALS:
            n, wr, avg, pf = stats(collect(dfs, S, E, [k], args.hold, "single"))
            if n >= 300:
                base.append((k, pf))
        base.sort(key=lambda x: -x[1])
        top = [k for k, _ in base[:8]]
        print(f"=== 2개 AND 조합 (단일 PF 상위 8: {top}) ===")
        print(f'{"조합":<32}{"거래":>7}{"승률":>7}{"평균":>9}{"PF":>7}')
        combos = []
        for a, b in itertools.combinations(top, 2):
            n, wr, avg, pf = stats(collect(dfs, S, E, [a, b], args.hold, "AND"))
            if n >= 100:
                combos.append((f"{a}+{b}", n, wr, avg, pf))
        combos.sort(key=lambda x: -x[4])
        for name, n, wr, avg, pf in combos[:20]:
            print(f"{name:<32}{n:>7}{wr:>6.0f}%{avg:>+8.2f}%{pf:>7.2f}")
    elif args.stage == "triple":  # short_cover+supply_div 베이스 + 제3신호 + 강건성
        base = ["short_cover", "supply_div"]
        print(f"=== short_cover+supply_div + 제3신호 AND (D+{args.hold}, PF순) ===")
        print(f'{"조합":<34}{"거래":>7}{"승률":>7}{"평균":>9}{"PF":>7}')
        combos = []
        for k in SIGNALS:
            if k in base:
                continue
            n, wr, avg, pf = stats(collect(dfs, S, E, base + [k], args.hold, "AND"))
            if n >= 80:
                combos.append((f"sc+sd+{k}", n, wr, avg, pf))
        combos.sort(key=lambda x: -x[4])
        for name, n, wr, avg, pf in combos[:15]:
            print(f"{name:<34}{n:>7}{wr:>6.0f}%{avg:>+8.2f}%{pf:>7.2f}")
        print(f"\n=== short_cover+supply_div 강건성 (구간 × 보유기간, 워크포워드) ===")
        for hold in [5, 10, 20]:
            for s, e, lbl in [("2023-06-01", "2024-12-31", "23~24약세"), ("2025-01-01", "2026-05-29", "25~26강세")]:
                n, wr, avg, pf = stats(collect(dfs, s, e, base, hold, "AND"))
                print(f"  D+{hold:<2} {lbl:<10} 거래{n:>6} 승률{wr:>3.0f}% 평균{avg:>+6.2f}% PF{pf:>5.2f}")
    else:  # live: 공매도(short_cover) 제외 + 25~26강세 실전 가능 조합 탐색
        s2, e2 = "2025-01-01", "2026-05-29"
        keys = [k for k in SIGNALS if k != "short_cover"]  # 공매도 데이터 단종 제외
        singles = []
        for k in keys:
            n, wr, avg, pf = stats(collect(dfs, s2, e2, [k], args.hold, "single"))
            if n >= 200:
                singles.append((k, n, pf))
        singles.sort(key=lambda x: -x[2])
        print(f"=== 25~26강세 단일 신호 (short_cover 제외=공매도 단종, D+{args.hold}, PF순) ===")
        for k, n, pf in singles[:10]:
            print(f"  {k:<16} 거래{n:>6} PF{pf:>5.2f}")
        topk = [k for k, _, _ in singles[:8]]
        print(f"\n=== 25~26강세 2개 AND 조합 (단일 상위 {len(topk)}개) ===")
        print(f'{"조합":<32}{"거래":>7}{"승률":>7}{"평균":>9}{"PF":>7}')
        combos = []
        for a, b in itertools.combinations(topk, 2):
            n, wr, avg, pf = stats(collect(dfs, s2, e2, [a, b], args.hold, "AND"))
            if n >= 60:
                combos.append((f"{a}+{b}", n, wr, avg, pf))
        combos.sort(key=lambda x: -x[4])
        for name, n, wr, avg, pf in combos[:15]:
            print(f"{name:<32}{n:>7}{wr:>6.0f}%{avg:>+8.2f}%{pf:>7.2f}")
    print("\n★ 생존자편향(현재 생존종목만). PASS 전까지 가설. 사후 최적화 금지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
