# -*- coding: utf-8 -*-
"""상한가 수급선행 신호화 백테스트 — 판정: ❌ 기각 (재현율≠정밀도 + 커버리지 편향).

사후학습 트랙 발견 "상한가의 53%가 D-1 수급선행"(재현율)의 역방향 검증:
  streak 보유 종목을 익일 시가에 사면 초과수익/상한가 적중이 나오는가?

★결과 (2026-07-07, 상세: data/backtest/limit_up_precursor_report.md):
1. 상한가 정밀도 = 기저율(0.3% vs 0.35%) — 수급선행으로 상한가 예측 불가.
2. D+5 초과수익 헤드라인(+0.5~0.8%p, t=5)은 **커버리지 편향**: 2026-03 KIS 백필로
   수급적재 종목일 비율 25%→100% 급증. 수급적재 종목만 대조군으로 쓰면
   외인3~5 +0.24%p(t=2.3)로 축소, 쌍끌이3은 조정장 -1.21%p(t=-2.9) 역효과.
3. 왕복비용 0.35% > 초과 0.2%p — 단독 신호화 가치 없음. 가점 승격도 부적합.
→ 사후학습 트랙은 '놓친 상한가 원인 태깅'(본래 목적)으로만 유지.

★방법론 교훈: 대조군은 반드시 데이터 커버리지 동질 집단으로 (커버리지 변화 시점 확인 필수).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
START = "2025-04-21"   # KIS 11세분 수급 신뢰 구간
MIN_TVAL = 5e8         # 당일 거래대금 5억 미만 제외(실행가능성)
COST_RT = 0.35         # 왕복 비용(수수료+거래세+슬리피지, %)
COLS = ["open", "close", "volume", "foreign_consecutive_buy",
        "inst_consecutive_buy", "foreign_net_5d", "inst_net_5d"]


def build_panel() -> pd.DataFrame:
    rows = []
    for p in sorted(PROCESSED.glob("*.parquet")):
        try:
            df = pd.read_parquet(p, columns=COLS)
        except Exception:
            continue
        df = df[df.index >= START]
        if len(df) < 15:
            continue
        df = df.copy()
        o_next = df["open"].shift(-1)
        df["ret_d1"] = (df["close"].shift(-1) / o_next - 1) * 100
        df["ret_d5"] = (df["close"].shift(-5) / o_next - 1) * 100
        df["next_chg"] = (df["close"].shift(-1) / df["close"] - 1) * 100
        df["code"] = p.stem
        tval = df["close"] * df["volume"]  # trading_value 컬럼은 결측(0) 잦음 — 직접 계산
        df = df[(tval >= MIN_TVAL) & (o_next > 0)]
        df.index.name = "date"
        rows.append(df.reset_index()[["date", "code", "foreign_consecutive_buy",
                                      "inst_consecutive_buy", "foreign_net_5d",
                                      "inst_net_5d", "ret_d1", "ret_d5", "next_chg"]])
    return pd.concat(rows, ignore_index=True).dropna(subset=["ret_d1"])


def excess_t(sub: pd.DataFrame, ctrl: pd.DataFrame, col: str):
    """일별 (신호평균 - 대조평균) 시계열 → t. 시장 드리프트·횡단면 중복 제거."""
    ex = (sub.groupby("date")[col].mean() - ctrl.groupby("date")[col].mean()).dropna()
    if len(ex) < 3:
        return (np.nan, np.nan, len(ex))
    m, s = ex.mean(), ex.std(ddof=1)
    return (m, m / (s / np.sqrt(len(ex))) if s > 0 else np.nan, len(ex))


def main() -> int:
    panel = build_panel()
    f = panel["foreign_consecutive_buy"].fillna(0)
    i = panel["inst_consecutive_buy"].fillna(0)
    # ★커버리지 프록시: 수급 5d 합이 0이 아니면 '수급 적재됨' — 대조군 동질화의 핵심
    covered = (panel["foreign_net_5d"].fillna(0).abs()
               + panel["inst_net_5d"].fillna(0).abs()) > 0
    print(f"패널: {panel['date'].min().date()} ~ {panel['date'].max().date()} | "
          f"{len(panel):,} 종목일 | 커버리지 {covered.mean()*100:.0f}%")
    cov_m = panel.assign(ym=panel["date"].dt.to_period("M"), c=covered).groupby("ym")["c"].mean()
    print("월별 커버리지:", {str(k): f"{v*100:.0f}%" for k, v in cov_m.items()})

    base_lu = (panel["next_chg"] >= 25).mean() * 100
    sigs = {
        "쌍끌이1 (f>0&i>0)": (f > 0) & (i > 0),
        "외인3 (f>=3)": f >= 3,
        "기관3 (i>=3)": i >= 3,
        "쌍끌이3 (f>=3&i>=3)": (f >= 3) & (i >= 3),
        "외인 3~5 밴드": (f >= 3) & (f <= 5),
        "과매집 (f>=8)": f >= 8,
    }

    print(f"\n상한가 정밀도 (기저 {base_lu:.2f}%):")
    for name, mask in sigs.items():
        sub = panel[mask]
        print(f"  {name:20s} n={len(sub):6,}  P(익일+25%)={ (sub['next_chg'] >= 25).mean()*100:.2f}%")

    print("\n★커버드 대조군 D+5 초과수익 (구간 분해):")
    windows = [("2025-04-01", "2026-02-28", "강세(25/04~26/02)"),
               ("2026-03-01", "2026-06-30", "조정(26/03~06)")]
    for name, mask in sigs.items():
        row = f"  {name:20s}"
        for a, b, tag in windows:
            w = (panel["date"] >= a) & (panel["date"] <= b)
            sub, ctrl = panel[mask & w & covered], panel[w & covered]
            m, t, _ = excess_t(sub, ctrl, "ret_d5")
            row += f" | {tag[:2]} {m:+5.2f}%p t={t:+5.2f} (n={len(sub):,})"
        print(row)

    print(f"\n※ 왕복비용 가정 {COST_RT}% — 최대 초과(+0.2~0.8%p/5d)로도 단독 트레이딩 비경제.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
