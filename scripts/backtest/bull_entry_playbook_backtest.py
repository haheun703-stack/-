# -*- coding: utf-8 -*-
"""BULL 전환 진입 플레이북 백테스트 — 판정: ✅ 채택 (시나리오 v1 ENTRY_WINDOW 사전 검증).

질문: V3b 레짐 BULL 전환일(견고 이벤트 40회, 2019~2026)에 무엇을 사는 게 최선인가.

★결과 (2026-07-07, 상세: data/backtest/bull_entry_playbook_report.md):
1. 전환 이벤트 자체 유효: 유니버스 D+20 +1.49%(t=2.57)·D+40 +3.95%(t=3.20)
2. 종목 = 저PER 하위20%: D+20 초과 +0.90%p(t=3.13)·이벤트 승률 76%·연도 견고(5/6년 양수)
   — 대칭 확인: 고PER 상위20%는 -0.80%p(t=-2.75). FV 워치리스트(저PER 축) 설계 지지.
3. 낙폭과대 잡기 금지(-0.46%p)·모멘텀 상위는 D+40 장기 보조(+1.45%p t=2.25)
4. 진입 지연 비용: T+1 +1.49% → T+3 +1.37%(92% 보존) → T+6 +1.15% — 검증 후 진입도 합리적
한계: 현존 1,166종 생존편향(유니버스 내 상대비교로 완화)·fund_PER 2025 정적 오염(19~24만 사용).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# stdout 래핑은 아래 모듈 import가 1회 수행 — 여기서 중복 래핑 금지(detach 크래시)
from scripts.backtest.brain_bull_relax_backtest import build_features, classify, load_kospi

import numpy as np
import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
MIN_TVAL20 = 1e9


def bull_events() -> list[pd.Timestamp]:
    """V3b BULL 진입 + 직전 5일 이상 non-BULL (플리커 제외)."""
    f = build_features(load_kospi())
    reg = classify(f, f["drv_pct"] < 60)
    reg = reg[reg.index >= "2019-01-01"]
    is_bull = (reg == "BULL").astype(int)
    enter = is_bull.diff() == 1
    return [reg.index[i] for i in range(len(reg))
            if enter.iloc[i] and is_bull.iloc[max(0, i - 6):i].sum() == 0]


def build_event_panel(events: list[pd.Timestamp]) -> pd.DataFrame:
    ev_set = set(events)
    rows = []
    for p in sorted(PROCESSED.glob("*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["open", "close", "volume", "fund_PER"])
        except Exception:
            continue
        if len(df) < 300:
            continue
        df = df.copy()
        o = df["open"].shift(-1)
        for h in (10, 20, 40):
            df[f"ret_d{h}"] = (df["close"].shift(-h) / o - 1) * 100
        df["ret_d20_lag3"] = (df["close"].shift(-20) / df["open"].shift(-3) - 1) * 100
        df["ret_d20_lag6"] = (df["close"].shift(-20) / df["open"].shift(-6) - 1) * 100
        df["mom20"] = df["close"].pct_change(20) * 100
        df["off_high"] = (df["close"] / df["close"].rolling(252).max() - 1) * 100
        df["tval20"] = (df["close"] * df["volume"]).rolling(20).mean()
        sel = df[df.index.isin(ev_set) & (df["tval20"] >= MIN_TVAL20) & (o > 0)]
        if len(sel) == 0:
            continue
        sel = sel.copy()
        sel["code"] = p.stem
        sel.index.name = "date"
        rows.append(sel.reset_index()[["date", "code", "fund_PER", "mom20", "off_high",
                                       "ret_d10", "ret_d20", "ret_d40",
                                       "ret_d20_lag3", "ret_d20_lag6"]])
    return pd.concat(rows, ignore_index=True)


def event_excess(panel: pd.DataFrame, mask: pd.Series, col: str):
    """이벤트별 (팩터군 평균 - 유니버스 평균) → 이벤트 간 mean/t (클러스터)."""
    sig = panel[mask].groupby("date")[col].mean()
    uni = panel.groupby("date")[col].mean()
    ex = (sig - uni).dropna()
    if len(ex) < 3:
        return (np.nan, np.nan, len(ex))
    m, s = ex.mean(), ex.std(ddof=1)
    return (m, m / (s / np.sqrt(len(ex))), len(ex))


def main() -> int:
    events = bull_events()
    print(f"BULL 전환 견고 이벤트: {len(events)}회 ({events[0].date()} ~ {events[-1].date()})")
    panel = build_event_panel(events)
    panel["yr"] = panel["date"].dt.year
    print(f"이벤트 패널: {len(panel):,} 종목·이벤트 | 이벤트당 평균 {len(panel) / len(events):.0f}종")

    for col in ["ret_d10", "ret_d20", "ret_d40"]:
        uni = panel.groupby("date")[col].mean()
        m, s = uni.mean(), uni.std(ddof=1)
        print(f"  유니버스 평균 {col[4:]:>3}: {m:+5.2f}% (이벤트간 t={m / (s / np.sqrt(len(uni))):+.2f}, n={len(uni)})")

    per_ok = (panel["yr"] <= 2024) & (panel["fund_PER"] > 0)
    panel["per_rank"] = panel[per_ok].groupby("date")["fund_PER"].rank(pct=True)
    panel["mom_rank"] = panel.groupby("date")["mom20"].rank(pct=True)
    panel["off_rank"] = panel.groupby("date")["off_high"].rank(pct=True)

    facs = {
        "저PER 하위20% (19~24)": panel["per_rank"] <= 0.2,
        "고PER 상위20% (19~24)": panel["per_rank"] >= 0.8,
        "모멘텀 상위20%": panel["mom_rank"] >= 0.8,
        "모멘텀 하위20%": panel["mom_rank"] <= 0.2,
        "고점근접 상위20%": panel["off_rank"] >= 0.8,
        "낙폭과대 하위20%": panel["off_rank"] <= 0.2,
    }
    print("\n팩터별 초과수익 (vs 이벤트 유니버스 평균, 이벤트 클러스터 t):")
    for name, mask in facs.items():
        out = f"  {name:22s}"
        for col in ["ret_d10", "ret_d20", "ret_d40"]:
            m, t, n = event_excess(panel, mask, col)
            out += f" | {col[4:]:>3} {m:+5.2f}%p t={t:+5.2f}"
        print(out + f" (ev n={n})")

    print("\n연도별 (저PER 하위20%, D+20 초과):")
    for yr, g in panel[panel["yr"] <= 2024].groupby("yr"):
        m, t, n = event_excess(g, g["per_rank"] <= 0.2, "ret_d20")
        print(f"  {yr}: {m:+5.2f}%p (t={t:+.2f}, 이벤트 {n})")
    sig = panel[panel["per_rank"] <= 0.2].groupby("date")["ret_d20"].mean()
    uni = panel.groupby("date")["ret_d20"].mean()
    ex = (sig - uni).dropna()
    print(f"  이벤트 승률: {(ex > 0).mean() * 100:.0f}% ({(ex > 0).sum()}/{len(ex)})")

    print("\n진입 지연 비용 (유니버스 평균, D+20 종가 청산 기준):")
    for col, tag in [("ret_d20", "T+1 시가"), ("ret_d20_lag3", "T+3 시가"), ("ret_d20_lag6", "T+6 시가")]:
        uni = panel.groupby("date")[col].mean().dropna()
        print(f"  {tag}: {uni.mean():+5.2f}% (n={len(uni)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
