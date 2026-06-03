"""거시 조기신호 선행성 적대검증 (read-only / shadow).

사장님 ② (2026-06-03): 거시 신호(VIX/장단기역전/달러급등/금리급등/HY)가
self C60(레버 ETF 자체 60선)의 약세전환보다 "먼저" 2022형 시나리오 B를 켰는가?

어제 regime_obs_adversarial 방법론 그대로:
  base_rate    : 상시점등(노이즈) 여부 — 외국인은 0.63으로 NOISE 탈락했음
  precision@K  : 경고 rising edge 후 K거래일 내 self C60 약세전환 발생 비율
  honest_lead  : 약세전환 전 W거래일 내 rising edge까지의 거리(선행 거래일)

★기간 설계: 2022 약세장 '안'만 보면 VIX·역전이 내내 켜져 base_rate가 1에 가까워
   선행성이 안 보인다. 2021 강세 끝물(2021-06)부터 봐서 '약세 진입 전에 켜졌나'를 측정.

사용: python -u -X utf8 scripts/research/macro_leadlag_adversarial.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from scripts.research.split_buy_backtest import prepare
from src.etf.macro_signals import WARN_LABEL, attach_warnings, load_macro

TICKERS = {"SOXL": "반도체3x", "TQQQ": "나스닥3x"}
START, END = "2021-06-01", "2023-06-01"  # 강세 끝물 + 2022 약세 + 회복 시작
PRECISION_HORIZON = 10   # rising edge 후 K거래일 내 self C60 약세전환?
RECALL_WINDOW = 30       # 약세전환 전 W거래일 내 rising edge?


def self_c60_bear_switches(ticker: str, idx: list) -> list[int]:
    """self C60(close>ma60, look-ahead0) BULL->BEAR 전환 인덱스 (idx 기준)."""
    _, bull = prepare(ticker, 1.0)
    bsub = bull.reindex(idx).fillna(False).astype(bool).tolist()
    return [i for i in range(1, len(bsub)) if (not bsub[i]) and bsub[i - 1]]


def analyze(ticker: str, name: str, macro_warn: pd.DataFrame) -> dict:
    lp, bull = prepare(ticker, 1.0)
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    idx = [d for d in bull.index if s <= d <= e]
    switches = self_c60_bear_switches(ticker, idx)

    # 거시 경고를 ETF 영업일에 정렬 (ffill)
    warn = macro_warn.reindex(idx, method="ffill")
    n = len(idx)

    out = {"ticker": ticker, "name": name, "rows": n, "bear_switches": len(switches), "obs": {}}
    for col, label in WARN_LABEL.items():
        if col not in warn:
            continue
        w = warn[col].fillna(False).astype(bool).reset_index(drop=True)
        base_rate = float(w.mean())
        rising = (w & (~w.shift(1, fill_value=False)))
        rising_idx = rising[rising].index.tolist()

        # precision: rising edge 후 K거래일 내 약세전환
        if rising_idx:
            hits = sum(
                1 for ri in rising_idx
                if set(range(ri, min(n, ri + PRECISION_HORIZON + 1))) & set(switches)
            )
            precision = round(hits / len(rising_idx), 3)
        else:
            precision = None

        # recall + honest lead: 전환 전 W거래일 내 rising edge
        leads, covered = [], 0
        for sw in switches:
            lo = max(0, sw - RECALL_WINDOW)
            edges = [ri for ri in rising_idx if lo <= ri <= sw]
            if edges:
                covered += 1
                leads.append(sw - edges[0])
        recall = round(covered / len(switches), 3) if switches else None
        honest_lead = round(sum(leads) / len(leads), 2) if leads else None

        out["obs"][label] = {
            "base_rate": round(base_rate, 3),
            "edges": len(rising_idx),
            "precision": precision,
            "recall": recall,
            "honest_lead": honest_lead,
        }
    return out


def verdict(o: dict) -> str:
    br, prec = o["base_rate"], o["precision"]
    if br is None:
        return "판정불가"
    if br >= 0.40:
        return "노이즈(상시점등) — 약세확인일뿐 선행X"
    if prec is not None and prec >= 0.5 and br < 0.25:
        return "★조기경보 후보 — 승격검토"
    return "약함 — 로그유지"


def main() -> None:
    print("=" * 76)
    print("거시 조기신호 선행성 적대검증 (self C60 약세전환 vs 거시 경고)")
    print(f"  기간 {START}~{END} (강세끝물+2022약세+회복) / "
          f"precision: edge후 {PRECISION_HORIZON}일내 전환 / recall: 전환전 {RECALL_WINDOW}일내 edge")
    print("=" * 76)
    macro = attach_warnings(load_macro(start="2021-01-01", end=END, include_hy=True))
    for ticker, name in TICKERS.items():
        res = analyze(ticker, name, macro)
        print(f"\n[{ticker}] {name}  ({res['rows']}일, self C60 약세전환 {res['bear_switches']}회)")
        for label, o in res["obs"].items():
            print(f"  {label:16s} base={o['base_rate']} edges={o['edges']} "
                  f"prec={o['precision']} recall={o['recall']} lead={o['honest_lead']}")
            print(f"  {'':16s} → {verdict(o)}")
    print("\n" + "=" * 76)
    print("base 높음=약세장 내내 켜짐(선행X) / prec 높고 base 낮음=진짜 조기경보")
    print("HY스프레드는 2023-06~만 → 2022 검증 불가(현재 모니터링용)")
    print("=" * 76)


if __name__ == "__main__":
    main()
