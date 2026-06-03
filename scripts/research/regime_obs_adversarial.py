"""보조 관측 선행성 적대적 재검증 (read-only / shadow).

목적: regime_monitor가 보고한 "보조 관측 lead 평균"이 진짜 선행 신호인지,
아니면 상시 점등(노이즈)으로 인한 착시인지 가린다. [[feedback_adversarial_self_validation]]

단순 lead 평균의 함정:
  보조 경고가 거의 매일 켜져 있으면(상시 점등), 전환 전 윈도우 시작부터 이미
  경고 상태 → lead = 윈도우 최대값으로 고정 → "먼저 경고했다"는 착시.

진짜 선행 신호의 조건:
  1. base rate 낮음        — 평소엔 조용해야 함 (상시 점등 X)
  2. precision 높음        — 경고가 새로 켜지면(rising edge) 곧 실제 전환이 와야 함
  3. honest lead 합리적    — rising edge 기준으로만 잰 선행 거래일

사용: python -u -X utf8 scripts/research/regime_obs_adversarial.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SHADOW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "shadow"
TICKERS = {
    "488080": "반도체레버",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}
WARN_COLS = {
    "vol_cluster_warn": "변동성클러스터",
    "kospi_warn": "KOSPI60이탈",
    "foreign_warn": "외국인순매도",
}
PRECISION_HORIZON = 5   # 경고 rising edge 후 K거래일 내 전환 발생?
RECALL_WINDOW = 20      # 전환 전 W거래일 내 rising edge 있었나?


def load_ledger(ticker: str) -> pd.DataFrame:
    path = SHADOW_DIR / f"{ticker}_regime_ledger.json"
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.load(open(path, encoding="utf-8"))).reset_index(drop=True)
    return df


def analyze(ticker: str, name: str) -> dict:
    df = load_ledger(ticker)
    if df.empty:
        return {"ticker": ticker, "error": "no ledger"}

    n = len(df)
    # 실제 약세 전환 인덱스 (BULL -> BEAR_TRANSITION)
    bear_switches = df.index[
        (df["regime"] == "BEAR_TRANSITION") & (df["regime_change"])
    ].tolist()

    out = {
        "ticker": ticker,
        "name": name,
        "rows": n,
        "bear_switches": len(bear_switches),
        "observations": {},
    }

    for col, label in WARN_COLS.items():
        warn = df[col].fillna(False).astype(bool)

        # 외국인은 데이터 가용 구간만 base rate 산정 (NaN/없는 구간 제외)
        if col == "foreign_warn":
            avail = df["foreign_net"].notna()
            denom = int(avail.sum())
            base_rate = float(warn[avail].mean()) if denom else None
        else:
            denom = n
            base_rate = float(warn.mean())

        # rising edge = 평소 꺼져 있다가 새로 켜진 날
        rising = warn & (~warn.shift(1, fill_value=False))
        rising_idx = df.index[rising].tolist()

        # precision: rising edge 후 K거래일 내 실제 전환 발생 비율
        if rising_idx:
            hits = 0
            for ri in rising_idx:
                horizon = set(range(ri, min(n, ri + PRECISION_HORIZON + 1)))
                if horizon & set(bear_switches):
                    hits += 1
            precision = round(hits / len(rising_idx), 3)
        else:
            precision = None

        # recall + honest lead: 전환 전 W거래일 내 rising edge가 있던 전환 비율
        honest_leads = []
        covered = 0
        for sw in bear_switches:
            lo = max(0, sw - RECALL_WINDOW)
            window_edges = [ri for ri in rising_idx if lo <= ri <= sw]
            if window_edges:
                covered += 1
                honest_leads.append(sw - window_edges[0])  # 가장 이른 edge 기준
        recall = round(covered / len(bear_switches), 3) if bear_switches else None
        honest_lead = round(sum(honest_leads) / len(honest_leads), 2) if honest_leads else None

        out["observations"][label] = {
            "base_rate": round(base_rate, 3) if base_rate is not None else None,
            "base_denom": denom,
            "rising_edges": len(rising_idx),
            "precision@%d" % PRECISION_HORIZON: precision,
            "recall@%d" % RECALL_WINDOW: recall,
            "honest_lead_days": honest_lead,
        }
    return out


def verdict(obs: dict) -> str:
    """노이즈 vs 신호 1차 판정 (보수적)."""
    br = obs["base_rate"]
    prec = obs.get("precision@%d" % PRECISION_HORIZON)
    if br is None:
        return "판정불가(데이터부족)"
    if br >= 0.40:
        return "노이즈(상시점등) — gate 부적합"
    if prec is not None and prec >= 0.5 and br < 0.25:
        return "신호 후보 — 추적가치"
    return "약함 — 로그 유지"


def main() -> None:
    print("=" * 72)
    print("보조 관측 선행성 적대적 재검증 (노이즈 vs 신호)")
    print(f"  precision: rising edge 후 {PRECISION_HORIZON}거래일 내 전환 / "
          f"recall: 전환 전 {RECALL_WINDOW}거래일 내 edge")
    print("=" * 72)
    for ticker, name in TICKERS.items():
        res = analyze(ticker, name)
        if res.get("error"):
            print(f"\n[{ticker}] {res['error']}")
            continue
        print(f"\n[{ticker}] {name}  ({res['rows']}일, 약세전환 {res['bear_switches']}회)")
        for label, o in res["observations"].items():
            print(f"  {label:14s} base_rate={o['base_rate']} (n={o['base_denom']}) "
                  f"edges={o['rising_edges']} "
                  f"precision={o['precision@%d' % PRECISION_HORIZON]} "
                  f"recall={o['recall@%d' % RECALL_WINDOW]} "
                  f"lead={o['honest_lead_days']}")
            print(f"  {'':14s} → {verdict(o)}")
    print("\n" + "=" * 72)
    print("base_rate 높음=상시점등(착시) / precision 높음=경고가 실제 전환을 예고")
    print("=" * 72)


if __name__ == "__main__":
    main()
