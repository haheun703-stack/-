"""레버리지 물타기(DCA) vs C60 검증 — "안 팔고 물타면 되지 않나" 방어 자료.

사장님 (2026-06-03): 단순 인라인으로 흘릴 게 아니라 기준 자료로 남긴다.
"안 팔고 무한 물타기"는 V자 강세장엔 좋아 보이나, 2022형 추세하락에선 계좌를
크게 망가뜨린다. 앞으로 물타기 유혹을 받을 때마다 막아주는 방어 자산.

비교 전략 (SOXL 반도체3x):
  A 일괄 buyhold       — 첫날 전액, 손절 없음 (기준선)
  B 일괄 + C60         — 전액 + 60선 이탈 청산
  C 조정분할 + C60     — 분할 진입 + 60선 청산 (우리 확정 룰)
  무한 물타기(DCA, 손절X) — 직전고점 -10/-20/-30/-40/-50%마다 추가, 60선 무시·청산 없음

구간: ①강세장 V자(2023~2026, 3월 -85% V자 포함) ②3월 V자 단기 ③2022 추세하락.
look-ahead 0 / 비용 0.1% / 시드 1.0 / read-only / 실주문 0.

사용: python -u -X utf8 scripts/research/leverage_dca_vs_c60.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from scripts.research.split_buy_backtest import COST, _mdd, prepare, sim

ASSETS = Path(__file__).resolve().parent.parent.parent / "docs" / "02-design" / "assets"
TICKER = "SOXL"
DCA_DIPS = [-0.10, -0.20, -0.30, -0.40, -0.50]  # 직전고점 대비 추가매수 트리거
DCA_AMT = 1.0 / 6.0                              # 첫날 + 5단계 = 6분할

PERIODS = [
    ("강세장 V자 2023-2026 (3월 -85% V자 포함)", "2023-01-01", "2026-06-01"),
    ("3월 V자 단기 2026-02~06", "2026-02-01", "2026-06-01"),
    ("2022 추세하락 (약세장)", "2022-01-01", "2022-12-31"),
]
STRATS = [
    ("buyhold", "A 일괄buyhold"),
    ("c60", "B 일괄+C60"),
    ("split", "C 조정분할+C60"),
    ("dca", "무한물타기(손절X)"),
]


def martingale_dca(lev_price: pd.Series, start, end) -> dict:
    """무한 물타기: 직전 고점 대비 dip마다 추가, 손절(60선 청산) 없음, 자금 소진까지."""
    idx = [d for d in lev_price.index if start <= d <= end]
    if not idx:
        return {"final_return_pct": 0.0, "mdd_pct": 0.0, "cash_used": 0.0}
    shares = 0.0
    cash = 1.0
    peak = None
    done: set[int] = set()
    curve: list[float] = []
    for d in idx:
        p = float(lev_price[d])
        if peak is None:  # 첫날 1/6 진입
            shares += DCA_AMT / p
            cash -= DCA_AMT * (1 + COST)
            peak = p
        else:
            peak = max(peak, p)
            for k, dip in enumerate(DCA_DIPS, 1):
                if k not in done and p <= peak * (1 + dip) and cash >= DCA_AMT:
                    shares += DCA_AMT / p
                    cash -= DCA_AMT * (1 + COST)
                    done.add(k)
        curve.append(shares * p + cash)
    return {
        "final_return_pct": round((curve[-1] - 1.0) * 100, 1),
        "mdd_pct": round(_mdd(curve) * 100, 1),
        "cash_used": round(min(1.0 - cash, 1.0), 2),
    }


def run() -> dict:
    lev_price, bull_sig = prepare(TICKER, 1.0)
    out: dict = {"ticker": TICKER, "periods": {}}
    for name, s, e in PERIODS:
        sd, ed = pd.Timestamp(s), pd.Timestamp(e)
        row = {
            "buyhold": sim(lev_price, bull_sig, "A", sd, ed),
            "c60": sim(lev_price, bull_sig, "B", sd, ed),
            "split": sim(lev_price, bull_sig, "C", sd, ed),
            "dca": martingale_dca(lev_price, sd, ed),
        }
        out["periods"][name] = {
            k: {"ret": v["final_return_pct"], "mdd": v["mdd_pct"]} for k, v in row.items()
        }
    return out


def main() -> None:
    result = run()
    print("=" * 78)
    print(f"레버리지 물타기(DCA) vs C60 — {result['ticker']} (안 팔고 물타면 되나?)")
    print("=" * 78)
    for name, row in result["periods"].items():
        print(f"\n[{name}]")
        print(f"  {'전략':18}{'수익%':>10}{'MDD%':>9}")
        for key, label in STRATS:
            r = row[key]
            print(f"  {label:18}{r['ret']:>10}{r['mdd']:>9}")
    print("\n" + "=" * 78)
    print("물타기=V자엔 최고수익이나 MDD 극심(안팜) / 추세하락엔 -68%. C60=추세하락 방패.")
    print("=" * 78)

    ASSETS.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS / "leverage_dca_vs_c60.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {out_path.relative_to(ASSETS.parent.parent.parent)}")


if __name__ == "__main__":
    main()
