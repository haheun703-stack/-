"""장전 브리핑 → 레버리지/인버스 매매 검증 (사장님 6/2 아이디어).

브리핑 상승확률(kospi_pred_model: 전일 US soxx/vix/spy/qqq → KOSPI 방향 로지스틱) 재현.
적중률 기보고: 전체62%/상승69%/하락53%(하락=동전). 그걸로 레버리지/인버스 매매:
  A 레버리지 buy&hold (기준, 어제 +1069%)
  B prob>thr 레버리지 / else 현금 (하락 회피)
  C prob>thr 레버리지 / else 인버스 (풀 스위칭)
레버리지=122630, 인버스=114800. ★2025.6~2026.5. KOSPI 대비. 전환비용 0.1%.
look-ahead 0 (전일 US로 당일 KOSPI 매매).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pykrx import stock

SWITCH = 0.001


def briefing_prob():
    m = json.load(open(PROJECT_ROOT / "data" / "us_market" / "kospi_pred_model.json", encoding="utf-8"))
    coef = np.array(m["coef"]); mean = np.array(m["scaler_mean"]); scale = np.array(m["scaler_scale"]); b = m["intercept"]
    df = pd.read_parquet(PROJECT_ROOT / "data" / "us_market" / "us_daily.parquet").sort_index()
    f = pd.DataFrame(index=df.index)
    f["soxx"] = df["soxx_ret_1d"] * 100
    f["vix_chg"] = df["vix_close"].pct_change() * 100
    f["vix_level"] = df["vix_close"]
    f["spy"] = df["spy_ret_1d"] * 100
    f["qqq"] = df["qqq_ret_1d"] * 100
    f["stoxx50"] = mean[5]  # us_daily에 없음 → mean(scaled 0)
    f = f.dropna()
    z = (f.values - mean) / scale
    logit = z @ coef + b
    prob = 1 / (1 + np.exp(-logit))
    return pd.Series(prob, index=f.index)  # US date 기준 (다음 KOSPI일 예측)


def main() -> int:
    prob = briefing_prob()
    k = pd.read_csv(PROJECT_ROOT / "data" / "kospi_index.csv")
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"]); k = k.set_index("date").sort_index()
    S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-05-29")

    # 레버리지/인버스 ETF 일별
    def etf(code):
        try:
            df = stock.get_market_ohlcv("20250601", "20260529", code)
            return df["종가"].astype(float)
        except Exception:
            return None
    lev = etf("122630"); inv = etf("114800")
    if lev is None or inv is None:
        print("ETF 수신 실패"); return 1
    lev.index = pd.to_datetime(lev.index); inv.index = pd.to_datetime(inv.index)

    # KOSPI 거래일에 "전일 US prob" 매핑 (merge_asof: KOSPI일 직전 US date)
    kdays = [d for d in k.index if S <= d <= E]
    pdf = pd.DataFrame({"usdate": prob.index, "prob": prob.values}).sort_values("usdate")
    kmap = pd.merge_asof(pd.DataFrame({"kdate": kdays}), pdf, left_on="kdate", right_on="usdate", direction="backward")
    kmap = kmap.set_index("kdate")["prob"]

    lev_r = lev.pct_change().fillna(0); inv_r = inv.pct_change().fillna(0)

    def sim(mode, thr):
        v = 1.0; prev = None; curve = []
        for d in kdays:
            p = kmap.get(d, 0.5)
            if mode == "A":
                cur = "lev"
            elif mode == "B":
                cur = "lev" if p > thr else "cash"
            else:
                cur = "lev" if p > thr else "inv"
            r = lev_r.get(d, 0) if cur == "lev" else (inv_r.get(d, 0) if cur == "inv" else 0)
            v *= (1 + r)
            if prev is not None and cur != prev:
                v *= (1 - SWITCH)
            curve.append(v); prev = cur
        eq = pd.Series(curve, index=kdays)
        ret = (eq.iloc[-1] - 1) * 100
        dr = eq.pct_change().dropna(); sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        return ret, mdd, sh

    ks = k[(k.index >= S) & (k.index <= E)]["close"]; kos = (ks.iloc[-1] / ks.iloc[0] - 1) * 100
    print(f"브리핑 → 레버리지/인버스 매매 — 2025.6~2026.5 (KOSPI {kos:+.0f}%)")
    print(f"브리핑 상승확률 분포: 평균{prob[(prob.index>=S)].mean():.2f} / >0.5 비율 {(prob[prob.index>=S]>0.5).mean()*100:.0f}%\n")
    print(f'{"전략":<28}{"수익":>9}{"MDD":>7}{"샤프":>6}')
    rA = sim("A", 0); print(f'{"A 레버리지 buy&hold":<28}{rA[0]:>+8.0f}%{rA[1]:>6.0f}%{rA[2]:>6.2f}')
    for thr in (0.5, 0.55, 0.6):
        rB = sim("B", thr); print(f'{f"B 상승예측만 레버(thr{thr})":<28}{rB[0]:>+8.0f}%{rB[1]:>6.0f}%{rB[2]:>6.2f}')
    for thr in (0.5, 0.55, 0.6):
        rC = sim("C", thr); print(f'{f"C 레버/인버 스위칭(thr{thr})":<28}{rC[0]:>+8.0f}%{rC[1]:>6.0f}%{rC[2]:>6.2f}')
    print("\n★ B(하락회피)가 A(+1069%급) 넘으면 = 브리핑으로 레버 타이밍 가치. C(인버스)가 B보다 나으면 인버스도 가치(단 하락53% 동전이라 의문).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
