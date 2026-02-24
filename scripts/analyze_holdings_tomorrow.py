"""보유 종목 내일 전망 분석 스크립트."""
import sys, json, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pykrx import stock as pykrx_stock

holdings = [
    {"name": "GS건설",    "ticker": "006360", "qty": 50,  "price": 23250, "pnl_pct": 9.2},
    {"name": "현대제철",   "ticker": "004020", "qty": 30,  "price": 36850, "pnl_pct": 7.6},
    {"name": "삼성전자",   "ticker": "005930", "qty": 36,  "price": 200000,"pnl_pct": 6.8},
    {"name": "S-Oil",     "ticker": "010950", "qty": 27,  "price": 114100,"pnl_pct": 5.7},
    {"name": "셀트리온",   "ticker": "068270", "qty": 13,  "price": 248500,"pnl_pct": 2.5},
    {"name": "카카오뱅크", "ticker": "323410", "qty": 73,  "price": 28450, "pnl_pct": 2.0},
    {"name": "클로봇",    "ticker": "466100", "qty": 48,  "price": 69000, "pnl_pct": -0.8},
    {"name": "아이지넷",   "ticker": "462980", "qty": 700, "price": 2020,  "pnl_pct": -1.4},
    {"name": "대한전선",   "ticker": "001440", "qty": 60,  "price": 33900, "pnl_pct": -3.0},
    {"name": "스맥",      "ticker": "099440", "qty": 550, "price": 6570,  "pnl_pct": -3.5},
    {"name": "한화엔진",   "ticker": "082740", "qty": 35,  "price": 55400, "pnl_pct": -4.3},
]


def calc_indicators(ticker):
    df = pykrx_stock.get_market_ohlcv_by_date("20250224", "20260224", ticker)
    if len(df) < 60:
        return None
    df = df.sort_index()
    c = df["종가"].astype(float)
    v = df["거래량"].astype(float)
    high = df["고가"].astype(float)
    low = df["저가"].astype(float)

    # RSI 14
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # BB%
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_pct = (c - bb_lower) / (bb_upper - bb_lower) * 100

    # MA
    ma5 = c.rolling(5).mean()
    ma60 = c.rolling(60).mean()

    # 거래량 비율
    vol_avg = v.rolling(20).mean()
    vol_ratio = v.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1

    # 수익률
    ret5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) >= 6 else 0
    ret20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) >= 21 else 0

    # TRIX
    ema12 = c.ewm(span=12).mean()
    ema12_2 = ema12.ewm(span=12).mean()
    ema12_3 = ema12_2.ewm(span=12).mean()
    trix = ema12_3.pct_change() * 100
    trix_bull = bool(trix.iloc[-1] > trix.iloc[-2])

    # Stoch
    low14 = c.rolling(14).min()
    high14 = c.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14) * 100
    stoch_d = stoch_k.rolling(3).mean()
    stoch_gx = bool(stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-2] <= stoch_d.iloc[-2])

    # ADX
    tr = pd.concat([high - low, abs(high - c.shift(1)), abs(low - c.shift(1))], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_dm = ((high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)).clip(lower=0)
    minus_dm = ((low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = dx.rolling(14).mean()

    return {
        "rsi": round(float(rsi.iloc[-1]), 1),
        "bb_pct": round(float(bb_pct.iloc[-1]), 1),
        "adx": round(float(adx.iloc[-1]), 1),
        "above_ma20": bool(c.iloc[-1] > ma20.iloc[-1]),
        "above_ma60": bool(c.iloc[-1] > ma60.iloc[-1]),
        "ma5_above_ma20": bool(ma5.iloc[-1] > ma20.iloc[-1]),
        "vol_ratio": round(float(vol_ratio), 2),
        "ret5": round(float(ret5), 1),
        "ret20": round(float(ret20), 1),
        "trix_bull": trix_bull,
        "stoch_gx": stoch_gx,
        "stoch_k": round(float(stoch_k.iloc[-1]), 1),
    }


def judge(h, ind):
    score = 0
    signals = []

    # RSI
    if ind["rsi"] < 30:
        score += 3; signals.append(f'RSI과매도({ind["rsi"]})')
    elif ind["rsi"] < 45:
        score += 2; signals.append(f'RSI적정({ind["rsi"]})')
    elif ind["rsi"] < 60:
        score += 1; signals.append(f'RSI중립({ind["rsi"]})')
    elif ind["rsi"] < 70:
        score += 0; signals.append(f'RSI높음({ind["rsi"]})')
    else:
        score -= 2; signals.append(f'RSI과열({ind["rsi"]})')

    # BB%
    if ind["bb_pct"] < 20:
        score += 2; signals.append("BB하단(반등기대)")
    elif ind["bb_pct"] < 50:
        score += 1; signals.append(f'BB중하단({ind["bb_pct"]:.0f}%)')
    elif ind["bb_pct"] > 90:
        score -= 2; signals.append(f'BB상단과열({ind["bb_pct"]:.0f}%)')
    elif ind["bb_pct"] > 70:
        score -= 1; signals.append(f'BB상단({ind["bb_pct"]:.0f}%)')
    else:
        signals.append(f'BB중간({ind["bb_pct"]:.0f}%)')

    # 추세
    if ind["above_ma20"] and ind["above_ma60"]:
        score += 2; signals.append("MA20+60위(상승)")
    elif ind["above_ma20"]:
        score += 1; signals.append("MA20위")
    elif ind["above_ma60"]:
        signals.append("MA20아래MA60위")
    else:
        score -= 1; signals.append("MA60아래(하락)")

    # TRIX
    if ind["trix_bull"]:
        score += 1; signals.append("TRIX상승전환")
    else:
        score -= 1; signals.append("TRIX하락")

    # Stoch
    if ind["stoch_gx"]:
        score += 2; signals.append("Stoch골든크로스!")
    elif ind["stoch_k"] < 20:
        score += 1; signals.append("Stoch과매도(반등)")
    elif ind["stoch_k"] > 80:
        score -= 1; signals.append("Stoch과매수")

    # 거래량
    if ind["vol_ratio"] > 2:
        score += 1; signals.append(f'거래량폭발({ind["vol_ratio"]}x)')
    elif ind["vol_ratio"] < 0.5:
        score -= 1; signals.append(f'거래량부진({ind["vol_ratio"]}x)')

    # 5일 모멘텀
    if ind["ret5"] > 5:
        score -= 1; signals.append(f'5일급등+{ind["ret5"]}%(과열)')
    elif ind["ret5"] < -5:
        score += 1; signals.append(f'5일급락{ind["ret5"]}%(반등)')

    # 판정
    if score >= 5:
        outlook, emoji, advice = "강세", "🟢", "보유+추가검토"
    elif score >= 3:
        outlook, emoji, advice = "양호", "🟢", "보유유지"
    elif score >= 1:
        outlook, emoji, advice = "중립", "🟡", "보유관망"
    elif score >= -1:
        outlook, emoji, advice = "주의", "🟠", "일부정리 검토"
    else:
        outlook, emoji, advice = "약세", "🔴", "정리검토"

    return score, outlook, emoji, advice, signals


def main():
    # 수급 데이터
    try:
        foreign = {}
        inst = {}
        for h in holdings:
            t = h["ticker"]
            df_inv = pykrx_stock.get_market_trading_value_by_date("20260217", "20260224", t)
            if len(df_inv) > 0:
                foreign[t] = int(df_inv["외국인합계"].tail(5).sum()) if "외국인합계" in df_inv.columns else 0
                inst[t] = int(df_inv["기관합계"].tail(5).sum()) if "기관합계" in df_inv.columns else 0
    except Exception:
        foreign, inst = {}, {}

    print("=" * 60)
    print("  보유 11종목 — 내일(2/25) 기술적 전망")
    print("=" * 60)

    results_list = []
    for h in holdings:
        ind = calc_indicators(h["ticker"])
        if ind is None:
            print(f'\n{h["name"]}({h["ticker"]}): 데이터 부족')
            continue

        score, outlook, emoji, advice, signals = judge(h, ind)
        fg = foreign.get(h["ticker"], 0)
        it = inst.get(h["ticker"], 0)

        # 수급 보너스
        supply_note = ""
        if fg > 0 and it > 0:
            score += 1
            supply_note = "외인+기관 동시매수"
        elif fg > 0:
            supply_note = "외인 순매수"
        elif it > 0:
            supply_note = "기관 순매수"
        elif fg < 0 and it < 0:
            score -= 1
            supply_note = "외인+기관 동시매도"

        # 재판정
        if score >= 5:
            outlook, emoji, advice = "강세", "🟢", "보유+추가검토"
        elif score >= 3:
            outlook, emoji, advice = "양호", "🟢", "보유유지"
        elif score >= 1:
            outlook, emoji, advice = "중립", "🟡", "보유관망"
        elif score >= -1:
            outlook, emoji, advice = "주의", "🟠", "일부정리 검토"
        else:
            outlook, emoji, advice = "약세", "🔴", "정리검토"

        eval_amt = h["qty"] * h["price"]

        print(f'\n{"─" * 60}')
        print(f'{emoji} {h["name"]}({h["ticker"]}) | {h["price"]:,}원 x {h["qty"]}주 = {eval_amt:,}원 | 수익 {h["pnl_pct"]:+.1f}%')
        print(f'  전망: {outlook} (점수 {score}) → {advice}')
        print(f'  RSI {ind["rsi"]} | BB% {ind["bb_pct"]:.0f} | ADX {ind["adx"]} | Stoch {ind["stoch_k"]:.0f}')
        ma_str = f'MA20{"↑" if ind["above_ma20"] else "↓"} MA60{"↑" if ind["above_ma60"] else "↓"}'
        print(f'  {ma_str} | TRIX{"↑" if ind["trix_bull"] else "↓"} | Vol {ind["vol_ratio"]}x | 5일{ind["ret5"]:+.1f}% 20일{ind["ret20"]:+.1f}%')
        if supply_note:
            print(f'  수급5일: 외인{fg/1e8:+.1f}억 기관{it/1e8:+.1f}억 → {supply_note}')
        print(f'  시그널: {", ".join(signals)}')

        results_list.append({
            "name": h["name"], "ticker": h["ticker"],
            "price": h["price"], "qty": h["qty"], "pnl_pct": h["pnl_pct"],
            "score": score, "outlook": outlook, "advice": advice,
            **ind,
            "foreign_5d": fg, "inst_5d": it,
        })

    # 정리 제안 요약
    print(f'\n{"=" * 60}')
    print("  정리 우선순위 제안")
    print("=" * 60)

    sorted_results = sorted(results_list, key=lambda x: x["score"])
    for r in sorted_results:
        emoji = "🔴" if r["score"] <= -1 else "🟠" if r["score"] <= 0 else "🟡" if r["score"] <= 2 else "🟢"
        print(f'  {emoji} {r["name"]:8s} | 수익{r["pnl_pct"]:+5.1f}% | 점수{r["score"]:+3d} | {r["advice"]}')

    # JSON 저장
    out = {"date": "2026-02-24", "analysis": results_list}
    Path(ROOT / "data" / "daily_snapshots" / "2026-02-24" / "holdings_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f'\n저장: data/daily_snapshots/2026-02-24/holdings_analysis.json')


if __name__ == "__main__":
    main()
