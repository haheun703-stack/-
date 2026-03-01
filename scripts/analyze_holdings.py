"""보유종목 심층 분석 — KIS API 실시간 + parquet 기술적 지표"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CSV_DIR = Path("stock_data_daily")
PROCESSED_DIR = Path("data/processed")

UAE_TICKERS = {
    "012450", "047810", "272210", "042660", "079550", "064350",
    "082740", "011210", "034020", "052690", "051600", "298040",
    "083650", "000720", "028260", "047040", "375500",
}


def analyze():
    from src.adapters.kis_order_adapter import KisOrderAdapter
    adapter = KisOrderAdapter()
    balance = adapter.fetch_balance()

    holdings = balance["holdings"]
    cash = balance["available_cash"]

    if not holdings:
        print("보유종목 없음")
        return

    total_invest = sum(h["avg_price"] * h["quantity"] for h in holdings)
    total_eval = sum(h["eval_amount"] for h in holdings)
    total_pnl = sum(h["pnl_amount"] for h in holdings)

    print("=" * 100)
    print(f"보유종목 심층 분석 ({len(holdings)}종목)")
    print("=" * 100)
    print(f"총 투자금: {total_invest:,.0f}원 | 평가금: {total_eval:,.0f}원 | "
          f"손익: {total_pnl:+,.0f}원 ({total_pnl/total_invest*100:+.2f}%)")
    print(f"예수금: {cash:,.0f}원 | 총 자산: {total_eval + cash:,.0f}원")

    for h in holdings:
        _analyze_single(h, total_invest)

    # 요약
    winners = [h for h in holdings if h["pnl_pct"] > 0]
    losers = [h for h in holdings if h["pnl_pct"] <= 0]
    print(f"\n{'=' * 100}")
    print("포트폴리오 요약")
    print(f"{'=' * 100}")
    print(f"  수익 종목: {len(winners)}개 | 손실 종목: {len(losers)}개")
    print(f"  최대 수익: {max(h['pnl_pct'] for h in holdings):+.2f}% | "
          f"최대 손실: {min(h['pnl_pct'] for h in holdings):+.2f}%")
    print(f"  총 손익: {total_pnl:+,.0f}원 ({total_pnl/total_invest*100:+.2f}%)")


def _analyze_single(h: dict, total_invest: float):
    ticker = h["ticker"]
    name = h["name"]
    qty = h["quantity"]
    avg_price = h["avg_price"]
    close = h["current_price"]
    pnl_pct = h["pnl_pct"]
    pnl_amt = h["pnl_amount"]
    invest = avg_price * qty
    weight = invest / total_invest * 100

    pq = PROCESSED_DIR / f"{ticker}.parquet"
    if not pq.exists():
        print(f"\n--- {name} ({ticker}) --- parquet 없음")
        return

    df = pd.read_parquet(pq)
    if len(df) < 20:
        print(f"\n--- {name} ({ticker}) --- 데이터 부족")
        return

    last = df.iloc[-1]

    # 이동평균
    ma5 = df["close"].tail(5).mean()
    ma20 = df["close"].tail(20).mean()
    ma60 = df["close"].tail(60).mean() if len(df) >= 60 else 0
    ma120 = df["close"].tail(120).mean() if len(df) >= 120 else 0

    rsi = float(last.get("rsi_14", 0))
    adx = float(last.get("adx_14", 0))
    macd_val = float(last.get("macd", 0))
    macd_sig = float(last.get("macd_signal", 0))
    macd_hist = float(last.get("macd_histogram", 0))
    stoch_k = float(last.get("stoch_slow_k", 50))
    bb_pos = float(last.get("bb_position", 50))

    vol = float(last.get("volume", 0))
    vol_avg20 = df["volume"].tail(20).mean()
    vol_ratio = vol / vol_avg20 if vol_avg20 > 0 else 0

    # 수급 (최근 5일)
    f5 = i5 = 0.0
    if "외국인합계" in df.columns:
        f5 = float(np.nansum(df.tail(5)["외국인합계"].values))
        i5 = float(np.nansum(df.tail(5)["기관합계"].values))
    if f5 == 0 and i5 == 0:
        csvs = list(CSV_DIR.glob(f"*_{ticker}.csv"))
        if csvs:
            cdf = pd.read_csv(csvs[0], parse_dates=["Date"]).sort_values("Date").tail(5)
            if "Foreign_Net" in cdf.columns:
                f5 = float(cdf["Foreign_Net"].sum())
                i5 = float(cdf["Inst_Net"].sum())

    closes = df["close"].values
    ret_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
    ret_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0

    high_52 = float(df["high"].tail(250).max()) if len(df) >= 250 else float(df["high"].max())
    low_52 = float(df["low"].tail(250).min()) if len(df) >= 250 else float(df["low"].min())
    from_high = (close / high_52 - 1) * 100
    from_low = (close / low_52 - 1) * 100

    if ma5 > ma20 > ma60 > 0:
        ma_align = "정배열"
    elif ma60 > ma20 > ma5 and ma5 > 0:
        ma_align = "역배열"
    else:
        ma_align = "혼조"

    signals = []
    warns = []

    if rsi > 70:
        warns.append(f"RSI {rsi:.0f} 과매수")
    elif rsi < 30:
        warns.append(f"RSI {rsi:.0f} 과매도")
    elif 40 <= rsi <= 60:
        signals.append(f"RSI {rsi:.0f} 적정")

    if stoch_k > 80:
        warns.append(f"Stoch {stoch_k:.0f} 과열")
    elif stoch_k < 20:
        signals.append(f"Stoch {stoch_k:.0f} 침체")

    if macd_hist > 0 and macd_val > macd_sig:
        signals.append("MACD 양호")
    elif macd_hist < 0:
        warns.append("MACD 하락")

    if close > ma20:
        signals.append("MA20 위")
    else:
        warns.append("MA20 아래")

    if close > ma60 and ma60 > 0:
        signals.append("MA60 위")
    elif ma60 > 0:
        warns.append("MA60 아래")

    if f5 > 0:
        signals.append("외인5d 순매수")
    elif f5 < 0:
        warns.append("외인5d 순매도")
    if i5 > 0:
        signals.append("기관5d 순매수")
    elif i5 < 0:
        warns.append("기관5d 순매도")

    # 판정
    danger = 0.0
    if pnl_pct < -10:
        danger += 3
    elif pnl_pct < -5:
        danger += 2
    elif pnl_pct < -3:
        danger += 1
    if close < ma20:
        danger += 1
    if close < ma60 and ma60 > 0:
        danger += 1
    if f5 < 0 and i5 < 0:
        danger += 1
    if macd_hist < 0:
        danger += 0.5

    if pnl_pct > 3 and (rsi > 70 or stoch_k > 80):
        verdict = "이익실현 고려"
    elif danger >= 4:
        verdict = "손절 검토"
    elif danger >= 3:
        verdict = "주의 관찰"
    elif pnl_pct > 0 and len(signals) > len(warns):
        verdict = "보유 유지"
    elif len(signals) >= len(warns):
        verdict = "관망"
    else:
        verdict = "주의"

    uae_tag = " [UAE 수혜]" if ticker in UAE_TICKERS else ""

    print(f"\n{'=' * 100}")
    print(f"{name} ({ticker}){uae_tag} | 비중 {weight:.1f}% | 투자 {invest:,.0f}원")
    print(f"{'~' * 100}")
    print(f"  매수가: {avg_price:,.0f}  ->  현재가: {close:,.0f}  |  "
          f"수익률: {pnl_pct:+.2f}% ({pnl_amt:+,}원)")
    print(f"  MA5: {ma5:,.0f} | MA20: {ma20:,.0f} | MA60: {ma60:,.0f} | "
          f"MA120: {ma120:,.0f} | 정렬: {ma_align}")
    print(f"  RSI: {rsi:.1f} | ADX: {adx:.1f} | Stoch: {stoch_k:.1f} | BB위치: {bb_pos:.0f}%")
    print(f"  MACD: {macd_val:.1f} / Sig: {macd_sig:.1f} / Hist: {macd_hist:.1f}")
    print(f"  5일 {ret_5d:+.1f}% | 20일 {ret_20d:+.1f}% | "
          f"52주고점 대비 {from_high:+.1f}% | 52주저점 대비 {from_low:+.1f}%")
    print(f"  거래량비: {vol_ratio:.1f}x | "
          f"외인5d: {f5/1e6:+.1f}백만 | 기관5d: {i5/1e6:+.1f}백만")
    print(f"  장점: {' | '.join(signals) if signals else '없음'}")
    print(f"  주의: {' | '.join(warns) if warns else '없음'}")
    print(f"  >> 판정: [{verdict}]")


if __name__ == "__main__":
    analyze()
