"""상폐 종목 데이터 수집 — 생존편향 보정용 (사장님 5/31).

2023-06 상장 - 2026-05 상장 diff = 기간 내 상폐 종목.
각 종목 OHLCV(pykrx) + 외인/기관 수급 → supply_divergence(외인매도+기관매수=1) 계산.
data/delisted/{code}.parquet 저장 → 백테스트 유니버스에 합쳐 생존편향 보정.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pykrx import stock

OUT = PROJECT_ROOT / "data" / "delisted"
OUT.mkdir(parents=True, exist_ok=True)
S, E = "20230601", "20260529"


def main() -> int:
    t_old = set(stock.get_market_ticker_list(S, market="ALL"))
    t_now = set(stock.get_market_ticker_list(E, market="ALL"))
    delisted = sorted(t_old - t_now)
    print(f"기간 내 상폐 종목: {len(delisted)}개 수집 시작", flush=True)

    ok = 0; skip = 0; err = 0
    for n, code in enumerate(delisted, 1):
        try:
            o = stock.get_market_ohlcv(S, E, code)
            if o is None or len(o) < 42:
                skip += 1
                continue
            inv = stock.get_market_trading_value_by_date(S, E, code)
            df = pd.DataFrame(index=o.index)
            df["open"] = o["시가"]; df["high"] = o["고가"]; df["low"] = o["저가"]
            df["close"] = o["종가"]; df["volume"] = o["거래량"]
            df["trading_value"] = o["거래량"] * o["종가"]  # 종목별 OHLCV엔 거래대금 컬럼 없음 → volume×close
            f = inv["외국인합계"].reindex(o.index).fillna(0) if "외국인합계" in inv.columns else pd.Series(0, index=o.index)
            i = inv["기관합계"].reindex(o.index).fillna(0) if "기관합계" in inv.columns else pd.Series(0, index=o.index)
            df["외국인합계"] = f; df["기관합계"] = i
            div = pd.Series(0, index=df.index)
            div[(f < 0) & (i > 0)] = 1
            div[(f < 0) & (i < 0)] = -1
            df["supply_divergence"] = div
            df = df[df["close"] > 0]
            if len(df) < 42:
                skip += 1
                continue
            df.to_parquet(OUT / f"{code}.parquet")
            ok += 1
        except Exception:
            err += 1
        if n % 25 == 0:
            print(f"  {n}/{len(delisted)} (저장 {ok} / 스킵 {skip} / 오류 {err})", flush=True)
        time.sleep(0.15)  # KRX rate limit 완화
    print(f"\n완료: 저장 {ok} / 스킵 {skip} / 오류 {err} (총 {len(delisted)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
