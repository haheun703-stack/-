"""금융투자 수급 수집 — 우리가 안 담던 폭등 선행 신호 (사장님 5/31).

KRX 12세부 투자자 중 '금융투자'(증권사 자기매매·프로그램·위탁창구)가 MLCC 폭등의
선행 엔진(삼성전기 4/1 초입부터 +1.7조)이었으나 우리는 기관합계로 묶어 놓침.
거래대금 상위 종목의 금융투자/기타외국인 시계열 수집 → 폭등 전조 검증용.
data/financial_invest/{code}.parquet 저장.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pykrx import stock

OUT = PROJECT_ROOT / "data" / "financial_invest"
OUT.mkdir(parents=True, exist_ok=True)
S, E = "20240101", "20260529"   # 약세장(24) 포함 → 사후선택/베타 검증
TOP_N = 300


def main() -> int:
    # 거래대금 상위 TOP_N 선정 (processed parquet 기준)
    cand = []
    for f in glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["trading_value"])
            cand.append((Path(f).stem, float(df["trading_value"].tail(250).mean())))
        except Exception:
            pass
    cand.sort(key=lambda x: -x[1])
    codes = [c for c, _ in cand[:TOP_N]]
    print(f"거래대금 상위 {len(codes)}종목 금융투자 수집 시작 ({S}~{E})", flush=True)

    ok = 0; err = 0
    for n, code in enumerate(codes, 1):
        try:
            d = stock.get_market_trading_value_by_date(S, E, code, detail=True)
            o = stock.get_market_ohlcv(S, E, code)
            if d is None or o is None or len(d) < 60:
                err += 1
                continue
            df = pd.DataFrame(index=o.index)
            df["close"] = o["종가"]; df["open"] = o["시가"]
            df["volume"] = o["거래량"]
            df["trading_value"] = o["거래량"] * o["종가"]
            for col in ["금융투자", "기타외국인", "외국인", "연기금", "투신", "개인"]:
                df[col] = d[col].reindex(o.index).fillna(0) if col in d.columns else 0
            df = df[df["close"] > 0]
            if len(df) < 60:
                err += 1
                continue
            df.to_parquet(OUT / f"{code}.parquet")
            ok += 1
        except Exception:
            err += 1
        if n % 30 == 0:
            print(f"  {n}/{len(codes)} (저장 {ok} / 오류 {err})", flush=True)
        time.sleep(0.15)
    print(f"\n완료: 저장 {ok} / 오류 {err} (총 {len(codes)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
