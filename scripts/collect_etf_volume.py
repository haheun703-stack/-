"""
ETF-8: 섹터 ETF 거래량 모니터링 (선행지표 5)
=============================================
매일 장 마감 후 섹터 ETF 거래량 수집:
- etf_universe.json의 모든 섹터 ETF
- 당일 거래량 / 20일 평균 거래량 비율
- 2배 이상이면 "거래량 폭발" 태깅
- 저장: data/etf_volume_monitor.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def collect_etf_volume() -> dict:
    """섹터 ETF 거래량 수집 + 20일 평균 대비 비율 계산."""
    from pykrx import stock as krx

    # ETF 유니버스 로드
    universe_path = DATA_DIR / "etf_universe.json"
    with open(universe_path, encoding="utf-8") as f:
        universe = json.load(f)

    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=40)).strftime("%Y%m%d")  # 여유있게 40일

    results = []
    sector_etfs = universe.get("sector", [])

    print(f"섹터 ETF 거래량 수집 시작 ({len(sector_etfs)}종목)")

    for etf in sector_etfs:
        ticker = etf["ticker"]
        name = etf["name"]
        sector = etf.get("sector", "")

        try:
            df = krx.get_market_ohlcv_by_date(start_date, end_date, ticker)
            if df.empty or len(df) < 5:
                print(f"  {name} ({ticker}): 데이터 부족 ({len(df)}행)")
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "volume_today": 0,
                    "volume_avg_20d": 0,
                    "volume_ratio": 0.0,
                    "explosion": False,
                    "error": "데이터 부족",
                })
                continue

            # 컬럼명 정규화 (pykrx 한글 컬럼)
            vol_col = None
            for col in df.columns:
                if "거래량" in str(col):
                    vol_col = col
                    break
            if vol_col is None and "volume" in [c.lower() for c in df.columns]:
                vol_col = [c for c in df.columns if c.lower() == "volume"][0]
            if vol_col is None:
                # 6번째 컬럼이 거래량인 경우 (index: 시가,고가,저가,종가,거래량,거래대금,등락률)
                vol_col = df.columns[4] if len(df.columns) > 4 else df.columns[0]

            volumes = df[vol_col].values
            today_vol = int(volumes[-1])

            # 20일 평균 (당일 제외)
            lookback = volumes[:-1][-20:] if len(volumes) > 1 else volumes[-20:]
            avg_20d = float(lookback.mean()) if len(lookback) > 0 else 0

            ratio = today_vol / avg_20d if avg_20d > 0 else 0.0
            explosion = ratio >= 2.0

            result = {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "volume_today": today_vol,
                "volume_avg_20d": round(avg_20d),
                "volume_ratio": round(ratio, 2),
                "explosion": explosion,
            }
            results.append(result)

            tag = " *** 폭발 ***" if explosion else ""
            print(f"  {name}: {today_vol:,} / avg {avg_20d:,.0f} = {ratio:.2f}배{tag}")

        except Exception as e:
            print(f"  {name} ({ticker}): 오류 — {e}")
            results.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "volume_today": 0,
                "volume_avg_20d": 0,
                "volume_ratio": 0.0,
                "explosion": False,
                "error": str(e),
            })

        time.sleep(0.2)  # pykrx rate limit

    # 결과 저장
    output = {
        "date": today.strftime("%Y-%m-%d"),
        "generated_at": today.strftime("%Y-%m-%d %H:%M"),
        "etf_count": len(results),
        "explosions": sum(1 for r in results if r.get("explosion")),
        "etfs": sorted(results, key=lambda x: x.get("volume_ratio", 0), reverse=True),
    }

    out_path = DATA_DIR / "etf_volume_monitor.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {out_path}")
    print(f"총 {len(results)}종목, 거래량 폭발 {output['explosions']}건")

    return output


if __name__ == "__main__":
    collect_etf_volume()
