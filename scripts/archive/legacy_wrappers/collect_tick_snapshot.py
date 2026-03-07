"""
전종목 체결정보 스냅샷 수집 → parquet

수집 항목: 시간, 현재가, 전일대비, 등락률, 매도호가, 매수호가,
          체결강도, 누적거래량, 체결량(최근)

사용법:
  python scripts/collect_tick_snapshot.py              # 전체 유니버스
  python scripts/collect_tick_snapshot.py --dry-run    # 005930 1종목 테스트
  python scripts/collect_tick_snapshot.py --tickers 005930,000660
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.adapters.kis_intraday_adapter import KisIntradayAdapter, _rate_limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tick_snapshot")

DATA_ROOT = Path("data/intraday/tick_snapshot")
UNIVERSE_CSV = Path("data/universe.csv")


def load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return df["ticker"].astype(str).str.zfill(6).tolist()


def fetch_tick_detail(adapter: KisIntradayAdapter, ticker: str) -> dict | None:
    """
    단일 종목 체결정보 수집.

    KIS API 2개 조합:
      1) fetch_price (FHKST01010100): 현재가, 호가, 등락률, 거래량
      2) _api_get FHKST01010300: 최근 체결 내역 → 체결량
    """
    _rate_limit()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 1) 현재가 시세 API
        data = adapter.broker.fetch_price(ticker)
        out = data.get("output", {})
        if not out:
            return None

        current_price = int(out.get("stck_prpr", 0))
        if current_price == 0:
            return None

        result = {
            "ticker": ticker,
            "timestamp": now_str,
            "current_price": current_price,
            "prev_close_diff": int(out.get("prdy_vrss", 0)),
            "change_pct": float(out.get("prdy_ctrt", 0)),
            "ask_price": int(out.get("askp1", current_price)),
            "bid_price": int(out.get("bidp1", current_price)),
            "strength": float(out.get("tday_rltv", out.get("seln_cntg_smtn", 0))),
            "cum_volume": int(out.get("acml_vol", 0)),
            "cum_trade_value": int(out.get("acml_tr_pbmn", 0)),
            "open_price": int(out.get("stck_oprc", 0)),
            "high_price": int(out.get("stck_hgpr", 0)),
            "low_price": int(out.get("stck_lwpr", 0)),
        }

        # 2) 체결 API로 최근 체결량 수집
        _rate_limit()
        trade_data = adapter._api_get(
            path="uapi/domestic-stock/v1/quotations/inquire-ccnl",
            tr_id="FHKST01010300",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
            },
        )
        trades = trade_data.get("output", [])
        if isinstance(trades, list) and trades:
            # 가장 최근 체결
            latest = trades[0]
            result["last_exec_vol"] = int(latest.get("cntg_vol", 0))
            result["last_exec_time"] = latest.get("stck_cntg_hour", "")
            # 최근 30건 평균 체결량
            exec_vols = [int(t.get("cntg_vol", 0)) for t in trades if t.get("cntg_vol")]
            result["avg_exec_vol"] = round(sum(exec_vols) / len(exec_vols), 1) if exec_vols else 0
        else:
            result["last_exec_vol"] = 0
            result["last_exec_time"] = ""
            result["avg_exec_vol"] = 0

        return result

    except Exception as e:
        logger.error("[체결] %s 수집 오류: %s", ticker, e)
        return None


def collect_all_ticks(adapter: KisIntradayAdapter, tickers: list[str]) -> dict:
    """전종목 체결정보 수집 → parquet 저장"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H%M")

    all_rows = []
    for i, ticker in enumerate(tickers, 1):
        row = fetch_tick_detail(adapter, ticker)
        if row:
            all_rows.append(row)
            if i % 50 == 0:
                logger.info("[%d/%d] %d건 수집 중...", i, len(tickers), len(all_rows))

    if not all_rows:
        logger.warning("수집 데이터 없음")
        return {"stocks": 0, "rows": 0, "path": ""}

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 저장
    folder = DATA_ROOT / date_str
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"snapshot_{time_str}.parquet"
    df.to_parquet(path, index=False)

    logger.info("체결 스냅샷 저장: %s (%d종목)", path, len(df))
    return {"stocks": len(df), "rows": len(df), "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description="전종목 체결정보 스냅샷")
    parser.add_argument("--dry-run", action="store_true", help="005930 1종목 테스트")
    parser.add_argument("--tickers", type=str, help="종목 (쉼표 구분)")
    args = parser.parse_args()

    if args.dry_run:
        tickers = ["005930"]
        logger.info("[dry-run] 삼성전자 1종목")
    elif args.tickers:
        tickers = [t.strip().zfill(6) for t in args.tickers.split(",")]
    else:
        tickers = load_universe()

    logger.info("수집 대상: %d종목", len(tickers))

    adapter = KisIntradayAdapter()

    t0 = time.time()
    result = collect_all_ticks(adapter, tickers)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"체결 스냅샷 완료 ({elapsed:.1f}초)")
    print(f"  종목: {result['stocks']}개")
    print(f"  저장: {result['path']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
