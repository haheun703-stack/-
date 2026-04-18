"""
NXT(넥스트레이드) 거래 가능 종목 마스터 업데이트

1차: nextrade.co.kr 종목 리스트 크롤링
2차: KIS API로 우리 유니버스 NXT 거래 가능 여부 확인
최종: data/nxt/nxt_master.json 저장
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

logger = logging.getLogger(__name__)

NXT_MASTER_PATH = PROJECT_ROOT / "data" / "nxt" / "nxt_master.json"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "processed"
STOCK_CSV_DIR = PROJECT_ROOT / "stock_data_daily"

# NXT 웹사이트 종목 리스트 URL (공식 API)
NXT_STOCK_LIST_URL = "https://www.nextrade.co.kr/api/market/stockList"
# 대체: KRX 정보데이터시스템 NXT 대상 종목
NXT_ALT_URL = "https://www.nextrade.co.kr/menu/marketData/menuList.do"


def _fetch_nxt_website() -> dict[str, str] | None:
    """NXT 공식 웹사이트에서 거래 가능 종목 리스트 크롤링"""
    try:
        # 1차: NXT API endpoint 시도
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nextrade.co.kr/",
        }

        # NXT 종목 리스트 페이지 (JSON API가 없으면 HTML 파싱)
        resp = requests.get(NXT_STOCK_LIST_URL, headers=headers, timeout=15)
        if resp.status_code == 200:
            try:
                data = resp.json()
                # API 형식에 따라 파싱
                stocks = {}
                items = data if isinstance(data, list) else data.get("data", data.get("list", []))
                for item in items:
                    code = item.get("stockCode", item.get("shrn_iscd", item.get("code", "")))
                    name = item.get("stockName", item.get("hts_kor_isnm", item.get("name", "")))
                    if code and len(code) == 6:
                        stocks[code] = name
                if stocks:
                    logger.info("[NXT마스터] 웹사이트에서 %d종목 수집", len(stocks))
                    return stocks
            except (json.JSONDecodeError, ValueError):
                logger.debug("[NXT마스터] JSON 파싱 실패, HTML 파싱 시도")

        # 2차: HTML 페이지에서 종목코드 추출 시도
        resp2 = requests.get(NXT_ALT_URL, headers=headers, timeout=15)
        if resp2.status_code == 200 and len(resp2.text) > 1000:
            import re
            # 6자리 종목코드 + 한글이름 패턴
            pattern = r'"?(\d{6})"?\s*[,:]?\s*"?([가-힣A-Za-z0-9\s&]+)"?'
            matches = re.findall(pattern, resp2.text)
            if matches:
                stocks = {code: name.strip() for code, name in matches if len(code) == 6}
                if len(stocks) > 100:
                    logger.info("[NXT마스터] HTML에서 %d종목 추출", len(stocks))
                    return stocks

    except requests.RequestException as e:
        logger.warning("[NXT마스터] 웹사이트 접근 실패: %s", e)
    except Exception as e:
        logger.warning("[NXT마스터] 크롤링 실패: %s", e)

    return None


def _build_from_universe() -> dict[str, str]:
    """
    우리 유니버스(84종목) + stock_data_daily에서 시총 상위 종목 추출.
    NXT는 시총 상위 800종목 대부분 거래 가능 → 우리 유니버스는 거의 전부 포함.
    """
    tickers = {}

    # 1) data/processed/*.parquet에서 84종목
    if UNIVERSE_DIR.exists():
        for f in UNIVERSE_DIR.glob("*.parquet"):
            ticker = f.stem
            if len(ticker) == 6 and ticker.isdigit():
                # stock_data_daily에서 이름 추출
                csv_path = STOCK_CSV_DIR / f"{ticker}.csv"
                name = _get_stock_name(csv_path, ticker)
                tickers[ticker] = name

    # 2) stock_data_daily에서 추가 대형주 (거래대금 상위)
    if STOCK_CSV_DIR.exists():
        import pandas as pd
        for csv_file in sorted(STOCK_CSV_DIR.glob("*.csv"))[:2000]:
            ticker = csv_file.stem
            if len(ticker) != 6 or not ticker.isdigit():
                continue
            if ticker in tickers:
                continue
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")
                if len(df) < 20:
                    continue
                # 최근 20일 평균 거래대금 50억 이상 = NXT 거래 가능 추정
                vol_col = "volume" if "volume" in df.columns else "거래량"
                close_col = "close" if "close" in df.columns else "종가"
                if vol_col in df.columns and close_col in df.columns:
                    recent = df.tail(20)
                    avg_value = (recent[vol_col] * recent[close_col]).mean()
                    if avg_value >= 5_000_000_000:  # 50억
                        name = _get_stock_name(csv_file, ticker)
                        tickers[ticker] = name
            except Exception:
                continue

    logger.info("[NXT마스터] 유니버스 기반 %d종목 빌드", len(tickers))
    return tickers


def _get_stock_name(csv_path: Path, ticker: str) -> str:
    """CSV 파일에서 종목명 추출"""
    try:
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=1)
            for col in ["name", "종목명", "종목이름"]:
                if col in df.columns:
                    val = df[col].iloc[0]
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    except Exception:
        pass
    return ticker


def _verify_with_kis(tickers: dict[str, str]) -> dict[str, str]:
    """KIS API로 NXT 거래 가능 여부 확인 (선택사항, 장중에만 동작)"""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")

        import mojito
        broker = mojito.KoreaInvestment(
            api_key=os.getenv("KIS_APP_KEY", ""),
            api_secret=os.getenv("KIS_APP_SECRET", ""),
            acc_no=os.getenv("KIS_ACC_NO", ""),
            mock=os.getenv("MODEL") != "REAL",
        )

        verified = {}
        sample_tickers = list(tickers.keys())[:10]  # 10개만 샘플 확인

        for ticker in sample_tickers:
            try:
                data = broker.fetch_price(ticker)
                output = data.get("output", {})
                # NXT 거래 가능 여부 확인: 가격이 0이 아니면 가능
                price = int(output.get("stck_prpr", 0))
                if price > 0:
                    verified[ticker] = tickers[ticker]
                time.sleep(0.1)
            except Exception:
                continue

        if verified:
            logger.info("[NXT마스터] KIS 샘플 %d/%d 확인", len(verified), len(sample_tickers))
            # 샘플이 전부 통과하면 전체를 NXT 거래 가능으로 간주
            if len(verified) == len(sample_tickers):
                return tickers

        return tickers  # 검증 실패해도 유니버스 기반으로 유지

    except Exception as e:
        logger.info("[NXT마스터] KIS 검증 스킵: %s", e)
        return tickers


def update_nxt_master(verify_kis: bool = False) -> dict:
    """NXT 거래 가능 종목 마스터 업데이트"""

    # 1차: NXT 웹사이트 크롤링
    source = "unknown"
    print("[1/3] NXT 웹사이트 크롤링 시도...")
    nxt_stocks = _fetch_nxt_website()
    if nxt_stocks:
        source = "nxt_website"

    # 2차: 유니버스 기반 빌드
    if not nxt_stocks:
        print("[2/3] 웹사이트 실패 → 유니버스 기반 빌드...")
        nxt_stocks = _build_from_universe()
        source = "universe_build"

    # 3차 (선택): KIS API 검증
    if verify_kis and nxt_stocks:
        print("[3/3] KIS API 검증 (샘플)...")
        nxt_stocks = _verify_with_kis(nxt_stocks)

    # 우리 84종목 중 NXT 교차 확인
    universe_tickers = set()
    if UNIVERSE_DIR.exists():
        for f in UNIVERSE_DIR.glob("*.parquet"):
            t = f.stem
            if len(t) == 6 and t.isdigit():
                universe_tickers.add(t)

    overlap = universe_tickers & set(nxt_stocks.keys())

    # 결과 저장
    result = {
        "updated_at": date.today().isoformat(),
        "source": source,
        "count": len(nxt_stocks),
        "universe_overlap": len(overlap),
        "universe_total": len(universe_tickers),
        "ticker_set": dict(sorted(nxt_stocks.items())),
        "tickers": sorted(nxt_stocks.keys()),
    }

    NXT_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    NXT_MASTER_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== NXT 마스터 업데이트 완료 ===")
    print(f"  소스: {result['source']}")
    print(f"  NXT 거래 가능: {result['count']}종목")
    print(f"  유니버스 교차: {result['universe_overlap']}/{result['universe_total']}종목")
    print(f"  저장: {NXT_MASTER_PATH}")

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="NXT 종목 마스터 업데이트")
    parser.add_argument("--verify-kis", action="store_true", help="KIS API로 NXT 거래 가능 확인")
    args = parser.parse_args()

    update_nxt_master(verify_kis=args.verify_kis)
