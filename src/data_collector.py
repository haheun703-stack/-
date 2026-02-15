"""
Step 1: data_collector.py — KOSPI/KOSDAQ 전종목 OHLCV + 투자자 매매동향 수집

데이터 소스: pykrx (KRX 공식 데이터, API 키 불필요)
기간: settings.yaml의 backtest.start_date ~ end_date
저장: data/raw/{ticker}.parquet
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── pykrx import (로컬 실행 시 사용) ──
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    logger.warning("pykrx 미설치. pip install pykrx 실행 필요")


class DataCollector:
    """KRX에서 전종목 OHLCV, 투자자 매매동향, 기본 펀더멘탈 수집"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.start_date = self.config["backtest"]["start_date"].replace("-", "")
        self.end_date = self.config["backtest"]["end_date"].replace("-", "")
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # 이미 수집 완료된 종목 (resume 용)
        self.completed = set()
        self._load_completed()

    def _load_completed(self):
        """이미 저장된 parquet 파일에서 완료 티커 추출"""
        for f in self.raw_dir.glob("*.parquet"):
            ticker = f.stem  # 예: 005930
            self.completed.add(ticker)
        if self.completed:
            logger.info(f"이미 수집 완료: {len(self.completed)}종목 (resume 모드)")

    # ──────────────────────────────────────────────
    # 1. 종목 리스트 가져오기
    # ──────────────────────────────────────────────
    def get_all_tickers(self) -> pd.DataFrame:
        """KOSPI + KOSDAQ 전종목 티커와 종목명 반환"""
        if not PYKRX_AVAILABLE:
            raise RuntimeError("pykrx 미설치")

        date = self.end_date
        tickers = []

        for market in ["KOSPI", "KOSDAQ"]:
            ticker_list = krx.get_market_ticker_list(date, market=market)
            for t in ticker_list:
                name = krx.get_market_ticker_name(t)
                tickers.append({
                    "ticker": t,
                    "name": name,
                    "market": market,
                })
            time.sleep(0.3)

        df = pd.DataFrame(tickers)
        logger.info(f"전체 종목 수: {len(df)} (KOSPI: {len(df[df.market=='KOSPI'])}, KOSDAQ: {len(df[df.market=='KOSDAQ'])})")
        return df

    # ──────────────────────────────────────────────
    # 2. 개별 종목 OHLCV 수집
    # ──────────────────────────────────────────────
    def download_ohlcv(self, ticker: str) -> pd.DataFrame:
        """일봉 OHLCV + 거래대금"""
        df = krx.get_market_ohlcv_by_date(
            self.start_date, self.end_date, ticker, adjusted=True
        )
        if df.empty:
            return df

        df.index.name = "date"
        df.columns = ["open", "high", "low", "close", "volume", "trading_value", "price_change"]
        return df

    # ──────────────────────────────────────────────
    # 3. 투자자별 매매동향
    # ──────────────────────────────────────────────
    def download_investor_trading(self, ticker: str) -> pd.DataFrame:
        """기관/외국인/개인 순매수 금액"""
        try:
            df = krx.get_market_trading_value_by_date(
                self.start_date, self.end_date, ticker
            )
            if df.empty:
                return df
            df.index.name = "date"
            # 컬럼: 기관합계, 기타법인, 개인, 외국인합계, 전체
            # 필요한 것만 선택
            cols_needed = []
            for col in df.columns:
                if "기관" in col or "외국인" in col or "개인" in col:
                    cols_needed.append(col)
            if cols_needed:
                df = df[cols_needed]
            return df
        except Exception as e:
            logger.warning(f"{ticker} 투자자 매매동향 수집 실패: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    # 4. 기본 펀더멘탈 (PER, PBR, EPS 등)
    # ──────────────────────────────────────────────
    def download_fundamental(self, ticker: str) -> pd.DataFrame:
        """Trailing PER, PBR, EPS, BPS 등"""
        try:
            df = krx.get_market_fundamental_by_date(
                self.start_date, self.end_date, ticker
            )
            if df.empty:
                return df
            df.index.name = "date"
            return df
        except Exception as e:
            logger.warning(f"{ticker} 펀더멘탈 수집 실패: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    # 5. 전종목 일괄 수집
    # ──────────────────────────────────────────────
    def collect_all(self):
        """전종목 데이터 수집 + parquet 저장 (resume 지원)"""
        tickers_df = self.get_all_tickers()

        # 종목 리스트 저장
        tickers_df.to_csv("data/universe/all_tickers.csv", index=False, encoding="utf-8-sig")

        pending = [t for t in tickers_df["ticker"] if t not in self.completed]
        logger.info(f"수집 대상: {len(pending)}종목 (이미 완료: {len(self.completed)})")

        failed = []

        for ticker in tqdm(pending, desc="📊 데이터 수집"):
            try:
                # OHLCV
                ohlcv = self.download_ohlcv(ticker)
                if ohlcv.empty:
                    logger.debug(f"{ticker}: OHLCV 없음, 건너뜀")
                    continue
                time.sleep(0.5)

                # 투자자 매매동향
                investor = self.download_investor_trading(ticker)
                time.sleep(0.5)

                # 펀더멘탈
                fundamental = self.download_fundamental(ticker)
                time.sleep(0.5)

                # 병합
                combined = ohlcv.copy()
                if not investor.empty:
                    combined = combined.join(investor, how="left")
                if not fundamental.empty:
                    # 펀더멘탈 컬럼명 충돌 방지
                    fund_cols = {c: f"fund_{c}" for c in fundamental.columns}
                    fundamental = fundamental.rename(columns=fund_cols)
                    combined = combined.join(fundamental, how="left")

                # 저장
                save_path = self.raw_dir / f"{ticker}.parquet"
                combined.to_parquet(save_path)
                self.completed.add(ticker)

            except Exception as e:
                logger.error(f"{ticker} 수집 실패: {e}")
                failed.append(ticker)
                time.sleep(1)  # 에러 시 조금 더 대기

        # 실패 종목 기록
        if failed:
            pd.Series(failed).to_csv("data/raw/failed_tickers.csv", index=False)
            logger.warning(f"수집 실패: {len(failed)}종목 → data/raw/failed_tickers.csv")

        logger.info(f"✅ 수집 완료: {len(self.completed)}종목")
        return self.completed


# ──────────────────────────────────────────────
# 샘플 데이터 생성 (pykrx 없을 때 테스트용)
# ──────────────────────────────────────────────
def generate_sample_data(output_dir: str = "data/raw", n_stocks: int = 30, seed: int = 42):
    """
    백테스트 로직 검증용 합성 데이터 생성.
    실제 한국 대형주 스타일의 가격 패턴을 시뮬레이션.
    """
    import numpy as np
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 대형주 모의 종목 (실제 코드/이름은 샘플)
    sample_stocks = [
        ("005930", "삼성전자", "반도체", 70000),
        ("000660", "SK하이닉스", "반도체", 120000),
        ("005380", "현대차", "자동차", 200000),
        ("051910", "LG화학", "화학", 500000),
        ("006400", "삼성SDI", "배터리", 400000),
        ("035420", "NAVER", "IT", 300000),
        ("035720", "카카오", "IT", 60000),
        ("068270", "셀트리온", "바이오", 180000),
        ("105560", "KB금융", "은행", 60000),
        ("055550", "신한지주", "은행", 40000),
        ("003550", "LG", "기타", 80000),
        ("066570", "LG전자", "기타", 100000),
        ("012330", "현대모비스", "자동차", 230000),
        ("028260", "삼성물산", "건설", 130000),
        ("032830", "삼성생명", "기타", 80000),
        ("009150", "삼성전기", "반도체", 150000),
        ("017670", "SK텔레콤", "통신", 50000),
        ("030200", "KT", "통신", 35000),
        ("034730", "SK", "기타", 170000),
        ("010130", "고려아연", "철강", 500000),
        ("036570", "엔씨소프트", "IT", 250000),
        ("003670", "포스코퓨처엠", "배터리", 300000),
        ("086790", "하나금융지주", "은행", 45000),
        ("018260", "삼성에스디에스", "IT", 150000),
        ("011170", "롯데케미칼", "화학", 200000),
        ("096770", "SK이노베이션", "화학", 150000),
        ("015760", "한국전력", "기타", 25000),
        ("000270", "기아", "자동차", 90000),
        ("033780", "KT&G", "식품", 90000),
        ("316140", "우리금융지주", "은행", 14000),
    ]

    stocks = sample_stocks[:n_stocks]
    dates = pd.bdate_range("2019-01-02", "2024-12-30")

    sector_map = {}
    for ticker, name, sector, _ in stocks:
        sector_map[ticker] = {"name": name, "sector": sector}

    for ticker, name, sector, base_price in tqdm(stocks, desc="📊 샘플 데이터 생성"):
        n_days = len(dates)

        # 가격 시뮬레이션: 추세 + 변동성 + 사이클
        trend = np.random.uniform(-0.0001, 0.0003)  # 일간 추세
        volatility = np.random.uniform(0.015, 0.035)  # 일간 변동성

        # 랜덤워크 + 평균회귀 성분
        returns = np.random.normal(trend, volatility, n_days)
        # 사이클 추가 (6개월 주기)
        cycle = 0.15 * np.sin(np.linspace(0, 8 * np.pi, n_days))
        cumulative = np.cumsum(returns) + cycle

        close = base_price * np.exp(cumulative)
        close = np.maximum(close, base_price * 0.3)  # 최소가 보장

        # OHLCV 생성
        daily_range = close * np.random.uniform(0.005, 0.03, n_days)
        high = close + daily_range * np.random.uniform(0.3, 1.0, n_days)
        low = close - daily_range * np.random.uniform(0.3, 1.0, n_days)
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n_days)

        # 거래량: 기본 + 변동 (가격 급변 시 거래량 증가)
        base_vol = np.random.uniform(500000, 5000000)
        vol_noise = np.random.lognormal(0, 0.5, n_days)
        price_change = np.abs(np.diff(close, prepend=close[0])) / close
        vol_spike = 1 + price_change * 30  # 가격 변동 클수록 거래량 증가
        volume = (base_vol * vol_noise * vol_spike).astype(int)
        trading_value = (volume * close).astype(int)

        # 투자자 매매동향 (기관/외국인/개인)
        inst_base = np.random.normal(0, trading_value * 0.1)
        foreign_base = np.random.normal(0, trading_value * 0.08)
        personal = -(inst_base + foreign_base)  # 제로섬

        # 펀더멘탈 (Trailing PER, PBR, EPS)
        sector_per_map = {
            "반도체": 15, "자동차": 7, "배터리": 25, "바이오": 30,
            "은행": 5.5, "화학": 10, "철강": 6, "IT": 18,
            "통신": 10, "건설": 7, "식품": 14, "기타": 11,
        }
        base_per = sector_per_map.get(sector, 11)
        per_series = base_per + np.random.normal(0, base_per * 0.15, n_days)
        per_series = np.maximum(per_series, 2)  # 최소 PER 2

        eps_series = close / per_series
        pbr_series = np.random.uniform(0.5, 3.0, n_days) + np.random.normal(0, 0.1, n_days)

        df = pd.DataFrame({
            "open": np.round(open_price).astype(int),
            "high": np.round(high).astype(int),
            "low": np.round(low).astype(int),
            "close": np.round(close).astype(int),
            "volume": volume,
            "trading_value": trading_value,
            "price_change": np.round(np.diff(close, prepend=close[0])).astype(int),
            "기관합계": np.round(inst_base).astype(int),
            "외국인합계": np.round(foreign_base).astype(int),
            "개인": np.round(personal).astype(int),
            "fund_BPS": np.round(close * np.random.uniform(0.5, 1.5, n_days)).astype(int),
            "fund_PER": np.round(per_series, 2),
            "fund_PBR": np.round(pbr_series, 2),
            "fund_EPS": np.round(eps_series).astype(int),
            "fund_DIV": np.round(np.random.uniform(0.5, 4.0, n_days), 2),
            "fund_DPS": np.round(close * np.random.uniform(0.005, 0.03, n_days)).astype(int),
        }, index=dates[:n_days])
        df.index.name = "date"

        save_path = output_dir / f"{ticker}.parquet"
        df.to_parquet(save_path)

    # 종목 메타정보 저장
    universe_dir = Path("data/universe")
    universe_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.DataFrame([
        {"ticker": t, "name": n, "market": "KOSPI", "sector": s, "base_price": p}
        for t, n, s, p in stocks
    ])
    meta.to_csv(universe_dir / "all_tickers.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(universe_dir / "sector_map.csv", index=False, encoding="utf-8-sig")

    logger.info(f"✅ 샘플 데이터 생성 완료: {n_stocks}종목")
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if PYKRX_AVAILABLE:
        collector = DataCollector()
        collector.collect_all()
    else:
        print("⚠️ pykrx 미설치 → 샘플 데이터로 대체 생성")
        generate_sample_data()
