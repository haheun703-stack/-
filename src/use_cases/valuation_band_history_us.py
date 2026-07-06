"""US 역사 PER 밴드 (자기 5년 이력 대비 현재 밸류에이션 위치) — 미국판 미래가치 엔진 V축.

한국 valuation_band_history.py의 이식판. 로직(TTM EPS·point-in-time lag·흑자게이트·5년 분위
pct_rank·recency 가드)은 동일하되, 데이터원만 교체:
  - 한국: dart_cache/fundamentals_historical.csv(10년 분기 EPS) + processed/{ticker}.parquet(일봉)
  - 미국: yfinance(전부) — 분기 재무 이력이 ~5분기로 짧아, **연간 Diluted EPS(최근 4년 백본)**
    + **분기 rolling TTM(최근 정밀화)** 하이브리드로 TTM EPS 스텝 시계열을 구성.
    가격은 yfinance history(6y) 일봉.

★데이터 검증(7/6 VPS 실측): 분기합 TTM(JNJ 11.05) ≈ 연간 Diluted EPS(11.03) 정합.
  연간은 최고참 연도가 NaN이라 유효 ~4년 → coverage_years ≈ 4(reliable 임계 3년 통과).

★★ VPS 전용(yfinance rate-limit) · `./venv/bin/python3.11`. 캐시(data/us_market/per_band/)로
  fetch 1회 후 백테스트 point-in-time(as_of) 재사용 — leader_cycle과 동형.

Usage:
    ./venv/bin/python3.11 src/use_cases/valuation_band_history_us.py --fetch   # 캐시 수집
    ./venv/bin/python3.11 src/use_cases/valuation_band_history_us.py           # 밴드 출력
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:  # 프로젝트 규칙(새 스크립트 load_dotenv 필수) — yfinance 비밀키 불필요하나 관례 준수
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "us_market" / "per_band"

# 한국 모듈과 동일 파라미터(파리티)
DISCLOSURE_LAG_DAYS = 60      # 결산일 → 공시 가용일 (US 10-Q ~40d·10-K ~75d, 60=lookahead 안전 절충)
BAND_WINDOW_YEARS = 5
MIN_BAND_DAYS = 250
LOW_PCT, HIGH_PCT = 0.25, 0.75
RECENCY_MAX_DAYS = 20
HISTORY_PERIOD = "6y"         # 5년 밴드 + lag 여유

_ttm_cache: dict[str, pd.Series | None] = {}
_close_cache: dict[str, pd.Series | None] = {}
_meta_cache: dict[str, dict | None] = {}

# basis 가드: 내 계산 PER vs yfinance trailingPE 허용 배율. 벗어나면 통화/ADR-비율 불일치
#   (해외통화 재무 ADR: TSM=TWD·NVO=DKK·TM=JPY → USD가격과 불일치 → PER 오염) → 제외.
#   진짜 이익급감 고PER(ABBV 등)은 yfinance trailingPE도 같이 높아 배율≈1 → 보존.
BASIS_LO, BASIS_HI = 0.4, 2.5


# ═══════════════════════════════════════════════
#  캐시 수집 (yfinance → 디스크)
# ═══════════════════════════════════════════════

def _build_ttm_eps_from_yf(t) -> pd.Series | None:
    """yfinance Ticker → TTM EPS 시계열(결산일 색인, lag 미적용). 연간 백본 + 분기 rolling TTM.

    - 연간 Diluted EPS: 각 연도값 = 그 회계연도 TTM EPS(결산일 색인).
    - 분기 Diluted EPS: 4개 연속분기 합 = rolling TTM(최근 구간 정밀화). 동일 결산일은
      분기 우선(더 최신/정밀). 4분기 span 240~300일만 인정(결측분기 오염 방지 = 한국 연속성 가드).
    """
    points: dict[pd.Timestamp, float] = {}
    # 연간 백본
    try:
        a = t.income_stmt
        if a is not None and "Diluted EPS" in a.index:
            for col, val in a.loc["Diluted EPS"].items():
                if pd.notna(val):
                    points[pd.Timestamp(col).normalize()] = float(val)
    except Exception:  # noqa: BLE001
        pass
    # 분기 rolling TTM (최근 정밀화 — 연간 위에 덮어씀)
    try:
        q = t.quarterly_income_stmt
        if q is not None and "Diluted EPS" in q.index:
            qs = q.loc["Diluted EPS"].dropna().sort_index()  # 오름차순
            vals, idx = qs.values, qs.index
            for i in range(3, len(qs)):
                span = (pd.Timestamp(idx[i]) - pd.Timestamp(idx[i - 3])).days
                if 240 <= span <= 300:  # 4개 연속분기(≈9개월 span)만
                    points[pd.Timestamp(idx[i]).normalize()] = float(vals[i - 3:i + 1].sum())
    except Exception:  # noqa: BLE001
        pass

    if not points:
        return None
    s = pd.Series(points).sort_index()
    return s[~s.index.duplicated(keep="last")]


def fetch_and_cache(tickers: list[str], delay: float = 0.5) -> int:
    """유니버스 close(6y 일봉) + TTM EPS 캐시. 반환: 성공 종목 수."""
    import yfinance as yf
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, tk in enumerate(tickers, 1):
        try:
            t = yf.Ticker(tk)
            hist = t.history(period=HISTORY_PERIOD)
            if hist is None or hist.empty or "Close" not in hist.columns:
                logger.warning("스킵 %s: 가격 이력 없음", tk)
                continue
            close = pd.Series(hist["Close"].values,
                              index=pd.to_datetime(hist.index).tz_localize(None)).sort_index()
            ttm = _build_ttm_eps_from_yf(t)
            if ttm is None or ttm.empty:
                logger.warning("스킵 %s: EPS 이력 없음", tk)
                continue
            close.to_frame("Close").to_parquet(CACHE_DIR / f"{tk}_close.parquet")
            (CACHE_DIR / f"{tk}_ttm.json").write_text(json.dumps(
                {d.strftime("%Y-%m-%d"): v for d, v in ttm.items()}), encoding="utf-8")

            # basis 가드 메타: 최신 내 PER vs yfinance trailingPE 대조(통화/ADR-비율 불일치 탐지)
            my_per = None
            if ttm[ttm > 0].size:
                latest_eps = float(ttm.sort_index().iloc[-1])
                if latest_eps > 0:
                    my_per = float(close.iloc[-1]) / latest_eps
            yf_per = None
            try:
                yf_per = t.info.get("trailingPE")
            except Exception:  # noqa: BLE001
                pass
            basis_ok = True
            if my_per and yf_per and yf_per > 0:
                ratio = my_per / yf_per
                basis_ok = BASIS_LO <= ratio <= BASIS_HI
            (CACHE_DIR / f"{tk}_meta.json").write_text(json.dumps(
                {"basis_ok": basis_ok, "my_current_per": round(my_per, 2) if my_per else None,
                 "yf_trailing_pe": round(yf_per, 2) if yf_per else None}), encoding="utf-8")

            ok += 1
            logger.info("[%d/%d] %s: close %d행(%s~) · TTM %d점%s",
                        i, len(tickers), tk, len(close),
                        close.index[0].strftime("%Y-%m"), len(ttm),
                        "" if basis_ok else "  ⚠️basis불일치(통화/ADR)→제외")
        except Exception as e:  # noqa: BLE001
            logger.warning("실패 %s: %s", tk, e)
        finally:
            if i < len(tickers):
                time.sleep(delay)
    return ok


# ═══════════════════════════════════════════════
#  캐시 로드 + PER 밴드 (한국 compute_per_band 동형)
# ═══════════════════════════════════════════════

def _load_ttm(ticker: str) -> pd.Series | None:
    """TTM EPS 시계열(공시 가용일=결산일+lag 색인)."""
    if ticker in _ttm_cache:
        return _ttm_cache[ticker]
    p = CACHE_DIR / f"{ticker}_ttm.json"
    if not p.exists():
        _ttm_cache[ticker] = None
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in raw.items()}).sort_index()
    s.index = s.index + pd.Timedelta(days=DISCLOSURE_LAG_DAYS)  # 결산일 → 가용일(point-in-time)
    s = s[~s.index.duplicated(keep="last")]
    _ttm_cache[ticker] = s if len(s) else None
    return _ttm_cache[ticker]


def _load_meta(ticker: str) -> dict | None:
    if ticker in _meta_cache:
        return _meta_cache[ticker]
    p = CACHE_DIR / f"{ticker}_meta.json"
    _meta_cache[ticker] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _meta_cache[ticker]


def _load_close(ticker: str) -> pd.Series | None:
    if ticker in _close_cache:
        return _close_cache[ticker]
    p = CACHE_DIR / f"{ticker}_close.parquet"
    if not p.exists():
        _close_cache[ticker] = None
        return None
    try:
        df = pd.read_parquet(p)
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        _close_cache[ticker] = pd.Series(df["Close"].values, index=idx).sort_index()
    except Exception:  # noqa: BLE001
        _close_cache[ticker] = None
    return _close_cache[ticker]


def _per_series(ticker: str, as_of: pd.Timestamp | None = None) -> pd.Series | None:
    """일별 트레일링 PER (TTM EPS>0 구간만). as_of 이후 절단(point-in-time, 백테스트용)."""
    ttm = _load_ttm(ticker)
    close = _load_close(ticker)
    if ttm is None or close is None or close.empty:
        return None
    if as_of is not None:
        close = close[close.index <= as_of]
        ttm = ttm[ttm.index <= as_of]
    if close.empty or ttm.empty:
        return None
    eps_aligned = ttm.reindex(close.index, method="ffill")
    per = close / eps_aligned
    per = per[(eps_aligned > 0) & np.isfinite(per) & (per > 0) & (per < 500)]
    return per if len(per) else None


def compute_per_band_us(ticker: str, as_of: pd.Timestamp | str | None = None) -> dict | None:
    """US 종목 역사 PER 밴드 + 현재 위치. 한국 compute_per_band와 동일 반환 스키마."""
    if isinstance(as_of, str):
        as_of = pd.Timestamp(as_of)
    # basis 가드: 통화/ADR-비율 불일치 종목(TSM·NVO·TM 등)은 PER 밴드 무의미 → 제외
    meta = _load_meta(ticker)
    if meta is not None and not meta.get("basis_ok", True):
        return None
    per = _per_series(ticker, as_of)
    if per is None:
        return None
    close = _load_close(ticker)
    ref_date = close.index[-1] if close is not None and len(close) else per.index[-1]
    if as_of is not None:
        ref_date = min(ref_date, as_of)
    if (ref_date - per.index[-1]).days > RECENCY_MAX_DAYS:
        return None
    window_start = per.index[-1] - pd.Timedelta(days=int(BAND_WINDOW_YEARS * 365.25))
    band = per[per.index >= window_start]
    if len(band) < MIN_BAND_DAYS:
        return None
    current = float(band.iloc[-1])
    p25, p50, p75 = (float(band.quantile(q)) for q in (0.25, 0.50, 0.75))
    pct_rank = float((band < current).mean())
    coverage_years = round((band.index[-1] - band.index[0]).days / 365.25, 1)

    ttm = _load_ttm(ticker)
    if as_of is not None and ttm is not None:
        ttm = ttm[ttm.index <= as_of]
    pos_ratio = float((ttm > 0).mean()) if ttm is not None and len(ttm) else 0.0

    if pct_rank <= LOW_PCT:
        signal = "저평가(밴드하단)"
    elif pct_rank >= HIGH_PCT:
        signal = "고평가(밴드상단)"
    else:
        signal = "중립"
    reliable = coverage_years >= 3.0 and pos_ratio >= 0.8

    return {
        "current_per": round(current, 1),
        "band_p25": round(p25, 1), "band_median": round(p50, 1), "band_p75": round(p75, 1),
        "pct_rank": round(pct_rank, 3),
        "n_days": len(band), "coverage_years": coverage_years,
        "earnings_positive_ratio": round(pos_ratio, 2),
        "signal": signal, "reliable": reliable,
    }


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def _default_universe() -> list[str]:
    # 클린아키텍처: use_cases 소유 정의 사용(scripts 역참조 금지)
    try:
        from src.use_cases.valuation_band import us_fv_universe
        return us_fv_universe()
    except Exception:  # noqa: BLE001
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "JNJ", "MU", "AVGO", "LLY", "ORCL"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="US 역사 PER 밴드")
    parser.add_argument("--fetch", action="store_true", help="yfinance 캐시 수집")
    parser.add_argument("--ticker", type=str, default=None, help="단일 종목")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    universe = [args.ticker] if args.ticker else _default_universe()

    if args.fetch:
        n = fetch_and_cache(universe, delay=args.delay)
        logger.info("캐시 완료: %d/%d종목 → %s", n, len(universe), CACHE_DIR)
        return

    logger.info("%-6s %8s %8s %8s %8s %7s %6s %8s %s", "종목", "현재PER", "p25", "중앙값", "p75",
                "pct_rank", "커버년", "흑자비율", "신호/신뢰")
    logger.info("-" * 90)
    for tk in universe:
        b = compute_per_band_us(tk)
        if not b:
            logger.info("%-6s  (데이터부족/캐시없음)", tk)
            continue
        logger.info("%-6s %8.1f %8.1f %8.1f %8.1f %7.2f %6.1f %8.2f  %s%s",
                    tk, b["current_per"], b["band_p25"], b["band_median"], b["band_p75"],
                    b["pct_rank"], b["coverage_years"], b["earnings_positive_ratio"],
                    b["signal"], " ★reliable" if b["reliable"] else "")


if __name__ == "__main__":
    main()
