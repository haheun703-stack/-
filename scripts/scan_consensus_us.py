"""US 컨센서스 스크리너 — 미국 대형주 중기 투자 후보 자동 발굴 (한국 scan_consensus 미러).

한국판(wisereport 스크래핑)을 yfinance-native로 이식. yfinance `Ticker.info`의
애널리스트 컨센서스(목표주가/투자의견/Forward EPS/PER)를 수집해, 한국 consensus_screening과
**동일 스키마**(ticker/name/close/target_price/upside_pct/forward_eps/forward_per/…)로 산출.
→ 미국판 미래가치 엔진(future_value_engine_us)의 V축 입력.

한국 대비 차이:
  - 유니버스: KRX CSV 대신 US 대형주 고정 풀(valuation_band.US_POOL 35 + 리더 반도체/제약).
  - 데이터원: wisereport HTML 스크래퍼 대신 yfinance.info(목표가·forwardEps·recommendationMean).
  - 투자의견: recommendationMean(1=강력매수~5=매도, 낮을수록 좋음)을 6-mean으로 뒤집어
    한국식 opinion_score(높을수록 좋음, 4+=강력)로 정규화.
  - 우선주/스팩 제외 로직 불필요(US 대형주 풀은 전부 보통주).
  - 기술적 축은 v1에서 제외(정직) — 컨센서스 4축(상승여력·forwardPER·애널리스트·배당)만.
    us_daily 기반 기술축은 백테스트(④)에서 예측력 확인 후 추가.

★★ VPS 전용 — 로컬은 yfinance rate-limit 차단(고정IP만 통과). `./venv/bin/python3.11` 사용.

Usage:
    ./venv/bin/python3.11 scripts/scan_consensus_us.py                 # 기본 (TOP 15)
    ./venv/bin/python3.11 scripts/scan_consensus_us.py --top 20
    ./venv/bin/python3.11 scripts/scan_consensus_us.py --ticker NVDA   # 단일 종목
    ./venv/bin/python3.11 scripts/scan_consensus_us.py --min-upside 15
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 프로젝트 규칙(새 스크립트 load_dotenv 필수). yfinance는 비밀키 불필요하나 관례 준수·향후 안전.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "consensus_screening_us.json"

# ═══════════════════════════════════════════════
#  US 유니버스 (대형주 고정 풀)
#    valuation_band.US_POOL(시총 상위 35) + 미래가치 리더(반도체·제약, config/global_leaders
#    US 섹션 중 US-상장·애널리스트 커버 우량). 해외 핑크시트(RHM.DE·CATL·BASFY·NTDOY·TCEHY)는
#    yfinance 컨센서스 신뢰도 낮아 제외 — v1은 US-상장/주요 ADR만(정직).
# ═══════════════════════════════════════════════

def build_universe() -> list[str]:
    """US 대형주 유니버스 티커 리스트 — use_cases 소유 정의 재사용(클린아키텍처)."""
    try:
        from src.use_cases.valuation_band import us_fv_universe
        return us_fv_universe()
    except Exception:  # noqa: BLE001 — import 실패 시 최소 풀로 폴백
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "LLY"]


def _setup_ssl() -> None:
    """yfinance/requests SSL — 한글 경로 cacert를 ASCII 임시경로로 복사(윈도우 error 77 우회).

    리눅스(VPS)에선 무해·무동작. valuation_band._setup_ssl과 동일 패턴(파리티).
    """
    try:
        import certifi
        ascii_ca = os.path.join(os.environ.get("TEMP", "/tmp"), "cacert_ascii.pem")
        if not os.path.exists(ascii_ca):
            shutil.copy(certifi.where(), ascii_ca)
        for k in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
            os.environ[k] = ascii_ca
    except Exception:  # noqa: BLE001
        pass


# ═══════════════════════════════════════════════
#  컨센서스 수집 (yfinance.info)
# ═══════════════════════════════════════════════

def _opinion_from_mean(rec_mean: float | None) -> float | None:
    """recommendationMean(1=강력매수~5=매도) → 한국식 opinion_score(높을수록 좋음).

    6 - mean 변환: 강력매수 1.0→5.0, 매수 2.0→4.0, 중립 3.0→3.0. 한국 opinion_score와
    동일 스케일(4.0+ = 강력매수권)이라 하류 스코어링/필터 재사용 가능.
    """
    if rec_mean is None or rec_mean <= 0:
        return None
    return round(6.0 - float(rec_mean), 2)


def fetch_consensus(tickers: list[str], delay: float = 0.5) -> list[dict]:
    """yfinance.info에서 컨센서스 수집. 목표가·애널리스트수 없는 종목은 graceful skip."""
    _setup_ssl()
    import yfinance as yf

    out: list[dict] = []
    for i, tk in enumerate(tickers, 1):
        try:
            info = yf.Ticker(tk).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            target = info.get("targetMeanPrice")
            n_analysts = info.get("numberOfAnalystOpinions")

            # poison-pill 방어(7/5 교훈): 핵심 필드 없으면 조용히 제외(가짜 채움 금지)
            if not price or price <= 0 or not target or target <= 0:
                logger.debug("스킵 %s: price/target 결측", tk)
                continue

            fwd_eps = info.get("forwardEps")
            fwd_per = info.get("forwardPE")
            # forwardPE 결측 시 price/forwardEps로 보완(둘 다 양수일 때만)
            if (not fwd_per or fwd_per <= 0) and fwd_eps and fwd_eps > 0:
                fwd_per = round(price / fwd_eps, 2)

            # dividendYield 단위 = 퍼센트(VPS 실측 확인: ACN 4.8·PFE 7.1·AAPL 0.35 = 그대로 %).
            #   ★분수 반환형(0.048)이면 하류 임계(>=3,>=4) 사망 — 실측상 퍼센트라 그대로 사용.
            div_y = info.get("dividendYield")

            out.append({
                "ticker": tk,
                "name": (info.get("shortName") or tk)[:24],
                "close": round(float(price), 2),
                "target_price": round(float(target), 2),
                "target_high": info.get("targetHighPrice"),
                "opinion_score": _opinion_from_mean(info.get("recommendationMean")),
                "recommendation": info.get("recommendationKey"),
                "analyst_count": int(n_analysts) if n_analysts else 0,
                "forward_eps": round(float(fwd_eps), 2) if fwd_eps else None,
                "forward_per": round(float(fwd_per), 2) if fwd_per and fwd_per > 0 else None,
                "trailing_per": round(float(info["trailingPE"]), 2) if info.get("trailingPE") else None,
                "dividend_yield": round(float(div_y), 2) if div_y else 0.0,
                "market_cap": info.get("marketCap"),
            })
        except Exception as e:  # noqa: BLE001
            logger.debug("수집 실패 %s: %s", tk, e)
        finally:
            if i < len(tickers):
                time.sleep(delay)  # rate-limit 방어
    return out


# ═══════════════════════════════════════════════
#  복합 스코어링 (컨센서스 4축 100점)
#    한국 5축에서 기술축(20점) 제거 후 재분배 — US 기술데이터는 백테스트 검증 후 v2 추가.
#    축1 상승여력 40 · 축2 forwardPER 25 · 축3 애널리스트 확신 20 · 축4 배당 15.
# ═══════════════════════════════════════════════

def calc_composite_score(item: dict) -> float:
    score = 0.0

    # 축1: 컨센서스 상승여력 (40점) — 30%+면 만점
    upside = item.get("upside_pct", 0)
    if upside > 0:
        score += min(40.0, upside / 30 * 40)

    # 축2: Forward PER 매력도 (25점) — PER 5이하 만점, 30이상 0점(한국과 동일 로그 스케일)
    fper = item.get("forward_per")
    if fper and fper > 0:
        if fper <= 5:
            score += 25.0
        elif fper <= 30:
            score += max(0.0, 25.0 * (1 - np.log(fper / 5) / np.log(6)))

    # 축3: 애널리스트 확신도 (20점) — 의견(14) + 기관수(6)
    opinion = item.get("opinion_score") or 0
    analysts = item.get("analyst_count") or 0
    if opinion >= 4.0:
        score += 14.0
    elif opinion >= 3.5:
        score += 7.0
    if analysts >= 15:
        score += 6.0
    elif analysts >= 5:
        score += 6.0 * (analysts / 15)

    # 축4: 배당 보너스 (15점) — 배당수익률 3%+ 만점
    div_yield = item.get("dividend_yield", 0)
    if div_yield >= 3:
        score += 15.0
    elif div_yield > 0:
        score += min(15.0, div_yield / 3 * 15)

    return round(score, 1)


def grade_from_score(score: float) -> str:
    if score >= 80:
        return "S"
    if score >= 65:
        return "A"
    if score >= 50:
        return "B"
    if score >= 35:
        return "C"
    return "D"


# ═══════════════════════════════════════════════
#  메인 스캔 파이프라인
# ═══════════════════════════════════════════════

def run_scan(
    top_n: int = 15,
    min_upside: float = 5,
    min_analysts: int = 5,
    single_ticker: str | None = None,
    delay: float = 0.5,
) -> dict:
    start = time.time()

    universe = [single_ticker] if single_ticker else build_universe()
    logger.info("US 컨센서스 수집 시작 (%d종목, delay=%.1f초) — VPS yfinance", len(universe), delay)

    consensus = fetch_consensus(universe, delay=delay)
    logger.info("컨센서스 보유: %d/%d종목", len(consensus), len(universe))

    # ★전 컨센서스 종목 스코어링(필터 무관) — all_picks는 FV 엔진 유니버스라 밸류트랩
    #   (음수 괴리·저상승여력 방어주)까지 포함해야 엔진이 고평가 회피를 평가 가능(렌즈1 #4).
    scored = []
    for c in consensus:
        close = c["close"]
        target = c["target_price"]
        upside = round((target / close - 1) * 100, 1)
        item = dict(c)
        item["upside_pct"] = upside
        item["composite_score"] = calc_composite_score(item)
        item["grade"] = grade_from_score(item["composite_score"])
        scored.append(item)
    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    # 필터 통과분 = 랭킹 리포트/스크리닝용(TOP N). 엔진은 all_picks(전체) 사용.
    passed = [p for p in scored if single_ticker
              or (p["upside_pct"] >= min_upside and (p.get("analyst_count") or 0) >= min_analysts)]
    top_picks = passed[:top_n]
    elapsed = round(time.time() - start, 1)

    result = {
        "scan_date": datetime.now().strftime("%Y-%m-%d"),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market": "US",
        "universe_size": len(universe),
        "with_consensus": len(consensus),
        "passed_filter": len(passed),
        "elapsed_sec": elapsed,
        "top_picks": [],
        "all_picks": [],
    }

    # ── 콘솔 리포트 ──
    logger.info("")
    logger.info("=" * 92)
    logger.info(" US 컨센서스 스크리닝 (%s)", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info(" 유니버스 %d → 컨센서스 %d → 필터통과 %d → TOP %d (%.1f초)",
                len(universe), len(consensus), len(passed), len(top_picks), elapsed)
    logger.info("=" * 92)
    logger.info("%2s %-8s %9s %10s %8s %6s %5s %7s %6s %5s %4s",
                "#", "종목", "종가", "목표가", "상승여력", "의견", "기관", "F-PER", "배당", "점수", "등급")
    logger.info("-" * 92)
    for idx, p in enumerate(top_picks, 1):
        op = f"{p['opinion_score']:.2f}" if p.get("opinion_score") else "N/A"
        fp = f"{p['forward_per']:.1f}" if p.get("forward_per") else "N/A"
        dv = f"{p['dividend_yield']:.1f}%" if p.get("dividend_yield", 0) > 0 else "-"
        logger.info("%2d %-8s %9.2f %10.2f %+7.1f%% %6s %5d %7s %6s %5.1f %4s",
                    idx, p["ticker"], p["close"], p["target_price"], p["upside_pct"],
                    op, p.get("analyst_count", 0), fp, dv, p["composite_score"], p["grade"])
        pj = dict(p)
        pj["rank"] = idx
        result["top_picks"].append(pj)
    logger.info("")

    result["all_picks"] = scored
    logger.info("전체 풀: %d종목 (future_value_engine_us 연동용·필터무관 밸류트랩 포함)", len(scored))

    # ── JSON 저장 ──
    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"{type(o).__name__} not serializable")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_default)
    logger.info("저장: %s", OUTPUT_PATH)

    return result


def main():
    parser = argparse.ArgumentParser(description="US 컨센서스 스크리너 (yfinance)")
    parser.add_argument("--top", type=int, default=15, help="TOP N (default: 15)")
    parser.add_argument("--min-upside", type=float, default=5, help="최소 상승여력 %% (default: 5)")
    parser.add_argument("--min-analysts", type=int, default=5, help="최소 애널리스트 수 (default: 5)")
    parser.add_argument("--ticker", type=str, default=None, help="단일 종목(진단, 필터 우회)")
    parser.add_argument("--delay", type=float, default=0.5, help="종목간 지연 초 (rate-limit)")
    args = parser.parse_args()

    run_scan(
        top_n=args.top,
        min_upside=args.min_upside,
        min_analysts=args.min_analysts,
        single_ticker=args.ticker,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
