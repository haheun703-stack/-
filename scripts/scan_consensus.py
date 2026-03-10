"""컨센서스 스크리너 — 3~6개월 중기 투자 후보 자동 발굴

wisereport 컨센서스(목표주가/투자의견/Forward EPS) + 기술적 분석 결합.
5축 100점 복합 스코어로 상승여력 TOP 종목 자동 리포트.

Usage:
    python scripts/scan_consensus.py                    # 기본 (TOP 15)
    python scripts/scan_consensus.py --top 20           # TOP 20
    python scripts/scan_consensus.py --send             # 텔레그램 발송
    python scripts/scan_consensus.py --ticker 000660    # 단일 종목
    python scripts/scan_consensus.py --min-upside 15    # 최소 상승여력
    python scripts/scan_consensus.py --no-tech          # 기술적 스킵
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CSV_DIR = PROJECT_ROOT / "stock_data_daily"
PARQUET_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "data" / "consensus_screening.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ═══════════════════════════════════════════════
#  1. 유니버스 구성 (CSV 기반 대형주 필터)
# ═══════════════════════════════════════════════

def build_universe(min_turnover: float = 100, min_price: int = 5000) -> list[dict]:
    """CSV 전종목에서 대형주 유니버스 구성.

    Args:
        min_turnover: 최소 거래대금 (억원)
        min_price: 최소 종가

    Returns:
        [{"ticker": "000660", "name": "SK하이닉스", "close": 938000, "volume": ...}, ...]
    """
    universe = []

    for f in glob.glob(str(CSV_DIR / "*.csv")):
        try:
            bn = os.path.basename(f).replace(".csv", "")
            parts = bn.rsplit("_", 1)
            if len(parts) != 2:
                continue
            name, ticker = parts

            # 우선주/스팩/ETF 제외
            if ticker.endswith(("5", "7", "8", "9")) and len(ticker) == 6:
                # 일반 종목도 5/7로 끝날 수 있으므로 이름 기반 필터
                pass
            if any(kw in name for kw in ["스팩", "SPAC", "ETN"]):
                continue

            df = pd.read_csv(f)
            if len(df) < 60:
                continue

            last = df.iloc[-1]
            close = last.get("Close", 0)
            volume = last.get("Volume", 0)

            if close < min_price or volume < 10000:
                continue

            turnover = close * volume / 1e8  # 억원
            if turnover < min_turnover:
                continue

            universe.append({
                "ticker": ticker,
                "name": name,
                "close": int(close),
                "volume": int(volume),
                "turnover": round(turnover, 0),
            })

        except Exception:
            continue

    logger.info(f"유니버스: {len(universe)}개 (거래대금 {min_turnover}억+, 종가 {min_price}원+)")
    return universe


# ═══════════════════════════════════════════════
#  2. 기술적 분석 (parquet/CSV)
# ═══════════════════════════════════════════════

def get_technical(ticker: str, close: int) -> dict:
    """parquet 또는 CSV에서 기술적 지표 추출."""
    tech = {
        "rsi": None,
        "above_ma20": None,
        "above_ma60": None,
        "above_ma120": None,
        "sar_trend": None,
        "dist_high_52w": None,
    }

    # parquet 우선
    pq_path = PARQUET_DIR / f"{ticker}.parquet"
    if pq_path.exists():
        try:
            pq = pd.read_parquet(pq_path)
            if len(pq) > 0:
                last = pq.iloc[-1]
                if "rsi14" in pq.columns:
                    tech["rsi"] = round(float(last.get("rsi14", 0)), 1) or None
                if "ma20" in pq.columns:
                    tech["above_ma20"] = close > float(last.get("ma20", 0))
                if "ma60" in pq.columns:
                    tech["above_ma60"] = close > float(last.get("ma60", 0))
                if "ma120" in pq.columns:
                    ma120 = float(last.get("ma120", 0))
                    tech["above_ma120"] = close > ma120 if ma120 > 0 else None
                if "sar_trend" in pq.columns:
                    tech["sar_trend"] = int(last.get("sar_trend", 0))

                # 52주 고점
                closes = pq["stck_clpr"].tail(252) if "stck_clpr" in pq.columns else None
                if closes is not None and len(closes) > 0:
                    high_52 = closes.max()
                    if high_52 > 0:
                        tech["dist_high_52w"] = round((close / high_52 - 1) * 100, 1)
            return tech
        except Exception:
            pass

    # CSV fallback
    csv_matches = list(CSV_DIR.glob(f"*_{ticker}.csv"))
    if csv_matches:
        try:
            df = pd.read_csv(csv_matches[0])
            if len(df) >= 60:
                closes = df["Close"].tail(252)
                tech["above_ma20"] = close > closes.tail(20).mean()
                tech["above_ma60"] = close > closes.tail(60).mean()
                if len(closes) >= 120:
                    tech["above_ma120"] = close > closes.tail(120).mean()
                high_52 = df["High"].tail(252).max() if "High" in df.columns else closes.max()
                if high_52 > 0:
                    tech["dist_high_52w"] = round((close / high_52 - 1) * 100, 1)

                # RSI 간이 계산
                delta = df["Close"].diff().tail(15)
                gain = delta.where(delta > 0, 0).mean()
                loss = (-delta.where(delta < 0, 0)).mean()
                if loss > 0:
                    tech["rsi"] = round(100 - (100 / (1 + gain / loss)), 1)
        except Exception:
            pass

    return tech


# ═══════════════════════════════════════════════
#  3. 복합 스코어링 (5축 100점)
# ═══════════════════════════════════════════════

def calc_composite_score(item: dict, cfg: dict) -> float:
    """5축 복합 스코어 계산.

    축1: 컨센서스 상승여력 (35점) — 30%+면 만점
    축2: Forward PER 매력도 (20점) — PER 5이하 만점, 30이상 0점
    축3: 애널리스트 확신도 (15점) — 의견 4.0+/기관수 15+
    축4: 기술적 건전성 (20점) — MA60↑(8) + RSI 30~65(6) + SAR↑(6)
    축5: 배당 보너스 (10점) — 배당수익률 3%+
    """
    w = cfg.get("scoring", {})
    w1 = w.get("upside_weight", 35)
    w2 = w.get("per_weight", 20)
    w3 = w.get("analyst_weight", 15)
    w4 = w.get("technical_weight", 20)
    w5 = w.get("dividend_weight", 10)

    score = 0.0

    # 축1: 상승여력
    upside = item.get("upside_pct", 0)
    if upside > 0:
        score += min(w1, upside / 30 * w1)

    # 축2: Forward PER 매력도
    fper = item.get("forward_per")
    if fper and fper > 0:
        if fper <= 5:
            score += w2
        elif fper <= 30:
            # 로그 스케일: PER 5=만점, PER 30=0점
            score += max(0, w2 * (1 - np.log(fper / 5) / np.log(6)))
        # PER 30 이상: 0점

    # 축3: 애널리스트 확신도
    opinion = item.get("opinion_score") or 0
    analysts = item.get("analyst_count") or 0
    # 의견 점수 (0~10점)
    if opinion >= 4.0:
        score += w3 * 0.67  # 10점
    elif opinion >= 3.5:
        score += w3 * 0.33  # 5점
    # 기관수 (0~5점)
    if analysts >= 15:
        score += w3 * 0.33
    elif analysts >= 5:
        score += w3 * 0.33 * (analysts / 15)

    # 축4: 기술적 건전성
    tech = item.get("technical", {})
    if tech.get("above_ma60"):
        score += w4 * 0.4  # 8점
    rsi = tech.get("rsi")
    if rsi and 30 <= rsi <= 65:
        score += w4 * 0.3  # 6점
    elif rsi and 65 < rsi <= 75:
        score += w4 * 0.15  # 3점
    if tech.get("sar_trend") == 1:
        score += w4 * 0.3  # 6점

    # 축5: 배당 보너스
    div_yield = item.get("dividend_yield", 0)
    if div_yield >= 3:
        score += w5
    elif div_yield > 0:
        score += min(w5, div_yield / 3 * w5)

    return round(score, 1)


def grade_from_score(score: float) -> str:
    if score >= 80:
        return "S"
    elif score >= 65:
        return "A"
    elif score >= 50:
        return "B"
    elif score >= 35:
        return "C"
    return "D"


# ═══════════════════════════════════════════════
#  4. 메인 스캔 파이프라인
# ═══════════════════════════════════════════════

def run_scan(
    top_n: int = 15,
    min_upside: float = 10,
    single_ticker: str | None = None,
    skip_tech: bool = False,
    send_telegram: bool = False,
) -> dict:
    """컨센서스 스크리닝 전체 파이프라인."""

    from src.adapters.consensus_scraper import ConsensusScraper

    # 설정 로드
    cfg = {}
    if SETTINGS_PATH.exists():
        import yaml
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            all_cfg = yaml.safe_load(f) or {}
            cfg = all_cfg.get("consensus_scanner", {})

    min_turnover = cfg.get("min_turnover_억", 100)
    min_price = cfg.get("min_price", 5000)
    delay = cfg.get("fetch_delay_sec", 0.5)
    thresholds = cfg.get("thresholds", {})
    min_upside = thresholds.get("min_upside_pct", min_upside)
    min_analysts = thresholds.get("min_analyst_count", 3)
    min_opinion = thresholds.get("min_opinion_score", 3.0)

    start_time = time.time()

    # ── 1. 유니버스 구성 ──
    if single_ticker:
        # 단일 종목 모드
        csv_match = list(CSV_DIR.glob(f"*_{single_ticker}.csv"))
        if csv_match:
            bn = csv_match[0].stem
            name = bn.rsplit("_", 1)[0]
            df = pd.read_csv(csv_match[0])
            close = int(df.iloc[-1]["Close"])
            universe = [{"ticker": single_ticker, "name": name, "close": close, "volume": 0, "turnover": 0}]
        else:
            universe = [{"ticker": single_ticker, "name": "", "close": 0, "volume": 0, "turnover": 0}]
    else:
        universe = build_universe(min_turnover, min_price)

    tickers = [u["ticker"] for u in universe]
    ticker_info = {u["ticker"]: u for u in universe}

    # ── 2. 컨센서스 수집 ──
    logger.info(f"wisereport 컨센서스 수집 시작 ({len(tickers)}종목, delay={delay}초)")
    scraper = ConsensusScraper()
    consensus_data = scraper.fetch_batch(tickers, delay=delay)
    logger.info(f"컨센서스 보유: {len(consensus_data)}/{len(tickers)}종목")

    # ── 3. 스코어링 ──
    picks = []

    for c in consensus_data:
        ticker = c["ticker"]
        info = ticker_info.get(ticker, {})
        close = info.get("close", 0)
        if close <= 0:
            continue

        target = c.get("target_price", 0) or 0
        if target <= 0:
            continue

        upside = round((target / close - 1) * 100, 1)
        if upside < min_upside:
            continue

        # 최소 기관수/의견 필터
        if (c.get("analyst_count") or 0) < min_analysts:
            continue
        if (c.get("opinion_score") or 0) < min_opinion:
            continue

        # 기술적 분석
        tech = get_technical(ticker, close) if not skip_tech else {}

        # 배당수익률
        div_est = c.get("dividend_est") or 0
        div_yield = round(div_est / close * 100, 2) if close > 0 and div_est > 0 else 0

        item = {
            "ticker": ticker,
            "name": c.get("name", info.get("name", "")),
            "close": close,
            "target_price": target,
            "upside_pct": upside,
            "opinion_score": c.get("opinion_score"),
            "analyst_count": c.get("analyst_count"),
            "forward_eps": c.get("forward_eps"),
            "forward_bps": c.get("forward_bps"),
            "forward_per": c.get("forward_per"),
            "forward_pbr": c.get("forward_pbr"),
            "dividend_yield": div_yield,
            "technical": tech,
        }

        item["composite_score"] = calc_composite_score(item, cfg)
        item["grade"] = grade_from_score(item["composite_score"])
        picks.append(item)

    # 스코어 순 정렬
    picks.sort(key=lambda x: x["composite_score"], reverse=True)

    # TOP N
    top_picks = picks[:top_n]

    elapsed = round(time.time() - start_time, 1)

    # ── 4. 결과 출력 ──
    result = {
        "scan_date": datetime.now().strftime("%Y-%m-%d"),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "universe_size": len(tickers),
        "with_consensus": len(consensus_data),
        "passed_filter": len(picks),
        "elapsed_sec": elapsed,
        "top_picks": [],
    }

    logger.info("")
    logger.info(f"{'='*90}")
    logger.info(f" 컨센서스 스크리닝 결과 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    logger.info(f" 유니버스 {len(tickers)} → 컨센서스 {len(consensus_data)} → 필터 {len(picks)} → TOP {len(top_picks)}")
    logger.info(f" 소요시간: {elapsed}초")
    logger.info(f"{'='*90}")
    logger.info("")
    logger.info(
        f"{'#':>2} {'종목':>12} {'코드':>7} {'종가':>9} {'목표가':>10} {'상승여력':>7} "
        f"{'의견':>5} {'기관':>4} {'F-PER':>6} {'배당':>5} {'점수':>5} {'등급':>3}"
    )
    logger.info("-" * 90)

    for idx, p in enumerate(top_picks, 1):
        opinion_s = f"{p['opinion_score']:.2f}" if p['opinion_score'] else "N/A"
        fper_s = f"{p['forward_per']:.1f}" if p['forward_per'] else "N/A"
        div_s = f"{p['dividend_yield']:.1f}%" if p['dividend_yield'] > 0 else "-"

        logger.info(
            f"{idx:>2} {p['name']:>12} {p['ticker']:>7} {p['close']:>9,} {p['target_price']:>10,} "
            f"{p['upside_pct']:>+6.1f}% {opinion_s:>5} {p['analyst_count'] or 0:>4} "
            f"{fper_s:>6} {div_s:>5} {p['composite_score']:>5.1f} {p['grade']:>3}"
        )

        # JSON용 (technical 제거해서 가독성 높임)
        pick_json = {k: v for k, v in p.items() if k != "technical"}
        pick_json["rank"] = idx
        pick_json["rsi"] = p["technical"].get("rsi") if p.get("technical") else None
        pick_json["above_ma60"] = p["technical"].get("above_ma60") if p.get("technical") else None
        pick_json["sar_trend"] = p["technical"].get("sar_trend") if p.get("technical") else None
        pick_json["dist_high_52w"] = p["technical"].get("dist_high_52w") if p.get("technical") else None
        result["top_picks"].append(pick_json)

    logger.info("")

    # all_picks: 전체 필터 통과 종목 (tomorrow_picks 컨센서스 풀용)
    all_picks_json = []
    for p in picks:
        pj = {k: v for k, v in p.items() if k != "technical"}
        pj["rsi"] = p["technical"].get("rsi") if p.get("technical") else None
        pj["above_ma60"] = p["technical"].get("above_ma60") if p.get("technical") else None
        pj["sar_trend"] = p["technical"].get("sar_trend") if p.get("technical") else None
        all_picks_json.append(pj)
    result["all_picks"] = all_picks_json
    logger.info(f"전체 풀: {len(all_picks_json)}종목 (tomorrow_picks 연동용)")

    # ── 5. JSON 저장 ──
    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_default)
    logger.info(f"저장: {OUTPUT_PATH}")

    # ── 6. 텔레그램 발송 ──
    if send_telegram and top_picks:
        _send_telegram(result)

    # ── 7. HTML 보고서 ──
    try:
        from src.consensus_report import generate_consensus_report
        html_path, png_path = generate_consensus_report(result)
        logger.info(f"HTML: {html_path}")
        if png_path:
            logger.info(f"PNG: {png_path}")
            if send_telegram:
                _send_telegram_photo(png_path)
    except ImportError:
        logger.debug("consensus_report 미구현 — HTML 스킵")
    except Exception as e:
        logger.warning(f"HTML 보고서 실패: {e}")

    return result


def _send_telegram(result: dict):
    """텔레그램 텍스트 메시지 발송."""
    try:
        from src.telegram_sender import send_message

        picks = result.get("top_picks", [])[:10]
        lines = [
            f"[컨센서스 스크리닝 {result['scan_date']}]",
            f"유니버스 {result['universe_size']} → 컨센서스 {result['with_consensus']} → TOP {len(picks)}",
            "",
        ]

        for p in picks:
            fper = f"PER{p['forward_per']:.0f}" if p.get("forward_per") else ""
            div = f" 배당{p['dividend_yield']:.1f}%" if p.get("dividend_yield", 0) > 1 else ""
            lines.append(
                f"#{p['rank']} {p['name']}({p['ticker']}) "
                f"+{p['upside_pct']:.0f}% {fper}{div} [{p['grade']}]{p['composite_score']:.0f}점"
            )

        send_message("\n".join(lines))
        logger.info("텔레그램 발송 완료")
    except Exception as e:
        logger.warning(f"텔레그램 실패: {e}")


def _send_telegram_photo(png_path: Path):
    """텔레그램 PNG 이미지 발송."""
    try:
        from src.telegram_sender import send_photo
        send_photo(str(png_path), caption="컨센서스 스크리닝 리포트")
        logger.info("텔레그램 PNG 발송 완료")
    except Exception as e:
        logger.warning(f"텔레그램 PNG 실패: {e}")


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="컨센서스 스크리너")
    parser.add_argument("--top", type=int, default=15, help="TOP N (default: 15)")
    parser.add_argument("--min-upside", type=float, default=10, help="최소 상승여력 %% (default: 10)")
    parser.add_argument("--ticker", type=str, default=None, help="단일 종목 코드")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--no-tech", action="store_true", help="기술적 분석 스킵")
    args = parser.parse_args()

    run_scan(
        top_n=args.top,
        min_upside=args.min_upside,
        single_ticker=args.ticker,
        skip_tech=args.no_tech,
        send_telegram=args.send,
    )


if __name__ == "__main__":
    main()
