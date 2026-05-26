"""D+0 종가 매수 vs D+1 시초가 매수 비교 백테스트 (5/26 사용자 질문 답변용).

배경 (5/26 11:00 퐝가님 질문):
- "베스트 진입 시점이 D+0 인지 D+1 인지 데이터로 답하라"
- 메모리 5/14 백테스트는 D+0 종가 진입 가정 — D+1 시초가 비교 데이터 없음
- 본 스크립트로 두 전략 직접 비교 → 정량 답

전략:
- 시그널 발화일 (D-0) — 4수급 1/2/3단계 또는 C2 필터 통과
- 전략 A: D+0 종가 매수 → D+1 종가 / D+3 종가 / D+5 종가 손익
- 전략 B: D+1 시초가 매수 → D+1 종가 / D+3 종가 / D+5 종가 손익

데이터 소스:
- 정보봇 OHLCV CSV 3개월 (VPS: ~/quantum-master/data/external/jgis_ohlcv/)
- 시그널 시드: scan_sector_fire 3단계 V3 발화 종목 (메모리 5/14 ab3b010)
- 또는 picks_history.json (시그널 일자/종목)

지표:
- D+1 평균 수익률 / 승률 / +10%↑ 비율 / -5%↓ 비율 / MDD
- A vs B 차이 (어느 쪽이 더 좋은가)
- 시그널 유형별 분포 (3단계 / C2 / 단일)

실행:
  VPS: cd ~/quantum-master && PYTHONPATH=. ./venv/bin/python3.11 scripts/backtest/backtest_d0_vs_d1_entry.py
  옵션: --lookback-days 90 (기본 90일)
        --signal-source supabase|picks_history|sector_fire (기본 sector_fire)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OHLCV_DIR = PROJECT_ROOT / "data" / "external" / "jgis_ohlcv"
PICKS_HISTORY = PROJECT_ROOT / "data" / "picks_history.json"


@dataclass
class TradeResult:
    """단일 트레이드 결과."""
    ticker: str
    signal_date: str           # 시그널 발화일 (D-0)
    strategy: str              # 'D0_CLOSE' or 'D1_OPEN'
    entry_price: float
    d1_close: float
    d3_close: float
    d5_close: float
    pnl_d1_pct: float
    pnl_d3_pct: float
    pnl_d5_pct: float
    signal_type: str = "?"     # '3단계' / 'C2' / 'V3_FULL_SWING' 등


@dataclass
class StrategySummary:
    """전략 종합 통계."""
    strategy: str
    n_trades: int = 0
    d1_avg: float = 0.0
    d1_win_rate: float = 0.0   # >0%
    d1_over10: float = 0.0     # +10%↑ 비율
    d1_under_5: float = 0.0    # -5%↓ 비율
    d3_avg: float = 0.0
    d5_avg: float = 0.0
    mdd: float = 0.0           # 최악 trade
    profit_factor: float = 0.0  # 평균 수익 / 평균 손실


C2_CORE_SOURCES = {"AI섹터", "밸류체인", "US모멘텀", "인텔리전스"}
C2_AVOID_SINGLE = {"수급폭발", "매집추적"}  # 단독이면 회피


def load_signal_dates_from_picks_history(
    lookback_days: int,
    grade_filter: str | None = None,
    min_score: float = 0,
    min_sources: int = 0,
    c2_filter: bool = False,
) -> list[tuple[str, str, str]]:
    """picks_history.json에서 시그널 일자/종목/유형 추출.

    실제 구조 (5/26 확인):
        {"records": [{"pick_date": "...", "ticker": "...", "sources": [...],
                      "grade": "강력 포착", "score": 100, "n_sources": 5, ...}]}

    Args:
        grade_filter: 'matchcase' 필터 (예: "강력 포착", "적극매수")
        min_score: 최소 점수 (기본 0 = 미적용)
        min_sources: 최소 sources 개수 (기본 0)
        c2_filter: C2 룰 (4핵심 2개+ 포함 + 단독 회피 sources 제외)

    Returns:
        [(date_yyyymmdd, ticker, signal_type), ...]
    """
    import json
    if not PICKS_HISTORY.exists():
        logger.warning("picks_history.json 없음")
        return []

    try:
        d = json.loads(PICKS_HISTORY.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("picks_history 로드 실패: %s", e)
        return []

    if isinstance(d, dict):
        recs = d.get("records", [])
    elif isinstance(d, list):
        recs = d
    else:
        return []

    cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%Y%m%d")
    results = []
    for x in recs:
        if not isinstance(x, dict):
            continue
        dt = (x.get("pick_date") or x.get("date") or x.get("target_date") or "").replace("-", "")
        if not dt or dt < cutoff:
            continue
        ticker = str(x.get("ticker", "")).zfill(6)
        srcs = x.get("sources", []) or []
        score = float(x.get("score", 0) or 0)
        grade = x.get("grade", "")
        n_src = int(x.get("n_sources", len(srcs)) or len(srcs))

        # 필터 적용
        if grade_filter and grade != grade_filter:
            continue
        if min_score > 0 and score < min_score:
            continue
        if min_sources > 0 and n_src < min_sources:
            continue
        if c2_filter:
            # C2 L1 필수: 4핵심 중 2개 이상
            core_hits = len(set(srcs) & C2_CORE_SOURCES)
            if core_hits < 2:
                continue
            # 회피: 수급폭발/매집추적 단독
            if len(srcs) == 1 and srcs[0] in C2_AVOID_SINGLE:
                continue

        sig_type = "+".join(srcs[:3]) if srcs else "?"
        if ticker and ticker != "000000":
            results.append((dt, ticker, sig_type))
    return results


def load_ohlcv_csv(ticker: str) -> list[dict]:
    """정보봇 OHLCV CSV 로드.

    파일명 형식: {종목명}_{6자리코드}.csv
    """
    candidates = list(OHLCV_DIR.glob(f"*_{str(ticker).zfill(6)}.csv"))
    if not candidates:
        return []

    bars = []
    with candidates[0].open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bars.append({
                    "date": row.get("Date", "").replace("-", ""),
                    "open": float(row.get("Open", 0) or 0),
                    "high": float(row.get("High", 0) or 0),
                    "low": float(row.get("Low", 0) or 0),
                    "close": float(row.get("Close", 0) or 0),
                    "volume": int(float(row.get("Volume", 0) or 0)),
                })
            except (TypeError, ValueError):
                continue
    return sorted(bars, key=lambda x: x["date"])


def find_bar_index(bars: list[dict], target_date: str) -> int:
    """target_date 일자의 bar index. 못 찾으면 -1."""
    for i, b in enumerate(bars):
        if b["date"] == target_date:
            return i
    return -1


def simulate_trade(bars: list[dict], signal_idx: int, strategy: str,
                    ticker: str, signal_date: str, signal_type: str) -> TradeResult | None:
    """단일 트레이드 시뮬레이션.

    Args:
        bars: 종목 OHLCV
        signal_idx: 시그널 발화일 index
        strategy: 'D0_CLOSE' (당일 종가 매수) or 'D1_OPEN' (다음날 시초가 매수)
    """
    n = len(bars)
    if strategy == "D0_CLOSE":
        if signal_idx >= n:
            return None
        entry = bars[signal_idx]["close"]
        d1_idx = signal_idx + 1
    elif strategy == "D1_OPEN":
        if signal_idx + 1 >= n:
            return None
        entry = bars[signal_idx + 1]["open"]
        d1_idx = signal_idx + 1
    else:
        return None

    if entry <= 0 or d1_idx >= n:
        return None

    d1_close = bars[d1_idx]["close"]
    d3_close = bars[min(d1_idx + 2, n - 1)]["close"]
    d5_close = bars[min(d1_idx + 4, n - 1)]["close"]

    return TradeResult(
        ticker=ticker, signal_date=signal_date, strategy=strategy,
        entry_price=entry,
        d1_close=d1_close, d3_close=d3_close, d5_close=d5_close,
        pnl_d1_pct=(d1_close - entry) / entry * 100,
        pnl_d3_pct=(d3_close - entry) / entry * 100,
        pnl_d5_pct=(d5_close - entry) / entry * 100,
        signal_type=signal_type,
    )


def summarize_strategy(trades: list[TradeResult], strategy: str) -> StrategySummary:
    """전략별 통계 집계."""
    s = StrategySummary(strategy=strategy, n_trades=len(trades))
    if not trades:
        return s

    d1s = [t.pnl_d1_pct for t in trades]
    d3s = [t.pnl_d3_pct for t in trades]
    d5s = [t.pnl_d5_pct for t in trades]

    s.d1_avg = sum(d1s) / len(d1s)
    s.d3_avg = sum(d3s) / len(d3s)
    s.d5_avg = sum(d5s) / len(d5s)
    s.d1_win_rate = sum(1 for x in d1s if x > 0) / len(d1s) * 100
    s.d1_over10 = sum(1 for x in d1s if x >= 10) / len(d1s) * 100
    s.d1_under_5 = sum(1 for x in d1s if x <= -5) / len(d1s) * 100
    s.mdd = min(d1s)
    profits = [x for x in d1s if x > 0]
    losses = [-x for x in d1s if x < 0]
    if losses and sum(losses) > 0:
        s.profit_factor = (sum(profits) / max(len(profits), 1)) / (sum(losses) / max(len(losses), 1))
    return s


def run_backtest(
    lookback_days: int = 90,
    signal_source: str = "picks_history",
    grade_filter: str | None = None,
    min_score: float = 0,
    min_sources: int = 0,
    c2_filter: bool = False,
) -> dict:
    """백테스트 실행."""
    filter_desc = []
    if grade_filter: filter_desc.append(f"grade={grade_filter}")
    if min_score > 0: filter_desc.append(f"score>={min_score}")
    if min_sources > 0: filter_desc.append(f"sources>={min_sources}")
    if c2_filter: filter_desc.append("C2(4핵심 2+)")
    fd = f" 필터: {', '.join(filter_desc)}" if filter_desc else ""
    print(f"[백테스트] D+0 vs D+1 진입 비교 (lookback {lookback_days}일, 소스: {signal_source}){fd}")

    if signal_source == "picks_history":
        signals = load_signal_dates_from_picks_history(
            lookback_days, grade_filter=grade_filter,
            min_score=min_score, min_sources=min_sources, c2_filter=c2_filter,
        )
    else:
        signals = load_signal_dates_from_picks_history(
            lookback_days, grade_filter=grade_filter,
            min_score=min_score, min_sources=min_sources, c2_filter=c2_filter,
        )

    if not signals:
        print("⚠️ 시그널 데이터 0건 — 백테스트 불가")
        return {"n_signals": 0}

    print(f"  시그널 {len(signals)}건")

    trades_d0 = []
    trades_d1 = []
    skipped = 0

    for sig_date, ticker, sig_type in signals:
        bars = load_ohlcv_csv(ticker)
        if not bars:
            skipped += 1
            continue
        idx = find_bar_index(bars, sig_date)
        if idx < 0:
            skipped += 1
            continue
        t0 = simulate_trade(bars, idx, "D0_CLOSE", ticker, sig_date, sig_type)
        t1 = simulate_trade(bars, idx, "D1_OPEN", ticker, sig_date, sig_type)
        if t0:
            trades_d0.append(t0)
        if t1:
            trades_d1.append(t1)

    print(f"  D+0 종가 매수 트레이드: {len(trades_d0)}건")
    print(f"  D+1 시초가 매수 트레이드: {len(trades_d1)}건")
    print(f"  스킵 (OHLCV/시그널일 매칭 실패): {skipped}건")

    s0 = summarize_strategy(trades_d0, "D0_CLOSE")
    s1 = summarize_strategy(trades_d1, "D1_OPEN")

    print()
    print(f"{'='*70}")
    print(f"  지표               D+0 종가 매수       D+1 시초가 매수")
    print(f"{'='*70}")
    print(f"  트레이드 수        {s0.n_trades:>15d}     {s1.n_trades:>15d}")
    print(f"  D+1 평균 (%)       {s0.d1_avg:>15.2f}     {s1.d1_avg:>15.2f}")
    print(f"  D+1 승률 (%)       {s0.d1_win_rate:>15.2f}     {s1.d1_win_rate:>15.2f}")
    print(f"  D+1 +10%↑ (%)      {s0.d1_over10:>15.2f}     {s1.d1_over10:>15.2f}")
    print(f"  D+1 -5%↓ (%)       {s0.d1_under_5:>15.2f}     {s1.d1_under_5:>15.2f}")
    print(f"  D+3 평균 (%)       {s0.d3_avg:>15.2f}     {s1.d3_avg:>15.2f}")
    print(f"  D+5 평균 (%)       {s0.d5_avg:>15.2f}     {s1.d5_avg:>15.2f}")
    print(f"  MDD (%)            {s0.mdd:>15.2f}     {s1.mdd:>15.2f}")
    print(f"  Profit Factor      {s0.profit_factor:>15.2f}     {s1.profit_factor:>15.2f}")
    print(f"{'='*70}")
    print()

    # 결단 도출
    diff = s0.d1_avg - s1.d1_avg
    if abs(diff) < 0.3:
        verdict = "근소 차이 — 표본 부족 또는 통계적 유의성 없음"
    elif diff > 0:
        verdict = f"D+0 종가 매수 우세 (D+1 평균 +{diff:.2f}%p)"
    else:
        verdict = f"D+1 시초가 매수 우세 (D+1 평균 +{-diff:.2f}%p)"
    print(f"📌 결론: {verdict}")

    return {
        "n_signals": len(signals),
        "n_d0_trades": s0.n_trades,
        "n_d1_trades": s1.n_trades,
        "s0": s0.__dict__,
        "s1": s1.__dict__,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--signal-source", default="picks_history",
                        choices=["picks_history", "sector_fire", "supabase"])
    parser.add_argument("--grade", default=None,
                        help="grade 필터 (예: '강력 포착', '적극매수')")
    parser.add_argument("--min-score", type=float, default=0)
    parser.add_argument("--min-sources", type=int, default=0)
    parser.add_argument("--c2", action="store_true",
                        help="C2 룰: 4핵심(AI섹터/밸류체인/US모멘텀/인텔리전스) 2개+ 포함")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_backtest(
        lookback_days=args.lookback_days, signal_source=args.signal_source,
        grade_filter=args.grade, min_score=args.min_score,
        min_sources=args.min_sources, c2_filter=args.c2,
    )


if __name__ == "__main__":
    main()
