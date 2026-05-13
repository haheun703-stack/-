#!/usr/bin/env python3
"""아르띠 피보나치 전략 백테스트 — 한국 주식 일봉 검증

아르띠(Arti Buri) 피보나치 전략 4가지 + 수급 조합 백테스트:
  S1. Fibonacci Only (baseline)
  S2. Fibonacci + RSI (과매도 확인)
  S3. Fibonacci + Volume Dry-up (거래량 감소 = 매도세 약화)
  S4. Fibonacci + S/R (전고/전저 지지 일치)
  S5. All Combined (S2+S3+S4)
  S6. Fibonacci + 수급 (외인+기관 순매수)
  S7. All + 수급 (궁극 조합)

핵심 설계:
  - Zigzag으로 스윙 고점/저점 탐지
  - 상승 추세(SL→SH) 후 되돌림 구간에서 피보나치 레벨 도달 시 매수
  - 0.382 (얕은 되돌림), 0.5 (중간), 0.618 (황금비율) 레벨별 분석
  - 0.786 이탈 = 손절 (추세 전환 판정)

Usage:
    python scripts/backtest_fibonacci_artie.py
    python scripts/backtest_fibonacci_artie.py --days 500
    python scripts/backtest_fibonacci_artie.py --swing 15
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "stock_data_daily"

# ─── 피보나치 레벨 ───
FIB_LEVELS = {
    "fib_382": 0.382,
    "fib_500": 0.500,
    "fib_618": 0.618,
}
FIB_STOP = 0.786  # 이탈 시 추세 전환 판정

# ─── 파라미터 ───
SWING_LOOKBACK = 10      # 스윙 포인트 좌우 확인 봉 수
FIB_TOLERANCE = 0.02     # 피보나치 레벨 ±2% 허용 오차
RSI_OVERSOLD = 40        # RSI 과매도 기준 (일봉이므로 40 사용)
VOL_DRYUP_RATIO = 0.8    # 거래량 < MA20의 80% = 매도세 약화
SR_TOLERANCE = 0.03      # 지지/저항 일치 ±3%
SUPPLY_MIN = 5.0         # 수급 순매수 최소 (억원)
MIN_SWING_PCT = 8.0      # 최소 스윙 폭 (%) — 너무 작은 움직임 제외
MIN_TRADING_VALUE = 10.0 # 최소 거래대금 (억원) — 유동성 필터


@dataclass
class FibSignal:
    """피보나치 되돌림 시그널."""
    ticker: str
    name: str
    date: str
    close: float
    fib_level: str       # fib_382, fib_500, fib_618
    fib_ratio: float     # 실제 되돌림 비율
    swing_high: float
    swing_low: float
    swing_pct: float     # 스윙 폭 (%)
    rsi: float
    vol_ratio: float     # Volume / Vol_MA20
    has_sr_support: bool  # 전고/전저 지지 여부
    foreign_net: float
    inst_net: float
    # 수익률
    ret_d1: float = np.nan
    ret_d3: float = np.nan
    ret_d5: float = np.nan
    ret_d10: float = np.nan
    # 최대 역행폭 (MDD within D+5)
    max_drawdown_d5: float = np.nan
    # 0.786 이탈 여부 (D+5 내)
    broke_786: bool = False


def load_csv(path: Path, lookback: int = 400) -> pd.DataFrame | None:
    """CSV 로드."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if len(df) < 60:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) > lookback:
            df = df.iloc[-lookback:].reset_index(drop=True)
        return df
    except Exception:
        return None


def find_swing_points(highs: np.ndarray, lows: np.ndarray,
                      lookback: int = SWING_LOOKBACK) -> tuple[list, list]:
    """Zigzag 방식으로 스윙 고점/저점 탐지.

    Returns:
        swing_highs: [(index, price), ...]
        swing_lows: [(index, price), ...]
    """
    n = len(highs)
    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        # 스윙 고점: 좌우 lookback 봉 중 최고
        left_high = np.max(highs[i - lookback:i])
        right_high = np.max(highs[i + 1:i + lookback + 1])
        if highs[i] >= left_high and highs[i] >= right_high:
            swing_highs.append((i, highs[i]))

        # 스윙 저점: 좌우 lookback 봉 중 최저
        left_low = np.min(lows[i - lookback:i])
        right_low = np.min(lows[i + 1:i + lookback + 1])
        if lows[i] <= left_low and lows[i] <= right_low:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def find_previous_sr_levels(swing_highs: list, swing_lows: list,
                             before_idx: int) -> list[float]:
    """특정 인덱스 이전의 지지/저항 레벨들 반환."""
    levels = []
    for idx, price in swing_highs:
        if idx < before_idx:
            levels.append(price)
    for idx, price in swing_lows:
        if idx < before_idx:
            levels.append(price)
    return levels


def check_sr_support(price: float, sr_levels: list[float],
                     tolerance: float = SR_TOLERANCE) -> bool:
    """가격이 기존 지지/저항선 근처인지 확인."""
    for level in sr_levels:
        if abs(price - level) / level <= tolerance:
            return True
    return False


def scan_fibonacci_signals(df: pd.DataFrame, ticker: str, name: str,
                            lookback_days: int, swing_lb: int,
                            fib_tol: float = FIB_TOLERANCE) -> list[FibSignal]:
    """하나의 종목에서 피보나치 되돌림 시그널 탐색."""
    df = df.copy()
    n = len(df)

    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    # 거래량 MA20 계산
    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    # RSI — CSV에 있으면 사용, 없으면 계산
    if "RSI" in df.columns:
        rsi_vals = df["RSI"].values.astype(float)
    else:
        delta = pd.Series(closes).diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_vals = (100 - 100 / (1 + rs)).values

    # 수급 데이터
    foreign_net = df["Foreign_Net"].values.astype(float) if "Foreign_Net" in df.columns else np.zeros(n)
    inst_net = df["Inst_Net"].values.astype(float) if "Inst_Net" in df.columns else np.zeros(n)

    # 스윙 포인트 탐지
    swing_highs, swing_lows = find_swing_points(highs, lows, swing_lb)

    if not swing_highs or not swing_lows:
        return []

    # 백테스트 범위
    start_idx = max(swing_lb * 2 + 20, n - lookback_days)
    end_idx = n - 10  # D+10 추적 여유

    signals = []

    for i in range(start_idx, end_idx):
        # 거래대금 필터
        tv = closes[i] * volumes[i] / 1e8
        if tv < MIN_TRADING_VALUE:
            continue

        # 가장 최근 스윙 고점 (현재 봉 이전)
        recent_sh = None
        for idx, price in reversed(swing_highs):
            if idx < i:
                recent_sh = (idx, price)
                break

        if recent_sh is None:
            continue

        # 스윙 고점 이전의 스윙 저점 (상승 추세의 시작점)
        sh_idx, sh_price = recent_sh
        recent_sl = None
        for idx, price in reversed(swing_lows):
            if idx < sh_idx:
                recent_sl = (idx, price)
                break

        if recent_sl is None:
            continue

        sl_idx, sl_price = recent_sl

        # 최소 스윙 폭 확인
        swing_pct = (sh_price - sl_price) / sl_price * 100
        if swing_pct < MIN_SWING_PCT:
            continue

        # 현재 가격이 스윙 고점 아래 + 스윙 저점 위여야 (되돌림 구간)
        if closes[i] >= sh_price or closes[i] <= sl_price:
            continue

        # 피보나치 되돌림 비율 계산
        # ratio = 0이면 고점, 1이면 저점
        fib_ratio = (sh_price - closes[i]) / (sh_price - sl_price)

        # 각 피보나치 레벨 근처인지 확인
        for level_name, level_val in FIB_LEVELS.items():
            if abs(fib_ratio - level_val) <= fib_tol:
                # 시그널 발견!
                vol_ratio = volumes[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0
                rsi = rsi_vals[i] if not np.isnan(rsi_vals[i]) else 50.0

                # S/R 지지 확인
                sr_levels = find_previous_sr_levels(swing_highs, swing_lows, sh_idx)
                has_sr = check_sr_support(closes[i], sr_levels)

                # D+1, D+3, D+5, D+10 수익률
                ret_d1 = (closes[i + 1] / closes[i] - 1) * 100 if i + 1 < n else np.nan
                ret_d3 = (closes[i + 3] / closes[i] - 1) * 100 if i + 3 < n else np.nan
                ret_d5 = (closes[i + 5] / closes[i] - 1) * 100 if i + 5 < n else np.nan
                ret_d10 = (closes[i + 10] / closes[i] - 1) * 100 if i + 10 < n else np.nan

                # D+5 최대 역행폭
                mdd_d5 = np.nan
                broke_786 = False
                if i + 5 < n:
                    future_lows = lows[i + 1:i + 6]
                    min_low = np.min(future_lows)
                    mdd_d5 = (min_low / closes[i] - 1) * 100
                    # 0.786 이탈 여부
                    fib_786_price = sh_price - FIB_STOP * (sh_price - sl_price)
                    if min_low < fib_786_price:
                        broke_786 = True

                sig = FibSignal(
                    ticker=ticker,
                    name=name,
                    date=str(df.iloc[i]["Date"].date()),
                    close=closes[i],
                    fib_level=level_name,
                    fib_ratio=round(fib_ratio, 4),
                    swing_high=sh_price,
                    swing_low=sl_price,
                    swing_pct=round(swing_pct, 1),
                    rsi=round(rsi, 1),
                    vol_ratio=round(vol_ratio, 2),
                    has_sr_support=has_sr,
                    foreign_net=round(foreign_net[i], 1),
                    inst_net=round(inst_net[i], 1),
                    ret_d1=round(ret_d1, 2) if not np.isnan(ret_d1) else np.nan,
                    ret_d3=round(ret_d3, 2) if not np.isnan(ret_d3) else np.nan,
                    ret_d5=round(ret_d5, 2) if not np.isnan(ret_d5) else np.nan,
                    ret_d10=round(ret_d10, 2) if not np.isnan(ret_d10) else np.nan,
                    max_drawdown_d5=round(mdd_d5, 2) if not np.isnan(mdd_d5) else np.nan,
                    broke_786=broke_786,
                )
                signals.append(sig)
                break  # 한 봉에서 하나의 레벨만

    return signals


def evaluate_strategy(signals: list[FibSignal], name: str,
                      filter_fn=None) -> dict | None:
    """전략 필터 적용 후 성과 지표 계산."""
    filtered = [s for s in signals if (filter_fn is None or filter_fn(s))]

    if not filtered:
        return None

    rets_d1 = [s.ret_d1 for s in filtered if not np.isnan(s.ret_d1)]
    rets_d3 = [s.ret_d3 for s in filtered if not np.isnan(s.ret_d3)]
    rets_d5 = [s.ret_d5 for s in filtered if not np.isnan(s.ret_d5)]
    rets_d10 = [s.ret_d10 for s in filtered if not np.isnan(s.ret_d10)]
    mdds = [s.max_drawdown_d5 for s in filtered if not np.isnan(s.max_drawdown_d5)]
    broke_count = sum(1 for s in filtered if s.broke_786)

    def calc_stats(rets: list) -> dict:
        if not rets:
            return {"n": 0, "avg": 0, "wr": 0, "pf": 0, "median": 0}
        arr = np.array(rets)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / len(arr) * 100 if len(arr) > 0 else 0
        pf = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999.0
        return {
            "n": len(arr),
            "avg": round(np.mean(arr), 2),
            "wr": round(wr, 1),
            "pf": round(pf, 2),
            "median": round(np.median(arr), 2),
        }

    return {
        "name": name,
        "n": len(filtered),
        "d1": calc_stats(rets_d1),
        "d3": calc_stats(rets_d3),
        "d5": calc_stats(rets_d5),
        "d10": calc_stats(rets_d10),
        "avg_mdd_d5": round(np.mean(mdds), 2) if mdds else 0,
        "broke_786_rate": round(broke_count / len(filtered) * 100, 1) if filtered else 0,
    }


def classify_alpha(stats: dict) -> str:
    """알파 등급 판정."""
    if stats is None or stats["n"] < 20:
        return "INSUFFICIENT (n<20)"
    d5 = stats["d5"]
    if d5["avg"] >= 2.0 and d5["wr"] >= 55 and d5["pf"] >= 1.8:
        return "★ STRONG_ALPHA"
    elif d5["avg"] >= 1.0 and d5["wr"] >= 50 and d5["pf"] >= 1.3:
        return "▲ WEAK_ALPHA"
    elif d5["avg"] >= 0.5 and d5["pf"] >= 1.1:
        return "△ MARGINAL"
    else:
        return "✗ NO_ALPHA"


def print_results(results: list[dict]):
    """결과 출력."""
    print("\n" + "=" * 90)
    print("아르띠 피보나치 전략 백테스트 결과")
    print("=" * 90)

    for r in results:
        if r is None:
            continue
        alpha = classify_alpha(r)
        print(f"\n{'─' * 80}")
        print(f"  {r['name']}  |  n={r['n']}  |  {alpha}")
        print(f"{'─' * 80}")
        for period, label in [("d1", "D+1"), ("d3", "D+3"), ("d5", "D+5"), ("d10", "D+10")]:
            s = r[period]
            print(f"  {label}: avg {s['avg']:+.2f}%  |  WR {s['wr']:.1f}%  |  "
                  f"PF {s['pf']:.2f}  |  median {s['median']:+.2f}%  |  n={s['n']}")
        print(f"  MDD(D+5 avg): {r['avg_mdd_d5']:.2f}%  |  "
              f"0.786 이탈률: {r['broke_786_rate']:.1f}%")


def print_fib_level_breakdown(signals: list[FibSignal]):
    """피보나치 레벨별 성과 분석."""
    print(f"\n{'=' * 90}")
    print("피보나치 레벨별 성과 비교")
    print(f"{'=' * 90}")

    for level in ["fib_382", "fib_500", "fib_618"]:
        level_sigs = [s for s in signals if s.fib_level == level]
        if not level_sigs:
            continue
        rets = [s.ret_d5 for s in level_sigs if not np.isnan(s.ret_d5)]
        if not rets:
            continue
        arr = np.array(rets)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / len(arr) * 100
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
        print(f"  {level}: n={len(arr):,}  |  D+5 avg {np.mean(arr):+.2f}%  |  "
              f"WR {wr:.1f}%  |  PF {pf:.2f}  |  median {np.median(arr):+.2f}%")


def print_swing_size_breakdown(signals: list[FibSignal]):
    """스윙 폭별 성과 분석."""
    print(f"\n{'=' * 90}")
    print("스윙 폭별 성과 비교 (상승 추세 크기)")
    print(f"{'=' * 90}")

    bins = [(8, 15, "8-15%"), (15, 25, "15-25%"), (25, 40, "25-40%"), (40, 999, "40%+")]
    for lo, hi, label in bins:
        sigs = [s for s in signals if lo <= s.swing_pct < hi]
        if not sigs:
            continue
        rets = [s.ret_d5 for s in sigs if not np.isnan(s.ret_d5)]
        if not rets:
            continue
        arr = np.array(rets)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / len(arr) * 100
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
        print(f"  스윙 {label}: n={len(arr):,}  |  D+5 avg {np.mean(arr):+.2f}%  |  "
              f"WR {wr:.1f}%  |  PF {pf:.2f}")


def main():
    parser = argparse.ArgumentParser(description="아르띠 피보나치 전략 백테스트")
    parser.add_argument("--days", type=int, default=260, help="백테스트 거래일 수 (default: 260)")
    parser.add_argument("--swing", type=int, default=SWING_LOOKBACK, help="스윙 포인트 lookback (default: 10)")
    parser.add_argument("--tolerance", type=float, default=FIB_TOLERANCE, help="피보 레벨 허용 오차 (default: 0.02)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        logger.error(f"데이터 디렉토리 없음: {DATA_DIR}")
        sys.exit(1)

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]
    logger.info(f"CSV 파일: {len(csv_files)}개, 백테스트 범위: 최근 {args.days}거래일")

    fib_tol = args.tolerance

    all_signals: list[FibSignal] = []
    processed = 0
    t0 = time.time()

    for csv_path in csv_files:
        fname = csv_path.stem
        parts = fname.rsplit("_", 1)
        if len(parts) == 2:
            name, ticker = parts
        else:
            name, ticker = fname, fname

        df = load_csv(csv_path, lookback=args.days + 100)
        if df is None:
            continue

        sigs = scan_fibonacci_signals(df, ticker, name, args.days, args.swing, fib_tol)
        all_signals.extend(sigs)
        processed += 1

        if processed % 500 == 0:
            logger.info(f"  진행: {processed}/{len(csv_files)}  시그널: {len(all_signals)}")

    elapsed = time.time() - t0
    logger.info(f"스캔 완료: {processed}종목, {len(all_signals)}개 시그널, {elapsed:.1f}초")

    if not all_signals:
        logger.error("시그널 0건 — 파라미터 조정 필요")
        sys.exit(1)

    # ─── 7가지 전략 평가 ───

    results = []

    # S1: Fibonacci Only (baseline)
    results.append(evaluate_strategy(
        all_signals, "S1. Fibonacci Only (baseline)"))

    # S2: Fibonacci + RSI (과매도)
    results.append(evaluate_strategy(
        all_signals, "S2. Fib + RSI (<40)",
        lambda s: s.rsi < RSI_OVERSOLD))

    # S3: Fibonacci + Volume Dry-up (거래량 감소 = 매도세 약화)
    results.append(evaluate_strategy(
        all_signals, "S3. Fib + Volume Dry-up (<0.8x MA20)",
        lambda s: s.vol_ratio < VOL_DRYUP_RATIO))

    # S4: Fibonacci + S/R 지지
    results.append(evaluate_strategy(
        all_signals, "S4. Fib + S/R Support (전고/전저 일치)",
        lambda s: s.has_sr_support))

    # S5: All Combined (RSI + Volume + S/R)
    results.append(evaluate_strategy(
        all_signals, "S5. Fib + RSI + Volume + S/R (All)",
        lambda s: s.rsi < RSI_OVERSOLD and s.vol_ratio < VOL_DRYUP_RATIO and s.has_sr_support))

    # S6: Fibonacci + 수급 (외인 or 기관 순매수)
    results.append(evaluate_strategy(
        all_signals, "S6. Fib + 수급 (외인|기관 >= 5억)",
        lambda s: s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN))

    # S7: Fibonacci + 수급 + RSI
    results.append(evaluate_strategy(
        all_signals, "S7. Fib + 수급 + RSI (<40)",
        lambda s: (s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN) and s.rsi < RSI_OVERSOLD))

    # S8: Fibonacci + 수급 + Volume Dry-up
    results.append(evaluate_strategy(
        all_signals, "S8. Fib + 수급 + Volume Dry-up",
        lambda s: (s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN) and s.vol_ratio < VOL_DRYUP_RATIO))

    # S9: 궁극 조합 (수급 + RSI + Volume + S/R)
    results.append(evaluate_strategy(
        all_signals, "S9. Fib + 수급 + RSI + Volume + S/R (궁극)",
        lambda s: (s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN)
                  and s.rsi < RSI_OVERSOLD and s.vol_ratio < VOL_DRYUP_RATIO and s.has_sr_support))

    # S10: 0.618 전용 + 수급 (황금비율 최적화)
    results.append(evaluate_strategy(
        all_signals, "S10. Fib 0.618 Only + 수급 (황금비율)",
        lambda s: s.fib_level == "fib_618" and (s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN)))

    # S11: 0.618 + 수급 + RSI (황금비율 궁극)
    results.append(evaluate_strategy(
        all_signals, "S11. Fib 0.618 + 수급 + RSI (황금비율 궁극)",
        lambda s: s.fib_level == "fib_618"
                  and (s.foreign_net >= SUPPLY_MIN or s.inst_net >= SUPPLY_MIN)
                  and s.rsi < RSI_OVERSOLD))

    # ─── 결과 출력 ───
    print_results(results)
    print_fib_level_breakdown(all_signals)
    print_swing_size_breakdown(all_signals)

    # ─── 알파 등급 요약 ───
    print(f"\n{'=' * 90}")
    print("알파 등급 요약")
    print(f"{'=' * 90}")
    for r in results:
        if r is None:
            continue
        alpha = classify_alpha(r)
        d5 = r["d5"]
        print(f"  {r['name']}")
        print(f"    → {alpha}  |  n={r['n']}  |  D+5 avg {d5['avg']:+.2f}%  |  "
              f"WR {d5['wr']:.1f}%  |  PF {d5['pf']:.2f}")

    # ─── CSV 저장 ───
    if all_signals:
        out_path = PROJECT_ROOT / "data" / "alpha_backtest" / "fibonacci_artie_signals.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in all_signals:
            rows.append({
                "ticker": s.ticker, "name": s.name, "date": s.date,
                "close": s.close, "fib_level": s.fib_level,
                "fib_ratio": s.fib_ratio, "swing_pct": s.swing_pct,
                "rsi": s.rsi, "vol_ratio": s.vol_ratio,
                "has_sr": s.has_sr_support,
                "foreign_net": s.foreign_net, "inst_net": s.inst_net,
                "ret_d1": s.ret_d1, "ret_d3": s.ret_d3,
                "ret_d5": s.ret_d5, "ret_d10": s.ret_d10,
                "mdd_d5": s.max_drawdown_d5, "broke_786": s.broke_786,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"시그널 CSV 저장: {out_path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
