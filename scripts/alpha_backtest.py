"""알파 백테스트 — 시그널별 실제 수익률 검증.

사용법:
    python scripts/alpha_backtest.py                    # 전체 백테스트
    python scripts/alpha_backtest.py --signal inst      # 기관매집만
    python scripts/alpha_backtest.py --signal foreign    # 외국인매집만
    python scripts/alpha_backtest.py --signal dual       # 쌍끌이만
    python scripts/alpha_backtest.py --signal picks      # 추천종목 사후검증

핵심 질문: "이 시그널이 발생한 날 매수했으면 D+1/5/10/20에 얼마나 벌었나?"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULT_DIR = DATA_DIR / "alpha_backtest"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 유틸리티 ───

def load_parquet(ticker: str) -> pd.DataFrame | None:
    """종목 일봉 parquet 로드. 컬럼: open,high,low,close,volume,기관합계,외국인합계,..."""
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def future_returns(df: pd.DataFrame, signal_dates: list[str], horizons: list[int] = None) -> pd.DataFrame:
    """시그널 발생일 기준 D+N 수익률 계산.

    Args:
        df: 종목 일봉 (index=date, columns=[close,...])
        signal_dates: 시그널 발생 날짜 리스트 (YYYY-MM-DD)
        horizons: [1, 5, 10, 20] 등 보유기간

    Returns:
        DataFrame: 각 시그널일별 D+1/5/10/20 수익률
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    results = []
    for d_str in signal_dates:
        d = pd.Timestamp(d_str)
        # 시그널일의 종가 = 매수가 (장마감 후 시그널 발생 → 다음날 시가 매수가 맞지만, 보수적으로 당일 종가)
        if d not in df.index:
            # 가장 가까운 거래일 찾기
            after = df.index[df.index >= d]
            if after.empty:
                continue
            d = after[0]

        entry_price = df.loc[d, "close"]
        if entry_price <= 0:
            continue

        row = {"signal_date": d.strftime("%Y-%m-%d"), "entry_price": entry_price}

        for h in horizons:
            future_idx = df.index[df.index > d]
            if len(future_idx) < h:
                row[f"D+{h}"] = np.nan
            else:
                future_close = df.loc[future_idx[h - 1], "close"]
                row[f"D+{h}"] = round((future_close - entry_price) / entry_price * 100, 2)

        results.append(row)

    return pd.DataFrame(results)


# ─── 시그널 1: 기관 연속 매수 ───

def signal_institutional_accumulation(min_consec: int = 3, min_amount_억: float = 50) -> dict:
    """기관 연속 순매수 시그널 백테스트.

    조건: 기관합계 N일 연속 순매수 & 누적 금액 >= min_amount_억
    """
    logger.info("=== 시그널: 기관 연속매수 (min_consec=%d, min_amount=%d억) ===", min_consec, min_amount_억)

    all_results = []
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        df = load_parquet(ticker)
        if df is None or len(df) < 60:
            continue
        if "기관합계" not in df.columns:
            continue

        # 기관 연속 매수일 계산
        inst = df["기관합계"].fillna(0)
        is_buy = (inst > 0).astype(int)

        # 연속 매수일 수 계산
        consec = is_buy.copy()
        for idx in range(1, len(consec)):
            if consec.iloc[idx] == 1:
                consec.iloc[idx] = consec.iloc[idx - 1] + 1
            else:
                consec.iloc[idx] = 0

        # 시그널 발생: 연속매수 min_consec일 도달한 날
        signal_mask = (consec == min_consec)
        signal_dates = df.index[signal_mask].strftime("%Y-%m-%d").tolist()

        if not signal_dates:
            continue

        # 해당 시점의 누적 금액 필터 (직전 N일 합)
        filtered_dates = []
        for sd in signal_dates:
            sd_ts = pd.Timestamp(sd)
            idx_pos = df.index.get_loc(sd_ts)
            start = max(0, idx_pos - min_consec + 1)
            total_amt = inst.iloc[start:idx_pos + 1].sum()
            # 금액 단위: 원 → 억 변환 (KIS API 기준 원 단위)
            total_억 = total_amt / 1e8
            if total_억 >= min_amount_억:
                filtered_dates.append(sd)

        if not filtered_dates:
            continue

        ret_df = future_returns(df, filtered_dates)
        if ret_df.empty:
            continue

        ret_df["ticker"] = ticker
        ret_df["signal"] = "INST_ACCUM_%dd" % min_consec
        all_results.append(ret_df)

        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d", i + 1, total)

    if not all_results:
        logger.warning("  시그널 발생 0건")
        return {}

    result = pd.concat(all_results, ignore_index=True)
    return _summarize(result, "INST_ACCUM_%dd" % min_consec)


# ─── 시그널 2: 외국인 연속 매수 ───

def signal_foreign_accumulation(min_consec: int = 3, min_amount_억: float = 30) -> dict:
    """외국인 연속 순매수 시그널 백테스트."""
    logger.info("=== 시그널: 외국인 연속매수 (min_consec=%d, min_amount=%d억) ===", min_consec, min_amount_억)

    all_results = []
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        df = load_parquet(ticker)
        if df is None or len(df) < 60:
            continue
        if "외국인합계" not in df.columns:
            continue

        foreign = df["외국인합계"].fillna(0)
        is_buy = (foreign > 0).astype(int)

        consec = is_buy.copy()
        for idx in range(1, len(consec)):
            if consec.iloc[idx] == 1:
                consec.iloc[idx] = consec.iloc[idx - 1] + 1
            else:
                consec.iloc[idx] = 0

        signal_mask = (consec == min_consec)
        signal_dates = df.index[signal_mask].strftime("%Y-%m-%d").tolist()

        if not signal_dates:
            continue

        filtered_dates = []
        for sd in signal_dates:
            sd_ts = pd.Timestamp(sd)
            idx_pos = df.index.get_loc(sd_ts)
            start = max(0, idx_pos - min_consec + 1)
            total_amt = foreign.iloc[start:idx_pos + 1].sum()
            total_억 = total_amt / 1e8
            if total_억 >= min_amount_억:
                filtered_dates.append(sd)

        if not filtered_dates:
            continue

        ret_df = future_returns(df, filtered_dates)
        if ret_df.empty:
            continue

        ret_df["ticker"] = ticker
        ret_df["signal"] = "FOREIGN_ACCUM_%dd" % min_consec
        all_results.append(ret_df)

        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d", i + 1, total)

    if not all_results:
        logger.warning("  시그널 발생 0건")
        return {}

    result = pd.concat(all_results, ignore_index=True)
    return _summarize(result, "FOREIGN_ACCUM_%dd" % min_consec)


# ─── 시그널 3: 쌍끌이 (기관+외인 동시 매수) ───

def signal_dual_buying(min_consec: int = 3) -> dict:
    """기관+외인 동시 순매수 시그널 백테스트."""
    logger.info("=== 시그널: 쌍끌이 (min_consec=%d) ===", min_consec)

    all_results = []
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        df = load_parquet(ticker)
        if df is None or len(df) < 60:
            continue
        if "기관합계" not in df.columns or "외국인합계" not in df.columns:
            continue

        inst = df["기관합계"].fillna(0)
        foreign = df["외국인합계"].fillna(0)

        # 둘 다 양수인 날
        dual = ((inst > 0) & (foreign > 0)).astype(int)

        consec = dual.copy()
        for idx in range(1, len(consec)):
            if consec.iloc[idx] == 1:
                consec.iloc[idx] = consec.iloc[idx - 1] + 1
            else:
                consec.iloc[idx] = 0

        signal_mask = (consec == min_consec)
        signal_dates = df.index[signal_mask].strftime("%Y-%m-%d").tolist()

        if not signal_dates:
            continue

        ret_df = future_returns(df, signal_dates)
        if ret_df.empty:
            continue

        ret_df["ticker"] = ticker
        ret_df["signal"] = "DUAL_BUY_%dd" % min_consec
        all_results.append(ret_df)

        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d", i + 1, total)

    if not all_results:
        logger.warning("  시그널 발생 0건")
        return {}

    result = pd.concat(all_results, ignore_index=True)
    return _summarize(result, "DUAL_BUY_%dd" % min_consec)


# ─── 시그널 4: picks_history 사후 검증 ───

def signal_picks_history() -> dict:
    """기존 추천종목 사후 수익률 검증."""
    logger.info("=== 시그널: picks_history 사후검증 ===")

    path = DATA_DIR / "picks_history.json"
    if not path.exists():
        logger.warning("  picks_history.json 없음")
        return {}

    with open(path, encoding="utf-8") as f:
        ph = json.load(f)

    records = ph.get("records", [])
    if not records:
        return {}

    all_results = []
    for r in records:
        ticker = r.get("ticker", "")
        pick_date = r.get("pick_date", "")
        grade = r.get("grade", "?")
        if not ticker or not pick_date:
            continue

        df = load_parquet(ticker)
        if df is None:
            continue

        ret_df = future_returns(df, [pick_date])
        if ret_df.empty:
            continue

        ret_df["ticker"] = ticker
        ret_df["name"] = r.get("name", "")
        ret_df["grade"] = grade
        ret_df["score"] = r.get("score", 0)
        ret_df["signal"] = "PICKS_%s" % grade
        all_results.append(ret_df)

    if not all_results:
        return {}

    result = pd.concat(all_results, ignore_index=True)

    # 전체 + grade별 요약
    summary = _summarize(result, "PICKS_ALL")

    # grade별 세분화
    for grade in result["grade"].unique():
        subset = result[result["grade"] == grade]
        if len(subset) >= 3:
            grade_summary = _summarize(subset, "PICKS_%s" % grade)
            summary["by_grade"] = summary.get("by_grade", {})
            summary["by_grade"][grade] = grade_summary

    return summary


# ─── 요약 통계 ───

def _summarize(df: pd.DataFrame, signal_name: str) -> dict:
    """시그널 결과를 요약 통계로 변환."""
    horizons = [c for c in df.columns if c.startswith("D+")]
    n = len(df)

    summary = {
        "signal": signal_name,
        "total_signals": n,
        "period": {
            "start": df["signal_date"].min() if "signal_date" in df.columns else "?",
            "end": df["signal_date"].max() if "signal_date" in df.columns else "?",
        },
    }

    for h in horizons:
        col = df[h].dropna()
        if col.empty:
            continue
        wins = (col > 0).sum()
        summary[h] = {
            "mean": round(col.mean(), 2),
            "median": round(col.median(), 2),
            "std": round(col.std(), 2),
            "win_rate": round(wins / len(col) * 100, 1),
            "avg_win": round(col[col > 0].mean(), 2) if wins > 0 else 0,
            "avg_loss": round(col[col <= 0].mean(), 2) if (len(col) - wins) > 0 else 0,
            "profit_factor": round(
                abs(col[col > 0].sum() / col[col <= 0].sum()), 2
            ) if col[col <= 0].sum() != 0 else float("inf"),
            "n": len(col),
        }

    # 판정
    d5 = summary.get("D+5", {})
    if d5:
        if d5["mean"] > 1.0 and d5["win_rate"] > 55:
            summary["verdict"] = "STRONG_ALPHA"
        elif d5["mean"] > 0.5 and d5["win_rate"] > 50:
            summary["verdict"] = "WEAK_ALPHA"
        elif d5["mean"] > 0 and d5["win_rate"] > 45:
            summary["verdict"] = "MARGINAL"
        else:
            summary["verdict"] = "NO_ALPHA"
    else:
        summary["verdict"] = "INSUFFICIENT_DATA"

    return summary


# ─── 리포트 출력 ───

def print_report(results: dict):
    """콘솔 리포트 출력."""
    signal = results.get("signal", "?")
    n = results.get("total_signals", 0)
    verdict = results.get("verdict", "?")
    period = results.get("period", {})

    print()
    print("=" * 60)
    print("  %s  (n=%d, %s ~ %s)" % (signal, n, period.get("start", "?"), period.get("end", "?")))
    print("=" * 60)
    print()
    print("  %-8s %8s %8s %8s %8s %8s %8s" % ("보유기간", "평균수익", "중앙값", "승률", "PF", "평균이익", "평균손실"))
    print("  " + "-" * 56)

    for h in ["D+1", "D+3", "D+5", "D+10", "D+20"]:
        data = results.get(h)
        if not data:
            continue
        print(
            "  %-8s %7.2f%% %7.2f%% %7.1f%% %8.2f %7.2f%% %7.2f%%"
            % (
                h,
                data["mean"],
                data["median"],
                data["win_rate"],
                data["profit_factor"],
                data["avg_win"],
                data["avg_loss"],
            )
        )

    print()
    verdict_emoji = {
        "STRONG_ALPHA": "★★★ 강한 알파",
        "WEAK_ALPHA": "★★☆ 약한 알파",
        "MARGINAL": "★☆☆ 한계적",
        "NO_ALPHA": "☆☆☆ 알파 없음",
        "INSUFFICIENT_DATA": "??? 데이터 부족",
    }
    print("  판정: %s" % verdict_emoji.get(verdict, verdict))
    print()

    # grade별 세분화 (picks용)
    by_grade = results.get("by_grade", {})
    if by_grade:
        print("  --- grade별 세분화 ---")
        for grade, gdata in by_grade.items():
            d5 = gdata.get("D+5", {})
            if d5:
                print(
                    "  [%s] n=%d, D+5 평균=%.2f%%, 승률=%.1f%%, PF=%.2f → %s"
                    % (grade, gdata["total_signals"], d5["mean"], d5["win_rate"], d5["profit_factor"], gdata.get("verdict", "?"))
                )
        print()


# ─── 메인 ───

def main():
    parser = argparse.ArgumentParser(description="알파 백테스트")
    parser.add_argument("--signal", choices=["inst", "foreign", "dual", "picks", "all"], default="all")
    parser.add_argument("--min-consec", type=int, default=3, help="최소 연속 매수일 (기본 3)")
    parser.add_argument("--save", action="store_true", help="결과를 JSON으로 저장")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.signal in ("inst", "all"):
        for consec in [3, 5, 7, 10]:
            r = signal_institutional_accumulation(min_consec=consec)
            if r:
                print_report(r)
                all_results[r["signal"]] = r

    if args.signal in ("foreign", "all"):
        for consec in [3, 5, 7]:
            r = signal_foreign_accumulation(min_consec=consec)
            if r:
                print_report(r)
                all_results[r["signal"]] = r

    if args.signal in ("dual", "all"):
        for consec in [3, 5]:
            r = signal_dual_buying(min_consec=consec)
            if r:
                print_report(r)
                all_results[r["signal"]] = r

    if args.signal in ("picks", "all"):
        r = signal_picks_history()
        if r:
            print_report(r)
            all_results[r["signal"]] = r

    # 종합 판정
    if all_results:
        print()
        print("=" * 60)
        print("  종합 판정")
        print("=" * 60)
        for name, r in all_results.items():
            d5 = r.get("D+5", {})
            print(
                "  %-25s n=%-5d D+5 avg=%-7s WR=%-6s → %s"
                % (
                    name,
                    r["total_signals"],
                    "%.2f%%" % d5["mean"] if d5 else "N/A",
                    "%.1f%%" % d5["win_rate"] if d5 else "N/A",
                    r.get("verdict", "?"),
                )
            )

    if args.save and all_results:
        save_path = RESULT_DIR / ("backtest_%s.json" % datetime.now().strftime("%Y%m%d_%H%M"))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info("결과 저장: %s", save_path)


if __name__ == "__main__":
    main()
