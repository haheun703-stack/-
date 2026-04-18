"""수급-가격 디버전스 백테스트

핵심 질문: "주가는 떨어지는데 기관/외인은 사고 있다" → D+5 수익률이 양(+)인가?

조건 변형 8개:
  DIV_INST_5d_5pct   : 5일 -5% 이상 하락 + 기관 3일+ 순매수
  DIV_INST_5d_10pct  : 5일 -10% 이상 하락 + 기관 3일+ 순매수
  DIV_FOR_5d_5pct    : 5일 -5% 이상 하락 + 외인 3일+ 순매수
  DIV_FOR_5d_10pct   : 5일 -10% 이상 하락 + 외인 3일+ 순매수
  DIV_DUAL_5d_5pct   : 5일 -5% 이상 하락 + 기관+외인 동시 3일+ 순매수
  DIV_DUAL_5d_10pct  : 5일 -10% 이상 하락 + 기관+외인 동시 3일+ 순매수
  DIV_INST_10d_10pct : 10일 -10% 이상 하락 + 기관 5일+ 순매수
  DIV_DUAL_10d_10pct : 10일 -10% 이상 하락 + 기관+외인 동시 5일+ 순매수

Usage (VPS):
    python -u scripts/backtest_divergence.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULT_PATH = DATA_DIR / "alpha_backtest" / "divergence_result.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── 유틸 ──

def load_parquet(ticker: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def future_returns(df: pd.DataFrame, signal_dates: list, horizons=None) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]
    results = []
    for d in signal_dates:
        if d not in df.index:
            after = df.index[df.index >= d]
            if after.empty:
                continue
            d = after[0]
        entry = df.loc[d, "close"]
        if entry <= 0:
            continue
        row = {"signal_date": d.strftime("%Y-%m-%d"), "entry_price": entry}
        for h in horizons:
            future = df.index[df.index > d]
            if len(future) < h:
                row[f"D+{h}"] = np.nan
            else:
                row[f"D+{h}"] = round((df.loc[future[h-1], "close"] - entry) / entry * 100, 2)
        results.append(row)
    return pd.DataFrame(results)


def summarize(df: pd.DataFrame, name: str) -> dict:
    horizons = [c for c in df.columns if c.startswith("D+")]
    n = len(df)
    s = {"signal": name, "n": n}
    for h in horizons:
        col = df[h].dropna()
        if col.empty:
            continue
        wins = (col > 0).sum()
        losses = len(col) - wins
        s[h] = {
            "mean": round(col.mean(), 2),
            "median": round(col.median(), 2),
            "wr": round(wins / len(col) * 100, 1),
            "pf": round(abs(col[col > 0].sum() / col[col <= 0].sum()), 2) if col[col <= 0].sum() != 0 else 99,
            "n": len(col),
        }
    # 판정
    d5 = s.get("D+5", {})
    if d5:
        if d5["mean"] > 1.5 and d5["wr"] > 55:
            s["verdict"] = "STRONG_ALPHA"
        elif d5["mean"] > 0.8 and d5["wr"] > 52:
            s["verdict"] = "WEAK_ALPHA"
        elif d5["mean"] > 0 and d5["wr"] > 48:
            s["verdict"] = "MARGINAL"
        else:
            s["verdict"] = "NO_ALPHA"
    return s


# ── 디버전스 시그널 생성 ──

CONFIGS = [
    # (이름, 하락기간, 하락률, 수급유형, 수급연속일)
    ("DIV_INST_5d_5pct",    5,  -5,  "inst",  3),
    ("DIV_INST_5d_10pct",   5,  -10, "inst",  3),
    ("DIV_FOR_5d_5pct",     5,  -5,  "for",   3),
    ("DIV_FOR_5d_10pct",    5,  -10, "for",   3),
    ("DIV_DUAL_5d_5pct",    5,  -5,  "dual",  3),
    ("DIV_DUAL_5d_10pct",   5,  -10, "dual",  3),
    ("DIV_INST_10d_10pct",  10, -10, "inst",  5),
    ("DIV_DUAL_10d_10pct",  10, -10, "dual",  5),
]


def find_divergence_signals(df: pd.DataFrame, price_window: int, price_drop_pct: float,
                            flow_type: str, flow_consec: int) -> list:
    """수급-가격 디버전스 시그널 발생일 찾기.

    Args:
        df: 종목 일봉 (close, 기관합계, 외국인합계 필수)
        price_window: 가격 하락 관찰 기간 (일)
        price_drop_pct: 최소 하락률 (%, 음수: -5 = 5% 하락)
        flow_type: "inst" / "for" / "dual"
        flow_consec: 수급 연속 순매수 최소 일수

    Returns:
        시그널 발생일 리스트 (pd.Timestamp)
    """
    if "기관합계" not in df.columns or "외국인합계" not in df.columns:
        return []

    close = df["close"]
    inst = df["기관합계"].fillna(0)
    foreign = df["외국인합계"].fillna(0)

    # 가격 변화율 (N일 수익률)
    price_ret = close.pct_change(price_window) * 100

    # 수급 연속 순매수 일수 계산
    if flow_type == "inst":
        is_buy = (inst > 0).astype(int)
    elif flow_type == "for":
        is_buy = (foreign > 0).astype(int)
    elif flow_type == "dual":
        is_buy = ((inst > 0) & (foreign > 0)).astype(int)
    else:
        return []

    # 연속 일수 계산
    consec = is_buy.copy()
    for i in range(1, len(consec)):
        if consec.iloc[i] == 1:
            consec.iloc[i] = consec.iloc[i-1] + 1
        else:
            consec.iloc[i] = 0

    # 디버전스 시그널: 가격 하락 + 수급 연속 매수
    mask = (price_ret <= price_drop_pct) & (consec >= flow_consec)

    return df.index[mask].tolist()


def run_backtest():
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)
    logger.info("총 %d 종목 parquet 로드", total)

    all_results = {}

    for cfg_name, pw, pd_pct, ft, fc in CONFIGS:
        all_results[cfg_name] = []

    for i, pf in enumerate(parquet_files):
        ticker = pf.stem
        df = load_parquet(ticker)
        if df is None or len(df) < 60:
            continue

        for cfg_name, pw, pd_pct, ft, fc in CONFIGS:
            signals = find_divergence_signals(df, pw, pd_pct, ft, fc)
            if not signals:
                continue

            ret_df = future_returns(df, signals)
            if ret_df.empty:
                continue

            ret_df["ticker"] = ticker
            all_results[cfg_name].append(ret_df)

        if (i + 1) % 200 == 0:
            logger.info("  진행: %d/%d", i + 1, total)

    # 결과 요약
    final = {}
    for cfg_name, _, _, _, _ in CONFIGS:
        dfs = all_results[cfg_name]
        if not dfs:
            logger.info("  %s: 시그널 0건", cfg_name)
            final[cfg_name] = {"signal": cfg_name, "n": 0, "verdict": "NO_DATA"}
            continue

        combined = pd.concat(dfs, ignore_index=True)
        s = summarize(combined, cfg_name)
        final[cfg_name] = s

        d5 = s.get("D+5", {})
        logger.info("  %s: n=%d, D+5 avg=%.2f%%, WR=%.1f%%, PF=%.2f → %s",
                    cfg_name, s["n"],
                    d5.get("mean", 0), d5.get("wr", 0), d5.get("pf", 0),
                    s.get("verdict", "?"))

    # 저장
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장: %s", RESULT_PATH)

    # 리포트 출력
    print()
    print("=" * 80)
    print("  수급-가격 디버전스 백테스트 결과")
    print("=" * 80)
    print()
    print("  %-25s %6s %8s %6s %6s %10s" % ("시그널", "건수", "D+5 avg", "WR", "PF", "판정"))
    print("  " + "-" * 65)
    for cfg_name, _, _, _, _ in CONFIGS:
        s = final[cfg_name]
        d5 = s.get("D+5", {})
        print("  %-25s %6d %7.2f%% %5.1f%% %6.2f %10s" % (
            cfg_name, s["n"],
            d5.get("mean", 0), d5.get("wr", 0), d5.get("pf", 0),
            s.get("verdict", "NO_DATA"),
        ))

    # STRONG_ALPHA 강조
    print()
    strong = [k for k, v in final.items() if v.get("verdict") == "STRONG_ALPHA"]
    weak = [k for k, v in final.items() if v.get("verdict") == "WEAK_ALPHA"]
    if strong:
        print("  ★ STRONG_ALPHA: %s" % ", ".join(strong))
    if weak:
        print("  ○ WEAK_ALPHA: %s" % ", ".join(weak))
    if not strong and not weak:
        print("  ✗ 알파 시그널 없음")


if __name__ == "__main__":
    run_backtest()
