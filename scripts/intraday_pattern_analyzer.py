"""장후 패턴 분석기 (Phase 12b)

intraday_learner가 수집한 1분봉 DB를 분석:
- 당일 +5%↑ 종목들의 09:00~10:00 공통 패턴 추출
- 체결강도 / 거래량 가속 / OHLC 분포
- 누적 학습 데이터로 다음날 진입 추천 시그널 생성

입력:
- data/intraday/intraday_minute_{YYYYMMDD}.db

출력:
- data/intraday/patterns_{YYYYMMDD}.json (당일 결과)
- data/intraday/intraday_patterns_cumulative.json (누적 통계)
- data/intraday/intraday_signals_{tomorrow}.json (다음날 진입 시그널)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import FinanceDataReader as fdr

DATA_DIR = PROJECT_ROOT / "data"
INTRA_DIR = DATA_DIR / "intraday"

LOG_PATH = PROJECT_ROOT / "logs" / "intraday_pattern.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("intraday_pattern")

EARLY_END = "1000"  # 09:00~10:00 패턴 추출
SURGE_THRESHOLD = 5.0  # 당일 +5%↑ = 급등 라벨


def load_minute_db(date_compact: str) -> pd.DataFrame:
    db = INTRA_DIR / f"intraday_minute_{date_compact}.db"
    if not db.exists():
        logger.error(f"[load] DB 없음: {db}")
        return pd.DataFrame()
    conn = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM intraday_minute ORDER BY code, minute", conn)
    conn.close()
    logger.info(f"[load] {db.name}: {len(df)}행, {df['code'].nunique()}종목")
    return df


def load_daily_close(date_str: str, codes: list) -> dict:
    """당일 종가 + 전일 종가 비교 → 일간 수익률."""
    rets = {}
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end = target_dt.strftime("%Y-%m-%d")
    for code in codes:
        try:
            df = fdr.DataReader(code, start, end)
            if len(df) >= 2:
                ret = df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
                rets[code] = round(ret * 100, 2)
        except Exception:
            continue
    return rets


def extract_early_features(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 09:00~10:00 early window 특징 추출."""
    early = df[df["minute"] < EARLY_END].copy()
    feat_rows = []
    for code, sub in early.groupby("code"):
        if len(sub) < 5:
            continue
        # OHLC
        first = sub.iloc[0]
        last = sub.iloc[-1]
        early_ret = (last["close"] - first["open"]) / first["open"] * 100 if first["open"] else 0
        # 체결강도
        strength_avg = sub["strength_avg"].mean()
        strength_max = sub["strength_avg"].max()
        # 거래량
        vol_sum = sub["volume"].sum()
        cum_vol_10 = last["cum_volume"]
        # 매수/매도 비율
        buy_total = sub["buy_count"].iloc[-1] - sub["buy_count"].iloc[0]
        sell_total = sub["sell_count"].iloc[-1] - sub["sell_count"].iloc[0]
        buy_ratio = buy_total / (buy_total + sell_total) if (buy_total + sell_total) else 0.5
        feat_rows.append({
            "code": code,
            "early_ret_pct": round(early_ret, 2),
            "strength_avg": round(strength_avg, 2),
            "strength_max": round(strength_max, 2),
            "vol_early": int(vol_sum),
            "cum_vol_10": int(cum_vol_10),
            "buy_ratio": round(buy_ratio, 3),
            "minute_count": len(sub),
        })
    return pd.DataFrame(feat_rows)


def analyze_day(date_str: str) -> dict:
    """당일 분석: 급등 종목들의 공통 early 패턴 추출."""
    date_compact = date_str.replace("-", "")
    minute_df = load_minute_db(date_compact)
    if minute_df.empty:
        return {}

    features = extract_early_features(minute_df)
    if features.empty:
        logger.warning(f"[analyze] {date_str} early features 없음")
        return {}

    codes = features["code"].tolist()
    daily_rets = load_daily_close(date_str, codes)
    features["day_ret_pct"] = features["code"].map(daily_rets).fillna(0)

    # 라벨: 급등(>=5%), 정상(0~5%), 하락(<0%)
    features["label"] = pd.cut(
        features["day_ret_pct"],
        bins=[-100, 0, SURGE_THRESHOLD, 100],
        labels=["DOWN", "FLAT", "SURGE"],
    )

    # 그룹별 통계
    summary = features.groupby("label", observed=True).agg(
        n=("code", "count"),
        early_ret_avg=("early_ret_pct", "mean"),
        strength_avg=("strength_avg", "mean"),
        strength_max=("strength_max", "mean"),
        buy_ratio=("buy_ratio", "mean"),
        cum_vol_10_avg=("cum_vol_10", "mean"),
    ).round(2)
    logger.info(f"\n[summary] {date_str}\n{summary}")

    # 급등 종목 상세
    surges = features[features["label"] == "SURGE"].sort_values("day_ret_pct", ascending=False)
    logger.info(f"\n[surges] {len(surges)}종목\n{surges.to_string(index=False)}")

    out = {
        "date": date_str,
        "total_codes": len(features),
        "summary": summary.reset_index().to_dict(orient="records"),
        "surges": surges.to_dict(orient="records"),
        "features": features.to_dict(orient="records"),
    }

    # 당일 결과 저장
    out_path = INTRA_DIR / f"patterns_{date_compact}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"[save] {out_path}")
    return out


def update_cumulative(today_result: dict):
    """누적 통계 업데이트 (모든 분석 결과 누적)."""
    cum_path = INTRA_DIR / "intraday_patterns_cumulative.json"
    if cum_path.exists():
        try:
            cum = json.loads(cum_path.read_text(encoding="utf-8"))
        except Exception:
            cum = {"days": [], "surge_features": []}
    else:
        cum = {"days": [], "surge_features": []}

    date = today_result.get("date")
    if date and date not in cum["days"]:
        cum["days"].append(date)
        for s in today_result.get("surges", []):
            s["date"] = date
            cum["surge_features"].append(s)

    # 누적 통계 (급등 종목의 평균 early 패턴)
    if cum["surge_features"]:
        sdf = pd.DataFrame(cum["surge_features"])
        cum["surge_thresholds"] = {
            "n_total_surges": len(sdf),
            "n_days": len(cum["days"]),
            "early_ret_p50": round(sdf["early_ret_pct"].median(), 2),
            "early_ret_p75": round(sdf["early_ret_pct"].quantile(0.75), 2),
            "strength_avg_p50": round(sdf["strength_avg"].median(), 2),
            "strength_avg_p75": round(sdf["strength_avg"].quantile(0.75), 2),
            "buy_ratio_p50": round(sdf["buy_ratio"].median(), 3),
            "buy_ratio_p75": round(sdf["buy_ratio"].quantile(0.75), 3),
        }
        logger.info(f"[cumulative] {cum['surge_thresholds']}")

    cum_path.write_text(json.dumps(cum, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"[save] {cum_path}")


def generate_next_signals(today_result: dict, tomorrow_str: str):
    """누적 임계치 기반으로 내일 진입 후보 시그널 생성.

    조건 (학습 누적 후 자동 보정):
    - early_ret >= P50 (학습된 급등 종목 중앙값)
    - strength_avg >= P50
    - buy_ratio >= P50

    오늘 features 중 위 조건 통과한 종목을 내일 BUY 후보로.
    """
    cum_path = INTRA_DIR / "intraday_patterns_cumulative.json"
    if not cum_path.exists():
        logger.info("[signals] 누적 데이터 없음, 시그널 생성 스킵")
        return
    try:
        cum = json.loads(cum_path.read_text(encoding="utf-8"))
    except Exception:
        return

    th = cum.get("surge_thresholds")
    if not th or th.get("n_total_surges", 0) < 5:
        logger.info(f"[signals] 학습 표본 부족 ({th.get('n_total_surges', 0) if th else 0}/5), 스킵")
        return

    er_th = th["early_ret_p50"]
    st_th = th["strength_avg_p50"]
    br_th = th["buy_ratio_p50"]
    logger.info(f"[signals] 임계: early_ret>={er_th}%, strength>={st_th}, buy_ratio>={br_th}")

    candidates = []
    for f in today_result.get("features", []):
        if (
            f["early_ret_pct"] >= er_th
            and f["strength_avg"] >= st_th
            and f["buy_ratio"] >= br_th
        ):
            candidates.append({
                "code": f["code"],
                "early_ret_pct": f["early_ret_pct"],
                "strength_avg": f["strength_avg"],
                "buy_ratio": f["buy_ratio"],
                "day_ret_pct": f.get("day_ret_pct", 0),
            })

    candidates.sort(key=lambda x: x["early_ret_pct"], reverse=True)
    sig_path = INTRA_DIR / f"intraday_signals_{tomorrow_str.replace('-', '')}.json"
    sig_path.write_text(
        json.dumps({
            "date": tomorrow_str,
            "thresholds": th,
            "candidates": candidates[:10],
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"[signals] {tomorrow_str} 후보 {len(candidates[:10])}종목 저장")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (기본: today)")
    args = parser.parse_args()

    today = args.date or datetime.now().strftime("%Y-%m-%d")
    result = analyze_day(today)
    if result:
        update_cumulative(result)
        tomorrow = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        generate_next_signals(result, tomorrow)


if __name__ == "__main__":
    main()
