"""금요일 오후 투매 → 월요일 반등 패턴 실증 분석

parquet 데이터 (3년, 1030종목)로 통계 검증:
  1) 요일별 수익률 비교
  2) 금요일 당일 고가→종가 하락 (오후 투매 프록시)
  3) 투매 후 월요일 반등 확률
  4) 기술지표 필터별 효과

사용법:
  python -u -X utf8 scripts/analyze_friday_pattern.py
  python -u -X utf8 scripts/analyze_friday_pattern.py --telegram  # 결과 텔레그램 발송
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "data" / "friday_pattern_analysis.json"


def load_universe(min_rows: int = 500) -> dict[str, pd.DataFrame]:
    """parquet 유니버스 로드 (최소 500일 이상)."""
    universe = {}
    for f in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["open", "high", "low", "close", "volume",
                                              "rsi_14", "bb_position", "외국인합계", "기관합계"])
            if len(df) >= min_rows:
                df.index = pd.to_datetime(df.index)
                universe[f.stem] = df
        except Exception:
            pass
    return universe


def analyze_weekday_returns(universe: dict[str, pd.DataFrame]) -> dict:
    """분석 1: 요일별 수익률 비교."""
    all_returns = {i: [] for i in range(5)}  # 0=월 ~ 4=금

    for ticker, df in universe.items():
        df["ret"] = df["close"].pct_change()
        df["weekday"] = df.index.weekday
        for wd in range(5):
            mask = df["weekday"] == wd
            rets = df.loc[mask, "ret"].dropna()
            all_returns[wd].extend(rets.tolist())

    day_names = ["월", "화", "수", "목", "금"]
    result = {}
    for wd in range(5):
        arr = np.array(all_returns[wd])
        result[day_names[wd]] = {
            "mean_pct": round(float(np.mean(arr)) * 100, 4),
            "median_pct": round(float(np.median(arr)) * 100, 4),
            "positive_rate": round(float(np.mean(arr > 0)) * 100, 2),
            "count": len(arr),
        }
    return result


def analyze_friday_to_monday(universe: dict[str, pd.DataFrame]) -> dict:
    """분석 2: 금요일 종가 → 월요일 시가/종가 반등."""
    fri_to_mon_open = []
    fri_to_mon_close = []

    for ticker, df in universe.items():
        df["weekday"] = df.index.weekday
        fridays = df[df["weekday"] == 4].copy()

        for idx, fri_row in fridays.iterrows():
            # 다음 거래일 (월요일) 찾기
            next_days = df.loc[df.index > idx].head(3)
            monday = next_days[next_days.index.weekday == 0]
            if monday.empty:
                # 공휴일 등으로 월요일이 아닐 수 있음 → 다음 거래일
                if len(next_days) > 0:
                    monday = next_days.iloc[[0]]
                else:
                    continue

            mon_row = monday.iloc[0]
            fri_close = fri_row["close"]
            if fri_close <= 0:
                continue

            gap_open = (mon_row["open"] - fri_close) / fri_close
            gap_close = (mon_row["close"] - fri_close) / fri_close

            fri_to_mon_open.append(gap_open)
            fri_to_mon_close.append(gap_close)

    arr_open = np.array(fri_to_mon_open)
    arr_close = np.array(fri_to_mon_close)

    return {
        "fri_to_mon_open": {
            "mean_pct": round(float(np.mean(arr_open)) * 100, 4),
            "median_pct": round(float(np.median(arr_open)) * 100, 4),
            "positive_rate": round(float(np.mean(arr_open > 0)) * 100, 2),
            "count": len(arr_open),
        },
        "fri_to_mon_close": {
            "mean_pct": round(float(np.mean(arr_close)) * 100, 4),
            "median_pct": round(float(np.median(arr_close)) * 100, 4),
            "positive_rate": round(float(np.mean(arr_close > 0)) * 100, 2),
            "count": len(arr_close),
        },
    }


def analyze_friday_dumping(universe: dict[str, pd.DataFrame],
                           thresholds: list[float] = [-2, -3, -5]) -> dict:
    """분석 3: 금요일 당일 고가 대비 종가 하락 (투매 프록시) → 월요일 반등."""
    results = {}

    for thr in thresholds:
        entries = []
        for ticker, df in universe.items():
            df["weekday"] = df.index.weekday
            df["high_to_close_pct"] = (df["close"] - df["high"]) / df["high"] * 100
            fridays = df[(df["weekday"] == 4) & (df["high_to_close_pct"] <= thr)]

            for idx, fri_row in fridays.iterrows():
                next_days = df.loc[df.index > idx].head(3)
                if len(next_days) == 0:
                    continue
                mon_row = next_days.iloc[0]
                fri_close = fri_row["close"]
                if fri_close <= 0:
                    continue

                mon_open_ret = (mon_row["open"] - fri_close) / fri_close * 100
                mon_close_ret = (mon_row["close"] - fri_close) / fri_close * 100

                entries.append({
                    "mon_open_ret": mon_open_ret,
                    "mon_close_ret": mon_close_ret,
                    "fri_drop": fri_row["high_to_close_pct"],
                    "rsi": fri_row.get("rsi_14", np.nan),
                    "bb_pos": fri_row.get("bb_position", np.nan),
                    "foreign": fri_row.get("외국인합계", 0),
                    "inst": fri_row.get("기관합계", 0),
                })

        if not entries:
            results[f"drop_{abs(thr):.0f}pct"] = {"count": 0}
            continue

        edf = pd.DataFrame(entries)
        results[f"drop_{abs(thr):.0f}pct"] = {
            "count": len(edf),
            "mon_open_ret_mean": round(float(edf["mon_open_ret"].mean()), 4),
            "mon_close_ret_mean": round(float(edf["mon_close_ret"].mean()), 4),
            "mon_close_positive_rate": round(float((edf["mon_close_ret"] > 0).mean()) * 100, 2),
            "mon_close_median": round(float(edf["mon_close_ret"].median()), 4),
            "avg_fri_drop": round(float(edf["fri_drop"].mean()), 2),
        }

    return results


def analyze_filtered_dumping(universe: dict[str, pd.DataFrame]) -> dict:
    """분석 4: 기술지표 필터 + 투매 → 반등 효과."""
    entries = []

    for ticker, df in universe.items():
        df["weekday"] = df.index.weekday
        df["high_to_close_pct"] = (df["close"] - df["high"]) / df["high"] * 100
        fridays = df[(df["weekday"] == 4) & (df["high_to_close_pct"] <= -2)]

        for idx, fri_row in fridays.iterrows():
            next_days = df.loc[df.index > idx].head(3)
            if len(next_days) == 0:
                continue
            mon_row = next_days.iloc[0]
            fri_close = fri_row["close"]
            if fri_close <= 0:
                continue

            mon_ret = (mon_row["close"] - fri_close) / fri_close * 100
            rsi = fri_row.get("rsi_14", np.nan)
            bb = fri_row.get("bb_position", np.nan)
            foreign = fri_row.get("외국인합계", 0)
            inst = fri_row.get("기관합계", 0)

            entries.append({
                "mon_ret": mon_ret,
                "rsi": rsi,
                "bb_pos": bb,
                "foreign_sell": foreign < 0,
                "inst_sell": inst < 0,
                "both_sell": foreign < 0 and inst < 0,
                "foreign_buy_30d": False,  # 30일 추세는 parquet 컬럼에서 확인
            })

    if not entries:
        return {"no_data": True}

    edf = pd.DataFrame(entries)
    base_rate = float((edf["mon_ret"] > 0).mean()) * 100
    base_mean = float(edf["mon_ret"].mean())

    filters = {}

    # RSI < 40
    mask = edf["rsi"] < 40
    if mask.sum() > 10:
        sub = edf[mask]
        filters["rsi_under_40"] = {
            "count": int(mask.sum()),
            "win_rate": round(float((sub["mon_ret"] > 0).mean()) * 100, 2),
            "mean_ret": round(float(sub["mon_ret"].mean()), 4),
        }

    # RSI > 60 (과매수인데 투매 = 강한 역전?)
    mask = edf["rsi"] > 60
    if mask.sum() > 10:
        sub = edf[mask]
        filters["rsi_over_60"] = {
            "count": int(mask.sum()),
            "win_rate": round(float((sub["mon_ret"] > 0).mean()) * 100, 2),
            "mean_ret": round(float(sub["mon_ret"].mean()), 4),
        }

    # BB < 30%
    mask = edf["bb_pos"] < 0.3
    if mask.sum() > 10:
        sub = edf[mask]
        filters["bb_under_30"] = {
            "count": int(mask.sum()),
            "win_rate": round(float((sub["mon_ret"] > 0).mean()) * 100, 2),
            "mean_ret": round(float(sub["mon_ret"].mean()), 4),
        }

    # 외+기 쌍매도
    mask = edf["both_sell"]
    if mask.sum() > 10:
        sub = edf[mask]
        filters["both_selling"] = {
            "count": int(mask.sum()),
            "win_rate": round(float((sub["mon_ret"] > 0).mean()) * 100, 2),
            "mean_ret": round(float(sub["mon_ret"].mean()), 4),
        }

    # 외국인만 매도 (기관은 매수)
    mask = (edf["foreign_sell"]) & (~edf["inst_sell"])
    if mask.sum() > 10:
        sub = edf[mask]
        filters["foreign_sell_only"] = {
            "count": int(mask.sum()),
            "win_rate": round(float((sub["mon_ret"] > 0).mean()) * 100, 2),
            "mean_ret": round(float(sub["mon_ret"].mean()), 4),
        }

    return {
        "base_count": len(edf),
        "base_win_rate": round(base_rate, 2),
        "base_mean_ret": round(base_mean, 4),
        "filters": filters,
    }


def analyze_all_weekday_dumping(universe: dict[str, pd.DataFrame]) -> dict:
    """비교군: 금요일뿐 아니라 모든 요일에서 고가→종가 -2% 이상 → 다음날 반등."""
    day_names = ["월", "화", "수", "목", "금"]
    results = {}

    for wd in range(5):
        entries = []
        for ticker, df in universe.items():
            df["weekday"] = df.index.weekday
            df["high_to_close_pct"] = (df["close"] - df["high"]) / df["high"] * 100
            target_days = df[(df["weekday"] == wd) & (df["high_to_close_pct"] <= -2)]

            for idx, row in target_days.iterrows():
                next_days = df.loc[df.index > idx].head(1)
                if len(next_days) == 0:
                    continue
                nxt = next_days.iloc[0]
                if row["close"] <= 0:
                    continue
                ret = (nxt["close"] - row["close"]) / row["close"] * 100
                entries.append(ret)

        if entries:
            arr = np.array(entries)
            results[day_names[wd]] = {
                "count": len(arr),
                "win_rate": round(float(np.mean(arr > 0)) * 100, 2),
                "mean_ret": round(float(np.mean(arr)), 4),
                "median_ret": round(float(np.median(arr)), 4),
            }

    return results


def format_telegram_message(analysis: dict) -> str:
    """텔레그램 메시지 포맷."""
    lines = [
        "📊 [금요일 오후 투매 패턴 분석]",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    # 요일별 수익률
    wd = analysis["weekday_returns"]
    lines.append("1) 요일별 평균 수익률")
    for day in ["월", "화", "수", "목", "금"]:
        d = wd[day]
        lines.append(f"  {day}: {d['mean_pct']:+.4f}% (양봉 {d['positive_rate']:.1f}%)")

    # 금→월 갭
    lines.append("")
    lines.append("2) 금요일→월요일")
    f2m = analysis["friday_to_monday"]
    lines.append(f"  시가갭: {f2m['fri_to_mon_open']['mean_pct']:+.4f}% (양봉 {f2m['fri_to_mon_open']['positive_rate']:.1f}%)")
    lines.append(f"  종가갭: {f2m['fri_to_mon_close']['mean_pct']:+.4f}% (양봉 {f2m['fri_to_mon_close']['positive_rate']:.1f}%)")

    # 투매 후 반등
    lines.append("")
    lines.append("3) 금요일 투매(고가→종가) 후 월요일")
    dump = analysis["friday_dumping"]
    for key in ["drop_2pct", "drop_3pct", "drop_5pct"]:
        d = dump.get(key, {})
        if d.get("count", 0) > 0:
            lines.append(f"  {key}: {d['count']}건, 승률 {d['mon_close_positive_rate']:.1f}%, 평균 {d['mon_close_ret_mean']:+.4f}%")

    # 요일별 투매 비교
    lines.append("")
    lines.append("4) 요일별 투매(-2%) 다음날 반등 비교")
    awd = analysis["all_weekday_dumping"]
    for day in ["월", "화", "수", "목", "금"]:
        d = awd.get(day, {})
        if d:
            lines.append(f"  {day}: {d['count']}건, 승률 {d['win_rate']:.1f}%, 평균 {d['mean_ret']:+.4f}%")

    # 필터 효과
    lines.append("")
    lines.append("5) 필터별 효과 (금요일 -2% 투매)")
    filt = analysis.get("filtered_dumping", {})
    if filt.get("base_count"):
        lines.append(f"  기본: {filt['base_count']}건, 승률 {filt['base_win_rate']:.1f}%, 평균 {filt['base_mean_ret']:+.4f}%")
        for name, f in filt.get("filters", {}).items():
            lines.append(f"  {name}: {f['count']}건, 승률 {f['win_rate']:.1f}%, 평균 {f['mean_ret']:+.4f}%")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 최종 판정
    dump3 = dump.get("drop_3pct", {})
    if dump3.get("mon_close_positive_rate", 0) >= 55:
        lines.append("✅ 패턴 존재! 금요일 -3% 투매 → 월요일 반등 유효")
    elif dump3.get("mon_close_positive_rate", 0) >= 50:
        lines.append("⚠️ 경계선 — 필터 조합으로 승률 개선 필요")
    else:
        lines.append("❌ 패턴 미약 — 금요일 투매 역매수 전략 재검토 필요")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="금요일 오후 투매 패턴 분석")
    parser.add_argument("--telegram", action="store_true", help="결과 텔레그램 발송")
    parser.add_argument("--min-rows", type=int, default=500, help="최소 데이터 행수")
    args = parser.parse_args()

    print("=" * 60)
    print("금요일 오후 투매 → 월요일 반등 패턴 분석")
    print("=" * 60)

    print("\n유니버스 로딩...")
    universe = load_universe(min_rows=args.min_rows)
    print(f"  {len(universe)}종목 로드")

    print("\n분석 1) 요일별 수익률...")
    weekday_returns = analyze_weekday_returns(universe)
    for day, d in weekday_returns.items():
        print(f"  {day}: {d['mean_pct']:+.4f}% (양봉 {d['positive_rate']:.1f}%, {d['count']:,}건)")

    print("\n분석 2) 금→월 갭...")
    fri_to_mon = analyze_friday_to_monday(universe)
    print(f"  시가: {fri_to_mon['fri_to_mon_open']['mean_pct']:+.4f}% ({fri_to_mon['fri_to_mon_open']['count']:,}건)")
    print(f"  종가: {fri_to_mon['fri_to_mon_close']['mean_pct']:+.4f}% ({fri_to_mon['fri_to_mon_close']['count']:,}건)")

    print("\n분석 3) 투매 후 반등...")
    dumping = analyze_friday_dumping(universe)
    for key, d in dumping.items():
        if d.get("count", 0) > 0:
            print(f"  {key}: {d['count']}건, 승률 {d['mon_close_positive_rate']:.1f}%, 평균 {d['mon_close_ret_mean']:+.4f}%")

    print("\n분석 4) 요일별 투매 비교...")
    all_wd = analyze_all_weekday_dumping(universe)
    for day, d in all_wd.items():
        print(f"  {day}: {d['count']}건, 승률 {d['win_rate']:.1f}%, 평균 {d['mean_ret']:+.4f}%")

    print("\n분석 5) 필터 효과...")
    filtered = analyze_filtered_dumping(universe)
    if filtered.get("base_count"):
        print(f"  기본: 승률 {filtered['base_win_rate']:.1f}%, 평균 {filtered['base_mean_ret']:+.4f}%")
        for name, f in filtered.get("filters", {}).items():
            print(f"  {name}: {f['count']}건, 승률 {f['win_rate']:.1f}%, 평균 {f['mean_ret']:+.4f}%")

    # 결과 저장
    analysis = {
        "weekday_returns": weekday_returns,
        "friday_to_monday": fri_to_mon,
        "friday_dumping": dumping,
        "all_weekday_dumping": all_wd,
        "filtered_dumping": filtered,
        "universe_count": len(universe),
        "analyzed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    OUTPUT_PATH.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n저장: {OUTPUT_PATH}")

    # 텔레그램
    msg = format_telegram_message(analysis)
    print(f"\n{msg}")

    if args.telegram:
        try:
            from src.telegram_sender import send_message
            send_message(msg)
            print("\n텔레그램 발송 완료!")
        except Exception as e:
            print(f"\n텔레그램 발송 실패: {e}")


if __name__ == "__main__":
    main()
