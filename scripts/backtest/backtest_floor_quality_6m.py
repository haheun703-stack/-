"""6개월 워크포워드 백테스트 — 4중결합 + 수정가설 OOS 검증.

목적: 1차(in-sample)에서 4중결합이 base를 못 이겼다. 수정 가설(횡보 우선 / stock_specific 회피 /
단독 수급)을 다른 기간(out-of-sample)으로 재검증한다. threshold 사후튜닝 금지 — 가설 3개만 반영.

비교군 6:
  base          : 차트 4조건 단독
  legacy4       : 기존 4중결합 (in-sample에서 base 못이김)
  hypoA         : 횡보(바닥다지기) + stock_specific 회피
  hypoB         : 횡보 + 단독수급(외인/기관 accumulation)
  hypoC         : 횡보 + stock_specific 회피 + 단독수급
  supply_solo   : 단독수급만

look-ahead 안전: build_candidate(hist<=asof), 다음거래일 시가 진입, D+N 종가, point-in-time,
액면분할/0행 점프가드(진입~D+10 구간 close<=0 또는 40%+ 점프 제외). 실주문 0 / dry.

유니버스: 전체상장 1064(탐색) + 유동성 273(성과판정 subset). 시총 3분위(대형/중형/소형).

사용:
  python -u -X utf8 scripts/backtest/backtest_floor_quality_6m.py --tag in_sample  --start 2025-12-08 --end 2026-06-05
  python -u -X utf8 scripts/backtest/backtest_floor_quality_6m.py --tag oos        --start 2025-06-09 --end 2025-12-05
  python -u -X utf8 scripts/backtest/backtest_floor_quality_6m.py --tag dry --sample 30 --start 2026-04-01
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.paper_track import build_candidate  # noqa: E402
from src.trading_calendar import is_kr_trading_day  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
UNIVERSE_CSV = ROOT / "data" / "universe.csv"
LIQUID_CSV = ROOT / "data" / "mechanical_universe_final.csv"
KOSPI_CSV = ROOT / "data" / "kospi_index.csv"
RESULTS = ROOT / "results"

HORIZONS = [3, 5, 10]


def load_universe() -> tuple[list[str], set[str], dict]:
    u = pd.read_csv(UNIVERSE_CSV)
    u["ticker"] = u["ticker"].astype(str).str.zfill(6)
    full = u["ticker"].tolist()
    liq = set(pd.read_csv(LIQUID_CSV)["ticker"].astype(str).str.zfill(6).tolist())
    # 시총 3분위 (대형/중형/소형)
    size = {}
    if "market_cap" in u.columns:
        mc = u.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
        n = len(mc)
        for i, (_, r) in enumerate(mc.iterrows()):
            size[r["ticker"]] = "대형" if i < n // 3 else ("중형" if i < 2 * n // 3 else "소형")
    return full, liq, size


def trading_days(start: date, end: date) -> set:
    days, d = set(), start
    while d <= end:
        if is_kr_trading_day(d):
            days.add(pd.Timestamp(d))
        d += timedelta(days=1)
    return days


def load_kospi(path: Path = KOSPI_CSV) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    k = pd.read_csv(path)
    dc = "Date" if "Date" in k.columns else k.columns[0]
    k[dc] = pd.to_datetime(k[dc])
    k = k.sort_values(dc)
    return pd.Series(pd.to_numeric(k["close"], errors="coerce").values, index=k[dc]).dropna()


def fwd_returns(df: pd.DataFrame, sig_pos: int) -> dict | None:
    """신호일 다음 거래일 시가 진입 → D+3/5/10 종가 수익률(%). 0/inf/액면분할 제외."""
    entry_pos = sig_pos + 1
    if entry_pos >= len(df):
        return None
    entry = float(df.iloc[entry_pos]["open"])
    if entry <= 0 or not np.isfinite(entry):
        return None
    window = df.iloc[entry_pos:entry_pos + 11]["close"].astype(float)
    if (window <= 0).any() or (window.pct_change().abs() > 0.4).any():
        return None
    out = {"entry": entry}
    for n in HORIZONS:
        jpos = entry_pos + n
        if jpos < len(df):
            c = float(df.iloc[jpos]["close"])
            out[f"d{n}"] = round((c / entry - 1) * 100, 2) if (c > 0 and np.isfinite(c)) else None
        else:
            out[f"d{n}"] = None
    return out


def kospi_fwd(kospi: pd.Series, sig_date: pd.Timestamp, n: int) -> float | None:
    k = kospi.loc[kospi.index > sig_date]
    if len(k) < n + 1:
        return None
    return round((float(k.iloc[n]) / float(k.iloc[0]) - 1) * 100, 2)


def in_group(g: str, floor: dict, mkt: dict, sup: dict) -> bool:
    label = floor.get("label")
    score = floor.get("floor_quality_score", 0)
    drop = mkt.get("drop_context")
    state = sup.get("supply_state")
    sideways = label == "바닥다지기후보"                                  # 횡보 매집
    not_specific = drop != "stock_specific_drop"                          # 개별급락 회피
    solo = state in ("foreign_accumulation", "institution_accumulation")  # 단독 수급
    if g == "base":
        return True
    if g == "legacy4":
        return ((label in ("진짜바닥후보", "바닥다지기후보") or score >= 2)
                and drop == "resilient_pullback"
                and state in ("dual_buying", "foreign_accumulation", "institution_accumulation"))
    if g == "hypoA":
        return sideways and not_specific
    if g == "hypoB":
        return sideways and solo
    if g == "hypoC":
        return sideways and not_specific and solo
    if g == "supply_solo":
        return solo
    return False


GROUPS = ["base", "legacy4", "hypoA", "hypoB", "hypoC", "supply_solo"]
GROUP_LABEL = {
    "base": "차트4조건 단독",
    "legacy4": "기존 4중결합",
    "hypoA": "가설A 횡보+급락회피",
    "hypoB": "가설B 횡보+단독수급",
    "hypoC": "가설C 횡보+급락회피+단독수급",
    "supply_solo": "수급단독(외인/기관)",
}


def summarize(trades: list[dict], group: str, liquid_only: bool) -> dict:
    rows = [t for t in trades if in_group(group, t["floor"], t["market"], t["supply"])]
    if liquid_only:
        rows = [t for t in rows if t["in_liquid"]]
    if not rows:
        return {"group": group, "entries": 0}
    out = {"group": group, "entries": len(rows)}
    for n in HORIZONS:
        arr = np.array([t[f"d{n}"] for t in rows if t.get(f"d{n}") is not None])
        if len(arr):
            out[f"win{n}"] = round(float((arr > 0).mean() * 100), 1)
            out[f"avg{n}"] = round(float(arr.mean()), 2)
            out[f"med{n}"] = round(float(np.median(arr)), 2)
            out[f"worst{n}"] = round(float(arr.min()), 2)
    exc = [t["d10"] - t["kospi_d10"] for t in rows
           if t.get("d10") is not None and t.get("kospi_d10") is not None]
    out["excess_d10"] = round(float(np.mean(exc)), 2) if exc else None
    # 시총별 진입수
    for tier in ("대형", "중형", "소형"):
        out[f"n_{tier}"] = sum(1 for t in rows if t.get("size") == tier)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="in_sample")
    ap.add_argument("--start", default="2025-12-08")
    ap.add_argument("--end", default="2026-06-05")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--kospi-csv", default=str(KOSPI_CSV),
                    help="KOSPI 소스. 2022 약세장은 data/kospi_index_bak_3_20.csv(2011~) 사용")
    args = ap.parse_args()

    full, liq, size = load_universe()
    if args.sample > 0:
        full = full[:args.sample]
    days = trading_days(date.fromisoformat(args.start), date.fromisoformat(args.end))
    kospi = load_kospi(Path(args.kospi_csv))
    RESULTS.mkdir(exist_ok=True)

    print(f"[{args.tag}] 유니버스 {len(full)} (유동성 {len(liq)}) | 거래일 {len(days)} | {args.start}~{args.end}")
    t0 = time.time()
    trades: list[dict] = []
    done = 0
    for t in full:
        f = PROCESSED / f"{t}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
        except Exception:
            continue
        for ts in [d for d in df.index if d in days]:
            close = float(df.loc[ts, "close"])
            if close <= 0:
                continue
            trade = {"ticker": t, "entry_date": ts.strftime("%Y-%m-%d"), "entry_price": int(close)}
            cand = build_candidate(trade, df)
            if cand.get("decision") != "진입":
                continue
            pos = df.index.get_loc(ts)
            if isinstance(pos, slice):
                continue
            fr = fwd_returns(df, pos)
            if not fr:
                continue
            trades.append({
                "ticker": t, "asof": ts.strftime("%Y-%m-%d"),
                "floor": cand.get("floor_quality", {}),
                "market": cand.get("market_context", {}),
                "supply": cand.get("supply_confirmation", {}),
                "d3": fr["d3"], "d5": fr["d5"], "d10": fr["d10"],
                "kospi_d10": kospi_fwd(kospi, ts, 10),
                "in_liquid": t in liq, "size": size.get(t, "소형"),
            })
        done += 1
        if done % 150 == 0:
            print(f"  {done}/{len(full)} | 진입 {len(trades)} | {time.time()-t0:.0f}s")

    # CSV
    csv_path = RESULTS / f"backtest_floor_quality_{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["ticker", "asof", "floor_label", "floor_score", "drop_context",
                    "supply_state", "d3", "d5", "d10", "kospi_d10", "in_liquid", "size"])
        for t in trades:
            w.writerow([t["ticker"], t["asof"], t["floor"].get("label"),
                        t["floor"].get("floor_quality_score"), t["market"].get("drop_context"),
                        t["supply"].get("supply_state"), t["d3"], t["d5"], t["d10"],
                        t["kospi_d10"], t["in_liquid"], t["size"]])

    # summary
    md = [f"# 백테스트 [{args.tag}] — 4중결합 + 수정가설",
          f"기간 {args.start}~{args.end} | 유니버스 {len(full)}(유동성 {len(liq)}) | 진입(base) {len(trades)}",
          f"규칙 고정 / look-ahead 금지 / 실주문0. 생성 {time.strftime('%Y-%m-%d %H:%M')}\n"]
    for liquid_only, title in [(False, "## 1. 전체상장 (탐색)"), (True, "## 2. 유동성 통과 (실전 성과판정 — 메인)")]:
        md.append(title)
        md.append("| 비교군 | 진입 | 승률D10 | 평균D3 | 평균D5 | 평균D10 | 중앙D10 | 최악D10 | KOSPI대비 | 대/중/소 |")
        md.append("|--------|------|---------|--------|--------|---------|---------|---------|-----------|----------|")
        for g in GROUPS:
            s = summarize(trades, g, liquid_only)
            if s.get("entries", 0) == 0:
                md.append(f"| {GROUP_LABEL[g]} | 0 | - | - | - | - | - | - | - | - |")
                continue
            md.append(f"| {GROUP_LABEL[g]} | {s['entries']} | {s.get('win10','-')}% | {s.get('avg3','-')}% | "
                      f"{s.get('avg5','-')}% | {s.get('avg10','-')}% | {s.get('med10','-')}% | "
                      f"{s.get('worst10','-')}% | {s.get('excess_d10')}%p | "
                      f"{s.get('n_대형',0)}/{s.get('n_중형',0)}/{s.get('n_소형',0)} |")
        md.append("")
    (RESULTS / f"backtest_floor_quality_{args.tag}_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[완료] 진입 {len(trades)} | {time.time()-t0:.0f}s | {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
