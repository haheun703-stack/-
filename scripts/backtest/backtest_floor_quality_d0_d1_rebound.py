"""하락→반등 구간 가설C(floor_quality) D0_CLOSE vs D1_OPEN 백테스트 (6/7 사장님 지시).

★read-only 분석. 운영코드 무변경(FLOWX/SmartEntry/C60/scheduler/SAJANG/PAPER_OPEN 0).
재사용(호출만): backtest_floor_quality_6m(load_universe/trading_days/in_group),
paper_track.build_candidate(가설C 후보 과거재현, look-ahead 안전), paper_smart_entry.classify_tier(SSOT),
price_axis_regime.build_price_axis_labels(라벨), show_me_report._d0d1_verdict(4분면). 실주문 0.

목적: 6/8~6/12 forward는 표본이 얇다. 과거 하락→반등 replay로 "가설C가 D1_OPEN 기준으로도
돈이 되는가 + 17 검증항목을 버티는가"를 본다. 과거 replay와 6/8~12 forward가 같은 방향이면 실전 근거.

하락→반등 정의(★종목별 — KOSPI 가공본 회피): 직전 lookback(20) 거래일 고점 대비 -drop%(5) 이상
하락 후, 신호일 종가가 5일선 위(단기 반등). 그 신호일에 가설C(hypoC) 후보면 트레이드.

진입 2기준: D0_CLOSE(신호일 종가) / D1_OPEN(다음 거래일 시초가, 체결 현실 근접). 둘 다 D+3/5/10 종가
수익률 + MFE/MAE. look-ahead 안전(라벨은 신호일까지 슬라이스, 0행/액면분할 가드).

사용:
  python -u -X utf8 scripts/backtest/backtest_floor_quality_d0_d1_rebound.py --tag rebound_long --start 2025-06-09 --end 2026-06-05
  python -u -X utf8 ... --tag mar2026 --start 2026-02-15 --end 2026-04-30
  python -u -X utf8 ... --tag may_jun2026 --start 2026-05-01 --end 2026-06-05
  python -u -X utf8 ... --tag bear2022 --start 2022-01-01 --end 2022-12-31
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest.backtest_floor_quality_6m import (  # noqa: E402  재사용(호출만)
    in_group,
    load_universe,
    trading_days,
)
from scripts.paper_smart_entry import classify_tier  # noqa: E402  SSOT
from scripts.paper_track import build_candidate  # noqa: E402  가설C 후보 재현
from src.use_cases.price_axis_regime import build_price_axis_labels  # noqa: E402
from src.use_cases.show_me_report import _d0d1_verdict  # noqa: E402  4분면

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
HORIZONS = [3, 5, 10]
JUDGE_H = "d10"  # 4분면 판정 기준


def is_rebound_signal(df: pd.DataFrame, sig_pos: int, lookback: int = 20, drop_pct: float = 5.0) -> bool:
    """직전 lookback 고점 대비 -drop%↓ 하락 후 신호일 종가가 5일선 위(단기 반등). 과거만 본다."""
    if sig_pos < lookback or sig_pos < 5:
        return False
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    prior_high = float(highs.iloc[sig_pos - lookback:sig_pos].max())
    cur = float(closes.iloc[sig_pos])
    if prior_high <= 0 or cur <= 0:
        return False
    drawdown = (cur / prior_high - 1) * 100
    if drawdown > -drop_pct:  # 충분히 안 빠졌으면 '하락 후'가 아님
        return False
    ma5 = float(closes.iloc[sig_pos - 4:sig_pos + 1].mean())
    return cur >= ma5  # 단기 반등(5일선 회복)


def _fwd(closes: pd.Series, entry: float, base_pos: int) -> dict:
    out = {}
    for n in HORIZONS:
        jpos = base_pos + n
        if jpos < len(closes):
            c = float(closes.iloc[jpos])
            out[f"d{n}"] = round((c / entry - 1) * 100, 2) if (c > 0 and np.isfinite(c)) else None
        else:
            out[f"d{n}"] = None
    return out


def _mfe_mae(df: pd.DataFrame, entry: float, start_pos: int, end_pos: int) -> tuple:
    seg = df.iloc[start_pos:end_pos + 1]
    if seg.empty or entry <= 0:
        return None, None
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    return round((hi / entry - 1) * 100, 2), round((lo / entry - 1) * 100, 2)


def d0_d1_metrics(df: pd.DataFrame, sig_pos: int) -> dict | None:
    """D0_CLOSE / D1_OPEN 진입 성과 + 체결품질. look-ahead 안전(0행/액면분할 가드)."""
    n_bars = len(df)
    if sig_pos + 1 >= n_bars:
        return None
    closes = df["close"].astype(float)
    d0_close = float(closes.iloc[sig_pos])
    d1_open = float(df.iloc[sig_pos + 1]["open"])
    if d0_close <= 0 or d1_open <= 0 or not np.isfinite(d1_open):
        return None
    win = closes.iloc[sig_pos:sig_pos + 12]
    if (win <= 0).any() or (win.pct_change().abs() > 0.4).any():
        return None  # 액면분할/0행 구간 제외
    d0 = _fwd(closes, d0_close, sig_pos)
    d1 = _fwd(closes, d1_open, sig_pos + 1)
    d0_mfe, d0_mae = _mfe_mae(df, d0_close, sig_pos, min(sig_pos + 10, n_bars - 1))
    d1_mfe, d1_mae = _mfe_mae(df, d1_open, sig_pos + 1, min(sig_pos + 11, n_bars - 1))
    # 체결품질: D1 갭상승률 / D1 당일 open-low 낙폭 / D1 종가 vs 시가
    d1_bar = df.iloc[sig_pos + 1]
    d1_low = float(d1_bar["low"])
    d1_close = float(d1_bar["close"])
    return {
        "d0_close": int(d0_close), "d1_open": int(d1_open),
        "d0": d0, "d1": d1,
        "d0_mfe": d0_mfe, "d0_mae": d0_mae, "d1_mfe": d1_mfe, "d1_mae": d1_mae,
        "gap_pct": round((d1_open / d0_close - 1) * 100, 2),
        "d1_open_low": round((d1_low / d1_open - 1) * 100, 2),
        "d1_close_vs_open": round((d1_close / d1_open - 1) * 100, 2),
    }


def _quad_label(v: str) -> str:
    for k in ("BOTH_GOOD", "D0GOOD_D1BAD", "BOTH_BAD", "D0BAD_D1GOOD"):
        if v.startswith(k):
            return k
    return "PENDING"


def _stats(vals: list) -> dict:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if not len(arr):
        return {}
    return {
        "n": int(len(arr)),
        "avg": round(float(arr.mean()), 2),
        "med": round(float(np.median(arr)), 2),
        "win": round(float((arr > 0).mean() * 100), 1),
        "worst": round(float(arr.min()), 2),
    }


def _profit_factor(vals: list) -> float | None:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    gains = float(arr[arr > 0].sum())
    losses = float(-arr[arr < 0].sum())
    return round(gains / losses, 2) if losses > 0 else None


def _common_labels(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "weekly_open": dict(Counter(r.get("weekly_open_state") for r in rows)),
        "half_year_open": dict(Counter(r.get("half_year_open_state") for r in rows)),
        "overheat": dict(Counter(r.get("overheat") for r in rows)),
        "tier": dict(Counter(r.get("tier") for r in rows)),
        "sector_top": dict(Counter(r.get("sector") for r in rows).most_common(3)),
        "avg_gap_pct": round(float(np.mean([r["gap_pct"] for r in rows])), 2),
        "avg_d1_open_low": round(float(np.mean([r["d1_open_low"] for r in rows])), 2),
    }


def report(trades: list[dict], tag: str) -> dict:
    print("=" * 72)
    print(f"하락→반등 가설C D0/D1 백테스트 — {tag}")
    print("=" * 72)
    n = len(trades)
    print(f"총 후보(가설C·하락후반등): {n}")
    if not n:
        print("후보 0 — 구간/조건(drop_pct·lookback) 재검토 필요")
        return {"tag": tag, "n": 0, "verdict": "NO_SAMPLE"}

    d0_d10 = [t["d0"].get("d10") for t in trades]
    d1_d10 = [t["d1"].get("d10") for t in trades]
    s0, s1 = _stats(d0_d10), _stats(d1_d10)
    print(f"\n[1] D0_CLOSE D+10: {s0}")
    print(f"[2] D1_OPEN  D+10: {s1}")
    print(f"[3] 손익비 D0/D1: {_profit_factor(d0_d10)} / {_profit_factor(d1_d10)}")

    # top1 제외(한 종목 착시 점검)
    d1_sorted = sorted([v for v in d1_d10 if v is not None], reverse=True)
    ex_top1 = d1_sorted[1:] if len(d1_sorted) > 1 else []
    ex_top1_avg = round(float(np.mean(ex_top1)), 2) if ex_top1 else None
    ex_top1_med = round(float(np.median(ex_top1)), 2) if ex_top1 else None
    print(f"[4] D1 top1 제외 평균/median: {ex_top1_avg} / {ex_top1_med}  (median이 핵심)")

    # 4분면
    quad = Counter(_quad_label(t["verdict"]) for t in trades)
    print(f"[5] 4분면: BOTH_GOOD {quad.get('BOTH_GOOD',0)} / D0GOOD_D1BAD {quad.get('D0GOOD_D1BAD',0)} "
          f"/ BOTH_BAD {quad.get('BOTH_BAD',0)} / D0BAD_D1GOOD {quad.get('D0BAD_D1GOOD',0)} / PENDING {quad.get('PENDING',0)}")

    # 체결품질
    gaps = [t["gap_pct"] for t in trades]
    gap_over7 = sum(1 for g in gaps if g is not None and g >= 7)
    print(f"[6] 체결품질: 평균 갭 {round(float(np.mean(gaps)),2)}% / 갭+7%↑(진입난이도) {gap_over7}건 "
          f"/ 평균 D1 open-low {round(float(np.mean([t['d1_open_low'] for t in trades])),2)}%")

    # 섹터 집중도
    sec = Counter(t["sector"] for t in trades)
    top_sec, top_n = sec.most_common(1)[0]
    print(f"[7] 섹터 집중도: {len(sec)}개 섹터 / 최다 {top_sec} {top_n}건({round(top_n/n*100)}%) "
          f"| 시총 {dict(Counter(t['size'] for t in trades))}")

    # tier별
    print("[8] tier별 D1_OPEN D+10:")
    tier_stats = {}
    for tier in ("CORE", "WATCH", "CONTROL"):
        ts = _stats([t["d1"].get("d10") for t in trades if t["tier"] == tier])
        tier_stats[tier] = ts
        print(f"    {tier}: {ts}")
    core_med = tier_stats.get("CORE", {}).get("med")
    ctrl_med = tier_stats.get("CONTROL", {}).get("med")
    core_gt_ctrl = (core_med is not None and ctrl_med is not None and core_med > ctrl_med)
    print(f"    → CORE>CONTROL(median): {core_gt_ctrl}")

    # 성공/실패 공통 라벨
    good = [t for t in trades if _quad_label(t["verdict"]) == "BOTH_GOOD"]
    bad = [t for t in trades if _quad_label(t["verdict"]) in ("BOTH_BAD", "D0GOOD_D1BAD")]
    print(f"[9] 성공(BOTH_GOOD {len(good)}) 공통: {_common_labels(good)}")
    print(f"[10] 실패(BOTH_BAD+D0GOOD_D1BAD {len(bad)}) 공통: {_common_labels(bad)}")

    # ── LIVE_READY_CANDIDATE 게이트 — ★CORE(가설C 주력) 기준 ──
    core_rows = [t for t in trades if t["tier"] == "CORE"]
    core_d1_vals = [t["d1"].get("d10") for t in core_rows]
    core_d1 = _stats(core_d1_vals)
    core_sorted = sorted([v for v in core_d1_vals if v is not None], reverse=True)
    core_ex_med = round(float(np.median(core_sorted[1:])), 2) if len(core_sorted) > 1 else None
    core_bothgood = sum(1 for t in core_rows if _quad_label(t["verdict"]) == "BOTH_GOOD")
    gate = {
        "①CORE D1 median+": bool(core_d1.get("med") is not None and core_d1["med"] > 0),
        "②CORE D1 승률55%↑": bool(core_d1.get("win") is not None and core_d1["win"] >= 55),
        "③CORE top1제외+": bool(core_ex_med is not None and core_ex_med > 0),
        "④CORE>CONTROL": core_gt_ctrl,
        "⑤CORE BOTH_GOOD 2건↑": core_bothgood >= 2,
    }
    passed = sum(gate.values())
    print(f"\n[게이트] {gate} → {passed}/5")
    if all(gate.values()):
        verdict = "LIVE_READY_CANDIDATE 가능 (단 ledger 무결성·positions/KIS reconcile은 운영점검 별건)"
    elif passed >= 3:
        verdict = "ESCALATE/TUNE — 일부만 통과(연장·파라미터 조정)"
    elif s1.get("med") is not None and s1["med"] > 0:
        verdict = "KEEP — D1 median 양이나 게이트 부족"
    else:
        verdict = "DROP — D1_OPEN 기준 재현성 부족"
    print(f"[결론] {verdict}")
    print("[안전] 실주문 0 / 운영코드 변경 0 / read-only 백테스트")

    return {
        "tag": tag, "n": n, "d0_d10": s0, "d1_d10": s1, "quad": dict(quad),
        "ex_top1_med": ex_top1_med, "tier": tier_stats, "core_gt_ctrl": core_gt_ctrl,
        "gate": gate, "gate_passed": passed, "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="하락→반등 가설C D0/D1 백테스트 (read-only)")
    ap.add_argument("--tag", default="rebound_long")
    ap.add_argument("--start", default="2025-06-09")
    ap.add_argument("--end", default="2026-06-05")
    ap.add_argument("--drop-pct", type=float, default=5.0)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0, help="유니버스 앞 N개만(스모크)")
    args = ap.parse_args()

    full, liq, size = load_universe()
    if args.sample > 0:
        full = full[:args.sample]
    days = trading_days(date.fromisoformat(args.start), date.fromisoformat(args.end))
    try:
        from src.use_cases.morning_plan_07 import load_sector_map
        sector_map = load_sector_map() or {}
    except Exception:
        sector_map = {}

    RESULTS.mkdir(exist_ok=True)
    print(f"[{args.tag}] 유니버스 {len(full)} | 거래일 {len(days)} | {args.start}~{args.end} "
          f"| 하락-{args.drop_pct}%(직전{args.lookback}일고점)후 5일선반등 | 가설C(hypoC)만")

    trades: list[dict] = []
    t0 = time.time()
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
            pos = df.index.get_loc(ts)
            if isinstance(pos, slice):
                continue
            if not is_rebound_signal(df, pos, args.lookback, args.drop_pct):
                continue
            close = float(df.loc[ts, "close"])
            if close <= 0:
                continue
            cand = build_candidate(
                {"ticker": t, "entry_date": ts.strftime("%Y-%m-%d"), "entry_price": int(close)}, df
            )
            if cand.get("decision") != "진입":
                continue
            # ★in_group(hypoC) 필터 제거: hypoC 조건=CORE 정의라 전부 CORE가 됨.
            #   진입 후보 전체를 classify_tier로 CORE/WATCH/CONTROL 분류해야 CORE vs CONTROL 비교 가능.
            m = d0_d1_metrics(df, pos)
            if not m:
                continue
            try:
                labels = build_price_axis_labels(df.iloc[:pos + 1])  # point-in-time
            except Exception:
                labels = {}
            pa = (labels.get("price_axis") or {}) if isinstance(labels, dict) else {}
            ao = (labels.get("annual_overheat") or {}) if isinstance(labels, dict) else {}
            trades.append({
                "ticker": t, "asof": ts.strftime("%Y-%m-%d"),
                "tier": classify_tier(cand), "sector": sector_map.get(t, "기타"),
                "size": size.get(t, "소형"), **m,
                "weekly_open_state": pa.get("weekly_open_state"),
                "half_year_open_state": pa.get("half_year_open_state"),
                "overheat": ao.get("overheat_grade"),
                "verdict": _d0d1_verdict(m["d0"].get(JUDGE_H), m["d1"].get(JUDGE_H)),
            })
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(full)} | 트레이드 {len(trades)} | {time.time()-t0:.0f}s")

    summary = report(trades, args.tag)

    # CSV 저장(트레이드 + 요약)
    csv_path = RESULTS / f"backtest_rebound_d0d1_{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["ticker", "asof", "tier", "sector", "size", "d0_close", "d1_open",
                    "d0_d10", "d1_d10", "d0_mae", "d1_mae", "gap_pct", "d1_open_low",
                    "weekly_open_state", "half_year_open_state", "overheat", "verdict"])
        for t in trades:
            w.writerow([t["ticker"], t["asof"], t["tier"], t["sector"], t["size"],
                        t["d0_close"], t["d1_open"], t["d0"].get("d10"), t["d1"].get("d10"),
                        t["d0_mae"], t["d1_mae"], t["gap_pct"], t["d1_open_low"],
                        t["weekly_open_state"], t["half_year_open_state"], t["overheat"], t["verdict"]])
    print(f"\n저장: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
