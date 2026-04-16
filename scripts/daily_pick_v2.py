#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Pick v2 — 내일 진입 후보 스크리닝 (퐝가님 상시 운영용)

백테스트 (2026-04-02, 874029c) + KISBOT 교차검증 (2026-04-16) 반영.

포지티브 시그널 (백테스트 상위 패턴):
  - 쌍끌이(각50+)     : D+3 평균 +7.83%, WR 100%, PF ∞
  - 개인강주도(+100)  : D+3 평균 +8.57%, WR 92.3%, PF 26.9
  - 외인단독(기매도)  : D+3 평균 +6.75%, WR 92.0%, PF 47

네거티브 필터 (KISBOT 결함 반영):
  - 외인 대량매도 감점 (-300억/-500억)
  - ret60 과열 감점 (+50%/+80%)
  - 기관 분배 Phase 감점 (5일중 -100억 3일↑)
  - 오늘 급락 감점 (-3% 이하)
  - 20MA 극단 이격 감점 (±12%↑)

사용:
  python scripts/daily_pick_v2.py              # 최신 날짜 자동
  python scripts/daily_pick_v2.py --date 2026-04-16
  python scripts/daily_pick_v2.py --min-score 30
"""
from __future__ import annotations
import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UNIVERSE_CSV = ROOT / "data" / "universe.csv"
RAW_DIR = ROOT / "data" / "raw"
SURGE_PATTERN = ROOT / "data" / "analysis_surge_{date}.csv"


def load_universe() -> dict[str, str]:
    if not UNIVERSE_CSV.exists():
        return {}
    uni = pd.read_csv(UNIVERSE_CSV, dtype={"ticker": str})
    uni["ticker"] = uni["ticker"].str.zfill(6)
    return dict(zip(uni["ticker"], uni["name"]))


def scan_one(fp: Path, today: pd.Timestamp, name_map: dict[str, str]) -> dict | None:
    t = fp.stem
    if not t.isdigit() or len(t) != 6:
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None
    if df.empty or today not in df.index:
        return None

    df = df.tail(70)
    today_row = df.loc[today]
    last5 = df.tail(5)
    last20 = df.tail(20)

    # 수급 (억)
    fgn5 = last5["외국인합계"].sum() / 1e8
    inst5 = last5["기관합계"].sum() / 1e8
    ind5 = last5["개인"].sum() / 1e8
    fgn_t = today_row["외국인합계"] / 1e8
    inst_t = today_row["기관합계"] / 1e8
    ind_t = today_row["개인"] / 1e8

    # 유동성 (20일 평균 거래대금, volume*close 추정)
    avg20_tv = (last20["volume"] * last20["close"]).mean() / 1e8

    # 수익률
    def _ret(days: int) -> float:
        if len(df) <= days:
            return 0.0
        return (today_row["close"] / df.iloc[-(days + 1)]["close"] - 1) * 100

    ret1 = _ret(1)
    ret5 = _ret(5)
    ret20 = _ret(20)
    ret60 = _ret(60)

    # 20MA 이격
    ma20 = last20["close"].mean()
    gap20 = (today_row["close"] / ma20 - 1) * 100

    # 분배 Phase 감지
    inst_heavy_sell_days = int(((last5["기관합계"] / 1e8) <= -100).sum())
    fgn_heavy_sell_days = int(((last5["외국인합계"] / 1e8) <= -100).sum())

    return {
        "ticker": t,
        "name": name_map.get(t, t),
        "close": float(today_row["close"]),
        "ret1": round(ret1, 2),
        "ret5": round(ret5, 2),
        "ret20": round(ret20, 2),
        "ret60": round(ret60, 2),
        "gap20": round(gap20, 2),
        "fgn5": round(fgn5, 1),
        "inst5": round(inst5, 1),
        "ind5": round(ind5, 1),
        "fgn_t": round(fgn_t, 1),
        "inst_t": round(inst_t, 1),
        "ind_t": round(ind_t, 1),
        "avg20_tv": round(avg20_tv, 1),
        "inst_heavy_sell_days": inst_heavy_sell_days,
        "fgn_heavy_sell_days": fgn_heavy_sell_days,
    }


def score_row(r: dict) -> tuple[int, str, str]:
    f, i, p = r["fgn5"], r["inst5"], r["ind5"]
    ft, it, pt = r["fgn_t"], r["inst_t"], r["ind_t"]
    s, tags, warns = 0, [], []

    # === POSITIVE ===
    if f >= 50 and i >= 50:
        s += 28
        tags.append("쌍끌이50+")
    elif f >= 30 and i >= 30:
        s += 12
        tags.append("쌍끌이30+")
    if p >= 100 and f > -200:
        s += 25
        tags.append("개인강주도+")
    elif p >= 50 and f > -100:
        s += 12
        tags.append("개인주도+")
    if f >= 100 and i <= -30:
        s += 22
        tags.append("외인강단독")
    elif f >= 50 and i <= -30:
        s += 18
        tags.append("외인단독")
    if ft >= 30 and it >= 30:
        s += 10
        tags.append("T쌍끌이")
    if pt >= 50 and ft > 0:
        s += 8
        tags.append("T개인건전추격")
    if -5 <= r["gap20"] <= 3 and (f >= 30 or i >= 30 or p >= 50):
        s += 8
        tags.append("20MA눌림수급")

    # === NEGATIVE (KISBOT 결함 반영) ===
    if f <= -500:
        s -= 30
        warns.append("외인초대량매도")
    elif f <= -300:
        s -= 20
        warns.append("외인대량매도")
    elif f <= -150:
        s -= 10
        warns.append("외인매도")
    if r["ret60"] >= 80:
        s -= 25
        warns.append("60일과열80+")
    elif r["ret60"] >= 50:
        s -= 15
        warns.append("60일과열50+")
    if r["inst_heavy_sell_days"] >= 3:
        s -= 18
        warns.append("기관분배3일+")
    elif r["inst_heavy_sell_days"] == 2:
        s -= 8
        warns.append("기관분배2일")
    if r["fgn_heavy_sell_days"] >= 3:
        s -= 15
        warns.append("외인연속매도")
    if r["ret1"] <= -3:
        s -= 15
        warns.append("오늘급락")
    if r["gap20"] >= 12:
        s -= 10
        warns.append("20MA과열")
    elif r["gap20"] <= -12:
        s -= 8
        warns.append("20MA이상급락")

    return s, ",".join(tags) if tags else "-", ",".join(warns) if warns else "-"


def generate_report(df: pd.DataFrame, today: pd.Timestamp, out_path: Path) -> None:
    """MD 형식 보고서 생성 (퐝가님 포맷)"""
    next_day = (today + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# 내일({next_day}) 최종 진입 리스트 (v2)")
    lines.append(f"기준일: {today.strftime('%Y-%m-%d')} | 총 {len(df)}개 후보 (score 20+)")
    lines.append("")

    # 🥇 무경고 + score 30+
    tier1 = df[(df["warns"] == "-") & (df["score"] >= 30)].head(15)
    lines.append(f"## 🥇 무경고 + 최우수 (score 30+, warns 전혀 없음) — 안전 핵심 [{len(tier1)}개]")
    lines.append("")
    if len(tier1):
        lines.append("| 순위 | 종목 | 종가 | score | tags | ret60 | gap20 | 5일수급(외/기/개) |")
        lines.append("|------|------|------|-------|------|-------|-------|------------------|")
        for i, (_, r) in enumerate(tier1.iterrows(), 1):
            lines.append(
                f"| {i} | **{r['name']}** ({r['ticker']}) | {int(r['close']):,} | "
                f"**{r['score']}** | {r['tags']} | {r['ret60']:+.1f}% | {r['gap20']:+.1f}% | "
                f"{r['fgn5']:+.0f}/{r['inst5']:+.0f}/{r['ind5']:+.0f} |"
            )
        lines.append("")

    # 🥈 1경고 (주의)
    tier2 = df[(df["warns"] != "-") & (df["score"] >= 25)].head(10)
    lines.append(f"## 🥈 경고 있음 — 주의 필요 [{len(tier2)}개]")
    lines.append("")
    if len(tier2):
        lines.append("| 종목 | score | tags | **warns** | ret60 | gap20 |")
        lines.append("|------|-------|------|----------|-------|-------|")
        for _, r in tier2.iterrows():
            lines.append(
                f"| **{r['name']}** ({r['ticker']}) | {r['score']} | {r['tags']} | "
                f"⚠️ {r['warns']} | {r['ret60']:+.1f}% | {r['gap20']:+.1f}% |"
            )
        lines.append("")

    # 저득점 제외 안내
    lines.append("## 📋 추천 포트폴리오 (5~7종목 분산 제안)")
    lines.append("")
    if len(tier1) >= 5:
        top5 = tier1.head(7)
        for _, r in top5.iterrows():
            lines.append(f"- **{r['name']}** ({r['ticker']}): {r['tags']} / 종가 {int(r['close']):,}원")
    lines.append("")
    lines.append("## ⚠️ 실전 진입 원칙")
    lines.append("- 9시 시가 + 10시 수급 재확인 후 진입")
    lines.append("- Shield 레짐별 MAX_POSITIONS 준수 (RED=3, YELLOW=5, GREEN=8)")
    lines.append("- 손절 -7% 기계적 엄수")
    lines.append("")
    lines.append(f"---\n_Generated by daily_pick_v2.py @ {datetime.now().isoformat()}_")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Pick v2 (퐝가님 상시 운영)")
    parser.add_argument("--date", help="기준일 YYYY-MM-DD (기본: 최신 parquet 날짜)")
    parser.add_argument("--min-score", type=int, default=20, help="최소 스코어 (기본 20)")
    parser.add_argument("--tv-min", type=float, default=100.0, help="20일 평균 거래대금 최소(억)")
    parser.add_argument("--tv-max", type=float, default=15000.0, help="20일 평균 거래대금 최대(억)")
    parser.add_argument("--exclude-surge", action="store_true", default=True, help="당일 급등주 제외")
    args = parser.parse_args()

    # 기준일 자동 결정
    if args.date:
        today = pd.Timestamp(args.date)
    else:
        # 가장 최근 parquet의 마지막 날짜
        sample = pd.read_parquet(RAW_DIR / "005930.parquet")
        today = sample.index[-1]
    print(f"[daily_pick_v2] 기준일: {today.strftime('%Y-%m-%d')}")

    name_map = load_universe()
    print(f"[daily_pick_v2] universe: {len(name_map)}개")

    # 스캔
    rows: list[dict] = []
    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    for fp in files:
        r = scan_one(Path(fp), today, name_map)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"[daily_pick_v2] 스캔: {len(df)}개")

    # 1~3차 필터
    df = df[(df["avg20_tv"] >= args.tv_min) & (df["avg20_tv"] <= args.tv_max)].copy()
    df = df[df["ret5"].between(-15, 15)].copy()
    df = df[df["gap20"].between(-15, 15)].copy()
    df = df[df["ret1"] > -5].copy()
    print(f"[daily_pick_v2] 필터 후: {len(df)}개")

    # 스코어링
    df[["score", "tags", "warns"]] = df.apply(lambda r: pd.Series(score_row(r.to_dict())), axis=1)
    df = df[df["score"] >= args.min_score].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # 당일 급등주 제외
    if args.exclude_surge:
        surge_path = Path(str(SURGE_PATTERN).format(date=today.strftime("%Y%m%d")))
        if surge_path.exists():
            surge = pd.read_csv(surge_path, dtype={"ticker": str})
            surge["ticker"] = surge["ticker"].str.zfill(6)
            before = len(df)
            df = df[~df["ticker"].isin(surge["ticker"])].copy()
            print(f"[daily_pick_v2] 당일 급등주 {before - len(df)}개 제외")

    print(f"[daily_pick_v2] 최종: {len(df)}개 (score 20+)")

    # 저장
    date_str = today.strftime("%Y%m%d")
    csv_path = ROOT / "data" / f"picks_v2_{date_str}.csv"
    md_path = ROOT / "data" / f"picks_v2_{date_str}_report.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    generate_report(df, today, md_path)
    print(f"[daily_pick_v2] 저장: {csv_path.name}, {md_path.name}")

    # 콘솔 요약 TOP 10
    tier1 = df[(df["warns"] == "-") & (df["score"] >= 30)].head(10)
    print()
    print("=" * 80)
    print(f"🥇 내일 진입 TOP {len(tier1)} (무경고)")
    print("=" * 80)
    for i, (_, r) in enumerate(tier1.iterrows(), 1):
        print(
            f"{i:2}. [{r['ticker']}] {r['name']:<12} score={r['score']:>3}  "
            f"tags={r['tags']:<35}  ret60={r['ret60']:+6.1f}%  tv={r['avg20_tv']:,.0f}억"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
