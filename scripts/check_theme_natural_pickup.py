"""
한타바이러스/에볼라 테마 11종목 자연 픽업 로깅
====================================================
목적
----
퐝가님(5/20) 인사이트("한타/에볼라 종목들이 오른다") 검증.
차트영웅 룰(picks_v2 산출)이 이 종목들을 자연 픽업하는지 매일 기록.

운영
----
- 매일 18:00 cron (picks_v2_{date}.csv 17:45 산출 후)
- jgis OHLCV(VPS /home/ubuntu/jgis/stock_data_daily/) 일봉 데이터 사용
- picks_v2_{date}.csv 자연 픽업 여부 + 점수 + 태그 기록
- 출력: data/theme_pickup_log.csv (append, 같은 date+ticker 덮어쓰기)

회고
----
5/27(화) — 1주차 회고:
- picks_v2 자연 픽업 0건 + 종목들 +5%↑ → Phase 2 테마 가중치 정식 통합 근거
- picks_v2 자연 픽업 OK + 매수 결과 좋음 → 룰 신뢰 확정
- picks_v2 자연 픽업 OK + 매수 결과 나쁨 → 룰 자체 재점검
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
# 테마 종목 풀 (theme_dictionary.yaml 한타바이러스/에볼라 통합)
# ─────────────────────────────────────────────
THEME_STOCKS = [
    ("006280", "녹십자", "한타바이러스", 1),
    ("005250", "녹십자홀딩스", "한타바이러스", 1),
    ("007570", "일양약품", "한타+에볼라", 1),
    ("142280", "녹십자엠에스", "한타바이러스", 2),
    ("950130", "엑세스바이오", "한타+에볼라", 2),
    ("253840", "수젠텍", "한타+에볼라", 2),
    ("084650", "랩지노믹스", "한타+에볼라", 2),
    ("011000", "진원생명과학", "한타+에볼라", 2),
    ("009290", "광동제약", "한타바이러스", 3),
    ("041960", "코미팜", "한타바이러스", 3),
    ("068270", "셀트리온", "에볼라", 1),
]

# ─────────────────────────────────────────────
# 경로 (VPS 기준; 로컬 dry-run은 환경변수로 override)
# ─────────────────────────────────────────────
JGIS_BASE = Path(os.environ.get("JGIS_BASE", "/home/ubuntu/jgis/stock_data_daily"))
QUANTUM_DATA = Path(os.environ.get("QUANTUM_DATA", "/home/ubuntu/quantum-master/data"))
LOG_FILE = QUANTUM_DATA / "theme_pickup_log.csv"


def find_jgis_csv(ticker: str) -> Path | None:
    for fp in JGIS_BASE.glob(f"*_{ticker}.csv"):
        return fp
    return None


def load_picks_v2(date_str: str) -> pd.DataFrame | None:
    fp = QUANTUM_DATA / f"picks_v2_{date_str}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df


def analyze_one(ticker: str, name: str, theme: str, order: int, picks: pd.DataFrame | None) -> dict:
    row = {
        "ticker": ticker, "name": name, "theme": theme, "order": order,
        "jgis_date": "", "close": 0, "prev_close": 0, "chg_pct": 0.0,
        "vol_ratio": 0.0, "foreign_net_5d": 0, "inst_net_5d": 0,
        "supply_pass": False, "picked_in_picks_v2": False, "score": "", "tags": "",
        "note": "",
    }

    fp = find_jgis_csv(ticker)
    if not fp:
        row["note"] = "JGIS_MISSING"
        return row

    try:
        df = pd.read_csv(fp)
    except Exception as e:
        row["note"] = f"READ_ERR:{str(e)[:30]}"
        return row

    if len(df) < 6:
        row["note"] = f"JGIS_TOO_FEW:{len(df)}"
        return row

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last.get("Close", 0))
    prev_close = float(prev.get("Close", 0))
    chg_pct = (close / prev_close - 1) * 100 if prev_close > 0 else 0.0
    vol_r = float(last.get("Vol_Ratio", 0))
    fn5 = int(df["Foreign_Net"].tail(5).sum()) if "Foreign_Net" in df.columns else 0
    in5 = int(df["Inst_Net"].tail(5).sum()) if "Inst_Net" in df.columns else 0

    row.update({
        "jgis_date": str(last.get("Date", ""))[:10],
        "close": close, "prev_close": prev_close,
        "chg_pct": round(chg_pct, 2), "vol_ratio": round(vol_r, 2),
        "foreign_net_5d": fn5, "inst_net_5d": in5,
        "supply_pass": (fn5 + in5) >= 0,
    })

    if picks is not None:
        match = picks[picks["ticker"] == ticker]
        if len(match) > 0:
            m = match.iloc[0]
            row["picked_in_picks_v2"] = True
            row["score"] = m.get("score", "")
            row["tags"] = str(m.get("tags", ""))[:100]

    return row


def merge_log(date_str: str, new_df: pd.DataFrame) -> pd.DataFrame:
    new_df = new_df.copy()
    new_df.insert(0, "date", date_str)
    new_df["ticker"] = new_df["ticker"].astype(str)

    if not LOG_FILE.exists():
        return new_df

    existing = pd.read_csv(LOG_FILE, dtype={"date": str, "ticker": str})
    mask = (existing["date"] == date_str) & (existing["ticker"].isin(new_df["ticker"]))
    existing = existing[~mask]
    return pd.concat([existing, new_df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="콘솔 출력만, 파일 미저장")
    args = parser.parse_args()

    date_str = args.date or datetime.date.today().strftime("%Y%m%d")
    print(f"[theme_pickup] date={date_str} jgis={JGIS_BASE} data={QUANTUM_DATA}")

    picks = load_picks_v2(date_str)
    if picks is None:
        print(f"  ⚠️  picks_v2_{date_str}.csv 없음 (아직 미산출 또는 휴장) — 종목 일봉만 기록")

    rows = [analyze_one(t, n, th, o, picks) for t, n, th, o in THEME_STOCKS]
    new_df = pd.DataFrame(rows)

    # 요약 출력
    print()
    print(f"{'종목':<14}{'테마':<14}{'O':<3}{'jgis일자':<12}{'종가':>10}{'D-1%':>8}{'Vol배':>7}{'외+기5합':>14}{'수급':>6}{'픽업':>6}{'점수':>6}")
    print("-" * 110)
    for r in rows:
        if r.get("note"):
            print(f"{r['name']:<14}{r['theme']:<14}{r['order']:<3}{r['note']}")
            continue
        sup = "✅" if r["supply_pass"] else "❌"
        pic = "✅" if r["picked_in_picks_v2"] else "·"
        sf = r["foreign_net_5d"] + r["inst_net_5d"]
        print(f"{r['name']:<14}{r['theme']:<14}{r['order']:<3}{r['jgis_date']:<12}{r['close']:>10,.0f}{r['chg_pct']:>+7.1f}%{r['vol_ratio']:>7.2f}{sf:>+14,}{sup:>5}{pic:>5}{str(r['score']):>6}")

    picked = sum(1 for r in rows if r.get("picked_in_picks_v2"))
    passed = sum(1 for r in rows if r.get("supply_pass"))
    print()
    print(f"[요약] picks_v2 자연 픽업 {picked}/11 | 외+기 5일 합 통과 {passed}/11")

    if args.dry_run:
        print("[dry-run] 파일 미저장")
        return

    merged = merge_log(date_str, new_df)
    QUANTUM_DATA.mkdir(parents=True, exist_ok=True)
    merged.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    print(f"[저장] {LOG_FILE} (누적 {len(merged)}행)")


if __name__ == "__main__":
    main()
