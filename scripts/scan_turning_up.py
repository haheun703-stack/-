"""전환 감지기 — "지금 고개 드는 종목" 스캔 (사장님 5/31 "너도 나처럼 신호 잡아야지").

30일 평균 selector가 놓친 '최근 며칠 전환'을 포착. 사장님이 5/28 LG엔솔 +15.3% 전환을
눈으로 본 것을, 데이터로 잡는다. MLCC 6종목 폭등 초입 패턴 = 바닥 + 기관매수 전환 + 첫 급등.

전환 신호(최근 기준):
  ① 최근 5일 급등 (하루라도 +7%↑ = 첫 캔들)
  ② 기관 매수 전환 (최근 5일 순매수 + 직전 5일 대비 가속)
  ③ 외인 매도 흡수 개선 (최근 10일 외인매도일 평균등락 > 0 = 수익화)
  ④ 바닥권 (직전 20일 약세였다 = 아직 폭등 전 = 전야)
  ⑤ 거래대금 충분 (≥100억, 전환엔 자금 필요)
퀀트봇=연구자. 실매수 HOLD. 신호 포착/기록만.
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
NAME_DIR = PROJECT_ROOT / "stock_data_daily"
OUT_FILE = PROJECT_ROOT / "data" / "turning_up_candidates.json"
KST = timezone(timedelta(hours=9))
EOK = 1e8


def _name_map() -> dict:
    m = {}
    for p in glob.glob(str(NAME_DIR / "*.csv")):
        stem = Path(p).stem
        if "_" in stem:
            name, code = stem.rsplit("_", 1)
            m[code.zfill(6)] = name
    return m


def main() -> int:
    names = _name_map()
    rows = []
    latest = ""
    for f in glob.glob(str(PROCESSED / "*.parquet")):
        code = Path(f).stem.zfill(6)
        try:
            df = pd.read_parquet(f).sort_index()
            df = df[df["close"] > 0]
        except Exception:
            continue
        if len(df) < 25:
            continue
        if not all(c in df.columns for c in ["trading_value", "외국인합계", "기관합계"]):
            continue
        latest = max(latest, str(df.index[-1].date()))
        ret = df["close"].pct_change() * 100
        d5, d10 = df.tail(5), df.tail(10)
        tv = float(df["trading_value"].iloc[-1]) / EOK
        if tv < 100:  # 거래대금 100억 미만 제외
            continue
        maxret5 = float(ret.tail(5).max())
        inst5 = float(d5["기관합계"].sum()) / EOK
        inst_prev5 = float(df["기관합계"].iloc[-10:-5].sum()) / EOK
        foreign5 = float(d5["외국인합계"].sum()) / EOK
        sret = ret.where(df["외국인합계"] < 0)
        absorb10 = float(sret.tail(10).mean()) if sret.tail(10).notna().any() else 0.0
        pos20 = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100  # 20일 수익(바닥권?)
        close = float(df["close"].iloc[-1])

        # 전환 점수 (사장님 눈 = 고개 드는 신호)
        score = 0
        if maxret5 >= 7: score += 3          # 첫 급등 캔들
        elif maxret5 >= 4: score += 1
        if inst5 > 0: score += 1             # 기관 매수
        if inst5 > inst_prev5 and inst5 > 0: score += 2  # 기관 매수 가속 전환 ★
        if absorb10 > 0: score += 1          # 외인 매도 흡수(수익화)
        if pos20 < 20: score += 1            # 아직 안 오름(전야)
        if score < 4:
            continue
        rows.append({
            "ticker": code, "name": names.get(code, ""),
            "score": score, "maxret5": round(maxret5, 1),
            "inst5_eok": round(inst5, 0), "inst_accel": round(inst5 - inst_prev5, 0),
            "foreign5_eok": round(foreign5, 0), "absorb10": round(absorb10, 2),
            "pos20": round(pos20, 1), "tv_eok": round(tv, 0), "close": round(close, 0),
        })
    rows.sort(key=lambda x: (-x["score"], -x["maxret5"]))

    print(f"=== 고개 드는 종목 (전환 감지) {len(rows)}건 / 최신 {latest} ===")
    print(f'{"종목":<18}{"점수":>4}{"5일최대등락":>9}{"기관5일":>9}{"기관가속":>9}{"외인5일":>9}{"흡수도":>7}{"20일":>7}{"거래대금":>8}')
    for c in rows[:30]:
        label = f"{c['name']}({c['ticker']})" if c["name"] else c["ticker"]
        print(f"{label:<18}{c['score']:>4}{c['maxret5']:>+8.1f}%{c['inst5_eok']:>+8,.0f}{c['inst_accel']:>+8,.0f}"
              f"{c['foreign5_eok']:>+8,.0f}{c['absorb10']:>+6.1f}%{c['pos20']:>+6.0f}%{c['tv_eok']:>7,.0f}억")

    OUT_FILE.write_text(json.dumps({
        "generated": datetime.now(tz=KST).isoformat(), "rule": "turning_up",
        "parquet_latest": latest, "count": len(rows), "candidates": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_FILE} ({len(rows)}건)")
    print("★ 사장님이 본 LG엔솔/삼성SDI/현대차가 상위에 잡히면 = 전환 감지 작동. 실매수 HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
