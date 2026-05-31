"""수급 D+20 스윙 selector — ⚠️ 실거래 부적합(연구 보류). 5/31 재검증.

★ 경고: 기존 "확정 룰 PF 1.68/강세장 2.40" 주장은 재현 불가 + 허수였음.
  supply_d20_backtest.py 재실행(5/31, 단타봇 직접 검증) 결과:
    - 기본재현(필터X)      전체3년 PF 1.32  (주장된 1.68 아님)
    - + 거래대금≥10억      전체3년 PF 0.72 / 최근25~26 PF 0.24·평균 -19.83%
    - + 슬리피지 0.5%      전체3년 PF 0.69
    - + 손절 -8%          최근25~26 승률 0%
  → 못 사는 잡주(생존자편향)가 PF를 부풀린 허수. 실거래 가능 종목만 보면 역엣지.
  실라이브 금지. 룰 재설계(다이버전스 방향) 전까지 후보 산출/기록만(HOLD).

룰(passes 동일):
  foreign_net_5d>0 AND inst_net_5d>0
  AND (foreign_consecutive_buy>=3 OR inst_consecutive_buy>=3)
  AND supply_divergence>0
  ⚠️ 죽은 조건: supply_divergence>0 = 당일 외인매도 → 그날 foreign_consecutive_buy가
     0으로 리셋 → foreign_consec>=3은 영원히 거짓. 실질 OR = inst_consec>=3만 작동.

data/processed/*.parquet 각 종목 최신일을 스캔해 룰 통과 후보를 선정.
퀀트봇 = 스윙 연구자(청사진). 실매수 HOLD — 후보 산출/기록만.
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

PROCESSED = PROJECT_ROOT / "data" / "processed"
NAME_DIR = PROJECT_ROOT / "stock_data_daily"  # {이름}_{코드}.csv 매핑용
OUT_FILE = PROJECT_ROOT / "data" / "supply_swing_candidates.json"
KST = timezone(timedelta(hours=9))
HOLD_DAYS = 20  # 백테스트 확정 보유기간


def _name_map() -> dict:
    m = {}
    for p in glob.glob(str(NAME_DIR / "*.csv")):
        stem = Path(p).stem  # 3S_060310
        if "_" in stem:
            name, code = stem.rsplit("_", 1)
            m[code.zfill(6)] = name
    return m


def passes(r: dict) -> bool:
    # ⚠️ foreign_consecutive_buy>=3은 죽은 조건(supply_divergence>0=당일 외인매도와
    #    상호배타 → 영원히 거짓). 실질 OR = inst_consecutive_buy>=3. (5/31 재검증)
    return (
        r.get("foreign_net_5d", 0) > 0
        and r.get("inst_net_5d", 0) > 0
        and (r.get("foreign_consecutive_buy", 0) >= 3 or r.get("inst_consecutive_buy", 0) >= 3)
        and r.get("supply_divergence", 0) > 0
    )


def main() -> int:
    import pandas as pd

    names = _name_map()
    files = glob.glob(str(PROCESSED / "*.parquet"))
    cands = []
    latest_seen = ""
    for f in files:
        code = Path(f).stem.zfill(6)
        try:
            df = pd.read_parquet(f).sort_index()
        except Exception:
            continue
        if len(df) < 5:
            continue
        r = df.iloc[-1].to_dict()
        d = str(df.index[-1].date())
        latest_seen = max(latest_seen, d)
        if passes(r):
            cands.append({
                "ticker": code,
                "name": names.get(code, ""),
                "date": d,
                "foreign_consec": int(r.get("foreign_consecutive_buy", 0)),
                "inst_consec": int(r.get("inst_consecutive_buy", 0)),
                "supply_div": float(r.get("supply_divergence", 0)),
                "accum_eff": round(float(r.get("accumulation_efficiency", 0) or 0), 3),
                "close": float(r.get("close", 0)),
                "rsi": round(float(r.get("rsi_14", 0) or 0), 1),
            })
    # 매집효율 강한 순 정렬
    cands.sort(key=lambda x: -x["accum_eff"])

    print(f"=== 수급 D+20 스윙 후보 {len(cands)}건 (parquet 최신일 {latest_seen}) ===")
    print(f"{'종목':<18}{'외인연속':>7}{'기관연속':>7}{'매집효율':>9}{'RSI':>6}{'종가':>10}")
    for c in cands[:25]:
        label = f"{c['name']}({c['ticker']})" if c["name"] else c["ticker"]
        print(f"{label:<18}{c['foreign_consec']:>7}{c['inst_consec']:>7}"
              f"{c['accum_eff']:>9.2f}{c['rsi']:>6.0f}{c['close']:>10,.0f}")

    OUT_FILE.write_text(json.dumps({
        "generated": datetime.now(tz=KST).isoformat(),
        "rule": "supply_swing_D20",
        "hold_days": HOLD_DAYS,
        "parquet_latest": latest_seen,
        "count": len(cands),
        "candidates": cands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_FILE} ({len(cands)}건)")
    if latest_seen < "2026-05":
        print(f"⚠️ parquet 최신일 {latest_seen} — BAT-D 갱신 멈춤(작업2 선결). 현 후보는 과거 기준.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
