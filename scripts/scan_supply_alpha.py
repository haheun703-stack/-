"""수급 알파 통합 selector (백테스트 검증 룰 — 5/31).

검증 근거(engine_ensemble_backtest + 워크포워드 + 레짐필터, 거래대금복구 후):
  supply_divergence>0 (+외인/기관 동반) + 중기 D+40 보유 + KOSPI 과열 회피
  → 2023~2025 시장 대비 +2.7~5.4%p 초과(약세장 포함), 과열회피로 2026도 양수.
  순수 기술신호(is_bullish/sar)는 베타뿐이라 제외.

현재 시점 후보 산출(lookahead 무관). KOSPI 과열(120일선+15%) 시 '신규진입 대기' 경고.
퀀트봇=연구자. 실매수 HOLD. 후보 산출/기록만. PASS 전 가설(생존편향·OOS 추가검증 필요).
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

import pandas as pd

PROCESSED = PROJECT_ROOT / "data" / "processed"
NAME_DIR = PROJECT_ROOT / "stock_data_daily"
KOSPI = PROJECT_ROOT / "data" / "kospi_index.csv"
OUT_FILE = PROJECT_ROOT / "data" / "supply_alpha_candidates.json"
KST = timezone(timedelta(hours=9))

HOLD_DAYS = 40         # 중기 (백테스트 알파 최대 구간)
MIN_TRADING_VALUE = 1e9  # 거래대금 10억
OVERHEAT_MULT = 1.15   # KOSPI 120일선 +15% = 과열


def _name_map() -> dict:
    m = {}
    for p in glob.glob(str(NAME_DIR / "*.csv")):
        stem = Path(p).stem
        if "_" in stem:
            name, code = stem.rsplit("_", 1)
            m[code.zfill(6)] = name
    return m


def _kospi_overheat() -> tuple[bool, float, float]:
    """KOSPI 현재 과열 여부 (120일선 +15% 이상)."""
    k = pd.read_csv(KOSPI)
    k.columns = [c.strip().lower() for c in k.columns]
    k["date"] = pd.to_datetime(k["date"])
    k = k.set_index("date").sort_index()
    ma120 = k["close"].rolling(120).mean().iloc[-1]
    close = k["close"].iloc[-1]
    return bool(close > ma120 * OVERHEAT_MULT), float(close), float(ma120)


def main() -> int:
    names = _name_map()
    overheat, kclose, kma120 = _kospi_overheat()
    files = glob.glob(str(PROCESSED / "*.parquet"))
    cands = []
    latest = ""
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
        latest = max(latest, d)
        # 핵심 알파: supply_divergence>0 + 거래대금 + (외인/기관 동반 가점)
        sd = float(r.get("supply_divergence", 0) or 0)
        tv = float(r.get("trading_value", 0) or 0)
        if sd <= 0 or tv < MIN_TRADING_VALUE:
            continue
        fc = int(r.get("foreign_consecutive_buy", 0) or 0)
        ic = int(r.get("inst_consecutive_buy", 0) or 0)
        dual = fc >= 3 and ic >= 3  # 외인+기관 동반(강한 알파 조합)
        close = float(r.get("close", 0) or 0)
        cands.append({
            "ticker": code,
            "name": names.get(code, ""),
            "date": d,
            "supply_div": round(sd, 3),
            "dual_supply": dual,
            "foreign_consec": fc,
            "inst_consec": ic,
            "accum_eff": round(float(r.get("accumulation_efficiency", 0) or 0), 3),
            "close": round(close, 0),
            "trading_value_eok": round(tv / 1e8, 1),
            "target_d40_ref": "중기 D+40 보유 (백테스트 알파 구간)",
        })
    # 외인기관 동반 우선 + supply_div 강한 순
    cands.sort(key=lambda x: (not x["dual_supply"], -x["supply_div"]))

    print(f"=== 수급 알파 후보 {len(cands)}건 (최신 {latest}) ===")
    print(f"KOSPI: close {kclose:,.0f} / MA120 {kma120:,.0f} → {'🔴 과열(신규진입 대기 권고)' if overheat else '🟢 정상(진입 가능 국면)'}")
    if overheat:
        print("  ※ 과열 국면: 백테스트상 수급 알파가 극강세장에서 무력 → 신규진입 보류 권고.")
    print(f"{'종목':<18}{'동반':>5}{'외인':>5}{'기관':>5}{'supply_div':>11}{'매집효율':>9}{'거래대금억':>10}{'종가':>10}")
    for c in cands[:30]:
        label = f"{c['name']}({c['ticker']})" if c["name"] else c["ticker"]
        print(f"{label:<18}{'★' if c['dual_supply'] else '-':>5}{c['foreign_consec']:>5}{c['inst_consec']:>5}"
              f"{c['supply_div']:>11.2f}{c['accum_eff']:>9.2f}{c['trading_value_eok']:>10.1f}{c['close']:>10,.0f}")

    OUT_FILE.write_text(json.dumps({
        "generated": datetime.now(tz=KST).isoformat(),
        "rule": "supply_alpha_D40_regime",
        "hold_days": HOLD_DAYS,
        "kospi_overheat": overheat,
        "parquet_latest": latest,
        "count": len(cands),
        "dual_count": sum(1 for c in cands if c["dual_supply"]),
        "candidates": cands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_FILE} ({len(cands)}건, 외인기관동반 {sum(1 for c in cands if c['dual_supply'])}건)")
    print("★ 검증된 방향(수급+중기+과열회피)이나 생존편향·OOS 추가검증 전 가설. 실매수 HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
