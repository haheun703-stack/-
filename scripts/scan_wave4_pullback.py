"""3파4파 눌림목 selector (코덱스 3단계 룰화, 5/31).

엘리어트 4파 눌림 진입 룰 ("3파,4파 상승.txt" 자막 기반 + geometric_engine 재사용):
  1. IMPULSE 상승 파동 식별 (ElliottWaveAnalyzer.analyze, confidence>=60)
  2. current_wave == "4" 또는 "4~5 전환" (4파 조정 진행 중)
  3. 규칙3 통과: 4파 저점 > 1파 고점 (파동중첩 금지) ← 자막 핵심 + 손절선
  4. 4파 되돌림 23.6~61.8% (3파 길이 대비, 38.2% 중심) ← 자막 "38.2% 지지"
  5. 거래량 감소 (4파 횡보 — volume_ma5 < volume_ma20) ← 자막 "파동 교대법칙"
진입(참고): 38.2% 지지권 + 거래량 감소 확인 시 후보. 5파 상승 노림.
손절: 4파 저점(=1파 고점 근처) 하회 = 파동 무효.
목표: 5파 0.618 / 1.000 (wave.fib_targets).

★ lookahead 주의: analyze는 df 끝점 기준 판정이라 '현재 후보 스캔'은 안전.
  과거 PF 백테스트는 매 시점 df.iloc[:i+1] 재호출 필요(무거움) → 별도 신중 설계
  (수급D20 허수 교훈: 거래대금 필터/슬리피지/생존자편향 처음부터 반영).
퀀트봇 = 연구자. 실매수 HOLD. 후보 산출/기록만.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.geometric_engine import ElliottWaveAnalyzer, WaveType

PROCESSED = PROJECT_ROOT / "data" / "processed"
NAME_DIR = PROJECT_ROOT / "stock_data_daily"
OUT_FILE = PROJECT_ROOT / "data" / "wave4_pullback_candidates.json"
KST = timezone(timedelta(hours=9))

CONF_MIN = 60.0
RETRACE_MIN, RETRACE_MAX = 0.236, 0.618  # 4파 되돌림 허용폭 (38.2% 중심)


def _name_map() -> dict:
    m = {}
    for p in glob.glob(str(NAME_DIR / "*.csv")):
        stem = Path(p).stem
        if "_" in stem:
            name, code = stem.rsplit("_", 1)
            m[code.zfill(6)] = name
    return m


def check(df: pd.DataFrame, analyzer: ElliottWaveAnalyzer) -> dict | None:
    if len(df) < 80:
        return None
    try:
        wave = analyzer.analyze(df, lookback=200)
    except Exception:
        return None
    if wave is None:
        return None
    if wave.wave_type != WaveType.IMPULSE or wave.direction != "up":
        return None
    if wave.current_wave not in ("4", "4~5 전환"):
        return None
    if wave.confidence < CONF_MIN:
        return None
    # 규칙3 중첩금지 통과 (4파 저점 > 1파 고점) — 자막 핵심 + 무효화 기준
    if not any(r.startswith("규칙3") for r in wave.rules_passed):
        return None

    w = wave.waves
    try:
        w1_high = float(w["1"][3])
        w3_start = float(w["3"][2])
        w3_end = float(w["3"][3])
        w4_low = float(w["4"][3])
    except (KeyError, IndexError, TypeError):
        return None
    w3_len = w3_end - w3_start
    if w3_len <= 0:
        return None
    retrace = (w3_end - w4_low) / w3_len  # 4파 되돌림 비율
    if not (RETRACE_MIN <= retrace <= RETRACE_MAX):
        return None

    last = df.iloc[-1]
    vma5 = float(last.get("volume_ma5", 0) or 0)
    vma20 = float(last.get("volume_ma20", 0) or 0)
    vol_contract = vma20 > 0 and vma5 < vma20  # 거래량 수축(교대법칙)
    close = float(last["close"])
    tv = float(last.get("trading_value", 0) or 0)

    return {
        "current_wave": wave.current_wave,
        "confidence": round(wave.confidence, 1),
        "retrace_382": round(retrace, 3),
        "vol_contract": bool(vol_contract),
        "w1_high": round(w1_high, 0),
        "w4_low": round(w4_low, 0),
        "close": round(close, 0),
        "trading_value_eok": round(tv / 1e8, 1),  # 거래대금(억)
        "fib_5_0618": wave.fib_targets.get("5파_0.618"),
        "fib_5_1000": wave.fib_targets.get("5파_1.000"),
        "stop_below": round(w1_high, 0),  # 손절: 1파 고점 하회
    }


def main() -> int:
    analyzer = ElliottWaveAnalyzer(zigzag_pct=3.0, min_bars=3)
    names = _name_map()
    files = glob.glob(str(PROCESSED / "*.parquet"))
    cands = []
    latest_seen = ""
    scanned = 0
    for f in files:
        code = Path(f).stem.zfill(6)
        try:
            df = pd.read_parquet(f).sort_index()
        except Exception:
            continue
        scanned += 1
        if len(df):
            latest_seen = max(latest_seen, str(df.index[-1].date()))
        res = check(df, analyzer)
        if res:
            res["ticker"] = code
            res["name"] = names.get(code, "")
            cands.append(res)

    # 거래량 수축 우선 + 38.2% 근접 + 신뢰도
    cands.sort(key=lambda x: (not x["vol_contract"], abs(x["retrace_382"] - 0.382), -x["confidence"]))

    print(f"=== 3파4파 눌림 후보 {len(cands)}건 / 스캔 {scanned}종목 (최신 {latest_seen}) ===")
    print(f"{'종목':<18}{'파동':>8}{'신뢰':>6}{'되돌림':>7}{'거래량↓':>7}{'거래대금억':>9}{'종가':>10}")
    for c in cands[:30]:
        label = f"{c['name']}({c['ticker']})" if c["name"] else c["ticker"]
        print(f"{label:<18}{c['current_wave']:>8}{c['confidence']:>6.0f}"
              f"{c['retrace_382']:>7.2f}{'O' if c['vol_contract'] else '-':>7}"
              f"{c['trading_value_eok']:>9.1f}{c['close']:>10,.0f}")

    OUT_FILE.write_text(json.dumps({
        "generated": datetime.now(tz=KST).isoformat(),
        "rule": "wave4_pullback",
        "parquet_latest": latest_seen,
        "count": len(cands),
        "candidates": cands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_FILE} ({len(cands)}건)")
    print("★ 후보 스캔(현재 시점, lookahead 안전). 과거 PF 백테스트는 별도 신중 설계 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
