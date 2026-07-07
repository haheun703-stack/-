"""시나리오 v1 일일 러너 (BAT-D G5.8) — 레짐(V0+V3b 섀도)·모드·워치리스트 기록.

7/7 퐝가님 지시("우리만의 시나리오"). 관측 전용·실주문 0·graceful exit 0.
- 레짐 V0 = 현행 규칙(close>MA20 & rv_pct<50 → BULL). 모드 판정의 기준.
- 레짐 V3b = 승격 후보(close>MA20 & 하방변동 252일 백분위<60) — 섀도 병행 기록,
  divergence 축적이 교체 결정(퐝가님)의 근거. 근거=data/backtest/brain_bull_relax_report.md
- 워치리스트 = FV 엔진 상위(fv_long) — 약세장 수급 팔로우는 기각됨(bear_accumulation_report).
- 행동표 = docs/scenario_v1.md

사용: ./venv/bin/python3.11 scripts/run_scenario_v1.py
"""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("scenario_v1")

DATA = PROJECT_ROOT / "data"
OUT = DATA / "shadow" / "scenario_v1.json"
HIST = DATA / "shadow" / "scenario_v1_history.jsonl"

MODE_BY_REGIME = {
    "CRISIS": ("DEFENSE", "신규 0·현금·관측만"),
    "BEAR": ("OBSERVE", "매수 없음·FV 워치리스트 축적·손절만 가동"),
    "BEAR/CAUTION": ("OBSERVE", "매수 없음·FV 워치리스트 축적"),
    "CAUTION": ("PREPARE", "워치리스트∩사이클 초중반 후보화·소규모 페이퍼만"),
    "BULL": ("ENTRY_WINDOW", "후보 중 3단계 수급 확인 종목만 진입(역추세 금지)"),
}


def _regimes() -> dict:
    """KOSPI CSV → V0(현행)·V3b(섀도) 레짐 동시 산출."""
    import numpy as np
    import pandas as pd
    k = pd.read_csv(DATA / "kospi_index.csv")
    k["Date"] = pd.to_datetime(k["Date"])
    k = k.sort_values("Date").set_index("Date")
    close = pd.to_numeric(k["close"], errors="coerce").dropna()
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    ret = close.pct_change()
    rv = ret.rolling(20).std() * np.sqrt(252) * 100
    rv_pct = (rv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50)
    dn = ret.where(ret < 0, 0.0)
    drv = dn.rolling(20).std() * np.sqrt(252) * 100
    drv_pct = (drv.rolling(252, min_periods=60).rank(pct=True) * 100).fillna(50)

    c, m20, m60 = close.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]

    def _reg(bull_ok: bool) -> str:
        if c > m20:
            return "BULL" if bull_ok else "CAUTION"
        if c > m60:
            return "BEAR"
        return "CRISIS"

    return {
        "asof": str(close.index[-1].date()),
        "kospi_close": round(float(c), 2),
        "regime_v0": _reg(float(rv_pct.iloc[-1]) < 50),
        "regime_v3b": _reg(float(drv_pct.iloc[-1]) < 60),
        "rv_pct": round(float(rv_pct.iloc[-1]), 1),
        "drv_pct": round(float(drv_pct.iloc[-1]), 1),
    }


def _watchlist(top_n: int = 10) -> list[dict]:
    """FV 엔진 산출 상위 — BEAR 워치리스트(수급 무관, 자격만)."""
    fv_path = DATA / "shadow" / "future_value.json"
    if not fv_path.exists():
        return []
    try:
        fv = json.loads(fv_path.read_text(encoding="utf-8"))
        cards = fv.get("scorecards", [])
        cards = [c for c in cards if c.get("cycle_signal") not in ("late", "over")]
        cards.sort(key=lambda c: c.get("fv_long") or 0, reverse=True)
        return [{"ticker": c["ticker"], "name": c.get("name"),
                 "fv_long": c.get("fv_long"), "tags": (c.get("tags") or [])[:4]}
                for c in cards[:top_n]]
    except Exception as e:  # noqa: BLE001
        logger.warning("[G5.8] FV 워치리스트 로드 실패(graceful): %s", e)
        return []


def main() -> int:
    try:
        reg = _regimes()
    except Exception as e:  # noqa: BLE001
        logger.warning("[G5.8] 레짐 산출 실패(graceful): %s", e)
        return 0
    mode, action = MODE_BY_REGIME.get(reg["regime_v0"], ("OBSERVE", "기본 관측"))
    wl = _watchlist()
    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **reg,
        "mode": mode,
        "action": action,
        "v3b_divergence": reg["regime_v0"] != reg["regime_v3b"],
        "watchlist": wl,
        "doc": "docs/scenario_v1.md",
        "validation": "shadow_observation_only",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(HIST, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: out[k] for k in
                            ("asof", "kospi_close", "regime_v0", "regime_v3b", "mode",
                             "v3b_divergence")}, ensure_ascii=False) + "\n")
    logger.info("[시나리오v1] %s | V0=%s V3b=%s%s | 모드=%s — %s",
                reg["asof"], reg["regime_v0"], reg["regime_v3b"],
                " ★divergence" if out["v3b_divergence"] else "", mode, action)
    if wl:
        logger.info("[시나리오v1] 워치리스트: %s",
                    ", ".join(f"{w['name']}({w['fv_long']})" for w in wl[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
