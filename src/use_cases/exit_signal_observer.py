"""FLOWX Market OS v1 exit_signal_observer (6단계).

SHADOW_OPEN 후보에 대해 "팔았으면 어땠나"(exit signal)를 **관찰·기록만** 한다.
실제 매도·청산·주문인텐트 생성·보유수량 변경 일절 없음(5/27 자동매도 사고 재발 방지).
기능 이름은 exit이지만 실제로는 observer다.

★절대 금지(import·호출 0): 스마트매도·매도브레인·오너룰·시장가매도·지정가매도·
주문인텐트게이트·주문어댑터. import는 OHLCV 로더와 5단계 산출물·국면 상수뿐.
계좌/포지션 파일도 건드리지 않는다(읽기 없음, 산출물은 data_store/exit_observer만).

설계: docs/02-design/flowx_market_os_v1.md §6, 진행 지시서 6단계.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.etf.c60_shadow import normalize_ohlcv
from src.etf.samsung_single_leverage_shadow import load_daily_ohlcv
from src.use_cases.engine_policy_map import MARKET_DATA_UNAVAILABLE, MARKET_R1
from src.use_cases.smart_entry_adapter import run_smart_entry_adapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXIT_OBSERVER_DIR = PROJECT_ROOT / "data_store" / "exit_observer"

OBSERVER_VERSION = "exit_signal_observer_v1"

# exit 기준(관찰용 라벨, 실주문 아님)
STOP_LEVELS = [-3.0, -5.0, -8.0]
TARGET_LEVELS = [3.0, 5.0, 7.0, 10.0]
TIME_HORIZONS = [1, 2, 3, 5, 10]

HOLD_OBSERVING = "HOLD_OBSERVING"          # R4 정상 관찰
HOLD_RISK_EXIT = "RISK_EXIT_OBSERVED"      # R1 관찰상 EXIT(실매도 X)
HOLD_BLOCKED = "HOLD_BLOCKED"              # DATA_UNAVAILABLE / OHLCV 없음


def _pct(price: float, entry: float) -> float | None:
    if not entry:
        return None
    return round((price - entry) / entry * 100, 2)


def compute_exit_signals(entry_price: float, ohlcv_after: pd.DataFrame | None) -> dict:
    """진입 후 OHLCV → MFE/MAE + exit 후보(손절/익절/추세이탈/시간청산). 순수 함수.

    ohlcv_after: entry_date 이후 일봉 DataFrame([open,high,low,close], DatetimeIndex).
    매도 실행이 아니라 "각 기준이 닿았는가"만 관찰한다.
    """
    if not entry_price or ohlcv_after is None or ohlcv_after.empty:
        return {
            "data_available": False,
            "current_close": None,
            "mfe_pct": None,
            "mae_pct": None,
            "exit_signals_triggered": [],
        }

    high_max = float(ohlcv_after["high"].max())
    low_min = float(ohlcv_after["low"].min())
    closes = ohlcv_after["close"]
    current_close = float(closes.iloc[-1])
    days = len(ohlcv_after)

    mfe_pct = _pct(high_max, entry_price)
    mae_pct = _pct(low_min, entry_price)

    triggered: list[dict] = []
    # 손절형(관찰): 저가가 손절선 이하로 닿았나
    for lv in STOP_LEVELS:
        if mae_pct is not None and mae_pct <= lv:
            triggered.append({"type": "stop", "level_pct": lv, "kind": "loss_cut"})
    # 익절형(관찰): 고가가 목표선 이상 닿았나
    for lv in TARGET_LEVELS:
        if mfe_pct is not None and mfe_pct >= lv:
            triggered.append({"type": "target", "level_pct": lv, "kind": "take_profit"})
    # 추세이탈형: 종가 MA5/MA10 이탈, 전일 저점 이탈
    if days >= 5:
        ma5 = float(closes.rolling(5).mean().iloc[-1])
        if current_close < ma5:
            triggered.append({"type": "trend", "kind": "below_ma5", "ref": round(ma5, 1)})
    if days >= 10:
        ma10 = float(closes.rolling(10).mean().iloc[-1])
        if current_close < ma10:
            triggered.append({"type": "trend", "kind": "below_ma10", "ref": round(ma10, 1)})
    if days >= 2:
        prev_low = float(ohlcv_after["low"].iloc[-2])
        if current_close < prev_low:
            triggered.append({"type": "trend", "kind": "below_prev_low", "ref": round(prev_low, 1)})
    # 시간청산형: D+N 종가 수익률(관찰)
    for h in TIME_HORIZONS:
        if days > h:
            triggered.append({"type": "time", "horizon": f"D+{h}", "return_pct": _pct(float(closes.iloc[h]), entry_price)})

    return {
        "data_available": True,
        "current_close": int(current_close),
        "highest_price": int(high_max),
        "lowest_price": int(low_min),
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "days_observed": days,
        "exit_signals_triggered": triggered,
    }


def _best_worst(signals: list[dict]) -> tuple[dict | None, dict | None]:
    """관찰 라벨: best=가장 크게 살릴 수 있던 익절, worst=가장 빨리 끊은 손절."""
    targets = [s for s in signals if s.get("type") == "target"]
    stops = [s for s in signals if s.get("type") == "stop"]
    best = max(targets, key=lambda s: s["level_pct"]) if targets else None
    worst = max(stops, key=lambda s: s["level_pct"]) if stops else None  # -3%가 가장 빨리 끊음
    return best, worst


def build_exit_observation(
    entry: dict, ohlcv_after: pd.DataFrame | None, market_regime: str, observation_date: str | None = None
) -> dict:
    """단일 후보 → exit observation. 순수 함수(주문/계좌/sell 경로 없음)."""
    entry_price = float(entry.get("virtual_entry_price") or 0)
    calc = compute_exit_signals(entry_price, ohlcv_after)

    if market_regime == MARKET_DATA_UNAVAILABLE or not calc["data_available"]:
        hold_status = HOLD_BLOCKED
        blocked_reason = "DATA_UNAVAILABLE" if market_regime == MARKET_DATA_UNAVAILABLE else "no_ohlcv"
    elif market_regime == MARKET_R1:
        hold_status = HOLD_RISK_EXIT
        blocked_reason = None
    else:
        hold_status = HOLD_OBSERVING
        blocked_reason = None

    signals = calc.get("exit_signals_triggered", [])
    best, worst = _best_worst(signals)

    # ── D+1 시가 기준(시가 진입) 별도 계산 ──
    # d1_open_filled면 ohlcv를 D+1(인덱스 1)부터 슬라이스해 시가 진입가 기준 재계산.
    # 미충전(다음날 시가 미도래)이면 pending(None). ★진입가만 다를 뿐 매도/주문 0.
    d1_filled = bool(entry.get("d1_open_filled"))
    d1_price = entry.get("virtual_entry_price_d1_open")
    calc_d1 = None
    if d1_filled and d1_price and ohlcv_after is not None and len(ohlcv_after) > 1:
        calc_d1 = compute_exit_signals(float(d1_price), ohlcv_after.iloc[1:])

    return {
        "date": observation_date or datetime.now().strftime("%Y-%m-%d"),
        "ticker": entry.get("ticker"),
        "name": entry.get("name", entry.get("ticker")),
        "source_type": entry.get("source_type", "smart_entry_adapter"),
        "tier": entry.get("tier"),
        "entry_date": entry.get("entry_date"),
        # ── D0 종가 기준(기존, 하위호환) ──
        "virtual_entry_price": int(entry_price) if entry_price else None,
        "current_close": calc.get("current_close"),
        "mfe_pct": calc.get("mfe_pct"),
        "mae_pct": calc.get("mae_pct"),
        "exit_signals_triggered": signals,
        "best_exit_candidate": best,
        "worst_exit_candidate": worst,
        "hold_status": hold_status,
        "blocked_reason": blocked_reason,
        # ── D+1 시가 기준(시가 진입) 관찰 — pending이면 None/빈배열 ──
        "virtual_entry_price_d1_open": (int(d1_price) if d1_price else None),
        "d1_open_filled": d1_filled,
        "mfe_pct_d1": (calc_d1.get("mfe_pct") if calc_d1 else None),
        "mae_pct_d1": (calc_d1.get("mae_pct") if calc_d1 else None),
        "exit_signals_triggered_d1": (calc_d1.get("exit_signals_triggered") if calc_d1 else []),
        # ── 안전 못박기(관찰 전용) ──
        "real_order": False,
        "sell_automation": "BLOCKED",
        "order_intent_created": False,
    }


def build_observer_document(
    entries: list[dict], market_regime: str, ohlcv_map: dict, observation_date: str | None = None
) -> dict:
    """여러 후보 → observer 문서. ohlcv_map: {ticker: ohlcv_after_df} 주입(테스트 가능)."""
    obs = [
        build_exit_observation(e, (ohlcv_map or {}).get(e.get("ticker")), market_regime, observation_date)
        for e in entries
    ]
    return {
        "version": OBSERVER_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_date": observation_date or datetime.now().strftime("%Y-%m-%d"),
        "market_regime": market_regime,
        "observations": obs,
        "counts": {"observed": len(obs)},
        "safety": {
            "real_order": False,
            "order_adapter": "None",
            "dry_run": True,
            "sell_automation": "BLOCKED",
            "order_intent_created": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "position_modified": False,
        },
    }


def build_observer_markdown(document: dict) -> str:
    md = [
        f"# FLOWX exit 관찰 — {document['observation_date']}",
        "",
        f"> 매도 자동화 없음. \"팔았으면 어땠나\"만 관찰. (실주문 0 / sell_automation BLOCKED)",
        "",
        f"- 시장국면: **{document['market_regime']}**",
        f"- 관찰 종목 수: {document['counts']['observed']}",
        "",
        "| 종목 | tier | 진입참조 | 현재가 | MFE% | MAE% | 상태 | 도달 exit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in document["observations"]:
        kinds = ", ".join(
            s.get("kind") or s.get("horizon") or str(s.get("level_pct"))
            for s in o.get("exit_signals_triggered", [])
        ) or "-"
        md.append(
            f"| {o['name']}({o['ticker']}) | {o['tier']} | {o['virtual_entry_price']} | "
            f"{o['current_close']} | {o['mfe_pct']} | {o['mae_pct']} | {o['hold_status']} | {kinds} |"
        )
    md.append("")
    md.append("---")
    md.append("**실주문 0 / 매도 자동화 BLOCKED / order_intent 0 / 보유수량 변경 0 / scheduler·SAJANG 변경 0**")
    return "\n".join(md) + "\n"


def save_observer(document: dict, output_dir: Path = EXIT_OBSERVER_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = str(document.get("observation_date") or datetime.now().strftime("%Y-%m-%d"))
    json_path = output_dir / f"exit_observer_{d}.json"
    md_path = output_dir / f"exit_observer_{d}.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_observer_markdown(document), encoding="utf-8")
    return json_path, md_path


def _slice_after(ticker: str, entry_date: str | None, days: int = 260, prefer_remote: bool = True):
    """entry_date 이후 일봉 OHLCV 슬라이스. 실패 시 None(관찰 차단)."""
    if not ticker or not entry_date:
        return None
    try:
        df = normalize_ohlcv(load_daily_ohlcv(ticker, days=days, prefer_remote=prefer_remote))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    after = df.loc[df.index >= pd.Timestamp(entry_date)]
    return after if not after.empty else None


def run_exit_observer(
    days: int = 1300, prefer_remote: bool = True, write: bool = True
) -> tuple[dict, Path | None, Path | None]:
    """5단계 SHADOW_OPEN 후보 → exit 관찰 문서. 매도 실행 없음."""
    doc5, _ = run_smart_entry_adapter(days=days, prefer_remote=prefer_remote, write=False)
    market_regime = doc5.get("market_regime")
    entry_date = doc5.get("as_of_date")
    entries = [
        {
            "ticker": e.get("ticker"),
            "name": e.get("name"),
            "tier": e.get("tier"),
            "virtual_entry_price": e.get("virtual_entry_price_d0_close") or e.get("close"),
            "virtual_entry_price_d1_open": e.get("virtual_entry_price_d1_open"),
            "d1_open_filled": e.get("d1_open_filled", False),
            "entry_date": entry_date,
            "source_type": "smart_entry_adapter",
        }
        for e in doc5.get("shadow_entries", [])
    ]
    ohlcv_map = {e["ticker"]: _slice_after(e["ticker"], e["entry_date"], prefer_remote=prefer_remote) for e in entries}
    document = build_observer_document(entries, market_regime, ohlcv_map, observation_date=entry_date)
    if write:
        json_path, md_path = save_observer(document)
        return document, json_path, md_path
    return document, None, None
