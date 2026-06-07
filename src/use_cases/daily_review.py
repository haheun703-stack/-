"""FLOWX Market OS v1 daily_review (7단계).

장마감 후 FLOWX 관측 결과를 복기한다. 매매 판단·자동 실행이 아니라 **후보선정 성능과
실행 성능을 분리**해서 리뷰만 한다(설계도 §5 "사과 vs 오렌지 안 섞기").

★기준 분리(가장 중요):
- 후보선정 성능 = **as_of 종가** 기준 (종목을 잘 골랐는가). CORE/WATCH/CONTROL 공통.
- 실행 성능   = **virtual_entry_price** 기준 (진입 타이밍이 좋았는가). SHADOW_OPEN만.
두 기준가를 섞지 않는다.

매수·매도·주문·계좌 실행 경로를 일절 import·호출하지 않는다(리뷰는 읽기·계산·기록만).
리뷰 결과로 정책/스케줄러/SAJANG을 바꾸지 않는다.

설계: docs/02-design/flowx_market_os_v1.md §5, 진행 지시서 7단계.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.use_cases.exit_signal_observer import _slice_after, run_exit_observer
from src.use_cases.morning_plan_07 import run_morning_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_DIR = PROJECT_ROOT / "data_store" / "reviews"

REVIEW_VERSION = "daily_review_v1"
HORIZONS = (1, 3, 5, 10)
MISSED_WINNER_PCT = 8.0   # CONTROL이 D+10 이만큼 오르면 놓친 승자


def _pct(price: float, base: float) -> float | None:
    if not base:
        return None
    return round((price - base) / base * 100, 2)


def _fwd_returns(closes, base: float) -> dict:
    out = {}
    for h in HORIZONS:
        out[f"D+{h}"] = _pct(float(closes.iloc[h]), base) if len(closes) > h else None
    return out


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _label_groups(rows: list[dict], key_fn) -> dict:
    """후보 행을 라벨로 묶어 D+1/D+3/D+10/MFE 평균·건수 집계(관측 비교용)."""
    groups: dict = {}
    for r in rows:
        if not r.get("data_available"):
            continue
        key = key_fn(r)
        if key is None:
            continue
        g = groups.setdefault(key, {"count": 0, "d1": [], "d3": [], "d10": [], "mfe": []})
        fwd = r.get("raw_fwd_pct") or {}
        g["count"] += 1
        g["d1"].append(fwd.get("D+1"))
        g["d3"].append(fwd.get("D+3"))
        g["d10"].append(fwd.get("D+10"))
        g["mfe"].append(r.get("mfe_pct"))
    return {
        k: {
            "count": g["count"],
            "mean_d1": _mean(g["d1"]), "mean_d3": _mean(g["d3"]),
            "mean_d10": _mean(g["d10"]), "mean_mfe": _mean(g["mfe"]),
        }
        for k, g in groups.items()
    }


def _sl(r: dict, *path):
    """후보 행의 shadow_labels에서 중첩 키 안전 추출."""
    node = r.get("shadow_labels") or {}
    for p in path:
        node = (node or {}).get(p) if isinstance(node, dict) else None
    return node


def build_label_performance(rows: list[dict]) -> dict:
    """관측 라벨별 D+N 성과 비교(지시서 §8 검증질문 대응). 순수 함수.

    매수/매도 판단이 아니라 "어떤 라벨이 실제로 성과가 좋았나"를 6/12에 보기 위한
    누적 집계다. 라벨 없는 후보는 자동 제외(degrade).
    """
    return {
        "weekly_open": _label_groups(rows, lambda r: _sl(r, "price_axis", "weekly_open_state")),
        "half_year_open": _label_groups(rows, lambda r: _sl(r, "price_axis", "half_year_open_state")),
        "half_year_leader_grade": _label_groups(rows, lambda r: _sl(r, "half_year_leader", "half_year_leader_grade")),
        "candle_turn": _label_groups(rows, lambda r: (_sl(r, "candle_turn", "label") or None)),
        "annual_overheat": _label_groups(rows, lambda r: (_sl(r, "annual_overheat", "overheat_grade") or "NONE")),
        "ipo_reversion": _label_groups(rows, lambda r: _sl(r, "ipo_reversion", "ipo_reversion_state")),
        "note": "라벨은 관측용. 좋아도 hard gate 승격은 사장님 승인 별도(즉시 진입룰 승격 금지).",
    }


def build_candidate_review(candidates: list[dict], ohlcv_map: dict) -> dict:
    """A. 후보선정 성능 — as_of 종가 기준. CORE/WATCH/CONTROL 전부(공정 비교).

    ohlcv_map: {ticker: as_of_date 이후 OHLCV DataFrame([o,h,l,c], idx[0]=as_of일)}.
    """
    rows: list[dict] = []
    tier_counts = {"CORE": 0, "WATCH": 0, "CONTROL": 0}
    missed_winner: list[dict] = []
    false_positive: list[dict] = []

    for c in candidates:
        tier = c.get("tier")
        if tier in tier_counts:
            tier_counts[tier] += 1
        df = (ohlcv_map or {}).get(c.get("ticker"))
        if df is None or df.empty:
            rows.append({
                "ticker": c.get("ticker"), "name": c.get("name"), "tier": tier,
                "as_of_close": None, "raw_fwd_pct": {f"D+{h}": None for h in HORIZONS},
                "mfe_pct": None, "mae_pct": None, "data_available": False,
                "shadow_labels": c.get("shadow_labels"),
            })
            continue
        closes = df["close"]
        base = float(closes.iloc[0])  # as_of 종가
        fwd = _fwd_returns(closes, base)
        rows.append({
            "ticker": c.get("ticker"), "name": c.get("name"), "tier": tier,
            "as_of_close": int(base), "raw_fwd_pct": fwd,
            "mfe_pct": _pct(float(df["high"].max()), base),
            "mae_pct": _pct(float(df["low"].min()), base),
            "data_available": True,
            "shadow_labels": c.get("shadow_labels"),
        })
        d10 = fwd.get("D+10")
        if d10 is not None:
            if tier == "CONTROL" and d10 >= MISSED_WINNER_PCT:
                missed_winner.append({"ticker": c.get("ticker"), "name": c.get("name"), "d10_pct": d10})
            if tier in ("CORE", "WATCH") and d10 < 0:
                false_positive.append({"ticker": c.get("ticker"), "name": c.get("name"), "d10_pct": d10})

    return {
        "basis": "as_of_close",
        "candidate_count": len(candidates),
        "tier_counts": tier_counts,
        "candidates": rows,
        "missed_winner": missed_winner,
        "false_positive": false_positive,
        "label_performance": build_label_performance(rows),
    }


def build_execution_review(observations: list[dict]) -> dict:
    """B. 실행 성능 — virtual_entry_price 기준. 6단계 exit observer 산출 재사용."""
    entries: list[dict] = []
    for o in observations or []:
        times = {
            s["horizon"]: s.get("return_pct")
            for s in o.get("exit_signals_triggered", [])
            if s.get("type") == "time"
        }
        entries.append({
            "ticker": o.get("ticker"), "name": o.get("name"), "tier": o.get("tier"),
            "entry_price": o.get("virtual_entry_price"),
            "pnl_pct": {f"D+{h}": times.get(f"D+{h}") for h in HORIZONS},
            "mfe_pct": o.get("mfe_pct"),
            "mae_pct": o.get("mae_pct"),
            "best_exit_candidate": o.get("best_exit_candidate"),
            "worst_exit_candidate": o.get("worst_exit_candidate"),
            "exit_signal_triggered": o.get("exit_signals_triggered", []),
            "avoid_loss_pct": o.get("mae_pct"),      # 손절로 회피 가능했던 최대 손실
            "missed_upside_pct": o.get("mfe_pct"),    # 못 챙긴 최대 수익
            "hold_status": o.get("hold_status"),
        })
    return {"basis": "virtual_entry_price", "entry_count": len(entries), "entries": entries}


def build_exit_summary(observations: list[dict]) -> dict:
    """C. exit observer 연결 — 어떤 exit 룰이 얼마나 트리거됐나(관찰 집계)."""
    type_counts = {"stop": 0, "target": 0, "trend": 0, "time": 0}
    for o in observations or []:
        for s in o.get("exit_signals_triggered", []):
            t = s.get("type")
            if t in type_counts:
                type_counts[t] += 1
    return {
        "exit_type_trigger_counts": type_counts,
        "note": "어떤 exit 룰이 손실을 줄였나/수익을 빨리 끊었나는 6/12 누적 비교(관찰만, 실매도 X)",
    }


def build_review_document(
    observation_date: str, market_regime: str,
    candidate_review: dict, execution_review: dict, exit_summary: dict,
    data_warnings: list | None = None,
) -> dict:
    cr, er = candidate_review, execution_review
    one_line = (
        f"후보 {cr['candidate_count']}(CORE {cr['tier_counts']['CORE']}/WATCH "
        f"{cr['tier_counts']['WATCH']}/CONTROL {cr['tier_counts']['CONTROL']}) · "
        f"실행관찰 {er['entry_count']} · missed_winner {len(cr['missed_winner'])} · "
        f"false_positive {len(cr['false_positive'])} (복기 전용, 실주문 0)"
    )
    return {
        "version": REVIEW_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_date": observation_date,
        "market_regime": market_regime,
        "one_line": one_line,
        "candidate_performance": cr,   # A: as_of 종가 기준
        "execution_performance": er,   # B: virtual_entry_price 기준
        "exit_observer_summary": exit_summary,  # C
        "data_warnings": data_warnings or [],
        "safety": {
            "real_order": False,
            "sell_automation": "BLOCKED",
            "order_intent_created": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "policy_changed": False,
            "position_modified": False,
        },
    }


def _fmt_label_groups(title: str, groups: dict) -> str:
    """라벨 그룹 → '버킷(n=건수, D+10평균 X%, MFE Y%)' 한 줄."""
    if not groups:
        return f"- {title}: (라벨 데이터 없음)"
    parts = []
    for bucket, g in groups.items():
        parts.append(
            f"{bucket}(n={g['count']}, D+10 {g['mean_d10']}%, MFE {g['mean_mfe']}%)"
        )
    return f"- {title}: " + " | ".join(parts)


def build_review_markdown(doc: dict) -> str:
    cr = doc["candidate_performance"]
    er = doc["execution_performance"]
    ex = doc["exit_observer_summary"]
    lp = cr.get("label_performance", {}) or {}
    md = [
        f"# FLOWX 일일 복기 — {doc['observation_date']}",
        "",
        f"> {doc['one_line']}",
        "",
        f"- 데이터 기준일: {doc['observation_date']}",
        f"- 시장국면: **{doc['market_regime']}**",
        "",
        "## 1. 후보선정 성능 (기준: as_of 종가)",
        f"- 후보 {cr['candidate_count']} · CORE {cr['tier_counts']['CORE']} / "
        f"WATCH {cr['tier_counts']['WATCH']} / CONTROL {cr['tier_counts']['CONTROL']}",
        "",
        "| 종목 | tier | as_of종가 | D+1 | D+3 | D+5 | D+10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cr["candidates"]:
        f = r["raw_fwd_pct"]
        md.append(
            f"| {r['name']}({r['ticker']}) | {r['tier']} | {r['as_of_close']} | "
            f"{f['D+1']} | {f['D+3']} | {f['D+5']} | {f['D+10']} |"
        )
    md += [
        "",
        "## 2. SmartEntry 실행 성능 (기준: virtual_entry_price)",
        f"- 실행 관찰 {er['entry_count']}",
        "",
        "| 종목 | tier | 진입가 | D+1 | D+3 | D+5 | D+10 | MFE | MAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in er["entries"]:
        p = e["pnl_pct"]
        md.append(
            f"| {e['name']}({e['ticker']}) | {e['tier']} | {e['entry_price']} | "
            f"{p['D+1']} | {p['D+3']} | {p['D+5']} | {p['D+10']} | {e['mfe_pct']} | {e['mae_pct']} |"
        )
    md += [
        "",
        "## 3. Exit observer 요약",
        f"- exit 트리거: {ex['exit_type_trigger_counts']}",
        f"- {ex['note']}",
        "",
        "## 4. missed_winner (CONTROL D+10 ≥ +8%)",
        "\n".join(f"- {m['name']}({m['ticker']}) D+10 {m['d10_pct']}%" for m in cr["missed_winner"]) or "- (없음)",
        "",
        "## 5. false_positive (CORE/WATCH D+10 < 0)",
        "\n".join(f"- {m['name']}({m['ticker']}) D+10 {m['d10_pct']}%" for m in cr["false_positive"]) or "- (없음)",
        "",
        "## 6. 라벨별 성과 비교 (관측 — 지시서 §8 검증질문)",
        _fmt_label_groups("반기시가 위/아래", lp.get("half_year_open", {})),
        _fmt_label_groups("주봉시가 위/아래", lp.get("weekly_open", {})),
        _fmt_label_groups("반기 주도주 등급", lp.get("half_year_leader_grade", {})),
        _fmt_label_groups("음양/양음 전환", lp.get("candle_turn", {})),
        _fmt_label_groups("연간 과열등급", lp.get("annual_overheat", {})),
        _fmt_label_groups("IPO 되돌림", lp.get("ipo_reversion", {})),
        f"- {lp.get('note', '')}",
        "",
    ]
    if doc["data_warnings"]:
        md.append("## 7. 데이터 경고")
        md.append("\n".join(f"- {w}" for w in doc["data_warnings"]))
        md.append("")
    md.append("---")
    md.append("**실주문 0 / 매도 자동화 BLOCKED / order_intent 0 / 정책·scheduler·SAJANG 변경 0 (복기 전용)**")
    return "\n".join(md) + "\n"


def save_review(doc: dict, output_dir: Path = REVIEW_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = str(doc.get("observation_date") or datetime.now().strftime("%Y-%m-%d"))
    json_path = output_dir / f"daily_review_{d}.json"
    md_path = output_dir / f"daily_review_{d}.md"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_review_markdown(doc), encoding="utf-8")
    return json_path, md_path


def run_daily_review(
    days: int = 1300, prefer_remote: bool = True, write: bool = True
) -> tuple[dict, Path | None, Path | None]:
    """6단계 exit observer(B/C) + morning_plan tiers(A) → 분리 복기 리포트."""
    obs_doc, _, _ = run_exit_observer(days=days, prefer_remote=prefer_remote, write=False)
    plan, _, _ = run_morning_plan(days=days, prefer_remote=prefer_remote, write=False)

    as_of_date = obs_doc.get("observation_date")
    market_regime = obs_doc.get("market_regime")
    observations = obs_doc.get("observations", [])

    tiers = plan.get("tiers", {})
    candidates = list(tiers.get("CORE", [])) + list(tiers.get("WATCH", [])) + list(tiers.get("CONTROL", []))
    cand_ohlcv = {
        c["ticker"]: _slice_after(c.get("ticker"), as_of_date, prefer_remote=prefer_remote)
        for c in candidates
    }
    warnings = [f"{c['ticker']}: OHLCV 없음(관찰 차단)" for c in candidates if cand_ohlcv.get(c["ticker"]) is None]

    doc = build_review_document(
        as_of_date, market_regime,
        build_candidate_review(candidates, cand_ohlcv),
        build_execution_review(observations),
        build_exit_summary(observations),
        data_warnings=warnings,
    )
    if write:
        json_path, md_path = save_review(doc)
        return doc, json_path, md_path
    return doc, None, None
