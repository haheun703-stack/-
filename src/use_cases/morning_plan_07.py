"""FLOWX Market OS v1 morning plan (3단계).

engine_policy_map 정책 + 최신 candidate_log를 합쳐, 아침 7시에 사람이 보는
당일 작전계획(JSON+MD)을 만든다. 계획 생성만 한다 — 스캐너/SmartEntry/주문/스케줄러
실행은 없다. 실주문 0 / 매도 BLOCKED / PAPER_OPEN 금지.

설계: docs/02-design/flowx_market_os_v1.md §1, 진행 지시서 3단계.
엔진 허용/차단의 단일진실은 2단계 engine_policy_map 정책표다(여기서 재해석 금지).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.use_cases.engine_policy_map import (
    MARKET_DATA_UNAVAILABLE,
    MARKET_R1,
    MARKET_R4,
    MODE_ALLOWED_SHADOW,
    MODE_PAPER_ONLY,
    run_policy_map,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLAN_DIR = PROJECT_ROOT / "data_store" / "plans"
LEDGER_PATH = PROJECT_ROOT / "data" / "paper_ledger.json"

PLAN_VERSION = "morning_plan_07_v1"
POLICY_SOURCE = "engine_policy_map_v1"

# 후보 관찰 라벨 (정책 smart_entry 모드 → 사람이 읽는 관찰 상태)
OBS_SMARTENTRY = "장중 SmartEntry 관찰 대상"
OBS_SHADOW = "shadow 관찰만 (SmartEntry 실행 안 함)"
OBS_COMPARISON = "missed-winner 비교군 (진입 대상 아님)"

ENGINE_LABELS = {
    "hypothesis_c": "가설C 후보선정",
    "smart_entry": "SmartEntry 장중 타이밍",
    "bc_rotation": "B/C 로테이션",
    "event_dart": "Event/DART",
    "sell": "매도 자동화",
}
ALLOWED_MODES = {MODE_PAPER_ONLY, MODE_ALLOWED_SHADOW}


def _candidate_summary(row: dict, observation: str) -> dict:
    return {
        "ticker": row.get("ticker"),
        "name": row.get("name", row.get("ticker")),
        "tier": row.get("_tier"),
        "ref_close": row.get("close"),
        "stop_loss": row.get("stop_loss"),
        "target": row.get("target_price"),
        "floor_label": row.get("_floor_label"),
        "drop_context": row.get("_drop_context"),
        "supply_state": row.get("_supply_state"),
        "observation": observation,
    }


def _candidate_log_meta(logs: list[dict]) -> dict:
    """최신 candidate_log 묶음 → 요약 메타. 여러 source면 합산."""
    if not logs:
        return {"as_of_date": None, "source": None, "total": 0, "enter_count": 0, "avoid_count": 0}
    sources = sorted({l.get("source") for l in logs if l.get("source")})
    return {
        "as_of_date": logs[0].get("as_of_date"),
        "source": sources[0] if len(sources) == 1 else sources,
        "total": sum(int(l.get("total", 0) or 0) for l in logs),
        "enter_count": sum(int(l.get("enter_count", 0) or 0) for l in logs),
        "avoid_count": sum(int(l.get("avoid_count", 0) or 0) for l in logs),
    }


def build_plan_document(policy: dict, picks: list, control: list, candidate_log_meta: dict) -> dict:
    """정책 + 분류된 후보 → plan JSON. 순수 함수(파일/네트워크 없음).

    엔진 허용/차단은 policy(2단계)를 그대로 따른다. 여기서 재해석하지 않는다.
    """
    market = policy.get("market_regime")
    engines = policy.get("engines", {})
    smart_mode = engines.get("smart_entry")
    # R4(ALLOWED_SHADOW)면 CORE/WATCH는 장중 SmartEntry 관찰, 아니면 shadow만
    obs = OBS_SMARTENTRY if smart_mode == MODE_ALLOWED_SHADOW else OBS_SHADOW

    core = [_candidate_summary(p, obs) for p in picks if p.get("_tier") == "CORE"]
    watch = [_candidate_summary(p, obs) for p in picks if p.get("_tier") == "WATCH"]
    # CONTROL은 정책과 무관하게 항상 비교군(진입 대상 아님)
    ctrl = [_candidate_summary(c, OBS_COMPARISON) for c in control]

    blocked_or_shadow = []
    for key, mode in engines.items():
        if mode not in ALLOWED_MODES:
            blocked_or_shadow.append(f"{ENGINE_LABELS.get(key, key)}: {mode}")

    data_warnings = []
    if market == MARKET_DATA_UNAVAILABLE:
        data_warnings.append("기초자산 데이터 없음 → 시장 종합 DATA_UNAVAILABLE, 신규진입 차단")
    for ticker, info in (policy.get("per_ticker") or {}).items():
        if not info.get("data_available"):
            data_warnings.append(f"{ticker}: DATA_UNAVAILABLE")

    as_of_date = (
        policy.get("as_of_date")
        or candidate_log_meta.get("as_of_date")
        or datetime.now().strftime("%Y-%m-%d")
    )

    return {
        "version": PLAN_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "policy_source": POLICY_SOURCE,
        "market_regime": market,
        "market_regime_rule": policy.get("market_regime_rule"),
        "engines": engines,
        "hysteresis": policy.get("hysteresis"),
        "per_ticker": policy.get("per_ticker", {}),
        "shadow_advisories": policy.get("shadow_advisories", []),
        "paper_open_allowed": bool(policy.get("paper_open_allowed", False)),
        "sell_automation": policy.get("sell_automation", "BLOCKED"),
        "candidate_log": {
            "as_of_date": candidate_log_meta.get("as_of_date"),
            "source": candidate_log_meta.get("source"),
            "total": candidate_log_meta.get("total", 0),
            "enter_count": candidate_log_meta.get("enter_count", 0),
            "avoid_count": candidate_log_meta.get("avoid_count", 0),
        },
        "tiers": {"CORE": core, "WATCH": watch, "CONTROL": ctrl},
        "blocked_or_shadow_reason": blocked_or_shadow,
        "data_warnings": data_warnings,
        "safety": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "paper_open_default": False,
            "sell_automation": "BLOCKED",
        },
    }


def _one_line_conclusion(plan: dict) -> str:
    market = plan["market_regime"]
    core_n = len(plan["tiers"]["CORE"])
    watch_n = len(plan["tiers"]["WATCH"])
    if market == MARKET_DATA_UNAVAILABLE:
        return "데이터 없음 — 오늘은 신규진입 차단, 전부 shadow 관찰만."
    if market == MARKET_R1:
        return "C60 약세/위험 — 신규진입 차단, 후보는 관찰만, 매도는 신호만."
    if market == MARKET_R4:
        return f"C60 강세 — CORE {core_n}/WATCH {watch_n} 장중 SmartEntry 관찰 (실주문 0, PAPER_OPEN 금지)."
    return "시장국면 미정 — 보수적 차단."


def _fmt_candidates(rows: list) -> str:
    if not rows:
        return "- (없음)"
    lines = []
    for r in rows:
        lines.append(
            f"- {r['name']}({r['ticker']}) | {r.get('floor_label') or '-'} | "
            f"기준가 {r.get('ref_close')} 손절 {r.get('stop_loss')} 목표 {r.get('target')} | {r['observation']}"
        )
    return "\n".join(lines)


def build_plan_markdown(plan: dict) -> str:
    engines = plan["engines"]
    allowed = [ENGINE_LABELS.get(k, k) for k, m in engines.items() if m in ALLOWED_MODES]
    blocked = plan["blocked_or_shadow_reason"]

    per_ticker_lines = []
    for ticker, info in (plan.get("per_ticker") or {}).items():
        if not info.get("data_available"):
            per_ticker_lines.append(f"- {ticker}: DATA_UNAVAILABLE")
            continue
        hyst = " (히스테리시스 대기)" if info.get("in_hysteresis_window") else ""
        per_ticker_lines.append(
            f"- {ticker}: {info.get('hard_gate_regime')} (raw={info.get('c60_regime_raw')}){hyst}"
        )

    cl = plan["candidate_log"]
    md = [
        f"# FLOWX 아침 작전계획 — {plan['as_of_date']}",
        "",
        f"> {_one_line_conclusion(plan)}",
        "",
        f"- 기준일: {plan['as_of_date']}",
        f"- 시장 종합 정책: **{plan['market_regime']}**",
        f"  - {plan.get('market_regime_rule') or ''}",
        "",
        "## 기초자산별 C60 상태",
        "\n".join(per_ticker_lines) if per_ticker_lines else "- (없음)",
        "",
        "## 허용 엔진",
        "\n".join(f"- {a}" for a in allowed) if allowed else "- (없음)",
        "",
        "## 차단/Shadow 엔진",
        "\n".join(f"- {b}" for b in blocked) if blocked else "- (없음)",
        "",
        f"## 후보 (candidate_log {cl.get('as_of_date')} · source={cl.get('source')} · "
        f"진입 {cl.get('enter_count')}/회피 {cl.get('avoid_count')})",
        "",
        "### CORE 후보",
        _fmt_candidates(plan["tiers"]["CORE"]),
        "",
        "### WATCH 후보",
        _fmt_candidates(plan["tiers"]["WATCH"]),
        "",
        "### CONTROL 비교군 (진입 대상 아님)",
        _fmt_candidates(plan["tiers"]["CONTROL"]),
        "",
    ]
    if plan["data_warnings"]:
        md.append("## 데이터 경고")
        md.append("\n".join(f"- {w}" for w in plan["data_warnings"]))
        md.append("")
    md.append("---")
    md.append("**실주문 0 / 매도 BLOCKED / scheduler 변경 0 / SAJANG 변경 0 / PAPER_OPEN 금지**")
    return "\n".join(md) + "\n"


def save_plan(plan: dict, output_dir: Path = PLAN_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(plan.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))
    json_path = output_dir / f"plan_{as_of}.json"
    md_path = output_dir / f"plan_{as_of}.md"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_plan_markdown(plan), encoding="utf-8")
    return json_path, md_path


def _load_candidates() -> tuple[list, list, list]:
    """최신 candidate_log → (picks=CORE+WATCH, control=CONTROL, logs).

    scripts/paper_smart_entry.py 순수 분류 로직 재사용(지시서 §3.3). lazy import로
    use_case import-time 의존을 최소화한다. ledger 기록·main 실행 함수는 호출하지
    않는다(읽기 전용 분류만).
    """
    if not LEDGER_PATH.exists():
        return [], [], []
    from scripts.paper_smart_entry import (  # noqa: E402  lazy 재사용
        _latest_candidate_logs,
        build_picks_from_candidate_log,
    )

    picks, control, ledger = build_picks_from_candidate_log()
    logs = _latest_candidate_logs(ledger)
    return picks, control, logs


def run_morning_plan(
    days: int = 1300, prefer_remote: bool = True, write: bool = True
) -> tuple[dict, Path | None, Path | None]:
    policy, _ = run_policy_map(days=days, prefer_remote=prefer_remote, write=write)
    picks, control, logs = _load_candidates()
    meta = _candidate_log_meta(logs)
    plan = build_plan_document(policy, picks, control, meta)
    json_path, md_path = save_plan(plan) if write else (None, None)
    return plan, json_path, md_path
