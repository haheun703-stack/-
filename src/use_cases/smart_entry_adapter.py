"""FLOWX Market OS v1 smart_entry_adapter (5단계).

morning_plan_07의 CORE/WATCH 후보를 SmartEntry가 읽을 수 있는 payload(shadow entry
intent)로 **변환·줄세우기만** 한다. SmartEntry 엔진을 실행하지 않는다 — 실주문 0,
주문어댑터 None, dry_run. "SmartEntry 실행"이 아니라 "SmartEntry가 볼 수 있는 형태로
후보를 정렬"하는 단계다(사장님 5단계 지시).

경계: 기본 SHADOW_OPEN. PAPER_OPEN은 6/8 전 금지(run/CLI에서 강제 차단, build에만
테스트용 분기). 주문 실행엔진·장중 데이터 어댑터·주문 경로를 일절 import·호출하지
않는다(import는 morning_plan_07·engine_policy_map 상수뿐).

설계: docs/02-design/flowx_market_os_v1.md §1·§7, 진행 지시서 5단계.
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
)
from src.use_cases.morning_plan_07 import run_morning_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_ENTRY_DIR = PROJECT_ROOT / "data_store" / "shadow_entries"

ADAPTER_VERSION = "smart_entry_adapter_v1"
# SmartEntry load_picks valid_grades 통과용 관찰 라벨(실주문 아님).
SHADOW_GRADE = "포착"

STATUS_SHADOW_OPEN = "SHADOW_OPEN"
STATUS_PAPER_OPEN = "PAPER_OPEN"        # 6/8 전 금지. build 테스트 분기 전용.
STATUS_CONTROL_ONLY = "CONTROL_ONLY"
STATUS_BLOCKED_REGIME = "BLOCKED_BY_REGIME"
STATUS_BLOCKED_DATA = "BLOCKED_BY_DATA"


def _to_payload(row: dict, status: str) -> dict:
    """morning_plan tier row → SmartEntry load_picks 호환 payload + 5단계 메타.

    load_picks 요구 필드(ticker/name/grade/close/stop_loss/target_price/total_score/
    paper_mode)를 채우되 실주문 아님(real_order=False, paper_mode=True)을 못박는다.
    """
    return {
        # ── SmartEntry load_picks 호환 ──
        "ticker": row.get("ticker"),
        "name": row.get("name", row.get("ticker")),
        "grade": SHADOW_GRADE,
        "close": row.get("ref_close"),
        "stop_loss": row.get("stop_loss"),
        "target_price": row.get("target"),
        "total_score": 0,
        "paper_mode": True,
        # ── 5단계 메타 ──
        "tier": row.get("tier"),
        "status": status,
        "real_order": False,
        "observation": row.get("observation"),
        "floor_label": row.get("floor_label"),
        "drop_context": row.get("drop_context"),
        "supply_state": row.get("supply_state"),
        # 관측 레이어 라벨 pass-through(진입 조건에 일절 쓰지 않음 — 표시·기록용).
        "shadow_labels": row.get("shadow_labels"),
        "_source": "morning_plan_07",
    }


def build_shadow_entries(plan: dict, paper_open: bool = False) -> dict:
    """morning_plan → shadow entry intent. 순수 함수(파일/네트워크/주문 없음).

    R4 + CORE/WATCH → SHADOW_OPEN(paper_open이면 PAPER_OPEN, 테스트 전용).
    CONTROL → CONTROL_ONLY(진입 후보 제외). R1 → BLOCKED_BY_REGIME.
    DATA_UNAVAILABLE → BLOCKED_BY_DATA. 그 외 → 보수적 차단.
    """
    market = plan.get("market_regime")
    smart_mode = (plan.get("engines") or {}).get("smart_entry")
    tiers = plan.get("tiers") or {}
    open_status = STATUS_PAPER_OPEN if paper_open else STATUS_SHADOW_OPEN

    # CONTROL은 정책 무관 항상 비교군(SmartEntry 진입 후보 아님)
    control_only = [_to_payload(r, STATUS_CONTROL_ONLY) for r in tiers.get("CONTROL", [])]
    core_watch = list(tiers.get("CORE", [])) + list(tiers.get("WATCH", []))

    shadow_entries: list[dict] = []
    blocked: list[dict] = []

    if market == MARKET_DATA_UNAVAILABLE:
        blocked = [_to_payload(r, STATUS_BLOCKED_DATA) for r in core_watch]
    elif market == MARKET_R1:
        blocked = [_to_payload(r, STATUS_BLOCKED_REGIME) for r in core_watch]
    elif market == MARKET_R4 and smart_mode == MODE_ALLOWED_SHADOW:
        shadow_entries = [_to_payload(r, open_status) for r in core_watch]
    else:
        # 방어: 알 수 없는 국면/모드 → 보수적 차단
        blocked = [_to_payload(r, STATUS_BLOCKED_REGIME) for r in core_watch]

    return {
        "version": ADAPTER_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": plan.get("as_of_date"),
        "market_regime": market,
        "smart_entry_mode": smart_mode,
        "paper_open": bool(paper_open),
        "shadow_entries": shadow_entries,
        "control_only": control_only,
        "blocked": blocked,
        "counts": {
            "shadow_open": len(shadow_entries),
            "control_only": len(control_only),
            "blocked": len(blocked),
        },
        "safety": {
            "real_order": False,
            "order_adapter": "None",
            "dry_run": True,
            "scheduler_changed": False,
            "sajang_changed": False,
            "paper_open_default": False,
            "sell_automation": "BLOCKED",
        },
    }


def save_shadow_entries(document: dict, output_dir: Path = SHADOW_ENTRY_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(document.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))
    path = output_dir / f"shadow_entries_{as_of}.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_smart_entry_adapter(
    days: int = 1300, prefer_remote: bool = True, write: bool = True
) -> tuple[dict, Path | None]:
    """morning_plan → shadow entry intent. PAPER_OPEN은 여기서 강제 차단(6/8 전 금지).

    paper_open 인자를 노출하지 않음 = run/CLI 경로로는 PAPER_OPEN을 켤 수 없다.
    """
    plan, _, _ = run_morning_plan(days=days, prefer_remote=prefer_remote, write=write)
    document = build_shadow_entries(plan, paper_open=False)  # 항상 SHADOW_OPEN
    path = save_shadow_entries(document) if write else None
    return document, path
