"""페이퍼 트레이닝 레이어2 — 장중 진입 (SmartEntry: VWAP/호가/체결강도/5분봉).

페이퍼 2레이어 결합:
  레이어1 (일봉, 장마감): candidate_log 진입후보 + floor_quality   ← paper_track.py
  레이어2 (장중, 실시간): SmartEntry로 VWAP/호가/체결강도 → 실제 진입 타이밍  ← 본 스크립트

후보풀(일봉)이 "무엇을 살까"라면, SmartEntry(장중)는 "언제 살까"다.
폭락한 반도체 소부장 바닥도 일봉이 아니라 장중 VWAP 회복·체결강도 전환으로 잡는다.

안전: 실주문 0 / KIS read-only(호가·체결강도 조회만) / dry_run=True / executor_bot=quant.
사용:
  python -u -X utf8 scripts/paper_smart_entry.py --analysis   # 장외: picks 생성 + 엔진 분석
  python -u -X utf8 scripts/paper_smart_entry.py              # 장중 풀세션 (6/8 09:00~, VPS)
  python -u -X utf8 scripts/paper_smart_entry.py --picks-only # picks 생성만 (KIS 미접촉)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "data" / "paper_ledger.json"
SMART_PICKS = ROOT / "data" / "paper_smart_picks.json"

VALID_GRADE = "포착"  # SmartEntry.load_picks valid_grades 통과용


def classify_tier(c: dict) -> str:
    """C-rule 3-tier 분류 (paper rule, ★약세장 미검증 — 관찰 전용).

    CORE  : 횡보 바닥다지기 + stock_specific 아님 + 외인/기관 단독 accumulation (최우선 관찰)
    WATCH : 횡보 + stock_specific 아님 (수급 neutral/dual_buying/약한단독 허용) — 관찰, CORE와 분리
    CONTROL: 차트 4조건은 통과했으나 C-rule 탈락 — 진입X, missed-winner 비교군

    OOS 2구간(in/out-of-sample 강세장)에서 CORE(가설C)가 base 압도. 약세장은 미검증.
    stock_specific_drop은 CORE/WATCH(진입 대상)에서 제외 — CONTROL 추적만 허용.
    """
    fq = c.get("floor_quality", {})
    mc = c.get("market_context", {})
    sc = c.get("supply_confirmation", {})
    sideways = fq.get("label") == "바닥다지기후보"               # 횡보 매집
    not_specific = mc.get("drop_context") != "stock_specific_drop"  # 개별급락 회피
    solo_strong = sc.get("supply_state") in ("foreign_accumulation", "institution_accumulation")
    if sideways and not_specific and solo_strong:
        return "CORE"
    if sideways and not_specific:
        return "WATCH"
    return "CONTROL"


def build_picks_from_candidate_log() -> tuple[list, list, dict]:
    """candidate_log 진입후보 → C-rule 3-tier 분류.

    반환: (picks=CORE+WATCH SmartEntry 관찰대상, control=CONTROL 비교군(진입X), ledger)
    """
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    logs = ledger.get("candidate_log", [])
    # 최신 기준일만 사용 (stale 후보 제외)
    if logs:
        latest = max((l.get("as_of_date", "") for l in logs), default="")
        logs = [l for l in logs if l.get("as_of_date") == latest]
    picks, control, seen = [], [], set()
    for log in logs:
        for c in log.get("candidates", []):
            if c.get("decision") != "진입" or c.get("ticker") in seen:
                continue
            seen.add(c["ticker"])
            tier = classify_tier(c)
            rr = c.get("risk_reward", {})
            entry = int(rr.get("entry_price", 0) or 0)
            floor = c.get("floor_quality", {})
            mc = c.get("market_context", {})
            sc = c.get("supply_confirmation", {})
            row = {
                "ticker": c["ticker"],
                "name": c.get("name", c["ticker"]),
                "grade": VALID_GRADE,
                "close": entry,
                "stop_loss": int(rr.get("stop_loss_price", 0) or (entry * 0.92 if entry else 0)),
                "target_price": int(rr.get("target_reference_price", 0) or 0),
                "total_score": floor.get("floor_quality_score", 0),
                "paper_mode": True,
                "_tier": tier,
                "_floor_label": floor.get("label", ""),
                "_drop_context": mc.get("drop_context"),
                "_supply_state": sc.get("supply_state"),
                "_source": log.get("source", "candidate_log"),
            }
            (control if tier == "CONTROL" else picks).append(row)
    picks.sort(key=lambda p: (p["_tier"] != "CORE", -(p["total_score"] or 0)))  # CORE 먼저
    return picks, control, ledger


def record_control_pool(control: list, ledger: dict) -> int:
    """CONTROL 비교군(진입X) → ledger shadow_control 기록. 6/12 missed-winner 비교용. 실주문 0."""
    if not control:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    pool = ledger.setdefault("shadow_control", [])
    existing = {(r.get("date"), r.get("ticker")) for r in pool}
    added = 0
    for c in control:
        if (today, c["ticker"]) in existing:
            continue
        pool.append({
            "date": today, "ticker": c["ticker"], "name": c["name"], "tier": "CONTROL",
            "reason_excluded": "C-rule 탈락 (차트 4조건은 통과)",
            "floor_label": c["_floor_label"], "drop_context": c["_drop_context"],
            "supply_state": c["_supply_state"], "ref_price": c["close"],
            "real_order": False, "tracking": [],
        })
        added += 1
    if added:
        ledger["_last_control_pool_at"] = datetime.now().isoformat(timespec="seconds")
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def _candidate_feature_map(ledger: dict) -> dict:
    """candidate_log → ticker별 최신 feature(floor/market/supply) 매핑."""
    out = {}
    for log in ledger.get("candidate_log", []):
        for c in log.get("candidates", []):
            out[c.get("ticker")] = c
    return out


def record_entries(report: dict, ledger: dict, paper_open: bool = False) -> int:
    """SmartEntry 진입(buy) → 4중결합 feature + 장중 기록 (실주문 0).

    ★6/8 기본 = SHADOW_OPEN(관찰)로 shadow_observations에 기록 (PAPER_OPEN 금지).
    --paper-open 명시 시에만 PAPER_OPEN으로 paper_trades 기록. tier(CORE/WATCH)도 같이 박는다.

    사장님 지정 12개 기록 항목: floor_quality / market_beta_20d / relative_return_5d·20d /
    drop_context / supply_state / supply_confirmation_score / smart_entry_trigger_time /
    trigger_reason / entry_price / vwap_at_entry / intraday_low_before_entry /
    market_index_state_at_entry.
    """
    status = "PAPER_OPEN" if paper_open else "SHADOW_OPEN"
    key = "paper_trades" if paper_open else "shadow_observations"
    cand_map = _candidate_feature_map(ledger)
    opened = 0
    for d in report.get("details", []):
        if d.get("decision") != "buy":
            continue
        c = cand_map.get(d["ticker"], {})
        fq = c.get("floor_quality", {})
        mc = c.get("market_context", {})
        sc = c.get("supply_confirmation", {})
        trade = {
            "id": f"PAPER-SMART-{datetime.now().strftime('%Y%m%d')}-{d['ticker']}",
            "ticker": d["ticker"],
            "name": d.get("name", d["ticker"]),
            "status": status,
            "tier": classify_tier(c),
            "real_order": False,
            "entry_layer": "L2_smart_entry_intraday",
            # ── 레이어1 일봉 4중결합 feature ──
            "candidate_features": {
                "floor_quality_score": fq.get("floor_quality_score"),
                "floor_label": fq.get("label"),
                "market_beta_20d": mc.get("market_beta_20d"),
                "relative_return_5d": mc.get("relative_return_5d"),
                "relative_return_20d": mc.get("relative_return_20d"),
                "drop_context": mc.get("drop_context"),
                "supply_state": sc.get("supply_state"),
                "supply_confirmation_score": sc.get("supply_confirmation_score"),
            },
            # ── 레이어2 SmartEntry 장중 기록 (필드명 best-effort, 6/8 실행 시 확정) ──
            "smart_entry": {
                "smart_entry_trigger_time": d.get("trigger_time") or d.get("filled_at")
                or datetime.now().isoformat(timespec="seconds"),
                "trigger_reason": d.get("decision_reasons") or d.get("reason"),
                "entry_price": d.get("order_price") or d.get("entry_price"),
                "vwap_at_entry": d.get("vwap") or d.get("vwap_at_entry"),
                "intraday_low_before_entry": d.get("intraday_low") or d.get("low"),
                "market_index_state_at_entry": d.get("market_state") or d.get("market_index_state"),
                "gap_pct": d.get("gap_pct"),
                "gap_type": d.get("gap_type"),
            },
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "tracking": [],
        }
        ledger.setdefault(key, []).append(trade)
        opened += 1
    if opened:
        ledger["_last_smart_entry_at"] = datetime.now().isoformat(timespec="seconds")
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return opened


def main() -> int:
    ap = argparse.ArgumentParser(description="페이퍼 레이어2 장중 진입 (SmartEntry) — C-rule 3-tier 관찰")
    ap.add_argument("--analysis", action="store_true", help="장외 분석만 (run_analysis_only)")
    ap.add_argument("--picks-only", action="store_true", help="picks 생성만 (KIS 미접촉)")
    ap.add_argument("--paper-open", action="store_true",
                    help="PAPER_OPEN 실제 기록 (기본 OFF=SHADOW 관찰만). ★6/8 관찰일엔 사용 금지")
    args = ap.parse_args()

    picks, control, ledger = build_picks_from_candidate_log()
    n_core = sum(1 for p in picks if p["_tier"] == "CORE")
    n_watch = sum(1 for p in picks if p["_tier"] == "WATCH")
    SMART_PICKS.write_text(
        json.dumps({"picks": picks, "control": control,
                    "_tiers": {"CORE": n_core, "WATCH": n_watch, "CONTROL": len(control)},
                    "_rule": "C-rule 3-tier (★약세장 미검증, paper 관찰 전용)",
                    "_generated_from": "candidate_log 진입후보",
                    "_safety": {"real_order": False, "dry_run": True, "kis": "read-only",
                                "paper_open": bool(args.paper_open)}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[C-rule 3-tier · ★약세장 미검증] CORE {n_core} / WATCH {n_watch} / CONTROL {len(control)}")
    print("─ Tier1 CORE (최우선: 횡보+급락회피+단독수급) ─")
    for p in (x for x in picks if x["_tier"] == "CORE"):
        print(f"  ★ {p['ticker']} {p['name']:12} | floor={p['_floor_label']}({p['total_score']:+d}) "
              f"| 수급={p['_supply_state']} | 진입참조 {p['close']:,} | 손절 {p['stop_loss']:,}")
    print("─ Tier2 WATCH (관찰: 횡보+급락회피, 수급 완화) ─")
    for p in (x for x in picks if x["_tier"] == "WATCH"):
        print(f"    {p['ticker']} {p['name']:12} | floor={p['_floor_label']}({p['total_score']:+d}) "
              f"| 수급={p['_supply_state']} | 진입참조 {p['close']:,}")
    print("─ Tier3 CONTROL (비교군·진입X: 차트4통과·C-rule탈락 → missed-winner 추적) ─")
    for p in control:
        print(f"    {p['ticker']} {p['name']:12} | floor={p['_floor_label']} "
              f"| drop={p['_drop_context']} | 수급={p['_supply_state']}")

    # CONTROL 비교군 → ledger 기록 (진입 아님, 6/12 missed-winner 비교용)
    n_ctrl = record_control_pool(control, ledger)
    if n_ctrl:
        print(f"[shadow_control] CONTROL {n_ctrl}건 추적 기록 (진입X / 실주문 0)")

    if not picks:
        print("CORE/WATCH 관찰 대상 없음 — CONTROL만 추적. 종료")
        return 0
    if args.picks_only:
        print("[picks-only] KIS 미접촉, 3-tier picks 생성 완료.")
        return 0

    # ── SmartEntry (dry_run, read-only) ──
    import yaml
    config = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")) or {}
    try:
        from src.adapters.kis_intraday_adapter import KisIntradayAdapter
        intraday = KisIntradayAdapter()
    except Exception as e:  # noqa
        print(f"[KIS] intraday adapter 초기화 실패(장외 또는 IP 화이트리스트): {e}")
        print("→ 장중 VPS(13.209.153.221, KIS 화이트리스트)에서 실행 필요. picks 생성은 완료.")
        return 0

    from src.use_cases.smart_entry import SmartEntryEngine
    engine = SmartEntryEngine(
        intraday_adapter=intraday, order_adapter=None, dry_run=True,
        config=config, mode="paper", executor_bot="quant",
    )
    engine.load_picks(SMART_PICKS)
    report = engine.run_analysis_only() if args.analysis else engine.run_full_session()

    print(f"\n[SmartEntry 결과] 대상 {report.get('total_candidates', 0)} | "
          f"체결 {report.get('filled', 0)} | 스킵 {report.get('skipped', 0)}")
    for d in report.get("details", []):
        mark = {"buy": "O", "skip": "X", "wait": "-", "holding": "="}.get(d.get("decision"), "?")
        print(f"  [{mark}] {d.get('name')}({d.get('ticker')}) → {d.get('decision')} "
              f"@ {d.get('order_price', 0):,}")

    opened = record_entries(report, ledger, paper_open=args.paper_open)
    status = "PAPER_OPEN" if args.paper_open else "SHADOW_OPEN(관찰)"
    print(f"[ledger] {status} {opened}건 기록 (실주문 0 / KIS read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
