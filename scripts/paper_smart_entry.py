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


def build_picks_from_candidate_log() -> tuple[list, dict]:
    """candidate_log의 진입 후보 → SmartEntry picks 포맷."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    logs = ledger.get("candidate_log", [])
    # 최신 기준일만 사용 (stale 후보 제외)
    if logs:
        latest = max((l.get("as_of_date", "") for l in logs), default="")
        logs = [l for l in logs if l.get("as_of_date") == latest]
    picks, seen = [], set()
    for log in logs:
        for c in log.get("candidates", []):
            if c.get("decision") != "진입" or c.get("ticker") in seen:
                continue
            seen.add(c["ticker"])
            rr = c.get("risk_reward", {})
            entry = int(rr.get("entry_price", 0) or 0)
            floor = c.get("floor_quality", {})
            picks.append({
                "ticker": c["ticker"],
                "name": c.get("name", c["ticker"]),
                "grade": VALID_GRADE,
                "close": entry,
                "stop_loss": int(rr.get("stop_loss_price", 0) or (entry * 0.92 if entry else 0)),
                "target_price": int(rr.get("target_reference_price", 0) or 0),
                "total_score": floor.get("floor_quality_score", 0),
                "paper_mode": True,
                "_floor_label": floor.get("label", ""),
                "_source": log.get("source", "candidate_log"),
            })
    return picks, ledger


def _candidate_feature_map(ledger: dict) -> dict:
    """candidate_log → ticker별 최신 feature(floor/market/supply) 매핑."""
    out = {}
    for log in ledger.get("candidate_log", []):
        for c in log.get("candidates", []):
            out[c.get("ticker")] = c
    return out


def record_entries(report: dict, ledger: dict) -> int:
    """SmartEntry 진입(buy) → paper_trades PAPER_OPEN + 4중결합 feature + 장중 기록 (실주문 0).

    사장님 지정 12개 기록 항목: floor_quality / market_beta_20d / relative_return_5d·20d /
    drop_context / supply_state / supply_confirmation_score / smart_entry_trigger_time /
    trigger_reason / entry_price / vwap_at_entry / intraday_low_before_entry /
    market_index_state_at_entry.
    """
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
            "status": "PAPER_OPEN",
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
        ledger.setdefault("paper_trades", []).append(trade)
        opened += 1
    if opened:
        ledger["_last_smart_entry_at"] = datetime.now().isoformat(timespec="seconds")
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return opened


def main() -> int:
    ap = argparse.ArgumentParser(description="페이퍼 레이어2 장중 진입 (SmartEntry)")
    ap.add_argument("--analysis", action="store_true", help="장외 분석만 (run_analysis_only)")
    ap.add_argument("--picks-only", action="store_true", help="picks 생성만 (KIS 미접촉)")
    args = ap.parse_args()

    picks, ledger = build_picks_from_candidate_log()
    SMART_PICKS.write_text(
        json.dumps({"picks": picks, "_generated_from": "candidate_log 진입후보",
                    "_safety": {"real_order": False, "dry_run": True, "kis": "read-only"}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[레이어1→2] 일봉 진입후보 {len(picks)}개 → SmartEntry picks: {SMART_PICKS.name}")
    for p in picks:
        print(f"  {p['ticker']} {p['name']:14} | floor={p['_floor_label']}({p['total_score']:+d}) "
              f"| 진입참조 {p['close']:,} | 손절 {p['stop_loss']:,}")

    if not picks:
        print("진입 후보 없음 — 종료")
        return 0
    if args.picks_only:
        print("[picks-only] KIS 미접촉, picks 생성 완료.")
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

    opened = record_entries(report, ledger)
    print(f"[ledger] PAPER_OPEN {opened}건 기록 (실주문 0 / KIS read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
