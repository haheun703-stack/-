"""FLOWX Market OS v1 show_me_report (8단계 — 마지막).

FLOWX 1~7단계 산출을 사장님이 눈으로 판단할 수 있게 그림·표로 보여준다.
판단을 대신하지 않는다 — 숫자를 그림으로 바꿔줄 뿐이다.

★SHOW ME는 정책 변경권이 0이다. 그림이 좋아 보여도 PAPER_OPEN·SAJANG·scheduler를
열지 않는다. 매수·매도·주문·계좌 실행 경로를 일절 import·호출하지 않는다.
후보선정(as_of 종가)과 실행(virtual_entry_price) 기준을 리포트에서도 섞지 않는다.

설계: docs/02-design/flowx_market_os_v1.md §1·§5, 진행 지시서 8단계.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.use_cases.daily_review import run_daily_review
from src.use_cases.regime_router_v1 import run_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "data_store" / "reports"

SHOW_ME_VERSION = "show_me_report_v1"


def _c60_panel(route_doc: dict) -> list[dict]:
    """1. C60 국면 패널: 488080/삼성/SK 종가·MA60·국면·히스테리시스."""
    panel = []
    for ticker, route in (route_doc.get("routes") or {}).items():
        if not route.get("data_available"):
            panel.append({"ticker": ticker, "data_available": False})
            continue
        panel.append({
            "ticker": ticker,
            "name": route.get("name"),
            "close": route.get("current_close"),
            "ma60": route.get("current_ma60"),
            "close_vs_ma60_pct": route.get("close_vs_ma60_pct"),
            "c60_regime_raw": route.get("c60_regime_raw"),
            "effective_regime": route.get("effective_regime"),
            "hard_gate_regime": route.get("hard_gate_regime"),
            "in_hysteresis_window": route.get("in_hysteresis_window"),
            "data_available": True,
        })
    return panel


def build_show_me_document(review_doc: dict, route_doc: dict) -> dict:
    """7단계 daily_review + 라우터 C60 → SHOW ME 통합 문서. 순수 함수."""
    cp = review_doc.get("candidate_performance", {}) or {}
    ep = review_doc.get("execution_performance", {}) or {}
    ex = review_doc.get("exit_observer_summary", {}) or {}

    return {
        "version": SHOW_ME_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_date": review_doc.get("observation_date"),
        "market_regime": review_doc.get("market_regime"),
        "one_line": review_doc.get("one_line"),
        # 1. 시장 국면
        "regime_show_me": _c60_panel(route_doc),
        # 2. 후보 흐름
        "candidate_flow": {
            "candidate_count": cp.get("candidate_count", 0),
            "tier_counts": cp.get("tier_counts", {"CORE": 0, "WATCH": 0, "CONTROL": 0}),
        },
        # 3. 후보선정 성능 (basis=as_of_close)
        "candidate_performance": cp,
        # 4. 실행 성능 (basis=virtual_entry_price)
        "execution_performance": ep,
        # 5. exit observer
        "exit_observer_summary": ex,
        # 6. missed/false
        "missed_false": {
            "missed_winner": cp.get("missed_winner", []),
            "false_positive": cp.get("false_positive", []),
        },
        "data_warnings": review_doc.get("data_warnings", []),
        # 7. 안전선 패널
        "safety_panel": {
            "real_order": False,
            "scheduler_changed": False,
            "sajang_changed": False,
            "paper_open_allowed": False,
            "sell_automation": "BLOCKED",
            "policy_changed": False,
            "order_symbol_grep": 0,
            "note": "SHOW ME는 정책 변경권 0. 그림이 좋아도 PAPER_OPEN/SAJANG/scheduler 안 연다.",
        },
    }


def _perf_table(rows: list[dict], price_key: str) -> list[str]:
    out = [
        f"| 종목 | tier | {price_key} | D+1 | D+3 | D+5 | D+10 | MFE | MAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fwd = r.get("raw_fwd_pct") or r.get("pnl_pct") or {}
        base = r.get("as_of_close") if "as_of_close" in r else r.get("entry_price")
        out.append(
            f"| {r.get('name')}({r.get('ticker')}) | {r.get('tier')} | {base} | "
            f"{fwd.get('D+1')} | {fwd.get('D+3')} | {fwd.get('D+5')} | {fwd.get('D+10')} | "
            f"{r.get('mfe_pct')} | {r.get('mae_pct')} |"
        )
    return out


def build_show_me_markdown(doc: dict) -> str:
    cp = doc["candidate_performance"]
    ep = doc["execution_performance"]
    ex = doc["exit_observer_summary"]
    mf = doc["missed_false"]
    sp = doc["safety_panel"]

    md = [
        f"# FLOWX SHOW ME — {doc['observation_date']}",
        "",
        f"> {doc['one_line']}",
        "",
        "## 1. 오늘 한 줄 결론",
        f"- {doc['one_line']}",
        "",
        "## 2. 시장 국면 SHOW ME (C60)",
        "| 기초자산 | 종가 | MA60 | 이격% | raw | effective | 정책 | 히스테리시스 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in doc["regime_show_me"]:
        if not p.get("data_available"):
            md.append(f"| {p['ticker']} | DATA_UNAVAILABLE | | | | | | |")
            continue
        hy = "대기" if p.get("in_hysteresis_window") else "-"
        md.append(
            f"| {p.get('name')}({p['ticker']}) | {p.get('close')} | {p.get('ma60')} | "
            f"{p.get('close_vs_ma60_pct')} | {p.get('c60_regime_raw')} | {p.get('effective_regime')} | "
            f"{p.get('hard_gate_regime')} | {hy} |"
        )
    cf = doc["candidate_flow"]
    md += [
        "",
        "## 3. 후보 흐름 SHOW ME",
        f"- 후보 {cf['candidate_count']} · CORE {cf['tier_counts'].get('CORE')} / "
        f"WATCH {cf['tier_counts'].get('WATCH')} / CONTROL {cf['tier_counts'].get('CONTROL')}",
        "",
        "## 4. 후보선정 성능 SHOW ME (기준: as_of 종가)",
    ]
    md += _perf_table(cp.get("candidates", []), "as_of종가")
    md += [
        "",
        "## 5. SmartEntry 실행 성능 SHOW ME (기준: virtual_entry_price)",
    ]
    md += _perf_table(ep.get("entries", []), "진입가")
    md += [
        "",
        "## 6. Exit observer SHOW ME",
        f"- exit 트리거 집계: {ex.get('exit_type_trigger_counts')}",
        f"- {ex.get('note', '')}",
        "",
        "## 7. missed_winner / false_positive",
        "- missed_winner: " + (", ".join(f"{m['name']}({m['ticker']}) {m['d10_pct']}%" for m in mf["missed_winner"]) or "(없음)"),
        "- false_positive: " + (", ".join(f"{m['name']}({m['ticker']}) {m['d10_pct']}%" for m in mf["false_positive"]) or "(없음)"),
        "",
    ]
    if doc["data_warnings"]:
        md.append("## 8. 데이터 경고")
        md.append(f"- OHLCV 없음 {len(doc['data_warnings'])}건 (관찰 차단)")
        md.append("")
    md += [
        "## 9. 안전선 확인",
        f"- real_order={sp['real_order']} / sell_automation={sp['sell_automation']} / "
        f"paper_open_allowed={sp['paper_open_allowed']}",
        f"- scheduler_changed={sp['scheduler_changed']} / sajang_changed={sp['sajang_changed']} / "
        f"policy_changed={sp['policy_changed']} / order_symbol_grep={sp['order_symbol_grep']}",
        f"- {sp['note']}",
        "",
        "## 10. 다음 관측 포인트",
        "- 6/8 실관측 데이터 누적 → 6/12 그림·숫자 사후비교(어떤 exit 룰/tier가 좋았나).",
        "- SHOW ME는 관측만. 승격/PAPER_OPEN은 사장님 승인 별도.",
        "",
        "---",
        "**실주문 0 / 매도 자동화 BLOCKED / PAPER_OPEN 금지 / 정책·scheduler·SAJANG 변경 0 (관측 전용)**",
    ]
    return "\n".join(md) + "\n"


def render_charts(doc: dict, output_dir: Path = REPORT_DIR) -> list[Path]:
    """matplotlib best-effort. 없거나 실패해도 MD/JSON은 별개로 생성된다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    d = str(doc.get("observation_date") or datetime.now().strftime("%Y-%m-%d"))
    paths: list[Path] = []
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tc = doc["candidate_flow"]["tier_counts"] or {}
        axes[0].bar(list(tc.keys()), list(tc.values()), color=["#c0392b", "#e67e22", "#7f8c8d"])
        axes[0].set_title("Tier distribution (CORE/WATCH/CONTROL)")
        panel = [p for p in doc["regime_show_me"] if p.get("data_available")]
        names = [p.get("ticker") for p in panel]
        dist = [p.get("close_vs_ma60_pct") or 0 for p in panel]
        axes[1].bar(names, dist, color="#2980b9")
        axes[1].axhline(0, color="k", lw=0.6)
        axes[1].set_title("C60 close vs MA60 (%)")
        fig.suptitle(f"FLOWX SHOW ME — {d} (observation only, real_order=0)")
        fig.tight_layout()
        p = output_dir / f"show_me_{d}_overview.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    except Exception:
        pass
    return paths


def save_show_me(doc: dict, output_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = str(doc.get("observation_date") or datetime.now().strftime("%Y-%m-%d"))
    json_path = output_dir / f"show_me_{d}.json"
    md_path = output_dir / f"show_me_{d}.md"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_show_me_markdown(doc), encoding="utf-8")
    return json_path, md_path


def run_show_me(
    days: int = 1300, prefer_remote: bool = True, write: bool = True, render_png: bool = True
) -> tuple[dict, Path | None, Path | None, list[Path]]:
    """7단계 daily_review + 라우터 C60 → SHOW ME 리포트(MD/JSON/PNG). 관측 전용."""
    review_doc, _, _ = run_daily_review(days=days, prefer_remote=prefer_remote, write=False)
    route_doc, _ = run_router(days=days, prefer_remote=prefer_remote, write=False)
    doc = build_show_me_document(review_doc, route_doc)
    if not write:
        return doc, None, None, []
    json_path, md_path = save_show_me(doc)
    png_paths = render_charts(doc) if render_png else []
    return doc, json_path, md_path, png_paths
