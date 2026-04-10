"""CTO (Chief Technology Officer) — 시스템 기술 성과 관리자

3가지 핵심 기능:
  1. 시그널 성과 추적: 소스별 적중률, 시그널 감쇠 감지
  2. 팩터 기여도 분석: SD/M/V/Q 4축별 수익 기여 분해
  3. 데이터 건강 모니터: 스테일/결측 데이터 경보

산출물: data/cto_report.json
  → BRAIN (가중치 조정), Unified Scorer (팩터 조정), COO (건강 경보)

실행:
  python -u -X utf8 scripts/run_cto.py
  python -u -X utf8 scripts/run_cto.py --send
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CTO] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cto")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "cto_report.json"


# ═══════════════════════════════════════════════════════
#  1. 시그널 성과 추적
# ═══════════════════════════════════════════════════════

def analyze_signal_performance() -> dict:
    """picks_history.json + daily_performance.json → 소스별 적중률"""

    # picks_history 분석
    picks_path = DATA_DIR / "picks_history.json"
    if not picks_path.exists():
        return {"error": "picks_history.json 없음", "sources": {}}

    with open(picks_path, encoding="utf-8") as f:
        history = json.load(f)

    records = history.get("records", [])
    if not records:
        return {"error": "기록 없음", "sources": {}}

    # 소스별 집계
    source_stats: dict[str, dict] = {}
    recent_7d: dict[str, dict] = {}  # 최근 7일
    recent_30d: dict[str, dict] = {}  # 최근 30일

    today = datetime.now()
    d7 = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    d30 = (today - timedelta(days=30)).strftime("%Y-%m-%d")

    for r in records:
        sources = r.get("sources", [])
        if not sources:
            sources = [r.get("strategy", "unknown")]

        settled_return = r.get("settled_return") or r.get("latest_return") or r.get("day1_return")
        if settled_return is None:
            continue

        is_win = settled_return > 0
        pick_date = r.get("pick_date", "")

        for src in sources:
            src = src.strip()
            if not src:
                continue

            # 전체
            if src not in source_stats:
                source_stats[src] = {"wins": 0, "losses": 0, "total_return": 0.0, "returns": []}
            source_stats[src]["returns"].append(settled_return)
            if is_win:
                source_stats[src]["wins"] += 1
            else:
                source_stats[src]["losses"] += 1
            source_stats[src]["total_return"] += settled_return

            # 최근 7일
            if pick_date >= d7:
                if src not in recent_7d:
                    recent_7d[src] = {"wins": 0, "losses": 0, "returns": []}
                recent_7d[src]["returns"].append(settled_return)
                if is_win:
                    recent_7d[src]["wins"] += 1
                else:
                    recent_7d[src]["losses"] += 1

            # 최근 30일
            if pick_date >= d30:
                if src not in recent_30d:
                    recent_30d[src] = {"wins": 0, "losses": 0, "returns": []}
                recent_30d[src]["returns"].append(settled_return)
                if is_win:
                    recent_30d[src]["wins"] += 1
                else:
                    recent_30d[src]["losses"] += 1

    # 소스별 요약
    sources_summary = {}
    decay_alerts = []

    for src, stats in source_stats.items():
        total = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total if total > 0 else 0
        avg_return = stats["total_return"] / total if total > 0 else 0
        returns = stats["returns"]
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = abs(np.mean([r for r in returns if r <= 0])) if any(r <= 0 for r in returns) else 0

        # 최근 7일 성과
        r7 = recent_7d.get(src, {})
        r7_total = r7.get("wins", 0) + r7.get("losses", 0)
        r7_win_rate = r7.get("wins", 0) / r7_total if r7_total > 0 else None

        # 최근 30일 성과
        r30 = recent_30d.get(src, {})
        r30_total = r30.get("wins", 0) + r30.get("losses", 0)
        r30_win_rate = r30.get("wins", 0) / r30_total if r30_total > 0 else None

        # 시그널 감쇠 감지
        decay = False
        if r7_win_rate is not None and r30_win_rate is not None:
            if r7_win_rate < r30_win_rate - 0.15:
                decay = True
                decay_alerts.append({
                    "source": src,
                    "win_rate_30d": round(r30_win_rate, 3),
                    "win_rate_7d": round(r7_win_rate, 3),
                    "drop": round(r30_win_rate - r7_win_rate, 3),
                    "recommendation": "가중치 하향 또는 비활성화 권고",
                })

        sources_summary[src] = {
            "total_picks": total,
            "win_rate": round(win_rate, 3),
            "avg_return": round(avg_return, 2),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "win_rate_7d": round(r7_win_rate, 3) if r7_win_rate is not None else None,
            "win_rate_30d": round(r30_win_rate, 3) if r30_win_rate is not None else None,
            "signal_decay": decay,
            "picks_7d": r7_total,
            "picks_30d": r30_total,
        }

    return {
        "total_records": len(records),
        "sources": dict(sorted(sources_summary.items(), key=lambda x: x[1]["total_picks"], reverse=True)),
        "decay_alerts": decay_alerts,
    }


# ═══════════════════════════════════════════════════════
#  2. 팩터 기여도 분석
# ═══════════════════════════════════════════════════════

def analyze_factor_attribution() -> dict:
    """daily_performance.json → v3_picks 수익률 기반 팩터 기여 추정"""

    perf_path = DATA_DIR / "daily_performance.json"
    if not perf_path.exists():
        return {"error": "daily_performance.json 없음"}

    with open(perf_path, encoding="utf-8") as f:
        perf_data = json.load(f)

    if not isinstance(perf_data, list) or not perf_data:
        return {"error": "데이터 없음"}

    # 레짐별 성과 집계
    regime_perf: dict[str, list] = {}
    total_picks = 0
    total_wins = 0

    for day in perf_data:
        regime = day.get("regime_predicted", "unknown")
        regime_hit = day.get("regime_hit", False)
        picks = day.get("v3_picks_results", [])
        avg_pnl = day.get("v3_picks_avg_pnl", 0)

        if picks:
            total_picks += len(picks)
            total_wins += sum(1 for p in picks if (p.get("pnl_pct", 0) or 0) > 0)

            if regime not in regime_perf:
                regime_perf[regime] = []
            regime_perf[regime].append({
                "date": day.get("date", ""),
                "avg_pnl": avg_pnl,
                "picks_count": len(picks),
                "regime_hit": regime_hit,
            })

    # 레짐별 요약
    regime_summary = {}
    for regime, entries in regime_perf.items():
        pnls = [e["avg_pnl"] for e in entries if e["avg_pnl"] is not None]
        hits = sum(1 for e in entries if e["regime_hit"])
        regime_summary[regime] = {
            "days": len(entries),
            "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
            "regime_accuracy": round(hits / len(entries), 3) if entries else 0,
            "total_picks": sum(e["picks_count"] for e in entries),
        }

    overall_win_rate = total_wins / total_picks if total_picks > 0 else 0

    return {
        "total_days": len(perf_data),
        "total_picks": total_picks,
        "overall_win_rate": round(overall_win_rate, 3),
        "regime_performance": regime_summary,
    }


# ═══════════════════════════════════════════════════════
#  3. 데이터 건강 모니터
# ═══════════════════════════════════════════════════════

def check_data_health() -> dict:
    """핵심 데이터 파일 신선도 + 크기 점검"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    checks = [
        # (이름, 경로, 최대허용경과시간(시간), 최소크기(바이트))
        ("brain_decision", DATA_DIR / "brain_decision.json", 26, 500),
        ("shield_report", DATA_DIR / "shield_report.json", 26, 200),
        ("tomorrow_picks", DATA_DIR / "tomorrow_picks.json", 26, 100),
        ("picks_history", DATA_DIR / "picks_history.json", 48, 1000),
        ("paper_portfolio", DATA_DIR / "paper_portfolio.json", 48, 200),
        ("daily_performance", DATA_DIR / "daily_performance.json", 48, 500),
        ("market_news", DATA_DIR / "market_news.json", 26, 100),
        ("ai_brain_judgment", DATA_DIR / "ai_brain_judgment.json", 26, 500),
        ("ai_v3_picks", DATA_DIR / "ai_v3_picks.json", 26, 200),
        # coo_run_log 제거 — 레거시 COO 오케스트레이터 폐기, BAT-D 직접 실행으로 전환
        ("consensus_screening", DATA_DIR / "consensus_screening.json", 48, 1000),
        ("earnings_acceleration", DATA_DIR / "earnings_acceleration.json", 48, 200),
        ("cfo_report", DATA_DIR / "cfo_report.json", 48, 200),
        ("etf_rotation_result", DATA_DIR / "etf_rotation_result.json", 26, 200),
        ("sector_composite", DATA_DIR / "sector_rotation/sector_composite.json", 26, 200),
    ]

    results = []
    healthy = 0
    stale = 0
    missing = 0

    for name, path, max_hours, min_size in checks:
        if not path.exists():
            results.append({
                "name": name,
                "status": "MISSING",
                "detail": "파일 없음",
            })
            missing += 1
            continue

        stat = path.stat()
        age_hours = (today - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
        size = stat.st_size

        if age_hours > max_hours:
            results.append({
                "name": name,
                "status": "STALE",
                "detail": f"경과 {age_hours:.0f}시간 (허용 {max_hours}시간)",
                "age_hours": round(age_hours, 1),
                "size_bytes": size,
            })
            stale += 1
        elif size < min_size:
            results.append({
                "name": name,
                "status": "EMPTY",
                "detail": f"크기 {size}B < 최소 {min_size}B",
                "age_hours": round(age_hours, 1),
                "size_bytes": size,
            })
            stale += 1
        else:
            results.append({
                "name": name,
                "status": "OK",
                "age_hours": round(age_hours, 1),
                "size_bytes": size,
            })
            healthy += 1

    # COO 마지막 실행 확인
    coo_log_path = DATA_DIR / "coo_run_log.json"
    last_coo = None
    if coo_log_path.exists():
        try:
            with open(coo_log_path, encoding="utf-8") as f:
                coo_log = json.load(f)
            last_coo = coo_log.get("end_time") or coo_log.get("start_time")
        except Exception:
            pass

    # Parquet 유니버스 점검
    processed_dir = DATA_DIR / "processed"
    pq_count = len(list(processed_dir.glob("*.parquet"))) if processed_dir.exists() else 0

    return {
        "checked_at": today.isoformat(),
        "total_checks": len(checks),
        "healthy": healthy,
        "stale": stale,
        "missing": missing,
        "health_score": round(healthy / len(checks) * 100, 1) if checks else 0,
        "parquet_count": pq_count,
        "last_coo_run": last_coo,
        "details": results,
        "alerts": [r for r in results if r["status"] != "OK"],
    }


# ═══════════════════════════════════════════════════════
#  4. 가중치 조정 제안
# ═══════════════════════════════════════════════════════

def generate_weight_suggestions(
    signal_perf: dict, factor_attr: dict, data_health: dict
) -> list[dict]:
    """성과 + 건강 기반 가중치 조정 제안"""
    suggestions = []

    # 시그널 감쇠 경보 기반
    for alert in signal_perf.get("decay_alerts", []):
        suggestions.append({
            "type": "SIGNAL_DECAY",
            "target": alert["source"],
            "action": "가중치 하향",
            "reason": f"7일 적중률 {alert['win_rate_7d']:.0%} < 30일 {alert['win_rate_30d']:.0%} (차이 {alert['drop']:.0%})",
            "severity": "HIGH",
        })

    # 전체 적중률 낮은 소스
    for src, stats in signal_perf.get("sources", {}).items():
        if stats["total_picks"] >= 5 and stats["win_rate"] < 0.40:
            suggestions.append({
                "type": "LOW_WIN_RATE",
                "target": src,
                "action": "비활성화 검토",
                "reason": f"적중률 {stats['win_rate']:.0%} ({stats['total_picks']}건) — 40% 미만",
                "severity": "MEDIUM",
            })

    # 데이터 건강 문제
    if data_health.get("stale", 0) + data_health.get("missing", 0) >= 3:
        suggestions.append({
            "type": "DATA_HEALTH",
            "target": "파이프라인",
            "action": "COO 점검 필요",
            "reason": f"스테일/누락 {data_health['stale']+data_health['missing']}건",
            "severity": "HIGH",
        })

    # 레짐 예측 정확도
    regime_perf = factor_attr.get("regime_performance", {})
    for regime, stats in regime_perf.items():
        if stats["days"] >= 3 and stats["regime_accuracy"] < 0.40:
            suggestions.append({
                "type": "REGIME_ACCURACY",
                "target": f"레짐 '{regime}'",
                "action": "레짐 판단 모델 재검토",
                "reason": f"정확도 {stats['regime_accuracy']:.0%} ({stats['days']}일)",
                "severity": "MEDIUM",
            })

    return suggestions


# ═══════════════════════════════════════════════════════
#  통합 실행
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CTO 시스템 기술 성과 리포트")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CTO 시스템 성과 리포트")
    print(f"{'='*60}")

    # 1. 시그널 성과
    print("\n[1/4] 시그널 성과 분석...")
    signal_perf = analyze_signal_performance()

    # 2. 팩터 기여도
    print("[2/4] 팩터 기여도 분석...")
    factor_attr = analyze_factor_attribution()

    # 3. 데이터 건강
    print("[3/4] 데이터 건강 점검...")
    data_health = check_data_health()

    # 4. 가중치 제안
    print("[4/4] 가중치 조정 제안 생성...")
    suggestions = generate_weight_suggestions(signal_perf, factor_attr, data_health)

    # 통합 리포트
    report = {
        "generated_at": datetime.now().isoformat(),
        "signal_performance": signal_perf,
        "factor_attribution": factor_attr,
        "data_health": data_health,
        "weight_suggestions": suggestions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    _print_report(report)

    # 텔레그램
    if args.send:
        try:
            from src.adapters.telegram_adapter import send_message
            msg = _format_telegram(report)
            send_message(msg)
            print("\n[텔레그램] 발송 완료")
        except Exception as e:
            print(f"\n[텔레그램] 발송 실패: {e}")

    print(f"\n[저장] {OUTPUT_PATH}")
    return report


def _print_report(report: dict):
    """콘솔 리포트"""
    sp = report["signal_performance"]
    fa = report["factor_attribution"]
    dh = report["data_health"]
    ws = report["weight_suggestions"]

    # 시그널 성과
    print(f"\n[시그널 성과] 총 {sp.get('total_records', 0)}건 추적")
    sources = sp.get("sources", {})
    for src, stats in list(sources.items())[:10]:
        decay = " 🔻감쇠" if stats.get("signal_decay") else ""
        w7 = f"7d:{stats['win_rate_7d']:.0%}" if stats['win_rate_7d'] is not None else ""
        print(f"  {src:16s} 적중 {stats['win_rate']:.0%} ({stats['total_picks']:3d}건) "
              f"평균 {stats['avg_return']:+.1f}% {w7}{decay}")

    if sp.get("decay_alerts"):
        print(f"\n  ⚠ 감쇠 경보: {len(sp['decay_alerts'])}건")
        for a in sp["decay_alerts"]:
            print(f"    {a['source']} — 30d {a['win_rate_30d']:.0%} → 7d {a['win_rate_7d']:.0%}")

    # 팩터 기여도
    print(f"\n[팩터 분석] 총 {fa.get('total_days', 0)}일, {fa.get('total_picks', 0)}건")
    print(f"  전체 적중률: {fa.get('overall_win_rate', 0):.0%}")
    for regime, stats in fa.get("regime_performance", {}).items():
        print(f"  {regime:8s} {stats['days']:3d}일 평균PnL {stats['avg_pnl']:+.1f}% "
              f"레짐정확 {stats['regime_accuracy']:.0%}")

    # 데이터 건강
    print(f"\n[데이터 건강] 점수: {dh.get('health_score', 0):.0f}/100")
    print(f"  정상: {dh['healthy']} | 스테일: {dh['stale']} | 누락: {dh['missing']}")
    print(f"  Parquet: {dh['parquet_count']}종목 | 마지막 COO: {dh.get('last_coo_run', 'N/A')}")
    for alert in dh.get("alerts", [])[:5]:
        print(f"  ⚠ {alert['name']}: {alert['status']} — {alert.get('detail', '')}")

    # 가중치 제안
    if ws:
        print(f"\n[가중치 제안] {len(ws)}건")
        for s in ws:
            icon = "🔴" if s["severity"] == "HIGH" else "🟡"
            print(f"  {icon} [{s['type']}] {s['target']} → {s['action']}")
            print(f"     {s['reason']}")


def _format_telegram(report: dict) -> str:
    """텔레그램 메시지"""
    dh = report["data_health"]
    sp = report["signal_performance"]
    ws = report["weight_suggestions"]

    lines = [f"🔧 CTO 리포트 — 데이터 {dh['health_score']:.0f}/100"]

    # 시그널 요약
    sources = sp.get("sources", {})
    top3 = list(sources.items())[:3]
    if top3:
        lines.append("\n📊 시그널 성과 TOP 3:")
        for src, stats in top3:
            lines.append(f"  {src}: {stats['win_rate']:.0%} ({stats['total_picks']}건)")

    # 감쇠
    if sp.get("decay_alerts"):
        lines.append(f"\n🔻 감쇠 경보 {len(sp['decay_alerts'])}건")
        for a in sp["decay_alerts"][:3]:
            lines.append(f"  {a['source']}: {a['win_rate_30d']:.0%}→{a['win_rate_7d']:.0%}")

    # 건강 문제
    alerts = dh.get("alerts", [])
    if alerts:
        lines.append(f"\n⚠ 데이터 이상 {len(alerts)}건")
        for a in alerts[:3]:
            lines.append(f"  {a['name']}: {a['status']}")

    # 제안
    high_ws = [s for s in ws if s["severity"] == "HIGH"]
    if high_ws:
        lines.append(f"\n🔴 긴급 제안 {len(high_ws)}건")
        for s in high_ws[:3]:
            lines.append(f"  {s['target']}: {s['action']}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
