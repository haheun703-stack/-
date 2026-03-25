"""BAT-D (COO) 타이밍 분석 — 그룹별/단계별 소요시간 리포트

coo_run_log.json을 읽어서 병목 구간을 식별합니다.

사용법:
  python scripts/measure_batd_timing.py            # 최근 로그 분석
  python scripts/measure_batd_timing.py --send      # 텔레그램 발송
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "coo_run_log.json"


def analyze_timing() -> dict:
    """coo_run_log.json에서 타이밍 분석."""
    if not LOG_PATH.exists():
        return {"error": "coo_run_log.json 없음 — COO가 한 번도 실행되지 않았습니다"}

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        log = json.load(f)

    summary = log.get("summary", {})
    groups = log.get("groups", [])

    # 그룹별 소요시간
    group_timings = []
    all_steps = []

    for g in groups:
        elapsed = g.get("elapsed_seconds", 0)
        group_timings.append({
            "group": g["group"],
            "elapsed_sec": elapsed,
            "elapsed_min": round(elapsed / 60, 1),
            "steps": len(g.get("steps", [])),
            "failed": g.get("failed_count", 0),
            "success": g.get("success", False),
        })

        # 단계별 소요시간 (started_at ~ finished_at)
        for step in g.get("steps", []):
            started = step.get("started_at")
            finished = step.get("finished_at")
            if started and finished:
                try:
                    s = datetime.fromisoformat(started)
                    e = datetime.fromisoformat(finished)
                    step_elapsed = (e - s).total_seconds()
                except Exception:
                    step_elapsed = 0
            else:
                step_elapsed = 0

            all_steps.append({
                "group": g["group"],
                "name": step["name"],
                "elapsed_sec": round(step_elapsed, 1),
                "success": step.get("success", False),
                "error": step.get("error"),
            })

    # 정렬: 느린 순
    group_timings.sort(key=lambda x: x["elapsed_sec"], reverse=True)
    all_steps.sort(key=lambda x: x["elapsed_sec"], reverse=True)

    # 병목 TOP 10
    bottlenecks = all_steps[:10]

    return {
        "date": log.get("started_at", "")[:10],
        "total_elapsed_min": summary.get("elapsed_minutes", 0),
        "total_elapsed_sec": summary.get("elapsed_seconds", 0),
        "success_rate": summary.get("success_rate", 0),
        "dry_run": log.get("dry_run", False),
        "group_timings": group_timings,
        "bottleneck_steps": bottlenecks,
        "total_steps": summary.get("total_steps", 0),
        "failed_steps": summary.get("failed_steps", 0),
    }


def format_report(analysis: dict) -> str:
    """분석 결과 → 리포트 텍스트."""
    if "error" in analysis:
        return analysis["error"]

    lines = []
    mode = "DRY-RUN" if analysis["dry_run"] else "LIVE"
    lines.append(f"=== BAT-D 타이밍 분석 ({analysis['date']}, {mode}) ===")
    lines.append(f"총 소요: {analysis['total_elapsed_min']:.1f}분 ({analysis['total_elapsed_sec']:.0f}초)")
    lines.append(f"성공률: {analysis['success_rate']}% ({analysis['total_steps'] - analysis['failed_steps']}/{analysis['total_steps']})")
    lines.append("")

    # 그룹별
    lines.append("--- 그룹별 소요시간 (느린 순) ---")
    for g in analysis["group_timings"]:
        icon = "OK" if g["success"] else "FAIL"
        fail_str = f" (실패 {g['failed']}건)" if g["failed"] > 0 else ""
        lines.append(f"  [{icon}] {g['group']:<12} {g['elapsed_min']:>6.1f}분 ({g['steps']}단계{fail_str})")
    lines.append("")

    # 병목 TOP 10
    lines.append("--- 병목 TOP 10 (느린 단계) ---")
    for i, s in enumerate(analysis["bottleneck_steps"], 1):
        icon = "OK" if s["success"] else "FAIL"
        elapsed_min = s["elapsed_sec"] / 60
        lines.append(f"  {i:>2}. [{icon}] {s['name']:<24} {elapsed_min:>5.1f}분 ({s['elapsed_sec']:.0f}초)")

    # 개선 제안
    lines.append("")
    lines.append("--- 개선 포인트 ---")
    if analysis["bottleneck_steps"]:
        top = analysis["bottleneck_steps"][0]
        if top["elapsed_sec"] > 300:
            lines.append(f"  ! {top['name']}: {top['elapsed_sec']/60:.1f}분 — 최우선 최적화 대상")
    failed = [s for s in analysis["bottleneck_steps"] if not s["success"]]
    if failed:
        lines.append(f"  ! 실패 단계 {len(failed)}건: {', '.join(s['name'] for s in failed[:3])}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BAT-D 타이밍 분석")
    parser.add_argument("--send", action="store_true", help="텔레그램 발송")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()

    analysis = analyze_timing()

    if args.json:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return

    report = format_report(analysis)
    print(report)

    if args.send:
        try:
            from src.telegram_sender import send_message
            send_message(report)
            print("\n텔레그램 전송 완료")
        except Exception as e:
            print(f"\n텔레그램 전송 실패: {e}")


if __name__ == "__main__":
    main()
