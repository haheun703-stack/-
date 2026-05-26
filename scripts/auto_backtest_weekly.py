"""주간 자동 백테스트 — 옵션 C (5/27 신규).

배경: 매주 토요일 09:00 cron 자동 실행.
1. 지난 주 실측 데이터 (decision_logger + signal_snapshot) 분석
2. D+0 vs D+1 백테스트 재실행 (4시나리오)
3. 잠정 임계값 9개 보정 권장안 도출
4. 텔레그램 보고 + Markdown 저장

실행:
  PYTHONPATH=. ./venv/bin/python3.11 scripts/auto_backtest_weekly.py

cron: 0 9 * * 6  (매주 토 09:00)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DECISION_LOG = PROJECT_ROOT / "data" / "decision_logs.json"


def analyze_decision_logs(days: int = 7) -> dict:
    """decision_logger.json 1주간 분석.

    Returns:
        {"n_buys": int, "n_sells": int, "avg_pnl_pct": float, "win_rate": float,
         "trigger_counts": {mvp: count}, "stop_loss_hits": int}
    """
    if not DECISION_LOG.exists():
        return {"error": "decision_logs.json 없음"}

    try:
        with DECISION_LOG.open(encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        return {"error": f"파싱 실패: {e}"}

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    recent = [l for l in logs if l.get("timestamp", "") >= cutoff]

    buys = [l for l in recent if l.get("action") == "BUY"]
    sells = [l for l in recent if l.get("action") == "SELL"]

    # MVP별 카운트
    trigger_counts = {}
    for l in recent:
        mvp = l.get("extra", {}).get("mvp", "unknown")
        trigger_counts[mvp] = trigger_counts.get(mvp, 0) + 1

    # 손익 (매도 시 pnl_pct)
    pnls = []
    stop_loss_hits = 0
    for s in sells:
        extra = s.get("extra", {})
        pnl = extra.get("pnl_pct") or extra.get("loss_pct") or 0
        pnls.append(pnl)
        if extra.get("type") in ("STOP_LOSS", "MA_FULL_BEAR"):
            stop_loss_hits += 1

    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100 if pnls else 0

    return {
        "n_buys": len(buys),
        "n_sells": len(sells),
        "avg_pnl_pct": avg_pnl,
        "win_rate": win_rate,
        "trigger_counts": trigger_counts,
        "stop_loss_hits": stop_loss_hits,
        "days": days,
    }


def run_d0_vs_d1_backtest() -> str:
    """D+0/D+1 백테스트 4시나리오 실행."""
    scenarios = [
        ("전체", []),
        ("강력 포착", ["--grade", "강력 포착"]),
        ("C2 필터", ["--c2"]),
        ("강력+C2", ["--grade", "강력 포착", "--c2"]),
    ]
    output_lines = ["## D+0 vs D+1 백테스트 4시나리오\n"]
    for name, args in scenarios:
        try:
            result = subprocess.run(
                [sys.executable, "scripts/backtest/backtest_d0_vs_d1_entry.py",
                 "--lookback-days", "90"] + args,
                cwd=PROJECT_ROOT, capture_output=True, text=True,
                encoding="utf-8", timeout=180,
            )
            output = result.stdout
            # 핵심 라인만 추출
            for line in output.split("\n"):
                if "D+1 평균" in line or "트레이드 수" in line or "결론" in line or "승률" in line:
                    output_lines.append(f"  [{name}] {line.strip()}")
        except Exception as e:
            output_lines.append(f"  [{name}] 실패: {e}")
    return "\n".join(output_lines)


def derive_threshold_recommendations(decisions: dict) -> str:
    """실측 데이터 기반 임계값 보정 권장."""
    if "error" in decisions:
        return f"⚠️ 실측 데이터 없음: {decisions['error']}"

    avg_pnl = decisions.get("avg_pnl_pct", 0)
    win_rate = decisions.get("win_rate", 0)
    n_sells = decisions.get("n_sells", 0)
    stop_loss_hits = decisions.get("stop_loss_hits", 0)
    trigger_counts = decisions.get("trigger_counts", {})

    recs = []

    if n_sells == 0:
        recs.append("⚠️ 1주간 매도 0건 — 임계값 데이터 부족")
        return "\n".join(recs)

    # MVP-2.6 -5% 손절 발화 빈도
    if stop_loss_hits / max(n_sells, 1) > 0.3:
        recs.append(
            f"🟡 손절 발화 {stop_loss_hits}/{n_sells} ({stop_loss_hits/n_sells*100:.0f}%) — "
            f"진입 게이트 강화 또는 손절 -5% → -4%로 빠른 차단 검토"
        )

    # MVP-2.5 trailing
    if "2_5" in trigger_counts:
        recs.append(
            f"📊 MVP-2.5 trailing 발화 {trigger_counts.get('2_5', 0)}건 — "
            f"+10% 임계 유지/완화 검토"
        )

    # MVP-2.7 D+3 익절
    if "2_7" in trigger_counts:
        recs.append(
            f"⏰ MVP-2.7 D+3/D+5 발화 {trigger_counts.get('2_7', 0)}건 — "
            f"임계 D+3 +3% 적정 검토"
        )

    # 평균 손익
    if avg_pnl < 0:
        recs.append(f"🚨 평균 손익 {avg_pnl:+.2f}% < 0% — 시스템 점검 필수")
    elif avg_pnl < 1.0:
        recs.append(f"🟡 평균 손익 {avg_pnl:+.2f}% — 진입 게이트 강화 권장")
    else:
        recs.append(f"✅ 평균 손익 {avg_pnl:+.2f}% — 정상")

    # 승률
    if win_rate < 50:
        recs.append(f"🟡 승률 {win_rate:.1f}% < 50% — 시그널 필터 검토")
    else:
        recs.append(f"✅ 승률 {win_rate:.1f}% — 정상")

    return "\n".join(recs)


def send_telegram_report(decisions: dict, backtest_result: str, recs: str):
    """텔레그램 + Markdown 파일 저장."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.telegram_sender import send_message

        title = f"📊 [주간 자동 백테스트] {datetime.now():%Y-%m-%d}"
        summary = (
            f"{title}\n\n"
            f"## 지난 주 실측 (decision_logger)\n"
            f"  매수 {decisions.get('n_buys', 0)}건 / 매도 {decisions.get('n_sells', 0)}건\n"
            f"  평균 손익 {decisions.get('avg_pnl_pct', 0):+.2f}%\n"
            f"  승률 {decisions.get('win_rate', 0):.1f}%\n"
            f"  손절 {decisions.get('stop_loss_hits', 0)}건\n\n"
            f"## 임계값 권장\n{recs}"
        )
        send_message(summary)

        # 전체 보고서 (백테스트 포함)
        report_dir = PROJECT_ROOT / "logs" / "weekly_backtest"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"backtest_{datetime.now():%Y%m%d_%H%M}.md"
        full_report = (
            f"# {title}\n\n"
            f"## 지난 주 실측\n```json\n{json.dumps(decisions, ensure_ascii=False, indent=2)}\n```\n\n"
            f"{backtest_result}\n\n"
            f"## 임계값 권장\n{recs}"
        )
        report_file.write_text(full_report, encoding="utf-8")
        print(f"보고서 저장: {report_file}")
    except Exception as e:
        print(f"텔레그램 발송 실패: {e}")


def main():
    print(f"[주간 자동 백테스트] {datetime.now():%Y-%m-%d %H:%M:%S} 시작")

    decisions = analyze_decision_logs(days=7)
    print(f"실측: {decisions}")

    backtest_result = run_d0_vs_d1_backtest()
    print(backtest_result)

    recs = derive_threshold_recommendations(decisions)
    print(f"권장:\n{recs}")

    send_telegram_report(decisions, backtest_result, recs)


if __name__ == "__main__":
    main()
