"""
Tomorrow Checklist — 장 시작 전 시스템 상태 점검 + 텔레그램

BAT-M(08:00) 또는 수동 실행.
V2/LENS/COO/SHIELD/KILL_SWITCH 상태를 체크하고 텔레그램으로 보냅니다.

사용법:
  python scripts/tomorrow_checklist.py            # 체크 + 텔레그램
  python scripts/tomorrow_checklist.py --no-send  # 체크만 (콘솔 출력)
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"


def check_file_freshness(path: Path, max_age_hours: int = 24) -> dict:
    """파일 존재 여부 + 최신성 체크."""
    if not path.exists():
        return {"exists": False, "fresh": False, "age_hours": None, "path": str(path.name)}
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    age_hours = age.total_seconds() / 3600
    return {
        "exists": True,
        "fresh": age_hours < max_age_hours,
        "age_hours": round(age_hours, 1),
        "path": str(path.name),
    }


def run_checklist() -> dict:
    """전체 체크리스트 실행."""
    results = {}
    issues = []
    warnings = []

    # ── 1. V2 + LENS 설정 확인 ──
    try:
        import yaml
        with open(PROJECT_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        v2 = cfg.get("alpha_v2", {})
        v2_enabled = v2.get("enabled", False)
        lens_enabled = v2.get("lens_enabled", False)
        ea_enabled = cfg.get("execution_alpha", {}).get("enabled", False)
        results["v2_enabled"] = v2_enabled
        results["lens_enabled"] = lens_enabled
        results["execution_alpha"] = ea_enabled
        if not v2_enabled:
            warnings.append("V2 비활성 상태")
        if not lens_enabled:
            warnings.append("LENS 비활성 상태")
    except Exception as e:
        issues.append(f"settings.yaml 읽기 실패: {e}")

    # ── 2. KILL_SWITCH 상태 ──
    ks_path = DATA / "KILL_SWITCH"
    ks_exists = ks_path.exists()
    results["kill_switch"] = ks_exists
    if ks_exists:
        try:
            content = ks_path.read_text(encoding="utf-8").strip()
            results["kill_switch_reason"] = content[:80]
        except Exception:
            results["kill_switch_reason"] = "읽기 실패"
        warnings.append("KILL_SWITCH 활성 — 자동매매 차단됨")

    # ── 3. COO 최근 실행 로그 ──
    coo_log = DATA / "coo_run_log.json"
    coo_check = check_file_freshness(coo_log, max_age_hours=20)
    results["coo_log"] = coo_check
    if coo_check["exists"]:
        try:
            with open(coo_log, "r", encoding="utf-8") as f:
                coo = json.load(f)
            summary = coo.get("summary", {})
            results["coo_success_rate"] = summary.get("success_rate", 0)
            results["coo_failed_steps"] = summary.get("failed_steps", 0)
            results["coo_elapsed_min"] = summary.get("elapsed_minutes", 0)
            results["coo_dry_run"] = coo.get("dry_run", True)
            if summary.get("failed_steps", 0) > 0:
                warnings.append(f"COO 실패 {summary['failed_steps']}건")
            if coo.get("dry_run", True):
                warnings.append("COO 아직 dry-run만 실행됨")
        except Exception as e:
            issues.append(f"COO 로그 파싱 실패: {e}")
    else:
        issues.append("COO 로그 없음 — 한 번도 실행되지 않음")

    # ── 4. SHIELD 레벨 ──
    shield_path = DATA / "shield_report.json"
    shield_check = check_file_freshness(shield_path, max_age_hours=20)
    results["shield"] = shield_check
    if shield_check["exists"]:
        try:
            with open(shield_path, "r", encoding="utf-8") as f:
                shield = json.load(f)
            results["shield_level"] = shield.get("overall_level", "UNKNOWN")
            if shield.get("overall_level") in ("RED", "ORANGE"):
                warnings.append(f"SHIELD {shield['overall_level']} — 방어 모드")
        except Exception as e:
            issues.append(f"SHIELD 파싱 실패: {e}")
    else:
        issues.append("SHIELD 리포트 없음")

    # ── 5. BRAIN 배분 ──
    brain_path = DATA / "brain_decision.json"
    brain_check = check_file_freshness(brain_path, max_age_hours=20)
    results["brain"] = brain_check
    if brain_check["exists"]:
        try:
            with open(brain_path, "r", encoding="utf-8") as f:
                brain = json.load(f)
            results["regime"] = brain.get("effective_regime", "UNKNOWN")
            results["cash_pct"] = brain.get("cash_pct", 0)
            results["invest_pct"] = brain.get("total_invest_pct", 0)
        except Exception as e:
            issues.append(f"BRAIN 파싱 실패: {e}")
    else:
        issues.append("BRAIN 결정 없음")

    # ── 6. US Overnight Signal ──
    us_path = DATA / "us_market" / "overnight_signal.json"
    us_check = check_file_freshness(us_path, max_age_hours=20)
    results["us_overnight"] = us_check
    if us_check["exists"]:
        try:
            with open(us_path, "r", encoding="utf-8") as f:
                us = json.load(f)
            results["us_date"] = us.get("us_close_date", "?")
        except Exception:
            pass
    else:
        warnings.append("US 오버나이트 시그널 없음")

    # ── 7. Paper Portfolio 상태 ──
    paper_path = DATA / "paper_portfolio.json"
    paper_check = check_file_freshness(paper_path, max_age_hours=48)
    results["paper_portfolio"] = paper_check
    if paper_check["exists"]:
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                paper = json.load(f)
            positions = paper.get("positions", {})
            stats = paper.get("stats", {})
            results["paper_positions"] = len(positions)
            results["paper_total_trades"] = stats.get("total_trades", 0)
            results["paper_mdd"] = stats.get("mdd", 0)
            results["paper_wins"] = stats.get("wins", 0)
            results["paper_losses"] = stats.get("losses", 0)
        except Exception as e:
            issues.append(f"Paper portfolio 파싱 실패: {e}")

    # ── 8. 핵심 데이터 파일 freshness ──
    critical_files = [
        DATA / "processed_indicators",  # directory
        DATA / "brain_decision.json",
        DATA / "shield_report.json",
        DATA / "regime_macro_signal.json",
    ]
    stale = []
    for f in critical_files:
        if f.is_dir():
            continue
        c = check_file_freshness(f, max_age_hours=20)
        if c["exists"] and not c["fresh"]:
            stale.append(f"{f.name} ({c['age_hours']}h)")
    if stale:
        warnings.append(f"오래된 데이터: {', '.join(stale)}")

    results["issues"] = issues
    results["warnings"] = warnings
    results["check_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return results


def format_report(r: dict) -> str:
    """텔레그램 메시지 포맷."""
    lines = []
    lines.append("=== MORNING CHECKLIST ===")
    lines.append(f"시간: {r.get('check_time', '?')}")
    lines.append("")

    # 시스템 상태
    v2 = "ON" if r.get("v2_enabled") else "OFF"
    lens = "ON" if r.get("lens_enabled") else "OFF"
    ks = "ON (차단)" if r.get("kill_switch") else "OFF (해제)"
    lines.append(f"V2: {v2} | LENS: {lens}")
    lines.append(f"KILL_SWITCH: {ks}")

    # 레짐 + SHIELD
    regime = r.get("regime", "?")
    shield = r.get("shield_level", "?")
    cash = r.get("cash_pct", 0)
    lines.append(f"레짐: {regime} | SHIELD: {shield} | 현금: {cash}%")
    lines.append("")

    # COO 상태
    coo = r.get("coo_log", {})
    if coo.get("exists"):
        dr = "DRY" if r.get("coo_dry_run") else "LIVE"
        rate = r.get("coo_success_rate", 0)
        failed = r.get("coo_failed_steps", 0)
        elapsed = r.get("coo_elapsed_min", 0)
        lines.append(f"COO: {dr} | 성공률 {rate}% | 실패 {failed}건 | {elapsed}분")
    else:
        lines.append("COO: 미실행")

    # Paper Trading
    if r.get("paper_portfolio", {}).get("exists"):
        pos = r.get("paper_positions", 0)
        trades = r.get("paper_total_trades", 0)
        mdd = r.get("paper_mdd", 0)
        wins = r.get("paper_wins", 0)
        losses = r.get("paper_losses", 0)
        wr = f"{wins/(wins+losses)*100:.0f}%" if (wins + losses) > 0 else "N/A"
        lines.append(f"Paper: {pos}종목 보유 | {trades}거래 | 승률 {wr} | MDD {mdd}%")
    lines.append("")

    # KILL_SWITCH 3조건
    lines.append("--- KILL_SWITCH 해제 3조건 ---")
    # 1) 승률
    if r.get("paper_total_trades", 0) > 0:
        wins = r.get("paper_wins", 0)
        losses = r.get("paper_losses", 0)
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        check1 = "PASS" if wr >= 55 else "FAIL"
        lines.append(f"1. 승률 55%+: {wr:.1f}% [{check1}]")
    else:
        lines.append("1. 승률 55%+: 데이터 없음 [WAIT]")

    # 2) MDD
    mdd = abs(r.get("paper_mdd", 0))
    if r.get("paper_total_trades", 0) > 0:
        check2 = "PASS" if mdd <= 8 else "FAIL"
        lines.append(f"2. MDD -8%: -{mdd}% [{check2}]")
    else:
        lines.append(f"2. MDD -8%: 백테스트 -8.24% [NEAR]")

    # 3) COO 10회
    lines.append(f"3. COO 10회+: {'미실행' if not coo.get('exists') else '실행 중'} [WAIT]")
    lines.append("")

    # 이슈/경고
    if r.get("issues"):
        lines.append("!!! ISSUES !!!")
        for i in r["issues"]:
            lines.append(f"  - {i}")
    if r.get("warnings"):
        lines.append("WARNINGS:")
        for w in r["warnings"]:
            lines.append(f"  - {w}")

    if not r.get("issues") and not r.get("warnings"):
        lines.append("ALL GREEN")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Morning Checklist")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미발송")
    args = parser.parse_args()

    results = run_checklist()
    report = format_report(results)

    print(report)

    # JSON 저장
    out_path = DATA / "morning_checklist.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")

    if not args.no_send:
        try:
            from src.telegram_sender import send_message
            send_message(report)
            print("텔레그램 전송 완료")
        except Exception as e:
            print(f"텔레그램 전송 실패: {e}")


if __name__ == "__main__":
    main()
