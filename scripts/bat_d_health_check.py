"""BAT-D 자동 메트릭 수집 + 이상 감지 + 텔레그램 알림

목적:
- 매일 BAT-D 끝나면 자동으로 단계별 시간/에러/누락 분석
- 평소 평균 대비 이상 발생 시 즉시 텔레그램 (3단계: INFO/WARN/CRIT)
- 5/16 아침에 +17분 발견한 것을 5/15 18:40에 알 수 있게

흐름:
- cron 로그 파싱 → 단계 시간 + 에러 카운트
- data/metrics/bat_d_daily.jsonl 누적
- 최근 5거래일 평균 vs 오늘 비교
- 임계값 초과 시 텔레그램

실행 (BAT-D 끝부분에서 자동):
    python -X utf8 scripts/bat_d_health_check.py
"""

import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

LOG_DIR = PROJECT_ROOT / "logs"
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = METRICS_DIR / "bat_d_daily.jsonl"

# 단계 마커: (key, regex)
MARKERS = [
    ("bat_d_start", r"=== BAT-D 시작"),
    ("update_daily_start", r"CSV 병렬 업데이트 시작"),
    ("sync_done", r"동기화 완료: \[scalper\]"),
    ("scan_nationality_done", r"수집 완료: 20\d{6} / \d+종목 / \d+행 / OK"),
    ("foreign_exh_done", r"수집 완료: \d{3,}종목"),
    ("china_money_done", r"차이나머니 수급 감지 완료"),
    ("v3_brain_done", r"v3 AI Brain 러너 완료"),
    ("wisereport_done", r"컨센서스 스크리닝 결과"),
    ("killer_picks_done", r"킬러픽 저장 완료"),
    ("etf_signals_done", r"etf_theme_signals_\d{8}"),
    ("bat_d_done", r"=== BAT-D 완료"),
]

# 이상 감지 임계값 (5/15 +17% 미감지 → 더 민감하게 조정)
THRESHOLDS = {
    "bat_d_duration_pct": 1.15,         # 평균 +15% 이상 → WARN
    "bat_d_duration_pct_crit": 1.30,    # 평균 +30% 이상 → CRIT
    "bat_d_duration_abs_crit": 140,     # 절대값 140분 이상 → CRIT (history 부족 시 fallback)
    "kis_errors_critical": 5,            # 5건 이상
    "update_daily_errors_warn": 5,       # 5건 이상
}


def _to_sec(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def parse_log(date_str: str) -> dict | None:
    """cron 로그에서 단계 시각 + 에러 카운트 추출.

    Args:
        date_str: YYYYMMDD
    """
    log_path = LOG_DIR / f"cron_{date_str}.log"
    if not log_path.exists():
        return None

    content = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.split("\n")

    # 단계 시각 추출
    found = {}
    for line in lines:
        tm = re.search(r"(\d{2}:\d{2}:\d{2})", line)
        if not tm:
            continue
        ts = tm.group(1)
        for name, pattern in MARKERS:
            if name in found:
                continue
            if re.search(pattern, line):
                found[name] = ts

    metrics = {
        "date": date_str,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "stages": found,
    }

    # 단계별 소요 시간 (분)
    if "bat_d_start" in found and "bat_d_done" in found:
        metrics["bat_d_min"] = round(
            (_to_sec(found["bat_d_done"]) - _to_sec(found["bat_d_start"])) / 60, 1
        )

    # 에러 카운트
    metrics["kis_errors"] = (
        content.count("EGW00133")
        + content.count("401 Unauthorized")
        + content.count("토큰 발급 실패")
    )
    update_log = (PROJECT_ROOT / "stock_data_daily" / "_update_log.txt").read_text(encoding="utf-8", errors="ignore") if (PROJECT_ROOT / "stock_data_daily" / "_update_log.txt").exists() else ""
    # 오늘 errors=N 추출
    m = re.search(rf"target={date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}.*errors=(\d+)", update_log)
    metrics["update_daily_errors"] = int(m.group(1)) if m else None

    # FAIL_COUNT 추출 (BAT-D 완료 라인에서)
    m = re.search(r"=== BAT-D 완료 \(실패: (\d+)건\)", content)
    metrics["bat_d_fails"] = int(m.group(1)) if m else None

    return metrics


def load_history(days: int = 5) -> list[dict]:
    """최근 N일 메트릭 로드"""
    if not METRICS_FILE.exists():
        return []
    rows = []
    for line in METRICS_FILE.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-days:]


def save_metric(metric: dict):
    """JSONL 누적 저장"""
    with METRICS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric, ensure_ascii=False) + "\n")


def detect_anomalies(today: dict, history: list[dict]) -> list[tuple[str, str]]:
    """이상 감지 → (severity, message) 리스트"""
    alerts = []

    # 1. BAT-D 시간 — 평균 +15% (WARN), +30% (CRIT), 절대값 140분+ (CRIT, history 부족 시)
    durations = [h.get("bat_d_min") for h in history if h.get("bat_d_min")]
    if today.get("bat_d_min"):
        bat_min = today["bat_d_min"]
        # 절대값 임계 (history 부족 시 fallback)
        if bat_min >= THRESHOLDS["bat_d_duration_abs_crit"]:
            alerts.append(("CRIT", f"🔴 BAT-D 절대시간 위기: *{bat_min:.1f}분* (임계 {THRESHOLDS['bat_d_duration_abs_crit']}분+)"))
        # 평균 대비 비율
        if durations:
            avg = sum(durations) / len(durations)
            ratio = bat_min / avg
            if ratio >= THRESHOLDS["bat_d_duration_pct_crit"]:
                alerts.append(("CRIT", f"🔴 BAT-D 시간 *+{(ratio - 1) * 100:.0f}%*: 오늘 {bat_min:.1f}분 (평균 {avg:.1f}분)"))
            elif ratio >= THRESHOLDS["bat_d_duration_pct"]:
                alerts.append(("WARN", f"⚠️ BAT-D 시간 *+{(ratio - 1) * 100:.0f}%*: 오늘 {bat_min:.1f}분 (평균 {avg:.1f}분, 최근 {len(durations)}일)"))

    # 2. KIS 에러 5건+ → CRITICAL
    if today.get("kis_errors", 0) >= THRESHOLDS["kis_errors_critical"]:
        alerts.append((
            "CRIT",
            f"🔴 KIS API 에러 폭증: *{today['kis_errors']}건* (임계값 {THRESHOLDS['kis_errors_critical']})",
        ))

    # 3. update_daily 에러 5건+ → WARN (효성 fix 후 0건이 정상)
    if today.get("update_daily_errors", 0) and today["update_daily_errors"] >= THRESHOLDS["update_daily_errors_warn"]:
        alerts.append((
            "WARN",
            f"⚠️ update_daily 에러 *{today['update_daily_errors']}건* (효성 fix 후 0건 기대)",
        ))

    # 4. BAT-D 실패 1건+ → WARN
    if today.get("bat_d_fails", 0) and today["bat_d_fails"] > 0:
        alerts.append((
            "WARN",
            f"⚠️ BAT-D 실패 *{today['bat_d_fails']}건* (정상 0건)",
        ))

    return alerts


def send_telegram(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] TELEGRAM 환경변수 미설정")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[WARN] 텔레그램 에러: {e}")
        return False


def format_report(today: dict, alerts: list[tuple[str, str]]) -> str:
    """텔레그램 메시지 포맷"""
    dt = today["date"]
    bat_min = today.get("bat_d_min", "?")
    kis_err = today.get("kis_errors", 0)
    upd_err = today.get("update_daily_errors", "?")
    fails = today.get("bat_d_fails", "?")

    lines = [
        f"🤖 *BAT-D 자동 건강 점검 ({dt})*",
        "",
        f"⏱ 전체 시간: *{bat_min}분*",
        f"❌ BAT-D 실패: {fails}건 / KIS 에러: {kis_err}건 / update_daily 에러: {upd_err}건",
        "",
    ]
    if alerts:
        lines.append("🚨 *이상 감지*")
        for sev, msg in alerts:
            lines.append(f"• {msg}")
    else:
        lines.append("✅ *모든 지표 정상*")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD (기본: 오늘)")
    parser.add_argument("--no-save", action="store_true", help="JSONL 저장 안 함 (dry-run)")
    args = parser.parse_args()

    today_str = args.date or date.today().strftime("%Y%m%d")
    print(f"[bat_d_health] target={today_str}")

    today = parse_log(today_str)
    if not today:
        print(f"[ERROR] 로그 없음: cron_{today_str}.log")
        return

    if not args.no_save:
        save_metric(today)
        print(f"[save] {METRICS_FILE} (누적)")
    else:
        print(f"[dry-run] 저장 skip")

    history = load_history(days=5)
    alerts = detect_anomalies(today, history)

    print(f"\n=== 메트릭 ===")
    print(f"  BAT-D 시간: {today.get('bat_d_min', '?')}분")
    print(f"  KIS 에러: {today.get('kis_errors', 0)}건")
    print(f"  update_daily 에러: {today.get('update_daily_errors', '?')}건")
    print(f"  BAT-D 실패: {today.get('bat_d_fails', '?')}건")

    print(f"\n=== 이상 감지 ({len(alerts)}건) ===")
    for sev, msg in alerts:
        print(f"  [{sev}] {msg}")

    msg = format_report(today, alerts)
    print(f"\n=== 텔레그램 메시지 ===")
    print(msg)

    if alerts or os.getenv("BAT_D_HEALTH_ALWAYS_NOTIFY") == "1":
        ok = send_telegram(msg)
        print(f"\n[telegram] {'발송 OK' if ok else '발송 SKIP'}")


if __name__ == "__main__":
    main()
