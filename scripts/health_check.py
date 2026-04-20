#!/usr/bin/env python3
"""
Quantum Master — 데이터 신선도 자동 복구 (health_check.py)

역할:
  1. 핵심 데이터 파일의 날짜를 확인
  2. 오늘 날짜가 아니면 → 해당 BAT를 자동 재실행
  3. 재실행 후에도 실패하면 → 텔레그램으로 "수동 개입 필요" 알림

cron 등록 (KST):
  0 18 * * 1-5 bash /home/ubuntu/quantum-master/scripts/cron/run_bat.sh HEALTH
  또는 직접:
  0 18 * * 1-5 /home/ubuntu/quantum-master/venv/bin/python3.11 /home/ubuntu/quantum-master/scripts/health_check.py

사용법:
  python scripts/health_check.py              # 자동 복구 모드
  python scripts/health_check.py --check-only # 확인만 (재실행 안 함)
  python scripts/health_check.py --bat D      # 특정 BAT만 강제 재실행
"""

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

# ── 경로 설정 ──
QM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QM))
os.chdir(QM)

from dotenv import load_dotenv
load_dotenv(QM / ".env")

from src.telegram_sender import send_message
from src.adapters.flowx_uploader import FlowxUploader

TODAY = date.today().isoformat()  # "2026-03-31"
TODAY_COMPACT = date.today().strftime("%Y%m%d")  # "20260331"
RUN_BAT = QM / "scripts" / "cron" / "run_bat.sh"
PY = QM / "venv" / "bin" / "python3.11"
LOG = QM / "logs" / f"health_{datetime.now():%Y%m%d_%H%M}.log"


# ══════════════════════════════════════════
# 1. 검사 대상 정의
# ══════════════════════════════════════════

# BAT별 핵심 데이터 파일 + 날짜 키 + 개별 복구 스크립트
# "해당 파일의 해당 키 값이 오늘 날짜를 포함하면 정상"
#
# recover_script: 개별 복구 스크립트 (BAT 전체 재실행 대신 단독 실행)
# recover_args: 스크립트 실행 인자 (list)
# recover_timeout: 개별 실행 타임아웃 (초)
# priority: 의존성 순서 (낮은 숫자 먼저 실행)
#
# recover_script 없으면 → BAT 전체 재실행 fallback
CHECKS = {
    "D": {
        "name": "장마감 전체 파이프라인",
        "files": [
            {
                "path": "data/institutional_flow/accumulation_alert.json",
                "date_key": "detected_at",
                "label": "기관수급",
                "recover_script": "scripts/institutional_flow_collector.py",
                "recover_timeout": 900,
                "priority": 10,
            },
            {
                "path": "data/volume_spike_watchlist.json",
                "date_key": "date",
                "label": "거래량급등",
                "recover_script": "scripts/scan_volume_spike.py",
                "recover_timeout": 300,
                "priority": 20,
            },
            {
                "path": "data/shield_report.json",
                "date_key": "timestamp",
                "label": "SHIELD",
                "recover_script": "scripts/run_shield.py",
                "recover_args": ["--send"],
                "recover_timeout": 300,
                "priority": 30,
            },
            {
                "path": "data/brain_decision.json",
                "date_key": "timestamp",
                "label": "BRAIN",
                "recover_script": "scripts/run_brain.py",
                "recover_timeout": 300,
                "priority": 40,
            },
            {
                "path": "data/tomorrow_picks.json",
                "date_key": "generated_at",
                "label": "추천종목",
                "recover_script": "scripts/scan_tomorrow_picks.py",
                "recover_timeout": 600,
                "priority": 50,
            },
        ],
    },
    "A": {
        "name": "미장마감",
        "files": [
            {
                "path": "data/ai_strategic_analysis.json",
                "date_key": "analysis_date",
                "label": "미장분석",
                # recover_script 없음 → BAT-A 전체 재실행 fallback
            },
        ],
    },
}


# ══════════════════════════════════════════
# 2. 검사 로직
# ══════════════════════════════════════════

def check_freshness(bat_id: str) -> dict:
    """BAT별 데이터 신선도 검사. 반환: {label: bool}"""
    spec = CHECKS.get(bat_id)
    if not spec:
        return {}

    results = {}
    for f in spec["files"]:
        fpath = QM / f["path"]
        label = f["label"]
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            date_val = str(data.get(f["date_key"], ""))
            # TODAY("2026-03-31")가 값에 포함되면 신선
            results[label] = TODAY in date_val
        except Exception:
            results[label] = False

    return results


def is_bat_fresh(bat_id: str) -> bool:
    """BAT의 모든 핵심 파일이 오늘 날짜이면 True"""
    results = check_freshness(bat_id)
    if not results:
        return True  # 검사 대상 없으면 패스
    return all(results.values())


# ══════════════════════════════════════════
# 2-B. FLOWX Supabase 업로드 검증
# ══════════════════════════════════════════

def check_flowx_uploaded() -> bool:
    """Supabase quant_jarvis 테이블에 오늘 날짜 데이터 존재 여부 확인."""
    try:
        uploader = FlowxUploader()
        if not uploader.is_active:
            log("[FLOWX] Supabase 미연결 — FLOWX 검증 스킵")
            return True  # 연결 안 되면 검증 불가, 패스
        result = uploader.client.table("quant_jarvis").select("date").eq("date", TODAY).execute()
        exists = bool(result.data)
        return exists
    except Exception as e:
        log(f"[FLOWX] Supabase 조회 실패: {e}")
        return False


def check_signals_uploaded() -> bool:
    """Supabase signals 테이블에 오늘 created_at 존재 여부 확인."""
    try:
        uploader = FlowxUploader()
        if not uploader.is_active:
            log("[SIGNALS] Supabase 미연결 — 검증 스킵")
            return True
        today_start = TODAY + "T00:00:00"
        result = (
            uploader.client.table("signals")
            .select("created_at")
            .gte("created_at", today_start)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception as e:
        log(f"[SIGNALS] Supabase 조회 실패: {e}")
        return False


def check_relay_uploaded() -> bool:
    """Supabase dashboard_relay 테이블에 오늘 날짜 존재 여부 확인.

    BAT-D G2(릴레이) 완료 전에 BAT-F(17:15) FLOWX 일괄 업로드가 먼저 실행되면
    relay_trading_signal.json이 아직 없어 relay=False 순서 문제 발생.
    HEALTH는 18:00에 실행되므로 파일이 이미 있어 재업로드 성공 가능.
    """
    try:
        uploader = FlowxUploader()
        if not uploader.is_active:
            log("[RELAY] Supabase 미연결 — 검증 스킵")
            return True
        result = (
            uploader.client.table("dashboard_relay")
            .select("date")
            .eq("date", TODAY)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception as e:
        log(f"[RELAY] Supabase 조회 실패: {e}")
        return False


def rerun_relay_upload() -> bool:
    """FLOWX relay 업로드만 단독 재실행 (FlowxUploader.upload_relay)."""
    log("[RELAY-RERUN] dashboard_relay 재업로드 시작")
    try:
        uploader = FlowxUploader()
        if not uploader.is_active:
            log("[RELAY-RERUN] Supabase 미연결")
            return False
        ok = uploader.upload_relay(TODAY)
        log(f"[RELAY-RERUN] 결과: {ok}")
        return bool(ok)
    except Exception as e:
        log(f"[RELAY-RERUN] 실행 오류: {e}")
        return False


def rerun_signal_logger() -> bool:
    """signal_logger.py 단독 재실행 — tomorrow_picks.json → signals 테이블 기록."""
    log("[SIGNALS-RERUN] signal_logger.py 재실행 시작")
    try:
        result = subprocess.run(
            [str(PY), "scripts/signal_logger.py"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(QM),
        )
        log(f"[SIGNALS-RERUN] 종료 (exit={result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log("[SIGNALS-RERUN] 타임아웃 (180초 초과)")
        return False
    except Exception as e:
        log(f"[SIGNALS-RERUN] 실행 오류: {e}")
        return False


def rerun_flowx() -> bool:
    """upload_flowx.py 단독 재실행."""
    log("[FLOWX-RERUN] upload_flowx.py 재실행 시작")
    try:
        result = subprocess.run(
            [str(PY), "scripts/upload_flowx.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(QM),
        )
        log(f"[FLOWX-RERUN] 종료 (exit={result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log("[FLOWX-RERUN] 타임아웃 (300초 초과)")
        return False
    except Exception as e:
        log(f"[FLOWX-RERUN] 실행 오류: {e}")
        return False


# ══════════════════════════════════════════
# 2-C. 투자자수급 DB + CSV 검증 (L6, L7)
# ══════════════════════════════════════════

def check_investor_db() -> bool:
    """investor_daily.db에 오늘 데이터 존재 확인."""
    db_path = QM / "data" / "investor_flow" / "investor_daily.db"
    if not db_path.exists():
        log("[INVESTOR-DB] DB 파일 없음")
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM investor_daily WHERE date = ?",
            (TODAY_COMPACT,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        if count > 0:
            log(f"[INVESTOR-DB] 오늘 데이터 {count}행 — 정상")
            return True
        log("[INVESTOR-DB] 오늘 데이터 0행!")
        return False
    except Exception as e:
        log(f"[INVESTOR-DB] 조회 실패: {e}")
        return False


def rerun_investor_collect() -> bool:
    """collect_investor_bulk --core-only + sync_investor_to_csv 재실행."""
    log("[INVESTOR-RERUN] collect_investor_bulk --core-only 시작")
    try:
        r1 = subprocess.run(
            [str(PY), "scripts/collect_investor_bulk.py", "--core-only"],
            capture_output=True, text=True, timeout=600, cwd=str(QM),
        )
        if r1.returncode != 0:
            log(f"[INVESTOR-RERUN] collect 실패 (exit={r1.returncode})")
            return False
        r2 = subprocess.run(
            [str(PY), "scripts/sync_investor_to_csv.py"],
            capture_output=True, text=True, timeout=300, cwd=str(QM),
        )
        log(f"[INVESTOR-RERUN] sync 완료 (exit={r2.returncode})")
        return r2.returncode == 0
    except subprocess.TimeoutExpired:
        log("[INVESTOR-RERUN] 타임아웃")
        return False
    except Exception as e:
        log(f"[INVESTOR-RERUN] 실행 오류: {e}")
        return False


def check_investor_csv_quality() -> bool:
    """stock_data_daily/ 랜덤 10개 CSV에서 Foreign_Net이 실제 값인지."""
    import csv as csv_mod
    import random
    csv_dir = QM / "stock_data_daily"
    if not csv_dir.exists():
        log("[INVESTOR-CSV] stock_data_daily/ 없음")
        return False
    csvs = list(csv_dir.glob("*.csv"))
    if len(csvs) < 10:
        log(f"[INVESTOR-CSV] CSV {len(csvs)}개 — 부족")
        return False
    samples = random.sample(csvs, 10)
    has_data = 0
    for p in samples:
        try:
            with open(p, "r", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    val = row.get("Foreign_Net", "")
                    if val and val not in ("", "0", "0.0", "nan", "None"):
                        has_data += 1
                        break
        except Exception:
            continue
    log(f"[INVESTOR-CSV] 샘플 10개 중 {has_data}개에 수급 데이터 있음")
    return has_data >= 3  # 유니버스만 수급 있으므로 30% 이상이면 OK


# ══════════════════════════════════════════
# 2-D. FLOWX 14테이블 전수 검증 (L8)
# ══════════════════════════════════════════

FLOWX_TABLES = [
    ("quant_jarvis", "date"),
    ("quant_scenario_dashboard", "date"),
    ("quant_sector_momentum", "date"),
    ("quant_fib_scanner", "date"),
    ("quant_alpha_scanner", "date"),
    ("quant_market_ranking", "date"),
    ("quant_bluechip_checkup", "date"),
    ("dashboard_crash_bounce", "date"),
    ("sector_rotation", "date"),
    ("etf_signals", "date"),
    ("china_flow", "date"),
    ("short_signals", "date"),
    ("morning_briefings", "date"),
    ("dashboard_relay", "date"),
]


def check_flowx_all_tables() -> tuple:
    """14개 FLOWX 테이블 전수 검증. (fresh_list, stale_list) 반환."""
    try:
        uploader = FlowxUploader()
        if not uploader.is_active:
            log("[FLOWX-ALL] Supabase 미연결 — 스킵")
            return [], []
    except Exception:
        return [], []

    fresh, stale = [], []
    for table, date_col in FLOWX_TABLES:
        try:
            result = (
                uploader.client.table(table)
                .select(date_col)
                .order(date_col, desc=True)
                .limit(1)
                .execute()
            )
            if result.data:
                latest = str(result.data[0][date_col])[:10]
                if latest == TODAY:
                    fresh.append(table)
                else:
                    stale.append(f"{table}({latest})")
            else:
                stale.append(f"{table}(EMPTY)")
        except Exception:
            stale.append(f"{table}(ERR)")
    log(f"[FLOWX-ALL] {len(fresh)}/{len(FLOWX_TABLES)} 신선, STALE: {len(stale)}")
    return fresh, stale


# ══════════════════════════════════════════
# 2-E. AI 모델 에러 감지 (L9)
# ══════════════════════════════════════════

def check_ai_model_errors() -> tuple:
    """오늘 cron 로그에서 AI 폴백/404 횟수 카운트. (fallback_count, error_404_count)"""
    log_path = QM / "logs" / f"cron_{datetime.now():%Y%m%d}.log"
    if not log_path.exists():
        return 0, 0
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        fallback = content.count("Haiku 폴백") + content.count("haiku fallback")
        err_404 = content.count("model_not_found") + content.lower().count("404 not found")
        return fallback, err_404
    except Exception:
        return 0, 0


# ══════════════════════════════════════════
# 3. 자동 재실행
# ══════════════════════════════════════════

def rerun_bat(bat_id: str) -> bool:
    """run_bat.sh를 통해 BAT 재실행. 성공 시 True. (Fallback — recover_script 없을 때만)"""
    log(f"[RERUN] BAT-{bat_id} 자동 재실행 시작")
    try:
        result = subprocess.run(
            ["bash", str(RUN_BAT), bat_id],
            capture_output=True,
            text=True,
            timeout=3600,  # 최대 1시간
            cwd=str(QM),
        )
        log(f"[RERUN] BAT-{bat_id} 종료 (exit={result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"[RERUN] BAT-{bat_id} 타임아웃 (1시간 초과)")
        return False
    except Exception as e:
        log(f"[RERUN] BAT-{bat_id} 실행 오류: {e}")
        return False


def rerun_script(file_spec: dict) -> bool:
    """개별 복구 스크립트 단독 실행. BAT 전체 재실행보다 훨씬 빠름.

    file_spec 필수 키: recover_script
    file_spec 선택 키: recover_args (list), recover_timeout (초), label
    """
    script = file_spec.get("recover_script")
    if not script:
        return False

    args = file_spec.get("recover_args", []) or []
    timeout = int(file_spec.get("recover_timeout", 300))
    label = file_spec.get("label", script)

    log(f"[RECOVER] {label} → {script} {' '.join(args)} (timeout={timeout}s)")
    try:
        result = subprocess.run(
            [str(PY), script] + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(QM),
        )
        ok = result.returncode == 0
        status = "OK" if ok else f"FAIL(exit={result.returncode})"
        log(f"[RECOVER] {label} {status}")
        if not ok and result.stderr:
            # 마지막 5줄만 로그에 남김
            tail = "\n".join(result.stderr.strip().splitlines()[-5:])
            log(f"[RECOVER] {label} stderr:\n{tail}")
        return ok
    except subprocess.TimeoutExpired:
        log(f"[RECOVER] {label} 타임아웃 ({timeout}초 초과)")
        return False
    except Exception as e:
        log(f"[RECOVER] {label} 실행 오류: {e}")
        return False


# ══════════════════════════════════════════
# 4. 알림 + 로그
# ══════════════════════════════════════════

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def notify(msg: str):
    log(f"[NOTIFY] {msg}")
    try:
        # QUIET 모드에서도 통과하도록 [HEALTH] 태그 추가
        tagged = msg if msg.startswith("[HEALTH]") else f"[HEALTH] {msg}"
        send_message(tagged)
    except Exception as e:
        log(f"[NOTIFY] 텔레그램 전송 실패: {e}")


# ══════════════════════════════════════════
# 5. 메인 로직
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="데이터 신선도 자동 복구")
    parser.add_argument("--check-only", action="store_true", help="확인만 (재실행 안 함)")
    parser.add_argument("--bat", type=str, help="특정 BAT만 강제 재실행")
    args = parser.parse_args()

    log(f"=== Health Check 시작 ({TODAY}) ===")

    # 주말이면 스킵
    if datetime.now().weekday() >= 5:
        log("주말 — 스킵")
        return

    # 특정 BAT 강제 재실행
    if args.bat:
        bat_id = args.bat.upper()
        log(f"[FORCE] BAT-{bat_id} 강제 재실행")
        rerun_bat(bat_id)
        # 재실행 후 검증
        results = check_freshness(bat_id)
        fresh = all(results.values()) if results else True
        if fresh:
            notify(f"✅ BAT-{bat_id} 강제 재실행 성공 — 데이터 정상")
        else:
            stale = [k for k, v in results.items() if not v]
            notify(f"❌ BAT-{bat_id} 강제 재실행 후에도 실패: {', '.join(stale)}")
        return

    # 전체 BAT 검사
    all_ok = True
    recovered = []
    failed = []

    for bat_id, spec in CHECKS.items():
        results = check_freshness(bat_id)
        # stale 파일들을 file_spec 리스트로 수집 (recover_script 매핑 접근 위해)
        stale_files = [f for f in spec["files"] if not results.get(f["label"], False)]

        if not stale_files:
            log(f"[OK] BAT-{bat_id} ({spec['name']}): 전체 신선")
            continue

        # 낡은 데이터 발견
        all_ok = False
        stale_labels = [f["label"] for f in stale_files]
        log(f"[STALE] BAT-{bat_id} ({spec['name']}): {', '.join(stale_labels)}")

        if args.check_only:
            failed.append(f"BAT-{bat_id}: {', '.join(stale_labels)}")
            continue

        # ── 선택적 복구: 우선순위 순으로 개별 스크립트 실행 ──
        # priority가 있는 파일은 의존 순서대로 실행 (기관수급 → 거래량 → SHIELD → BRAIN → picks)
        stale_files.sort(key=lambda f: f.get("priority", 999))

        recovered_labels = []
        fallback_needed = False

        for f in stale_files:
            if f.get("recover_script"):
                ok = rerun_script(f)
                if ok:
                    recovered_labels.append(f["label"])
            else:
                # 매핑 없으면 BAT 전체 재실행 fallback 필요
                fallback_needed = True
                log(f"[RECOVER] {f['label']}: recover_script 미정의 → BAT 전체 재실행 필요")

        if fallback_needed:
            log(f"[RERUN] BAT-{bat_id} 전체 재실행 fallback 시작")
            rerun_bat(bat_id)

        # 재실행 후 검증
        results2 = check_freshness(bat_id)
        stale2 = [k for k, v in results2.items() if not v]

        if not stale2:
            recovered.append(f"BAT-{bat_id}({', '.join(recovered_labels)})" if recovered_labels else f"BAT-{bat_id}")
            log(f"[RECOVERED] BAT-{bat_id}: 복구 성공")
        else:
            failed.append(f"BAT-{bat_id}: {', '.join(stale2)}")
            log(f"[FAILED] BAT-{bat_id}: 복구 실패 — {', '.join(stale2)}")

    # ── Universe CSV 연령 검사 (pykrx 야간 불안정 대비) ──
    universe_csv = QM / "data" / "universe.csv"
    if universe_csv.exists():
        csv_mtime = datetime.fromtimestamp(universe_csv.stat().st_mtime)
        csv_age_days = (datetime.now() - csv_mtime).days
        if csv_age_days <= 7:
            log(f"[UNIVERSE] CSV {csv_age_days}일전 갱신 — 정상")
        else:
            all_ok = False
            log(f"[UNIVERSE] CSV {csv_age_days}일전 갱신 — 장중 갱신 필요!")
            if not args.check_only:
                log("[UNIVERSE] rebuild_universe.py --incremental 시도")
                ok = rerun_script({
                    "recover_script": "scripts/rebuild_universe.py",
                    "recover_args": ["--incremental"],
                    "recover_timeout": 900,
                    "label": "유니버스CSV",
                })
                # ★ exit code만 믿지 않고 실제 CSV mtime 재확인 (거짓 OK 방지)
                if ok:
                    new_mtime = datetime.fromtimestamp(universe_csv.stat().st_mtime)
                    new_age = (datetime.now() - new_mtime).days
                    if new_age <= 7:
                        recovered.append("유니버스CSV")
                        log(f"[UNIVERSE] 복구 검증 통과 — CSV {new_age}일전 갱신")
                    else:
                        # --incremental은 CSV 미기록 구조 → 풀 rebuild fallback 시도
                        log(f"[UNIVERSE] --incremental 후에도 CSV {new_age}일전 — 풀 rebuild fallback 시도")
                        ok2 = rerun_script({
                            "recover_script": "scripts/rebuild_universe.py",
                            "recover_args": [],  # 풀 모드 (기본 --min-cap 0.2)
                            "recover_timeout": 1200,
                            "label": "유니버스CSV(풀)",
                        })
                        if ok2:
                            new_mtime2 = datetime.fromtimestamp(universe_csv.stat().st_mtime)
                            new_age2 = (datetime.now() - new_mtime2).days
                            if new_age2 <= 1:
                                recovered.append("유니버스CSV(풀rebuild)")
                                log(f"[UNIVERSE] 풀 rebuild 복구 성공 — CSV {new_age2}일전 갱신")
                            else:
                                failed.append(f"유니버스CSV: 풀 rebuild 후에도 {new_age2}일 미갱신 (pykrx 이상, BAT-H 장중 대기)")
                                log(f"[UNIVERSE] 풀 rebuild도 실패 — CSV {new_age2}일전")
                        else:
                            failed.append("유니버스CSV: 풀 rebuild 실행 실패 (pykrx 야간 불안정 가능, BAT-H 장중 대기)")
                            log("[UNIVERSE] 풀 rebuild 실행 실패")
                else:
                    failed.append("유니버스CSV: --incremental 실행 실패 (장중 BAT-H 대기)")
            else:
                failed.append(f"유니버스CSV: {csv_age_days}일 미갱신")
    else:
        all_ok = False
        log("[UNIVERSE] CSV 파일 없음!")
        failed.append("유니버스CSV: 파일 없음")

    # ── FLOWX Supabase 업로드 검증 (3차 안전장치) ──
    flowx_ok = check_flowx_uploaded()
    if flowx_ok:
        log("[FLOWX] Supabase 오늘 데이터 확인 — 정상")
    else:
        log("[FLOWX] Supabase 오늘 데이터 없음 — 재업로드 시도")
        all_ok = False
        if not args.check_only:
            rerun_ok = rerun_flowx()
            # 재실행 후 다시 확인
            if rerun_ok and check_flowx_uploaded():
                recovered.append("FLOWX")
                log("[FLOWX] 재업로드 성공")
            else:
                failed.append("FLOWX: Supabase 업로드 실패")
                log("[FLOWX] 재업로드 후에도 실패")
        else:
            failed.append("FLOWX: Supabase 오늘 데이터 없음")

    # ── signals 테이블 (tracker) 신선도 검증 (4차 안전장치) ──
    signals_ok = check_signals_uploaded()
    if signals_ok:
        log("[SIGNALS] Supabase signals 오늘 데이터 확인 — 정상")
    else:
        log("[SIGNALS] Supabase signals 오늘 데이터 없음 — signal_logger 재실행 시도")
        all_ok = False
        if not args.check_only:
            rerun_ok = rerun_signal_logger()
            if rerun_ok and check_signals_uploaded():
                recovered.append("signals(tracker)")
                log("[SIGNALS] 재실행 성공")
            else:
                failed.append("signals(tracker): Supabase 업로드 실패 (tomorrow_picks 부재 가능)")
                log("[SIGNALS] 재실행 후에도 실패")
        else:
            failed.append("signals(tracker): Supabase 오늘 데이터 없음")

    # ── dashboard_relay 테이블 신선도 검증 (5차 안전장치) ──
    # BAT-D G2(17:19 relay_engine) < BAT-F(17:15 upload_flowx) 순서 역전으로
    # relay=False 발생 가능. HEALTH 18:00에는 파일이 있으므로 재업로드하면 성공.
    relay_ok = check_relay_uploaded()
    if relay_ok:
        log("[RELAY] Supabase dashboard_relay 오늘 데이터 확인 — 정상")
    else:
        log("[RELAY] Supabase dashboard_relay 오늘 데이터 없음 — 단독 재업로드 시도")
        all_ok = False
        if not args.check_only:
            rerun_ok = rerun_relay_upload()
            if rerun_ok and check_relay_uploaded():
                recovered.append("dashboard_relay")
                log("[RELAY] 재업로드 성공")
            else:
                failed.append("dashboard_relay: 재업로드 실패 (relay_trading_signal.json 확인 필요)")
                log("[RELAY] 재업로드 후에도 실패")
        else:
            failed.append("dashboard_relay: Supabase 오늘 데이터 없음")

    # ── L6: 투자자수급 DB ──
    investor_ok = check_investor_db()
    if not investor_ok:
        all_ok = False
        if not args.check_only:
            rerun_ok = rerun_investor_collect()
            if rerun_ok and check_investor_db():
                recovered.append("수급DB")
                log("[INVESTOR-DB] 재수집 성공")
            else:
                failed.append("수급DB: 재수집 후에도 0행")
        else:
            failed.append("수급DB: 오늘 데이터 없음")

    # ── L7: 수급 CSV 품질 ──
    csv_ok = check_investor_csv_quality()
    if not csv_ok:
        all_ok = False
        if not args.check_only:
            if investor_ok or check_investor_db():
                log("[INVESTOR-CSV] sync_investor_to_csv 재실행")
                subprocess.run(
                    [str(PY), "scripts/sync_investor_to_csv.py"],
                    capture_output=True, timeout=300, cwd=str(QM),
                )
                if check_investor_csv_quality():
                    recovered.append("수급CSV")
                else:
                    failed.append("수급CSV: sync 재실행 후에도 품질 미달")
            else:
                failed.append("수급CSV: DB 데이터 없어 sync 불가")
        else:
            failed.append("수급CSV: Foreign_Net 데이터 없음")

    # ── L8: FLOWX 14테이블 전수 ──
    fresh_tables, stale_tables = check_flowx_all_tables()
    if stale_tables:
        all_ok = False
        log(f"[FLOWX-ALL] STALE: {', '.join(stale_tables)}")
        if not args.check_only:
            rerun_flowx()
            fresh2, stale2 = check_flowx_all_tables()
            newly_fixed = len(stale_tables) - len(stale2)
            if newly_fixed > 0:
                recovered.append(f"FLOWX({newly_fixed}개 복구)")
            if stale2:
                failed.append(f"FLOWX: {', '.join(stale2[:5])}")
        else:
            failed.append(f"FLOWX: {', '.join(stale_tables[:5])}")
    else:
        log("[FLOWX-ALL] 14테이블 전체 신선 — 정상")

    # ── L9: AI 모델 에러 ──
    fallback_cnt, err_404_cnt = check_ai_model_errors()
    if fallback_cnt >= 3 or err_404_cnt >= 1:
        all_ok = False
        failed.append(f"AI모델: 폴백 {fallback_cnt}회, 404 {err_404_cnt}회")
        log(f"[AI-MODEL] 폴백={fallback_cnt}, 404={err_404_cnt} — 모델 설정 확인 필요")
    elif fallback_cnt > 0:
        log(f"[AI-MODEL] 폴백 {fallback_cnt}회 (정상 범위)")

    # ══════════════════════════════════════════
    # 종합 리포트 (항상 발송 — 정상/이상 무관)
    # ══════════════════════════════════════════
    data_fresh = is_bat_fresh("D")
    flowx_total = len(FLOWX_TABLES)
    flowx_ok_count = len(fresh_tables) if not stale_tables else len(fresh_tables)

    summary_lines = [
        f"📊 BAT-D 종합 검수 ({TODAY})",
        "",
        f"📂 데이터 JSON: {'✅' if data_fresh else '⚠️'}",
        f"📈 수급DB: {'✅' if investor_ok else '❌'}",
        f"📊 수급CSV: {'✅' if csv_ok else '❌'}",
        f"🌐 FLOWX: {flowx_ok_count}/{flowx_total}" + (f" ⚠️STALE {len(stale_tables)}" if stale_tables else " ✅"),
        f"🤖 AI모델: {'✅' if (fallback_cnt < 3 and err_404_cnt == 0) else '⚠️'}" + (f" (폴백{fallback_cnt})" if fallback_cnt else ""),
    ]

    if recovered:
        summary_lines.append(f"\n🔄 자동복구: {', '.join(recovered)}")
    if failed:
        summary_lines.append(f"\n❌ 수동확인:\n" + "\n".join(f"  • {f}" for f in failed))
    if not recovered and not failed and all_ok:
        summary_lines.append("\n✅ 전체 정상 — 내일 BAT 준비 완료")

    notify("\n".join(summary_lines))
    log("=== Health Check 완료 ===")


if __name__ == "__main__":
    main()
