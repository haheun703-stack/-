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
RUN_BAT = QM / "scripts" / "cron" / "run_bat.sh"
PY = QM / "venv" / "bin" / "python3.11"
LOG = QM / "logs" / f"health_{datetime.now():%Y%m%d_%H%M}.log"


# ══════════════════════════════════════════
# 1. 검사 대상 정의
# ══════════════════════════════════════════

# BAT별 핵심 데이터 파일 + 날짜 키
# "해당 파일의 해당 키 값이 오늘 날짜를 포함하면 정상"
CHECKS = {
    "D": {
        "name": "장마감 전체 파이프라인",
        "files": [
            {
                "path": "data/brain_decision.json",
                "date_key": "timestamp",
                "label": "BRAIN",
            },
            {
                "path": "data/shield_report.json",
                "date_key": "timestamp",
                "label": "SHIELD",
            },
            {
                "path": "data/institutional_flow/accumulation_alert.json",
                "date_key": "detected_at",
                "label": "기관수급",
            },
            {
                "path": "data/volume_spike_watchlist.json",
                "date_key": "date",
                "label": "거래량급등",
            },
            {
                "path": "data/tomorrow_picks.json",
                "date_key": "generated_at",
                "label": "추천종목",
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
# 3. 자동 재실행
# ══════════════════════════════════════════

def rerun_bat(bat_id: str) -> bool:
    """run_bat.sh를 통해 BAT 재실행. 성공 시 True."""
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
        stale = [k for k, v in results.items() if not v]

        if not stale:
            log(f"[OK] BAT-{bat_id} ({spec['name']}): 전체 신선")
            continue

        # 낡은 데이터 발견
        all_ok = False
        log(f"[STALE] BAT-{bat_id} ({spec['name']}): {', '.join(stale)}")

        if args.check_only:
            failed.append(f"BAT-{bat_id}: {', '.join(stale)}")
            continue

        # 자동 재실행
        rerun_bat(bat_id)

        # 재실행 후 검증
        results2 = check_freshness(bat_id)
        stale2 = [k for k, v in results2.items() if not v]

        if not stale2:
            recovered.append(f"BAT-{bat_id}")
            log(f"[RECOVERED] BAT-{bat_id}: 자동 복구 성공")
        else:
            failed.append(f"BAT-{bat_id}: {', '.join(stale2)}")
            log(f"[FAILED] BAT-{bat_id}: 복구 실패 — {', '.join(stale2)}")

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

    # 결과 알림
    if all_ok:
        log("=== 전체 정상 ===")
        # 정상일 때는 알림 안 보냄 (조용히)
        return

    msg_parts = []
    if recovered:
        msg_parts.append(f"🔄 자동 복구 성공: {', '.join(recovered)}")
    if failed:
        msg_parts.append(f"❌ 수동 개입 필요: {chr(10).join(failed)}")

    if msg_parts:
        notify("\n".join(msg_parts))
    elif args.check_only and not all_ok:
        stale_summary = "; ".join(failed) if failed else "일부 파일 낡음"
        notify(f"⚠️ 데이터 신선도 경고: {stale_summary}")

    log("=== Health Check 완료 ===")


if __name__ == "__main__":
    main()
