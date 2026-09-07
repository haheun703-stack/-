"""quant_bot_advisory 소급 정리 — 데이터계약(260724) 이전 적재분 (B-74 잔여).

**퐝가님 2026-09-07 승인분.** 기본은 dry-run이고 `--apply`를 줘야 쓴다.

    # VPS에서
    cd ~/quantum-master
    ./venv/bin/python3.11 -u -X utf8 scripts/backfill_advisory_contract.py           # 조회만
    ./venv/bin/python3.11 -u -X utf8 scripts/backfill_advisory_contract.py --apply   # 실제 정리

범위
  계약 시행일(2026-07-27) 이후 · 우리 생산분(`alert_codes`) · 스크럽 배포(9/7 11:25) 이전.
  계약 **이전**(5/18~7/26, 약 2,160행)은 당시 규칙상 정상이라 대상이 아니다.
  타 봇 적재분(alert_codes 불일치)도 대상이 아니다.

하는 일 (행은 지우지 않는다 — 수신 이력은 남긴다)
  · `reasoning` → 허용키만 남김 (`src/adapters/advisory_contract.ALLOWED_REASONING_KEYS`)
  · `related_tickers` → 빈 배열
  · `title`·`body` → **금지 어휘가 있는 것만** 최소 문구로 교체

안전장치
  · 실행 전 전량을 `data/advisory_backfill_backup_20260907.json`에 백업
  · 단일 트랜잭션. 실패 시 롤백
  · 종료 전 같은 판정으로 사후 검증(잔존 위반 0을 확인)
"""
import os, sys, json, argparse
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from src.adapters.advisory_contract import scrub_reasoning, check_text

SINCE_DEPLOY = "2026-09-07T11:25:00+09:00"
CONTRACT_FROM = "2026-07-27"
CODES = ["SNAPSHOT-AUTO", "MORNING-BRIEFING"]

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="실제 UPDATE (없으면 dry-run)")
args = ap.parse_args()

con = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=15)
cur = con.cursor()
cur.execute("""SELECT id, advisory_date, msg_type, title, body, related_tickers, reasoning
               FROM quant_bot_advisory
               WHERE alert_codes && %s::text[]
                 AND advisory_date >= %s
                 AND created_at < %s::timestamptz
               ORDER BY id""", (CODES, CONTRACT_FROM, SINCE_DEPLOY))
rows = cur.fetchall()
print(f"대상 {len(rows)}행 (계약 시행 {CONTRACT_FROM} 이후, 배포 이전)")

backup = []
plan = []
for rid, d, mt, title, body, rt, rs in rows:
    backup.append({"id": rid, "date": str(d), "msg_type": mt, "title": title,
                   "body": body, "related_tickers": rt, "reasoning": rs})
    new_rs, removed = scrub_reasoning(rs if isinstance(rs, dict) else {})
    new_rt = []
    new_title, new_body = title, body
    if check_text(title):
        new_title = f"[advisory {mt}] {d} — 시장 진단(매매판단 필드 소급 제거)"
    if check_text(body):
        new_body = (f"{d} {mt} 시장 진단. 계산 산출물이며 매매 판단이 아니다. "
                    f"(2026-09-07 데이터계약 소급 정리로 본문 대체)")
    changed = (new_rs != (rs or {})) or bool(rt) or new_title != title or new_body != body
    if changed:
        plan.append((rid, new_title, new_body, new_rt, new_rs))

print(f"변경 예정 {len(plan)}행 / 무변경 {len(rows) - len(plan)}행")
bp = str(ROOT / "data" / "advisory_backfill_backup_20260907.json")
with open(bp, "w", encoding="utf-8") as f:
    json.dump(backup, f, ensure_ascii=False, default=str)
print(f"백업 저장: {bp} ({len(backup)}행)")

if plan[:1]:
    rid, t, b, rt, rs = plan[0]
    print(f"\n샘플 #{rid}\n  title: {t[:70]}\n  reasoning 키: {sorted(rs)}\n  related_tickers: {rt}")

if not args.apply:
    print("\n[DRY-RUN] --apply 없으면 쓰지 않는다.")
    con.close(); sys.exit(0)

try:
    for rid, t, b, rt, rs in plan:
        cur.execute("""UPDATE quant_bot_advisory
                       SET title=%s, body=%s, related_tickers=%s, reasoning=%s
                       WHERE id=%s""", (t, b, rt, Json(rs), rid))
    con.commit()
    print(f"\n✅ UPDATE 커밋 완료 — {len(plan)}행")
except Exception as e:
    con.rollback()
    print(f"\n❌ 실패·롤백: {e}")
    sys.exit(1)

# 사후 검증
from src.adapters.advisory_contract import audit_row
cur.execute("""SELECT id, title, body, related_tickers, reasoning FROM quant_bot_advisory
               WHERE alert_codes && %s::text[] AND advisory_date >= %s
                 AND created_at < %s::timestamptz""", (CODES, CONTRACT_FROM, SINCE_DEPLOY))
bad = [r[0] for r in cur.fetchall()
       if audit_row({"title": r[1], "body": r[2], "related_tickers": r[3], "reasoning": r[4]})]
print(f"사후 검증: 잔존 위반 {len(bad)}행" + (f" — {bad[:5]}" if bad else " ✅"))
con.close()
