"""scripts/sql/bot_collaboration_v1.sql 자동 적용 (5/18 신규)

목적: quant_bot_advisory + scalper_bot_feedback 테이블 Supabase 생성
멱등성: CREATE TABLE IF NOT EXISTS 사용 → 재실행 안전

Usage:
    python scripts/db/apply_bot_collaboration.py             # 적용 + 검증
    python scripts/db/apply_bot_collaboration.py --dry-run   # SQL 출력만
    python scripts/db/apply_bot_collaboration.py --verify    # 적용 안 하고 현재 상태만
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

SQL_PATH = PROJECT_ROOT / "scripts" / "sql" / "bot_collaboration_v1.sql"
TARGET_TABLES = ["quant_bot_advisory", "scalper_bot_feedback"]


def get_connection():
    import psycopg2

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL 미설정 — .env 확인")
    return psycopg2.connect(url, connect_timeout=10)


def verify_tables(con) -> dict:
    """타겟 테이블 존재 + 행 수 확인."""
    result = {}
    cur = con.cursor()
    for tname in TARGET_TABLES:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=%s)",
            (tname,),
        )
        exists = cur.fetchone()[0]
        if exists:
            cur.execute(f"SELECT COUNT(*) FROM {tname}")
            n_rows = cur.fetchone()[0]
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
                (tname,),
            )
            cols = [r[0] for r in cur.fetchall()]
            result[tname] = {"exists": True, "n_rows": n_rows, "n_columns": len(cols)}
        else:
            result[tname] = {"exists": False}
    return result


def apply_sql(con, sql: str) -> None:
    cur = con.cursor()
    cur.execute(sql)
    con.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="적용 안 하고 현재 상태만 확인")
    args = parser.parse_args()

    if not SQL_PATH.exists():
        print(f"❌ SQL 파일 없음: {SQL_PATH}")
        return 1

    sql = SQL_PATH.read_text(encoding="utf-8")
    print(f"SQL 파일: {SQL_PATH.name} ({len(sql):,} bytes)")

    if args.dry_run:
        print("[DRY-RUN] 실행 안 함 — SQL 내용:")
        print("=" * 70)
        print(sql)
        return 0

    print("DATABASE_URL 연결 시도...")
    con = get_connection()
    print("연결 OK")

    # 1) 적용 전 상태
    print("\n[BEFORE] 테이블 상태:")
    before = verify_tables(con)
    for tname, info in before.items():
        status = "✅ 존재" if info["exists"] else "❌ 없음"
        rows = f" ({info.get('n_rows', 0):,}행)" if info["exists"] else ""
        print(f"  {tname}: {status}{rows}")

    if args.verify:
        con.close()
        return 0

    # 2) 적용
    print("\n[APPLY] SQL 실행...")
    try:
        apply_sql(con, sql)
        print("✅ 적용 성공")
    except Exception as e:
        print(f"❌ 적용 실패: {e}")
        con.rollback()
        con.close()
        return 1

    # 3) 적용 후 상태
    print("\n[AFTER] 테이블 상태:")
    after = verify_tables(con)
    for tname, info in after.items():
        status = "✅ 존재" if info["exists"] else "❌ 없음"
        rows = f" ({info.get('n_rows', 0):,}행, {info.get('n_columns', 0)} 컬럼)" if info["exists"] else ""
        print(f"  {tname}: {status}{rows}")

    # 4) 첫 advisory 확인
    print("\n[FIRST ADVISORY] 시드 INSERT 확인:")
    cur = con.cursor()
    cur.execute(
        "SELECT id, advisory_date, advisory_time, msg_type, market_regime, title "
        "FROM quant_bot_advisory ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    if row:
        print(f"  id={row[0]} date={row[1]} time={row[2]} type={row[3]} regime={row[4]}")
        print(f"  title: {row[5]}")
    else:
        print("  (시드 없음)")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
