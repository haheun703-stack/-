"""데이터계약(260724) 이행 자가검사 — 중단 테이블 신규 레코드 차단 검증.

용도: 적재 차단(b7ae38b·409ee34) 배포 후, 중단 대상 테이블에 신규 레코드가
      실제로 들어가지 않는지 Supabase 실물로 확인해 운영자·웹봇에 제출한다.

읽기 전용(SELECT만). 쓰기·삭제 없음.

실행:
    python -u -X utf8 scripts/verify_contract_suspension.py [--asof 20260728]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.flowx_uploader import SUSPENDED_TABLES  # noqa: E402

# 유지 확정 테이블 — 차단되면 안 되는 것들(역방향 검증: 오늘자가 있어야 정상)
KEEP_TABLES = [
    "quant_scenario_dashboard",   # /scenario 유일 소스
    "quant_leader_cycle",         # /leader-cycle
    "quant_sector_flow",
    "dashboard_smart_money",       # ★접두 quant_ 아님(uploader 실측)
    "quant_valuation_gap",
]

# 회색지대(B-24 ①) — 운영자 판단 대기. 현재 적재 중인지 실측만 한다.
GRAY_TABLES = [
    "quant_bluechip_checkup",
    "quant_us_macro",
    "quant_market_ranking",
]

# 날짜 컬럼 후보 (우선순위 순)
DATE_COL_CANDIDATES = [
    "date", "trade_date", "signal_date", "as_of", "asof", "base_date",
    "created_at", "updated_at", "inserted_at",
]


def _client():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import os
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("[FATAL] SUPABASE_URL/KEY 미설정")
        sys.exit(2)
    from supabase import create_client
    return create_client(url, key)


def _pick_date_col(cols: list[str]) -> str | None:
    for cand in DATE_COL_CANDIDATES:
        if cand in cols:
            return cand
    return None


def _norm(value) -> str:
    """날짜/타임스탬프 값을 YYYYMMDD로 정규화."""
    if value is None:
        return ""
    s = str(value)
    digits = "".join(ch for ch in s[:10] if ch.isdigit())
    return digits[:8]


# 퀀트봇 FLOWX 업로드 시각대(KST) — 이 시각 레코드가 있으면 퀀트봇 적재분이다.
# BAT-D 종반 FLOWX upload(18:40) + BAT-F 재시도(18:47). 타 봇은 16:30~17:00대에 적재한다.
QUANT_UPLOAD_KST_HOURS = {18, 19}


def _kst(ts: str) -> str:
    """UTC ISO 타임스탬프 → KST HH:MM."""
    try:
        h = (int(ts[11:13]) + 9) % 24
        return f"{h:02d}:{ts[14:16]}"
    except Exception:
        return "?"


def _producer_times(client, table: str, date_col: str, asof: str) -> tuple[list[str], bool]:
    """당일 레코드의 적재 시각(KST) 분포와 '퀀트봇 시각대 포함' 여부."""
    ts_col = None
    try:
        head = client.table(table).select("*").limit(1).execute()
        cols = list(head.data[0].keys()) if head.data else []
        for c in ("created_at", "updated_at", "inserted_at"):
            if c in cols:
                ts_col = c
                break
        if ts_col is None:
            return [], False
        d = datetime.strptime(asof, "%Y%m%d").date()
        q = client.table(table).select(ts_col)
        if date_col in ("created_at", "updated_at", "inserted_at"):
            nxt = (d + timedelta(days=1)).isoformat()
            r = q.gte(date_col, d.isoformat()).lt(date_col, nxt).execute()
        else:
            r = q.eq(date_col, d.isoformat()).execute()
        times = sorted({_kst(x[ts_col]) for x in (r.data or []) if x.get(ts_col)})
        quant = any(int(t[:2]) in QUANT_UPLOAD_KST_HOURS for t in times if t != "?")
        return times, quant
    except Exception:
        return [], False


def probe(client, table: str, asof: str) -> dict:
    """테이블 1개를 조사해 최신 날짜와 asof 당일 건수를 돌려준다."""
    out = {"table": table, "status": "", "date_col": "", "latest": "", "today_rows": None,
           "note": "", "times": [], "quant_hour": False}
    try:
        head = client.table(table).select("*").limit(1).execute()
    except Exception as e:
        out["status"] = "ERROR"
        out["note"] = f"조회 실패: {type(e).__name__}: {str(e)[:120]}"
        return out

    if not head.data:
        out["status"] = "EMPTY"
        out["note"] = "테이블 비어 있음(레코드 0건)"
        return out

    cols = list(head.data[0].keys())
    dc = _pick_date_col(cols)
    if dc is None:
        out["status"] = "NO_DATE_COL"
        out["note"] = f"날짜 컬럼 없음 — 보유 컬럼: {','.join(cols[:12])}"
        return out
    out["date_col"] = dc

    try:
        latest = client.table(table).select(dc).order(dc, desc=True).limit(1).execute()
        out["latest"] = _norm(latest.data[0][dc]) if latest.data else ""
    except Exception as e:
        out["status"] = "ERROR"
        out["note"] = f"최신일 조회 실패: {str(e)[:100]}"
        return out

    # asof 당일 건수 — date형은 eq, timestamp형은 [asof, asof+1) 범위
    try:
        d = datetime.strptime(asof, "%Y%m%d").date()
        nxt = (d + timedelta(days=1)).isoformat()
        q = client.table(table).select(dc, count="exact")
        if dc in ("created_at", "updated_at", "inserted_at"):
            res = q.gte(dc, d.isoformat()).lt(dc, nxt).execute()
        else:
            res = q.eq(dc, d.isoformat()).execute()
        out["today_rows"] = res.count if res.count is not None else len(res.data or [])
    except Exception as e:
        out["note"] = f"당일건수 조회 실패: {str(e)[:80]}"
        out["today_rows"] = -1

    if out["today_rows"] and out["today_rows"] > 0:
        out["times"], out["quant_hour"] = _producer_times(client, table, dc, asof)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().strftime("%Y%m%d"),
                    help="검사 기준일 YYYYMMDD (기본: 오늘)")
    args = ap.parse_args()
    asof = args.asof

    client = _client()
    print(f"# 데이터계약(260724) 이행 자가검사 — 기준일 {asof}")
    print(f"# 실행 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} / 읽기전용(SELECT)")
    print()

    violations: list[dict] = []   # 퀀트봇이 적재한 것 = 진짜 계약 위반
    foreign: list[dict] = []      # 타 봇이 적재한 것 = 퀀트봇 소관 밖
    errors: list[dict] = []

    # ── 1) 중단 대상: 당일 신규 레코드가 0이어야 정상 ────────────────
    print(f"## 1. 중단 대상 {len(SUSPENDED_TABLES)}종 — 당일({asof}) 신규 0건이어야 정상")
    print()
    print(f"> 생산자 판별: 퀀트봇 FLOWX 업로드는 **KST {sorted(QUANT_UPLOAD_KST_HOURS)}시대**"
          f"(BAT-D 종반 18:40 + BAT-F 재시도 18:47)에만 일어난다. "
          f"당일 레코드의 적재 시각이 그 시간대 밖이면 **타 봇 적재분**이다.")
    print()
    print("| 테이블 | 날짜컬럼 | 최신일 | 당일건수 | 적재시각(KST) | 판정 |")
    print("|---|---|---|---|---|---|")
    for t in sorted(SUSPENDED_TABLES):
        r = probe(client, t, asof)
        if r["status"] in ("ERROR",):
            errors.append(r)
            verdict = "⚠️ 조회불가"
        elif r["status"] == "EMPTY":
            verdict = "✅ 빈 테이블"
        elif r["status"] == "NO_DATE_COL":
            verdict = "⚠️ 날짜컬럼없음"
        elif r["today_rows"] and r["today_rows"] > 0:
            if r["quant_hour"]:
                violations.append(r)
                verdict = "🔴 **위반(퀀트봇 적재)**"
            else:
                foreign.append(r)
                verdict = "🟠 타봇 적재(퀀트봇분 없음)"
        elif r["today_rows"] == -1:
            verdict = "⚠️ 건수확인불가"
        else:
            verdict = "✅ 차단됨"
        times = ",".join(r["times"]) if r["times"] else "-"
        print(f"| `{t}` | {r['date_col'] or '-'} | {r['latest'] or '-'} | "
              f"{r['today_rows'] if r['today_rows'] is not None else '-'} | {times} | {verdict} |")
        if r["note"]:
            print(f"|  ↳ | | | | | {r['note']} |")
    print()

    # ── 2) 유지 확정: 당일 레코드가 있어야 정상(역방향 검증) ──────────
    print(f"## 2. 유지 확정 {len(KEEP_TABLES)}종 — 당일 레코드가 **있어야** 정상")
    print()
    print("| 테이블 | 날짜컬럼 | 최신일 | 당일건수 | 판정 |")
    print("|---|---|---|---|---|")
    keep_missing: list[str] = []
    for t in KEEP_TABLES:
        r = probe(client, t, asof)
        if r["status"] == "ERROR":
            verdict = "⚠️ 조회불가"
            errors.append(r)
        elif r["today_rows"] and r["today_rows"] > 0:
            verdict = "✅ 정상 적재"
        else:
            verdict = "🟡 당일 없음"
            keep_missing.append(t)
        print(f"| `{t}` | {r['date_col'] or '-'} | {r['latest'] or '-'} | "
              f"{r['today_rows'] if r['today_rows'] is not None else '-'} | {verdict} |")
    print()

    # ── 3) 회색지대: 실측만(판단 대기) ──────────────────────────────
    print(f"## 3. 회색지대 {len(GRAY_TABLES)}종 — 운영자 판단 대기(B-24 ①), 현황 실측만")
    print()
    print("| 테이블 | 날짜컬럼 | 최신일 | 당일건수 | 현황 |")
    print("|---|---|---|---|---|")
    for t in GRAY_TABLES:
        r = probe(client, t, asof)
        state = "적재 중" if (r["today_rows"] or 0) > 0 else "당일 없음"
        if r["status"] == "ERROR":
            state = "조회불가"
        print(f"| `{t}` | {r['date_col'] or '-'} | {r['latest'] or '-'} | "
              f"{r['today_rows'] if r['today_rows'] is not None else '-'} | {state} |")
    print()

    # ── 요약 ────────────────────────────────────────────────────
    print("## 요약")
    print()
    if violations:
        print(f"- 🔴 **퀀트봇 계약 위반 {len(violations)}건** — 퀀트봇 업로드 시각대에 신규 적재됨: "
              + ", ".join(f"`{v['table']}`({v['today_rows']}건, {','.join(v['times'])})"
                          for v in violations))
    else:
        print(f"- ✅ **퀀트봇 적재분 0건** — 중단 {len(SUSPENDED_TABLES)}종 전부에서 "
              f"퀀트봇 업로드 시각대(KST {sorted(QUANT_UPLOAD_KST_HOURS)}시) 레코드 소멸. 차단 정상 동작")
    if foreign:
        print(f"- 🟠 **타 봇 적재 {len(foreign)}건 — 퀀트봇 소관 밖(운영자 조치 필요)**: "
              + ", ".join(f"`{f['table']}`({f['today_rows']}건, KST {','.join(f['times'])})"
                          for f in foreign))
    if keep_missing:
        print(f"- 🟡 유지 대상인데 당일 레코드 없음: {', '.join(f'`{t}`' for t in keep_missing)}")
    else:
        print("- ✅ 유지 확정 테이블 전부 당일 적재 정상")
    if errors:
        print(f"- ⚠️ 조회 실패 {len(errors)}건: "
              + ", ".join(f"`{e['table']}`" for e in errors))
    print()
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
