"""데이터계약(260724) 이행 자가검사 — 중단 테이블 신규 레코드 차단 검증.

용도: 적재 차단(b7ae38b·409ee34) 배포 후, 중단 대상 테이블에 신규 레코드가
      실제로 들어가지 않는지 Supabase 실물로 확인해 운영자·웹봇에 제출한다.

읽기 전용(SELECT만). 쓰기·삭제 없음.

실행:
    python -u -X utf8 scripts/verify_contract_suspension.py [--asof 20260728]
    python -u -X utf8 scripts/verify_contract_suspension.py --alert   # cron 상시화(B-23)
                                                                      # 위반·유지누락 시에만 텔레그램, 평시 무음
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
# BAT-D 종반 FLOWX upload(18:40) + BAT-F 재시도(18:47). 타 봇 실측 적재는 06시·16시대.
# 20~23시 포함(7/30 검수 F5): BAT-D 지연으로 업로드가 20시 이후로 밀리면 미탐되던 구멍.
# 늦은 시각대 오탐은 사람이 확인하면 되는 안전측, 미탐은 아무도 못 본다.
QUANT_UPLOAD_KST_HOURS = {18, 19, 20, 21, 22, 23}


def _kst(ts: str) -> str:
    """UTC ISO 타임스탬프 → KST HH:MM."""
    try:
        h = (int(ts[11:13]) + 9) % 24
        return f"{h:02d}:{ts[14:16]}"
    except Exception:
        return "?"


def _producer_times(client, table: str, date_col: str, asof: str) -> tuple[list[str], bool | None]:
    """당일 레코드의 적재 시각(KST) 분포와 '퀀트봇 시각대 포함' 여부.

    반환 quant: True=퀀트봇 시각대 포함 / False=전부 타 시각대 / **None=판별불가**
    (ts 컬럼 부재·조회 실패). ★None을 False와 구분하는 이유(7/30 검수 F1):
    판별 실패를 '타봇'으로 접으면 진짜 위반이 조용히 미탐된다 — 7/28 오귀속
    사고의 역방향이 코드에 내장되는 꼴.
    """
    ts_col = None
    try:
        head = client.table(table).select("*").limit(1).execute()
        cols = list(head.data[0].keys()) if head.data else []
        for c in ("created_at", "updated_at", "inserted_at"):
            if c in cols:
                ts_col = c
                break
        if ts_col is None:
            return [], None
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
        return [], None


def probe(client, table: str, asof: str) -> dict:
    """테이블 1개를 조사해 최신 날짜와 asof 당일 건수를 돌려준다.

    quant_hour: True=퀀트봇 시각대 / False=타 시각대 / None=판별불가(7/30 검수 F1)
    """
    out = {"table": table, "status": "", "date_col": "", "latest": "", "today_rows": None,
           "note": "", "times": [], "quant_hour": None}
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


def count_on(client, table: str, date_col: str, day: str) -> int:
    """이미 알아낸 date_col로 특정 일자 건수만 재조회 (전일 소급검사용, 실패 시 -1)."""
    try:
        d = datetime.strptime(day, "%Y%m%d").date()
        q = client.table(table).select(date_col, count="exact")
        if date_col in ("created_at", "updated_at", "inserted_at"):
            res = q.gte(date_col, d.isoformat()).lt(
                date_col, (d + timedelta(days=1)).isoformat()).execute()
        else:
            res = q.eq(date_col, d.isoformat()).execute()
        return res.count if res.count is not None else len(res.data or [])
    except Exception:
        return -1


# 값 정합성 검사에서 제외할 컬럼(식별자·메타) — 나머지 수치 컬럼만 본다.
_SANITY_SKIP_COLS = {
    "id", "date", "ticker", "code", "name", "sector", "signal_type", "grade",
    "created_at", "updated_at", "inserted_at", "trade_date", "signal_date",
}
# 이 행수 미만이면 판정하지 않는다 — 1~2행이 우연히 0인 건 흔하다(오탐 억제).
_SANITY_MIN_ROWS = 5


def _const_numeric_fields(rows: list[dict], date_col: str | None = None,
                          table: str = "", prefix: str = "") -> tuple[list[str], list[str]]:
    """행 묶음에서 수치 필드가 통째로 0/null인 것을 찾는다.

    수치가 아닌 값이 하나라도 섞이면 그 필드는 건너뛴다(문자열 상수는 정상일 수
    있다). bool은 수치로 보지 않는다 — False가 전부인 플래그는 흔하고 정상이다.

    ★오탐 억제(8\1): '전량 0'은 **0이 정당한 값**과 **채우지 못한 0**을 구분하지
    못한다(7/31 첫 실행 경고 3건이 전부 오탐). `const_sanity`가 우리 페이로드
    이력·사람 판정으로 걸러낸다. 억제된 것은 버리지 않고 함께 반환해 리포트에
    드러낸다 — 7/30에 봉쇄한 "확인불가를 정상으로 접던" 실패와 같은 부류이기 때문.

    반환: (경고할 항목, 억제된 항목)
    """
    from src.adapters import const_sanity

    out: list[str] = []
    suppressed: list[str] = []
    for col in rows[0]:
        if col in _SANITY_SKIP_COLS or (date_col and col == date_col):
            continue
        vals = [r.get(col) for r in rows]
        if not all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
                   for v in vals):
            continue
        if all(v is None for v in vals):
            desc = f"{col}=전량 null"
        elif all((v or 0) == 0 for v in vals):
            desc = f"{col}=전량 0"
        else:
            continue
        if table:
            warn, why = const_sanity.should_warn(table, prefix + col)
            if not warn:
                suppressed.append(f"{prefix}{col}({why})")
                continue
        out.append(desc)
    return out, suppressed


def _batch_groups(rows: list[dict], ts_col: str) -> list[tuple[str, list[dict]]]:
    """적재 시각(분 단위)으로 배치를 가른다.

    ★왜 배치인가(7\31 실측): "전 행 0" 규칙은 **같은 날 같은 테이블에 여러 봇이
    쓰면 서로의 실값에 가려 미탐된다**. 7/30 `dashboard_smart_money`는 18시대에
    퀀트봇 42행만 있어서 잡혔지만, 7/29는 정보봇이 19시대에 193행(실값)을 넣어
    퀀트봇 38행의 전량 0이 묻혔다 — **7/30에 잡힌 건 운이었다.**
    같은 cron 실행은 created_at이 초 단위로 뭉치므로 분 단위로 가르면
    생산자별 배치가 자연히 분리되고, upsert가 created_at을 갱신하지 않는
    문제와도 무관해진다(배치가 나뉘어 보일 뿐이다).
    """
    from collections import defaultdict
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ts = r.get(ts_col)
        g[_kst(str(ts)) if ts else "?"].append(r)
    return sorted(g.items())


def _record_groups(payload) -> list[tuple[str, list[dict]]]:
    """JSON payload에서 '행 묶음'을 뽑는다 — `list[dict]` 또는 `dict[str, dict]`.

    ★B-26 ②: 유지 3종(scenario_dashboard·leader_cycle·sector_flow)은 하루 **1행**이고
    실제 값이 `data` JSON 안에 배열/맵으로 들어 있다. 행 단위 상수 판정이 성립하지
    않아 §2.5 1차에서는 통째로 미검사였다 — 유지 5종 중 2종만 보고 있었다는 뜻이다.
    배열을 펼쳐 같은 판정을 걸면 나머지 3종도 사각에서 나온다.
    """
    groups: list[tuple[str, list[dict]]] = []
    if not isinstance(payload, dict):
        return groups
    for k, v in payload.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            groups.append((k, v))
        elif isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            groups.append((k, list(v.values())))
    return groups


def value_sanity(client, table: str, date_col: str, asof: str) -> tuple[list[str], str]:
    """유지 테이블의 당일 값이 **거짓 상수**(수치 컬럼이 통째로 0/null)인지 본다. (B-26 ①)

    ★왜 필요한가(7\31 실사건): `dashboard_smart_money.price/change_pct`가 0으로
    하드코딩돼 6주간 매일 적재됐고, 웹 `/smart-money`가 score 상위 50을 읽는 구조라
    화면이 4거래일 내내 공란이었다. 기존 자가검사는 **"적재됐는가"만 보고 "값이
    맞는가"를 안 봐서** 전혀 못 잡았고, 웹봇이 화면을 보고 알려줘서 알았다.

    ★공동적재 테이블은 퀀트봇 행만 걸러야 한다. 그날 실측이 정확히 그랬다 —
    60행 중 퀀트봇 50행만 price=0이고 정보봇 10행은 실값이라, 전체를 보면
    "전 행 0"이 성립하지 않아 미탐된다. 생산자 판별(적재 시각대)을 재사용한다.

    반환: (의심 컬럼 설명 목록, 판정 근거 메모)
    """
    try:
        d = datetime.strptime(asof, "%Y%m%d").date()
        q = client.table(table).select("*")
        if date_col in ("created_at", "updated_at", "inserted_at"):
            q = q.gte(date_col, d.isoformat()).lt(date_col, (d + timedelta(days=1)).isoformat())
        else:
            q = q.eq(date_col, d.isoformat())
        rows = (q.limit(1000).execute().data) or []
    except Exception as e:  # noqa: BLE001
        return [], f"조회실패({str(e)[:40]})", False

    if not rows:
        return [], "당일 0행", False

    ts_col = next((c for c in ("created_at", "updated_at", "inserted_at")
                   if c in rows[0]), None)

    findings: list[str] = []
    notes: list[str] = []
    suppressed: list[str] = []
    checked = False

    # (A) 행 단위 — **적재 배치별로** 본다(플랫 컬럼 테이블: smart_money·valuation_gap).
    #
    # ★생산자 시각대로 거르지 않는 이유(7\31 실측 2건):
    #   ⑴ `created_at`은 행이 **처음 만들어진** 시각이라 `on_conflict` upsert로
    #      우리가 덮어써도 갱신되지 않는다. `quant_sector_flow`는 퀀트봇이 18:40:33·
    #      18:50:55에 POST 200을 받았는데 created_at은 4일 내내 16:56이다 —
    #      "퀀트봇 시각대 행 0"이 "우리가 안 썼다"를 뜻하지 않는다.
    #   ⑵ 시각대로 거르고 남은 것을 한 덩어리로 보면, 같은 시간대에 타 봇 실값이
    #      섞였을 때 우리 배치의 전량 0이 묻힌다(7/29 정보봇 19시 193행이
    #      퀀트봇 38행을 덮어 미탐. 7/30에 잡힌 건 우연히 우리만 있었기 때문).
    # 분 단위 배치로 가르면 둘 다 해소되고, 어느 배치가 문제인지 시각으로 지목된다.
    if ts_col:
        batches = [(bt, br) for bt, br in _batch_groups(rows, ts_col)
                   if len(br) >= _SANITY_MIN_ROWS]
        if batches:
            for bt, br in batches:
                f_out, f_sup = _const_numeric_fields(br, date_col, table)
                findings += [f"[{bt}] {f}" for f in f_out]
                suppressed += f_sup
            notes.append(f"{len(rows)}행/배치 {len(batches)}개("
                         + ",".join(f"{bt}×{len(br)}" for bt, br in batches) + ")")
            checked = True
        else:
            notes.append(f"{len(rows)}행 — 배치별 최소 {_SANITY_MIN_ROWS}행 미만")
    elif len(rows) >= _SANITY_MIN_ROWS:
        f_out, f_sup = _const_numeric_fields(rows, date_col, table)
        findings += f_out
        suppressed += f_sup
        notes.append(f"전체 {len(rows)}행(적재시각 컬럼 없음)")
        checked = True
    else:
        notes.append(f"전체 {len(rows)}행(행단위 판정 최소 {_SANITY_MIN_ROWS} 미만)")

    # (B) JSON `data` 내부 행 묶음 — 하루 1행 3종(B-26 ②)
    import json as _json
    checked_groups = 0
    for r in rows:
        payload = r.get("data")
        if isinstance(payload, str):
            try:
                payload = _json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
        for gname, grows in _record_groups(payload):
            if len(grows) < _SANITY_MIN_ROWS:
                continue  # 표본 부족 — 조용히 넘기되 아래 집계에서 빠진다
            checked_groups += 1
            f_out, f_sup = _const_numeric_fields(grows, None, table, f"data.{gname}.")
            findings += [f"data.{gname}.{f}" for f in f_out]
            suppressed += f_sup
    if checked_groups:
        notes.append(f"data 배열 {checked_groups}묶음")
        checked = True

    # 억제분은 감추지 않는다 — 무엇을 왜 넘겼는지가 리포트에 남아야 한다.
    if suppressed:
        notes.append(f"억제 {len(suppressed)}건: " + ", ".join(suppressed[:4])
                     + (" 외" if len(suppressed) > 4 else ""))

    return findings, " / ".join(notes), checked


def _is_trading_day(asof: str) -> bool:
    """휴장일 판정 (B-13 어댑터 재사용). 판정 불가 시 장날로 폴백 — 경보를 잠재우는 쪽이 더 위험."""
    try:
        from src.adapters.kis_holiday_adapter import is_trading_day
        return is_trading_day(datetime.strptime(asof, "%Y%m%d").date())
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 휴장일 판정 실패 — 장날로 간주: {e}")
        return True


def _send_alert(violations: list[dict], keep_missing: list[str], asof: str,
                unknowns: list[dict] | None = None,
                keep_unknown: list[str] | None = None,
                errors: list[dict] | None = None,
                prev_violations: list[dict] | None = None,
                prev_asof: str = "",
                stale_values: list[str] | None = None) -> None:
    """위반·유지누락·판별불가·검사불능 시에만 텔레그램 발송. 평시 무음(로그만).

    - 위반(퀀트봇 시각대 적재)은 휴장 여부와 무관하게 알람 — 있어선 안 되는 레코드다.
    - 유지 테이블 당일 누락·확인불가는 장날에만 알람 — 휴장일엔 업로드가 없는 게 정상.
    - 조회 실패 2건 이상이면 "검사 불완전" 알람(7/30 검수 F3: 키 회전 등 전면 장애 시
      violations·keep_missing이 모두 비어 완전 침묵하던 구멍).
    - "[HEALTH]" 선두 태그: 텔레그램 QUIET 모드 허용 목록 통과용.
    """
    unknowns = unknowns or []
    keep_unknown = keep_unknown or []
    errors = errors or []
    prev_violations = prev_violations or []
    stale_values = stale_values or []

    lines: list[str] = []
    if violations:
        lines.append(f"[HEALTH] 🔴 데이터계약 자가검사 — 퀀트봇 위반 {len(violations)}건 ({asof})")
        lines += [f"• {v['table']}: {v['today_rows']}건, KST {','.join(v['times'])}"
                  for v in violations]
    if prev_violations:
        lines.append(f"[HEALTH] 🔴 전일({prev_asof}) 소급 위반 {len(prev_violations)}건 — 검사시각 이후 적재분")
        lines += [f"• {v['table']}: {v['today_rows']}건, KST {','.join(v['times'])}"
                  for v in prev_violations]
    if unknowns:
        lines.append(f"[HEALTH] 🟠 중단 테이블 적재有·생산자 판별불가 {len(unknowns)}건 ({asof}): "
                     + ", ".join(u["table"] for u in unknowns))

    trading = None
    if keep_missing or keep_unknown:
        trading = _is_trading_day(asof)
    if keep_missing:
        if trading:
            lines.append(f"[HEALTH] 🟡 계약 유지 테이블 당일({asof}) 적재 누락: "
                         + ", ".join(keep_missing))
        else:
            print("[INFO] 휴장일 — 유지 테이블 당일 없음은 정상, 알람 생략")
    if keep_unknown and trading:
        lines.append(f"[HEALTH] 🟠 유지 테이블 적재 확인불가({asof}): " + ", ".join(keep_unknown))
    if stale_values:
        # 휴장 게이트 불필요 — 당일 행이 없으면 애초에 findings가 안 나온다.
        lines.append(f"[HEALTH] 🟡 유지 테이블 값 거짓 상수 의심 {len(stale_values)}건 ({asof}): "
                     + ", ".join(stale_values))
    if len(errors) >= 2:
        lines.append(f"[HEALTH] ⚠️ 자가검사 조회실패 {len(errors)}건 — 검사 불완전: "
                     + ", ".join(f"`{e['table']}`" for e in errors[:8]))

    if not lines:
        print("[alert] 위반·누락 없음 — 발송 생략(무음)")
        return
    try:
        from src.telegram_sender import send_message
        ok = send_message("\n".join(lines))
        print(f"[alert] 텔레그램 발송 {'OK' if ok else '실패'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 텔레그램 발송 실패: {e}")


def _default_asof() -> str:
    """기본 기준일 — 자정~06시 실행은 전일로 앵커(7/30 검수 F6).

    BAT-D(16:30 시작)가 크게 지연돼 자정을 넘겨 실행되면 date.today()가 다음날이 되어
    ①유지 5종 전면 오탐 ②당일 위반 전면 미탐으로 감시가 뒤집힌다. 하필 파이프라인이
    고장난 날에 그렇게 되므로, 새벽 실행은 직전 영업분으로 본다.
    """
    now = datetime.now()
    d = now.date() - timedelta(days=1) if now.hour < 6 else now.date()
    return d.strftime("%Y%m%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=_default_asof(),
                    help="검사 기준일 YYYYMMDD (기본: 오늘, 00~06시 실행은 전일)")
    ap.add_argument("--alert", action="store_true",
                    help="위반·유지누락 시 텔레그램 [HEALTH] 알람 (cron 상시화용, 평시 무음)")
    ap.add_argument("--sweep-prev", action="store_true",
                    help="전일분 중단테이블 소급 재검사 (검사시각 이후 적재분 포착, 7/30 검수 F8)")
    args = ap.parse_args()
    asof = args.asof

    client = _client()
    print(f"# 데이터계약(260724) 이행 자가검사 — 기준일 {asof}")
    print(f"# 실행 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} / 읽기전용(SELECT)")
    print()

    violations: list[dict] = []   # 퀀트봇이 적재한 것 = 진짜 계약 위반
    foreign: list[dict] = []      # 타 봇이 적재한 것 = 퀀트봇 소관 밖
    unknowns: list[dict] = []     # 적재는 있는데 생산자 판별 불가 = 위반 배제 못 함
    errors: list[dict] = []
    date_cols: dict[str, str] = {}   # 전일 소급검사에서 재사용

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
        if r["date_col"]:
            date_cols[t] = r["date_col"]
        if r["status"] in ("ERROR",):
            errors.append(r)
            verdict = "⚠️ 조회불가"
        elif r["status"] == "EMPTY":
            verdict = "✅ 빈 테이블"
        elif r["status"] == "NO_DATE_COL":
            verdict = "⚠️ 날짜컬럼없음"
        elif r["today_rows"] and r["today_rows"] > 0:
            if r["quant_hour"] is True:
                violations.append(r)
                verdict = "🔴 **위반(퀀트봇 적재)**"
            elif r["quant_hour"] is None:
                # 판별 실패를 '타봇'으로 접으면 진짜 위반이 미탐된다(F1) → 별도 분류·알람
                unknowns.append(r)
                verdict = "🟠 적재有·생산자 **판별불가**"
            else:
                foreign.append(r)
                verdict = "🟠 타봇 적재(퀀트봇분 없음)"
        elif r["today_rows"] == -1:
            errors.append(r)
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
    keep_unknown: list[str] = []
    keep_date_cols: dict[str, str] = {}
    for t in KEEP_TABLES:
        r = probe(client, t, asof)
        if r["date_col"]:
            keep_date_cols[t] = r["date_col"]
        if r["status"] == "ERROR":
            verdict = "⚠️ 조회불가"
            errors.append(r)
            keep_unknown.append(t)
        elif r["today_rows"] == -1:
            # 건수 쿼리만 실패한 경우 — '누락'으로 접으면 오탐 알람(F4). 별도 분류
            verdict = "⚠️ 건수확인불가"
            errors.append(r)
            keep_unknown.append(t)
        elif r["today_rows"] and r["today_rows"] > 0:
            verdict = "✅ 정상 적재"
        else:
            verdict = "🟡 당일 없음"
            keep_missing.append(t)
        print(f"| `{t}` | {r['date_col'] or '-'} | {r['latest'] or '-'} | "
              f"{r['today_rows'] if r['today_rows'] is not None else '-'} | {verdict} |")
    print()

    # ── 2.5) 유지 테이블 값 정합성 — 거짓 상수 탐지 (B-26 ①) ──────────
    print("## 2.5 유지 테이블 값 정합성 — 수치 컬럼이 통째로 0/null인가")
    print()
    print("> 적재 여부만 보면 `price=0`을 6주간 매일 내보내도 ✅로 보인다"
          "(7\\31 dashboard_smart_money 실사건). 공동적재 테이블은 **퀀트봇 적재분만**"
          " 걸러서 본다 — 타 봇 실값이 섞이면 '전량 0'이 성립하지 않아 미탐된다."
          " 하루 1행 JSON 테이블은 `data` 안의 배열/맵을 펼쳐서 같은 판정을 건다(B-26 ②).")
    print()
    print("| 테이블 | 검사범위 | 의심 컬럼 | 판정 |")
    print("|---|---|---|---|")
    stale_values: list[str] = []
    sanity_unchecked: list[str] = []
    for t in KEEP_TABLES:
        dc = keep_date_cols.get(t)
        if not dc:
            print(f"| `{t}` | - | - | ⚠️ 날짜컬럼없음 |")
            sanity_unchecked.append(t)
            continue
        findings, note, checked = value_sanity(client, t, dc, asof)
        if findings:
            stale_values.append(f"{t}({', '.join(findings)})")
            verdict = "🟡 **거짓 상수 의심**"
        elif note.startswith("조회실패"):
            verdict = "⚠️ 조회불가"
            sanity_unchecked.append(t)
        elif not checked:
            # ★"검사하지 않았다"를 ✅로 찍으면 안 된다 — 유지 5종을 매일 값까지 본다는
            # 착각을 주고, 영구 미검사가 조용히 굳는다(7\30 F4와 같은 실패 모드).
            verdict = "➖ 미검사"
            sanity_unchecked.append(t)
        else:
            verdict = "✅ 정상"
        print(f"| `{t}` | {note} | {', '.join(findings) or '-'} | {verdict} |")
    print()
    if sanity_unchecked:
        print(f"> ➖ 값 검사 미적용 {len(sanity_unchecked)}종: "
              + ", ".join(f"`{t}`" for t in sanity_unchecked)
              + " — 표본이 최소행수 미만이거나 조회 불가. 검사한 종수는 아래 요약 참조.")
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

    # ── 3.5) 전일 소급 재검사 (F8: 검사시각 이후 적재분은 영구 사각) ──────
    prev_violations: list[dict] = []
    prev_asof = ""
    if args.sweep_prev and date_cols:
        prev_asof = (datetime.strptime(asof, "%Y%m%d").date()
                     - timedelta(days=1)).strftime("%Y%m%d")
        print(f"## 3.5 전일({prev_asof}) 소급 재검사 — 어제 검사시각 이후 적재분 포착")
        print()
        print("| 테이블 | 전일건수 | 적재시각(KST) | 판정 |")
        print("|---|---|---|---|")
        for t in sorted(date_cols):
            n = count_on(client, t, date_cols[t], prev_asof)
            if n <= 0:
                continue
            times, quant = _producer_times(client, t, date_cols[t], prev_asof)
            if quant is True:
                r = {"table": t, "today_rows": n, "times": times}
                prev_violations.append(r)
                verdict = "🔴 **소급 위반**"
            elif quant is None:
                verdict = "🟠 판별불가"
            else:
                verdict = "🟠 타봇"
            print(f"| `{t}` | {n} | {','.join(times) or '-'} | {verdict} |")
        if not prev_violations:
            print("|  | | | (퀀트봇 소급 위반 없음) |")
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
    if unknowns:
        print(f"- 🟠 **생산자 판별불가 {len(unknowns)}건 — 위반 배제 못 함(수동 확인 필요)**: "
              + ", ".join(f"`{u['table']}`({u['today_rows']}건)" for u in unknowns))
    if prev_violations:
        print(f"- 🔴 **전일({prev_asof}) 소급 위반 {len(prev_violations)}건**: "
              + ", ".join(f"`{v['table']}`({v['today_rows']}건, {','.join(v['times'])})"
                          for v in prev_violations))
    if keep_missing:
        print(f"- 🟡 유지 대상인데 당일 레코드 없음: {', '.join(f'`{t}`' for t in keep_missing)}")
    elif not keep_unknown:
        print("- ✅ 유지 확정 테이블 전부 당일 적재 정상")
    if keep_unknown:
        print(f"- 🟠 유지 테이블 적재 확인불가: {', '.join(f'`{t}`' for t in keep_unknown)}")
    if stale_values:
        print(f"- 🟡 **유지 테이블 값 거짓 상수 의심 {len(stale_values)}건** — 적재는 됐으나 "
              f"수치가 통째로 0/null: {', '.join(f'`{s}`' for s in stale_values)}")
    elif len(sanity_unchecked) < len(KEEP_TABLES):
        print(f"- ✅ 값 정합성 검사한 {len(KEEP_TABLES) - len(sanity_unchecked)}종에서 거짓 상수 없음"
              + (f" (미검사 {len(sanity_unchecked)}종)" if sanity_unchecked else ""))
    else:
        print(f"- ➖ 값 정합성: 유지 {len(KEEP_TABLES)}종 **전부 미검사** — 검사가 아무것도 보지 않았다")
    if errors:
        print(f"- ⚠️ 조회 실패 {len(errors)}건 — **검사 불완전**: "
              + ", ".join(f"`{e['table']}`" for e in errors))
    print()

    if args.alert:
        _send_alert(violations, keep_missing, asof,
                    unknowns=unknowns, keep_unknown=keep_unknown, errors=errors,
                    prev_violations=prev_violations, prev_asof=prev_asof,
                    stale_values=stale_values)

    # exit 1 조건: 퀀트봇 위반(당일·소급) 또는 판별불가 또는 검사 불완전(2건+)
    # — 침묵하며 exit 0 하던 F1·F3 구멍 봉쇄
    if violations or prev_violations or unknowns or len(errors) >= 2:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
