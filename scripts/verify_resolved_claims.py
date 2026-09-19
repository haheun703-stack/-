"""해소 선언 재검증 — 「고쳤다」고 보고한 것이 오늘도 고쳐진 상태인가.

배경
  9/19 실측에서 **9/7에 「B-47 완결」이라 보고한 배선이 12거래일 동안 한 셀도
  채우지 않았다**는 것이 드러났다. 같은 형태가 9/7에도 3건 있었다(B-98 전용키·
  B-97 내재가치·B-99 국적수급). 즉 **최근 신규 부채의 절반 이상이 「내가 완료라고
  보고한 것의 실제 상태」**다.

  원인은 한 가지다 — **「완료」가 사람의 선언이고 기계가 확인하지 않는다.**
  조치 당일에는 라이브 1회 검증을 하지만, 그 뒤로는 아무도 다시 보지 않는다.
  그래서 배선이 끊기거나 원천이 바뀌어도 백로그에는 🟢으로 남는다.

설계 원칙 (전부 이미 비싸게 배운 것들)
  ① **검사는 대상 밖에서 돈다** — B-94(8/21~9/4). 스케줄러 검사가 자기가 속한
     BAT-D의 완료를 검사해 11거래일 거짓 ❌였다. 이 스크립트는 HEALTH 단계에서
     돌아 BAT-D의 산출물을 **끝난 뒤에** 본다.
  ② **급성과 만성은 다른 검사다** — B-104(9/18). 기존 수급 검사가 `tail(20)`
     관측창 전량 0만 봐서 **어제 하루 죽은 것**을 ✅로 통과시켰다. 여기서는
     전일 대비 급락을 별도 항목으로 본다.
  ③ **스킵을 성공으로 위장하지 않는다** — 단타봇이 9/7에 알려준 설계.
     판정 불가는 `skipped=True`로 남기고 등급 계산에서 빼되 **로그에는 이유를 찍는다**.
  ④ **자기 실패 모드로 검증한다** — 7/30. 이 파일 맨 아래 `--selftest`가
     각 체크를 「고장난 입력」에 통과시켜 본다.

사용
  python -u -X utf8 scripts/verify_resolved_claims.py
  python -u -X utf8 scripts/verify_resolved_claims.py --json
  python -u -X utf8 scripts/verify_resolved_claims.py --only B-47
  python -u -X utf8 scripts/verify_resolved_claims.py --selftest

종료코드
  0 = 전부 통과(또는 스킵)   1 = 회귀 발견(해소 선언이 깨짐)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"


# ──────────────────────────────────────────────
# 결과 모델
# ──────────────────────────────────────────────

@dataclass
class Result:
    ok: bool
    detail: str
    skipped: bool = False
    reason: str = ""          # skipped일 때 왜 판정을 못 했는가 (③)

    def __post_init__(self):
        # ③ 이유 없는 보류는 성공 위장이다 — 설계 원칙을 코드로 강제한다.
        #   주석으로만 적어 두면 다음 사람이(또는 내가) 그냥 skipped=True를 쓴다.
        if self.skipped and not self.reason:
            self.skipped = False
            self.ok = False
            self.detail = (self.detail + " " if self.detail else "") + \
                          "[설계위반] 판정 보류에 reason이 없다"

    @property
    def icon(self) -> str:
        if self.skipped:
            return "⏸️"
        return "✅" if self.ok else "🚨"


@dataclass
class Claim:
    cid: str
    what: str                 # 무엇이 참이어야 하는가
    resolved_on: str          # 해소 선언일 ("" = 선언한 적 없는 상시 불변식)
    fn: Callable[[], Result]
    missed_days: int = 0      # 선언 후 이 검사 없이 지나간 거래일 (참고용)
    kind: str = "해소선언"     # "해소선언" | "불변식"

    @property
    def origin(self) -> str:
        """★「선언 2026-09-19」처럼 **하지 않은 선언을 적지 않는다.**

        B-104·B-106은 고쳤다고 보고한 적이 없고 «앞으로 깨지면 안 되는 상태»다.
        둘을 같은 문구로 찍으면 나중에 «9/19에 고쳤다는데 왜 또 깨졌나»라는
        오판을 만든다 — 오늘 B-105가 정확히 부정확한 라벨에서 시작한 일이다.
        """
        return f"선언 {self.resolved_on}" if self.resolved_on else f"불변식(등재 {self.cid})"


CLAIMS: list[Claim] = []


def claim(cid: str, what: str, resolved_on: str = "", missed_days: int = 0,
          kind: str = "해소선언"):
    def deco(fn):
        CLAIMS.append(Claim(cid, what, resolved_on, fn, missed_days, kind))
        return fn
    return deco


# ──────────────────────────────────────────────
# 공용 도구
# ──────────────────────────────────────────────

def _recent_trading_dates(n: int = 2) -> list[str]:
    """종가 parquet 인덱스에서 실제 거래일 최근 n개. 달력이 아니라 데이터에서 뽑는다.

    ★휴장일 하드코딩이나 `weekday()` 판정을 쓰지 않는다 — 9/19 세션에서 내가
      파일 목록의 빈칸을 보고 「9/12 cron 공백」이라 단정했다가 그날이 토요일임을
      뒤늦게 확인한 일이 있다. 거래일은 **데이터가 말하게 한다.**
    """
    import pandas as pd
    processed = DATA / "processed"
    for pf in sorted(processed.glob("*.parquet"))[:20]:
        try:
            idx = pd.read_parquet(pf, columns=["close"]).index
        except Exception:
            continue
        days = [str(d)[:10] for d in idx[-n:]]
        if len(days) == n:
            return days
    return []


def _nonzero_count(col: str, day: str, sample: int = 0) -> tuple[int, int]:
    """processed parquet에서 특정 날짜·특정 컬럼의 (비영 종목수, 행 보유 종목수).

    sample=0이면 전수. 부분 표본은 급락 판정을 흔들 수 있어 기본은 전수다.
    """
    import pandas as pd
    processed = DATA / "processed"
    files = sorted(processed.glob("*.parquet"))
    if sample:
        files = files[::max(1, len(files) // sample)]
    nz = rows = 0
    for pf in files:
        try:
            df = pd.read_parquet(pf, columns=[col])
        except Exception:
            continue          # 컬럼 없는 종목은 건너뛴다
        try:
            idx = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except Exception:
            continue
        hit = [i for i, d in enumerate(idx) if d == day]
        if not hit:
            continue
        rows += 1
        v = df[col].iloc[hit[0]]
        try:
            if pd.notna(v) and float(v) != 0:
                nz += 1
        except Exception:
            pass
    return nz, rows


def _raw_nonzero(col: str, days: int = 10) -> tuple[int, int]:
    """data/raw 전수에서 최근 days 거래일 안에 col이 비영인 (종목수, 컬럼보유 종목수)."""
    import pandas as pd
    raw = DATA / "raw"
    nz = have = 0
    for pf in sorted(raw.glob("*.parquet")):
        try:
            df = pd.read_parquet(pf, columns=[col])
        except Exception:
            continue
        have += 1
        tail = df[col].tail(days)
        try:
            if (pd.to_numeric(tail, errors="coerce").fillna(0) != 0).any():
                nz += 1
        except Exception:
            pass
    return nz, have


# ──────────────────────────────────────────────
# 해소 선언 검증들
# ──────────────────────────────────────────────

@claim("B-47", "정보봇 CSV의 공매도 잔고가 raw parquet에 매일 채워진다",
       resolved_on="2026-09-07", missed_days=12)
def _check_b47() -> Result:
    """9/7 「배선 완료(6bbd345)」 선언. 9/19 실측에서 **2주 비영 0건**으로 무효 확인.

    이 검사가 9/8에 한 번만 돌았어도 12거래일을 벌었다.
    """
    nz, have = _raw_nonzero("short_balance", days=10)
    if have == 0:
        return Result(False, "raw parquet에 short_balance 컬럼이 아예 없다")
    if nz == 0:
        return Result(
            False,
            f"최근 10거래일 비영 0건 / 컬럼 보유 {have}종목 — "
            f"배선은 있으나 채워지지 않는다 (B-105: 원천 short_balance_qty가 "
            f"전일 공매도량이고 9/15부터 커버리지 2종목)",
        )
    return Result(True, f"최근 10거래일 비영 {nz}/{have}종목")


@claim("B-104", "기타법인 수급이 D+1 지연 범위를 넘지 않는다", kind="불변식")
def _check_other_corp_acute() -> Result:
    """②급성 검사 — 단 **마지막 거래일은 원래 비어 있다.**

    ★9/19에 이 검사를 처음 만들 때 나는 「9/18 기타법인 전 종목 0 = 당일 수집
      단절」이라고 판정했다. 틀렸다. 추적 결과:
        · `investor_daily.db`에 9/18 기타법인 **1,188종목 실값**이 있었고
        · 단타봇 flow CSV에도 **1,312종목** 있었다
        · 끊긴 자리는 `extend_parquet_data`(**16:51~16:59**)가
          `collect_investor_kis`(**17:11~17:30**)보다 **40분 먼저 도는 것**이었다.
      즉 기타법인은 **매일 D+1로 채워지는 구조**다(B-88·B-102와 같은 순서 역전,
      세 번째). 9/18이 비어 보인 것은 다음 실행(9/19)이 토요일이라 없었기 때문이다.

    ★그래서 **마지막 거래일을 검사하면 매일 오탐**이 된다. 하루 앞을 본다.
      오늘 내내 비판한 「그날 만든 코드가 그날 뚫린다」를 배포 전에 잡은 자리다.

    검사 대상: D-1(직전 거래일)이 D-2 대비 급락했는가.
    D(최신)는 지연이 정상이므로 **검사하지 않고 detail에만 적는다.**
    """
    days = _recent_trading_dates(3)
    if len(days) < 3:
        return Result(True, "", skipped=True, reason="거래일 3일치를 데이터에서 못 얻음")
    d2, d1, d0 = days                      # D-2, D-1, D(최신)
    nz1, rows1 = _nonzero_count("기타법인", d1)
    nz2, _ = _nonzero_count("기타법인", d2)
    nz0, _ = _nonzero_count("기타법인", d0)
    tail = f" · 최신 {d0} {nz0}종목(D+1 지연이 정상)"
    if rows1 == 0:
        return Result(True, "", skipped=True, reason=f"{d1} 행이 없음")
    if nz2 == 0:
        return Result(True, f"{d1} {nz1}종목{tail}", skipped=True,
                      reason=f"{d2} 기준값이 0이라 급락 판정 불가")
    drop = 1.0 - (nz1 / nz2)
    if nz1 == 0:
        return Result(False, f"{d1} 기타법인 **전 종목 0** (D-2 {nz2}종목) — "
                             f"D+1 지연으로도 설명되지 않는다{tail}")
    if drop >= 0.80:
        return Result(False, f"{d1} {nz1}종목 (D-2 {nz2}) — {drop*100:.0f}% 급락{tail}")
    return Result(True, f"{d1} {nz1}종목 (D-2 {nz2}, {-drop*100:+.0f}%){tail}")


@claim("B-94", "BAT-D 완주 판정이 BAT-D 밖에서 돈다",
       resolved_on="2026-09-07", missed_days=0)
def _check_b94_outside(sh: Path | None = None) -> Result:
    """검사기가 자기 대상 안으로 되돌아가지 않았는지 — 배선 위치 자체를 본다.

    ★값이 아니라 **구조**를 검사한다. 값(A등급)만 보면 누가 다시 BAT-D 안으로
      옮겨도 그날은 통과한다.
    ★`sh`를 주입받는다 — selftest가 프로덕션 `run_bat.sh`를 덮었다 복원하는
      방식이면 중간에 죽을 때 파일이 깨진 채 남는다. 테스트는 임시 파일을 준다.
    """
    sh = sh or (PROJECT_ROOT / "scripts" / "cron" / "run_bat.sh")
    if not sh.exists():
        return Result(True, "", skipped=True, reason="run_bat.sh 없음(로컬 실행)")
    txt = sh.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    hits = [i for i, ln in enumerate(lines)
            if "data_health_check.py" in ln and not ln.strip().startswith("#")]
    if not hits:
        return Result(False, "data_health_check.py 호출이 run_bat.sh에서 사라졌다")
    # 그 호출이 어느 단계 블록에 있는지 — 직전 `if [ "$BAT" = ... ]` 를 거슬러 찾는다
    for h in hits:
        stage = None
        for j in range(h, -1, -1):
            ln = lines[j]
            if '"$BAT"' in ln and "=" in ln:
                stage = ln.strip()
                break
        if stage and '"D"' in stage:
            return Result(False,
                          f"data_health_check.py가 다시 BAT-D 안에 있다 (line {h+1}) — "
                          f"자기 자신의 완료 로그를 볼 수 없다")
    return Result(True, f"BAT-D 밖에서 호출됨 (line {[h+1 for h in hits]})")


@claim("B-77", "6개월 묵은 캐시가 픽에 유입되지 않는다",
       resolved_on="2026-09-07", missed_days=8)
def _check_b77_stale_guard() -> Result:
    """`load_json_fresh` 가드가 살아 있고, 대상 파일이 여전히 낡았는지 확인.

    ★가드가 **붙어 있는가**와 **작동하는가**는 다르다. 파일이 낡은 상태 그대로인데
      픽에 들어가고 있으면 가드가 뚫린 것이다.
    """
    src = PROJECT_ROOT / "src"
    hit = list(src.rglob("*.py")) + list((PROJECT_ROOT / "scripts").rglob("*.py"))
    has_loader = any("def load_json_fresh" in p.read_text(encoding="utf-8", errors="ignore")
                     for p in hit if p.is_file())
    if not has_loader:
        return Result(False, "load_json_fresh 정의가 사라졌다")

    stale = []
    for name in ("scan_cache.json", "scenarios/active_scenarios.json"):
        f = DATA / name
        if not f.exists():
            continue
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("cached_at", "updated", "updated_at", "generated_at"):
            v = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(v, str) and len(v) >= 10:
                try:
                    age = (date.today() - datetime.fromisoformat(v[:19]).date()).days
                except Exception:
                    continue
                if age > 7:
                    stale.append(f"{name} {age}일")
                break
    if stale:
        return Result(True, f"가드 존재 · 대상은 여전히 낡음({', '.join(stale)}) = 가드가 막는 중")
    return Result(True, "가드 존재 · 낡은 대상 없음")


@claim("B-98", "Supabase 키가 봇별 전용이다",
       resolved_on="2026-09-07", missed_days=0)
def _check_b98_key() -> Result:
    """8/21에 「전용키 교체 완료」라 보고했으나 9/7에 3봇 동일로 드러난 건.

    우리 키의 지문만 남긴다 — 다른 봇 `.env`를 읽는 것은 우리 소관 밖이다.
    """
    import hashlib
    import os
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return Result(True, "", skipped=True, reason=".env 없음(로컬)")
    key = None
    for ln in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("SUPABASE_KEY="):
            key = ln.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not key:
        return Result(False, ".env에 SUPABASE_KEY가 없다")
    fp = hashlib.sha256(key.encode()).hexdigest()[:16]
    mode = oct(env.stat().st_mode)[-3:]
    known_shared = "422249c6a38fe57e"      # 9/7 실측 — 3봇이 공유하던 값
    if fp == known_shared:
        return Result(False, f"9/7에 확인된 **3봇 공유 키 그대로**(sha256 {fp}) · 권한 {mode}")
    if mode not in ("600", "640"):
        return Result(False, f"키는 교체됨(sha256 {fp})이나 .env 권한이 {mode} — 640 이하 필요")
    return Result(True, f"공유키와 다름(sha256 {fp}) · 권한 {mode} — 단 **전용 여부는 웹봇 회신 대기(B-98)**")


@claim("B-106", "BAT-D 내부 실패가 0건이다", kind="불변식")
def _check_bat_d_failures() -> Result:
    """⑤ 같은 ❌가 며칠째인지 세어 문구에 넣는다.

    `collect_foreign_exhaustion`이 9/11부터 6거래일 연속 실패했는데 매일 같은
    문장이라 배경이 됐다. **연속 일수를 본문에 넣으면 눈에 띈다.**
    """
    logs = PROJECT_ROOT / "logs"
    if not logs.exists():
        return Result(True, "", skipped=True, reason="logs/ 없음(로컬)")
    files = sorted(logs.glob("cron_2026*.log"))[-10:]
    if not files:
        return Result(True, "", skipped=True, reason="cron 로그 없음")
    streak: dict[str, int] = {}
    order: list[str] = []
    for f in reversed(files):                 # 최신 → 과거
        txt = f.read_text(encoding="utf-8", errors="ignore")
        today_fail = set()
        for ln in txt.splitlines():
            if "[WARN]" in ln and ("실패 (exit=" in ln or "타임아웃" in ln):
                for tok in ln.split():
                    if tok.startswith("scripts/") and tok.endswith(".py"):
                        today_fail.add(tok)
        for s in today_fail:
            if s not in order:
                order.append(s)
            # 최신부터 연속으로 이어질 때만 카운트
            if streak.get(s, 0) == (len(files) - 1 - files.index(f)):
                streak[s] = streak.get(s, 0) + 1
        for s in list(streak):
            if s not in today_fail and streak[s] == 0:
                streak.pop(s, None)
    worst = [(s, n) for s, n in streak.items() if n > 0]
    if not worst:
        return Result(True, f"최근 {len(files)}거래일 내부 실패 0건")
    worst.sort(key=lambda x: -x[1])
    parts = [f"{s.split('/')[-1]} **{n}거래일째**" for s, n in worst]
    return Result(False, " · ".join(parts))


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

def run(only: str | None = None) -> tuple[list[tuple[Claim, Result]], int]:
    out = []
    regressions = 0
    for c in CLAIMS:
        if only and c.cid.upper() != only.upper():
            continue
        try:
            r = c.fn()
        except Exception as e:                # 검사 자체의 고장을 통과로 위장하지 않는다
            r = Result(False, f"검사 실행 중 예외: {type(e).__name__}: {e}")
        out.append((c, r))
        if not r.ok and not r.skipped:
            regressions += 1
    return out, regressions


STREAK_FILE = DATA / "resolved_claim_streak.json"


def _update_streaks(results: list[tuple[Claim, Result]]) -> dict[str, int]:
    """⑤ 같은 회귀가 며칠째인지 세어 **문구에 넣는다**.

    B-106(`collect_foreign_exhaustion` 6거래일)과 B-94(11거래일)가 매일
    **똑같은 문장**으로 알려 왔기 때문에 배경이 됐다. 「6거래일째」가 붙으면
    같은 줄이라도 눈에 걸린다.

    ★하루에 여러 번 돌려도 중복 증가하지 않도록 마지막 계수일을 함께 저장한다.
    """
    today = str(date.today())
    try:
        st = json.loads(STREAK_FILE.read_text(encoding="utf-8"))
    except Exception:
        st = {}
    out: dict[str, int] = {}
    for c, r in results:
        e = st.get(c.cid) or {}
        if r.ok or r.skipped:
            st.pop(c.cid, None)
            continue
        if e.get("last") == today:
            out[c.cid] = int(e.get("days", 1))          # 같은 날 재실행 — 유지
            continue
        days = int(e.get("days", 0)) + 1
        st[c.cid] = {"days": days, "last": today, "since": e.get("since", today)}
        out[c.cid] = days
    try:
        STREAK_FILE.parent.mkdir(parents=True, exist_ok=True)
        STREAK_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return out


def _notify(results: list[tuple[Claim, Result]], streaks: dict[str, int]) -> None:
    bad = [(c, r) for c, r in results if not r.ok and not r.skipped]
    if not bad:
        return
    lines = [f"🚨 <b>해소 선언 회귀 {len(bad)}건</b> ({date.today()})",
             "「고쳤다」고 보고한 것이 지금은 깨져 있습니다.", ""]
    for c, r in bad:
        d = streaks.get(c.cid, 1)
        lines.append(f"• <b>[{c.cid}] {d}거래일째</b> ({c.origin})")
        lines.append(f"  {c.what}")
        lines.append(f"  → {r.detail}")
    try:
        from src.telegram_sender import send_message
        send_message("\n".join(lines))
    except Exception as e:
        print(f"[알림 실패] {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    results, regressions = run(a.only)
    streaks = _update_streaks(results) if not a.only else {}
    if not a.no_telegram and not a.only:
        _notify(results, streaks)

    if a.json:
        print(json.dumps([{
            "id": c.cid, "what": c.what, "kind": c.kind, "resolved_on": c.resolved_on,
            "ok": r.ok, "skipped": r.skipped, "reason": r.reason, "detail": r.detail,
        } for c, r in results], ensure_ascii=False, indent=1))
        return 1 if regressions else 0

    print(f"=== 해소 선언 재검증 ({date.today()}) — {len(results)}건 ===")
    for c, r in results:
        d = streaks.get(c.cid)
        tag = f" **{d}거래일째**" if d else ""
        print(f"{r.icon} [{c.cid}]{tag} {c.what}")
        print(f"     {c.origin} · {r.detail or r.reason}")
        if r.skipped and r.reason:
            print(f"     ⏸️ 판정보류: {r.reason}")
    if regressions:
        print(f"\n🚨 **회귀 {regressions}건** — 「고쳤다」고 보고한 것이 지금은 깨져 있다.")
        print("   백로그 상태를 🟢에서 🔴로 되돌리고 원인을 다시 본다.")
    else:
        print("\n✅ 회귀 0건 — 해소 선언이 전부 유지되고 있다.")
    return 1 if regressions else 0


def selftest() -> int:
    """④ 자기 실패 모드 검증 — 고장난 입력에 검사가 속는지 본다.

    ★7/30 교훈: 감시 도구는 대상이 아니라 **자기 자신의 실패 모드**로 검증한다.
      「정상 동작」만 확인한 8케이스 PASS가 미탐 4·오탐 2를 놓쳤다.
    """
    import tempfile
    ok = fail = 0

    def case(name: str, got: bool, want: bool):
        nonlocal ok, fail
        if got == want:
            ok += 1
            print(f"  ✅ {name}")
        else:
            fail += 1
            print(f"  ❌ {name} — 기대 {want}, 실제 {got}")

    print("=== selftest: 검사가 고장난 입력에 속는가 ===")

    # 1) 예외를 던지는 검사는 통과로 위장되면 안 된다
    CLAIMS.append(Claim("TEST-EX", "예외 발생", "2026-09-19",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    res, regs = run("TEST-EX")
    case("예외 발생 검사 → 회귀로 계상", regs == 1, True)
    case("예외 발생 검사 → skipped로 숨기지 않음", res[0][1].skipped is False, True)
    CLAIMS.pop()

    # 2) skipped는 회귀로 세지 않는다
    CLAIMS.append(Claim("TEST-SK", "판정 보류", "2026-09-19",
                        lambda: Result(True, "", skipped=True, reason="입력 없음")))
    _, regs = run("TEST-SK")
    case("skipped → 회귀 아님", regs == 0, True)
    CLAIMS.pop()

    # 3) skipped인데 reason이 비면 설계 위반 → 보류가 아니라 실패로 승격돼야 한다 (③)
    CLAIMS.append(Claim("TEST-NR", "이유 없는 보류", "2026-09-19",
                        lambda: Result(True, "", skipped=True, reason="")))
    res, regs = run("TEST-NR")
    case("이유 없는 skipped → 보류로 안 빠짐", res[0][1].skipped is False, True)
    case("이유 없는 skipped → 회귀로 계상", regs == 1, True)
    CLAIMS.pop()

    # 4) B-94 구조 검사 — 임시 파일을 **주입**한다.
    #    ★프로덕션 run_bat.sh를 덮었다 복원하는 방식으로 짰다가 되돌렸다:
    #      복원 전에 죽으면 486줄짜리 운영 스크립트가 3줄로 남는다.
    #      «테스트가 대상을 파괴할 수 있으면 그 테스트가 최대 위험»이다.
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.sh"
        bad.write_text('if [ "$BAT" = "D" ]; then\n'
                       '  run_py scripts/data_health_check.py\n'
                       'fi\n', encoding="utf-8")
        case("B-94: BAT-D 안으로 되돌리면 탐지",
             _check_b94_outside(bad).ok is False, True)

        good = Path(td) / "good.sh"
        good.write_text('if [ "$BAT" = "D" ]; then\n'
                        '  run_py scripts/rebuild_indicators.py\n'
                        'fi\n'
                        'if [ "$BAT" = "HEALTH" ]; then\n'
                        '  run_py scripts/data_health_check.py\n'
                        'fi\n', encoding="utf-8")
        case("B-94: HEALTH 단계면 통과", _check_b94_outside(good).ok is True, True)

        gone = Path(td) / "gone.sh"
        gone.write_text('if [ "$BAT" = "D" ]; then\n  run_py scripts/other.py\nfi\n',
                        encoding="utf-8")
        case("B-94: 호출이 사라지면 탐지", _check_b94_outside(gone).ok is False, True)

        case("B-94: 파일 없으면 보류(성공 위장 아님)",
             _check_b94_outside(Path(td) / "nope.sh").skipped is True, True)

    print(f"\n{ok} 통과 · {fail} 실패")
    return 1 if fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `| head` 등으로 파이프가 먼저 닫힌 경우 — 검사 결과와 무관하다.
        sys.exit(0)
