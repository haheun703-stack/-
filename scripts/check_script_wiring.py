"""스크립트 배선 검증 — "만들어놓고 안 거는" 실패를 매일 잡는다.

배경 (2026-08-29)
  하루에 같은 실패가 두 번 나왔다.
    ⑴ B-47 `fill_short_from_jgis.py` — 8/21에 만들고 **수동 1회만** 돌린 뒤
       `run_bat.sh`에 걸지 않아 6거래일간 공매도 데이터가 0이었다.
       그런데 나는 "배선 완료"라고 보고했다.
    ⑵ `upload_flowx_strategy_scoreboard.py` — 만들어졌지만 어디에도 안 걸려
       매일 생성되는 36건이 flowx.kr로 나가지 못했고,
       화면에는 182일 묵은 아카이브가 대신 표시되고 있었다.

  둘 다 **"산출물은 만드는데 파이프가 안 이어진"** 같은 모양이다.
  사람이 기억으로 막을 수 없어서 검사로 만든다.

판정 규칙 (B-29 오진에서 배운 것)
  ★"참조 0"은 `run_bat.sh`만 보면 성립하지 않는다. 8/14에 그 판정으로
    라이브 cron 7개를 지울 뻔했다. 그래서 아래 **네 곳을 모두** 본다.
      ① scripts/cron/run_bat.sh
      ② VPS crontab (`--crontab` 파일로 전달, 없으면 그 사실을 출력)
      ③ 다른 .py의 import / subprocess 호출
      ④ *.bat / *.sh 스크립트
  ★신규(최근 N일 내 git 추가)인데 어디에도 없으면 **NEW-ORPHAN** — 가장 위험.
    오래된 고아는 의도적 보관일 수 있으므로 등급을 나눈다.

실행
  python -u -X utf8 scripts/check_script_wiring.py
  python -u -X utf8 scripts/check_script_wiring.py --days 14
  python -u -X utf8 scripts/check_script_wiring.py --crontab /tmp/crontab.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
RUN_BAT = SCRIPTS / "cron" / "run_bat.sh"

#: 배선이 필요 없는 것 — 라이브러리/연구/일회성
EXEMPT_DIRS = {"archive", "backtest", "research", "db", "sql", "data_accumulation"}
EXEMPT_PREFIX = ("backtest_", "_", "test_")
EXEMPT_NAMES = {
    "__init__.py",
    "check_script_wiring.py",   # 자기 자신
}


def git_added_within(days: int) -> set[str]:
    """최근 N일 내 git에 추가된 scripts/*.py 파일명 집합."""
    try:
        out = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--diff-filter=A",
             "--name-only", "--format=", "--", "scripts/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ git 조회 실패 — 신규 판정 생략: {e}")
        return set()
    return {Path(l.strip()).name for l in out.splitlines()
            if l.strip().endswith(".py")}


def gather_references(crontab_path: Path | None) -> tuple[str, list[str]]:
    """배선 텍스트를 한 덩어리로 모은다. (텍스트, 확인한 소스 목록)"""
    blobs: list[str] = []
    sources: list[str] = []

    if RUN_BAT.exists():
        blobs.append(RUN_BAT.read_text(encoding="utf-8", errors="ignore"))
        sources.append("run_bat.sh")

    if crontab_path and crontab_path.exists():
        blobs.append(crontab_path.read_text(encoding="utf-8", errors="ignore"))
        sources.append(f"crontab({crontab_path})")
    else:
        # ★없는 것을 조용히 넘기지 않는다 — B-29가 바로 그래서 났다.
        sources.append("crontab(미확인 ★)")

    # ★8/29 자기검증에서 뚫렸다: 처음엔 tests/를 포함해서 훑었더니
    #   `upload_flowx_strategy_scoreboard.py`가 미배선인데도 "참조 있음"으로 통과했다.
    #   `tests/test_upload_flowx_strategy_scoreboard.py`가 import하고 있었기 때문이다.
    #   **테스트 참조는 배선이 아니다.** 테스트가 있다는 것과 매일 도는 것은 다르다.
    #   (7/30 "감시 도구는 자기 실패 모드로 검증해야 한다"를 이 도구에 적용한 결과)
    EXCLUDE_PARTS = {"archive", "venv", "node_modules", "tests", "test",
                     ".next", "__pycache__", "flowx-web"}
    for pat in ("*.py", "*.sh", "*.bat"):
        for f in PROJECT_ROOT.rglob(pat):
            if EXCLUDE_PARTS & set(f.parts):
                continue
            if f.name.startswith("test_") or f.name == "check_script_wiring.py":
                continue
            try:
                blobs.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001
                continue
    sources.append("코드(.py/.sh/.bat, ★tests 제외)")
    return "\n".join(blobs), sources


def main() -> int:
    ap = argparse.ArgumentParser(description="스크립트 배선 검증")
    ap.add_argument("--days", type=int, default=30, help="신규 판정 기간(일)")
    ap.add_argument("--crontab", type=Path,
                    help="`crontab -l > 파일` 결과 경로 (VPS에서 생성)")
    args = ap.parse_args()

    blob, sources = gather_references(args.crontab)
    recent = git_added_within(args.days)

    targets = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name in EXEMPT_NAMES or f.name.startswith(EXEMPT_PREFIX):
            continue
        targets.append(f)

    new_orphans, old_orphans = [], []
    for f in targets:
        stem = f.stem
        # 파일명(확장자 포함/제외) 또는 모듈 import 형태로 언급되는가
        pats = [rf"\b{re.escape(f.name)}\b",
                rf"\bimport\s+{re.escape(stem)}\b",
                rf"\bfrom\s+{re.escape(stem)}\s+import\b",
                rf"\bscripts\.{re.escape(stem)}\b"]
        if any(re.search(p, blob) for p in pats):
            continue
        (new_orphans if f.name in recent else old_orphans).append(f.name)

    print("=" * 70)
    print("스크립트 배선 검증")
    print("=" * 70)
    print(f"확인한 소스: {', '.join(sources)}")
    print(f"검사 대상  : scripts/*.py {len(targets)}개 "
          f"(archive·backtest_·_ 접두 제외)")
    print(f"신규 기준  : 최근 {args.days}일 git 추가 {len(recent)}개")
    print()

    if new_orphans:
        print(f"🚨 NEW-ORPHAN {len(new_orphans)}건 — **최근에 만들었는데 아무 데서도 안 부른다**")
        print("   이게 B-47(공매도 6거래일 0)·스코어보드(182일 묵은 화면)를 만든 형태다.")
        for n in new_orphans:
            print(f"     - {n}")
        print()
    else:
        print("✅ NEW-ORPHAN 0건 — 최근 만든 스크립트는 전부 배선돼 있다")
        print()

    if old_orphans:
        print(f"🟡 기존 고아 {len(old_orphans)}건 (의도적 보관일 수 있음 — 삭제 전 반드시 실측)")
        for n in old_orphans[:15]:
            print(f"     - {n}")
        if len(old_orphans) > 15:
            print(f"     … 외 {len(old_orphans) - 15}건")
        print()

    if not args.crontab:
        print("⚠️ crontab을 안 봤다. VPS에서 아래로 만들어 다시 돌릴 것 —")
        print("   crontab -l > /tmp/crontab.txt && "
              "python -u -X utf8 scripts/check_script_wiring.py --crontab /tmp/crontab.txt")
        print("   ★crontab을 빼고 낸 '고아' 판정은 B-29에서 이미 한 번 틀렸다.")

    return 1 if new_orphans else 0


if __name__ == "__main__":
    raise SystemExit(main())
