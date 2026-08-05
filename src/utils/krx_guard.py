"""KRX 접근 차단 가드 (2026-08-05 신설).

KRXSession ID/PW 로그인은 매 실행(gap-fill 최대 7회)이 **CD007 계정잠금**을
유발한다. VPS 정본 `scripts/cron/run_bat.sh`에서는 6/22에 주석 처리됐으나,
로컬 윈도우 경로에 호출이 그대로 남아 있었다:

- `coo_orchestrator.py:91` → `scripts/scan_nationality.py --send`
- `scripts/schedule_D_original.bat:91` → `scripts/collect_short_selling.py`
- `scripts/schedule_D_original.bat:129` → `scripts/scan_nationality.py --send`

두 경로 모두 현재는 작업 스케줄러 등록 경로의 인코딩이 깨져 실행되지 않는다.
문제는 **누군가 그 경로를 "고치는" 순간 즉시 부활**한다는 것이다. 그래서
호출자를 하나씩 주석 처리하는 대신 **호출 대상에 가드**를 두어 모든 경로를
한 번에 닫는다. bat 파일은 인코딩이 이미 손상돼 있어 편집 시 파일 전체가
깨질 위험이 있는데, 이 방식은 bat을 건드리지 않는다는 이점도 있다.

해제: KRX 잠금 해제를 **확인한 뒤** 환경변수 `KRX_ACCESS_UNLOCKED=1`.
"""
from __future__ import annotations

import os

ENV_KEY = "KRX_ACCESS_UNLOCKED"


def krx_access_allowed(script_name: str = "") -> bool:
    """KRX 접근이 허용됐는지. 기본은 차단(fail-closed)."""
    if os.getenv(ENV_KEY) == "1":
        return True
    print(
        f"[KRX-GUARD] 차단: {script_name or '이 스크립트'}는 KRX 로그인을 수행합니다 "
        f"— CD007 계정잠금 방지 가드(2026-08-05). "
        f"해제하려면 잠금 해제 확인 후 {ENV_KEY}=1"
    )
    return False
