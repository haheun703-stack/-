"""거짓 상수 판정의 오탐 억제 — "0이 정당한 값"과 "채우지 못한 0"을 가른다.

★왜 필요한가(8\1): 7/31에 넣은 값 정합성 검사가 **첫 실행에서 경고 3건을 냈고
전부 오탐**이었다.

  - `quant_sector_fire.overheat_penalty` — `overheat = 0`에서 시작해 과열 조건마다
    감산만 하는 지표다(scan_sector_fire.py:225). 0이 하한이자 기본값이라
    폭락장(MA20 이격 -6.8%·RSI 41)에선 0이 정답이다.
  - `quant_sector_fire.market_kospi_stoch_k` — 0~100 범위 지표가 바닥에 붙은 것.
    6월 77~87 → 7/7 22 → 7/14 7.7 → 7/22부터 0이라는 궤적이 시장을 그대로 따라갔다.
  - `dashboard_smart_money.exec_strength` — 체결강도는 정보봇 산출값이라
    **우리 업로드 컬럼 목록에 아예 없다**. 우리가 안 보내서 DEFAULT 0이 박힌 것을
    "우리가 0을 채웠다"고 읽었다.

`price=0`(진짜 결함)과의 차이는 **0이 물리적으로 불가능한 값인가**인데, 이건
런타임 데이터만으로는 구분되지 않는다. 그래서 두 축으로 억제한다:

  ① **이력** — 오늘 처음 0이 된 것(급변)과 며칠째 0인 것(기보고 상태)을 가른다.
     급변은 언제나 알리고, 지속 상태는 `REMIND_EVERY`회마다만 다시 알린다.
     ★"과거에 실값이 있었으니 0도 가능한 값"으로 억제하면 안 된다 — 그러면
     **잘 돌던 컬럼이 고장나 0이 되는** 가장 흔한 결함이 영구 침묵한다(8/1 검증에서
     실제로 뚫렸다). 7/31 `price=0`이 잡힌 건 처음부터 0이었기 때문일 뿐이다.
  ② **판정** — 사람이 근거와 함께 "정상 0"으로 확정한 컬럼(`config/const_zero_verdicts.json`).
     이력만으로는 태생적 0(감산 지표)과 결함을 끝내 구분할 수 없어, 코드를 읽고
     내린 판단을 파일에 남긴다. 여기 등재된 것만 완전히 침묵한다.

★★이력을 DB에서 뽑지 않는 이유(이 모듈의 존재 이유): 공동적재 테이블은 타 봇
실값이 섞인다. `dashboard_smart_money.price`는 정보봇이 실값을 넣으므로 **DB 이력엔
0 아닌 값이 늘 있고**, DB를 기준으로 억제했다면 6주짜리 `price=0`을 영원히 못 잡는다.
우리가 무엇을 보냈는지는 우리만 안다 — **우리가 보낸 것의 이력만이 우리 판정의
근거가 된다.** B-32("판정 소스는 DB 시각 추정이 아니라 우리 업로드")와 같은 원칙이고,
B-19 ①("판정 소스는 대상 파이프라인과 독립이어야")의 재적용이다.

감시가 적재 경로를 죽이면 안 되므로 모든 공개 함수는 예외를 삼킨다. 다만 **판정
불가 시에는 억제하지 않는다** — 경보를 잠재우는 쪽이 더 위험하다(data_health_check.py:91).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATE_PATH = PROJECT_ROOT / "data" / "metrics" / "payload_const_state.json"
_VERDICT_PATH = PROJECT_ROOT / "config" / "const_zero_verdicts.json"

# 연속 0이 이어질 때 몇 회마다 다시 알릴지. 매일 같은 경고를 내면 사람이 무시하게
# 되고(경보 피로), 완전히 침묵하면 `price=0` 6주 방치가 그대로 재현된다.
REMIND_EVERY = 5

_verdict_cache: dict[str, Any] | None = None


def _key(table: str, col: str) -> str:
    return f"{table}.{col}"


def _load_verdicts() -> dict[str, Any]:
    """사람이 확정한 '정상 0' 판정 목록. 파일이 없으면 빈 판정으로 동작한다."""
    global _verdict_cache
    if _verdict_cache is not None:
        return _verdict_cache
    try:
        with open(_VERDICT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _verdict_cache = data.get("verdicts", {}) if isinstance(data, dict) else {}
    except FileNotFoundError:
        _verdict_cache = {}
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONST] 판정 파일 읽기 실패(무시): %s", e)
        _verdict_cache = {}
    return _verdict_cache


def _load_state() -> dict[str, Any]:
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONST] 이력 파일 읽기 실패(무시): %s", e)
        return {}


def _save_state(state: dict[str, Any]) -> None:
    """원자 저장 — 중간에 죽어도 반쯤 쓰인 JSON이 남지 않게 한다."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONST] 이력 저장 실패(무시): %s", e)


def observe(table: str, observations: dict[str, bool], date_str: str) -> None:
    """이번 업로드에서 각 수치 컬럼이 전량 0/null이었는지를 이력에 반영한다.

    `observations` = {컬럼명: 전량_0_또는_null 여부}. **0이 아니었던 컬럼도 반드시
    함께 넘겨야** 이력이 성립한다 — "한 번이라도 실값을 보냈다"가 억제의 근거이기
    때문이다. 같은 날 같은 테이블을 여러 번 올려도 마지막 관측으로 수렴한다.
    """
    try:
        if not observations:
            return
        state = _load_state()
        for col, is_zero in observations.items():
            k = _key(table, col)
            e = state.get(k) or {}
            e["obs_count"] = int(e.get("obs_count", 0)) + 1
            e["last_seen"] = date_str
            e.setdefault("first_seen", date_str)
            if is_zero:
                e["zero_streak"] = int(e.get("zero_streak", 0)) + 1
            else:
                e["zero_streak"] = 0
                e["last_nonzero"] = date_str
            state[k] = e
        _save_state(state)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONST] 이력 반영 실패(무시) %s: %s", table, e)


def should_warn(table: str, col: str) -> tuple[bool, str]:
    """이 컬럼의 '전량 0'을 경고할지 판정한다. `observe()` **호출 전에** 부른다.

    반환 `(경고할까, 사유)`. 사유는 로그·리포트에 그대로 붙여 왜 조용한지(혹은
    왜 시끄러운지)를 사람이 바로 알 수 있게 한다.

    ★억제 근거를 "과거에 실값이 있었다"로 잡으면 안 된다(8\1 검증에서 잡은 설계
    결함): 그러면 **정상 작동하던 컬럼이 갑자기 0으로 고장나는** 가장 흔한 시나리오가
    영구 침묵한다. 7/31 `price=0`이 잡힌 건 처음부터 0이었기 때문일 뿐이고, 실값을
    쓰다가 0으로 회귀했다면 그 규칙으로는 영원히 못 잡았다.

    그래서 억제 근거는 **"이미 알려진 지속 상태인가"**다:
      - 직전이 실값인데 오늘 전량 0 → **급변**. 언제나 경고한다.
      - 연속 0이 이어지는 중 → 이미 보고한 상태. `REMIND_EVERY`회마다만 재알림.
      - 사람이 근거와 함께 '정상 0'으로 확정 → 완전 침묵(그때만).

    판정 불가·오류 시에는 **경고하는 쪽으로 폴백**한다.
    """
    try:
        v = _load_verdicts().get(_key(table, col))
        if v:
            return False, f"정상 0 판정({v.get('why', '사유 미기재')})"

        e = _load_state().get(_key(table, col))
        if not e:
            return True, "이력 없음(첫 관측)"
        streak = int(e.get("zero_streak", 0)) + 1  # 오늘 것을 더한 연속 횟수
        if streak == 1:
            last = e.get("last_nonzero")
            return True, f"직전까지 실값{f'(~{last})' if last else ''} → 오늘 처음 0"
        if streak % REMIND_EVERY == 0:
            return True, f"{streak}회 연속 0 — 미해결"
        return False, f"{streak}회 연속 0 — 기보고"
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONST] 억제 판정 실패(경고 유지) %s.%s: %s", table, col, e)
        return True, "판정 실패"
