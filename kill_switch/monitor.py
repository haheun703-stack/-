"""kill_switch/monitor.py — L4 킬스위치 (RISK_ENGINE_SPEC_v2 §5, 협상 불가 영역).

설계 철학 (§0 요약 — 협상 불가):
1. 수익은 결과, 생존이 목표.
2. 리스크 관리는 사후 보고서가 아니라 사전 게이트.
3. 모든 모델은 틀린다 — 그래서 마지막 층은 모델이 아니라 하드 룰(이 파일)이다.
4. 노출(exposure)이 아니라 리스크(risk)를 본다.

발동 룰 (§5):
- K1 일일 실현+평가 손실 ≥ -3.5%  → 전량 청산 + 긴급 알림 + 24시간 거래 정지
- K2 누적 DD ≥ -10%              → 전량 청산 + 사람의 명시적 승인까지 무기한 정지
- K3 데이터 이상(가격 NaN / 시세 300초 이상 미수신 / API 인증 실패) → 신규 주문 즉시 차단
- K4 게이트 로그 없는 체결 감지   → 신규 주문 차단 + Critical 알림

★의도적 격리(§5): risk/ 패키지를 import하지 않는다. risk/ 쪽 문법 에러·버그가
  이 마지막 층까지 전파되는 것을 차단하기 위해 상수(KST 포함)를 여기에 중복 정의한다.
  상수 중복은 버그가 아니라 설계다.
★자동 복구 금지(§10-4): evaluate()가 정상 지표를 봐도 이미 tripped면 절대 자동 해제하지
  않는다. 해제는 사람의 수동 확인 코드(release)로만 가능하다.
★Phase 1a의 정직한 경계: '전량 청산' 실행은 주문 경로 배선(Phase 1b)의 몫이다.
  이 파일은 actions를 선언만 한다(주문/브로커 import 0). 발동 시 실제로 하는 일은
  state.json 기록 + data/KILL_SWITCH 플래그 보장 + 알림 스텁(logging 한 줄)이 전부다.
★data/KILL_SWITCH는 기존 EnvChecker 소유(2026-05-19 발동 중) — 절대 삭제/덮어쓰기
  금지. 발동 시 파일이 없으면 생성, 있으면 구분선+사유를 append만 한다.
  release()는 state.json만 풀고 이 플래그 파일은 무접촉(제거는 사람 몫).
★쓰기 3원칙: 기본 호출(CLI 인자 없음=status, evaluate)은 부작용 0. 쓰기는 명시적
  함수(trip/release/save_state)와 명시적 CLI 인자로만. json 기록은 KST ISO 타임스탬프로
  자기 상태를 선언. 모르면 차단(fail-closed: state.json 깨짐 → tripped로 간주).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("kill_switch.monitor")

# ── 상수: 의도적 중복(§5 격리 — risk/config.py와 값 일치, import 금지) ──
KST = timezone(timedelta(hours=9))          # 의도적 중복(§5 격리)
DAILY_KILL_LIMIT: float = -0.035            # K1 — 의도적 중복(§5 격리)
TOTAL_KILL_LIMIT: float = -0.10             # K2 — 의도적 중복(§5 격리)
QUOTE_STALE_MAX_SEC: int = 300              # K3 — 의도적 중복(§5 격리)
WATCHDOG_INTERVAL_SEC: int = 30             # §5 watchdog 주기

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH: Path = Path(__file__).resolve().parent / "state.json"
DEFAULT_KILL_FLAG: Path = PROJECT_ROOT / "data" / "KILL_SWITCH"

# 룰별 선언 actions (Phase 1a: 선언만 — 실행 배선은 1b)
_RULE_ACTIONS: dict[str, list[str]] = {
    "K1": ["LIQUIDATE_ALL", "HALT_24H"],
    "K2": ["LIQUIDATE_ALL", "HALT_INDEFINITE"],
    "K3": ["BLOCK_NEW_ORDERS"],
    "K4": ["BLOCK_NEW_ORDERS"],
}
_VALID_RULES = ("K1", "K2", "K3", "K4")


def _now_kst(now_kst: datetime | None = None) -> datetime:
    """KST aware datetime 반환. naive datetime.now() 금지(VPS UTC 9시간 어긋남 방어).

    now_kst 주입 가능(테스트 용이성). naive가 주입되면 KST로 간주해 tz를 부착한다.
    """
    if now_kst is None:
        return datetime.now(tz=KST)
    if now_kst.tzinfo is None:
        return now_kst.replace(tzinfo=KST)
    return now_kst.astimezone(KST)


def _is_nan(x: float | None) -> bool:
    """numpy 없이 NaN 판정(NaN != NaN). None은 NaN이 아니라 '데이터 없음=스킵'."""
    return x is not None and x != x


@dataclass
class KillMetrics:
    """킬스위치 입력 지표. None = 해당 데이터 없음 → 그 룰은 스킵(실데이터 배선=1b).

    주의: None(미배선)과 NaN(배선됐는데 깨진 값)은 다르다 — NaN은 K3 데이터 이상으로
    fail-closed 처리한다.
    """

    daily_pnl_pct: float | None = None          # K1: 일일 실현+평가 손익률 (예: -0.04)
    cumulative_dd_pct: float | None = None      # K2: 누적 드로다운 (예: -0.12)
    price_nan: bool | None = None               # K3: 가격 NaN 감지
    quote_stale_sec: float | None = None        # K3: 시세 미수신 경과초
    api_auth_fail: bool | None = None           # K3: API 인증 실패
    fills_without_gate_log: int | None = None   # K4: 게이트 로그 없는 체결 건수


@dataclass
class KillSwitchState:
    """킬스위치 영속 상태 — state.json에 저장, 서버 리부트로 해제되지 않는다."""

    tripped: bool = False
    rule: str | None = None
    reason: str | None = None
    activated_at: str | None = None     # KST ISO
    released_at: str | None = None      # KST ISO
    actions: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def load_state(path: Path = DEFAULT_STATE_PATH) -> KillSwitchState:
    """state.json 로드. 파일 없음 → untripped 기본 상태.

    ★fail-closed: JSON 깨짐/읽기 실패 등 '모르는 상태'면 tripped=True(K3)로 간주한다.
    킬스위치는 자기 상태를 모를 때 열려 있으면 안 된다.
    """
    p = Path(path)
    if not p.exists():
        return KillSwitchState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("state.json root is not an object")
        return KillSwitchState(
            tripped=bool(raw.get("tripped", False)),
            rule=raw.get("rule"),
            reason=raw.get("reason"),
            activated_at=raw.get("activated_at"),
            released_at=raw.get("released_at"),
            actions=list(raw.get("actions") or []),
            history=list(raw.get("history") or []),
        )
    except Exception:  # noqa: BLE001 — 어떤 깨짐이든 fail-closed
        logger.critical("state.json 손상 감지 — fail-closed로 차단 상태 간주: %s", p)
        return KillSwitchState(
            tripped=True,
            rule="K3",
            reason="state.json corrupted — fail-closed",
            actions=list(_RULE_ACTIONS["K3"]),
        )


def save_state(state: KillSwitchState, path: Path = DEFAULT_STATE_PATH) -> None:
    """상태 영속화 — tmp 파일에 쓰고 os.replace로 원자적 교체(쓰다 죽어도 반파일 없음).

    기록(json)은 KST ISO 타임스탬프(saved_at)로 자기 상태를 선언한다(쓰기 3원칙 ②).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["saved_at"] = _now_kst().isoformat()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


_RULE_PRIORITY: dict[str, int] = {"K2": 4, "K1": 3, "K4": 2, "K3": 1}


def _classify(metrics: KillMetrics) -> tuple[str, str] | None:
    """metrics → (rule, reason) 또는 None. 순수 판정. 우선순위 K2 > K1 > K4 > K3.

    None 지표는 스킵(미배선). NaN 숫자 지표(배선됐는데 깨짐)는 K3 데이터 이상으로 fail-closed.
    """
    # NaN 숫자 지표 = 배선됐는데 깨진 값 → K3 (fills_without_gate_log 포함 — 적대리뷰 P2)
    nan_fields = [
        name
        for name, val in (
            ("daily_pnl_pct", metrics.daily_pnl_pct),
            ("cumulative_dd_pct", metrics.cumulative_dd_pct),
            ("quote_stale_sec", metrics.quote_stale_sec),
            ("fills_without_gate_log", metrics.fills_without_gate_log),
        )
        if _is_nan(val)
    ]

    if (
        metrics.cumulative_dd_pct is not None
        and not _is_nan(metrics.cumulative_dd_pct)
        and metrics.cumulative_dd_pct <= TOTAL_KILL_LIMIT
    ):
        return ("K2", f"누적 DD {metrics.cumulative_dd_pct:.4f} ≤ {TOTAL_KILL_LIMIT} — 무기한 정지(사람 승인 필요)")
    if (
        metrics.daily_pnl_pct is not None
        and not _is_nan(metrics.daily_pnl_pct)
        and metrics.daily_pnl_pct <= DAILY_KILL_LIMIT
    ):
        return ("K1", f"일일 손실 {metrics.daily_pnl_pct:.4f} ≤ {DAILY_KILL_LIMIT} — 24시간 정지")
    if (
        metrics.fills_without_gate_log is not None
        and not _is_nan(metrics.fills_without_gate_log)  # ★NaN은 위에서 K3로(P2)
        and metrics.fills_without_gate_log > 0
    ):
        return ("K4", f"게이트 로그 없는 체결 {metrics.fills_without_gate_log}건 감지")

    k3_reasons: list[str] = []
    if metrics.price_nan is True:
        k3_reasons.append("가격 NaN")
    if (
        metrics.quote_stale_sec is not None
        and not _is_nan(metrics.quote_stale_sec)
        and metrics.quote_stale_sec >= QUOTE_STALE_MAX_SEC
    ):
        k3_reasons.append(f"시세 미수신 {metrics.quote_stale_sec:.0f}s ≥ {QUOTE_STALE_MAX_SEC}s")
    if metrics.api_auth_fail is True:
        k3_reasons.append("API 인증 실패")
    if nan_fields:
        k3_reasons.append(f"지표 NaN(fail-closed): {', '.join(nan_fields)}")
    if k3_reasons:
        return ("K3", "데이터 이상: " + " / ".join(k3_reasons))
    return None


def evaluate(
    metrics: KillMetrics,
    state: KillSwitchState,
    now_kst: datetime | None = None,
) -> tuple[KillSwitchState, list[str]]:
    """발동 판정 — 순수 함수(부작용 0, 입력 state 비변경). 영속화는 trip()의 몫.

    - 미발동 상태에서 룰 감지 → 발동 state 반환.
    - 이미 tripped → ★자동 해제 절대 금지(§10-4). 단 더 치명적인 룰(우선순위 ↑)이 새로
      감지되면 actions를 승급한다 — state.json 손상으로 K3(주문차단)만 걸린 채 진짜 K1/K2
      (전량청산)가 삼켜지던 버그(적대리뷰 P1) 방지. 최초 발동시각은 보존.
    반환: (새 상태, 발동 actions). 미발동이면 (입력 state 그대로, []).
    """
    fired = _classify(metrics)

    if state.tripped:
        if fired is not None and _RULE_PRIORITY[fired[0]] > _RULE_PRIORITY.get(state.rule or "", 0):
            rule, reason = fired
            upgraded = KillSwitchState(
                tripped=True,
                rule=rule,
                reason=f"승급({state.rule}→{rule}): {reason}",
                activated_at=state.activated_at,  # 최초 발동시각 보존
                released_at=None,
                actions=list(_RULE_ACTIONS[rule]),
                history=list(state.history),
            )
            return upgraded, list(_RULE_ACTIONS[rule])
        return state, list(state.actions)  # 해제 금지 — 같거나 낮은 룰은 현행 유지

    if fired is None:
        return state, []
    rule, reason = fired
    actions = list(_RULE_ACTIONS[rule])
    new_state = KillSwitchState(
        tripped=True,
        rule=rule,
        reason=reason,
        activated_at=_now_kst(now_kst).isoformat(),
        released_at=None,
        actions=actions,
        history=list(state.history),
    )
    return new_state, actions


def trip(
    rule: str,
    reason: str,
    state_path: Path = DEFAULT_STATE_PATH,
    flag_path: Path = DEFAULT_KILL_FLAG,
    now_kst: datetime | None = None,
) -> KillSwitchState:
    """발동(부작용 함수): state.json 저장 + KILL_SWITCH 플래그 append + 알림 스텁.

    - 플래그 파일은 기존 내용 보존(EnvChecker 소유) — 없으면 생성, 있으면 append만.
    - 이미 tripped 상태에서 다시 호출되면 원래 발동(rule/activated_at)은 보존하고
      history에 추가 발동 시도만 기록한다(첫 발동 증거 덮어쓰기 방지).
    - 알림은 Phase 1a에서는 logging.critical 한 줄 스텁(실알림 배선=1b).
    """
    if rule not in _VALID_RULES:
        raise ValueError(f"알 수 없는 룰: {rule!r} (허용: {_VALID_RULES})")
    ts = _now_kst(now_kst).isoformat()
    state = load_state(state_path)

    if state.tripped:
        if _RULE_PRIORITY[rule] > _RULE_PRIORITY.get(state.rule or "", 0):
            # 더 치명적 룰로 승급 — 최초 발동시각 보존, rule/reason/actions 갱신(적대리뷰 P1).
            state.history.append(
                {"event": "upgrade", "from": state.rule, "to": rule, "reason": reason, "at": ts}
            )
            state.rule = rule
            state.reason = reason
            state.actions = list(_RULE_ACTIONS[rule])
        else:
            state.history.append(
                {"event": "trip_while_tripped", "rule": rule, "reason": reason, "at": ts}
            )
    else:
        state.tripped = True
        state.rule = rule
        state.reason = reason
        state.activated_at = ts
        state.released_at = None
        state.actions = list(_RULE_ACTIONS[rule])
    save_state(state, state_path)

    # KILL_SWITCH 플래그: 기존 내용 보존 + 구분선/사유 append (덮어쓰기 절대 금지)
    fp = Path(flag_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(f"\n# --- kill_switch trip {ts} ---\n")
        f.write(f"rule={rule}\n")
        f.write(f"reason={reason}\n")
        f.write(f"actions={','.join(_RULE_ACTIONS[rule])} (Phase 1a: 선언만 — 실행 배선은 1b)\n")

    # 알림 스텁 (Phase 1a)
    logger.critical("[KILL_SWITCH] %s 발동 — %s (actions=%s)", rule, reason, _RULE_ACTIONS[rule])
    return state


def release_code(state: KillSwitchState) -> str:
    """수동 해제 확인 코드(6자리 숫자) 파생 — 사람이 --show-release-code로 확인.

    sha256(activated_at + ORDER_INTENTS_HMAC_KEY)에서 파생하므로 발동 시각과 키를
    아는 사람만 계산 가능. 자동 복구 경로가 아니라 '사람의 명시적 행동' 증명용.

    ★키 미설정 시 ValueError(fail-closed, 적대리뷰 P1): 키가 없으면 seed가 state.json
      평문(activated_at)만으로 줄어들어, state.json을 읽는 자동 스크립트가 K2 무기한
      정지를 사람 승인 없이 우회 해제할 수 있다 — 게이트는 키 부재 시 미서명(차단)인데
      킬스위치 해제만 약해지는 비대칭을 차단한다.
    """
    key = os.environ.get("ORDER_INTENTS_HMAC_KEY", "")
    if not key:
        raise ValueError(
            "ORDER_INTENTS_HMAC_KEY 미설정 — 해제 코드 생성 불가(fail-closed). "
            "비밀 없이 코드가 파생되면 사람 승인 없는 자동 해제가 가능해진다. 키 설정 후 재시도."
        )
    seed = (state.activated_at or "") + key
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return str(int(digest, 16) % 1_000_000).zfill(6)


def release(
    code: str,
    state_path: Path = DEFAULT_STATE_PATH,
    now_kst: datetime | None = None,
) -> KillSwitchState:
    """수동 해제 — 확인 코드 일치 시에만. 코드 불일치 → ValueError.

    state.json의 tripped만 풀고 발동~해제 1건을 history에 보존한다.
    ★data/KILL_SWITCH 플래그 파일은 무접촉 — 제거 판단은 사람 몫(EnvChecker 소유).
    """
    state = load_state(state_path)
    if not state.tripped:
        raise ValueError("해제 대상 없음 — 현재 tripped 상태가 아닙니다.")
    expected = release_code(state)
    if code != expected:
        raise ValueError("해제 코드 불일치 — --show-release-code로 확인하십시오.")

    ts = _now_kst(now_kst).isoformat()
    state.history.append(
        {
            "rule": state.rule,
            "reason": state.reason,
            "activated_at": state.activated_at,
            "released_at": ts,
            "actions": list(state.actions),
        }
    )
    state.tripped = False
    state.rule = None
    state.reason = None
    state.activated_at = None
    state.released_at = ts
    state.actions = []
    save_state(state, state_path)
    logger.warning("[KILL_SWITCH] 수동 해제 완료 (%s) — data/KILL_SWITCH 플래그는 무접촉(사람 몫)", ts)
    return state


def build_metrics_from_files(project_root: Path = PROJECT_ROOT) -> KillMetrics:
    """파일 기반 best-effort 지표 수집(Phase 1a 스텁) — 실급여원 배선은 1b.

    data/kis_balance.json에 명시적 키가 있으면 읽고, 없거나 못 읽으면 전부 None(스킵).
    파일 mtime으로 staleness를 유추하지 않는다(과거 파일로 인한 거짓 K3 방지 — 진짜
    시세 수신 시각 배선은 1b). 예외는 삼키고 None.
    """
    metrics = KillMetrics()
    try:
        balance_path = Path(project_root) / "data" / "kis_balance.json"
        if balance_path.exists():
            raw = json.loads(balance_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                v = raw.get("daily_pnl_pct")
                if isinstance(v, (int, float)):
                    metrics.daily_pnl_pct = float(v)
                v = raw.get("cumulative_dd_pct")
                if isinstance(v, (int, float)):
                    metrics.cumulative_dd_pct = float(v)
    except Exception:  # noqa: BLE001 — best-effort: 못 읽으면 None 유지(스킵)
        # 파일이 존재하는데 파싱 실패 = '깨진 급여원' → 조용한 실명 방지 위해 warning(적대리뷰 P2).
        # Phase 1b 실배선은 이 경우를 K3(데이터 이상)로 승격해야 한다(스텁 단계에선 경고만).
        logger.warning(
            "build_metrics_from_files: kis_balance.json 파싱 실패 — 지표 None 스킵. "
            "1b 배선 시 이 경우를 K3로 승격할 것.", exc_info=True,
        )
    return metrics


def run_watchdog(
    metrics_provider: Callable[[], KillMetrics],
    state_path: Path = DEFAULT_STATE_PATH,
    flag_path: Path = DEFAULT_KILL_FLAG,
    interval_sec: float = WATCHDOG_INTERVAL_SEC,
    max_loops: int | None = None,
) -> KillSwitchState:
    """30초 watchdog 루프 — 별도 프로세스 진입점. 메인 봇 생사와 무관하게 동작.

    load → evaluate → 새 발동이면 trip(영속화+플래그+알림). 이미 tripped면 관찰만
    (자동 해제 금지). metrics_provider 자체가 예외를 내면 '킬스위치가 눈을 잃은 것'
    이므로 fail-closed로 K3 발동한다. max_loops는 테스트용(None=무한).
    KeyboardInterrupt는 안전 종료(상태 무변경).
    """
    loops = 0
    state = load_state(state_path)
    try:
        while max_loops is None or loops < max_loops:
            loops += 1
            state = load_state(state_path)
            try:
                metrics = metrics_provider()
            except Exception as exc:  # noqa: BLE001 — fail-closed: 모르면 차단
                if not state.tripped:
                    state = trip(
                        "K3",
                        f"metrics provider 예외 — fail-closed: {type(exc).__name__}: {exc}",
                        state_path=state_path,
                        flag_path=flag_path,
                    )
                if max_loops is None or loops < max_loops:
                    time.sleep(interval_sec)
                continue

            new_state, actions = evaluate(metrics, state)
            # 신규 발동 또는 더 치명적 룰로 승급(rule 변경) → trip()으로 영속화(플래그·알림).
            # state.json 손상→K3 tripped인 채 진짜 K1/K2가 들어와도 승급이 영속화된다(P1).
            if new_state.tripped and (not state.tripped or new_state.rule != state.rule):
                state = trip(
                    new_state.rule or "K3",
                    new_state.reason or "unknown",
                    state_path=state_path,
                    flag_path=flag_path,
                )
            else:
                state = new_state
                if state.tripped:
                    logger.info("[KILL_SWITCH] 발동 유지 중 rule=%s actions=%s (자동 해제 금지)", state.rule, actions)

            if max_loops is None or loops < max_loops:
                time.sleep(interval_sec)
    except KeyboardInterrupt:
        logger.info("[KILL_SWITCH] watchdog 수동 중단(KeyboardInterrupt) — 상태 무변경 종료")
    return state


def _print_status(state_path: Path, flag_path: Path) -> None:
    """현재 상태 요약 출력 — 부작용 0(쓰기 3원칙 ①: 기본 호출은 읽기 전용)."""
    state = load_state(state_path)
    print("=== KILL SWITCH 상태 (읽기 전용) ===")
    print(f"state.json : {state_path}")
    print(f"tripped    : {state.tripped}")
    print(f"rule       : {state.rule}")
    print(f"reason     : {state.reason}")
    print(f"activated  : {state.activated_at}")
    print(f"released   : {state.released_at}")
    print(f"actions    : {state.actions}")
    print(f"history    : {len(state.history)}건")
    flag = Path(flag_path)
    print(f"KILL_SWITCH 플래그: {'존재(발동 중일 수 있음 — EnvChecker 소유)' if flag.exists() else '없음'} ({flag})")


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점. 인자 없음=status(부작용 0) / --watch / --show-release-code /
    --release CODE / --test-fire RULE(★기본 경로 보호 — 실상태 오염 방지).
    """
    parser = argparse.ArgumentParser(
        prog="kill_switch.monitor",
        description="L4 킬스위치 — 협상 불가 영역 (§5). 인자 없으면 상태 출력만(부작용 0).",
    )
    parser.add_argument("--watch", action="store_true", help="30초 watchdog 루프 가동")
    parser.add_argument("--show-release-code", action="store_true", help="수동 해제 코드 표시")
    parser.add_argument("--release", metavar="CODE", help="확인 코드로 수동 해제 (플래그 파일 무접촉)")
    parser.add_argument(
        "--test-fire", metavar="RULE", choices=list(_VALID_RULES),
        help="모의 발동 — ★--state-path/--flag-path 둘 다 기본값과 다를 때만 허용",
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--flag-path", type=Path, default=DEFAULT_KILL_FLAG)
    parser.add_argument("--interval-sec", type=float, default=WATCHDOG_INTERVAL_SEC)
    parser.add_argument("--max-loops", type=int, default=None, help="watchdog 루프 횟수 제한(테스트용)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.test_fire:
        # 실상태 오염 방지: 기본 state.json/KILL_SWITCH를 모의 발동으로 건드리지 않는다
        same_state = Path(args.state_path).resolve() == DEFAULT_STATE_PATH.resolve()
        same_flag = Path(args.flag_path).resolve() == DEFAULT_KILL_FLAG.resolve()
        if same_state or same_flag:
            print(
                "오류: --test-fire는 --state-path와 --flag-path 모두 기본 경로와 다르게 "
                "지정해야 합니다(실상태/EnvChecker 플래그 오염 방지).",
                file=sys.stderr,
            )
            return 2
        state = trip(
            args.test_fire,
            f"모의 발동(--test-fire {args.test_fire})",
            state_path=args.state_path,
            flag_path=args.flag_path,
        )
        print(f"모의 발동 완료: rule={state.rule} actions={state.actions}")
        try:
            print(f"해제 코드(모의): {release_code(state)}")
        except ValueError as exc:
            print(f"(해제 코드 생성 불가: {exc})")
        return 0

    if args.show_release_code:
        state = load_state(args.state_path)
        if not state.tripped:
            print("현재 tripped 상태가 아닙니다 — 해제 코드 불필요.")
            return 0
        try:
            print(f"해제 코드: {release_code(state)}")
        except ValueError as exc:
            print(f"해제 코드 생성 실패: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.release:
        try:
            release(args.release, state_path=args.state_path)
        except ValueError as exc:
            print(f"해제 실패: {exc}", file=sys.stderr)
            return 1
        print("해제 완료 — state.json만 해제. data/KILL_SWITCH 플래그는 무접촉(제거는 사람 몫).")
        return 0

    if args.watch:
        final = run_watchdog(
            build_metrics_from_files,
            state_path=args.state_path,
            flag_path=args.flag_path,
            interval_sec=args.interval_sec,
            max_loops=args.max_loops,
        )
        return 1 if final.tripped else 0

    # 기본: status 출력만 (부작용 0)
    _print_status(args.state_path, args.flag_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
