"""risk/nonce_store.py — 게이트 통행증 nonce 영속 저장소 (RISK_ENGINE Phase 1b / C-ii).

`verify_gate_token(seen_nonces=...)`에 주입하는 set 호환 객체.
pre_trade_gate 계약상 사용되는 연산은 `__contains__`(`in`)과 `add` 두 가지뿐이다.

영속화 목적 (체크리스트 C-ii):
  - 재시작 후에도 max_age(기본 300s) 내 토큰 replay 차단 — 인메모리 set은 재시작 시 소실되어
    '재부팅 후 max_age 창 동안 같은 토큰 재사용 가능' 구멍이 있었다(1b-i 교차검증 발견).
  - 같은 머신의 어댑터 인스턴스 간 공유 — 파일이 단일 진실원이라 인스턴스마다 set이 갈리지 않음.

설계:
  - 파일: data/risk/seen_gate_nonces.log, 한 줄 "nonce<TAB>KST_ISO".
  - add: 인메모리 추가 + 파일 1줄 append(작은 라인 append는 원자적).
  - __contains__: 파일을 재읽어 만료(>retention) 행 제외 후 검사 → 교차 프로세스 가시성 확보.
  - 만료 nonce는 그 토큰 자체가 verify의 max_age로 이미 거부되므로 replay 불가 →
    retention=max_age(+버퍼)면 충분. 만료행이 쌓이면 compact(재기록)로 파일 크기 제한.

신뢰 경계 (graceful degradation):
  - 파일 읽기/쓰기 실패 시 '전면 차단'(fail-closed)으로 가지 않고 경고 후 인메모리로 동작한다.
    nonce는 *replay 방지 층*이며, 1차 방어(HMAC 서명 + 만료 + 종목/사이즈 바인딩)는 그대로
    유효하다. 파일 글리치 하나로 모든 실주문을 막는(가용성 붕괴) 것이 더 큰 리스크라는 판단.
    진짜 fail-closed 백스톱은 킬스위치(L4)다.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from risk.config import KST

logger = logging.getLogger("risk.nonce_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NONCE_PATH = PROJECT_ROOT / "data" / "risk" / "seen_gate_nonces.log"
_DEFAULT_RETENTION_SEC = 360  # verify max_age(300) + 60s 시계 스큐 버퍼
_COMPACT_THRESHOLD = 500      # 만료행이 이만큼 쌓이면 파일 재기록(크기 제한)


def _now_kst(now_kst: datetime | None = None) -> datetime:
    if now_kst is None:
        return datetime.now(tz=KST)
    if now_kst.tzinfo is None:
        return now_kst.replace(tzinfo=KST)
    return now_kst.astimezone(KST)


class PersistentNonceSet:
    """파일 영속 + 인메모리 nonce set. `in`/`add`만 지원(verify_gate_token 계약).

    now_fn: 테스트용 시간 주입(인자 없는 호출자 → KST aware datetime). None이면 실시간.
    """

    def __init__(
        self,
        path: Path = DEFAULT_NONCE_PATH,
        retention_sec: int = _DEFAULT_RETENTION_SEC,
        now_fn=None,
    ) -> None:
        self.path = Path(path)
        self.retention_sec = retention_sec
        self._now_fn = now_fn or (lambda: datetime.now(tz=KST))
        self._mem: set[str] = set()
        self._refresh()  # 생성 시 파일에서 살아있는 nonce 로드(재시작 후 replay 차단)

    # ── set 호환 인터페이스 ────────────────────────────────────────────────
    def __contains__(self, nonce: str) -> bool:
        # 교차 프로세스 가시성: 매 검사 전 파일을 다시 읽어 다른 인스턴스가 쓴 nonce도 본다.
        # 실주문 빈도가 낮아(분 단위) 재읽기 비용은 무시 가능.
        self._refresh()
        return nonce in self._mem

    def add(self, nonce: str) -> None:
        if not nonce:
            return
        self._mem.add(nonce)
        self._append(nonce)

    # ── 내부 ──────────────────────────────────────────────────────────────
    def _cutoff_iso_ok(self, ts_iso: str, now: datetime) -> bool:
        """ts_iso가 retention 창 안(살아있음)이면 True. 파싱 불가/미래 = 보수적으로 살아있음 취급."""
        try:
            ts = datetime.fromisoformat(ts_iso)
        except (ValueError, TypeError):
            return True  # 깨진 타임스탬프 = 보수적으로 유지(검사에 남겨둠)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=KST)
        age = (now - ts).total_seconds()
        return age <= self.retention_sec

    def _refresh(self) -> None:
        """파일을 읽어 살아있는(retention 내) nonce만 인메모리에 적재. 만료행 누적 시 compact."""
        p = self.path
        if not p.exists():
            return
        now = self._now_fn()
        live: set[str] = set()
        live_lines: list[str] = []
        expired = 0
        try:
            with p.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    if not line:
                        continue
                    nonce, _, ts_iso = line.partition("\t")
                    if not nonce:
                        continue
                    if self._cutoff_iso_ok(ts_iso, now):
                        live.add(nonce)
                        live_lines.append(line)
                    else:
                        expired += 1
        except OSError as exc:
            # graceful: 파일 글리치로 전면 차단하지 않는다(서명+만료 층은 유효). 인메모리 유지.
            logger.warning("nonce_store 읽기 실패 — 인메모리로 계속(서명/만료 층 유효): %s", exc)
            return
        # 인메모리는 파일 살아있는 것 ∪ 이번 프로세스가 add한 것(소실 방지) 합집합
        self._mem |= live
        if expired >= _COMPACT_THRESHOLD:
            self._compact(live_lines)

    def _append(self, nonce: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ts = self._now_fn().isoformat()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"{nonce}\t{ts}\n")
        except OSError as exc:
            logger.warning("nonce_store 기록 실패 — 인메모리에는 남음(같은 프로세스 replay는 차단): %s", exc)

    def _compact(self, live_lines: list[str]) -> None:
        """만료행을 버리고 살아있는 행만 원자적 재기록(파일 크기 제한)."""
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                "".join(l + "\n" for l in live_lines), encoding="utf-8"
            )
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("nonce_store compact 실패(무해, 다음 기회 재시도): %s", exc)
