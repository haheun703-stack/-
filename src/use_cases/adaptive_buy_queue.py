"""적응형 포지션 매매법 MVP-2 — 분할매수 큐 (-10% / -20% / -30%).

배경 (퐝가님 5/23 흐름 + 5/24 일요일 작업):
  6단계 흐름 중 [4단계 분할매수 대기]:
    "내리면 사고, 끝까지 오르고 -3%면 팔고,
     조정의 시작점에서 다시 사고 — 단, 한 번에 사지 말고 3 단계로 나눠서."

  MVP-1 매도 후 천장가 + 가용 현금 알면 자동 큐 등록.
  종목별 3 단계 지정가:
    - L1: 천장 -10%  (가용 30%) — 1차 진입
    - L2: 천장 -20%  (가용 30%) — 평균 단가 낮춤
    - L3: 천장 -30%  (가용 40%) — 바닥 매수 (가장 큰 비중)

  평단가 효과:
    0.30 × 0.90 + 0.30 × 0.80 + 0.40 × 0.70
    = 0.27 + 0.24 + 0.28 = 0.79
    → 평단가 = 천장 × 79% (= 천장 대비 -21% 효과)

  매 30 분 cron으로 단계별 도달 여부 확인 + (옵션) 자동 매수.

MVP-2 기능:
  1. register_buy_queue: 매도 후 천장가 + 가용 현금 → 3 단계 큐 등록
  2. check_and_trigger_queues: cron에서 호출, 도달 단계 자동 매수 / 알림
  3. data/adaptive_buy_queue.json 영속 저장 (상태 추적)
  4. KILL_SWITCH 발동 시 정지
  5. 3 종목 한도 (ADAPTIVE_MAX_POSITIONS)
  6. 1주차 안전: ADAPTIVE_AUTO_BUY=0 → 알림만, 자동매수 X

상태 머신:
  PENDING ──도달──> TRIGGERED ──자동매수──> FILLED
                            └──알림만──> NOTIFIED
                                          └──다음 cron 재시도

사용:
  from src.use_cases.adaptive_buy_queue import register_buy_queue, check_and_trigger_queues
  register_buy_queue(ticker="240810", peak_price=25000, available_cash=3_000_000, name="원익IPS")
  results = check_and_trigger_queues(broker)  # cron
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QUEUE_PATH = PROJECT_ROOT / "data" / "adaptive_buy_queue.json"
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.flag"

# === P0-1 race condition 방어 (5/25 sub-agent A) ===
# cron `*/30 9-15 * * 1-5` + MVP-1 자동 매도 + MVP-2 큐 등록 동시 실행 시
# JSON read-modify-write 충돌 → 데이터 손실 방지.
# 우선순위: portalocker (cross-platform, VPS Linux + 로컬 Windows 모두 지원)
#         → fcntl (Linux fallback)
#         → 단순 .lock 파일 (atomic create) fallback
# atomic save: tmp 파일 쓰기 → os.replace로 단일 호출 교체 (Windows + Linux 보장)

_LOCK_TIMEOUT_SEC = float(os.getenv("ADAPTIVE_LOCK_TIMEOUT_SEC", "5"))
_LOCK_POLL_INTERVAL_SEC = 0.05

try:
    import portalocker  # type: ignore
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False

try:
    import fcntl  # type: ignore  # Linux only
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


# === 임계 (.env 동적, 1주차는 보수적) ===
def _parse_int_list(env_val: str, default: list[int]) -> list[int]:
    try:
        return [int(x.strip()) for x in env_val.split(",") if x.strip()]
    except (ValueError, AttributeError):
        return default


SPLIT_LEVELS = _parse_int_list(os.getenv("ADAPTIVE_SPLIT_LEVELS", "10,20,30"), [10, 20, 30])
SPLIT_RATIOS = _parse_int_list(os.getenv("ADAPTIVE_SPLIT_RATIOS", "30,30,40"), [30, 30, 40])
SPLIT_MAX_AMOUNT = int(os.getenv("ADAPTIVE_SPLIT_MAX_AMOUNT", "1000000"))     # 1단위 100만
SPLIT_MAX_QTY = int(os.getenv("ADAPTIVE_SPLIT_MAX_QTY", "0"))                  # 1주차 1주 cap (0=무제한)
MAX_POSITIONS = int(os.getenv("ADAPTIVE_MAX_POSITIONS", "3"))                  # 3종목 한도
AUTO_BUY = os.getenv("ADAPTIVE_AUTO_BUY", "0") == "1"                          # 1주차 알림만
QUEUE_EXPIRY_DAYS = int(os.getenv("ADAPTIVE_QUEUE_EXPIRY_DAYS", "60"))


# === 상태 상수 ===
STATUS_PENDING = "PENDING"            # 가격 미도달
STATUS_TRIGGERED = "TRIGGERED"        # 가격 도달 + 알림 (AUTO_BUY=0)
STATUS_FILLED = "FILLED"              # 자동매수 성공
STATUS_QUICK_ARMED = "QUICK_ARMED"    # MVP-2.5 +7% 도달 → trailing 시작 (5/24 보강)
STATUS_QUICK_SOLD = "QUICK_SOLD"      # MVP-2.5 trailing 꺾임 → 매도 완료
STATUS_EXPIRED = "EXPIRED"            # 만료
STATUS_FAILED = "FAILED"              # 매수 실패

# === MVP-2.5 빠른 익절 임계 ===
QUICK_PROFIT_PCT = float(os.getenv("ADAPTIVE_QUICK_PROFIT_PCT", "7"))   # +N% 도달 시 익절


@dataclass
class QueueStage:
    """큐 단계."""

    level: int                  # 1, 2, 3
    target_pct: float           # 0.90, 0.80, 0.70 (천장 대비)
    target_price: int           # 지정가
    alloc_ratio: float          # 0.30, 0.30, 0.40
    alloc_amount: int           # 배정 금액 (KRW)
    qty: int                    # 매수 수량
    status: str = STATUS_PENDING
    triggered_at: Optional[str] = None
    order_id: Optional[str] = None
    actual_price: int = 0       # 실제 체결가
    actual_qty: int = 0
    error: Optional[str] = None
    # === MVP-2.5 빠른 익절 (5/24 추가) ===
    quick_profit_target: int = 0      # 매수가 × 1.07 = trailing 진입 가격
    quick_profit_order_id: Optional[str] = None
    quick_profit_sold_at: Optional[str] = None
    quick_profit_sold_price: int = 0
    # === MVP-2.5 Trailing 보강 (5/24 퐝가님 지시: "15%까지 다 먹기") ===
    trailing_peak: int = 0            # +7% 진입 후 추적 중인 고점
    trailing_armed_at: Optional[str] = None  # trailing 시작 시각
    trailing_peak_updated_at: Optional[str] = None


def _is_kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


def _load_queues_raw() -> dict[str, Any]:
    """JSON load. Windows에서 atomic replace와 동시 read 시 짧은 race가 있어
    OSError 발생 가능 → 짧은 retry로 보호 (writer는 atomic이라 partial JSON
    노출은 불가).
    """
    if not QUEUE_PATH.exists():
        return {"queues": {}}
    last_err: Optional[BaseException] = None
    for _ in range(5):
        try:
            with QUEUE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            last_err = e
            time.sleep(0.01)
    logger.warning("queue load 실패: %s — 빈 큐로 시작", last_err)
    return {"queues": {}}


def _atomic_save_json(path: Path, data: dict[str, Any]) -> None:
    """tmp 파일 쓰기 → os.replace로 원자적 교체 (Windows + Linux 양쪽 보장).

    같은 디렉터리에 tmp 파일을 만들어야 cross-device move 회피.
    pid + uuid suffix로 동시 호출 시 tmp 파일명 충돌 방지.

    Windows: 동시 read 핸들이 열려있는 짧은 순간 os.replace가 PermissionError를
    낼 수 있어 짧은 retry로 보호. Linux는 단발 성공.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp_path = path.parent / tmp_name
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass  # fsync 미지원 환경 (메모리 fs 등)
        # os.replace는 POSIX에선 atomic + read-while-replace 무중단.
        # Windows에선 target이 read 핸들 잡혀있으면 일시 PermissionError → retry.
        # 총 ~5초 retry (락 타임아웃과 같은 한계).
        last_err: Optional[BaseException] = None
        deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.005)
        # 끝까지 실패 — raise
        raise last_err if last_err else RuntimeError("os.replace 실패")
    except Exception:
        # tmp 잔여 정리 (실패 흔적 남기지 않음)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _save_queues_raw(data: dict[str, Any]) -> None:
    """기존 호출처 보존용 — atomic save 위임."""
    _atomic_save_json(QUEUE_PATH, data)


class _SimpleFileLock:
    """portalocker/fcntl 둘 다 없을 때 fallback 락 (Windows venv 등).

    .lock 파일을 O_CREAT|O_EXCL로 만들어 mutual exclusion.
    타임아웃 내 재시도 → 실패 시 TimeoutError.
    """

    def __init__(self, target: Path, timeout: float):
        self.lock_path = target.parent / f".{target.name}.lock"
        self.timeout = timeout
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        last_err: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                self._fd = os.open(
                    str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                # pid 기록 (debug 용)
                try:
                    os.write(self._fd, str(os.getpid()).encode("ascii"))
                except OSError:
                    pass
                return
            except FileExistsError as e:
                last_err = e
                time.sleep(_LOCK_POLL_INTERVAL_SEC)
        raise TimeoutError(
            f"file lock 획득 실패 ({self.timeout}s): {self.lock_path} ({last_err})"
        )

    def release(self) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        except OSError:
            pass
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except OSError:
            pass


def _locked_read_modify_write(
    path: Path,
    modify_fn: Callable[[dict[str, Any]], Any],
) -> Any:
    """파일 락 잡고 load → modify_fn(data) → atomic save 일관 수행.

    Args:
        path: JSON 경로
        modify_fn: data (dict)를 받아 in-place 수정 + 반환값을 호출자에게 전달.
                   반환값 dict에 ``_skip_save=True`` 키가 있으면 save 생략 (검증 실패 등).

    Returns:
        modify_fn의 반환값 (호출자 컨텍스트별).

    Raises:
        TimeoutError: 락 획득 5초 타임아웃.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _inner() -> Any:
        data = _load_queues_raw()
        result = modify_fn(data)
        skip_save = isinstance(result, dict) and bool(result.get("_skip_save"))
        if not skip_save:
            _atomic_save_json(path, data)
        return result

    if _HAS_PORTALOCKER:
        lock_path = path.parent / f".{path.name}.lock"
        # portalocker.Lock은 with 진입 시 timeout 내 EXCLUSIVE 락 획득
        try:
            with portalocker.Lock(
                str(lock_path),
                mode="a+",
                timeout=_LOCK_TIMEOUT_SEC,
                flags=portalocker.LOCK_EX,
            ):
                return _inner()
        except portalocker.LockException as e:
            logger.error("portalocker 락 타임아웃 (%ss): %s", _LOCK_TIMEOUT_SEC, e)
            raise TimeoutError(f"portalocker lock timeout: {e}") from e

    if _HAS_FCNTL:
        lock_path = path.parent / f".{path.name}.lock"
        deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            acquired = False
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(_LOCK_POLL_INTERVAL_SEC)
            if not acquired:
                logger.error("fcntl 락 타임아웃 (%ss): %s", _LOCK_TIMEOUT_SEC, lock_path)
                raise TimeoutError(f"fcntl lock timeout: {lock_path}")
            return _inner()
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass

    # fallback: 단순 lock 파일
    lock = _SimpleFileLock(path, _LOCK_TIMEOUT_SEC)
    lock.acquire()
    try:
        return _inner()
    finally:
        lock.release()


def load_queues() -> dict[str, dict]:
    """저장된 큐 전체 dict 반환 ({ticker: queue_entry})."""
    return _load_queues_raw().get("queues", {})


def _count_active_positions(queues: dict[str, dict]) -> int:
    """활성 큐 수 (PENDING/TRIGGERED 단계 1개 이상 보유 종목)."""
    count = 0
    for entry in queues.values():
        for stage in entry.get("stages", []):
            if stage.get("status") in (STATUS_PENDING, STATUS_TRIGGERED):
                count += 1
                break
    return count


def _build_stages(peak_price: int, available_cash: int) -> list[QueueStage]:
    """3단계 큐 빌드 (LEVELS/RATIOS .env)."""
    stages: list[QueueStage] = []
    if len(SPLIT_LEVELS) != len(SPLIT_RATIOS):
        logger.warning("LEVELS/RATIOS 길이 불일치 — 기본값으로 fallback")
        levels, ratios = [10, 20, 30], [30, 30, 40]
    else:
        levels, ratios = SPLIT_LEVELS, SPLIT_RATIOS

    for i, (pct_drop, alloc_pct) in enumerate(zip(levels, ratios), start=1):
        target_pct = 1.0 - pct_drop / 100.0          # 0.90, 0.80, 0.70
        target_price = int(peak_price * target_pct)
        alloc_amount = min(int(available_cash * alloc_pct / 100), SPLIT_MAX_AMOUNT)
        qty = alloc_amount // max(target_price, 1)
        # 1주차 안전: SPLIT_MAX_QTY > 0이면 수량 cap (차트영웅 max-qty 1 동일 안전망)
        if SPLIT_MAX_QTY > 0:
            qty = min(qty, SPLIT_MAX_QTY)
        stages.append(QueueStage(
            level=i,
            target_pct=target_pct,
            target_price=target_price,
            alloc_ratio=alloc_pct / 100.0,
            alloc_amount=alloc_amount,
            qty=qty,
        ))
    return stages


def register_buy_queue(
    ticker: str,
    peak_price: int,
    available_cash: int,
    name: str = "",
) -> dict:
    """매도 후 천장가 + 가용 현금 → 3단계 큐 자동 등록.

    Args:
        ticker: 종목 코드
        peak_price: 천장가 (MVP-1 detect_peak_signal 결과)
        available_cash: 가용 현금 (KRW)
        name: 종목명 (알림용)

    Returns:
        {"success": bool, "ticker": str, "stages": [...], "error": str}
    """
    if _is_kill_switch_active():
        return {"success": False, "error": "KILL_SWITCH 발동 중 — 등록 정지"}

    if peak_price <= 0:
        return {"success": False, "error": f"peak_price 부적합: {peak_price}"}

    if available_cash < 100_000:
        return {"success": False, "error": f"가용 현금 부족: {available_cash:,}"}

    stages = _build_stages(peak_price, available_cash)
    stage_dicts = [stage.__dict__ for stage in stages]

    # P0-1: 락 잡고 read-modify-write 단일 단위로 처리 (race condition 방지)
    def _modify(raw: dict[str, Any]) -> dict:
        queues = raw.setdefault("queues", {})
        # 동일 종목 기존 큐 있으면 덮어씀 (천장 갱신 케이스)
        is_update = ticker in queues
        if not is_update and _count_active_positions(queues) >= MAX_POSITIONS:
            return {
                "success": False,
                "error": f"3종목 한도 도달 ({MAX_POSITIONS}) — 기존 종목 청산 후 재등록",
                "_skip_save": True,
            }
        queues[ticker] = {
            "ticker": ticker,
            "name": name,
            "peak_price": int(peak_price),
            "available_cash": int(available_cash),
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "stages": stage_dicts,
        }
        return {
            "success": True,
            "ticker": ticker,
            "name": name,
            "is_update": is_update,
            "peak_price": peak_price,
            "available_cash": available_cash,
            "stages": stage_dicts,
        }

    try:
        result = _locked_read_modify_write(QUEUE_PATH, _modify)
    except TimeoutError as e:
        logger.error("register_buy_queue 락 타임아웃: %s", e)
        return {"success": False, "error": f"락 타임아웃: {e}"}

    # _skip_save 마커는 호출자에게 노출하지 않음
    result.pop("_skip_save", None)
    return result


def _fetch_current_price(broker, ticker: str) -> int:
    """현재가 fetch (adaptive_position_manager와 동일 패턴)."""
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}
        return int(str(output.get("stck_prpr", 0)).replace(",", "") or 0)
    except Exception as e:
        logger.warning("price fetch %s 실패: %s", ticker, e)
        return 0


def _is_expired(registered_at: str, expiry_days: int | None = None) -> bool:
    """등록 후 만료 여부.

    ★ M6 fix (5/27 검수): expiry_days 인자 지원 (AI 동조 큐 3일 만료 honor).
    None이면 전역 QUEUE_EXPIRY_DAYS 사용 (기본 60일).
    """
    days_limit = expiry_days if expiry_days is not None else QUEUE_EXPIRY_DAYS
    try:
        reg = datetime.fromisoformat(registered_at)
        return (datetime.now() - reg).days > days_limit
    except (ValueError, TypeError):
        return False


def execute_auto_buy(broker, ticker: str, stage_dict: dict,
                       intraday_adapter=None, regime: str = "NEUTRAL",
                       *, mode: str | None = None, executor_bot: str | None = None) -> dict:
    """단계별 자동 매수 (ADAPTIVE_AUTO_BUY=1일 때만).

    5/26 통합: 매수 직전 adaptive_entry_gates (H4 VWAP + H5 호가 + H6 매물대 + H7 ATR)
    실행. 모든 게이트 통과 시에만 buy_limit. 환경변수 ADAPTIVE_ENTRY_GATES_ENABLED=0이면
    게이트 우회 (백업/안전망).

    Args:
        broker: KisOrderAdapter (fetch_price + fetch_ohlcv + buy_limit)
        ticker: 종목코드
        stage_dict: 큐의 단일 stage dict (target_price, qty, level 등 포함)
        intraday_adapter: KisIntradayAdapter (호가창 — Optional)
        regime: 시장 레짐 ('BULL'/'NEUTRAL'/'BEARISH') — ATR 배수 선택

    Returns:
        {"success": bool, "order_id": str, "price": int, "qty": int,
         "error": str, "block_reason": str, "atr_stop": dict | None,
         "gate_summary": dict}
    """
    if not AUTO_BUY:
        return {"success": False, "error": "ADAPTIVE_AUTO_BUY=0 — 알림만"}

    target_price = int(stage_dict.get("target_price", 0))
    qty = int(stage_dict.get("qty", 0))
    if target_price <= 0 or qty <= 0:
        return {"success": False, "error": f"target_price/qty 부적합 ({target_price}/{qty})"}

    # 진입 게이트 (H4/H5/H6/H7) — 환경변수로 ON/OFF
    gate_summary: dict = {}
    if os.getenv("ADAPTIVE_ENTRY_GATES_ENABLED", "0") == "1":
        try:
            from src.use_cases.adaptive_entry_gates import check_all_entry_gates
            gate = check_all_entry_gates(
                ticker=ticker,
                target_price=target_price,
                broker=broker,
                intraday_adapter=intraday_adapter,
                regime=regime,
            )
            gate_summary = {
                "vwap_reason": gate.vwap_reason,
                "vwap_dev_pct": gate.vwap_dev_pct,
                "orderbook_reason": gate.orderbook_reason,
                "supply_zone_reason": gate.supply_zone_reason,
                "supply_position": gate.supply_position,
                "is_vwap_dip": gate.is_vwap_dip,
                "is_strong_bid": gate.is_strong_bid,
                "is_poc_breakout": gate.is_poc_breakout,
            }
            if not gate.allow:
                logger.warning(
                    "[entry gates] %s L%s 차단: %s",
                    ticker, stage_dict.get("level"), gate.block_reason,
                )
                return {
                    "success": False,
                    "error": f"entry gate blocked: {gate.block_reason}",
                    "block_reason": gate.block_reason,
                    "gate_summary": gate_summary,
                }
            # ATR 동적 손익절 결과 (매수 성공 후 stage에 저장됨)
            atr_stop_dict = None
            if gate.atr_stop is not None:
                atr_stop_dict = {
                    "stop_price": gate.atr_stop.stop_price,
                    "target_price": gate.atr_stop.target_price,
                    "stop_pct": gate.atr_stop.stop_pct,
                    "target_pct": gate.atr_stop.target_pct,
                    "source": gate.atr_stop.source,
                    "atr_value": gate.atr_stop.atr_value,
                }
                gate_summary["atr_stop"] = atr_stop_dict
        except Exception as e:
            logger.warning(
                "[entry gates] %s 게이트 체크 실패 — fail-open: %s",
                ticker, e,
            )
            gate_summary["error"] = str(e)

    try:
        # 지정가 매수 (target_price) — 5/28 코덱스: mode/executor_bot 전달
        adapter_kwargs = {}
        if mode is not None or executor_bot is not None:
            adapter_kwargs = {"mode": mode, "executor_bot": executor_bot}
        order = broker.buy_limit(ticker, target_price, qty, **adapter_kwargs)
        order_id = getattr(order, "order_id", "") or ""
        return {
            "success": True,
            "order_id": order_id,
            "price": target_price,
            "qty": qty,
            "gate_summary": gate_summary,
        }
    except Exception as e:
        logger.error("auto buy %s L%s 실패: %s", ticker, stage_dict.get("level"), e)
        return {"success": False, "error": str(e), "gate_summary": gate_summary}


def check_and_trigger_queues(
    broker, intraday_adapter=None, regime: str = "NEUTRAL",
    *, mode: str | None = None, executor_bot: str | None = None,
) -> list[dict]:
    """모든 활성 큐 순회 + 가격 도달 단계 트리거.

    매 5분 cron에서 호출.
    5/28 코덱스 검수: mode/executor_bot 명시 시 execute_auto_buy 내부 buy_limit에 전달.

    Args:
        broker: KisOrderAdapter (fetch_price + buy_limit)
        intraday_adapter: KisIntradayAdapter (H5 호가 / 동시호가 / 4수급 게이트 필수) ★ C2 fix
        regime: 'BULL'/'NEUTRAL'/'BEARISH' — ATR 배수 선택

    Returns:
        [{ticker, name, level, status, target_price, current_price, ...}, ...]
    """
    triggers: list[dict] = []

    if _is_kill_switch_active():
        logger.info("KILL_SWITCH 발동 — 큐 트리거 정지")
        return triggers

    # P0-1: 락 잡고 read-modify-write 일관 처리.
    # broker.fetch_price / buy_limit는 외부 I/O라 락 안에서 호출되면 락 보유 시간이
    # 길어지지만, 5초 타임아웃 + cron 30분 간격이라 실용적으로 안전.
    def _modify(raw: dict[str, Any]) -> dict:
        queues = raw.setdefault("queues", {})
        modified = False
        for ticker, entry in list(queues.items()):
            # 만료 체크
            # ★ M6 fix (5/27): AI 동조 큐는 expiry_days=3, 기본은 60일 (entry별)
            if _is_expired(entry.get("registered_at", ""), entry.get("expiry_days")):
                for stage in entry.get("stages", []):
                    if stage.get("status") in (STATUS_PENDING, STATUS_TRIGGERED):
                        stage["status"] = STATUS_EXPIRED
                        modified = True
                triggers.append({
                    "ticker": ticker,
                    "name": entry.get("name", ""),
                    "event": "EXPIRED",
                    "registered_at": entry.get("registered_at", ""),
                })
                continue

            current_price = _fetch_current_price(broker, ticker)
            if current_price <= 0:
                continue

            for stage in entry.get("stages", []):
                if stage.get("status") != STATUS_PENDING:
                    continue

                target_price = int(stage.get("target_price", 0))
                if target_price <= 0:
                    continue

                # 도달: 현재가 ≤ 지정가 (떨어져서 지정가 도달)
                if current_price <= target_price:
                    stage["triggered_at"] = datetime.now().isoformat(timespec="seconds")

                    # 자동 매수 시도 (★ C2 fix + 5/28 코덱스: mode/executor_bot 전달)
                    buy_result = execute_auto_buy(
                        broker, ticker, stage,
                        intraday_adapter=intraday_adapter, regime=regime,
                        mode=mode, executor_bot=executor_bot,
                    )
                    if buy_result["success"]:
                        stage["status"] = STATUS_FILLED
                        stage["order_id"] = buy_result.get("order_id", "")
                        actual_price = buy_result.get("price", target_price)
                        stage["actual_price"] = actual_price
                        stage["actual_qty"] = buy_result.get("qty", stage.get("qty", 0))
                        # MVP-2.5: 매수 체결 즉시 빠른 익절 목표가 자동 계산
                        stage["quick_profit_target"] = int(actual_price * (1 + QUICK_PROFIT_PCT / 100))
                    elif buy_result.get("error", "").startswith("ADAPTIVE_AUTO_BUY=0"):
                        # 알림만 모드 — TRIGGERED로 표시 후 추후 수동/자동
                        stage["status"] = STATUS_TRIGGERED
                    else:
                        stage["status"] = STATUS_FAILED
                        stage["error"] = buy_result.get("error", "")

                    modified = True
                    triggers.append({
                        "ticker": ticker,
                        "name": entry.get("name", ""),
                        "level": stage.get("level"),
                        "status": stage.get("status"),
                        "target_price": target_price,
                        "current_price": current_price,
                        "peak_price": entry.get("peak_price", 0),
                        "qty": stage.get("qty", 0),
                        "alloc_amount": stage.get("alloc_amount", 0),
                        "order_id": stage.get("order_id"),
                        "error": stage.get("error"),
                    })
        # 변경 없으면 save 생략 (불필요 I/O 회피)
        return {"_skip_save": not modified}

    try:
        _locked_read_modify_write(QUEUE_PATH, _modify)
    except TimeoutError as e:
        logger.error("check_and_trigger_queues 락 타임아웃: %s", e)
        # triggers는 부분적으로 채워질 수 있으나 save 안 된 상태 — 호출자에게는
        # 결과 노출 X (다음 cron에서 재시도)
        return []

    return triggers


def get_queue_status(ticker: str) -> Optional[dict]:
    """특정 종목 큐 상태."""
    queues = load_queues()
    return queues.get(ticker)


def format_trigger_for_telegram(trigger: dict) -> str:
    """텔레그램 알림용 포맷."""
    name = trigger.get("name") or trigger.get("ticker", "")

    if trigger.get("event") == "EXPIRED":
        return (
            f"⏰ 분할매수 큐 만료 [{name}]\n"
            f"  등록일: {trigger.get('registered_at', '')[:10]}\n"
            f"  {QUEUE_EXPIRY_DAYS}일 경과 — 큐 자동 정리"
        )

    status = trigger.get("status", "")
    level = trigger.get("level", "?")
    target = int(trigger.get("target_price", 0))
    current = int(trigger.get("current_price", 0))
    peak = int(trigger.get("peak_price", 0))
    qty = int(trigger.get("qty", 0))
    alloc = int(trigger.get("alloc_amount", 0))

    pct_from_peak = ((current / peak) - 1) * 100 if peak > 0 else 0.0

    if status == STATUS_FILLED:
        head = f"✅ 분할매수 L{level} 체결 [{name}]"
        action_line = f"  주문ID: {trigger.get('order_id', '')}"
    elif status == STATUS_TRIGGERED:
        head = f"🔔 분할매수 L{level} 가격 도달 [{name}] (알림만)"
        action_line = f"  ADAPTIVE_AUTO_BUY=0 — 수동 매수 또는 활성화 검토"
    elif status == STATUS_FAILED:
        head = f"⚠️ 분할매수 L{level} 실패 [{name}]"
        action_line = f"  오류: {trigger.get('error', '')[:60]}"
    else:
        head = f"❓ 분할매수 L{level} 상태={status} [{name}]"
        action_line = ""

    lines = [
        head,
        f"  천장: {peak:,} → 현재: {current:,} ({pct_from_peak:+.2f}%)",
        f"  지정가: {target:,} (L{level} = 천장 -{int((1 - target/peak)*100) if peak else 0}%)",
        f"  배정: {alloc:,}원 / {qty}주",
    ]
    if action_line:
        lines.append(action_line)
    return "\n".join(lines)


def clear_queue(ticker: str) -> bool:
    """특정 종목 큐 삭제 (수동 청산용)."""

    def _modify(raw: dict[str, Any]) -> dict:
        queues = raw.setdefault("queues", {})
        if ticker in queues:
            del queues[ticker]
            return {"_removed": True}
        return {"_removed": False, "_skip_save": True}

    try:
        result = _locked_read_modify_write(QUEUE_PATH, _modify)
    except TimeoutError as e:
        logger.error("clear_queue 락 타임아웃: %s", e)
        return False
    return bool(result.get("_removed"))
