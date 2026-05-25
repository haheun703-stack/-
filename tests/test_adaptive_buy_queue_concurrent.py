"""test_adaptive_buy_queue_concurrent.py — P0-1 JSON race condition 회귀 방어.

배경 (5/25 sub-agent A — P0-1 분업):
  bkit:code-analyzer 5/24 검수 P0-1 판정 — "JSON race condition: 동시 쓰기 시
  데이터 손실 가능". 5/26(화) 09:30부터 cron `*/30 9-15 * * 1-5`로
  run_adaptive_cycle.py가 30분마다 실행. MVP-1 자동 매도 → MVP-2 큐 등록이
  같은 사이클에서 발생 → JSON 쓰기 충돌 가능.

수정 (`src/use_cases/adaptive_buy_queue.py`):
  - _atomic_save_json: tmp 파일 쓰기 → os.replace로 원자적 교체
  - _locked_read_modify_write: portalocker → fcntl → 단순 .lock 파일 fallback
  - register_buy_queue / check_and_trigger_queues / clear_queue 모두 적용

검증 시나리오:
  C-01. 5 스레드 동시 register_buy_queue → 5건 모두 저장 (race 손실 0건)
  C-02. mid-write 중 read 시 partial JSON 노출 0건 (atomic 보장)
  C-03. clear_queue 동시 호출 + register 동시 호출 → 일관 상태 유지
  C-04. _atomic_save_json 단위 동작 (tmp 잔여 정리 + 원자성)

실행:
  python -m pytest tests/test_adaptive_buy_queue_concurrent.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_broker(current_price: int = 0):
    broker = MagicMock()
    broker.fetch_price.return_value = {"output": {"stck_prpr": str(current_price)}}
    return broker


class TestAdaptiveBuyQueueConcurrent(unittest.TestCase):
    """P0-1 동시성/원자성 검증."""

    def setUp(self):
        # 격리: 모듈 reload + QUEUE_PATH/KILL_SWITCH_PATH 패치
        if "src.use_cases.adaptive_buy_queue" in sys.modules:
            del sys.modules["src.use_cases.adaptive_buy_queue"]

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        import src.use_cases.adaptive_buy_queue as mod
        self.mod = mod
        mod.QUEUE_PATH = self.tmp_path / "adaptive_buy_queue.json"
        mod.KILL_SWITCH_PATH = self.tmp_path / "kill_switch.flag"
        # 3종목 한도 우회 (테스트에서 5종목 등록 필요)
        mod.MAX_POSITIONS = 99

    def tearDown(self):
        self.tmpdir.cleanup()

    # ─────────────────────────────────────────────────────────
    # C-01. 5 스레드 동시 register → 5건 모두 저장
    # ─────────────────────────────────────────────────────────
    def test_C01_concurrent_register_no_data_loss(self):
        """5 스레드가 동시에 register_buy_queue → 5건 모두 저장."""
        tickers = [f"T{i:05d}" for i in range(5)]
        results: list[dict] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(len(tickers))

        def _worker(ticker: str):
            barrier.wait()  # 동시 출발
            r = self.mod.register_buy_queue(
                ticker=ticker, peak_price=10_000, available_cash=1_000_000, name=ticker
            )
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in tickers]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10)

        # 모든 스레드 종료
        for th in threads:
            self.assertFalse(th.is_alive(), f"스레드 미종료: {th}")

        # 모든 호출 성공
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertTrue(r.get("success"), f"등록 실패: {r}")

        # 최종 저장 상태: 5종목 모두 저장
        queues = self.mod.load_queues()
        self.assertEqual(len(queues), 5, f"저장된 종목 수 불일치: {sorted(queues.keys())}")
        for t in tickers:
            self.assertIn(t, queues)
            self.assertEqual(queues[t]["peak_price"], 10_000)
            self.assertEqual(len(queues[t]["stages"]), 3)

    # ─────────────────────────────────────────────────────────
    # C-02. atomic write 보장 — mid-write 중 read 시 partial JSON 0건
    # ─────────────────────────────────────────────────────────
    def test_C02_atomic_write_no_partial_read(self):
        """write 진행 중 read해도 항상 일관된 JSON (atomic 보장).

        register와 load_queues를 동시에 반복 실행해 partial JSON / 빈 dict 노출
        여부 검증.
        """
        # 초기 1건 저장
        self.mod.register_buy_queue("INIT0", 10_000, 1_000_000, name="INIT0")

        stop = threading.Event()
        partial_read_count = {"n": 0}
        partial_lock = threading.Lock()

        def _writer():
            # 200ms 동안 반복 register (덮어쓰기 케이스 → 매번 write 발생)
            end = time.monotonic() + 0.2
            i = 0
            while not stop.is_set() and time.monotonic() < end:
                self.mod.register_buy_queue(
                    f"W{i % 3}", 10_000 + i, 1_000_000, name=f"W{i}"
                )
                i += 1

        def _reader():
            end = time.monotonic() + 0.2
            while not stop.is_set() and time.monotonic() < end:
                try:
                    # _load_queues_raw는 락 외부 read이므로 atomic save 효과를 직접 검증
                    raw = self.mod._load_queues_raw()
                    if "queues" not in raw:
                        with partial_lock:
                            partial_read_count["n"] += 1
                    # 파일이 비어있거나 깨졌으면 _load_queues_raw가 빈 dict 반환 (atomic 안 됐을 때)
                    if raw.get("queues") is None:
                        with partial_lock:
                            partial_read_count["n"] += 1
                except Exception:
                    with partial_lock:
                        partial_read_count["n"] += 1

        writers = [threading.Thread(target=_writer) for _ in range(3)]
        readers = [threading.Thread(target=_reader) for _ in range(3)]
        for th in writers + readers:
            th.start()
        for th in writers + readers:
            th.join(timeout=5)
        stop.set()

        # partial read 0건 (atomic write 효과)
        self.assertEqual(
            partial_read_count["n"], 0,
            f"partial / corrupted read 발생: {partial_read_count['n']}회"
        )

        # 최종 상태: INIT0 + W0/W1/W2 = 4종목
        queues = self.mod.load_queues()
        self.assertIn("INIT0", queues)
        # W* 종목 1개 이상 저장
        self.assertTrue(
            any(t.startswith("W") for t in queues),
            f"W* 종목 미저장: {sorted(queues.keys())}"
        )

    # ─────────────────────────────────────────────────────────
    # C-03. register + clear 동시 호출 → 일관 상태
    # ─────────────────────────────────────────────────────────
    def test_C03_concurrent_register_and_clear(self):
        """register/clear 동시 호출 시 손실/이중 삭제 없이 일관 상태 유지."""
        # 사전 3건 저장
        for i in range(3):
            self.mod.register_buy_queue(f"X{i}", 10_000, 1_000_000, name=f"X{i}")

        def _register_more():
            for i in range(3, 8):
                self.mod.register_buy_queue(f"X{i}", 10_000, 1_000_000, name=f"X{i}")

        def _clear_some():
            for i in range(0, 3):
                self.mod.clear_queue(f"X{i}")

        threads = [
            threading.Thread(target=_register_more),
            threading.Thread(target=_clear_some),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5)

        queues = self.mod.load_queues()
        # X0~X2 삭제 + X3~X7 추가 → 최종 5종목 (X3, X4, X5, X6, X7)
        # 또는 register와 clear 순서에 따라 X0~X2가 살아남을 수도 있음
        # → 핵심은 "JSON 깨짐 없음 + 종목 수 ≥ 5"
        self.assertGreaterEqual(
            len(queues), 5,
            f"종목 수 부족 (race 손실 가능): {sorted(queues.keys())}"
        )
        # X3~X7는 register_more가 마지막 종료될 때까지 살아있어야 함
        for i in range(3, 8):
            self.assertIn(f"X{i}", queues)

    # ─────────────────────────────────────────────────────────
    # C-04. _atomic_save_json 단위 검증
    # ─────────────────────────────────────────────────────────
    def test_C04_atomic_save_unit(self):
        """_atomic_save_json: tmp 파일 잔여 0건 + 원자적 교체 확인."""
        target = self.tmp_path / "atomic_test.json"
        data = {"foo": "bar", "list": [1, 2, 3], "한글": "값"}
        self.mod._atomic_save_json(target, data)

        # 파일 존재 + 내용 일치
        self.assertTrue(target.exists())
        with target.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

        # tmp 파일 잔여 0건
        tmp_residue = [p for p in self.tmp_path.iterdir() if p.name.startswith(".atomic_test.json.tmp")]
        self.assertEqual(tmp_residue, [], f"tmp 잔여: {tmp_residue}")

    # ─────────────────────────────────────────────────────────
    # C-05. 락 타임아웃 동작 (옵션) — _SimpleFileLock 직접 검증
    # ─────────────────────────────────────────────────────────
    def test_C05_lock_timeout_raises(self):
        """fallback _SimpleFileLock 타임아웃 발동 검증."""
        target = self.tmp_path / "lock_test.json"
        # 락 인위적으로 잡아둠
        lock_path = target.parent / f".{target.name}.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            lock = self.mod._SimpleFileLock(target, timeout=0.2)
            with self.assertRaises(TimeoutError):
                lock.acquire()
        finally:
            os.close(fd)
            try:
                lock_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
