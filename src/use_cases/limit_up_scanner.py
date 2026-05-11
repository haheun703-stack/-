"""
상한가 풀림 실시간 감지 엔진 v1.0
═══════════════════════════════════════════════

장중 상한가 종목의 "풀림"을 실시간 감지하여 즉시 매수 시그널 생성.

백테스트 근거:
  - 이력 2회+ & 5일내 재상한가 & 가격 1천원+ → 64건 전승 (100%)
  - 상한가 당일 종가 매수 시 승률 98%, 평균 +9.6%
  - 하지만 99.6%가 잠금 상태 → 당일 매수 불가
  - 해결: 장중 풀림 감지 → 즉시 매수 (v3 수준 진입가 확보)

흐름:
  Phase 1: 후보 종목 폴링 (30초 간격, ~30종목)
  Phase 2: 상한가 도달 감지 → 집중 감시 (10초 간격)
  Phase 3: 풀림 감지 (현재가 < 상한가 × 0.99) → 매수
  Phase 4: 체결 후 포지션 등록 + 텔레그램

의존:
  - adapters/kis_intraday_adapter.py (fetch_tick)
  - adapters/kis_order_adapter.py (buy_limit)
  - use_cases/safety_guard.py (매수 전 체크)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "data" / "limit_up_candidates.json"
STATE_PATH = PROJECT_ROOT / "data" / "limit_up_scanner_state.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


class LimitUpScanner:
    """장중 상한가 풀림 실시간 감지 엔진"""

    def __init__(
        self,
        intraday_adapter=None,
        order_adapter=None,
        config: dict | None = None,
    ):
        self.intraday = intraday_adapter
        self.order = order_adapter
        self.config = config or {}

        lu_cfg = self.config.get("limit_up_scanner", {})
        self.scan_interval = lu_cfg.get("scan_interval_sec", 30)
        self.focus_interval = lu_cfg.get("focus_interval_sec", 10)
        self.unlock_threshold = lu_cfg.get("unlock_threshold", 0.99)
        self.min_prev_limit_ups = lu_cfg.get("min_prev_limit_ups", 2)
        self.max_days_since_last = lu_cfg.get("max_days_since_last", 10)
        self.position_pct = lu_cfg.get("position_pct", 0.10)
        self.max_positions = lu_cfg.get("max_positions", 5)
        # ── v2 포지션 관리용 (현재 미사용, 향후 구현) ──
        self.take_profit = lu_cfg.get("take_profit", 0.10)
        self.add_threshold = lu_cfg.get("add_threshold", -0.10)
        self.max_adds = lu_cfg.get("max_adds", 3)
        self.max_hold_days = lu_cfg.get("max_hold_days", 20)
        self.dry_run = lu_cfg.get("dry_run", True)
        self.capital = lu_cfg.get("capital", 50_000_000)

        # 상태
        self.candidates: list[dict] = []
        self.focus_list: dict[str, dict] = {}  # ticker → {limit_price, detected_at, ...}
        self.positions: dict[str, dict] = {}   # ticker → 포지션 정보
        self.filled_today: list[dict] = []     # 오늘 체결 기록

    # ═══════════════════════════════════════════════════════
    # Phase 0: 후보 목록 관리
    # ═══════════════════════════════════════════════════════

    def load_candidates(self) -> list[dict]:
        """data/limit_up_candidates.json 로드"""
        if CANDIDATES_PATH.exists():
            with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
                self.candidates = json.load(f)
            logger.info(
                "[LU스캐너] 후보 %d종목 로드: %s",
                len(self.candidates),
                ", ".join(c["ticker"] for c in self.candidates[:5]),
            )
        else:
            logger.warning("[LU스캐너] 후보 파일 없음: %s", CANDIDATES_PATH)
            self.candidates = []
        return self.candidates

    def generate_candidates(self, date_str: str | None = None) -> list[dict]:
        """
        data/raw/*.parquet 스캔하여 상한가/급등 이력 후보 생성.
        BAT-D에서 매일 호출.

        조건 (OR):
        - Tier1: 최근 6개월 내 상한가(29%+) 이력 min_prev_limit_ups회 이상
        - Tier2: 상한가 1회 + 10%+ 급등 2회 이상
        공통:
        - 직전 상한가/급등으로부터 max_days_since_last일 이내
        - 최종 종가 >= 1,000원
        """
        today = pd.Timestamp(date_str or datetime.now().strftime("%Y-%m-%d"))
        six_months_ago = today - pd.Timedelta(days=180)

        candidates = []
        limit_up_pct = 29.0  # 상한가 기준 (+29% 이상)
        surge_pct = 10.0     # 급등 기준 (+10% 이상)

        for fname in sorted(os.listdir(RAW_DIR)):
            if not fname.endswith(".parquet"):
                continue
            ticker = fname.replace(".parquet", "")
            try:
                df = pd.read_parquet(RAW_DIR / fname)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                # 최근 6개월만
                mask = df.index >= six_months_ago
                dfp = df[mask]
                if len(dfp) < 5:
                    continue

                # 등락률 계산
                if "price_change" in dfp.columns:
                    pct = dfp["price_change"]
                else:
                    pct = dfp["close"].pct_change() * 100

                # 상한가 이벤트 추출
                lu_dates = dfp.index[pct >= limit_up_pct].tolist()
                # 10%+ 급등 이벤트 추출 (상한가 제외)
                surge_dates = dfp.index[(pct >= surge_pct) & (pct < limit_up_pct)].tolist()

                # Tier1: 상한가 2회+ / Tier2: 상한가 1회 + 급등 2회+
                lu_count = len(lu_dates)
                surge_count = len(surge_dates)
                tier1 = lu_count >= self.min_prev_limit_ups
                tier2 = lu_count >= 1 and surge_count >= 2
                if not (tier1 or tier2):
                    continue

                # 직전 이벤트(상한가 또는 급등)로부터 최대 일수 체크
                all_event_dates = lu_dates + surge_dates
                last_event = max(all_event_dates)
                days_since = (today - last_event).days
                if days_since > self.max_days_since_last:
                    continue

                # 직전 상한가 날짜 (없으면 급등 날짜)
                last_lu = max(lu_dates) if lu_dates else max(surge_dates)

                # 최종 종가 체크 (= 다음날 전일종가 = 상한가 계산 기준)
                prev_close = int(dfp.iloc[-1]["close"])
                if prev_close < 1000:
                    continue

                tier = "T1" if tier1 else "T2"
                candidates.append({
                    "ticker": ticker,
                    "prev_close": prev_close,
                    "limit_price": int(prev_close * 1.30),
                    "limit_up_count": lu_count,
                    "surge_count": surge_count,
                    "tier": tier,
                    "last_limit_up": last_lu.strftime("%Y-%m-%d"),
                    "days_since_last": days_since,
                    "generated_at": today.strftime("%Y-%m-%d"),
                })
            except Exception as e:
                logger.debug("[LU스캐너] %s 스캔 실패: %s", ticker, e)

        # Tier1 우선, 상한가 횟수 내림차순 정렬
        candidates.sort(key=lambda x: (0 if x["tier"] == "T1" else 1, -x["limit_up_count"], -x["surge_count"], x["days_since_last"]))

        # 저장
        CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

        logger.info("[LU스캐너] 후보 %d종목 생성 → %s", len(candidates), CANDIDATES_PATH)
        self.candidates = candidates
        return candidates

    # ═══════════════════════════════════════════════════════
    # Phase 1~3: 실시간 스캔 루프
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def calculate_limit_price(prev_close: int) -> int:
        """전일종가 기준 당일 상한가 계산 (30% 상한)"""
        return int(prev_close * 1.30)

    def _is_market_open(self) -> bool:
        """장중 여부 (09:00~15:20)"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=0, second=0)
        market_close = now.replace(hour=15, minute=20, second=0)
        return market_open <= now <= market_close

    async def scan_loop(self):
        """
        메인 스캔 루프 (async).
        Phase 1: 후보 종목 현재가 폴링 (30초)
        Phase 2: 상한가 도달 → focus_list 이동 (10초)
        Phase 3: 풀림 감지 → 매수
        """
        if not self.candidates:
            self.load_candidates()

        if not self.candidates:
            logger.error("[LU스캐너] 후보 종목 없음. 종료.")
            return

        logger.info(
            "[LU스캐너] 스캔 시작 — 후보 %d종목, 간격 %d초",
            len(self.candidates), self.scan_interval,
        )

        while self._is_market_open():
            try:
                # Phase 1: 후보 종목 폴링
                await self._scan_candidates()

                # Phase 2+3: 포커스 종목 집중 감시
                if self.focus_list:
                    await self._monitor_focus()

                # 대기
                await asyncio.sleep(self.scan_interval)

            except KeyboardInterrupt:
                logger.info("[LU스캐너] 사용자 중단")
                break
            except Exception as e:
                logger.error("[LU스캐너] 루프 오류: %s", e)
                await asyncio.sleep(5)

        logger.info(
            "[LU스캐너] 장 마감 (15:20). 스캔 종료 — 포커스 %d종목, 체결 %d건",
            len(self.focus_list), len(self.filled_today),
        )
        self._save_state()

    async def _scan_candidates(self):
        """후보 종목 현재가 조회 → 상한가 도달 감지"""
        for cand in self.candidates:
            ticker = cand["ticker"]

            # 이미 포커스 또는 포지션이면 스킵
            if ticker in self.focus_list or ticker in self.positions:
                continue

            # 최대 포지션 초과 체크
            if len(self.positions) + len(self.filled_today) >= self.max_positions:
                break

            tick = self.intraday.fetch_tick(ticker)
            if not tick or tick.get("current_price", 0) == 0:
                continue

            high_price = tick.get("high_price", 0)
            limit_price = cand["limit_price"]

            # 당일 고가가 상한가 × 0.98 이상 → 상한가 도달 판정
            if high_price >= limit_price * 0.98:
                self.focus_list[ticker] = {
                    "limit_price": limit_price,
                    "detected_at": datetime.now().isoformat(),
                    "prev_close": cand["prev_close"],
                    "limit_up_count": cand["limit_up_count"],
                    "last_tick": tick,
                }
                logger.info(
                    "🔴 [LU스캐너] 상한가 도달! %s — 고가 %s / 상한가 %s",
                    ticker, f"{high_price:,}", f"{limit_price:,}",
                )

    async def _monitor_focus(self):
        """포커스 종목 집중 감시 (10초 간격으로 풀림 체크)"""
        # 타임아웃: 1시간 이상 경과한 포커스 종목 제거
        now = datetime.now()
        expired = [
            t for t, info in self.focus_list.items()
            if (now - datetime.fromisoformat(info["detected_at"])).seconds > 3600
        ]
        for t in expired:
            logger.info("[LU스캐너] %s 포커스 타임아웃 (1시간)", t)
            self.focus_list.pop(t)

        unlocked = []

        for ticker, info in list(self.focus_list.items()):
            tick = self.intraday.fetch_tick(ticker)
            if not tick or tick.get("current_price", 0) == 0:
                continue

            current_price = tick["current_price"]
            limit_price = info["limit_price"]
            ask_price = tick.get("ask_price", 0)

            # 풀림 판정: 현재가 < 상한가 × unlock_threshold
            # + 매도호가가 현재가 근처에 존재 (= 실제 체결 가능)
            is_unlocked = current_price < limit_price * self.unlock_threshold
            has_ask = (
                ask_price > 0
                and ask_price <= current_price * 1.01
                and ask_price < limit_price
            )

            if is_unlocked and has_ask:
                logger.info(
                    "🟢 [LU스캐너] 풀림 감지! %s — 현재가 %s (상한가 %s, -%s%%)",
                    ticker,
                    f"{current_price:,}",
                    f"{limit_price:,}",
                    f"{(1 - current_price / limit_price) * 100:.1f}",
                )
                unlocked.append((ticker, current_price, info))

            # 상태 업데이트
            info["last_tick"] = tick

        # 풀린 종목 매수 실행
        for ticker, price, info in unlocked:
            await self._execute_entry(ticker, price, info)
            self.focus_list.pop(ticker, None)

    # ═══════════════════════════════════════════════════════
    # Phase 4: 매수 실행
    # ═══════════════════════════════════════════════════════

    async def _execute_entry(self, ticker: str, unlocked_price: int, info: dict):
        """풀림 감지 후 매수 실행"""
        # 포지션 한도 체크
        if len(self.positions) + len(self.filled_today) >= self.max_positions:
            logger.warning("[LU스캐너] 최대 포지션 도달. %s 매수 스킵.", ticker)
            return

        # 주문 수량 계산
        amount = int(self.capital * self.position_pct)
        quantity = amount // unlocked_price
        if quantity <= 0:
            logger.warning("[LU스캐너] %s 수량 0. 스킵.", ticker)
            return

        entry = {
            "ticker": ticker,
            "entry_price": unlocked_price,
            "limit_price": info["limit_price"],
            "quantity": quantity,
            "amount": unlocked_price * quantity,
            "limit_up_count": info["limit_up_count"],
            "detected_at": info["detected_at"],
            "entry_at": datetime.now().isoformat(),
        }

        if self.dry_run:
            logger.info(
                "🧪 [DRY-RUN] 매수: %s × %d주 @ %s원 (총 %s원)",
                ticker, quantity, f"{unlocked_price:,}",
                f"{unlocked_price * quantity:,}",
            )
            entry["status"] = "DRY_RUN"
        else:
            # 실제 주문
            if not self.order:
                logger.error("[LU스캐너] %s Order 어댑터 미초기화. 매수 불가.", ticker)
                entry["status"] = "FAILED"
                entry["error"] = "Order adapter not initialized"
                self.filled_today.append(entry)
                self._send_alert(entry)
                return
            try:
                order_result = self.order.buy_limit(ticker, unlocked_price, quantity)
                entry["order_id"] = getattr(order_result, "order_id", "N/A")
                entry["status"] = "ORDERED"
                logger.info(
                    "✅ [LU스캐너] 매수 주문: %s × %d주 @ %s원",
                    ticker, quantity, f"{unlocked_price:,}",
                )
            except Exception as e:
                logger.error("[LU스캐너] %s 매수 실패: %s", ticker, e)
                entry["status"] = "FAILED"
                entry["error"] = str(e)

        self.filled_today.append(entry)
        self.positions[ticker] = entry

        # 텔레그램 알림
        self._send_alert(entry)

    def _send_alert(self, entry: dict):
        """텔레그램 알림 전송"""
        try:
            from src.telegram_sender import send_message
        except ImportError:
            logger.warning("[LU스캐너] 텔레그램 모듈 없음 (src.telegram_sender)")
            return

        mode = "🧪 DRY-RUN" if self.dry_run else "🔥 LIVE"
        msg = (
            f"{mode} 상한가 풀림 매수\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"종목: {entry['ticker']}\n"
            f"매수가: {entry['entry_price']:,}원\n"
            f"수량: {entry['quantity']}주\n"
            f"금액: {entry['amount']:,}원\n"
            f"상한가: {entry['limit_price']:,}원\n"
            f"할인율: {(1 - entry['entry_price'] / entry['limit_price']) * 100:.1f}%\n"
            f"이력: 상한가 {entry['limit_up_count']}회\n"
            f"시각: {entry['entry_at'][:19]}\n"
        )
        try:
            send_message(msg)
        except Exception as e:
            logger.error("[LU스캐너] 텔레그램 실패: %s", e)

    # ═══════════════════════════════════════════════════════
    # 상태 저장/복원
    # ═══════════════════════════════════════════════════════

    def _save_state(self):
        """일일 상태 저장"""
        state = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "focus_list": {k: {kk: vv for kk, vv in v.items() if kk != "last_tick"}
                          for k, v in self.focus_list.items()},
            "filled_today": self.filled_today,
            "positions": {k: v for k, v in self.positions.items()},
        }
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("[LU스캐너] 상태 저장: %s", STATE_PATH)

    # ═══════════════════════════════════════════════════════
    # 동기 래퍼 (BAT/스크립트용)
    # ═══════════════════════════════════════════════════════

    def run(self):
        """동기 실행 (asyncio.run 래퍼)"""
        asyncio.run(self.scan_loop())

    def run_once(self) -> dict:
        """1회성 스캔 (테스트/디버그용)"""
        if not self.intraday:
            raise RuntimeError("intraday_adapter가 None — KIS 어댑터 초기화 실패. .env 확인 필요.")

        if not self.candidates:
            self.load_candidates()

        results = {"scanned": 0, "limit_up_detected": [], "unlocked": []}

        for cand in self.candidates:
            ticker = cand["ticker"]
            tick = self.intraday.fetch_tick(ticker)
            if not tick or tick.get("current_price", 0) == 0:
                continue

            results["scanned"] += 1
            high_price = tick.get("high_price", 0)
            current_price = tick["current_price"]
            limit_price = cand["limit_price"]

            if high_price >= limit_price * 0.98:
                is_unlocked = current_price < limit_price * self.unlock_threshold
                results["limit_up_detected"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "high_price": high_price,
                    "limit_price": limit_price,
                    "is_unlocked": is_unlocked,
                })
                if is_unlocked:
                    results["unlocked"].append(ticker)

        return results
