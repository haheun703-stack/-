"""
장중 VWAP 모니터 — 5종목 실시간 추적 + 텔레그램 알림

BAT 스케줄: 08:55 시작 → 09:00 장 시작 대기 → 14:00 자동 종료

기능:
  1. 09:01   개장 갭 분석 (전일 대비)
  2. 09:00~09:30  매 3분 스냅샷 → VWAP 축적
  3. 09:30   VWAP 기준선 확정 → 텔레그램 전송
  4. 09:30~14:00  매 5분 모니터링
     - VWAP 대비 ±1.5% 이탈 시 알림
     - 눌림 후 VWAP 회복 시 "진입 기회" 알림
     - 30분마다 현황 요약
  5. 11:30   TradeAdvisor AI 분석 통합 (VWAP 컨텍스트 포함)
  6. 14:00   최종 요약 + 종료

사용법:
  python scripts/run_vwap_monitor.py               # 라이브 (장중)
  python scripts/run_vwap_monitor.py --simulate     # 시뮬레이션 (테스트)
  python scripts/run_vwap_monitor.py --targets 005930:10 000660:5

VWAP 계산:
  VWAP = Σ(Price × ΔVolume) / Σ(ΔVolume)
  - 매 폴링마다 현재가 × (누적거래량 변화) 가중합
  - 09:00~09:30 초반 30분 데이터로 기준선 설정
  - 이후 실시간 갱신
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vwap_monitor")


# ─── 대상 종목 (수동 설정) ───
DEFAULT_TARGETS = [
    {"ticker": "034020", "name": "두산에너빌리티", "qty": 20, "desc": "원전/SMR"},
    {"ticker": "000500", "name": "가온전선", "qty": 20, "desc": "원전 relay"},
    {"ticker": "011690", "name": "와이투솔루션", "qty": 200, "desc": "전력인프라"},
    {"ticker": "028050", "name": "삼성E&A", "qty": 60, "desc": "건설 Tier2"},
    {"ticker": "064350", "name": "현대로템", "qty": 5, "desc": "방산 Tier1"},
]

# ─── 설정 ───
POLL_SEC_OPENING = 180       # 09:00~09:30 → 3분 간격
POLL_SEC_NORMAL = 300        # 09:30~14:00 → 5분 간격
VWAP_DIP_PCT = -1.5          # VWAP 대비 이만큼 하향 → 눌림 알림
VWAP_HOT_PCT = 2.5           # VWAP 대비 이만큼 상향 → 과열 알림
VWAP_RECOVER_PCT = -0.3      # 눌림 후 이 수준 회복 → 진입 기회 알림
ALERT_COOLDOWN_SEC = 900     # 동일 종목 알림 쿨다운 (15분)
SUMMARY_INTERVAL_SEC = 1800  # 30분마다 현황 요약
API_RETRY_MAX = 3            # API 호출 실패 시 재시도 횟수
API_RETRY_DELAY = 10         # 재시도 대기 (초)
API_CIRCUIT_BREAKER_WAIT = 60  # 서킷브레이커 감지 시 대기 (초)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StockTracker — 개별 종목 VWAP 추적기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StockTracker:
    def __init__(self, ticker: str, name: str, qty: int, desc: str):
        self.ticker = ticker
        self.name = name
        self.qty = qty
        self.desc = desc

        # VWAP 누적
        self.sum_pv = 0.0       # Σ(price × Δvolume)
        self.sum_v = 0           # Σ(Δvolume)
        self.vwap = 0.0

        # 스냅샷
        self.snapshots: list[dict] = []
        self.opening_price = 0
        self.prev_close_pct = 0.0

        # 알림 상태
        self.below_vwap = False      # 현재 VWAP 아래인지
        self.dip_count = 0           # 눌림 횟수
        self.last_alert_time = None  # 마지막 알림 시각
        self.last_alert_type = ""    # 마지막 알림 종류

    def update(self, price_info: dict) -> dict | None:
        """새 스냅샷 추가 + VWAP 갱신. 실패 시 None."""
        price = price_info.get("current_price", 0)
        cum_vol = price_info.get("volume", 0)
        if price <= 0:
            return None

        snap = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "price": price,
            "cum_vol": cum_vol,
            "high": price_info.get("high", 0),
            "low": price_info.get("low", 0),
            "open": price_info.get("open", 0),
            "change_pct": price_info.get("change_pct", 0.0),
        }

        # 시가 기록
        if not self.opening_price and snap["open"] > 0:
            self.opening_price = snap["open"]
        self.prev_close_pct = snap["change_pct"]

        # VWAP 갱신
        if self.snapshots:
            prev_vol = self.snapshots[-1]["cum_vol"]
            delta_vol = cum_vol - prev_vol
            if delta_vol > 0:
                self.sum_pv += price * delta_vol
                self.sum_v += delta_vol
                self.vwap = self.sum_pv / self.sum_v
        else:
            # 첫 스냅샷: 초기 VWAP = 현재가
            self.vwap = price

        self.snapshots.append(snap)
        return snap

    @property
    def current_price(self) -> int:
        return self.snapshots[-1]["price"] if self.snapshots else 0

    @property
    def deviation_pct(self) -> float:
        """VWAP 대비 현재가 괴리율 (%)"""
        if self.vwap <= 0:
            return 0.0
        return (self.current_price - self.vwap) / self.vwap * 100

    def check_alert(self) -> str | None:
        """VWAP 이탈/회복 이벤트 판정. 알림 문자열 또는 None."""
        dev = self.deviation_pct
        now = datetime.now()

        # 쿨다운 체크
        if self.last_alert_time:
            elapsed = (now - self.last_alert_time).total_seconds()
            if elapsed < ALERT_COOLDOWN_SEC:
                return None

        alert = None

        # ① VWAP 하향 이탈 → 눌림 매수 기회
        if dev <= VWAP_DIP_PCT and not self.below_vwap:
            self.below_vwap = True
            self.dip_count += 1
            alert = (
                f"⬇️ {self.name} VWAP 눌림 (#{self.dip_count})\n"
                f"   현재 {self.current_price:,}원 vs VWAP {int(self.vwap):,}원 ({dev:+.1f}%)\n"
                f"   → 눌림 매수 검토 구간"
            )
            self.last_alert_type = "DIP"

        # ② VWAP 회복 (이전 하향이탈 후 복귀) → 진입 시점
        elif dev >= VWAP_RECOVER_PCT and self.below_vwap:
            self.below_vwap = False
            alert = (
                f"↩️ {self.name} VWAP 회복\n"
                f"   현재 {self.current_price:,}원 vs VWAP {int(self.vwap):,}원 ({dev:+.1f}%)\n"
                f"   → 반등 확인 — 진입 검토"
            )
            self.last_alert_type = "RECOVER"

        # ③ VWAP 상향 과열
        elif dev >= VWAP_HOT_PCT:
            alert = (
                f"⬆️ {self.name} VWAP 과열\n"
                f"   현재 {self.current_price:,}원 vs VWAP {int(self.vwap):,}원 ({dev:+.1f}%)\n"
                f"   → 추격매수 주의"
            )
            self.last_alert_type = "HOT"

        if alert:
            self.last_alert_time = now
        return alert


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VWAPMonitor — 메인 모니터 엔진
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VWAPMonitor:
    def __init__(self, targets: list[dict], simulate: bool = False):
        self.simulate = simulate
        self.trackers = {
            t["ticker"]: StockTracker(t["ticker"], t["name"], t["qty"], t.get("desc", ""))
            for t in targets
        }
        self.state_path = Path("data/vwap_monitor.json")
        self._consecutive_failures = 0  # API 연속 실패 카운터

        if not simulate:
            try:
                from src.adapters.kis_order_adapter import KisOrderAdapter
                self.adapter = KisOrderAdapter()
                logger.info("[INIT] KIS API 어댑터 초기화 성공")
            except Exception as e:
                logger.error("[INIT] KIS API 어댑터 초기화 실패: %s", e)
                logger.error("[INIT] .env 파일/KIS 키 확인 필요 — 모니터 종료")
                raise
        else:
            self.adapter = None

    # ── 데이터 수집 ──

    def poll_all(self) -> bool:
        """전 종목 현재가 스냅샷. 성공 시 True, 전체 실패 시 False."""
        success_count = 0
        fail_count = 0

        for ticker, tracker in self.trackers.items():
            for attempt in range(1, API_RETRY_MAX + 1):
                try:
                    if self.simulate:
                        info = self._simulate_price(ticker, tracker)
                    else:
                        info = self.adapter.fetch_current_price(ticker)

                    if not info or info.get("current_price", 0) <= 0:
                        logger.warning("[POLL] %s: 가격 0 또는 빈 응답 (시도 %d/%d)",
                                       tracker.name, attempt, API_RETRY_MAX)
                        if attempt < API_RETRY_MAX:
                            time.sleep(API_RETRY_DELAY)
                            continue
                        fail_count += 1
                        break

                    snap = tracker.update(info)
                    if snap:
                        logger.debug("[POLL] %s: %s원 (vol=%s)",
                                     tracker.name, snap["price"], snap["cum_vol"])
                        success_count += 1
                    break  # 성공 → 다음 종목

                except Exception as e:
                    logger.error("[POLL] %s 오류 (시도 %d/%d): %s",
                                 ticker, attempt, API_RETRY_MAX, e)
                    if attempt < API_RETRY_MAX:
                        time.sleep(API_RETRY_DELAY)
                    else:
                        fail_count += 1

            time.sleep(0.5)  # API 속도 제한

        # 연속 실패 카운터 관리
        if success_count == 0 and fail_count > 0:
            self._consecutive_failures += 1
            logger.warning("[POLL] 전 종목 실패 (%d회 연속) — API 장애 또는 서킷브레이커 의심",
                           self._consecutive_failures)
            if self._consecutive_failures >= 3:
                logger.warning("[POLL] 3회 연속 전체 실패 — %d초 대기 후 재시도",
                               API_CIRCUIT_BREAKER_WAIT)
                self.send(
                    "⚠️ [VWAP 모니터] API 3회 연속 실패\n"
                    f"서킷브레이커 또는 API 장애 의심\n"
                    f"{API_CIRCUIT_BREAKER_WAIT}초 대기 후 재시도합니다."
                )
                time.sleep(API_CIRCUIT_BREAKER_WAIT)
            return False
        else:
            self._consecutive_failures = 0
            return True

    def _simulate_price(self, ticker: str, tracker: StockTracker) -> dict:
        """시뮬레이션 모드: 가짜 가격 생성."""
        import random

        # 초기 기준가 (ticker별)
        base_prices = {
            "034020": 98000, "000500": 100300, "011690": 6990,
            "028050": 33450, "064350": 233000,
        }
        base = base_prices.get(ticker, 50000)

        # 이전 가격 기반 랜덤 워크
        if tracker.snapshots:
            prev = tracker.snapshots[-1]["price"]
            change = random.gauss(0, base * 0.003)  # ±0.3% 변동
            price = max(int(prev + change), int(base * 0.95))
        else:
            # 시가: 기준가 ± 2% 갭
            gap = random.uniform(-0.02, 0.02)
            price = int(base * (1 + gap))

        # 누적 거래량: 시간에 따라 증가
        elapsed_min = len(tracker.snapshots) * 3
        vol = int(100000 * (1 + elapsed_min / 30) * random.uniform(0.8, 1.2))

        return {
            "current_price": price,
            "volume": vol,
            "high": max(price, tracker.snapshots[-1]["high"] if tracker.snapshots else price),
            "low": min(price, tracker.snapshots[-1]["low"] if tracker.snapshots else price),
            "open": tracker.opening_price or price,
            "change_pct": round((price - base) / base * 100, 2),
        }

    # ── 메시지 포맷 ──

    def format_opening(self) -> str:
        """09:01 개장 갭 리포트."""
        lines = [
            "📊 [09:00 개장 갭 분석]",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        for t in self.trackers.values():
            if t.current_price <= 0:
                continue
            pct = t.prev_close_pct
            icon = "🔴" if pct < -1 else ("🟢" if pct > 1 else "⚪")
            lines.append(
                f"{icon} {t.name}: {t.current_price:,}원 ({pct:+.1f}%)"
            )
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("📌 09:30까지 VWAP 기준선 축적 중...")
        return "\n".join(lines)

    def format_vwap_baseline(self) -> str:
        """09:30 VWAP 기준선 확정."""
        lines = [
            "📊 [09:30 VWAP 기준선 확정]",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        for t in self.trackers.values():
            if t.vwap <= 0:
                continue
            dev = t.deviation_pct
            icon = "📈" if dev >= 0 else "📉"
            lines.append(
                f"{icon} {t.name}\n"
                f"   현재 {t.current_price:,}원 | VWAP {int(t.vwap):,}원 ({dev:+.1f}%)\n"
                f"   시가 {t.opening_price:,}원 | 전일비 {t.prev_close_pct:+.1f}%"
            )
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append(
            f"📌 모니터링 시작\n"
            f"   눌림 알림: VWAP {VWAP_DIP_PCT}% 이하\n"
            f"   과열 알림: VWAP +{VWAP_HOT_PCT}% 이상"
        )
        return "\n".join(lines)

    def format_summary(self) -> str:
        """30분 현황 요약."""
        now = datetime.now().strftime("%H:%M")
        lines = [
            f"📊 [{now} VWAP 현황]",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        dip_stocks = []
        for t in self.trackers.values():
            if t.vwap <= 0:
                continue
            dev = t.deviation_pct
            if dev < -1:
                status = "⬇️"
            elif dev > 1:
                status = "⬆️"
            else:
                status = "🟰"
            extra = f" [눌림#{t.dip_count}]" if t.dip_count > 0 else ""
            lines.append(
                f"{status} {t.name}: {t.current_price:,}원 "
                f"(VWAP {dev:+.1f}%){extra}"
            )
            if t.below_vwap:
                dip_stocks.append(t.name)

        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        if dip_stocks:
            lines.append(f"💡 현재 눌림 구간: {', '.join(dip_stocks)}")
        else:
            lines.append("💡 전 종목 VWAP 근처 안정")
        return "\n".join(lines)

    def format_final(self) -> str:
        """14:00 최종 요약."""
        lines = [
            "📊 [14:00 VWAP 모니터 종료]",
            "━━━━━━━━━━━━━━━━━━━━━",
        ]
        for t in self.trackers.values():
            if t.vwap <= 0:
                continue
            dev = t.deviation_pct
            day_range = ""
            if t.snapshots:
                prices = [s["price"] for s in t.snapshots if s["price"] > 0]
                if prices:
                    day_range = f" (장중 {min(prices):,}~{max(prices):,})"
            lines.append(
                f"{'📈' if dev >= 0 else '📉'} {t.name}: {t.current_price:,}원 "
                f"(VWAP {dev:+.1f}%) 눌림{t.dip_count}회{day_range}"
            )
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        total_dips = sum(t.dip_count for t in self.trackers.values())
        lines.append(f"📌 총 눌림 이벤트: {total_dips}회")
        return "\n".join(lines)

    # ── 텔레그램 ──

    def send(self, msg: str):
        """텔레그램 전송 (시뮬레이션 시 콘솔 출력)."""
        if self.simulate:
            print(f"\n{'='*40}")
            print(msg)
            print(f"{'='*40}\n")
        else:
            from src.telegram_sender import send_message
            send_message(msg)

    # ── 11:30 AI 분석 ──

    def run_ai_analysis(self):
        """11:30 TradeAdvisor AI 분석 + VWAP 컨텍스트."""
        from src.agents.trade_advisor import TradeAdvisor

        advisor = TradeAdvisor()
        results = []

        for ticker, tracker in self.trackers.items():
            if tracker.current_price <= 0:
                continue

            logger.info("[AI] %s 분석 중...", tracker.name)
            try:
                result = asyncio.run(
                    advisor.analyze_buy(ticker, tracker.qty, tracker.current_price)
                )
            except Exception as e:
                logger.error("[AI] %s 실패: %s", tracker.name, e)
                result = {"verdict": "ERROR", "error": str(e)}

            # VWAP 컨텍스트 추가
            result["vwap"] = int(tracker.vwap)
            result["vwap_dev"] = f"{tracker.deviation_pct:+.1f}%"
            result["dip_count"] = tracker.dip_count
            result["below_vwap"] = tracker.below_vwap

            results.append({
                "ticker": ticker,
                "name": tracker.name,
                "qty": tracker.qty,
                "current_price": tracker.current_price,
                "result": result,
            })

        # 메시지 포맷
        msg = self._format_ai_msg(results)
        self.send(msg)

        # JSON 저장 (기존 midday_analysis.json 호환)
        self._save_ai_results(results)

        return results

    def _format_ai_msg(self, results: list[dict]) -> str:
        lines = [
            "🔍 [11:30 AI + VWAP 통합 분석]",
            "━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]
        for r in results:
            ai = r["result"]
            verdict = ai.get("verdict", "ERROR")
            if verdict == "BUY_OK":
                icon = "✅"
            elif verdict == "WAIT":
                icon = "⏳"
            elif verdict == "SKIP":
                icon = "❌"
            else:
                icon = "⚠️"

            vwap_str = f"VWAP {ai.get('vwap', 0):,}원 ({ai.get('vwap_dev', '')})"
            dip_str = f" | 눌림#{ai.get('dip_count', 0)}" if ai.get("dip_count", 0) > 0 else ""
            below_str = " ⬇️현재눌림중" if ai.get("below_vwap") else ""

            lines.append(f"{icon} {r['name']}({r['ticker']})")
            lines.append(f"   현재 {r['current_price']:,}원 | {vwap_str}{dip_str}{below_str}")

            tech = ai.get("technical_summary", "")
            if tech:
                lines.append(f"   📊 {tech}")
            catalyst = ai.get("catalyst", "")
            if catalyst:
                lines.append(f"   📰 {catalyst}")
            risk = ai.get("risk_warning", "")
            if risk:
                lines.append(f"   ⚠️ {risk}")
            suggestion = ai.get("suggestion", "")
            if suggestion:
                lines.append(f"   💡 {suggestion}")

            conf = ai.get("confidence", 0)
            lines.append(f"   🤖 AI: {verdict} (신뢰도 {conf}%)")
            lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        buy_ok = [r["name"] for r in results if r["result"].get("verdict") == "BUY_OK"]
        dip_now = [r["name"] for r in results if r["result"].get("below_vwap")]
        if buy_ok:
            lines.append(f"✅ 매수 OK: {', '.join(buy_ok)}")
        if dip_now:
            lines.append(f"⬇️ 현재 눌림: {', '.join(dip_now)}")
        if not buy_ok and not dip_now:
            lines.append("💡 현재 즉시 매수 시그널 없음 — 눌림 대기")
        lines.append("")
        lines.append("📌 매수 시 텔레그램 명령:")
        for r in results:
            lines.append(f"   매수 {r['name']} {r['qty']}")

        return "\n".join(lines)

    def _save_ai_results(self, results: list[dict]):
        output = {
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "targets": results,
        }
        path = Path("data/midday_analysis.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        logger.info("[AI] 결과 저장: %s", path)

    # ── 상태 저장 ──

    def save_state(self):
        state = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stocks": {},
        }
        for ticker, t in self.trackers.items():
            prices = [s["price"] for s in t.snapshots if s["price"] > 0]
            state["stocks"][ticker] = {
                "name": t.name,
                "current_price": t.current_price,
                "vwap": int(t.vwap) if t.vwap else 0,
                "vwap_dev_pct": round(t.deviation_pct, 2),
                "opening_price": t.opening_price,
                "dip_count": t.dip_count,
                "below_vwap": t.below_vwap,
                "snapshots": len(t.snapshots),
                "day_high": max(prices) if prices else 0,
                "day_low": min(prices) if prices else 0,
            }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ── 메인 루프 ──

    def run(self):
        logger.info("=" * 50)
        logger.info("[VWAP Monitor] 시작 — %d종목 %s",
                     len(self.trackers),
                     "(시뮬레이션)" if self.simulate else "(LIVE)")
        logger.info("  대상: %s", ", ".join(
            f"{t.name}({t.ticker})" for t in self.trackers.values()
        ))
        logger.info("=" * 50)

        try:
            self._run_phases()
        except KeyboardInterrupt:
            logger.info("[VWAP Monitor] 사용자 중단 (Ctrl+C)")
        except Exception as e:
            logger.critical("[VWAP Monitor] 치명적 오류로 종료: %s", e, exc_info=True)
            try:
                self.send(
                    f"🚨 [VWAP 모니터 크래시]\n"
                    f"오류: {e}\n"
                    f"시각: {datetime.now().strftime('%H:%M:%S')}\n"
                    f"수집된 스냅샷: {self._snapshot_count()}개"
                )
            except Exception:
                pass
        finally:
            # 어떤 상황이든 마지막 상태 저장
            try:
                self.save_state()
                logger.info("[VWAP Monitor] 최종 상태 저장 완료")
            except Exception as e2:
                logger.error("[VWAP Monitor] 최종 상태 저장 실패: %s", e2)
            logger.info("[VWAP Monitor] 종료")

    def _snapshot_count(self) -> int:
        """전체 스냅샷 수."""
        return sum(len(t.snapshots) for t in self.trackers.values())

    def _run_phases(self):
        """Phase 0~5 순차 실행. 예외 발생 시 상위 run()이 처리."""

        # ━━ Phase 0: 장 시작 대기 ━━
        if not self.simulate:
            self._wait_until(dt_time(9, 0))
            time.sleep(30)  # 09:00:30 — 시가 확정 대기

        # ━━ Phase 1: 개장 갭 분석 (09:01) ━━
        logger.info("[Phase 1] 개장 관찰")
        self.poll_all()
        self.send(self.format_opening())

        # ━━ Phase 2: VWAP 축적 (09:00~09:30) ━━
        if self.simulate:
            for i in range(10):
                time.sleep(1)
                self.poll_all()
                logger.info("[Opening] 스냅샷 #%d", i + 2)
        else:
            target_0930 = datetime.now().replace(hour=9, minute=30, second=0)
            while datetime.now() < target_0930:
                time.sleep(POLL_SEC_OPENING)
                self.poll_all()
                n = len(list(self.trackers.values())[0].snapshots)
                logger.info("[Opening] 스냅샷 #%d", n)

        # ━━ Phase 3: VWAP 기준선 확정 (09:30) ━━
        logger.info("[Phase 2] VWAP 기준선 설정")
        self.send(self.format_vwap_baseline())
        self.save_state()

        # ━━ Phase 4: 모니터링 (09:30~14:00) ━━
        logger.info("[Phase 3] VWAP 모니터링 시작")
        last_summary = datetime.now()
        ai_done = False

        if self.simulate:
            end_iter = 20
            for i in range(end_iter):
                time.sleep(1)
                self.poll_all()
                self._check_and_send_alerts()
                if (i + 1) % 5 == 0:
                    self.send(self.format_summary())
                if i == 14 and not ai_done:
                    logger.info("[AI] 시뮬 AI 분석")
                    self.send("🔍 [11:30 AI 분석] (시뮬레이션 — AI 스킵)")
                    ai_done = True
                self.save_state()
        else:
            end_time = datetime.now().replace(hour=14, minute=0, second=0)
            while datetime.now() < end_time:
                time.sleep(POLL_SEC_NORMAL)

                try:
                    self.poll_all()
                except Exception as e:
                    logger.error("[Phase 4] poll_all 예외 (무시하고 계속): %s", e)

                try:
                    self._check_and_send_alerts()
                except Exception as e:
                    logger.error("[Phase 4] 알림 체크 예외: %s", e)

                # 30분 요약
                elapsed = (datetime.now() - last_summary).total_seconds()
                if elapsed >= SUMMARY_INTERVAL_SEC:
                    try:
                        self.send(self.format_summary())
                    except Exception as e:
                        logger.error("[Phase 4] 요약 전송 실패: %s", e)
                    last_summary = datetime.now()

                # 11:30 AI 분석
                now = datetime.now()
                if not ai_done and now.hour == 11 and now.minute >= 28:
                    logger.info("[AI] 11:30 TradeAdvisor 분석 시작")
                    try:
                        self.run_ai_analysis()
                    except Exception as e:
                        logger.error("[AI] 분석 실패: %s", e)
                        self.send(f"⚠️ [AI 분석 실패] {e}")
                    ai_done = True

                try:
                    self.save_state()
                except Exception as e:
                    logger.error("[Phase 4] 상태 저장 실패: %s", e)

        # ━━ Phase 5: 종료 (14:00) ━━
        self.send(self.format_final())
        self.save_state()

    def _check_and_send_alerts(self):
        """전 종목 VWAP 알림 체크 → 발생 시 즉시 텔레그램."""
        alerts = []
        for t in self.trackers.values():
            alert = t.check_alert()
            if alert:
                alerts.append(alert)
                logger.info("[ALERT] %s", alert.split("\n")[0])

        if alerts:
            msg = "🔔 [VWAP 알림]\n━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n\n".join(alerts)
            self.send(msg)

    def _wait_until(self, target: dt_time):
        """지정 시각까지 대기."""
        logger.info("[대기] %s까지 대기 중...", target.strftime("%H:%M"))
        while True:
            now = datetime.now()
            target_dt = now.replace(
                hour=target.hour, minute=target.minute, second=target.second
            )
            if now >= target_dt:
                break
            remaining = (target_dt - now).total_seconds()
            logger.debug("[대기] 남은 시간: %.0f초", remaining)
            time.sleep(min(30, remaining))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="장중 VWAP 모니터")
    parser.add_argument("--simulate", action="store_true",
                        help="시뮬레이션 모드 (가짜 데이터, 빠른 실행)")
    parser.add_argument("--targets", nargs="*",
                        help="종목 직접 지정 (예: 005930:10 000660:5)")
    args = parser.parse_args()

    # 대상 종목
    if args.targets:
        from src.stock_name_resolver import ticker_to_name
        targets = []
        for t in args.targets:
            parts = t.split(":")
            ticker = parts[0]
            qty = int(parts[1]) if len(parts) > 1 else 10
            name = ticker_to_name(ticker) or ticker
            targets.append({"ticker": ticker, "name": name, "qty": qty})
    else:
        targets = DEFAULT_TARGETS

    monitor = VWAPMonitor(targets, simulate=args.simulate)
    monitor.run()


if __name__ == "__main__":
    main()
