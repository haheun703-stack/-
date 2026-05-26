"""run_adaptive_cycle.py — 적응형 포지션 매매법 통합 사이클 (MVP-1 + 2 + 3 + 4).

매 30분 cron 1회 호출로 4단계 흐름 전체 처리:
  1. MVP-1: 천장 -3% 감지 → (옵션) 자동 매도 → MVP-2 큐 자동 등록
  2. MVP-2: 분할매수 큐 가격 도달 확인 → (옵션) 자동 매수
  3. MVP-3: 받침 패턴 감지 (소부장 후보 풀)
  4. MVP-4: 3중 검증 후 재진입 평가 → (옵션) 자동 매수

cron 등록 (VPS):
  # 적응형 매매법 통합 사이클 — 매 30분 (장중 09:00~15:30)
  */30 9-15 * * 1-5 cd /home/ubuntu/quantum-master && PYTHONPATH=. ./venv/bin/python3.11 -u -X utf8 scripts/run_adaptive_cycle.py --paper >> logs/adaptive_cycle.log 2>&1

  # 소부장 후보 풀 갱신 — 매일 18:00
  0 18 * * 1-5 cd /home/ubuntu/quantum-master && PYTHONPATH=. ./venv/bin/python3.11 -u -X utf8 scripts/step5_soubujang_pool.py >> logs/soubujang_pool.log 2>&1

  # 일요일 STEP 5 종합 (퐝가님 일요일 점검용)
  0 17 * * 0 cd /home/ubuntu/quantum-master && PYTHONPATH=. ./venv/bin/python3.11 -u -X utf8 scripts/step5_soubujang_pool.py >> logs/soubujang_pool.log 2>&1

1주차 안전 (5/26 ~ 5/31):
  --paper 모드 + ADAPTIVE_AUTO_SELL=0 + ADAPTIVE_AUTO_BUY=0 + ADAPTIVE_AUTO_REENTRY=0
  → 모든 시그널 텔레그램 알림만, 실제 KIS 주문 0건

사용:
  python scripts/run_adaptive_cycle.py --paper           # 시뮬레이션
  python scripts/run_adaptive_cycle.py --real            # 실전 (5/27 후 검토)
  python scripts/run_adaptive_cycle.py --paper --dry-run # KIS 호출 없이 mock
  python scripts/run_adaptive_cycle.py --skip-mvp4       # MVP-4 스킵
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# === 후보 풀 로드 ===
def load_latest_soubujang_pool() -> dict:
    """data/soubujang_pool/ 최신 파일 로드."""
    pool_dir = PROJECT_ROOT / "data" / "soubujang_pool"
    if not pool_dir.exists():
        return {}

    candidates = sorted(pool_dir.glob("soubujang_pool_*.json"), reverse=True)
    if not candidates:
        return {}

    try:
        with candidates[0].open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("pool 로드 실패: %s", e)
        return {}


def build_step5_lookup(pool: dict) -> dict:
    """MVP-4용 STEP 5 룩업 dict {ticker: {stars, upside}}."""
    lookup: dict[str, dict] = {}
    for ticker, info in pool.items():
        s5 = info.get("step5", {})
        upside = s5.get("upside_ratio") or 0
        try:
            upside_f = float(upside)
        except (ValueError, TypeError):
            upside_f = 0.0

        # 등급 → 별 개수 (★★★=3, ★★★★=4, ★★★★★=5)
        if upside_f >= 5.0:
            stars = 5
        elif upside_f >= 3.0:
            stars = 4
        elif upside_f >= 2.0:
            stars = 3
        elif upside_f >= 1.5:
            stars = 2
        else:
            stars = 1

        lookup[ticker] = {
            "name": info.get("name", ""),
            "stars": stars,
            "upside": upside_f,
            "passed": bool(info.get("passed", False)),
        }
    return lookup


def passed_candidates(pool: dict) -> list[tuple[str, str]]:
    """soubujang_pool 통과 종목만 [(ticker, name), ...]."""
    return [
        (ticker, info.get("name", ""))
        for ticker, info in pool.items()
        if info.get("passed")
    ]


# === 자비스 9 안전선 wrapper ===
def make_jarvis_safety_check(is_real: bool):
    """자비스 9 안전선 통합 체크 함수 생성.

    실시간 점수/매크로 가드/시간대 등을 종합. 모듈이 없으면 기본 통과.
    """
    try:
        # 기존 매크로 가드 + 실시간 점수
        from src.use_cases.market_regime_guard import is_market_safe_for_buy
    except ImportError:
        is_market_safe_for_buy = None  # type: ignore[assignment]

    def check(ticker: str) -> dict:
        failed: list[str] = []

        # 1) 매크로 가드 (BEARISH 차단)
        if is_market_safe_for_buy is not None:
            try:
                safe, reason = is_market_safe_for_buy()
                if not safe:
                    failed.append(f"매크로 가드: {reason}")
            except Exception as e:
                failed.append(f"매크로 가드 오류: {e}")

        # 2) 시간대 (장중 09:00~15:00만 신규 진입)
        now = dt.datetime.now().time()
        if now < dt.time(9, 0) or now > dt.time(15, 0):
            failed.append(f"시간대 외: {now:%H:%M}")

        # 3) 실전 모드 + KILL_SWITCH 발동 여부 (보조)
        if is_real and (PROJECT_ROOT / "data" / "kill_switch.flag").exists():
            failed.append("KILL_SWITCH 발동")

        return {"pass": len(failed) == 0, "failed": failed}

    return check


# === 텔레그램 알림 ===
def send_telegram(message: str) -> bool:
    """텔레그램 알림 (BOT_TOKEN/CHAT_ID 필요)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram 설정 누락 — 알림 미발송")
        return False
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        return r.status_code == 200
    except Exception as e:
        logger.warning("Telegram 발송 실패: %s", e)
        return False


# === P0-2 (5/25): REAL 모드 3중 안전 게이트 ===
def validate_real_mode_gate() -> tuple[bool, str]:
    """REAL 모드 가동 전 3중 안전 게이트 검증.

    게이트:
      1) AUTO_TRADING_ENABLED=1 환경변수 명시
      2) KIS_ACC_NO 형식 검증 (- 제거 후 최소 8자 숫자)

    (CLI 인자 --real은 호출 측에서 이미 검증)

    Returns:
        (passed: bool, fail_reason: str)
    """
    auto_trading = os.getenv("AUTO_TRADING_ENABLED", "0")
    acc_no = os.getenv("KIS_ACC_NO", "")
    acc_digits = acc_no.replace("-", "")

    if auto_trading != "1":
        return False, f"AUTO_TRADING_ENABLED={auto_trading!r} (기대 '1')"
    if not acc_no or len(acc_digits) < 8 or not acc_digits.isdigit():
        return False, f"KIS_ACC_NO 형식 이상 ({acc_no!r})"
    return True, ""


# === 메인 사이클 ===
def run_cycle(is_paper: bool, skip: set[str], dry_run: bool) -> dict:
    """4단계 통합 사이클 실행."""
    summary: dict = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "PAPER" if is_paper else "REAL",
        "dry_run": dry_run,
        "mvp1": {"executed": False, "triggers": 0, "errors": []},
        "mvp2": {"executed": False, "triggers": 0, "errors": []},
        "mvp2_5": {"executed": False, "triggers": 0, "errors": []},
        "mvp2_6": {"executed": False, "triggers": 0, "errors": []},  # 자동 손절 -5% (5/25)
        "mvp2_7": {"executed": False, "triggers": 0, "errors": []},  # H8/H9 시간 매도 (5/26)
        "mvp5": {"executed": False, "triggers": 0, "errors": []},     # AI 밸류체인 동조 (5/26)
        "mvp6": {"executed": False, "triggers": 0, "errors": []},     # 모멘텀 추격 매수 (5/26 20:30)
        "mvp3": {"executed": False, "triggers": 0, "errors": []},
        "mvp4": {"executed": False, "triggers": 0, "errors": []},
    }

    # broker 초기화
    if dry_run:
        from unittest.mock import MagicMock
        broker = MagicMock(name="MockBroker")
        logger.info("DRY_RUN 모드 — MockBroker 사용")
    else:
        # P0-2 (5/25): REAL 모드 3중 안전 게이트
        if not is_paper:
            passed, gate_fail = validate_real_mode_gate()
            if not passed:
                warn = (
                    f"🛑 P0-2 REAL 모드 차단 → PAPER fallback\n"
                    f"  사유: {gate_fail}\n"
                    f"  시각: {dt.datetime.now():%Y-%m-%d %H:%M:%S}"
                )
                logger.error(warn)
                send_telegram(warn)
                is_paper = True
                summary["mode"] = "PAPER_FALLBACK"
                summary["paper_fallback_reason"] = gate_fail
            else:
                acc_no = os.getenv("KIS_ACC_NO", "")
                masked = acc_no[:4] + "***" + acc_no[-2:] if len(acc_no) >= 6 else "***"
                logger.warning(
                    "🔴 REAL 모드 가동 (3중 게이트 통과): "
                    "AUTO_TRADING_ENABLED=1, ACC_NO=%s",
                    masked,
                )

        # 5/26 fix: mojito raw 객체에 buy_limit 없음 → KisOrderAdapter로 교체.
        # adapter는 mojito를 wrapping + buy_limit/sell_limit/가드레일/텔레그램알림 제공.
        # adapter는 환경변수 MODEL=REAL/PAPER 자동 인식 (paper 모드 시 mock=True).
        try:
            if is_paper:
                from src.adapters.paper_order_adapter import PaperOrderAdapter
                broker = PaperOrderAdapter()
                logger.info("PAPER 모드 — PaperOrderAdapter 사용 (mojito 미사용)")
            else:
                from src.adapters.kis_order_adapter import KisOrderAdapter
                broker = KisOrderAdapter()
                logger.warning("🔴 REAL 모드 — KisOrderAdapter 활성 (buy_limit/가드레일 포함)")
        except Exception as e:
            logger.error("broker 초기화 실패: %s", e)
            summary["error"] = f"broker init: {e}"
            return summary

    # 후보 풀 로드
    pool = load_latest_soubujang_pool()
    step5_lookup = build_step5_lookup(pool)
    candidates = passed_candidates(pool)
    logger.info("후보 풀: %d 종목 (통과 %d)", len(pool), len(candidates))

    # === 5/25 학습 모드: 신호 snapshot DB 초기화 + 후보 일괄 저장 ===
    learning_mode = os.getenv("ADAPTIVE_DAILY_LEARNING_MODE", "0") == "1"
    if learning_mode:
        try:
            from src.use_cases.signal_snapshot import init_db, snapshot_signals
            from src.use_cases.adaptive_buy_queue import _load_queues_raw

            init_db()
            queue_tickers = set((_load_queues_raw().get("queues") or {}).keys())
            snap_candidates = [
                {
                    "ticker": t,
                    "name": n,
                    "in_queue": int(t in queue_tickers),
                    "in_holdings": 0,  # broker.fetch_balance 호출 비용 절약
                }
                for t, n in candidates
            ]
            saved = snapshot_signals(snap_candidates)
            logger.info("학습 모드 snapshot: %d종목 저장 (큐: %d)", saved, len(queue_tickers))
        except Exception as e:
            logger.warning("snapshot 저장 실패: %s", e)

    # 자비스 안전선
    jarvis_check = make_jarvis_safety_check(is_real=not is_paper)

    # === MVP-1: 천장 감지 + 매도 ===
    if "mvp1" not in skip and candidates:
        from src.use_cases.adaptive_position_manager import (
            detect_peak_signal,
            execute_auto_sell,
            format_peak_signal_for_telegram,
        )

        summary["mvp1"]["executed"] = True
        for ticker, name in candidates:
            try:
                sig = detect_peak_signal(broker, ticker)
                if sig.trigger:
                    summary["mvp1"]["triggers"] += 1
                    msg = format_peak_signal_for_telegram(sig, name)
                    print(msg)
                    send_telegram(msg)

                    # 학습 로그: MVP-1 천장 trigger
                    if learning_mode:
                        try:
                            from src.use_cases.decision_logger import log_decision
                            log_decision(
                                "SELL" if sig.auto_sell_eligible else "ALERT",
                                ticker, name=name,
                                current_price=sig.current_price,
                                target_price=sig.peak_price,
                                peak_drop_pct=sig.pct_from_peak,
                                pass_reasons=sig.reasons_pass,
                                fail_reasons=sig.reasons_fail,
                                extra={"mvp": "1", "days_since_peak": sig.days_since_peak},
                            )
                        except Exception as _e:
                            logger.warning("MVP-1 log_decision 실패: %s", _e)

                    # AUTO_SELL=1 시 매도 실행 + MVP-2 큐 자동 등록
                    if sig.auto_sell_eligible:
                        # 보유 수량 가져오기 (단순 구현)
                        try:
                            holdings = broker.fetch_balance() if hasattr(broker, "fetch_balance") else {}
                            qty = 0
                            for h in (holdings.get("output1", []) if isinstance(holdings, dict) else []):
                                if h.get("pdno") == ticker:
                                    qty = int(h.get("hldg_qty", 0))
                                    break
                            if qty > 0:
                                res = execute_auto_sell(broker, sig, qty)
                                send_telegram(f"✅ 매도 결과: {res}")
                        except Exception as e:
                            summary["mvp1"]["errors"].append(f"{ticker} 매도: {e}")
            except Exception as e:
                summary["mvp1"]["errors"].append(f"{ticker}: {e}")

    # === MVP-2: 분할매수 큐 트리거 ===
    if "mvp2" not in skip:
        from src.use_cases.adaptive_buy_queue import (
            check_and_trigger_queues,
            format_trigger_for_telegram,
        )

        summary["mvp2"]["executed"] = True
        try:
            triggers = check_and_trigger_queues(broker)
            summary["mvp2"]["triggers"] = len(triggers)
            for t in triggers:
                msg = format_trigger_for_telegram(t)
                print(msg)
                send_telegram(msg)

                # 학습 로그: MVP-2 큐 trigger (BUY or TRIGGERED or EXPIRED)
                if learning_mode:
                    try:
                        from src.use_cases.decision_logger import log_decision
                        status = t.get("status", t.get("event", "?"))
                        log_decision(
                            "BUY" if status == "FILLED" else "QUEUE_TRIGGER",
                            t.get("ticker", "?"), name=t.get("name", ""),
                            current_price=t.get("current_price", 0),
                            qty=t.get("qty", 0),
                            amount=t.get("alloc_amount", 0),
                            target_price=t.get("target_price", 0),
                            extra={
                                "mvp": "2",
                                "level": t.get("level"),
                                "status": status,
                                "order_id": t.get("order_id"),
                                "peak_price": t.get("peak_price", 0),
                            },
                        )
                    except Exception as _e:
                        logger.warning("MVP-2 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp2"]["errors"].append(str(e))

    # === MVP-2.5: 빠른 익절 (+7%) — FILLED stage 모니터링 ===
    if "mvp2_5" not in skip:
        from src.use_cases.adaptive_quick_profit import (
            check_quick_profit_triggers,
            format_quick_profit_for_telegram,
        )

        summary["mvp2_5"]["executed"] = True
        try:
            triggers = check_quick_profit_triggers(broker)
            summary["mvp2_5"]["triggers"] = len(triggers)
            for t in triggers:
                msg = format_quick_profit_for_telegram(t)
                print(msg)
                send_telegram(msg)

                # 학습 로그: MVP-2.5 Trailing Quick Profit (ARMED or SOLD)
                if learning_mode:
                    try:
                        from src.use_cases.decision_logger import log_decision
                        status = t.get("status", "?")
                        log_decision(
                            "SELL" if status == "QUICK_SOLD" else "ALERT",
                            t.get("ticker", "?"), name=t.get("name", ""),
                            current_price=t.get("current_price", 0),
                            trailing_drop_pct=t.get("trailing_drop_pct", 0.0),
                            extra={
                                "mvp": "2_5",
                                "status": status,
                                "trailing_peak": t.get("trailing_peak", 0),
                                "actual_buy": t.get("actual_buy", 0),
                                "profit_pct": t.get("profit_pct", 0.0),
                            },
                        )
                    except Exception as _e:
                        logger.warning("MVP-2.5 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp2_5"]["errors"].append(str(e))

    # === MVP-2.6: 자동 손절 -5% — FILLED stage 모니터링 (5/25 백테스트 R2) ===
    if "mvp2_6" not in skip:
        from src.use_cases.adaptive_stop_loss import (
            check_stop_loss_triggers,
            format_stop_loss_for_telegram,
        )

        summary["mvp2_6"]["executed"] = True
        try:
            triggers = check_stop_loss_triggers(broker)
            summary["mvp2_6"]["triggers"] = len(triggers)
            for t in triggers:
                msg = format_stop_loss_for_telegram(t)
                print(msg)
                send_telegram(msg)

                # 학습 로그: MVP-2.6 손절 매도
                if learning_mode:
                    try:
                        from src.use_cases.decision_logger import log_decision
                        log_decision(
                            "SELL", t.get("ticker", "?"), name=t.get("name", ""),
                            current_price=t.get("current_price", 0),
                            qty=t.get("qty", 0),
                            extra={
                                "mvp": "2_6",
                                "type": "STOP_LOSS",
                                "actual_buy": t.get("actual_buy", 0),
                                "loss_pct": t.get("loss_pct", 0.0),
                                "level": t.get("level"),
                                "order_id": t.get("order_id"),
                            },
                        )
                    except Exception as _e:
                        logger.warning("MVP-2.6 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp2_6"]["errors"].append(str(e))

    # === MVP-2.7: 시간 매도 (H8 D+3 익절 + H9 D+5 데드라인) — 5/26 PDCA ===
    # 백테스트 기반 (5/14): D+1 +2.02% / D+3 +4.00% peak / D+5 -0.35% 음수 진입
    # 환경변수 ADAPTIVE_TIME_EXIT_ENABLED=1일 때만 발동.
    if "mvp2_7" not in skip:
        try:
            from src.use_cases.adaptive_time_exit import (
                scan_queue_for_time_exits, execute_time_exit,
            )
            # 큐 상태 로드
            import json
            from pathlib import Path
            queue_path = Path(__file__).resolve().parent.parent / "data" / "adaptive_buy_queue.json"
            queue_state = {}
            if queue_path.exists():
                queue_state = json.loads(queue_path.read_text(encoding="utf-8"))

            time_sigs = scan_queue_for_time_exits(queue_state, broker)
            summary.setdefault("mvp2_7", {"executed": True, "triggers": 0, "errors": []})
            summary["mvp2_7"]["executed"] = True
            summary["mvp2_7"]["triggers"] = len(time_sigs)

            for sig in time_sigs:
                msg = (
                    f"⏰ [시간 매도] {sig.exit_type}\n"
                    f"  {sig.ticker} D+{sig.trade_days_elapsed} {sig.pnl_pct:+.2f}%\n"
                    f"  매수 {sig.entry_price:,} → 현재 {sig.current_price:,}\n"
                    f"  사유: {sig.reason}"
                )
                print(msg)
                send_telegram(msg)
                # 실행
                exec_result = execute_time_exit(broker, sig)
                # 학습 로그
                if learning_mode:
                    try:
                        from src.use_cases.decision_logger import log_decision
                        log_decision(
                            "SELL", sig.ticker, name="",
                            current_price=sig.current_price,
                            qty=sig.qty,
                            extra={
                                "mvp": "2_7",
                                "type": sig.exit_type,
                                "trade_days_elapsed": sig.trade_days_elapsed,
                                "pnl_pct": sig.pnl_pct,
                                "entry_price": sig.entry_price,
                                "exec_success": exec_result.get("success", False),
                            },
                        )
                    except Exception as _e:
                        logger.warning("MVP-2.7 log_decision 실패: %s", _e)
        except Exception as e:
            summary.setdefault("mvp2_7", {"executed": False, "triggers": 0, "errors": []})
            summary["mvp2_7"]["errors"].append(str(e))

    # === MVP-3: 받침 패턴 감지 ===
    if "mvp3" not in skip and candidates:
        from src.use_cases.support_pattern_detector import (
            scan_pool_for_support,
            format_support_signal_for_telegram,
        )

        summary["mvp3"]["executed"] = True
        try:
            tickers_only = [t for t, _ in candidates]
            results = scan_pool_for_support(broker, tickers_only)
            for sig in results:
                if sig.trigger:
                    summary["mvp3"]["triggers"] += 1
                    name = next((n for t, n in candidates if t == sig.ticker), sig.ticker)
                    msg = format_support_signal_for_telegram(sig, name)
                    print(msg)
                    send_telegram(msg)

                    # 학습 로그: MVP-3 받침 패턴 (ALERT — 매수는 MVP-4가)
                    if learning_mode:
                        try:
                            from src.use_cases.decision_logger import log_decision
                            log_decision(
                                "ALERT", sig.ticker, name=name,
                                current_price=getattr(sig, "current_price", 0),
                                volume_ratio=getattr(sig, "volume_ratio", 0.0),
                                bullish_ratio=getattr(sig, "bullish_ratio", 0.0),
                                pass_reasons=sig.reasons_pass,
                                extra={"mvp": "3", "support_type": "받침패턴"},
                            )
                        except Exception as _e:
                            logger.warning("MVP-3 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp3"]["errors"].append(str(e))

    # === MVP-4: 자동 재진입 ===
    if "mvp4" not in skip and candidates:
        from src.use_cases.adaptive_reentry import (
            scan_pool_for_reentry,
            execute_auto_reentry,
            format_reentry_for_telegram,
        )

        summary["mvp4"]["executed"] = True
        try:
            decisions = scan_pool_for_reentry(
                broker, candidates,
                step5_pool=step5_lookup,
                jarvis_safety_check=jarvis_check,
            )
            for dec in decisions:
                if dec.trigger:
                    summary["mvp4"]["triggers"] += 1
                    if dec.auto_reentry_eligible:
                        res = execute_auto_reentry(broker, dec)
                        logger.info("재진입 실행: %s → %s", dec.ticker, res)
                    msg = format_reentry_for_telegram(dec)
                    print(msg)
                    send_telegram(msg)

                    # 학습 로그: MVP-4 재진입 3중 검증 통과 (BUY or ALERT)
                    if learning_mode:
                        try:
                            from src.use_cases.decision_logger import log_decision
                            log_decision(
                                "BUY" if dec.auto_reentry_eligible else "ALERT",
                                dec.ticker, name=getattr(dec, "name", ""),
                                current_price=getattr(dec, "current_price", 0),
                                qty=getattr(dec, "target_qty", 0),
                                target_price=getattr(dec, "target_price", 0),
                                pass_reasons=[
                                    *(["받침 통과"] if dec.support_pass else []),
                                    *(["STEP5 통과"] if dec.step5_pass else []),
                                    *(["자비스 통과"] if dec.jarvis_pass else []),
                                ],
                                fail_reasons=getattr(dec, "jarvis_failed_checks", []),
                                extra={
                                    "mvp": "4",
                                    "step5_stars": getattr(dec, "step5_stars", 0),
                                    "step5_upside": getattr(dec, "step5_upside", 0.0),
                                },
                            )
                        except Exception as _e:
                            logger.warning("MVP-4 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp4"]["errors"].append(str(e))

    # === MVP-5: AI 밸류체인 동조화 검출 (5/26 신규, 서브에이전트 보고 기반) ===
    # 4개 AI 세부 섹터 (검사/PCB/소재/산업소재) 동시 폭등 감지
    # 5/26 ISC+16.9%/인텍플러스+18.9%/코리아써키트+12.4%/두산+10.1%/동진쎄미켐+10.2% 사례
    if "mvp5" not in skip:
        try:
            from src.use_cases.ai_chain_detector import (
                detect_ai_chain_sync,
                format_ai_chain_signal_for_telegram,
            )
            from src.use_cases.ai_chain_auto_watchlist import (
                add_to_ai_chain_watchlist,
                format_added_for_telegram,
            )
            from src.use_cases.adaptive_position_manager import _load_protected_tickers

            summary["mvp5"]["executed"] = True
            ai_sig = detect_ai_chain_sync(broker)
            if ai_sig.triggered:
                summary["mvp5"]["triggers"] = 1
                msg = format_ai_chain_signal_for_telegram(ai_sig)
                print(msg)
                send_telegram(msg)

                # 워치리스트 자동 추가 (보호 종목 + 보유 종목 제외)
                try:
                    protected = _load_protected_tickers()
                    held = set()
                    if hasattr(broker, "fetch_balance"):
                        bal = broker.fetch_balance()
                        for h in bal.get("holdings", []):
                            t = str(h.get("ticker", "")).zfill(6)
                            if t:
                                held.add(t)
                    add_result = add_to_ai_chain_watchlist(
                        ai_sig.surge_stocks,
                        protected_tickers=protected,
                        held_tickers=held,
                    )
                    if add_result.get("added"):
                        wl_msg = format_added_for_telegram(add_result)
                        print(wl_msg)
                        send_telegram(wl_msg)
                except Exception as _e:
                    logger.warning("[MVP-5] 워치리스트 자동 추가 실패: %s", _e)

                # ── 5/27 실매매 진입 핵심: AI 동조 자동 큐 등록 (강세장 적응) ──
                # 환경변수 AI_CHAIN_QUEUE_AUTO_REGISTER=1일 때만 발동
                # peak=현재가 / L1 -3% / L2 -7% / L3 -12% / 만료 3일
                try:
                    from src.use_cases.ai_chain_queue_auto_register import (
                        register_ai_chain_queues,
                        merge_into_queue_state,
                        format_registration_for_telegram,
                    )
                    import json
                    from pathlib import Path
                    queue_path = Path(__file__).resolve().parent.parent / "data" / "adaptive_buy_queue.json"
                    queue_state = {}
                    if queue_path.exists():
                        queue_state = json.loads(queue_path.read_text(encoding="utf-8"))
                    reg_result = register_ai_chain_queues(
                        ai_sig.surge_stocks,
                        protected_tickers=protected,
                        held_tickers=held,
                        queue_state=queue_state,
                    )
                    if reg_result.registered:
                        merge_into_queue_state(queue_state, reg_result.registered)
                        tmp = queue_path.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps(queue_state, ensure_ascii=False, indent=2), encoding="utf-8")
                        tmp.replace(queue_path)
                        q_msg = format_registration_for_telegram(reg_result)
                        print(q_msg)
                        send_telegram(q_msg)
                        logger.warning(
                            "[MVP-5] AI 동조 큐 자동 등록: %d종목", len(reg_result.registered),
                        )
                except Exception as _e:
                    logger.warning("[MVP-5] AI 동조 큐 자동 등록 실패: %s", _e)

                # 학습 로그
                if learning_mode:
                    try:
                        from src.use_cases.decision_logger import log_decision
                        log_decision(
                            "ALERT", "AI_CHAIN", name="AI 밸류체인 동조",
                            current_price=0, qty=0,
                            extra={
                                "mvp": "5",
                                "type": "AI_CHAIN_SYNC",
                                "fire_sectors": ai_sig.fire_sectors,
                                "fire_count": ai_sig.fire_sector_count,
                                "surge_count": len(ai_sig.surge_stocks),
                                "top_surge": [s["ticker"] for s in ai_sig.surge_stocks[:5]],
                            },
                        )
                    except Exception as _e:
                        logger.warning("MVP-5 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp5"]["errors"].append(str(e))

    # === MVP-6: 모멘텀 추격 매수 (5/26 20:30 신규, 불장 추격 매매) ===
    # 5분봉 +3% + 거래량 평균 3배+ + 양봉 + 종일 1~20% → 즉시 1주 매수
    # 후보: intraday_eye 워치리스트 + AI 동조 워치리스트 + sector_fire AI 6섹터
    if "mvp6" not in skip:
        try:
            from src.use_cases.momentum_chase import (
                scan_momentum_candidates, format_momentum_for_telegram,
            )
            summary["mvp6"]["executed"] = True

            # 후보 풀 수집
            cand_set = set()
            # 1) 워치리스트
            try:
                import yaml as _yaml
                cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
                if cfg_path.exists():
                    _cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                    cand_set.update(_cfg.get("intraday_eye", {}).get("watchlist", []))
            except Exception:
                pass

            # 2) AI 동조 워치리스트
            try:
                from src.use_cases.ai_chain_auto_watchlist import get_ai_chain_watchlist_tickers
                cand_set.update(get_ai_chain_watchlist_tickers())
            except Exception:
                pass

            # 3) sector_fire AI 6섹터 (오늘 5/26 추가)
            try:
                import yaml as _yaml
                sm_path = Path(__file__).resolve().parent.parent / "config" / "sector_fire_map.yaml"
                if sm_path.exists():
                    _sm = _yaml.safe_load(sm_path.read_text(encoding="utf-8")) or {}
                    for s_name in ["AI반도체", "AI반도체검사", "AI반도체PCB", "AI반도체장비설계",
                                    "AI반도체소재", "AI산업소재", "AI보안소프트웨어"]:
                        s_info = _sm.get("sectors", {}).get(s_name, {})
                        cand_set.update(s_info.get("tickers", []))
            except Exception:
                pass

            # 보호 + 보유 제외용 정보
            from src.use_cases.adaptive_position_manager import _load_protected_tickers
            protected = _load_protected_tickers()
            held = set()
            if hasattr(broker, "fetch_balance"):
                try:
                    bal = broker.fetch_balance()
                    for h in bal.get("holdings", []):
                        t = str(h.get("ticker", "")).zfill(6)
                        if t:
                            held.add(t)
                except Exception:
                    pass

            # KisIntradayAdapter 인스턴스 (5분봉 + 호가)
            intraday_for_momentum = intraday_adapter if 'intraday_adapter' in dir() else None
            if intraday_for_momentum is None:
                try:
                    from src.adapters.kis_intraday_adapter import KisIntradayAdapter
                    intraday_for_momentum = KisIntradayAdapter()
                except Exception as _e:
                    logger.warning("[MVP-6] KisIntradayAdapter 초기화 실패: %s", _e)

            if intraday_for_momentum and cand_set:
                # 후보 풀이 너무 크면 상위 30개만 (API 호출 제한)
                cand_list = list(cand_set)[:30]
                fires = scan_momentum_candidates(
                    intraday_for_momentum, cand_list,
                    held_tickers=held, protected_tickers=protected,
                )
                summary["mvp6"]["triggers"] = len(fires)

                for sig in fires:
                    msg = format_momentum_for_telegram(sig)
                    print(msg)
                    send_telegram(msg)
                    logger.warning(
                        "[MVP-6] 모멘텀 추격 발화: %s %s",
                        sig.ticker, sig.reason,
                    )
                    # 학습 로그
                    if learning_mode:
                        try:
                            from src.use_cases.decision_logger import log_decision
                            log_decision(
                                "ALERT", sig.ticker, name=sig.name,
                                current_price=sig.current_price,
                                qty=1,
                                extra={
                                    "mvp": "6",
                                    "type": "MOMENTUM_CHASE",
                                    "five_min_pct": sig.five_min_pct,
                                    "vol_ratio": sig.vol_ratio,
                                    "daily_pct": sig.daily_pct,
                                    "target_price": sig.target_price,
                                    "stop_price": sig.stop_price,
                                    "profit_target": sig.profit_target,
                                },
                            )
                        except Exception as _e:
                            logger.warning("MVP-6 log_decision 실패: %s", _e)
        except Exception as e:
            summary["mvp6"]["errors"].append(str(e))

    summary["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
    return summary


def main():
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="paper 시뮬 모드 (KIS mock)")
    mode_group.add_argument("--real", action="store_true", help="실전 모드 (KIS 실거래)")
    parser.add_argument("--dry-run", action="store_true",
                        help="MockBroker 사용, KIS 호출 0건")
    parser.add_argument("--skip-mvp1", action="store_true")
    parser.add_argument("--skip-mvp2", action="store_true")
    parser.add_argument("--skip-mvp2_5", action="store_true")
    parser.add_argument("--skip-mvp2_6", action="store_true")
    parser.add_argument("--skip-mvp3", action="store_true")
    parser.add_argument("--skip-mvp4", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    is_paper = args.paper
    skip: set[str] = set()
    if args.skip_mvp1: skip.add("mvp1")
    if args.skip_mvp2: skip.add("mvp2")
    if args.skip_mvp2_5: skip.add("mvp2_5")
    if args.skip_mvp3: skip.add("mvp3")
    if args.skip_mvp4: skip.add("mvp4")

    print("=" * 70)
    print(f"🔁 적응형 포지션 매매법 통합 사이클 ({dt.datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"   모드: {'PAPER' if is_paper else '🔴 REAL'}"
          f"{'  (DRY_RUN)' if args.dry_run else ''}")
    print(f"   스킵: {skip or '없음'}")
    print("=" * 70)

    summary = run_cycle(is_paper=is_paper, skip=skip, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("📊 사이클 요약:")
    for mvp in ("mvp1", "mvp2", "mvp2_5", "mvp2_6", "mvp2_7", "mvp3", "mvp4", "mvp5", "mvp6"):
        s = summary.get(mvp, {"executed": False, "triggers": 0, "errors": []})
        status = "✓" if s["executed"] else "⏭ SKIP"
        print(f"  {mvp.upper()}: {status}  트리거 {s['triggers']}건"
              f"{'  오류 ' + str(len(s['errors'])) + '건' if s['errors'] else ''}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
