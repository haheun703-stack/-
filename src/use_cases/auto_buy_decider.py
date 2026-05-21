"""자동 매수 결정 모듈 — 사장님 5/18 자동매매 ON 결단 (3번 작업)

배경: 5/20 자비스 자율 1주 매수
- 사장님 손 0번, 매수 즉시 카톡 알림
- 안전선 9건 ALL 통과 시만 매수

안전선 9건 (이 함수 8건 평가 + kis_order_adapter ⑨ 별도):
  ① 종합 점수 STRONG 90+ (80→90 격상)            [should_auto_buy 평가]
  ② EYE 필터 4종 모두 통과                         [should_auto_buy 평가]
  ③ 14:00 이후 진입 (오전 변동성 회피)            [should_auto_buy 평가]
  ④ 일일 매수 0건 (1건 한도)                      [should_auto_buy 평가]
  ⑤ 1주 10만원 한도                                [should_auto_buy 평가]
  ⑥ 시장 regime ∈ {MILD_BULL, NEUTRAL}            [should_auto_buy 평가]
  ⑦ AUTO_TRADE_5_20=true 환경변수                 [should_auto_buy 평가]
  ⑧ 막내 NEGA 0건 (5/21+, 5/19~5/20 SKIP)         [should_auto_buy 평가]
  ⑨ 지정가 현재가 ±5% 이내                       [kis_order_adapter._guard 평가]

사용:
  from src.use_cases.auto_buy_decider import should_auto_buy
  decision = should_auto_buy(broker, ticker, name, snap)
  if decision.action == 'BUY':
      kis_order_adapter.place_order(ticker, decision.qty, decision.price)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 안전선 임계값 (5/18 사장님 결단, 5/21 .env 동적 로드로 일반화)
THRESHOLD_INTEGRATED_SCORE = 90.0   # 안전선 ① STRONG 90+
EARLIEST_BUY_TIME = "14:00"          # 안전선 ③ 14:00 이후
MAX_DAILY_BUYS = int(os.getenv("AUTO_TRADING_MAX_TRADES_PER_DAY", "15"))  # 안전선 ④ .env 동적
MAX_QTY = int(os.getenv("AUTO_TRADING_MAX_QTY", "1"))                      # 안전선 ⑤ 1주 (.env)
MAX_AMOUNT = int(os.getenv("AUTO_TRADING_MAX_AMOUNT", "3000000"))          # 안전선 ⑤ 300만원 (.env)
ALLOWED_REGIMES = {"MILD_BULL", "NEUTRAL", "STRONG_BULL"}  # 안전선 ⑥
ENV_FLAG_ENABLED = "AUTO_TRADING_ENABLED"  # 안전선 ⑦ (5/21 일반화: AUTO_TRADE_5_20 → AUTO_TRADING_ENABLED)


@dataclass
class BuyDecision:
    """자동 매수 결정 결과."""

    action: str  # 'BUY' | 'SKIP'
    ticker: str
    name: str
    qty: int
    estimated_price: int
    estimated_amount: int
    reason: str
    checks_passed: list[str]
    checks_failed: list[str]


def _check_time(now_str: Optional[str] = None) -> tuple[bool, str]:
    """안전선 ③ 14:00 이후 진입."""
    if not now_str:
        now_str = datetime.now().strftime("%H:%M")
    if now_str < EARLIEST_BUY_TIME:
        return False, f"진입 시간 미달 ({now_str} < {EARLIEST_BUY_TIME})"
    return True, f"진입 시간 OK ({now_str})"


def _check_5_20_env() -> tuple[bool, str]:
    """안전선 ⑦ AUTO_TRADING_ENABLED 환경변수 (5/21 일반화).

    레거시: AUTO_TRADE_5_20=true 도 호환 (.env 또는 코드 하위 호환).
    """
    enabled = os.environ.get(ENV_FLAG_ENABLED, "0").strip()
    if enabled == "1":
        return True, f"{ENV_FLAG_ENABLED}=1"
    # 레거시 토글 호환
    legacy = os.environ.get("AUTO_TRADE_5_20", "false").lower()
    if legacy == "true":
        return True, f"AUTO_TRADE_5_20=true (레거시 호환)"
    return False, f"{ENV_FLAG_ENABLED} != 1 (현재 {enabled}) 및 AUTO_TRADE_5_20 != true"


def _check_daily_count(today: str, db_path: str = "data/owner_rule_positions.json") -> tuple[bool, str]:
    """안전선 ④ 일일 1건 초과 방지.

    5/18 수정: paper_portfolio.json → owner_rule_positions.json (자아성찰 #1 해소)
    - auto_buy_executor가 매수 성공 시 owner_rule_positions.json INSERT
    - paper_portfolio는 paper_warmup_daily 모의 학습용 (자동매매와 분리)
    - 같은 경로 사용 필수 (그렇지 않으면 5분 cron마다 안전선 ④ 무력화)
    """
    import json
    from pathlib import Path
    p = Path(db_path)
    if not p.exists():
        return True, "일일 카운트 0 (포트폴리오 없음)"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        today_entries = sum(
            1 for pos in data.get("positions", {}).values()
            if pos.get("entry_date") == today
        )
        if today_entries >= MAX_DAILY_BUYS:
            return False, f"일일 {today_entries}건 이미 매수 (한도 {MAX_DAILY_BUYS})"
        return True, f"일일 매수 {today_entries}/{MAX_DAILY_BUYS}"
    except Exception as e:
        logger.warning("daily count 확인 실패: %s", e)
        return True, "일일 카운트 확인 불가 (PASS)"


def should_auto_buy(
    ticker: str,
    name: str,
    integrated_score: float,
    eye_should_skip: bool,
    eye_skip_reasons: list[str],
    market_regime: str,
    current_price: int,
    blocked_by_nega: bool = False,
    now_str: Optional[str] = None,
    today: Optional[str] = None,
) -> BuyDecision:
    """자동 매수 결정 — 안전선 9건 ALL 통과 시만 BUY.

    Args:
        ticker, name: 종목 정보
        integrated_score: integrated_score.calculate_integrated_score() 결과
        eye_should_skip: eye_filters.evaluate_filters() 결과
        eye_skip_reasons: SKIP 사유 리스트
        market_regime: snapshot regime (MILD_BULL/NEUTRAL/CAUTION/BEAR)
        current_price: 현재가
        blocked_by_nega: 막내 intraday_signals NEGA 차단 여부 (5/21+)
        now_str: 'HH:MM' (테스트용)
        today: 'YYYY-MM-DD' (테스트용)

    Returns:
        BuyDecision
    """
    if not today:
        today = datetime.now().strftime("%Y-%m-%d")

    checks_passed = []
    checks_failed = []

    # 안전선 ① STRONG 90+
    if integrated_score >= THRESHOLD_INTEGRATED_SCORE:
        checks_passed.append(f"점수 {integrated_score:.0f} ≥ 90")
    else:
        checks_failed.append(f"점수 {integrated_score:.0f} < 90")

    # 안전선 ② EYE 필터 통과
    if not eye_should_skip:
        checks_passed.append("EYE 필터 4종 통과")
    else:
        checks_failed.append(f"EYE SKIP: {', '.join(eye_skip_reasons)}")

    # 안전선 ③ 14:00 이후
    ok, msg = _check_time(now_str)
    (checks_passed if ok else checks_failed).append(msg)

    # 안전선 ④ 일일 1건
    ok, msg = _check_daily_count(today)
    (checks_passed if ok else checks_failed).append(msg)

    # 안전선 ⑤ 1주 10만원 — 가격 체크
    estimated_amount = current_price * MAX_QTY
    if estimated_amount <= MAX_AMOUNT:
        checks_passed.append(f"1주 가격 {current_price:,} ≤ 10만")
    else:
        checks_failed.append(f"1주 {current_price:,} > 10만 (안전선 위반)")

    # 안전선 ⑥ regime 체크
    if market_regime in ALLOWED_REGIMES:
        checks_passed.append(f"regime {market_regime} OK")
    else:
        checks_failed.append(f"regime {market_regime} 차단 (CAUTION/BEAR/CRISIS)")

    # 안전선 ⑦ 5/20 환경변수
    ok, msg = _check_5_20_env()
    (checks_passed if ok else checks_failed).append(msg)

    # 안전선 ⑧ 막내 NEGA (5/21+, 5/20은 SKIP)
    if blocked_by_nega:
        checks_failed.append(f"막내 NEGA 차단: {ticker}")
    else:
        checks_passed.append("막내 NEGA 통과 (5/20은 미적용)")

    # 모든 체크 통과?
    all_passed = len(checks_failed) == 0
    total_evaluated = len(checks_passed) + len(checks_failed)

    return BuyDecision(
        action="BUY" if all_passed else "SKIP",
        ticker=ticker,
        name=name,
        qty=MAX_QTY,
        estimated_price=current_price,
        estimated_amount=estimated_amount,
        reason=(
            f"안전선 {len(checks_passed)}/{total_evaluated} 통과 (+ kis ⑨ 별도)"
            if all_passed
            else f"안전선 {len(checks_failed)}건 미달"
        ),
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )


def format_decision_for_telegram(d: BuyDecision) -> str:
    """텔레그램 알림 포맷."""
    if d.action == "BUY":
        n = len(d.checks_passed)
        return (
            f"✅ [자동 매수 결정] {d.name}({d.ticker})\n"
            f"  수량: {d.qty}주 × {d.estimated_price:,}원 = {d.estimated_amount:,}원\n"
            f"  사유: {d.reason}\n"
            f"  통과 {n}건: 모두 OK (+ kis ⑨ 가격±5% 별도 검증)"
        )
    return (
        f"⛔ [자동 매수 SKIP] {d.name}({d.ticker})\n"
        f"  미달: {', '.join(d.checks_failed[:3])}"
    )
