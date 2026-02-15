"""
텔레그램 메시지 포맷터 — "퀀텀전략" v3.0

KISBOT v4.3 스타일 참조:
  - 시장 상태, 백테스트 결과, 매수 추천 종목을 구조화된 메시지로 생성
  - 6-Layer Pipeline 진단, 트리거별/등급별 성과 포함

기호 참조 (ICONS):
  시장 상태:  BULL / BEAR / SIDEWAYS
  등급:       A / B / C
  트리거:     IMPULSE / CONFIRM / BREAKOUT
  수익/손실:  WIN / LOSS
  레짐:       ADVANCE / DISTRIB / ACCUM
  6-Layer:    PASS / BLOCK
  랭킹:       RANK_1 / RANK_2 / RANK_3
"""

from datetime import datetime


# ============================================================
# 기호 참조 맵 (KISBOT 스타일 + 우리 시스템 확장)
# ============================================================
ICONS = {
    # 시장 상태
    "BULL": "\U0001f7e2",           # 🟢
    "BEAR": "\U0001f534",           # 🔴
    "SIDEWAYS": "\U0001f7e1",       # 🟡

    # 등급
    "A": "\U0001f31f",              # 🌟
    "B": "\U0001f44d",              # 👍
    "C": "\u26a0\ufe0f",            # ⚠️

    # 트리거
    "IMPULSE": "\u26a1",            # ⚡
    "CONFIRM": "\U0001f3af",        # 🎯
    "BREAKOUT": "\U0001f680",       # 🚀

    # 수익/손실
    "WIN": "\U0001f7e2",            # 🟢
    "LOSS": "\U0001f534",           # 🔴
    "EVEN": "\u26aa",               # ⚪

    # 레짐 (HMM)
    "ADVANCE": "\U0001f4c8",        # 📈
    "DISTRIB": "\U0001f4c9",        # 📉
    "ACCUM": "\U0001f4e6",          # 📦

    # 6-Layer
    "PASS": "\u2705",               # ✅
    "BLOCK": "\u274c",              # ❌

    # 랭킹
    "RANK_1": "\U0001f3c6",         # 🏆
    "RANK_2": "\U0001f948",         # 🥈
    "RANK_3": "\U0001f949",         # 🥉

    # 뉴스 등급 (v3.1)
    "NEWS_A": "\U0001f4f0\U0001f31f",  # 📰🌟
    "NEWS_B": "\U0001f4f0\U0001f50d",  # 📰🔍
    "NEWS_C": "\U0001f4f0",            # 📰

    # 매집 단계 (v3.1)
    "ACCUM_1": "\U0001f331",        # 🌱 Phase 1 (초기)
    "ACCUM_2": "\U0001f33f",        # 🌿 Phase 2 (중기)
    "ACCUM_3": "\U0001f332",        # 🌲 Phase 3 (점화)
    "DUMP": "\U0001f6a8",           # 🚨 투매

    # 기타
    "CHART": "\U0001f4ca",          # 📊
    "ALERT": "\U0001f514",          # 🔔
    "MONEY": "\U0001f4b0",          # 💰
    "FIRE": "\U0001f525",           # 🔥
    "CLOCK": "\u23f0",              # ⏰
    "STAR": "\u2b50",               # ⭐
    "CHECK": "\u2714\ufe0f",        # ✔️
    "MEMO": "\U0001f4dd",           # 📝
    "GEAR": "\u2699\ufe0f",         # ⚙️
    "LINE": "\u2500" * 30,          # ──────────
}


def _icon(key: str) -> str:
    """아이콘 키 → 유니코드 변환 (없으면 빈 문자열)"""
    return ICONS.get(key, "")


def _sign(value: float) -> str:
    """양수는 +, 음수는 자동"""
    return f"+{value}" if value > 0 else f"{value}"


def _comma(value) -> str:
    """천 단위 콤마"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


# ============================================================
# 메인 포맷터
# ============================================================

def format_scan_result(
    stats: dict,
    signals: list[dict] = None,
    diagnostic: dict = None,
    scan_date: str = None,
) -> str:
    """
    전체 스캔 결과 텔레그램 메시지 생성.

    Args:
        stats: calc_full_metrics() 반환값
        signals: signal_engine.scan_universe() 결과 리스트
        diagnostic: SignalDiagnostic.summarize() 결과
        scan_date: 스캔 날짜 (없으면 오늘)

    Returns:
        텔레그램 메시지 문자열
    """
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")

    parts = []

    # ── 헤더 ──
    parts.append(_format_header(scan_date))

    # ── 백테스트 성과 ──
    if stats:
        parts.append(_format_backtest_stats(stats))

    # ── 6-Layer 진단 ──
    if diagnostic:
        parts.append(_format_diagnostic(diagnostic))

    # ── 매수 추천 종목 ──
    if signals:
        parts.append(_format_recommendations(signals))

    # ── 푸터 ──
    parts.append(_format_footer())

    return "\n".join(parts)


def format_trade_alert(signal: dict, action: str = "BUY") -> str:
    """
    실시간 매매 알림 메시지.

    Args:
        signal: 시그널 dict (ticker, grade, trigger_type, entry_price, ...)
        action: "BUY" / "SELL"
    """
    trigger = signal.get("trigger_type", "confirm").upper()
    trigger_icon = _icon(trigger)
    grade = signal.get("grade", "C")
    grade_icon = _icon(grade)

    if action == "BUY":
        msg = (
            f"{_icon('ALERT')} 매수 시그널 발생\n"
            f"{ICONS['LINE']}\n"
            f"  종목: {signal.get('ticker', '?')}\n"
            f"  등급: {grade_icon} {grade}등급 (BES {signal.get('zone_score', 0):.2f})\n"
            f"  트리거: {trigger_icon} {trigger}\n"
            f"  진입가: {_comma(signal.get('entry_price', 0))}원\n"
            f"  손절가: {_comma(signal.get('stop_loss', 0))}원\n"
            f"  목표가: {_comma(signal.get('target_price', 0))}원\n"
            f"  손익비: 1:{signal.get('risk_reward_ratio', 0):.1f}\n"
            f"{ICONS['LINE']}\n"
        )
    else:
        pnl_pct = signal.get("pnl_pct", 0)
        pnl_icon = _icon("WIN") if pnl_pct > 0 else _icon("LOSS")
        msg = (
            f"{pnl_icon} 매도 완료\n"
            f"{ICONS['LINE']}\n"
            f"  종목: {signal.get('ticker', '?')}\n"
            f"  매도가: {_comma(signal.get('exit_price', 0))}원\n"
            f"  수익률: {_sign(pnl_pct)}%\n"
            f"  사유: {signal.get('exit_reason', '?')}\n"
            f"  보유일: {signal.get('hold_days', 0)}일\n"
            f"{ICONS['LINE']}\n"
        )

    return msg


# ============================================================
# 내부 포맷 함수
# ============================================================

def _format_header(scan_date: str) -> str:
    return (
        f"{_icon('CHART')} [퀀텀전략 v3.0] {scan_date} 6-Layer 스캔\n"
        f"{ICONS['LINE']}"
    )


def _format_backtest_stats(stats: dict) -> str:
    """백테스트 성과 섹션"""
    total = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0)
    avg_win = stats.get("avg_win_pct", 0)
    avg_loss = stats.get("avg_loss_pct", 0)
    expectancy = stats.get("expectancy", 0)
    pf = stats.get("profit_factor", 0)
    sharpe = stats.get("sharpe_ratio", 0)
    sortino = stats.get("sortino_ratio", 0)
    calmar = stats.get("calmar_ratio", 0)
    mdd = stats.get("max_drawdown_pct", 0)
    total_return = stats.get("total_return_pct", 0)
    cagr = stats.get("cagr_pct", 0)
    hold_days = stats.get("avg_hold_days", 0)

    # 성과 이모지 판정
    perf_icon = _icon("WIN") if total_return > 0 else _icon("LOSS")

    lines = [
        f"\n{_icon('CHART')} [ 백테스트 성과 ]",
        f"  거래: {total}건 | 승률: {win_rate:.1f}%",
        f"  평균이익: {_sign(avg_win)}% | 평균손실: {avg_loss:.2f}%",
        f"  기대값: {_sign(expectancy)}%/거래",
        f"  Profit Factor: {pf:.2f}",
        f"  평균보유: {hold_days}일",
        "",
        f"  {perf_icon} 총수익률: {_sign(total_return)}%",
        f"  CAGR: {_sign(cagr)}% | MDD: {mdd:.1f}%",
        f"  Sharpe: {sharpe:.2f} | Sortino: {sortino:.3f} | Calmar: {calmar:.3f}",
    ]

    # 트리거별 분석
    tb = stats.get("trigger_breakdown", {})
    if tb:
        lines.append(f"\n  {_icon('FIRE')} [트리거별]")
        for ttype, t_stats in tb.items():
            t_icon = _icon(ttype.upper())
            lines.append(
                f"    {t_icon} {ttype}: {t_stats['count']}건 "
                f"승률={t_stats['win_rate']:.1f}% "
                f"PF={t_stats['profit_factor']:.2f} "
                f"E={t_stats['expectancy']:.2f}%"
            )

    # 등급별 분석
    gb = stats.get("grade_breakdown", {})
    if gb:
        lines.append(f"\n  {_icon('STAR')} [등급별]")
        for grade, g_stats in gb.items():
            g_icon = _icon(grade)
            lines.append(
                f"    {g_icon} {grade}등급: {g_stats['count']}건 "
                f"승률={g_stats['win_rate']:.1f}% "
                f"PF={g_stats['profit_factor']:.2f}"
            )

    return "\n".join(lines)


def _format_diagnostic(diagnostic: dict) -> str:
    """6-Layer Pipeline 진단 섹션"""
    total_eval = diagnostic.get("total_evaluations", 0)
    final_sig = diagnostic.get("final_signals", 0)
    sig_rate = diagnostic.get("signal_rate", 0)
    layers = diagnostic.get("layers", {})

    lines = [
        f"\n{_icon('GEAR')} [ 6-Layer Pipeline 진단 ]",
        f"  총 평가: {_comma(total_eval)}건 -> 시그널: {final_sig}건 ({sig_rate}%)",
    ]

    for name, l_stats in layers.items():
        reached = l_stats.get("reached", 0)
        passed = l_stats.get("passed", 0)
        rate = l_stats.get("pass_rate", 0)
        icon = _icon("PASS") if rate >= 50 else _icon("BLOCK")

        # 간이 게이지 (5단계)
        gauge_filled = int(rate / 20)
        gauge = "\u2588" * gauge_filled + "\u2591" * (5 - gauge_filled)

        lines.append(f"  {icon} {name:16s} [{gauge}] {rate:5.1f}% ({passed}/{reached})")

    return "\n".join(lines)


def _format_recommendations(signals: list[dict]) -> str:
    """매수 추천 종목 섹션"""
    if not signals:
        return f"\n{_icon('MEMO')} [ 매수 후보 ]\n  해당 없음"

    # zone_score 기준 정렬
    sorted_signals = sorted(signals, key=lambda s: s.get("zone_score", 0), reverse=True)

    lines = []

    # 1순위 추천
    top = sorted_signals[0]
    trigger = top.get("trigger_type", "confirm").upper()
    grade = top.get("grade", "C")

    lines.append(f"\n{_icon('RANK_1')} [ 1순위 추천 매수 ]")
    lines.append(f"  종목: {top.get('ticker', '?')}")
    lines.append(f"  등급: {_icon(grade)} {grade}등급 | BES: {top.get('zone_score', 0):.2f}")
    lines.append(f"  트리거: {_icon(trigger)} {trigger}")
    lines.append(f"  진입가: {_comma(top.get('entry_price', 0))}원")
    lines.append(f"  손절가: {_comma(top.get('stop_loss', 0))}원")
    lines.append(f"  목표가: {_comma(top.get('target_price', 0))}원")
    lines.append(f"  손익비: 1:{top.get('risk_reward_ratio', 0):.1f}")

    # 2순위 이하
    if len(sorted_signals) > 1:
        lines.append(f"\n{_icon('MEMO')} [ 매수 후보 ]")
        for i, sig in enumerate(sorted_signals[1:], start=2):
            rank_icon = _icon(f"RANK_{i}") if i <= 3 else f" {i}."
            t_type = sig.get("trigger_type", "confirm").upper()
            g = sig.get("grade", "C")
            lines.append(
                f"  {rank_icon} {sig.get('ticker', '?')} "
                f"{_icon(g)}{g} "
                f"{_icon(t_type)}{t_type} "
                f"BES={sig.get('zone_score', 0):.2f} "
                f"RR=1:{sig.get('risk_reward_ratio', 0):.1f}"
            )

    # 전체 요약
    lines.append(f"\n  {_icon('CHECK')} 전체 후보: {len(sorted_signals)}종목")

    return "\n".join(lines)


def _format_footer() -> str:
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"\n{ICONS['LINE']}\n"
        f"{_icon('CLOCK')} {now} | 6-Layer Pipeline Quant v3.1\n"
        f"  4단계 부분청산(2R/4R/8R/10R) | HMM 레짐 | OU 필터 | News Gate"
    )


# ============================================================
# 편의 함수: 백테스트 완료 후 전체 리포트 메시지
# ============================================================

def format_backtest_report(results: dict, scan_date: str = None) -> str:
    """
    BacktestEngine._compile_results() 반환값으로 전체 리포트 메시지 생성.

    Args:
        results: {"stats": dict, "trades_df": df, "equity_df": df,
                  "signals_df": df, "diagnostic": dict}
    """
    stats = results.get("stats", {})
    diagnostic = results.get("diagnostic", {})

    # signals_df에서 최근 시그널 추출
    signals_df = results.get("signals_df", None)
    recent_signals = []
    if signals_df is not None and not signals_df.empty:
        # 마지막 날짜 시그널만
        last_date = signals_df["date"].max()
        recent = signals_df[signals_df["date"] == last_date]
        for _, row in recent.iterrows():
            recent_signals.append(row.to_dict())

    return format_scan_result(
        stats=stats,
        signals=recent_signals if recent_signals else None,
        diagnostic=diagnostic,
        scan_date=scan_date,
    )


# ============================================================
# v3.1 뉴스/매집 알림 포맷터
# ============================================================

def format_news_alert(
    ticker: str,
    grade: str,
    action: str,
    reason: str = "",
    news_text: str = "",
    param_overrides: dict = None,
    entry_price: float = 0,
    target_price: float = 0,
    stop_loss: float = 0,
    pipeline_passed: bool = False,
) -> str:
    """
    v3.1 뉴스 등급 알림 메시지.

    Args:
        ticker: 종목코드
        grade: "A" / "B" / "C"
        action: EventDrivenAction 값 문자열
        reason: 등급 판정 사유
        news_text: 뉴스 요약
        param_overrides: 파라미터 조정 dict
        entry_price: 진입가 (A급, 파이프라인 통과 시)
        target_price: 목표가
        stop_loss: 손절가
        pipeline_passed: 6-Layer 통과 여부
    """
    grade_key = f"NEWS_{grade}" if f"NEWS_{grade}" in ICONS else "NEWS_C"
    grade_icon = _icon(grade_key)

    parts = [
        f"{grade_icon} [v3.1 News Gate] {grade}등급 뉴스 감지",
        ICONS["LINE"],
        f"  종목: {ticker}",
        f"  등급: {grade}등급",
        f"  판정: {action}",
    ]

    if reason:
        parts.append(f"  사유: {reason}")
    if news_text:
        parts.append(f"  뉴스: {news_text[:100]}{'...' if len(news_text) > 100 else ''}")

    # A급: 파라미터 조정 + 진입 정보
    if grade == "A" and param_overrides:
        parts.append(f"\n  {_icon('GEAR')} [파라미터 조정]")
        rr = param_overrides.get("rr_min", "")
        rsi = param_overrides.get("rsi_entry_max", "")
        pos = param_overrides.get("position_size_pct", "")
        hold = param_overrides.get("max_hold_days", "")
        if rr:
            parts.append(f"    손익비 하한: 1:{rr}")
        if rsi:
            parts.append(f"    RSI 상한: {rsi}")
        if pos:
            parts.append(f"    포지션: {pos}%")
        if hold:
            parts.append(f"    최대 보유: {hold}일")

    # 파이프라인 통과 시 진입 정보
    if pipeline_passed and entry_price > 0:
        pip_icon = _icon("PASS")
        parts.append(f"\n  {pip_icon} [6-Layer 통과 → 진입 가능]")
        parts.append(f"    진입가: {_comma(entry_price)}원")
        if stop_loss > 0:
            parts.append(f"    손절가: {_comma(stop_loss)}원")
        if target_price > 0:
            parts.append(f"    목표가: {_comma(target_price)}원")
    elif grade == "A" and not pipeline_passed:
        parts.append(f"\n  {_icon('BLOCK')} 6-Layer 미통과 — 진입 보류")

    # B급: 관찰 안내
    if grade == "B":
        parts.append(f"\n  {_icon('MEMO')} 관찰 리스트 등록 (14일 모니터링)")

    parts.append(ICONS["LINE"])
    return "\n".join(parts)


def format_accumulation_alert(
    ticker: str,
    phase: int,
    confidence: float = 0,
    bonus_score: int = 0,
    inst_streak: int = 0,
    foreign_streak: int = 0,
    obv_divergence: str = "",
    description: str = "",
) -> str:
    """
    v3.1 매집 단계 알림 메시지.

    Args:
        ticker: 종목코드
        phase: 매집 단계 (1/2/3) 또는 -1(투매)
        confidence: 신뢰도 (0~100)
        bonus_score: 보너스 점수 (+5/+10/+15/-20)
        inst_streak: 기관 연속 순매수 일수
        foreign_streak: 외국인 연속 순매수 일수
        obv_divergence: "bullish" / "bearish" / ""
        description: 상세 설명
    """
    if phase == -1:
        phase_icon = _icon("DUMP")
        phase_name = "투매 감지"
    elif phase == 3:
        phase_icon = _icon("ACCUM_3")
        phase_name = "Phase 3 (점화)"
    elif phase == 2:
        phase_icon = _icon("ACCUM_2")
        phase_name = "Phase 2 (중기 매집)"
    elif phase == 1:
        phase_icon = _icon("ACCUM_1")
        phase_name = "Phase 1 (초기 매집)"
    else:
        phase_icon = _icon("MEMO")
        phase_name = "매집 미감지"

    parts = [
        f"{phase_icon} [Smart Money v2] 매집 분석",
        ICONS["LINE"],
        f"  종목: {ticker}",
        f"  단계: {phase_name}",
        f"  신뢰도: {confidence:.0f}%",
        f"  보너스: {_sign(bonus_score)}점",
    ]

    # 수급 정보
    if inst_streak != 0 or foreign_streak != 0:
        parts.append(f"\n  {_icon('MONEY')} [수급 현황]")
        if inst_streak != 0:
            direction = "순매수" if inst_streak > 0 else "순매도"
            parts.append(f"    기관: {abs(inst_streak)}일 연속 {direction}")
        if foreign_streak != 0:
            direction = "순매수" if foreign_streak > 0 else "순매도"
            parts.append(f"    외국인: {abs(foreign_streak)}일 연속 {direction}")

    # OBV 다이버전스
    if obv_divergence:
        div_icon = _icon("ADVANCE") if obv_divergence == "bullish" else _icon("DISTRIB")
        div_name = "불리시" if obv_divergence == "bullish" else "베어리시"
        parts.append(f"    OBV: {div_icon} {div_name} 다이버전스")

    if description:
        parts.append(f"\n  {description}")

    parts.append(ICONS["LINE"])
    return "\n".join(parts)


# ──────────────────────────────────────────
# v4.0 라이브 트레이딩 포맷
# ──────────────────────────────────────────

def format_order_result(order, action: str = "BUY") -> str:
    """주문 체결/실패 알림 포맷."""
    icon = _icon("BUY") if action == "BUY" else _icon("SELL")
    status_icon = _icon("ADVANCE") if order.status.value in ("filled", "partial") else _icon("WARN")

    parts = [
        f"{icon} [{action}] 주문 결과",
        ICONS["LINE"],
        f"  종목: {order.ticker}",
        f"  상태: {status_icon} {order.status.value.upper()}",
        f"  주문가: {order.price:,}원",
        f"  수량: {order.quantity:,}주",
    ]

    if order.filled_quantity > 0:
        parts.append(f"  체결: {order.filled_quantity:,}주 @ {order.filled_price:,.0f}원")
        parts.append(f"  체결금액: {order.filled_quantity * order.filled_price:,.0f}원")

    if order.order_id:
        parts.append(f"  주문번호: {order.order_id}")

    parts.append(ICONS["LINE"])
    return "\n".join(parts)


def format_position_summary(positions: list) -> str:
    """보유종목 현황표 포맷."""
    if not positions:
        return f"{_icon('INFO')} 보유종목 없음"

    parts = [
        f"{_icon('MONEY')} [보유종목 현황]",
        ICONS["LINE"],
    ]

    total_invested = 0
    total_eval = 0

    for p in positions:
        pnl_pct = p.unrealized_pnl_pct if hasattr(p, "unrealized_pnl_pct") else 0
        pnl_icon = _icon("ADVANCE") if pnl_pct >= 0 else _icon("DECLINE")
        eval_amount = p.current_price * p.shares

        parts.append(
            f"  {p.ticker} ({p.name})\n"
            f"    매입: {p.entry_price:,.0f}  현재: {p.current_price:,.0f}\n"
            f"    수량: {p.shares:,}주  {pnl_icon} {pnl_pct:+.1f}%\n"
            f"    손절: {p.stop_loss:,.0f}  부분청산: {p.partial_exits_done}/4"
        )

        total_invested += p.entry_price * p.shares
        total_eval += eval_amount

    total_pnl_pct = ((total_eval / total_invested) - 1) * 100 if total_invested > 0 else 0
    total_icon = _icon("ADVANCE") if total_pnl_pct >= 0 else _icon("DECLINE")

    parts.append(ICONS["LINE"])
    parts.append(
        f"  총 투자: {total_invested:,.0f}원\n"
        f"  총 평가: {total_eval:,.0f}원\n"
        f"  {total_icon} 수익률: {total_pnl_pct:+.1f}%\n"
        f"  종목수: {len(positions)}개"
    )

    parts.append(ICONS["LINE"])
    return "\n".join(parts)


def format_daily_performance(perf) -> str:
    """일일 성과 리포트 포맷."""
    pnl = perf.realized_pnl + perf.unrealized_pnl
    balance_change = perf.ending_balance - perf.starting_balance
    change_pct = (balance_change / perf.starting_balance) * 100 if perf.starting_balance > 0 else 0
    icon = _icon("ADVANCE") if change_pct >= 0 else _icon("DECLINE")

    win_rate = (perf.win_trades / perf.trades_executed * 100) if perf.trades_executed > 0 else 0

    parts = [
        f"{_icon('INFO')} [일일 성과 리포트] {perf.date}",
        ICONS["LINE"],
        f"  시작잔고: {perf.starting_balance:,.0f}원",
        f"  종료잔고: {perf.ending_balance:,.0f}원",
        f"  {icon} 변동: {balance_change:+,.0f}원 ({change_pct:+.2f}%)",
        ICONS["LINE"],
        f"  실현손익: {perf.realized_pnl:+,.0f}원",
        f"  미실현손익: {perf.unrealized_pnl:+,.0f}원",
        f"  매매횟수: {perf.trades_executed}건",
        f"  승률: {win_rate:.0f}% ({perf.win_trades}W/{perf.loss_trades}L)",
        ICONS["LINE"],
    ]
    return "\n".join(parts)


def format_emergency_alert(reason: str) -> str:
    """긴급 알림 포맷."""
    parts = [
        f"{_icon('WARN')}{_icon('WARN')}{_icon('WARN')} [긴급 알림]",
        ICONS["LINE"],
        f"  사유: {reason}",
        f"  조치: 전종목 시장가 청산 실행",
        f"  STOP.signal 생성됨",
        ICONS["LINE"],
        f"  수동 해제: STOP.signal 파일 삭제",
        f"  자동 해제: 00:00 일일 리셋",
        ICONS["LINE"],
    ]
    return "\n".join(parts)


def format_scheduler_status(phase: str, status: str, detail: str = "") -> str:
    """스케줄러 상태 포맷."""
    status_icon = _icon("ADVANCE") if status == "완료" else (
        _icon("WARN") if status == "실패" else _icon("INFO")
    )

    parts = [
        f"{_icon('INFO')} [스케줄러] {phase}",
        f"  상태: {status_icon} {status}",
    ]

    if detail:
        parts.append(f"  상세: {detail}")

    return "\n".join(parts)
