"""
전문 퀀트 성과지표 모듈

기존 Sharpe/MDD에 추가:
  - Sortino Ratio (하방 변동성 기준)
  - Calmar Ratio (CAGR / MDD)
  - Profit Factor (총이익 / 총손실)
  - Win/Loss Ratio, Expectancy
  - 트리거별/등급별 분석
  - v4.6: 통계적 신뢰도 등급 (표본수 기반)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 최소 표본수 임계값 ──
MIN_TRADES_CRITICAL = 30    # 30건 미만: 통계적 의미 없음
MIN_TRADES_WARNING = 50     # 50건 미만: 신뢰도 낮음
MIN_TRADES_RELIABLE = 100   # 100건 이상: 신뢰도 양호


def calc_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.03,
    annualize: int = 252,
) -> float:
    """
    Sortino Ratio = (연간 수익률 - Rf) / 하방편차

    하방편차: 음수 수익률만으로 계산 → 하락 리스크에 집중
    """
    if len(daily_returns) < 2:
        return 0.0

    excess = daily_returns - risk_free_rate / annualize
    downside = daily_returns[daily_returns < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    downside_std = downside.std() * np.sqrt(annualize)
    annual_return = daily_returns.mean() * annualize - risk_free_rate

    return round(annual_return / downside_std, 3)


def calc_calmar_ratio(
    daily_returns: pd.Series,
    equity_series: pd.Series,
    annualize: int = 252,
) -> float:
    """
    Calmar Ratio = CAGR / |MDD|

    MDD가 깊을수록 페널티. 리스크 조정 수익률.
    """
    if len(equity_series) < 2:
        return 0.0

    n_years = len(equity_series) / annualize
    if n_years <= 0:
        return 0.0

    total_return = equity_series.iloc[-1] / equity_series.iloc[0]
    if total_return <= 0:
        return 0.0

    cagr = total_return ** (1 / n_years) - 1

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    mdd = abs(drawdown.min())

    if mdd == 0:
        return 0.0

    return round(cagr / mdd, 3)


def calc_max_drawdown(equity_series: pd.Series) -> dict:
    """
    MDD 상세 분석.

    Returns:
        mdd_pct, mdd_start, mdd_end, mdd_duration_days
    """
    if len(equity_series) < 2:
        return {"mdd_pct": 0.0, "mdd_start": None, "mdd_end": None, "mdd_duration": 0}

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak

    mdd_pct = round(drawdown.min() * 100, 2)
    mdd_end_idx = drawdown.idxmin()

    # MDD 시작점 (직전 고점)
    peak_before = equity_series.loc[:mdd_end_idx]
    mdd_start_idx = peak_before.idxmax()

    # 회복 시점
    recovery_idx = None
    if mdd_end_idx is not None:
        after_trough = equity_series.loc[mdd_end_idx:]
        recovered = after_trough[after_trough >= peak.loc[mdd_start_idx]]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]

    duration = 0
    try:
        if mdd_start_idx is not None and mdd_end_idx is not None:
            duration = (pd.Timestamp(mdd_end_idx) - pd.Timestamp(mdd_start_idx)).days
    except Exception:
        pass

    return {
        "mdd_pct": mdd_pct,
        "mdd_start": str(mdd_start_idx),
        "mdd_end": str(mdd_end_idx),
        "mdd_duration": duration,
        "recovery_date": str(recovery_idx) if recovery_idx else None,
    }


def calc_profit_factor(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """
    Profit Factor = 총 이익 / 총 손실

    PF > 1.5 양호, PF > 2.0 우수
    """
    if trades_df.empty or pnl_col not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df[pnl_col] > 0][pnl_col].sum()
    losses = abs(trades_df[trades_df[pnl_col] <= 0][pnl_col].sum())

    if losses == 0:
        return float("inf") if wins > 0 else 0.0

    return round(wins / losses, 3)


def calc_expectancy(trades_df: pd.DataFrame, pnl_pct_col: str = "pnl_pct") -> float:
    """
    기대값 = 승률 × 평균이익 - 패률 × 평균손실

    거래당 기대 수익률(%). 양수여야 수익 시스템.
    """
    if trades_df.empty or pnl_pct_col not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df[pnl_pct_col] > 0]
    losses = trades_df[trades_df[pnl_pct_col] <= 0]

    n = len(trades_df)
    win_rate = len(wins) / n if n > 0 else 0
    loss_rate = 1 - win_rate

    avg_win = wins[pnl_pct_col].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses[pnl_pct_col].mean()) if len(losses) > 0 else 0

    return round(win_rate * avg_win - loss_rate * avg_loss, 3)


def calc_full_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float = 50_000_000,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    전체 성과지표 계산.

    Returns:
        dict with all quant metrics
    """
    metrics = {
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_rr_ratio": 0.0,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "avg_hold_days": 0,
        "total_commission": 0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "mdd_detail": {},
        "grade_breakdown": {},
        "trigger_breakdown": {},
    }

    if trades_df.empty or equity_df.empty:
        return metrics

    # 기본 통계
    metrics["total_trades"] = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    metrics["win_rate"] = round(len(wins) / len(trades_df) * 100, 1)
    metrics["avg_hold_days"] = int(trades_df["hold_days"].mean())
    metrics["total_commission"] = int(trades_df["commission"].sum())
    metrics["avg_win_pct"] = round(wins["pnl_pct"].mean(), 2) if len(wins) > 0 else 0
    metrics["avg_loss_pct"] = round(losses["pnl_pct"].mean(), 2) if len(losses) > 0 else 0

    # 평균 손익비
    avg_win = abs(wins["pnl_pct"].mean()) if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 1
    metrics["avg_rr_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    # 에쿼티 기반 지표
    equity_series = equity_df["portfolio_value"]
    daily_returns = equity_series.pct_change().dropna()

    # 총 수익률
    final_value = equity_series.iloc[-1]
    metrics["total_return_pct"] = round((final_value / initial_capital - 1) * 100, 1)

    # CAGR
    n_years = len(equity_df) / 252
    if n_years > 0 and final_value > 0:
        metrics["cagr_pct"] = round(
            ((final_value / initial_capital) ** (1 / n_years) - 1) * 100, 1
        )

    # MDD
    mdd_detail = calc_max_drawdown(equity_series)
    metrics["max_drawdown_pct"] = mdd_detail["mdd_pct"]
    metrics["mdd_detail"] = mdd_detail

    # Sharpe
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        excess = daily_returns.mean() - risk_free_rate / 252
        metrics["sharpe_ratio"] = round(
            excess / daily_returns.std() * np.sqrt(252), 2
        )

    # Sortino
    metrics["sortino_ratio"] = calc_sortino_ratio(daily_returns, risk_free_rate)

    # Calmar
    metrics["calmar_ratio"] = calc_calmar_ratio(daily_returns, equity_series)

    # Profit Factor
    metrics["profit_factor"] = calc_profit_factor(trades_df)

    # Expectancy
    metrics["expectancy"] = calc_expectancy(trades_df)

    # 등급별 분석
    for grade in ["A", "B", "C"]:
        gt = trades_df[trades_df["grade"] == grade]
        if len(gt) > 0:
            gw = gt[gt["pnl"] > 0]
            metrics["grade_breakdown"][grade] = {
                "count": len(gt),
                "win_rate": round(len(gw) / len(gt) * 100, 1),
                "avg_pnl_pct": round(gt["pnl_pct"].mean(), 2),
                "profit_factor": calc_profit_factor(gt),
            }

    # 트리거별 분석
    if "trigger_type" in trades_df.columns:
        for ttype in ["impulse", "confirm", "breakout"]:
            tt = trades_df[trades_df["trigger_type"] == ttype]
            if len(tt) > 0:
                tw = tt[tt["pnl"] > 0]
                metrics["trigger_breakdown"][ttype] = {
                    "count": len(tt),
                    "win_rate": round(len(tw) / len(tt) * 100, 1),
                    "avg_pnl_pct": round(tt["pnl_pct"].mean(), 2),
                    "avg_hold_days": int(tt["hold_days"].mean()),
                    "total_pnl": int(tt["pnl"].sum()),
                    "profit_factor": calc_profit_factor(tt),
                    "expectancy": calc_expectancy(tt),
                }

    # v4.6: 통계적 신뢰도 등급
    metrics["statistical_reliability"] = assess_reliability(
        n_trades=len(trades_df),
        bootstrap_results=None,
    )

    return metrics


def assess_reliability(
    n_trades: int,
    bootstrap_results: dict = None,
) -> dict:
    """
    v4.6 통계적 신뢰도 등급 평가.

    등급:
      A: 100건+ & Bootstrap 유의 → 높은 신뢰도
      B: 50~99건 또는 Bootstrap 미유의 → 보통 신뢰도
      C: 30~49건 → 낮은 신뢰도
      F: 30건 미만 → 통계적 의미 없음
    """
    # 기본 등급: 표본수 기준
    if n_trades >= MIN_TRADES_RELIABLE:
        grade = "A"
        message = "통계적으로 신뢰할 수 있는 표본수"
    elif n_trades >= MIN_TRADES_WARNING:
        grade = "B"
        message = "보통 수준의 표본수 (100건 이상 권장)"
    elif n_trades >= MIN_TRADES_CRITICAL:
        grade = "C"
        message = "표본수 부족 - 결과 해석에 주의 필요"
    else:
        grade = "F"
        message = f"표본수 심각 부족 ({n_trades}건) - 통계적 의미 없음"

    # Bootstrap 유의성 반영
    bs_significant = False
    if bootstrap_results and bootstrap_results.get("status") == "completed":
        sharpe_sig = bootstrap_results.get("shuffle_sharpe", {}).get("significant", False)
        return_sig = bootstrap_results.get("shuffle_total_return", {}).get("significant", False)
        bs_significant = sharpe_sig or return_sig

        if grade == "A" and not bs_significant:
            grade = "B"
            message += " (Bootstrap 미유의: 랜덤 대비 우위 불확실)"
        elif grade == "B" and bs_significant:
            message += " (Bootstrap 유의: 랜덤 대비 우위 확인)"

    return {
        "grade": grade,
        "n_trades": n_trades,
        "min_recommended": MIN_TRADES_RELIABLE,
        "bootstrap_significant": bs_significant,
        "message": message,
    }


def print_metrics(metrics: dict):
    """성과지표 콘솔 출력"""
    log = logging.getLogger(__name__)

    log.info(f"\n{'='*60}")
    log.info(f"v4.6 Quant Metrics Report")
    log.info(f"{'='*60}")

    # v4.6: 표본수 경고
    n_trades = metrics["total_trades"]
    if n_trades < MIN_TRADES_CRITICAL:
        log.warning(
            f"  [표본수 경고] {n_trades}건 < {MIN_TRADES_CRITICAL}건: "
            f"통계적 의미 없음. 아래 지표를 신뢰하지 마세요."
        )
    elif n_trades < MIN_TRADES_WARNING:
        log.warning(
            f"  [표본수 주의] {n_trades}건 < {MIN_TRADES_WARNING}건: "
            f"신뢰도 낮음. 추가 데이터 필요."
        )

    log.info(f"  총 거래: {n_trades}건")
    log.info(f"  승률: {metrics['win_rate']:.1f}%")
    log.info(f"  평균 손익비: 1:{metrics['avg_rr_ratio']:.2f}")
    log.info(f"  기대값: {metrics['expectancy']:.3f}%/거래")
    log.info(f"{'─'*60}")
    log.info(f"  총 수익률: {metrics['total_return_pct']:.1f}%")
    log.info(f"  CAGR: {metrics['cagr_pct']:.1f}%")
    log.info(f"  MDD: {metrics['max_drawdown_pct']:.1f}%")
    log.info(f"{'─'*60}")
    log.info(f"  Sharpe:  {metrics['sharpe_ratio']:.2f}")
    log.info(f"  Sortino: {metrics['sortino_ratio']:.3f}")
    log.info(f"  Calmar:  {metrics['calmar_ratio']:.3f}")
    log.info(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    log.info(f"{'─'*60}")

    # 트리거별
    tb = metrics.get("trigger_breakdown", {})
    if tb:
        log.info(f"  [트리거별]")
        for ttype, stats in tb.items():
            log.info(f"    {ttype:10s}: {stats['count']}건 "
                     f"승률={stats['win_rate']:.1f}% "
                     f"PF={stats['profit_factor']:.2f} "
                     f"E={stats['expectancy']:.2f}%")

    # 등급별
    gb = metrics.get("grade_breakdown", {})
    if gb:
        log.info(f"  [등급별]")
        for grade, stats in gb.items():
            log.info(f"    {grade}등급: {stats['count']}건 "
                     f"승률={stats['win_rate']:.1f}% "
                     f"PF={stats['profit_factor']:.2f}")

    # v4.6: 통계적 신뢰도 등급
    rel = metrics.get("statistical_reliability", {})
    if rel:
        log.info(f"{'─'*60}")
        grade_label = rel.get("grade", "?")
        log.info(f"  [통계적 신뢰도] {grade_label}등급 ({rel.get('message', '')})")

    log.info(f"{'='*60}")
