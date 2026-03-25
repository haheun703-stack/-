"""
=============================================================
  Quantum Master FLOWX — 데이터 커넥터 v3 (풀 공개)
  dashboard_data.py
=============================================================
  역할: 실제 시스템 데이터 파일들을 읽어 dashboard_state.json 생성
  실행: python dashboard_data.py              (1회 실행)
        python dashboard_data.py --upload     (Supabase 업로드 포함)
=============================================================
  데이터 소스 (모두 BAT-D/D2에서 자동 생성):
    Zone 1: brain_decision.json + regime_macro_signal.json + lens_context.json
    Zone 2: tomorrow_picks.json (ai_largecap + picks) — 기술지표 풀 공개
    Zone 3: paper_portfolio.json (통합 Paper Trading)
    Zone 4: sector_rotation/sector_momentum.json + etf_trading_signal + group_relay
    Zone 5: krx_nationality/nationality_signal.json + sd_pattern_daily
    Zone 6: market_learning/signal_accuracy.json
    Zone 7: etf_recommendations.json + etf_trading_signal.json — ETF 추천
=============================================================
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = PROJECT_ROOT / "website" / "data" / "dashboard_state.json"


def _load_json(path: str | Path, default=None):
    p = Path(path) if not isinstance(path, Path) else path
    if not p.is_absolute():
        p = DATA_DIR / p
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


# ──────────────────────────────────────────────
# FLOWX 상황별 메시지 생성기
# ──────────────────────────────────────────────

def _build_flowx_message(
    regime: str,
    shield_level: str,
    brain: dict,
    regime_data: dict,
    vix: float,
) -> dict:
    """
    3가지 상황별 FLOWX 메시지 생성.
    상황 1: 방어 (SHIELD RED/ORANGE 또는 BEAR/CRISIS)
    상황 2: 공격 (RECOVERY/BULL)
    상황 3: 관망 (그 외, 신호 충돌)
    """
    confidence = brain.get("confidence", 0.5)
    nw_score = brain.get("nightwatch_score", 0)
    fgn_days = brain.get("foreign_consecutive_buy_days", 0)

    # SHIELD RED 사유 추출
    shield_warnings = brain.get("shield_warnings", [])
    red_reason = shield_warnings[0] if shield_warnings else "리스크 상승"

    # 상황 1: 방어
    if shield_level in ("RED", "ORANGE") or regime in ("BEAR", "CRISIS", "PRE_CRISIS", "PANIC"):
        return {
            "situation": "방어",
            "headline": "지금은 지키는 게 버는 겁니다",
            "reason": f"SHIELD {shield_level}: {red_reason}. 레짐 {regime}. 신규 진입 없음.",
            "action_down": "보유 종목 손절선 엄수. 이탈 시 즉시 청산.",
            "action_up": "반등해도 추매 없음. 레짐 전환 확인 후 재진입.",
            "next_signal": "외국인 순매수 전환 + KOSPI RSI 45 돌파 시 재판단",
        }

    # 상황 2: 공격
    if regime in ("RECOVERY", "BULL", "PRE_BULL"):
        return {
            "situation": "공격",
            "headline": "상승 초입 — 올라탈 타이밍입니다",
            "reason": (
                f"레짐 {regime}. "
                f"외국인 {fgn_days}일 순매수. "
                f"VIX {vix:.1f}. NightWatch {nw_score:+.2f}."
            ),
            "action_up": "추매 조건: 20일선 위 + 거래량 1.5배 양봉 확인 시 25% 추가.",
            "action_partial": "+8% 도달 시 절반 익절, 나머지는 목표가 추적.",
            "action_down": "손절선 유지. -3% 이탈 시 전량 청산.",
        }

    # 상황 3: 관망
    bull_signals = 0
    bear_signals = 0
    if nw_score > 0.2:
        bull_signals += 1
    elif nw_score < -0.2:
        bear_signals += 1
    if vix < 20:
        bull_signals += 1
    elif vix > 25:
        bear_signals += 1
    if fgn_days > 3:
        bull_signals += 1
    elif fgn_days < -3:
        bear_signals += 1

    return {
        "situation": "관망",
        "headline": "방향이 불명확합니다 — 현금이 포지션입니다",
        "reason": (
            f"상승신호 {bull_signals}개 vs 하락신호 {bear_signals}개. "
            f"레짐 {regime}. 우위 없음."
        ),
        "action": "신규 진입 없음. 기존 포지션 손절선 유지.",
        "next_signal": "다음 판단: 오늘 14:30 프리클로즈 스캔",
    }


# ──────────────────────────────────────────────
# ZONE 1: 오늘의 판단 (BRAIN + LENS + REGIME)
# ──────────────────────────────────────────────

def build_zone1() -> dict:
    brain = _load_json("brain_decision.json")
    regime = _load_json("regime_macro_signal.json")
    lens = _load_json("lens_context.json")
    shield = _load_json("shield_report.json")

    # BRAIN 핵심
    effective_regime = brain.get("effective_regime", "UNKNOWN")
    vix = brain.get("vix_level", 0)
    invest_pct = brain.get("total_invest_pct", 0)
    cash_pct = round(100 - invest_pct)

    # 판단 매핑
    if effective_regime in ("PANIC", "CRISIS"):
        verdict = "회피"
    elif effective_regime in ("BEAR", "CAUTION"):
        verdict = "관망"
    elif effective_regime in ("NEUTRAL",):
        verdict = "중립"
    elif effective_regime in ("RECOVERY", "BULL"):
        verdict = "매수"
    else:
        verdict = "관망"

    # REGIME 전환 방향
    transition = regime.get("transition_direction", "")
    trans_prob = regime.get("transition_probability", 0)
    macro_grade = regime.get("macro_grade", "")

    # LENS 요약
    game_board = lens.get("game_board", {})
    lens_summary = game_board.get("summary", "") if isinstance(game_board, dict) else ""

    # SHIELD 상태
    shield_level = shield.get("overall_level", "NORMAL")

    # 9-ARM 점수 합산
    arms = brain.get("arms", [])
    arm_total = sum(a.get("adjusted_pct", 0) for a in arms) if isinstance(arms, list) else 0

    # KOSPI 데이터 (regime에서)
    kospi_close = regime.get("kospi_close", 0)
    kospi_chg = regime.get("kospi_change_pct", 0)

    # ── FLOWX 상황별 메시지 생성 ──
    flowx_message = _build_flowx_message(
        effective_regime, shield_level, brain, regime, vix,
    )

    return {
        "verdict": verdict,
        "cash_pct": cash_pct,
        "buy_pct": round(invest_pct),
        "regime": effective_regime,
        "regime_transition": f"{transition}" if transition else "",
        "transition_prob": round(trans_prob, 1),
        "macro_grade": macro_grade,
        "vix": round(vix, 1),
        "kospi": kospi_close,
        "kospi_chg": round(kospi_chg, 2),
        "brain_score": round(arm_total, 1),
        "shield_status": shield_level,
        "lens_summary": lens_summary[:100] if lens_summary else "",
        "updated_at": brain.get("timestamp", ""),
        "flowx_message": flowx_message,
    }


# ──────────────────────────────────────────────
# ZONE 2: 액션 (tomorrow_picks → 오늘 할 일)
# ──────────────────────────────────────────────

def build_zone2(max_items: int = 10) -> list[dict]:
    """종목 풀 공개 — 기술지표, 점수 분해, 매집단계, 안전마진, AI 판단 전부 노출"""
    picks_data = _load_json("tomorrow_picks.json")
    if not picks_data:
        return []

    actions = []
    seen = set()

    # picks에서 상세 데이터 인덱스 구축 (ticker → pick dict)
    picks_index = {}
    for pick in picks_data.get("picks", []):
        t = pick.get("ticker", "")
        if t:
            picks_index[t] = pick

    # AI 대형주 (confidence 순)
    for item in picks_data.get("ai_largecap", [])[:5]:
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        conf = item.get("confidence", 0)
        grade = "AA" if conf >= 0.85 else "A" if conf >= 0.75 else "B"

        entry = {
            "ticker": ticker,
            "name": item.get("name", ""),
            "action": "BUY" if conf >= 0.7 else "WATCH",
            "grade": grade,
            "score": round(conf * 100),
            "reason": item.get("reasoning", ""),
            "strategy": "AI_BRAIN",
            "ai_confidence": round(conf * 100),
            "ai_urgency": item.get("urgency", ""),
            "expected_impact": item.get("expected_impact_pct", 0),
        }

        # picks에 같은 종목이 있으면 기술지표 병합
        if ticker in picks_index:
            _merge_pick_detail(entry, picks_index[ticker])

        actions.append(entry)

    # 전략 종합 picks (score 순)
    picks_sorted = sorted(
        picks_data.get("picks", []),
        key=lambda x: x.get("total_score", 0),
        reverse=True,
    )
    for pick in picks_sorted[:8]:
        ticker = pick.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        grade_kr = pick.get("grade", "")
        score = pick.get("total_score", 0)

        if grade_kr == "적극매수":
            action, grade = "BUY", "AA"
        elif grade_kr == "매수":
            action, grade = "BUY", "A"
        elif score >= 45:
            action, grade = "WATCH", "B"
        else:
            action, grade = "WATCH", "C"

        reasons = pick.get("reasons", [])

        entry = {
            "ticker": ticker,
            "name": pick.get("name", ""),
            "action": action,
            "grade": grade,
            "score": round(score),
            "reason": " · ".join(reasons[:4]),
            "strategy": "SCAN",
        }
        _merge_pick_detail(entry, pick)
        actions.append(entry)

    return actions[:max_items]


def _merge_pick_detail(entry: dict, pick: dict):
    """picks 상세 데이터를 entry에 병합 — 기술지표 풀 공개"""
    # 가격 정보
    entry["close"] = pick.get("close", 0)
    entry["price_change"] = round(pick.get("price_change", 0), 2)
    entry["stop_loss"] = pick.get("stop_loss", 0)
    entry["target_price"] = pick.get("target_price", 0)
    entry["entry_price"] = pick.get("entry_price", 0)
    entry["entry_condition"] = pick.get("entry_condition", "")

    # 기술지표
    entry["rsi"] = round(pick.get("rsi", 0), 1)
    entry["adx"] = round(pick.get("adx", 0), 1)
    entry["stoch_k"] = round(pick.get("stoch_k", 0), 1)
    entry["stoch_d"] = round(pick.get("stoch_d", 0), 1) if pick.get("stoch_d") is not None else None
    entry["bb_position"] = round(pick.get("bb_position", 0), 2)
    entry["above_ma20"] = pick.get("above_ma20", False)
    entry["above_ma60"] = pick.get("above_ma60", False)
    entry["ma5_gap"] = round(pick.get("ma5_gap_pct", 0), 1)
    entry["sar_trend"] = pick.get("sar_trend", 0)

    # 수급
    entry["foreign_5d"] = pick.get("foreign_5d", 0)
    entry["inst_5d"] = pick.get("inst_5d", 0)

    # 매집 단계
    entry["accum_phase"] = pick.get("accum_phase", "")
    entry["accum_days"] = pick.get("accum_days", 0)
    entry["accum_return"] = round(pick.get("accum_return", 0), 1)

    # 안전마진
    entry["safety_signal"] = pick.get("safety_signal", "")
    entry["safety_label"] = pick.get("safety_label", "")

    # 점수 분해
    sb = pick.get("score_breakdown", {})
    if sb:
        entry["score_breakdown"] = {
            "multi": sb.get("multi", 0),
            "individual": round(sb.get("individual", 0), 1),
            "tech": round(sb.get("tech", 0), 1),
            "flow": round(sb.get("flow", 0), 1),
            "safety": round(sb.get("safety", 0), 1),
            "overheat": round(sb.get("overheat", 0), 1),
        }

    # AI 판단 (picks 내장)
    if pick.get("ai_action"):
        entry["ai_action"] = pick["ai_action"]
        entry["ai_tag"] = pick.get("ai_tag", "")
    if pick.get("ai_bonus"):
        entry["ai_bonus"] = round(pick["ai_bonus"], 1)

    # 과열 경고
    entry["overheat_flags"] = pick.get("overheat_flags", [])

    # 52주 낙폭
    entry["drawdown"] = round(pick.get("drawdown", 0), 1)

    # 컨센서스
    entry["consensus_upside"] = round(pick.get("consensus_upside", 0), 1)

    # 전략 타입
    if pick.get("strategy"):
        entry["trade_strategy"] = pick["strategy"]


# ──────────────────────────────────────────────
# ZONE 3: 성과 (Paper Trading 포트폴리오)
# ──────────────────────────────────────────────

def build_zone3() -> dict:
    pf = _load_json("paper_portfolio.json")
    if not pf:
        return {"equity": 0, "total_return_pct": 0, "positions": []}

    initial = pf.get("initial_capital", 30_000_000)
    equity_list = pf.get("daily_equity", [])
    latest_equity = equity_list[-1]["equity"] if equity_list else initial

    total_return = round((latest_equity / initial - 1) * 100, 2) if initial > 0 else 0

    # 주간/월간 수익
    week_return = 0
    month_return = 0
    if len(equity_list) >= 5:
        week_eq = equity_list[-min(5, len(equity_list))]["equity"]
        week_return = round((latest_equity / week_eq - 1) * 100, 2) if week_eq > 0 else 0
    if len(equity_list) >= 20:
        month_eq = equity_list[-min(20, len(equity_list))]["equity"]
        month_return = round((latest_equity / month_eq - 1) * 100, 2) if month_eq > 0 else 0

    # 통계
    stats = pf.get("stats", {})
    total_trades = stats.get("total_trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0
    mdd = stats.get("mdd", 0)

    # PF (Profit Factor)
    closed = pf.get("closed_trades", [])
    gross_profit = sum(t["pnl_pct"] for t in closed if t.get("pnl_pct", 0) > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in closed if t.get("pnl_pct", 0) < 0))
    pf_ratio = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

    # 보유 포지션
    positions = []
    for ticker, pos in pf.get("positions", {}).items():
        # processed parquet에서 현재가
        cur_price = 0
        pq_path = DATA_DIR / "processed" / f"{ticker}.parquet"
        if pq_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(pq_path)
                if len(df) > 0:
                    cur_price = float(df.iloc[-1]["close"])
            except Exception:
                pass

        avg_price = pos.get("avg_price", 0)
        pnl_pct = round((cur_price / avg_price - 1) * 100, 2) if avg_price > 0 and cur_price > 0 else 0

        entry_date = pos.get("entry_date", "")
        days_held = 0
        if entry_date:
            try:
                from datetime import datetime as dt
                days_held = (dt.now() - dt.strptime(entry_date, "%Y-%m-%d")).days
            except Exception:
                pass

        positions.append({
            "ticker": ticker,
            "name": pos.get("name", ""),
            "pnl_pct": pnl_pct,
            "days": days_held,
            "strategy": pos.get("strategy", ""),
            "grade": pos.get("grade", ""),
        })

    # 최근 청산 거래 (최근 5건)
    recent_trades = closed[-5:] if closed else []

    return {
        "equity": latest_equity,
        "initial_capital": initial,
        "total_return_pct": total_return,
        "week_return_pct": week_return,
        "month_return_pct": month_return,
        "win_rate": win_rate,
        "pf": pf_ratio,
        "mdd": mdd,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "positions": positions,
        "recent_trades": recent_trades,
    }


# ──────────────────────────────────────────────
# ZONE 4: 섹터 흐름 + 릴레이
# ──────────────────────────────────────────────

def build_zone4(top_n: int = 10) -> list[dict]:
    sm = _load_json("sector_rotation/sector_momentum.json")
    relay = _load_json("group_relay/group_relay_today.json")
    etf_sig = _load_json("sector_rotation/etf_trading_signal.json")

    # 릴레이 fired 그룹
    fired_set = set()
    for g in relay.get("fired_groups", []):
        fired_set.add(g.get("group", ""))

    # ETF 시그널 인덱스 (섹터명 → 상세)
    etf_index = {}
    for item in etf_sig.get("watch_list", []) + etf_sig.get("smart_sectors", []) + etf_sig.get("smart_money_etf", []):
        s = item.get("sector", "")
        if s:
            etf_index[s] = item

    sectors = []
    for sec in sm.get("sectors", [])[:top_n]:
        name = sec.get("sector", "")
        score = round(sec.get("momentum_score", 0), 1)
        ret_5 = round(sec.get("ret_5", 0), 2)
        rsi = round(sec.get("rsi_14", 0), 1)
        rank = sec.get("rank", 0)

        # 시그널 판단
        if score >= 70:
            signal = "매수"
        elif score >= 50:
            signal = "관찰"
        elif score >= 30:
            signal = "주의"
        else:
            signal = "회피"

        # 릴레이 연결
        relay_target = None
        if name in fired_set:
            relay_target = "FIRE"

        row = {
            "name": name,
            "score": score,
            "ret_5d": ret_5,
            "rsi": rsi,
            "rank": rank,
            "signal": signal,
            "relay": relay_target,
        }

        # ETF 코드 연결
        if name in etf_index:
            ei = etf_index[name]
            row["etf_code"] = ei.get("etf_code", "")
            row["etf_signal"] = ei.get("signal", "")
            row["etf_sizing"] = ei.get("sizing", "")
            row["bb_pct"] = round(ei.get("bb_pct", 0), 1)
            row["stoch_k"] = round(ei.get("stoch_k", 0), 1)
            row["adx"] = round(ei.get("adx", 0), 1)

        sectors.append(row)

    return sectors


# ──────────────────────────────────────────────
# ZONE 5: 수급 레이더 (외인/기관 + SD V2 패턴)
# ──────────────────────────────────────────────

def build_zone5() -> dict:
    # 외국인 자금 흐름
    nat_signal = _load_json("krx_nationality/nationality_signal.json")
    foreign_flow = []
    for sig in nat_signal.get("signals", [])[:5]:
        foreign_flow.append({
            "ticker": sig.get("ticker", ""),
            "name": sig.get("name", ""),
            "direction": sig.get("foreign_direction", "NEUTRAL"),
            "score": sig.get("score", 0),
            "z_score": round(sig.get("inst_zscore", 0), 2),
        })

    # SD V2 패턴 (최신)
    sd_dir = DATA_DIR / "sd_pattern_daily"
    sd_patterns = []
    if sd_dir.exists():
        sd_files = sorted(sd_dir.glob("*.json"), reverse=True)
        if sd_files:
            sd_data = _load_json(sd_files[0])
            # 상위 패턴 종목
            if isinstance(sd_data, dict):
                items = sorted(
                    sd_data.items(),
                    key=lambda x: abs(x[1].get("sd_score", 0)) if isinstance(x[1], dict) else 0,
                    reverse=True,
                )[:6]
                for ticker, info in items:
                    if not isinstance(info, dict):
                        continue
                    sd_score = info.get("sd_score", 0)
                    pattern_name = info.get("pattern_name", "")
                    if sd_score > 0.3:
                        grade, pat_type = "A", "매집"
                    elif sd_score > 0:
                        grade, pat_type = "B", "관찰"
                    elif sd_score > -0.3:
                        grade, pat_type = "C", "관찰"
                    else:
                        grade, pat_type = "F", "분산"
                    sd_patterns.append({
                        "ticker": ticker,
                        "name": info.get("name", ticker),
                        "grade": grade,
                        "pattern": pat_type,
                        "pattern_name": pattern_name,
                        "sd_score": round(sd_score, 3),
                    })

    # KOSPI 수급 총합 (etf_investor_flow에서 최근일 합산)
    etf_flow = _load_json("etf_investor_flow.json")
    supply_summary = {"foreign": 0, "inst": 0, "indiv": 0}
    if isinstance(etf_flow, dict):
        for _ticker, info in etf_flow.get("etfs", {}).items():
            days = info.get("days", [])
            if days:
                last = days[-1]
                supply_summary["foreign"] += last.get("foreign_net", 0)
                supply_summary["inst"] += last.get("inst_net", 0)
                supply_summary["indiv"] += last.get("individual_net", 0)

    return {
        "foreign_flow": foreign_flow,
        "sd_patterns": sd_patterns,
        "supply_summary": supply_summary,
    }


# ──────────────────────────────────────────────
# ZONE 6: 시스템 신뢰도 (학습 데이터 기반)
# ──────────────────────────────────────────────

def build_zone6() -> dict:
    sig_acc = _load_json("market_learning/signal_accuracy.json")
    signals = sig_acc.get("signals", {})

    def get_rate(name):
        s = signals.get(name, {})
        return round(s.get("hit_rate", 0), 1)

    # 최근 10건 적중 (daily_log에서)
    daily_log = sig_acc.get("daily_log", [])
    recent_10 = []
    for entry in daily_log[-10:]:
        if isinstance(entry, dict):
            recent_10.append(1 if entry.get("hit", False) else 0)
        else:
            recent_10.append(0)

    # 패딩
    while len(recent_10) < 10:
        recent_10.insert(0, 0)

    # 활성 시그널 수
    active_count = sum(1 for s in signals.values() if s.get("total", 0) > 0)

    return {
        "tomorrow_picks": get_rate("tomorrow_picks"),
        "whale_detect": get_rate("whale_detect"),
        "volume_spike": get_rate("volume_spike") or get_rate("pullback_scan"),
        "brain": get_rate("accumulation_tracker"),
        "recent_10": recent_10,
        "active_signals": active_count,
    }


# ──────────────────────────────────────────────
# ZONE 7: ETF 추천 (매크로 + 섹터 + 테마)
# ──────────────────────────────────────────────

def build_zone7() -> list[dict]:
    """ETF 추천 풀 공개 — 카테고리, 신뢰도, 진입타이밍, 손절, 비중"""
    recs = _load_json("etf_recommendations.json")
    if not recs:
        return []

    etfs = []

    # 시장 방향
    mkt = recs.get("market_direction", {})

    # ETF 추천
    for pick in recs.get("etf_picks", []):
        reasons = pick.get("reason", [])
        if isinstance(reasons, str):
            reasons = [reasons]

        etfs.append({
            "category": pick.get("category", ""),
            "ticker": pick.get("ticker", ""),
            "name": pick.get("name", ""),
            "action": pick.get("action", "WATCH"),
            "confidence": round(pick.get("confidence", 0) * 100),
            "holding_period": pick.get("holding_period", ""),
            "portfolio_pct": pick.get("portfolio_pct", 0),
            "stop_loss": pick.get("stop_loss", ""),
            "target": pick.get("target", ""),
            "entry_timing": pick.get("entry_timing", ""),
            "reasons": reasons[:3],
        })

    # 핫 섹터 ETF (중복 제거)
    seen_tickers = {e["ticker"] for e in etfs}
    for hs in recs.get("hot_sectors", []):
        etf_info = hs.get("etf", {})
        ticker = etf_info.get("ticker", "")
        if ticker and ticker not in seen_tickers:
            seen_tickers.add(ticker)
            reasons = hs.get("reasons", [])
            etfs.append({
                "category": "섹터",
                "ticker": ticker,
                "name": etf_info.get("name", ""),
                "action": "WATCH",
                "confidence": round(hs.get("score", 0)),
                "holding_period": "",
                "portfolio_pct": 0,
                "stop_loss": "",
                "target": "",
                "entry_timing": "",
                "reasons": reasons[:3],
            })

    # 시장 방향 정보 첨부
    if etfs and mkt:
        etfs.insert(0, {
            "_market_direction": mkt.get("direction", ""),
            "_market_score": round(mkt.get("score", 0), 2),
            "_market_confidence": round(mkt.get("confidence", 0) * 100),
            "_regime": mkt.get("regime", ""),
            "_vix": mkt.get("vix_level", 0),
            "_reasons": mkt.get("reasons", [])[:3],
        })

    return etfs


# ──────────────────────────────────────────────
# ZONE 8: Market Pulse (실시간 시장 인텔리전스)
# ──────────────────────────────────────────────

def build_zone8() -> dict:
    """Market Pulse 데이터 (market_pulse.json) 로드."""
    pulse = _load_json("market_pulse.json")
    if not pulse:
        return {"status": "no_data", "message": "Market Pulse 미실행"}

    mkt = pulse.get("market_direction", {})
    picks = pulse.get("top_picks", [])
    etf_recs = pulse.get("etf_recommendations", [])
    sectors = pulse.get("sector_rankings", {})

    # 섹터 랭킹 정렬
    sector_ranking = sorted(
        [{"sector": k, **v} for k, v in sectors.items()],
        key=lambda x: x.get("avg_change_pct", 0),
        reverse=True,
    )

    return {
        "timestamp": pulse.get("timestamp", ""),
        "direction": mkt.get("direction", "UNKNOWN"),
        "direction_comment": mkt.get("comment", ""),
        "tomorrow": mkt.get("tomorrow", ""),
        "etf_call": mkt.get("etf_call", ""),
        "kospi_lever_pct": mkt.get("kospi_lever_pct", 0),
        "etf_recommendations": [
            {
                "ticker": e.get("ticker", ""),
                "name": e.get("name", ""),
                "action": e.get("action", ""),
                "reason": e.get("reason", ""),
            }
            for e in etf_recs[:5]
        ],
        "sector_ranking": [
            {
                "sector": s["sector"],
                "change_pct": s.get("avg_change_pct", 0),
                "leader": s.get("leader", ""),
                "leader_change": s.get("leader_change", 0),
            }
            for s in sector_ranking[:5]
        ],
        "top_picks": [
            {
                "rank": p.get("rank", 0),
                "ticker": p.get("ticker", ""),
                "name": p.get("name", ""),
                "sector": p.get("sector", ""),
                "price": p.get("price", 0),
                "change_pct": p.get("change_pct", 0),
                "vwap_est": p.get("vwap_est", 0),
                "entry_price": p.get("entry_price", 0),
                "target_price": p.get("target_price", 0),
                "stop_price": p.get("stop_price", 0),
                "risk_reward": p.get("risk_reward", 0),
                "position_comment": p.get("position_comment", ""),
                "supply_comment": p.get("supply_comment", ""),
            }
            for p in picks[:5]
        ],
    }


# ──────────────────────────────────────────────
# 통합 빌드 & 저장
# ──────────────────────────────────────────────

def build_state() -> dict:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 대시보드 데이터 빌드 중...")

    state = {
        "generated_at": datetime.now().isoformat(),
        "zone1": build_zone1(),
        "zone2": build_zone2(),
        "zone3": build_zone3(),
        "zone4": build_zone4(),
        "zone5": build_zone5(),
        "zone6": build_zone6(),
        "zone7": build_zone7(),
        "zone8": build_zone8(),
    }

    # 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[완료] {OUTPUT_FILE} 저장")

    # 요약 출력
    z1 = state["zone1"]
    z3 = state["zone3"]
    z6 = state["zone6"]
    z7_count = len([e for e in state["zone7"] if "ticker" in e])
    z8 = state["zone8"]
    print(f"  Zone1: {z1['verdict']} (현금{z1['cash_pct']}%, 레짐={z1['regime']})")
    print(f"  Zone2: {len(state['zone2'])}종목 풀 공개")
    print(f"  Zone3: 자산 {z3['equity']:,}원 ({z3['total_return_pct']:+.1f}%) 보유 {len(z3['positions'])}종목")
    print(f"  Zone4: {len(state['zone4'])}섹터")
    print(f"  Zone5: SD {len(state['zone5'].get('sd_patterns', []))}종목")
    print(f"  Zone6: picks {z6['tomorrow_picks']}% whale {z6['whale_detect']}%")
    print(f"  Zone7: ETF {z7_count}종목")
    print(f"  Zone8: Pulse {z8.get('direction', '?')} TOP {len(z8.get('top_picks', []))}종목")

    return state


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLOWX 대시보드 데이터 빌드")
    parser.add_argument("--upload", action="store_true", help="Supabase 업로드 포함")
    args = parser.parse_args()

    state = build_state()

    if args.upload:
        try:
            from src.adapters.flowx_uploader import FlowxUploader
            uploader = FlowxUploader()
            if uploader.is_active:
                # dashboard_state를 Supabase에 업로드 (테이블 추가 필요)
                print("[FLOWX] Supabase 업로드는 웹봇 테이블 설정 후 활성화")
            else:
                print("[FLOWX] Supabase 미연결")
        except Exception as e:
            print(f"[FLOWX] 업로드 실패: {e}")
