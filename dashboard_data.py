"""
=============================================================
  Quantum Master FLOWX — 데이터 커넥터 v2
  dashboard_data.py
=============================================================
  역할: 실제 시스템 데이터 파일들을 읽어 dashboard_state.json 생성
  실행: python dashboard_data.py              (1회 실행)
        python dashboard_data.py --upload     (Supabase 업로드 포함)
=============================================================
  데이터 소스 (모두 BAT-D/D2에서 자동 생성):
    Zone 1: brain_decision.json + regime_macro_signal.json + lens_context.json
    Zone 2: tomorrow_picks.json (ai_largecap + picks)
    Zone 3: paper_portfolio.json (통합 Paper Trading)
    Zone 4: sector_rotation/sector_momentum.json + group_relay
    Zone 5: krx_nationality/nationality_signal.json + sd_pattern_daily
    Zone 6: market_learning/signal_accuracy.json
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
    }


# ──────────────────────────────────────────────
# ZONE 2: 액션 (tomorrow_picks → 오늘 할 일)
# ──────────────────────────────────────────────

def build_zone2(max_items: int = 6) -> list[dict]:
    picks_data = _load_json("tomorrow_picks.json")
    if not picks_data:
        return []

    actions = []
    seen = set()

    # AI 대형주 (confidence 순)
    for item in picks_data.get("ai_largecap", [])[:3]:
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        conf = item.get("confidence", 0)
        grade = "AA" if conf >= 0.85 else "A" if conf >= 0.75 else "B"

        actions.append({
            "ticker": ticker,
            "name": item.get("name", ""),
            "action": "BUY" if conf >= 0.7 else "WATCH",
            "grade": grade,
            "score": round(conf * 100),
            "reason": item.get("reasoning", "")[:60],
            "strategy": "AI_BRAIN",
        })

    # 전략 종합 picks (score 순)
    picks_sorted = sorted(
        picks_data.get("picks", []),
        key=lambda x: x.get("total_score", 0),
        reverse=True,
    )
    for pick in picks_sorted[:5]:
        ticker = pick.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        grade_kr = pick.get("grade", "")
        score = pick.get("total_score", 0)
        close = pick.get("close", 0)

        if grade_kr == "적극매수":
            action, grade = "BUY", "AA"
        elif grade_kr == "매수":
            action, grade = "BUY", "A"
        elif score >= 45:
            action, grade = "WATCH", "B"
        else:
            action, grade = "WATCH", "C"

        reasons = pick.get("reasons", [])
        reason_str = reasons[0] if reasons else ""

        actions.append({
            "ticker": ticker,
            "name": pick.get("name", ""),
            "action": action,
            "grade": grade,
            "score": round(score),
            "close": close,
            "stop_loss": pick.get("stop_loss", 0),
            "target_price": pick.get("target_price", 0),
            "reason": reason_str[:60],
            "strategy": "SCAN",
        })

    return actions[:max_items]


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
        week_eq = equity_list[-5]["equity"]
        week_return = round((latest_equity / week_eq - 1) * 100, 2) if week_eq > 0 else 0
    if len(equity_list) >= 20:
        month_eq = equity_list[-20]["equity"]
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

    # 릴레이 fired 그룹
    fired_set = set()
    for g in relay.get("fired_groups", []):
        fired_set.add(g.get("group", ""))

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

        sectors.append({
            "name": name,
            "score": score,
            "ret_5d": ret_5,
            "rsi": rsi,
            "rank": rank,
            "signal": signal,
            "relay": relay_target,
        })

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

    # KOSPI 수급 총합 (etf_investor_flow에서)
    etf_flow = _load_json("etf_investor_flow.json")
    supply_summary = {}
    if isinstance(etf_flow, dict):
        supply_summary = {
            "foreign": etf_flow.get("foreign_total", 0),
            "inst": etf_flow.get("inst_total", 0),
            "indiv": etf_flow.get("indiv_total", 0),
        }

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
    print(f"  Zone1: {z1['verdict']} (현금{z1['cash_pct']}%, 레짐={z1['regime']})")
    print(f"  Zone2: {len(state['zone2'])}종목 액션")
    print(f"  Zone3: 자산 {z3['equity']:,}원 ({z3['total_return_pct']:+.1f}%) 보유 {len(z3['positions'])}종목")
    print(f"  Zone4: {len(state['zone4'])}섹터")
    print(f"  Zone5: SD {len(state['zone5'].get('sd_patterns', []))}종목")
    print(f"  Zone6: picks {z6['tomorrow_picks']}% whale {z6['whale_detect']}%")

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
