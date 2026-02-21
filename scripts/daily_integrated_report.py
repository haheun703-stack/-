"""통합 데일리 리포트 — 6대 시그널 통합 + HTML/PNG + 텔레그램 발송.

6대 시그널:
  1. US Overnight Signal (5등급)
  2. KOSPI 레짐 (4단계)
  3. Quantum v10 매수 후보 (Kill→Rank→Tag)
  4. 섹터 릴레이 (발화→후행 섹터)
  5. 그룹 릴레이 (대장주→계열사, 참고)
  6. ETF 매매 시그널 (Smart Money + Theme)

사용법:
  python scripts/daily_integrated_report.py              # 전체 실행 + 텔레그램
  python scripts/daily_integrated_report.py --no-send    # 보고서만 생성
  python scripts/daily_integrated_report.py --no-scan    # Quantum 스캔 건너뜀
  python scripts/daily_integrated_report.py --text-only  # 텍스트만 (HTML 생략)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PORTFOLIO = 30_000_000
SCAN_CACHE = PROJECT_ROOT / "data" / "scan_cache.json"


# ── 시장 스탠스 매트릭스 ──
STANCE_MATRIX = {
    ("STRONG_BULL", "BULL"): "적극 매수",
    ("STRONG_BULL", "CAUTION"): "선별 매수",
    ("MILD_BULL", "BULL"): "선별 매수",
    ("MILD_BULL", "CAUTION"): "선별 매수",
    ("MILD_BULL", "BEAR"): "관망",
    ("NEUTRAL", "BULL"): "관망(소량)",
    ("NEUTRAL", "CAUTION"): "관망",
    ("NEUTRAL", "BEAR"): "매수 자제",
    ("MILD_BEAR", "BULL"): "관망",
    ("MILD_BEAR", "CAUTION"): "매수 자제",
    ("MILD_BEAR", "BEAR"): "매수 자제",
    ("MILD_BEAR", "CRISIS"): "매수 금지",
    ("STRONG_BEAR", "BULL"): "매수 자제",
    ("STRONG_BEAR", "CAUTION"): "매수 금지",
    ("STRONG_BEAR", "BEAR"): "매수 금지",
    ("STRONG_BEAR", "CRISIS"): "전량 청산 검토",
}


# ═══════════════════════════════════════════════
# 1. 시그널 수집
# ═══════════════════════════════════════════════

def collect_us_overnight() -> dict:
    """US Overnight Signal JSON 로드."""
    signal_path = PROJECT_ROOT / "data" / "us_market" / "overnight_signal.json"
    if not signal_path.exists():
        return {"grade": "NEUTRAL", "combined_score_100": 0, "composite": "neutral"}
    try:
        with open(signal_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"grade": "NEUTRAL", "combined_score_100": 0, "composite": "neutral"}


def collect_kospi_regime() -> dict:
    """KOSPI 레짐 판정 (scan_buy_candidates.get_kospi_regime 로직 재구현)."""
    kospi_path = PROJECT_ROOT / "data" / "kospi_index.csv"
    if not kospi_path.exists():
        return {"regime": "CAUTION", "slots": 3, "close": 0, "ma20": 0, "ma60": 0, "rv_pct": 0.5}

    df = pd.read_csv(kospi_path, index_col="Date", parse_dates=True).sort_index()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["rv20_pct"] = df["rv20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    if len(df) < 60:
        return {"regime": "CAUTION", "slots": 3, "close": 0, "ma20": 0, "ma60": 0, "rv_pct": 0.5}

    row = df.iloc[-1]
    close = float(row["close"])
    ma20 = float(row["ma20"]) if not pd.isna(row["ma20"]) else 0
    ma60 = float(row["ma60"]) if not pd.isna(row["ma60"]) else 0
    rv_pct = float(row.get("rv20_pct", 0.5)) if not pd.isna(row.get("rv20_pct", 0.5)) else 0.5

    if ma20 == 0 or ma60 == 0:
        regime, slots = "CAUTION", 3
    elif close > ma20:
        regime, slots = ("BULL", 5) if rv_pct < 0.50 else ("CAUTION", 3)
    elif close > ma60:
        regime, slots = "BEAR", 2
    else:
        regime, slots = "CRISIS", 0

    return {"regime": regime, "slots": slots, "close": close, "ma20": ma20, "ma60": ma60, "rv_pct": rv_pct}


def collect_quantum_candidates(
    run_scan: bool = True,
    grade_filter: str = "AB",
    use_news: bool = False,
) -> tuple[list[dict], dict]:
    """Quantum v10 매수 후보 수집."""
    if not run_scan:
        return _load_scan_cache()
    try:
        # scan_buy_candidates는 stdout 래핑이 있어서 별도 프로세스에서 실행하거나
        # 직접 import 시 주의 필요 (sys.stdout 충돌)
        import importlib
        mod = importlib.import_module("scan_buy_candidates")
        candidates, stats = mod.scan_all(
            grade_filter=grade_filter,
            use_news=use_news,
            universe="parquet",
        )
        _save_scan_cache(candidates, stats)
        return candidates, stats
    except Exception as e:
        print(f"  Quantum 스캔 실패: {e}")
        import traceback
        traceback.print_exc()
        return _load_scan_cache()


def _load_scan_cache() -> tuple[list[dict], dict]:
    """스캔 캐시 로드."""
    if SCAN_CACHE.exists():
        try:
            data = json.loads(SCAN_CACHE.read_text(encoding="utf-8"))
            return data.get("candidates", []), data.get("stats", {})
        except Exception:
            pass
    return [], {}


def _save_scan_cache(candidates: list[dict], stats: dict):
    """스캔 결과 캐시 저장."""
    # 직렬화 가능한 필드만 추출
    safe_candidates = []
    for c in candidates:
        safe = {}
        for k, v in c.items():
            if isinstance(v, (str, int, float, bool, type(None), list)):
                safe[k] = v
            elif isinstance(v, np.floating):
                safe[k] = float(v)
            elif isinstance(v, np.integer):
                safe[k] = int(v)
        safe_candidates.append(safe)
    try:
        SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCAN_CACHE, "w", encoding="utf-8") as f:
            json.dump({"candidates": safe_candidates, "stats": stats,
                        "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M")},
                       f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def collect_relay_signals() -> dict:
    """릴레이 시그널 수집."""
    try:
        from relay_report import generate_relay_report
        return generate_relay_report(portfolio=PORTFOLIO, top_picks=3)
    except Exception as e:
        print(f"  릴레이 스캔 실패: {e}")
        return {"fired_sectors": [], "relay_signals": []}


def collect_group_relay_signals() -> dict:
    """그룹 릴레이 시그널 수집 (참고 정보)."""
    try:
        from group_relay_detector import generate_group_relay_report
        return generate_group_relay_report(fire_threshold=3.0)
    except Exception as e:
        print(f"  그룹 릴레이 스캔 실패: {e}")
        return {"fired_groups": [], "no_fire": True, "summary": "스캔 실패"}


def collect_etf_signals() -> dict:
    """ETF 매매 시그널 수집 (etf_trading_signal.json 로드)."""
    signal_path = PROJECT_ROOT / "data" / "sector_rotation" / "etf_trading_signal.json"
    default = {"smart_money_etf": [], "theme_money_etf": [], "watch_list": [],
               "smart_sectors": [], "summary": {}}
    if not signal_path.exists():
        return default
    try:
        with open(signal_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ETF 시그널 로드 실패: {e}")
        return default


def _load_quantum_positions() -> dict:
    """Quantum positions.json 직접 로드."""
    pos_file = PROJECT_ROOT / "data" / "positions.json"
    default = {"capital": 100_000_000, "positions": []}
    if not pos_file.exists():
        return default
    try:
        data = json.loads(pos_file.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "positions" in data:
            return data
        return default
    except Exception:
        return default


def collect_positions() -> dict:
    """Quantum + Relay 보유 포지션 통합 수집."""
    # Quantum
    quantum_data = _load_quantum_positions()
    quantum_positions = quantum_data.get("positions", [])

    # Relay
    from relay_positions import (
        load_positions as load_relay_pos,
        get_current_price,
    )
    relay_positions = load_relay_pos()

    # Relay 현재가 + 수익률 계산
    relay_details = []
    for pos in relay_positions:
        cur = get_current_price(pos["ticker"])
        pnl = (cur - pos["entry_price"]) / pos["entry_price"] * 100 if cur > 0 and pos["entry_price"] > 0 else 0
        relay_details.append({
            **pos,
            "current_price": cur,
            "pnl_pct": round(pnl, 2),
        })

    # 합산
    total_quantum_invested = sum(p.get("amount", 0) for p in quantum_positions)
    total_relay_invested = sum(p.get("investment", 0) for p in relay_positions)
    total_relay_pnl = sum(
        (d["current_price"] - d["entry_price"]) * d["quantity"]
        for d in relay_details if d["current_price"] > 0
    )

    return {
        "quantum": {
            "count": len(quantum_positions),
            "positions": quantum_positions,
            "capital": quantum_data.get("capital", 100_000_000),
        },
        "relay": {
            "count": len(relay_positions),
            "positions": relay_details,
        },
        "total_count": len(quantum_positions) + len(relay_positions),
        "total_invested": total_quantum_invested + total_relay_invested,
        "total_relay_pnl": int(total_relay_pnl),
    }


def collect_all_signals(
    run_scan: bool = True,
    grade_filter: str = "AB",
    use_news: bool = False,
) -> dict:
    """6개 시그널 시스템 데이터 통합 수집."""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'=' * 60}")
    print(f"  통합 데일리 리포트 — {today}")
    print(f"{'=' * 60}")

    # 1. US Overnight
    print("\n[1/6] US Overnight Signal...")
    us_data = collect_us_overnight()
    us_grade = us_data.get("grade", "NEUTRAL")
    us_score = us_data.get("combined_score_100", 0)
    print(f"  {us_grade} ({us_score:+.1f})")

    # 2. KOSPI 레짐
    print("\n[2/6] KOSPI 레짐...")
    kospi_data = collect_kospi_regime()
    print(f"  {kospi_data['regime']} ({kospi_data['slots']}슬롯)")

    # 3. Quantum 스캔
    print(f"\n[3/6] Quantum v10 {'스캔' if run_scan else '캐시 로드'}...")
    candidates, stats = collect_quantum_candidates(run_scan, grade_filter, use_news)
    print(f"  {len(candidates)}종목 통과")

    # 4. 섹터 릴레이
    print("\n[4/6] 섹터 릴레이 시그널...")
    relay_data = collect_relay_signals()
    fired_count = len(relay_data.get("fired_sectors", []))
    signal_count = len(relay_data.get("relay_signals", []))
    print(f"  발화 {fired_count}개, 시그널 {signal_count}개")

    # 5. 그룹 릴레이
    print("\n[5/6] 그룹 릴레이 시그널...")
    group_relay_data = collect_group_relay_signals()
    gr_fired = len(group_relay_data.get("fired_groups", []))
    print(f"  발화 {gr_fired}개 그룹")

    # 6. ETF 시그널
    print("\n[6/6] ETF 시그널...")
    etf_data = collect_etf_signals()
    etf_summary = etf_data.get("summary", {})
    smart_buy = etf_summary.get("smart_buy", 0)
    theme_buy = etf_summary.get("theme_buy", 0)
    watch = etf_summary.get("watch", 0)
    print(f"  SMART {smart_buy}개, THEME {theme_buy}개, 관찰 {watch}개")

    # 포지션
    print("\n포지션 로드...")
    positions = collect_positions()
    print(f"  Quantum {positions['quantum']['count']}건, Relay {positions['relay']['count']}건")

    return {
        "date": today,
        "generated_at": now,
        "us_overnight": us_data,
        "kospi_regime": kospi_data,
        "quantum": {"candidates": candidates, "stats": stats},
        "relay": relay_data,
        "group_relay": group_relay_data,
        "etf": etf_data,
        "positions": positions,
    }


# ═══════════════════════════════════════════════
# 2. 액션 플랜 빌드
# ═══════════════════════════════════════════════

def build_action_plan(data: dict) -> dict:
    """통합 데이터에서 액션 플랜 추출."""
    us_grade = data["us_overnight"].get("grade", "NEUTRAL")
    kospi_regime = data["kospi_regime"].get("regime", "CAUTION")

    market_stance = STANCE_MATRIX.get((us_grade, kospi_regime), "관망")
    can_buy = "매수" in market_stance or "소량" in market_stance

    buys = []
    watches = []
    sells = []

    # Quantum 후보
    for c in data["quantum"]["candidates"][:5]:
        grade = c.get("grade", "D")
        entry = {
            "source": "Quantum",
            "name": c.get("name", ""),
            "ticker": c.get("ticker", ""),
            "grade": grade,
            "entry_price": c.get("entry_price", 0),
            "target_price": c.get("target_price", 0),
            "stop_loss": c.get("stop_loss", 0),
            "risk_reward": c.get("risk_reward", 0),
        }
        if grade in ("S", "A") and can_buy:
            buys.append(entry)
        else:
            entry["reason"] = "등급 부족" if grade not in ("S", "A") else f"시장: {market_stance}"
            watches.append(entry)

    # Relay 후보 — 후행 섹터 대기중 + 신뢰도 HIGH만
    for sig in data["relay"].get("relay_signals", []):
        follow_ret = sig.get("follow_stats", {}).get("avg_return", 0)
        conf = sig["pattern"]["confidence"]

        # 이미 움직인 후행 섹터 제외
        if follow_ret >= 3.0:
            continue

        if conf == "HIGH" and can_buy:
            for pick in sig.get("picks", [])[:2]:
                buys.append({
                    "source": "Relay",
                    "name": pick.get("name", ""),
                    "ticker": pick.get("ticker", ""),
                    "sector": sig["follow_sector"],
                    "fired_sector": sig["lead_sector"],
                    "lead_return": sig.get("lead_return", 0),
                    "win_rate": sig["pattern"]["win_rate"],
                    "weight_pct": sig["sizing"]["weight_pct"],
                })
        elif conf in ("HIGH", "MED"):
            watches.append({
                "source": "Relay",
                "name": f"{sig['lead_sector']}→{sig['follow_sector']}",
                "ticker": "",
                "reason": f"신뢰도 {conf}" if conf != "HIGH" else f"시장: {market_stance}",
            })

    # Relay 매도 대상 — 현재가 기반 체크
    for pos in data["positions"]["relay"]["positions"]:
        pnl = pos.get("pnl_pct", 0)
        target = pos.get("target_pct", 6.0)
        days = pos.get("trading_days_held", 0)
        timeout = pos.get("timeout_days", 2)

        reason = None
        if pnl >= target:
            reason = f"목표 수익 도달 ({pnl:+.1f}%)"
        elif pnl <= -5.0:
            reason = f"손절 ({pnl:+.1f}%)"
        elif days >= timeout and pnl <= 0:
            reason = f"래그 타임아웃 ({days}일)"

        if reason:
            sells.append({
                "source": "Relay",
                "name": pos.get("name", ""),
                "ticker": pos.get("ticker", ""),
                "pnl_pct": pnl,
                "reason": reason,
            })

    return {
        "market_stance": market_stance,
        "us_grade": us_grade,
        "kospi_regime": kospi_regime,
        "buys": buys,
        "sells": sells,
        "watches": watches,
    }


# ═══════════════════════════════════════════════
# 3. 텍스트 메시지 포맷
# ═══════════════════════════════════════════════

def format_text_message(data: dict) -> str:
    """텔레그램 텍스트 메시지 포맷."""
    plan = data["action_plan"]
    us = data["us_overnight"]
    kospi = data["kospi_regime"]
    positions = data["positions"]

    lines = []
    lines.append(f"[Quantum Master] {data['date']} {data['generated_at'].split(' ')[-1]}")
    lines.append("-" * 30)

    # ── 시장 온도 ──
    us_grade = us.get("grade", "NEUTRAL")
    us_score = us.get("combined_score_100", 0)
    idx = us.get("index_direction", {})
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)
    vix = us.get("vix", {})
    vix_level = vix.get("level", 0)
    vix_status = vix.get("status", "")

    regime = kospi["regime"]
    slots = kospi["slots"]
    close = kospi.get("close", 0)
    ma20 = kospi.get("ma20", 0)
    ma60 = kospi.get("ma60", 0)

    lines.append(f"US: {us_grade}({us_score:+.1f}) | EWY{ewy_ret:+.1f}% SPY{spy_ret:+.1f}% QQQ{qqq_ret:+.1f}%")
    lines.append(f"VIX: {vix_level:.0f}[{vix_status}] | KOSPI: {regime}({slots}슬롯) {close:,.0f}")
    lines.append(f">> {plan['market_stance']}")

    special = us.get("special_rules", [])
    if special:
        rules_str = ", ".join(r.get("name", "") for r in special)
        lines.append(f"!! {rules_str}")

    # ── Quantum 매수 후보 ──
    candidates = data["quantum"]["candidates"]
    lines.append("")
    if candidates:
        lines.append(f"[Quantum {len(candidates[:5])}종목]")
        for i, c in enumerate(candidates[:5], 1):
            grade = c.get("grade", "?")
            name = c.get("name", "")
            ticker = c.get("ticker", "")
            entry = c.get("entry_price", 0)
            target = c.get("target_price", 0)
            stop = c.get("stop_loss", 0)
            rr = c.get("risk_reward", 0)
            target_pct = (target - entry) / entry * 100 if entry > 0 else 0
            stop_pct = (stop - entry) / entry * 100 if entry > 0 else 0
            lines.append(f"{grade}{i}. {name}({ticker}) {entry:,.0f}원 RR1:{rr:.1f}")
            lines.append(f"   T{target_pct:+.1f}% S{stop_pct:+.1f}%")
    else:
        lines.append("[Quantum] 통과 종목 없음 (v8 게이트 대기)")

    # ── 릴레이 시그널 ──
    relay = data["relay"]
    fired = relay.get("fired_sectors", [])
    signals = relay.get("relay_signals", [])

    # 분류: 진입적기 vs 이미움직임
    actionable = []
    already_moved = []
    for s in signals:
        follow_ret = s.get("follow_stats", {}).get("avg_return", 0)
        conf = s["pattern"]["confidence"]
        if follow_ret < 3.0 and conf in ("HIGH", "MED"):
            actionable.append(s)
        elif follow_ret >= 3.0:
            already_moved.append(s)

    lines.append("")
    if actionable:
        lines.append(f"[Relay {len(actionable)}건 진입적기]")
        for sig in actionable:
            p = sig["pattern"]
            follow_ret = sig.get("follow_stats", {}).get("avg_return", 0)
            lines.append(f"{sig['lead_sector']}{sig['lead_return']:+.1f}% -> "
                         f"{sig['follow_sector']}(+{follow_ret:.1f}%) "
                         f"lag{p['best_lag']} W{p['win_rate']:.0f}%[{p['confidence']}]")
            picks = sig.get("picks", [])
            if picks:
                pick_parts = []
                for pk in picks[:3]:
                    fs = pk.get("foreign_streak", 0)
                    is_ = pk.get("inst_streak", 0)
                    flow = ""
                    if fs > 0 or is_ > 0:
                        fp = []
                        if fs > 0:
                            fp.append(f"외{fs}")
                        if is_ > 0:
                            fp.append(f"기{is_}")
                        flow = f"({'/'.join(fp)}일)"
                    pick_parts.append(f"{pk['name']}{flow}")
                lines.append(f"   -> {', '.join(pick_parts)}")
    elif fired:
        lines.append(f"[Relay] 발화{len(fired)}개 (전부 이미움직임)")
    else:
        lines.append("[Relay] 발화 없음")

    if already_moved:
        moved_str = ", ".join(f"{s['follow_sector']}(+{s.get('follow_stats', {}).get('avg_return', 0):.1f}%)"
                              for s in already_moved[:3])
        lines.append(f"   (이미움직임: {moved_str})")

    # ── 그룹 릴레이 (참고) ──
    gr = data.get("group_relay", {})
    gr_fired = gr.get("fired_groups", [])
    if gr_fired:
        lines.append("")
        lines.append(f"[그룹릴레이 {len(gr_fired)}건]")
        for fg in gr_fired:
            leader_chg = fg.get("leader_change", 0)
            lines.append(f"  {fg['group']}: {fg['leader_name']}{leader_chg:+.1f}% → 계열사 대기")
            for sub in fg.get("waiting_subsidiaries", [])[:3]:
                lines.append(f"   -> {sub['name']}({sub['change_pct']:+.1f}%) "
                              f"S{sub['score']:.0f} [{sub['tier']}]")

    # ── ETF 시그널 ──
    etf = data.get("etf", {})
    etf_smart = etf.get("smart_money_etf", [])
    etf_theme = etf.get("theme_money_etf", [])
    etf_watch = etf.get("watch_list", [])
    smart_sectors = etf.get("smart_sectors", [])

    etf_all = etf_smart + etf_theme
    lines.append("")
    if etf_all:
        lines.append(f"[ETF {len(etf_all)}건 매수]")
        if smart_sectors:
            lines.append(f"  SMART섹터: {', '.join(smart_sectors)}")
        for e in etf_all:
            signal = e.get("signal", "")
            sizing = e.get("sizing", "")
            foreign = e.get("foreign_5d_bil", 0)
            inst = e.get("inst_5d_bil", 0)
            lines.append(f"  {e['sector']}({e['etf_code']}) [{signal}] {sizing}")
            lines.append(f"   RSI{e.get('rsi', 0):.0f} BB{e.get('bb_pct', 0):.0f}% "
                         f"외인{foreign:+,}억 기관{inst:+,}억")
    elif etf_watch:
        lines.append(f"[ETF] 매수 0건, 관찰 {len(etf_watch)}건")
    else:
        lines.append("[ETF] 시그널 없음")

    if etf_watch:
        watch_str = ", ".join(
            f"{e['sector']}({e.get('signal', '')})"
            for e in etf_watch[:4]
        )
        lines.append(f"  관찰: {watch_str}")

    # ── 보유 포지션 ──
    q_pos = positions["quantum"]["positions"]
    r_pos = positions["relay"]["positions"]
    total_count = positions["total_count"]

    lines.append("")
    if total_count > 0:
        lines.append(f"[포지션 {total_count}건]")
        for p in q_pos:
            name = p.get("name", p.get("ticker", ""))
            pnl = p.get("unrealized_pnl_pct", 0)
            entry_date = p.get("entry_date", "")
            # 보유일수 계산
            days_str = ""
            if entry_date:
                try:
                    from datetime import datetime as _dt
                    d = _dt.strptime(entry_date, "%Y-%m-%d")
                    days = (datetime.now() - d).days
                    days_str = f" {days}일"
                except Exception:
                    pass
            lines.append(f"Q. {name} {pnl:+.1f}%{days_str}")
        for p in r_pos:
            pnl = p.get("pnl_pct", 0)
            days = p.get("trading_days_held", 0)
            lines.append(f"R. {p['name']} {pnl:+.1f}% {days}일 | "
                         f"{p.get('fired_sector', '')}>{p.get('sector', '')}")
        if positions["total_invested"] > 0:
            lines.append(f"   투입{positions['total_invested']:,}원 P/L{positions['total_relay_pnl']:+,}원")
    else:
        lines.append("[포지션] 없음")

    # ── 액션 플랜 ──
    stance = plan["market_stance"]
    lines.append("")
    lines.append("[Action]")
    if plan["buys"]:
        for b in plan["buys"]:
            src = "Q" if b["source"] == "Quantum" else "R"
            if src == "R":
                lines.append(f"  BUY[{src}] {b['name']} "
                             f"| {b.get('fired_sector', '')}>{b.get('sector', '')} W{b.get('win_rate', 0):.0f}%")
            else:
                lines.append(f"  BUY[{src}] {b['name']} "
                             f"| {b.get('entry_price', 0):,.0f}원 RR1:{b.get('risk_reward', 0):.1f}")
    else:
        lines.append(f"  BUY: 없음")

    if plan["sells"]:
        for s in plan["sells"]:
            lines.append(f"  SELL {s['name']} | {s.get('reason', '')}")

    if plan["watches"]:
        # 감시 항목 — 동일 follow를 머지
        follow_map: dict[str, list[str]] = {}
        other_watches = []
        for w in plan["watches"][:5]:
            name = w.get("name", "")
            if "->" in name or "\u2192" in name:
                # lead→follow 형태
                parts = name.replace("\u2192", "->").split("->")
                if len(parts) == 2:
                    lead = parts[0].strip()
                    follow = parts[1].strip()
                    follow_map.setdefault(follow, []).append(lead)
                    continue
            other_watches.append(name)
        if follow_map:
            merged = []
            for follow, leads in follow_map.items():
                merged.append(f"{follow}({'+'.join(leads)})")
            lines.append(f"  WATCH: {', '.join(merged)}")
            if "관망" in stance or "자제" in stance:
                lines.append(f"    * {stance} -- 발동시 소량만")
        for ow in other_watches:
            lines.append(f"  WATCH: {ow}")

    # ── 서보성 원칙 ──
    try:
        import yaml as _yaml
        with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f)
        _pos = _cfg.get("live_trading", {}).get("position", {})
        _reserve = _pos.get("cash_reserve_pct", 0.20)
        _max_pos = _pos.get("max_positions", 5)
        _sb = _pos.get("split_buy", {})
        _e1 = int(_sb.get("entry_1st", 0.50) * 100)
        _e2 = int(_sb.get("entry_2nd", 0.30) * 100)
        _e3 = int(_sb.get("entry_3rd", 0.20) * 100)
        lines.append(f"[원칙] 현금유보{int(_reserve*100)}% 최대{_max_pos}종목 분할{_e1}/{_e2}/{_e3}")
    except Exception:
        pass

    lines.append("-" * 30)
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 4. 실행
# ═══════════════════════════════════════════════

def run_report(
    send: bool = True,
    run_scan: bool = True,
    grade_filter: str = "AB",
    use_news: bool = False,
    text_only: bool = False,
    no_html: bool = False,
) -> dict:
    """통합 리포트 생성 + 발송. 외부에서 호출 가능한 진입점."""
    # 1. 데이터 수집
    data = collect_all_signals(run_scan, grade_filter, use_news)

    # 2. 액션 플랜
    data["action_plan"] = build_action_plan(data)

    # 3. 텍스트 메시지
    text_msg = format_text_message(data)
    print(text_msg)

    # 4. HTML 보고서
    png_path = None
    html_path = None
    if not no_html and not text_only:
        try:
            from src.html_integrated_report import generate_integrated_report
            html_path, png_path = generate_integrated_report(data)
            print(f"\nHTML: {html_path}")
            if png_path:
                print(f"PNG:  {png_path}")
        except Exception as e:
            print(f"\nHTML 생성 실패: {e}")
            import traceback
            traceback.print_exc()

    # 5. 텔레그램 발송
    if send:
        try:
            from src.telegram_sender import send_message
            send_message(text_msg)
            print("\n텔레그램 텍스트 발송 완료")
        except Exception as e:
            print(f"\n텔레그램 텍스트 발송 실패: {e}")

        if png_path and png_path.exists():
            try:
                from src.html_report import send_report_to_telegram
                caption = (f"[통합 데일리 리포트] {data['date']} | "
                           f"{data['action_plan']['market_stance']}")
                send_report_to_telegram(png_path, caption)
                print("텔레그램 이미지 발송 완료")
            except Exception as e:
                print(f"텔레그램 이미지 발송 실패: {e}")

    # 6. JSON 저장
    _save_report_json(data)

    return data


def _save_report_json(data: dict):
    """통합 리포트 JSON 저장."""
    out_path = PROJECT_ROOT / "data" / "integrated_report.json"
    plan = data.get("action_plan", {})
    payload = {
        "date": data["date"],
        "generated_at": data["generated_at"],
        "market_stance": plan.get("market_stance", ""),
        "us_grade": data["us_overnight"].get("grade", ""),
        "us_score": data["us_overnight"].get("combined_score_100", 0),
        "kospi_regime": data["kospi_regime"].get("regime", ""),
        "kospi_slots": data["kospi_regime"].get("slots", 0),
        "quantum_count": len(data["quantum"]["candidates"]),
        "relay_fired": len(data["relay"].get("fired_sectors", [])),
        "relay_signals": len(data["relay"].get("relay_signals", [])),
        "positions_total": data["positions"]["total_count"],
        "buys": len(plan.get("buys", [])),
        "sells": len(plan.get("sells", [])),
    }
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"저장: {out_path}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="통합 데일리 리포트")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 미발송")
    parser.add_argument("--no-scan", action="store_true", help="Quantum 스캔 건너뜀")
    parser.add_argument("--no-html", action="store_true", help="HTML 보고서 건너뜀")
    parser.add_argument("--text-only", action="store_true", help="텍스트만 발송")
    parser.add_argument("--grade", type=str, default="AB", help="Grade 필터")
    parser.add_argument("--no-news", action="store_true", help="뉴스 건너뜀")
    args = parser.parse_args()

    run_report(
        send=not args.no_send,
        run_scan=not args.no_scan,
        grade_filter=args.grade,
        use_news=not args.no_news,
        text_only=args.text_only,
        no_html=args.no_html,
    )


if __name__ == "__main__":
    main()
