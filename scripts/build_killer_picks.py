#!/usr/bin/env python
"""킬러픽 자비스 보고서 — 전 데이터소스 교차검증 자동화.

기관 매집 + 개인 지지 + 컨센서스 + 섹터 순위 + ETF + 과거 적중률을
교차 검증하여 '진짜 살 만한 종목'을 뽑는 자비스 컨트롤타워 스타일 리포트.

Output: data/killer_picks.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "killer_picks.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── helpers ───────────────────────────────────────────────


def _load(rel_path: str, default=None):
    """JSON 안전 로드 — 없으면 default 반환."""
    fp = DATA_DIR / rel_path
    if not fp.exists():
        return default if default is not None else {}
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default if default is not None else {}


def _bil(val) -> float:
    """원 → 억 변환 (이미 억이면 그대로)."""
    if val is None:
        return 0
    v = float(val)
    if abs(v) > 1_000_000_000:
        return round(v / 100_000_000, 1)
    return round(v, 1)


# ─── 1. 시장 환경 ──────────────────────────────────────────


def build_market_environment() -> dict:
    brain = _load("brain_decision.json")
    shield = _load("shield_report.json")
    intel = _load("market_intelligence.json")
    o1 = _load("o1_deep_analysis.json")

    regime = brain.get("effective_regime") or brain.get("regime") or "UNKNOWN"
    vix = brain.get("vix_level") or brain.get("vix") or 0
    cash_pct = brain.get("cash_pct") or brain.get("cash_ratio") or 0
    shield_level = shield.get("overall_level") or shield.get("status") or "UNKNOWN"

    mdd = shield.get("mdd_status", {})
    mdd_pct = mdd.get("current_mdd_pct", 0)

    us_mood = intel.get("us_market_mood", "UNKNOWN")
    geo_risk = o1.get("macro_analysis", {}).get("geopolitical_risk_score", 0)

    # 요약문 자동 생성
    summaries = {
        "NORMAL": "정상 모드 — 적극적 투자 가능",
        "CAUTION": "주의 모드 — 선별적 매수, 현금 확보",
        "BEAR": "약세 모드 — 방어 중심, 신규 매수 자제",
        "CRISIS": "위기 모드 — 전량 현금화 권고",
    }
    summary = summaries.get(regime, f"{regime} 모드")
    if vix >= 25:
        summary += f" (VIX {vix})"

    return {
        "regime": regime,
        "vix": vix,
        "shield": shield_level,
        "mdd_pct": round(mdd_pct, 1),
        "cash_pct": round(cash_pct, 1),
        "us_mood": us_mood,
        "geopolitical_risk": geo_risk,
        "summary": summary,
    }


# ─── 2. 시그널 검증 (과거 적중률) ──────────────────────────


def build_signal_validation() -> dict:
    acc = _load("market_learning/signal_accuracy.json")
    signals = acc.get("signals", {})
    daily_log = acc.get("daily_log", [])

    # 최고 시그널 찾기
    best_name, best_hr, best_ret = "", 0, 0
    for name, s in signals.items():
        hr = s.get("hit_rate", 0)
        if hr > best_hr and s.get("total", 0) >= 10:
            best_name = name
            best_hr = hr
            best_ret = s.get("avg_ret", 0)

    # 일별 트렌드 (최근 5일)
    trend = []
    for d in daily_log[-5:]:
        trend.append({
            "date": d.get("date", ""),
            "accum_hr": d.get("accumulation_tracker_hr", 0),
            "accum_ret": d.get("accumulation_tracker_ret", 0),
            "picks_hr": d.get("tomorrow_picks_hr", 0),
            "picks_ret": d.get("tomorrow_picks_ret", 0),
        })

    # 반등일 적중률 계산
    rebound_hits = [t for t in trend if t["accum_hr"] >= 60]
    rebound_insight = ""
    if rebound_hits:
        avg_hr = sum(t["accum_hr"] for t in rebound_hits) / len(rebound_hits)
        rebound_insight = f"매집 시그널이 반등일 평균 {avg_hr:.0f}% 적중"
    else:
        rebound_insight = f"최고 시그널: {best_name} ({best_hr:.1f}%)"

    return {
        "best_signal": best_name,
        "best_hit_rate": round(best_hr, 1),
        "best_avg_ret": round(best_ret, 2),
        "signal_summary": {
            k: {"hit_rate": v.get("hit_rate", 0), "avg_ret": v.get("avg_ret", 0), "total": v.get("total", 0)}
            for k, v in signals.items()
            if v.get("total", 0) > 0
        },
        "daily_trend": trend,
        "insight": rebound_insight,
    }


# ─── 3. 기관 매집 분석 ─────────────────────────────────────


def build_institutional_picks() -> list[dict]:
    alerts = _load("institutional_flow/accumulation_alert.json")
    consensus = _load("consensus_screening.json")
    sector_comp = _load("sector_rotation/sector_composite.json")

    # 컨센서스 lookup
    cons_map = {}
    for tp in consensus.get("top_picks", []):
        cons_map[tp["ticker"]] = tp

    # 섹터 순위 lookup
    sectors = sector_comp.get("sectors", [])
    sector_rank_map = {}
    for i, s in enumerate(sectors):
        sector_rank_map[s.get("sector", "")] = i + 1

    items = alerts.get("stock_alerts", [])
    # 기관 연속매수 3일+ 또는 STRONG/MODERATE 또는 EARLY_* (조기 감지)
    EARLY_GRADES = ("EARLY_DUAL", "EARLY_ACCEL", "EARLY_SURGE")
    result = []
    for item in items:
        inst_consec = item.get("inst_consecutive", 0)
        grade = item.get("grade", "")
        if inst_consec < 3 and grade not in ("STRONG", "MODERATE") and grade not in EARLY_GRADES:
            continue

        ticker = item.get("ticker", "")
        cons = cons_map.get(ticker, {})
        sector = item.get("sector", "")

        entry = {
            "ticker": ticker,
            "name": item.get("name", ""),
            "sector": sector,
            "inst_consecutive": inst_consec,
            "inst_5d_bil": item.get("inst_5d_억", 0),
            "inst_20d_bil": item.get("inst_20d_억", 0),
            "foreign_5d_bil": item.get("foreign_5d_억", 0),
            "dual_buying": item.get("dual_buying", False),
            "grade": grade,
        }

        # 컨센서스 매칭
        if cons:
            entry["consensus"] = {
                "target": cons.get("target_price", 0),
                "upside": cons.get("upside_pct", 0),
                "per": cons.get("forward_per", 0),
                "pbr": cons.get("forward_pbr", 0),
                "dividend": cons.get("dividend_yield", 0),
                "grade": cons.get("grade", ""),
            }

        entry["sector_rank"] = sector_rank_map.get(sector, 99)

        # verdict 자동 생성
        parts = []
        # 조기 감지 태그
        if grade == "EARLY_DUAL":
            parts.append("조기감지: 기관+외인 동시진입")
        elif grade == "EARLY_ACCEL":
            parts.append("조기감지: 기관 가속매수")
        elif grade == "EARLY_SURGE":
            parts.append("조기감지: 기관 대량매수")

        if inst_consec >= 5:
            parts.append(f"기관 {inst_consec}일 연속매수")
        elif inst_consec >= 3:
            parts.append(f"기관 {inst_consec}일 연속")
        elif inst_consec >= 2:
            parts.append(f"기관 {inst_consec}일 연속")
        if item.get("inst_20d_억", 0) >= 1000:
            parts.append(f"20일 +{item['inst_20d_억']:,.0f}억 대량 매집")
        if item.get("dual_buying") or item.get("dual_today"):
            parts.append("외인+기관 동반매수")
        if item.get("foreign_5d_억", 0) < -500:
            parts.append("외인 매도 중 → 기관이 흡수")
        entry["verdict"] = ", ".join(parts) if parts else grade

        result.append(entry)

    # 등급 우선순위 + 기관 연속매수일 + 20일 누적으로 정렬
    grade_priority = {"STRONG": 0, "MODERATE": 1, "NOTABLE": 2,
                      "EARLY_DUAL": 3, "EARLY_ACCEL": 4, "EARLY_SURGE": 5, "WATCH": 6}
    result.sort(key=lambda x: (
        grade_priority.get(x.get("grade", ""), 9),
        -x["inst_consecutive"],
        -abs(x.get("inst_20d_bil", 0)),
    ))
    return result[:15]  # 10 → 15: EARLY 포함하여 더 많이 노출


# ─── 4. 개인 지지 분석 ─────────────────────────────────────


def build_retail_support() -> list[dict]:
    nat = _load("krx_nationality/nationality_signal.json")
    signals = nat.get("signals", [])

    result = []
    for sig in signals:
        pattern = sig.get("retail_pattern", "")
        if pattern not in ("RETAIL_SUPPORT", "SMART_DIVERGE"):
            continue

        retail_5d = sig.get("retail_net_5d", 0)
        retail_20d = sig.get("retail_net_20d", 0)
        absorb = sig.get("retail_foreign_ratio", 0)
        consec = sig.get("retail_consecutive", 0)

        result.append({
            "ticker": sig.get("ticker", ""),
            "name": sig.get("name", ""),
            "retail_net_5d_bil": _bil(retail_5d),
            "retail_net_20d_bil": _bil(retail_20d),
            "retail_consecutive": consec,
            "absorb_rate": round(abs(absorb) * 100, 0) if absorb else 0,
            "pattern": pattern,
            "vol_price_pattern": sig.get("vol_price_pattern", ""),
            "verdict": _retail_verdict(sig),
        })

    result.sort(key=lambda x: (-x["retail_net_5d_bil"]))
    return result[:5]


def _retail_verdict(sig: dict) -> str:
    parts = []
    consec = sig.get("retail_consecutive", 0)
    absorb = sig.get("retail_foreign_ratio", 0)
    pattern = sig.get("retail_pattern", "")

    if consec >= 5:
        parts.append(f"개인 {consec}일 연속 매수")
    elif consec >= 2:
        parts.append(f"개인 {consec}일 연속")

    if absorb and abs(absorb) > 0.5:
        parts.append(f"외인 매도 {abs(absorb)*100:.0f}% 흡수")

    if pattern == "SMART_DIVERGE":
        parts.append("스마트머니 괴리 (기관 매수 vs 개인 매도)")

    return ", ".join(parts) if parts else pattern


# ─── 5. 교차검증 TOP 5 ─────────────────────────────────────


def build_cross_validated_top5() -> list[dict]:
    """여러 시스템에서 동시에 포착된 종목을 교차 검증."""
    inst_alerts = _load("institutional_flow/accumulation_alert.json")
    consensus = _load("consensus_screening.json")
    pullback = _load("pullback_scan.json")
    sector_comp = _load("sector_rotation/sector_composite.json")
    nat = _load("krx_nationality/nationality_signal.json")
    tomorrow = _load("tomorrow_picks.json")

    # 각 소스별 ticker set 구성
    # 기관 매집 (3일+ 또는 STRONG/MODERATE)
    inst_map = {}
    for item in inst_alerts.get("stock_alerts", []):
        ic = item.get("inst_consecutive", 0)
        g = item.get("grade", "")
        if ic >= 3 or g in ("STRONG", "MODERATE"):
            inst_map[item["ticker"]] = item

    # 컨센서스 A등급 이상
    cons_map = {}
    for tp in consensus.get("top_picks", []):
        if tp.get("grade") in ("S", "A"):
            cons_map[tp["ticker"]] = tp

    # 눌림목 매수대기
    pull_map = {}
    for c in pullback.get("candidates", []):
        if c.get("grade") == "매수대기":
            pull_map[c["ticker"]] = c

    # 국적별 BUY 시그널
    nat_map = {}
    for sig in nat.get("signals", []):
        if sig.get("signal") == "BUY" or sig.get("retail_pattern") == "RETAIL_SUPPORT":
            nat_map[sig["ticker"]] = sig

    # 기존 추천 (매수/관찰)
    picks_map = {}
    for p in tomorrow.get("picks", []):
        if p.get("grade") in ("강력 포착", "포착", "관심", "관찰", "적극매수", "매수", "관심매수"):
            picks_map[p["ticker"]] = p

    # 섹터 기관 수급 강한 섹터의 종목들
    top_sectors = set()
    for s in sector_comp.get("sectors", [])[:5]:
        if s.get("inst_5d_억", 0) > 500:
            top_sectors.add(s.get("sector", ""))

    # 교차 카운트
    all_tickers = set()
    all_tickers.update(inst_map.keys(), cons_map.keys(), pull_map.keys(), nat_map.keys(), picks_map.keys())

    candidates = []
    for ticker in all_tickers:
        signals_matched = []
        if ticker in inst_map:
            ic = inst_map[ticker].get("inst_consecutive", 0)
            signals_matched.append(f"기관{ic}일연속")
        if ticker in cons_map:
            cg = cons_map[ticker].get("grade", "")
            signals_matched.append(f"컨센서스{cg}")
        if ticker in pull_map:
            signals_matched.append("눌림목매수대기")
        if ticker in nat_map:
            ns = nat_map[ticker].get("signal", "")
            rp = nat_map[ticker].get("retail_pattern", "")
            if ns == "BUY":
                signals_matched.append("국적별BUY")
            if rp == "RETAIL_SUPPORT":
                signals_matched.append("개인지지")
        if ticker in picks_map:
            pg = picks_map[ticker].get("grade", "")
            signals_matched.append(f"추천{pg}")

        # 해당 종목의 섹터가 top_sectors에 있으면 추가 시그널
        sector = inst_map.get(ticker, {}).get("sector", "")
        if sector in top_sectors:
            signals_matched.append(f"{sector}섹터강세")

        if len(signals_matched) < 2:
            continue

        # 종목 정보 모으기
        name = (
            inst_map.get(ticker, {}).get("name")
            or cons_map.get(ticker, {}).get("name", "")
            or pull_map.get(ticker, {}).get("name", "")
            or nat_map.get(ticker, {}).get("name", "")
            or picks_map.get(ticker, {}).get("name", "")
            or ""
        )

        entry = {
            "ticker": ticker,
            "name": name,
            "signals_matched": len(signals_matched),
            "matched_from": signals_matched,
        }

        # 컨센서스 정보
        if ticker in cons_map:
            c = cons_map[ticker]
            entry["consensus"] = {
                "target": c.get("target_price", 0),
                "upside": c.get("upside_pct", 0),
                "per": c.get("forward_per", 0),
                "pbr": c.get("forward_pbr", 0),
                "dividend": c.get("dividend_yield", 0),
            }

        # 기관 정보
        if ticker in inst_map:
            ii = inst_map[ticker]
            entry["inst_consecutive"] = ii.get("inst_consecutive", 0)
            entry["inst_20d_bil"] = ii.get("inst_20d_억", 0)

        # 추천 정보 (진입가/손절/목표)
        if ticker in picks_map:
            pp = picks_map[ticker]
            entry["entry_price"] = pp.get("entry_price", pp.get("close", 0))
            entry["stop_loss"] = pp.get("stop_loss", 0)
            entry["target_price"] = pp.get("target_price", 0)
        elif ticker in cons_map:
            cc = cons_map[ticker]
            entry["entry_price"] = cc.get("close", 0)
            entry["stop_loss"] = round(cc.get("close", 0) * 0.95)
            entry["target_price"] = cc.get("target_price", 0)

        # conviction
        n = len(signals_matched)
        if n >= 4:
            entry["conviction"] = "HIGH"
        elif n >= 3:
            entry["conviction"] = "MEDIUM"
        else:
            entry["conviction"] = "LOW"

        # action
        if n >= 3 and ticker in cons_map:
            entry["action"] = "매수"
        elif n >= 3:
            entry["action"] = "관심매수"
        else:
            entry["action"] = "관찰"

        candidates.append(entry)

    candidates.sort(key=lambda x: (-x["signals_matched"], -x.get("inst_20d_bil", 0)))
    # 순위 부여
    for i, c in enumerate(candidates[:10]):
        c["rank"] = i + 1

    return candidates[:10]


# ─── 6. ETF TOP 5 ──────────────────────────────────────────


def build_etf_top5() -> list[dict]:
    etf_sig = _load("sector_rotation/etf_trading_signal.json")
    etf_rec = _load("etf_recommendations.json")
    sector_comp = _load("sector_rotation/sector_composite.json")

    result = []
    rank = 0

    # 섹터 순위에서 기관 수급 강한 섹터의 ETF
    for s in sector_comp.get("sectors", [])[:5]:
        inst_5d = s.get("inst_5d_억", 0)
        if inst_5d < 200:
            continue
        rank += 1
        etf_code = s.get("etf_code", "")
        result.append({
            "rank": rank,
            "ticker": etf_code,
            "name": f"KODEX {s.get('sector', '')}",
            "category": "섹터",
            "signal": f"기관 5일 +{inst_5d:,.0f}억",
            "sector_score": s.get("composite_score", 0),
            "regime": s.get("regime", ""),
            "action": "BUY" if s.get("regime") == "STRONG_ROTATION" else "관심",
            "sizing": "10~20%",
            "reason": f"{s.get('sector', '')} 섹터 기관+외인 유입",
        })

    # SMART_BUY ETF
    for sig in etf_sig.get("smart_money_etf", []):
        if sig.get("signal") == "SMART_BUY" and sig.get("sector") not in [r.get("name", "").replace("KODEX ", "") for r in result]:
            rank += 1
            result.append({
                "rank": rank,
                "ticker": sig.get("etf_code", ""),
                "name": f"KODEX {sig.get('sector', '')}",
                "category": "스마트머니",
                "signal": f"SMART_BUY RSI{sig.get('rsi', 0):.0f} BB{sig.get('bb_pct', 0):.0f}%",
                "action": "BUY",
                "sizing": sig.get("sizing", "FULL"),
                "reason": sig.get("reason", ""),
            })

    # 방어형 ETF (인버스/금/달러)
    for pick in etf_rec.get("etf_picks", []):
        if pick.get("action") == "BUY":
            rank += 1
            result.append({
                "rank": rank,
                "ticker": pick.get("ticker", ""),
                "name": pick.get("name", ""),
                "category": pick.get("category", "방어"),
                "signal": " + ".join(pick.get("reason", [])[:2]),
                "action": "BUY",
                "sizing": f"{pick.get('portfolio_pct', 0)}%",
                "holding_period": pick.get("holding_period", ""),
                "entry_timing": pick.get("entry_timing", ""),
                "stop_loss": pick.get("stop_loss", ""),
                "reason": pick.get("reason", [""])[0] if pick.get("reason") else "",
            })

    return result[:8]


# ─── 7. 포트폴리오 제안 ────────────────────────────────────


def build_portfolio_suggestion(env: dict, top5: list, etf5: list) -> dict:
    regime = env.get("regime", "NORMAL")

    # 레짐별 방어/공격 비율
    ratios = {
        "CRISIS": (90, 10),
        "BEAR": (75, 25),
        "CAUTION": (60, 40),
        "NORMAL": (40, 60),
        "BULL": (20, 80),
    }
    defense_pct, offense_pct = ratios.get(regime, (50, 50))

    # 방어 포트폴리오
    defense = []
    # 방어형 ETF 추가
    for e in etf5:
        cat = e.get("category", "")
        if cat in ("인버스", "헤지", "매크로", "방어"):
            pct = int(str(e.get("sizing", "10")).replace("%", "").strip() or "10")
            defense.append({"name": e["name"], "ticker": e["ticker"], "pct": pct})
    # 섹터 ETF 중 방어적인 것 (은행/금융)
    for e in etf5:
        if e.get("category") == "섹터" and any(k in e.get("name", "") for k in ("은행", "금융", "보험")):
            defense.append({"name": e["name"], "ticker": e["ticker"], "pct": 15})
    # 남은 비중은 현금
    used = sum(d["pct"] for d in defense)
    if used < defense_pct:
        defense.append({"name": "현금", "ticker": "cash", "pct": defense_pct - used})

    # 공격 포트폴리오 (교차검증 TOP + 공격형 ETF)
    offense = []
    for i, pick in enumerate(top5[:4]):
        pct = min(15, offense_pct // min(4, len(top5)))
        offense.append({"name": pick["name"], "ticker": pick["ticker"], "pct": pct})
    # 공격형 섹터 ETF
    for e in etf5:
        if e.get("category") in ("섹터", "스마트머니") and not any(k in e.get("name", "") for k in ("은행", "금융", "보험")):
            if len(offense) < 6:
                offense.append({"name": e["name"], "ticker": e["ticker"], "pct": 10})

    return {
        "defense_pct": defense_pct,
        "offense_pct": offense_pct,
        "defense": defense,
        "offense": offense,
    }


# ─── 8. 섹터 분석 ──────────────────────────────────────────


def build_sector_analysis() -> dict:
    sector_comp = _load("sector_rotation/sector_composite.json")
    o1 = _load("o1_deep_analysis.json")
    intel = _load("market_intelligence.json")

    sectors = sector_comp.get("sectors", [])

    top_sectors = []
    for s in sectors[:5]:
        entry = {
            "sector": s.get("sector", ""),
            "score": s.get("composite_score", 0),
            "regime": s.get("regime", ""),
            "inst_5d": s.get("inst_5d_억", 0),
            "foreign_5d": s.get("foreign_5d_억", 0),
            "ret_5d": s.get("ret_5", 0),
        }
        top_sectors.append(entry)

    # 회피 섹터 (하위 5개)
    avoid_sectors = [s.get("sector", "") for s in sectors[-5:] if s.get("composite_score", 0) < 30]

    # 자금 흐름 요약
    cross = o1.get("cross_sector_dynamics", "")
    money_flow = cross[:100] if cross else ""

    return {
        "top_sectors": top_sectors,
        "avoid_sectors": avoid_sectors,
        "money_flow": money_flow,
    }


# ─── 9. 개별 딥다이브 ──────────────────────────────────────


def build_deep_dive(inst_picks: list, env: dict) -> list[dict]:
    """기관 연속매수 상위 종목에 대한 bull/bear 분석."""
    o1 = _load("o1_deep_analysis.json")
    brain = _load("brain_decision.json")
    targets = _load("institutional_targets.json").get("targets", {})

    micro = {}
    for m in o1.get("micro_analysis", []):
        micro[m.get("sector", "")] = m

    result = []
    for item in inst_picks[:3]:
        ticker = item.get("ticker", "")
        sector = item.get("sector", "")
        sector_info = micro.get(sector, {})
        target_info = targets.get(ticker, {})

        bull_parts = []
        bear_parts = []

        # Bull case
        ic = item.get("inst_consecutive", 0)
        if ic >= 5:
            bull_parts.append(f"기관 {ic}일 연속 매집")
        i20 = item.get("inst_20d_bil", 0)
        if i20 >= 1000:
            bull_parts.append(f"20일 +{i20:,.0f}억 대량 유입")
        if sector_info.get("earnings_cycle") in ("recovery", "expansion"):
            bull_parts.append(f"{sector} 실적 {sector_info['earnings_cycle']}")
        if sector_info.get("conviction", 0) >= 7:
            bull_parts.append(f"AI conviction {sector_info['conviction']}/10")
        if target_info:
            gap = target_info.get("gap_pct", 0)
            if gap < -5:
                bull_parts.append(f"목표가 대비 {abs(gap):.1f}% 저평가")

        # Bear case
        f5 = item.get("foreign_5d_bil", 0)
        if f5 < -500:
            bear_parts.append(f"외인 5일 {f5:,.0f}억 이탈")
        mdd = env.get("mdd_pct", 0)
        if mdd < -20:
            bear_parts.append(f"MDD {mdd:.1f}%")
        shield = env.get("shield", "")
        if shield == "RED":
            bear_parts.append("SHIELD RED")
        if sector_info.get("supply_demand_shift") == "분배":
            bear_parts.append(f"{sector} 자금 분배 단계")

        # Verdict
        if len(bull_parts) >= 3 and len(bear_parts) <= 1:
            verdict = "매수 유력 — 기관 확신 강함"
        elif len(bear_parts) >= 3:
            verdict = "관망 — 리스크 우선 확인"
        else:
            verdict = "분할매수 대기 — 지지선 확인 후"

        cons = item.get("consensus", {})
        result.append({
            "ticker": ticker,
            "name": item.get("name", ""),
            "sector": sector,
            "question": f"{item.get('name', '')} 지금 매수 적기인가?",
            "bull_case": ", ".join(bull_parts) if bull_parts else "특별한 강점 없음",
            "bear_case": ", ".join(bear_parts) if bear_parts else "특별한 위험 없음",
            "verdict": verdict,
            "entry_condition": cons.get("target", "지지선 확인 후"),
            "target": f"{cons.get('target', 0):,} ({cons.get('upside', 0):.1f}%)" if cons else "",
        })

    return result


# ─── MAIN ───────────────────────────────────────────────────


def build_killer_picks() -> dict:
    log.info("킬러픽 생성 시작...")

    # 날짜 결정 (tomorrow_picks의 target_date 사용)
    tp = _load("tomorrow_picks.json")
    target_date = tp.get("target_date", datetime.now().strftime("%Y-%m-%d"))
    target_label = tp.get("target_date_label", "")

    # 각 섹션 빌드
    env = build_market_environment()
    log.info(f"  시장환경: {env['regime']} / VIX {env['vix']} / SHIELD {env['shield']}")

    validation = build_signal_validation()
    log.info(f"  시그널검증: 최고={validation['best_signal']} ({validation['best_hit_rate']}%)")

    inst_picks = build_institutional_picks()
    log.info(f"  기관매집: {len(inst_picks)}개 종목")

    retail = build_retail_support()
    log.info(f"  개인지지: {len(retail)}개 종목")

    top5 = build_cross_validated_top5()
    log.info(f"  교차검증: {len(top5)}개 종목 (TOP: {top5[0]['name'] if top5 else 'N/A'})")

    etf5 = build_etf_top5()
    log.info(f"  ETF: {len(etf5)}개")

    portfolio = build_portfolio_suggestion(env, top5, etf5)
    log.info(f"  포트폴리오: 방어{portfolio['defense_pct']}% / 공격{portfolio['offense_pct']}%")

    sector = build_sector_analysis()
    log.info(f"  섹터: TOP={sector['top_sectors'][0]['sector'] if sector['top_sectors'] else 'N/A'}")

    deep_dive = build_deep_dive(inst_picks, env)
    log.info(f"  딥다이브: {len(deep_dive)}개 종목")

    result = {
        "date": target_date,
        "target_label": target_label,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market_environment": env,
        "signal_validation": validation,
        "institutional_picks": inst_picks,
        "retail_support": retail,
        "cross_validated_top5": top5,
        "etf_top5": etf5,
        "portfolio_suggestion": portfolio,
        "sector_analysis": sector,
        "individual_deep_dive": deep_dive,
    }

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    log.info(f"킬러픽 저장 완료: {OUTPUT_PATH} ({size_kb:.1f}KB)")
    log.info(f"  교차검증 TOP: {[c['name'] for c in top5[:5]]}")
    log.info(f"  ETF: {[e['name'] for e in etf5[:5]]}")

    return result


if __name__ == "__main__":
    build_killer_picks()
