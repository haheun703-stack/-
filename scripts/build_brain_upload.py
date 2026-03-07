"""brain_data_upload.json 빌드 스크립트 — 대시보드용 전체 데이터 통합"""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "website" / "data" / "brain_data_upload.json"


def load(name, default=None):
    p = DATA / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default or {}


def build_strategic():
    raw = load("ai_strategic_analysis.json")
    if not raw:
        return {}
    return raw


def build_sector_focus():
    return load("ai_sector_focus.json")


def build_group_relay():
    raw = load("group_relay/group_relay_today.json")
    if not raw:
        return {"fired_groups": [], "scan_time": ""}
    return raw


def build_whale():
    raw = load("whale_detect.json")
    if not raw:
        return {"items": [], "stats": {}}
    # 상위 15개만
    items = raw.get("items", [])[:15]
    return {
        "updated_at": raw.get("updated_at", ""),
        "total_detected": raw.get("total_detected", 0),
        "stats": raw.get("stats", {}),
        "items": items,
    }


def build_dual_buying():
    raw = load("dual_buying_watch.json")
    if not raw:
        return {"s_grade": [], "a_grade": [], "b_grade": [], "stats": {}}
    return raw


def build_v3_picks():
    return load("ai_v3_picks.json")


def build_news():
    """AI Brain 뉴스 판단 데이터 — ai_brain_judgment.json 기반"""
    raw = load("ai_brain_judgment.json")
    if not raw:
        return {"market_sentiment": "neutral", "key_themes": [], "direction": "neutral", "sector_outlook": {}}
    return {
        "market_sentiment": raw.get("market_sentiment", raw.get("sentiment", "neutral")),
        "key_themes": raw.get("key_themes", raw.get("themes", [])),
        "direction": raw.get("direction", raw.get("dir", "neutral")),
        "sector_outlook": raw.get("sector_outlook", {}),
    }


def build_journal():
    """매매일지 — AI 추천 vs 실제 결과"""
    v3 = load("ai_v3_picks.json")
    tp = load("tomorrow_picks.json")

    days = []

    # v3 Brain 추천 (3/2 기준)
    if v3 and v3.get("buys"):
        v3_picks = []
        for b in v3["buys"]:
            v3_picks.append({
                "ticker": b["ticker"],
                "name": b["name"],
                "grade": f"conviction {b.get('conviction', '?')}",
                "score": b.get("conviction", 0) * 10,
                "entry_price": b.get("entry_price", 0),
                "current_price": b.get("entry_price", 0),  # 주말이라 동일
                "result_pct": 0.0,
                "hit": False,
                "reasoning": b.get("reasoning", ""),
            })
        days.append({
            "date": f"{v3.get('decision_date', '2026-03-02')} (AI Brain v3)",
            "day_return": 0.0,
            "picks": v3_picks,
        })

    # tomorrow_picks TOP 7 (3/3 대상)
    if tp and tp.get("picks"):
        tp_picks = []
        for p in tp["picks"][:7]:
            tp_picks.append({
                "ticker": p["ticker"],
                "name": p["name"],
                "grade": p.get("grade", ""),
                "score": p.get("total_score", 0),
                "entry_price": 0,
                "current_price": 0,
                "result_pct": 0.0,
                "hit": False,
                "reasoning": ", ".join(p.get("sources", [])),
            })
        days.append({
            "date": f"{tp.get('target_date', '2026-03-03')} (TOP 7 추천)",
            "day_return": 0.0,
            "picks": tp_picks,
        })

    # AI 대형주 추천 (3/3 대상)
    if tp and tp.get("ai_largecap"):
        lc_picks = []
        for a in tp["ai_largecap"][:5]:
            lc_picks.append({
                "ticker": a["ticker"],
                "name": a["name"],
                "grade": f"AI {a.get('urgency', '')}",
                "score": int(a.get("confidence", 0) * 100),
                "entry_price": 0,
                "current_price": 0,
                "result_pct": 0.0,
                "hit": False,
                "reasoning": a.get("reasoning", ""),
            })
        days.append({
            "date": f"{tp.get('target_date', '2026-03-03')} (AI 대형주)",
            "day_return": 0.0,
            "picks": lc_picks,
        })

    total = sum(len(d["picks"]) for d in days)
    return {
        "total_picks": total,
        "hit_rate": 0.0,
        "avg_return": 0.0,
        "period": "2026-03-02 ~ 2026-03-03",
        "days": days,
    }


def build_brain_decision():
    """BRAIN 자본배분 결정 데이터"""
    raw = load("brain_decision.json")
    if not raw:
        return {}
    return {
        "timestamp": raw.get("timestamp", ""),
        "effective_regime": raw.get("effective_regime", ""),
        "kospi_regime": raw.get("kospi_regime", ""),
        "nightwatch_score": raw.get("nightwatch_score", 0),
        "vix_level": raw.get("vix_level", 0),
        "confidence": raw.get("confidence", 0),
        "arms": raw.get("arms", []),
        "total_invest_pct": raw.get("total_invest_pct", 0),
        "cash_pct": raw.get("cash_pct", 0),
        "adjustments": raw.get("adjustments", []),
        "warnings": raw.get("warnings", []),
    }


def build_shield_report():
    """SHIELD 포트폴리오 방어 리포트"""
    raw = load("shield_report.json")
    if not raw:
        return {}
    return {
        "timestamp": raw.get("timestamp", ""),
        "overall_level": raw.get("overall_level", "GREEN"),
        "mdd_status": raw.get("mdd_status", {}),
        "stock_alerts": raw.get("stock_alerts", []),
        "systemic_risk": raw.get("systemic_risk", {}),
        "sector_overlaps": raw.get("sector_overlaps", []),
    }


def build_china_money():
    """차이나머니 수급 시그널 데이터"""
    raw = load("china_money/china_money_signal.json")
    if not raw:
        return {"date": "", "summary": {}, "signals": [], "top_foreign_buyers": []}
    return {
        "date": raw.get("date", ""),
        "generated_at": raw.get("generated_at", ""),
        "total_stocks": raw.get("total_stocks", 0),
        "summary": raw.get("summary", {}),
        "signals": raw.get("signals", [])[:15],
        "top_foreign_buyers": raw.get("top_foreign_buyers", [])[:10],
    }


def main():
    brain = {
        "strategic": build_strategic(),
        "sector_focus": build_sector_focus(),
        "group_relay": build_group_relay(),
        "whale": build_whale(),
        "dual_buying": build_dual_buying(),
        "v3_picks": build_v3_picks(),
        "news": build_news(),
        "journal": build_journal(),
        "china_money": build_china_money(),
        "brain_decision": build_brain_decision(),
        "shield_report": build_shield_report(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(brain, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"brain_data_upload.json saved ({OUT.stat().st_size:,} bytes)")
    print(f"  strategic regime: {brain['strategic'].get('regime', 'N/A')}")
    print(f"  whale items: {len(brain['whale'].get('items', []))}")
    print(f"  v3 buys: {len(brain['v3_picks'].get('buys', []))}")
    print(f"  journal days: {len(brain['journal'].get('days', []))}")
    print(f"  journal total picks: {brain['journal'].get('total_picks', 0)}")
    print(f"  news sentiment: {brain['news'].get('market_sentiment', 'N/A')}")
    print(f"  risk factors: {len(brain['strategic'].get('risk_factors', []))}")
    cm = brain["china_money"]
    print(f"  china_money: {cm.get('summary', {}).get('SURGE', 0)} SURGE / {cm.get('summary', {}).get('INFLOW', 0)} INFLOW")
    bd = brain["brain_decision"]
    print(f"  brain_decision: regime={bd.get('effective_regime', 'N/A')}, invest={bd.get('total_invest_pct', 0):.1f}%, cash={bd.get('cash_pct', 0):.1f}%")
    sr = brain["shield_report"]
    print(f"  shield_report: level={sr.get('overall_level', 'N/A')}, alerts={len(sr.get('stock_alerts', []))}")


if __name__ == "__main__":
    main()
