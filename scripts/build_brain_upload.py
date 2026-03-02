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
    # 이란 전쟁 리스크 반영 확인 — 이미 risk_factors에 없으면 추가
    risks = raw.get("risk_factors", [])
    iran_keywords = ["이란", "Iran", "중동"]
    has_iran = any(any(kw in r for kw in iran_keywords) for r in risks)
    if not has_iran:
        risks.insert(0, "미국/이스라엘 vs 이란 군사충돌 — 중동 전면전 리스크 급등")
        risks.insert(1, "유가 급등 가능성 (WTI $85+ 전망) — 호르무즈 해협 봉쇄 시나리오")
        risks.insert(2, "안전자산 선호 심리 → 원/달러 환율 상승 압력")
        raw["risk_factors"] = risks
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
    """3/2 기준 이란 전쟁 반영 뉴스 데이터"""
    # ai_brain_judgment.json이 최신이면 사용, 아니면 직접 생성
    raw = load("ai_brain_judgment.json")
    date_str = raw.get("date", "") if raw else ""

    # 이란 전쟁 반영 — 미장 선물 급락 실시간 데이터 강제 적용
    # (ai_brain_judgment은 선물 급락 전 뉴스 기반이라 bullish → cautious 보정)
    if True:  # 이란 전쟁 기간 동안 강제 적용
        return {
            "market_sentiment": "cautious",
            "key_themes": [
                "미국/이스라엘 vs 이란 군사 충돌 — 중동 전면전 리스크",
                "유가 급등 전망 (호르무즈 해협 봉쇄 시나리오)",
                "방산주 글로벌 급등 — K-방산 수출 확대 수혜",
                "글로벌 기술주 랠리 (S&P +1.8%, NASDAQ +1.3%)",
                "대만 AI/반도체 수출 역대급 — HBM 수요 폭증",
                "한국은행 금리 동결 전망 — 유동성 우호적",
                "안전자산 선호 확대 — 금/달러 강세",
            ],
            "direction": "cautious",
            "sector_outlook": {
                "정유/에너지": {"direction": "positive", "reason": "이란 전쟁으로 유가 급등 전망, 정유사 마진 확대"},
                "방산": {"direction": "positive", "reason": "중동 전쟁 확대로 글로벌 방산 수주 급증, K-방산 최대 수혜"},
                "반도체": {"direction": "positive", "reason": "대만 AI 수출 호조 + Nvidia 실적 기대, 공급 타이트"},
                "IT서비스": {"direction": "positive", "reason": "반도체 릴레이 수혜 + 엔터프라이즈 AI 확산"},
                "금융": {"direction": "neutral", "reason": "금리 동결 전망이나 중동 리스크로 변동성 확대"},
                "건설": {"direction": "negative", "reason": "금리 불확실성 + 원자재 상승 압력으로 마진 악화"},
                "2차전지": {"direction": "negative", "reason": "유럽 EV 보조금 축소 + 원자재 상승 부담"},
                "유틸리티": {"direction": "neutral", "reason": "방어주 성격이나 유가 상승 시 발전원가 부담"},
                "조선": {"direction": "positive", "reason": "중동 리스크로 해상 운송 우회 수요 + LNG선 수주"},
                "철강": {"direction": "neutral", "reason": "원자재 상승 혼조 — 원가 부담 vs 제품가 전가"},
                "자동차": {"direction": "neutral", "reason": "현대차 그룹 릴레이 진행 중이나 환율 불확실성"},
            },
        }
    # 기존 데이터 활용
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


if __name__ == "__main__":
    main()
