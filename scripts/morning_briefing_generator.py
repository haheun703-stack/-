"""FLOWX 모닝 브리핑 생성기 — STEP 1

8레이어 데이터 소스를 수집하여 구조화된 브리핑 JSON 생성.
Supabase morning_briefings 테이블에 upsert 할 수 있는 형태로 반환.

데이터 소스:
  L1: data/us_market/overnight_signal.json (US 야간)
  L2: data/market_intelligence.json (시장 인텔리전스)
  L3: data/regime_macro_signal.json (레짐/매크로)
  L4: data/ai_strategic_analysis.json (AI 전략)
  L5: data/ai_sector_focus.json (섹터 포커스)
  L6: data/tomorrow_picks.json (추천 종목)
  L7: data/morning_reports.json (증권사 리포트)
  L8: data/theme_alerts.json (테마 알림)

Usage:
    python scripts/morning_briefing_generator.py [--dry-run]
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trading_calendar import (
    should_run_bat, last_us_trading_day, get_stale_data_warning,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── 헬퍼 ──────────────────────────────────────────

def _load_json(path: Path) -> dict | list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── KOSPI 상승 확률 계산 ──────────────────────────

def _calc_kospi_probability(sig: dict) -> dict:
    """로지스틱 회귀 기반 KOSPI 상승/하락 확률.

    데이터 정합성 체크:
      - 모든 US 수익률이 0%이면 "데이터 없음" → 50/50 반환
      - est_low/est_high와 up_prob 방향 정합 (이중 경로 모순 방지)
    """
    idx = sig.get("index_direction", {})
    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    vix = sig.get("vix", {}).get("level", 20)

    us_vec = sig.get("l2_pattern", {}).get("today_us_vector", {})
    sox_ret = us_vec.get("us_soxx_chg", 0)
    vix_chg = us_vec.get("us_vix_chg", 0)

    # ── 데이터 정합성 체크: 모든 주요 지수 0%이면 비거래일 데이터 ──
    all_zero = (
        abs(ewy_ret) < 0.01 and abs(spy_ret) < 0.01
        and abs(qqq_ret) < 0.01 and abs(sox_ret) < 0.01
    )
    if all_zero:
        logger.warning(
            "US 수익률 전부 0% → 비거래일(주말/공휴일) 데이터. "
            "KOSPI 예측 50/50 반환."
        )
        return {
            "up_prob": 50,
            "down_prob": 50,
            "est_low": -0.3,
            "est_high": 0.3,
            "stale_data": True,
        }

    # STOXX50 (yfinance)
    stoxx50_ret = 0.0
    try:
        import yfinance as yf
        d = yf.download("^STOXX50E", period="5d", progress=False)
        if len(d) >= 2:
            d.columns = d.columns.droplevel(1) if d.columns.nlevels > 1 else d.columns
            closes = d["Close"].dropna()
            if len(closes) >= 2:
                stoxx50_ret = round(float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100), 2)
    except Exception:
        pass

    # 모델 로드
    model_path = DATA_DIR / "us_market" / "kospi_pred_model.json"
    model = _load_json(model_path) if model_path.exists() else {}

    if model and model.get("coef"):
        features = model["features"]
        mean = model["scaler_mean"]
        scale = model["scaler_scale"]
        coef = model["coef"]
        intercept = model["intercept"]

        raw = {
            "ewy": ewy_ret, "soxx": sox_ret, "vix_chg": vix_chg,
            "vix_level": vix, "spy": spy_ret, "qqq": qqq_ret,
            "stoxx50": stoxx50_ret,
        }

        logit = intercept
        for i, feat in enumerate(features):
            val = raw.get(feat, 0.0)
            scaled = (val - mean[i]) / scale[i] if scale[i] != 0 else 0
            logit += coef[i] * scaled

        up_prob = round(_sigmoid(logit) * 100)
    else:
        up_prob = 50
        if sox_ret > 2: up_prob += 12
        elif sox_ret < -2: up_prob -= 12
        if spy_ret > 1: up_prob += 8
        elif spy_ret < -1: up_prob -= 8
        if vix > 25: up_prob -= 8
        up_prob = max(5, min(95, up_prob))

    down_prob = 100 - up_prob

    # ── 레인지 추정 (est_low/est_high) ──
    base = sox_ret * 0.35 + spy_ret * 0.35 + stoxx50_ret * 0.3
    if abs(base) > 0.1:
        est_low = round(base * 0.5, 1)
        est_high = round(base * 1.1, 1)
    else:
        est_low, est_high = -0.3, 0.3

    # ── 정합성 보정: up_prob와 est 방향이 모순되면 조정 ──
    # 예: 모델이 65% 상승인데 est가 -0.7%~-0.3% → 하한을 0으로 올림
    est_lo = min(est_low, est_high)
    est_hi = max(est_low, est_high)

    if up_prob >= 60 and est_hi < 0:
        # 상승 우세인데 레인지가 전부 음수 → 레인지를 모델 기준으로 조정
        est_lo = round(base * 0.3, 1) if base > 0 else -0.2
        est_hi = round(base * 1.0, 1) if base > 0 else 0.5
    elif down_prob >= 60 and est_lo > 0:
        # 하락 우세인데 레인지가 전부 양수 → 반대로 조정
        est_lo = round(base * 1.0, 1) if base < 0 else -0.5
        est_hi = round(base * 0.3, 1) if base < 0 else 0.2

    return {
        "up_prob": up_prob,
        "down_prob": down_prob,
        "est_low": min(est_lo, est_hi),
        "est_high": max(est_lo, est_hi),
    }


# ── 시장 상태 판정 ──────────────────────────────

def _determine_market_status(prob: dict, regime: str) -> str:
    """BULL / BEAR / NEUTRAL / CAUTION 판정."""
    # 비거래일 데이터이면 NEUTRAL 반환
    if prob.get("stale_data"):
        return "NEUTRAL"
    if regime == "CRISIS":
        return "BEAR"
    if prob["up_prob"] >= 60:
        return "BULL"
    if prob["down_prob"] >= 60:
        return "BEAR"
    if regime in ("BEAR", "CAUTION"):
        return "CAUTION"
    return "NEUTRAL"


# ── 메인 생성 함수 ──────────────────────────────

def generate_morning_briefing(date_str: str = "") -> dict:
    """8레이어 데이터를 수집하여 Supabase용 브리핑 dict 반환.

    Returns:
        {
            "date": "2026-03-18",
            "market_status": "CAUTION",
            "us_summary": "SPY -0.3%, QQQ +0.1%, VIX 22.4 경계",
            "kr_summary": "KOSPI 상승확률 48%, 보합권 예상",
            "kospi_prob": {"up_prob": 48, ...},
            "news_picks": [{"ticker": "005930", "title": "..."}],
            "sector_focus": ["반도체", "방산"],
            "briefing_full": "...(텔레그램 전체용 텍스트)",
            "briefing_summary": "...(FLOWX 요약용 텍스트)",
        }
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # ── L1: US 오버나이트 ──
    sig = _load_json(DATA_DIR / "us_market" / "overnight_signal.json")
    idx = sig.get("index_direction", {})
    spy_ret = idx.get("SPY", {}).get("ret_1d", 0)
    qqq_ret = idx.get("QQQ", {}).get("ret_1d", 0)
    dia_ret = idx.get("DIA", {}).get("ret_1d", 0)
    vix_data = sig.get("vix", {})
    vix = vix_data.get("level", 20)
    vix_status = vix_data.get("status", "정상")
    us_grade = sig.get("grade", "NEUTRAL")

    us_vec = sig.get("l2_pattern", {}).get("today_us_vector", {})
    sox_ret = us_vec.get("us_soxx_chg", 0)
    ewy_ret = idx.get("EWY", {}).get("ret_1d", 0)

    # ── 데이터 신선도 체크 ──
    sig_date = sig.get("us_close_date", "")
    stale_warning = get_stale_data_warning(sig_date)
    if stale_warning:
        logger.warning(f"Overnight Signal 데이터 경고: {stale_warning}")

    us_summary = f"SPY {spy_ret:+.1f}%, QQQ {qqq_ret:+.1f}%, SOXX {sox_ret:+.1f}%, VIX {vix:.1f} {vix_status}"
    if stale_warning:
        us_summary += f" [⚠️ {sig_date} 기준, 최신 아님]"

    # ── L2: 시장 인텔리전스 ──
    intel = _load_json(DATA_DIR / "market_intelligence.json")
    kr_forecast = intel.get("kr_open_forecast", "보합")
    kr_reason = intel.get("kr_forecast_reason", "")
    key_events = intel.get("key_events", [])

    # ── L3: 레짐/매크로 ──
    regime_data = _load_json(DATA_DIR / "regime_macro_signal.json")
    current_regime = regime_data.get("current_regime", "UNKNOWN")
    macro_grade = regime_data.get("macro_grade", "")

    # ── KOSPI 확률 ──
    prob = _calc_kospi_probability(sig)

    # 시장 상태 판정
    market_status = _determine_market_status(prob, current_regime)

    # KR 요약
    dir_label = (
        "상승 우세" if prob["up_prob"] >= 60 else
        "하락 우세" if prob["down_prob"] >= 60 else
        "보합권"
    )
    kr_summary = f"KOSPI 상승확률 {prob['up_prob']}% ({dir_label}), 레짐 {current_regime}"

    # ── L4: AI 전략 ──
    strat = _load_json(DATA_DIR / "ai_strategic_analysis.json")
    risk_factors = strat.get("risk_factors", [])

    # ── L5: 섹터 포커스 ──
    focus_data = _load_json(DATA_DIR / "ai_sector_focus.json")
    focus_sectors_raw = focus_data.get("focus_sectors", [])
    sector_focus = [s.get("sector", "") for s in focus_sectors_raw if s.get("sector")][:5]

    # ── L6: 추천 종목 ──
    picks_data = _load_json(DATA_DIR / "tomorrow_picks.json")
    picks = picks_data.get("picks", [])
    top_picks = [p for p in picks if p.get("grade") in ("적극매수", "매수")][:5]

    news_picks = []
    for p in top_picks:
        ticker = p.get("ticker", "")
        name = p.get("name", "")
        grade = p.get("grade", "")
        score = p.get("total_score", 0)
        sources = p.get("sources", [])
        if isinstance(sources, list) and sources:
            if isinstance(sources[0], dict):
                source_names = [s.get("source", "") for s in sources[:3]]
            else:
                source_names = [str(s) for s in sources[:3]]
        else:
            source_names = []
        news_picks.append({
            "ticker": ticker,
            "name": name,
            "grade": grade,
            "score": round(score, 1),
            "signals": source_names,
        })

    # ── L7: 증권사 리포트 ──
    morning_reports = _load_json(DATA_DIR / "morning_reports.json")
    reports = morning_reports.get("reports", [])
    buy_reports = [r for r in reports if r.get("opinion_type") == "매수"][:5]

    # ── L8: 테마 알림 ──
    theme_data = _load_json(DATA_DIR / "theme_alerts.json")
    themes = []
    if isinstance(theme_data, list):
        themes = [t.get("theme_name", "") for t in theme_data if t.get("theme_name")][:5]

    # ── 전체 브리핑 텍스트 (텔레그램 개인채널용) ──
    full_lines = _build_full_briefing(
        date_str, prob, us_grade, spy_ret, qqq_ret, dia_ret,
        sox_ret, vix, vix_status, ewy_ret, current_regime,
        market_status, sector_focus, news_picks, buy_reports,
        themes, risk_factors, key_events, kr_forecast,
    )
    briefing_full = "\n".join(full_lines)

    # ── 요약 브리핑 (FLOWX 채널용) ──
    summary_lines = _build_summary_briefing(
        date_str, prob, us_grade, spy_ret, qqq_ret, sox_ret,
        vix, market_status, news_picks[:3],
    )
    briefing_summary = "\n".join(summary_lines)

    return {
        "date": date_str,
        "market_status": market_status,
        "us_summary": us_summary,
        "kr_summary": kr_summary,
        "kospi_prob": prob,
        "news_picks": news_picks,
        "sector_focus": sector_focus,
        "briefing_full": briefing_full,
        "briefing_summary": briefing_summary,
        # 추가 메타데이터
        "us_grade": us_grade,
        "regime": current_regime,
        "macro_grade": macro_grade,
        "themes": themes,
        "risk_factors": risk_factors[:3],
        "broker_reports": [
            {"company": r.get("company", ""), "broker": r.get("broker", ""),
             "target_price": r.get("target_price", "")}
            for r in buy_reports
        ],
    }


# ── 전체 브리핑 텍스트 ──────────────────────────

def _build_full_briefing(
    date_str, prob, us_grade, spy_ret, qqq_ret, dia_ret,
    sox_ret, vix, vix_status, ewy_ret, regime,
    market_status, sectors, picks, reports, themes,
    risks, events, kr_forecast,
) -> list[str]:
    L = []
    L.append(f"📊 FLOWX 모닝 브리핑 | {date_str}")
    L.append("━" * 28)

    # 1. KOSPI 예측
    if prob["down_prob"] >= 60:
        icon, label = "🔴", "하락 우세"
    elif prob["up_prob"] >= 60:
        icon, label = "🟢", "상승 우세"
    elif prob["down_prob"] >= 55:
        icon, label = "🟠", "하락 소폭"
    elif prob["up_prob"] >= 55:
        icon, label = "🟢", "상승 소폭"
    else:
        icon, label = "⚪", "보합권"

    filled = round(prob["up_prob"] / 100 * 15)
    bar = "█" * filled + "░" * (15 - filled)
    L.append(f"\n{icon} KOSPI {label}")
    L.append(f"  상승 {prob['up_prob']}% {bar} 하락 {prob['down_prob']}%")
    L.append(f"  레인지: {prob['est_low']:+.1f}% ~ {prob['est_high']:+.1f}%")

    # 2. US 야간
    L.append(f"\n🌍 US 야간 | Signal: {us_grade}")
    L.append(f"  SPY {spy_ret:+.1f}% | QQQ {qqq_ret:+.1f}% | DIA {dia_ret:+.1f}%")
    L.append(f"  SOXX {sox_ret:+.1f}% | EWY {ewy_ret:+.1f}%")
    vix_icon = "⚠️" if vix >= 20 else "✅"
    L.append(f"  VIX {vix:.1f} {vix_icon} {vix_status}")

    # 3. 레짐 + 개장 전망
    L.append(f"\n📋 레짐: {regime} | 개장전망: {kr_forecast}")

    # 4. 리스크
    if risks:
        L.append(f"\n🚨 리스크")
        for r in risks[:3]:
            L.append(f"  🔴 {r}")

    # 5. 주요 이벤트
    high_events = [e for e in events if e.get("impact") == "high"][:3]
    if high_events:
        L.append(f"\n⚡ 주요 이벤트")
        for e in high_events:
            L.append(f"  • {e.get('event', '')}")

    # 6. 섹터 포커스
    if sectors:
        L.append(f"\n🎯 섹터 포커스: {' | '.join(sectors)}")

    # 7. AI 추천 종목
    if picks:
        L.append(f"\n💰 AI 추천 TOP {len(picks)}")
        for i, p in enumerate(picks, 1):
            sigs = ", ".join(p.get("signals", [])[:2])
            L.append(f"  {i}. {p['name']}({p['ticker']}) [{p['grade']}] {p['score']}점")
            if sigs:
                L.append(f"     시그널: {sigs}")

    # 8. 증권사 리포트
    if reports:
        L.append(f"\n📝 증권사 매수의견 ({len(reports)}건)")
        for r in reports[:4]:
            target = r.get("target_price", "")
            target_str = f" 목표{target}" if target else ""
            L.append(f"  • {r.get('company', '')}{target_str} ({r.get('broker', '')})")

    # 9. 테마
    if themes:
        L.append(f"\n🔥 테마: {' | '.join(themes)}")

    # 10. 전략 요약
    L.append("")
    if prob["down_prob"] >= 60:
        L.append("➜ 갭다운 예상 — 우량주 저가매수 기회")
    elif prob["up_prob"] >= 60:
        L.append("➜ 갭업 예상 — 목표가 부근 분할매도")
    else:
        L.append("➜ 혼조 예상 — 시가 확인 후 대응")

    L.append("\n⚠️ 투자 판단은 본인 책임 | FLOWX")

    return L


# ── 요약 브리핑 (FLOWX 채널용) ─────────────────

def _build_summary_briefing(
    date_str, prob, us_grade, spy_ret, qqq_ret, sox_ret,
    vix, market_status, top3_picks,
) -> list[str]:
    """상위 3개 뉴스만 포함된 요약본."""
    L = []
    L.append(f"📊 FLOWX 모닝 | {date_str}")

    # 시장 상태
    status_map = {"BULL": "🟢 BULL", "BEAR": "🔴 BEAR", "CAUTION": "🟡 CAUTION", "NEUTRAL": "⚪ NEUTRAL"}
    L.append(f"시장: {status_map.get(market_status, market_status)}")

    # KOSPI
    L.append(f"KOSPI 상승확률: {prob['up_prob']}%")

    # US 1줄
    L.append(f"US: SPY{spy_ret:+.1f}% QQQ{qqq_ret:+.1f}% VIX{vix:.0f}")

    # TOP 3
    if top3_picks:
        L.append(f"\n🔥 AI 추천 TOP 3")
        for i, p in enumerate(top3_picks, 1):
            L.append(f"  {i}. {p['name']} [{p['grade']}] {p['score']}점")

    L.append("\n🔓 전체 브리핑은 FLOWX에서 확인")
    L.append("flowx.kr")

    return L


# ── CLI ──────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FLOWX 모닝 브리핑 생성")
    parser.add_argument("--dry-run", action="store_true", help="생성만 하고 업로드 안 함")
    parser.add_argument("--date", default="", help="날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    briefing = generate_morning_briefing(args.date)

    print(f"\n[브리핑 생성 완료] {briefing['date']}")
    print(f"  시장상태: {briefing['market_status']}")
    print(f"  US: {briefing['us_summary']}")
    print(f"  KR: {briefing['kr_summary']}")
    print(f"  섹터: {briefing['sector_focus']}")
    print(f"  추천: {len(briefing['news_picks'])}종목")
    print(f"  전체 브리핑: {len(briefing['briefing_full'])}자")
    print(f"  요약 브리핑: {len(briefing['briefing_summary'])}자")

    if not args.dry_run:
        # Supabase 업로드
        from src.adapters.flowx_uploader import FlowxUploader
        uploader = FlowxUploader()
        if uploader.is_active:
            ok = uploader.upload_morning_briefing(briefing)
            print(f"\n  Supabase: {'OK' if ok else 'FAIL'}")
        else:
            print("\n  [WARN] Supabase 미연결")

    print("\n[전체 브리핑]")
    print("=" * 40)
    print(briefing["briefing_full"])
    print("=" * 40)
    print("\n[요약 브리핑]")
    print(briefing["briefing_summary"])


if __name__ == "__main__":
    main()
