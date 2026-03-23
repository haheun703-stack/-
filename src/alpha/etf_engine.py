"""
ETF 추천 엔진 — "대한민국 1등 추세저격"
=========================================
개별종목과 독립된 ETF 전용 파이프라인.
시장 방향 + 섹터 흐름 + 파생 시그널을 종합하여
"지금 어떤 ETF를, 왜, 얼마나" 추천.

입력:
  brain_decision.json, overnight_signal.json, derivatives_signal.json,
  regime_macro_signal.json, active_scenarios.json, scenario_chains.json,
  investor_flow.json, etf_universe.json, active_macro_chains.json

출력:
  data/etf_recommendations.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# 유틸리티
# ============================================================

def _load_json(path: Path) -> dict | list | None:
    """JSON 안전 로드 — 없거나 깨지면 None."""
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("JSON 로드 실패 %s: %s", path.name, e)
    return None


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# ETF Direction Engine — 시장 3일 방향 판단
# ============================================================

class ETFDirectionEngine:
    """US Overnight + 파생 + 수급 + BRAIN 종합 → 시장 방향."""

    def __init__(self):
        self.overnight = _load_json(DATA_DIR / "us_market" / "overnight_signal.json") or {}
        self.derivatives = _load_json(DATA_DIR / "derivatives" / "derivatives_signal.json") or {}
        self.macro = _load_json(DATA_DIR / "regime_macro_signal.json") or {}
        self.brain = _load_json(DATA_DIR / "brain_decision.json") or {}

    def compute_market_direction(self) -> dict:
        """시장 3일 방향 판단. score: -1.0(극약세) ~ +1.0(극강세)."""
        score = 0.0
        reasons = []

        # ── 1. US Overnight (가중치 25%) ──
        combined = self.overnight.get("combined_score_100", 0)
        us_norm = max(min(combined / 100, 1.0), -1.0)
        score += us_norm * 0.25

        grade = self.overnight.get("grade", "")
        if grade in ("STRONG_BEAR", "MILD_BEAR"):
            reasons.append(f"US {grade} ({combined:.0f}점)")
        elif grade in ("STRONG_BULL", "MILD_BULL"):
            reasons.append(f"US {grade} ({combined:.0f}점)")

        # VIX
        vix = self.overnight.get("vix", {})
        vix_level = vix.get("level", 20)
        if vix_level >= 30:
            score -= 0.10
            reasons.append(f"VIX {vix_level:.1f} (극공포)")
        elif vix_level >= 25:
            score -= 0.05
            reasons.append(f"VIX {vix_level:.1f} (공포)")

        # ── 2. 파생 시그널 (가중치 30%) — 가장 선행 ──
        composite = self.derivatives.get("composite", {})
        deriv_score_raw = composite.get("score", 50)
        # score 범위: 0~100, 50 중립
        deriv_norm = (deriv_score_raw - 50) / 50  # -1 ~ +1
        score += deriv_norm * 0.30

        deriv_grade = composite.get("grade", "NEUTRAL")
        if deriv_grade != "NEUTRAL":
            reasons.append(f"파생 {deriv_grade} ({deriv_score_raw:.0f}점)")

        # 풋콜비율 세부
        pcr = self.derivatives.get("put_call_proxy", {}).get("pc_ratio", 0.3)
        if pcr > 1.0:
            score -= 0.05
            reasons.append(f"풋콜비율 {pcr:.2f} (약세)")
        elif pcr < 0.2:
            score += 0.03

        # 선물 베이시스
        basis = self.derivatives.get("futures_basis", {})
        basis_1d = basis.get("basis_1d", 0)
        if abs(basis_1d) > 0.3:
            b_contrib = max(min(basis_1d * 0.1, 0.05), -0.05)
            score += b_contrib
            if abs(basis_1d) > 0.5:
                reasons.append(f"선물베이시스 {basis_1d:.2f}%")

        # ── 3. 한국 수급 (가중치 25%) ──
        # regime_macro_signal의 signals에서 추출
        signals = self.macro.get("signals", {})
        macro_score_raw = self.macro.get("macro_score", 50)
        macro_norm = (macro_score_raw - 50) / 50  # -1 ~ +1
        score += macro_norm * 0.15

        # EWY 5일 수익률 (한국 ETF 방향)
        ewy_5d = signals.get("ewy_5d", 0)
        if ewy_5d < -5:
            score -= 0.05
            reasons.append(f"EWY 5d {ewy_5d:.1f}%")
        elif ewy_5d > 5:
            score += 0.05
            reasons.append(f"EWY 5d +{ewy_5d:.1f}%")

        # ── 4. BRAIN 레짐 (가중치 20%) ──
        regime = self.brain.get("effective_regime", "CAUTION")
        regime_map = {"BULL": 0.10, "CAUTION": 0.0, "BEAR": -0.10, "CRISIS": -0.20}
        score += regime_map.get(regime, 0) * 1.0  # 직접 반영

        if regime in ("BEAR", "CRISIS"):
            reasons.append(f"레짐 {regime}")

        # contrarian
        if self.brain.get("contrarian_opportunity"):
            reasons.append("BRAIN contrarian 감지")

        # ── 판정 ──
        if score >= 0.15:
            direction = "BULLISH"
        elif score <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "score": round(score, 3),
            "confidence": round(min(abs(score) * 5, 1.0), 2),
            "reasons": reasons,
            "holding_period": "3일" if abs(score) > 0.2 else "1~2일",
            "vix_level": vix_level,
            "regime": regime,
        }


# ============================================================
# ETF Sector Engine — 5가지 선행 지표로 섹터 전조 감지
# ============================================================

class ETFSectorEngine:
    """섹터별 추세 전조 감지 — 5가지 선행 지표."""

    def __init__(self):
        self.overnight = _load_json(DATA_DIR / "us_market" / "overnight_signal.json") or {}
        self.scenarios = _load_json(DATA_DIR / "scenarios" / "active_scenarios.json") or {}
        self.chains = _load_json(DATA_DIR / "scenarios" / "scenario_chains.json") or {}
        self.flow = _load_json(DATA_DIR / "sector_rotation" / "investor_flow.json") or {}
        self.etf_universe = _load_json(DATA_DIR / "etf_universe.json") or {}
        self.etf_volume = _load_json(DATA_DIR / "etf_volume_monitor.json") or {}
        self.macro_chains = _load_json(DATA_DIR / "active_macro_chains.json") or {}

    def compute_sector_signals(self) -> list[dict]:
        """섹터별 추세 전조 감지 → HOT 섹터 Top 3."""
        sectors: dict[str, dict] = {}

        self._indicator_1_us_relay(sectors)
        self._indicator_2_scenario_chain(sectors)
        self._indicator_3_sector_flow(sectors)
        self._indicator_4_news_sentiment(sectors)
        self._indicator_5_etf_volume(sectors)
        self._indicator_6_macro_chain(sectors)

        # 상위 3개 섹터 = HOT (최소 2개 선행지표 일치 = 40점)
        ranked = sorted(sectors.items(), key=lambda x: x[1]["score"], reverse=True)
        hot_sectors = []
        for sector, data in ranked[:3]:
            if data["score"] >= 40:
                etf = self._match_sector_to_etf(sector)
                hot_sectors.append({
                    "sector": sector,
                    "score": data["score"],
                    "reasons": data["reasons"],
                    "etf": etf,
                })

        return hot_sectors

    # ── 선행지표 1: US→KR 릴레이 (1~3일 선행) ──
    def _indicator_1_us_relay(self, sectors: dict) -> None:
        """overnight_signal의 sector_momentum에서 US 섹터 방향 추출."""
        sector_mom = self.overnight.get("sector_momentum", {})
        for sector, data in sector_mom.items():
            ret_1d = data.get("ret_1d_pct", 0)
            ret_5d = data.get("ret_5d_pct", 0)
            consecutive_up = data.get("consecutive_up", 0)

            # US 섹터가 5일 +5% 이상 또는 3일 연속 상승
            if ret_5d >= 5.0 or consecutive_up >= 3:
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] += 25
                us_etf = data.get("us_etf", "")
                sectors[sector]["reasons"].append(
                    f"US 릴레이: {us_etf} 5d {ret_5d:+.1f}%"
                )
            # US 섹터 강한 하락 → COLD (음수 점수)
            elif ret_5d <= -5.0:
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] -= 15
                sectors[sector]["reasons"].append(
                    f"US 약세: {data.get('us_etf', '')} 5d {ret_5d:.1f}%"
                )

    # ── 선행지표 2: 시나리오 체인 Phase 매칭 ──
    def _indicator_2_scenario_chain(self, sectors: dict) -> None:
        """활성 시나리오의 현재 Phase에서 HOT 섹터 추출."""
        active = self.scenarios.get("scenarios", {})
        chain_list = self.chains.get("scenarios", [])

        # 체인 ID→데이터 맵 구축
        chain_map = {c["id"]: c for c in chain_list if isinstance(c, dict) and "id" in c}

        for scenario_id, state in active.items():
            if not isinstance(state, dict):
                continue
            current_phase = state.get("current_phase", 0)
            score_val = state.get("score", 0)
            if score_val < 40:
                continue  # 약한 시나리오 무시

            chain = chain_map.get(scenario_id, {})
            phases = chain.get("chain", [])

            for phase_data in phases:
                if phase_data.get("phase") == current_phase:
                    # 같은 시나리오에서 중복 섹터 방지 (방산+방위산업ETF → 방산 1건)
                    seen_in_scenario = set()
                    for hot in phase_data.get("hot_sectors", []):
                        normalized = self._normalize_sector(hot)
                        if normalized in seen_in_scenario:
                            continue
                        seen_in_scenario.add(normalized)
                        sectors.setdefault(normalized, {"score": 0, "reasons": []})
                        sectors[normalized]["score"] += 30
                        name = state.get("name", chain.get("name", scenario_id))
                        sectors[normalized]["reasons"].append(
                            f"시나리오 {name} Phase{current_phase} ({score_val}점)"
                        )
                    break

    # ── 선행지표 3: 섹터 수급 (외인+기관 5일 누적) ──
    def _indicator_3_sector_flow(self, sectors: dict) -> None:
        """investor_flow.json에서 외인+기관 누적 순매수 상위 섹터."""
        flow_sectors = self.flow.get("sectors", [])
        for item in flow_sectors:
            sector = item.get("sector", "")
            foreign = item.get("foreign_cum_bil", 0)
            inst = item.get("inst_cum_bil", 0)
            net = foreign + inst

            if net > 500:  # 500억 이상 순유입
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] += 20
                sectors[sector]["reasons"].append(f"수급 유입 {net:.0f}억")

            # 스텔스 매집 (가격 하락 중 외인 매수)
            if item.get("stealth_buying") and foreign > 300:
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] += 10
                sectors[sector]["reasons"].append(
                    f"스텔스 매집 (외인 {foreign:.0f}억, 가격 {item.get('price_change_5', 0):+.1f}%)"
                )

    # ── 선행지표 4: 뉴스/센티먼트 (overnight sector_signals) ──
    def _indicator_4_news_sentiment(self, sectors: dict) -> None:
        """overnight_signal의 sector_signals에서 센티먼트 추출."""
        sector_signals = self.overnight.get("sector_signals", {})
        for sector, data in sector_signals.items():
            if not isinstance(data, dict):
                continue
            sig = data.get("signal", "")
            sig_score = data.get("score", 0)

            if sig in ("STRONG_BUY", "BUY") or sig_score >= 70:
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] += 15
                sectors[sector]["reasons"].append(
                    f"US 섹터 시그널 {sig} ({sig_score}점)"
                )

    # ── 선행지표 5: 섹터 ETF 거래량 폭발 (가장 비밀스러운 선행지표) ──
    def _indicator_5_etf_volume(self, sectors: dict) -> None:
        """섹터 ETF 거래량이 20일 평균의 2배 이상이면 개별종목보다 선행."""
        if not self.etf_volume:
            return

        for item in self.etf_volume.get("etfs", []):
            ratio = item.get("volume_ratio", 0)
            if ratio >= 2.0:
                sector = item.get("sector", "")
                sectors.setdefault(sector, {"score": 0, "reasons": []})
                sectors[sector]["score"] += 25
                sectors[sector]["reasons"].append(
                    f"ETF 거래량 {ratio:.1f}배 폭발 ({item.get('name', '')})"
                )

    # ── 선행지표 6: 매크로 체인 (원자재/환율/금리 연동) ──
    def _indicator_6_macro_chain(self, sectors: dict) -> None:
        """활성 매크로 체인의 수혜/피해 섹터 점수 반영."""
        active = self.macro_chains.get("active_chains", [])
        if not active:
            return

        for chain in active:
            chain_name = chain.get("name", "")

            # 수혜 섹터 +20
            for b in chain.get("beneficiaries", []):
                if b.get("type") == "sector":
                    sector = self._normalize_sector(b.get("sector", ""))
                    if not sector:
                        continue
                    sectors.setdefault(sector, {"score": 0, "reasons": []})
                    sectors[sector]["score"] += 20
                    sectors[sector]["reasons"].append(
                        f"매크로 수혜: {chain_name} → {b.get('reason', '')}"
                    )

            # 피해 섹터 -15
            for v in chain.get("victims", []):
                if v.get("type") == "sector":
                    sector = self._normalize_sector(v.get("sector", ""))
                    if not sector:
                        continue
                    sectors.setdefault(sector, {"score": 0, "reasons": []})
                    sectors[sector]["score"] -= 15
                    sectors[sector]["reasons"].append(
                        f"매크로 피해: {chain_name} → {v.get('reason', '')}"
                    )

    # ── 헬퍼 ──
    def _normalize_sector(self, raw: str) -> str:
        """비표준 섹터명 정규화."""
        mapping = {
            "방위산업ETF": "방산",
            "방위산업": "방산",
            "정유": "에너지",
            "에너지화학": "에너지",
            "금": "금",
            "달러": "달러",
            "채권": "채권",
        }
        return mapping.get(raw, raw)

    def _match_sector_to_etf(self, sector: str) -> dict | None:
        """섹터명 → etf_universe에서 매칭되는 ETF 반환."""
        for etf in self.etf_universe.get("sector", []):
            if etf.get("sector") == sector:
                return etf
        # safe_haven에서도 검색
        for etf in self.etf_universe.get("safe_haven", []):
            if etf.get("sector") == sector:
                return etf
        return None


# ============================================================
# ETF Recommendation Generator — 방향 + 섹터 → 구체적 ETF 추천
# ============================================================

class ETFRecommendationEngine:
    """방향 판단 + 섹터 판단을 합쳐서 구체적 ETF 추천 생성."""

    def __init__(self):
        self.direction_engine = ETFDirectionEngine()
        self.sector_engine = ETFSectorEngine()
        self.etf_universe = _load_json(DATA_DIR / "etf_universe.json") or {}
        self.brain = _load_json(DATA_DIR / "brain_decision.json") or {}
        self.macro_chains = _load_json(DATA_DIR / "active_macro_chains.json") or {}

    def generate_recommendations(self) -> dict:
        """ETF 추천 생성 → data/etf_recommendations.json 저장."""
        direction = self.direction_engine.compute_market_direction()
        hot_sectors = self.sector_engine.compute_sector_signals()

        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "market_direction": direction,
            "hot_sectors": hot_sectors,
            "etf_picks": [],
        }

        # ── 1. 인버스 추천 (BEARISH) ──
        if direction["direction"] == "BEARISH":
            confidence = direction["confidence"]
            if confidence >= 0.5:
                # 강한 약세 → 인버스 2X, 약한 약세 → 인버스 1X
                if confidence >= 0.7:
                    inv = self._find_etf("inverse", "inverse_2x")
                    pct = 10
                else:
                    inv = self._find_etf("inverse", "inverse_1x", underlying="KOSPI200")
                    pct = 7

                if inv:
                    recommendations["etf_picks"].append({
                        "category": "인버스",
                        "ticker": inv["ticker"],
                        "name": inv["name"],
                        "action": "BUY",
                        "reason": direction["reasons"],
                        "confidence": confidence,
                        "holding_period": "1~3일",
                        "portfolio_pct": pct,
                        "stop_loss": "기초지수 +2% 반등 시",
                        "target": "기초지수 -3% 추가 하락 시 익절",
                    })

        # ── 2. 레버리지 추천 (BULLISH + 고신뢰) ──
        elif direction["direction"] == "BULLISH" and direction["confidence"] >= 0.6:
            lev = self._find_etf("leverage", "leverage_2x", underlying="KOSPI200")
            if lev:
                recommendations["etf_picks"].append({
                    "category": "레버리지",
                    "ticker": lev["ticker"],
                    "name": lev["name"],
                    "action": "BUY",
                    "reason": direction["reasons"],
                    "confidence": direction["confidence"],
                    "holding_period": "3~5일",
                    "portfolio_pct": 10,
                    "stop_loss": "기초지수 -1.5% 하락 시",
                    "target": "기초지수 +2% 상승 시 익절",
                })

        # ── 3. 섹터 ETF 추천 (HOT 섹터) ──
        for sector_info in hot_sectors:
            etf = sector_info.get("etf")
            if etf:
                recommendations["etf_picks"].append({
                    "category": "섹터",
                    "ticker": etf["ticker"],
                    "name": etf["name"],
                    "action": "BUY",
                    "reason": sector_info["reasons"],
                    "confidence": round(min(sector_info["score"] / 100, 1.0), 2),
                    "holding_period": "5~10일",
                    "portfolio_pct": 8,
                    "stop_loss": "섹터 수급 이탈 시",
                    "target": "시나리오 Phase 전환 시 재평가",
                })

        # ── 4. 안전자산 ETF (BEARISH + VIX 높을 때) ──
        regime = self.brain.get("effective_regime", "CAUTION")
        if direction["direction"] == "BEARISH" or regime in ("BEAR", "CRISIS"):
            vix = direction.get("vix_level", 20)
            if vix >= 25 or self.brain.get("contrarian_opportunity"):
                gold = self._find_etf("safe_haven", "gold")
                if gold:
                    recommendations["etf_picks"].append({
                        "category": "헤지",
                        "ticker": gold["ticker"],
                        "name": gold["name"],
                        "action": "BUY",
                        "reason": [
                            f"VIX {vix:.1f}",
                            f"레짐 {regime}",
                            "안전자산 수요",
                        ],
                        "confidence": 0.7,
                        "holding_period": "보유 유지",
                        "portfolio_pct": 5,
                        "stop_loss": "VIX 20 이하 안정 시 축소",
                        "target": "위기 지속 시 비중 확대",
                    })

        # ── 5. 매크로 체인 직접 ETF 추천 (원자재/환율 수혜 ETF) ──
        for chain in self.macro_chains.get("active_chains", []):
            chain_name = chain.get("name", "")
            for b in chain.get("beneficiaries", []):
                if b.get("type") == "etf":
                    ticker = b.get("ticker", "")
                    if ticker:
                        recommendations["etf_picks"].append({
                            "category": "매크로",
                            "ticker": ticker,
                            "name": b.get("name", ticker),
                            "action": "BUY",
                            "reason": [f"매크로: {chain_name}", b.get("reason", "")],
                            "confidence": 0.65,
                            "holding_period": "3~7일",
                            "portfolio_pct": 5,
                            "stop_loss": "매크로 조건 해소 시",
                            "target": "추세 지속 시 유지",
                        })

        # BRAIN ARM 비중 참조 — etf_dollar ARM이 높으면 달러 ETF 추가
        for arm in self.brain.get("arms", []):
            if arm.get("name") == "etf_dollar" and arm.get("adjusted_pct", 0) >= 8:
                dollar = self._find_etf("safe_haven", "dollar")
                if dollar:
                    recommendations["etf_picks"].append({
                        "category": "헤지",
                        "ticker": dollar["ticker"],
                        "name": dollar["name"],
                        "action": "BUY",
                        "reason": [f"BRAIN etf_dollar {arm['adjusted_pct']:.0f}%", "원화 약세 헤지"],
                        "confidence": 0.6,
                        "holding_period": "보유 유지",
                        "portfolio_pct": arm["adjusted_pct"],
                        "stop_loss": "USD/KRW 하락 전환 시",
                        "target": "원화 약세 지속 시 유지",
                    })

        # ── 중복 제거 (같은 ticker 2번 추천 방지) ──
        seen = set()
        unique_picks = []
        for pick in recommendations["etf_picks"]:
            if pick["ticker"] not in seen:
                seen.add(pick["ticker"])
                unique_picks.append(pick)
        recommendations["etf_picks"] = unique_picks

        # ── 저장 ──
        out_path = DATA_DIR / "etf_recommendations.json"
        _save_json(out_path, recommendations)
        logger.info("ETF 추천 %d건 생성 → %s", len(unique_picks), out_path)

        return recommendations

    def _find_etf(self, category: str, etf_type: str, underlying: str = None) -> dict | None:
        """유니버스에서 카테고리+타입으로 ETF 검색."""
        for etf in self.etf_universe.get(category, []):
            if etf.get("type") == etf_type:
                if underlying and etf.get("underlying") != underlying:
                    continue
                return etf
        # 타입 무시하고 첫 번째 반환 (fallback)
        items = self.etf_universe.get(category, [])
        return items[0] if items else None


# ============================================================
# 텔레그램 포맷터
# ============================================================

def format_etf_telegram(recommendations: dict) -> str:
    """ETF 추천 결과를 텔레그램 메시지 문자열로 변환."""
    direction = recommendations.get("market_direction", {})
    picks = recommendations.get("etf_picks", [])

    if not picks:
        return ""

    lines = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"━━━ ETF 추천 ({today}) ━━━")
    lines.append("")

    # 시장 방향 요약
    dir_str = direction.get("direction", "NEUTRAL")
    dir_score = direction.get("score", 0)
    dir_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(dir_str, "⚪")
    lines.append(f"{dir_emoji} 시장방향: {dir_str} (점수 {dir_score:+.2f})")

    dir_reasons = direction.get("reasons", [])
    if dir_reasons:
        lines.append(f"  근거: {', '.join(dir_reasons[:3])}")
    lines.append("")

    # ETF 추천 목록
    cat_emoji = {
        "인버스": "🔴",
        "레버리지": "🟢",
        "섹터": "🟢",
        "매크로": "🟠",
        "헤지": "🟡",
    }

    for pick in picks:
        cat = pick.get("category", "")
        emoji = cat_emoji.get(cat, "⚪")
        lines.append(f"{emoji} {cat} | {pick['name']} ({pick['ticker']})")

        # 근거
        reasons = pick.get("reason", [])
        if isinstance(reasons, list):
            reason_str = ", ".join(str(r) for r in reasons[:3])
        else:
            reason_str = str(reasons)
        lines.append(f"  근거: {reason_str}")

        # 목표/비중
        lines.append(
            f"  목표: {pick.get('holding_period', '?')}, "
            f"포트 {pick.get('portfolio_pct', 0)}%"
        )
        lines.append(f"  손절: {pick.get('stop_loss', '-')}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI 실행 — BAT-D step 20에서 호출
# ============================================================

def run_etf_engine() -> dict:
    """ETF 추천 엔진 실행 + 매크로 체인 감지 + 텔레그램 3단 발송."""
    print("=" * 50)
    print("ETF 추천 엔진 실행 (매크로 체인 통합)")
    print("=" * 50)

    # ── 1단계: 매크로 체인 감지 (가장 먼저!) ──
    from src.alpha.macro_chain_detector import MacroChainDetector, format_macro_telegram
    detector = MacroChainDetector()
    active_chains = detector.detect()
    print(f"\n매크로 체인: {len(active_chains)}건 활성")
    for chain in active_chains:
        print(f"  [{chain['id']}] {chain['name']}")

    # ── 2단계: ETF 추천 엔진 ──
    engine = ETFRecommendationEngine()
    result = engine.generate_recommendations()

    direction = result["market_direction"]
    picks = result["etf_picks"]
    hot = result.get("hot_sectors", [])

    print(f"\n시장 방향: {direction['direction']} (점수 {direction['score']:+.3f})")
    if direction["reasons"]:
        print(f"  근거: {', '.join(direction['reasons'])}")

    if hot:
        labels = [f"{h['sector']}({h['score']}점)" for h in hot]
        print(f"\nHOT 섹터: {', '.join(labels)}")

    print(f"\nETF 추천 {len(picks)}건:")
    for p in picks:
        print(f"  [{p['category']}] {p['name']} ({p['ticker']}) — 포트 {p['portfolio_pct']}%")
        reasons = p.get("reason", [])
        if isinstance(reasons, list):
            print(f"    근거: {', '.join(str(r) for r in reasons[:3])}")
        else:
            print(f"    근거: {reasons}")

    # ── 3단계: 텔레그램 3단 발송 (매크로→ETF→종목) ──
    telegram_parts = []

    # 1단: 매크로 시그널
    macro_text = format_macro_telegram(active_chains)
    if macro_text:
        telegram_parts.append(macro_text)

    # 2단: ETF 추천
    etf_text = format_etf_telegram(result)
    if etf_text:
        telegram_parts.append(etf_text)

    # 3단 합체 후 발송
    if telegram_parts:
        combined = "\n\n".join(telegram_parts)
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.telegram_sender import send_message
            send_message(combined)
            print("\n텔레그램 3단 발송 완료 (매크로→ETF)")
        except Exception as e:
            print(f"\n텔레그램 발송 실패: {e}")
    else:
        print("\n시그널 없음 — 텔레그램 미발송")

    return result


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_etf_engine()
