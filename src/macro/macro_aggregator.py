"""MacroAggregator — 30+ JSON 시그널 파일을 하나의 컨텍스트로 통합

Master Brain에 전달할 데이터를 수집/요약합니다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class MacroAggregator:
    """모든 시그널 JSON을 수집하여 단일 dict로 반환."""

    SOURCES = {
        # ── 거시 ──
        "overnight": "us_market/overnight_signal.json",
        "regime_macro": "regime_macro_signal.json",
        "cpi": "macro/cpi_data.json",
        "cot": "cot/cot_signal.json",
        "china_money": "china_money/china_money_signal.json",
        "liquidity": "liquidity_cycle/liquidity_signal.json",
        # ── 개별종목 ──
        "tomorrow_picks": "tomorrow_picks.json",
        "ai_v3_picks": "ai_v3_picks.json",
        "ai_brain_judgment": "ai_brain_judgment.json",
        "ai_strategic_analysis": "ai_strategic_analysis.json",
        "consensus": "consensus_screening.json",
        "pullback": "pullback_scan.json",
        "volume_spike": "volume_spike_watchlist.json",
        "whale_detect": "whale_detect.json",
        "dual_buying": "dual_buying_watch.json",
        "dart_events": "dart_event_signals.json",
        "institutional": "institutional_targets.json",
        "accumulation": "accumulation_tracker.json",
        # ── ETF ──
        "etf_rotation": "etf_rotation_result.json",
        "sector_momentum": "sector_rotation/sector_momentum.json",
        "etf_leverage": "leverage_etf/leverage_etf_scan.json",
        "etf_investor_flow": "etf_investor_flow.json",
        "relay_signal": "relay/relay_signal.json",
        "group_relay": "group_relay/group_relay_today.json",
        "value_chain": "value_chain_relay.json",
        # ── AI 판단 ──
        "brain_decision": "brain_decision.json",
        "shield_report": "shield_report.json",
        "portfolio_allocation": "portfolio_allocation.json",
        "force_hybrid": "force_hybrid.json",
        # ── 포트폴리오 ──
        "positions": "paper_portfolio.json",
        "equity": "equity_tracker.json",
        "daily_performance": "daily_performance.json",
    }

    def aggregate(self) -> dict:
        """모든 소스를 읽어 하나의 dict로 반환. 없는 파일은 skip."""
        result = {}
        for key, rel_path in self.SOURCES.items():
            path = DATA_DIR / rel_path
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    result[key] = data
                except Exception as e:
                    logger.warning(f"  {key} ({rel_path}): 파싱 실패 — {e}")
                    result[key] = None
            else:
                result[key] = None
        loaded = sum(1 for v in result.values() if v is not None)
        logger.info(f"MacroAggregator: {loaded}/{len(self.SOURCES)} 소스 로드")
        return result

    def summarize_for_prompt(self, data: dict) -> str:
        """Claude 프롬프트용 텍스트 요약 (토큰 절약)."""
        # 소스 키 → summarizer 매핑
        summarizers = {
            # 거시
            "overnight": self._summarize_overnight,
            "regime_macro": self._summarize_regime,
            "cpi": self._summarize_cpi,
            "cot": self._summarize_cot,
            "china_money": self._summarize_china,
            "liquidity": self._summarize_liquidity,
            # 개별종목
            "tomorrow_picks": self._summarize_picks,
            "ai_v3_picks": self._summarize_v3,
            "ai_brain_judgment": self._summarize_ai_brain,
            "ai_strategic_analysis": self._summarize_strategic,
            "consensus": self._summarize_consensus,
            "pullback": self._summarize_pullback,
            "volume_spike": self._summarize_volume_spike,
            "whale_detect": self._summarize_whale,
            "dual_buying": self._summarize_dual_buying,
            "dart_events": self._summarize_dart,
            "institutional": self._summarize_institutional,
            "accumulation": self._summarize_accumulation,
            # ETF
            "etf_rotation": self._summarize_etf,
            "sector_momentum": self._summarize_momentum,
            "etf_leverage": self._summarize_leverage,
            "relay_signal": self._summarize_relay,
            "group_relay": self._summarize_group_relay,
            "value_chain": self._summarize_value_chain,
            # AI/포트폴리오
            "brain_decision": self._summarize_brain,
            "shield_report": self._summarize_shield,
            "portfolio_allocation": self._summarize_allocation,
            "force_hybrid": self._summarize_force_hybrid,
            "positions": self._summarize_positions,
            "equity": self._summarize_equity,
            "daily_performance": self._summarize_daily_perf,
        }

        sections = []
        for key, fn in summarizers.items():
            d = data.get(key)
            if d:
                try:
                    s = fn(d)
                    if s:
                        sections.append(s)
                except Exception as e:
                    logger.warning(f"summarize {key} 실패: {e}")

        return "\n\n".join(sections)

    # ══════════════════════════════════════════════════════════
    # 거시 (Macro)
    # ══════════════════════════════════════════════════════════

    def _summarize_overnight(self, d: dict) -> str:
        grade = d.get("grade", "?")
        score = d.get("combined_score_100", 0)
        summary = d.get("summary", "")
        shock = d.get("shock_type", {})
        shock_type = shock.get("shock_type", "NONE") if isinstance(shock, dict) else str(shock)

        rules = d.get("special_rules", [])
        rule_names = [r.get("name", "") for r in rules] if rules else []

        commod = d.get("commodities", {})
        commod_parts = []
        for key in ["gold", "oil", "copper", "silver", "natgas", "uranium"]:
            c = commod.get(key, {})
            if c and "ret_1d" in c:
                commod_parts.append(f"{key}:{c['ret_1d']:+.1f}%")

        vix = d.get("vix", {})
        vix_str = ""
        if isinstance(vix, dict):
            vix_str = f"VIX: {vix.get('close', '?')} ({vix.get('ret_1d', 0)*100:+.1f}%)"
        elif isinstance(vix, (int, float)):
            vix_str = f"VIX: {vix}"

        lines = ["## US Overnight Signal"]
        lines.append(f"등급: {grade} (점수: {score:+.1f})")
        if vix_str:
            lines.append(vix_str)
        if shock_type not in ("NONE", "None", ""):
            lines.append(f"충격: {shock_type}")
        if rule_names:
            lines.append(f"특수룰: {', '.join(rule_names)}")
        if commod_parts:
            lines.append(f"원자재: {', '.join(commod_parts)}")
        lines.append(f"요약: {summary}")
        return "\n".join(lines)

    def _summarize_regime(self, d: dict) -> str:
        regime = d.get("current_regime", d.get("regime", "?"))
        score = d.get("macro_score", d.get("total_score", 0))
        grade = d.get("macro_grade", "?")
        multiplier = d.get("position_multiplier", "?")
        transition = d.get("transition_direction", "?")
        rec = d.get("recommendation", "")
        signals = d.get("signals", {})
        sig_parts = []
        if isinstance(signals, dict):
            for k, v in list(signals.items())[:6]:
                sig_parts.append(f"{k}={v}")
        sig_str = ", ".join(sig_parts)
        lines = ["## 매크로 레짐"]
        lines.append(f"레짐: {regime} (등급: {grade}, 점수: {score})")
        lines.append(f"포지션 배수: {multiplier}, 전환 방향: {transition}")
        if sig_str:
            lines.append(f"시그널: {sig_str}")
        if rec:
            lines.append(f"추천: {rec}")
        return "\n".join(lines)

    def _summarize_cpi(self, d: dict) -> str:
        cpi = d.get("cpi_yoy", "?")
        pce = d.get("core_pce_yoy", "?")
        unemp = d.get("unemployment_rate", "?")
        stag = d.get("stagflation_signal", "?")
        desc = d.get("stagflation_description", "")
        fed = d.get("fed_funds_rate", "?")
        cpi_trend = d.get("cpi_trend", "?")
        return (
            f"## CPI/인플레이션\n"
            f"CPI YoY: {cpi}%, Core PCE: {pce}%, 실업률: {unemp}%\n"
            f"기준금리: {fed}%, CPI 추세: {cpi_trend}\n"
            f"스태그플레이션: {stag} — {desc}"
        )

    def _summarize_cot(self, d: dict) -> str:
        direction = d.get("composite_direction", "?")
        score = d.get("composite_score", 0)
        signals = d.get("signals", {})
        sig_parts = []
        if isinstance(signals, dict):
            for k, v in signals.items():
                if isinstance(v, bool):
                    sig_parts.append(f"{k}={'Y' if v else 'N'}")
                else:
                    sig_parts.append(f"{k}={v}")
        contracts = d.get("contracts", {})
        con_parts = []
        if isinstance(contracts, dict):
            for name, info in contracts.items():
                if isinstance(info, dict):
                    pos = info.get("net_position_chg", info.get("net_change", "?"))
                    con_parts.append(f"{name}:{pos:+.0f}" if isinstance(pos, (int, float)) else f"{name}:{pos}")
        lines = ["## COT 포지셔닝"]
        lines.append(f"방향: {direction} (점수: {score})")
        if sig_parts:
            lines.append(f"시그널: {', '.join(sig_parts)}")
        if con_parts:
            lines.append(f"계약: {', '.join(con_parts)}")
        return "\n".join(lines)

    def _summarize_china(self, d: dict) -> str:
        summary = d.get("summary", "")
        total = d.get("total_stocks", 0)
        signals = d.get("signals", [])
        top_buyers = d.get("top_foreign_buyers", [])
        lines = ["## 중국/외국인 자금 흐름"]
        lines.append(f"분석 종목: {total}개")
        if summary:
            lines.append(f"요약: {summary}")
        # 시그널 발생 종목
        if signals:
            sig_names = [f"{s.get('name','?')}({s.get('signal','?')})" for s in signals[:5]]
            lines.append(f"시그널: {', '.join(sig_names)}")
        # 외국인 순매수 상위
        if top_buyers:
            buy_names = []
            for b in top_buyers[:3]:
                if isinstance(b, dict):
                    buy_names.append(b.get("name", "?"))
                else:
                    buy_names.append(str(b))
            if buy_names:
                lines.append(f"외국인 순매수 상위: {', '.join(buy_names)}")
        return "\n".join(lines)

    def _summarize_liquidity(self, d: dict) -> str:
        regime = d.get("regime", "?")
        direction = d.get("composite_direction", "?")
        score = d.get("composite_score", 0)
        signals = d.get("signals", {})
        sig_parts = []
        if isinstance(signals, dict):
            for k, v in signals.items():
                if isinstance(v, bool):
                    sig_parts.append(f"{k}={'Y' if v else 'N'}")
        indicators = d.get("indicators", {})
        ind_parts = []
        if isinstance(indicators, dict):
            nl = indicators.get("net_liquidity", {})
            if isinstance(nl, dict) and "value" in nl:
                ind_parts.append(f"순유동성:{nl['value']/1e12:.2f}T")
            m2 = indicators.get("m2_yoy_pct", {})
            if isinstance(m2, dict) and "value" in m2:
                ind_parts.append(f"M2 YoY:{m2['value']:.1f}%")
        lines = ["## 유동성 사이클"]
        lines.append(f"레짐: {regime}, 방향: {direction} (점수: {score})")
        if sig_parts:
            lines.append(f"시그널: {', '.join(sig_parts)}")
        if ind_parts:
            lines.append(f"지표: {', '.join(ind_parts)}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    # 개별종목
    # ══════════════════════════════════════════════════════════

    def _summarize_picks(self, d) -> str:
        if isinstance(d, dict):
            picks_list = d.get("picks", d.get("top5", []))
        elif isinstance(d, list):
            picks_list = d
        else:
            return ""
        if not picks_list:
            return ""
        lines = ["## 내일 추천 (상위 5)"]
        for p in picks_list[:5]:
            name = p.get("name", "?")
            score = p.get("total_score", 0)
            grade = p.get("grade", "?")
            sources = p.get("sources", [])
            strategy = p.get("strategy", "")
            src_str = f" [{','.join(sources[:3])}]" if sources else ""
            strat_str = f" ({strategy})" if strategy else ""
            lines.append(f"  {name}: {score}점 {grade}등급{src_str}{strat_str}")
        return "\n".join(lines)

    def _summarize_v3(self, d: dict) -> str:
        regime = d.get("regime", "?")
        slots = d.get("available_slots", "?")
        buys = d.get("buys", [])
        skipped = d.get("skipped", [])
        reasoning = d.get("reasoning", "")
        lines = ["## v3 AI 매매 판단"]
        lines.append(f"레짐: {regime}, 가용 슬롯: {slots}")
        if buys:
            for b in buys[:3]:
                name = b.get("name", "?")
                lines.append(f"  매수: {name}")
        else:
            lines.append("  매수: 없음")
        if skipped:
            skip_names = [f"{s.get('name','?')}({s.get('skip_reason','?')[:20]})" for s in skipped[:3]]
            lines.append(f"  스킵: {', '.join(skip_names)}")
        if reasoning:
            lines.append(f"  판단: {str(reasoning)[:150]}")
        return "\n".join(lines)

    def _summarize_ai_brain(self, d: dict) -> str:
        sentiment = d.get("market_sentiment", "?")
        themes = d.get("key_themes", [])
        judgments = d.get("stock_judgments", [])
        sector_outlook = d.get("sector_outlook", {})

        lines = ["## AI Brain 판단"]
        lines.append(f"시장 심리: {sentiment}")
        if themes:
            theme_str = ", ".join(themes[:5]) if isinstance(themes, list) else str(themes)[:100]
            lines.append(f"핵심 테마: {theme_str}")
        # 종목 판단 상위 5
        if judgments:
            for j in judgments[:5]:
                name = j.get("name", "?")
                action = j.get("action", "?")
                conf = j.get("confidence", 0)
                reason = j.get("reasoning", "")[:60]
                lines.append(f"  {name}: {action} (신뢰도:{conf}) — {reason}")
        # 섹터 전망
        if sector_outlook and isinstance(sector_outlook, dict):
            outlook_parts = []
            for sector, info in list(sector_outlook.items())[:6]:
                if isinstance(info, dict):
                    outlook = info.get("outlook", info.get("action", "?"))
                    outlook_parts.append(f"{sector}:{outlook}")
                else:
                    outlook_parts.append(f"{sector}:{info}")
            if outlook_parts:
                lines.append(f"섹터전망: {', '.join(outlook_parts)}")
        return "\n".join(lines)

    def _summarize_strategic(self, d: dict) -> str:
        regime = d.get("regime", d.get("market_regime", "?"))
        confidence = d.get("regime_confidence", "?")
        sectors = d.get("sector_priority", [])
        max_buys = d.get("max_new_buys", "?")
        cash_sug = d.get("cash_reserve_suggestion", "?")
        summary = d.get("global_summary", "")
        risks = d.get("risk_factors", [])

        lines = ["## 전략 분석"]
        lines.append(f"레짐: {regime} (신뢰도: {confidence})")
        lines.append(f"최대 신규 매수: {max_buys}건, 현금 추천: {cash_sug}%")
        if isinstance(sectors, list) and sectors:
            sec_str = ", ".join(
                s if isinstance(s, str) else s.get("name", s.get("sector", "?"))
                for s in sectors[:5]
            )
            lines.append(f"섹터 우선순위: {sec_str}")
        if risks:
            risk_str = "; ".join(str(r)[:50] for r in risks[:3])
            lines.append(f"리스크: {risk_str}")
        if summary:
            lines.append(f"요약: {str(summary)[:150]}")
        return "\n".join(lines)

    def _summarize_consensus(self, d) -> str:
        if isinstance(d, dict):
            all_picks = d.get("all_picks", d.get("top_picks", []))
            total = d.get("universe_size", len(all_picks))
            passed = d.get("passed_filter", len(all_picks))
        elif isinstance(d, list):
            all_picks = d
            total = len(d)
            passed = total
        else:
            return ""
        lines = [f"## 컨센서스 풀"]
        lines.append(f"유니버스: {total}종목, 필터 통과: {passed}종목")
        for p in all_picks[:5]:
            name = p.get("name", "?")
            upside = p.get("upside_pct", "?")
            grade = p.get("grade", "?")
            fper = p.get("forward_per", "?")
            lines.append(f"  {name}: 상승여력 {upside}%, 등급 {grade}, F-PER {fper}")
        return "\n".join(lines)

    def _summarize_pullback(self, d: dict) -> str:
        candidates = d.get("candidates", [])
        uptrend = d.get("uptrend_count", 0)
        grades = d.get("grade_counts", {})
        if not candidates:
            return ""
        lines = ["## 눌림목 후보"]
        lines.append(f"상승추세: {uptrend}종목, 등급분포: {grades}")
        for c in candidates[:5]:
            name = c.get("name", "?")
            score = c.get("score", 0)
            grade = c.get("grade", "?")
            rsi = c.get("rsi", "?")
            reasons = c.get("reasons", [])
            reason_str = f" [{', '.join(reasons[:2])}]" if reasons else ""
            lines.append(f"  {name}: {score}점 {grade}등급 RSI={rsi}{reason_str}")
        return "\n".join(lines)

    def _summarize_volume_spike(self, d: dict) -> str:
        signals = d.get("signals", [])
        regime = d.get("regime", "?")
        gate = d.get("regime_gate", True)
        stats = d.get("stats", {})
        if not signals:
            return ""
        lines = ["## 거래량 급증"]
        lines.append(f"레짐: {regime}, 게이트: {'OPEN' if gate else 'CLOSED'}")
        if stats:
            lines.append(f"통계: {stats}")
        for s in signals[:5]:
            if isinstance(s, dict):
                name = s.get("name", "?")
                pullback = s.get("pullback_pct", "?")
                days = s.get("days_since_spike", "?")
                vol_z = s.get("vol_z_at_spike", "?")
                score = s.get("score", "?")
                lines.append(f"  {name}: 눌림 {pullback}%, {days}일차, Z={vol_z}, 점수:{score}")
        return "\n".join(lines)

    def _summarize_whale(self, d) -> str:
        if isinstance(d, list):
            items = d
        elif isinstance(d, dict):
            items = d.get("items", d.get("whales", d.get("detected", [])))
        else:
            return ""
        if not items:
            return ""
        total = d.get("total_detected", len(items)) if isinstance(d, dict) else len(items)
        lines = [f"## 고래 감지 ({total}건)"]
        for w in items[:5]:
            name = w.get("name", "?")
            vol = w.get("volume_surge_ratio", w.get("vol_ratio", "?"))
            strength = w.get("strength", "?")
            grade = w.get("grade", "?")
            lines.append(f"  {name}: 거래량비 {vol}, 강도 {strength}, 등급 {grade}")
        return "\n".join(lines)

    def _summarize_dual_buying(self, d: dict) -> str:
        s_grade = d.get("s_grade", [])
        a_grade = d.get("a_grade", [])
        b_grade = d.get("b_grade", [])
        core_watch = d.get("core_watch", [])
        if not s_grade and not a_grade and not b_grade:
            return ""
        lines = ["## 기관+외인 쌍끌이"]
        if s_grade:
            names = [x.get("name", "?") if isinstance(x, dict) else str(x) for x in s_grade[:3]]
            lines.append(f"  S등급: {', '.join(names)}")
        if a_grade:
            names = [x.get("name", "?") if isinstance(x, dict) else str(x) for x in a_grade[:3]]
            lines.append(f"  A등급: {', '.join(names)}")
        if b_grade:
            names = [x.get("name", "?") if isinstance(x, dict) else str(x) for x in b_grade[:5]]
            lines.append(f"  B등급: {', '.join(names)}")
        return "\n".join(lines)

    def _summarize_dart(self, d) -> str:
        if isinstance(d, list):
            events = d
        elif isinstance(d, dict):
            events = d.get("signals", d.get("events", []))
        else:
            return ""
        if not events:
            return ""
        actionable = d.get("actionable_count", len(events)) if isinstance(d, dict) else len(events)
        lines = [f"## DART 이벤트 ({actionable}건 유효)"]
        for e in events[:5]:
            name = e.get("name", "?")
            ticker = e.get("ticker", "?")
            event = e.get("event", e.get("report_nm", "?"))
            action = e.get("action", "?")
            tier = e.get("tier", "?")
            score = e.get("event_score", 0)
            lines.append(f"  {name}({ticker}): {event} [{action}] 등급:{tier} 점수:{score}")
        return "\n".join(lines)

    def _summarize_institutional(self, d) -> str:
        if isinstance(d, list):
            return ""
        if not isinstance(d, dict):
            return ""
        total = d.get("total_stocks", 0)
        calculated = d.get("calculated", 0)
        zone_dist = d.get("zone_distribution", {})
        vel_dist = d.get("velocity_distribution", {})
        lines = ["## 기관 타겟"]
        lines.append(f"분석 종목: {total}개, 계산 완료: {calculated}개")
        if zone_dist:
            lines.append(f"구간 분포: {zone_dist}")
        if vel_dist:
            lines.append(f"속도 분포: {vel_dist}")
        return "\n".join(lines)

    def _summarize_accumulation(self, d: dict) -> str:
        total = d.get("total_detected", 0)
        phase_stats = d.get("phase_stats", {})
        items = d.get("items", d.get("top20", []))
        if not total and not items:
            return ""
        lines = ["## 매집 추적"]
        lines.append(f"감지: {total}종목, 단계분포: {phase_stats}")
        for item in items[:5]:
            name = item.get("name", "?")
            phase = item.get("phase", "?")
            score = item.get("total_score", 0)
            lines.append(f"  {name}: {phase} ({score}점)")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    # ETF
    # ══════════════════════════════════════════════════════════

    def _summarize_etf(self, d: dict) -> str:
        regime = d.get("regime", "?")
        leading = d.get("leading_regime", "?")
        order_queue = d.get("order_queue", [])
        sector_result = d.get("sector_result", {})
        leverage_result = d.get("leverage_result", {})
        ai_filter = d.get("ai_filter", {})
        predator = d.get("predator_result", {})

        lines = ["## ETF 로테이션"]
        lines.append(f"레짐: {regime}, 리딩: {leading}")

        # 주문 큐
        if order_queue:
            for o in order_queue[:5]:
                if isinstance(o, dict):
                    name = o.get("name", o.get("etf_name", "?"))
                    action = o.get("action", "?")
                    reason = o.get("reason", "")[:40]
                    lines.append(f"  {action} {name} — {reason}")
        else:
            lines.append("  주문 큐: 없음")

        # 섹터 결과 요약
        if isinstance(sector_result, dict):
            buys = sector_result.get("buys", [])
            sells = sector_result.get("sells", [])
            if buys:
                buy_names = [b.get("name", "?") if isinstance(b, dict) else str(b) for b in buys[:3]]
                lines.append(f"  섹터매수: {', '.join(buy_names)}")
            if sells:
                sell_names = [s.get("name", "?") if isinstance(s, dict) else str(s) for s in sells[:3]]
                lines.append(f"  섹터매도: {', '.join(sell_names)}")

        # AI 필터
        if isinstance(ai_filter, dict):
            decisions = ai_filter.get("decisions", [])
            kills = [x for x in decisions if isinstance(x, dict) and x.get("decision") == "KILL"]
            if kills:
                kill_names = [k.get("name", "?") for k in kills[:3]]
                lines.append(f"  AI KILL: {', '.join(kill_names)}")

        return "\n".join(lines)

    def _summarize_momentum(self, d) -> str:
        if isinstance(d, dict):
            rankings = d.get("sectors", d.get("rankings", []))
        elif isinstance(d, list):
            rankings = d
        else:
            return ""
        if not rankings:
            return ""
        lines = ["## 섹터 모멘텀 (상위 5)"]
        for r in rankings[:5]:
            rank = r.get("rank", "?")
            name = r.get("sector", r.get("name", "?"))
            score = r.get("momentum_score", r.get("score", "?"))
            ret5 = r.get("ret_5", "?")
            vol_ratio = r.get("vol_ratio", "?")
            rank_chg = r.get("rank_change", 0)
            chg_str = f" (순위변화: {rank_chg:+d})" if isinstance(rank_chg, int) and rank_chg != 0 else ""
            lines.append(f"  {rank}. {name}: 모멘텀 {score}, 5일수익 {ret5}%, 거래량비 {vol_ratio}{chg_str}")
        return "\n".join(lines)

    def _summarize_leverage(self, d: dict) -> str:
        rec = d.get("recommendation", "?")
        regime = d.get("kospi_regime", "?")
        us = d.get("us_overnight", "?")
        etfs = d.get("etfs", [])
        lines = ["## 레버리지/인버스 ETF"]
        lines.append(f"레짐: {regime}, US: {us}, 추천: {rec}")
        if isinstance(etfs, list):
            for e in etfs[:5]:
                if isinstance(e, dict):
                    name = e.get("name", e.get("etf_name", "?"))
                    signal = e.get("signal", e.get("action", "?"))
                    lines.append(f"  {name}: {signal}")
        return "\n".join(lines)

    def _summarize_relay(self, d: dict) -> str:
        rec = d.get("recommendation", "")
        active = d.get("active_alerts", 0)
        total_score = d.get("total_alert_score", 0)
        sectors = d.get("sectors", {})
        summary = d.get("telegram_summary", "")

        lines = ["## 릴레이 시그널"]
        lines.append(f"활성 알림: {active}건 (총점: {total_score})")
        if rec:
            lines.append(f"추천: {rec}")

        if isinstance(sectors, dict):
            for name, info in sectors.items():
                if isinstance(info, dict):
                    status = info.get("status", info.get("alert_level", "?"))
                    alert_score = info.get("alert_score", 0)
                    if alert_score > 0:
                        lines.append(f"  {name}: {status} (점수: {alert_score})")
        return "\n".join(lines)

    def _summarize_group_relay(self, d: dict) -> str:
        fired = d.get("fired_groups", [])
        summary = d.get("summary", "")
        if not fired and not summary:
            return ""
        lines = ["## 그룹 릴레이"]
        if summary:
            lines.append(f"요약: {summary}")
        for g in fired[:3]:
            if isinstance(g, dict):
                name = g.get("group", g.get("name", "?"))
                lines.append(f"  발동: {name}")
            else:
                lines.append(f"  발동: {g}")
        return "\n".join(lines)

    def _summarize_value_chain(self, d: dict) -> str:
        fired = d.get("fired_sectors", [])
        no_fire = d.get("no_fire_sectors", [])
        if not fired:
            return ""
        lines = ["## 밸류체인 릴레이"]
        for s in fired[:3]:
            if isinstance(s, dict):
                name = s.get("sector", s.get("name", "?"))
                trigger = s.get("trigger", "?")
                lines.append(f"  발동: {name} — {trigger}")
            else:
                lines.append(f"  발동: {s}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    # AI / 포트폴리오
    # ══════════════════════════════════════════════════════════

    def _summarize_brain(self, d: dict) -> str:
        regime = d.get("effective_regime", d.get("kospi_regime", "?"))
        confidence = d.get("confidence", "?")
        cash = d.get("cash_pct", "?")
        invest = d.get("total_invest_pct", "?")
        arms = d.get("arms", [])
        briefing = d.get("briefing", "")
        warnings = d.get("warnings", [])

        lines = ["## BRAIN 배분"]
        lines.append(f"레짐: {regime}, 신뢰도: {confidence}")
        lines.append(f"투자: {invest}%, 현금: {cash}%")
        if isinstance(arms, list):
            arm_parts = []
            for a in arms:
                if isinstance(a, dict):
                    name = a.get("name", "?")
                    pct = a.get("adjusted_pct", a.get("base_pct", 0))
                    arm_parts.append(f"{name}:{pct}%")
            if arm_parts:
                lines.append(f"ARM 배분: {', '.join(arm_parts)}")
        if briefing:
            lines.append(f"브리핑: {str(briefing)[:150]}")
        if warnings:
            lines.append(f"경고: {', '.join(str(w)[:40] for w in warnings[:3])}")
        return "\n".join(lines)

    def _summarize_shield(self, d: dict) -> str:
        level = d.get("overall_level", d.get("status", "?"))
        overlaps = d.get("sector_overlaps", [])
        mdd = d.get("mdd_status", {})
        stock_alerts = d.get("stock_alerts", [])
        warnings = d.get("warnings", [])
        breakdowns = d.get("correlation_breakdowns", [])

        lines = ["## SHIELD 보고서"]
        lines.append(f"위험 수준: {level}")
        if isinstance(mdd, dict):
            cur_mdd = mdd.get("current_mdd_pct", "?")
            lines.append(f"현재 MDD: {cur_mdd}%")
        # 섹터 집중 위험
        severe = [o for o in overlaps if isinstance(o, dict) and o.get("severity") in ("HIGH", "CRITICAL")]
        if severe:
            sev_names = [o.get("sector", "?") for o in severe[:3]]
            lines.append(f"섹터 집중 위험: {', '.join(sev_names)}")
        # 상관관계 붕괴
        broken = [b for b in breakdowns if isinstance(b, dict) and b.get("is_breakdown")]
        if broken:
            br_names = [b.get("pair_name", "?") for b in broken[:3]]
            lines.append(f"상관관계 붕괴: {', '.join(br_names)}")
        if warnings:
            lines.append(f"경고: {', '.join(str(w)[:40] for w in warnings[:3])}")
        return "\n".join(lines)

    def _summarize_allocation(self, d: dict) -> str:
        regime = d.get("regime", "?")
        macro_grade = d.get("macro_grade", "?")
        multiplier = d.get("position_multiplier", "?")
        cash = d.get("cash_reserve_pct", "?")
        allocations = d.get("allocations", {})
        lines = ["## 포트폴리오 배분"]
        lines.append(f"레짐: {regime}, 등급: {macro_grade}, 배수: {multiplier}, 현금: {cash}%")
        if isinstance(allocations, dict):
            for strategy, info in allocations.items():
                if isinstance(info, dict):
                    pct = info.get("budget_pct", info.get("allocation_pct", "?"))
                    lines.append(f"  {strategy}: {pct}%")
                else:
                    lines.append(f"  {strategy}: {info}")
        return "\n".join(lines)

    def _summarize_force_hybrid(self, d: dict) -> str:
        health = d.get("supply_demand_health", {})
        anomaly = d.get("anomaly", {})
        radar = d.get("event_radar", {})
        insights = d.get("cross_insights", [])
        lines = ["## 포스 하이브리드"]
        if isinstance(health, dict) and health:
            score = health.get("score", "?")
            alert = health.get("alert", "?")
            alert_desc = health.get("alert_desc", "")
            lines.append(f"수급 건강: {score}점 ({alert}) — {alert_desc}")
        if isinstance(anomaly, dict) and anomaly:
            detected = anomaly.get("detected", anomaly.get("count", 0))
            lines.append(f"이상 감지: {detected}건")
        if isinstance(radar, dict) and radar:
            mood = radar.get("mood", "?")
            mood_desc = radar.get("mood_desc", "")
            event_count = radar.get("event_count", 0)
            high = radar.get("high_impact", 0)
            lines.append(f"이벤트 레이더: {mood} — {event_count}건 (HIGH:{high})")
            if mood_desc:
                lines.append(f"  {mood_desc}")
            # 테마 히트 요약 (URL 제외)
            themes = radar.get("theme_hits", [])
            if themes:
                theme_names = [t.get("theme", "?") for t in themes[:5] if isinstance(t, dict)]
                lines.append(f"  테마: {', '.join(theme_names)}")
        if insights:
            for i in insights[:2]:
                lines.append(f"  인사이트: {str(i)[:80]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _summarize_positions(self, d) -> str:
        if isinstance(d, dict):
            positions = d.get("positions", [])
            capital = d.get("capital", "?")
        elif isinstance(d, list):
            positions = d
            capital = "?"
        else:
            return ""
        lines = ["## 포트폴리오"]
        lines.append(f"자본: {capital}")
        if not positions:
            lines.append("보유 종목 없음")
        else:
            for h in positions[:10]:
                name = h.get("name", h.get("ticker", "?"))
                pnl = h.get("pnl_pct", h.get("return_pct", 0))
                qty = h.get("qty", h.get("quantity", "?"))
                avg = h.get("avg_price", "?")
                pnl_str = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else str(pnl)
                lines.append(f"  {name}: {pnl_str} (수량:{qty}, 평균가:{avg})")
        return "\n".join(lines)

    def _summarize_equity(self, d: dict) -> str:
        current = d.get("current_equity", d.get("total_equity", "?"))
        peak = d.get("peak_equity", "?")
        mdd = d.get("current_mdd_pct", "?")
        max_mdd = d.get("max_mdd_pct", "?")
        lines = ["## 자산 현황"]
        lines.append(f"현재 자산: {current}, 최고점: {peak}")
        mdd_str = f"{mdd:.1f}%" if isinstance(mdd, (int, float)) else str(mdd)
        max_mdd_str = f"{max_mdd:.1f}%" if isinstance(max_mdd, (int, float)) else str(max_mdd)
        lines.append(f"현재 MDD: {mdd_str}, 최대 MDD: {max_mdd_str}")
        return "\n".join(lines)

    def _summarize_daily_perf(self, d) -> str:
        if isinstance(d, list) and d:
            latest = d[-1] if d else {}
            regime = latest.get("regime_predicted", "?")
            hit = latest.get("regime_hit", "?")
            kospi = latest.get("kospi_change_pct", "?")
            v3_pnl = latest.get("v3_picks_avg_pnl", "?")
            return f"## 일일 성과 (최근)\n레짐예측: {regime} (적중: {hit}), KOSPI변동: {kospi}%, v3 평균PnL: {v3_pnl}%"
        return ""
