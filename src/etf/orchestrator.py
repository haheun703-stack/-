"""
ETF 3축 통합 오케스트레이터
====================================
매일 저녁 실행:
  Step 1: 레짐 → 비중 매트릭스
  Step 2: 섹터 ETF 스캔
  Step 3: 레버리지 판단
  Step 4: 지수 ETF 비중 체크
  Step 5: 통합 리스크 체크
  Step 6: 주문 큐 생성
  Step 7: 텔레그램 리포트
"""

import json
from datetime import datetime
from dataclasses import dataclass

from src.etf.config import load_settings, get_allocation
from src.etf.sector_engine import SectorETFEngine
from src.etf.leverage_engine import LeverageEngine
from src.etf.index_engine import IndexETFEngine
from src.etf.risk_manager import ETFRiskManager
from src.etf.predator_engine import PredatorEngine


class ETFOrchestrator:
    """ETF 3축 통합 오케스트레이터."""

    def __init__(self, settings: dict = None):
        self.settings = settings or load_settings()
        self.sector_engine = SectorETFEngine(self.settings)
        self.leverage_engine = LeverageEngine(self.settings)
        self.index_engine = IndexETFEngine(self.settings)
        self.risk_manager = ETFRiskManager(self.settings)
        self.predator_engine = PredatorEngine(self.settings)

        self.current_regime: str = "CAUTION"
        self.previous_regime: str | None = None

    def run(
        self,
        regime: str,
        previous_regime: str = None,
        kospi_ma20_above: bool = True,
        kospi_ma60_above: bool = True,
        total_portfolio_value: float = 100_000_000,
        individual_stock_sectors: set[str] = None,
        momentum_data: dict = None,
        smart_money_data: dict = None,
        supply_data: dict = None,
        us_overnight: dict = None,
        five_axis_score: float = 0,
        index_holdings: dict = None,
        # 프레데터 모드 파라미터
        prev_momentum_data: dict = None,
        sector_returns_1d: dict = None,
        supply_flow_data: dict = None,
    ) -> dict:
        """
        3축 통합 실행.

        Returns:
            dict with regime, allocation, 3축 results, risk_check, order_queue, telegram_report
        """
        self.current_regime = regime.upper()
        self.previous_regime = previous_regime
        individual_stock_sectors = individual_stock_sectors or set()
        us_overnight = us_overnight or {"grade": 3, "signal": "neutral"}

        print(f"\n{'='*60}")
        print(f"🤖 ETF 3축 로테이션 오케스트레이터")
        print(f"{'='*60}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 레짐: {self.current_regime} | 포트: {total_portfolio_value:,.0f}원")

        # Step 1: 비중 매트릭스
        allocation = get_allocation(self.current_regime, self.settings)
        print(f"\n📋 Step 1: 비중 배분")
        print(f"   섹터 {allocation['sector']}% | 레버 {allocation['leverage']}% | "
              f"지수 {allocation['index']}% | 현금 {allocation['cash']}%")

        # Step 2: 섹터 ETF 스캔
        print(f"\n🎯 Step 2: 섹터 ETF 스캔")
        sector_result = {"buy_candidates": [], "sell_signals": [], "current_positions": [], "summary": "스킵"}
        if allocation["sector"] > 0 and momentum_data:
            sector_result = self.sector_engine.run(
                momentum_data=momentum_data or {},
                smart_money_data=smart_money_data or {},
                supply_data=supply_data or {},
                individual_sectors=individual_stock_sectors,
            )
            print(f"   {sector_result['summary']}")
        else:
            print(f"   ⏭️ 섹터 ETF 비중 0% - 스킵")

        # Step 2.5: 프레데터 모드 (가속도 + 확신 집중)
        predator_enabled = self.settings.get("predator", {}).get("enabled", False)
        predator_result = None
        if predator_enabled and momentum_data and prev_momentum_data:
            print(f"\n🦅 Step 2.5: 프레데터 모드")
            predator_result = self.predator_engine.run(
                current_ranks=momentum_data,
                prev_ranks=prev_momentum_data,
                supply_data=supply_flow_data,
                total_sector_pct=allocation["sector"],
                sector_returns_1d=sector_returns_1d or {},
                regime_changed=(previous_regime is not None and previous_regime.upper() != self.current_regime),
                regime_direction=f"{previous_regime}→{self.current_regime}" if previous_regime else "",
                us_overnight_grade=us_overnight.get("grade", 3) if us_overnight else 3,
            )
            print(f"   {predator_result['summary']}")

        # Step 3: 레버리지 판단
        print(f"\n⚡ Step 3: 레버리지 판단")
        leverage_result = self.leverage_engine.run(
            regime=self.current_regime,
            us_overnight=us_overnight,
            five_axis_score=five_axis_score,
            previous_regime=self.previous_regime,
            momentum_data=momentum_data,
        )
        print(f"   {leverage_result['summary']}")

        # Step 4: 지수 ETF 비중
        print(f"\n📈 Step 4: 지수 ETF 비중")
        if index_holdings:
            self.index_engine.set_current_holdings(index_holdings)
        index_result = self.index_engine.run(
            regime=self.current_regime,
            ma_20_above=kospi_ma20_above,
            ma_60_above=kospi_ma60_above,
        )
        print(f"   {index_result['summary']}")

        # Step 5: 통합 리스크 체크
        print(f"\n🛡️ Step 5: 리스크 체크")
        etf_sectors = self._extract_etf_sectors(sector_result)
        sector_exposure = self._calc_sector_exposure(sector_result, individual_stock_sectors)
        leverage_pct = allocation["leverage"]
        total_invest_pct = 100 - allocation["cash"]

        risk_check = self.risk_manager.run_checks(
            portfolio_value=total_portfolio_value,
            sector_exposure=sector_exposure,
            leverage_exposure_pct=leverage_pct,
            total_investment_pct=total_invest_pct,
            individual_stock_sectors=individual_stock_sectors,
            etf_sectors=etf_sectors,
            regime=self.current_regime,
            previous_regime=self.previous_regime,
        )
        print(f"   {risk_check.summary}")

        # Step 6: 주문 큐
        print(f"\n📝 Step 6: 주문 큐 생성")
        order_queue = self._build_order_queue(sector_result, leverage_result, index_result, risk_check, allocation)
        if order_queue:
            for order in order_queue:
                emoji = "🟢" if order["action"] == "BUY" else "🔴"
                print(f"   {emoji} [{order['priority']}] {order['name']} {order['action']} | {order['reason']}")
        else:
            print(f"   📭 주문 없음")

        # Step 7: 텔레그램 리포트
        telegram_report = self._build_telegram_report(allocation, sector_result, leverage_result, index_result, risk_check, order_queue, predator_result)

        result = {
            "regime": self.current_regime,
            "allocation": allocation,
            "sector_result": {k: v for k, v in sector_result.items() if k != "timestamp"},
            "leverage_result": {k: v for k, v in leverage_result.items() if k != "timestamp"},
            "index_result": {k: v for k, v in index_result.items() if k != "timestamp"},
            "predator_result": predator_result,
            "risk_check": {"passed": risk_check.passed, "level": risk_check.level, "violations": risk_check.violations, "summary": risk_check.summary},
            "order_queue": order_queue,
            "telegram_report": telegram_report,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"✅ 오케스트레이터 실행 완료")
        print(f"{'='*60}")

        return result

    def _build_order_queue(self, sector_result, leverage_result, index_result, risk_check, allocation) -> list[dict]:
        """3축 결과 → 통합 주문 큐."""
        orders = []

        # 킬스위치 발동 시
        if risk_check.level == "KILLSWITCH":
            return self._killswitch_orders(risk_check)

        # 매도 우선
        for sig in sector_result.get("sell_signals", []):
            if sig.get("signal") == "SELL":
                orders.append({"axis": "sector", "code": sig["code"], "name": sig["name"], "action": "SELL", "target_weight_pct": 0, "reason": sig["reason"], "priority": 2})

        lev_decision = leverage_result.get("decision", {})
        if lev_decision.get("signal") in ["SELL", "EMERGENCY_SELL"]:
            priority = 1 if lev_decision["signal"] == "EMERGENCY_SELL" else 2
            orders.append({"axis": "leverage", "code": lev_decision.get("etf_code", ""), "name": lev_decision.get("etf_name", ""), "action": "SELL", "target_weight_pct": 0, "reason": lev_decision.get("reason", ""), "priority": priority})

        for t in index_result.get("targets", []):
            if t["action"] == "SELL":
                orders.append({"axis": "index", "code": t["code"], "name": t["name"], "action": "SELL", "target_weight_pct": 0, "reason": t["reason"], "priority": 2})

        # 매수 (레버리지 > 섹터 > 지수)
        if lev_decision.get("signal") == "BUY":
            orders.append({"axis": "leverage", "code": lev_decision["etf_code"], "name": lev_decision["etf_name"], "action": "BUY", "target_weight_pct": allocation["leverage"], "reason": lev_decision["reason"], "priority": 2})

        buy_candidates = sector_result.get("buy_candidates", [])
        per_etf_weight = allocation["sector"] / max(len(buy_candidates), 1)
        for c in buy_candidates:
            if c.get("signal") == "BUY":
                orders.append({"axis": "sector", "code": c["code"], "name": c["name"], "action": "BUY", "target_weight_pct": round(per_etf_weight, 2), "reason": c["reason"], "priority": 2})

        for t in index_result.get("targets", []):
            if t["action"] in ["BUY", "REBALANCE"]:
                orders.append({"axis": "index", "code": t["code"], "name": t["name"], "action": "BUY", "target_weight_pct": t["target_weight_pct"], "reason": t["reason"], "priority": 3})

        orders.sort(key=lambda x: (x["priority"], x["action"] != "SELL"))
        return orders

    def _killswitch_orders(self, risk_check) -> list[dict]:
        orders = []
        for adj in risk_check.adjustments:
            if adj.get("severity") == "KILLSWITCH":
                action_type = adj.get("type", "")
                if action_type == "close_leverage":
                    orders.append({"axis": "leverage", "code": "", "name": "레버리지 전체", "action": "SELL", "target_weight_pct": 0, "reason": adj["message"], "priority": 1})
                elif action_type == "close_sector":
                    orders.append({"axis": "sector", "code": "", "name": "섹터 ETF 전체", "action": "SELL", "target_weight_pct": 0, "reason": adj["message"], "priority": 1})
                elif action_type in ["close_all", "close_all_longs_next_open"]:
                    orders.append({"axis": "all", "code": "", "name": "전체 포지션", "action": "SELL", "target_weight_pct": 0, "reason": adj["message"], "priority": 1})
        return orders

    def _extract_etf_sectors(self, sector_result: dict) -> set[str]:
        sectors = set()
        for c in sector_result.get("buy_candidates", []):
            sectors.add(c.get("sector", ""))
        for p in sector_result.get("current_positions", []):
            sectors.add(p.get("sector", ""))
        return sectors

    def _calc_sector_exposure(self, sector_result: dict, individual_sectors: set) -> dict[str, float]:
        exposure = {}
        for c in sector_result.get("buy_candidates", []):
            sector = c.get("sector", "unknown")
            exposure[sector] = exposure.get(sector, 0) + 13
        for sector in individual_sectors:
            exposure[sector] = exposure.get(sector, 0) + 10
        return exposure

    def _build_telegram_report(self, allocation, sector_result, leverage_result, index_result, risk_check, order_queue, predator_result=None) -> str:
        lines = []
        lines.append("━" * 28)
        lines.append("🤖 ETF 3축 로테이션 리포트")
        lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("━" * 28)

        regime_emoji = {"BULL": "🟢", "PRE_BULL": "🟢", "CAUTION": "🟡",
                       "PRE_BEAR": "🟠", "BEAR": "🟠", "PRE_CRISIS": "🔴", "CRISIS": "🔴"}
        r = self.current_regime
        pre_tag = " (선행)" if r.startswith("PRE_") else ""
        lines.append(f"\n{regime_emoji.get(r, '⚪')} 레짐: {r}{pre_tag}")
        lines.append(f"📊 섹터 {allocation['sector']}% | 레버 {allocation['leverage']}% | 지수 {allocation['index']}% | 현금 {allocation['cash']}%")

        # 섹터 ETF
        lines.append(f"\n🎯 [섹터 ETF]")
        if sector_result.get("buy_candidates"):
            for c in sector_result["buy_candidates"]:
                lines.append(f"  📈 {c['name']} ({c['composite_score']:.0f}점)")
                lines.append(f"     {c['reason']}")
        else:
            lines.append("  ─ 매수 후보 없음")

        sell_sigs = [s for s in sector_result.get("sell_signals", []) if s.get("signal") == "SELL"]
        if sell_sigs:
            for s in sell_sigs:
                lines.append(f"  📉 {s['name']} SELL - {s['reason']}")

        # 프레데터 모드
        if predator_result:
            lines.append(f"\n🦅 [프레데터 모드]")
            # 가속도 TOP 3
            accels = predator_result.get("accelerations", [])[:3]
            if accels:
                accel_str = " > ".join([f"{a['sector']}({a['acceleration_score']:.0f})" for a in accels])
                lines.append(f"  가속도: {accel_str}")

            # 확신도 배분
            convictions = predator_result.get("convictions", [])
            if convictions:
                for c in convictions:
                    level_emoji = {"HIGH": "🔥", "MID": "📌", "LOW": "📎"}.get(c["level"], "")
                    lines.append(f"  {level_emoji} {c['sector']} {c['level']} → {c['weight_pct']:.1f}%")
                    if c.get("reasons"):
                        lines.append(f"     {', '.join(c['reasons'][:3])}")

            # 이벤트 트리거
            triggers = predator_result.get("event_triggers", [])
            if triggers:
                for t in triggers:
                    lines.append(f"  ⚡ {t['trigger_type']}: {t['reason']}")
            else:
                lines.append(f"  트리거: 없음 (정상)")

        # 레버리지
        lines.append(f"\n⚡ [레버리지]")
        lines.append(f"  {leverage_result['summary']}")

        # 지수 ETF
        lines.append(f"\n📈 [지수 ETF]")
        lines.append(f"  {index_result['summary']}")

        # 리스크
        lines.append(f"\n🛡️ [리스크]")
        lines.append(f"  {risk_check.summary}")

        # 내일 액션
        lines.append(f"\n📋 [내일 액션]")
        if order_queue:
            for order in order_queue[:5]:
                emoji = "🟢" if order["action"] == "BUY" else "🔴"
                lines.append(f"  {emoji} {order['name']} {order['action']}")
        else:
            lines.append("  ─ 주문 없음 (유지)")

        lines.append("\n" + "━" * 28)
        return "\n".join(lines)

    def to_json(self, result: dict) -> str:
        """결과를 JSON 문자열로 직렬화."""
        serializable = {k: v for k, v in result.items() if k != "telegram_report"}
        return json.dumps(serializable, ensure_ascii=False, indent=2)
