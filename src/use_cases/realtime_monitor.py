"""실시간 수급 모니터링 유스케이스 - 장중 N분 간격으로 수급 이상 감지"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.entities.models import InvestorFlow
from src.use_cases.ports import InvestorFlowPort, OutputPort, StockDataPort


def _to_eok(shares: int, price: float) -> float:
    """주수 x 가격 -> 억원 (float)"""
    return shares * price / 1_0000_0000


@dataclass
class RealtimeMonitorInteractor:
    """실시간 수급 모니터링 오케스트레이터

    flow_port   : 네이버/KIS 등 수급 데이터 조회
    stock_port  : KIS 등 현재가 조회 (억원 변환용)
    output      : 콘솔/파일 등 출력
    """

    flow_port: InvestorFlowPort
    stock_port: StockDataPort
    output: OutputPort
    _prev_flows: dict[str, InvestorFlow] = field(default_factory=dict)
    _prices: dict[str, float] = field(default_factory=dict)
    _names: dict[str, str] = field(default_factory=dict)
    _stock_info: dict[str, dict] = field(default_factory=dict)  # total_shares, frgn 등

    async def monitor(
        self,
        codes: list[str],
        interval: int = 300,
        threshold_eok: int = 50,
    ) -> None:
        """장중 반복 모니터링 (Ctrl+C 종료)

        Args:
            codes: 종목코드 리스트
            interval: 조회 간격 (초, 기본 300 = 5분)
            threshold_eok: 외국인 알림 기준 (억원, 기본 50)
        """
        self.output.display(
            f"  [모니터링 시작] {interval}초 간격, {len(codes)}종목, "
            f"알림기준: 외국인 {threshold_eok}억 / 기관 {int(threshold_eok * 0.6)}억"
        )
        self.output.display(f"  {'='*50}")

        # 첫 사이클에서 종목 이름/가격 캐싱을 위해 stock_port 조회
        await self._refresh_prices(codes)

        cycle = 0
        try:
            while True:
                cycle += 1
                now = datetime.now()
                next_time = now + timedelta(seconds=interval)
                self.output.display(
                    f"\n  [{now.strftime('%H:%M')}] #{cycle} 수급 체크 중..."
                )

                alerts = await self._check_once(codes, threshold_eok)

                if alerts:
                    for code, flow, reasons in alerts:
                        name = self._names.get(code, code)
                        price = self._prices.get(code, 0)
                        self._print_alert(code, name, price, flow, reasons)
                else:
                    self.output.display("  -> 특이사항 없음")

                self.output.display(
                    f"  [체크 완료] (다음: {next_time.strftime('%H:%M')})"
                )
                await asyncio.sleep(interval)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.output.display(f"\n  {'='*50}")
            self.output.display(f"  [모니터링 종료] 총 {cycle}회 체크")

    async def _refresh_prices(self, codes: list[str]) -> None:
        """stock_port에서 종목 이름/현재가/발행주식수 갱신"""
        for code in codes:
            try:
                stock, chart_data = await self.stock_port.fetch(code, period_days=5)
                self._names[code] = stock.name
                if chart_data.latest_close:
                    self._prices[code] = chart_data.latest_close
                # KIS fetch_stock_info로 발행주식수/외국인보유 캐싱
                if hasattr(self.stock_port, "fetch_stock_info"):
                    info = self.stock_port.fetch_stock_info(code)
                    if info.get("total_shares", 0) > 0:
                        self._stock_info[code] = info
            except Exception:
                pass
            await asyncio.sleep(0.3)

    async def _check_once(
        self,
        codes: list[str],
        threshold_eok: int,
    ) -> list[tuple[str, InvestorFlow, list[str]]]:
        """한 사이클: 전 종목 수급 조회 + 알림 감지

        Returns:
            [(종목코드, InvestorFlow, [알림사유, ...]), ...]
        """
        results: list[tuple[str, InvestorFlow, list[str]]] = []

        for code in codes:
            try:
                flow = await self.flow_port.fetch(code)
                reasons = self._detect_alerts(code, flow, threshold_eok)
                if reasons:
                    results.append((code, flow, reasons))
                self._prev_flows[code] = flow
            except Exception as e:
                self.output.display(f"  [오류] {code} 수급 조회 실패: {e}")

            await asyncio.sleep(1)  # rate limit

        return results

    def _detect_alerts(
        self,
        code: str,
        flow: InvestorFlow,
        threshold_eok: int,
    ) -> list[str]:
        """수급 이상 감지 -> 알림 사유 리스트

        조건:
        1. 외국인 순매매 절대값 >= threshold_eok 억
        2. 기관 순매매 절대값 >= threshold_eok * 0.6 억
        3. 외국인+기관 동시 순매수 (둘 다 양수)
        4. 이전 체크 대비 외국인 변화량 >= 30억
        5. 거래량이 전일 동시간 대비 300% 이상 (향후 확장)
        """
        reasons: list[str] = []
        price = self._prices.get(code, 70_000)  # fallback 7만원
        inst_threshold_eok = int(threshold_eok * 0.6)

        # 주수 -> 억원 변환
        foreign_eok = _to_eok(flow.foreign_net, price)
        inst_eok = _to_eok(flow.inst_net, price)

        # (1) 외국인 대량
        if abs(foreign_eok) >= threshold_eok:
            direction = "순매수" if foreign_eok > 0 else "순매도"
            reasons.append(f"외국인 {foreign_eok:+,.0f}억 ({direction})")

        # (2) 기관 대량
        if abs(inst_eok) >= inst_threshold_eok:
            direction = "순매수" if inst_eok > 0 else "순매도"
            reasons.append(f"기관 {inst_eok:+,.0f}억 ({direction})")

        # (3) 외국인+기관 동시 순매수
        if flow.foreign_net > 0 and flow.inst_net > 0:
            combined_eok = foreign_eok + inst_eok
            if combined_eok >= inst_threshold_eok:
                reasons.append(
                    f"외국인+기관 동시 순매수! "
                    f"(외 {foreign_eok:+,.0f}억 / 기 {inst_eok:+,.0f}억)"
                )

        # (4) 이전 체크 대비 변화량
        prev = self._prev_flows.get(code)
        if prev:
            change_shares = flow.foreign_net - prev.foreign_net
            change_eok = _to_eok(change_shares, price)
            if abs(change_eok) >= 30:
                direction = "증가" if change_eok > 0 else "감소"
                reasons.append(
                    f"외국인 변화 {direction} {abs(change_eok):,.0f}억 "
                    f"({_to_eok(prev.foreign_net, price):+,.0f}억 -> {foreign_eok:+,.0f}억)"
                )

        return reasons

    def _print_alert(
        self,
        code: str,
        name: str,
        price: float,
        flow: InvestorFlow,
        reasons: list[str],
    ) -> None:
        """수급 알림 포맷 출력 (발행주식수 + 외국인 보유 포함)"""
        now = datetime.now().strftime("%H:%M")
        foreign_eok = _to_eok(flow.foreign_net, price)
        inst_eok = _to_eok(flow.inst_net, price)
        indiv_eok = _to_eok(flow.individual_net, price)

        # 발행주식수/외국인보유: InvestorFlow 우선, KIS 캐시 보완
        total_shares = flow.total_shares
        frgn_qty = flow.foreign_holding_qty
        frgn_rto = flow.foreign_holding_ratio

        kis_info = self._stock_info.get(code, {})
        if not total_shares and kis_info.get("total_shares"):
            total_shares = kis_info["total_shares"]
        if not frgn_qty and kis_info.get("frgn_hldn_qty"):
            frgn_qty = kis_info["frgn_hldn_qty"]
        if not frgn_rto and kis_info.get("frgn_hldn_rto"):
            frgn_rto = kis_info["frgn_hldn_rto"]

        sep = "=" * 40
        self.output.display(f"\n  {sep}")
        self.output.display(f"  {name} ({code}) - 현재가 {price:,.0f}원  [{now}]")
        self.output.display(f"  {sep}")

        if total_shares > 0:
            self.output.display(f"  발행주식수: {total_shares:,}주")
        if frgn_qty > 0 and frgn_rto:
            self.output.display(f"  외국인 보유: {frgn_qty:,}주 ({frgn_rto:.2f}%)")
        elif frgn_rto:
            self.output.display(f"  외국인 보유율: {frgn_rto:.2f}%")

        self.output.display(f"  {'-'*40}")
        self.output.display(f"  {'투자자':<8} {'억원':>8}    {'주수':>14}")
        self.output.display(f"  {'-'*40}")
        self.output.display(
            f"  {'외국인':<8} {foreign_eok:>+8,.0f}억    {flow.foreign_net:>+14,}주"
        )
        self.output.display(
            f"  {'기관':<8} {inst_eok:>+8,.0f}억    {flow.inst_net:>+14,}주"
        )
        self.output.display(
            f"  {'개인':<8} {indiv_eok:>+8,.0f}억    {flow.individual_net:>+14,}주"
        )
        self.output.display(f"  {'-'*40}")
        for reason in reasons:
            self.output.display(f"  >> {reason}")
        self.output.display(f"  {sep}")
