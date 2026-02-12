"""콘솔 출력 어댑터 - 실시간 모니터링 결과를 콘솔에 출력"""

from __future__ import annotations

from datetime import datetime

from src.entities.models import InvestorFlow
from src.use_cases.ports import OutputPort


def to_eok(shares: int, price: float = 70_000) -> str:
    """주수 x 가격 -> 억원 문자열 (가격 기본값 7만원)"""
    amount = shares * price / 1_0000_0000
    if abs(amount) >= 1:
        return f"{amount:+,.0f}억"
    return f"{shares:+,}주"


class ConsoleOutputAdapter(OutputPort):
    """콘솔에 메시지를 출력하는 어댑터 (OutputPort 구현)"""

    def display(self, message: str) -> None:
        print(message)

    def print_alert(
        self,
        code: str,
        name: str,
        price: float,
        flow: InvestorFlow,
        alert_reasons: list[str],
        total_shares: int = 0,
        frgn_holding_qty: int = 0,
        frgn_holding_rto: float | None = None,
    ) -> None:
        """수급 알림 상세 출력 (발행주식수 + 외국인 보유 포함)

        삼성전자 (005930) - 현재가 167,800원
        ========================================
        발행주식수: 5,969,782,550주
        외국인 보유: 3,012,345,678주 (50.46%)
        ----------------------------------------
        투자자     억원        주수
        외국인    +6,177억    +3,681,106주
        ...
        ----------------------------------------
        >> 알림 사유
        ========================================
        """
        now = datetime.now().strftime("%H:%M")
        sep = "=" * 40

        # 발행주식수/외국인보유: flow 엔티티에서 우선 추출
        ts = total_shares or flow.total_shares
        fq = frgn_holding_qty or flow.foreign_holding_qty
        fr = frgn_holding_rto if frgn_holding_rto is not None else flow.foreign_holding_ratio

        print(f"\n  {sep}")
        print(f"  {name} ({code}) - 현재가 {price:,.0f}원  [{now}]")
        print(f"  {sep}")

        if ts > 0:
            print(f"  발행주식수: {ts:,}주")
        if fq > 0 and fr:
            print(f"  외국인 보유: {fq:,}주 ({fr:.2f}%)")
        elif fr:
            print(f"  외국인 보유율: {fr:.2f}%")

        print(f"  {'-'*40}")
        print(f"  {'투자자':<8} {'억원':>8}    {'주수':>14}")
        print(f"  {'-'*40}")
        print(f"  {'외국인':<8} {to_eok(flow.foreign_net, price):>8}    {flow.foreign_net:>+14,}주")
        print(f"  {'기관':<8} {to_eok(flow.inst_net, price):>8}    {flow.inst_net:>+14,}주")
        print(f"  {'개인':<8} {to_eok(flow.individual_net, price):>8}    {flow.individual_net:>+14,}주")
        print(f"  {'-'*40}")
        for reason in alert_reasons:
            print(f"  >> {reason}")
        print(f"  {sep}")

    def print_scan(self, results: list[dict]) -> None:
        """전체 종목 요약 테이블 출력

        results: [{"name": str, "price": float, "change_pct": float,
                   "flow": InvestorFlow, "vol_ratio": float}, ...]
        """
        header = (
            f"  {'종목':<10} {'현재가':>10} {'등락':>7}  "
            f"{'외국인':>8}  {'기관':>8}  {'거래량%':>6}"
        )
        print(f"\n  {'='*65}")
        print(header)
        print(f"  {'-'*65}")

        for r in results:
            name = r["name"]
            price = r["price"]
            change_pct = r["change_pct"]
            flow: InvestorFlow = r["flow"]
            vol_ratio = r.get("vol_ratio", 100)

            foreign_str = to_eok(flow.foreign_net, price)
            inst_str = to_eok(flow.inst_net, price)

            flags = ""
            if flow.foreign_net > 0 and flow.inst_net > 0:
                flags += " [동시매수]"
            if vol_ratio >= 300:
                flags += " [거래량!]"

            print(
                f"  {name:<10} {price:>10,.0f} {change_pct:>+6.1f}%  "
                f"{foreign_str:>8}  {inst_str:>8}  {vol_ratio:>5.0f}%{flags}"
            )

        print(f"  {'='*65}")
