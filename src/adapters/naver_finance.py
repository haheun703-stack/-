"""네이버 금융 크롤링 어댑터 - 투자자별 매매동향 수집"""

from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

from src.entities.models import InvestorFlow
from src.use_cases.ports import InvestorFlowPort

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


class NaverFinanceAdapter(InvestorFlowPort):
    """네이버 금융에서 외국인/기관 매매동향을 크롤링하는 어댑터"""

    async def fetch(self, ticker: str) -> InvestorFlow:
        url = f"https://finance.naver.com/item/frgn.naver?code={ticker}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "euc-kr"

            tables = pd.read_html(StringIO(resp.text), header=0)

            # 9컬럼 테이블 찾기 (날짜/종가/전일비/등락률/거래량/기관순매매/외국인순매매/보유주수/보유율)
            flow_table = None
            for t in tables:
                if len(t.columns) == 9:
                    cols = [str(c) for c in t.columns]
                    if any("외국인" in c for c in cols):
                        flow_table = t
                        break

            if flow_table is None or flow_table.empty:
                return InvestorFlow(ticker=ticker, date=date.today())

            # NaN 행 제거
            flow_table = flow_table.dropna(how="all")
            if len(flow_table) < 2:
                return InvestorFlow(ticker=ticker, date=date.today())

            # Row 0은 서브헤더 (순매매량/순매매량/보유주수/보유율) → Row 1이 실제 최신 데이터
            row = flow_table.iloc[1]

            # 컬럼 인덱스: 5=기관순매매량, 6=외국인순매매량, 7=보유주수, 8=외국인보유율
            inst_net = _parse_int(row.iloc[5])
            foreign_net = _parse_int(row.iloc[6])
            foreign_holding_qty = _parse_int(row.iloc[7])
            holding_ratio = _parse_float(row.iloc[8])

            # 발행주식수 계산: 보유주수 / (보유율/100)
            total_shares = 0
            if holding_ratio and holding_ratio > 0 and foreign_holding_qty > 0:
                total_shares = int(foreign_holding_qty / (holding_ratio / 100))

            return InvestorFlow(
                ticker=ticker,
                date=date.today(),
                foreign_net=foreign_net,
                inst_net=inst_net,
                individual_net=-(foreign_net + inst_net),
                total_shares=total_shares,
                foreign_holding_qty=foreign_holding_qty,
                foreign_holding_ratio=holding_ratio,
            )

        except Exception:
            return InvestorFlow(ticker=ticker, date=date.today())


def _parse_int(val) -> int:
    """문자열에서 정수 파싱 (콤마, +/- 부호 처리)"""
    try:
        s = str(val).replace(",", "").replace("+", "").strip()
        return int(s)
    except (ValueError, TypeError):
        return 0


def _parse_float(val) -> float | None:
    """퍼센트 문자열에서 float 파싱 (예: '51.28%' → 51.28)"""
    try:
        s = str(val).replace("%", "").replace(",", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return None
