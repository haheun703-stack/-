"""V3: DCF + RIM 내재가치 서브팩터

DCF (Discounted Cash Flow): FCF 5년 예측 + 터미널밸류
RIM (Residual Income Model): BPS + 잔여이익(ROE - Ke) 5년

복합 fair_value → 현재가 대비 upside → Z-Score(0~1.0) 정규화.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha.factors.value_ebitda_ev import _zscore_normalize

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ═══════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════
DEFAULT_WACC = 0.10             # 한국시장 평균 자본비용
DEFAULT_TERMINAL_GROWTH = 0.02  # 장기 명목 성장률
DEFAULT_KE = 0.10               # 자기자본비용
MAX_GROWTH_RATE = 0.20          # FCF 성장률 상한
MIN_GROWTH_RATE = -0.05         # FCF 성장률 하한
DCF_YEARS = 5
RIM_YEARS = 5

# 분기 레이블 끝 → 연환산 배수
_ANNUALIZE_MAP = {"Q1": 4.0, "Q2": 2.0, "Q3": 4.0 / 3, "Q4": 1.0}


# ═══════════════════════════════════════════════
# DCF 계산기
# ═══════════════════════════════════════════════

class IntrinsicDCF:
    """Discounted Cash Flow 내재가치"""

    def __init__(
        self,
        wacc: float = DEFAULT_WACC,
        terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
        years: int = DCF_YEARS,
    ):
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.years = years

    def fair_value_per_share(
        self,
        fcf_annual: float,
        growth_rate: float,
        shares: float,
        net_debt: float = 0,
    ) -> float | None:
        """DCF 적정 주가.

        Args:
            fcf_annual: 연환산 FCF (원)
            growth_rate: FCF 예상 성장률 (0.10 = 10%)
            shares: 발행주식수
            net_debt: 순부채 (양수=부채 초과)
        """
        if fcf_annual <= 0 or shares <= 0:
            return None

        g = max(MIN_GROWTH_RATE, min(growth_rate, MAX_GROWTH_RATE))

        # 5년 FCF 현재가치
        pv_fcf = 0.0
        projected = fcf_annual
        for t in range(1, self.years + 1):
            projected *= (1 + g)
            pv_fcf += projected / (1 + self.wacc) ** t

        # Terminal Value (Gordon Growth)
        tv_fcf = projected * (1 + self.terminal_growth)
        tv = tv_fcf / (self.wacc - self.terminal_growth)
        pv_tv = tv / (1 + self.wacc) ** self.years

        equity_value = pv_fcf + pv_tv - net_debt
        if equity_value <= 0:
            return None

        return equity_value / shares


# ═══════════════════════════════════════════════
# RIM 계산기
# ═══════════════════════════════════════════════

class IntrinsicRIM:
    """Residual Income Model 내재가치"""

    def __init__(self, ke: float = DEFAULT_KE, years: int = RIM_YEARS):
        self.ke = ke
        self.years = years

    def fair_value_per_share(
        self,
        bps: float,
        roe: float,
        retention_ratio: float = 0.6,
    ) -> float | None:
        """RIM 적정 주가.

        Args:
            bps: 주당순자산 (원)
            roe: ROE (0.10 = 10%)
            retention_ratio: 내부유보율
        """
        if bps <= 0 or roe <= 0:
            return None

        bps_growth = roe * retention_ratio
        pv_ri = 0.0
        current_bps = bps

        for t in range(1, self.years + 1):
            ri = current_bps * (roe - self.ke)
            pv_ri += ri / (1 + self.ke) ** t
            current_bps *= (1 + bps_growth)

        fair_value = bps + pv_ri
        return fair_value if fair_value > 0 else None


# ═══════════════════════════════════════════════
# V3 통합 팩터
# ═══════════════════════════════════════════════

class ValueIntrinsic:
    """V3 내재가치 팩터: DCF + RIM 합성 → Z-Score"""

    def __init__(
        self,
        financial_data: dict | None = None,
        market_cap_data: dict | None = None,
    ):
        if financial_data is None:
            financial_data = _load_financial()
        if market_cap_data is None:
            market_cap_data = _load_market_cap()

        self._bs_data = financial_data.get("bs_data", {})
        self._cf_data = financial_data.get("cf_data", {})
        self._quality = financial_data.get("quality", {})
        self._market_cap = market_cap_data

        self._fund_daily = _load_latest_fundamentals()

        self._dcf = IntrinsicDCF()
        self._rim = IntrinsicRIM()
        self._details: dict[str, dict] = {}

    # ── 데이터 접근 헬퍼 ──

    def _get_fund(self, ticker: str) -> dict:
        return self._fund_daily.get(ticker, {})

    def _get_current_price(self, ticker: str) -> float | None:
        f = self._get_fund(ticker)
        bps = f.get("BPS")
        pbr = f.get("PBR")
        if bps and pbr and bps > 0 and pbr > 0:
            return bps * pbr
        return None

    def _get_shares(self, ticker: str) -> float | None:
        f = self._get_fund(ticker)
        bps = f.get("BPS")
        if not bps or bps <= 0:
            return None
        bs = self._bs_data.get(ticker, {})
        for label in ("2025Q3", "2025Q2", "2025Q1", "2024Q4"):
            equity = bs.get(label, {}).get("equity")
            if equity and equity > 0:
                return equity / bps
        return None

    def _get_fcf_annual(self, ticker: str) -> float | None:
        cf = self._cf_data.get(ticker, {})
        op_cf = cf.get("operating_cf")
        if op_cf is None:
            return None

        capex = 0.0
        if cf.get("capex_tangible") is not None:
            capex += abs(cf["capex_tangible"])
        if cf.get("capex_intangible") is not None:
            capex += abs(cf["capex_intangible"])

        fcf_cum = op_cf - capex

        period = cf.get("period", "")
        for suffix, factor in _ANNUALIZE_MAP.items():
            if period.endswith(suffix):
                return fcf_cum * factor
        return fcf_cum

    def _estimate_growth(self, ticker: str) -> float | None:
        bs = self._bs_data.get(ticker, {})
        if not bs:
            return None

        yoy_pairs = [
            ("2025Q3", "2024Q3"),
            ("2025Q2", "2024Q2"),
            ("2025Q1", "2024Q1"),
            ("2024Q4", "2023Q4"),
        ]

        growths: list[float] = []
        for cur_label, prev_label in yoy_pairs:
            cur_rev = bs.get(cur_label, {}).get("revenue_cum")
            prev_rev = bs.get(prev_label, {}).get("revenue_cum")
            if cur_rev and prev_rev and prev_rev > 0:
                growths.append((cur_rev - prev_rev) / abs(prev_rev))

        return float(np.median(growths)) if growths else None

    def _get_net_debt(self, ticker: str) -> float:
        bs = self._bs_data.get(ticker, {})
        for label in ("2025Q3", "2025Q2", "2025Q1", "2024Q4"):
            debt = bs.get(label, {}).get("total_debt")
            if debt is not None:
                return debt
        return 0.0

    # ── 스코어링 ──

    def score_raw(self, ticker: str) -> dict | None:
        """단일 종목 DCF/RIM 계산 → upside dict."""
        price = self._get_current_price(ticker)
        if not price or price <= 0:
            return None

        shares = self._get_shares(ticker)
        fund = self._get_fund(ticker)
        bps = fund.get("BPS")
        q = self._quality.get(ticker, {})
        roe = q.get("roe_mean")
        payout = q.get("dividend_payout")

        # DCF
        dcf_val = None
        fcf_ann = self._get_fcf_annual(ticker)
        growth = self._estimate_growth(ticker)
        net_debt = self._get_net_debt(ticker)

        if fcf_ann and fcf_ann > 0 and growth is not None and shares and shares > 0:
            dcf_val = self._dcf.fair_value_per_share(
                fcf_annual=fcf_ann,
                growth_rate=growth,
                shares=shares,
                net_debt=net_debt,
            )

        # RIM
        rim_val = None
        if bps and bps > 0 and roe and roe > 0:
            retention = 1 - (payout if payout and 0 < payout < 1 else 0.4)
            retention = max(0.2, min(retention, 0.95))
            rim_val = self._rim.fair_value_per_share(
                bps=bps, roe=roe, retention_ratio=retention,
            )

        values = [v for v in (dcf_val, rim_val) if v is not None]
        if not values:
            return None

        fair = float(np.median(values))
        upside = (fair / price) - 1.0

        return {
            "dcf_value": round(dcf_val, 0) if dcf_val else None,
            "rim_value": round(rim_val, 0) if rim_val else None,
            "fair_value": round(fair, 0),
            "current_price": round(price, 0),
            "upside": round(upside, 4),
            "models_used": len(values),
        }

    def score_universe(self) -> dict[str, float]:
        """전체 유니버스 upside → Z-Score(0~1.0)."""
        raw: dict[str, float] = {}

        all_tickers = set(self._quality.keys()) & set(self._market_cap.keys())
        for ticker in all_tickers:
            result = self.score_raw(ticker)
            if result is not None:
                raw[ticker] = result["upside"]
                self._details[ticker] = result

        if raw:
            upsides = list(raw.values())
            logger.info(
                "V3 내재가치: %d종목 (DCF+RIM), upside [%.0f%% ~ %.0f%%]",
                len(raw),
                min(upsides) * 100,
                max(upsides) * 100,
            )

        return _zscore_normalize(raw)

    def get_detail(self, ticker: str) -> dict | None:
        return self._details.get(ticker)


# ═══════════════════════════════════════════════
# 데이터 로더 (value_ebitda_ev.py와 동일 소스)
# ═══════════════════════════════════════════════

def _load_financial() -> dict:
    path = PROJECT_ROOT / "data" / "v2_migration" / "financial_quarterly.json"
    if not path.exists():
        logger.warning("financial_quarterly.json 없음")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_market_cap() -> dict:
    path = PROJECT_ROOT / "data" / "market_cap_cache.json"
    if not path.exists():
        logger.warning("market_cap_cache.json 없음")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_latest_fundamentals() -> dict[str, dict]:
    """fundamental_daily.parquet에서 종목별 최신 BPS/PBR/EPS/PER 로드."""
    path = PROJECT_ROOT / "data" / "fundamental_cache" / "fundamental_daily.parquet"
    if not path.exists():
        logger.warning("fundamental_daily.parquet 없음")
        return {}

    df = pd.read_parquet(path)
    df = df.sort_values("date")
    latest = df.groupby("ticker").last()

    result: dict[str, dict] = {}
    for ticker, row in latest.iterrows():
        entry = {}
        for col in ("BPS", "PBR", "PER", "EPS", "DIV", "DPS"):
            val = row.get(col)
            if pd.notna(val) and val != 0:
                entry[col] = float(val)
        if entry:
            result[str(ticker)] = entry

    logger.info("Fundamental daily: %d종목 최신 데이터 로드", len(result))
    return result
