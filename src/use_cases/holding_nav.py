"""src/use_cases/holding_nav.py — 지주사 NAV(순자산가치) 디스카운트 엔진.

지주·관계사가 보유한 상장 자회사 지분을 시가로 평가해 NAV를 계산하고, 지주사 자신의
시총과 비교해 할인율을 산출한다. "삼성전자가 오르면 삼성물산 NAV도 오르는데 지주사 할인
때문에 시차/언더슈팅" — 그 갭(반사이익)을 정량화하는 게 목적.

  NAV = Σ(상장 자회사 시총 × 지분율) + 비상장가치 + 자체사업가치 − 순부채
  할인율 = (지주사 시총 − NAV) / NAV        (음수 = NAV보다 싸게 거래 = 할인)

★관측/백테스트 전용 — hard gate 아님, 실주문 0. valuation_band.py와 동일 계층(관측 라벨).
  지분율·순부채·비상장가치·자체사업가치는 config/holding_nav.yaml 수동 입력(사업보고서 기준,
  분기 갱신). 이 모듈은 순수 계산만 한다(I/O 없음) — 시총 dict를 주입받는다(scan 스크립트가 조달).

밸류트랩 주의: 할인율이 깊다고 매수 신호가 아니다(할인 영구화 함정). 신호는 별도 — 할인율이
  자기 역사적 밴드 하단 + 핵심 자회사 반등 시작이 겹칠 때만(scan/백테스트에서 판정).
"""
from __future__ import annotations

from dataclasses import dataclass, field

EOK = 100_000_000  # 1억 (원). YAML 입력은 억원 단위 → ×EOK로 원 환산.


@dataclass(frozen=True)
class Stake:
    """상장 자회사 1건의 보유 지분."""
    ticker: str
    name: str
    pct: float  # 보유 지분율 (%)


@dataclass(frozen=True)
class Holding:
    """지주·관계사 1곳의 NAV 구성 (config/holding_nav.yaml에서 로드)."""
    ticker: str
    name: str
    kind: str                       # "pure_holding"(순수투자지주) | "operating_holding"(사업지주)
    listed_stakes: list[Stake]
    other_nav_eok: float = 0.0      # 비상장 자회사 가치 (억원, 보수적 장부가)
    own_business_eok: float = 0.0   # 자체 영업가치 (억원, 사업지주만)
    net_debt_eok: float = 0.0       # 순부채(+) / 순현금(−) (억원)
    as_of: str = ""                 # 데이터 기준 시점 (예: "2026-Q1")
    source: str = ""                # 출처/검증 메모

    @classmethod
    def from_dict(cls, ticker: str, d: dict) -> "Holding":
        stakes = [
            Stake(ticker=str(s["ticker"]).zfill(6), name=s.get("name", ""), pct=float(s["pct"]))
            for s in d.get("listed_stakes", [])
        ]
        return cls(
            ticker=str(ticker).zfill(6),
            name=d.get("name", ""),
            kind=d.get("kind", "pure_holding"),
            listed_stakes=stakes,
            other_nav_eok=float(d.get("other_nav_eok", 0) or 0),
            own_business_eok=float(d.get("own_business_eok", 0) or 0),
            net_debt_eok=float(d.get("net_debt_eok", 0) or 0),
            as_of=str(d.get("as_of", "")),
            source=str(d.get("source", "")),
        )


@dataclass(frozen=True)
class StakeValue:
    """상장 자회사 1건의 시가 평가 결과."""
    ticker: str
    name: str
    pct: float
    sub_market_cap: float | None    # 자회사 전체 시총 (원). None = 시총 결측
    stake_value: float              # 보유 지분 가치 = sub_market_cap × pct/100 (원, 결측 시 0)
    missing: bool = False


@dataclass(frozen=True)
class NavResult:
    ticker: str
    name: str
    kind: str
    holding_market_cap: float | None        # 지주사 자신의 시총 (원)
    stake_values: list[StakeValue]
    listed_stake_value: float               # Σ 상장 지분 가치 (원)
    other_nav: float                        # 비상장 (원)
    own_business: float                     # 자체사업 (원)
    net_debt: float                         # 순부채 (원, +가 부채)
    nav: float                              # 최종 NAV (원)
    as_of: str = ""
    source: str = ""
    missing_subs: list[str] = field(default_factory=list)  # 시총 결측 자회사

    @property
    def discount(self) -> float | None:
        """할인율 = (지주사 시총 − NAV) / NAV. 음수 = 할인거래(NAV보다 싸다). NAV<=0이면 None."""
        if self.holding_market_cap is None or self.nav <= 0:
            return None
        return (self.holding_market_cap - self.nav) / self.nav

    @property
    def discount_pct(self) -> float | None:
        d = self.discount
        return round(d * 100, 1) if d is not None else None

    @property
    def nav_per_market_cap(self) -> float | None:
        """NAV / 시총 배율. 2.0이면 시총의 2배 가치를 들고 있다는 뜻(50% 할인)."""
        if self.holding_market_cap and self.holding_market_cap > 0:
            return round(self.nav / self.holding_market_cap, 3)
        return None

    @property
    def stake_breakdown(self) -> list[tuple[str, float]]:
        """NAV에서 각 상장 지분이 차지하는 비중 (name, weight 0~1) — 내림차순."""
        if self.nav <= 0:
            return []
        rows = [(sv.name, sv.stake_value / self.nav) for sv in self.stake_values if sv.stake_value > 0]
        return sorted(rows, key=lambda x: x[1], reverse=True)


def compute_nav(holding: Holding, market_caps: dict[str, float | None]) -> NavResult:
    """지주사 NAV를 계산한다.

    market_caps: {ticker: 시총(원)} — 지주사 자신 + 모든 상장 자회사 시총을 담은 dict.
                 결측(None/누락)인 자회사는 가치 0으로 처리하고 missing_subs에 기록한다.
    """
    stake_values: list[StakeValue] = []
    missing: list[str] = []
    total_stake = 0.0
    for s in holding.listed_stakes:
        cap = market_caps.get(s.ticker)
        if cap is None:
            missing.append(s.ticker)
            stake_values.append(StakeValue(s.ticker, s.name, s.pct, None, 0.0, missing=True))
            continue
        val = cap * (s.pct / 100.0)
        total_stake += val
        stake_values.append(StakeValue(s.ticker, s.name, s.pct, cap, val))

    other_nav = holding.other_nav_eok * EOK
    own_business = holding.own_business_eok * EOK
    net_debt = holding.net_debt_eok * EOK
    nav = total_stake + other_nav + own_business - net_debt

    return NavResult(
        ticker=holding.ticker,
        name=holding.name,
        kind=holding.kind,
        holding_market_cap=market_caps.get(holding.ticker),
        stake_values=stake_values,
        listed_stake_value=total_stake,
        other_nav=other_nav,
        own_business=own_business,
        net_debt=net_debt,
        nav=nav,
        as_of=holding.as_of,
        source=holding.source,
        missing_subs=missing,
    )


def discount_signal(discount: float | None, hist_median: float | None,
                    hist_std: float | None, z_threshold: float = 1.0) -> str:
    """할인율의 역사적 z-score 기반 관측 라벨 (관측 전용, 매수신호 아님).

    할인이 역사적 평균보다 z_threshold 이상 깊으면 "할인확대"(평균회귀 후보),
    반대로 얕으면 "할인축소"(반사이익 진행/소멸). 밸류트랩 회피를 위해 핵심 자회사
    반등 동반 여부는 scan/백테스트가 별도 판정한다.
    """
    if discount is None or hist_median is None or hist_std is None or hist_std <= 0:
        return "데이터부족"
    z = (discount - hist_median) / hist_std
    if z <= -z_threshold:
        return "할인확대"   # 평소보다 더 싸다 → 평균회귀 매수 후보(단, 핵심자산 반등 확인 필수)
    if z >= z_threshold:
        return "할인축소"   # 평소보다 덜 싸다 → 반사이익 진행/차익 구간
    return "정상밴드"
