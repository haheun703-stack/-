"""매크로 전략가 서브에이전트 — 시장 레짐, 섹터 로테이션, 시장 폭 분석

시장 전체의 매크로 환경을 진단하고 섹터별 순환 타이밍,
글로벌 리스크 성향을 종합 판단하는 에이전트.
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.entities.macro_models import (
    MacroRegimeAnalysis,
    MarketBreadth,
    SectorRotation,
)
from src.use_cases.ports import MacroAnalystPort

# ─── 섹터 목록 (config/settings.yaml 동일) ──────────────────
SECTORS = [
    "반도체", "자동차", "배터리", "바이오", "금융",
    "화학", "철강금속", "유통소비", "IT_SW", "IT_HW",
    "건설", "조선해운", "통신미디어", "에너지",
]

# ─── 시스템 프롬프트 ─────────────────────────────────────────

REGIME_SYSTEM_PROMPT = """당신은 한국 주식시장 매크로 전략가(Macro Analyst)입니다.
시장 전체의 거시적 환경을 분석하여 현재 시장 레짐을 판별합니다.

## 역할
- KOSPI/KOSDAQ 지수, 환율, 금리, 글로벌 지수를 종합하여 시장 레짐 분류
- 레짐 전환 확률과 방향 예측
- 주요 매크로 요인 식별

## 레짐 분류 기준
- bull: 강세장 — 지수 상승 추세, 외국인 순매수, 글로벌 위험선호
- recovery: 회복기 — 바닥 탈출, 선행지표 개선, 거래량 증가
- sideways: 횡보 — 방향성 부재, 혼조세, 거래량 감소
- correction: 조정기 — 상승 후 눌림, 이격도 과열 해소
- bear: 약세장 — 하락 추세, 외국인 순매도, 글로벌 위험회피
- crisis: 위기 — 급락, 공포 지수 급등, 유동성 경색

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "regime": "bull|recovery|sideways|correction|bear|crisis",
  "regime_confidence": 0.75,
  "regime_duration_days": 15,
  "transition_probability": 0.3,
  "transition_direction": "전환 방향 (예: sideways→recovery)",
  "key_factors": ["요인1", "요인2", "요인3"],
  "reasoning": "종합 판단 근거"
}
```"""

SECTOR_SYSTEM_PROMPT = """당신은 한국 주식시장 매크로 전략가(Macro Analyst)입니다.
섹터별 상대강도와 로테이션 타이밍을 분석합니다.

## 역할
- 14개 섹터의 상대 모멘텀 판별 (leading / improving / weakening / lagging)
- 경기 사이클 기반 섹터 로테이션 시그널 판단
- 매수 유망 섹터와 회피 섹터 추천

## 섹터 목록
반도체, 자동차, 배터리, 바이오, 금융, 화학, 철강금속, 유통소비,
IT_SW, IT_HW, 건설, 조선해운, 통신미디어, 에너지

## 로테이션 시그널 분류
- none: 뚜렷한 로테이션 없음
- early_cycle: 초기 회복 (금융, 경기민감주 선도)
- mid_cycle: 중기 확장 (IT, 산업재 선도)
- late_cycle: 후기 과열 (에너지, 소재 선도)
- defensive: 방어 전환 (유틸리티, 통신, 필수소비재 선도)

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "leading_sectors": [
    {"name": "반도체", "momentum": "leading", "score": 85}
  ],
  "lagging_sectors": [
    {"name": "건설", "momentum": "lagging", "score": 25}
  ],
  "rotation_signal": "none|early_cycle|mid_cycle|late_cycle|defensive",
  "recommended_sectors": ["반도체", "자동차"],
  "avoid_sectors": ["건설", "철강금속"],
  "reasoning": "섹터 로테이션 판단 근거"
}
```"""

BREADTH_SYSTEM_PROMPT = """당신은 한국 주식시장 매크로 전략가(Macro Analyst)입니다.
시장 폭(Market Breadth)을 분석하여 시장 내부 건전성을 진단합니다.

## 역할
- 상승/하락 종목비 (Advance-Decline Ratio) 해석
- 이동평균선 위 종목 비율로 시장 참여도 판단
- 신고가/신저가 비율로 추세 강도 측정
- 시장 폭 다이버전스 (지수는 상승하나 폭이 좁아지는 현상) 감지

## 시장 폭 시그널 분류
- bullish: 상승 폭 양호 — 다수 종목 참여, 건전한 상승
- neutral: 혼조 — 뚜렷한 방향성 없음
- bearish: 하락 폭 확대 — 소수 종목만 지수 견인, 취약
- divergence: 다이버전스 — 지수와 폭의 괴리 (추세 전환 경고)

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "advance_decline_ratio": 1.3,
  "new_highs": 45,
  "new_lows": 12,
  "pct_above_ma20": 0.62,
  "pct_above_ma60": 0.48,
  "pct_above_ma200": 0.55,
  "breadth_thrust": false,
  "breadth_signal": "bullish|neutral|bearish|divergence",
  "reasoning": "시장 폭 판단 근거"
}
```"""


# ─── 데이터 포매팅 함수 ─────────────────────────────────────

def _format_market_data(market_data: dict) -> str:
    """시장 데이터를 레짐 분석용 프롬프트 텍스트로 변환"""
    lines = ["[한국 시장 지수]"]

    # KOSPI / KOSDAQ
    kospi = market_data.get("kospi", {})
    if kospi:
        lines.append(
            f"KOSPI: {kospi.get('close', 'N/A'):,} "
            f"(전일대비 {kospi.get('change_pct', 0):+.2f}%, "
            f"5일 {kospi.get('change_5d_pct', 0):+.2f}%, "
            f"20일 {kospi.get('change_20d_pct', 0):+.2f}%)"
        )
    kosdaq = market_data.get("kosdaq", {})
    if kosdaq:
        lines.append(
            f"KOSDAQ: {kosdaq.get('close', 'N/A'):,} "
            f"(전일대비 {kosdaq.get('change_pct', 0):+.2f}%, "
            f"5일 {kosdaq.get('change_5d_pct', 0):+.2f}%, "
            f"20일 {kosdaq.get('change_20d_pct', 0):+.2f}%)"
        )

    # 환율 / 금리
    lines.append("\n[환율 및 금리]")
    usd_krw = market_data.get("usd_krw", {})
    if usd_krw:
        lines.append(
            f"USD/KRW: {usd_krw.get('close', 'N/A'):,.1f} "
            f"(전일대비 {usd_krw.get('change_pct', 0):+.2f}%)"
        )
    bond_10y = market_data.get("bond_10y", {})
    if bond_10y:
        lines.append(
            f"한국 10년물 국채금리: {bond_10y.get('yield_pct', 'N/A'):.2f}% "
            f"(전일대비 {bond_10y.get('change_bp', 0):+.1f}bp)"
        )

    # VKOSPI (변동성 지수)
    vkospi = market_data.get("vkospi", {})
    if vkospi:
        lines.append(
            f"VKOSPI: {vkospi.get('close', 'N/A'):.2f} "
            f"(전일대비 {vkospi.get('change_pct', 0):+.2f}%)"
        )

    # 글로벌 지수
    global_indices = market_data.get("global_indices", {})
    if global_indices:
        lines.append("\n[글로벌 지수 (전일 종가 기준)]")
        for name, data in global_indices.items():
            if isinstance(data, dict):
                lines.append(
                    f"{name}: {data.get('close', 'N/A'):,.2f} "
                    f"({data.get('change_pct', 0):+.2f}%)"
                )
            else:
                lines.append(f"{name}: {data}")

    # 외국인/기관 동향
    investor_flow = market_data.get("investor_flow", {})
    if investor_flow:
        lines.append("\n[투자자 동향 (억원)]")
        lines.append(
            f"외국인 순매수: {investor_flow.get('foreign_net', 0):+,.0f}억 "
            f"(5일 누적: {investor_flow.get('foreign_net_5d', 0):+,.0f}억)"
        )
        lines.append(
            f"기관 순매수: {investor_flow.get('inst_net', 0):+,.0f}억 "
            f"(5일 누적: {investor_flow.get('inst_net_5d', 0):+,.0f}억)"
        )

    # 추가 컨텍스트
    extra = market_data.get("extra_context", "")
    if extra:
        lines.append(f"\n[추가 컨텍스트]\n{extra}")

    return "\n".join(lines)


def _format_sector_data(sector_data: dict) -> str:
    """섹터 데이터를 로테이션 분석용 프롬프트 텍스트로 변환"""
    lines = ["[섹터별 수익률 및 모멘텀]"]

    sectors = sector_data.get("sectors", [])
    if sectors:
        lines.append(
            f"{'섹터':<12} {'1일':>8} {'5일':>8} {'20일':>8} "
            f"{'거래대금비':>10} {'외국인순매수':>12}"
        )
        lines.append("-" * 70)
        for s in sectors:
            lines.append(
                f"{s.get('name', ''):<12} "
                f"{s.get('change_1d_pct', 0):>+7.2f}% "
                f"{s.get('change_5d_pct', 0):>+7.2f}% "
                f"{s.get('change_20d_pct', 0):>+7.2f}% "
                f"{s.get('volume_ratio', 1.0):>9.2f}x "
                f"{s.get('foreign_net', 0):>+11,.0f}억"
            )

    # 시장 사이클 힌트
    cycle_hint = sector_data.get("cycle_hint", "")
    if cycle_hint:
        lines.append(f"\n[경기 사이클 참고]\n{cycle_hint}")

    # 상대강도 상위/하위
    top_sectors = sector_data.get("top_rs", [])
    if top_sectors:
        lines.append(f"\n[상대강도 상위] {', '.join(top_sectors)}")
    bottom_sectors = sector_data.get("bottom_rs", [])
    if bottom_sectors:
        lines.append(f"[상대강도 하위] {', '.join(bottom_sectors)}")

    return "\n".join(lines)


def _format_breadth_data(breadth_data: dict) -> str:
    """시장 폭 데이터를 분석용 프롬프트 텍스트로 변환"""
    lines = ["[시장 폭 데이터]"]

    # 상승/하락 종목 수
    lines.append(
        f"상승 종목: {breadth_data.get('advancing', 0):,}개 / "
        f"하락 종목: {breadth_data.get('declining', 0):,}개 / "
        f"보합: {breadth_data.get('unchanged', 0):,}개"
    )
    adv = breadth_data.get("advancing", 0)
    dec = breadth_data.get("declining", 1)
    if dec > 0:
        lines.append(f"A/D 비율: {adv / dec:.2f}")

    # 신고가/신저가
    lines.append(
        f"\n신고가: {breadth_data.get('new_highs', 0):,}개 / "
        f"신저가: {breadth_data.get('new_lows', 0):,}개"
    )

    # 이동평균선 위 종목 비율
    lines.append("\n[이동평균선 위 종목 비율]")
    lines.append(f"20일선 위: {breadth_data.get('pct_above_ma20', 0):.1%}")
    lines.append(f"60일선 위: {breadth_data.get('pct_above_ma60', 0):.1%}")
    lines.append(f"200일선 위: {breadth_data.get('pct_above_ma200', 0):.1%}")

    # 최근 A/D 라인 추세
    ad_line_trend = breadth_data.get("ad_line_trend", "")
    if ad_line_trend:
        lines.append(f"\n[A/D 라인 추세] {ad_line_trend}")

    # 지수 대비 폭 비교 (다이버전스 탐지용)
    index_vs_breadth = breadth_data.get("index_vs_breadth", "")
    if index_vs_breadth:
        lines.append(f"[지수 vs 시장 폭] {index_vs_breadth}")

    # 거래대금 집중도
    concentration = breadth_data.get("volume_concentration", "")
    if concentration:
        lines.append(f"[거래대금 집중도] {concentration}")

    return "\n".join(lines)


# ─── 에이전트 구현 ─────────────────────────────────────────

class MacroAnalystAgent(BaseAgent, MacroAnalystPort):
    """매크로 전략가 에이전트 — MacroAnalystPort 구현

    시장 레짐 분류, 섹터 로테이션 타이밍, 시장 폭 분석을 수행하여
    포트폴리오 전략의 매크로 방향성을 제공한다.
    """

    async def analyze_regime(self, market_data: dict) -> dict:
        """시장 레짐 분석 — 현재 시장 국면 판별 및 전환 확률 예측

        Args:
            market_data: KOSPI/KOSDAQ 지수, 환율, 금리, VKOSPI,
                         글로벌 지수, 투자자 동향 등 매크로 데이터

        Returns:
            MacroRegimeAnalysis를 딕셔너리로 변환한 결과
        """
        user_prompt = _format_market_data(market_data)
        data = await self._ask_claude_json(REGIME_SYSTEM_PROMPT, user_prompt)

        result = MacroRegimeAnalysis(
            regime=data.get("regime", "sideways"),
            regime_confidence=data.get("regime_confidence", 0.5),
            regime_duration_days=data.get("regime_duration_days", 0),
            transition_probability=data.get("transition_probability", 0.0),
            transition_direction=data.get("transition_direction", ""),
            key_factors=data.get("key_factors", []),
            reasoning=data.get("reasoning", ""),
        )
        return result.to_dict()

    async def analyze_sector_rotation(self, sector_data: dict) -> dict:
        """섹터 로테이션 분석 — 섹터별 모멘텀과 순환 타이밍 판별

        Args:
            sector_data: 14개 섹터별 수익률, 거래대금, 외국인 수급 등

        Returns:
            SectorRotation을 딕셔너리로 변환한 결과
        """
        user_prompt = _format_sector_data(sector_data)
        data = await self._ask_claude_json(SECTOR_SYSTEM_PROMPT, user_prompt)

        result = SectorRotation(
            leading_sectors=data.get("leading_sectors", []),
            lagging_sectors=data.get("lagging_sectors", []),
            rotation_signal=data.get("rotation_signal", "none"),
            recommended_sectors=data.get("recommended_sectors", []),
            avoid_sectors=data.get("avoid_sectors", []),
            reasoning=data.get("reasoning", ""),
        )
        return result.__dict__

    async def analyze_breadth(self, breadth_data: dict) -> dict:
        """시장 폭 분석 — 시장 내부 건전성 진단

        Args:
            breadth_data: 상승/하락 종목 수, 신고가/신저가,
                          이동평균선 위 비율, A/D 라인 추세 등

        Returns:
            MarketBreadth를 딕셔너리로 변환한 결과
        """
        user_prompt = _format_breadth_data(breadth_data)
        data = await self._ask_claude_json(BREADTH_SYSTEM_PROMPT, user_prompt)

        result = MarketBreadth(
            advance_decline_ratio=data.get("advance_decline_ratio", 1.0),
            new_highs=data.get("new_highs", 0),
            new_lows=data.get("new_lows", 0),
            pct_above_ma20=data.get("pct_above_ma20", 0.5),
            pct_above_ma60=data.get("pct_above_ma60", 0.5),
            pct_above_ma200=data.get("pct_above_ma200", 0.5),
            breadth_thrust=data.get("breadth_thrust", False),
            breadth_signal=data.get("breadth_signal", "neutral"),
            reasoning=data.get("reasoning", ""),
        )
        return result.__dict__
