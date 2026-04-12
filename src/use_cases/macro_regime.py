"""매크로 시장 흐름 엔진 — 금리/물가/환율/ERP 기반 시장 판단

유저의 매크로 프레임워크를 시스템화:
  1. 금리 흐름: 기준금리 추세 + 국고채 장단기 스프레드 → 긴축/완화
  2. 물가 흐름: CPI 추세 → 인플레/안정/둔화
  3. 환율 흐름: 원/달러 추세 → 원화강세/약세 → 수출주/내수주
  4. 주식 매력도 (ERP): 주식 vs 채권 상대 매력
  5. 섹터 유불리: 금리/환율/물가 조합별 수혜 섹터 추천

모든 판단의 핵심: "이전에 얼마였는데 → 지금 얼마다 → 그래서 이 방향이다"
ECOS API 시계열 데이터(1/3개월 비교)를 직접 사용하므로 히스토리 축적 불필요.

입력:
  - data/ecos_macro.json         (ECOS 시계열 + 최신값)
  - data/regime_macro_signal.json (KOSPI 시장 국면)

출력:
  - data/macro/macro_regime.json (매크로 종합 판단)

사용:
  python -u -X utf8 -m src.use_cases.macro_regime
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
MACRO_DIR = DATA_DIR / "macro"
ECOS_JSON = DATA_DIR / "ecos_macro.json"
REGIME_SIGNAL_JSON = DATA_DIR / "regime_macro_signal.json"
OUTPUT_PATH = MACRO_DIR / "macro_regime.json"

# ═══════════════════════════════════════════════════════════
#  섹터 유불리 매핑 (매크로 조건별)
# ═══════════════════════════════════════════════════════════

RATE_DOWN_BENEFIT = ["성장주", "IT/반도체", "바이오", "부동산/건설", "유틸리티"]
RATE_UP_BENEFIT = ["은행/금융", "보험"]
KRW_WEAK_BENEFIT = ["자동차", "조선", "반도체", "IT하드웨어", "화학"]
KRW_STRONG_BENEFIT = ["내수소비", "항공/여행", "수입유통"]
INFLATION_BENEFIT = ["에너지", "원자재/소재", "금융", "필수소비재"]
DEFLATION_BENEFIT = ["성장주", "IT/반도체", "임의소비재"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── 금리 ─────────────────────────────────────────────────


def _judge_rate(indicators: dict) -> dict:
    """금리 흐름 판단 — ECOS 시계열 기반.

    핵심 비교:
      - 기준금리: 3개월전 vs 현재 → 인상/동결/인하
      - 국고채10년: 1개월전 vs 현재 → 시장금리 방향
      - 장단기 스프레드(10년-3년): 역전 여부 → 경기 전망
    """
    base_data = indicators.get("base_rate", {})
    bond_3y_data = indicators.get("bond_3y", {})
    bond_10y_data = indicators.get("bond_10y", {})

    base = base_data.get("value")
    bond_3y = bond_3y_data.get("value")
    bond_10y = bond_10y_data.get("value")

    result = {
        "base_rate": base,
        "bond_3y": bond_3y,
        "bond_10y": bond_10y,
        "spread_10y_3y": None,
        "direction": "보합",  # 긴축 / 완화 / 보합
        "signal": "",
        "details": [],
        "score": 50,
    }

    details = []
    score = 50

    # ── 1) 기준금리 추이 ──
    base_prev_3m = base_data.get("prev_3m")
    base_prev_1m = base_data.get("prev_1m")
    if base is not None and base_prev_3m is not None:
        change = base - base_prev_3m
        if change < -0.25:
            details.append(f"기준금리: 3개월전 {base_prev_3m}% → 현재 {base}% (인하 {change:+.2f}%p)")
            score += 15  # 금리 인하 = 주식에 우호
        elif change > 0.25:
            details.append(f"기준금리: 3개월전 {base_prev_3m}% → 현재 {base}% (인상 {change:+.2f}%p)")
            score -= 15
        else:
            details.append(f"기준금리: {base}% 동결 유지 (3개월전도 {base_prev_3m}%)")

    # ── 2) 국고채10년 추이 (시장금리) ──
    bond10_chg_1m = bond_10y_data.get("change_1m")
    bond10_chg_3m = bond_10y_data.get("change_3m")
    bond10_prev_1m = bond_10y_data.get("prev_1m")
    bond10_prev_3m = bond_10y_data.get("prev_3m")

    if bond_10y is not None and bond10_prev_1m is not None:
        if bond10_chg_1m > 0.3:
            details.append(
                f"국고채10년: 1개월전 {bond10_prev_1m}% → 현재 {bond_10y}% "
                f"(+{bond10_chg_1m:.2f}%p 급등, 긴축 신호)"
            )
            score -= 15
        elif bond10_chg_1m < -0.3:
            details.append(
                f"국고채10년: 1개월전 {bond10_prev_1m}% → 현재 {bond_10y}% "
                f"({bond10_chg_1m:+.2f}%p 하락, 완화 신호)"
            )
            score += 15
        else:
            details.append(
                f"국고채10년: 1개월전 {bond10_prev_1m}% → 현재 {bond_10y}% "
                f"({bond10_chg_1m:+.2f}%p 소폭 변동)"
            )

    # 3개월 추세도 참고
    if bond10_chg_3m is not None and bond10_prev_3m is not None:
        if bond10_chg_3m > 0.5:
            details.append(f"3개월간 국고채10년 +{bond10_chg_3m:.2f}%p 상승 (금리 상승 추세)")
            score -= 5
        elif bond10_chg_3m < -0.5:
            details.append(f"3개월간 국고채10년 {bond10_chg_3m:+.2f}%p 하락 (금리 하락 추세)")
            score += 5

    # ── 3) 장단기 스프레드 ──
    if bond_10y is not None and bond_3y is not None:
        spread = round(bond_10y - bond_3y, 3)
        result["spread_10y_3y"] = spread
        if spread < 0:
            details.append(f"장단기 금리 역전 ({spread:+.3f}%p) — 경기침체 경고")
            score -= 10
        elif spread < 0.15:
            details.append(f"장단기 스프레드 축소 ({spread:.3f}%p) — 경기둔화 가능성")
            score -= 3
        else:
            details.append(f"장단기 스프레드 {spread:.3f}%p — 정상")

    # 방향 결정
    score = max(0, min(100, score))
    if score >= 60:
        result["direction"] = "완화"
    elif score <= 40:
        result["direction"] = "긴축"
    else:
        result["direction"] = "보합"

    result["score"] = score
    result["details"] = details
    result["signal"] = details[0] if details else ""
    return result


# ─── 물가 ─────────────────────────────────────────────────


def _judge_inflation(indicators: dict) -> dict:
    """물가 흐름 판단 — CPI 시계열 기반."""
    cpi_data = indicators.get("cpi", {})
    cpi = cpi_data.get("value")

    result = {
        "cpi": cpi,
        "direction": "안정",  # 상승 / 안정 / 둔화
        "signal": "",
        "details": [],
        "score": 50,
    }

    if cpi is None:
        return result

    details = []
    score = 55  # 기본: 약간 우호 (물가 자체가 없으면 걱정할 게 없다)

    cpi_prev_1m = cpi_data.get("prev_1m")
    cpi_prev_3m = cpi_data.get("prev_3m")
    cpi_chg_1m = cpi_data.get("change_1m")
    cpi_chg_3m = cpi_data.get("change_3m")
    cpi_chg_1m_pct = cpi_data.get("change_1m_pct")
    cpi_chg_3m_pct = cpi_data.get("change_3m_pct")

    if cpi_prev_1m is not None:
        if cpi_chg_1m_pct > 0.3:
            details.append(f"CPI: 1개월전 {cpi_prev_1m} → 현재 {cpi} (+{cpi_chg_1m_pct:.1f}%, 물가 상승 압력)")
            score -= 15
        elif cpi_chg_1m_pct < -0.1:
            details.append(f"CPI: 1개월전 {cpi_prev_1m} → 현재 {cpi} ({cpi_chg_1m_pct:+.1f}%, 물가 하락)")
            score += 10
        else:
            details.append(f"CPI: 1개월전 {cpi_prev_1m} → 현재 {cpi} ({cpi_chg_1m_pct:+.1f}%, 안정)")

    if cpi_prev_3m is not None:
        details.append(f"3개월간 CPI {cpi_prev_3m} → {cpi} ({cpi_chg_3m_pct:+.1f}%)")
        if cpi_chg_3m_pct > 1.5:
            score -= 10
        elif cpi_chg_3m_pct < 0:
            score += 5

    score = max(0, min(100, score))
    if score >= 55:
        result["direction"] = "안정"
    elif score <= 35:
        result["direction"] = "상승"
    else:
        result["direction"] = "소폭상승"

    result["score"] = score
    result["details"] = details
    result["signal"] = details[0] if details else f"CPI {cpi}"
    return result


# ─── 환율 ─────────────────────────────────────────────────


def _judge_fx(indicators: dict) -> dict:
    """환율 흐름 판단 — 원/달러 시계열 기반.

    원/달러 상승 = 원화약세 = 수출주 유리, 수입비용 증가
    원/달러 하락 = 원화강세 = 내수주 유리, 수입비용 감소
    """
    fx_data = indicators.get("usd_krw", {})
    usd_krw = fx_data.get("value")

    result = {
        "usd_krw": usd_krw,
        "direction": "보합",  # 원화약세 / 원화강세 / 보합
        "signal": "",
        "details": [],
        "score": 50,
    }

    if usd_krw is None:
        return result

    details = []
    score = 50

    # 절대 수준
    if usd_krw >= 1450:
        level = "고환율권"
        score -= 8  # 고환율 = 불안 요소
    elif usd_krw >= 1300:
        level = "중고환율권"
    elif usd_krw >= 1150:
        level = "중저환율권"
        score += 5
    else:
        level = "저환율권"
        score += 8

    # 1개월 추세
    fx_prev_1m = fx_data.get("prev_1m")
    fx_chg_1m = fx_data.get("change_1m")
    fx_chg_1m_pct = fx_data.get("change_1m_pct")

    if fx_prev_1m is not None:
        if fx_chg_1m_pct > 2.0:
            details.append(
                f"원/달러: 1개월전 {fx_prev_1m:.0f}원 → 현재 {usd_krw:.0f}원 "
                f"(+{fx_chg_1m:.0f}원, {fx_chg_1m_pct:+.1f}% 급등, 원화 급락)"
            )
            score -= 15
        elif fx_chg_1m_pct > 0.5:
            details.append(
                f"원/달러: 1개월전 {fx_prev_1m:.0f}원 → 현재 {usd_krw:.0f}원 "
                f"(+{fx_chg_1m:.0f}원, 원화 약세 진행)"
            )
            score -= 8
        elif fx_chg_1m_pct < -2.0:
            details.append(
                f"원/달러: 1개월전 {fx_prev_1m:.0f}원 → 현재 {usd_krw:.0f}원 "
                f"({fx_chg_1m:.0f}원, 원화 강세 전환)"
            )
            score += 10
        elif fx_chg_1m_pct < -0.5:
            details.append(
                f"원/달러: 1개월전 {fx_prev_1m:.0f}원 → 현재 {usd_krw:.0f}원 "
                f"({fx_chg_1m:.0f}원, 원화 소폭 강세)"
            )
            score += 5
        else:
            details.append(
                f"원/달러: 1개월전 {fx_prev_1m:.0f}원 → 현재 {usd_krw:.0f}원 (안정)"
            )

    # 3개월 추세
    fx_prev_3m = fx_data.get("prev_3m")
    fx_chg_3m = fx_data.get("change_3m")
    fx_chg_3m_pct = fx_data.get("change_3m_pct")
    if fx_prev_3m is not None:
        details.append(f"3개월간 {fx_prev_3m:.0f}원 → {usd_krw:.0f}원 ({fx_chg_3m_pct:+.1f}%)")

    details.append(f"현재 {level} ({usd_krw:.0f}원)")

    score = max(0, min(100, score))
    if score >= 55:
        result["direction"] = "원화강세"
    elif score <= 40:
        result["direction"] = "원화약세"
    else:
        result["direction"] = "보합"

    result["score"] = score
    result["details"] = details
    result["signal"] = details[0] if details else f"{usd_krw:.0f}원"
    return result


# ─── ERP (주식 매력도) ────────────────────────────────────


def _judge_erp(ecos_data: dict) -> dict:
    """주식 매력도 (ERP = 주식 기대수익률 - 무위험수익률)."""
    erp = ecos_data.get("erp")
    kospi = ecos_data.get("kospi", {})
    per = kospi.get("per")
    bond_10y = ecos_data.get("indicators", {}).get("bond_10y", {}).get("value")

    result = {
        "erp": erp,
        "kospi_per": per,
        "bond_10y": bond_10y,
        "verdict": "판단불가",
        "signal": "",
        "details": [],
        "score": 50,
    }

    if erp is None:
        if per and bond_10y:
            earnings_yield = (1 / per) * 100
            erp = round(earnings_yield - bond_10y, 2)
            result["erp"] = erp
        else:
            result["signal"] = "주식매력도 계산 불가 (KOSPI PER 미수집, 평일 자동 수집)"
            result["details"] = ["주말/공휴일에는 KOSPI PER 데이터가 없어 ERP 계산 불가"]
            return result

    earnings_yield = round((1 / per) * 100, 2) if per and per > 0 else None
    details = []

    if earnings_yield and bond_10y:
        details.append(
            f"주식 기대수익률 {earnings_yield:.1f}% (PER {per:.1f}배의 역수) "
            f"vs 국고채10년 {bond_10y:.2f}%"
        )
        details.append(f"ERP = {earnings_yield:.1f}% - {bond_10y:.2f}% = {erp:+.1f}%p")

    if erp >= 4:
        result["verdict"] = "주식 매우 유리"
        result["signal"] = f"ERP {erp:.1f}%p — 채권 대비 주식이 매우 매력적"
        result["score"] = 85
    elif erp >= 2:
        result["verdict"] = "주식 우위"
        result["signal"] = f"ERP {erp:.1f}%p — 주식이 채권보다 유리"
        result["score"] = 70
    elif erp >= 0:
        result["verdict"] = "비슷"
        result["signal"] = f"ERP {erp:.1f}%p — 주식과 채권 매력 비슷"
        result["score"] = 50
    else:
        result["verdict"] = "채권 유리"
        result["signal"] = f"ERP {erp:.1f}%p — 채권이 주식보다 매력적"
        result["score"] = 30

    result["details"] = details
    return result


# ─── 섹터 유불리 ──────────────────────────────────────────


def _recommend_sectors(rate: dict, inflation: dict, fx: dict) -> dict:
    """매크로 조건별 수혜/피해 섹터 추천."""
    from collections import Counter

    benefit = []
    avoid = []

    rd = rate["direction"]
    if rd == "완화":
        benefit.extend(RATE_DOWN_BENEFIT)
        avoid.extend(RATE_UP_BENEFIT)
    elif rd == "긴축":
        benefit.extend(RATE_UP_BENEFIT)
        avoid.extend(["부동산/건설", "성장주"])

    fd = fx["direction"]
    if fd == "원화약세":
        benefit.extend(KRW_WEAK_BENEFIT)
        avoid.extend(["항공/여행"])
    elif fd == "원화강세":
        benefit.extend(KRW_STRONG_BENEFIT)
        avoid.extend(["조선"])

    inf_d = inflation["direction"]
    if inf_d in ("상승", "소폭상승"):
        benefit.extend(INFLATION_BENEFIT)
        avoid.extend(["성장주", "IT/반도체"])
    elif inf_d == "둔화":
        benefit.extend(DEFLATION_BENEFIT)

    # 중복 제거 + 상쇄
    benefit_count = Counter(benefit)
    avoid_count = Counter(avoid)
    for sec in list(benefit_count.keys()):
        if sec in avoid_count:
            b, a = benefit_count[sec], avoid_count[sec]
            if b > a:
                del avoid_count[sec]
                benefit_count[sec] = b - a
            elif a > b:
                del benefit_count[sec]
                avoid_count[sec] = a - b
            else:
                del benefit_count[sec]
                del avoid_count[sec]

    return {
        "benefit_sectors": [s for s, _ in benefit_count.most_common(5)],
        "avoid_sectors": [s for s, _ in avoid_count.most_common(3)],
        "logic": f"금리={rd}, 환율={fd}, 물가={inf_d}",
    }


# ─── 대응 전략 ────────────────────────────────────────────


def _build_response_strategy(rate: dict, inflation: dict, fx: dict,
                             erp: dict, sectors: dict, overall_score: float) -> dict:
    """매크로 조건별 대응 전략 생성.

    각 지표의 direction을 보고 "그래서 어떻게 해야 하나"를 구체적으로 알려준다.
    """
    actions = []  # 개별 조건별 행동 지침
    portfolio_tips = []  # 포트폴리오 조정 팁

    # ── 금리 대응 ──
    rd = rate.get("direction", "보합")
    if rd == "긴축":
        actions.append({
            "condition": "금리 상승 (긴축)",
            "actions": [
                "은행/금융/보험주 비중 확대 — 금리 상승 수혜",
                "성장주/바이오 신규 진입 자제 — 할인율 상승으로 밸류 부담",
                "부동산/건설주 경계 — 대출 부담 증가",
            ],
        })
        portfolio_tips.append("고배당·가치주 중심 포트폴리오 유지")
    elif rd == "완화":
        actions.append({
            "condition": "금리 하락 (완화)",
            "actions": [
                "성장주/IT/바이오 진입 기회 — 할인율 하락으로 밸류 개선",
                "부동산/건설주 관심 — 대출 부담 완화",
                "은행주 비중 축소 고려 — 예대마진 축소 가능",
            ],
        })
        portfolio_tips.append("성장주 비중 확대 기회 탐색")
    else:
        actions.append({
            "condition": "금리 보합",
            "actions": ["금리 방향 전환 시그널 주시 — 현재 포지션 유지"],
        })

    # ── 환율 대응 ──
    fd = fx.get("direction", "보합")
    usd_krw = fx.get("usd_krw", 0)
    if fd == "원화약세":
        actions.append({
            "condition": f"원화 약세 ({usd_krw:.0f}원)",
            "actions": [
                "수출주(자동차/조선/반도체) 비중 확대 — 환율 수혜",
                "항공/여행주 회피 — 원가 부담 증가",
                "달러 자산 보유 유지 — 환차익 기대",
            ],
        })
        portfolio_tips.append("수출 비중 높은 대형주 선호")
    elif fd == "원화강세":
        actions.append({
            "condition": f"원화 강세 ({usd_krw:.0f}원)",
            "actions": [
                "내수/소비주 비중 확대 — 구매력 개선",
                "항공/여행주 관심 — 해외 비용 절감",
                "수출주 비중 축소 고려 — 환율 역풍",
            ],
        })
        portfolio_tips.append("내수 소비 관련주로 회전")
    else:
        actions.append({
            "condition": "환율 보합",
            "actions": ["환율 변동성 확대 대비 — 수출/내수 균형 포트폴리오 유지"],
        })

    # ── 물가 대응 ──
    inf_d = inflation.get("direction", "안정")
    if inf_d in ("상승", "소폭상승"):
        actions.append({
            "condition": f"물가 {inf_d}",
            "actions": [
                "에너지/원자재/소재주 관심 — 인플레 수혜",
                "필수소비재 방어 포지션 — 가격 전가력 보유 기업",
                "고PER 성장주 경계 — 실질 금리 상승 부담",
            ],
        })
        portfolio_tips.append("인플레 방어 자산(에너지/금융) 비중 확대")
    elif inf_d == "둔화":
        actions.append({
            "condition": "물가 둔화",
            "actions": [
                "성장주/IT 진입 기회 — 금리 인하 기대 동반",
                "임의소비재 관심 — 소비 여력 회복",
            ],
        })
    else:
        actions.append({
            "condition": "물가 안정",
            "actions": ["물가 리스크 낮음 — 밸류에이션 부담 제한적"],
        })

    # ── ERP 대응 ──
    erp_verdict = erp.get("verdict", "판단불가")
    if erp_verdict in ("주식 매우 유리", "주식 우위"):
        actions.append({
            "condition": f"주식 매력도: {erp_verdict}",
            "actions": [
                "주식 비중 확대 유효 — 채권 대비 기대수익 우위",
                "분할 매수 전략 — 매크로 불확실성 감안",
            ],
        })
    elif erp_verdict == "채권 유리":
        actions.append({
            "condition": f"주식 매력도: {erp_verdict}",
            "actions": [
                "주식 비중 축소, 현금/단기채권 비중 확대",
                "신규 매수 보류 — 채권 수익률이 더 매력적",
            ],
        })

    # ── 종합 대응 ──
    if overall_score >= 70:
        summary = "매크로 우호적 — 적극적 포지션 운영 가능"
        cash_action = "현금비중 20% 수준 유지, 나머지는 주식 비중 확대"
        entry_action = "신규 진입 적극 검토, 수혜 섹터 중심 매수"
    elif overall_score >= 55:
        summary = "매크로 양호 — 기본 포지션 유지하며 기회 탐색"
        cash_action = "현금비중 30~40% 유지"
        entry_action = "검증된 종목 위주 선별적 진입"
    elif overall_score >= 40:
        summary = "매크로 주의 — 신중한 접근 필요"
        cash_action = "현금비중 50~60%로 확대"
        entry_action = "신규 진입 최소화, 기존 포지션 리스크 점검"
    else:
        summary = "매크로 비우호적 — 방어 모드 전환"
        cash_action = "현금비중 70~80%로 확대, 손절 라인 엄격 관리"
        entry_action = "신규 매수 자제, 수출주/금융주 등 방어주만 제한적 검토"

    benefit = sectors.get("benefit_sectors", [])
    avoid = sectors.get("avoid_sectors", [])
    sector_tip = ""
    if benefit:
        sector_tip += f"수혜 섹터({', '.join(benefit[:3])}) 중심으로 편입"
    if avoid:
        sector_tip += f", {', '.join(avoid[:2])} 등은 비중 축소"

    return {
        "summary": summary,
        "cash_action": cash_action,
        "entry_action": entry_action,
        "sector_action": sector_tip,
        "portfolio_tips": portfolio_tips,
        "condition_actions": actions,
    }


# ─── 종합 ─────────────────────────────────────────────────


def _compute_overall(rate: dict, inflation: dict, fx: dict, erp: dict,
                     market_signal: dict) -> dict:
    """매크로 종합 점수 + 판정."""
    rate_score = rate.get("score", 50)
    inf_score = inflation.get("score", 50)
    fx_score = fx.get("score", 50)
    erp_score = erp.get("score", 50)
    market_score = market_signal.get("macro_score", 50)

    total = round(
        rate_score * 0.25
        + inf_score * 0.20
        + fx_score * 0.15
        + erp_score * 0.25
        + market_score * 0.15,
        1,
    )

    if total >= 70:
        grade = "우호적"
        stance = "적극적"
        desc = "매크로 환경 우호적 — 포지션 확대 유효"
    elif total >= 55:
        grade = "중립(양호)"
        stance = "기본 유지"
        desc = "매크로 중립~약간 우호 — 기본 포지션 유지"
    elif total >= 40:
        grade = "중립(주의)"
        stance = "신중"
        desc = "매크로 약간 비우호 — 신규 진입 신중"
    else:
        grade = "비우호적"
        stance = "방어적"
        desc = "매크로 비우호적 — 현금비중 확대 권장"

    if total >= 70:
        suggested_cash = 20
    elif total >= 55:
        suggested_cash = 40
    elif total >= 40:
        suggested_cash = 60
    else:
        suggested_cash = 80

    return {
        "score": total,
        "grade": grade,
        "stance": stance,
        "description": desc,
        "suggested_cash_pct": suggested_cash,
        "components": {
            "rate": {"weight": 0.25, "score": rate_score},
            "inflation": {"weight": 0.20, "score": inf_score},
            "fx": {"weight": 0.15, "score": fx_score},
            "erp": {"weight": 0.25, "score": erp_score},
            "market": {"weight": 0.15, "score": market_score},
        },
    }


# ─── 메인 ─────────────────────────────────────────────────


def analyze_macro_regime() -> dict:
    """매크로 시장 흐름 분석 실행."""
    logger.info("=" * 60)
    logger.info("  매크로 시장 흐름 분석")
    logger.info("=" * 60)

    ecos_data = _load_json(ECOS_JSON)
    if not ecos_data.get("indicators"):
        logger.warning("ECOS 데이터 없음 — fetch_ecos_macro.py 먼저 실행 필요")
        return {}

    market_signal = _load_json(REGIME_SIGNAL_JSON)
    indicators = ecos_data["indicators"]

    # 1. 금리 흐름
    rate = _judge_rate(indicators)
    logger.info(f"  금리: {rate['direction']} (점수 {rate['score']})")
    for d in rate["details"]:
        logger.info(f"    {d}")

    # 2. 물가 흐름
    inflation = _judge_inflation(indicators)
    logger.info(f"  물가: {inflation['direction']} (점수 {inflation['score']})")
    for d in inflation["details"]:
        logger.info(f"    {d}")

    # 3. 환율 흐름
    fx = _judge_fx(indicators)
    logger.info(f"  환율: {fx['direction']} (점수 {fx['score']})")
    for d in fx["details"]:
        logger.info(f"    {d}")

    # 4. 주식 매력도
    erp = _judge_erp(ecos_data)
    logger.info(f"  주식매력도: {erp['verdict']} (점수 {erp['score']})")
    for d in erp["details"]:
        logger.info(f"    {d}")

    # 5. 섹터 유불리
    sectors = _recommend_sectors(rate, inflation, fx)
    logger.info(f"  수혜 섹터: {sectors['benefit_sectors']}")
    logger.info(f"  주의 섹터: {sectors['avoid_sectors']}")

    # 6. 종합
    overall = _compute_overall(rate, inflation, fx, erp, market_signal)
    logger.info(f"  종합: {overall['score']}/100 → {overall['grade']} ({overall['stance']})")
    logger.info(f"  → {overall['description']}")

    # 7. 대응 전략
    strategy = _build_response_strategy(rate, inflation, fx, erp, sectors, overall["score"])
    logger.info(f"  대응: {strategy['summary']}")
    logger.info(f"    현금: {strategy['cash_action']}")
    logger.info(f"    진입: {strategy['entry_action']}")
    if strategy["sector_action"]:
        logger.info(f"    섹터: {strategy['sector_action']}")
    for tip in strategy["portfolio_tips"]:
        logger.info(f"    팁: {tip}")
    for ca in strategy["condition_actions"]:
        logger.info(f"    [{ca['condition']}]")
        for a in ca["actions"]:
            logger.info(f"      • {a}")

    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rate": rate,
        "inflation": inflation,
        "fx": fx,
        "erp": erp,
        "sectors": sectors,
        "overall": overall,
        "strategy": strategy,
        "market_phase": market_signal.get("current_regime", ""),
        "market_phase_score": market_signal.get("macro_score", 0),
    }

    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  저장: {OUTPUT_PATH}")

    return output


def main():
    analyze_macro_regime()


if __name__ == "__main__":
    main()
