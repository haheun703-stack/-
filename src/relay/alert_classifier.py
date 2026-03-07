"""
섹터 경보 분류기
========================
섹터 유형별(persistent/event/conditional) 차등 경보 로직.

경보 레벨 0~5:
  5: 전 조건 충족 (EXECUTE)
  4: US 확인 + 뉴스 강 (KR_READY)
  3: US 2개+ 강세 (CONFIRM)
  2: US 1개 강세 or 뉴스만 (WATCH)
  1: 약한 신호
  0: 비활성 (INACTIVE)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def classify_alert(
    sector_key: str,
    sector_config: dict,
    us_leaders_data: dict,
    us_overnight: dict,
    news_score: int,
    news_keywords_matched: list[str],
    kr_leaders_strong: bool,
    negative_news_found: bool = False,
) -> dict:
    """섹터 하나의 경보 레벨을 판정.

    Args:
        sector_key: 섹터 식별자
        sector_config: relay_sectors.yaml의 해당 섹터 설정
        us_leaders_data: us_tracker 결과 중 해당 섹터
        us_overnight: data/us_market/overnight_signal.json
        news_score: 뉴스 키워드 매칭 점수 (0~10)
        news_keywords_matched: 매칭된 키워드 목록
        kr_leaders_strong: 한국 대장주 전일 강세 여부
        negative_news_found: 부정 키워드 발견 여부 (conditional용)

    Returns:
        {"phase": 0~4, "phase_name": str, "alert_level": 0~5,
         "reasons": list, "kill_reason": str|None}
    """
    sector_type = sector_config.get("type", "persistent")
    min_strong = sector_config.get("us_leader_min_strong", 2)

    # US 대장주 강세 수
    leaders = us_leaders_data.get("leaders", {})
    strong_count = sum(1 for v in leaders.values() if v.get("is_strong"))
    weak_count = sum(1 for v in leaders.values() if v.get("is_weak"))

    # US 2차 연동 ETF 확인 (기존 overnight에서)
    sec_etf = sector_config.get("us_secondaries_etf", "")
    us_secondary_confirm = _check_us_secondary(sec_etf, us_overnight)

    reasons = []
    kill_reason = None

    # ── conditional 타입 특별 처리 ──
    if sector_type == "conditional":
        if negative_news_found:
            kill_reason = f"부정 키워드 발견 → {sector_config.get('name', sector_key)} 비활성"
            return _make_result(0, "INACTIVE", 0, reasons, kill_reason)
        if news_score < 3:
            return _make_result(0, "INACTIVE", 0, ["조건부 뉴스 미충족"], None)

    # ── Phase 판정 ──
    phase = 0
    alert_level = 0

    # Phase 1: WATCH — US 1개 강세 or 뉴스만
    if strong_count >= 1 or news_score >= 3:
        phase = 1
        alert_level = 2
        if strong_count >= 1:
            reasons.append(f"US 대장주 {strong_count}/{len(leaders)} 강세")
        if news_score >= 3:
            reasons.append(f"뉴스 점수 {news_score} (키워드: {', '.join(news_keywords_matched[:3])})")

    # Phase 2: CONFIRM — US min_strong개 이상 강세 + US 2차 확산
    if strong_count >= min_strong:
        phase = 2
        alert_level = 3
        reasons.append(f"US 대장주 {strong_count}/{len(leaders)} ≥ {min_strong} 강세 → 본경보")
        if us_secondary_confirm:
            alert_level = max(alert_level, 4)
            reasons.append(f"US 2차 ETF({sec_etf}) 동반 강세")

    # Phase 3: KR_READY — 한국 대장주 전일 강세 확인
    if phase >= 2 and kr_leaders_strong:
        phase = 3
        alert_level = max(alert_level, 4)
        reasons.append("KR 대장주 전일 강세 확인 → 실행 준비")

    # Phase 4: EXECUTE — 뉴스 + US 확인 + KR 확인 (경보 3개 이상)
    # 경보 요소: US 대장주 강세, US 2차 확산, 뉴스, KR 대장주 강세
    alert_factors = sum([
        strong_count >= min_strong,
        us_secondary_confirm,
        news_score >= 3,
        kr_leaders_strong,
    ])
    if alert_factors >= 3:
        phase = 4
        alert_level = 5
        reasons.append(f"경보 {alert_factors}/4 요소 충족 → 실행 가능")

    # ── 이벤트형 보정 ──
    if sector_type == "event":
        # 이벤트형은 뉴스 없으면 최대 Phase 1
        if news_score < 2 and phase > 1:
            phase = min(phase, 1)
            alert_level = min(alert_level, 2)
            reasons.append("이벤트형: 뉴스 부족 → Phase 하향")

    # ── 약세 패널티 ──
    if weak_count >= 2:
        phase = max(0, phase - 1)
        alert_level = max(0, alert_level - 2)
        reasons.append(f"US 대장주 {weak_count}개 약세 → 하향 조정")

    phase_names = {0: "INACTIVE", 1: "WATCH", 2: "CONFIRM", 3: "KR_READY", 4: "EXECUTE"}
    return _make_result(phase, phase_names.get(phase, "INACTIVE"), alert_level, reasons, kill_reason)


def _check_us_secondary(etf_ticker: str, us_overnight: dict) -> bool:
    """기존 US Overnight 데이터에서 2차 ETF 동반 강세 확인."""
    if not us_overnight or not etf_ticker:
        return False

    sector_momentum = us_overnight.get("sector_momentum", {})

    # ETF 티커 → 한국 섹터 매핑 (역방향)
    etf_to_kr_sector = {
        "SOXX": "반도체",
        "XLE": "에너지",
        "XLI": "조선",
        "XLK": "IT",
        "XLF": "금융",
        "XLV": "헬스케어",
    }
    kr_sector = etf_to_kr_sector.get(etf_ticker.upper(), "")
    if kr_sector and kr_sector in sector_momentum:
        mom = sector_momentum[kr_sector]
        ret_1d = mom.get("ret_1d_pct", 0)
        return ret_1d > 0  # 양수 수익률이면 동반 강세

    # index_direction에서 확인
    idx_dir = us_overnight.get("index_direction", {})
    for key, val in idx_dir.items():
        if key.upper() == etf_ticker.upper():
            return val.get("ret_1d", 0) > 0

    return False


def _make_result(phase: int, phase_name: str, alert_level: int,
                 reasons: list, kill_reason: str | None) -> dict:
    return {
        "phase": phase,
        "phase_name": phase_name,
        "alert_level": min(alert_level, 5),
        "reasons": reasons,
        "kill_reason": kill_reason,
    }
