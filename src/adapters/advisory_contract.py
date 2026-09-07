"""quant_bot_advisory 데이터계약 가드 (B-74 · B-55⑴, 2026-09-07).

배경
  7/27 데이터계약 전환으로 매매판단 테이블 16종의 적재를 끊었는데, 봇간 통신 테이블
  `quant_bot_advisory`가 psycopg2 직결이라 `flowx_uploader`의 guard를 거치지 않았고
  `reasoning` 한 컬럼으로 차단 테이블의 내용물(ETF 추천 action=BUY_LONG·픽 종목·
  등급)이 7/27 이후 하루도 안 끊기고 나갔다(9/7 실측 1,351행).

원칙
  - 이 테이블에 남기는 것은 **시장 수준의 계산 산출물**뿐이다: 레짐 라벨, 체결강도
    평균/중앙값, 인버스 ETF 강도, 이벤트 횟수, 표본 집계치.
  - 종목 코드·종목별 판단·action/grade/가격 지시·매수/매도/청산/목표가 어휘는
    어떤 컬럼에도 싣지 않는다(`related_tickers`는 빈 배열).
  - 생산자는 둘(`snapshot_session.py`·`run_morning_briefing.py`)이고 둘 다 여기의
    `scrub_reasoning`·`check_text`를 거친다. 자가검사(`verify_contract_suspension.py`
    §2.6)도 같은 허용 목록으로 판정한다 — 허용 목록은 이 파일 한 곳에만 둔다.
  - 소비자(단타봇)는 컬럼 `market_regime`·`market_strength_avg`·`inverse_etf_strength`
    만 기능적으로 읽는다(9/7 단타봇 코드 실측: auto_trader·brain_state_builder·
    quant_advisory_subscriber). `reasoning`·`related_tickers`·`body`는 표시용뿐이라
    비워도 게이트가 깨지지 않는다.
"""
from __future__ import annotations

import re

#: reasoning(JSONB)에 남길 수 있는 키 전부. 여기 없는 키는 INSERT 전에 제거된다.
ALLOWED_REASONING_KEYS: frozenset[str] = frozenset({
    # 장중 스냅샷 (snapshot_session)
    "market_strength_mean", "market_strength_median",
    "inverse_etf_strength", "inverse_etf_buy_ratio",
    "eye_event_counts", "intraday_signals_count",
    "sample_avg_chg_pct", "sample_positive_count", "sample_total",
    # 장전 브리핑 메타 (run_morning_briefing.parse_briefing_meta)
    "kospi_up_pct", "kospi_range", "vix", "ewy_chg",
    "us_spy_chg", "us_qqq_chg", "us_soxx_chg", "us_dia_chg",
})

#: 어떤 텍스트 컬럼(title·body)에도 있어선 안 되는 어휘. 대소문자 무시.
#: "매수비율"(체결 매수비율 = 데이터 명칭)처럼 정당한 합성어를 오탐하지 않도록
#: 단어 경계가 있는 영문 토큰과 명백한 한국어 지시 구문만 건다.
_FORBIDDEN_WORDS = re.compile(
    r"\b(BUY_LONG|BUY_SHORT|SELL_SHORT|BUY|SELL|HOLD|TRIM)\b", re.IGNORECASE)
_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "target_price", "stop_loss", "entry_price", "size_won", "action=",
    "적극 매수", "강력 추천", "진입 결정", "진입 보류", "분할매도", "분할 매도",
    "청산", "목표가", "관망", "추천", "매수적기", "비중확대", "신규진입",
)


def scrub_reasoning(reasoning: dict) -> tuple[dict, list[str]]:
    """허용 목록 밖의 키를 전부 제거한다. (정리본, 제거된 키 목록) 반환."""
    if not isinstance(reasoning, dict):
        return {}, ["<non-dict reasoning>"]
    removed = sorted(k for k in reasoning if k not in ALLOWED_REASONING_KEYS)
    clean = {k: v for k, v in reasoning.items() if k in ALLOWED_REASONING_KEYS}
    return clean, removed


def check_text(*texts: str | None) -> list[str]:
    """title·body 등 자유 텍스트에서 금지 어휘를 찾는다. 없으면 빈 목록."""
    hits: list[str] = []
    for t in texts:
        if not t:
            continue
        for m in _FORBIDDEN_WORDS.finditer(t):
            hits.append(m.group(0))
        low = t.lower()
        for p in _FORBIDDEN_PHRASES:
            if p.lower() in low:
                hits.append(p)
    return sorted(set(hits))


def audit_row(row: dict) -> list[str]:
    """자가검사용 — 적재된 한 행이 계약을 지키는지. 위반 사유 목록(빈 목록=정상)."""
    problems: list[str] = []
    reasoning = row.get("reasoning")
    if isinstance(reasoning, dict):
        _, removed = scrub_reasoning(reasoning)
        if removed:
            problems.append("reasoning 금지키 " + ",".join(removed[:6])
                            + ("…" if len(removed) > 6 else ""))
    elif reasoning not in (None, {}):
        problems.append("reasoning 비정형")
    rt = row.get("related_tickers")
    if rt:
        problems.append(f"related_tickers {len(rt)}건")
    hits = check_text(row.get("title"), row.get("body"))
    if hits:
        problems.append("어휘 " + ",".join(hits[:5]))
    return problems
