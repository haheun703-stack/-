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
    r"(?<![A-Za-z])(BUY_LONG|BUY_SHORT|SELL_SHORT|STRONG_BUY|STRONG_SELL"
    r"|BUY|SELL|HOLD|TRIM)(?![A-Za-z])", re.IGNORECASE)

#: 한국어 지시 구문 — 띄어쓰기 유무를 모두 흡수한다.
#: ★9/7 오후 신설: 첫 판은 띄어쓴 형태만 담아 `적극매수`처럼 붙여 쓰면 통과했고,
#:   실제 브리핑 결론 문구 3종 중 2종(`우량주 저가매수 기회`·`시가 확인 후 대응`)이
#:   목록에 아예 없었다. 백필 대상 31행 중 17행이 미탐이었다(검수 1팀 + DB 실측).
_FORBIDDEN_KO = re.compile(
    r"(적극\s*매수|강력\s*매수|저가\s*매수|추격\s*매수|분할\s*매도|전량\s*매도"
    r"|강력\s*추천|진입\s*결정|진입\s*보류|신규\s*진입|비중\s*확대|비중\s*축소"
    r"|매수\s*적기|매도\s*적기|손절|익절|목표가|목표\s*주가"
    r"|확인\s*후\s*대응|갭업\s*예상|갭다운\s*예상)")

#: 6자리 종목코드 — 텍스트 컬럼으로 종목이 나가는 것을 막는다(검수 1팀 F-12).
#: `related_tickers`만 비우고 본문을 열어 두면 같은 정보가 그대로 나간다.
_TICKER_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")

_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "target_price", "stop_loss", "entry_price", "size_won", "action=",
    "청산", "관망", "추천",
)


def _scrub_value(value):
    """허용키 **안의 값**에서 종목코드·판단 어휘를 걸러낸다. (정리값, 사유) 반환.

    ★9/7 오후 신설(검수 1팀 F-1·F-2) — 첫 판은 **키 이름만** 보고 값은 통과시켰다.
    그런데 `eye_event_counts`는 `{종목코드: 횟수}` dict이고 `kospi_range`는 정규식으로
    뽑은 자유 문자열이다. 즉 허용키를 통해 종목코드와 문구가 그대로 나갈 수 있었다 —
    이 파일 docstring이 "종목 코드는 **어떤 컬럼에도** 싣지 않는다"고 선언한 바로 그것이다.
    (9/7 실측으로 배포 후 20행은 마침 빈 dict였으나, EYE 이벤트가 있는 날이면 들어간다.)
    """
    reasons: list[str] = []
    if isinstance(value, dict):
        # 키가 종목코드인 맵은 **집계치로 축약**한다 — 종목별 랭킹을 내보내지 않는다.
        code_keys = [k for k in value if _TICKER_RE.fullmatch(str(k))]
        if code_keys:
            try:
                total = sum(int(v) for v in value.values())
            except (TypeError, ValueError):
                total = len(value)
            return ({"tickers": len(value), "events": total},
                    [f"종목코드 키 {len(code_keys)}개 → 집계로 축약"])
        out = {}
        for k, v in value.items():
            nv, r = _scrub_value(v)
            out[k] = nv
            reasons += r
        return out, reasons
    if isinstance(value, list):
        # 리스트는 종목 목록일 가능성이 높다 — 길이만 남긴다.
        if value:
            return len(value), [f"리스트 {len(value)}건 → 길이로 축약"]
        return value, reasons
    if isinstance(value, str):
        hits = check_text(value)
        if hits:
            return None, ["값 어휘 " + ",".join(hits[:3])]
    return value, reasons


def scrub_reasoning(reasoning: dict) -> tuple[dict, list[str]]:
    """허용 목록 밖의 키를 제거하고 **남은 값도 검사**한다. (정리본, 사유) 반환."""
    if not isinstance(reasoning, dict):
        return {}, ["<non-dict reasoning>"]
    removed = [k for k in reasoning if k not in ALLOWED_REASONING_KEYS]
    clean: dict = {}
    for k, v in reasoning.items():
        if k not in ALLOWED_REASONING_KEYS:
            continue
        nv, reasons = _scrub_value(v)
        clean[k] = nv
        removed += [f"{k}:{r}" for r in reasons]
    return clean, sorted(set(removed))


def check_text(*texts: str | None) -> list[str]:
    """title·body 등 자유 텍스트에서 금지 어휘를 찾는다. 없으면 빈 목록."""
    hits: list[str] = []
    for t in texts:
        if not t:
            continue
        for m in _FORBIDDEN_WORDS.finditer(t):
            hits.append(m.group(0))
        for m in _FORBIDDEN_KO.finditer(t):
            hits.append(m.group(0))
        for m in _TICKER_RE.finditer(t):
            hits.append(f"종목코드:{m.group(0)}")
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
        # ★9/7: 문자열이면 len()이 글자 수가 되어 오보가 난다(검수 1팀 F-14).
        n = len(rt) if isinstance(rt, (list, tuple, set)) else 1
        problems.append(f"related_tickers {n}건")
    hits = check_text(row.get("title"), row.get("body"))
    if hits:
        problems.append("어휘 " + ",".join(hits[:5]))
    return problems
