"""국적별 외국인 수급 시그널 엔진

KRX 국적별 거래량 + KIS 외국인 순매수 → 매수/매도 방향 추론

핵심 로직:
  1. 일별 국적별 거래량 추이 (7일)
  2. KIS 외국인 순매수와 교차 분석 → 방향성 추론
  3. 패턴 감지 (아시아 매집, 기관 매집, 헤지펀드, 이탈 등)
  4. 종합 스코어 (-50 ~ +50) → 시그널 등급

국가 그룹:
  - 기관 (Institutional): 영국, 룩셈부르크, 아일랜드, 노르웨이
  - 아시아 (Asia): 중국, 홍콩, 싱가포르, 일본, 대만, 말레이시아
  - 헤지펀드 (Hedge): 케이맨 제도, 영국령 버진아일랜드
  - 중동 (Middle East): 카타르, 사우디아라비아, 쿠웨이트, 아랍에미리트
  - 기타: 미국, 캐나다, 프랑스, 스위스, 독일, 오스트레일리아
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "krx_nationality" / "nationality.db"
CHINA_SIGNAL_PATH = PROJECT_ROOT / "data" / "china_money" / "china_money_signal.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "krx_nationality" / "nationality_signal.json"


# ═══════════════════════════════════════════════════
# 국가 그룹 정의
# ═══════════════════════════════════════════════════

COUNTRY_GROUPS = {
    "institutional": ["영국", "룩셈부르크", "아일랜드", "노르웨이"],
    "asia": ["중국", "홍콩", "싱가포르", "일본", "대만", "말레이시아"],
    "hedge": ["케이맨 제도", "영국령 버진아일랜드"],
    "middle_east": ["카타르", "사우디아라비아", "쿠웨이트", "아랍에미리트"],
    "americas": ["미국", "캐나다"],
    "europe": ["프랑스", "스위스", "독일", "오스트레일리아"],
}

# 역방향 매핑: 국가 → 그룹
COUNTRY_TO_GROUP = {}
for group, countries in COUNTRY_GROUPS.items():
    for c in countries:
        COUNTRY_TO_GROUP[c] = group


# ═══════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════

@dataclass
class NationalitySignal:
    """종목별 국적 수급 시그널."""
    ticker: str
    name: str
    signal: str          # STRONG_BUY / BUY / NEUTRAL / CAUTION / SELL
    score: int           # -50 ~ +50
    reasons: list[str] = field(default_factory=list)

    # 그룹별 거래량 추이 (최근 3일 평균 vs 이전 4일 평균)
    inst_trend: float = 0.0     # 기관 추이 (양수=증가)
    asia_trend: float = 0.0     # 아시아 추이
    hedge_trend: float = 0.0    # 헤지펀드 추이

    # KIS 외국인 순매수 (있으면)
    foreign_net_5d: int = 0
    foreign_direction: str = ""  # BUY / SELL / NEUTRAL

    # 주요 국가 상세
    top_countries: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════
# DB 쿼리
# ═══════════════════════════════════════════════════

def _get_daily_data(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = 7,
) -> list[dict]:
    """최근 N 거래일의 국적별 데이터.

    Returns:
        [{"date": "20260310", "countries": {"영국": 7533072, "중국": 138071, ...}}, ...]
    """
    rows = conn.execute(
        """
        SELECT DISTINCT date FROM nationality_daily
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, days),
    ).fetchall()

    if not rows:
        return []

    dates = [r[0] for r in rows]
    result = []
    for d in sorted(dates):  # 오래된 → 최신 순
        country_rows = conn.execute(
            "SELECT country, trade_vol FROM nationality_daily "
            "WHERE ticker = ? AND date = ?",
            (ticker, d),
        ).fetchall()
        countries = {cn: vol for cn, vol in country_rows}
        result.append({"date": d, "countries": countries})

    return result


def _get_all_tickers(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """DB에 있는 모든 종목."""
    rows = conn.execute(
        "SELECT DISTINCT ticker, name FROM nationality_daily ORDER BY ticker"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# ═══════════════════════════════════════════════════
# 그룹별 거래량 집계
# ═══════════════════════════════════════════════════

def _group_volumes(countries: dict[str, int]) -> dict[str, int]:
    """국가별 거래량 → 그룹별 합산."""
    groups = {g: 0 for g in COUNTRY_GROUPS}
    for cn, vol in countries.items():
        g = COUNTRY_TO_GROUP.get(cn, "other")
        if g in groups:
            groups[g] += vol
    return groups


def _calc_trend(daily_data: list[dict], group_name: str) -> float:
    """최근 3일 평균 vs 이전 데이터 평균 → 변화율.

    Returns:
        변화율 (예: 0.5 = 50% 증가, -0.3 = 30% 감소)
    """
    if len(daily_data) < 4:
        return 0.0

    group_countries = COUNTRY_GROUPS.get(group_name, [])

    recent_vols = []
    older_vols = []

    for i, day in enumerate(daily_data):
        vol = sum(day["countries"].get(cn, 0) for cn in group_countries)
        if i >= len(daily_data) - 3:  # 최근 3일
            recent_vols.append(vol)
        else:
            older_vols.append(vol)

    avg_recent = sum(recent_vols) / len(recent_vols) if recent_vols else 0
    avg_older = sum(older_vols) / len(older_vols) if older_vols else 0

    if avg_older == 0:
        return 1.0 if avg_recent > 0 else 0.0

    return (avg_recent - avg_older) / avg_older


def _detect_consecutive(daily_data: list[dict], group_name: str) -> int:
    """그룹 거래량 연속 증가/감소 일수. 양수=연속증가, 음수=연속감소."""
    if len(daily_data) < 2:
        return 0

    group_countries = COUNTRY_GROUPS.get(group_name, [])
    vols = []
    for day in daily_data:
        vol = sum(day["countries"].get(cn, 0) for cn in group_countries)
        vols.append(vol)

    # 뒤에서부터 연속 증가/감소 카운트
    consecutive = 0
    for i in range(len(vols) - 1, 0, -1):
        if vols[i] > vols[i - 1]:
            if consecutive >= 0:
                consecutive += 1
            else:
                break
        elif vols[i] < vols[i - 1]:
            if consecutive <= 0:
                consecutive -= 1
            else:
                break
        else:
            break

    return consecutive


# ═══════════════════════════════════════════════════
# KIS 외국인 순매수 데이터 로드
# ═══════════════════════════════════════════════════

def _load_kis_foreign() -> dict[str, dict]:
    """china_money_signal.json에서 KIS 외국인 순매수 데이터 로드.

    Returns:
        {"207940": {"foreign_net_5d": 43392, "signal": "INFLOW", ...}, ...}
    """
    result = {}
    try:
        if CHINA_SIGNAL_PATH.exists():
            data = json.loads(CHINA_SIGNAL_PATH.read_text(encoding="utf-8"))
            for sig in data.get("signals", []):
                ticker = sig.get("ticker", "")
                if ticker:
                    result[ticker] = sig
    except Exception as e:
        logger.warning(f"KIS 외국인 데이터 로드 실패: {e}")
    return result


# ═══════════════════════════════════════════════════
# 시그널 판정
# ═══════════════════════════════════════════════════

def analyze_stock(
    conn: sqlite3.Connection,
    ticker: str,
    name: str,
    kis_data: dict | None = None,
) -> NationalitySignal | None:
    """단일 종목의 국적별 수급 시그널 생성."""
    daily = _get_daily_data(conn, ticker, days=7)
    if len(daily) < 2:
        return None

    # 그룹별 추이
    inst_trend = _calc_trend(daily, "institutional")
    asia_trend = _calc_trend(daily, "asia")
    hedge_trend = _calc_trend(daily, "hedge")

    # 연속 증가/감소
    asia_consec = _detect_consecutive(daily, "asia")
    inst_consec = _detect_consecutive(daily, "institutional")
    hedge_consec = _detect_consecutive(daily, "hedge")

    # 최신 날짜의 그룹별 거래량
    latest = daily[-1]["countries"]
    groups = _group_volumes(latest)

    # KIS 외국인 방향
    foreign_net_5d = 0
    foreign_dir = "NEUTRAL"
    if kis_data:
        foreign_net_5d = kis_data.get("foreign_net_5d", 0)
        if foreign_net_5d > 10000:
            foreign_dir = "BUY"
        elif foreign_net_5d < -10000:
            foreign_dir = "SELL"

    # ─── 스코어링 ───
    score = 0
    reasons = []

    # 1) 기관(영국) 추이 (최대 ±15점)
    if inst_trend > 0.3:
        score += 15
        reasons.append(f"영국 기관 급증 (+{inst_trend:.0%})")
    elif inst_trend > 0.1:
        score += 8
        reasons.append(f"영국 기관 증가 (+{inst_trend:.0%})")
    elif inst_trend < -0.3:
        score -= 12
        reasons.append(f"영국 기관 급감 ({inst_trend:.0%})")
    elif inst_trend < -0.1:
        score -= 6
        reasons.append(f"영국 기관 감소 ({inst_trend:.0%})")

    # 2) 아시아 추이 (최대 ±15점)
    if asia_trend > 0.3:
        score += 15
        reasons.append(f"아시아 매집 (+{asia_trend:.0%})")
    elif asia_trend > 0.1:
        score += 8
        reasons.append(f"아시아 증가 (+{asia_trend:.0%})")
    elif asia_trend < -0.3:
        score -= 15
        reasons.append(f"아시아 이탈 ({asia_trend:.0%})")
    elif asia_trend < -0.1:
        score -= 8
        reasons.append(f"아시아 감소 ({asia_trend:.0%})")

    # 3) 아시아 연속 매집 (최대 +10점)
    if asia_consec >= 3:
        score += 10
        reasons.append(f"아시아 {asia_consec}일 연속 증가")
    elif asia_consec <= -3:
        score -= 10
        reasons.append(f"아시아 {abs(asia_consec)}일 연속 감소")

    # 4) 헤지펀드 추이 (경고 신호)
    if hedge_trend > 0.5 and hedge_consec >= 4:
        score -= 5  # 차익실현 임박 경고
        reasons.append(f"헤지펀드 {hedge_consec}일+ 활성 (차익실현 임박)")
    elif hedge_trend > 0.2:
        score += 3
        reasons.append("헤지펀드 유입")

    # 5) KIS 외국인 순매수 방향 (최대 ±10점)
    if foreign_dir == "BUY":
        score += 10
        reasons.append(f"외국인 순매수 {foreign_net_5d:,}주 (5일)")
    elif foreign_dir == "SELL":
        score -= 10
        reasons.append(f"외국인 순매도 {foreign_net_5d:,}주 (5일)")

    # 6) 복합 패턴 보너스 (기관+아시아 동시 증가 + 외국인 순매수)
    if inst_trend > 0.1 and asia_trend > 0.1 and foreign_dir == "BUY":
        score += 5
        reasons.append("기관+아시아 복합 매수")

    # ─── 등급 판정 ───
    if score >= 30:
        signal = "STRONG_BUY"
    elif score >= 15:
        signal = "BUY"
    elif score > -10:
        signal = "NEUTRAL"
    elif score > -25:
        signal = "CAUTION"
    else:
        signal = "SELL"

    # TOP 국가 (최신일)
    sorted_countries = sorted(latest.items(), key=lambda x: x[1], reverse=True)
    top_countries = [
        {"country": cn, "volume": vol, "group": COUNTRY_TO_GROUP.get(cn, "other")}
        for cn, vol in sorted_countries[:8]
    ]

    return NationalitySignal(
        ticker=ticker,
        name=name,
        signal=signal,
        score=score,
        reasons=reasons,
        inst_trend=round(inst_trend, 3),
        asia_trend=round(asia_trend, 3),
        hedge_trend=round(hedge_trend, 3),
        foreign_net_5d=foreign_net_5d,
        foreign_direction=foreign_dir,
        top_countries=top_countries,
    )


# ═══════════════════════════════════════════════════
# 전체 분석 실행
# ═══════════════════════════════════════════════════

def run_analysis(db_path: Path = DB_PATH) -> list[NationalitySignal]:
    """전 종목 국적별 수급 시그널 생성."""
    conn = sqlite3.connect(str(db_path))
    tickers = _get_all_tickers(conn)
    kis_foreign = _load_kis_foreign()

    signals = []
    for ticker, name in tickers:
        kis = kis_foreign.get(ticker)
        sig = analyze_stock(conn, ticker, name, kis)
        if sig:
            signals.append(sig)

    conn.close()

    # 스코어 순 정렬
    signals.sort(key=lambda s: s.score, reverse=True)

    # JSON 저장
    output = {
        "analyzed_at": datetime.now().isoformat(),
        "total_stocks": len(signals),
        "signals": [s.to_dict() for s in signals],
        "summary": {
            "strong_buy": sum(1 for s in signals if s.signal == "STRONG_BUY"),
            "buy": sum(1 for s in signals if s.signal == "BUY"),
            "neutral": sum(1 for s in signals if s.signal == "NEUTRAL"),
            "caution": sum(1 for s in signals if s.signal == "CAUTION"),
            "sell": sum(1 for s in signals if s.signal == "SELL"),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"분석 완료: {len(signals)}종목 | "
        f"강매수={output['summary']['strong_buy']} "
        f"매수={output['summary']['buy']} "
        f"중립={output['summary']['neutral']} "
        f"주의={output['summary']['caution']} "
        f"매도={output['summary']['sell']}"
    )

    return signals


def format_telegram(signals: list[NationalitySignal]) -> str:
    """텔레그램 메시지 포맷."""
    EMOJI = {
        "STRONG_BUY": "\U0001F534",   # 🔴
        "BUY": "\U0001F7E0",          # 🟠
        "NEUTRAL": "\u26AA",          # ⚪
        "CAUTION": "\U0001F7E1",      # 🟡
        "SELL": "\U0001F535",          # 🔵
    }
    LABEL = {
        "STRONG_BUY": "강력매수",
        "BUY": "매수",
        "NEUTRAL": "중립",
        "CAUTION": "주의",
        "SELL": "매도",
    }

    lines = ["<b>🌍 국적별 외국인 수급 시그널</b>", ""]

    # 비중립만 표시
    active = [s for s in signals if s.signal != "NEUTRAL"]
    if not active:
        lines.append("모든 종목 중립 — 특이 시그널 없음")
        return "\n".join(lines)

    for sig in active:
        emoji = EMOJI.get(sig.signal, "⚪")
        label = LABEL.get(sig.signal, sig.signal)
        score_str = f"+{sig.score}" if sig.score > 0 else str(sig.score)

        lines.append(f"{emoji} <b>{label}</b>  {sig.name}  {score_str}점")
        for reason in sig.reasons[:3]:
            lines.append(f"    · {reason}")
        lines.append("")

    # T-1→T+1 전략 요약
    lines.append("<b>📊 핵심 전략</b>")
    strong_buys = [s for s in signals if s.signal == "STRONG_BUY"]
    cautions = [s for s in signals if s.signal in ("CAUTION", "SELL")]

    if strong_buys:
        names = ", ".join(s.name for s in strong_buys[:3])
        lines.append(f"  기관+아시아 복합매수 → {names}")
    if cautions:
        names = ", ".join(s.name for s in cautions[:3])
        lines.append(f"  아시아 이탈 주의 → {names}")

    return "\n".join(lines)
