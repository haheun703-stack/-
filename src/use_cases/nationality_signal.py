"""국적별 외국인 수급 시그널 엔진 v2 — 7 Secrets Edition

KRX 국적별 거래량 + KIS 외국인 순매수 + 주가 데이터 → 고정밀 매수/매도 시그널

7 Secrets:
  #1. 거래량 가속도 (2차 미분) — 매집 속도 변화 감지
  #2. 집중도 지수 (HHI) — 확신 매수 vs 패시브 분산
  #3. 조용한 매집 (Quiet Accumulation) — 최고 수익률 패턴
  #4. 교차 종목 흐름 (Cross-Stock Flow) — 섹터 로테이션 감지
  #5. 이상치 탐지 (Z-Score) — 평소와 다른 극단적 움직임
  #6. 선행-후행 관계 (Lead-Lag) — 영국 선행 → 아시아 후행 선점
  #7. 거래량-가격 괴리 (Volume-Price Divergence) — 흡수 매집 감지

스코어 범위: -23 ~ +48
등급: STRONG_BUY (≥35) / BUY (≥20) / NEUTRAL (≥5) / CAUTION (≥-10) / SELL (<-10)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "krx_nationality" / "nationality.db"
CHINA_SIGNAL_PATH = PROJECT_ROOT / "data" / "china_money" / "china_money_signal.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "krx_nationality" / "nationality_signal.json"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"


# ═══════════════════════════════════════════════════
# 국가 그룹 정의
# ═══════════════════════════════════════════════════

COUNTRY_GROUPS = {
    "institutional": [
        "영국", "룩셈부르크", "아일랜드", "노르웨이",
        "스위스", "네덜란드", "프랑스", "독일", "스웨덴",
    ],
    "asia": ["중국", "홍콩", "싱가포르", "일본", "대만", "말레이시아"],
    "hedge": ["케이맨 제도", "케이맨제도", "케이맨",
              "버뮤다", "영국령 버진아일랜드", "영국령 버진아일랜", "영국령버진아일"],
    "middle_east": ["카타르", "사우디아라비아", "사우디", "쿠웨이트", "아랍에미리트"],
    "americas": ["미국", "캐나다"],
    "europe_other": ["오스트레일리아", "호주", "벨기에", "이탈리아", "핀란드"],
}

# 역방향 매핑: 국가 → 그룹
COUNTRY_TO_GROUP: dict[str, str] = {}
for _group, _countries in COUNTRY_GROUPS.items():
    for _c in _countries:
        COUNTRY_TO_GROUP[_c] = _group

# 섹터 분류 (교차 종목 흐름용)
TICKER_SECTOR = {
    "005930": "반도체", "000660": "반도체", "042700": "반도체", "000990": "반도체",
    "012450": "방산", "064350": "방산", "047810": "방산", "272210": "방산",
    "009540": "조선", "329180": "조선",
    "000270": "자동차", "005380": "자동차",
    "373220": "배터리", "006400": "배터리",
    "071050": "금융", "105560": "금융", "055550": "금융",
    "207940": "바이오", "068270": "바이오",
    "035420": "IT", "035720": "IT",
    "028260": "지주", "402340": "지주", "180640": "지주",
    "307950": "전기장비",
    "103140": "소재", "010130": "소재",
    "090430": "소비재", "051900": "소비재",
    "096770": "에너지",
    "352820": "엔터",
}


# ═══════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════

@dataclass
class NationalitySignal:
    """종목별 국적 수급 시그널 (v2)."""
    ticker: str
    name: str
    signal: str          # STRONG_BUY / BUY / NEUTRAL / CAUTION / SELL
    score: int           # -23 ~ +48

    # Secret 상세
    reasons: list[str] = field(default_factory=list)
    pattern: str = ""    # QUIET_ACCUM / INST_ACCEL / ASIA_ENTRY / HEDGE_OVERHEAT / FULL_EXIT / etc

    # 그룹별 추이
    inst_trend: float = 0.0
    asia_trend: float = 0.0
    hedge_trend: float = 0.0

    # Secret #1: 가속도
    inst_acceleration: float = 0.0   # 기관 거래량 가속도
    asia_acceleration: float = 0.0

    # Secret #2: 집중도
    hhi: float = 0.0                 # 허핀달 지수 (0~1)
    dominant_country: str = ""       # 집중 거래 국가
    dominant_group: str = ""         # 해당 국가의 그룹

    # Secret #5: Z-Score
    inst_zscore: float = 0.0
    asia_zscore: float = 0.0
    hedge_zscore: float = 0.0

    # Secret #7: 가격 괴리
    price_change_pct: float = 0.0    # 최근 3일 가격 변화율
    vol_price_pattern: str = ""      # ABSORB / NORMAL / TRAP / ABANDON

    # KIS 외국인
    foreign_net_5d: int = 0
    foreign_direction: str = ""

    # 주요 국가
    top_countries: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════
# DB 쿼리
# ═══════════════════════════════════════════════════

def _get_daily_data(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = 10,
) -> list[dict]:
    """최근 N 거래일의 국적별 데이터.

    Returns:
        [{"date": "20260310", "countries": {"영국": 7533072, ...}}, ...]
        날짜 오름차순 (오래된 → 최신)
    """
    rows = conn.execute(
        "SELECT DISTINCT date FROM nationality_daily "
        "WHERE ticker = ? ORDER BY date DESC LIMIT ?",
        (ticker, days),
    ).fetchall()

    if not rows:
        return []

    dates = [r[0] for r in rows]
    result = []
    for d in sorted(dates):
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
# 그룹별 집계 유틸리티
# ═══════════════════════════════════════════════════

def _group_volumes(countries: dict[str, int]) -> dict[str, int]:
    """국가별 거래량 → 그룹별 합산."""
    groups: dict[str, int] = {g: 0 for g in COUNTRY_GROUPS}
    for cn, vol in countries.items():
        g = COUNTRY_TO_GROUP.get(cn, "other")
        if g in groups:
            groups[g] += vol
    return groups


def _group_daily_series(
    daily_data: list[dict], group_name: str
) -> list[int]:
    """일별 데이터에서 특정 그룹의 거래량 시계열 추출."""
    group_countries = COUNTRY_GROUPS.get(group_name, [])
    return [
        sum(day["countries"].get(cn, 0) for cn in group_countries)
        for day in daily_data
    ]


# ═══════════════════════════════════════════════════
# Secret #1: 거래량 가속도 (2차 미분)
# ═══════════════════════════════════════════════════

def _calc_acceleration(vols: list[int]) -> float:
    """거래량 시계열의 가속도 (2차 미분의 최근 평균).

    Returns:
        가속도 값 (양수=가속 매집, 음수=감속/이탈)
        거래량 단위로 정규화: (avg_vol > 0일 때) acceleration / avg_vol
    """
    if len(vols) < 3:
        return 0.0

    # 1차 미분 (변화량)
    deltas = [vols[i] - vols[i - 1] for i in range(1, len(vols))]

    # 2차 미분 (가속도)
    accels = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]

    if not accels:
        return 0.0

    # 최근 2일 가속도 평균
    recent_accel = sum(accels[-2:]) / len(accels[-2:])

    # 평균 거래량으로 정규화
    avg_vol = sum(vols) / len(vols)
    if avg_vol == 0:
        return 0.0

    return recent_accel / avg_vol


def _calc_trend(vols: list[int]) -> float:
    """최근 3일 평균 vs 이전 평균 → 변화율."""
    if len(vols) < 4:
        return 0.0

    recent = vols[-3:]
    older = vols[:-3]

    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older) if older else 0

    if avg_older == 0:
        return 1.0 if avg_recent > 0 else 0.0

    return (avg_recent - avg_older) / avg_older


# ═══════════════════════════════════════════════════
# Secret #2: 집중도 지수 (HHI)
# ═══════════════════════════════════════════════════

def _calc_hhi(
    daily_data: list[dict],
) -> tuple[float, str, str, bool]:
    """허핀달-허쉬만 지수 계산 (v2: 과거 대비 집중도 변화 감지).

    Returns:
        (hhi, dominant_country, dominant_group, is_unusual)
        hhi: 0~1 (1에 가까울수록 집중)
        is_unusual: 과거 평균 HHI 대비 유의미하게 높은 경우 True
    """
    if not daily_data:
        return 0.0, "", "", False

    latest = daily_data[-1]["countries"]
    total = sum(latest.values())
    if total == 0:
        return 0.0, "", "", False

    # 현재 HHI
    hhi = 0.0
    max_share = 0.0
    dominant_country = ""
    for cn, vol in latest.items():
        share = vol / total
        hhi += share ** 2
        if share > max_share:
            max_share = share
            dominant_country = cn

    dominant_group = COUNTRY_TO_GROUP.get(dominant_country, "other")

    # 과거 HHI 평균과 비교 → 평소보다 집중된 경우만 시그널
    is_unusual = False
    if len(daily_data) >= 4:
        past_hhis = []
        for day in daily_data[:-1]:
            t = sum(day["countries"].values())
            if t == 0:
                continue
            h = sum((v / t) ** 2 for v in day["countries"].values())
            past_hhis.append(h)
        if past_hhis:
            avg_past_hhi = sum(past_hhis) / len(past_hhis)
            # 현재 HHI가 과거 평균보다 20%+ 높으면 비정상 집중
            if hhi > avg_past_hhi * 1.20 and hhi >= 0.30:
                is_unusual = True

    return round(hhi, 4), dominant_country, dominant_group, is_unusual


# ═══════════════════════════════════════════════════
# Secret #3: 조용한 매집 (Quiet Accumulation) 탐지
# ═══════════════════════════════════════════════════

def _detect_quiet_accumulation(
    inst_vols: list[int],
    hedge_vols: list[int],
    price_change_pct: float,
    foreign_dir: str,
) -> bool:
    """조용한 매집 4가지 조건 동시 충족 여부.

    조건:
      ① 기관 거래량 3일+ 소폭 증가 (5~30% 범위 내)
      ② 헤지펀드 거래량 변화 없거나 감소
      ③ 주가 횡보 또는 소폭 하락 (-3% ~ +2%)
      ④ KIS 순매수 양수 (또는 중립)
    """
    if len(inst_vols) < 4 or len(hedge_vols) < 4:
        return False

    # ① 기관 3일 연속 소폭 증가
    inst_increasing = 0
    for i in range(-3, 0):
        if len(inst_vols) + i < 1:
            continue
        prev = inst_vols[i - 1] if abs(i - 1) <= len(inst_vols) else 0
        curr = inst_vols[i]
        if prev > 0 and 0.03 <= (curr - prev) / prev <= 0.40:
            inst_increasing += 1
    if inst_increasing < 2:
        return False

    # ② 헤지 변화 없거나 감소
    hedge_trend = _calc_trend(hedge_vols)
    if hedge_trend > 0.10:
        return False

    # ③ 주가 횡보/소폭 하락
    if not (-5.0 <= price_change_pct <= 3.0):
        return False

    # ④ 순매수 방향 (SELL이 아니면 OK)
    if foreign_dir == "SELL":
        return False

    return True


# ═══════════════════════════════════════════════════
# Secret #5: 이상치 탐지 (Z-Score)
# ═══════════════════════════════════════════════════

def _calc_zscore(vols: list[int]) -> float:
    """최근 거래량의 Z-Score (마지막 값 기준).

    시계열이 짧으면 (< 5일) 0 반환.
    """
    if len(vols) < 5:
        return 0.0

    # 마지막 값 제외한 기간으로 평균/표준편차
    baseline = vols[:-1]
    mean = sum(baseline) / len(baseline)
    variance = sum((v - mean) ** 2 for v in baseline) / len(baseline)
    std = math.sqrt(variance) if variance > 0 else 0

    if std == 0:
        return 0.0

    return (vols[-1] - mean) / std


# ═══════════════════════════════════════════════════
# Secret #6: 선행-후행 관계 (Lead-Lag)
# ═══════════════════════════════════════════════════

def _detect_lead_lag(
    inst_vols: list[int],
    asia_vols: list[int],
) -> str:
    """영국 기관 선행 → 아시아 후행 패턴 탐지.

    Returns:
        "LEAD" — 영국 이미 매집 시작, 아시아 아직 → 선점 기회
        "FOLLOW" — 아시아도 따라온 상태
        "NONE" — 패턴 없음
    """
    if len(inst_vols) < 5 or len(asia_vols) < 5:
        return "NONE"

    # 영국 기관: 최근 2~3일 전부터 증가
    inst_early_trend = _calc_trend(inst_vols[-5:])  # 5일 기준

    # 아시아: 최근 1~2일은 아직 변화 없거나 미미
    asia_recent = asia_vols[-2:]
    asia_older = asia_vols[-5:-2]
    if not asia_older:
        return "NONE"
    avg_asia_recent = sum(asia_recent) / len(asia_recent)
    avg_asia_older = sum(asia_older) / len(asia_older)

    asia_change = 0.0
    if avg_asia_older > 0:
        asia_change = (avg_asia_recent - avg_asia_older) / avg_asia_older

    # 영국 증가 중 + 아시아 아직 미미 → LEAD (선점 기회)
    if inst_early_trend > 0.15 and asia_change < 0.10:
        return "LEAD"

    # 영국 증가 + 아시아도 증가 → FOLLOW (이미 시작됨)
    if inst_early_trend > 0.10 and asia_change > 0.10:
        return "FOLLOW"

    return "NONE"


# ═══════════════════════════════════════════════════
# Secret #7: 거래량-가격 괴리 (주가 데이터 로드)
# ═══════════════════════════════════════════════════

def _load_price_data(ticker: str, days: int = 10) -> list[float]:
    """stock_data_daily CSV에서 최근 N일 종가 로드.

    Returns:
        [price_old, ..., price_recent] (오름차순)
    """
    # CSV 파일 찾기 (종목명_ticker.csv)
    pattern = f"*_{ticker}.csv"
    matches = list(CSV_DIR.glob(pattern))
    if not matches:
        return []

    try:
        df = pd.read_csv(matches[0], encoding="utf-8-sig")
        if "Close" not in df.columns:
            return []
        closes = df["Close"].dropna().tail(days).tolist()
        return closes
    except Exception:
        return []


def _calc_price_change(prices: list[float], days: int = 3) -> float:
    """최근 N일 가격 변화율(%)."""
    if len(prices) < days + 1:
        return 0.0
    old = prices[-(days + 1)]
    new = prices[-1]
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100


def _classify_vol_price(
    inst_trend: float,
    price_change_pct: float,
) -> str:
    """거래량-가격 관계 분류.

    Returns:
        ABSORB: 기관↑ + 주가↓ — 매도 흡수 중 (★최고★)
        NORMAL: 기관↑ + 주가↑ — 일반적 매수
        TRAP:   기관↓ + 주가↑ — 개인 주도 함정
        ABANDON: 기관↓ + 주가↓ — 관심 상실
        FLAT: 변화 미미
    """
    inst_up = inst_trend > 0.08
    inst_down = inst_trend < -0.08
    price_up = price_change_pct > 1.0
    price_down = price_change_pct < -1.0

    if inst_up and price_down:
        return "ABSORB"
    elif inst_up and price_up:
        return "NORMAL"
    elif inst_down and price_up:
        return "TRAP"
    elif inst_down and price_down:
        return "ABANDON"
    else:
        return "FLAT"


# ═══════════════════════════════════════════════════
# KIS 외국인 순매수 데이터
# ═══════════════════════════════════════════════════

def _load_kis_foreign() -> dict[str, dict]:
    """china_money_signal.json에서 KIS 외국인 순매수."""
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
# 연속 증가/감소 탐지
# ═══════════════════════════════════════════════════

def _detect_consecutive(vols: list[int]) -> int:
    """거래량 연속 증가/감소 일수. 양수=연속증가, 음수=연속감소."""
    if len(vols) < 2:
        return 0

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
# 핵심: 종목별 시그널 판정 (7 Secrets 통합)
# ═══════════════════════════════════════════════════

def analyze_stock(
    conn: sqlite3.Connection,
    ticker: str,
    name: str,
    kis_data: dict | None = None,
) -> NationalitySignal | None:
    """단일 종목 7 Secrets 통합 분석."""
    daily = _get_daily_data(conn, ticker, days=10)
    if len(daily) < 3:
        return None

    # ─── 그룹별 시계열 추출 ───
    inst_vols = _group_daily_series(daily, "institutional")
    asia_vols = _group_daily_series(daily, "asia")
    hedge_vols = _group_daily_series(daily, "hedge")

    # ─── 기본 추이 ───
    inst_trend = _calc_trend(inst_vols)
    asia_trend = _calc_trend(asia_vols)
    hedge_trend = _calc_trend(hedge_vols)

    inst_consec = _detect_consecutive(inst_vols)
    asia_consec = _detect_consecutive(asia_vols)

    # ─── KIS 외국인 방향 ───
    foreign_net_5d = 0
    foreign_dir = "NEUTRAL"
    if kis_data:
        foreign_net_5d = kis_data.get("foreign_net_5d", 0)
        if foreign_net_5d > 10000:
            foreign_dir = "BUY"
        elif foreign_net_5d < -10000:
            foreign_dir = "SELL"

    # ─── 주가 데이터 ───
    prices = _load_price_data(ticker)
    price_change = _calc_price_change(prices, days=3)

    # ─── Secret #1: 가속도 ───
    inst_accel = _calc_acceleration(inst_vols)
    asia_accel = _calc_acceleration(asia_vols)

    # ─── Secret #2: 집중도 (HHI) ───
    latest_countries = daily[-1]["countries"]
    hhi, dominant_country, dominant_group, hhi_unusual = _calc_hhi(daily)

    # ─── Secret #5: Z-Score ───
    inst_z = _calc_zscore(inst_vols)
    asia_z = _calc_zscore(asia_vols)
    hedge_z = _calc_zscore(hedge_vols)

    # ─── Secret #6: Lead-Lag ───
    lead_lag = _detect_lead_lag(inst_vols, asia_vols)

    # ─── Secret #7: 거래량-가격 괴리 ───
    vol_price_pattern = _classify_vol_price(inst_trend, price_change)

    # ─── Secret #3: 조용한 매집 ───
    # 기관 추세가 실제 양수 + Z-score가 극단적 음수가 아닐 때만 유효
    is_quiet_accum = (
        _detect_quiet_accumulation(inst_vols, hedge_vols, price_change, foreign_dir)
        and inst_trend > -0.05    # 기관 추세 실질 양수~보합
        and inst_z > -1.5         # Z-score 극단적 감소 아님
    )

    # ═══════════════════════════════════════════
    # 스코어링 (총 -23 ~ +48)
    # ═══════════════════════════════════════════
    score = 0
    reasons: list[str] = []
    pattern = ""

    # --- Secret #1: 가속도 (최대 +8) ---
    if inst_accel > 0.15 and len(inst_vols) >= 4:
        # 가속도가 2일+ 양수이며 강한 수준
        score += 8
        reasons.append(f"기관 매집 가속 ({inst_accel:+.2f})")
    elif inst_accel > 0.05:
        score += 4
        reasons.append(f"기관 매집 가속 소폭 ({inst_accel:+.2f})")
    elif inst_accel < -0.15:
        score -= 5
        reasons.append(f"기관 매집 감속 ({inst_accel:+.2f})")

    # --- Secret #2: 집중도 (최대 +5) ---
    # 평소보다 비정상적으로 집중된 경우에만 시그널 (hhi_unusual)
    if hhi_unusual and dominant_group == "institutional":
        score += 5
        reasons.append(f"기관 집중 매집 이상치 (HHI={hhi:.2f}, {dominant_country})")
    elif hhi_unusual and dominant_group == "asia":
        score += 4
        reasons.append(f"아시아 집중 거래 이상치 (HHI={hhi:.2f}, {dominant_country})")
    elif hhi_unusual and dominant_group == "hedge":
        score -= 2
        reasons.append(f"헤지 집중 과열 (HHI={hhi:.2f}, 변동성 경고)")

    # --- Secret #3: 조용한 매집 (최대 +10) --- ★최고 패턴★
    if is_quiet_accum:
        score += 10
        pattern = "QUIET_ACCUM"
        reasons.append("★ 조용한 매집 감지 (기관↑ 헤지- 주가횡보 순매수+)")

    # --- Secret #4: 교차 종목 흐름 ---
    # (이건 전체 분석 단계에서 처리 — 아래 _apply_cross_flow에서 가산)

    # --- Secret #5: 이상치 Z-Score (최대 +7 / -5) ---
    if inst_z >= 2.0:
        z_score_pts = min(7, int(inst_z * 2.5))
        score += z_score_pts
        reasons.append(f"기관 이상치 거래량 (Z={inst_z:.1f}σ)")
    elif inst_z <= -2.0:
        score -= 3
        reasons.append(f"기관 거래량 급감 (Z={inst_z:.1f}σ)")

    if asia_z >= 2.5:
        score += 4
        reasons.append(f"아시아 이상치 (Z={asia_z:.1f}σ)")

    if hedge_z >= 2.5:
        score -= 3
        reasons.append(f"헤지 과열 (Z={hedge_z:.1f}σ, 피크 경고)")

    # --- Secret #6: 선행 선점 (최대 +5) ---
    if lead_lag == "LEAD":
        score += 5
        reasons.append("영국 선행 매집, 아시아 미진입 → 선점 기회")
    elif lead_lag == "FOLLOW":
        score += 2
        reasons.append("영국+아시아 동반 진입 중")

    # --- Secret #7: 가격 괴리 (최대 +8 / -8) ---
    if vol_price_pattern == "ABSORB":
        score += 8
        reasons.append(f"흡수 매집 (기관↑ 주가{price_change:+.1f}% → 매도 흡수)")
    elif vol_price_pattern == "TRAP":
        score -= 8
        reasons.append(f"함정 (기관↓ 주가{price_change:+.1f}% → 개인 주도)")
    elif vol_price_pattern == "ABANDON":
        score -= 4
        reasons.append(f"관심 상실 (기관↓ 주가↓)")

    # --- 기존 추세 보강 (Secret에 안 잡힌 기본 시그널) ---
    # 기관 연속 매집/이탈 (Secret #1/5로 안 잡힌 경우만)
    if inst_consec >= 4 and "기관" not in " ".join(reasons):
        score += 3
        reasons.append(f"기관 {inst_consec}일 연속 거래 증가")
    elif inst_consec <= -3 and "기관" not in " ".join(reasons):
        score -= 3
        reasons.append(f"기관 {abs(inst_consec)}일 연속 거래 감소")

    # 아시아 연속
    if asia_consec >= 3 and "아시아" not in " ".join(reasons):
        score += 3
        reasons.append(f"아시아 {asia_consec}일 연속 증가")
    elif asia_consec <= -3:
        score -= 3
        reasons.append(f"아시아 {abs(asia_consec)}일 연속 감소")

    # KIS 외국인 방향 보강
    if foreign_dir == "BUY" and "순매수" not in " ".join(reasons):
        score += 3
        reasons.append(f"외국인 순매수 {foreign_net_5d:,}주")
    elif foreign_dir == "SELL" and "순매도" not in " ".join(reasons):
        score -= 3
        reasons.append(f"외국인 순매도 {foreign_net_5d:,}주")

    # --- 패턴 라벨링 ---
    if not pattern:
        if inst_accel > 0.10 and inst_z >= 1.5:
            pattern = "INST_ACCEL"
        elif asia_z >= 2.0 and asia_trend > 0.15:
            pattern = "ASIA_ENTRY"
        elif hedge_z >= 2.0 and hedge_trend > 0.3:
            pattern = "HEDGE_OVERHEAT"
        elif inst_trend < -0.2 and asia_trend < -0.2:
            pattern = "FULL_EXIT"
        elif vol_price_pattern == "ABSORB":
            pattern = "ABSORB_BUY"
        elif vol_price_pattern == "TRAP":
            pattern = "TRAP_WARNING"
        elif lead_lag == "LEAD":
            pattern = "LEAD_PREEMPT"
        else:
            pattern = "MIXED"

    # ═══════════════════════════════════════════
    # 등급 판정 (v2 기준)
    # ═══════════════════════════════════════════
    if score >= 35:
        signal = "STRONG_BUY"
    elif score >= 20:
        signal = "BUY"
    elif score >= 5:
        signal = "NEUTRAL"
    elif score >= -10:
        signal = "CAUTION"
    else:
        signal = "SELL"

    # TOP 국가
    sorted_countries = sorted(
        latest_countries.items(), key=lambda x: x[1], reverse=True
    )
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
        pattern=pattern,
        inst_trend=round(inst_trend, 3),
        asia_trend=round(asia_trend, 3),
        hedge_trend=round(hedge_trend, 3),
        inst_acceleration=round(inst_accel, 4),
        asia_acceleration=round(asia_accel, 4),
        hhi=hhi,
        dominant_country=dominant_country,
        dominant_group=dominant_group,
        inst_zscore=round(inst_z, 2),
        asia_zscore=round(asia_z, 2),
        hedge_zscore=round(hedge_z, 2),
        price_change_pct=round(price_change, 2),
        vol_price_pattern=vol_price_pattern,
        foreign_net_5d=foreign_net_5d,
        foreign_direction=foreign_dir,
        top_countries=top_countries,
    )


# ═══════════════════════════════════════════════════
# Secret #4: 교차 종목 흐름 (전체 분석 후 적용)
# ═══════════════════════════════════════════════════

def _apply_cross_flow(signals: list[NationalitySignal]) -> None:
    """섹터별 기관 흐름 방향 분석 → 동조 종목에 보너스/감점.

    같은 섹터 내 2개+ 종목이 기관 증가 → 섹터 전체 유입 확인 → +5점
    같은 섹터 내 2개+ 종목이 기관 감소 → 섹터 전체 이탈 → -5점
    다른 섹터 기관 자금과 역방향 → 자금 유출 섹터 -3점
    """
    # 섹터별 기관 추이 집계
    sector_inst: dict[str, list[float]] = {}
    for sig in signals:
        sector = TICKER_SECTOR.get(sig.ticker, "기타")
        if sector not in sector_inst:
            sector_inst[sector] = []
        sector_inst[sector].append(sig.inst_trend)

    # 섹터 방향 판별
    sector_direction: dict[str, str] = {}
    for sector, trends in sector_inst.items():
        if len(trends) < 2:
            sector_direction[sector] = "UNKNOWN"
            continue
        avg = sum(trends) / len(trends)
        up_count = sum(1 for t in trends if t > 0.05)
        down_count = sum(1 for t in trends if t < -0.05)

        if up_count >= 2 and avg > 0.08:
            sector_direction[sector] = "INFLOW"
        elif down_count >= 2 and avg < -0.08:
            sector_direction[sector] = "OUTFLOW"
        else:
            sector_direction[sector] = "MIXED"

    # 종목별 보너스/감점 적용
    inflow_sectors = [s for s, d in sector_direction.items() if d == "INFLOW"]
    outflow_sectors = [s for s, d in sector_direction.items() if d == "OUTFLOW"]
    mixed_sectors = [s for s, d in sector_direction.items() if d == "MIXED"]

    # 핵심 수정: 전체 시장이 같은 방향이면 변별력 없음
    # 유입+이탈 섹터가 모두 존재할 때만 교차 흐름 시그널 적용 (자금 이동 감지)
    has_divergence = len(inflow_sectors) >= 1 and len(outflow_sectors) >= 1

    for sig in signals:
        sector = TICKER_SECTOR.get(sig.ticker, "기타")

        if not has_divergence:
            # 전체 동일 방향 → 교차 흐름 스킵 (변별력 없음)
            continue

        if sector_direction.get(sector) == "INFLOW":
            sig.score += 5
            sig.reasons.append(f"섹터({sector}) 기관 동반 유입 확인")
        elif sector_direction.get(sector) == "OUTFLOW":
            sig.score -= 3  # 전체 이탈이 아닐 때만, 감점도 완화
            sig.reasons.append(f"섹터({sector}) 기관 이탈 (타 섹터 유입 중)")

        # 자금 이동 경로: 유출 섹터 → 유입 섹터 가산
        if sector in inflow_sectors and outflow_sectors:
            from_sectors = ", ".join(outflow_sectors[:2])
            sig.score += 2
            sig.reasons.append(f"자금 이동: {from_sectors} → {sector}")

    # 교차 흐름 적용 후 등급 재판정
    for sig in signals:
        if sig.score >= 35:
            sig.signal = "STRONG_BUY"
        elif sig.score >= 20:
            sig.signal = "BUY"
        elif sig.score >= 5:
            sig.signal = "NEUTRAL"
        elif sig.score >= -10:
            sig.signal = "CAUTION"
        else:
            sig.signal = "SELL"


# ═══════════════════════════════════════════════════
# 전체 분석 실행
# ═══════════════════════════════════════════════════

def run_analysis(db_path: Path = DB_PATH) -> list[NationalitySignal]:
    """전 종목 국적별 수급 시그널 생성 (v2 — 7 Secrets)."""
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

    # Secret #4: 교차 종목 흐름 (전체 결과에 대해 적용)
    _apply_cross_flow(signals)

    # 스코어 순 정렬
    signals.sort(key=lambda s: s.score, reverse=True)

    # JSON 저장
    output = {
        "engine_version": "v2_7secrets",
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
        "sector_flow": _summarize_sector_flow(signals),
        "patterns": _summarize_patterns(signals),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"v2 분석 완료: {len(signals)}종목 | "
        f"강매수={output['summary']['strong_buy']} "
        f"매수={output['summary']['buy']} "
        f"중립={output['summary']['neutral']} "
        f"주의={output['summary']['caution']} "
        f"매도={output['summary']['sell']}"
    )

    return signals


def _summarize_sector_flow(signals: list[NationalitySignal]) -> dict:
    """섹터별 흐름 요약."""
    sector_data: dict[str, list] = {}
    for sig in signals:
        sector = TICKER_SECTOR.get(sig.ticker, "기타")
        if sector not in sector_data:
            sector_data[sector] = []
        sector_data[sector].append({
            "ticker": sig.ticker,
            "name": sig.name,
            "score": sig.score,
            "inst_trend": sig.inst_trend,
            "pattern": sig.pattern,
        })

    result = {}
    for sector, stocks in sector_data.items():
        avg_score = sum(s["score"] for s in stocks) / len(stocks) if stocks else 0
        avg_inst = sum(s["inst_trend"] for s in stocks) / len(stocks) if stocks else 0
        direction = "INFLOW" if avg_inst > 0.08 else "OUTFLOW" if avg_inst < -0.08 else "MIXED"
        result[sector] = {
            "direction": direction,
            "avg_score": round(avg_score, 1),
            "avg_inst_trend": round(avg_inst, 3),
            "stocks": len(stocks),
        }
    return result


def _summarize_patterns(signals: list[NationalitySignal]) -> dict:
    """패턴별 종목 요약."""
    patterns: dict[str, list[str]] = {}
    for sig in signals:
        if sig.pattern and sig.pattern != "MIXED":
            if sig.pattern not in patterns:
                patterns[sig.pattern] = []
            patterns[sig.pattern].append(sig.name)
    return patterns


# ═══════════════════════════════════════════════════
# 텔레그램 포맷
# ═══════════════════════════════════════════════════

def format_telegram(signals: list[NationalitySignal]) -> str:
    """텔레그램 메시지 포맷 (v2 — 패턴 포함)."""
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
    PATTERN_LABEL = {
        "QUIET_ACCUM": "조용한 매집",
        "INST_ACCEL": "기관 가속 매집",
        "ASIA_ENTRY": "아시아 진입",
        "ABSORB_BUY": "흡수 매집",
        "LEAD_PREEMPT": "선점 기회",
        "HEDGE_OVERHEAT": "헤지 과열",
        "FULL_EXIT": "전면 이탈",
        "TRAP_WARNING": "함정 경고",
    }

    lines = ["<b>🌍 국적별 외국인 수급 v2 (7 Secrets)</b>", ""]

    # 비중립만 표시
    active = [s for s in signals if s.signal != "NEUTRAL"]
    if not active:
        lines.append("모든 종목 중립 — 특이 시그널 없음")
        return "\n".join(lines)

    for sig in active[:15]:  # 상위 15개
        emoji = EMOJI.get(sig.signal, "⚪")
        label = LABEL.get(sig.signal, sig.signal)
        score_str = f"+{sig.score}" if sig.score > 0 else str(sig.score)
        pat_label = PATTERN_LABEL.get(sig.pattern, "")
        pat_str = f" [{pat_label}]" if pat_label else ""

        lines.append(f"{emoji} <b>{label}</b>  {sig.name}  {score_str}점{pat_str}")
        for reason in sig.reasons[:4]:
            lines.append(f"    · {reason}")
        lines.append("")

    # 섹터 흐름
    sector_inflow = []
    sector_outflow = []
    for sig in signals:
        for r in sig.reasons:
            if "동반 유입" in r and "섹터" in r:
                sector = r.split("(")[1].split(")")[0] if "(" in r else ""
                if sector and sector not in sector_inflow:
                    sector_inflow.append(sector)
            elif "동반 이탈" in r and "섹터" in r:
                sector = r.split("(")[1].split(")")[0] if "(" in r else ""
                if sector and sector not in sector_outflow:
                    sector_outflow.append(sector)

    if sector_inflow or sector_outflow:
        lines.append("<b>📊 섹터 자금 흐름</b>")
        if sector_inflow:
            lines.append(f"  유입: {', '.join(sector_inflow)}")
        if sector_outflow:
            lines.append(f"  이탈: {', '.join(sector_outflow)}")
        lines.append("")

    # 핵심 패턴 요약
    quiet = [s.name for s in signals if s.pattern == "QUIET_ACCUM"]
    absorb = [s.name for s in signals if s.pattern == "ABSORB_BUY"]
    lead = [s.name for s in signals if s.pattern == "LEAD_PREEMPT"]
    traps = [s.name for s in signals if s.pattern == "TRAP_WARNING"]

    if quiet or absorb or lead:
        lines.append("<b>🔑 핵심 패턴</b>")
        if quiet:
            lines.append(f"  조용한 매집: {', '.join(quiet[:3])}")
        if absorb:
            lines.append(f"  흡수 매집: {', '.join(absorb[:3])}")
        if lead:
            lines.append(f"  선점 기회: {', '.join(lead[:3])}")
        if traps:
            lines.append(f"  ⚠️ 함정: {', '.join(traps[:3])}")

    return "\n".join(lines)
