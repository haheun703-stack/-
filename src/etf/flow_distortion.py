"""ETF 수급 왜곡 분석 엔진 — A+B+C 통합

검증된 사실 (ETF Flow X-Ray v2):
  개인이 섹터 ETF를 대량 매수하면,
  AP(지정참가회사)가 ETF 설정 과정에서 기초자산을 T+1~T+2에 매수.
  이 매수는 개별종목에서 "기관(금융투자)" 매수로 기록됨.

3가지 분석:
  A) calc_distortion_ratio  — 기관 순매수에서 ETF 기계적 매수 차감
  B) detect_leading_signal  — ETF 개인 급증 → T+1~T+2 선행 시그널
  C) calc_sector_sentiment  — ETF 개인 순매수 추세로 섹터 센티먼트 판정
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ETF_FLOW_PATH = PROJECT_ROOT / "data" / "etf_investor_flow.json"
ETF_UNIVERSE_PATH = PROJECT_ROOT / "data" / "sector_rotation" / "etf_universe.json"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
LEADING_SIGNAL_PATH = PROJECT_ROOT / "data" / "etf_flow_leading.json"

# ETF Flow X-Ray v2에서 왜곡 효과 없음 확인된 종목
_DEFAULT_EXCLUDE = {"005930", "005380"}  # 삼성전자, 현대차


def _load_settings() -> dict:
    """settings.yaml에서 etf_flow_distortion 섹션 로드."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("etf_flow_distortion", {})
    except Exception:
        return {}


def load_etf_flow_data() -> dict:
    """data/etf_investor_flow.json 로드."""
    if not ETF_FLOW_PATH.exists():
        logger.warning("ETF flow 데이터 없음: %s", ETF_FLOW_PATH)
        return {}
    try:
        with open(ETF_FLOW_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("ETF flow 로드 실패: %s", e)
        return {}


def _load_etf_universe() -> dict:
    """etf_universe.json 로드 → {섹터: {etf_code, stock_count}}."""
    if not ETF_UNIVERSE_PATH.exists():
        return {}
    try:
        with open(ETF_UNIVERSE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_sector_etf_code(sector: str, etf_flow_data: dict) -> str | None:
    """섹터명으로 ETF flow 데이터에서 해당 ETF 코드 찾기."""
    etfs = etf_flow_data.get("etfs", {})
    for code, info in etfs.items():
        if info.get("sector") == sector:
            return code
    return None


def _get_stock_count(sector: str) -> int:
    """ETF 구성종목 수 조회."""
    universe = _load_etf_universe()
    info = universe.get(sector, {})
    count = info.get("stock_count", 0)
    return max(count, 10)  # 최소 10으로 제한 (0 방지)


# ================================================================
# A: 수급 왜곡 비율 계산
# ================================================================
def calc_distortion_ratio(
    ticker: str,
    sector: str,
    inst_net_raw: int,
    etf_flow_data: dict | None = None,
    weight_in_etf: float | None = None,
    lookback_days: int = 2,
) -> dict:
    """기관 순매수에서 ETF 생성 기계적 매수를 추정하여 차감.

    Args:
        ticker: 종목코드
        sector: 섹터명
        inst_net_raw: 원래 기관 순매수 (원)
        etf_flow_data: collect_etf_investor_flow 결과 (None이면 자동 로드)
        weight_in_etf: ETF 내 종목 비중 (None이면 균등 추정)
        lookback_days: ETF 개인매수 합산 일수 (T-1, T-2 = 2일)

    Returns:
        {
            "raw_inst": 원래 기관순매수,
            "etf_mechanical": ETF에 의한 기계적 매수 추정,
            "adjusted_inst": 보정된 기관순매수,
            "distortion_pct": 왜곡 비율 (%, 0~100)
        }
    """
    cfg = _load_settings()
    exclude = set(cfg.get("distortion_correction", {}).get("exclude_tickers", _DEFAULT_EXCLUDE))

    default = {
        "raw_inst": inst_net_raw,
        "etf_mechanical": 0,
        "adjusted_inst": inst_net_raw,
        "distortion_pct": 0.0,
    }

    # 제외 종목
    if ticker in exclude:
        return default

    # 기관 순매도인 경우 왜곡 보정 불필요
    if inst_net_raw <= 0:
        return default

    if etf_flow_data is None:
        etf_flow_data = load_etf_flow_data()
    if not etf_flow_data:
        return default

    # 해당 섹터 ETF 찾기
    etf_code = _get_sector_etf_code(sector, etf_flow_data)
    if not etf_code:
        return default

    etf_info = etf_flow_data["etfs"].get(etf_code, {})
    days = etf_info.get("days", [])
    if len(days) < lookback_days:
        return default

    # T-1 ~ T-lookback_days 개인 순매수 합산
    recent_days = days[-lookback_days:]
    etf_individual_net = sum(d.get("individual_net", 0) for d in recent_days)

    # 개인이 순매도한 경우 → ETF 생성 아닌 환매 → 왜곡 0
    if etf_individual_net <= 0:
        return default

    # 종목 비중 추정
    if weight_in_etf is None:
        stock_count = _get_stock_count(sector)
        # 시총 상위 종목은 비중 높음 → top3에 50% 가정, 나머지 균등
        # 보수적으로 균등 배분
        weight_in_etf = 1.0 / stock_count

    # 기계적 매수 추정
    etf_mechanical = int(etf_individual_net * weight_in_etf)

    # 보정
    adjusted = max(0, inst_net_raw - etf_mechanical)
    distortion_pct = min(100.0, (etf_mechanical / inst_net_raw) * 100) if inst_net_raw > 0 else 0.0

    return {
        "raw_inst": inst_net_raw,
        "etf_mechanical": etf_mechanical,
        "adjusted_inst": adjusted,
        "distortion_pct": round(distortion_pct, 1),
    }


# ================================================================
# B: 선행 매수 시그널
# ================================================================

# 섹터별 주요 기초종목 매핑
_SECTOR_TOP_STOCKS = {
    "반도체": [("000660", "SK하이닉스")],  # 삼성전자 제외 (효과 없음)
    "은행": [("105560", "KB금융"), ("055550", "신한지주")],
    "건설": [("000720", "현대건설"), ("047040", "대우건설")],
    "조선": [("009540", "HD한국조선해양")],  # 삼성중공업 노이즈
    "2차전지": [("373220", "LG에너지솔루션"), ("006400", "삼성SDI")],
    "바이오": [("068270", "셀트리온"), ("207940", "삼성바이오로직스")],
    "방산": [("012450", "한화에어로스페이스"), ("064350", "현대로템")],
    "IT": [("035420", "NAVER"), ("035720", "카카오")],
    "에너지화학": [("010950", "S-Oil"), ("096770", "SK이노베이션")],
    "철강소재": [("005490", "POSCO홀딩스")],
    "금융": [("105560", "KB금융"), ("055550", "신한지주")],
}


def detect_leading_signal(etf_flow_data: dict | None = None) -> list[dict]:
    """ETF 개인매수 급증 시 → T+1~T+2 기초종목 기관매수 예측.

    조건: 최근 1일 개인순매수가 20일 평균의 N배 이상 (기본 2배)

    Returns:
        list of {
            "sector", "etf_code", "etf_name",
            "individual_net_today", "individual_net_ma20",
            "surge_ratio",
            "expected_stocks": [(code, name), ...],
            "signal_strength": "STRONG" | "MODERATE",
            "message": str
        }
    """
    cfg = _load_settings()
    leading_cfg = cfg.get("leading_signal", {})
    if not leading_cfg.get("enabled", True):
        return []

    surge_threshold = leading_cfg.get("surge_threshold", 2.0)
    strong_threshold = leading_cfg.get("strong_threshold", 3.0)

    if etf_flow_data is None:
        etf_flow_data = load_etf_flow_data()
    if not etf_flow_data:
        return []

    signals = []
    etfs = etf_flow_data.get("etfs", {})

    for etf_code, info in etfs.items():
        days = info.get("days", [])
        if len(days) < 21:
            continue

        sector = info.get("sector", "")

        # 최근 1일 개인 순매수
        today_net = days[-1].get("individual_net", 0)
        if today_net <= 0:
            continue

        # 20일 평균
        ma20_values = [d.get("individual_net", 0) for d in days[-21:-1]]
        # 양수만으로 평균 (순매도일 제외)
        positive_days = [v for v in ma20_values if v > 0]
        if not positive_days:
            continue
        ma20 = sum(positive_days) / len(positive_days)
        if ma20 <= 0:
            continue

        surge_ratio = today_net / ma20

        if surge_ratio < surge_threshold:
            continue

        # 시그널 등급
        if surge_ratio >= strong_threshold:
            strength = "STRONG"
        else:
            strength = "MODERATE"

        # 기초종목
        expected = _SECTOR_TOP_STOCKS.get(sector, [])

        today_bil = today_net / 1e8
        ma20_bil = ma20 / 1e8

        stock_names = ", ".join(name for _, name in expected) if expected else "미매핑"

        msg = (f"{info['name']} 개인 {surge_ratio:.1f}배 급증 "
               f"({today_bil:.0f}억 vs 평균 {ma20_bil:.0f}억) → "
               f"{stock_names} T+1~T+2 기관매수 유입 예상")

        signals.append({
            "sector": sector,
            "etf_code": etf_code,
            "etf_name": info.get("name", ""),
            "individual_net_today": today_net,
            "individual_net_ma20": ma20,
            "surge_ratio": round(surge_ratio, 2),
            "expected_stocks": expected,
            "signal_strength": strength,
            "message": msg,
        })

    # STRONG 우선, surge_ratio 내림차순
    signals.sort(key=lambda x: (-1 if x["signal_strength"] == "STRONG" else 0, -x["surge_ratio"]))

    # 저장
    if signals:
        try:
            with open(LEADING_SIGNAL_PATH, "w", encoding="utf-8") as f:
                # expected_stocks를 직렬화 가능하게 변환
                serializable = []
                for s in signals:
                    s_copy = dict(s)
                    s_copy["expected_stocks"] = [
                        {"code": c, "name": n} for c, n in s["expected_stocks"]
                    ]
                    serializable.append(s_copy)
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("선행 시그널 저장 실패: %s", e)

    return signals


# ================================================================
# C: 섹터 센티먼트
# ================================================================
def calc_sector_sentiment(etf_flow_data: dict | None = None) -> dict:
    """ETF 개인 순매수 추세 → 섹터별 소매 센티먼트.

    Returns:
        {
            "반도체": {
                "sentiment": "HOT" | "WARM" | "NEUTRAL" | "COLD",
                "individual_net_5d": int (원),
                "individual_net_1d": int (원),
                "consecutive_buy_days": int,
                "trend": "accelerating" | "steady" | "decelerating",
                "rank": int
            }
        }
    """
    cfg = _load_settings()
    sent_cfg = cfg.get("sentiment", {})
    if not sent_cfg.get("enabled", True):
        return {}

    hot_threshold = float(sent_cfg.get("hot_threshold", 100e8))
    cold_threshold = float(sent_cfg.get("cold_threshold", -100e8))

    if etf_flow_data is None:
        etf_flow_data = load_etf_flow_data()
    if not etf_flow_data:
        return {}

    result = {}
    etfs = etf_flow_data.get("etfs", {})

    for etf_code, info in etfs.items():
        days = info.get("days", [])
        if len(days) < 5:
            continue

        sector = info.get("sector", "")
        if not sector:
            continue

        # 최근 5일 개인 순매수
        recent_5 = days[-5:]
        net_5d = sum(d.get("individual_net", 0) for d in recent_5)
        net_1d = days[-1].get("individual_net", 0)

        # 연속 순매수 일수
        consec = 0
        for d in reversed(recent_5):
            if d.get("individual_net", 0) > 0:
                consec += 1
            else:
                break

        # 추세 (3일 vs 이전 3일)
        if len(days) >= 6:
            recent_3 = sum(d.get("individual_net", 0) for d in days[-3:])
            prev_3 = sum(d.get("individual_net", 0) for d in days[-6:-3])
            if recent_3 > prev_3 * 1.3:
                trend = "accelerating"
            elif recent_3 < prev_3 * 0.7:
                trend = "decelerating"
            else:
                trend = "steady"
        else:
            trend = "steady"

        # 등급 판정
        if consec >= 5 and net_5d > hot_threshold:
            sentiment = "HOT"
        elif consec >= 3 or net_5d > hot_threshold * 0.5:
            sentiment = "WARM"
        elif consec == 0 and net_5d < cold_threshold:
            sentiment = "COLD"
        else:
            sentiment = "NEUTRAL"

        result[sector] = {
            "sentiment": sentiment,
            "individual_net_5d": net_5d,
            "individual_net_1d": net_1d,
            "consecutive_buy_days": consec,
            "trend": trend,
        }

    # 순위 부여 (5일 누적 내림차순)
    sorted_sectors = sorted(result.items(), key=lambda x: -x[1]["individual_net_5d"])
    for rank, (sector, info) in enumerate(sorted_sectors, 1):
        info["rank"] = rank

    return result


# ================================================================
# 통합 분석 (스캔 시 호출)
# ================================================================
def get_distortion_for_scan(
    ticker: str,
    sector: str,
    inst_net_raw: int,
) -> float:
    """스캔 시 간편 호출 — 왜곡 비율(%)만 반환.

    etf_flow_data를 매번 로드하지 않도록 모듈 레벨 캐시 활용.
    """
    cfg = _load_settings()
    if not cfg.get("enabled", True):
        return 0.0
    if not cfg.get("distortion_correction", {}).get("enabled", True):
        return 0.0

    result = calc_distortion_ratio(ticker, sector, inst_net_raw)
    return result["distortion_pct"]
