"""투자자 수급 통합본 시그널 스캐너 (pykrx KRX + KIS)

정보봇이 수집한 investor_flow.json에서 6종 시그널을 판정하고
종목별 수급 점수를 산출합니다.

데이터 소스: D:/Global_Stock_Overview_Scripter_정보봇/data/supply_daily/{date}_investor_flow.json

기본 6 시그널:
  시장 — FOREIGN_MASS_SELL/BUY, INST_BUYING_HEAVY/SELLING_HEAVY
  종목 — FOREIGN_MEGA_BUY/SELL

고급 전략 5종 (advanced):
  1. 역행 매수 (Contrarian) — 외인 MASS_SELL + 기관 TOP 매수 = 확신 매수
  2. 기관 흡수 (Absorption) — 외인 매도를 기관이 흡수 (비율 측정)
  3. 수급 연속성 (Consecutive) — TOP 리스트 N일 연속 등장
  4. 코스닥 로테이션 (Rotation) — KOSPI↔KOSDAQ 자금 이동
  5. 연기금 바닥 (Pension) — 연기금 대량 매수 = 바닥 시그널

출력: data/investor_flow/investor_flow_signal.json

Usage:
    python scripts/scan_investor_flow.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── 경로 설정 ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.jgis_short_adapter import (
    load_investor_flow,
    load_investor_flow_history,
    load_investor_flow_summary_from_intel,
)
from src.utils.atomic_io import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "config" / "settings.yaml"
OUTPUT_DIR = ROOT / "data" / "investor_flow"
OUTPUT_PATH = OUTPUT_DIR / "investor_flow_signal.json"
SHORT_FACTOR_PATH = ROOT / "data" / "short_selling" / "jgis_short_factor.json"

_BILLION = 1_0000_0000  # 1억 = 1e8


# ══════════════════════════════════════════════════════════
# 설정 로드
# ══════════════════════════════════════════════════════════

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ══════════════════════════════════════════════════════════
# 시장 레벨 시그널 (4종)
# ══════════════════════════════════════════════════════════

def detect_market_signals(
    current: dict,
    history: list[dict],
    cfg: dict,
) -> list[dict]:
    """시장 전체 수급 시그널 판정.

    Returns:
        [{"type": "FOREIGN_MASS_SELL", "msg": "외인 2일 연속 ..."}]
    """
    signals = []
    mkt_cfg = cfg.get("market_signals", {})
    mass_th = mkt_cfg.get("foreign_mass_threshold", 3000)
    inst_th = mkt_cfg.get("inst_heavy_threshold", 5000)

    f_today = current.get("foreign", 0)
    i_today = current.get("institution", 0)

    # 전일 데이터 — current.date와 일치하는 history 인덱스 찾아 그 직전 항목 선택
    # (fallback 로드로 current.date와 history[-1].date가 같을 수 있음)
    f_prev = None
    cur_date = current.get("date")
    if len(history) >= 2:
        prev_idx = None
        if cur_date:
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("date") == cur_date and i >= 1:
                    prev_idx = i - 1
                    break
        # 매칭 실패 시 마지막에서 2번째 (기존 동작 fallback)
        if prev_idx is None:
            prev_idx = -2
        prev = history[prev_idx]
        # current와 prev가 같은 날짜면 무효화 (2일 연속 판정 불가)
        if prev.get("date") != cur_date:
            f_prev = prev.get("foreign", 0)

    # 1) FOREIGN_MASS_SELL: 2일 연속 -threshold
    if f_prev is not None and f_today <= -mass_th and f_prev <= -mass_th:
        signals.append({
            "type": "FOREIGN_MASS_SELL",
            "msg": f"외인 2일 연속 대량 매도 ({f_prev:+,.0f}→{f_today:+,.0f}억)",
        })

    # 2) FOREIGN_MASS_BUY: 2일 연속 +threshold
    if f_prev is not None and f_today >= mass_th and f_prev >= mass_th:
        signals.append({
            "type": "FOREIGN_MASS_BUY",
            "msg": f"외인 2일 연속 대량 매수 ({f_prev:+,.0f}→{f_today:+,.0f}억)",
        })

    # 3) INST_BUYING_HEAVY: 기관 당일 +threshold
    if i_today >= inst_th:
        signals.append({
            "type": "INST_BUYING_HEAVY",
            "msg": f"기관 당일 대량 매수 ({i_today:+,.0f}억)",
        })

    # 4) INST_SELLING_HEAVY: 기관 당일 -threshold
    if i_today <= -inst_th:
        signals.append({
            "type": "INST_SELLING_HEAVY",
            "msg": f"기관 당일 대량 매도 ({i_today:+,.0f}억)",
        })

    return signals


# ══════════════════════════════════════════════════════════
# 종목 레벨 시그널 + 수급 점수
# ══════════════════════════════════════════════════════════

def _calc_tier_score(
    net_억: float,
    tiers: dict,
    side: str = "buy",
) -> int:
    """yaml 정의된 컷에 따라 100/70/40/20 점수 계산.

    Args:
        net_억: 순매수 금액 (억원, 매도면 음수)
        tiers: {"mega": 1000, "large": 500, "medium": 100, "small": 0}
        side: "buy"면 양수 점수, "sell"면 음수 점수
    """
    sign = 1 if side == "buy" else -1
    abs_amt = abs(net_억)
    mega = tiers.get("mega", 1000)
    large = tiers.get("large", 500)
    medium = tiers.get("medium", 100)
    if abs_amt >= mega:
        return 100 * sign
    elif abs_amt >= large:
        return 70 * sign
    elif abs_amt >= medium:
        return 40 * sign
    return 20 * sign


def compute_stock_signals(
    flow: dict,
    cfg: dict,
) -> tuple[dict, list[str], list[str], list[dict]]:
    """종목별 수급 점수 + MEGA 시그널.

    Returns:
        (stock_scores, mega_buy_tickers, mega_sell_tickers, stock_signals)
    """
    stk_cfg = cfg.get("stock_signals", {})
    int_cfg = cfg.get("integration", {})
    mega_th = stk_cfg.get("mega_threshold", 1000)
    inst_weight = int_cfg.get("inst_weight", 0.5)
    tiers = stk_cfg.get("score_tiers", {"mega": 1000, "large": 500, "medium": 100, "small": 0})

    stock_scores: dict[str, dict] = {}
    mega_buy: list[str] = []
    mega_sell: list[str] = []
    stock_signals: list[dict] = []

    # 외인 매수 TOP
    for item in flow.get("foreign_top_buy", []):
        ticker = item.get("ticker", "")
        name = item.get("name", "")
        net_amt = item.get("net_amt", 0)
        net_qty = item.get("net_qty", 0)
        net_억 = net_amt / _BILLION

        if net_억 >= mega_th:
            mega_buy.append(ticker)
            stock_signals.append({
                "type": "FOREIGN_MEGA_BUY",
                "ticker": ticker,
                "name": name,
                "msg": f"외인 {name} +{net_억:,.0f}억 순매수",
            })

        # 수급 점수 (외인 매수)
        f_score = _calc_tier_score(net_억, tiers, side="buy")

        stock_scores[ticker] = {
            "name": name,
            "foreign_amt_억": round(net_억, 1),
            "foreign_qty": net_qty,
            "foreign_score": f_score,
            "inst_score": 0,
            "flow_score": f_score,
            "side": "buy",
        }

    # 외인 매도 TOP
    for item in flow.get("foreign_top_sell", []):
        ticker = item.get("ticker", "")
        name = item.get("name", "")
        net_amt = item.get("net_amt", 0)
        net_qty = item.get("net_qty", 0)
        net_억 = net_amt / _BILLION  # 음수

        if net_억 <= -mega_th:
            mega_sell.append(ticker)
            stock_signals.append({
                "type": "FOREIGN_MEGA_SELL",
                "ticker": ticker,
                "name": name,
                "msg": f"외인 {name} {net_억:,.0f}억 순매도",
            })

        # 매도 종목은 음수 점수로
        f_score = _calc_tier_score(net_억, tiers, side="sell")

        if ticker not in stock_scores:
            stock_scores[ticker] = {
                "name": name,
                "foreign_amt_억": round(net_억, 1),
                "foreign_qty": net_qty,
                "foreign_score": f_score,
                "inst_score": 0,
                "flow_score": f_score,
                "side": "sell",
            }
        else:
            # 매수 리스트에도 있으면 합산 (극히 드문 경우)
            stock_scores[ticker]["foreign_amt_억"] += round(net_억, 1)
            stock_scores[ticker]["foreign_score"] += f_score

    # 기관 매수 TOP → 가중치 적용
    for item in flow.get("institution_top_buy", []):
        ticker = item.get("ticker", "")
        name = item.get("name", "")
        net_amt = item.get("net_amt", 0)
        net_qty = item.get("net_qty", 0)
        net_억 = net_amt / _BILLION

        i_score = _calc_tier_score(net_억, tiers, side="buy")
        weighted = round(i_score * inst_weight)

        if ticker in stock_scores:
            stock_scores[ticker]["inst_amt_억"] = round(net_억, 1)
            stock_scores[ticker]["inst_qty"] = net_qty
            stock_scores[ticker]["inst_score"] = weighted
            stock_scores[ticker]["flow_score"] += weighted
        else:
            stock_scores[ticker] = {
                "name": name,
                "foreign_amt_억": 0,
                "foreign_qty": 0,
                "foreign_score": 0,
                "inst_amt_억": round(net_억, 1),
                "inst_qty": net_qty,
                "inst_score": weighted,
                "flow_score": weighted,
                "side": "inst_buy",
            }

    # 기관 매도 TOP
    for item in flow.get("institution_top_sell", []):
        ticker = item.get("ticker", "")
        name = item.get("name", "")
        net_amt = item.get("net_amt", 0)
        net_억 = net_amt / _BILLION

        i_score = _calc_tier_score(net_억, tiers, side="sell")
        weighted = round(i_score * inst_weight)

        if ticker in stock_scores:
            stock_scores[ticker]["inst_amt_억"] = round(net_억, 1)
            stock_scores[ticker]["inst_score"] = weighted
            stock_scores[ticker]["flow_score"] += weighted
        else:
            stock_scores[ticker] = {
                "name": name,
                "foreign_amt_억": 0,
                "foreign_qty": 0,
                "foreign_score": 0,
                "inst_amt_억": round(net_억, 1),
                "inst_qty": 0,
                "inst_score": weighted,
                "flow_score": weighted,
                "side": "inst_sell",
            }

    return stock_scores, mega_buy, mega_sell, stock_signals


# ══════════════════════════════════════════════════════════
# 외인-기관 크로스 / 공매도 교차분석
# ══════════════════════════════════════════════════════════

def detect_cross_divergence(flow: dict) -> list[str]:
    """외인 매도 + 기관 매수 → 기관 흡수 종목."""
    foreign_sell = {x["ticker"] for x in flow.get("foreign_top_sell", [])}
    inst_buy = {x["ticker"] for x in flow.get("institution_top_buy", [])}
    return sorted(foreign_sell & inst_buy)


def detect_critical_risk(mega_sell_tickers: list[str]) -> list[str]:
    """공매도 극단 + 외인 대량 매도 → CRITICAL 리스크.

    jgis_short_factor.json에서 SHORT_EXTREME 또는 SHORT_CREDIT_DIVERGE 종목과 교차.
    """
    if not SHORT_FACTOR_PATH.exists():
        return []

    try:
        with open(SHORT_FACTOR_PATH, encoding="utf-8") as f:
            short_data = json.load(f)
    except Exception:
        return []

    all_res = short_data.get("all_results", {})
    short_extreme_tickers = set()
    for ticker, d in all_res.items():
        sigs = d.get("active_signals", [])
        if "SHORT_EXTREME" in sigs or "SHORT_CREDIT_DIVERGE" in sigs:
            short_extreme_tickers.add(ticker)

    critical = sorted(set(mega_sell_tickers) & short_extreme_tickers)
    return critical


# ══════════════════════════════════════════════════════════
# 시장 포지션 멀티플라이어 산출
# ══════════════════════════════════════════════════════════

def calc_market_position_mult(foreign_total: float, cfg: dict) -> float:
    """외인 시장합계 → 포지션 멀티플라이어.

    -10000억 이하: 0.85 (= 1 + -0.15)
    -5000억 이하:  0.93 (= 1 + -0.07)
    +5000억 이상:  1.05 (= 1 + 0.05)
    그 외:         1.00
    """
    int_cfg = cfg.get("integration", {})
    severe = int_cfg.get("market_penalty_severe", -0.15)
    moderate = int_cfg.get("market_penalty_moderate", -0.07)
    boost = int_cfg.get("market_boost", 0.05)

    if foreign_total <= -10000:
        return round(1.0 + severe, 4)
    elif foreign_total <= -5000:
        return round(1.0 + moderate, 4)
    elif foreign_total >= 5000:
        return round(1.0 + boost, 4)
    return 1.0


# ══════════════════════════════════════════════════════════
# 고급 전략 5종 ("시크릿 무기")
# ══════════════════════════════════════════════════════════

def detect_contrarian(
    flow: dict,
    market_signal_types: list[str],
    adv_cfg: dict,
) -> list[dict]:
    """전략1: 역행 매수 — 외인 MASS_SELL 장에서 기관이 TOP 매수한 종목.

    외인이 시장 전체에서 패닉 매도하는 중에도 기관이 대규모 매수하는 종목은
    '확신 매수'로 해석. 역사적으로 약세장에서 기관 선행 매수 종목은
    반등 시 초과수익을 보이는 경향.
    """
    if not adv_cfg.get("contrarian", {}).get("enabled", False):
        return []
    if "FOREIGN_MASS_SELL" not in market_signal_types:
        return []

    result = []
    for item in flow.get("institution_top_buy", []):
        net_억 = item.get("net_amt", 0) / _BILLION
        if net_억 >= 100:  # 기관 100억+ 매수만
            result.append({
                "ticker": item["ticker"],
                "name": item.get("name", ""),
                "inst_amt_억": round(net_억),
            })
    return result


def detect_absorption(
    flow: dict,
    cross_divergence: list[str],
    adv_cfg: dict,
) -> list[dict]:
    """전략2: 기관 흡수 — 외인 매도를 기관이 받아내는 종목 + 흡수율 측정.

    흡수율(absorption_ratio) = |기관 매수금| / |외인 매도금|
    ≥50% = FULL(완전 흡수), ≥30% = PARTIAL(부분), <30% = WEAK(약)
    높은 흡수율은 기관이 저가 물량을 적극 확보 중이라는 의미.
    """
    if not adv_cfg.get("absorption", {}).get("enabled", False):
        return []
    if not cross_divergence:
        return []

    # 매도/매수 금액 맵 구축
    foreign_sell_map: dict[str, float] = {}
    for item in flow.get("foreign_top_sell", []):
        foreign_sell_map[item["ticker"]] = item.get("net_amt", 0) / _BILLION

    inst_buy_map: dict[str, float] = {}
    inst_name_map: dict[str, str] = {}
    for item in flow.get("institution_top_buy", []):
        inst_buy_map[item["ticker"]] = item.get("net_amt", 0) / _BILLION
        inst_name_map[item["ticker"]] = item.get("name", "")

    result = []
    for ticker in cross_divergence:
        f_amt = foreign_sell_map.get(ticker, 0)  # 음수
        i_amt = inst_buy_map.get(ticker, 0)      # 양수
        if f_amt == 0:
            continue
        ratio = abs(i_amt) / abs(f_amt)
        grade = "FULL" if ratio >= 0.5 else "PARTIAL" if ratio >= 0.3 else "WEAK"
        result.append({
            "ticker": ticker,
            "name": inst_name_map.get(ticker, ""),
            "ratio": round(ratio, 2),
            "grade": grade,
            "foreign_amt_억": round(f_amt),
            "inst_amt_억": round(i_amt),
        })
    return result


def detect_consecutive_flow(
    flow: dict,
    history: list[dict],
    adv_cfg: dict,
) -> dict[str, int]:
    """전략3: 수급 연속성 — TOP 리스트에 N일 연속 등장 종목 추적.

    외인/기관 TOP 매수 리스트에 연속 등장 = 지속적 관심 = 추세 확인.
    5일 연속이면 강한 확신, 3일이면 추세 확인, 2일이면 초기 관심.
    """
    if not adv_cfg.get("consecutive", {}).get("enabled", False):
        return {}

    # 오늘 TOP 매수 종목
    today_buys = set()
    for item in flow.get("foreign_top_buy", []):
        today_buys.add(item["ticker"])
    for item in flow.get("institution_top_buy", []):
        today_buys.add(item["ticker"])

    if not today_buys or len(history) < 2:
        return {}

    # 오늘을 제외한 히스토리 역순으로 연속성 체크
    # history[-1]이 오늘, [-2]가 어제, ... 순서
    consecutive: dict[str, int] = {}
    for ticker in today_buys:
        count = 1  # 오늘 이미 1일
        for past_flow in reversed(history[:-1]):  # 어제부터 역순
            past_buys = set()
            for item in past_flow.get("foreign_top_buy", []):
                past_buys.add(item["ticker"])
            for item in past_flow.get("institution_top_buy", []):
                past_buys.add(item["ticker"])
            if ticker in past_buys:
                count += 1
            else:
                break
        if count >= 2:
            consecutive[ticker] = count

    return consecutive


def detect_rotation(
    market: dict,
    adv_cfg: dict,
) -> str | None:
    """전략4: 시장 로테이션 — KOSPI↔KOSDAQ 외인 수급 발산 감지.

    KOSPI 외인 대량 매도 + KOSDAQ 외인 순매수 = 소형주 로테이션.
    대형주 회피 속 코스닥 관심 증가는 중소형주 강세 시그널.
    """
    if not adv_cfg.get("rotation", {}).get("enabled", False):
        return None

    threshold = adv_cfg["rotation"].get("divergence_threshold", 10000)
    kospi_f = market.get("kospi", {}).get("foreign", 0)
    kosdaq_f = market.get("kosdaq", {}).get("foreign", 0)

    if kospi_f <= -threshold and kosdaq_f >= 0:
        return "KOSDAQ_FAVOR"
    elif kosdaq_f <= -threshold and kospi_f >= 0:
        return "KOSPI_FAVOR"
    return None


def detect_pension_signal(
    market: dict,
    adv_cfg: dict,
) -> tuple[str | None, float]:
    """전략5: 연기금 바닥 — 약세장에서 연기금 대량 매수 = 바닥 시그널.

    국민연금은 장기 가치투자 성향으로 하락 시 저가 매수.
    연기금 대규모 매수는 역사적으로 시장 바닥 근처에서 발생하는 경향.
    """
    if not adv_cfg.get("pension", {}).get("enabled", False):
        return None, 0

    threshold = adv_cfg["pension"].get("buy_threshold", 500)
    kospi_pension = market.get("kospi", {}).get("pension", 0)
    kosdaq_pension = market.get("kosdaq", {}).get("pension", 0)
    total_pension = kospi_pension + kosdaq_pension

    if total_pension >= threshold:
        return "PENSION_BUY", total_pension
    elif total_pension <= -threshold:
        return "PENSION_SELL", total_pension
    return None, total_pension


def build_kosdaq_ticker_set(flow: dict) -> set[str]:
    """investor_flow.json TOP 리스트에서 코스닥 종목 판별 불가 →
    pykrx로 코스닥 전 종목 SET를 캐시하여 반환.

    실패 시 빈 set (로테이션 전략 비활성화됨).
    """
    try:
        from pykrx import stock as pykrx_stock
        from datetime import datetime, timedelta
        # 최근 거래일 기준
        for offset in range(5):
            dt = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
            tickers = pykrx_stock.get_market_ticker_list(dt, market="KOSDAQ")
            if tickers:
                return set(tickers)
    except Exception as e:
        logger.debug("pykrx 코스닥 목록 조회 실패: %s", e)
    return set()


def compute_advanced_signals(
    flow: dict,
    market: dict,
    market_signal_types: list[str],
    cross_divergence: list[str],
    history: list[dict],
    adv_cfg: dict,
) -> dict:
    """고급 전략 5종 통합 산출."""
    # 전략1: 역행 매수
    contrarian = detect_contrarian(flow, market_signal_types, adv_cfg)

    # 전략2: 기관 흡수
    absorption = detect_absorption(flow, cross_divergence, adv_cfg)

    # 전략3: 수급 연속성
    consecutive = detect_consecutive_flow(flow, history, adv_cfg)

    # 전략4: 코스닥 로테이션
    rotation_signal = detect_rotation(market, adv_cfg)

    # 전략5: 연기금 바닥
    pension_signal, pension_total = detect_pension_signal(market, adv_cfg)

    # 로테이션용 코스닥 종목 SET (필요 시)
    kosdaq_tickers = []
    if rotation_signal:
        _kd_set = build_kosdaq_ticker_set(flow)
        if _kd_set:
            kosdaq_tickers = sorted(_kd_set)

    return {
        "contrarian_tickers": contrarian,
        "absorption_details": absorption,
        "consecutive_flow": consecutive,
        "rotation_signal": rotation_signal,
        "kosdaq_tickers": kosdaq_tickers,
        "pension_signal": pension_signal,
        "pension_total": pension_total,
    }


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════

def main():
    config = load_config()
    if_cfg = config.get("investor_flow_pykrx", {})

    if not if_cfg.get("enabled", False):
        print("[투자자 수급] 비활성 (investor_flow_pykrx.enabled=false) — 스킵")
        return

    flow_dir = if_cfg.get("flow_dir")

    # 1) 최신 investor_flow 로드
    flow = load_investor_flow(flow_dir=flow_dir)
    if not flow:
        # fallback: daily_intelligence.json
        intel_summary = load_investor_flow_summary_from_intel()
        if intel_summary:
            logger.info("investor_flow 파일 없음 → daily_intelligence fallback 사용")
            flow = intel_summary
        else:
            print("[투자자 수급] 데이터 없음 — 스킵")
            return

    date_str = flow.get("date", "unknown")
    source = flow.get("source", "unknown")
    f_total = flow.get("foreign", 0)
    i_total = flow.get("institution", 0)
    ind_total = flow.get("individual", 0)

    print(f"[투자자 수급] {date_str} | source={source}")
    print(f"  외인: {f_total:+,.0f}억 | 기관: {i_total:+,.0f}억 | 개인: {ind_total:+,.0f}억")

    # KOSPI/KOSDAQ 분리
    kospi = flow.get("kospi", {})
    kosdaq = flow.get("kosdaq", {})
    if kospi:
        print(f"  KOSPI — 외인: {kospi.get('foreign', 0):+,.0f}억 | "
              f"기관: {kospi.get('institution', 0):+,.0f}억")
    if kosdaq:
        print(f"  KOSDAQ — 외인: {kosdaq.get('foreign', 0):+,.0f}억 | "
              f"기관: {kosdaq.get('institution', 0):+,.0f}억")

    # 2) 히스토리 로드 (연속 매수/매도 판정용)
    history = load_investor_flow_history(
        days=if_cfg.get("lookback_days", 5),
        flow_dir=flow_dir,
    )

    # 3) 시장 시그널
    market_signals = detect_market_signals(flow, history, if_cfg)
    market_mult = calc_market_position_mult(f_total, if_cfg)

    if market_signals:
        for sig in market_signals:
            print(f"  ★ {sig['type']}: {sig['msg']}")
    print(f"  포지션 멀티플라이어: ×{market_mult}")

    # 4) 종목 시그널 + 수급 점수
    has_top = bool(flow.get("foreign_top_buy") or flow.get("foreign_top_sell"))
    stock_scores, mega_buy, mega_sell, stock_signals = {}, [], [], []
    if has_top:
        stock_scores, mega_buy, mega_sell, stock_signals = \
            compute_stock_signals(flow, if_cfg)

    # 5) 크로스 분석
    cross_div = detect_cross_divergence(flow) if has_top else []
    critical_risk = detect_critical_risk(mega_sell) if mega_sell else []

    # 5.5) 고급 전략 5종
    adv_cfg = if_cfg.get("advanced", {})
    market_signal_types = [s["type"] for s in market_signals]
    market_dict = {
        "foreign": f_total,
        "institution": i_total,
        "individual": ind_total,
        "kospi": kospi,
        "kosdaq": kosdaq,
    }
    advanced = compute_advanced_signals(
        flow, market_dict, market_signal_types, cross_div, history, adv_cfg,
    )

    # 6) 출력 빌드
    all_signals = [s["type"] for s in market_signals] + [s["type"] for s in stock_signals]
    output = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "market": {
            "foreign": f_total,
            "institution": i_total,
            "individual": ind_total,
            "kospi": kospi,
            "kosdaq": kosdaq,
        },
        "market_signals": [s["type"] for s in market_signals],
        "market_signals_detail": market_signals,
        "market_position_mult": market_mult,
        "stock_scores": stock_scores,
        "mega_buy_tickers": mega_buy,
        "mega_sell_tickers": mega_sell,
        "cross_divergence": cross_div,
        "critical_risk": critical_risk,
        "signal_count": len(all_signals),
        "top_foreign_buy": flow.get("foreign_top_buy", [])[:10],
        "top_foreign_sell": flow.get("foreign_top_sell", [])[:10],
        "top_inst_buy": flow.get("institution_top_buy", [])[:10],
        "top_inst_sell": flow.get("institution_top_sell", [])[:10],
        "advanced": advanced,
    }

    # 저장 (atomic write — scan_tomorrow_picks가 동시 읽을 때 부분 기록 차단)
    atomic_write_json(OUTPUT_PATH, output)
    logger.info("저장: %s (%d bytes)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size)

    # 요약 출력
    print(f"\n{'━'*60}")
    print(f"  투자자 수급 통합본 시그널 결과")
    print(f"{'━'*60}")
    print(f"  날짜: {date_str} | 소스: {source}")
    print(f"  시장 시그널: {[s['type'] for s in market_signals] or '없음'}")
    print(f"  포지션 멀티플라이어: ×{market_mult}")
    if has_top:
        print(f"  종목 스코어: {len(stock_scores)}종목")
        print(f"  외인 MEGA 매수: {len(mega_buy)}종목 {mega_buy[:5]}")
        print(f"  외인 MEGA 매도: {len(mega_sell)}종목 {mega_sell[:5]}")
        print(f"  크로스(외인매도+기관매수): {len(cross_div)}종목 {cross_div[:5]}")
        if critical_risk:
            print(f"  ⚠ CRITICAL (공매도+외인대매도): {critical_risk}")
    else:
        print(f"  [!] TOP 종목 리스트 없음 (pykrx 미수집일) — 시장 시그널만 활성")

    # 고급 전략 요약
    _adv_items = []
    if advanced.get("contrarian_tickers"):
        ct_names = [x["name"] for x in advanced["contrarian_tickers"][:3]]
        _adv_items.append(f"역행매수:{len(advanced['contrarian_tickers'])}종목({','.join(ct_names)})")
    if advanced.get("absorption_details"):
        ab_full = sum(1 for x in advanced["absorption_details"] if x["grade"] == "FULL")
        _adv_items.append(f"기관흡수:{len(advanced['absorption_details'])}종목(FULL:{ab_full})")
    if advanced.get("consecutive_flow"):
        cc = advanced["consecutive_flow"]
        _adv_items.append(f"연속매수:{len(cc)}종목(max:{max(cc.values())}일)")
    if advanced.get("rotation_signal"):
        _adv_items.append(f"로테이션:{advanced['rotation_signal']}")
    if advanced.get("pension_signal"):
        _adv_items.append(f"연기금:{advanced['pension_signal']}({advanced['pension_total']:+,.0f}억)")
    if _adv_items:
        print(f"  [고급] {' | '.join(_adv_items)}")
    else:
        print(f"  [고급] 발동 전략 없음")
    print(f"{'━'*60}")


if __name__ == "__main__":
    main()
