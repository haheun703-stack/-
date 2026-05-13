"""투자자 수급 통합본 시그널 스캐너 (pykrx KRX + KIS)

정보봇이 수집한 investor_flow.json에서 6종 시그널을 판정하고
종목별 수급 점수를 산출합니다.

데이터 소스: D:/Global_Stock_Overview_Scripter_정보봇/data/supply_daily/{date}_investor_flow.json

6 시그널:
  시장 — FOREIGN_MASS_SELL/BUY, INST_BUYING_HEAVY/SELLING_HEAVY
  종목 — FOREIGN_MEGA_BUY/SELL

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

    # 전일 데이터 (history의 마지막-1이 전일)
    f_prev = None
    if len(history) >= 2:
        prev = history[-2]
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

        # 수급 점수 (외인)
        if net_억 >= 1000:
            f_score = 100
        elif net_억 >= 500:
            f_score = 70
        elif net_억 >= 100:
            f_score = 40
        else:
            f_score = 20

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
        if net_억 <= -1000:
            f_score = -100
        elif net_억 <= -500:
            f_score = -70
        elif net_억 <= -100:
            f_score = -40
        else:
            f_score = -20

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

        if net_억 >= 1000:
            i_score = 100
        elif net_억 >= 500:
            i_score = 70
        elif net_억 >= 100:
            i_score = 40
        else:
            i_score = 20

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

        if net_억 <= -1000:
            i_score = -100
        elif net_억 <= -500:
            i_score = -70
        elif net_억 <= -100:
            i_score = -40
        else:
            i_score = -20

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
    }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
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
    print(f"{'━'*60}")


if __name__ == "__main__":
    main()
