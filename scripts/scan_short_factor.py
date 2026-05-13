"""공매도 3종 통합 8시그널 + 4팩터 스캐너

정보봇(KIS API) 공매도·신용·대차 데이터를 기반으로:
  - 8개 시그널 판정
  - 4개 팩터 계산 (Short Cover / Credit Risk / Inst Pressure / Divergence)
  - universe.csv 대상 일괄 처리

데이터 소스: D:/Global_Stock_Overview_Scripter_정보봇/data/supply_tracker/{ticker}.csv
출력: data/short_selling/jgis_short_factor.json

Usage:
  python -u -X utf8 scripts/scan_short_factor.py
  python -u -X utf8 scripts/scan_short_factor.py --test     # 10종목 테스트
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.adapters.jgis_short_adapter import JgisShortAdapter, _safe_float, _safe_int
from src.utils.atomic_io import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "short_selling"
OUTPUT_PATH = OUTPUT_DIR / "jgis_short_factor.json"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# 시그널 우선순위 (낮을수록 높은 우선순위)
SIGNAL_PRIORITY = {
    "SHORT_CREDIT_DIVERGE": 0,
    "SHORT_EXTREME": 1,
    "SHORT_COVER_RALLY": 2,
    "LOAN_SURGE": 3,
    "CREDIT_OVERHEAT": 4,
    "SHORT_SURGE": 5,
    "LOAN_MOMENTUM": 6,
    "SHORT_BALANCE_HIGH": 7,
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_universe() -> dict[str, str]:
    """universe.csv → {ticker: name}."""
    if not UNIVERSE_PATH.exists():
        logger.warning("universe.csv 없음")
        return {}
    result = {}
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["ticker"]] = row.get("name", row["ticker"])
    return result


# ═══════════════════════════════════════════════════
# 8 시그널 판정
# ═══════════════════════════════════════════════════
def compute_signals(rows: list[dict], cfg: dict) -> dict[str, bool]:
    """종목 CSV 데이터(최근 N일) → 8 시그널 판정.

    Args:
        rows: JgisShortAdapter.load_ticker_csv() 결과 (시간순 정렬)
        cfg: jgis_short_selling.signals 설정
    """
    signals = {
        "SHORT_EXTREME": False,
        "SHORT_SURGE": False,
        "SHORT_BALANCE_HIGH": False,
        "CREDIT_OVERHEAT": False,
        "LOAN_SURGE": False,
        "SHORT_COVER_RALLY": False,
        "LOAN_MOMENTUM": False,
        "SHORT_CREDIT_DIVERGE": False,
    }

    if len(rows) < 1:
        return signals

    latest = rows[-1]
    prev = rows[-2] if len(rows) >= 2 else {}

    # 최신 값
    short_qty = _safe_int(latest.get("short_selling_qty"))
    price = _safe_float(latest.get("price"))
    credit_rate = _safe_float(latest.get("credit_balance_rate"))
    loan_bal = _safe_int(latest.get("loan_balance_qty"))
    loan_new = _safe_int(latest.get("loan_new_qty"))
    loan_repay = _safe_int(latest.get("loan_repay_qty"))
    loan_bal_rate = _safe_float(latest.get("loan_balance_rate"))

    # 전일 값
    prev_loan_bal = _safe_int(prev.get("loan_balance_qty"))
    prev_short_qty = _safe_int(prev.get("short_selling_qty"))

    # short_ratio: short_overheat 컬럼이 있으면 비중, 아니면 추정
    # CSV에는 short_selling_qty만 있고 total_volume이 없음
    # loan_balance_rate를 공매도 잔고비율로 사용
    short_ratio = loan_bal_rate  # 대차잔고비율을 공매도 잔고율 프록시로 사용

    # 1. SHORT_EXTREME: 공매도 거래비중 ≥ 20%
    # (CSV에 거래비중 직접 컬럼 없음 → daily_intelligence.json의 short_ratio 사용 시 적용)
    # 여기서는 short_overheat 플래그로 대체
    if _safe_int(latest.get("short_overheat")):
        signals["SHORT_EXTREME"] = True

    # 2. SHORT_SURGE: 전일대비 공매도 급증
    if prev_short_qty > 0 and short_qty > 0:
        surge_ratio = (short_qty - prev_short_qty) / max(prev_short_qty, 1)
        if surge_ratio >= cfg.get("short_surge_delta", 5.0) / 100:
            # 비중 기준이 아닌 거래량 비율 5배 이상으로 판정
            pass  # CSV에 비중(%p) 직접 비교 어려움 → JSON fallback에서 처리

    # 3. SHORT_BALANCE_HIGH: 공매도 잔고비율 ≥ 3%
    if loan_bal_rate >= cfg.get("short_balance_high", 3.0):
        signals["SHORT_BALANCE_HIGH"] = True

    # 4. CREDIT_OVERHEAT: 신용잔고율 ≥ 5%
    if credit_rate >= cfg.get("credit_overheat", 5.0):
        signals["CREDIT_OVERHEAT"] = True

    # 5. LOAN_SURGE: 대차잔고 전일대비 +20%
    if prev_loan_bal > 0 and loan_bal > 0:
        loan_chg = (loan_bal - prev_loan_bal) / prev_loan_bal * 100
        if loan_chg >= cfg.get("loan_surge_pct", 20.0):
            signals["LOAN_SURGE"] = True

    # 6. SHORT_COVER_RALLY: 대차상환 > 신규 × 3
    if loan_new > 0 and loan_repay > loan_new * cfg.get("cover_ratio_threshold", 3.0):
        signals["SHORT_COVER_RALLY"] = True

    # 7. LOAN_MOMENTUM: 대차잔고 5일 변화 ±15%+
    if len(rows) >= 6:
        loan_5d_ago = _safe_int(rows[-6].get("loan_balance_qty"))
        if loan_5d_ago > 0 and loan_bal > 0:
            mom_5d = (loan_bal - loan_5d_ago) / loan_5d_ago * 100
            if abs(mom_5d) >= cfg.get("loan_momentum_5d_pct", 15.0):
                signals["LOAN_MOMENTUM"] = True

    # 8. SHORT_CREDIT_DIVERGE: 공매도 잔고 높음 & 신용 높음
    diverge_short = cfg.get("diverge_short_min", 10.0)
    diverge_credit = cfg.get("diverge_credit_min", 3.0)
    # 공매도 잔고비율 >= diverge_short (% 단위가 다를 수 있음)
    # loan_balance_rate는 보통 0.01~5% 범위 → 1% 이상이면 높은 편
    # 실용적으로: 대차잔고 비율 >= 1% AND 신용잔고율 >= 3%
    if loan_bal_rate >= 1.0 and credit_rate >= diverge_credit:
        signals["SHORT_CREDIT_DIVERGE"] = True

    return signals


# ═══════════════════════════════════════════════════
# JSON 기반 시그널 보강 (daily_intelligence.json TOP 30)
# ═══════════════════════════════════════════════════
def enrich_from_jgis_json(
    ticker: str,
    signals: dict[str, bool],
    jgis_data: dict[str, dict],
    cfg: dict,
) -> tuple[dict[str, bool], dict]:
    """JSON의 short_selling_summary로 시그널 보강 + short_ratio 등 원시데이터 추출."""
    extra = {}
    row = jgis_data.get(ticker)
    if not row:
        return signals, extra

    sr = _safe_float(row.get("short_ratio"))
    cr = _safe_float(row.get("credit_balance_rate"))
    cover = _safe_float(row.get("cover_ratio"))
    mom5 = _safe_float(row.get("loan_momentum_5d"))
    sr_delta = _safe_float(row.get("short_ratio_delta"))

    extra = {
        "json_short_ratio": sr,
        "json_credit_rate": cr,
        "json_cover_ratio": cover,
        "json_loan_momentum_5d": mom5,
    }

    # SHORT_EXTREME: short_ratio >= 20%
    if sr >= cfg.get("short_extreme_threshold", 20.0):
        signals["SHORT_EXTREME"] = True

    # SHORT_SURGE: 전일대비 +5%p
    if sr_delta >= cfg.get("short_surge_delta", 5.0):
        signals["SHORT_SURGE"] = True

    # SHORT_CREDIT_DIVERGE: short_ratio >= 10 AND credit >= 3
    if sr >= cfg.get("diverge_short_min", 10.0) and cr >= cfg.get("diverge_credit_min", 3.0):
        signals["SHORT_CREDIT_DIVERGE"] = True

    return signals, extra


# ═══════════════════════════════════════════════════
# 4 팩터 계산
# ═══════════════════════════════════════════════════
def compute_factors(rows: list[dict], signals: dict, cfg: dict) -> dict[str, float]:
    """4 팩터 계산 (각 0~100).

    Factor 1: Short Cover Factor (높을수록 숏커버 랠리 가능성)
    Factor 2: Credit Risk Factor (높을수록 반대매매 리스크)
    Factor 3: Inst Short Pressure (높을수록 기관 숏 압력)
    Factor 4: Divergence Factor (공매도+신용 동시 과열)
    """
    factors = {
        "short_cover_factor": 0.0,
        "credit_risk_factor": 0.0,
        "inst_short_pressure": 0.0,
        "divergence_factor": 0.0,
    }

    if not rows:
        return factors

    latest = rows[-1]
    credit_rate = _safe_float(latest.get("credit_balance_rate"))
    loan_bal = _safe_int(latest.get("loan_balance_qty"))
    loan_new = _safe_int(latest.get("loan_new_qty"))
    loan_repay = _safe_int(latest.get("loan_repay_qty"))
    loan_bal_rate = _safe_float(latest.get("loan_balance_rate"))
    change_pct = _safe_float(latest.get("change_pct"))
    short_qty = _safe_int(latest.get("short_selling_qty"))

    # ── Factor 1: Short Cover Factor ──
    # cover_ratio, loan_trend (5일), short_ratio 기반
    score_sc = 0.0
    cover_ratio = loan_repay / max(loan_new, 1) if loan_new > 0 or loan_repay > 0 else 0

    if cover_ratio >= 3:
        score_sc += 40
    elif cover_ratio >= 2:
        score_sc += 25
    elif cover_ratio >= 1.5:
        score_sc += 15

    # 5일 대차잔고 추세 (감소일수록 높은 점수)
    if len(rows) >= 6:
        loan_5d_ago = _safe_int(rows[-6].get("loan_balance_qty"))
        if loan_5d_ago > 0 and loan_bal > 0:
            loan_trend = (loan_bal - loan_5d_ago) / loan_5d_ago * 100
            if loan_trend <= -15:
                score_sc += 30
            elif loan_trend <= -5:
                score_sc += 15
            elif loan_trend <= 0:
                score_sc += 5

    # 공매도/대차 잔고 높을수록 숏커버 잠재력
    if loan_bal_rate >= 2.0:
        score_sc += 30
    elif loan_bal_rate >= 1.0:
        score_sc += 15
    elif loan_bal_rate >= 0.5:
        score_sc += 5

    factors["short_cover_factor"] = min(max(score_sc, 0), 100)

    # ── Factor 2: Credit Risk Factor ──
    # 신용잔고율 기반 위험도
    if credit_rate >= 7:
        risk = 100
    elif credit_rate >= 5:
        risk = 70
    elif credit_rate >= 3:
        risk = 40
    elif credit_rate >= 1:
        risk = 15
    else:
        risk = 0

    # 주가 하락 중 + 신용과열 → 위험 증폭
    if change_pct < -3 and credit_rate >= 5:
        risk = min(risk + 30, 100)
    elif change_pct < -2 and credit_rate >= 3:
        risk = min(risk + 15, 100)

    factors["credit_risk_factor"] = min(max(risk, 0), 100)

    # ── Factor 3: Institutional Short Pressure ──
    # 5일 대차잔고 모멘텀 + 당일 대차 변화 + 공매도 거래량
    pressure = 0.0

    if len(rows) >= 6:
        loan_5d_ago = _safe_int(rows[-6].get("loan_balance_qty"))
        if loan_5d_ago > 0 and loan_bal > 0:
            mom_5d = (loan_bal - loan_5d_ago) / loan_5d_ago * 100
            if mom_5d >= 30:
                pressure += 50
            elif mom_5d >= 15:
                pressure += 30
            elif mom_5d >= 5:
                pressure += 10

    # 당일 대차 순증가 (신규 > 상환)
    if loan_new > 0 and loan_repay > 0:
        daily_net = loan_new - loan_repay
        if daily_net > 0 and loan_bal > 0:
            daily_pct = daily_net / loan_bal * 100
            if daily_pct >= 5:
                pressure += 30
            elif daily_pct >= 2:
                pressure += 15

    # 대차잔고 비율 높음
    if loan_bal_rate >= 3:
        pressure += 20
    elif loan_bal_rate >= 1.5:
        pressure += 10

    factors["inst_short_pressure"] = min(max(pressure, 0), 100)

    # ── Factor 4: Divergence Factor ──
    # 공매도(대차) + 신용 동시 과열
    if loan_bal_rate >= 1.0 and credit_rate >= 3:
        div_score = (min(loan_bal_rate, 5) / 5 + min(credit_rate, 10) / 10) * 50
        factors["divergence_factor"] = min(max(div_score, 0), 100)
    elif loan_bal_rate >= 0.5 and credit_rate >= 2:
        factors["divergence_factor"] = 20.0

    return factors


# ═══════════════════════════════════════════════════
# 메인 스캔 로직
# ═══════════════════════════════════════════════════
def scan_all(
    adapter: JgisShortAdapter,
    universe: dict[str, str],
    cfg: dict,
    jgis_top30: dict[str, dict],
) -> list[dict]:
    """유니버스 전체 스캔 → 결과 리스트."""

    sig_cfg = cfg.get("jgis_short_selling", {}).get("signals", {})
    results = []

    # 배치 로드
    tickers = list(universe.keys())
    logger.info("배치 로드 시작: %d종목", len(tickers))
    batch = adapter.load_batch(tickers, max_workers=8)
    logger.info("배치 로드 완료: %d종목 (데이터 있음)", len(batch))

    for ticker, name in universe.items():
        rows = batch.get(ticker, [])
        if not rows:
            continue

        latest = rows[-1]

        # 유효 데이터 체크: 최소한 price가 있어야 함
        if _safe_float(latest.get("price")) <= 0:
            continue

        # 8 시그널 판정
        sigs = compute_signals(rows, sig_cfg)

        # JSON TOP 30 보강
        extra = {}
        if ticker in jgis_top30:
            sigs, extra = enrich_from_jgis_json(ticker, sigs, jgis_top30, sig_cfg)

        # 4 팩터 계산
        factors = compute_factors(rows, sigs, sig_cfg)

        # 활성 시그널 리스트 (우선순위순)
        active = sorted(
            [s for s, v in sigs.items() if v],
            key=lambda s: SIGNAL_PRIORITY.get(s, 99),
        )

        # 원시 데이터 추출
        loan_bal = _safe_int(latest.get("loan_balance_qty"))
        loan_new = _safe_int(latest.get("loan_new_qty"))
        loan_repay = _safe_int(latest.get("loan_repay_qty"))
        prev_loan = _safe_int(rows[-2].get("loan_balance_qty")) if len(rows) >= 2 else 0

        loan_change_pct = 0.0
        if prev_loan > 0 and loan_bal > 0:
            loan_change_pct = (loan_bal - prev_loan) / prev_loan * 100

        loan_mom_5d = 0.0
        if len(rows) >= 6:
            loan_5d_ago = _safe_int(rows[-6].get("loan_balance_qty"))
            if loan_5d_ago > 0 and loan_bal > 0:
                loan_mom_5d = (loan_bal - loan_5d_ago) / loan_5d_ago * 100

        cover_ratio = loan_repay / max(loan_new, 1) if loan_new > 0 or loan_repay > 0 else 0

        result = {
            "ticker": ticker,
            "date": latest.get("date", ""),
            "name": name,
            "active_signals": active,
            "short_cover_factor": round(factors["short_cover_factor"], 1),
            "credit_risk_factor": round(factors["credit_risk_factor"], 1),
            "inst_short_pressure": round(factors["inst_short_pressure"], 1),
            "divergence_factor": round(factors["divergence_factor"], 1),
            "short_ratio": round(extra.get("json_short_ratio", _safe_float(latest.get("loan_balance_rate"))), 2),
            "credit_balance_rate": round(_safe_float(latest.get("credit_balance_rate")), 2),
            "loan_balance_qty": loan_bal,
            "loan_new_qty": loan_new,
            "loan_repay_qty": loan_repay,
            "cover_ratio": round(cover_ratio, 2),
            "loan_change_pct": round(loan_change_pct, 2),
            "loan_momentum_5d": round(loan_mom_5d, 2),
            "change_pct": round(_safe_float(latest.get("change_pct")), 2),
            "price": _safe_float(latest.get("price")),
        }
        results.append(result)

    return results


def build_output(results: list[dict]) -> dict:
    """스캔 결과 → JSON 출력 구조 생성."""

    # 시그널 요약
    sig_summary: dict[str, int] = {}
    for r in results:
        for s in r["active_signals"]:
            sig_summary[s] = sig_summary.get(s, 0) + 1

    # 팩터 요약
    credit_high = [r for r in results if r["credit_risk_factor"] >= 70]
    cover_high = [r for r in results if r["short_cover_factor"] >= 70]
    pressure_high = [r for r in results if r["inst_short_pressure"] >= 70]
    diverge_high = [r for r in results if r["divergence_factor"] >= 50]

    # 시그널 2개 이상 종목 (위험도 순)
    multi_signal = [r for r in results if len(r["active_signals"]) >= 2]
    multi_signal.sort(
        key=lambda r: min(SIGNAL_PRIORITY.get(s, 99) for s in r["active_signals"]),
    )

    # 숏커버 TOP (매수 기회)
    cover_top = sorted(
        [r for r in results if r["short_cover_factor"] > 0],
        key=lambda r: -r["short_cover_factor"],
    )[:15]

    # Credit Risk 킬 대상
    credit_kills = sorted(credit_high, key=lambda r: -r["credit_risk_factor"])

    # all_results dict
    all_results = {r["ticker"]: r for r in results}

    today = results[0]["date"] if results else datetime.now().strftime("%Y-%m-%d")

    return {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "total_scanned": len(results),
        "signal_summary": sig_summary,
        "factor_summary": {
            "credit_risk_high": len(credit_high),
            "short_cover_high": len(cover_high),
            "inst_pressure_high": len(pressure_high),
            "divergence_high": len(diverge_high),
        },
        "top_signals": [
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "signals": r["active_signals"],
                "credit_risk": r["credit_risk_factor"],
                "short_cover": r["short_cover_factor"],
                "price": r["price"],
            }
            for r in multi_signal[:20]
        ],
        "top_short_cover": [
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "short_cover_factor": r["short_cover_factor"],
                "cover_ratio": r["cover_ratio"],
                "loan_momentum_5d": r["loan_momentum_5d"],
                "price": r["price"],
            }
            for r in cover_top
        ],
        "credit_risk_kills": [
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "credit_risk_factor": r["credit_risk_factor"],
                "credit_balance_rate": r["credit_balance_rate"],
                "change_pct": r["change_pct"],
                "price": r["price"],
            }
            for r in credit_kills[:15]
        ],
        "all_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="공매도 3종 8시그널 + 4팩터 스캐너")
    parser.add_argument("--test", action="store_true", help="10종목 테스트 모드")
    args = parser.parse_args()

    config = load_config()
    jgis_cfg = config.get("jgis_short_selling", {})

    if not jgis_cfg.get("enabled", False):
        logger.info("jgis_short_selling.enabled=false → 스킵")
        return

    adapter = JgisShortAdapter(config)

    if not adapter.csv_dir.exists():
        logger.warning("CSV 디렉토리 없음: %s → 스킵", adapter.csv_dir)
        return

    # 유니버스 로드
    universe = load_universe()
    if args.test:
        universe = dict(list(universe.items())[:10])
    logger.info("유니버스: %d종목", len(universe))

    # JSON TOP 30 로드 (보강용)
    jgis_top30 = adapter.load_jgis_top30()
    if jgis_top30:
        logger.info("JGIS TOP 30: %d종목 (JSON 보강)", len(jgis_top30))

    # 스캔 실행
    results = scan_all(adapter, universe, config, jgis_top30)
    logger.info("스캔 완료: %d종목 (유효 데이터)", len(results))

    if not results:
        logger.warning("유효 결과 없음 → 종료")
        return

    # 출력 생성 + 저장 (atomic write — scan_tomorrow_picks 동시 읽기 안전)
    output = build_output(results)
    atomic_write_json(OUTPUT_PATH, output)
    logger.info("저장: %s (%d bytes)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size)

    # 결과 출력
    sig = output["signal_summary"]
    fac = output["factor_summary"]
    print(f"\n{'━'*60}")
    print(f"  공매도 3종 8시그널 + 4팩터 결과")
    print(f"{'━'*60}")
    print(f"  스캔: {output['total_scanned']}종목 | 날짜: {output['date']}")
    print(f"\n  [시그널 요약]")
    for s_name in sorted(sig.keys(), key=lambda x: SIGNAL_PRIORITY.get(x, 99)):
        print(f"    {s_name:30s}: {sig[s_name]:3d}종목")

    print(f"\n  [팩터 요약]")
    print(f"    Credit Risk ≥70 (킬게이트):  {fac['credit_risk_high']:3d}종목")
    print(f"    Short Cover ≥70 (숏커버기회): {fac['short_cover_high']:3d}종목")
    print(f"    Inst Pressure ≥70 (숏압력):   {fac['inst_pressure_high']:3d}종목")
    print(f"    Divergence ≥50 (동시과열):    {fac['divergence_high']:3d}종목")

    # TOP 시그널 종목
    top = output["top_signals"][:10]
    if top:
        print(f"\n  [다중 시그널 종목 TOP 10]")
        for r in top:
            sigs_str = "+".join(r["signals"])
            print(f"    {r['name']:12s}({r['ticker']}) — {sigs_str}")
            print(f"      CR={r['credit_risk']:.0f} SC={r['short_cover']:.0f}")

    # 숏커버 기회 TOP 5
    cover = output["top_short_cover"][:5]
    if cover:
        print(f"\n  [숏커버 랠리 기회 TOP 5]")
        for r in cover:
            print(f"    {r['name']:12s}({r['ticker']}) — SC={r['short_cover_factor']:.0f} "
                  f"커버비={r['cover_ratio']:.1f} 5일mom={r['loan_momentum_5d']:+.1f}%")

    # Credit Risk 킬 대상
    kills = output["credit_risk_kills"][:5]
    if kills:
        print(f"\n  [Credit Risk 킬게이트 (≥70)]")
        for r in kills:
            print(f"    {r['name']:12s}({r['ticker']}) — CR={r['credit_risk_factor']:.0f} "
                  f"신용율={r['credit_balance_rate']:.1f}% 등락={r['change_pct']:+.1f}%")

    print(f"{'━'*60}")


if __name__ == "__main__":
    main()
