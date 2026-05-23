"""소부장 후보 풀 발굴 + STEP 5 ★★★ 필터 (퐝가님 5/23 흐름).

배경:
  본 흐름 (천장-3% / 분할매수 / 받침 재진입)은 중소형 소부장에서 가장 잘 작동.
  config/soubujang_universe.yaml 7섹터 × ~33종목 대상으로:
    1. KIS API 현재 시총 + 거래대금 fetch
    2. DART API 8분기 영업이익 + CAGR
    3. STEP 5 산식 적용 (step5_fetch_10stocks.py와 동일)
    4. 시총 1,000억~10조 + 거래대금 50억+ + STEP 5 ★★★ 필터
    5. 결과 → data/soubujang_pool/<date>.json + 표 출력

실행:
  source venv/Scripts/activate
  python -u -X utf8 scripts/step5_soubujang_pool.py

cron 권장 (5/24~5/25 등록 검토):
  0 18 * * 1-5   매일 18:00 자동 갱신
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = PROJECT_ROOT / "config" / "soubujang_universe.yaml"

# === 필터 임계 (.env 동적) ===
MIN_MARKET_CAP_EOK = int(os.getenv("SOUBUJANG_MIN_MCAP_EOK", "1000"))     # 1,000억
MAX_MARKET_CAP_EOK = int(os.getenv("SOUBUJANG_MAX_MCAP_EOK", "100000"))   # 10조
MIN_DAILY_AMOUNT_EOK = int(os.getenv("SOUBUJANG_MIN_AMOUNT_EOK", "50"))   # 일평균 50억
MIN_UPSIDE = float(os.getenv("SOUBUJANG_MIN_UPSIDE", "2.0"))               # STEP 5 ★★★ 이상


def load_universe() -> dict:
    """soubujang_universe.yaml 로드."""
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_broker():
    import mojito
    is_mock = os.getenv("MODEL") != "REAL"
    return mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=is_mock,
    )


def _safe_int(v, default=0):
    try:
        return int(str(v).replace(",", "")) if v else default
    except (ValueError, TypeError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(str(v).replace(",", "")) if v else default
    except (ValueError, TypeError):
        return default


def fetch_kis_basics(broker, ticker: str) -> dict:
    """현재가/시총/PER/EPS + 1년 OHLCV 거래대금 평균."""
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}

        # 1년 OHLCV
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=60)).strftime("%Y%m%d")
        ohlcv_res = broker.fetch_ohlcv(ticker, timeframe="D", start_day=start_day, end_day=end_day)
        rows = ohlcv_res.get("output2", []) if ohlcv_res else []

        # 20일 평균 거래대금
        amounts = []
        peaks = []  # 30일 최고가용
        for r in rows[:30]:
            try:
                close = float(r.get("stck_clpr", 0))
                vol = float(r.get("acml_vol", 0))
                amounts.append(close * vol)
                peaks.append(float(r.get("stck_hgpr", 0)))
            except (ValueError, TypeError):
                continue
        avg_amount_eok = sum(amounts[:20]) / 20 / 100_000_000 if len(amounts) >= 20 else 0
        peak_30d = max(peaks) if peaks else 0

        return {
            "current_price": _safe_int(output.get("stck_prpr")),
            "market_cap_eok": _safe_int(output.get("hts_avls")),
            "shares": _safe_int(output.get("lstn_stcn")),
            "per": _safe_float(output.get("per")),
            "eps": _safe_float(output.get("eps")),
            "sector": output.get("bstp_kor_isnm", ""),
            "avg_amount_20d_eok": round(avg_amount_eok, 1),
            "peak_30d": int(peak_30d),
        }
    except Exception as e:
        logger.warning("KIS %s 실패: %s", ticker, e)
        return {}


def fetch_dart_8q_batch(tickers: list[str]) -> dict:
    """DART 8분기 일괄."""
    try:
        from src.adapters.dart_financial_adapter import DartFinancialAdapter
        adapter = DartFinancialAdapter()
        if not adapter.is_available:
            return {}
        return adapter.collect_bs_all(tickers)
    except Exception as e:
        logger.error("DART fetch 실패: %s", e)
        return {}


def calculate_step5(kis: dict, dart: dict) -> dict:
    """STEP 5 산식 (step5_fetch_10stocks.py와 동일)."""
    out = {
        "current_market_cap_jo": None,
        "ttm_op_jo": None,
        "cagr_avg": None,
        "upside_ratio": None,
        "annual_return": None,
        "quality": "incomplete",
    }
    mcap_eok = kis.get("market_cap_eok", 0)
    if mcap_eok > 0:
        out["current_market_cap_jo"] = round(mcap_eok / 10_000, 2)

    if not dart:
        return out

    periods = sorted(dart.keys())
    if len(periods) < 5:
        return out

    try:
        op_cums = {p: (dart[p].get("op_income_cum", 0) or 0) for p in periods}
        latest_label = periods[-1]
        latest_op = op_cums[latest_label]

        # Q 식별
        q_num = 3
        for q_marker, q in (("Q1", 1), ("Q2", 2), ("Q3", 3), ("Q4", 4)):
            if q_marker in latest_label:
                q_num = q
                break

        # TTM 계산
        ttm_op = None
        if q_num < 4:
            yr = int(latest_label[:4])
            prev_full = f"{yr-1}Q4"
            prev_partial = f"{yr-1}Q{q_num}"
            if prev_full in op_cums and prev_partial in op_cums:
                ttm_op = op_cums[latest_label] + op_cums[prev_full] - op_cums[prev_partial]
        elif q_num == 4:
            ttm_op = latest_op

        if ttm_op and ttm_op > 0:
            out["ttm_op_jo"] = round(ttm_op / 1_000_000_000_000, 3)
        elif latest_op > 0:
            out["ttm_op_jo"] = round(latest_op * 4 / q_num / 1_000_000_000_000, 3)

        # CAGR 1y + 2y 평균
        cagr_1y = None
        cagr_2y = None
        if len(periods) >= 5 and op_cums.get(periods[-5], 0) > 0 and latest_op > 0:
            cagr_1y = (latest_op / op_cums[periods[-5]]) - 1
        if len(periods) >= 8 and op_cums.get(periods[-8], 0) > 0 and latest_op > 0:
            ratio_2y = latest_op / op_cums[periods[-8]]
            if ratio_2y > 0:
                cagr_2y = (ratio_2y ** 0.5) - 1

        cagrs = [c for c in (cagr_1y, cagr_2y) if c is not None]
        if cagrs:
            cagr_avg = sum(cagrs) / len(cagrs)
            cagr_avg = max(-0.3, min(0.5, cagr_avg))
            out["cagr_avg"] = round(cagr_avg * 100, 1)

        # 미래 시총 두 모델 평균
        per = kis.get("per", 0)
        if out["ttm_op_jo"] and out["cagr_avg"] is not None and out["current_market_cap_jo"]:
            cagr_dec = out["cagr_avg"] / 100
            upside_A = (1 + cagr_dec) ** 5
            future_op = out["ttm_op_jo"] * (1 + cagr_dec) ** 5
            if per > 0:
                future_mcap_B = future_op * per
                upside_B = future_mcap_B / out["current_market_cap_jo"]
                out["upside_ratio"] = round((upside_A + upside_B) / 2, 2)
            else:
                out["upside_ratio"] = round(upside_A, 2)
            if out["upside_ratio"] > 0:
                out["annual_return"] = round((out["upside_ratio"] ** 0.2 - 1) * 100, 1)

        out["quality"] = "complete" if out["upside_ratio"] else "partial"
    except Exception as e:
        out["error"] = str(e)

    return out


def grade_step5(upside: float | None) -> str:
    """STEP 5 등급."""
    if upside is None:
        return "N/A"
    if upside >= 5.0:
        return "★★★★★"
    if upside >= 3.5:
        return "★★★★"
    if upside >= 2.0:
        return "★★★"
    if upside >= 1.2:
        return "★★"
    if upside >= 0.8:
        return "★"
    return "❌"


def apply_filters(ticker: str, name: str, kis: dict, step5: dict, sector: str) -> tuple[bool, list[str]]:
    """5단계 필터 적용."""
    reasons_pass = []
    reasons_fail = []

    # 1. 시총
    mcap_eok = kis.get("market_cap_eok", 0)
    if MIN_MARKET_CAP_EOK <= mcap_eok <= MAX_MARKET_CAP_EOK:
        reasons_pass.append(f"시총 {mcap_eok:,}억 OK")
    else:
        reasons_fail.append(f"시총 {mcap_eok:,}억 ({MIN_MARKET_CAP_EOK}~{MAX_MARKET_CAP_EOK} 범위 외)")

    # 2. 거래대금
    amt = kis.get("avg_amount_20d_eok", 0)
    if amt >= MIN_DAILY_AMOUNT_EOK:
        reasons_pass.append(f"거래대금 {amt:.1f}억 OK")
    else:
        reasons_fail.append(f"거래대금 {amt:.1f}억 < {MIN_DAILY_AMOUNT_EOK}억")

    # 3. STEP 5 ★★★ 이상
    upside = step5.get("upside_ratio")
    if upside and upside >= MIN_UPSIDE:
        reasons_pass.append(f"STEP 5 {upside:.2f}x ({grade_step5(upside)})")
    else:
        u_str = f"{upside:.2f}x" if upside else "N/A"
        reasons_fail.append(f"STEP 5 {u_str} ({grade_step5(upside)})")

    return len(reasons_fail) == 0, reasons_pass + reasons_fail


def main():
    logger.info("=" * 70)
    logger.info("소부장 후보 풀 발굴 시작 (5/23 퐝가님 흐름)")
    logger.info("=" * 70)

    universe = load_universe()
    sectors_block = {k: v for k, v in universe.items() if not k.startswith("_") and k != "whitelist"}

    broker = get_broker()

    # 전체 ticker 수집
    all_tickers = []
    ticker_to_sector = {}
    for sector_key, sector_info in sectors_block.items():
        for item in sector_info.get("tickers", []):
            t = item["ticker"]
            all_tickers.append(t)
            ticker_to_sector[t] = (sector_key, sector_info["name"], item["name"], item.get("reason", ""))

    logger.info("총 후보: %d종목", len(all_tickers))

    # DART 8분기 일괄 fetch
    dart_all = fetch_dart_8q_batch(all_tickers)

    # 종목별 분석
    results = {}
    for ticker in all_tickers:
        sector_key, sector_name, name, reason = ticker_to_sector[ticker]
        logger.info("[%s %s]", ticker, name)
        kis = fetch_kis_basics(broker, ticker)
        if not kis:
            continue
        s5 = calculate_step5(kis, dart_all.get(ticker, {}))
        passed, reasons = apply_filters(ticker, name, kis, s5, sector_name)
        results[ticker] = {
            "name": name,
            "sector_key": sector_key,
            "sector_name": sector_name,
            "reason": reason,
            "kis": kis,
            "step5": s5,
            "passed": passed,
            "filter_reasons": reasons,
        }

    # 저장
    out_dir = PROJECT_ROOT / "data" / "soubujang_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"soubujang_pool_{today}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", out_file)

    # 콘솔 표
    print("\n" + "=" * 130)
    print(f"{'섹터':25} {'종목':16} {'시총(억)':>10} {'거래대금':>10} "
          f"{'PER':>7} {'CAGR%':>8} {'업사이드':>9} {'등급':6} {'결과':4}")
    print("=" * 130)

    # 섹터별 정렬
    sector_order = list(sectors_block.keys())
    sorted_tickers = sorted(results.keys(),
                            key=lambda t: (sector_order.index(results[t]["sector_key"]),
                                          -(results[t]["step5"].get("upside_ratio") or 0)))

    pass_count = 0
    for ticker in sorted_tickers:
        r = results[ticker]
        kis = r["kis"]
        s5 = r["step5"]
        upside = s5.get("upside_ratio")
        grade = grade_step5(upside)
        status = "✅PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            pass_count += 1

        print(
            f"{r['sector_name'][:25]:25} "
            f"{r['name'][:16]:16} "
            f"{kis.get('market_cap_eok', 0):>10,} "
            f"{kis.get('avg_amount_20d_eok', 0):>10,.1f} "
            f"{kis.get('per', 0):>7.1f} "
            f"{s5.get('cagr_avg') or '-':>8} "
            f"{upside or '-':>9} "
            f"{grade:6} "
            f"{status:4}"
        )
    print("=" * 130)
    print(f"\n✅ 통과: {pass_count}/{len(results)}종목 (STEP 5 ★★★ 이상 + 시총/거래대금)")

    # 통과 종목만 별도 출력
    print("\n" + "=" * 80)
    print("★ 적응형 포지션 매매법 후보 풀 (3중 검증 1차 통과)")
    print("=" * 80)
    print(f"{'종목':16} {'섹터':20} {'업사이드':>9} {'등급':6} {'30일 천장':>11} {'현재가':>10}")
    print("=" * 80)
    for ticker in sorted_tickers:
        r = results[ticker]
        if not r["passed"]:
            continue
        kis = r["kis"]
        s5 = r["step5"]
        print(
            f"{r['name'][:16]:16} "
            f"{r['sector_name'][:20]:20} "
            f"{s5.get('upside_ratio') or '-':>9} "
            f"{grade_step5(s5.get('upside_ratio')):6} "
            f"{kis.get('peak_30d', 0):>11,} "
            f"{kis.get('current_price', 0):>10,}"
        )
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
