"""STEP 5 모델 10종목 실측 데이터 fetch + 계산.

근거: 강방천 회장 STEP 5 모델 (에셋플러스자산운용)
지시: 2026-05-23 퐝가님 "추정 X, 직접 계산 + 숫자 증명"

소스 (정보봇 의존 X, 퀀트봇 자체 API):
  - KIS API: 현재가 + 시가총액 + PER + EPS + 발행주식수 + 1년 OHLCV
  - DART API: 8분기 영업이익/순이익/매출 (DartFinancialAdapter)
  - 피보나치: 1년 일봉 직접 계산 (0.382/0.5/0.618)

STEP 5 산식 (실측 기반):
  CAGR = (최근 4Q OP / 1년 전 4Q OP)^(1/1) - 1  (1년 기준)
  5년 후 OP = 최근 4Q OP × (1 + CAGR)^5
  미래 시총 = 5년 후 OP × 현재 PER (KIS 제공)
  업사이드 = 미래 시총 / 현재 시총

산식 단순화 의도:
  - PER 조정 X (KIS의 실제 PER을 그대로 적용 — 시장이 평가한 PER 신뢰)
  - CAGR 추정 X (DART 실측 추세 그대로 적용)

실행:
  source venv/Scripts/activate
  python -u -X utf8 scripts/step5_fetch_10stocks.py

출력:
  data/step5/step5_10stocks_<date>.json — 원본 데이터
  콘솔: 표 출력
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 10종목 (퐝가님 5/23 지정)
STOCKS = [
    ("006260", "LS"),
    ("066570", "LG전자"),
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("298040", "효성중공업"),
    ("009150", "삼성전기"),
    ("006400", "삼성SDI"),
    ("005380", "현대차"),
    ("277810", "레인보우로보틱스"),
    ("005490", "POSCO홀딩스"),
]


def get_broker():
    """KIS broker 인스턴스 (mock vs real)."""
    import mojito
    is_mock = os.getenv("MODEL") != "REAL"
    logger.info("KIS mode=%s (MODEL=%s)", "mock" if is_mock else "real", os.getenv("MODEL"))
    return mojito.KoreaInvestment(
        api_key=os.getenv("KIS_APP_KEY"),
        api_secret=os.getenv("KIS_APP_SECRET"),
        acc_no=os.getenv("KIS_ACC_NO"),
        mock=is_mock,
    )


def fetch_kis_basics(broker, ticker: str) -> dict:
    """KIS API로 현재가/시총/PER/EPS/주식수 조회."""
    try:
        res = broker.fetch_price(ticker)
        output = res.get("output", {}) if res else {}

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

        return {
            "current_price": _safe_int(output.get("stck_prpr")),
            "market_cap_eok": _safe_int(output.get("hts_avls")),  # 억원
            "shares": _safe_int(output.get("lstn_stcn")),
            "per": _safe_float(output.get("per")),
            "pbr": _safe_float(output.get("pbr")),
            "eps": _safe_float(output.get("eps")),
            "bps": _safe_float(output.get("bps")),
            "w52_high": _safe_int(output.get("w52_hgpr")),
            "w52_low": _safe_int(output.get("w52_lwpr")),
            "sector": output.get("bstp_kor_isnm", ""),
            "raw_output": output,  # 디버깅용
        }
    except Exception as e:
        logger.warning("KIS basics %s 실패: %s", ticker, e)
        return {}


def fetch_fibonacci(broker, ticker: str, days: int = 240) -> dict:
    """1년 일봉 → 피보나치 0.382/0.5/0.618 직접 계산."""
    try:
        end_day = date.today().strftime("%Y%m%d")
        start_day = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
        res = broker.fetch_ohlcv(ticker, timeframe="D", start_day=start_day, end_day=end_day)
        rows = res.get("output2", []) if res else []
        if not rows:
            logger.warning("OHLCV %s 빈 응답", ticker)
            return {}
        highs = []
        lows = []
        for r in rows:
            try:
                h = float(r.get("stck_hgpr", 0))
                l = float(r.get("stck_lwpr", 0))
                if h > 0 and l > 0:
                    highs.append(h)
                    lows.append(l)
            except (ValueError, TypeError):
                continue
        if not highs or not lows:
            return {}
        max_h = max(highs)
        min_l = min(lows)
        diff = max_h - min_l
        return {
            "high_1y": int(max_h),
            "low_1y": int(min_l),
            "range": int(diff),
            "fib_0_382": int(min_l + diff * 0.382),
            "fib_0_500": int(min_l + diff * 0.5),
            "fib_0_618": int(min_l + diff * 0.618),
            "samples": len(highs),
        }
    except Exception as e:
        logger.warning("Fibonacci %s 실패: %s", ticker, e)
        return {}


def fetch_dart_8q(tickers: list[str]) -> dict:
    """DART API로 최근 8분기 영업이익/순이익/매출 일괄 fetch."""
    try:
        from src.adapters.dart_financial_adapter import DartFinancialAdapter
        adapter = DartFinancialAdapter()
        if not adapter.is_available:
            logger.warning("DART API 미사용 — DART_API_KEY 미설정?")
            return {}
        logger.info("DART 8분기 fetch 시작 (10종목 일괄)")
        data = adapter.collect_bs_all(tickers)
        logger.info("DART fetch 완료 — %d종목 데이터", len(data))
        return data
    except Exception as e:
        logger.error("DART fetch 실패: %s", e, exc_info=True)
        return {}


def calculate_step5(kis: dict, dart: dict) -> dict:
    """STEP 5 산식 적용 (실측 기반).

    Returns:
        {
            current_market_cap_jo: 현재 시총 (조원),
            op_4q_total: 최근 4Q 영업이익 합 (조원),
            cagr_yoy: 1년 영업이익 증가율 (%),
            op_5y_future: 5년 후 영업이익 (조원),
            future_market_cap_jo: 미래 시총 (조원),
            upside_ratio: 업사이드 배수,
            annual_return: 연환산 수익률 (%),
            data_quality: 데이터 신뢰도 라벨
        }
    """
    out = {
        "current_market_cap_jo": None,
        "op_4q_total": None,
        "cagr_yoy": None,
        "op_5y_future": None,
        "future_market_cap_jo": None,
        "upside_ratio": None,
        "annual_return": None,
        "data_quality": "incomplete",
        "notes": [],
    }

    # 현재 시총 (KIS 억 → 조)
    mcap_eok = kis.get("market_cap_eok", 0)
    if mcap_eok > 0:
        out["current_market_cap_jo"] = round(mcap_eok / 10_000, 2)

    # DART에서 8분기 추세 추출 (단년도 CAGR의 일회성 효과 보정)
    # collect_bs_all returns: {ticker: {period_label: {op_income_cum, ...}}}
    # period_label 예: '2025Q3', '2024Q4', ...
    # reprt 코드: 11013=Q1누적 / 11012=Q2누적(반기) / 11014=Q3누적 / 11011=Q4누적(사업보고서)
    if dart:
        periods = sorted(dart.keys())  # 알파벳 정렬: 2023Q4 < 2024Q1 < ... < 2025Q3
        try:
            # 모든 분기의 누적 OP 추출
            op_cums = {}
            for label in periods:
                amt = dart[label].get("op_income_cum", 0) or 0
                op_cums[label] = amt

            # 최신 라벨 (보통 2025Q3)의 누적 OP → 연환산 (×4/3)
            latest_label = periods[-1]
            latest_op_cum = op_cums.get(latest_label, 0)

            # Q 식별 (라벨 끝의 Q1/Q2/Q3/Q4)
            q_in_latest = 3  # 기본값 (Q3 누적 = 3분기치)
            for q_marker, q_num in (("Q1", 1), ("Q2", 2), ("Q3", 3), ("Q4", 4)):
                if q_marker in latest_label:
                    q_in_latest = q_num
                    break

            # TTM (Trailing 12 Months) 영업이익 계산:
            # TTM = 최신 Qn 누적 + 직전년 연간 - 직전년 Qn 누적
            ttm_op = None
            if q_in_latest < 4 and len(periods) >= 5:
                # 예: latest=2025Q3, prev_year=2024Q4, prev_year_partial=2024Q3
                # 후보 라벨 찾기
                yr = int(latest_label[:4])
                prev_year_full = f"{yr-1}Q4"
                prev_year_partial = f"{yr-1}Q{q_in_latest}"
                if prev_year_full in op_cums and prev_year_partial in op_cums:
                    ttm_op = op_cums[latest_label] + op_cums[prev_year_full] - op_cums[prev_year_partial]
                    out["notes"].append(
                        f"TTM = {latest_label}({op_cums[latest_label]/1e12:.2f}조) + "
                        f"{prev_year_full}({op_cums[prev_year_full]/1e12:.2f}조) - "
                        f"{prev_year_partial}({op_cums[prev_year_partial]/1e12:.2f}조) "
                        f"= {ttm_op/1e12:.2f}조"
                    )
            elif q_in_latest == 4:
                ttm_op = latest_op_cum  # 사업보고서 = 연간

            if ttm_op and ttm_op > 0:
                out["op_4q_total"] = round(ttm_op / 1_000_000_000_000, 3)  # 원 → 조원
            elif latest_op_cum > 0:
                # 폴백: Qn 누적을 연환산
                annualized = latest_op_cum * 4 / q_in_latest
                out["op_4q_total"] = round(annualized / 1_000_000_000_000, 3)
                out["notes"].append(f"TTM 계산 불가 — Q{q_in_latest} 누적을 연환산")

            # CAGR: 2년 추세 (2023Q4 vs 2025Q3 같은 비교) — 일회성 효과 완화
            # 우선 2년 전 동일 분기 누적 vs 최근 누적
            cagr_2y = None
            cagr_1y = None
            if len(periods) >= 8:
                two_years_ago = periods[-8] if len(periods) >= 8 else None
                if two_years_ago and op_cums.get(two_years_ago, 0) > 0 and latest_op_cum > 0:
                    ratio_2y = latest_op_cum / op_cums[two_years_ago]
                    if ratio_2y > 0:
                        cagr_2y = (ratio_2y ** 0.5) - 1  # 2년 CAGR
            # 1년 전 동일 분기와도 비교
            if len(periods) >= 5:
                one_year_ago = periods[-5]
                if op_cums.get(one_year_ago, 0) > 0 and latest_op_cum > 0:
                    cagr_1y = (latest_op_cum / op_cums[one_year_ago]) - 1

            # 1년/2년 CAGR 평균 (있는 것만)
            cagrs = [c for c in (cagr_1y, cagr_2y) if c is not None]
            if cagrs:
                cagr_avg = sum(cagrs) / len(cagrs)
                out["cagr_yoy"] = round(cagr_avg * 100, 1)
                if cagr_1y is not None and cagr_2y is not None:
                    out["notes"].append(
                        f"CAGR 1y={cagr_1y*100:.1f}% / 2y={cagr_2y*100:.1f}% → 평균 {cagr_avg*100:.1f}%"
                    )
        except Exception as e:
            out["notes"].append(f"DART 추출 예외: {e}")

    # 5년 후 OP = 현재 TTM × (1 + CAGR_평균)^5
    if out["op_4q_total"] and out["cagr_yoy"] is not None:
        cagr_decimal = out["cagr_yoy"] / 100
        # 클립 (보수적 안전선)
        if cagr_decimal < -0.3:
            cagr_decimal = -0.3
            out["notes"].append("CAGR 클립: -30% (5년 지속 가정)")
        elif cagr_decimal > 0.5:
            cagr_decimal = 0.5
            out["notes"].append("CAGR 클립: +50% (5년 지속 한도)")
        out["op_5y_future"] = round(out["op_4q_total"] * (1 + cagr_decimal) ** 5, 3)

    # 미래 시총 — 2 모델 병행 (서로 검증)
    # 모델 A: PER 일정 가정 → 업사이드 = (1+CAGR)^5
    # 모델 B: 5년 후 OP × 현재 PER (영업이익 ≈ 순이익 단순화)
    per = kis.get("per", 0)
    if out["cagr_yoy"] is not None and out["current_market_cap_jo"]:
        cagr_clipped = max(-0.3, min(0.5, out["cagr_yoy"] / 100))
        out["upside_model_A"] = round((1 + cagr_clipped) ** 5, 2)
        out["future_mcap_model_A"] = round(out["current_market_cap_jo"] * out["upside_model_A"], 2)

    if out["op_5y_future"] and per > 0:
        out["future_mcap_model_B"] = round(out["op_5y_future"] * per, 2)
        if out["current_market_cap_jo"]:
            out["upside_model_B"] = round(out["future_mcap_model_B"] / out["current_market_cap_jo"], 2)

    # 두 모델 평균 (둘 다 있을 때) — 최종 업사이드
    upside_A = out.get("upside_model_A")
    upside_B = out.get("upside_model_B")
    upsides = [u for u in (upside_A, upside_B) if u is not None]
    if upsides:
        out["upside_ratio"] = round(sum(upsides) / len(upsides), 2)
        out["future_market_cap_jo"] = round(out["current_market_cap_jo"] * out["upside_ratio"], 2) if out["current_market_cap_jo"] else None
        if out["upside_ratio"] > 0:
            out["annual_return"] = round((out["upside_ratio"] ** 0.2 - 1) * 100, 1)

    # 데이터 품질
    if all([out["current_market_cap_jo"], out["op_4q_total"], out["cagr_yoy"], out["upside_ratio"]]):
        out["data_quality"] = "complete"
    elif out["current_market_cap_jo"]:
        out["data_quality"] = "partial"

    return out


def main():
    logger.info("=" * 60)
    logger.info("STEP 5 10종목 실측 fetch 시작 (강방천 모델)")
    logger.info("=" * 60)

    broker = get_broker()

    # 1. 모든 종목 KIS basics + Fibonacci
    results = {}
    for ticker, name in STOCKS:
        logger.info("[%s %s] KIS basics + Fibonacci...", ticker, name)
        kis = fetch_kis_basics(broker, ticker)
        fib = fetch_fibonacci(broker, ticker)
        results[ticker] = {
            "name": name,
            "kis": kis,
            "fibonacci": fib,
            "dart": None,
            "step5": None,
        }

    # 2. DART 8분기 일괄 (10종목 한 번에)
    tickers_only = [t for t, _ in STOCKS]
    dart_all = fetch_dart_8q(tickers_only)
    for ticker in tickers_only:
        results[ticker]["dart"] = dart_all.get(ticker, {})

    # 3. STEP 5 계산
    for ticker, name in STOCKS:
        r = results[ticker]
        r["step5"] = calculate_step5(r["kis"], r["dart"])

    # 4. 저장
    out_dir = Path("data/step5")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"step5_10stocks_{today}.json"

    # raw_output 제거 (용량)
    for r in results.values():
        if "raw_output" in r.get("kis", {}):
            del r["kis"]["raw_output"]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("저장: %s", out_file)

    # 5. 콘솔 표 출력 (두 모델 비교)
    print("\n" + "=" * 145)
    print(f"{'종목':12} {'현재가':>10} {'시총(조)':>10} {'PER':>7} "
          f"{'TTM_OP(조)':>11} {'CAGR%':>8} {'5y_OP':>9} "
          f"{'업사이드_A':>11} {'업사이드_B':>11} {'평균업사이드':>13} {'연환산%':>8}")
    print(f"{'':12} {'':10} {'':10} {'':7} "
          f"{'(영업)':>11} {'(1y+2y평균)':>8} {'(보수클립)':>9} "
          f"{'(PER일정)':>11} {'(5yOP×PER)':>11} {'(A,B평균)':>13} {'':8}")
    print("=" * 145)
    for ticker, name in STOCKS:
        r = results[ticker]
        kis = r["kis"]
        s5 = r["step5"]

        def fmt_num(v, default="-", precision=2):
            if v is None or v == 0:
                return default
            if isinstance(v, (int, float)):
                if precision == 0:
                    return f"{v:,.0f}"
                return f"{v:,.{precision}f}"
            return str(v)

        print(
            f"{name:12} "
            f"{kis.get('current_price', 0):>10,} "
            f"{fmt_num(s5.get('current_market_cap_jo')):>10} "
            f"{kis.get('per', 0):>7.1f} "
            f"{fmt_num(s5.get('op_4q_total'), precision=3):>11} "
            f"{fmt_num(s5.get('cagr_yoy'), precision=1):>8} "
            f"{fmt_num(s5.get('op_5y_future'), precision=2):>9} "
            f"{fmt_num(s5.get('upside_model_A')):>11} "
            f"{fmt_num(s5.get('upside_model_B')):>11} "
            f"{fmt_num(s5.get('upside_ratio')):>13} "
            f"{fmt_num(s5.get('annual_return'), precision=1):>8}"
        )
    print("=" * 145)

    # 6. 피보나치 표
    print("\n" + "=" * 100)
    print(f"{'종목':12} {'현재가':>10} {'1y고점':>10} {'1y저점':>10} "
          f"{'피보 0.382':>12} {'피보 0.5':>12} {'피보 0.618':>12}")
    print("=" * 100)
    for ticker, name in STOCKS:
        r = results[ticker]
        kis = r["kis"]
        fib = r["fibonacci"]
        print(
            f"{name:12} "
            f"{kis.get('current_price', 0):>10,} "
            f"{fib.get('high_1y', 0):>10,} "
            f"{fib.get('low_1y', 0):>10,} "
            f"{fib.get('fib_0_382', 0):>12,} "
            f"{fib.get('fib_0_500', 0):>12,} "
            f"{fib.get('fib_0_618', 0):>12,}"
        )
    print("=" * 100)

    logger.info("완료")
    return results


if __name__ == "__main__":
    main()
