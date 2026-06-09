"""KODEX/대형주 가격 진위 검증 — source_truth 판정 (read-only).

현재 환경 데이터(pykrx)가 실제 한국장 가격인지, 가공/스케일링/시뮬인지 판정한다.
배경: hedge shadow가 KODEX200 13만원·삼성 32만원 등 실제(4만·7만)와 4~5배 차이.
★다음 주 실전 전환은 source_truth=REAL 종목만 허용(사장님 지시).

판정(pykrx vs 네이버 실시간):
  - 차이 < 3%            → REAL       (실전 검증 가능)
  - 차이 3~10%           → UNKNOWN    (보류)
  - 차이 > 10% & 배율일관 → SCALED     (스케일링 시뮬, 실전 금지)
  - 차이 > 10% & 배율불규칙→ SIMULATED  (가공, 실전 금지)

매매 무관·read-only·저장 0(콘솔). KIS는 화이트리스트 IP(VPS)라 로컬 제약 → 네이버 기준.
사용: python -u -X utf8 scripts/kodex_price_truth_check.py
"""
from __future__ import annotations

import contextlib
import io
import logging

import requests

TARGETS = {
    "069500": "KODEX200",
    "122630": "KODEX레버리지",
    "114800": "KODEX인버스",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}


def pykrx_close(code: str) -> float | None:
    from pykrx import stock as krx
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = krx.get_market_ohlcv("20260525", "20260609", code)
    finally:
        logging.disable(logging.NOTSET)
    if df is None or df.empty:
        return None
    return float(df["종가"].iloc[-1])


def naver_close(code: str) -> float | None:
    """네이버 금융 실시간 종가(실제 시장가). 여러 엔드포인트 폴백."""
    headers = {"User-Agent": "Mozilla/5.0"}
    # 1) 모바일 stock API
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/basic", timeout=10, headers=headers)
        if r.status_code == 200:
            d = r.json()
            v = d.get("closePrice") or d.get("nowPrice")
            if v:
                return float(str(v).replace(",", ""))
    except Exception:
        pass
    # 2) polling realtime API
    try:
        r = requests.get(
            f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}",
            timeout=10, headers=headers,
        )
        if r.status_code == 200:
            d = r.json()
            datas = d.get("datas") or []
            if datas:
                v = datas[0].get("closePrice")
                if v:
                    return float(str(v).replace(",", ""))
    except Exception:
        pass
    return None


def classify(pykrx_v: float | None, naver_v: float | None) -> tuple[str, float | None, float | None]:
    if not pykrx_v or not naver_v:
        return "UNKNOWN", None, None
    diff = abs(pykrx_v - naver_v) / naver_v * 100
    ratio = pykrx_v / naver_v
    if diff < 3:
        return "REAL", round(diff, 1), round(ratio, 2)
    if diff <= 10:
        return "UNKNOWN", round(diff, 1), round(ratio, 2)
    if 2.5 <= ratio <= 6.0:
        return "SCALED", round(diff, 1), round(ratio, 2)
    return "SIMULATED", round(diff, 1), round(ratio, 2)


def main():
    print(f"{'종목':<14}{'pykrx':>12}{'네이버':>12}{'차이%':>8}{'배율':>7}  source_truth")
    print("-" * 64)
    results = {}
    ratios = []
    for code, nm in TARGETS.items():
        p = pykrx_close(code)
        n = naver_close(code)
        st, diff, ratio = classify(p, n)
        results[code] = st
        if ratio:
            ratios.append(ratio)
        p_s = f"{p:,.0f}" if p else "None"
        n_s = f"{n:,.0f}" if n else "None"
        d_s = f"{diff:.1f}%" if diff is not None else "-"
        r_s = f"{ratio:.2f}x" if ratio else "-"
        print(f"{nm:<14}{p_s:>12}{n_s:>12}{d_s:>8}{r_s:>7}  {st}")
    print("-" * 64)
    real_n = sum(1 for v in results.values() if v == "REAL")
    print(f"REAL {real_n}/{len(TARGETS)}  |  실전 전환 허용 종목: "
          f"{[c for c, v in results.items() if v == 'REAL'] or '(없음)'}")
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"평균 배율(pykrx/네이버) {avg:.2f}x")
        if avg > 2.5:
            print("→ ★전 종목 스케일링 시뮬 데이터. 실전 전환 금지. 실제 가격 소스 확보 필요.")
    print("(KIS 교차검증은 화이트리스트 IP=VPS에서만 — 로컬 생략)")
    print("안전: read-only · 매매 무관 · 저장 0")


if __name__ == "__main__":
    main()
