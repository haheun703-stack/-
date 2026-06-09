"""경기선행지수 순환변동치 → KOSPI 고점 변곡점 신호 적대검증 (read-only 탐색)

영상 주장: 선행지수 순환변동치 101 돌파 후 "꺾이면" 고점 변곡점,
그 후 빠르면 3개월·보통 6개월 뒤 조정/하락장. "단 한 번도 안 빗나갔다".

이 스크립트는 그 주장을 ECOS 36년 데이터 + KOSPI로 적대검증한다.
- 별도 트랙: FLOWX 본체 무관, 실주문 0, 파일 저장 0 (콘솔 출력만)
- 검증 포인트: ①peak 후 forward return vs base rate ②임계값 민감도
  ③발표지연(2개월) look-ahead 보정 ④peak 정의 민감도

사용: python -u -X utf8 scripts/explore_leading_cycle_signal.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fetch_cycle(key: str, item: str) -> pd.Series:
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/600"
        f"/901Y067/M/199001/202612/{item}"
    )
    rows = requests.get(url, timeout=25).json()["StatisticSearch"]["row"]
    data = {r["TIME"]: float(r["DATA_VALUE"]) for r in rows if r.get("DATA_VALUE")}
    s = pd.Series(data)
    s.index = pd.to_datetime(s.index, format="%Y%m")
    return s.sort_index()


def find_peaks(lead: pd.Series, threshold: float, confirm: int = 1) -> list[pd.Timestamp]:
    """국소 peak(전월 상승→당월 하락 전환) 중 peak값 >= threshold.

    confirm: 꺾임 확인에 필요한 연속 하락 개월 수 (1=첫 하락, 2=2개월 연속 하락).
    """
    vals = lead.values
    idx = lead.index
    peaks = []
    for i in range(1, len(vals) - confirm):
        # i가 국소 peak: 직전 상승, 이후 confirm개월 연속 하락
        if vals[i] < vals[i - 1]:
            continue  # i 자체가 직전보다 낮으면 peak 아님
        rising_before = vals[i] >= vals[i - 1]
        falling_after = all(vals[i + k] < vals[i + k - 1] for k in range(1, confirm + 1))
        if rising_before and falling_after and vals[i] >= threshold:
            peaks.append(idx[i])
    return peaks


def _load_kospi() -> pd.Series:
    """KOSPI 일별 장기 종가. pykrx(1001) → FDR(KS11) fallback.
    (로컬 kospi_index.csv는 강세장 가공본이라 검증에 부적합 → 미사용)
    """
    # 1) pykrx (KRX 직접, fetch_ecos_macro에서 작동 확인)
    try:
        import contextlib
        import io as _io
        import logging
        from pykrx import stock as krx
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
                df = krx.get_index_ohlcv("19900101", "20261231", "1001")
        finally:
            logging.disable(logging.NOTSET)
        if df is not None and not df.empty:
            ks = df["종가"].dropna()
            ks.index = pd.to_datetime(ks.index)
            print(f"  KOSPI 소스: pykrx 1001 ({len(ks)}행)")
            return ks
    except Exception as e:
        print(f"  pykrx 1001 실패({e}) → FDR fallback")
    # 2) FDR
    import FinanceDataReader as fdr
    ks = fdr.DataReader("KS11")["Close"].dropna()
    ks.index = pd.to_datetime(ks.index)
    print(f"  KOSPI 소스: FDR KS11 ({len(ks)}행)")
    return ks


def fwd_return(kospi: pd.Series, t: pd.Timestamp, months: int) -> float | None:
    """t시점 대비 months개월 후 KOSPI 수익률(%)."""
    base = kospi.asof(t)
    target_date = t + pd.DateOffset(months=months)
    if target_date > kospi.index[-1]:
        return None
    fut = kospi.asof(target_date)
    if base is None or fut is None or base != base or fut != fut:
        return None
    return round((fut / base - 1) * 100, 2)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ECOS_API_KEY", "").strip()
    if not key:
        print("ECOS_API_KEY 없음")
        return

    lead = fetch_cycle(key, "I16E")  # 선행지수 순환변동치
    coin = fetch_cycle(key, "I16D")  # 동행지수 순환변동치
    ks = _load_kospi()
    ks_m = ks.resample("MS").last()  # 월별(월초 인덱스)

    overlap_start = max(lead.index[0], ks_m.index[0])
    print(f"선행 순환변동치: {lead.index[0].date()} ~ {lead.index[-1].date()} ({len(lead)}개월)")
    print(f"KOSPI(FDR):     {ks_m.index[0].date()} ~ {ks_m.index[-1].date()}")
    print(f"검증 겹침 구간:  {overlap_start.date()} ~ {min(lead.index[-1], ks_m.index[-1]).date()}")
    print(f"현재 선행={lead.iloc[-1]} ({lead.index[-1].date()}) / 동행={coin.iloc[-1]}")
    print("=" * 70)

    # ── base rate: 전체 시점의 forward return ──
    all_t = [t for t in lead.index if t >= overlap_start]
    for horizon in (3, 6):
        rs = [fwd_return(ks_m, t, horizon) for t in all_t]
        rs = [r for r in rs if r is not None]
        neg = sum(1 for r in rs if r < 0) / len(rs) * 100
        neg10 = sum(1 for r in rs if r < -10) / len(rs) * 100
        print(f"[BASE RATE] 임의시점 +{horizon}M: 평균 {sum(rs)/len(rs):+.2f}% · "
              f"하락(<0) {neg:.0f}% · 조정(<-10%) {neg10:.0f}% (n={len(rs)})")
    print("=" * 70)

    # ── peak 신호 검증 (임계값 민감도 × 꺾임확인 민감도) ──
    for confirm in (1, 2):
        print(f"\n### 꺾임 정의: {confirm}개월 연속 하락 확인 ###")
        for threshold in (100, 101, 102):
            peaks = [p for p in find_peaks(lead, threshold, confirm) if p >= overlap_start]
            if not peaks:
                print(f"  [임계 {threshold}] peak 0건")
                continue
            for horizon in (3, 6):
                # 이론(look-ahead 有): peak 월 기준
                th = [fwd_return(ks_m, p, horizon) for p in peaks]
                th = [r for r in th if r is not None]
                # 실전(발표지연 2개월 보정): peak 인지 시점 = peak+2개월
                pr = [fwd_return(ks_m, p + pd.DateOffset(months=2), horizon) for p in peaks]
                pr = [r for r in pr if r is not None]
                if th:
                    neg = sum(1 for r in th if r < 0) / len(th) * 100
                    line = (f"  [임계 {threshold}] peak {len(peaks)}건 · +{horizon}M "
                            f"이론 평균{sum(th)/len(th):+.2f}%/하락{neg:.0f}%")
                    if pr:
                        negp = sum(1 for r in pr if r < 0) / len(pr) * 100
                        line += f" · 발표지연보정 평균{sum(pr)/len(pr):+.2f}%/하락{negp:.0f}%"
                    print(line)

    # ── 최근 30년 peak 이벤트 상세 (임계 101, 첫 꺾임) ──
    print("\n" + "=" * 70)
    print("[peak 이벤트 상세] 임계 101 · 첫 꺾임 (이론 기준)")
    print(f"{'peak월':<10}{'선행값':>7}{'+3M':>9}{'+6M':>9}")
    for p in [p for p in find_peaks(lead, 101, 1) if p >= overlap_start]:
        r3 = fwd_return(ks_m, p, 3)
        r6 = fwd_return(ks_m, p, 6)
        print(f"{str(p.date()):<10}{lead.asof(p):>7.1f}"
              f"{(f'{r3:+.1f}%' if r3 is not None else 'NA'):>9}"
              f"{(f'{r6:+.1f}%' if r6 is not None else 'NA'):>9}")


if __name__ == "__main__":
    main()
