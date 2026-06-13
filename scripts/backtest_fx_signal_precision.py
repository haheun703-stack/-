"""fx-liquidity P0-2 — 환율 급등 신호의 KOSPI 하락 선행성 적대검증 (read-only).

질문: usdkrw 1일 +1% 또는 20일 신고가가 KOSPI 익일~5일 하락을 선행하는가?
방법: base rate / precision / lift / 신호후 중앙수익률 / 2022 환율위기 vs 2024+ 구간 분리.

★사전 등록 (지시서 P0-2, 데이터 고문 방지): 2026 초강세장(환율↑·주가↑ 동행)이라 선행성이
  노이즈에 묻힐 수 있고 2022는 단일 에피소드 → **"NOISE 또는 표본 불충분" 판정 자체가 정상 답**.
  lift ≈ 1.0(base rate 대비 차별성 없음) 또는 신호 표본 < 30이면 "선행성 미입증"으로 종결한다.
  신호가 나올 때까지 임계값(+1%/20일 신고가)을 바꿔 재실행하지 않는다.

데이터: ECOS 731Y001(원/달러 일별) + pykrx 1001(KOSPI 종가 일별). 수집·계산만, 파일 write 0.
사용: python -u -X utf8 scripts/backtest_fx_signal_precision.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
START = "20220101"
END = datetime.now().strftime("%Y%m%d")


def fetch_usdkrw() -> pd.Series:
    """ECOS 731Y001 원/달러 일별 종가 (장기). 영업일만."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ECOS_API_KEY", "").strip()
    if not key:
        raise SystemExit("ECOS_API_KEY 없음 — .env 확인")
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000"
        f"/731Y001/D/{START}/{END}/0000001"
    )
    r = requests.get(url, timeout=30)
    payload = r.json()
    if "StatisticSearch" not in payload:
        raise SystemExit(f"ECOS 응답 이상: {payload.get('RESULT', payload)}")
    rows = payload["StatisticSearch"].get("row", [])
    data = {}
    for row in rows:
        try:
            data[pd.Timestamp(row["TIME"])] = float(row["DATA_VALUE"].replace(",", ""))
        except (ValueError, KeyError):
            continue
    return pd.Series(data).sort_index()


def fetch_kospi() -> pd.Series:
    """KOSPI 종가 일별 — Yahoo ^KS11. 실지수(가공본 kospi_index.csv 아님).

    ★pykrx 대신 Yahoo v8 chart(four_signal_gate 패턴 재사용): pykrx는 import 시 KRX 로그인을
      시도하는데 로컬 .env KRX 비번 만료로 차단 중(KRX_DATA_PW 만료 이슈 — 메모리). Yahoo는
      KRX 로그인 무관이라 P0-2 분석을 비번 갱신 대기 없이 진행할 수 있다.
    """
    import time as _t

    p1 = int(_t.mktime(_t.strptime(START, "%Y%m%d")))
    p2 = int(_t.mktime(_t.strptime(END, "%Y%m%d"))) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/%5EKS11"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"}, timeout=30)
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    data = {
        pd.Timestamp(t, unit="s").normalize(): c
        for t, c in zip(ts, closes)
        if c is not None
    }
    return pd.Series(data).sort_index()


def _summary(df: pd.DataFrame, sub: pd.DataFrame, label: str) -> None:
    base = df["fwd5_down"].mean()
    n = len(sub)
    print(f"\n[{label}] 신호 {n}건 / 전체 {len(df)}일")
    if n == 0:
        print("  → 신호 0건 (표본 불충분)")
        return
    prec = sub["fwd5_down"].mean()
    med1 = sub["fwd1"].median() * 100
    med5 = sub["fwd5"].median() * 100
    base_med5 = df["fwd5"].median() * 100
    lift = prec / base if base else float("nan")
    print(f"  base rate(전체 5일후 하락): {base*100:.1f}%  |  신호후 5일 하락: {prec*100:.1f}%  (lift {lift:.2f}x)")
    print(f"  신호후 중앙수익률: 1일 {med1:+.2f}% / 5일 {med5:+.2f}%  (전체 5일 중앙 {base_med5:+.2f}%)")
    verdict = "선행성 미입증(NOISE)" if (n < 30 or 0.85 <= lift <= 1.15) else "차별성 관찰 — 2차 검증 대상"
    print(f"  → 판정: {verdict}")


def main() -> None:
    print(f"=== fx-liquidity P0-2: 환율→KOSPI 선행성 적대검증 ({START}~{END}) ===")
    krw = fetch_usdkrw()
    kospi = fetch_kospi()
    print(f"수집: 환율 {len(krw)}일 / KOSPI {len(kospi)}일")

    df = pd.DataFrame({"krw": krw, "kospi": kospi}).dropna()
    if len(df) < 100:
        print(f"교집합 {len(df)}일 — 표본 불충분(<100), 분석 중단")
        return
    print(f"교집합 {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")

    df["krw_chg_1d"] = df["krw"].pct_change() * 100
    df["krw_20d_high"] = df["krw"] >= df["krw"].rolling(20).max()
    for h in (1, 2, 3, 5):
        df[f"fwd{h}"] = df["kospi"].shift(-h) / df["kospi"] - 1
    df["fwd5_down"] = df["fwd5"] < 0
    df = df.dropna(subset=["krw_chg_1d", "fwd5"])

    # ── 전체 구간 ──
    _summary(df, df[df["krw_chg_1d"] >= 1.0], "전체 · 환율 1일 +1%")
    _summary(df, df[df["krw_20d_high"]], "전체 · 환율 20일 신고가")

    # ── 1~2일 지연 강건성 (신호 다음날 진입 시) ──
    sig = df[df["krw_chg_1d"] >= 1.0]
    if len(sig) >= 30:
        lag_down = (sig["fwd1"] < 0).mean()
        print(f"\n[지연 강건성] 환율+1% 신호 익일(fwd1) 하락률 {lag_down*100:.1f}% (n={len(sig)})")

    # ── 구간 분리: 2022 환율위기 vs 2024+ 현재장 ──
    print("\n── 구간 분리 ──")
    for yr_label, mask in [
        ("2022 환율위기(1400+)", df.index.year == 2022),
        ("2024+ 현재장", df.index.year >= 2024),
    ]:
        seg = df[mask]
        if len(seg) < 30:
            print(f"[{yr_label}] {len(seg)}일 — 표본 불충분")
            continue
        s1 = seg[seg["krw_chg_1d"] >= 1.0]
        base = seg["fwd5_down"].mean()
        prec = s1["fwd5_down"].mean() if len(s1) else float("nan")
        lift = prec / base if (base and len(s1)) else float("nan")
        print(f"[{yr_label}] {len(seg)}일 / 환율+1% 신호 {len(s1)}건 / "
              f"base {base*100:.1f}% / precision {prec*100:.1f}% / lift {lift:.2f}x")

    print("\n※ 사전등록(데이터 고문 방지): lift≈1.0 또는 표본<30이면 '선행성 미입증'으로 종결.")
    print("  임계값을 신호 나올 때까지 바꿔 재실행하지 않는다. 결과 정직 반영(NOISE도 정상 답).")


if __name__ == "__main__":
    main()
