"""
3단 예측 체인 백테스트 — 일봉 근사 (1년)

일봉 근사 방법:
  - DAX 30분 → DAX 당일 시가→종가 (장중 수익률)
  - AUD/JPY, CNH → 전일 종가 대비 당일 종가 변화율
  - 타겟: S&P500 당일 수익률 (전일 종가→당일 종가)

사용법:
    python -u -X utf8 scripts/backtest_predict_chain.py
    python -u -X utf8 scripts/backtest_predict_chain.py --start 2024-03-01 --end 2026-03-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────
# 티커 목록
# ──────────────────────────────────────────
TICKERS = {
    # Stage 1: 아시안 리스크
    "AUDJPY=X": "AUD/JPY",
    "CNY=X": "USD/CNY",  # CNH 대신 CNY (일봉 제공)
    "ES=F": "ES선물",
    # Stage 2: 유럽 오픈
    "^GDAXI": "DAX",
    "EURUSD=X": "EUR/USD",
    # Stage 3: 괴리 감지
    "HYG": "하이일드",
    "^TNX": "10Y금리",
    "GC=F": "금선물",
    "CL=F": "원유",
    # 타겟
    "^GSPC": "S&P500",
}

# ──────────────────────────────────────────
# 설정 (predict_chain.py와 동일)
# ──────────────────────────────────────────
CFG = {
    "asian_risk": {
        "audjpy_threshold_pct": 0.3,
        "cnh_threshold_pct": 0.2,
        "weight_audjpy": 0.50,
        "weight_cnh": 0.30,
        "weight_futures": 0.20,
    },
    "europe_open": {
        "dax_threshold_pct": 0.3,
        "eurusd_threshold_pct": 0.15,
        "weight_dax": 0.55,
        "weight_eurusd": 0.25,
        "weight_futures": 0.20,
    },
    "signal": {
        "min_agreement_score": 0.4,
        "direction_match_required": True,
    },
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_ret(df: pd.DataFrame, idx: int, mode: str = "cc") -> float | None:
    """수익률 계산. cc=close-to-close, oc=open-to-close."""
    try:
        if mode == "cc":
            if idx == 0:
                return None
            prev_c = float(df["Close"].iloc[idx - 1])
            curr_c = float(df["Close"].iloc[idx])
            if prev_c == 0 or np.isnan(prev_c):
                return None
            return (curr_c - prev_c) / prev_c
        elif mode == "oc":
            o = float(df["Open"].iloc[idx])
            c = float(df["Close"].iloc[idx])
            if o == 0 or np.isnan(o):
                return None
            return (c - o) / o
    except (IndexError, KeyError):
        return None
    return None


# ──────────────────────────────────────────
# 3단 엔진 (일봉 근사)
# ──────────────────────────────────────────
def compute_signal(
    date_idx: int,
    sp500: pd.DataFrame,
    dax: pd.DataFrame | None,
    audjpy: pd.DataFrame | None,
    cnh: pd.DataFrame | None,
    es: pd.DataFrame | None,
    eurusd: pd.DataFrame | None,
    hyg: pd.DataFrame | None,
    tnx: pd.DataFrame | None,
    gold: pd.DataFrame | None,
) -> dict | None:
    """일봉 기반 3단 예측 체인 시그널 생성."""

    sp_date = sp500.index[date_idx]

    # ── Stage 1: 아시안 리스크 ──
    cfg1 = CFG["asian_risk"]
    s1_score = 0.0

    # AUD/JPY (전일→당일 종가)
    aj_ret = _aligned_ret(audjpy, sp_date, mode="cc")
    if aj_ret is not None:
        th = cfg1["audjpy_threshold_pct"] / 100
        w = cfg1["weight_audjpy"]
        if aj_ret > th:
            s1_score += w
        elif aj_ret < -th:
            s1_score -= w

    # USD/CNH
    cnh_ret = _aligned_ret(cnh, sp_date, mode="cc")
    if cnh_ret is not None:
        th = cfg1["cnh_threshold_pct"] / 100
        w = cfg1["weight_cnh"]
        if cnh_ret > th:
            s1_score -= w  # CNH 약세 = risk-off
        elif cnh_ret < -th:
            s1_score += w * 0.5

    # ES 선물
    es_ret = _aligned_ret(es, sp_date, mode="cc")
    if es_ret is not None:
        w = cfg1["weight_futures"]
        s1_score += clamp(es_ret * 100, -1, 1) * w

    s1_score = clamp(s1_score, -1, 1)
    s1_dir = "BULL" if s1_score > 0.15 else ("BEAR" if s1_score < -0.15 else "NEUTRAL")

    # ── Stage 2: 유럽 오픈 (DAX 시가→종가 = 장중 수익률) ──
    cfg2 = CFG["europe_open"]
    s2_score = 0.0

    # DAX 장중 수익률 (시가→종가) = DAX 30분의 일봉 근사
    dax_ret = _aligned_ret(dax, sp_date, mode="oc")
    if dax_ret is not None:
        th = cfg2["dax_threshold_pct"] / 100
        w = cfg2["weight_dax"]
        if dax_ret > th:
            s2_score += w
        elif dax_ret < -th:
            s2_score -= w
        else:
            s2_score += clamp(dax_ret * 100, -1, 1) * w * 0.5

    # EUR/USD
    eu_ret = _aligned_ret(eurusd, sp_date, mode="cc")
    if eu_ret is not None:
        th = cfg2["eurusd_threshold_pct"] / 100
        w = cfg2["weight_eurusd"]
        if eu_ret > th:
            s2_score += w
        elif eu_ret < -th:
            s2_score -= w

    # ES 30분 → ES 장중 수익률로 근사
    es_oc = _aligned_ret(es, sp_date, mode="oc")
    if es_oc is not None:
        w = cfg2["weight_futures"]
        s2_score += clamp(es_oc * 100, -1, 1) * w

    s2_score = clamp(s2_score, -1, 1)
    s2_dir = "BULL" if s2_score > 0.15 else ("BEAR" if s2_score < -0.15 else "NEUTRAL")

    # ── Stage 3: 괴리 감지 ──
    invalidate = False
    boost = False

    # HYG 히든 스트레스
    hyg_ret = _aligned_ret(hyg, sp_date, mode="cc")
    es_cc = _aligned_ret(es, sp_date, mode="cc")
    if hyg_ret is not None and es_cc is not None:
        if abs(es_cc) < 0.003 and hyg_ret < -0.005:
            invalidate = True

    # 채권↑ + 금↓ = 리스크온
    tnx_ret = _aligned_ret(tnx, sp_date, mode="cc")
    gold_ret = _aligned_ret(gold, sp_date, mode="cc")
    if tnx_ret is not None and gold_ret is not None:
        if tnx_ret > 0.002 and gold_ret < -0.003:
            boost = True

    # ── 결합 ──
    combined = s1_score * 0.35 + s2_score * 0.65
    if invalidate:
        combined = min(combined, 0)
    if boost:
        combined *= 1.3
    combined = clamp(combined, -1, 1)

    direction_match = s1_dir == s2_dir and s1_dir != "NEUTRAL"
    confidence = abs(combined)
    if direction_match:
        confidence = min(1.0, confidence * 1.5)

    min_score = CFG["signal"]["min_agreement_score"]
    if CFG["signal"]["direction_match_required"] and not direction_match:
        signal = "NEUTRAL"
        confidence *= 0.5
    elif combined >= min_score:
        signal = "BULL"
    elif combined <= -min_score:
        signal = "BEAR"
    else:
        signal = "NEUTRAL"

    # ── S&P500 실제 수익률 (타겟) ──
    sp_ret = safe_ret(sp500, date_idx, mode="cc")
    if sp_ret is None:
        return None

    sp_dir = "BULL" if sp_ret > 0.001 else ("BEAR" if sp_ret < -0.001 else "NEUTRAL")

    return {
        "date": sp_date.strftime("%Y-%m-%d"),
        "signal": signal,
        "score": round(combined * 100, 2),
        "confidence": round(confidence * 100, 1),
        "s1_dir": s1_dir,
        "s1_score": round(s1_score, 4),
        "s2_dir": s2_dir,
        "s2_score": round(s2_score, 4),
        "direction_match": direction_match,
        "invalidate": invalidate,
        "boost": boost,
        "sp_ret_pct": round(sp_ret * 100, 4),
        "sp_direction": sp_dir,
        "dax_ret_pct": round((dax_ret or 0) * 100, 4),
        "audjpy_ret_pct": round((aj_ret or 0) * 100, 4),
    }


def _aligned_ret(
    df: pd.DataFrame | None, target_date: pd.Timestamp, mode: str = "cc"
) -> float | None:
    """타겟 날짜에 가장 가까운 데이터로 수익률 계산."""
    if df is None or df.empty:
        return None

    # 정확한 날짜 매칭
    if target_date in df.index:
        idx = df.index.get_loc(target_date)
        return safe_ret(df, idx, mode)

    # 가장 가까운 이전 날짜 (최대 3일 전까지)
    mask = df.index <= target_date
    if not mask.any():
        return None
    closest = df.index[mask][-1]
    if (target_date - closest).days > 3:
        return None
    idx = df.index.get_loc(closest)
    return safe_ret(df, idx, mode)


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3단 예측 체인 백테스트")
    parser.add_argument("--start", default="2025-03-01")
    parser.add_argument("--end", default="2026-03-01")
    args = parser.parse_args()

    print(f"═══ 3단 예측 체인 백테스트 ═══")
    print(f"기간: {args.start} ~ {args.end}")
    print(f"근사: DAX 30분 → DAX 시가→종가, AUD/JPY → 전일 종가 대비")
    print()

    # 1. 데이터 다운로드
    print("[1] 데이터 다운로드...")
    data = {}
    for ticker, label in TICKERS.items():
        try:
            df = yf.download(ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
            # MultiIndex 컬럼 평탄화
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and len(df) > 10:
                data[ticker] = df
                print(f"  {ticker:12s} ({label}): {len(df):>4d}일")
            else:
                print(f"  {ticker:12s} ({label}): 데이터 부족")
        except Exception as e:
            print(f"  {ticker:12s} ({label}): 실패 — {e}")

    sp500 = data.get("^GSPC")
    if sp500 is None or len(sp500) < 50:
        print("ERROR: S&P500 데이터 없음. 종료.")
        sys.exit(1)

    print(f"\n  S&P500 거래일: {len(sp500)}일")

    # 2. 백테스트 실행
    print("\n[2] 백테스트 실행...")
    results = []
    for i in range(1, len(sp500)):
        row = compute_signal(
            i, sp500,
            dax=data.get("^GDAXI"),
            audjpy=data.get("AUDJPY=X"),
            cnh=data.get("CNY=X"),
            es=data.get("ES=F"),
            eurusd=data.get("EURUSD=X"),
            hyg=data.get("HYG"),
            tnx=data.get("^TNX"),
            gold=data.get("GC=F"),
        )
        if row:
            results.append(row)

    df_results = pd.DataFrame(results)
    total = len(df_results)
    print(f"  총 시그널: {total}건")

    # 3. 분석
    print("\n" + "═" * 60)
    print("  3단 예측 체인 백테스트 결과")
    print("═" * 60)

    # 3-1. 시그널 분포
    sig_counts = df_results["signal"].value_counts()
    print(f"\n[시그널 분포]")
    for sig in ["BULL", "BEAR", "NEUTRAL"]:
        cnt = sig_counts.get(sig, 0)
        pct = cnt / total * 100
        print(f"  {sig:10s}: {cnt:>4d}건 ({pct:5.1f}%)")

    # 3-2. 방향 예측 정확도 (NEUTRAL 제외)
    active = df_results[df_results["signal"] != "NEUTRAL"].copy()
    if len(active) > 0:
        active["correct"] = active["signal"] == active["sp_direction"]
        hit_rate = active["correct"].mean() * 100
        print(f"\n[방향 예측 정확도] (NEUTRAL 제외)")
        print(f"  시그널 발동: {len(active)}건 / {total}건 ({len(active)/total*100:.1f}%)")
        print(f"  적중률: {hit_rate:.1f}%")

        # BULL/BEAR 별 적중률
        for sig in ["BULL", "BEAR"]:
            sub = active[active["signal"] == sig]
            if len(sub) > 0:
                sr = sub["correct"].mean() * 100
                print(f"    {sig}: {len(sub)}건, 적중 {sr:.1f}%")
    else:
        print("\n  활성 시그널 없음")

    # 3-3. 수익률 분석 (시그널 따라 매매 시)
    print(f"\n[가상 수익률 분석]")
    df_results["position_ret"] = 0.0
    # BULL → 매수, BEAR → 매도(숏), NEUTRAL → 관망
    bull_mask = df_results["signal"] == "BULL"
    bear_mask = df_results["signal"] == "BEAR"
    df_results.loc[bull_mask, "position_ret"] = df_results.loc[bull_mask, "sp_ret_pct"]
    df_results.loc[bear_mask, "position_ret"] = -df_results.loc[bear_mask, "sp_ret_pct"]

    total_ret = df_results["position_ret"].sum()
    avg_ret = df_results.loc[bull_mask | bear_mask, "position_ret"].mean() if (bull_mask | bear_mask).any() else 0
    cumulative = df_results["position_ret"].cumsum()
    mdd = (cumulative - cumulative.cummax()).min()

    print(f"  누적 수익률: {total_ret:+.2f}%")
    print(f"  건당 평균: {avg_ret:+.4f}%")
    print(f"  MDD: {mdd:.2f}%")

    # 승/패 분석
    active_rets = df_results.loc[bull_mask | bear_mask, "position_ret"]
    if len(active_rets) > 0:
        wins = (active_rets > 0).sum()
        losses = (active_rets < 0).sum()
        flat = (active_rets == 0).sum()
        win_avg = active_rets[active_rets > 0].mean() if wins > 0 else 0
        loss_avg = active_rets[active_rets < 0].mean() if losses > 0 else 0
        pf = abs(active_rets[active_rets > 0].sum() / active_rets[active_rets < 0].sum()) if losses > 0 else float("inf")

        print(f"  승: {wins}건 (평균 +{win_avg:.3f}%)")
        print(f"  패: {losses}건 (평균 {loss_avg:.3f}%)")
        print(f"  보합: {flat}건")
        print(f"  Profit Factor: {pf:.2f}")

    # 3-4. 방향 일치 효과
    print(f"\n[방향 일치 효과]")
    matched = df_results[df_results["direction_match"]]
    unmatched = df_results[~df_results["direction_match"]]
    print(f"  방향 일치: {len(matched)}건 ({len(matched)/total*100:.1f}%)")
    print(f"  방향 불일치: {len(unmatched)}건 ({len(unmatched)/total*100:.1f}%)")

    if len(matched) > 0:
        m_active = matched[matched["signal"] != "NEUTRAL"]
        if len(m_active) > 0:
            m_correct = (m_active["signal"] == m_active["sp_direction"]).mean() * 100
            print(f"  일치 시 적중률: {m_correct:.1f}%")

    # 3-5. DAX vs S&P500 직접 상관
    print(f"\n[DAX → S&P500 방향 상관]")
    dax_valid = df_results[df_results["dax_ret_pct"] != 0].copy()
    if len(dax_valid) > 0:
        dax_valid["dax_dir"] = dax_valid["dax_ret_pct"].apply(
            lambda x: "BULL" if x > 0.1 else ("BEAR" if x < -0.1 else "NEUTRAL")
        )
        dax_active = dax_valid[dax_valid["dax_dir"] != "NEUTRAL"]
        if len(dax_active) > 0:
            dax_correct = (dax_active["dax_dir"] == dax_active["sp_direction"]).mean() * 100
            print(f"  DAX 방향 → S&P 방향 일치: {dax_correct:.1f}% ({len(dax_active)}건)")
        else:
            print(f"  DAX 방향 판정 건수 부족")

        # 피어슨 상관
        corr = dax_valid["dax_ret_pct"].corr(dax_valid["sp_ret_pct"])
        print(f"  DAX-S&P 수익률 상관계수: {corr:.4f}")

    # 3-6. 월별 적중률
    print(f"\n[월별 적중률]")
    df_results["month"] = pd.to_datetime(df_results["date"]).dt.to_period("M")
    for month, group in df_results.groupby("month"):
        g_active = group[group["signal"] != "NEUTRAL"]
        if len(g_active) > 0:
            hr = (g_active["signal"] == g_active["sp_direction"]).mean() * 100
            g_ret = group["position_ret"].sum()
            print(f"  {month}: 발동 {len(g_active):>2d}건, 적중 {hr:5.1f}%, 수익 {g_ret:+6.2f}%")
        else:
            print(f"  {month}: 발동 0건")

    # 3-7. 괴리 감지 효과
    print(f"\n[괴리 감지 효과]")
    inv_days = df_results[df_results["invalidate"]].copy()
    boost_days = df_results[df_results["boost"]].copy()
    print(f"  HYG 히든 스트레스 (invalidate): {len(inv_days)}건")
    if len(inv_days) > 0:
        inv_sp = inv_days["sp_ret_pct"].mean()
        print(f"    → 해당일 S&P 평균 수익: {inv_sp:+.3f}% (실제로 하락했나?)")
    print(f"  스마트머니 리스크온 (boost): {len(boost_days)}건")
    if len(boost_days) > 0:
        boost_sp = boost_days["sp_ret_pct"].mean()
        print(f"    → 해당일 S&P 평균 수익: {boost_sp:+.3f}% (실제로 상승했나?)")

    print(f"\n{'═' * 60}")
    print(f"  ※ 일봉 근사치임. DAX 30분봉 = DAX 시가→종가로 대체")
    print(f"  ※ 실제 5분봉 기반 성능은 이보다 높을 수 있음")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
