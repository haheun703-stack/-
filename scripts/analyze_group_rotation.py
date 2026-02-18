"""
그룹 순환매 상관관계 분석
- 그룹 간 상대강도 순환
- 그룹 ETF vs EWY 상대강도
- 개별 종목 교차 상관 (8×8)
- 삼성생명/삼성화재 중복 검증
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import pandas as pd
import numpy as np

DATA_DIR = Path("data/group_rotation")
PQ_DIR = Path("data/processed")
ANALYSIS_START = "2025-01-01"
ANALYSIS_END = "2026-02-13"


def load_etf(filename):
    df = pd.read_csv(DATA_DIR / filename, index_col="Date", parse_dates=True).sort_index()
    return df


def load_stock(ticker):
    pq = PQ_DIR / f"{ticker}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = DATA_DIR / f"stock_{ticker}.csv"
    if csv.exists():
        return pd.read_csv(csv, index_col="Date", parse_dates=True).sort_index()
    return None


def main():
    with open(DATA_DIR / "members.json", "r", encoding="utf-8") as f:
        groups = json.load(f)

    # ETF 로드
    etf_h = load_etf("etf_hyundai.csv")
    etf_s = load_etf("etf_samsung.csv")
    etf_ewy = load_etf("etf_ewy.csv")

    print("=" * 65)
    print("  그룹 순환매 상관관계 분석")
    print("=" * 65)
    print(f"  분석 기간: {ANALYSIS_START} ~ {ANALYSIS_END}")
    print(f"  현대차그룹 ETF: {len(etf_h)}일 | 삼성그룹 ETF: {len(etf_s)}일 | EWY: {len(etf_ewy)}일")

    # ── 분석 1: 그룹 간 상대강도 순환 ──
    print(f"\n{'=' * 65}")
    print("  [분석 1] 그룹 간 상대강도 순환")
    print(f"{'=' * 65}")

    # 20일 수익률
    etf_h["ret20"] = etf_h["close"].pct_change(20) * 100
    etf_s["ret20"] = etf_s["close"].pct_change(20) * 100
    etf_ewy["ret20"] = etf_ewy["close"].pct_change(20) * 100

    # 공통 날짜
    common = etf_h.index.intersection(etf_s.index)
    common = common[(common >= ANALYSIS_START) & (common <= ANALYSIS_END)]

    h_ret = etf_h.loc[common, "ret20"].dropna()
    s_ret = etf_s.loc[common, "ret20"].dropna()
    common2 = h_ret.index.intersection(s_ret.index)
    h_ret = h_ret.loc[common2]
    s_ret = s_ret.loc[common2]

    # 전체 상관
    overall_corr = h_ret.corr(s_ret)
    print(f"\n  현대차그룹 vs 삼성그룹 20일수익률 상관: {overall_corr:.3f}")

    # Rolling 60일 상관
    df_pair = pd.DataFrame({"hyundai": h_ret, "samsung": s_ret})
    roll_corr = df_pair["hyundai"].rolling(60).corr(df_pair["samsung"])
    roll_valid = roll_corr.dropna()
    print(f"  60일 롤링 상관: 평균 {roll_valid.mean():.3f}, 범위 [{roll_valid.min():.3f}, {roll_valid.max():.3f}]")

    # 크로스 코릴레이션 (lag -10 ~ +10)
    print(f"\n  크로스 코릴레이션 (현대차 → 삼성, lag일):")
    best_lag, best_corr = 0, 0
    for lag in range(-10, 11):
        if lag >= 0:
            c = h_ret.iloc[:len(h_ret)-lag if lag > 0 else len(h_ret)].values
            d = s_ret.iloc[lag:].values
        else:
            c = h_ret.iloc[-lag:].values
            d = s_ret.iloc[:len(s_ret)+lag].values
        min_len = min(len(c), len(d))
        if min_len < 30:
            continue
        corr = np.corrcoef(c[:min_len], d[:min_len])[0, 1]
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
        if lag in [-5, -3, -1, 0, 1, 3, 5]:
            print(f"    lag {lag:+3d}일: {corr:.3f}")

    print(f"  최대 상관 lag: {best_lag:+d}일 (corr={best_corr:.3f})")
    if best_lag > 0:
        print(f"  → 현대차가 {best_lag}일 선행 후 삼성 따라옴")
    elif best_lag < 0:
        print(f"  → 삼성이 {-best_lag}일 선행 후 현대차 따라옴")
    else:
        print(f"  → 동시 움직임 (선행/후행 관계 약함)")

    # 스프레드 분석
    spread = h_ret - s_ret
    print(f"\n  스프레드 (현대차 - 삼성) 20일수익률:")
    print(f"    평균: {spread.mean():.2f}%p, 표준편차: {spread.std():.2f}%p")
    print(f"    범위: [{spread.min():.1f}%p, {spread.max():.1f}%p]")
    extreme_high = (spread > spread.mean() + 1.5 * spread.std()).sum()
    extreme_low = (spread < spread.mean() - 1.5 * spread.std()).sum()
    print(f"    극단값 (±1.5σ): 현대차 우위 {extreme_high}일, 삼성 우위 {extreme_low}일")

    # ── 분석 2: 그룹 ETF vs EWY ──
    print(f"\n{'=' * 65}")
    print("  [분석 2] 그룹 ETF vs EWY 상대강도")
    print(f"{'=' * 65}")

    common_ewy = etf_h.index.intersection(etf_ewy.index)
    common_ewy = common_ewy[(common_ewy >= ANALYSIS_START) & (common_ewy <= ANALYSIS_END)]

    for name, etf, label in [("현대차그룹", etf_h, "hyundai"), ("삼성그룹", etf_s, "samsung")]:
        e_ret = etf.loc[common_ewy, "ret20"].dropna()
        ewy_ret = etf_ewy.loc[common_ewy, "ret20"].dropna()
        ci = e_ret.index.intersection(ewy_ret.index)
        e_ret = e_ret.loc[ci]
        ewy_ret = ewy_ret.loc[ci]

        rel = e_ret - ewy_ret
        corr_ewy = e_ret.corr(ewy_ret)

        print(f"\n  {name} vs EWY:")
        print(f"    상관: {corr_ewy:.3f}")
        print(f"    상대강도 (그룹 - EWY): 평균 {rel.mean():.2f}%p, σ {rel.std():.2f}%p")

        # 그룹이 EWY 대비 언더퍼폼하는 기간
        under = (rel < -rel.std()).sum()
        total = len(rel)
        print(f"    EWY 대비 언더퍼폼 (-1σ 이하): {under}일 ({under/total*100:.0f}%)")

    # ── 분석 3: 8종목 교차 상관 ──
    print(f"\n{'=' * 65}")
    print("  [분석 3] 개별 종목 교차 상관 (8×8)")
    print(f"{'=' * 65}")

    all_members = []
    for gname, gdata in groups.items():
        for m in gdata["members"]:
            all_members.append({"ticker": m["ticker"], "name": m["name"], "group": gname})

    # 20일 수익률
    stock_rets = {}
    for m in all_members:
        df = load_stock(m["ticker"])
        if df is None:
            continue
        close = df["close"]
        ret = close.pct_change(20) * 100
        ret = ret[(ret.index >= ANALYSIS_START) & (ret.index <= ANALYSIS_END)]
        stock_rets[m["ticker"]] = ret

    # 공통 날짜 정렬
    all_dates = None
    for ret in stock_rets.values():
        if all_dates is None:
            all_dates = ret.dropna().index
        else:
            all_dates = all_dates.intersection(ret.dropna().index)

    ret_matrix = pd.DataFrame({t: stock_rets[t].reindex(all_dates) for t in stock_rets})
    ret_matrix = ret_matrix.dropna()

    print(f"\n  공통 데이터: {len(ret_matrix)}일, {len(ret_matrix.columns)}종목")

    # 상관 행렬
    corr_matrix = ret_matrix.corr()

    # 이름 매핑
    name_map = {m["ticker"]: m["name"] for m in all_members}

    # 헤더
    print(f"\n  {'':>10}", end="")
    for t in corr_matrix.columns:
        print(f" {name_map.get(t, t):>8}", end="")
    print()
    print(f"  {'-' * (10 + 9 * len(corr_matrix.columns))}")

    for t1 in corr_matrix.index:
        print(f"  {name_map.get(t1, t1):>10}", end="")
        for t2 in corr_matrix.columns:
            val = corr_matrix.loc[t1, t2]
            marker = " *" if t1 != t2 and abs(val) > 0.7 else "  "
            print(f" {val:>6.2f}{marker}", end="")
        print()

    # 고상관 쌍 추출
    print(f"\n  고상관 쌍 (|r| > 0.7, 같은 종목 제외):")
    pairs_found = set()
    for i, t1 in enumerate(corr_matrix.index):
        for j, t2 in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            val = corr_matrix.loc[t1, t2]
            if abs(val) > 0.7:
                pair_key = tuple(sorted([t1, t2]))
                if pair_key not in pairs_found:
                    pairs_found.add(pair_key)
                    g1 = next((m["group"] for m in all_members if m["ticker"] == t1), "?")
                    g2 = next((m["group"] for m in all_members if m["ticker"] == t2), "?")
                    cross = " [교차그룹]" if g1 != g2 else ""
                    print(f"    {name_map[t1]} × {name_map[t2]}: {val:.3f}{cross}")

    if not pairs_found:
        print(f"    없음 (모든 쌍 |r| < 0.7)")

    # 저상관 쌍 (분산 효과)
    print(f"\n  저상관 쌍 (|r| < 0.3, 교차그룹):")
    low_pairs = []
    for i, t1 in enumerate(corr_matrix.index):
        for j, t2 in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            val = corr_matrix.loc[t1, t2]
            g1 = next((m["group"] for m in all_members if m["ticker"] == t1), "?")
            g2 = next((m["group"] for m in all_members if m["ticker"] == t2), "?")
            if g1 != g2 and abs(val) < 0.3:
                low_pairs.append((name_map[t1], name_map[t2], val))

    if low_pairs:
        for n1, n2, val in sorted(low_pairs, key=lambda x: abs(x[2])):
            print(f"    {n1} × {n2}: {val:.3f} → 동시 보유 시 분산 효과 큼")
    else:
        print(f"    없음 (교차그룹 모든 쌍 |r| >= 0.3)")

    # ── 삼성생명 vs 삼성화재 특별 분석 ──
    print(f"\n{'=' * 65}")
    print("  [특별] 삼성생명 vs 삼성화재 중복 검증")
    print(f"{'=' * 65}")

    life_ret = stock_rets.get("032830")
    fire_ret = stock_rets.get("000810")
    if life_ret is not None and fire_ret is not None:
        ci = life_ret.dropna().index.intersection(fire_ret.dropna().index)
        life_ret = life_ret.loc[ci]
        fire_ret = fire_ret.loc[ci]

        corr_lf = life_ret.corr(fire_ret)
        print(f"\n  20일 수익률 상관: {corr_lf:.3f}")

        # 5일 수익률 상관도
        life_df = load_stock("032830")
        fire_df = load_stock("000810")
        l5 = life_df["close"].pct_change(5).reindex(ci).dropna()
        f5 = fire_df["close"].pct_change(5).reindex(ci).dropna()
        ci5 = l5.index.intersection(f5.index)
        corr_5 = l5.loc[ci5].corr(f5.loc[ci5])
        print(f"  5일 수익률 상관: {corr_5:.3f}")

        # 동시 하락/상승 비율
        both_down = ((life_ret < 0) & (fire_ret < 0)).sum()
        both_up = ((life_ret > 0) & (fire_ret > 0)).sum()
        total = len(life_ret)
        print(f"  동시 상승: {both_up}일 ({both_up/total*100:.0f}%)")
        print(f"  동시 하락: {both_down}일 ({both_down/total*100:.0f}%)")
        print(f"  역방향: {total - both_up - both_down}일 ({(total-both_up-both_down)/total*100:.0f}%)")

        if corr_lf > 0.7:
            print(f"\n  판정: 상관 {corr_lf:.3f} > 0.7 → 하나 제거 권장")
            # 어느 것을 남길지
            life_vol = life_ret.std()
            fire_vol = fire_ret.std()
            print(f"    삼성생명 변동성: {life_vol:.2f} | 삼성화재 변동성: {fire_vol:.2f}")
            if life_vol > fire_vol:
                print(f"    → 삼성생명 남기기 (변동성 높음 = 순환매 기회 많음)")
            else:
                print(f"    → 삼성화재 남기기 (변동성 높음 = 순환매 기회 많음)")
        elif corr_lf > 0.5:
            print(f"\n  판정: 상관 {corr_lf:.3f} — 중간, 동시 보유 제한 권장")
        else:
            print(f"\n  판정: 상관 {corr_lf:.3f} < 0.5 → 둘 다 유지 가능")

    # ── 요약 + 추천 ──
    print(f"\n{'=' * 65}")
    print("  분석 요약 + 종목 추천")
    print(f"{'=' * 65}")

    print(f"""
  1. 그룹 간 상관 {overall_corr:.3f} → {'높음 (동시 움직임)' if overall_corr > 0.7 else '중간' if overall_corr > 0.5 else '낮음 (분산 효과 기대)'}
  2. 크로스 코릴레이션 최대 lag: {best_lag:+d}일
  3. 삼성생명/화재 상관: {corr_lf:.3f}
  4. 교차그룹 저상관 쌍: {len(low_pairs)}개

  추천 종목 구성:
    현대차그룹: 현대차, 기아, 현대제철 (기존 C모드 유지)
    삼성그룹: [아래 상관 분석 기반 확정]""")


if __name__ == "__main__":
    main()
