"""
MLCC/전자부품 섹터 폭등 전 거래량·매집 신호 심화 분석
대상: 삼성전기(009150), 삼화콘덴서(001820), SAMT(031330), LG이노텍(011070)
기간: 2026-01-01 ~ 2026-05-29
"""
import sys
sys.path.insert(0, 'D:/sub-agent-project_퀀트봇')

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:,.1f}'.format)

STOCKS = {
    '009150': '삼성전기',
    '001820': '삼화콘덴서',
    '031330': 'SAMT',
    '011070': 'LG이노텍',
}
START = '2026-01-01'
END   = '2026-05-29'

def load_data():
    data = {}
    for code, name in STOCKS.items():
        df = fdr.DataReader(code, START, END)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # 거래대금 계산 (원 단위)
        df['거래대금'] = df['Close'] * df['Volume']
        # 이동평균
        df['Vol_MA5']  = df['Volume'].rolling(5).mean()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Amt_MA5']  = df['거래대금'].rolling(5).mean()
        df['Amt_MA20'] = df['거래대금'].rolling(20).mean()
        # 비율
        df['Vol_5_20_ratio'] = df['Vol_MA5'] / df['Vol_MA20']
        df['Vol_day_20_ratio'] = df['Volume'] / df['Vol_MA20']
        df['Amt_day_20_ratio'] = df['거래대금'] / df['Amt_MA20']
        # 수익률
        df['Ret_1d'] = df['Close'].pct_change()
        df['Ret_5d'] = df['Close'].pct_change(5)
        df['Ret_20d'] = df['Close'].pct_change(20)
        # 연속 거래량 증가일
        df['Vol_inc'] = (df['Volume'] > df['Volume'].shift(1)).astype(int)
        df['Vol_consec_inc'] = df['Vol_inc'].groupby(
            (df['Vol_inc'] != df['Vol_inc'].shift()).cumsum()
        ).cumsum() * df['Vol_inc']
        data[code] = df
        print(f"[로드 완료] {name}({code}): {len(df)}행, {df.index[0].date()} ~ {df.index[-1].date()}")
    return data

def find_surge_start(df, threshold=0.30):
    """누적 수익률 +30% 최초 달성 시점 탐색 (20일 롤링)"""
    for i in range(20, len(df)):
        ret = (df['Close'].iloc[i] - df['Close'].iloc[i-20]) / df['Close'].iloc[i-20]
        if ret >= threshold:
            # 시작점을 역으로 찾기: 최저점 이후 30% 상승
            window = df.iloc[i-20:i+1]
            start_idx = window['Close'].idxmin()
            peak_ret = (df['Close'].iloc[i] - window['Close'].min()) / window['Close'].min()
            return start_idx, df.index[i], peak_ret
    return None, None, None

def analyze_pre_surge(df, name, surge_start, n_days=20):
    """폭등 시작 전 N일 거래량 패턴 분석"""
    if surge_start is None:
        return None

    loc = df.index.get_loc(surge_start)
    pre_start = max(0, loc - n_days)
    pre_df = df.iloc[pre_start:loc+1].copy()

    print(f"\n{'='*60}")
    print(f"[{name}] 폭등 시작일: {surge_start.date()}")
    print(f"  사전 {n_days}일 구간: {pre_df.index[0].date()} ~ {pre_df.index[-1].date()}")

    # 구간 기초 통계
    vol_mean = pre_df['Volume'].mean()
    vol_std  = pre_df['Volume'].std()

    print(f"\n  [거래량 통계 — 폭등 전 {n_days}일]")
    print(f"  평균거래량    : {vol_mean:,.0f}")
    print(f"  최대거래량    : {pre_df['Volume'].max():,.0f} ({pre_df['Volume'].idxmax().date()})")
    print(f"  Vol_5/20비율  : {pre_df['Vol_5_20_ratio'].iloc[-1]:.3f} (최종일)")

    # 거래대금 분석
    amt_mean = pre_df['거래대금'].mean()
    print(f"\n  [거래대금 통계 — 폭등 전 {n_days}일]")
    print(f"  평균거래대금   : {amt_mean/1e8:,.1f}억")
    print(f"  최대거래대금   : {pre_df['거래대금'].max()/1e8:,.1f}억 ({pre_df['거래대금'].idxmax().date()})")

    # 매집 패턴 분류: 점진적 증가 vs 스파이크
    # 점진적 증가: 후반 10일 평균 / 전반 10일 평균
    n_half = min(10, len(pre_df)//2)
    if n_half >= 3:
        vol_first_half = pre_df['Volume'].iloc[:n_half].mean()
        vol_second_half = pre_df['Volume'].iloc[-n_half:].mean()
        gradual_ratio = vol_second_half / vol_first_half if vol_first_half > 0 else 0

        # 스파이크: 단일 최대 / 전체 평균
        spike_ratio = pre_df['Volume'].max() / vol_mean if vol_mean > 0 else 0

        pattern = "점진적 매집" if gradual_ratio > 1.3 and spike_ratio < 3.0 else \
                  "스파이크 후 매집" if spike_ratio >= 3.0 else "혼합형"

        print(f"\n  [매집 패턴]")
        print(f"  후반10일/전반10일 비율 : {gradual_ratio:.2f}x  → {'점진적 증가' if gradual_ratio > 1.3 else '변화 없음'}")
        print(f"  최대/평균 스파이크비율  : {spike_ratio:.2f}x")
        print(f"  패턴 판정              : {pattern}")

    # 거래량 임계 돌파일 (20일 평균 × 2배, 3배)
    print(f"\n  [거래량 임계 돌파일 — 폭등 전 {n_days}일 내]")
    for mult in [1.5, 2.0, 3.0]:
        hit = pre_df[pre_df['Vol_day_20_ratio'] >= mult]
        if not hit.empty:
            first_hit = hit.index[0]
            days_before = (surge_start - first_hit).days
            # 영업일 수 계산
            bdays = len(df.loc[first_hit:surge_start]) - 1
            print(f"  Vol ≥ {mult:.1f}×MA20 최초 돌파: {first_hit.date()} (폭등 {bdays}영업일 전, ratio={hit['Vol_day_20_ratio'].iloc[0]:.2f}x)")
        else:
            print(f"  Vol ≥ {mult:.1f}×MA20 최초 돌파: 미달 (해당 없음)")

    # 거래대금 임계 돌파일
    print(f"\n  [거래대금 임계 돌파일]")
    for mult in [1.5, 2.0, 3.0]:
        hit = pre_df[pre_df['Amt_day_20_ratio'] >= mult]
        if not hit.empty:
            first_hit = hit.index[0]
            bdays = len(df.loc[first_hit:surge_start]) - 1
            amt_b = hit['거래대금'].iloc[0] / 1e8
            print(f"  거래대금 ≥ {mult:.1f}×MA20 최초 돌파: {first_hit.date()} (폭등 {bdays}영업일 전, {amt_b:.0f}억)")
        else:
            print(f"  거래대금 ≥ {mult:.1f}×MA20 최초 돌파: 미달")

    # 연속 증가 최대 구간
    max_consec = pre_df['Vol_consec_inc'].max()
    max_consec_end = pre_df['Vol_consec_inc'].idxmax()
    print(f"\n  [연속 거래량 증가]")
    print(f"  최대 연속 증가 : {max_consec}일 (종료일: {max_consec_end.date()})")

    return {
        'name': name,
        'surge_start': surge_start,
        'vol_mean_pre': vol_mean,
        'amt_mean_pre_b': amt_mean/1e8,
        'gradual_ratio': gradual_ratio if n_half >= 3 else None,
        'spike_ratio': spike_ratio if n_half >= 3 else None,
        'max_consec_inc': max_consec,
    }

def cross_stock_timing(data):
    """선도-후행 시차 분석"""
    print(f"\n{'='*60}")
    print("[선도-후행 거래량 시차 분석]")

    results = {}
    for code, name in STOCKS.items():
        df = data[code]
        # Vol ≥ 2×MA20 첫 돌파일
        hit = df[df['Vol_day_20_ratio'] >= 2.0]
        # 2월 이후로 제한
        hit = hit[hit.index >= '2026-02-01']
        if not hit.empty:
            first_hit = hit.index[0]
            results[code] = {'name': name, 'first_2x': first_hit, 'ratio': hit['Vol_day_20_ratio'].iloc[0]}
        else:
            results[code] = {'name': name, 'first_2x': None, 'ratio': None}

    print(f"\n  Vol ≥ 2×MA20 최초 돌파일 (2026-02 이후):")
    for code, r in results.items():
        if r['first_2x']:
            print(f"  {r['name']:12s}: {r['first_2x'].date()} (당일비율 {r['ratio']:.2f}x)")
        else:
            print(f"  {r['name']:12s}: 해당 없음")

    # 삼성전기 기준 시차
    if results['009150']['first_2x'] and results['001820']['first_2x']:
        delta_820 = (results['001820']['first_2x'] - results['009150']['first_2x']).days
        print(f"\n  삼성전기 → 삼화콘덴서 시차: {delta_820}일 ({delta_820}일 후)")
    if results['009150']['first_2x'] and results['031330']['first_2x']:
        delta_330 = (results['031330']['first_2x'] - results['009150']['first_2x']).days
        print(f"  삼성전기 → SAMT 시차       : {delta_330}일 ({delta_330}일 후)")
    if results['009150']['first_2x'] and results['011070']['first_2x']:
        delta_070 = (results['011070']['first_2x'] - results['009150']['first_2x']).days
        print(f"  삼성전기 → LG이노텍 시차   : {delta_070}일 ({delta_070}일 후)")

    return results

def detailed_timeline(data, code, name, start_date, end_date):
    """특정 구간 일별 상세 타임라인"""
    df = data[code]
    mask = (df.index >= start_date) & (df.index <= end_date)
    seg = df[mask].copy()

    print(f"\n{'='*60}")
    print(f"[{name}({code})] {start_date} ~ {end_date} 일별 타임라인")
    print(f"{'날짜':12s} {'종가':>8s} {'거래량':>12s} {'V/MA20':>8s} {'A/MA20':>8s} {'5일MA':>8s} {'거래대금(억)':>10s}")
    print("-"*72)
    for idx, row in seg.iterrows():
        flag = ""
        if row['Vol_day_20_ratio'] >= 3.0:
            flag = " ★★★"
        elif row['Vol_day_20_ratio'] >= 2.0:
            flag = " ★★"
        elif row['Vol_day_20_ratio'] >= 1.5:
            flag = " ★"
        print(f"{str(idx.date()):12s} {row['Close']:>8,.0f} {row['Volume']:>12,.0f} "
              f"{row['Vol_day_20_ratio']:>8.2f} {row['Amt_day_20_ratio']:>8.2f} "
              f"{row['Vol_5_20_ratio']:>8.3f} {row['거래대금']/1e8:>10.1f}{flag}")

def obv_analysis(data, code, name, start_date, end_date):
    """OBV 분석"""
    df = data[code]
    mask = (df.index >= start_date) & (df.index <= end_date)
    seg = df[mask].copy()

    seg['OBV'] = 0
    for i in range(1, len(seg)):
        if seg['Close'].iloc[i] > seg['Close'].iloc[i-1]:
            seg.iloc[i, seg.columns.get_loc('OBV')] = seg['OBV'].iloc[i-1] + seg['Volume'].iloc[i]
        elif seg['Close'].iloc[i] < seg['Close'].iloc[i-1]:
            seg.iloc[i, seg.columns.get_loc('OBV')] = seg['OBV'].iloc[i-1] - seg['Volume'].iloc[i]
        else:
            seg.iloc[i, seg.columns.get_loc('OBV')] = seg['OBV'].iloc[i-1]

    seg['OBV_MA10'] = seg['OBV'].rolling(10).mean()

    print(f"\n{'='*60}")
    print(f"[{name}({code})] OBV 분석 ({start_date} ~ {end_date})")
    print(f"{'날짜':12s} {'OBV':>15s} {'OBV_MA10':>15s} {'OBV vs MA10':>12s}")
    print("-"*55)
    for idx, row in seg.iterrows():
        if pd.notna(row['OBV_MA10']):
            diff = row['OBV'] - row['OBV_MA10']
            flag = " [상향]" if diff > 0 else ""
            print(f"{str(idx.date()):12s} {row['OBV']:>15,.0f} {row['OBV_MA10']:>15,.0f} {diff:>+12,.0f}{flag}")

def derive_rules(data):
    """조기 포착 거래량 룰 도출"""
    print(f"\n{'='*60}")
    print("[조기 포착 거래량 룰 도출 — 폭등 8~10일 전 기준]")

    for code, name in STOCKS.items():
        df = data[code]
        # 2월 폭등 구간 탐색: 2026-02-01 이후 30일 윈도우
        surge_start, surge_end, peak_ret = find_surge_start(df[df.index >= '2026-01-15'])
        if surge_start is None:
            print(f"\n  {name}: 폭등 시점 미탐지")
            continue

        print(f"\n  [{name}] 폭등 시작: {surge_start.date()}, 최고 수익률: {peak_ret*100:.1f}%")

        loc = df.index.get_loc(surge_start)
        # 8~10영업일 전
        target_loc = max(0, loc - 10)
        target_date = df.index[target_loc]

        row = df.iloc[target_loc]
        print(f"  폭등 10영업일 전({target_date.date()}) 지표:")
        print(f"    종가={row['Close']:,.0f}, Vol={row['Volume']:,.0f}")
        print(f"    Vol/MA20={row['Vol_day_20_ratio']:.2f}x, Vol5/MA20={row['Vol_5_20_ratio']:.3f}")
        print(f"    거래대금={row['거래대금']/1e8:.0f}억, 거래대금/MA20={row['Amt_day_20_ratio']:.2f}x")

        # 8~10일 전 구간에서 첫 매집 신호일
        window_10 = df.iloc[max(0, loc-15):loc]
        hit_15 = window_10[window_10['Vol_day_20_ratio'] >= 1.5]
        hit_20 = window_10[window_10['Vol_day_20_ratio'] >= 2.0]

        if not hit_15.empty:
            bdays = len(df.loc[hit_15.index[0]:surge_start]) - 1
            print(f"    Vol≥1.5×MA20 첫 신호: {hit_15.index[0].date()} (폭등 {bdays}영업일 전)")
        if not hit_20.empty:
            bdays = len(df.loc[hit_20.index[0]:surge_start]) - 1
            print(f"    Vol≥2.0×MA20 첫 신호: {hit_20.index[0].date()} (폭등 {bdays}영업일 전)")

def main():
    print("MLCC/전자부품 폭등 전 거래량 매집 신호 심화 분석")
    print(f"기간: {START} ~ {END}")
    print("="*60)

    data = load_data()

    # ─── 1. 폭등 시작일 탐지 + 사전 매집 패턴 ───
    print(f"\n{'='*60}")
    print("[1단계] 폭등 시작일 탐지 (누적 +30% 기준, 2026-01-15 이후)")
    surge_info = {}
    for code, name in STOCKS.items():
        df_from_jan = data[code][data[code].index >= '2026-01-15']
        s, e, ret = find_surge_start(df_from_jan)
        if s:
            print(f"  {name}({code}): 폭등 시작={s.date()}, 고점 도달={e.date()}, 수익률={ret*100:.1f}%")
            surge_info[code] = s
        else:
            print(f"  {name}({code}): 탐지 실패")

    # ─── 2. 각 종목 상세 사전 분석 ───
    for code, name in STOCKS.items():
        if code in surge_info:
            analyze_pre_surge(data[code], name, surge_info[code], n_days=25)

    # ─── 3. 선도-후행 시차 ───
    cross_stock_timing(data)

    # ─── 4. 일별 타임라인 (1/20~3/10) ───
    for code, name in STOCKS.items():
        detailed_timeline(data, code, name, '2026-01-20', '2026-03-10')

    # ─── 5. OBV 분석 ───
    for code, name in STOCKS.items():
        obv_analysis(data, code, name, '2026-01-20', '2026-03-15')

    # ─── 6. 조기 포착 룰 ───
    derive_rules(data)

    # ─── 7. 엔진 적합성 평가 요약 ───
    print(f"\n{'='*60}")
    print("[기존 엔진 적합성 평가]")
    print("""
  scan_investor_flow     : 외인/기관 수급 중심. 거래량 비율(Vol/MA20) 미계산 → 보완 필요
  accumulation_tracker   : OBV + 거래량 추세 추적. 매집 패턴 정량화 가능 → 핵심 활용 가능
  volume_power_tracker   : 체결강도 기반. Vol_5/MA20 비율 임계(90+) 추적 → 가장 직접적 활용 가능

  공통 보완 필요 항목:
    1) Vol/MA20 ≥ 1.5 연속 3일 이상 조건
    2) 거래대금 임계 돌파 조건 (소형주 필수)
    3) 선도주(삼성전기) 신호 발생 시 후행주 자동 스캔 로직
""")

if __name__ == '__main__':
    main()
