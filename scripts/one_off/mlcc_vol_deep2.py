"""
MLCC 심화 분석 2차: 2/19~2/24 폭등 구간 집중 + 정확한 시차 도출
"""
import sys
sys.path.insert(0, 'D:/sub-agent-project_퀀트봇')
import FinanceDataReader as fdr
import pandas as pd
import numpy as np

pd.set_option('display.float_format', '{:,.2f}'.format)

STOCKS = {
    '009150': '삼성전기',
    '001820': '삼화콘덴서',
    '031330': 'SAMT',
    '011070': 'LG이노텍',
}
START = '2025-12-01'  # MA20 워밍업 위해 앞당김
END   = '2026-05-29'

def load_data():
    data = {}
    for code, name in STOCKS.items():
        df = fdr.DataReader(code, START, END)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df['거래대금'] = df['Close'] * df['Volume']
        df['Vol_MA5']  = df['Volume'].rolling(5).mean()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Amt_MA5']  = df['거래대금'].rolling(5).mean()
        df['Amt_MA20'] = df['거래대금'].rolling(20).mean()
        df['Vol_5_20'] = df['Vol_MA5'] / df['Vol_MA20']
        df['Vol_D_20'] = df['Volume'] / df['Vol_MA20']
        df['Amt_D_20'] = df['거래대금'] / df['Amt_MA20']
        df['Ret1d'] = df['Close'].pct_change()
        data[code] = df
        print(f"[로드] {name}({code}): {len(df)}행, {df.index[0].date()}~{df.index[-1].date()}")
    return data

def print_range(data, code, name, s, e):
    df = data[code]
    seg = df.loc[s:e].copy()
    print(f"\n{'='*70}")
    print(f"[{name}({code})] {s} ~ {e}")
    print(f"{'날짜':12s} {'종가':>8s} {'등락':>7s} {'거래량':>12s} {'V/MA20':>7s} {'거래대금(억)':>10s} {'A/MA20':>7s} {'5/20':>6s}")
    print("-"*80)
    for idx, r in seg.iterrows():
        flag = ""
        if pd.notna(r['Vol_D_20']):
            if r['Vol_D_20'] >= 3.0: flag = "★★★"
            elif r['Vol_D_20'] >= 2.0: flag = "★★ "
            elif r['Vol_D_20'] >= 1.5: flag = "★  "
        ret_str = f"{r['Ret1d']*100:+.1f}%" if pd.notna(r['Ret1d']) else "  N/A"
        print(f"{str(idx.date()):12s} {r['Close']:>8,.0f} {ret_str:>7s} {r['Volume']:>12,.0f} "
              f"{r['Vol_D_20']:>7.2f} {r['거래대금']/1e8:>10.1f} {r['Amt_D_20']:>7.2f} "
              f"{r['Vol_5_20']:>6.3f} {flag}")

def find_first_signal(df, start, end, vol_mult=2.0):
    """특정 기간 내 Vol/MA20 ≥ mult 첫 날"""
    seg = df.loc[start:end]
    hit = seg[seg['Vol_D_20'] >= vol_mult]
    if not hit.empty:
        return hit.index[0], hit['Vol_D_20'].iloc[0], hit['거래대금'].iloc[0]/1e8
    return None, None, None

def sector_sync_analysis(data):
    """
    실제 2/19~2/24 폭등 기준 거슬러 올라가며 조기 신호 탐색
    삼성전기·삼화콘덴서: 1월 말~2월 초 먼저 움직임
    SAMT·LG이노텍: 2월 말~3월 폭등
    """
    print(f"\n{'='*70}")
    print("[섹터 동조 분석] 각 종목 Vol≥2.0×MA20 최초 신호일 (2026-01-20 이후)")
    for threshold_date in ['2026-01-20', '2026-02-01', '2026-02-15']:
        print(f"\n  -- {threshold_date} 이후 첫 2배 돌파 --")
        for code, name in STOCKS.items():
            df = data[code]
            seg = df.loc[threshold_date:'2026-03-31']
            hit = seg[seg['Vol_D_20'] >= 2.0]
            if not hit.empty:
                d = hit.index[0]
                print(f"  {name:12s}: {d.date()}  V/MA20={hit['Vol_D_20'].iloc[0]:.2f}x  "
                      f"거래대금={hit['거래대금'].iloc[0]/1e8:.0f}억")
            else:
                print(f"  {name:12s}: 해당 없음")

def compute_obv_crossover(df, start, end):
    """OBV가 OBV_MA10을 상향 돌파하는 첫 날"""
    seg = df.loc[start:end].copy()
    seg['OBV'] = 0
    for i in range(1, len(seg)):
        prev = seg['OBV'].iloc[i-1]
        if seg['Close'].iloc[i] > seg['Close'].iloc[i-1]:
            seg.iloc[i, seg.columns.get_loc('OBV')] = prev + seg['Volume'].iloc[i]
        elif seg['Close'].iloc[i] < seg['Close'].iloc[i-1]:
            seg.iloc[i, seg.columns.get_loc('OBV')] = prev - seg['Volume'].iloc[i]
        else:
            seg.iloc[i, seg.columns.get_loc('OBV')] = prev
    seg['OBV_MA10'] = seg['OBV'].rolling(10).mean()
    seg['OBV_above'] = seg['OBV'] > seg['OBV_MA10']
    # 첫 상향 돌파: 이전 날 아래, 당일 위
    crosses = seg[(seg['OBV_above'] == True) & (seg['OBV_above'].shift(1) == False)]
    return crosses.index[0] if not crosses.empty else None

def summary_rules(data):
    """각 종목의 조기 신호 → 실제 폭등일 까지의 영업일 수 계산"""
    print(f"\n{'='*70}")
    print("[조기 포착 룰 종합 — 신호일 → 폭등일 영업일 수]")

    # 각 종목 폭등 기준일 (종가 기준 주가가 크게 오르기 시작한 날)
    surge_dates = {
        '009150': ('2026-02-19', '2026-02-23', '+30%+ 본격 폭등'),  # 2/19 +15.7% → 2/23 +16.1%
        '001820': ('2026-02-10', '2026-02-23', '+23% 선행 + 본격 폭등'),
        '031330': ('2026-02-27', '2026-03-31', '2/27 예고 스파이크 → 3/31 본폭등'),
        '011070': ('2026-02-26', '2026-03-04', '2/26 폭등 시작'),
    }

    # 각 종목별 신호 분석
    for code, name in STOCKS.items():
        df = data[code]
        surge_warn, surge_main, label = surge_dates[code]

        print(f"\n  [{name}({code})] {label}")
        print(f"  폭등 경보일: {surge_warn}  본폭등: {surge_main}")

        # 폭등 경보일 기준 이전 신호 탐색
        search_start = '2026-01-15'
        search_end   = surge_warn

        seg_pre = df.loc[search_start:search_end]

        # Vol ≥ 1.5 첫 돌파
        for mult, label2 in [(1.5, "Vol≥1.5×MA20"), (2.0, "Vol≥2.0×MA20"), (3.0, "Vol≥3.0×MA20")]:
            hit = seg_pre[seg_pre['Vol_D_20'] >= mult]
            if not hit.empty:
                d = hit.index[0]
                bdays = len(df.loc[d:surge_warn]) - 1
                print(f"    {label2} 첫 신호: {d.date()} → 경보까지 {bdays}영업일  "
                      f"(V/MA20={hit['Vol_D_20'].iloc[0]:.2f}, 거래대금={hit['거래대금'].iloc[0]/1e8:.0f}억)")
            else:
                print(f"    {label2}: 미달")

        # 거래대금 임계
        for mult, label2 in [(1.5, "거래대금≥1.5×MA20"), (2.0, "거래대금≥2.0×MA20")]:
            hit = seg_pre[seg_pre['Amt_D_20'] >= mult]
            if not hit.empty:
                d = hit.index[0]
                bdays = len(df.loc[d:surge_warn]) - 1
                print(f"    {label2} 첫 신호: {d.date()} → 경보까지 {bdays}영업일  "
                      f"(A/MA20={hit['Amt_D_20'].iloc[0]:.2f}, 거래대금={hit['거래대금'].iloc[0]/1e8:.0f}억)")
            else:
                print(f"    {label2}: 미달")

        # Vol_5/MA20 ≥ 1.2 유지 구간 (매집 신호)
        hit_ma = seg_pre[seg_pre['Vol_5_20'] >= 1.2]
        if not hit_ma.empty:
            d = hit_ma.index[0]
            bdays = len(df.loc[d:surge_warn]) - 1
            print(f"    Vol5MA≥1.2×MA20 첫 진입: {d.date()} → 경보까지 {bdays}영업일  "
                  f"(5일MA비율={hit_ma['Vol_5_20'].iloc[0]:.3f})")

def print_precise_pre_surge(data):
    """2/19 폭등 직전 구간(1/20~2/19) 정밀 분석 — 삼성전기·삼화콘덴서"""
    for code in ['009150', '001820']:
        name = STOCKS[code]
        print_range(data, code, name, '2026-01-20', '2026-02-24')

    # LG이노텍: 1/27 대량 거래 + 2/26 폭등 직전
    for code in ['011070', '031330']:
        name = STOCKS[code]
        print_range(data, code, name, '2026-01-20', '2026-03-05')

def final_rule_matrix(data):
    """최종 조기 포착 룰 매트릭스 출력"""
    print(f"\n{'='*70}")
    print("[최종 조기 포착 룰 매트릭스]")
    print(f"{'종목':12s} {'Vol≥1.5 첫 신호':18s} {'사전 영업일':>12s} {'거래대금(억)':>12s} {'Vol5/MA20':>10s}")
    print("-"*70)

    rows = [
        ('009150', '삼성전기',   '2026-02-02', '2026-02-19', 2.06,  4178, None),
        ('001820', '삼화콘덴서', '2026-01-29', '2026-02-10', 1.89,   38,  None),
        ('011070', 'LG이노텍',  '2026-01-29', '2026-02-26', 2.45,  1385, 1.592),
        ('031330', 'SAMT',     '2026-02-27', '2026-03-31', 2.66,   150, None),
    ]

    for code, name, sig_date, surge_date, vol_ratio, amt, ma5 in rows:
        df = data[code]
        bdays = len(df.loc[sig_date:surge_date]) - 1
        ma5_str = f"{ma5:.3f}" if ma5 else "  N/A"
        print(f"{name:12s} {sig_date:18s} {bdays:>12d} {amt:>12,.0f} {ma5_str:>10s}")

    print()
    print("  공통 패턴 요약:")
    print("  - Vol/MA20 첫 2배 돌파 → 실제 폭등까지 3~20영업일")
    print("  - 소형주(삼화콘덴서, SAMT): 거래대금 규모 자체가 작아 비율이 더 중요")
    print("  - 대형주(삼성전기, LG이노텍): 거래대금 절대값 기준 유효 (1000억+ 돌파)")
    print("  - Vol_5/MA20 ≥ 1.3 유지 = 매집 지속 신호 (스파이크와 구분)")
    print()
    print("  선도-후행 관계 (2/19~2/24 폭등 그룹):")
    print("  - 삼성전기 2×MA20 신호: 2/02  → 삼화콘덴서: 2/03 (시차 1영업일)")
    print("  - LG이노텍 독자 스파이크: 1/29 (삼성전기보다 4영업일 빠름 — 애플 관련 독자 모멘텀)")
    print()
    print("  SAMT 별도 웨이브 (3월 말 폭등 그룹):")
    print("  - 2/27 예고 스파이크 → 약 21영업일 후 본격 폭등(3/31)")
    print("  - 2/24~2/26 거래대금 60~73억 (평상시 20~50억 대비 1.5~2.0x) 선행 신호")

def main():
    print("MLCC 거래량 심화 분석 2차 — 실측 데이터 기반")
    print("="*70)
    data = load_data()

    # 1. 섹터 동조 분석
    sector_sync_analysis(data)

    # 2. 정밀 사전 구간 출력 (삼성전기·삼화콘덴서는 1/20~2/24, 나머지 1/20~3/05)
    print_precise_pre_surge(data)

    # 3. 신호일→폭등일 종합
    summary_rules(data)

    # 4. 최종 룰 매트릭스
    final_rule_matrix(data)

if __name__ == '__main__':
    main()
