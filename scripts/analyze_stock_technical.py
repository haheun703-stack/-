"""
에스제이그룹(306040) 기술적 분석 스크립트
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_stock(ticker='306040'):
    # 데이터 로드
    df = pd.read_parquet(f'd:/sub-agent-project/data/processed/{ticker}.parquet')

    # 최근 30일 데이터
    df_recent = df.tail(30).copy()

    print('='*80)
    print(f'에스제이그룹 ({ticker}) 기술적 분석')
    print('='*80)
    print(f'\n분석 기준일: {df.index[-1]}')
    print(f'현재가: {df["Close"].iloc[-1]:,.0f}원')
    print(f'전일대비: {df["Close"].iloc[-1] - df["Close"].iloc[-2]:+,.0f}원 ({(df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100:+.2f}%)')

    # 1. 가격 추이 및 이동평균선
    print('\n' + '='*80)
    print('[1] 최근 10일 가격 추이 및 이동평균선')
    print('='*80)
    recent_summary = df_recent[['Close', 'MA5', 'MA20', 'MA60']].tail(10)
    print(recent_summary.to_string())

    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    current_price = df['Close'].iloc[-1]

    print(f'\n현재 MA5: {ma5:,.0f}원 (현재가 대비 {(current_price/ma5-1)*100:+.2f}%)')
    print(f'현재 MA20: {ma20:,.0f}원 (현재가 대비 {(current_price/ma20-1)*100:+.2f}%)')
    print(f'현재 MA60: {ma60:,.0f}원 (현재가 대비 {(current_price/ma60-1)*100:+.2f}%)')

    # 이동평균선 배열
    if ma5 > ma20 > ma60:
        ma_status = '✅ 정배열 (강세)'
    elif ma5 < ma20 < ma60:
        ma_status = '🚨 역배열 (약세)'
    else:
        ma_status = '⚠️ 혼조'
    print(f'\n이동평균선 배열: {ma_status}')

    # 골든크로스/데드크로스 체크 (최근 10일)
    gc_dc_status = []
    for i in range(-10, 0):
        if i == -len(df):
            continue
        if df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1] and df['MA5'].iloc[i] > df['MA20'].iloc[i]:
            gc_dc_status.append(f'{df.index[i]} 골든크로스 발생')
        elif df['MA5'].iloc[i-1] >= df['MA20'].iloc[i-1] and df['MA5'].iloc[i] < df['MA20'].iloc[i]:
            gc_dc_status.append(f'{df.index[i]} 데드크로스 발생')

    if gc_dc_status:
        print('최근 크로스: ' + ', '.join(gc_dc_status))
    else:
        print('최근 10일 내 MA 크로스 없음')

    # 2. 기술적 지표
    print('\n' + '='*80)
    print('[2] 기술적 지표 분석')
    print('='*80)

    # RSI
    rsi = df['RSI'].iloc[-1]
    print(f'RSI(14): {rsi:.1f}')
    if rsi > 70:
        rsi_status = '🚨 과매수 (조정 가능)'
    elif rsi > 50:
        rsi_status = '✅ 중립~강세'
    elif rsi > 30:
        rsi_status = '⚠️ 중립~약세'
    else:
        rsi_status = '✅ 과매도 (반등 가능)'
    print(f'  → 해석: {rsi_status}')

    # ADX
    adx = df['ADX'].iloc[-1]
    print(f'\nADX(14): {adx:.1f}')
    if adx > 40:
        adx_status = '✅ 강한 추세'
    elif adx > 25:
        adx_status = '✅ 추세 형성'
    else:
        adx_status = '⚠️ 박스권/약한 추세'
    print(f'  → 해석: {adx_status}')

    # MACD
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    print(f'\nMACD: {macd:.2f}')
    print(f'Signal: {macd_signal:.2f}')
    print(f'Histogram: {macd_hist:.2f}')

    if macd > macd_signal and macd_hist > 0:
        macd_status = '✅ 매수 신호'
    elif macd < macd_signal and macd_hist < 0:
        macd_status = '🚨 매도 신호'
    else:
        macd_status = '⚠️ 중립'
    print(f'  → 해석: {macd_status}')

    # MACD 크로스 체크 (최근 10일)
    macd_cross = []
    for i in range(-10, 0):
        if i == -len(df):
            continue
        if df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            macd_cross.append(f'{df.index[i]} MACD 골든크로스')
        elif df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]:
            macd_cross.append(f'{df.index[i]} MACD 데드크로스')

    if macd_cross:
        print('최근 MACD 크로스: ' + ', '.join(macd_cross))
    else:
        print('최근 10일 내 MACD 크로스 없음')

    # 3. 볼린저 밴드
    print('\n' + '='*80)
    print('[3] 볼린저 밴드 분석')
    print('='*80)

    bb_upper = df['BB_Upper'].iloc[-1]
    bb_middle = df['BB_Middle'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]

    print(f'상단: {bb_upper:,.0f}원')
    print(f'중간(MA20): {bb_middle:,.0f}원')
    print(f'하단: {bb_lower:,.0f}원')
    print(f'현재가: {current_price:,.0f}원')

    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
    print(f'\n밴드 내 위치: {bb_position:.1f}%')

    if bb_position > 80:
        bb_status = '🚨 상단 근접 (과매수 가능)'
    elif bb_position > 50:
        bb_status = '✅ 중상단 (강세)'
    elif bb_position > 20:
        bb_status = '⚠️ 중하단 (약세)'
    else:
        bb_status = '✅ 하단 근접 (반등 가능)'
    print(f'  → 해석: {bb_status}')

    # 4. 거래량 분석
    print('\n' + '='*80)
    print('[4] 거래량 분석')
    print('='*80)

    volume = df['Volume'].iloc[-1]
    volume_ma5 = df['Volume'].iloc[-5:].mean()
    volume_ma20 = df['Volume'].iloc[-20:].mean()

    print(f'금일 거래량: {volume:,.0f}')
    print(f'5일 평균: {volume_ma5:,.0f} (대비 {(volume/volume_ma5-1)*100:+.1f}%)')
    print(f'20일 평균: {volume_ma20:,.0f} (대비 {(volume/volume_ma20-1)*100:+.1f}%)')

    if volume > volume_ma20 * 1.5:
        volume_status = '✅ 거래 급증 (관심 증가)'
    elif volume > volume_ma20:
        volume_status = '✅ 평균 이상'
    else:
        volume_status = '⚠️ 평균 이하 (관심 약함)'
    print(f'  → 해석: {volume_status}')

    # OBV
    if 'OBV' in df.columns:
        obv = df['OBV'].iloc[-1]
        obv_prev = df['OBV'].iloc[-6]
        obv_change = (obv / obv_prev - 1) * 100

        print(f'\nOBV: {obv:,.0f}')
        print(f'5일 전 대비: {obv_change:+.1f}%')

        if obv_change > 5:
            obv_status = '✅ OBV 상승 (매수세 우세)'
        elif obv_change < -5:
            obv_status = '🚨 OBV 하락 (매도세 우세)'
        else:
            obv_status = '⚠️ OBV 보합'
        print(f'  → 해석: {obv_status}')

    # 5. 지지/저항 분석
    print('\n' + '='*80)
    print('[5] 지지/저항 분석 (최근 60일)')
    print('='*80)

    df_60 = df.tail(60)
    high_max = df_60['High'].max()
    low_min = df_60['Low'].min()

    print(f'60일 최고가: {high_max:,.0f}원 (현재가 대비 {(high_max/current_price-1)*100:+.1f}%)')
    print(f'60일 최저가: {low_min:,.0f}원 (현재가 대비 {(current_price/low_min-1)*100:+.1f}%)')

    # 매물대 분석 (가격 구간별 거래량)
    price_bins = pd.cut(df_60['Close'], bins=10)
    volume_by_price = df_60.groupby(price_bins)['Volume'].sum().sort_values(ascending=False)

    print('\n매물대 (거래량 기준 상위 3개 구간):')
    for i, (price_range, vol) in enumerate(volume_by_price.head(3).items(), 1):
        print(f'{i}. {price_range}: 거래량 {vol:,.0f}')

    # 6. 종합 판단
    print('\n' + '='*80)
    print('[6] 종합 판단')
    print('='*80)

    # 점수 계산
    score = 0
    reasons = []

    # MA 배열 (30점)
    if ma5 > ma20 > ma60:
        score += 30
        reasons.append('✅ 이동평균선 정배열')
    elif ma5 > ma20:
        score += 15
        reasons.append('⚠️ 단기 상승세')
    else:
        reasons.append('🚨 이동평균선 약세')

    # RSI (20점)
    if 40 <= rsi <= 60:
        score += 20
        reasons.append('✅ RSI 적정 수준')
    elif 30 <= rsi < 40:
        score += 15
        reasons.append('⚠️ RSI 약세이나 반등 가능')
    elif rsi > 70:
        reasons.append('🚨 RSI 과매수')
    else:
        score += 10

    # MACD (20점)
    if macd > macd_signal and macd_hist > 0:
        score += 20
        reasons.append('✅ MACD 매수 신호')
    elif macd > macd_signal:
        score += 10
        reasons.append('⚠️ MACD 상승 전환')
    else:
        reasons.append('🚨 MACD 약세')

    # 거래량 (15점)
    if volume > volume_ma20 * 1.2:
        score += 15
        reasons.append('✅ 거래량 증가')
    elif volume > volume_ma20:
        score += 10
        reasons.append('⚠️ 거래량 평균 이상')
    else:
        reasons.append('⚠️ 거래량 부족')

    # ADX (15점)
    if adx > 25:
        score += 15
        reasons.append('✅ 추세 강함')
    else:
        score += 5
        reasons.append('⚠️ 추세 약함')

    print(f'기술적 분석 점수: {score}/100점')
    print('\n주요 판단 근거:')
    for reason in reasons:
        print(f'  {reason}')

    # 매수 적합성
    print('\n' + '-'*80)
    if score >= 70:
        print('📊 매수 적합성: ✅ 양호')
        print('   - 기술적으로 긍정적인 신호가 우세합니다.')
    elif score >= 50:
        print('📊 매수 적합성: ⚠️ 보통')
        print('   - 일부 긍정적 신호가 있으나 신중한 접근이 필요합니다.')
    else:
        print('📊 매수 적합성: 🚨 부정적')
        print('   - 기술적으로 약세 신호가 많습니다. 진입 보류 권장.')

    # 진입 타이밍
    print('\n진입 타이밍:')
    if current_price < ma5:
        print(f'  - MA5({ma5:,.0f}원) 돌파 시 단기 진입')
    if current_price < ma20:
        print(f'  - MA20({ma20:,.0f}원) 돌파 시 중기 진입')
    if bb_position < 30:
        print(f'  - 볼린저 밴드 하단 근처로 반등 진입 가능')
    if rsi < 40:
        print(f'  - RSI 과매도 구간으로 반등 대기')

    # 손익 목표 (제공된 정보 기반)
    target_price = 14430
    stop_loss = 9476

    print('\n손익 목표:')
    print(f'  - 목표가: {target_price:,.0f}원 (+{(target_price/current_price-1)*100:.1f}%)')
    print(f'  - 손절가: {stop_loss:,.0f}원 ({(stop_loss/current_price-1)*100:.1f}%)')
    print(f'  - Risk/Reward: 1:15.9 (매우 양호)')

    # 주의점
    print('\n' + '='*80)
    print('[7] 주의점 및 대응 전략')
    print('='*80)

    print('\n✅ 유지해도 되는 조건:')
    print(f'  1. 현재가가 MA20({ma20:,.0f}원) 위에서 유지')
    print(f'  2. RSI가 30 이상 유지')
    print(f'  3. 거래량이 20일 평균({volume_ma20:,.0f}) 이상 유지')
    print(f'  4. MACD Histogram이 0선 위 유지')
    print(f'  5. OBV 상승 추세 유지')

    print('\n🚨 반드시 대응해야 할 조건:')
    print(f'  1. 손절가({stop_loss:,.0f}원) 이탈 시 즉시 청산')
    print(f'  2. MA20 하향 이탈 + 거래량 급증 시 주의')
    print(f'  3. RSI 30 이하 하락 시 추가 하락 가능성 대비')
    print(f'  4. MACD 데드크로스 발생 시 익절/손절 판단')
    print(f'  5. 볼린저 밴드 하단 이탈 시 추가 하락 경계')

    print('\n내일 흐름 예측:')

    # 상승 가능성
    if score >= 60 and macd > macd_signal and rsi > 45:
        print('  ✅ 상승 가능성 높음')
        print(f'    - 목표: {ma20 * 1.03:,.0f}원 ~ {bb_upper:,.0f}원')
    elif score >= 50:
        print('  ⚠️ 보합~약 상승 예상')
        print(f'    - 박스권: {ma20:,.0f}원 ~ {ma5 * 1.02:,.0f}원')
    else:
        print('  🚨 조정 가능성')
        print(f'    - 지지선: {bb_lower:,.0f}원 ~ {ma60:,.0f}원')

    print('\n' + '='*80)
    print('분석 완료')
    print('='*80)

if __name__ == '__main__':
    analyze_stock('306040')
