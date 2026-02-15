"""
HJ중공업 (097230) 기술적 분석 스크립트
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 데이터 로드
df = pd.read_parquet('d:/sub-agent-project/data/processed/097230.parquet')

# 최근 30일 데이터
df_recent = df.tail(30).copy()

print('━' * 80)
print('HJ중공업 (097230) 기술적 분석 보고서')
print('━' * 80)
print(f'\n분석 기간: {df_recent.index[0]} ~ {df_recent.index[-1]}')
print(f'데이터 행 수: {len(df_recent)}개')

# 최신 데이터
latest = df_recent.iloc[-1]
prev = df_recent.iloc[-2]

print('\n' + '=' * 80)
print('업체 정보')
print('=' * 80)
print(f'종목명: HJ중공업')
print(f'종목코드: 097230')
print(f'현재가: {latest["close"]:,.0f}원')
print(f'전일 대비: {((latest["close"]/prev["close"]-1)*100):+.2f}%')
print(f'거래량: {latest["volume"]:,.0f}주')

# 1. 이동평균선 분석
print('\n' + '=' * 80)
print('기술적 분석 - 이동평균선')
print('=' * 80)
print(f'MA5:  {latest["ma5"]:>10,.0f}원 | 현재가 대비: {((latest["close"]/latest["ma5"]-1)*100):>6.2f}%')
print(f'MA20: {latest["ma20"]:>10,.0f}원 | 현재가 대비: {((latest["close"]/latest["ma20"]-1)*100):>6.2f}%')
print(f'MA60: {latest["ma60"]:>10,.0f}원 | 현재가 대비: {((latest["close"]/latest["ma60"]-1)*100):>6.2f}%')

# 이동평균선 배열
if latest['ma5'] > latest['ma20'] > latest['ma60']:
    ma_align = '✓ 정배열 (상승추세)'
    ma_signal = '매수'
elif latest['ma5'] < latest['ma20'] < latest['ma60']:
    ma_align = '✗ 역배열 (하락추세)'
    ma_signal = '매도'
else:
    ma_align = '△ 혼조 (방향성 불명확)'
    ma_signal = '관망'

print(f'\n이동평균선 배열: {ma_align}')
print(f'신호: {ma_signal}')

# 골든크로스/데드크로스 확인
recent_5 = df_recent.tail(5)
cross_info = []
for i in range(1, len(recent_5)):
    prev_day = recent_5.iloc[i-1]
    curr_day = recent_5.iloc[i]

    if prev_day['ma5'] <= prev_day['ma20'] and curr_day['ma5'] > curr_day['ma20']:
        cross_info.append(f'✓ 골든크로스 발생: {recent_5.index[i]} (MA5 > MA20)')
    elif prev_day['ma5'] >= prev_day['ma20'] and curr_day['ma5'] < curr_day['ma20']:
        cross_info.append(f'✗ 데드크로스 발생: {recent_5.index[i]} (MA5 < MA20)')

if cross_info:
    print('\n최근 크로스:')
    for info in cross_info:
        print(f'  {info}')

# 2. 기술적 지표 분석
print('\n' + '=' * 80)
print('기술적 분석 - 보조지표')
print('=' * 80)

# RSI
print(f'\nRSI(14): {latest["rsi"]:.2f}')
if latest['rsi'] >= 70:
    rsi_status = '⚠ 과매수 (조정 가능성)'
    rsi_signal = '매도 관망'
elif latest['rsi'] <= 30:
    rsi_status = '✓ 과매도 (반등 가능성)'
    rsi_signal = '매수 기회'
elif 50 <= latest['rsi'] < 70:
    rsi_status = '△ 상승 모멘텀'
    rsi_signal = '매수 유지'
else:
    rsi_status = '△ 중립'
    rsi_signal = '관망'
print(f'  상태: {rsi_status}')
print(f'  신호: {rsi_signal}')

# ADX
print(f'\nADX: {latest["adx"]:.2f}')
if latest['adx'] >= 25:
    adx_status = '✓ 강한 추세'
    adx_signal = '추세 추종'
else:
    adx_status = '△ 약한 추세 (박스권)'
    adx_signal = '관망'
print(f'  상태: {adx_status}')
print(f'  신호: {adx_signal}')

# MACD
print(f'\nMACD: {latest["macd"]:.2f}')
print(f'Signal: {latest["macd_signal"]:.2f}')
print(f'Histogram: {latest["macd_hist"]:.2f}')

if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
    macd_status = '✓ 골든크로스 (매수 신호)'
    macd_signal = '매수'
elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
    macd_status = '✗ 데드크로스 (매도 신호)'
    macd_signal = '매도'
else:
    macd_status = '△ 중립'
    macd_signal = '관망'
print(f'  상태: {macd_status}')
print(f'  신호: {macd_signal}')

# 3. 거래량 및 OBV 분석
print('\n' + '=' * 80)
print('거래량 / 매물대 분석')
print('=' * 80)

print(f'\n현재 거래량: {latest["volume"]:,.0f}주')
print(f'거래량 MA5:  {latest["volume_ma5"]:,.0f}주')
print(f'거래량 MA20: {latest["volume_ma20"]:,.0f}주')

vol_ratio_5 = latest['volume'] / latest['volume_ma5'] if latest['volume_ma5'] > 0 else 0
vol_ratio_20 = latest['volume'] / latest['volume_ma20'] if latest['volume_ma20'] > 0 else 0

print(f'\n거래량 비율 (현재/MA5):  {vol_ratio_5:.2f}배')
print(f'거래량 비율 (현재/MA20): {vol_ratio_20:.2f}배')

if vol_ratio_20 >= 2.0:
    vol_status = '✓ 급등 (강한 관심)'
    vol_signal = '매집/이탈 주의'
elif vol_ratio_20 >= 1.5:
    vol_status = '✓ 증가 (관심 확대)'
    vol_signal = '거래 활성화'
elif vol_ratio_20 >= 1.0:
    vol_status = '△ 보통'
    vol_signal = '정상 범위'
else:
    vol_status = '△ 감소 (관심 축소)'
    vol_signal = '거래 위축'

print(f'  상태: {vol_status}')
print(f'  신호: {vol_signal}')

# OBV 분석
if 'obv' in latest:
    print(f'\nOBV: {latest["obv"]:,.0f}')
    obv_ma5 = df_recent['obv'].rolling(5).mean().iloc[-1]
    obv_ma20 = df_recent['obv'].rolling(20).mean().iloc[-1]
    print(f'OBV MA5:  {obv_ma5:,.0f}')
    print(f'OBV MA20: {obv_ma20:,.0f}')

    if latest['obv'] > obv_ma5 > obv_ma20:
        obv_status = '✓ OBV 상승 (매수세 우위)'
        obv_signal = '매수 신호'
    elif latest['obv'] < obv_ma5 < obv_ma20:
        obv_status = '✗ OBV 하락 (매도세 우위)'
        obv_signal = '매도 신호'
    else:
        obv_status = '△ OBV 혼조'
        obv_signal = '중립'

    print(f'  상태: {obv_status}')
    print(f'  신호: {obv_signal}')

# 4. 볼린저 밴드
print('\n' + '=' * 80)
print('볼린저 밴드')
print('=' * 80)

if 'bb_upper' in latest and 'bb_lower' in latest and 'bb_middle' in latest:
    print(f'\n상단: {latest["bb_upper"]:,.0f}원')
    print(f'중간: {latest["bb_middle"]:,.0f}원')
    print(f'하단: {latest["bb_lower"]:,.0f}원')
    print(f'현재: {latest["close"]:,.0f}원')

    bb_width = latest['bb_upper'] - latest['bb_lower']
    bb_position = (latest['close'] - latest['bb_lower']) / bb_width if bb_width > 0 else 0.5

    print(f'\n밴드 폭: {bb_width:,.0f}원')
    print(f'현재 위치: {bb_position*100:.1f}% (0%=하단, 100%=상단)')

    if bb_position >= 0.8:
        bb_status = '⚠ 상단 근접 (과열, 조정 가능성)'
        bb_signal = '익절/관망'
    elif bb_position <= 0.2:
        bb_status = '✓ 하단 근접 (과매도, 반등 가능성)'
        bb_signal = '매수 기회'
    elif 0.4 <= bb_position <= 0.6:
        bb_status = '△ 중간권 (안정)'
        bb_signal = '중립'
    elif bb_position > 0.6:
        bb_status = '△ 상단 이동 중'
        bb_signal = '상승 진행'
    else:
        bb_status = '△ 하단 이동 중'
        bb_signal = '하락 진행'

    print(f'  상태: {bb_status}')
    print(f'  신호: {bb_signal}')

# 5. 지지/저항 분석
print('\n' + '=' * 80)
print('지지 / 저항 분석 (최근 30일)')
print('=' * 80)

high_max = df_recent['high'].max()
low_min = df_recent['low'].min()
close_max = df_recent['close'].max()
close_min = df_recent['close'].min()

print(f'\n최고가: {high_max:,.0f}원')
print(f'최저가: {low_min:,.0f}원')
print(f'변동폭: {((high_max/low_min-1)*100):.2f}%')

price_position = (latest['close'] - low_min) / (high_max - low_min) if (high_max - low_min) > 0 else 0.5
print(f'\n현재가 위치: {price_position*100:.1f}% (최저 대비)')

# 저항선/지지선 계산 (피봇 포인트)
pivot = (latest['high'] + latest['low'] + latest['close']) / 3
r1 = 2 * pivot - latest['low']
s1 = 2 * pivot - latest['high']

print(f'\n피봇: {pivot:,.0f}원')
print(f'저항선(R1): {r1:,.0f}원 (현재가 대비: {((r1/latest["close"]-1)*100):+.2f}%)')
print(f'지지선(S1): {s1:,.0f}원 (현재가 대비: {((s1/latest["close"]-1)*100):+.2f}%)')

# 6. 최근 5일 가격 추이
print('\n' + '=' * 80)
print('최근 5일 가격 추이')
print('=' * 80)
print()
for idx, row in df_recent.tail(5).iterrows():
    daily_change = ((row['close']/row['open']-1)*100) if row['open'] > 0 else 0
    vol_change = ((row['volume']/row['volume_ma20']-1)*100) if row['volume_ma20'] > 0 else 0
    print(f'{idx}: {row["close"]:>8,.0f}원 ({daily_change:>+6.2f}%) | 거래량: {row["volume"]:>12,.0f}주 ({vol_change:>+6.1f}%)')

# 7. 종합 판단
print('\n' + '=' * 80)
print('핵심 분석 포인트')
print('=' * 80)

# 신호 집계
signals = []

# MA 신호
if latest['ma5'] > latest['ma20'] > latest['ma60']:
    signals.append(('MA배열', '매수', 1))
elif latest['ma5'] < latest['ma20'] < latest['ma60']:
    signals.append(('MA배열', '매도', -1))
else:
    signals.append(('MA배열', '중립', 0))

# RSI 신호
if latest['rsi'] <= 30:
    signals.append(('RSI', '매수', 1))
elif latest['rsi'] >= 70:
    signals.append(('RSI', '매도', -1))
else:
    signals.append(('RSI', '중립', 0))

# MACD 신호
if latest['macd'] > latest['macd_signal']:
    signals.append(('MACD', '매수', 1))
elif latest['macd'] < latest['macd_signal']:
    signals.append(('MACD', '매도', -1))
else:
    signals.append(('MACD', '중립', 0))

# 거래량 신호
if vol_ratio_20 >= 1.5:
    signals.append(('거래량', '활성', 1))
elif vol_ratio_20 < 0.8:
    signals.append(('거래량', '위축', -1))
else:
    signals.append(('거래량', '보통', 0))

# OBV 신호
if 'obv' in latest:
    obv_ma5 = df_recent['obv'].rolling(5).mean().iloc[-1]
    if latest['obv'] > obv_ma5:
        signals.append(('OBV', '매수세', 1))
    else:
        signals.append(('OBV', '매도세', -1))

print('\n신호 요약:')
total_score = 0
for indicator, signal, score in signals:
    total_score += score
    emoji = '✓' if score > 0 else ('✗' if score < 0 else '△')
    print(f'  {emoji} {indicator}: {signal}')

print(f'\n종합 점수: {total_score} / {len(signals)}')

if total_score >= 3:
    overall = '✓ 강한 매수 신호'
    recommendation = '적극 매수'
elif total_score >= 1:
    overall = '△ 약한 매수 신호'
    recommendation = '신중 매수'
elif total_score <= -3:
    overall = '✗ 강한 매도 신호'
    recommendation = '청산 검토'
elif total_score <= -1:
    overall = '△ 약한 매도 신호'
    recommendation = '관망/손절 준비'
else:
    overall = '△ 중립'
    recommendation = '관망'

print(f'\n종합 판단: {overall}')
print(f'투자 의견: {recommendation}')

# 8. 유지/대응 조건
print('\n' + '=' * 80)
print('✅ 유지해도 되는 조건 (Hold Conditions)')
print('=' * 80)

hold_conditions = []

# MA5 위 유지
if latest['close'] > latest['ma5']:
    hold_conditions.append(f'현재가 {latest["close"]:,.0f}원이 MA5({latest["ma5"]:,.0f}원) 위에 유지')

# RSI 30~70 구간
if 30 < latest['rsi'] < 70:
    hold_conditions.append(f'RSI {latest["rsi"]:.1f}가 과열/과매도 구간 벗어남')

# MACD 골든크로스 유지
if latest['macd'] > latest['macd_signal']:
    hold_conditions.append(f'MACD가 Signal선 위 유지 (모멘텀 유지)')

# OBV 상승
if 'obv' in latest:
    obv_ma5 = df_recent['obv'].rolling(5).mean().iloc[-1]
    if latest['obv'] > obv_ma5:
        hold_conditions.append(f'OBV 상승세 유지 (매수세 지속)')

# 거래량
if vol_ratio_20 >= 1.0:
    hold_conditions.append(f'거래량이 MA20 대비 {vol_ratio_20:.2f}배 유지 (관심 지속)')

if hold_conditions:
    for i, cond in enumerate(hold_conditions, 1):
        print(f'{i}. {cond}')
else:
    print('현재 유지 조건 미충족')

print('\n' + '=' * 80)
print('🚨 반드시 대응해야 할 조건 (Action Required)')
print('=' * 80)

action_conditions = []

# 손절선
stop_loss = latest['ma20'] * 0.95  # MA20 대비 -5%
action_conditions.append(f'손절: {stop_loss:,.0f}원 이하 이탈 시 (MA20 -5%)')

# MA5 이탈
action_conditions.append(f'단기 이탈: {latest["ma5"]:,.0f}원(MA5) 하향 이탈 시')

# 데드크로스 발생
action_conditions.append(f'데드크로스: MA5가 MA20 하향 돌파 시')

# RSI 과열
if latest['rsi'] >= 70:
    action_conditions.append(f'RSI 과열: 현재 {latest["rsi"]:.1f}, 70 이상 지속 시 익절 검토')

# 거래량 급감
action_conditions.append(f'거래량 급감: MA20 대비 0.5배 이하 시 (관심 소멸)')

# MACD 데드크로스
action_conditions.append(f'MACD 데드크로스: MACD가 Signal선 하향 돌파 시')

for i, cond in enumerate(action_conditions, 1):
    print(f'{i}. {cond}')

print('\n' + '=' * 80)
print('내일 예상 흐름')
print('=' * 80)

# 추세 예측
if total_score >= 2 and latest['macd'] > latest['macd_signal'] and latest['rsi'] < 70:
    trend = '상승 가능성 높음'
    reason = 'MA 정배열, MACD 골든크로스, RSI 과열 미도달'
elif total_score <= -2 and latest['macd'] < latest['macd_signal']:
    trend = '하락 가능성 높음'
    reason = 'MA 역배열, MACD 데드크로스'
else:
    trend = '박스권 횡보 예상'
    reason = '명확한 방향성 부재'

print(f'\n예상: {trend}')
print(f'근거: {reason}')

# 목표가/저항선
if latest['close'] < latest['ma20']:
    target = latest['ma20']
    print(f'\n1차 목표: {target:,.0f}원 (MA20 회복)')
else:
    target = r1
    print(f'\n1차 목표: {target:,.0f}원 (저항선 R1)')

print(f'지지선: {s1:,.0f}원 (S1)')

print('\n' + '━' * 80)
print('분석 완료')
print('━' * 80)
