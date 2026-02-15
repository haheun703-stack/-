#!/usr/bin/env python3
"""
HJ중공업 종합 기술적 분석 스크립트
"""
import sys
import os
sys.path.insert(0, "D:/sub-agent-project")

import pandas as pd
import numpy as np
from datetime import datetime

# CSV 파일 읽기
csv_path = r"D:\sub-agent-project\stock_data_daily\HJ중공업_097230.csv"
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 날짜 컬럼 변환
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 최근 데이터만 사용
recent_df = df.tail(200).copy()
latest_df = df.tail(20).copy()

print("=" * 80)
print("HJ중공업(097230) 종합 기술적 분석 보고서")
print("=" * 80)
print(f"분석 기준일: {df.iloc[-1]['Date'].strftime('%Y-%m-%d')}")
print(f"총 데이터: {len(df)}일 | 분석 대상: 최근 200일")
print("=" * 80)

# 1. 최근 20일 가격 추이
print("\n[1] 최근 20일 가격 추이 (고가/저가/종가/변동률)")
print("-" * 80)
latest_df['Change%'] = ((latest_df['Close'] - latest_df['Close'].shift(1)) / latest_df['Close'].shift(1) * 100).round(2)
latest_df['Range%'] = ((latest_df['High'] - latest_df['Low']) / latest_df['Low'] * 100).round(2)

for idx, row in latest_df.iterrows():
    print(f"{row['Date'].strftime('%Y-%m-%d')} | 고:{row['High']:>8,.0f} 저:{row['Low']:>8,.0f} "
          f"종:{row['Close']:>8,.0f} | 일변동:{row['Change%']:>6.2f}% 등락폭:{row['Range%']:>6.2f}%")

latest_close = df.iloc[-1]['Close']
prev_close = df.iloc[-2]['Close']
print(f"\n현재가: {latest_close:,.0f}원 (전일대비 {((latest_close - prev_close) / prev_close * 100):.2f}%)")

# 2. 이동평균선 배열
print("\n[2] 이동평균선 배열 (5/20/60/120일)")
print("-" * 80)
ma5 = df.iloc[-1]['MA5']
ma20 = df.iloc[-1]['MA20']
ma60 = df.iloc[-1]['MA60']
ma120 = df.iloc[-1]['MA120']

print(f"MA5   : {ma5:>10,.2f}원 | 현재가 대비: {((latest_close - ma5) / ma5 * 100):>6.2f}%")
print(f"MA20  : {ma20:>10,.2f}원 | 현재가 대비: {((latest_close - ma20) / ma20 * 100):>6.2f}%")
print(f"MA60  : {ma60:>10,.2f}원 | 현재가 대비: {((latest_close - ma60) / ma60 * 100):>6.2f}%")
print(f"MA120 : {ma120:>10,.2f}원 | 현재가 대비: {((latest_close - ma120) / ma120 * 100):>6.2f}%")

# 이동평균선 정배열/역배열 판단
if ma5 > ma20 > ma60 > ma120:
    ma_status = "정배열 (강세)"
elif ma5 < ma20 < ma60 < ma120:
    ma_status = "역배열 (약세)"
else:
    ma_status = "혼조"
print(f"\n이동평균선 배열: {ma_status}")

# 3. RSI, MACD, Stochastic, ADX 해석
print("\n[3] 주요 기술적 지표 해석")
print("-" * 80)
rsi = df.iloc[-1]['RSI']
macd = df.iloc[-1]['MACD']
macd_signal = df.iloc[-1]['MACD_Signal']
stoch_k = df.iloc[-1]['Stoch_K']
stoch_d = df.iloc[-1]['Stoch_D']
adx = df.iloc[-1]['ADX']

print(f"RSI(14)      : {rsi:>6.2f} ", end="")
if rsi > 70:
    print("→ 과매수 구간 (조정 가능성)")
elif rsi < 30:
    print("→ 과매도 구간 (반등 가능성)")
elif rsi > 50:
    print("→ 강세 구간")
else:
    print("→ 약세 구간")

print(f"MACD         : {macd:>6.2f}")
print(f"MACD Signal  : {macd_signal:>6.2f}")
macd_diff = macd - macd_signal
print(f"MACD 차이    : {macd_diff:>6.2f} ", end="")
if macd_diff > 0:
    print("→ 골든크로스 (매수 신호)")
else:
    print("→ 데드크로스 (매도 신호)")

print(f"Stochastic K : {stoch_k:>6.2f}")
print(f"Stochastic D : {stoch_d:>6.2f}")
if stoch_k > 80 and stoch_d > 80:
    print("               → 과매수 (단기 조정 가능성)")
elif stoch_k < 20 and stoch_d < 20:
    print("               → 과매도 (단기 반등 가능성)")
else:
    print("               → 중립")

print(f"ADX          : {adx:>6.2f} ", end="")
if adx > 40:
    print("→ 강한 추세")
elif adx > 25:
    print("→ 추세 진행 중")
else:
    print("→ 추세 약함 (박스권)")

# 4. 볼린저밴드
print("\n[4] 볼린저밴드 위치 및 스퀴즈 여부")
print("-" * 80)
upper_band = df.iloc[-1]['Upper_Band']
lower_band = df.iloc[-1]['Lower_Band']
bb_width = ((upper_band - lower_band) / ma20 * 100) if ma20 > 0 else 0
bb_position = ((latest_close - lower_band) / (upper_band - lower_band) * 100) if (upper_band - lower_band) > 0 else 50

print(f"상단밴드     : {upper_band:>10,.2f}원")
print(f"중심선(MA20) : {ma20:>10,.2f}원")
print(f"하단밴드     : {lower_band:>10,.2f}원")
print(f"밴드 폭      : {bb_width:>6.2f}% ", end="")
if bb_width < 10:
    print("→ 스퀴즈 (변동성 축소, 급등/급락 임박)")
elif bb_width > 30:
    print("→ 확장 (변동성 확대)")
else:
    print("→ 정상")

print(f"현재가 위치  : {bb_position:>6.2f}% ", end="")
if bb_position > 80:
    print("→ 상단 근접 (과매수)")
elif bb_position < 20:
    print("→ 하단 근접 (과매도)")
else:
    print("→ 중간")

# 5. 거래량 추이
print("\n[5] 거래량 추이 (20일 평균 대비 서지 비율)")
print("-" * 80)
recent_df['Volume_MA20'] = recent_df['Volume'].rolling(20).mean()
latest_volume = df.iloc[-1]['Volume']
volume_ma20 = recent_df.iloc[-1]['Volume_MA20']
volume_surge = ((latest_volume - volume_ma20) / volume_ma20 * 100) if volume_ma20 > 0 else 0

print(f"최근 거래량  : {latest_volume:>12,.0f}주")
print(f"20일 평균    : {volume_ma20:>12,.0f}주")
print(f"서지 비율    : {volume_surge:>6.2f}% ", end="")
if volume_surge > 100:
    print("→ 급증 (이벤트 발생 가능)")
elif volume_surge > 50:
    print("→ 증가")
elif volume_surge < -50:
    print("→ 감소 (관심 저하)")
else:
    print("→ 평균 수준")

# 최근 5일 거래량 추이
print("\n최근 5일 거래량:")
for idx, row in df.tail(5).iterrows():
    vol_change = ((row['Volume'] - volume_ma20) / volume_ma20 * 100) if volume_ma20 > 0 else 0
    print(f"  {row['Date'].strftime('%Y-%m-%d')} | {row['Volume']:>12,.0f}주 ({vol_change:>6.2f}%)")

# 6. 수급 분석 (외국인/기관)
print("\n[6] 수급 분석 (외국인/기관 순매수 추이)")
print("-" * 80)
foreign_net = df.iloc[-1]['Foreign_Net']
inst_net = df.iloc[-1]['Inst_Net']

# 최근 5일 누적
foreign_5d = df.tail(5)['Foreign_Net'].sum()
inst_5d = df.tail(5)['Inst_Net'].sum()

# 최근 20일 누적
foreign_20d = df.tail(20)['Foreign_Net'].sum()
inst_20d = df.tail(20)['Inst_Net'].sum()

print(f"금일 외국인 순매수: {foreign_net:>12,.0f}주")
print(f"금일 기관 순매수  : {inst_net:>12,.0f}주")
print(f"\n5일 외국인 누적   : {foreign_5d:>12,.0f}주 ", end="")
print("→ 순매수" if foreign_5d > 0 else "→ 순매도")
print(f"5일 기관 누적     : {inst_5d:>12,.0f}주 ", end="")
print("→ 순매수" if inst_5d > 0 else "→ 순매도")

print(f"\n20일 외국인 누적  : {foreign_20d:>12,.0f}주 ", end="")
print("→ 순매수" if foreign_20d > 0 else "→ 순매도")
print(f"20일 기관 누적    : {inst_20d:>12,.0f}주 ", end="")
print("→ 순매수" if inst_20d > 0 else "→ 순매도")

# 7. GeometricQuantEngine 분석
print("\n[7] v4.7 기하학적 분석 (GeometricQuantEngine)")
print("-" * 80)
try:
    from src.geometric_engine import GeometricQuantEngine

    engine = GeometricQuantEngine(config={"profile": "default", "lookback": 200})
    result = engine.generate_l7_result(recent_df, ticker="097230")

    print(result)
except Exception as e:
    print(f"GeometricQuantEngine 분석 실패: {e}")
    print("기하학적 분석은 스킵합니다.")

# 8. 매매 전략
print("\n[8] 매매 전략 (진입가/손절가/목표가/손익비)")
print("-" * 80)

# ATR 기반 손절/목표가 계산
atr = df.iloc[-1]['ATR']
entry_price = latest_close

# 손절가: 현재가 - 2*ATR
stop_loss = entry_price - (2 * atr)
# 목표가1: 현재가 + 3*ATR (1.5배)
target1 = entry_price + (3 * atr)
# 목표가2: 현재가 + 4*ATR (2배)
target2 = entry_price + (4 * atr)

# 손익비 계산
risk = entry_price - stop_loss
reward1 = target1 - entry_price
reward2 = target2 - entry_price
rr_ratio1 = reward1 / risk if risk > 0 else 0
rr_ratio2 = reward2 / risk if risk > 0 else 0

print(f"진입가      : {entry_price:>10,.0f}원")
print(f"손절가      : {stop_loss:>10,.0f}원 (-{((entry_price - stop_loss) / entry_price * 100):>5.2f}%)")
print(f"목표가1     : {target1:>10,.0f}원 (+{((target1 - entry_price) / entry_price * 100):>5.2f}%) | 손익비 1:{rr_ratio1:.2f}")
print(f"목표가2     : {target2:>10,.0f}원 (+{((target2 - entry_price) / entry_price * 100):>5.2f}%) | 손익비 1:{rr_ratio2:.2f}")
print(f"ATR(14)     : {atr:>10,.2f}원")

# 지지/저항선
support1 = lower_band
support2 = ma20
resistance1 = upper_band
resistance2 = df.tail(20)['High'].max()

print(f"\n지지선1     : {support1:>10,.0f}원 (볼린저 하단)")
print(f"지지선2     : {support2:>10,.0f}원 (MA20)")
print(f"저항선1     : {resistance1:>10,.0f}원 (볼린저 상단)")
print(f"저항선2     : {resistance2:>10,.0f}원 (20일 최고가)")

# 종합 판단
print("\n[종합 판단]")
print("-" * 80)

# 매수/매도 신호 점수화
buy_signals = 0
sell_signals = 0

if rsi < 30:
    buy_signals += 2
elif rsi > 70:
    sell_signals += 2

if macd_diff > 0:
    buy_signals += 1
else:
    sell_signals += 1

if stoch_k < 20 and stoch_d < 20:
    buy_signals += 1
elif stoch_k > 80 and stoch_d > 80:
    sell_signals += 1

if latest_close > ma5 > ma20:
    buy_signals += 1
elif latest_close < ma5 < ma20:
    sell_signals += 1

if bb_position < 20:
    buy_signals += 1
elif bb_position > 80:
    sell_signals += 1

if foreign_5d > 0 and inst_5d > 0:
    buy_signals += 2
elif foreign_5d < 0 and inst_5d < 0:
    sell_signals += 2

print(f"매수 신호 점수: {buy_signals}점")
print(f"매도 신호 점수: {sell_signals}점")

if buy_signals > sell_signals + 2:
    recommendation = "강력 매수"
elif buy_signals > sell_signals:
    recommendation = "매수"
elif sell_signals > buy_signals + 2:
    recommendation = "강력 매도"
elif sell_signals > buy_signals:
    recommendation = "매도"
else:
    recommendation = "중립 (관망)"

print(f"\n추천: {recommendation}")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)
