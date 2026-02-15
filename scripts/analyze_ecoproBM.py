"""
에코프로비엠 (247540) 기술적 분석 스크립트
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "D:/sub-agent-project")

# CSV 읽기
df = pd.read_csv('D:/sub-agent-project/stock_data_daily/에코프로비엠_247540.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 최근 20일 데이터
recent_20 = df.tail(20).copy()
latest = df.iloc[-1]

print('='*80)
print('에코프로비엠 (247540) 기술적 분석 보고서')
print('분석일시: 2026-02-14')
print('='*80)
print()

# 1. 최근 20일 가격 추이
print('[1] 최근 20일 가격 추이')
print('-'*80)
for idx, row in recent_20.iterrows():
    change = ((row['Close'] - row['Open']) / row['Open'] * 100) if row['Open'] != 0 else 0
    vol_ma = recent_20['Volume'].mean()
    vol_ratio = (row['Volume'] / vol_ma - 1) * 100 if vol_ma > 0 else 0
    print(f"{row['Date'].strftime('%Y-%m-%d')} | 시가: {row['Open']:>9,.0f} | 고가: {row['High']:>9,.0f} | 저가: {row['Low']:>9,.0f} | 종가: {row['Close']:>9,.0f} | 변동: {change:>6.2f}% | 거래량비: {vol_ratio:>6.1f}%")

print()
print(f"최신 종가: {latest['Close']:,.0f}원 (2026-02-12)")
print(f"20일 최고가: {recent_20['High'].max():,.0f}원")
print(f"20일 최저가: {recent_20['Low'].min():,.0f}원")
print()

# 2. 이동평균선 배열
print('[2] 이동평균선 분석')
print('-'*80)
ma5 = latest['MA5']
ma20 = latest['MA20']
ma60 = latest['MA60']
ma120 = latest['MA120']
current = latest['Close']

print(f"현재가: {current:,.0f}원")
print(f"MA5:   {ma5:,.0f}원 ({((current/ma5-1)*100):>+6.2f}%)")
print(f"MA20:  {ma20:,.0f}원 ({((current/ma20-1)*100):>+6.2f}%)")
print(f"MA60:  {ma60:,.0f}원 ({((current/ma60-1)*100):>+6.2f}%)")
print(f"MA120: {ma120:,.0f}원 ({((current/ma120-1)*100):>+6.2f}%)")
print()

# 이평선 배열 판단
if ma5 > ma20 > ma60 > ma120:
    print("이평선 배열: 정배열 (강세)")
elif ma5 < ma20 < ma60 < ma120:
    print("이평선 배열: 역배열 (약세)")
else:
    print("이평선 배열: 혼조세 (방향성 불명확)")
print()

# 3. RSI, MACD, Stochastic, ADX 해석
print('[3] 기술적 지표 분석')
print('-'*80)
rsi = latest['RSI']
macd = latest['MACD']
macd_signal = latest['MACD_Signal']
stoch_k = latest['Stoch_K']
stoch_d = latest['Stoch_D']
adx = latest['ADX']

print(f"RSI(14): {rsi:.2f}")
if rsi > 70:
    rsi_status = "과매수 구간 (조정 가능성)"
elif rsi < 30:
    rsi_status = "과매도 구간 (반등 가능성)"
else:
    rsi_status = "중립 구간"
print(f"  상태: {rsi_status}")
print()

print(f"MACD: {macd:.2f}")
print(f"MACD Signal: {macd_signal:.2f}")
print(f"MACD Histogram: {(macd - macd_signal):.2f}")
if macd > macd_signal:
    macd_status = "골든크로스 상태 (매수 신호)"
else:
    macd_status = "데드크로스 상태 (매도 신호)"
print(f"  상태: {macd_status}")
print()

print(f"Stochastic K: {stoch_k:.2f}")
print(f"Stochastic D: {stoch_d:.2f}")
if stoch_k > 80:
    stoch_status = "과매수 구간"
elif stoch_k < 20:
    stoch_status = "과매도 구간"
else:
    stoch_status = "중립 구간"
print(f"  상태: {stoch_status}")
print()

print(f"ADX: {adx:.2f}")
if adx > 40:
    adx_status = "강한 추세"
elif adx > 25:
    adx_status = "추세 형성 중"
else:
    adx_status = "추세 없음 (횡보)"
print(f"  상태: {adx_status}")
print()

# 4. 볼린저밴드 분석
print('[4] 볼린저밴드 분석')
print('-'*80)
bb_upper = latest['Upper_Band']
bb_lower = latest['Lower_Band']
bb_mid = (bb_upper + bb_lower) / 2
bb_width = bb_upper - bb_lower
bb_position = (current - bb_lower) / (bb_upper - bb_lower) * 100 if bb_width > 0 else 50

print(f"상단밴드: {bb_upper:,.0f}원")
print(f"하단밴드: {bb_lower:,.0f}원")
print(f"현재가 위치: {bb_position:.1f}% (0%=하단, 100%=상단)")
print(f"밴드폭: {bb_width:,.0f}원")

# 스퀴즈 판단 (밴드폭이 평균 대비 작은지)
recent_bb_width = (recent_20['Upper_Band'] - recent_20['Lower_Band']).mean()
if bb_width < recent_bb_width * 0.7:
    squeeze_status = "스퀴즈 상태 (변동성 확대 대기)"
else:
    squeeze_status = "정상 상태"
print(f"  상태: {squeeze_status}")
print()

# 5. 거래량 분석
print('[5] 거래량 분석')
print('-'*80)
vol_current = latest['Volume']
vol_ma20 = recent_20['Volume'].mean()
vol_surge = (vol_current / vol_ma20 - 1) * 100

print(f"당일 거래량: {vol_current:,.0f}주")
print(f"20일 평균: {vol_ma20:,.0f}주")
print(f"거래량 비율: {vol_surge:+.1f}%")

if vol_surge > 100:
    vol_status = "거래량 급증 (매우 강한 수급)"
elif vol_surge > 50:
    vol_status = "거래량 증가 (강한 수급)"
elif vol_surge > 0:
    vol_status = "평균 이상 (양호)"
else:
    vol_status = "평균 이하 (관심 감소)"
print(f"  상태: {vol_status}")
print()

# 6. 수급 분석 (외국인/기관)
print('[6] 수급 분석')
print('-'*80)
foreign_net = latest['Foreign_Net']
inst_net = latest['Inst_Net']

print(f"외국인 순매수: {foreign_net:,.0f}주")
print(f"기관 순매수: {inst_net:,.0f}주")

# 최근 5일 누적
foreign_5d = recent_20.tail(5)['Foreign_Net'].sum()
inst_5d = recent_20.tail(5)['Inst_Net'].sum()
print(f"외국인 5일 누적: {foreign_5d:,.0f}주")
print(f"기관 5일 누적: {inst_5d:,.0f}주")
print()

# 7. v4.7 기하학적 분석
print('[7] v4.7 기하학적 분석 (GeometricQuantEngine)')
print('-'*80)

try:
    from src.geometric_engine import GeometricQuantEngine

    engine = GeometricQuantEngine(config={"profile": "default", "lookback": 200})
    result = engine.generate_l7_result(df, ticker="247540")

    print(result)
    print()
except Exception as e:
    print(f"기하학적 분석 실행 실패: {e}")
    print()

# 8. 매매 전략
print('[8] 매매 전략 제안')
print('-'*80)

# ATR 기반 손절/목표가 설정
atr = latest['ATR']
entry_price = current

# 손절가: 현재가 - 2*ATR
stop_loss = entry_price - (2 * atr)
# 목표가1: 현재가 + 3*ATR (손익비 1.5:1)
target1 = entry_price + (3 * atr)
# 목표가2: 현재가 + 4*ATR (손익비 2:1)
target2 = entry_price + (4 * atr)

risk_reward1 = (target1 - entry_price) / (entry_price - stop_loss)
risk_reward2 = (target2 - entry_price) / (entry_price - stop_loss)

print(f"진입가 (현재가): {entry_price:,.0f}원")
print(f"손절가 (2*ATR): {stop_loss:,.0f}원 ({((stop_loss/entry_price-1)*100):+.2f}%)")
print(f"목표가1 (3*ATR): {target1:,.0f}원 ({((target1/entry_price-1)*100):+.2f}%) - 손익비 {risk_reward1:.2f}:1")
print(f"목표가2 (4*ATR): {target2:,.0f}원 ({((target2/entry_price-1)*100):+.2f}%) - 손익비 {risk_reward2:.2f}:1")
print(f"ATR(14): {atr:,.0f}원")
print()

print('[9] 핵심 분석 포인트')
print('-'*80)

# 종합 판단
signals = []
if rsi > 50:
    signals.append("RSI 중립 이상")
else:
    signals.append("RSI 약세")

if macd > macd_signal:
    signals.append("MACD 골든크로스")
else:
    signals.append("MACD 데드크로스")

if current > ma20:
    signals.append("20일선 상회")
else:
    signals.append("20일선 하회")

if vol_surge > 0:
    signals.append("거래량 양호")
else:
    signals.append("거래량 부진")

print("기술적 신호:")
for sig in signals:
    print(f"  - {sig}")
print()

# 유지/대응 조건
print('[10] 보유 판단 가이드')
print('='*80)
print()
print("✅ 유지해도 되는 조건:")
print(f"  1. 종가가 {ma20:,.0f}원 (MA20) 이상 유지")
print(f"  2. RSI가 30 이상 유지 (현재 {rsi:.1f})")
print(f"  3. 거래량이 {vol_ma20*0.5:,.0f}주 (평균의 50%) 이상")
print(f"  4. 손절가 {stop_loss:,.0f}원을 지키는 경우")
print()

print("🚨 반드시 대응해야 할 조건:")
print(f"  1. 종가가 {stop_loss:,.0f}원 (손절가) 이하로 하락 시 즉시 손절")
print(f"  2. RSI가 30 이하로 급락하며 거래량 급증 시 (공포 매도)")
print(f"  3. MACD 히스토그램이 -5000 이하로 급락 시")
print(f"  4. 5일 이동평균선({ma5:,.0f}원)을 거래량 동반하여 이탈 시")
print(f"  5. 외국인+기관이 3일 연속 순매도하며 거래량 증가 시")
print()

print('='*80)
print('분석 완료')
print('='*80)
