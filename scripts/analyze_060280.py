"""
큐렉소(060280) 기술적 분석 스크립트
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_stock():
    # 데이터 로드
    df = pd.read_csv('d:/sub-agent-project/stock_data_daily/큐렉소_060280.csv', index_col='Date', parse_dates=True)

    # 최근 30일 데이터
    df_recent = df.tail(30)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    print("=" * 80)
    print("큐렉소(060280) 기술적 분석 보고서")
    print(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # 1. 업체 정보 및 현재가
    print("\n[업체 정보]")
    print(f"종목명: 큐렉소 (060280)")
    print(f"현재가: {latest['close']:,.0f}원")
    print(f"전일대비: {latest['close'] - prev['close']:+,.0f}원 ({(latest['close']/prev['close']-1)*100:+.2f}%)")
    print(f"거래량: {latest['volume']:,.0f}주")

    # 2. 가격 추이 (최근 5일)
    print("\n[최근 5일 가격 추이]")
    print("-" * 80)
    print(f"{'날짜':<12} {'종가':>10} {'MA5':>10} {'MA20':>10} {'MA60':>10} {'거래량':>12}")
    print("-" * 80)
    for idx in range(-5, 0):
        row = df.iloc[idx]
        date_str = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        print(f"{date_str:<12} {row['close']:>10,.0f} {row['ma5']:>10,.0f} {row['ma20']:>10,.0f} {row['ma60']:>10,.0f} {row['volume']:>12,.0f}")

    # 3. 기술적 지표 현황
    print("\n[기술적 지표 분석]")
    print(f"\n▶ RSI (14일): {latest['rsi']:.2f}")
    if latest['rsi'] > 70:
        rsi_status = "과매수 구간 (조정 가능성)"
    elif latest['rsi'] < 30:
        rsi_status = "과매도 구간 (반등 가능성)"
    else:
        rsi_status = "중립 구간"
    print(f"  → {rsi_status}")

    print(f"\n▶ ADX (추세 강도): {latest['adx']:.2f}")
    if latest['adx'] > 25:
        adx_status = "강한 추세"
    elif latest['adx'] > 20:
        adx_status = "추세 형성 중"
    else:
        adx_status = "약한 추세 (박스권)"
    print(f"  → {adx_status}")

    print(f"\n▶ MACD:")
    print(f"  - MACD: {latest['macd']:.2f}")
    print(f"  - Signal: {latest['macd_signal']:.2f}")
    print(f"  - Histogram: {latest['macd'] - latest['macd_signal']:.2f}")
    if latest['macd'] > latest['macd_signal']:
        macd_status = "상승 신호 (골든크로스)"
    else:
        macd_status = "하락 신호 (데드크로스)"
    print(f"  → {macd_status}")

    # 4. 볼린저 밴드
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        print(f"\n▶ 볼린저 밴드:")
        print(f"  - 상단: {latest['bb_upper']:,.0f}원")
        print(f"  - 중심: {latest['ma20']:,.0f}원")
        print(f"  - 하단: {latest['bb_lower']:,.0f}원")
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100
        print(f"  - 현재가 위치: {bb_position:.1f}% (하단=0%, 상단=100%)")

    # 5. 이동평균선 배열
    print("\n[이동평균선 분석]")
    print(f"현재가: {latest['close']:,.0f}원")
    print(f"MA5:   {latest['ma5']:>10,.0f}원 ({latest['close']/latest['ma5']-1:+.2%})")
    print(f"MA20:  {latest['ma20']:>10,.0f}원 ({latest['close']/latest['ma20']-1:+.2%})")
    print(f"MA60:  {latest['ma60']:>10,.0f}원 ({latest['close']/latest['ma60']-1:+.2%})")

    # 골든크로스/데드크로스 확인
    ma5_cross = "상향" if latest['ma5'] > prev['ma5'] else "하향"
    ma20_cross = "상향" if latest['ma20'] > prev['ma20'] else "하향"

    print(f"\n이동평균선 배열:")
    if latest['ma5'] > latest['ma20'] > latest['ma60']:
        print("  → 정배열 (상승 추세)")
    elif latest['ma5'] < latest['ma20'] < latest['ma60']:
        print("  → 역배열 (하락 추세)")
    else:
        print("  → 혼조 (추세 전환기)")

    # 6. 거래량 분석
    print("\n[거래량 및 OBV 분석]")
    avg_volume_20 = df['volume'].tail(20).mean()
    volume_ratio = latest['volume'] / avg_volume_20
    print(f"금일 거래량: {latest['volume']:,.0f}주")
    print(f"20일 평균: {avg_volume_20:,.0f}주")
    print(f"거래량 비율: {volume_ratio:.2f}배")

    if 'obv' in df.columns:
        obv_trend = "상승" if latest['obv'] > df['obv'].tail(5).iloc[0] else "하락"
        print(f"\nOBV 추세: {obv_trend}")
        print(f"  - 현재 OBV: {latest['obv']:,.0f}")
        print(f"  - 5일전 OBV: {df['obv'].tail(6).iloc[0]:,.0f}")

    # 7. 지지/저항 분석
    print("\n[지지/저항 분석]")
    recent_high = df_recent['high'].max()
    recent_low = df_recent['low'].min()
    print(f"30일 고점: {recent_high:,.0f}원 (현재가 대비 {(recent_high/latest['close']-1)*100:+.2f}%)")
    print(f"30일 저점: {recent_low:,.0f}원 (현재가 대비 {(recent_low/latest['close']-1)*100:+.2f}%)")

    # 8. 핵심 분석 포인트
    print("\n" + "=" * 80)
    print("[핵심 분석 포인트]")
    print("=" * 80)

    points = []

    # RSI 분석
    if latest['rsi'] > 70:
        points.append("⚠ RSI 과매수 구간 - 단기 조정 가능성")
    elif latest['rsi'] < 30:
        points.append("✓ RSI 과매도 구간 - 저점 매수 기회")

    # MACD 분석
    if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
        points.append("✓ MACD 골든크로스 발생 - 상승 전환 신호")
    elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
        points.append("⚠ MACD 데드크로스 발생 - 하락 전환 신호")

    # 이동평균선
    if latest['close'] > latest['ma5'] > latest['ma20'] > latest['ma60']:
        points.append("✓ 이동평균선 정배열 - 강한 상승 추세")
    elif latest['close'] < latest['ma60']:
        points.append("⚠ 현재가가 MA60 하방 - 약세 구간")

    # 거래량
    if volume_ratio > 2.0:
        points.append("✓ 급증한 거래량 - 매집/매도 세력 활동")
    elif volume_ratio < 0.5:
        points.append("⚠ 거래량 부족 - 관망세 우세")

    for i, point in enumerate(points, 1):
        print(f"{i}. {point}")

    # 9. 유지/대응 조건
    print("\n" + "=" * 80)
    print("[투자 판단 가이드]")
    print("=" * 80)

    print("\n✅ 유지해도 되는 조건:")
    maintain_conditions = [
        f"현재가가 MA20({latest['ma20']:,.0f}원) 위에서 유지",
        f"거래량이 평균 이상 유지 (현재: {volume_ratio:.1f}배)",
        "RSI가 30~70 사이 유지",
        "MACD가 Signal선 위에서 유지"
    ]
    for i, cond in enumerate(maintain_conditions, 1):
        print(f"  {i}. {cond}")

    print("\n🚨 반드시 대응해야 할 조건:")
    alert_conditions = [
        f"MA5({latest['ma5']:,.0f}원) 하향 이탈 시 → 단기 손절 검토",
        f"MA20({latest['ma20']:,.0f}원) 하향 이탈 시 → 추세 전환, 청산 고려",
        f"30일 저점({recent_low:,.0f}원) 붕괴 시 → 즉시 손절",
        "거래량 급감과 함께 하락 시 → 매도세 우세, 관망",
        "RSI 70 초과 후 하락 전환 시 → 단기 차익 실현"
    ]
    for i, cond in enumerate(alert_conditions, 1):
        print(f"  {i}. {cond}")

    # 10. 종합 판단
    print("\n" + "=" * 80)
    print("[종합 판단]")
    print("=" * 80)

    score = 0
    reasons = []

    # 점수 계산
    if latest['rsi'] >= 30 and latest['rsi'] <= 70:
        score += 20
        reasons.append("RSI 적정 구간")

    if latest['macd'] > latest['macd_signal']:
        score += 20
        reasons.append("MACD 매수 신호")

    if latest['close'] > latest['ma20']:
        score += 20
        reasons.append("MA20 상방")

    if latest['ma5'] > latest['ma20']:
        score += 20
        reasons.append("단기 상승 추세")

    if volume_ratio >= 1.0:
        score += 20
        reasons.append("거래량 충분")
    elif volume_ratio < 0.5:
        score -= 10
        reasons.append("거래량 부족")

    print(f"\n매수 적합성 점수: {score}/100점")
    print(f"\n긍정 요인:")
    for reason in reasons:
        print(f"  - {reason}")

    if score >= 80:
        print(f"\n💡 판단: 매수 적합 (강력 추천)")
        print(f"   진입 타이밍: 즉시 매수 가능")
    elif score >= 60:
        print(f"\n💡 판단: 매수 가능 (조건부 추천)")
        print(f"   진입 타이밍: MA5 근처 조정 시 분할 매수")
    elif score >= 40:
        print(f"\n💡 판단: 관망 (중립)")
        print(f"   진입 타이밍: 추가 신호 확인 필요")
    else:
        print(f"\n💡 판단: 매수 부적합")
        print(f"   진입 타이밍: 추세 전환 확인 후 재검토")

    print("\n⚠ 주의사항:")
    warnings = [
        "기술적 분석은 참고 자료이며, 투자 판단은 본인의 책임입니다",
        "손절가는 반드시 설정하고 지켜야 합니다",
        "거래량 없는 상승은 지속성이 약합니다",
        "뉴스 및 재무제표 등 펀더멘털 분석도 병행해야 합니다"
    ]
    for warning in warnings:
        print(f"  - {warning}")

    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)

if __name__ == "__main__":
    analyze_stock()
