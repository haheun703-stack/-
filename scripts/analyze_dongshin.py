"""
동신건설(025950) 기술적 분석 스크립트
Parquet 파일 기반 상세 분석

실행: python scripts/analyze_dongshin.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def load_parquet_data(ticker: str = "025950") -> pd.DataFrame:
    """Parquet 파일 로드."""
    file_path = Path(f"data/processed/{ticker}.parquet")
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_parquet(file_path)

    # 날짜 인덱스 확인 및 정렬
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    return df


def analyze_moving_averages(df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> dict:
    """이동평균선 분석."""
    result = {}

    # 이동평균 정보
    ma_info = {}
    for ma in ['MA5', 'MA20', 'MA60', 'MA120']:
        if ma in latest.index and not pd.isna(latest[ma]):
            diff_pct = ((latest['close'] - latest[ma]) / latest[ma]) * 100
            ma_info[ma] = {
                'value': latest[ma],
                'diff_pct': diff_pct,
                'above': latest['close'] > latest[ma]
            }

    result['ma_values'] = ma_info

    # 정배열/역배열 판정
    if 'MA5' in ma_info and 'MA20' in ma_info and 'MA60' in ma_info:
        if ma_info['MA5']['value'] > ma_info['MA20']['value'] > ma_info['MA60']['value']:
            result['alignment'] = '완전 정배열'
        elif ma_info['MA5']['value'] < ma_info['MA20']['value'] < ma_info['MA60']['value']:
            result['alignment'] = '완전 역배열'
        else:
            result['alignment'] = '혼조'

    # 골든/데드크로스
    if 'MA5' in latest.index and 'MA20' in latest.index and 'MA5' in prev.index and 'MA20' in prev.index:
        if prev['MA5'] <= prev['MA20'] and latest['MA5'] > latest['MA20']:
            result['cross'] = '골든크로스 발생'
        elif prev['MA5'] >= prev['MA20'] and latest['MA5'] < latest['MA20']:
            result['cross'] = '데드크로스 발생'
        elif latest['MA5'] > latest['MA20']:
            result['cross'] = '골든크로스 유지'
        else:
            result['cross'] = '데드크로스 상태'

    return result


def analyze_indicators(df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> dict:
    """기술적 지표 분석."""
    result = {}

    # RSI
    if 'RSI' in latest.index and not pd.isna(latest['RSI']):
        rsi = latest['RSI']
        result['RSI'] = {
            'value': rsi,
            'signal': '과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'
        }

    # MACD
    if 'MACD' in latest.index and 'MACD_signal' in latest.index:
        macd = latest['MACD']
        signal = latest['MACD_signal']
        hist = macd - signal

        result['MACD'] = {
            'value': macd,
            'signal': signal,
            'histogram': hist,
            'status': '매수 신호' if hist > 0 else '매도 신호'
        }

    # ADX
    if 'ADX' in latest.index and not pd.isna(latest['ADX']):
        adx = latest['ADX']
        result['ADX'] = {
            'value': adx,
            'signal': '강한 추세' if adx > 25 else '약한 추세'
        }

    # Stochastic
    if 'stoch_k' in latest.index and 'stoch_d' in latest.index:
        result['Stochastic'] = {
            'k': latest['stoch_k'],
            'd': latest['stoch_d'],
            'signal': '과매수' if latest['stoch_k'] > 80 else '과매도' if latest['stoch_k'] < 20 else '중립'
        }

    return result


def analyze_volume(df: pd.DataFrame, latest: pd.Series) -> dict:
    """거래량 분석."""
    result = {}

    recent_30 = df.tail(30)
    vol_ma5 = recent_30['volume'].tail(5).mean()
    vol_ma20 = recent_30['volume'].tail(20).mean()
    vol_ratio = (latest['volume'] / vol_ma20) * 100

    result['current'] = latest['volume']
    result['ma5'] = vol_ma5
    result['ma20'] = vol_ma20
    result['ratio'] = vol_ratio

    # OBV
    if 'OBV' in latest.index:
        obv_5d = recent_30['OBV'].tail(5)
        obv_slope = obv_5d.iloc[-1] - obv_5d.iloc[0]

        result['OBV'] = {
            'value': latest['OBV'],
            'trend': '강한 상승' if obv_slope > 0 else '강한 하락' if obv_slope < 0 else '횡보',
            'change_5d': obv_slope
        }

    return result


def analyze_bollinger_bands(latest: pd.Series) -> dict:
    """볼린저 밴드 분석."""
    if not all(col in latest.index for col in ['BB_upper', 'BB_middle', 'BB_lower']):
        return {}

    upper = latest['BB_upper']
    middle = latest['BB_middle']
    lower = latest['BB_lower']
    close = latest['close']

    position = ((close - lower) / (upper - lower)) * 100 if upper != lower else 50

    if position > 80:
        signal = '상단 근접 - 과열 가능성'
    elif position < 20:
        signal = '하단 근접 - 반등 가능성'
    else:
        signal = '중립 구간'

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'position_pct': position,
        'signal': signal
    }


def analyze_support_resistance(df: pd.DataFrame, latest: pd.Series) -> dict:
    """지지/저항 분석."""
    recent_60 = df.tail(60)

    # 저항선
    resistance = [
        recent_60['high'].max(),
        recent_60.tail(30)['high'].max(),
        recent_60.tail(10)['high'].max()
    ]
    resistance = sorted(set(resistance), reverse=True)

    # 지지선
    support = [
        recent_60['low'].min(),
        recent_60.tail(30)['low'].min(),
        recent_60.tail(10)['low'].min()
    ]
    support = sorted(set(support))

    return {
        'resistance': [r for r in resistance if r > latest['close']],
        'support': [s for s in support if s < latest['close']]
    }


def analyze_volume_profile(df: pd.DataFrame) -> dict:
    """매물대 분석 (최근 60일)."""
    recent_60 = df.tail(60)

    # 가격대별 거래량 집계
    price_bins = 10
    price_min = recent_60['low'].min()
    price_max = recent_60['high'].max()
    price_range = price_max - price_min
    bin_size = price_range / price_bins

    volume_by_price = {}
    for _, row in recent_60.iterrows():
        price_level = int((row['close'] - price_min) / bin_size) if bin_size > 0 else 0
        price_level = min(price_level, price_bins - 1)

        if price_level not in volume_by_price:
            volume_by_price[price_level] = 0
        volume_by_price[price_level] += row['volume']

    # 상위 3개 매물대
    top_volumes = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)[:3]

    heavy_zones = []
    for level, vol in top_volumes:
        price_low = price_min + (level * bin_size)
        price_high = price_low + bin_size
        heavy_zones.append({
            'range': (price_low, price_high),
            'volume': vol
        })

    return {'heavy_zones': heavy_zones}


def print_report(df: pd.DataFrame, ticker: str = "025950", name: str = "동신건설"):
    """분석 리포트 출력."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    recent_30 = df.tail(30)

    # 기본 정보
    print("=" * 80)
    print(f"동신건설 (025950) 기술적 분석 보고서")
    print("=" * 80)
    print(f"분석일: {latest.name.strftime('%Y-%m-%d')}")
    print(f"현재가: {latest['close']:,.0f}원")
    print(f"데이터 범위: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}일)")
    print()

    # 1. 최근 30일 가격 추이
    print("■ 1. 최근 30일 가격 추이")
    print("-" * 80)
    print(f"최고가: {recent_30['high'].max():,.0f}원 ({recent_30['high'].idxmax().strftime('%m/%d')})")
    print(f"최저가: {recent_30['low'].min():,.0f}원 ({recent_30['low'].idxmin().strftime('%m/%d')})")
    print(f"변동폭: {((recent_30['high'].max() / recent_30['low'].min() - 1) * 100):.1f}%")
    print()

    # 이동평균선
    ma_analysis = analyze_moving_averages(df, latest, prev)
    if 'ma_values' in ma_analysis:
        print(f"현재가: {latest['close']:,.0f}원")
        for ma_name, ma_data in ma_analysis['ma_values'].items():
            symbol = "▲" if ma_data['above'] else "▼"
            print(f"{ma_name}:   {ma_data['value']:,.0f}원 {symbol} (현재가 대비 {ma_data['diff_pct']:+.1f}%)")
        print()

        if 'alignment' in ma_analysis:
            print(f"이동평균 배열: {ma_analysis['alignment']}")
        if 'cross' in ma_analysis:
            print(f"크로스 상태: {ma_analysis['cross']}")
    print()

    # 2. 기술적 지표
    print("■ 2. 기술적 지표 상태")
    print("-" * 80)
    indicators = analyze_indicators(df, latest, prev)

    if 'RSI' in indicators:
        print(f"RSI(14): {indicators['RSI']['value']:.1f} ({indicators['RSI']['signal']})")

    if 'ADX' in indicators:
        print(f"ADX: {indicators['ADX']['value']:.1f} ({indicators['ADX']['signal']})")

    if 'MACD' in indicators:
        m = indicators['MACD']
        print(f"MACD: {m['value']:.2f}")
        print(f"Signal: {m['signal']:.2f}")
        print(f"Histogram: {m['histogram']:+.2f} ({m['status']})")

    if 'Stochastic' in indicators:
        s = indicators['Stochastic']
        print(f"Stochastic: %K={s['k']:.1f}, %D={s['d']:.1f} ({s['signal']})")
    print()

    # 3. 거래량 분석
    print("■ 3. 거래량 분석")
    print("-" * 80)
    vol_analysis = analyze_volume(df, latest)
    print(f"당일 거래량: {vol_analysis['current']:,.0f}")
    print(f"5일 평균: {vol_analysis['ma5']:,.0f}")
    print(f"20일 평균: {vol_analysis['ma20']:,.0f}")
    print(f"거래량 비율: {vol_analysis['ratio']:.0f}% (20일 평균 대비)")

    if 'OBV' in vol_analysis:
        obv = vol_analysis['OBV']
        print(f"OBV: {obv['value']:,.0f} ({obv['trend']})")
        print(f"OBV 5일 추세: {obv['change_5d']:+,.0f}")
    print()

    # 4. 볼린저 밴드
    print("■ 4. 볼린저 밴드 분석")
    print("-" * 80)
    bb = analyze_bollinger_bands(latest)
    if bb:
        print(f"상단: {bb['upper']:,.0f}원")
        print(f"중단: {bb['middle']:,.0f}원")
        print(f"하단: {bb['lower']:,.0f}원")
        print(f"현재가 위치: {bb['position_pct']:.1f}% (0%=하단, 100%=상단)")
        print(f"상태: {bb['signal']}")
    else:
        print("볼린저 밴드 데이터 없음")
    print()

    # 5. 지지/저항
    print("■ 5. 지지/저항 분석")
    print("-" * 80)
    sr = analyze_support_resistance(df, latest)

    print("주요 저항선:")
    for i, level in enumerate(sr['resistance'][:3], 1):
        distance = ((level / latest['close'] - 1) * 100)
        print(f"  R{i}: {level:,.0f}원 (+{distance:.1f}%)")

    print()
    print("주요 지지선:")
    for i, level in enumerate(sr['support'][:3], 1):
        distance = ((latest['close'] / level - 1) * 100)
        print(f"  S{i}: {level:,.0f}원 (-{distance:.1f}%)")
    print()

    # 6. 매물대
    print("■ 6. 매물대 분석 (최근 60일)")
    print("-" * 80)
    vp = analyze_volume_profile(df)
    print("강한 매물대 (거래 집중 구간):")
    for zone in vp['heavy_zones']:
        low, high = zone['range']
        print(f"  {low:,.0f}~{high:,.0f}원 (거래량: {zone['volume']:,.0f})")
    print()

    # 7. 종합 판단
    print("=" * 80)
    print("■ 종합 판단 - 내일 매매 전략")
    print("=" * 80)

    # 매수 적합성 판정
    score = 0
    reasons = []
    warnings = []

    # 추세
    if 'alignment' in ma_analysis and ma_analysis['alignment'] == '완전 정배열':
        score += 2
        reasons.append("완전 정배열 (강한 상승 추세)")

    # RSI
    if 'RSI' in indicators:
        rsi_val = indicators['RSI']['value']
        if 40 < rsi_val < 70:
            score += 1
            reasons.append(f"RSI {rsi_val:.0f} (적정 수준)")
        elif rsi_val > 70:
            warnings.append(f"RSI 과매수 ({rsi_val:.0f})")

    # MACD
    if 'MACD' in indicators and indicators['MACD']['histogram'] > 0:
        score += 1
        reasons.append("MACD 매수 신호")

    # 거래량
    if vol_analysis['ratio'] > 120:
        score += 1
        reasons.append(f"거래량 증가 ({vol_analysis['ratio']:.0f}%)")

    # OBV
    if 'OBV' in vol_analysis and vol_analysis['OBV']['trend'] in ['강한 상승']:
        score += 1
        reasons.append("OBV 상승 추세")

    print(f"\n매수 적합성 점수: {score}/6")
    print()

    if score >= 4:
        print("✅ 매수 적합 - 강한 매수 신호")
    elif score >= 3:
        print("💡 매수 고려 - 긍정적 신호")
    else:
        print("⚠️ 관망 권장 - 신호 부족")

    print("\n긍정 요인:")
    for r in reasons:
        print(f"  • {r}")

    if warnings:
        print("\n주의 사항:")
        for w in warnings:
            print(f"  ⚠️ {w}")

    # 진입 타이밍
    print("\n" + "-" * 80)
    print("■ 진입 타이밍 및 주의점")
    print("-" * 80)

    # 저항선까지 거리
    if sr['resistance']:
        nearest_resistance = sr['resistance'][0]
        resistance_distance = ((nearest_resistance / latest['close'] - 1) * 100)
        print(f"가장 가까운 저항선: {nearest_resistance:,.0f}원 (+{resistance_distance:.1f}%)")

        if resistance_distance < 3:
            print("  ⚠️ 저항선 근접 - 단기 조정 가능성")
        else:
            print("  ✅ 저항선까지 여유")

    # 지지선까지 거리
    if sr['support']:
        nearest_support = sr['support'][-1]
        support_distance = ((latest['close'] / nearest_support - 1) * 100)
        print(f"가장 가까운 지지선: {nearest_support:,.0f}원 (-{support_distance:.1f}%)")

    print()
    print("✅ 유지해도 되는 조건:")
    print("  1. 5일선(MA5) 위에서 지지")
    print("  2. 거래량 증가 지속")
    print("  3. RSI 70 이하 유지")

    print()
    print("🚨 반드시 대응해야 할 조건:")
    if sr['support']:
        print(f"  1. {nearest_support:,.0f}원(지지선) 이탈 시 손절")
    print(f"  2. 5일선 하향 이탈 시 관망")
    print(f"  3. 거래량 급감 시 주의")

    print()
    print("=" * 80)


def main():
    """메인 실행."""
    try:
        df = load_parquet_data("025950")
        print_report(df)

        # 파일 저장
        output_path = Path("docs/03-analysis/dongsin_analysis.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import contextlib
        with open(output_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                print_report(df)

        print(f"\n분석 결과 저장: {output_path}")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
