"""
고영(098460) 종목 기술적 분석 스크립트
v4.7 기하학적 분석 + 종합 기술적 분석
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import numpy as np
from pathlib import Path

# Windows cp949 인코딩 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.geometric_engine import GeometricQuantEngine


def load_stock_data(ticker: str = "098460") -> tuple:
    """고영 종목 데이터 로드"""
    data_dir = Path("stock_data_daily")
    for f in data_dir.glob(f"*_{ticker}.csv"):
        print(f"[*] 데이터 파일 로드: {f}")
        df = pd.read_csv(f)

        # Date 컬럼이 있으면 인덱스로 설정
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # 컬럼명 정리
        df.columns = [c.strip() for c in df.columns]

        name = f.stem.split("_")[0]
        print(f"[*] 종목명: {name} | 전체 데이터: {len(df)}행")
        print(f"[*] 컬럼: {list(df.columns)}")
        print(f"[*] 데이터 기간: {df.index[0]} ~ {df.index[-1]}")

        return df, name

    raise FileNotFoundError(f"종목 {ticker} CSV 파일을 찾을 수 없습니다.")


def analyze_recent_price_trend(df: pd.DataFrame, days: int = 20):
    """최근 N일 가격 추이 분석"""
    recent = df.tail(days)

    print(f"\n{'='*80}")
    print(f"  [1] 최근 {days}일 가격 추이")
    print(f"{'='*80}")

    print(f"\n  기간: {recent.index[0].date()} ~ {recent.index[-1].date()}")
    print(f"  고가: {recent['High'].max():,.0f}원")
    print(f"  저가: {recent['Low'].min():,.0f}원")
    print(f"  시작가: {recent['Close'].iloc[0]:,.0f}원")
    print(f"  종가: {recent['Close'].iloc[-1]:,.0f}원")

    change_pct = (recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100
    print(f"  변동률: {change_pct:+.2f}%")

    # 일별 상세
    print(f"\n  [최근 10일 상세]")
    print(f"  {'날짜':<12} {'종가':>10} {'변동률':>8} {'거래량':>12}")
    print(f"  {'-'*50}")

    for i in range(min(10, len(recent))):
        row = recent.iloc[-(10-i)]
        idx = len(df) - (10-i)

        if idx > 0:
            prev_close = df.iloc[idx-1]['Close']
            change = (row['Close'] / prev_close - 1) * 100
        else:
            change = 0

        date_str = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        print(f"  {date_str:<12} {row['Close']:>10,.0f} {change:>7.2f}% {row['Volume']:>12,}")


def analyze_moving_averages(df: pd.DataFrame):
    """이동평균선 배열 분석"""
    latest = df.iloc[-1]

    print(f"\n{'='*80}")
    print(f"  [2] 이동평균선 배열 (5/20/60/120일)")
    print(f"{'='*80}")

    print(f"\n  현재가: {latest['Close']:,.0f}원")
    print(f"\n  {'이평선':<10} {'가격':>10} {'괴리율':>8} {'위치':>6}")
    print(f"  {'-'*40}")

    ma_list = []
    for ma_name in ['MA5', 'MA20', 'MA60', 'MA120']:
        if ma_name in df.columns and not pd.isna(latest[ma_name]):
            ma_val = latest[ma_name]
            diff_pct = (latest['Close'] / ma_val - 1) * 100
            position = "상단" if latest['Close'] > ma_val else "하단"

            print(f"  {ma_name:<10} {ma_val:>10,.0f} {diff_pct:>7.2f}% {position:>6}")
            ma_list.append((ma_name, ma_val))

    # 정배열 여부 확인
    if len(ma_list) >= 3:
        ma_values = [v for _, v in ma_list]
        is_aligned = all(ma_values[i] > ma_values[i+1] for i in range(len(ma_values)-1))

        if is_aligned:
            print(f"\n  >> 정배열 상태 (강한 상승 추세)")
        else:
            print(f"\n  >> 역배열 또는 혼재 상태")


def analyze_technical_indicators(df: pd.DataFrame):
    """RSI, MACD, Stochastic, ADX 해석"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    print(f"\n{'='*80}")
    print(f"  [3] 기술적 지표 분석")
    print(f"{'='*80}")

    # RSI
    if 'RSI' in df.columns and not pd.isna(latest['RSI']):
        rsi = latest['RSI']
        print(f"\n  [RSI(14)]")
        print(f"  현재값: {rsi:.1f}")

        if rsi > 70:
            signal = "과매수 구간 - 조정 가능성"
        elif rsi > 60:
            signal = "강세 구간"
        elif rsi > 40:
            signal = "중립 구간"
        elif rsi > 30:
            signal = "약세 구간"
        else:
            signal = "과매도 구간 - 반등 가능성"

        print(f"  해석: {signal}")

    # MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = latest['MACD']
        macd_sig = latest['MACD_Signal']
        hist = macd - macd_sig

        print(f"\n  [MACD]")
        print(f"  MACD: {macd:.2f}")
        print(f"  Signal: {macd_sig:.2f}")
        print(f"  Histogram: {hist:.2f}")

        # 골든/데드 크로스 확인
        prev_macd = prev['MACD']
        prev_sig = prev['MACD_Signal']

        if macd > macd_sig and prev_macd <= prev_sig:
            print(f"  해석: 골든크로스 발생 (매수 신호)")
        elif macd < macd_sig and prev_macd >= prev_sig:
            print(f"  해석: 데드크로스 발생 (매도 신호)")
        elif macd > macd_sig:
            print(f"  해석: 매수 신호 지속 중")
        else:
            print(f"  해석: 매도 신호 지속 중")

    # Stochastic
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        k = latest['Stoch_K']
        d = latest['Stoch_D']

        print(f"\n  [Stochastic]")
        print(f"  %K: {k:.1f}")
        print(f"  %D: {d:.1f}")

        if k > 80:
            signal = "과매수 구간"
        elif k < 20:
            signal = "과매도 구간"
        else:
            signal = "중립 구간"

        if k > d:
            signal += " / 골든크로스 (매수)"
        else:
            signal += " / 데드크로스 (매도)"

        print(f"  해석: {signal}")

    # ADX
    if 'ADX' in df.columns:
        adx = latest['ADX']
        plus_di = latest.get('Plus_DI', np.nan)
        minus_di = latest.get('Minus_DI', np.nan)

        print(f"\n  [ADX (추세 강도)]")
        print(f"  ADX: {adx:.1f}")

        if not pd.isna(plus_di) and not pd.isna(minus_di):
            print(f"  +DI: {plus_di:.1f}")
            print(f"  -DI: {minus_di:.1f}")

            if adx > 25:
                if plus_di > minus_di:
                    signal = "강한 상승 추세"
                else:
                    signal = "강한 하락 추세"
            elif adx < 20:
                signal = "추세 없음 (횡보)"
            else:
                signal = "약한 추세"
        else:
            if adx > 25:
                signal = "강한 추세"
            elif adx < 20:
                signal = "약한 추세 (횡보)"
            else:
                signal = "보통 추세"

        print(f"  해석: {signal}")


def analyze_bollinger_bands(df: pd.DataFrame):
    """볼린저밴드 위치 및 스퀴즈 여부"""
    latest = df.iloc[-1]

    print(f"\n{'='*80}")
    print(f"  [4] 볼린저밴드 분석")
    print(f"{'='*80}")

    if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
        upper = latest['Upper_Band']
        lower = latest['Lower_Band']
        close = latest['Close']

        # 밴드 내 위치 (%)
        if upper != lower:
            bb_pct = (close - lower) / (upper - lower) * 100
        else:
            bb_pct = 50

        # 밴드폭
        bb_width = (upper - lower) / close * 100

        print(f"\n  상단밴드: {upper:,.0f}원")
        print(f"  현재가: {close:,.0f}원")
        print(f"  하단밴드: {lower:,.0f}원")
        print(f"  밴드 내 위치: {bb_pct:.1f}%")
        print(f"  밴드폭: {bb_width:.2f}%")

        # 밴드폭 기반 스퀴즈 판단
        recent_bb_width = df['Upper_Band'].tail(20) - df['Lower_Band'].tail(20)
        recent_bb_width_pct = (recent_bb_width / df['Close'].tail(20) * 100)
        avg_width = recent_bb_width_pct.mean()

        print(f"  20일 평균 밴드폭: {avg_width:.2f}%")

        # 해석
        if bb_width < avg_width * 0.7:
            squeeze = "스퀴즈 발생 (변동성 수축 - 큰 움직임 임박)"
        elif bb_width > avg_width * 1.3:
            squeeze = "밴드 확장 (변동성 확대 중)"
        else:
            squeeze = "정상 범위"

        print(f"  스퀴즈 여부: {squeeze}")

        # 가격 위치 해석
        if bb_pct > 100:
            position = "상단 돌파 (과열 신호)"
        elif bb_pct > 80:
            position = "상단 근접 (매도 고려)"
        elif bb_pct < 0:
            position = "하단 돌파 (침체 신호)"
        elif bb_pct < 20:
            position = "하단 근접 (매수 고려)"
        else:
            position = "중립 구간"

        print(f"  가격 위치: {position}")


def analyze_volume(df: pd.DataFrame):
    """거래량 추이 분석"""
    latest = df.iloc[-1]

    print(f"\n{'='*80}")
    print(f"  [5] 거래량 분석")
    print(f"{'='*80}")

    recent_20 = df.tail(20)
    avg_vol = recent_20['Volume'].mean()

    print(f"\n  현재 거래량: {latest['Volume']:,}주")
    print(f"  20일 평균: {avg_vol:,.0f}주")

    surge_ratio = latest['Volume'] / avg_vol
    print(f"  서지 비율: {surge_ratio:.2f}x")

    if surge_ratio > 3:
        signal = "폭발적 거래량 - 중요 이벤트 발생"
    elif surge_ratio > 2:
        signal = "급증 - 관심 필요"
    elif surge_ratio > 1.5:
        signal = "증가 - 모멘텀 형성 중"
    elif surge_ratio > 0.7:
        signal = "보통 수준"
    else:
        signal = "감소 - 관심 저하"

    print(f"  해석: {signal}")

    # OBV 추세
    if 'OBV' in df.columns:
        obv_now = latest['OBV']
        obv_20ago = df.iloc[-20]['OBV'] if len(df) >= 20 else obv_now

        obv_change_pct = (obv_now / obv_20ago - 1) * 100 if obv_20ago != 0 else 0

        print(f"\n  [OBV (On-Balance Volume)]")
        print(f"  현재: {obv_now:,.0f}")
        print(f"  20일 전: {obv_20ago:,.0f}")
        print(f"  변화율: {obv_change_pct:+.2f}%")

        if obv_change_pct > 10:
            print(f"  해석: 매집 추세 강함")
        elif obv_change_pct > 0:
            print(f"  해석: 매집 추세")
        elif obv_change_pct > -10:
            print(f"  해석: 분산 추세")
        else:
            print(f"  해석: 분산 추세 강함")


def analyze_supply_demand(df: pd.DataFrame):
    """수급 분석 (외국인/기관 순매수 추이)"""
    latest = df.iloc[-1]

    print(f"\n{'='*80}")
    print(f"  [6] 수급 분석")
    print(f"{'='*80}")

    if 'Foreign_Net' in df.columns:
        foreign_net = latest['Foreign_Net']
        recent_foreign = df['Foreign_Net'].tail(5).sum()

        print(f"\n  [외국인 순매수]")
        print(f"  당일: {foreign_net:,.0f}주")
        print(f"  최근 5일 누적: {recent_foreign:,.0f}주")

        if recent_foreign > 0:
            print(f"  해석: 외국인 매수 우위")
        else:
            print(f"  해석: 외국인 매도 우위")

    if 'Inst_Net' in df.columns:
        inst_net = latest['Inst_Net']
        recent_inst = df['Inst_Net'].tail(5).sum()

        print(f"\n  [기관 순매수]")
        print(f"  당일: {inst_net:,.0f}주")
        print(f"  최근 5일 누적: {recent_inst:,.0f}주")

        if recent_inst > 0:
            print(f"  해석: 기관 매수 우위")
        else:
            print(f"  해석: 기관 매도 우위")


def analyze_geometric(df: pd.DataFrame, ticker: str = "098460"):
    """v4.7 기하학적 분석 (10지표)"""
    print(f"\n{'='*80}")
    print(f"  [7] v4.7 기하학적 분석 (GeometricQuantEngine)")
    print(f"{'='*80}")

    try:
        # GeometricQuantEngine 초기화
        config = {
            "profile": "default",
            "lookback": 200
        }

        engine = GeometricQuantEngine(config=config)

        # DataFrame 준비 (소문자 컬럼명)
        df_copy = df.copy()
        df_copy.columns = [c.lower() for c in df_copy.columns]

        # 필수 컬럼 확인
        required = ['close', 'high', 'low', 'volume']
        missing = [c for c in required if c not in df_copy.columns]

        if missing:
            print(f"\n  [!] 필수 컬럼 누락: {missing}")
            return

        # L7 결과 생성
        print(f"\n  [분석 시작...]")
        result = engine.generate_l7_result(df_copy, ticker=ticker)

        print(f"\n{result}")

    except Exception as e:
        print(f"\n  [!] 기하학적 분석 오류: {e}")
        import traceback
        traceback.print_exc()


def calculate_trading_strategy(df: pd.DataFrame):
    """매매 전략 (진입가/손절가/목표가/손익비)"""
    latest = df.iloc[-1]
    recent_10 = df.tail(10)

    print(f"\n{'='*80}")
    print(f"  [8] 매매 전략")
    print(f"{'='*80}")

    current_price = latest['Close']

    # 진입가: 현재가
    entry_price = current_price

    # 손절가: 최근 10일 저가 기준 -0.5%
    swing_low = recent_10['Low'].min()
    stop_loss = max(swing_low * 0.995, current_price * 0.97)

    # ATR 기반 목표가
    atr = latest.get('ATR', current_price * 0.02)
    target_1 = entry_price + atr * 2
    target_2 = entry_price + atr * 4
    target_3 = entry_price + atr * 8

    # 손익비
    risk = entry_price - stop_loss
    reward = target_1 - entry_price
    rr_ratio = reward / risk if risk > 0 else 0

    print(f"\n  [진입 전략]")
    print(f"  진입가: {entry_price:,.0f}원 (현재가)")
    print(f"  손절가: {stop_loss:,.0f}원 ({(stop_loss/entry_price-1)*100:+.2f}%)")
    print(f"  손실폭: {risk:,.0f}원")

    print(f"\n  [목표가 설정]")
    print(f"  1차 목표: {target_1:,.0f}원 (2R, {(target_1/entry_price-1)*100:+.2f}%) - 25% 청산")
    print(f"  2차 목표: {target_2:,.0f}원 (4R, {(target_2/entry_price-1)*100:+.2f}%) - 25% 청산")
    print(f"  3차 목표: {target_3:,.0f}원 (8R, {(target_3/entry_price-1)*100:+.2f}%) - 25% 청산")
    print(f"  장기 보유: 25% (트레일링 스탑)")

    print(f"\n  [손익비]")
    print(f"  1차 목표 기준: {rr_ratio:.2f} (위험 {risk:,.0f}원 / 보상 {reward:,.0f}원)")

    if rr_ratio >= 2:
        print(f"  평가: 우수한 손익비 (매수 적극 고려)")
    elif rr_ratio >= 1.5:
        print(f"  평가: 양호한 손익비 (매수 고려)")
    elif rr_ratio >= 1:
        print(f"  평가: 보통 손익비 (신중 검토)")
    else:
        print(f"  평가: 낮은 손익비 (매수 지양)")

    # RSI/Stochastic 기반 진입 타이밍
    print(f"\n  [진입 타이밍]")
    rsi = latest.get('RSI', 50)
    stoch_k = latest.get('Stoch_K', 50)

    if rsi < 30 and stoch_k < 20:
        timing = "과매도 구간 - 적극 매수 타이밍"
    elif rsi < 40 and stoch_k < 30:
        timing = "약세 구간 - 분할 매수 고려"
    elif rsi > 70 or stoch_k > 80:
        timing = "과매수 구간 - 진입 지양"
    else:
        timing = "중립 구간 - 추세 확인 후 진입"

    print(f"  {timing}")


def main():
    """메인 실행 함수"""
    print(f"\n{'#'*80}")
    print(f"#  고영(098460) 종합 기술적 분석 보고서")
    print(f"#  v4.7 Geometric Quant Engine + 8-Layer Analysis")
    print(f"{'#'*80}")

    # 데이터 로드
    df, name = load_stock_data("098460")

    # 1. 최근 20일 가격 추이
    analyze_recent_price_trend(df, 20)

    # 2. 이동평균선 배열
    analyze_moving_averages(df)

    # 3. RSI, MACD, Stochastic, ADX
    analyze_technical_indicators(df)

    # 4. 볼린저밴드
    analyze_bollinger_bands(df)

    # 5. 거래량 분석
    analyze_volume(df)

    # 6. 수급 분석
    analyze_supply_demand(df)

    # 7. v4.7 기하학적 분석
    analyze_geometric(df, "098460")

    # 8. 매매 전략
    calculate_trading_strategy(df)

    # 종합 의견
    print(f"\n{'='*80}")
    print(f"  [종합 의견]")
    print(f"{'='*80}")

    latest = df.iloc[-1]
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    macd_sig = latest.get('MACD_Signal', 0)

    print(f"\n  종목명: {name} (098460)")
    print(f"  분석일: {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)}")
    print(f"  현재가: {latest['Close']:,.0f}원")

    # 종합 판단
    bullish_count = 0
    total_signals = 0

    # 이평선 정배열
    if 'MA5' in df.columns and 'MA20' in df.columns and 'MA60' in df.columns:
        total_signals += 1
        if latest['Close'] > latest['MA5'] > latest['MA20'] > latest['MA60']:
            bullish_count += 1
            print(f"  ✅ 이평선 정배열")
        else:
            print(f"  ⬜ 이평선 역배열/혼재")

    # RSI
    total_signals += 1
    if 30 < rsi < 70:
        bullish_count += 0.5
        print(f"  ✅ RSI 중립/적정 구간")
    elif rsi < 30:
        bullish_count += 1
        print(f"  ✅ RSI 과매도 (반등 기회)")
    else:
        print(f"  ⬜ RSI 과매수")

    # MACD
    total_signals += 1
    if macd > macd_sig:
        bullish_count += 1
        print(f"  ✅ MACD 매수 신호")
    else:
        print(f"  ⬜ MACD 매도 신호")

    # 거래량
    total_signals += 1
    avg_vol = df['Volume'].tail(20).mean()
    if latest['Volume'] > avg_vol * 1.2:
        bullish_count += 1
        print(f"  ✅ 거래량 증가")
    else:
        print(f"  ⬜ 거래량 보통/감소")

    # 최종 점수
    score = (bullish_count / total_signals) * 100
    print(f"\n  [매수 신호 강도: {score:.0f}점/100점]")

    if score >= 75:
        recommendation = "적극 매수"
    elif score >= 50:
        recommendation = "매수 고려"
    elif score >= 25:
        recommendation = "관망"
    else:
        recommendation = "매수 지양"

    print(f"  [추천 전략: {recommendation}]")

    print(f"\n{'#'*80}")
    print(f"#  분석 완료")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
