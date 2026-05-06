"""
=============================================================
  퀀트봇 v2.0 — 4모듈 통합 자동매매 엔진
=============================================================
  모듈 1. 진입 타이밍 점수 (score_entry)
  모듈 2. 리스크 관리 / ATR 동적 손절 (risk_manager)
  모듈 3. 종목 필터링 파이프라인 (stock_filter)
  모듈 4. 레버리지/인버스 방향성 판단 (direction_signal)
  통합.  매일 장 마감 후 실행되는 메인 루틴 (run_daily)
=============================================================
  필요 라이브러리:
    pip install pykrx FinanceDataReader yfinance pandas numpy
=============================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from pykrx import stock as krx
    import FinanceDataReader as fdr
    import yfinance as yf
    LIVE_DATA = True
except ImportError:
    LIVE_DATA = False
    print("[경고] 데이터 라이브러리 미설치. 샘플 데이터로 시뮬레이션합니다.")
    print("       pip install pykrx FinanceDataReader yfinance")


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range 계산"""
    h, l, c = df['high'], df['low'], df['close']
    tr = np.maximum(h - l,
         np.maximum(abs(h - c.shift(1)),
                    abs(l - c.shift(1))))
    return tr.rolling(period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 기술적 지표 컬럼 추가"""
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'])
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['atr'] = compute_atr(df)
    df['ma5']  = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    return df.dropna()


# ──────────────────────────────────────────────
# 모듈 1. 진입 타이밍 점수
# ──────────────────────────────────────────────

def score_entry(df: pd.DataFrame, i: int = -1) -> dict:
    """
    기술적 복합 점수 계산 (최대 100점)
    반환: {'score': int, 'signals': list, 'action': str}
    """
    df = add_indicators(df)
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    signals = []
    score = 0

    # [1] RSI 반등 구간 (25점)
    if 30 <= row['rsi'] <= 50:
        score += 25
        signals.append(f"RSI 반등 구간 ({row['rsi']:.1f})")
    elif row['rsi'] < 30:
        score += 15
        signals.append(f"RSI 과매도 ({row['rsi']:.1f}) — 반등 주의")

    # [2] MACD 골든크로스 (25점)
    if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
        score += 25
        signals.append("MACD 골든크로스 발생")
    elif row['macd'] > row['macd_signal']:
        score += 10
        signals.append("MACD 상향 유지")

    # [3] 거래량 급증 (20점)
    vol_ratio = row['volume'] / row['vol_ma20'] if row['vol_ma20'] > 0 else 0
    if vol_ratio >= 2.0:
        score += 20
        signals.append(f"거래량 {vol_ratio:.1f}배 급증")
    elif vol_ratio >= 1.5:
        score += 10
        signals.append(f"거래량 {vol_ratio:.1f}배 증가")

    # [4] 강한 양봉 캔들 (15점)
    body = abs(row['close'] - row['open'])
    wick = row['high'] - row['low']
    if wick > 0 and row['close'] > row['open']:
        body_ratio = body / wick
        if body_ratio >= 0.6:
            score += 15
            signals.append(f"강한 양봉 (몸통 {body_ratio*100:.0f}%)")

    # [5] 이동평균 정배열 (15점)
    if row['close'] > row['ma5'] > row['ma20']:
        score += 15
        signals.append("단기 이동평균 정배열")
        if row['ma20'] > row['ma60']:
            score += 5  # 보너스
            signals.append("중장기 정배열 보너스 +5")

    # 액션 결정
    if score >= 70:
        action = "ENTER_FULL"       # 풀 포지션 진입
    elif score >= 40:
        action = "ENTER_PARTIAL"    # 25% 소량 진입
    else:
        action = "PASS"             # 패스

    return {'score': score, 'signals': signals, 'action': action}


# ──────────────────────────────────────────────
# 모듈 2. 리스크 관리 엔진
# ──────────────────────────────────────────────

def dynamic_stop(df: pd.DataFrame, entry_price: float, multiplier: float = 2.0) -> float:
    """ATR 기반 동적 손절가 계산"""
    df = add_indicators(df)
    current_atr = df['atr'].iloc[-1]
    stop = entry_price - (current_atr * multiplier)
    return round(stop, 0)

def position_size(capital: float, risk_pct: float,
                  entry: float, stop: float) -> dict:
    """
    capital  : 총 자금 (원)
    risk_pct : 허용 손실 비율 (예: 0.02 = 2%)
    entry    : 진입가
    stop     : 손절가
    """
    if entry <= stop:
        return {'shares': 0, 'invest': 0, 'max_loss': 0}

    risk_amount   = capital * risk_pct
    loss_per_unit = entry - stop
    shares        = int(risk_amount / loss_per_unit)
    invest_amount = shares * entry
    max_loss      = shares * loss_per_unit

    # 전체 자금 20% 상한선
    if invest_amount > capital * 0.20:
        shares        = int((capital * 0.20) / entry)
        invest_amount = shares * entry
        max_loss      = shares * loss_per_unit

    return {
        'shares':       shares,
        'invest':       int(invest_amount),
        'max_loss':     int(max_loss),
        'stop_price':   stop,
        'entry_price':  entry,
        'risk_pct_real': round(max_loss / capital * 100, 2)
    }

def trailing_stop(entry: float, highest: float,
                  atr_val: float, multiplier: float = 2.0) -> float:
    """트레일링 스탑 (최고가 기준 자동 상향)"""
    trail = highest - (atr_val * multiplier)
    return max(trail, entry)  # 진입가 아래로 절대 내려가지 않음

def check_monthly_drawdown(trade_log: list, capital: float,
                            limit_pct: float = 0.10) -> bool:
    """
    이번 달 손실이 한도 초과 시 True 반환 → 매매 중단 신호
    trade_log: [{'date': str, 'pnl': int}, ...]
    """
    now = datetime.now()
    month_pnl = sum(
        t['pnl'] for t in trade_log
        if datetime.strptime(t['date'], '%Y-%m-%d').month == now.month
           and datetime.strptime(t['date'], '%Y-%m-%d').year == now.year
    )
    drawdown = -month_pnl / capital
    if drawdown >= limit_pct:
        print(f"[경고] 월간 드로우다운 {drawdown*100:.1f}% — 매매 중단")
        return True
    return False


# ──────────────────────────────────────────────
# 모듈 3. 종목 필터링 파이프라인
# ──────────────────────────────────────────────

def fetch_ohlcv(code: str, days: int = 300) -> pd.DataFrame:
    """종목 OHLCV 데이터 가져오기"""
    if not LIVE_DATA:
        return _mock_ohlcv(days)

    end   = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=days)).strftime('%Y%m%d')
    try:
        df = krx.get_market_ohlcv_by_date(start, end, code)
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'value', 'count']
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        print(f"[{code}] 데이터 오류: {e}")
        return pd.DataFrame()

def get_all_kospi_codes() -> list:
    """코스피 전 종목 코드 반환"""
    if not LIVE_DATA:
        return ['005930', '000660', '035420', '051910', '006400',
                '035720', '068270', '207940', '005380', '028260']
    today = datetime.today().strftime('%Y%m%d')
    return krx.get_market_ticker_list(today, market='KOSPI')

def stock_filter(codes: list = None, top_n: int = 10) -> list:
    """
    전 종목 스캔 → 조건 통과 종목 상위 N개 반환
    """
    if codes is None:
        codes = get_all_kospi_codes()

    candidates = []
    print(f"[필터링] 총 {len(codes)}개 종목 스캔 시작...")

    for i, code in enumerate(codes):
        if i % 50 == 0:
            print(f"  진행: {i}/{len(codes)}")

        df = fetch_ohlcv(code)
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)
        latest = df.iloc[-1]
        score = 0

        # ── 거래대금 필터 (유동성) ──
        turnover = latest['volume'] * latest['close']
        if turnover < 5_000_000_000:
            continue

        # ── 변동성 필터 ──
        atr_pct = (latest['atr'] / latest['close']) * 100
        if atr_pct < 1.5 or atr_pct > 8.0:
            continue

        # ── 52주 위치 (저평가 구간) ──
        low52  = df['low'].rolling(252, min_periods=60).min().iloc[-1]
        high52 = df['high'].rolling(252, min_periods=60).max().iloc[-1]
        range52 = high52 - low52
        if range52 > 0:
            pos52 = (latest['close'] - low52) / range52
            if 0.05 <= pos52 <= 0.45:
                score += 30  # 저점에서 반등 구간

        # ── 거래량 폭발 ──
        vol_ratio = latest['volume'] / latest['vol_ma20'] if latest['vol_ma20'] > 0 else 0
        if vol_ratio >= 2.0:
            score += 25

        # ── 기술적 점수 합산 ──
        entry_result = score_entry(df)
        score += entry_result['score']

        candidates.append({
            'code':      code,
            'score':     score,
            'pos52':     round(pos52, 3) if range52 > 0 else 0,
            'vol_ratio': round(vol_ratio, 2),
            'atr_pct':   round(atr_pct, 2),
            'close':     int(latest['close']),
            'signals':   entry_result['signals'],
            'action':    entry_result['action']
        })

    result = sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_n]
    print(f"[필터링 완료] 상위 {len(result)}개 종목 선별")
    return result


# ──────────────────────────────────────────────
# 모듈 4. 레버리지/인버스 방향성 판단
# ──────────────────────────────────────────────

def fetch_market_indicators() -> dict:
    """시장 방향성 판단에 필요한 지표 수집"""
    if not LIVE_DATA:
        return _mock_market_data()

    data = {}
    try:
        # VIX
        vix_df = yf.download('^VIX', period='5d', progress=False)
        data['vix'] = float(vix_df['Close'].iloc[-1])

        # S&P500 선물 변화율
        spx_df = yf.download('ES=F', period='5d', progress=False)
        spx_close = spx_df['Close']
        data['sp500_futures_change'] = float(
            (spx_close.iloc[-1] - spx_close.iloc[-2]) / spx_close.iloc[-2] * 100
        )

        # KOSPI RSI
        kospi_df = fdr.DataReader('KS11', datetime.today() - timedelta(days=60))
        kospi_df.columns = [c.lower() for c in kospi_df.columns]
        data['kospi_rsi'] = float(compute_rsi(kospi_df['close']).iloc[-1])

        # 외국인 순매수 (최근 3일)
        today = datetime.today().strftime('%Y%m%d')
        start = (datetime.today() - timedelta(days=10)).strftime('%Y%m%d')
        inv = krx.get_market_trading_value_by_date(start, today, 'KOSPI')
        if '외국인합계' in inv.columns:
            foreign_3d = inv['외국인합계'].tail(3)
            data['foreign_net_buy_3d'] = int((foreign_3d > 0).sum() - (foreign_3d < 0).sum())
        else:
            data['foreign_net_buy_3d'] = 0

        data['short_ratio_change'] = 0  # 별도 API 필요

    except Exception as e:
        print(f"[지표 수집 오류] {e} — 기본값 사용")
        data = _mock_market_data()

    return data

def direction_signal(market_data: dict) -> dict:
    """
    레버리지 vs 인버스 vs HOLD 판단
    반환: {'direction': str, 'bull': int, 'bear': int, 'reasons': list}
    """
    vix  = market_data.get('vix', 20)
    spx  = market_data.get('sp500_futures_change', 0)
    fgn  = market_data.get('foreign_net_buy_3d', 0)
    rsi  = market_data.get('kospi_rsi', 50)
    srt  = market_data.get('short_ratio_change', 0)

    bull_score = 0
    bear_score = 0
    reasons = []

    # ── 상승 신호 ──
    if spx > 0.5:
        bull_score += 30
        reasons.append(f"S&P500 선물 +{spx:.1f}% (상승)")
    if fgn >= 2:
        bull_score += 25
        reasons.append(f"외국인 {fgn}일 연속 순매수")
    if vix < 15:
        bull_score += 20
        reasons.append(f"VIX 낮음 ({vix:.1f}) — 리스크온")
    if rsi < 45:
        bull_score += 15
        reasons.append(f"KOSPI RSI 과매도 ({rsi:.1f}) — 반등 기대")
    if srt < -0.2:
        bull_score += 10
        reasons.append("공매도 잔고 감소 — 숏 커버링")

    # ── 하락 신호 ──
    if spx < -0.5:
        bear_score += 30
        reasons.append(f"S&P500 선물 {spx:.1f}% (하락)")
    if fgn <= -2:
        bear_score += 25
        reasons.append(f"외국인 {abs(fgn)}일 연속 순매도")
    if vix > 25:
        bear_score += 20
        reasons.append(f"VIX 급등 ({vix:.1f}) — 공포 지수")
    if rsi > 70:
        bear_score += 15
        reasons.append(f"KOSPI RSI 과매수 ({rsi:.1f}) — 조정 가능")
    if srt > 0.3:
        bear_score += 10
        reasons.append("공매도 잔고 증가")

    # ── 방향 결정 ──
    if bull_score >= 60 and bull_score > bear_score:
        direction = 'LEVERAGE'
    elif bear_score >= 60 and bear_score > bull_score:
        direction = 'INVERSE'
    else:
        direction = 'HOLD'  # 애매하면 무조건 관망

    return {
        'direction':  direction,
        'bull_score': bull_score,
        'bear_score': bear_score,
        'reasons':    reasons
    }


# ──────────────────────────────────────────────
# 통합 메인 루틴
# ──────────────────────────────────────────────

def run_daily(capital: float = 10_000_000,
              risk_pct: float = 0.02,
              trade_log: list = None):
    """
    매일 장 마감 후 (15:30~) 실행하는 메인 루틴

    capital   : 총 투자 자금 (원)
    risk_pct  : 1회 최대 허용 손실 비율 (기본 2%)
    trade_log : 이전 거래 기록 리스트
    """
    print("\n" + "="*55)
    print(f"  퀀트봇 v2.0  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)

    if trade_log is None:
        trade_log = []

    # ── Step 1. 월간 드로우다운 체크 ──
    print("\n[Step 1] 월간 드로우다운 체크")
    if check_monthly_drawdown(trade_log, capital, limit_pct=0.10):
        print("→ 이번 달 매매 중단. 다음 달 재시작.")
        return []

    # ── Step 2. 시장 방향성 판단 ──
    print("\n[Step 2] 시장 방향성 판단")
    market_data = fetch_market_indicators()
    direction   = direction_signal(market_data)
    print(f"  VIX: {market_data.get('vix', 'N/A'):.1f}  |  "
          f"S&P선물: {market_data.get('sp500_futures_change', 0):+.1f}%  |  "
          f"외국인: {market_data.get('foreign_net_buy_3d', 0):+d}일")
    print(f"  방향 판단: {direction['direction']}  "
          f"(상승점수 {direction['bull_score']} vs 하락점수 {direction['bear_score']})")
    for r in direction['reasons']:
        print(f"    - {r}")

    # ── Step 3. 종목 필터링 ──
    print("\n[Step 3] 종목 필터링 실행")
    candidates = stock_filter(top_n=10)

    if not candidates:
        print("→ 조건 통과 종목 없음. 오늘은 관망.")
        return []

    # ── Step 4. 진입 후보 선정 & 포지션 계산 ──
    print("\n[Step 4] 진입 계획 생성")
    orders = []

    for c in candidates:
        if c['action'] == 'PASS':
            continue

        df = fetch_ohlcv(c['code'])
        if df is None or len(df) < 20:
            continue

        entry_price = int(df['close'].iloc[-1])
        stop_price  = dynamic_stop(df, entry_price, multiplier=2.0)
        sizing      = position_size(capital, risk_pct, entry_price, stop_price)

        if sizing['shares'] == 0:
            continue

        order = {
            'code':        c['code'],
            'entry_price': entry_price,
            'stop_price':  stop_price,
            'target_price': int(entry_price * 1.10),  # 10% 목표
            'shares':      sizing['shares'],
            'invest':      sizing['invest'],
            'max_loss':    sizing['max_loss'],
            'score':       c['score'],
            'action':      c['action'],
            'signals':     c['signals'],
        }
        orders.append(order)

        print(f"\n  [{c['code']}] 점수: {c['score']}  |  액션: {c['action']}")
        print(f"    진입가: {entry_price:,}원  |  손절: {stop_price:,}원  |"
              f"  목표: {order['target_price']:,}원")
        print(f"    수량: {sizing['shares']}주  |  투자금: {sizing['invest']:,}원  |"
              f"  최대손실: {sizing['max_loss']:,}원")
        for sig in c['signals'][:3]:
            print(f"    * {sig}")

    # ── Step 5. 레버리지/인버스 서브 포지션 ──
    print("\n[Step 5] 레버리지/인버스 판단")
    lev_budget = int(capital * 0.10)  # 전체 자금의 10%만

    if direction['direction'] == 'LEVERAGE':
        print(f"  → LEVERAGE ETF 진입  (예산: {lev_budget:,}원)")
        print("     추천: KODEX 레버리지 (122630) 또는 TIGER 레버리지")
        orders.append({
            'code':    'LEVERAGE_ETF',
            'invest':  lev_budget,
            'action':  'LEVERAGE',
            'signals': direction['reasons']
        })
    elif direction['direction'] == 'INVERSE':
        print(f"  → INVERSE ETF 진입  (예산: {lev_budget:,}원)")
        print("     추천: KODEX 인버스 (114800) 또는 KODEX 200선물인버스2X")
        orders.append({
            'code':    'INVERSE_ETF',
            'invest':  lev_budget,
            'action':  'INVERSE',
            'signals': direction['reasons']
        })
    else:
        print("  → HOLD — 방향성 불분명, 레버리지/인버스 미진입")

    # ── 최종 요약 ──
    print("\n" + "="*55)
    print(f"  오늘의 진입 후보: {len(orders)}건")
    total_invest = sum(o.get('invest', 0) for o in orders)
    print(f"  총 예정 투자금: {total_invest:,}원  ({total_invest/capital*100:.1f}%)")
    print("="*55 + "\n")

    return orders


# ──────────────────────────────────────────────
# 샘플 데이터 (라이브러리 미설치 시 사용)
# ──────────────────────────────────────────────

def _mock_ohlcv(days: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days, freq='B')
    close = 50000 + np.cumsum(np.random.randn(days) * 500)
    close = np.maximum(close, 10000)
    df = pd.DataFrame({
        'open':   close * (1 + np.random.randn(days) * 0.003),
        'high':   close * (1 + abs(np.random.randn(days)) * 0.008),
        'low':    close * (1 - abs(np.random.randn(days)) * 0.008),
        'close':  close,
        'volume': np.random.randint(500000, 5000000, days).astype(float)
    }, index=dates)
    # 마지막 날 거래량 급증 시뮬레이션
    df['volume'].iloc[-1] *= 2.5
    return df

def _mock_market_data() -> dict:
    return {
        'vix': 16.5,
        'sp500_futures_change': 0.7,
        'foreign_net_buy_3d': 2,
        'kospi_rsi': 44.0,
        'short_ratio_change': -0.3
    }


# ──────────────────────────────────────────────
# 실행 진입점
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── 설정 ──────────────────────────────────
    MY_CAPITAL   = 10_000_000   # 총 자금 (1천만원 예시)
    RISK_PER_TRADE = 0.02       # 1회 최대 손실 2%
    MY_TRADE_LOG   = []         # 실제 사용 시 DB나 CSV에서 로드
    # ─────────────────────────────────────────

    orders = run_daily(
        capital   = MY_CAPITAL,
        risk_pct  = RISK_PER_TRADE,
        trade_log = MY_TRADE_LOG
    )

    print(f"\n총 {len(orders)}개 주문 생성 완료.")
    print("내일 9:05 이후 각 주문을 증권사 API에 전달하세요.")
    print("\n[주의] 갭 상승 3% 초과 종목은 당일 진입 보류 권장")
