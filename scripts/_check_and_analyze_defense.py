"""BAT-D 완료 감지 → 방산 종목 3/18 수급 자동 분석"""
import sys, os, time
sys.path.insert(0, r"D:\sub-agent-project")

import pandas as pd
import json
from datetime import datetime

TARGET_DATE = "2026-03-18"
DEFENSE = {
    '103140': '풍산',
    '064350': '현대로템',
}

def check_parquet_updated():
    """핵심 종목 parquet이 3/18로 업데이트됐는지"""
    for ticker in DEFENSE:
        pq = f"data/processed/{ticker}.parquet"
        if not os.path.exists(pq):
            return False
        df = pd.read_parquet(pq, columns=['close'])
        if TARGET_DATE not in str(df.index[-1]):
            return False
    return True

def analyze():
    """3/18 수급 분석"""
    print(f"\n{'='*60}")
    print(f"  방산 핵심종목 3/18 수급 분석 ({datetime.now().strftime('%H:%M:%S')})")
    print(f"{'='*60}\n")
    
    for ticker, name in DEFENSE.items():
        pq = f"data/processed/{ticker}.parquet"
        df = pd.read_parquet(pq)
        close = df['close'].dropna()
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        chg = (current - prev) / prev * 100
        
        print(f"▶ {name} ({ticker})")
        print(f"  3/18 종가: {current:,.0f} (전일비 {chg:+.1f}%)")
        
        # RSI
        if 'rsi' in df.columns:
            print(f"  RSI(14): {float(df['rsi'].iloc[-1]):.1f}")
        
        # 볼린저
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_up = float(df['bb_upper'].iloc[-1])
            bb_low = float(df['bb_lower'].iloc[-1])
            bb_pct = (current - bb_low) / (bb_up - bb_low) * 100
            print(f"  볼린저: {bb_pct:.0f}%")
        
        # 기관/외인
        if '기관합계' in df.columns:
            inst_today = float(df['기관합계'].iloc[-1])
            inst_5 = df['기관합계'].tail(5).sum()
            print(f"  기관: 당일={inst_today/1e8:+,.0f}억 / 5일합={inst_5/1e8:+,.0f}억")
        
        if 'foreign_net_5d' in df.columns:
            fgn_5 = float(df['foreign_net_5d'].iloc[-1])
            print(f"  외인 5일: {fgn_5/1e8:+,.0f}억")
        
        if '개인' in df.columns:
            ind_today = float(df['개인'].iloc[-1])
            print(f"  개인: 당일={ind_today/1e8:+,.0f}억")
        
        # OBV
        if 'obv' in df.columns:
            obv = df['obv'].dropna()
            obv_chg = float(obv.iloc[-1]) - float(obv.iloc[-2])
            print(f"  OBV 변화: {'+' if obv_chg > 0 else ''}{obv_chg:,.0f}")
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            hist = float(df['macd'].iloc[-1]) - float(df['macd_signal'].iloc[-1])
            prev_hist = float(df['macd'].iloc[-2]) - float(df['macd_signal'].iloc[-2])
            direction = "수렴중(골든 접근)" if hist > prev_hist else "확산중(데드 심화)"
            print(f"  MACD hist: {hist:+,.0f} → {direction}")
        
        print()

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 방산 종목 3/18 데이터 감시 시작...")
    
    while True:
        if check_parquet_updated():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 3/18 데이터 확인! 분석 시작")
            analyze()
            
            # 결과를 파일로도 저장
            with open("data/defense_analysis_318.txt", "w", encoding="utf-8") as f:
                import io
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                analyze()
                sys.stdout = old_stdout
                f.write(buffer.getvalue())
            
            print("✅ 분석 완료 → data/defense_analysis_318.txt 저장됨")
            break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 아직 3/17... 2분 후 재확인")
            time.sleep(120)

if __name__ == "__main__":
    main()
