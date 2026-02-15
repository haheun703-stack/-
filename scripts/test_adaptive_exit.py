"""
v4.1 적응형 청산 시뮬레이션 테스트

하나투어(039130) 1/26~2/13 패턴 시뮬레이션:
  1/26: 진입 (Impulse 트리거) @ 약 69,000원
  1/27~1/29: 소폭 상승 → 최고가 71,500원
  1/30~2/4: 조정 (-3~5% 하락) — 이 구간에서 기존 -3% 손절이 발동
  2/5~2/13: 재상승 → 75,000원+

기존 시스템: -3% 도달 시 즉시 청산 → 이후 상승 놓침
적응형 시스템: 건강한 조정 판정 → MA20 기반 넓은 손절 → 추세 지속 포착
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.use_cases.adaptive_exit import AdaptiveExitManager
from src.use_cases.daily_hold_scorer import DailyHoldScorer


def create_hanatour_simulation() -> pd.DataFrame:
    """
    하나투어 1/26~2/13 패턴 합성 데이터.

    실제 패턴 기반:
    - 1/26: 시동 트리거 발동 (전일 고가 돌파, 거래량 폭증)
    - 1/27~29: 상승 지속 (OBV 상승, MA20 위, ADX 강화)
    - 1/30~2/4: 조정 (주가 -4%, 하지만 OBV 유지, 거래량 감소)
    - 2/5~13: 재상승 (골든크로스 확인, 스마트머니 유입)
    """
    dates = pd.date_range("2026-01-10", "2026-02-13", freq="B")  # 영업일
    n = len(dates)

    # 주가 시나리오 설계 (실제 하나투어 패턴 반영)
    # 핵심: 진입가 69,000 대비 조정 66,500(-3.6%)이 기존 -3% 손절 발동
    target_prices = {}
    for i, d in enumerate(dates):
        ds = d.strftime("%m-%d")
        if ds <= "01-23":
            target_prices[ds] = 64000 + i * 200  # 매집 구간
        elif ds == "01-26":
            target_prices[ds] = 69000  # 시동 트리거 진입 포인트
        elif ds == "01-27":
            target_prices[ds] = 70500
        elif ds == "01-28":
            target_prices[ds] = 72000  # 최고점
        elif ds == "01-29":
            target_prices[ds] = 71200  # 조정 시작
        elif ds == "01-30":
            target_prices[ds] = 69500
        elif ds == "02-02":
            target_prices[ds] = 68000
        elif ds == "02-03":
            target_prices[ds] = 67000
        elif ds == "02-04":
            target_prices[ds] = 66500  # 최저점 (진입가 대비 -3.6%)
        elif ds == "02-05":
            target_prices[ds] = 67200  # 반등 시작
        elif ds == "02-06":
            target_prices[ds] = 69000
        elif ds == "02-09":
            target_prices[ds] = 71000
        elif ds == "02-10":
            target_prices[ds] = 73000
        elif ds == "02-11":
            target_prices[ds] = 74500
        elif ds == "02-12":
            target_prices[ds] = 76000
        elif ds == "02-13":
            target_prices[ds] = 77500
        else:
            target_prices[ds] = 65000 + i * 150

    prices = []
    for i in range(n):
        d = dates[i]
        day_str = d.strftime("%m-%d")
        base_price = target_prices.get(day_str, 65000)
        # 최소 노이즈 (진입/조정 포인트 정확도 유지)
        noise = np.random.uniform(-100, 100)
        prices.append(max(base_price + noise, 60000))

    close = np.array(prices)

    # OHLC 생성
    high = close * (1 + np.random.uniform(0.005, 0.02, n))
    low = close * (1 - np.random.uniform(0.005, 0.02, n))
    open_p = close * (1 + np.random.uniform(-0.01, 0.01, n))

    # 거래량: 조정 구간에서 감소 (건강한 신호)
    volume = np.zeros(n)
    for i in range(n):
        d = dates[i].strftime("%m-%d")
        if d <= "01-24":
            volume[i] = np.random.uniform(300000, 500000)
        elif d <= "01-28":
            volume[i] = np.random.uniform(800000, 1500000)  # 폭증
        elif d <= "02-04":
            volume[i] = np.random.uniform(200000, 400000)  # 감소 (건강한 조정)
        else:
            volume[i] = np.random.uniform(500000, 900000)  # 재증가

    # 기술적 지표 계산
    df = pd.DataFrame({
        "date": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # MA
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_60"] = df["close"].rolling(min(20, n)).mean()  # 데이터 적으므로 20일로
    df["sma_120"] = df["close"].rolling(min(20, n)).mean()

    # RSI (14일)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - 100 / (1 + rs)

    # ADX (간이 계산)
    df["adx"] = np.linspace(20, 35, n) + np.random.uniform(-3, 3, n)

    # +DI, -DI
    df["plus_di"] = np.linspace(25, 35, n) + np.random.uniform(-2, 2, n)
    df["minus_di"] = np.linspace(20, 15, n) + np.random.uniform(-2, 2, n)

    # OBV
    obv = [0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i] * 0.3)  # OBV 하락폭 제한 (매집 유지)
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # 수급 (기관/외국인)
    inst_net = np.zeros(n)
    foreign_net = np.zeros(n)
    for i in range(n):
        d = dates[i].strftime("%m-%d")
        if d <= "01-24":
            inst_net[i] = np.random.uniform(-50000, 100000)
            foreign_net[i] = np.random.uniform(0, 80000)
        elif d <= "01-28":
            inst_net[i] = np.random.uniform(50000, 300000)  # 기관 매수
            foreign_net[i] = np.random.uniform(30000, 200000)
        elif d <= "02-04":
            # 조정 중에도 수급 유지 (건강한 조정의 핵심)
            inst_net[i] = np.random.uniform(-20000, 50000)
            foreign_net[i] = np.random.uniform(10000, 80000)
        else:
            inst_net[i] = np.random.uniform(50000, 250000)
            foreign_net[i] = np.random.uniform(50000, 200000)
    df["inst_net"] = inst_net
    df["foreign_net"] = foreign_net

    df = df.bfill().ffill()
    return df


def simulate_old_system(df: pd.DataFrame, entry_idx: int) -> dict:
    """기존 시스템 시뮬레이션 (고정 -3% 손절)"""
    entry_price = float(df.iloc[entry_idx]["close"])
    stop_loss_pct = 0.03
    highest_price = entry_price

    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        close = float(row["close"])
        high = float(row["high"])

        if high > highest_price:
            highest_price = high

        pct_loss = (close / entry_price - 1)
        if pct_loss <= -stop_loss_pct:
            return {
                "exit_idx": i,
                "exit_date": str(df.iloc[i]["date"])[:10],
                "exit_price": close,
                "pnl_pct": round(pct_loss * 100, 2),
                "reason": f"pct_stop (-{stop_loss_pct*100:.0f}%)",
                "days_held": i - entry_idx,
                "highest_price": highest_price,
            }

    # 기간 내 미청산
    last_close = float(df.iloc[-1]["close"])
    return {
        "exit_idx": len(df) - 1,
        "exit_date": str(df.iloc[-1]["date"])[:10],
        "exit_price": last_close,
        "pnl_pct": round((last_close / entry_price - 1) * 100, 2),
        "reason": "미청산 (보유 유지)",
        "days_held": len(df) - 1 - entry_idx,
        "highest_price": highest_price,
    }


def simulate_adaptive_system(df: pd.DataFrame, entry_idx: int) -> dict:
    """v4.1 적응형 시스템 시뮬레이션"""
    entry_price = float(df.iloc[entry_idx]["close"])
    highest_price = entry_price

    adaptive = AdaptiveExitManager()
    scorer = DailyHoldScorer()

    health_log = []
    score_log = []

    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        close = float(row["close"])
        high = float(row["high"])
        hold_days = i - entry_idx

        if high > highest_price:
            highest_price = high

        # 조정 건강도 평가
        health = adaptive.evaluate_pullback(
            df, i, entry_price, highest_price, "impulse",
        )
        health_log.append({
            "day": hold_days,
            "date": str(df.iloc[i]["date"])[:10],
            "close": close,
            "score": health.health_score,
            "class": health.classification,
            "adj_stop_pct": health.adjusted_stop_pct,
        })

        # 일일 보유 점수
        hold_result = scorer.score(
            df, i, entry_price, highest_price,
            trigger_type="impulse", hold_days=hold_days,
        )
        score_log.append({
            "day": hold_days,
            "total": hold_result.total_score,
            "action": hold_result.action,
            "tech": hold_result.technical_score,
            "sd": hold_result.supply_demand_score,
        })

        # 청산 판정
        pct_loss = (close / entry_price - 1)

        if hold_result.action == "exit":
            return {
                "exit_idx": i,
                "exit_date": str(df.iloc[i]["date"])[:10],
                "exit_price": close,
                "pnl_pct": round(pct_loss * 100, 2),
                "reason": f"hold_score_exit (score={hold_result.total_score:.0f})",
                "days_held": hold_days,
                "highest_price": highest_price,
                "health_log": health_log,
                "score_log": score_log,
            }

        if health.classification == "critical":
            return {
                "exit_idx": i,
                "exit_date": str(df.iloc[i]["date"])[:10],
                "exit_price": close,
                "pnl_pct": round(pct_loss * 100, 2),
                "reason": f"adaptive_critical (health={health.health_score:.0f})",
                "days_held": hold_days,
                "highest_price": highest_price,
                "health_log": health_log,
                "score_log": score_log,
            }

        if health.classification == "dangerous":
            if pct_loss <= -health.adjusted_stop_pct:
                return {
                    "exit_idx": i,
                    "exit_date": str(df.iloc[i]["date"])[:10],
                    "exit_price": close,
                    "pnl_pct": round(pct_loss * 100, 2),
                    "reason": f"adaptive_dangerous (stop={health.adjusted_stop_pct*100:.1f}%)",
                    "days_held": hold_days,
                    "highest_price": highest_price,
                    "health_log": health_log,
                    "score_log": score_log,
                }

        if health.classification == "healthy":
            if health.adjusted_stop_price > 0 and close <= health.adjusted_stop_price:
                return {
                    "exit_idx": i,
                    "exit_date": str(df.iloc[i]["date"])[:10],
                    "exit_price": close,
                    "pnl_pct": round(pct_loss * 100, 2),
                    "reason": f"adaptive_healthy_stop (stop={health.adjusted_stop_price:,.0f})",
                    "days_held": hold_days,
                    "highest_price": highest_price,
                    "health_log": health_log,
                    "score_log": score_log,
                }

    # 기간 내 미청산
    last_close = float(df.iloc[-1]["close"])
    return {
        "exit_idx": len(df) - 1,
        "exit_date": str(df.iloc[-1]["date"])[:10],
        "exit_price": last_close,
        "pnl_pct": round((last_close / entry_price - 1) * 100, 2),
        "reason": "미청산 (보유 유지)",
        "days_held": len(df) - 1 - entry_idx,
        "highest_price": highest_price,
        "health_log": health_log,
        "score_log": score_log,
    }


def main():
    print("=" * 70)
    print("v4.1 적응형 청산 시뮬레이션: 하나투어 1/26~2/13 패턴")
    print("=" * 70)

    np.random.seed(42)  # 재현 가능
    df = create_hanatour_simulation()

    # 진입일 찾기 (1/26 근처)
    entry_idx = None
    for i, row in df.iterrows():
        if row["date"].strftime("%m-%d") == "01-26":
            entry_idx = i
            break
    if entry_idx is None:
        entry_idx = 12  # fallback

    entry_price = float(df.iloc[entry_idx]["close"])
    print(f"\n진입일: {df.iloc[entry_idx]['date'].strftime('%Y-%m-%d')}")
    print(f"진입가: {entry_price:,.0f}원")

    # ── 기존 시스템 ──
    print(f"\n{'─' * 50}")
    print("  [기존] 고정 -3% 손절")
    print(f"{'─' * 50}")
    old = simulate_old_system(df, entry_idx)
    print(f"  청산일: {old['exit_date']}")
    print(f"  청산가: {old['exit_price']:,.0f}원")
    print(f"  수익률: {old['pnl_pct']:+.2f}%")
    print(f"  사유:   {old['reason']}")
    print(f"  보유일: {old['days_held']}일")
    print(f"  최고가: {old['highest_price']:,.0f}원")

    # ── 적응형 시스템 ──
    print(f"\n{'─' * 50}")
    print("  [v4.1] 적응형 청산 시스템")
    print(f"{'─' * 50}")
    new = simulate_adaptive_system(df, entry_idx)
    print(f"  청산일: {new['exit_date']}")
    print(f"  청산가: {new['exit_price']:,.0f}원")
    print(f"  수익률: {new['pnl_pct']:+.2f}%")
    print(f"  사유:   {new['reason']}")
    print(f"  보유일: {new['days_held']}일")
    print(f"  최고가: {new['highest_price']:,.0f}원")

    # ── 비교 ──
    diff = new["pnl_pct"] - old["pnl_pct"]
    print(f"\n{'=' * 50}")
    print(f"  수익률 차이: {diff:+.2f}%p")
    if diff > 0:
        print(f"  → 적응형 시스템이 {diff:.2f}%p 더 벌었다!")
    elif diff < 0:
        print(f"  → 기존 시스템이 {abs(diff):.2f}%p 더 벌었다")
    else:
        print(f"  → 동일 성과")
    print(f"{'=' * 50}")

    # ── 일일 건강도 + 보유 점수 로그 ──
    if "health_log" in new:
        print(f"\n{'─' * 70}")
        print("  일일 조정 건강도 + 보유 점수")
        print(f"{'─' * 70}")
        print(f"  {'일차':>4} | {'날짜':>10} | {'종가':>8} | {'건강':>5} | {'분류':>10} | {'점수':>5} | {'판단':>12}")
        print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*10}─┼─{'─'*5}─┼─{'─'*12}")

        for h, s in zip(new.get("health_log", []), new.get("score_log", [])):
            cls_emoji = {
                "healthy": "  OK  ",
                "caution": " WARN ",
                "dangerous": "DANGER",
                "critical": " CRIT ",
            }.get(h["class"], h["class"])

            action_emoji = {
                "strong_hold": "  STRONG ",
                "hold": "  HOLD   ",
                "tighten": " TIGHTEN ",
                "exit": "  EXIT   ",
            }.get(s["action"], s["action"])

            print(f"  D+{h['day']:>2} | {h['date']:>10} | {h['close']:>8,.0f} | "
                  f"{h['score']:>5.1f} | {cls_emoji:>10} | {s['total']:>5.1f} | {action_emoji:>12}")


if __name__ == "__main__":
    main()
