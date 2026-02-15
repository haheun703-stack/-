"""하나투어 - 과거 매수 시그널 역추적 (v3.2: 추세 지속 포함)."""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    f = Path("stock_data_daily/하나투어_039130.csv")
    df = pd.read_csv(f, index_col="Date", parse_dates=True)
    # 추가 지표
    df["volume_surge_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["slope_ma60"] = df["MA60"].pct_change(5) * 100
    return df


def check_trend_continuation(df, idx):
    """
    Grade F 우회: 추세 지속(Trend Continuation) 판정.

    7개 조건 중 5개 이상 충족 시 보수적 진입 허용.
    1. 종가 > MA20 AND MA60
    2. ADX >= 25
    3. +DI > -DI
    4. RSI 50~72
    5. MACD > Signal
    6. 거래량 >= 20일 평균 x 0.8
    7. OBV 5일 변화 > 0
    """
    if idx < 60:
        return None

    row = df.iloc[idx]
    close = row["Close"]
    ma20 = row.get("MA20", np.nan)
    ma60 = row.get("MA60", np.nan)
    adx = row.get("ADX", 0)
    plus_di = row.get("Plus_DI", 0)
    minus_di = row.get("Minus_DI", 0)
    rsi = row.get("RSI", 50)
    macd = row.get("MACD", 0) or 0
    macd_sig = row.get("MACD_Signal", 0) or 0
    vol = row.get("Volume", 0)
    vol_ma20 = df["Volume"].iloc[max(0, idx-19):idx+1].mean() if idx >= 19 else 0
    atr = row.get("ATR", 1)

    conditions = {}

    # 1. 종가 > MA20 AND MA60
    above_ma20 = not pd.isna(ma20) and close > ma20
    above_ma60 = not pd.isna(ma60) and close > ma60
    conditions["price_above_mas"] = above_ma20 and above_ma60

    # 2. ADX >= 25
    conditions["adx_strong"] = not pd.isna(adx) and adx >= 25

    # 3. +DI > -DI
    conditions["plus_di_above"] = (
        not pd.isna(plus_di) and not pd.isna(minus_di) and plus_di > minus_di
    )

    # 4. RSI 50~72
    conditions["rsi_strong"] = not pd.isna(rsi) and 50 <= rsi <= 72

    # 5. MACD > Signal
    conditions["macd_bullish"] = macd > macd_sig

    # 6. 거래량 >= 20일 평균 x 0.8
    conditions["volume_adequate"] = vol_ma20 > 0 and vol >= vol_ma20 * 0.8

    # 7. OBV 5일 변화 > 0
    if idx >= 5 and "OBV" in df.columns:
        obv_now = df["OBV"].iloc[idx]
        obv_prev = df["OBV"].iloc[idx - 5]
        conditions["obv_rising"] = (
            not pd.isna(obv_now) and not pd.isna(obv_prev)
            and obv_now > obv_prev
        )
    else:
        conditions["obv_rising"] = False

    met_count = sum(conditions.values())

    # 5/7 이상 + RSI 72 이하
    if met_count >= 5 and (pd.isna(rsi) or rsi <= 72):
        # 손익비 계산
        stop = close * 0.97  # -3% 손절
        target = close + atr * 3  # ATR x 3 목표
        risk = close - stop
        reward = target - close
        rr = reward / risk if risk > 0 else 0

        return {
            "met_count": met_count,
            "conditions": conditions,
            "stop": int(stop),
            "target": int(target),
            "rr": round(rr, 2),
            "signal": rr >= 1.2,  # 추세 지속은 RR 1.2 이상
        }

    return None


def check_entry(df, idx):
    """해당 날짜의 Zone Score + Trigger 판정."""
    row = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else row
    close = row["Close"]
    ma20 = row.get("MA20", close)
    atr = row.get("ATR", 1)
    rsi = row.get("RSI", 50)
    stoch_k = row.get("Stoch_K", 50)
    stoch_d = row.get("Stoch_D", 50)
    adx = row.get("ADX", 0)
    plus_di = row.get("Plus_DI", 0)
    minus_di = row.get("Minus_DI", 0)
    ma5 = row.get("MA5", close)

    if pd.isna(ma20) or pd.isna(atr) or atr == 0:
        return None

    # -- Zone Score --
    pullback_atr = (ma20 - close) / atr
    if 0.5 <= pullback_atr <= 1.5:
        atr_score = 30
        zone_name = "sweet_spot"
    elif 0.25 <= pullback_atr < 0.5 or 1.5 < pullback_atr <= 2.0:
        atr_score = 20
        zone_name = "healthy/deep"
    elif 0 <= pullback_atr < 0.25:
        atr_score = 10
        zone_name = "shallow"
    else:
        atr_score = 5
        zone_name = "noise/structural"

    if 30 <= rsi <= 45:
        rsi_score = 20
    elif 45 < rsi <= 55:
        rsi_score = 15
    elif rsi < 30:
        rsi_score = 10
    else:
        rsi_score = 5

    if stoch_k < 20:
        stoch_score = 15
    elif stoch_k < 40:
        stoch_score = 10
    else:
        stoch_score = 5

    zone_score = atr_score + rsi_score + stoch_score

    if zone_score >= 55:
        grade = "A"
    elif zone_score >= 40:
        grade = "B"
    elif zone_score >= 25:
        grade = "C"
    else:
        grade = "F"

    # -- Grade F: 추세 지속(Trend Continuation) 우회 시도 --
    if grade == "F":
        tc = check_trend_continuation(df, idx)
        if tc and tc["signal"]:
            return {
                "date": str(df.index[idx].date()),
                "close": int(close), "ma20": int(ma20),
                "pullback_atr": round(pullback_atr, 2),
                "zone_name": zone_name,
                "zone_score": zone_score,
                "grade": "T",  # Trend Continuation
                "rsi": round(rsi, 1), "stoch_k": round(stoch_k, 1),
                "trigger": f"TREND({tc['met_count']}/7)",
                "signal": True,
                "stop": tc["stop"],
                "target": tc["target"],
                "rr": tc["rr"],
                "tc_conditions": tc["conditions"],
            }

        return {
            "date": str(df.index[idx].date()),
            "close": int(close), "ma20": int(ma20),
            "pullback_atr": round(pullback_atr, 2),
            "zone_name": zone_name,
            "zone_score": zone_score, "grade": grade,
            "rsi": round(rsi, 1), "stoch_k": round(stoch_k, 1),
            "trigger": "N/A (grade F)",
            "signal": False,
        }

    # -- Trigger (기존 로직) --
    # Impulse
    imp = {}
    imp["close>MA5"] = close > ma5 if not pd.isna(ma5) else False
    imp["rsi_up"] = rsi > prev.get("RSI", 50) and rsi < 70 if not pd.isna(rsi) else False

    macd = row.get("MACD", 0) or 0
    macd_sig = row.get("MACD_Signal", 0) or 0
    hist = macd - macd_sig
    prev_hist = (prev.get("MACD", 0) or 0) - (prev.get("MACD_Signal", 0) or 0)
    imp["macd_hist_up"] = hist > prev_hist

    vol_surge = row.get("volume_surge_ratio", 1.0)
    if pd.isna(vol_surge):
        vol_surge = 1.0
    imp["vol_surge"] = vol_surge > 1.2

    imp["stoch_golden"] = stoch_k > stoch_d if not pd.isna(stoch_d) else False
    imp_count = sum(imp.values())

    # Confirm
    conf = {}
    conf["close>MA20"] = close > ma20
    conf["rsi>50"] = rsi > 50
    conf["adx>20"] = adx > 20
    conf["pdi>mdi"] = plus_di > minus_di
    conf_count = sum(conf.values())

    if imp_count >= 3:
        trigger = f"IMPULSE({imp_count}/5)"
    elif conf_count >= 3:
        trigger = f"CONFIRM({conf_count}/4)"
    else:
        trigger = f"NONE(imp:{imp_count}/5, conf:{conf_count}/4)"

    signal = (imp_count >= 3 or conf_count >= 3)

    # Risk check
    if signal:
        swing_low = df["Low"].iloc[max(0, idx-9):idx+1].min()
        stop = max(swing_low * 0.995, close * 0.97)
        target = close + atr * 3
        risk = close - stop
        reward = target - close
        rr = reward / risk if risk > 0 else 0
        risk_pass = rr >= 1.5
    else:
        stop = 0
        target = 0
        rr = 0
        risk_pass = False

    return {
        "date": str(df.index[idx].date()),
        "close": int(close), "ma20": int(ma20),
        "pullback_atr": round(pullback_atr, 2),
        "zone_name": zone_name,
        "zone_score": zone_score, "grade": grade,
        "rsi": round(rsi, 1), "stoch_k": round(stoch_k, 1),
        "trigger": trigger,
        "signal": signal and risk_pass,
        "stop": int(stop) if signal else 0,
        "target": int(target) if signal else 0,
        "rr": round(rr, 2) if signal else 0,
        "imp_detail": imp if imp_count >= 3 else None,
        "conf_detail": conf if conf_count >= 3 and imp_count < 3 else None,
    }


def main():
    df = load_data()

    # 2026-01~02 스캔
    print("=" * 110)
    print("  하나투어(039130) - 매수 시그널 역추적 v3.2 (추세 지속 포함)")
    print("  기간: 2026/01/01 ~ 2026/02/13")
    print("=" * 110)
    print(f"  {'날짜':>12} | {'종가':>7} | {'MA20':>7} | {'풀백ATR':>7} | {'Zone':>5} | {'등급':>2} | {'RSI':>5} | {'Stoch':>5} | {'트리거':>20} | {'시그널':>6}")
    print("-" * 110)

    signals_found = []
    trend_signals = []

    for idx in range(len(df)):
        date = df.index[idx]
        if date < pd.Timestamp("2026-01-01"):
            continue

        result = check_entry(df, idx)
        if result is None:
            continue

        sig_mark = ""
        if result["signal"] and result["grade"] == "T":
            sig_mark = ">> TREND"
        elif result["signal"]:
            sig_mark = ">> BUY"

        grade_display = result["grade"]

        print(f"  {result['date']:>12} | {result['close']:>7,} | {result['ma20']:>7,} | {result['pullback_atr']:>7.2f} | {result['zone_score']:>5} | {grade_display:>2} | {result['rsi']:>5.1f} | {result['stoch_k']:>5.1f} | {result['trigger']:>20} | {sig_mark}")

        if result["signal"]:
            signals_found.append(result)
            if result["grade"] == "T":
                trend_signals.append(result)

    print("-" * 110)

    # ---- 기존 BUY 시그널 ----
    pullback_signals = [s for s in signals_found if s["grade"] != "T"]
    if pullback_signals:
        print(f"\n  [풀백 매수 시그널] 총 {len(pullback_signals)}건")
        print("=" * 110)
        for s in pullback_signals:
            print(f"\n  ** {s['date']} | 진입: {s['close']:,}원 | 등급: {s['grade']}")
            print(f"     ATR풀백: {s['pullback_atr']:.2f} ({s['zone_name']}) | Zone Score: {s['zone_score']}")
            print(f"     트리거: {s['trigger']}")
            print(f"     손절: {s['stop']:,}원 | 목표: {s['target']:,}원 | 손익비: {s['rr']:.2f}")
            current = 50900
            pnl_pct = (current - s["close"]) / s["close"] * 100
            print(f"     -> 현재가(50,900) 대비: {pnl_pct:+.1f}%")

    # ---- 추세 지속 시그널 (신규) ----
    if trend_signals:
        print(f"\n  [추세 지속 시그널 (v3.2 신규)] 총 {len(trend_signals)}건")
        print("=" * 110)
        for s in trend_signals:
            print(f"\n  ** {s['date']} | 진입: {s['close']:,}원 | 등급: T (Trend Continuation)")
            print(f"     ATR풀백: {s['pullback_atr']:.2f} ({s['zone_name']}) | Zone Score: {s['zone_score']}")
            print(f"     트리거: {s['trigger']}")
            print(f"     손절: {s['stop']:,}원 | 목표: {s['target']:,}원 | 손익비: {s['rr']:.2f}")
            current = 50900
            pnl_pct = (current - s["close"]) / s["close"] * 100
            print(f"     -> 현재가(50,900) 대비: {pnl_pct:+.1f}%")
            if s.get("tc_conditions"):
                met = [k for k, v in s["tc_conditions"].items() if v]
                fail = [k for k, v in s["tc_conditions"].items() if not v]
                print(f"     충족: {', '.join(met)}")
                if fail:
                    print(f"     미충족: {', '.join(fail)}")

    if not signals_found:
        print("\n  매수 시그널 발동일 없음")

    # ──── 뉴스 부스트 시뮬레이션 ────
    print("\n" + "=" * 110)
    print("  [뉴스 부스트 시뮬레이션] - 하나투어 M&A 이슈 가정")
    print("=" * 110)
    print("  시나리오: '하나투어 M&A 인수설' B급 루머 (+0.08)")
    print("           + 살아있는 이슈: 기관 매집 지속 (impact 8, +0.05)")
    print("           + 실적 beat 예상 (+0.08) = 총 부스트 +0.21")
    print("-" * 110)

    # 뉴스 부스트로 Grade F → 가능한 승격 시뮬레이션
    NEWS_BOOST = 0.21  # A급과 유사한 강력한 뉴스
    for idx in range(len(df)):
        date = df.index[idx]
        if date < pd.Timestamp("2026-02-04"):
            continue

        row = df.iloc[idx]
        close = row["Close"]
        ma20 = row.get("MA20", close)
        atr = row.get("ATR", 1)
        rsi = row.get("RSI", 50)
        stoch_k = row.get("Stoch_K", 50)

        if pd.isna(ma20) or pd.isna(atr) or atr == 0:
            continue

        pullback_atr = (ma20 - close) / atr

        # 원래 Zone Score (0~1 스케일 근사)
        if pullback_atr < 0:
            atr_score_01 = 0.0
        elif 0.5 <= pullback_atr <= 1.5:
            atr_score_01 = 1.0
        else:
            atr_score_01 = 0.4

        rsi_01 = 0.5 if pd.isna(rsi) else (1.0 if 38 <= rsi <= 45 else (0.5 if rsi <= 55 else 0.2))
        stoch_01 = 0.5 if pd.isna(stoch_k) else (1.0 if stoch_k <= 25 else (0.4 if stoch_k <= 60 else 0.1))
        raw_zone = 0.35 * atr_score_01 + 0.35 * 0.5 + 0.30 * (rsi_01 * 0.3 + stoch_01 * 0.3 + 0.5 * 0.4)
        boosted_zone = min(raw_zone + NEWS_BOOST, 1.0)

        # 등급 판정 (config 기준: A>=0.85, B>=0.70, C>=0.55)
        if boosted_zone >= 0.85:
            grade = "A"
        elif boosted_zone >= 0.70:
            grade = "B"
        elif boosted_zone >= 0.55:
            grade = "C"
        else:
            grade = "F"

        orig_grade = "F" if raw_zone < 0.55 else ("C" if raw_zone < 0.70 else "B")
        upgraded = grade != orig_grade and grade != "F"
        mark = f"  ** {orig_grade}>{grade} (뉴스 부스트)" if upgraded else ""

        tc = check_trend_continuation(df, idx)
        tc_mark = ""
        if tc and tc["signal"]:
            tc_grade = "TN" if NEWS_BOOST >= 0.08 else "T"
            tc_mark = f"  + {tc_grade}({tc['met_count']}/7)"

        print(f"  {str(date.date()):>12} | {int(close):>7,} | Zone: {raw_zone:.3f}>{boosted_zone:.3f} | {grade} |{mark}{tc_mark}")

    print("-" * 110)
    print("  [결론]")
    print("  뉴스 부스트(+0.21)로:")
    print("  1. Grade F가 C 이상으로 승격 가능 -> 기존 Pipeline(L1~L6) 진입 가능")
    print("  2. Trend Continuation 조건 5->4로 완화 + 비중 50%->70%")
    print("  3. 두 경로 모두 활성화되어 추가 상승분 포착 가능")

    print("\n" + "=" * 110)
    print("  [시스템 설명 v3.2]")
    print("  - Grade A/B/C: ATR 풀백 매수 (기존)")
    print("  - Grade T: 추세 지속 매수 (v3.2 기술적 분석)")
    print("  - Grade TN: 추세 지속 + 뉴스 강화 매수 (v3.2 뉴스 연동)")
    print("  -- 뉴스 부스트 배점:")
    print("     A급 확정공시: +0.15 | B급 루머: +0.08")
    print("     살아있는 이슈(건당): +0.03~0.05")
    print("     어닝 서프라이즈(beat): +0.08 | 실적전 매집: +0.05")
    print("     긍정 뉴스 3건+: +0.03 | 최대 한도: 0.30")
    print("=" * 110)


if __name__ == "__main__":
    main()
