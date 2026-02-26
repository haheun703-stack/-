"""
진입 패턴 실증 분석 — 3가지 전략 백테스트

1. MACD 0선 크로스오버 전략
2. 수급 폭발 → 조정 → 급등 패턴
3. 이벤트(US 실적) → KR 섹터 사전 포지셔닝

84종목 processed parquet 기반, 3년 데이터
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED = DATA_DIR / "processed"
UNIVERSE = DATA_DIR / "mechanical_universe_final.csv"


# ──────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────

def load_universe() -> dict[str, str]:
    """유니버스 종목코드 → 이름 매핑."""
    if not UNIVERSE.exists():
        return {}
    df = pd.read_csv(UNIVERSE)
    return dict(zip(df["ticker"].astype(str).str.zfill(6), df["name"]))


def load_all_parquets() -> dict[str, pd.DataFrame]:
    """전체 processed parquet 로드."""
    result = {}
    for f in sorted(PROCESSED.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            if len(df) < 120:  # 최소 6개월
                continue
            df = df.sort_index()
            result[f.stem] = df
        except Exception:
            continue
    return result


def calc_forward_returns(df: pd.DataFrame, entry_idx: int, days: list[int] = [1, 3, 5, 10, 20]) -> dict:
    """진입일 기준 N일 후 수익률 계산."""
    returns = {}
    entry_close = df.iloc[entry_idx]["close"]
    for d in days:
        future_idx = entry_idx + d
        if future_idx < len(df):
            future_close = df.iloc[future_idx]["close"]
            returns[f"ret_{d}d"] = round((future_close / entry_close - 1) * 100, 2)
        else:
            returns[f"ret_{d}d"] = None
    return returns


# ══════════════════════════════════════════════════════════
# 분석 1: MACD 0선 크로스오버 전략
# ══════════════════════════════════════════════════════════

def analyze_macd_zero_cross(data: dict[str, pd.DataFrame], names: dict) -> dict:
    """
    MACD가 0선 근처(-1%~+1% 범위)에서 시그널 상향 돌파하는 시점 분석.

    조건:
    - MACD 값이 0 근처 (close 대비 ±1% 이내)
    - 전일: MACD < Signal (또는 MACD_hist < 0)
    - 당일: MACD >= Signal (또는 MACD_hist >= 0)  → 골든크로스
    - 추가: MACD가 상승 중 (macd > macd_prev)
    """
    print("\n" + "=" * 70)
    print("  분석 1: MACD 0선 크로스오버 전략")
    print("=" * 70)

    all_trades = []

    for ticker, df in data.items():
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            continue
        if "macd_histogram" not in df.columns or "macd_histogram_prev" not in df.columns:
            continue

        # 최근 3년만
        cutoff = df.index.max() - pd.Timedelta(days=365 * 3)
        df_cut = df[df.index >= cutoff].copy()
        if len(df_cut) < 60:
            continue

        for i in range(1, len(df_cut)):
            row = df_cut.iloc[i]
            prev = df_cut.iloc[i - 1]

            macd = row.get("macd", None)
            macd_sig = row.get("macd_signal", None)
            hist = row.get("macd_histogram", None)
            hist_prev = row.get("macd_histogram_prev", None)
            close = row.get("close", None)

            if any(pd.isna(v) for v in [macd, macd_sig, hist, hist_prev, close]):
                continue
            if close <= 0:
                continue

            # MACD 0선 근처: |MACD| < close * 0.01 (1% 범위)
            macd_near_zero = abs(macd) < close * 0.01

            # 히스토그램 양전환: 전일 음 → 당일 양 (골든크로스)
            golden_cross = hist_prev < 0 and hist >= 0

            # MACD 상승 중
            macd_rising = macd > prev.get("macd", macd)

            if macd_near_zero and golden_cross and macd_rising:
                rets = calc_forward_returns(df_cut, i)

                # 추가 컨텍스트
                rsi = row.get("rsi_14", 50)
                vol_surge = row.get("volume_surge_ratio", 1.0)
                foreign = row.get("foreign_net_5d", 0)

                all_trades.append({
                    "ticker": ticker,
                    "name": names.get(ticker, ticker),
                    "date": str(df_cut.index[i].date()),
                    "close": close,
                    "macd": round(macd, 2),
                    "rsi": round(rsi, 1) if not pd.isna(rsi) else 50,
                    "vol_surge": round(vol_surge, 2) if not pd.isna(vol_surge) else 1.0,
                    "foreign_5d": round(foreign, 0) if not pd.isna(foreign) else 0,
                    **rets,
                })

    # 통계
    if not all_trades:
        print("  시그널 없음")
        return {"signal_count": 0}

    df_trades = pd.DataFrame(all_trades)
    total = len(df_trades)

    print(f"\n  총 시그널: {total}건 (84종목, 3년)")

    result = {"signal_count": total, "by_period": {}}

    for period in ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"]:
        valid = df_trades[period].dropna()
        if len(valid) == 0:
            continue

        win_rate = (valid > 0).mean() * 100
        avg_ret = valid.mean()
        med_ret = valid.median()
        max_gain = valid.max()
        max_loss = valid.min()

        days = period.split("_")[1]
        print(f"\n  ── {days} 후 ──")
        print(f"     승률: {win_rate:.1f}%")
        print(f"     평균 수익률: {avg_ret:+.2f}%")
        print(f"     중앙값: {med_ret:+.2f}%")
        print(f"     최대 이익: {max_gain:+.2f}%  |  최대 손실: {max_loss:+.2f}%")

        result["by_period"][period] = {
            "count": len(valid),
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_ret, 2),
            "median_return": round(med_ret, 2),
            "max_gain": round(max_gain, 2),
            "max_loss": round(max_loss, 2),
        }

    # RSI 구간별 세분화
    print(f"\n  ── RSI 구간별 10일 수익률 ──")
    for rsi_lo, rsi_hi, label in [(0, 40, "과매도(<40)"), (40, 55, "중립저(40~55)"), (55, 70, "중립고(55~70)"), (70, 100, "과매수(>70)")]:
        subset = df_trades[(df_trades["rsi"] >= rsi_lo) & (df_trades["rsi"] < rsi_hi)]
        valid = subset["ret_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {label}: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    # 수급 동반 시
    print(f"\n  ── 외국인 순매수 동반 시 ──")
    foreign_buy = df_trades[df_trades["foreign_5d"] > 0]
    foreign_sell = df_trades[df_trades["foreign_5d"] <= 0]
    for label, subset in [("외인매수+MACD크로스", foreign_buy), ("외인매도+MACD크로스", foreign_sell)]:
        valid = subset["ret_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {label}: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    # 거래량 동반 시
    print(f"\n  ── 거래량 폭발 동반 시 (surge > 2.0) ──")
    vol_high = df_trades[df_trades["vol_surge"] >= 2.0]
    vol_normal = df_trades[df_trades["vol_surge"] < 2.0]
    for label, subset in [("거래량 폭발+MACD", vol_high), ("거래량 평범+MACD", vol_normal)]:
        valid = subset["ret_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {label}: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    return result


# ══════════════════════════════════════════════════════════
# 분석 2: 수급 폭발 → 조정 → 급등 패턴
# ══════════════════════════════════════════════════════════

def analyze_volume_spike_pullback_surge(data: dict[str, pd.DataFrame], names: dict) -> dict:
    """
    패턴: 거래량 폭발(3σ 이상) → 5~15일 조정(-5%~-15%) → 급등 발생 여부

    질문: 수급이 터진 후 조정이 오면 다시 올라가는 게 '통상적'인가?
    """
    print("\n" + "=" * 70)
    print("  분석 2: 수급 폭발 → 조정 → 급등 패턴")
    print("=" * 70)

    all_patterns = []

    for ticker, df in data.items():
        cutoff = df.index.max() - pd.Timedelta(days=365 * 3)
        df_cut = df[df.index >= cutoff].copy()
        if len(df_cut) < 60:
            continue

        vol_surge = df_cut.get("volume_surge_ratio")
        vol_z = df_cut.get("vol_z")
        if vol_surge is None or vol_z is None:
            continue

        # Step 1: 거래량 폭발 감지 (vol_z >= 3.0 또는 surge >= 3.0)
        for i in range(len(df_cut) - 25):
            row = df_cut.iloc[i]
            vz = row.get("vol_z", 0)
            vs = row.get("volume_surge_ratio", 1)

            if pd.isna(vz) or pd.isna(vs):
                continue

            is_spike = vz >= 3.0 or vs >= 3.0
            if not is_spike:
                continue

            spike_close = row["close"]
            spike_date = df_cut.index[i]
            spike_ret = row.get("ret1", 0)
            spike_direction = "상승" if spike_ret > 0 else "하락"

            # Step 2: 이후 5~15일 내 조정 확인 (고점 대비 -5% 이상 하락)
            window = df_cut.iloc[i + 1: i + 16]
            if len(window) < 5:
                continue

            peak_in_window = window["close"].max()
            trough_in_window = window["close"].min()
            trough_idx = window["close"].idxmin()

            # 조정 깊이 (스파이크 종가 대비)
            drawdown = (trough_in_window / spike_close - 1) * 100

            has_pullback = drawdown < -3  # 3% 이상 조정

            if not has_pullback:
                continue

            # Step 3: 조정 후 10/20일 반등 확인
            trough_position = df_cut.index.get_loc(trough_idx)
            if isinstance(trough_position, slice):
                trough_position = trough_position.start

            rebounds = {}
            for d in [5, 10, 20]:
                future = trough_position + d
                if future < len(df_cut):
                    future_close = df_cut.iloc[future]["close"]
                    rebounds[f"rebound_{d}d"] = round((future_close / trough_in_window - 1) * 100, 2)
                else:
                    rebounds[f"rebound_{d}d"] = None

            # 스파이크 대비 최종 수익률
            for d in [10, 20]:
                future = i + d
                if future < len(df_cut):
                    future_close = df_cut.iloc[future]["close"]
                    rebounds[f"from_spike_{d}d"] = round((future_close / spike_close - 1) * 100, 2)
                else:
                    rebounds[f"from_spike_{d}d"] = None

            # 외인/기관 수급
            foreign = row.get("foreign_net_5d", 0)

            all_patterns.append({
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "spike_date": str(spike_date.date()),
                "spike_close": spike_close,
                "spike_direction": spike_direction,
                "spike_vol_z": round(vz, 1),
                "spike_surge": round(vs, 1),
                "drawdown_pct": round(drawdown, 1),
                "foreign_5d": round(foreign, 0) if not pd.isna(foreign) else 0,
                **rebounds,
            })

    if not all_patterns:
        print("  패턴 없음")
        return {"pattern_count": 0}

    df_p = pd.DataFrame(all_patterns)
    total = len(df_p)

    print(f"\n  총 패턴: {total}건 (거래량 3σ 폭발 + 3%+ 조정)")

    result = {"pattern_count": total, "stats": {}}

    # 조정 후 반등 통계
    print(f"\n  ── 조정 저점 기준 반등률 ──")
    for col in ["rebound_5d", "rebound_10d", "rebound_20d"]:
        valid = df_p[col].dropna()
        if len(valid) < 5:
            continue
        days = col.split("_")[1]
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        med = valid.median()
        print(f"     {days} 후: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%, 중앙값 {med:+.2f}%")
        result["stats"][col] = {"count": len(valid), "win_rate": round(wr, 1), "avg": round(avg, 2), "median": round(med, 2)}

    # 스파이크 기준 수익률 (진입 시점이 스파이크 당일인 경우)
    print(f"\n  ── 스파이크 당일 진입 기준 ──")
    for col in ["from_spike_10d", "from_spike_20d"]:
        valid = df_p[col].dropna()
        if len(valid) < 5:
            continue
        days = col.split("_")[1]
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        med = valid.median()
        print(f"     {days} 후: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%, 중앙값 {med:+.2f}%")

    # 핵심: 스파이크 방향별 차이
    print(f"\n  ── 스파이크 방향별 (상승폭발 vs 하락폭발) ──")
    for direction in ["상승", "하락"]:
        subset = df_p[df_p["spike_direction"] == direction]
        valid = subset["rebound_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {direction} 스파이크 후 조정→10일 반등: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    # 외인 수급 동반 여부
    print(f"\n  ── 외인 순매수 동반 여부 ──")
    for label, cond in [("외인매수 동반", df_p["foreign_5d"] > 0), ("외인매도 동반", df_p["foreign_5d"] <= 0)]:
        subset = df_p[cond]
        valid = subset["rebound_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {label}: {len(valid)}건, 10일 반등 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    # 조정 깊이별 반등
    print(f"\n  ── 조정 깊이별 반등 (10일) ──")
    for lo, hi, label in [(-5, -3, "얕은 조정(-3~-5%)"), (-10, -5, "보통 조정(-5~-10%)"), (-30, -10, "깊은 조정(-10%+)")]:
        subset = df_p[(df_p["drawdown_pct"] >= lo) & (df_p["drawdown_pct"] < hi)]
        valid = subset["rebound_10d"].dropna()
        if len(valid) < 5:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        print(f"     {label}: {len(valid)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    return result


# ══════════════════════════════════════════════════════════
# 분석 3: US 이벤트 → KR 섹터 연동 (SOXX → 반도체)
# ══════════════════════════════════════════════════════════

def analyze_us_kr_sector_linkage(data: dict[str, pd.DataFrame], names: dict) -> dict:
    """
    US 반도체(SOXX) 급등 후 KR 반도체 종목 수익률.
    SOXX가 2% 이상 급등한 날 → 다음날/3일/5일 KR 반도체 종목 수익률.

    엔비디아 실적 같은 이벤트를 '사전 포지셔닝' 가능한지 검증.
    """
    print("\n" + "=" * 70)
    print("  분석 3: US 이벤트 → KR 섹터 연동 (SOXX → 반도체)")
    print("=" * 70)

    # 반도체 관련 종목 (유니버스 내)
    semi_keywords = ["반도체", "SK하이닉스", "삼성전자", "한미반도체", "리노공업",
                     "이오테크닉스", "테크윙", "주성엔지니어링", "ISC", "디아이",
                     "하나마이크론", "원익IPS", "HPSP", "솔브레인", "동진쎄미켐",
                     "티씨케이", "에스에프에이", "한솔케미칼", "실리콘투"]

    semi_tickers = []
    for ticker, name in names.items():
        for kw in semi_keywords:
            if kw in name:
                semi_tickers.append(ticker)
                break

    # 추가: 섹터 분류 파일이 있으면 활용
    sector_file = DATA_DIR / "dart_cache" / "fundamentals_all.csv"
    if sector_file.exists():
        fund_df = pd.read_csv(sector_file)
        fund_df["ticker"] = fund_df["ticker"].astype(str).str.zfill(6)
        semi_from_sector = fund_df[
            fund_df["sector_name"].str.contains("반도체", na=False)
        ]["ticker"].unique().tolist()
        semi_tickers = list(set(semi_tickers + semi_from_sector))

    semi_tickers = [t for t in semi_tickers if t in data]

    print(f"  반도체 유니버스: {len(semi_tickers)}종목")
    if not semi_tickers:
        print("  반도체 종목 없음")
        return {"semi_count": 0}

    # SOXX 데이터 추출 (parquet에 soxx_close 컬럼)
    sample_df = next(iter(data.values()))
    if "soxx_close" not in sample_df.columns:
        print("  soxx_close 컬럼 없음")
        return {"semi_count": 0, "error": "no soxx_close"}

    # 아무 종목에서든 soxx_close 추출
    soxx = sample_df[["soxx_close"]].dropna().copy()
    soxx["soxx_ret"] = soxx["soxx_close"].pct_change() * 100

    cutoff = soxx.index.max() - pd.Timedelta(days=365 * 3)
    soxx = soxx[soxx.index >= cutoff]

    # SOXX 급등일 (2% 이상)
    soxx_surge = soxx[soxx["soxx_ret"] >= 2.0]
    # SOXX 급락일 (-2% 이하)
    soxx_crash = soxx[soxx["soxx_ret"] <= -2.0]

    print(f"  SOXX 2%+ 급등일: {len(soxx_surge)}일")
    print(f"  SOXX 2%- 급락일: {len(soxx_crash)}일")

    result = {"semi_count": len(semi_tickers), "soxx_surge_days": len(soxx_surge)}

    def calc_sector_returns(event_dates, label):
        all_rets = {f"ret_{d}d": [] for d in [1, 3, 5, 10]}

        for event_date in event_dates:
            for ticker in semi_tickers:
                df = data[ticker]
                if event_date not in df.index:
                    continue
                idx = df.index.get_loc(event_date)
                if isinstance(idx, slice):
                    idx = idx.start

                for d in [1, 3, 5, 10]:
                    future = idx + d
                    if future < len(df):
                        entry = df.iloc[idx]["close"]
                        exit_p = df.iloc[future]["close"]
                        if entry > 0:
                            ret = (exit_p / entry - 1) * 100
                            all_rets[f"ret_{d}d"].append(ret)

        print(f"\n  ── {label} ──")
        for period, rets in all_rets.items():
            if len(rets) < 10:
                continue
            arr = np.array(rets)
            days = period.split("_")[1]
            wr = (arr > 0).mean() * 100
            avg = arr.mean()
            med = np.median(arr)
            print(f"     {days} 후: {len(arr)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%, 중앙값 {med:+.2f}%")
            result[f"{label}_{period}"] = {"count": len(arr), "win_rate": round(wr, 1), "avg": round(avg, 2)}

    calc_sector_returns(soxx_surge.index, "SOXX급등_후_KR반도체")
    calc_sector_returns(soxx_crash.index, "SOXX급락_후_KR반도체")

    # 핵심: SOXX 3일 연속 상승 후
    print(f"\n  ── SOXX 3일 연속 상승 후 (사전 포지셔닝 효과) ──")
    soxx["up_streak"] = (soxx["soxx_ret"] > 0).rolling(3).sum()
    streak_dates = soxx[soxx["up_streak"] >= 3].index
    print(f"     3일 연속 상승 발생: {len(streak_dates)}회")

    all_rets_streak = {f"ret_{d}d": [] for d in [1, 3, 5]}
    for event_date in streak_dates:
        for ticker in semi_tickers:
            df = data[ticker]
            if event_date not in df.index:
                continue
            idx = df.index.get_loc(event_date)
            if isinstance(idx, slice):
                idx = idx.start
            # 3일 전 진입 (사전 포지셔닝)
            entry_idx = max(0, idx - 3)
            entry_close = df.iloc[entry_idx]["close"]
            exit_close = df.iloc[idx]["close"]
            if entry_close > 0:
                ret = (exit_close / entry_close - 1) * 100
                all_rets_streak["ret_3d"].append(ret)
            # 이벤트 당일부터 5일
            future = idx + 5
            if future < len(df) and entry_close > 0:
                ret = (df.iloc[future]["close"] / entry_close - 1) * 100
                all_rets_streak["ret_5d"].append(ret)

    for period, rets in all_rets_streak.items():
        if len(rets) < 10:
            continue
        arr = np.array(rets)
        wr = (arr > 0).mean() * 100
        avg = arr.mean()
        if period == "ret_3d":
            print(f"     3일 전 사전진입 → 이벤트 당일 수익: {len(arr)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")
        else:
            print(f"     3일 전 사전진입 → 이벤트 후 5일 수익: {len(arr)}건, 승률 {wr:.1f}%, 평균 {avg:+.2f}%")

    return result


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  진입 패턴 실증 분석 — 84종목 × 3년")
    print("  실행: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    names = load_universe()
    print(f"\n유니버스: {len(names)}종목")

    print("parquet 로딩 중...")
    data = load_all_parquets()
    print(f"로딩 완료: {len(data)}종목")

    # ── 분석 1: MACD 0선 크로스오버 ──
    r1 = analyze_macd_zero_cross(data, names)

    # ── 분석 2: 수급 폭발 → 조정 → 급등 ──
    r2 = analyze_volume_spike_pullback_surge(data, names)

    # ── 분석 3: US → KR 섹터 연동 ──
    r3 = analyze_us_kr_sector_linkage(data, names)

    # ── 종합 ──
    print("\n" + "=" * 70)
    print("  종합 결론")
    print("=" * 70)

    # MACD 판정
    macd_10d = r1.get("by_period", {}).get("ret_10d", {})
    macd_wr = macd_10d.get("win_rate", 0)
    macd_avg = macd_10d.get("avg_return", 0)

    print(f"\n  [MACD 0선 크로스] 10일 승률 {macd_wr}%, 평균 {macd_avg:+.2f}%")
    if macd_wr >= 55 and macd_avg > 1:
        print("    → ✅ 유효한 전략. 스캐너 구현 권장.")
    elif macd_wr >= 50:
        print("    → ⚠️ 보통. 추가 필터(수급, RSI) 결합 시 개선 가능.")
    else:
        print("    → ❌ 단독으로는 부족. 보조 확인 용도만.")

    # 수급 폭발 패턴 판정
    rebound = r2.get("stats", {}).get("rebound_10d", {})
    rb_wr = rebound.get("win_rate", 0)
    rb_avg = rebound.get("avg", 0)

    print(f"\n  [수급폭발→조정→반등] 10일 반등 승률 {rb_wr}%, 평균 {rb_avg:+.2f}%")
    if rb_wr >= 60:
        print("    → ✅ 통상적 패턴! 조정 시 추가 매수 유효.")
    elif rb_wr >= 50:
        print("    → ⚠️ 반반. 외인 수급/조정 깊이에 따라 분기.")
    else:
        print("    → ❌ V자 반등은 소수. 조정=추세 전환 가능성 높음.")

    # US→KR 연동 판정
    print(f"\n  [US→KR 섹터 연동]")
    if "SOXX급등_후_KR반도체_ret_3d" in r3:
        s = r3["SOXX급등_후_KR반도체_ret_3d"]
        print(f"    SOXX 급등 → KR반도체 3일: 승률 {s['win_rate']}%, 평균 {s['avg']:+.2f}%")

    # JSON 저장
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "macd_zero_cross": r1,
        "volume_spike_pattern": r2,
        "us_kr_linkage": r3,
    }

    out_path = DATA_DIR / "entry_pattern_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
