"""디레버리징 종료 시그널 분석 — 숫자로 증명

5가지 분석:
1) KOSPI 과거 폭락 후 회복 패턴
2) 외국인/기관 매도 강도 추이
3) RSI 과매도 비율 추이
4) 거래량 클라이맥스 감지
5) S등급 종목 개별 바닥 신호
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def analyze_crash_recovery():
    """분석 1: KOSPI 과거 폭락 후 회복 패턴"""
    print("=== 분석 1: KOSPI 과거 폭락 후 회복 패턴 ===")
    print()

    kospi = pd.read_csv(
        PROJECT_ROOT / "data" / "kospi_index.csv",
        index_col=0, parse_dates=True,
    ).sort_index()
    kospi["ret"] = kospi["close"].pct_change() * 100

    crash_days = kospi[kospi["ret"] <= -5]
    print(f"KOSPI -5% 이상 폭락일 (3년): {len(crash_days)}일")
    print()

    for idx, row in crash_days.iterrows():
        date = idx.strftime("%Y-%m-%d")
        ret = row["ret"]
        close_val = row["close"]

        future = kospi.loc[idx:]
        if len(future) < 6:
            continue

        # 20일 내 추가 하락
        f20 = future.iloc[: min(21, len(future))]
        min_close = f20["close"].min()
        days_to_min = (f20["close"].idxmin() - idx).days
        add_drop = (min_close - close_val) / close_val * 100

        # N일 후 수익률
        rets = {}
        for d in [1, 3, 5, 10, 20]:
            if len(future) > d:
                rets[d] = (future["close"].iloc[d] - close_val) / close_val * 100
            else:
                rets[d] = None

        print(f"{date} | 당일 {ret:+.1f}% | 종가 {close_val:,.0f}")
        print(f"  추가하락: {add_drop:+.1f}% ({days_to_min}일 후 저점)")
        parts = []
        for d in [1, 3, 5, 10, 20]:
            v = rets[d]
            parts.append(f"{d}일:{v:+.1f}%" if v is not None else f"{d}일:N/A")
        print(f"  {' | '.join(parts)}")
        print()


def analyze_foreign_flow():
    """분석 2: 외국인/기관 매도 강도 추이"""
    print("=== 분석 2: 외국인/기관 매도 강도 추이 ===")
    print()

    # 거래대금 200억+ 종목만
    large_files = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            avg_val = (df["close"] * df["volume"]).tail(20).mean() / 1e8
            if avg_val >= 200:
                large_files.append(f)
        except Exception:
            continue

    print(f"분석 대상: 거래대금 200억+ {len(large_files)}종목")

    foreign_daily: dict[str, float] = {}
    inst_daily: dict[str, float] = {}

    for f in large_files:
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            recent = df.loc["2026-02-17":"2026-03-06"]

            for date, row in recent.iterrows():
                ds = date.strftime("%Y-%m-%d")
                if "외국인합계" in df.columns:
                    v = float(row.get("외국인합계", 0) or 0)
                    foreign_daily[ds] = foreign_daily.get(ds, 0) + v
                if "기관합계" in df.columns:
                    v = float(row.get("기관합계", 0) or 0)
                    inst_daily[ds] = inst_daily.get(ds, 0) + v
        except Exception:
            continue

    print()
    print(f"{'날짜':<12s} | {'외국인(억)':>12s} | {'기관(억)':>12s} | {'합계(억)':>12s} | 시그널")
    print("-" * 75)

    prev_f = None
    for date in sorted(foreign_daily.keys()):
        f_val = foreign_daily[date] / 1e8
        i_val = inst_daily.get(date, 0) / 1e8
        total = f_val + i_val

        # 디레버리징 종료 시그널: 외국인 매도 감소 + 기관 매수 증가
        signal = ""
        if prev_f is not None and f_val < 0 and prev_f < 0:
            if abs(f_val) < abs(prev_f) * 0.7:
                signal = "← 매도 둔화!"
        if f_val > 0:
            signal = "← 외국인 순매수 전환!"
        if i_val > 0 and f_val < 0:
            signal += " 기관 흡수"

        print(f"{date} | {f_val:>+12,.0f} | {i_val:>+12,.0f} | {total:>+12,.0f} | {signal}")
        prev_f = f_val

    # 매도 강도 추이 (3일 이동평균)
    print()
    dates = sorted(foreign_daily.keys())
    if len(dates) >= 5:
        last5 = dates[-5:]
        avg_last5 = np.mean([foreign_daily[d] / 1e8 for d in last5])
        first5 = dates[:5]
        avg_first5 = np.mean([foreign_daily[d] / 1e8 for d in first5])
        print(f"외국인 매도 추이: 초반5일 평균 {avg_first5:+,.0f}억 → 최근5일 평균 {avg_last5:+,.0f}억")
        if abs(avg_last5) < abs(avg_first5) * 0.5:
            print(">>> 매도 강도 50% 이상 감소 — 디레버리징 후반부!")
        elif avg_last5 > 0:
            print(">>> 외국인 순매수 전환! — 디레버리징 종료 시그널!")


def analyze_rsi_distribution():
    """분석 3: RSI 과매도 비율 추이"""
    print()
    print("=== 분석 3: RSI 분포 — 과매도 비율 추이 ===")
    print()

    dates_check = ["2026-02-27", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]
    rsi_dist = {d: {"under30": 0, "under40": 0, "under50": 0, "total": 0} for d in dates_check}

    for f in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["close", "rsi_14"])
            df.index = pd.to_datetime(df.index)

            for d in dates_check:
                if d in df.index.strftime("%Y-%m-%d").values:
                    row = df.loc[d]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    rsi = float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else None
                    if rsi is not None:
                        rsi_dist[d]["total"] += 1
                        if rsi < 30:
                            rsi_dist[d]["under30"] += 1
                        if rsi < 40:
                            rsi_dist[d]["under40"] += 1
                        if rsi < 50:
                            rsi_dist[d]["under50"] += 1
        except Exception:
            continue

    print(f"{'날짜':<12s} | {'RSI<30':>12s} | {'RSI<40':>12s} | {'RSI<50':>12s} | 전체")
    print("-" * 70)
    for d in dates_check:
        r = rsi_dist[d]
        t = max(r["total"], 1)
        print(
            f"{d} | {r['under30']:>4d} ({r['under30']/t*100:4.1f}%) "
            f"| {r['under40']:>4d} ({r['under40']/t*100:4.1f}%) "
            f"| {r['under50']:>4d} ({r['under50']/t*100:4.1f}%) "
            f"| {r['total']}"
        )

    # 3/4 vs 3/6 비교
    if rsi_dist["2026-03-04"]["total"] > 0 and rsi_dist["2026-03-06"]["total"] > 0:
        t4 = rsi_dist["2026-03-04"]
        t6 = rsi_dist["2026-03-06"]
        print()
        pct4 = t4["under30"] / max(t4["total"], 1) * 100
        pct6 = t6["under30"] / max(t6["total"], 1) * 100
        print(f"RSI<30 비율: 3/4 {pct4:.1f}% → 3/6 {pct6:.1f}%")
        if pct6 < pct4:
            print(">>> 과매도 종목 감소 — 바닥 확인 중!")
        else:
            print(">>> 과매도 종목 증가 — 아직 바닥 아님!")


def analyze_volume_climax():
    """분석 4: 거래량 셀링 클라이맥스"""
    print()
    print("=== 분석 4: KOSPI 거래량 추이 (셀링 클라이맥스 감지) ===")
    print()

    kospi = pd.read_csv(
        PROJECT_ROOT / "data" / "kospi_index.csv",
        index_col=0, parse_dates=True,
    ).sort_index()
    kospi["ret"] = kospi["close"].pct_change() * 100
    vol_20avg = kospi["volume"].rolling(20).mean()

    vol_data = kospi.loc["2026-02-20":"2026-03-06"]

    for idx, row in vol_data.iterrows():
        date = idx.strftime("%Y-%m-%d")
        vol = row["volume"]
        avg = vol_20avg.loc[idx] if idx in vol_20avg.index else 0
        ratio = vol / avg if avg > 0 else 0
        ret = row["ret"] if pd.notna(row["ret"]) else 0

        bar = "#" * int(ratio * 5)
        tag = ""
        if ratio > 1.3 and ret < -3:
            tag = " <<< SELLING CLIMAX"
        elif ratio > 1.2 and ret > 3:
            tag = " <<< REVERSAL BUYING"
        elif vol < avg * 0.8 and abs(ret) < 1:
            tag = " (거래량 감소 = 안정화)"

        print(f"{date} | {row['close']:>8,.0f} | {ret:+5.1f}% | 거래량 {vol:>10,} | {ratio:.2f}x | {bar}{tag}")

    # 최근 거래량 추세
    recent = vol_data.tail(3)
    if len(recent) >= 3:
        vol_trend = recent["volume"].values
        if vol_trend[-1] < vol_trend[-2] < vol_trend[-3]:
            print()
            print(">>> 거래량 3일 연속 감소 — 패닉 매도 소진 중!")


def analyze_bottom_signals():
    """분석 5: S등급 종목 개별 바닥 신호"""
    print()
    print("=== 분석 5: S등급 종목 바닥 신호 체크 ===")
    print()

    s_tickers = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "005380": "현대차",
        "000270": "기아",
        "105560": "KB금융",
        "035420": "NAVER",
        "003230": "삼양식품",
    }

    summary = []
    for ticker, name in s_tickers.items():
        try:
            f = DATA_DIR / f"{ticker}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)

            latest = df.iloc[-1]
            close = float(latest["close"])
            rsi = float(latest["rsi_14"]) if pd.notna(latest.get("rsi_14")) else 50
            vol = float(latest["volume"])
            vol_avg = float(df["volume"].tail(20).mean())
            vol_ratio = vol / vol_avg if vol_avg > 0 else 0
            bb = float(latest["bb_position"]) if pd.notna(latest.get("bb_position")) else 0.5

            # 5일간 수급
            foreign_5d = 0
            if "외국인합계" in df.columns:
                foreign_5d = float(df["외국인합계"].tail(5).sum()) / 1e8
            inst_5d = 0
            if "기관합계" in df.columns:
                inst_5d = float(df["기관합계"].tail(5).sum()) / 1e8

            # 3/4(폭락일) 대비 반등
            bounce = 0
            crash_date = "2026-03-04"
            if crash_date in df.index.strftime("%Y-%m-%d").values:
                cd = df.loc[crash_date]
                if isinstance(cd, pd.DataFrame):
                    cd = cd.iloc[-1]
                crash_close = float(cd["close"])
                bounce = (close - crash_close) / crash_close * 100

            # 20일 저점 대비 위치
            low_20d = float(df["close"].tail(20).min())
            from_low = (close - low_20d) / low_20d * 100

            # 바닥 신호 카운트
            signals = []
            if rsi < 35:
                signals.append(f"RSI={rsi:.0f}(극과매도)")
            elif rsi < 45:
                signals.append(f"RSI={rsi:.0f}(과매도)")
            if bb < 0.15:
                signals.append(f"BB={bb:.0%}(하단돌파)")
            elif bb < 0.3:
                signals.append(f"BB={bb:.0%}(하단근접)")
            if vol_ratio > 2.0:
                signals.append(f"거래량{vol_ratio:.1f}x(폭발)")
            if inst_5d > 0:
                signals.append(f"기관5일+{inst_5d:,.0f}억")
            if foreign_5d > 0:
                signals.append(f"외인5일+{foreign_5d:,.0f}억")
            if bounce > 3:
                signals.append(f"3/4대비+{bounce:.1f}%반등")

            n = len(signals)
            if n >= 4:
                verdict = ">>> 매수 GO"
            elif n >= 3:
                verdict = ">> 1차 분할 OK"
            elif n >= 2:
                verdict = "> 관찰 (대기)"
            else:
                verdict = "- 아직"

            print(f"{name:<10s}({ticker}) {close:>10,}원")
            print(f"  RSI {rsi:.0f} | BB {bb:.0%} | Vol {vol_ratio:.1f}x | 3/4대비 {bounce:+.1f}%")
            print(f"  외인5일 {foreign_5d:+,.0f}억 | 기관5일 {inst_5d:+,.0f}억")
            print(f"  바닥 신호 {n}개: {verdict}")
            if signals:
                print(f"  [{' | '.join(signals)}]")
            print()

            summary.append({
                "name": name, "ticker": ticker, "signals": n,
                "verdict": verdict, "rsi": rsi, "bounce": bounce,
            })

        except Exception as e:
            print(f"{name}: 오류 {e}")
            print()

    # ─── 종합 판정 ───
    print("=" * 60)
    print("=== 종합 타이밍 판정 ===")
    print("=" * 60)
    print()

    ready = [s for s in summary if s["signals"] >= 3]
    wait = [s for s in summary if s["signals"] < 3]

    if len(ready) >= 4:
        print("판정: 1차 분할매수 시점 도달!")
        print(f"  바닥 신호 3개+ 종목: {len(ready)}/{len(summary)}")
    elif len(ready) >= 2:
        print("판정: 선별적 1차 분할 가능")
        print(f"  바닥 신호 3개+ 종목: {len(ready)}/{len(summary)}")
    else:
        print("판정: 아직 대기. 디레버리징 진행 중")
        print(f"  바닥 신호 3개+ 종목: {len(ready)}/{len(summary)}")

    print()
    for s in ready:
        print(f"  GO: {s['name']} (신호 {s['signals']}개, RSI {s['rsi']:.0f})")
    for s in wait:
        print(f"  WAIT: {s['name']} (신호 {s['signals']}개, RSI {s['rsi']:.0f})")


def main():
    analyze_crash_recovery()
    analyze_foreign_flow()
    analyze_rsi_distribution()
    analyze_volume_climax()
    analyze_bottom_signals()

    # 텔레그램 발송
    from src.telegram_sender import send_message

    msg = (
        "[타이밍 분석 완료]\n"
        "5가지 디레버리징 종료 시그널 분석\n"
        "상세 결과는 콘솔 출력 확인"
    )
    send_message(msg)


if __name__ == "__main__":
    main()
