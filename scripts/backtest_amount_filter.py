"""
거래대금 필터 백테스트 — 3개월

수급 폭발 스캐너의 거래대금 AND 필터 효과를 검증.
비교: vol_z/vsr만 (기존) vs vol_z/vsr + 거래대금 3x (신규)

스파이크 감지 → 이후 10일 내 최대 수익률 vs 최대 손실률 비교.

Usage:
    python scripts/backtest_amount_filter.py
    python scripts/backtest_amount_filter.py --months 6
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_DIR = PROJECT_ROOT / "stock_data_daily"

# ── 파라미터 ──
SPIKE_VOL_Z = 3.0
SPIKE_VSR = 3.0
AMOUNT_RATIO_THRESHOLD = 3.0
FORWARD_DAYS = 10          # 스파이크 후 관찰 기간
PULLBACK_ENTRY = -0.03     # 스파이크 고가 대비 -3% 진입
STOP_LOSS = -0.15           # 손절
TARGET_PROFIT = 0.10        # 익절


def build_name_map() -> dict[str, str]:
    name_map = {}
    for csv in CSV_DIR.glob("*.csv"):
        parts = csv.stem.rsplit("_", 1)
        if len(parts) == 2:
            name_map[parts[1]] = parts[0]
    return name_map


def run_backtest(months: int = 3):
    """3개월 백테스트 실행."""
    name_map = build_name_map()

    # 날짜 범위 계산
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(months=months)

    print(f"[백테스트] 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({months}개월)")
    print(f"[파라미터] vol_z>={SPIKE_VOL_Z} or vsr>={SPIKE_VSR}, 거래대금>={AMOUNT_RATIO_THRESHOLD}x")
    print(f"[관찰] 스파이크 후 {FORWARD_DAYS}일 성과, 진입: 고가 대비 -{abs(PULLBACK_ENTRY)*100:.0f}%")

    # 결과 수집
    results_old = []   # 기존 (거래량만)
    results_new = []   # 신규 (거래량 + 거래대금)

    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    print(f"\n[스캔] {len(parquets)}종목 처리 중...")

    for pq in parquets:
        ticker = pq.stem
        try:
            df = pd.read_parquet(pq)
            if len(df) < 60:
                continue

            # 날짜 필터
            df = df.loc[df.index >= start_date]
            if len(df) < 21:
                continue

            # 거래대금 계산 (close * volume)
            df["_amount"] = df["close"] * df["volume"]
            df["_amount_ma20"] = df["_amount"].rolling(20).mean()
            df["_amount_ratio"] = np.where(
                df["_amount_ma20"] > 0,
                df["_amount"] / df["_amount_ma20"],
                0.0,
            )

            name = name_map.get(ticker, ticker)

            for i in range(20, len(df) - FORWARD_DAYS):
                row = df.iloc[i]
                vol_z = float(row.get("vol_z", 0) or 0)
                vsr = float(row.get("volume_surge_ratio", 0) or 0)

                if pd.isna(vol_z):
                    vol_z = 0
                if pd.isna(vsr):
                    vsr = 0

                # 기존 조건: vol_z OR vsr
                vol_pass = vol_z >= SPIKE_VOL_Z or vsr >= SPIKE_VSR
                if not vol_pass:
                    continue

                spike_date = df.index[i]
                spike_high = float(row["high"])
                spike_close = float(row["close"])
                amount_ratio = float(df["_amount_ratio"].iloc[i])

                # 이후 10일 성과 계산
                forward = df.iloc[i + 1: i + 1 + FORWARD_DAYS]
                if len(forward) == 0:
                    continue

                # 조정 진입가: 스파이크 고가 * (1 + PULLBACK_ENTRY)
                entry_price = spike_high * (1 + PULLBACK_ENTRY)

                # 진입 가능 여부: 이후 10일 내 저가가 entry_price 이하
                entered = False
                entry_day = -1
                for j, (fdate, frow) in enumerate(forward.iterrows()):
                    if float(frow["low"]) <= entry_price:
                        entered = True
                        entry_day = j
                        break

                if not entered:
                    # 진입 안 됨 → 결과에 포함하되 entry=False
                    record = {
                        "ticker": ticker,
                        "name": name,
                        "spike_date": str(spike_date.date()),
                        "spike_close": spike_close,
                        "spike_high": spike_high,
                        "vol_z": round(vol_z, 2),
                        "vsr": round(vsr, 2),
                        "amount_ratio": round(amount_ratio, 2),
                        "entered": False,
                        "max_gain_pct": 0,
                        "max_loss_pct": 0,
                        "day10_pct": 0,
                        "win": False,
                    }
                    results_old.append(record)
                    if amount_ratio >= AMOUNT_RATIO_THRESHOLD:
                        results_new.append(record)
                    continue

                # 진입 후 성과 (entry_price 기준)
                post_entry = forward.iloc[entry_day:]
                max_high = float(post_entry["high"].max())
                min_low = float(post_entry["low"].min())
                close_last = float(post_entry.iloc[-1]["close"])

                max_gain = (max_high / entry_price - 1) * 100
                max_loss = (min_low / entry_price - 1) * 100
                day10_pct = (close_last / entry_price - 1) * 100

                record = {
                    "ticker": ticker,
                    "name": name,
                    "spike_date": str(spike_date.date()),
                    "spike_close": spike_close,
                    "spike_high": spike_high,
                    "vol_z": round(vol_z, 2),
                    "vsr": round(vsr, 2),
                    "amount_ratio": round(amount_ratio, 2),
                    "entered": True,
                    "max_gain_pct": round(max_gain, 2),
                    "max_loss_pct": round(max_loss, 2),
                    "day10_pct": round(day10_pct, 2),
                    "win": day10_pct > 0,
                }

                results_old.append(record)
                if amount_ratio >= AMOUNT_RATIO_THRESHOLD:
                    results_new.append(record)

        except Exception as e:
            pass  # 개별 종목 에러 무시

    # ── 결과 분석 ──
    print_comparison(results_old, results_new)


def print_comparison(old: list[dict], new: list[dict]):
    """기존 vs 신규 비교 출력."""

    def stats(records: list[dict], label: str):
        df = pd.DataFrame(records)
        total = len(df)
        if total == 0:
            print(f"\n[{label}] 시그널 0건")
            return {}

        entered = df[df["entered"]]
        not_entered = df[~df["entered"]]
        n_entered = len(entered)

        if n_entered == 0:
            print(f"\n[{label}] 스파이크 {total}건, 진입 0건")
            return {}

        wins = entered[entered["win"]]
        win_rate = len(wins) / n_entered * 100

        avg_gain = entered["max_gain_pct"].mean()
        avg_loss = entered["max_loss_pct"].mean()
        avg_day10 = entered["day10_pct"].mean()
        median_day10 = entered["day10_pct"].median()

        # 손익비 (avg gain / abs(avg loss))
        pf = avg_gain / abs(avg_loss) if avg_loss != 0 else float("inf")

        print(f"\n{'═' * 60}")
        print(f"  [{label}]")
        print(f"{'═' * 60}")
        print(f"  스파이크 총 감지:      {total}건")
        print(f"  조정 진입(-3%):        {n_entered}건 ({n_entered/total*100:.0f}%)")
        print(f"  미진입(반등 강세):     {len(not_entered)}건")
        print(f"  ──────────────────────────────────")
        print(f"  승률:                  {win_rate:.1f}%  ({len(wins)}/{n_entered})")
        print(f"  평균 최대수익:         {avg_gain:+.2f}%")
        print(f"  평균 최대손실:         {avg_loss:+.2f}%")
        print(f"  손익비 (PF):           {pf:.2f}")
        print(f"  평균 10일 수익:        {avg_day10:+.2f}%")
        print(f"  중앙값 10일 수익:      {median_day10:+.2f}%")

        # 거래대금 분포
        if "amount_ratio" in df.columns:
            ar = df["amount_ratio"]
            print(f"  ──────────────────────────────────")
            print(f"  거래대금비율 평균:     {ar.mean():.1f}x")
            print(f"  거래대금비율 중앙값:   {ar.median():.1f}x")

        return {
            "total": total,
            "entered": n_entered,
            "win_rate": win_rate,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "pf": pf,
            "avg_day10": avg_day10,
        }

    s_old = stats(old, "기존 (거래량만)")
    s_new = stats(new, "신규 (거래량 + 거래대금 3x)")

    if s_old and s_new:
        print(f"\n{'═' * 60}")
        print(f"  [비교 요약]")
        print(f"{'═' * 60}")
        removed = s_old["total"] - s_new["total"]
        print(f"  필터링 제거:    {removed}건 ({removed/s_old['total']*100:.0f}% 감소)")
        wr_diff = s_new["win_rate"] - s_old["win_rate"]
        print(f"  승률 변화:      {s_old['win_rate']:.1f}% → {s_new['win_rate']:.1f}% ({wr_diff:+.1f}%p)")
        pf_diff = s_new["pf"] - s_old["pf"]
        print(f"  PF 변화:        {s_old['pf']:.2f} → {s_new['pf']:.2f} ({pf_diff:+.2f})")
        d10_diff = s_new["avg_day10"] - s_old["avg_day10"]
        print(f"  10일수익 변화:  {s_old['avg_day10']:+.2f}% → {s_new['avg_day10']:+.2f}% ({d10_diff:+.2f}%p)")
        print(f"{'═' * 60}")

        if s_new["win_rate"] > s_old["win_rate"] and s_new["pf"] > s_old["pf"]:
            print(f"  ✅ 거래대금 필터 효과 확인 — 승률+PF 모두 개선")
        elif s_new["win_rate"] > s_old["win_rate"]:
            print(f"  ⚠️ 승률은 개선, PF는 하락 — 추가 검토 필요")
        elif s_new["pf"] > s_old["pf"]:
            print(f"  ⚠️ PF는 개선, 승률은 하락 — 추가 검토 필요")
        else:
            print(f"  ❌ 거래대금 필터 효과 미미 — 기존 유지 검토")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3, help="백테스트 기간 (개월)")
    args = parser.parse_args()
    run_backtest(args.months)
