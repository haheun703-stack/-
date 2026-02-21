"""
Shakeout Detector 백테스트 (Step 1)

목적: "횡보 후 급락 → V자 회복" 패턴이 실제로 유효한지 검증
대상: parquet 유니버스 전체 + 보유 3종목 (pykrx)
기간: 최근 1년 (약 240 거래일)

패턴 정의:
  1) 횡보 구간: 직전 20일 일간수익률 std < threshold
  2) 급락 발생: 1~3일 내 -3% 이상 하락
  3) 거래량 급증: 하락일 거래량이 20일 평균 대비 2배 이상
  4) MA120 위: 하락 후에도 MA120 위에서 유지

회복 판정:
  - 3일 / 5일 / 10일 내 급락 전 가격 대비 -1% 이내 복귀
"""

import pandas as pd
import numpy as np
import glob
import sys
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────
# 1. 패턴 스캔 함수
# ──────────────────────────────────────────
def scan_shakeout_patterns(
    df: pd.DataFrame,
    std_threshold: float = 0.015,
    drop_threshold: float = -0.03,
    vol_mult: float = 2.0,
    lookback: int = 20,
    recovery_windows: list = None,
) -> list[dict]:
    """
    단일 종목 DataFrame에서 shakeout 패턴을 스캔.

    Parameters:
      df: OHLCV + sma_120 + volume_ma20 + ret1 컬럼 필요
      std_threshold: 횡보 판별 기준 (일간수익률 std)
      drop_threshold: 급락 기준 (예: -0.03 = -3%)
      vol_mult: 거래량 배수 기준
      lookback: 횡보 판별 기간
      recovery_windows: 회복 확인 기간 리스트

    Returns:
      패턴 발생 리스트 [{date, drop_pct, vol_ratio, above_ma120, ...}, ...]
    """
    if recovery_windows is None:
        recovery_windows = [3, 5, 10]

    results = []

    # 필요 컬럼 확인
    required = ["close", "volume", "ret1"]
    for c in required:
        if c not in df.columns:
            return results

    # sma_120 없으면 직접 계산
    if "sma_120" not in df.columns:
        df["sma_120"] = df["close"].rolling(120).mean()

    # volume_ma20 없으면 직접 계산
    if "volume_ma20" not in df.columns:
        df["volume_ma20"] = df["volume"].rolling(20).mean()

    # 최근 1년만 사용 (마지막 250거래일)
    df = df.iloc[-250:].copy()

    if len(df) < lookback + 30:
        return results

    for i in range(lookback + 5, len(df) - max(recovery_windows)):
        row = df.iloc[i]

        # ── 조건 1: 횡보 구간 (직전 20일 std 낮음) ──
        recent_rets = df["ret1"].iloc[i - lookback : i]
        std_val = recent_rets.std()
        if std_val >= std_threshold:
            continue

        # ── 조건 2: 급락 발생 (당일 또는 1~3일 누적) ──
        # 당일 급락
        day_ret = row["ret1"]
        if pd.isna(day_ret):
            continue

        # 1~3일 누적 하락 체크
        cum_drops = {}
        cum = 0
        for d in range(0, 3):
            idx = i + d
            if idx >= len(df):
                break
            r = df["ret1"].iloc[idx]
            if pd.isna(r):
                break
            cum += r
            cum_drops[d + 1] = cum

        # 최대 누적 하락 찾기
        best_drop = None
        best_days = 0
        for days, drop in cum_drops.items():
            if drop <= drop_threshold:
                if best_drop is None or drop < best_drop:
                    best_drop = drop
                    best_days = days

        if best_drop is None:
            continue

        # ── 조건 3: 거래량 급증 ──
        vol_ratio = row["volume"] / row["volume_ma20"] if row["volume_ma20"] > 0 else 0
        if pd.isna(vol_ratio) or vol_ratio < vol_mult:
            continue

        # ── 조건 4: MA120 위 유지 ──
        # 급락 후 최저점이 MA120 위인지
        drop_end_idx = i + best_days - 1
        low_in_drop = df["close"].iloc[i : drop_end_idx + 1].min()
        ma120_at_drop = df["sma_120"].iloc[i]
        if pd.isna(ma120_at_drop):
            continue
        above_ma120 = low_in_drop > ma120_at_drop

        # ── 회복 판정 ──
        pre_drop_price = df["close"].iloc[i - 1]  # 급락 전날 종가
        recovery_info = {}
        for win in recovery_windows:
            end_idx = drop_end_idx + win
            if end_idx >= len(df):
                recovery_info[f"recover_{win}d"] = None
                recovery_info[f"recover_{win}d_ret"] = None
                continue
            # win일 내 최고가 기준으로 회복 판단
            prices_after = df["close"].iloc[drop_end_idx + 1 : end_idx + 1]
            if len(prices_after) == 0:
                recovery_info[f"recover_{win}d"] = None
                recovery_info[f"recover_{win}d_ret"] = None
                continue
            max_price_after = prices_after.max()
            recovery_ret = (max_price_after / pre_drop_price - 1)
            recovered = recovery_ret >= -0.01  # -1% 이내 복귀
            recovery_info[f"recover_{win}d"] = recovered
            recovery_info[f"recover_{win}d_ret"] = round(recovery_ret * 100, 2)

        # 10일 후 수익률 (급락 저점 대비)
        future_10d_idx = drop_end_idx + 10
        if future_10d_idx < len(df):
            future_ret = (df["close"].iloc[future_10d_idx] / low_in_drop - 1)
        else:
            future_ret = None

        results.append({
            "date": df.index[i].strftime("%Y-%m-%d"),
            "pre_price": int(pre_drop_price),
            "drop_pct": round(best_drop * 100, 2),
            "drop_days": best_days,
            "vol_ratio": round(vol_ratio, 1),
            "std_20d": round(std_val, 4),
            "above_ma120": above_ma120,
            "ma120_margin": round((low_in_drop / ma120_at_drop - 1) * 100, 2),
            **recovery_info,
            "future_10d_ret": round(future_ret * 100, 2) if future_ret else None,
        })

    return results


# ──────────────────────────────────────────
# 2. 전체 유니버스 스캔
# ──────────────────────────────────────────
def run_universe_backtest(std_thresholds=None):
    """84종목 유니버스 + 전체 parquet 스캔."""
    if std_thresholds is None:
        std_thresholds = [0.010, 0.015, 0.020, 0.025]

    parquet_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "processed" / "*.parquet")))
    print(f"parquet 파일: {len(parquet_files)}개")

    # 종목명 매핑 (CSV에서)
    name_map = {}
    csv_dir = PROJECT_ROOT / "stock_data_daily"
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            parts = f.stem.split("_", 1)
            if len(parts) == 2:
                name_map[parts[0]] = parts[1]

    all_results = {}

    for std_th in std_thresholds:
        print(f"\n{'='*60}")
        print(f"  std 기준: {std_th}")
        print(f"{'='*60}")

        total_patterns = 0
        total_recover_3d = 0
        total_recover_5d = 0
        total_recover_10d = 0
        total_above_ma120 = 0
        stock_hits = []
        all_patterns = []

        for pf in parquet_files:
            code = Path(pf).stem
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue

            patterns = scan_shakeout_patterns(df, std_threshold=std_th)

            if patterns:
                name = name_map.get(code, code)
                stock_hits.append((code, name, len(patterns)))
                for p in patterns:
                    p["code"] = code
                    p["name"] = name
                    all_patterns.append(p)
                    total_patterns += 1
                    if p.get("recover_3d"):
                        total_recover_3d += 1
                    if p.get("recover_5d"):
                        total_recover_5d += 1
                    if p.get("recover_10d"):
                        total_recover_10d += 1
                    if p.get("above_ma120"):
                        total_above_ma120 += 1

        # 결과 요약
        print(f"\n총 패턴 발생: {total_patterns}건 (종목 {len(stock_hits)}개)")

        if total_patterns > 0:
            # MA120 필터 적용 시
            ma120_patterns = [p for p in all_patterns if p["above_ma120"]]
            ma120_count = len(ma120_patterns)
            ma120_r3 = sum(1 for p in ma120_patterns if p.get("recover_3d"))
            ma120_r5 = sum(1 for p in ma120_patterns if p.get("recover_5d"))
            ma120_r10 = sum(1 for p in ma120_patterns if p.get("recover_10d"))

            print(f"\n[전체 패턴] (MA120 무관)")
            print(f"  3일 회복률: {total_recover_3d}/{total_patterns} = {total_recover_3d/total_patterns*100:.1f}%")
            print(f"  5일 회복률: {total_recover_5d}/{total_patterns} = {total_recover_5d/total_patterns*100:.1f}%")
            print(f"  10일 회복률: {total_recover_10d}/{total_patterns} = {total_recover_10d/total_patterns*100:.1f}%")
            print(f"  MA120 위 비율: {total_above_ma120}/{total_patterns} = {total_above_ma120/total_patterns*100:.1f}%")

            if ma120_count > 0:
                print(f"\n[MA120 위 패턴만] ({ma120_count}건)")
                print(f"  3일 회복률: {ma120_r3}/{ma120_count} = {ma120_r3/ma120_count*100:.1f}%")
                print(f"  5일 회복률: {ma120_r5}/{ma120_count} = {ma120_r5/ma120_count*100:.1f}%")
                print(f"  10일 회복률: {ma120_r10}/{ma120_count} = {ma120_r10/ma120_count*100:.1f}%")

                # 10일 후 수익률 통계
                future_rets = [p["future_10d_ret"] for p in ma120_patterns if p["future_10d_ret"] is not None]
                if future_rets:
                    print(f"\n  [MA120 위] 급락 저점 → 10일 후 수익률:")
                    print(f"    평균: {np.mean(future_rets):+.2f}%")
                    print(f"    중앙: {np.median(future_rets):+.2f}%")
                    print(f"    최소: {np.min(future_rets):+.2f}%")
                    print(f"    최대: {np.max(future_rets):+.2f}%")
                    print(f"    양수 비율: {sum(1 for r in future_rets if r > 0)/len(future_rets)*100:.1f}%")

            # 상위 패턴 상세
            print(f"\n[패턴 상세 (MA120 위, 상위 15건)]")
            ma120_sorted = sorted(ma120_patterns, key=lambda x: x["drop_pct"])
            for p in ma120_sorted[:15]:
                r3 = "O" if p.get("recover_3d") else "X"
                r5 = "O" if p.get("recover_5d") else "X"
                r10 = "O" if p.get("recover_10d") else "X"
                fut = f"{p['future_10d_ret']:+.1f}%" if p["future_10d_ret"] else "N/A"
                print(f"  {p['name'][:8]:8s} {p['date']} | "
                      f"하락 {p['drop_pct']:+.1f}%({p['drop_days']}일) | "
                      f"거래량 {p['vol_ratio']:.1f}x | "
                      f"MA120 +{p['ma120_margin']:.1f}% | "
                      f"회복 3d:{r3} 5d:{r5} 10d:{r10} | "
                      f"10일후: {fut}")

        all_results[std_th] = {
            "total": total_patterns,
            "ma120_count": len([p for p in all_patterns if p["above_ma120"]]),
            "recover_3d": total_recover_3d,
            "recover_5d": total_recover_5d,
            "recover_10d": total_recover_10d,
            "patterns": all_patterns,
        }

    # ── 최종 비교표 ──
    print(f"\n\n{'='*70}")
    print(f"  최종 비교표: std 기준별 결과")
    print(f"{'='*70}")
    print(f"{'std 기준':>10} | {'전체':>6} | {'MA120위':>7} | {'3일회복':>8} | {'5일회복':>8} | {'10일회복':>8}")
    print(f"{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for th in std_thresholds:
        r = all_results[th]
        mc = r["ma120_count"]
        if mc > 0:
            ma120_pats = [p for p in r["patterns"] if p["above_ma120"]]
            r3 = sum(1 for p in ma120_pats if p.get("recover_3d"))
            r5 = sum(1 for p in ma120_pats if p.get("recover_5d"))
            r10 = sum(1 for p in ma120_pats if p.get("recover_10d"))
            print(f"   {th:.3f}  | {r['total']:>5}건 | {mc:>5}건 | "
                  f"{r3:>3}/{mc:<3} {r3/mc*100:4.0f}% | "
                  f"{r5:>3}/{mc:<3} {r5/mc*100:4.0f}% | "
                  f"{r10:>3}/{mc:<3} {r10/mc*100:4.0f}%")
        else:
            print(f"   {th:.3f}  | {r['total']:>5}건 | {mc:>5}건 | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")

    return all_results


# ──────────────────────────────────────────
# 3. 보유 3종목 pykrx 분석
# ──────────────────────────────────────────
def analyze_held_stocks():
    """보유 3종목을 pykrx에서 직접 수집하여 분석."""
    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        print("pykrx 미설치")
        return

    targets = {
        "036640": "HRS",
        "107590": "미원홀딩스",
        "036560": "KZ정밀",
    }

    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")

    print(f"\n\n{'='*60}")
    print(f"  보유 3종목 분석 (pykrx)")
    print(f"  기간: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    for code, name in targets.items():
        print(f"\n── {name} ({code}) ──")
        try:
            # OHLCV 수집
            df = pykrx_stock.get_market_ohlcv(start_date, end_date, code)
            if df.empty or len(df) < 130:
                print(f"  데이터 부족 ({len(df)}행)")
                continue

            # 컬럼 정리
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
            df = df.rename(columns=col_map)

            if "close" not in df.columns:
                # pykrx가 한글 컬럼인 경우
                if "종가" in df.columns:
                    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})

            df["ret1"] = df["close"].pct_change()
            df["sma_120"] = df["close"].rolling(120).mean()
            df["volume_ma20"] = df["volume"].rolling(20).mean()

            # 스캔
            for std_th in [0.010, 0.015, 0.020]:
                patterns = scan_shakeout_patterns(df, std_threshold=std_th)
                if patterns:
                    print(f"\n  [std < {std_th}] 패턴 {len(patterns)}건:")
                    for p in patterns:
                        r3 = "O" if p.get("recover_3d") else "X"
                        r5 = "O" if p.get("recover_5d") else "X"
                        r10 = "O" if p.get("recover_10d") else "X"
                        ma = "위" if p["above_ma120"] else "아래"
                        fut = f"{p['future_10d_ret']:+.1f}%" if p["future_10d_ret"] else "N/A"
                        print(f"    {p['date']} | 하락 {p['drop_pct']:+.1f}%({p['drop_days']}일) | "
                              f"거래량 {p['vol_ratio']:.1f}x | MA120 {ma} | "
                              f"회복 3d:{r3} 5d:{r5} 10d:{r10} | 10일후: {fut}")
                else:
                    print(f"  [std < {std_th}] 패턴 0건")

            # 투자자별 수급 (패턴 날짜 있으면)
            print(f"\n  [투자자 수급 조회]")
            try:
                investor_df = pykrx_stock.get_market_trading_volume_by_date(
                    start_date, end_date, code, detail=True
                )
                if not investor_df.empty:
                    print(f"    투자자 데이터 {len(investor_df)}행 확보")
                    # 최근 패턴 날짜의 수급 확인
                    all_pats = scan_shakeout_patterns(df, std_threshold=0.020)
                    for p in all_pats[-5:]:
                        pat_date = p["date"].replace("-", "")
                        if pat_date in investor_df.index.strftime("%Y%m%d").tolist():
                            inv_row = investor_df.loc[p["date"]]
                            print(f"    {p['date']}: 개인={inv_row.get('개인', 'N/A'):,}, "
                                  f"기관={inv_row.get('기관합계', 'N/A'):,}, "
                                  f"외인={inv_row.get('외국인합계', 'N/A'):,}")
            except Exception as e:
                print(f"    투자자 데이터 수집 실패: {e}")

        except Exception as e:
            print(f"  오류: {e}")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Shakeout Detector 백테스트 (Step 1)")
    print("  패턴: 횡보 → 급락 → V자 회복")
    print("=" * 60)

    # 1) 유니버스 전체 스캔
    results = run_universe_backtest()

    # 2) 보유 3종목
    analyze_held_stocks()

    print("\n\n" + "=" * 60)
    print("  백테스트 완료")
    print("=" * 60)
