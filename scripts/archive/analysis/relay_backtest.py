"""슈퍼섹터 릴레이 백테스팅 — 과거 패턴 검증.

"선행 섹터가 N% 이상 오른 다음날, 후행 섹터가 몇% 확률로 올랐나?"
를 2년치 데이터로 전수 검증하여 승률/래그/신뢰도를 계산한다.

NOTE: 수익률은 섹터 평균 등락률(gross). 실제 매매 시 왕복 수수료(0.03%)
      + 세금(0.18%) + 슬리피지(≈0.5%) ≈ 총 0.7%를 차감해야 함.

입력:
  - stock_data_daily/*.csv  (2,859종목 일봉)
  - naver_sector_map.csv    (네이버 79개 업종)

출력:
  - data/sector_rotation/relay_patterns.json
    → sector_relay_engine.py가 신뢰도 표시에 활용

사용법:
  python scripts/relay_backtest.py              # 전체 백테스팅
  python scripts/relay_backtest.py --threshold 3 # 발화 기준 3%
  python scripts/relay_backtest.py --days 756    # 3년치
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "sector_rotation"
DAILY_DIR = PROJECT_ROOT / "stock_data_daily"


# ─────────────────────────────────────────────
# 슈퍼섹터 정의 (네이버 79개 업종 기준)
# relay_order: 선행(앞) → 후행(뒤) 순서
# ─────────────────────────────────────────────

SUPER_SECTORS = {
    "금융": {
        "sectors": ["증권", "생명보험", "손해보험", "은행", "카드", "기타금융"],
        "relay_pairs": [
            ("증권", "생명보험"),
            ("증권", "손해보험"),
            ("증권", "은행"),
            ("생명보험", "손해보험"),
            ("은행", "생명보험"),
            ("은행", "손해보험"),
        ],
    },
    "IT/반도체": {
        "sectors": ["반도체와반도체장비", "디스플레이장비및부품", "전자장비와기기", "IT서비스", "소프트웨어"],
        "relay_pairs": [
            ("반도체와반도체장비", "디스플레이장비및부품"),
            ("반도체와반도체장비", "전자장비와기기"),
            ("반도체와반도체장비", "IT서비스"),
            ("디스플레이장비및부품", "전자장비와기기"),
        ],
    },
    "2차전지/에너지": {
        "sectors": ["전기제품", "화학", "에너지장비및서비스", "석유와가스"],
        "relay_pairs": [
            ("전기제품", "화학"),
            ("전기제품", "에너지장비및서비스"),
            ("화학", "에너지장비및서비스"),
            ("석유와가스", "화학"),
        ],
    },
    "방산/조선": {
        "sectors": ["우주항공과국방", "조선", "기계"],
        "relay_pairs": [
            ("우주항공과국방", "조선"),
            ("우주항공과국방", "기계"),
            ("조선", "기계"),
        ],
    },
    "건설/소재": {
        "sectors": ["건설", "건축자재", "철강", "비철금속"],
        "relay_pairs": [
            ("건설", "건축자재"),
            ("건설", "철강"),
            ("철강", "비철금속"),
        ],
    },
    "소비/유통": {
        "sectors": ["백화점과일반상점", "식품", "음료", "화장품"],
        "relay_pairs": [
            ("백화점과일반상점", "식품"),
            ("백화점과일반상점", "화장품"),
            ("화장품", "식품"),
        ],
    },
    "바이오/헬스": {
        "sectors": ["제약", "생물공학", "건강관리장비와용품", "생명과학도구및서비스"],
        "relay_pairs": [
            ("제약", "생물공학"),
            ("제약", "건강관리장비와용품"),
            ("생물공학", "건강관리장비와용품"),
        ],
    },
}


# ─────────────────────────────────────────────
# 1. 데이터 로드 + 섹터별 일별 등락률 빌드
# ─────────────────────────────────────────────

def load_naver_sector_map() -> dict[str, str]:
    """네이버 섹터맵 로드 → {ticker: sector}"""
    path = DATA_DIR / "naver_sector_map.csv"
    df = pd.read_csv(path, dtype={"ticker": str})
    return dict(zip(df["ticker"], df["sector"]))


def build_sector_daily_returns(
    ticker_to_sector: dict[str, str],
    min_stocks: int = 3,
) -> pd.DataFrame:
    """전종목 CSV에서 섹터별 일별 등락률 DataFrame 빌드.

    Returns:
        DataFrame index=date, columns=[sector1_ret, sector1_breadth, sector1_count, ...]
        Wide format → 이후 섹터명으로 직접 조회
    """
    logger.info("stock_data_daily CSV 로드 중...")

    # 모든 CSV에서 (ticker, date, close) 수집
    all_closes = {}  # {ticker: Series(date→close)}
    csv_files = list(DAILY_DIR.glob("*.csv"))
    loaded = 0

    for csv_path in csv_files:
        # 파일명: "종목명_티커.csv" → rsplit으로 티커 추출
        parts = csv_path.stem.rsplit("_", 1)
        if len(parts) < 2:
            continue
        ticker = parts[-1]
        if ticker not in ticker_to_sector:
            continue
        try:
            df = pd.read_csv(csv_path, usecols=["Date", "Close"], parse_dates=["Date"])
            df = df.dropna().set_index("Date").sort_index()
            if len(df) < 60:
                continue
            all_closes[ticker] = df["Close"]
            loaded += 1
        except Exception:
            continue

    logger.info("CSV 로드 완료: %d종목", loaded)

    # 종목별 일별 수익률 계산
    all_returns = {}  # {ticker: Series(date→pct_change)}
    for ticker, closes in all_closes.items():
        ret = closes.pct_change() * 100
        all_returns[ticker] = ret.dropna()

    # 섹터별 집계
    sectors = set(ticker_to_sector.values())
    sector_groups = {s: [] for s in sectors}
    for ticker, ret_series in all_returns.items():
        sector = ticker_to_sector[ticker]
        sector_groups[sector].append(ret_series)

    # 섹터별 일별 평균수익률 + breadth
    result_frames = []

    for sector, series_list in sector_groups.items():
        if len(series_list) < min_stocks:
            continue

        # DataFrame으로 합치기 (공통 날짜)
        df_combined = pd.DataFrame({f"s{i}": s for i, s in enumerate(series_list)})
        df_combined = df_combined.dropna(thresh=max(3, len(series_list) // 2))

        avg_ret = df_combined.mean(axis=1)
        breadth = (df_combined > 0).sum(axis=1) / df_combined.notna().sum(axis=1)
        count = df_combined.notna().sum(axis=1)

        frame = pd.DataFrame({
            f"{sector}__ret": avg_ret,
            f"{sector}__breadth": breadth,
            f"{sector}__count": count,
        })
        result_frames.append(frame)

    if not result_frames:
        return pd.DataFrame()

    sector_df = pd.concat(result_frames, axis=1).sort_index()
    logger.info(
        "섹터 일별 데이터: %d거래일, %d개 섹터",
        len(sector_df),
        len([c for c in sector_df.columns if c.endswith("__ret")]),
    )
    return sector_df


# ─────────────────────────────────────────────
# 2. 백테스팅 핵심 로직
# ─────────────────────────────────────────────

def backtest_relay_pair(
    sector_df: pd.DataFrame,
    lead: str,
    follow: str,
    fire_threshold: float = 5.0,
    breadth_threshold: float = 0.7,
    lag_days: list[int] | None = None,
    min_samples: int = 3,
) -> dict:
    """선행→후행 패턴 백테스팅.

    Returns:
        {lag1: {win_rate, avg_return, samples, confidence}, ..., best_lag}
    """
    if lag_days is None:
        lag_days = [1, 2, 3]

    lead_ret_col = f"{lead}__ret"
    lead_br_col = f"{lead}__breadth"
    follow_ret_col = f"{follow}__ret"

    # 컬럼 존재 확인
    for col in [lead_ret_col, lead_br_col, follow_ret_col]:
        if col not in sector_df.columns:
            return {"error": f"컬럼 없음: {col}", "samples": 0}

    # 발화일 추출
    fire_mask = (
        (sector_df[lead_ret_col] >= fire_threshold)
        & (sector_df[lead_br_col] >= breadth_threshold)
    )
    fire_dates = sector_df.index[fire_mask].tolist()

    if not fire_dates:
        return {
            "lead": lead, "follow": follow,
            "fire_dates": 0, "samples": 0,
            "error": "발화 샘플 없음",
        }

    # 래그별 후행 수익률 수집
    dates_list = sector_df.index.tolist()
    lag_results = {}

    for lag in lag_days:
        returns = []
        cases = []

        for fire_date in fire_dates:
            try:
                idx = dates_list.index(fire_date)
                target_idx = idx + lag
                if target_idx >= len(dates_list):
                    continue
                target_date = dates_list[target_idx]

                follow_ret = sector_df.loc[target_date, follow_ret_col]
                lead_ret = sector_df.loc[fire_date, lead_ret_col]

                if pd.isna(follow_ret):
                    continue

                returns.append(float(follow_ret))
                cases.append({
                    "fire_date": fire_date.strftime("%Y-%m-%d"),
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "lead_ret": round(float(lead_ret), 2),
                    "follow_ret": round(float(follow_ret), 2),
                })
            except (ValueError, IndexError):
                continue

        n = len(returns)
        if n < min_samples:
            lag_results[f"lag{lag}"] = {
                "win_rate": 0, "avg_return": 0,
                "samples": n, "confidence": "LOW",
            }
            continue

        win_rate = round(sum(1 for r in returns if r > 0) / n * 100, 1)
        avg_return = round(float(np.mean(returns)), 2)
        med_return = round(float(np.median(returns)), 2)
        std_return = round(float(np.std(returns)), 2)

        if win_rate >= 65 and n >= 8:
            confidence = "HIGH"
        elif win_rate >= 55 and n >= 5:
            confidence = "MED"
        else:
            confidence = "LOW"

        # 케이스를 후행 수익률 내림차순 정렬
        cases.sort(key=lambda x: x["follow_ret"], reverse=True)

        lag_results[f"lag{lag}"] = {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "med_return": med_return,
            "std_return": std_return,
            "samples": n,
            "confidence": confidence,
            "best_cases": cases[:3],
            "worst_cases": cases[-3:],
        }

    # 최적 래그 선택 (승률 × 평균수익 최대)
    best_lag = 1
    best_score = -999
    for lag in lag_days:
        key = f"lag{lag}"
        data = lag_results.get(key, {})
        if data.get("samples", 0) < min_samples:
            continue
        score = data.get("win_rate", 0) * max(data.get("avg_return", 0), 0.01)
        if score > best_score:
            best_score = score
            best_lag = lag

    return {
        "lead": lead,
        "follow": follow,
        "fire_dates": len(fire_dates),
        "best_lag": best_lag,
        **lag_results,
    }


# ─────────────────────────────────────────────
# 3. 전체 실행
# ─────────────────────────────────────────────

def run_full_backtest(
    fire_threshold: float = 5.0,
    breadth_threshold: float = 0.7,
    backtest_days: int = 504,
) -> dict:
    """모든 슈퍼섹터 릴레이 쌍 백테스팅."""

    print(f"\n{'=' * 60}")
    print(f"  슈퍼섹터 릴레이 백테스팅")
    print(f"  발화 기준: +{fire_threshold}% & 상승비율 {breadth_threshold*100:.0f}%")
    print(f"  기간: 최근 {backtest_days}거래일 (약 {backtest_days/252:.1f}년)")
    print(f"{'=' * 60}")

    # 섹터 매핑 로드
    ticker_to_sector = load_naver_sector_map()
    logger.info("네이버 섹터맵: %d종목", len(ticker_to_sector))

    # 섹터 일별 등락률 빌드
    sector_df = build_sector_daily_returns(ticker_to_sector)
    if sector_df.empty:
        print("  데이터 부족!")
        return {}

    # 최근 N거래일만
    sector_df = sector_df.tail(backtest_days)
    logger.info("백테스팅 기간: %s ~ %s", sector_df.index[0].strftime("%Y-%m-%d"),
                sector_df.index[-1].strftime("%Y-%m-%d"))

    # 릴레이 쌍 백테스팅
    all_results = {}

    for super_name, config in SUPER_SECTORS.items():
        print(f"\n▣ [{super_name}]")
        all_results[super_name] = {}

        for lead, follow in config["relay_pairs"]:
            result = backtest_relay_pair(
                sector_df, lead, follow,
                fire_threshold=fire_threshold,
                breadth_threshold=breadth_threshold,
            )
            all_results[super_name][f"{lead}→{follow}"] = result

            # 요약 출력
            best_lag = result.get("best_lag", 1)
            lag_data = result.get(f"lag{best_lag}", {})
            wr = lag_data.get("win_rate", 0)
            ret = lag_data.get("avg_return", 0)
            conf = lag_data.get("confidence", "?")
            n = lag_data.get("samples", 0)
            fire_n = result.get("fire_dates", 0)

            conf_icon = {"HIGH": "🟢", "MED": "🟡", "LOW": "🔴"}.get(conf, "⚫")
            print(
                f"  {lead:<16} → {follow:<16} "
                f"발화{fire_n:>3}회  "
                f"래그{best_lag}일  "
                f"승률{wr:>5.1f}%  "
                f"평균{ret:>+6.2f}%  "
                f"{conf_icon}[{conf}] n={n}"
            )

    return all_results


# ─────────────────────────────────────────────
# 4. 저장 + 요약
# ─────────────────────────────────────────────

def save_patterns(results: dict, fire_threshold: float,
                  breadth_threshold: float, backtest_days: int):
    """relay_patterns.json 저장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "relay_patterns.json"

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fire_threshold": fire_threshold,
        "breadth_threshold": breadth_threshold,
        "backtest_days": backtest_days,
        "super_sectors": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    logger.info("패턴 저장 → %s", path)
    return path


def print_summary(results: dict):
    """HIGH/MED 신뢰도 패턴 요약."""
    print(f"\n{'=' * 70}")
    print(f"  릴레이 패턴 요약 (MED 이상만)")
    print(f"{'=' * 70}")
    print(f"  {'슈퍼섹터':<10} {'선행→후행':<35} {'래그':>3} {'승률':>6} {'평균':>7} {'신뢰도'}")
    print(f"  {'─' * 65}")

    found = 0
    for super_name, pairs in results.items():
        for pair_name, data in pairs.items():
            best_lag = data.get("best_lag", 1)
            lag_data = data.get(f"lag{best_lag}", {})
            conf = lag_data.get("confidence", "")
            if conf not in ("HIGH", "MED"):
                continue

            wr = lag_data.get("win_rate", 0)
            ret = lag_data.get("avg_return", 0)
            n = lag_data.get("samples", 0)
            icon = "🟢" if conf == "HIGH" else "🟡"

            print(
                f"  {super_name:<10} {pair_name:<35} "
                f"{best_lag}일 {wr:>5.1f}% {ret:>+6.2f}%  "
                f"{icon} {conf} (n={n})"
            )
            found += 1

    if found == 0:
        print("  MED 이상 패턴 없음 (발화 기준 완화 필요 가능)")

    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="슈퍼섹터 릴레이 백테스팅")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="발화 기준 등락률 (기본 3.5%%)")
    parser.add_argument("--breadth", type=float, default=0.6,
                        help="발화 기준 상승비율 (기본 0.6)")
    parser.add_argument("--days", type=int, default=504,
                        help="백테스팅 기간 거래일 (기본 504=약2년)")
    args = parser.parse_args()

    results = run_full_backtest(
        fire_threshold=args.threshold,
        breadth_threshold=args.breadth,
        backtest_days=args.days,
    )

    if results:
        save_patterns(results, args.threshold, args.breadth, args.days)
        print_summary(results)
        print("  relay_patterns.json 생성 완료.")
        print("  → sector_relay_engine.py가 이 데이터를 참조합니다.")


if __name__ == "__main__":
    main()
