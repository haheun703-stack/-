"""ETF 전체 유니버스 1년치 데이터 백필 (yfinance).

TIGER 22종 + KODEX 추가 9종 + 레버리지 3종 + 지수 2종 = 총 36종
2025-03-01 ~ 현재까지 yfinance로 OHLCV 수집.

사용법:
  python scripts/backfill_etf_1year.py
  python scripts/backfill_etf_1year.py --start 2025-03-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DAILY_DIR = PROJECT_ROOT / "data" / "sector_rotation" / "etf_daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 전체 ETF 유니버스 (36종)
# ─────────────────────────────────────────────

ALL_ETFS = {
    # ── TIGER 섹터 22종 ──
    "157500": "TIGER 증권",
    "091220": "TIGER 은행",
    "140710": "TIGER 보험",
    "091230": "TIGER 반도체",
    "305540": "TIGER 2차전지테마",
    "364970": "TIGER 바이오TOP10",
    "143860": "TIGER 헬스케어",
    "139260": "TIGER 200 IT",
    "139220": "TIGER 200 건설",
    "139270": "TIGER 200 금융",
    "139250": "TIGER 200 에너지화학",
    "139240": "TIGER 200 철강소재",
    "157490": "TIGER 소프트웨어",
    "228810": "TIGER 미디어컨텐츠",
    "300610": "TIGER K게임",
    "365000": "TIGER 인터넷TOP10",
    "463250": "TIGER K방산&우주",
    "494670": "TIGER 조선TOP10",
    "138520": "TIGER 삼성그룹",
    "138540": "TIGER 현대차그룹플러스",
    "292160": "TIGER KRX300",
    "292150": "TIGER 코리아TOP10",
    # ── KODEX 추가 9종 ──
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "091180": "KODEX 자동차",
    "117700": "KODEX 건설",
    "117460": "KODEX 에너지화학",
    "305720": "KODEX 2차전지산업",
    "244580": "KODEX 바이오",
    "266360": "KODEX 핀테크",
    "140700": "KODEX 운송",
    # ── 레버리지/인버스 3종 ──
    "122630": "KODEX 레버리지",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    # ── 지수 ETF 2종 ──
    "069500": "KODEX 200",
    "278530": "KODEX MSCI Korea TR",
}


def backfill(start: str = "2025-03-01", end: str = "2026-02-28") -> None:
    """전체 ETF 유니버스 OHLCV 백필 (yfinance)."""
    total = len(ALL_ETFS)
    success = 0
    fail = 0
    skip = 0

    print("=" * 60)
    print(f"  ETF 1년치 백필 (yfinance): {start} ~ {end}")
    print(f"  대상: {total}종")
    print("=" * 60)

    for i, (code, name) in enumerate(ALL_ETFS.items(), 1):
        out_path = DAILY_DIR / f"{code}.parquet"

        # 이미 충분한 데이터 존재 시 스킵
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                if len(existing) >= 230:
                    first = existing.index[0]
                    if hasattr(first, "strftime"):
                        first_str = first.strftime("%Y-%m-%d")
                    else:
                        first_str = str(first)
                    if first_str <= start:
                        logger.info(
                            "[%d/%d] %s (%s): 이미 %d일 보유 — skip",
                            i, total, name, code, len(existing),
                        )
                        skip += 1
                        continue
            except Exception:
                pass

        ticker = f"{code}.KS"
        logger.info("[%d/%d] %s (%s) → %s 수집 중...", i, total, name, code, ticker)

        try:
            df = yf.download(ticker, start=start, end=end, progress=False)

            if df is None or df.empty:
                logger.warning("  → 데이터 없음")
                fail += 1
                continue

            # MultiIndex 컬럼 평탄화
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 컬럼명 소문자 통일
            df.columns = [c.lower() for c in df.columns]

            # 필요 컬럼만 유지
            keep = ["open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]

            df.index.name = "date"

            # 기존 parquet과 병합 (기존 데이터에 nav, trading_value 등 있을 수 있음)
            if out_path.exists():
                try:
                    old = pd.read_parquet(out_path)
                    # 기존 데이터에서 yfinance에 없는 컬럼 보존
                    extra_cols = [c for c in old.columns if c not in df.columns]
                    if extra_cols:
                        # 날짜 기준 join
                        for col in extra_cols:
                            df[col] = old.reindex(df.index)[col]

                    # 오래된 날짜 데이터도 병합
                    old_only = old[old.index < df.index.min()]
                    if not old_only.empty:
                        df = pd.concat([old_only[df.columns.intersection(old_only.columns)], df])
                        df = df[~df.index.duplicated(keep="last")]
                        df.sort_index(inplace=True)
                except Exception:
                    pass

            df.to_parquet(out_path)
            success += 1
            logger.info(
                "  → %d거래일 저장 (%s ~ %s)",
                len(df),
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
            )

        except Exception as e:
            logger.error("  → 실패: %s", e)
            fail += 1

    print()
    print("=" * 60)
    print(f"  완료: 성공 {success} / 스킵 {skip} / 실패 {fail} / 총 {total}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ETF 전체 유니버스 1년치 백필")
    parser.add_argument("--start", default="2025-03-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-28", help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()
    backfill(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
