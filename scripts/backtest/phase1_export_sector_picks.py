"""Phase 1: quant_sector_picks export + 정보봇 OHLCV 조인 정합성 검증

목표:
- Supabase quant_sector_picks 1,046행 (4/25~5/14 종목 단위 점수) 전체 export
- 정보봇 OHLCV (data/external/jgis_ohlcv/) 와 종목 조인 가능 여부 검증
- 결측 종목/일자 카운트 보고

출력:
- data/backtest/sector_picks_raw.parquet (전체 점수 데이터)
- data/backtest/phase1_report.md (검증 결과 보고서)

실행:
    python -X utf8 scripts/backtest/phase1_export_sector_picks.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "data" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JGIS_OHLCV = Path("/home/ubuntu/quantum-master/data/external/jgis_ohlcv")
LOCAL_OHLCV = PROJECT_ROOT / "stock_data_daily"


def export_picks():
    """Supabase quant_sector_picks 전체 export (페이지네이션)"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    sb = create_client(url, key)

    all_rows = []
    page = 0
    page_size = 1000
    while True:
        offset = page * page_size
        res = sb.table("quant_sector_picks").select("*").range(offset, offset + page_size - 1).execute()
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < page_size:
            break
        page += 1

    df = pd.DataFrame(all_rows)
    print(f"[export] quant_sector_picks 총 {len(df)}행")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  종목 수 (unique): {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
    return df


def verify_ohlcv_join(df_picks):
    """정보봇 + 퀀트봇 OHLCV에서 picks 종목 매칭 검증"""
    tickers = df_picks["ticker"].astype(str).str.zfill(6).unique() if "ticker" in df_picks.columns else []
    print(f"\n[조인 검증] picks 종목 {len(tickers)}개")

    # 로컬 OHLCV에서 검증 (VPS jgis는 SSH 필요해서 로컬로 대체)
    local_files = {f.stem.split("_")[-1]: f for f in LOCAL_OHLCV.glob("*.csv")}
    print(f"  로컬 stock_data_daily/: {len(local_files)}개 CSV")

    matched = [t for t in tickers if t in local_files]
    missing = [t for t in tickers if t not in local_files]
    print(f"  매칭: {len(matched)}/{len(tickers)} 종목 ({len(matched)/len(tickers)*100:.1f}%)")
    if missing:
        print(f"  누락 종목 샘플 (10개): {missing[:10]}")
    return matched, missing


def report(df_picks, matched, missing):
    """phase1_report.md 생성"""
    out_path = OUT_DIR / "phase1_report.md"
    content = f"""# Phase 1 데이터 수집 보고서

**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## quant_sector_picks 요약

- **총 행수**: {len(df_picks):,}
- **기간**: {df_picks['date'].min()} ~ {df_picks['date'].max()}
- **종목 수 (unique)**: {df_picks['ticker'].nunique() if 'ticker' in df_picks.columns else 'N/A'}
- **컬럼**: {len(df_picks.columns)}개 — {list(df_picks.columns)}

## 일자별 행수

```
{df_picks.groupby('date').size().to_string()}
```

## OHLCV 조인 정합성

- **picks 종목**: {len(matched) + len(missing)}개
- **로컬 OHLCV 매칭**: {len(matched)}개 ({len(matched)/(len(matched)+len(missing))*100:.1f}%)
- **누락**: {len(missing)}개

### 누락 종목 (전체 리스트)

```
{missing}
```

## 다음 단계 (Phase 2)

- D+1, D+3, D+5, D+10 수익률 계산
- 점수(final_score) vs 수익률 Pearson correlation
- BUY/WATCH 시그널별 적중률 측정

## 한계

- **데이터 기간이 20일 (4/25~5/14)** — 6개월 목표 대비 부족
- 통계적 유의성은 1~3개월 후 재검증 필요
- 시작은 방법론 정립 + 초기 결과 확인용
"""
    out_path.write_text(content, encoding="utf-8")
    print(f"\n[report] {out_path}")


def main():
    print("=" * 60)
    print("Phase 1: quant_sector_picks export + OHLCV 조인 검증")
    print("=" * 60)

    df = export_picks()
    df.to_parquet(OUT_DIR / "sector_picks_raw.parquet", index=False)
    print(f"\n[save] {OUT_DIR / 'sector_picks_raw.parquet'} ({len(df)}행)")

    matched, missing = verify_ohlcv_join(df)
    report(df, matched, missing)

    print(f"\n[OK] Phase 1 완료")


if __name__ == "__main__":
    main()
