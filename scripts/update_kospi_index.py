"""KOSPI 인덱스 업데이트 (레짐 판별용)"""
from pathlib import Path
import yfinance as yf

OUT = Path(__file__).resolve().parent.parent / "data" / "kospi_index.csv"

def main():
    df = yf.download("^KS11", period="3y", progress=False)
    df.columns = [c[0].lower() for c in df.columns]
    df.to_csv(OUT)
    print(f"KOSPI {len(df)} rows saved → {OUT.name}")

if __name__ == "__main__":
    main()
