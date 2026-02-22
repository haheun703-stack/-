"""기술지표 재계산 (raw → processed parquet)"""
from src.indicators import IndicatorEngine

def main():
    e = IndicatorEngine()
    e.process_all()

if __name__ == "__main__":
    main()
