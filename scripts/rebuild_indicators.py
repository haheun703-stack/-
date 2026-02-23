"""기술지표 재계산 (raw → processed parquet)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators import IndicatorEngine

def main():
    e = IndicatorEngine()
    e.process_all()

if __name__ == "__main__":
    main()
