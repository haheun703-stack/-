@echo off
REM ============================================================
REM  Quantum Master - BAT-D2: 수급 데이터 확정 후 수집
REM  스케줄: 매일 18:30 (KRX 투자자별 매매동향 확정 18:10 이후)
REM  등록: schtasks /create /tn "QM_D2_Supply" /tr "wscript.exe D:\sub-agent-project_퀀트봇\scripts\run_hidden.vbs D:\sub-agent-project_퀀트봇\scripts\schedule_D2_supply.bat" /sc daily /st 18:30
REM
REM  역할: parquet 수급 채워넣기 + collect_supply + SD V2 패턴 저장
REM  예상 소요: ~15분
REM ============================================================

echo [%date% %time%] ================================================== >> D:\sub-agent-project_퀀트봇\logs\schedule.log
echo [%date% %time%] BAT-D2 시작: 수급 데이터 확정 후 수집 >> D:\sub-agent-project_퀀트봇\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project_퀀트봇\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_퀀트봇
set PYTHONPATH=D:\sub-agent-project_퀀트봇

if not exist logs mkdir logs

REM ── 거래일 가드 ──
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-D2 스킵: 비거래일 >> logs\schedule.log
    goto :eof
)

REM ══════════════════════════════════════════════
REM  STEP 1: Parquet 수급 채워넣기 (--supply-only)
REM  OHLCV는 BAT-D(16:30)에서 이미 수집됨
REM  수급이 0인 최근 날짜만 찾아서 pykrx로 재수집
REM ══════════════════════════════════════════════
echo [%date% %time%] [D2-1/4] Parquet 수급 채워넣기 >> logs\schedule.log
python -u -X utf8 scripts\extend_parquet_data.py --supply-only >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] [D2-1/4] FAILED >> logs\schedule.log

REM ══════════════════════════════════════════════
REM  STEP 2: 수급 이면 데이터 수집 (collect_supply_data)
REM ══════════════════════════════════════════════
echo [%date% %time%] [D2-2/4] 수급 이면 데이터 수집 >> logs\schedule.log
python -u -X utf8 scripts\collect_supply_data.py >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] [D2-2/4] FAILED >> logs\schedule.log

REM ══════════════════════════════════════════════
REM  STEP 3: 기술지표 재계산 (수급 반영된 parquet로)
REM ══════════════════════════════════════════════
echo [%date% %time%] [D2-3/4] 기술지표 재계산 (수급 반영) >> logs\schedule.log
python -u -X utf8 scripts\rebuild_indicators.py >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] [D2-3/4] FAILED >> logs\schedule.log

REM ══════════════════════════════════════════════
REM  STEP 4: SD V2 패턴 일별 저장 (전환 추적용)
REM  scan_buy에서도 저장하지만, 수급 확정 후 재저장
REM ══════════════════════════════════════════════
echo [%date% %time%] [D2-4/4] SD V2 패턴 일별 저장 >> logs\schedule.log
python -u -X utf8 -c "
import sys
sys.path.insert(0, 'D:/sub-agent-project')
from datetime import date
from pathlib import Path
import pandas as pd
from src.alpha.factors.sd_score_v2 import compute_sd_features
from src.alpha.factors.sd_transition_tracker import save_daily_patterns

today_str = date.today().strftime('%%Y-%%m-%%d')
raw_dir = Path('data/processed')
patterns = {}
for p in sorted(raw_dir.glob('*.parquet')):
    ticker = p.stem
    try:
        df = pd.read_parquet(p)
        if len(df) < 20:
            continue
        idx = len(df) - 1
        feat = compute_sd_features(df, idx, ticker)
        patterns[ticker] = {
            'name': ticker,
            'pattern': feat.pattern,
            'pattern_name': feat.pattern_name,
            'sd_score': round(feat.sd_score, 4),
            'foreign_net_20d': round(feat.foreign_net_20d, 1),
            'inst_net_20d': round(feat.inst_net_20d, 1),
            'individual_net_20d': round(feat.individual_net_20d, 1),
        }
    except Exception:
        pass
save_daily_patterns(today_str, patterns)
print(f'SD V2 패턴 저장: {len(patterns)}종목')
" >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] [D2-4/4] FAILED >> logs\schedule.log

echo [%date% %time%] BAT-D2 완료 (4단계) >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
