@echo off
REM ============================================================
REM  Quantum Master - BAT-SNAP: ÀåÁß ¼ö±Þ ½º³À¼¦
REM  ½ºÄÉÁÙ: ¸ÅÀÏ 09:30 / 11:00 / 13:30 / 15:00 (4È¸)
REM  µî·Ï:
REM    schtasks /create /tn "QM_SNAP1" /tr "wscript.exe D:\sub-agent-project_ÄöÆ®º¿\scripts\run_hidden.vbs D:\sub-agent-project_ÄöÆ®º¿\scripts\schedule_SNAP_supply.bat 1" /sc daily /st 09:30
REM    schtasks /create /tn "QM_SNAP2" /tr "wscript.exe D:\sub-agent-project_ÄöÆ®º¿\scripts\run_hidden.vbs D:\sub-agent-project_ÄöÆ®º¿\scripts\schedule_SNAP_supply.bat 2" /sc daily /st 11:00
REM    schtasks /create /tn "QM_SNAP3" /tr "wscript.exe D:\sub-agent-project_ÄöÆ®º¿\scripts\run_hidden.vbs D:\sub-agent-project_ÄöÆ®º¿\scripts\schedule_SNAP_supply.bat 3" /sc daily /st 13:30
REM    schtasks /create /tn "QM_SNAP4" /tr "wscript.exe D:\sub-agent-project_ÄöÆ®º¿\scripts\run_hidden.vbs D:\sub-agent-project_ÄöÆ®º¿\scripts\schedule_SNAP_supply.bat 4" /sc daily /st 15:00
REM
REM  »ç°í ÀÌ·Â: µ¥¸ó ºñÈ°¼ºÈ­(093f9ea) ÈÄ 20ÀÏ°£ ½º³À¼¦ ¹Ì¼öÁý (2026-03-06~26)
REM ============================================================

echo [%date% %time%] ================================================== >> D:\sub-agent-project_ÄöÆ®º¿\logs\schedule.log
echo [%date% %time%] BAT-SNAP ½ÃÀÛ: ¼ö±Þ ½º³À¼¦ %1Â÷ >> D:\sub-agent-project_ÄöÆ®º¿\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project_ÄöÆ®º¿\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_ÄöÆ®º¿
set PYTHONPATH=D:\sub-agent-project_ÄöÆ®º¿

if not exist logs mkdir logs

REM ¦¡¦¡ °Å·¡ÀÏ °¡µå ¦¡¦¡
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-SNAP ½ºÅµ: ºñ°Å·¡ÀÏ >> logs\schedule.log
    goto :eof
)

REM ¦¡¦¡ ½º³À¼¦ ¹øÈ£ È®ÀÎ ¦¡¦¡
if "%1"=="" (
    echo [%date% %time%] BAT-SNAP ¿À·ù: ½º³À¼¦ ¹øÈ£ ¹ÌÁöÁ¤ >> logs\schedule.log
    goto :eof
)

echo [%date% %time%] [SNAP-%1] ¼ö±Þ ½º³À¼¦ ¼öÁý ½ÃÀÛ >> logs\schedule.log
python -u -X utf8 scripts\daily_scheduler.py --run-now snap%1 >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] [SNAP-%1] FAILED >> logs\schedule.log

echo [%date% %time%] BAT-SNAP %1Â÷ ¿Ï·á >> logs\schedule.log
echo [%date% %time%] ================================================== >> logs\schedule.log
