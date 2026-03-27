@echo off
REM ===============================================
REM  BAT-M: FLOWX ¸ð´× ºê¸®ÇÎ
REM  ½ºÄÉÁÙ: ¸ÅÀÏ 08:00 KST (QM_M_MorningBriefing)
REM ===============================================
set PYTHONPATH=D:\sub-agent-project_ÄöÆ®º¿
cd /d D:\sub-agent-project_ÄöÆ®º¿

call venv\Scripts\activate.bat

echo [%date% %time%] BAT-M ¸ð´× ºê¸®ÇÎ ½ÃÀÛ
python -u -X utf8 scripts/cron_morning_briefing.py >> logs\morning_briefing.log 2>&1
echo [%date% %time%] BAT-M ¸ð´× ºê¸®ÇÎ ¿Ï·á

deactivate
