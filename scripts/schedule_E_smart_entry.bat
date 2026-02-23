@echo off
chcp 65001 >nul
echo ========================================
echo [QM-E] AI 스마트 진입 (08:50~10:30)
echo ========================================

call D:\sub-agent-project\venv\Scripts\activate.bat
cd /d D:\sub-agent-project
set PYTHONPATH=D:\sub-agent-project

echo [1/1] AI 스마트 진입 실행 (DRY-RUN)...
python -u -X utf8 scripts/smart_entry_runner.py --analysis
if %ERRORLEVEL% NEQ 0 echo [FAIL] smart_entry_runner 실패 (code=%ERRORLEVEL%)

echo ========================================
echo [QM-E] 완료
echo ========================================
