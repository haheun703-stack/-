@echo off
chcp 65001 >nul
REM ============================================================
REM  Quantum Master - BAT-H: 11:30 ���� AI �м�
REM  ������: ���� 11:30 (��~��)
REM  ���: schtasks /create /tn "QM_H_Midday" /tr "D:\sub-agent-project_��Ʈ��\scripts\schedule_H_midday_analysis.bat" /sc daily /st 11:30
REM
REM  5���� TradeAdvisor �м� �� �ڷ��׷� ����
REM  12:00 �ż� �Ǵܿ� ������ ����
REM ============================================================

echo [%date% %time%] BAT-H ����: ���� AI �м� >> D:\sub-agent-project_��Ʈ��\logs\schedule.log

call D:\sub-agent-project_��Ʈ��\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_��Ʈ��
set PYTHONPATH=D:\sub-agent-project_��Ʈ��

REM ���� �ŷ��� ���� (trading_calendar ���) ����
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-H ��ŵ: ��ŷ��� >> logs\schedule.log
    goto :eof
)

echo ========================================
echo [QM-H] 11:30 ���� AI �м� (5����)
echo   TradeAdvisor �м� �� �ڷ��׷� ����
echo   12:00 �ż� �Ǵܿ� ������
echo ========================================

python -u -X utf8 scripts/run_midday_analysis.py >> logs\schedule.log 2>&1

REM [ARCHIVED 4/21] market_pulse.py — orphan 아카이브됨, VPS run_bat.sh H에서도 미호출

echo [%date% %time%] BAT-H �Ϸ� >> logs\schedule.log
