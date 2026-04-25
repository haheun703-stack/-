@echo off
REM ============================================================
REM  Quantum Master - BAT-A: ���� ���� + ��ħ �罺ĵ + �ڷ��׷�
REM  ������: ���� 06:10 (��~��, ���� ���� ����)
REM  ���: schtasks /create /tn "QM_A_USClose" /tr "D:\sub-agent-project_��Ʈ��\scripts\schedule_A_us_close.bat" /sc daily /st 06:10
REM
REM  [v3] ���� ������ �ݿ� + ������ �溸 + ��õ���� �罺ĵ + �ڷ��׷�
REM       �� BAT-E(08:50)�� �ֽ� ��õ���� �ڵ��ż�
REM ============================================================

echo [%date% %time%] BAT-A ����: ���� ���� + ������ + ��ħ �罺ĵ >> D:\sub-agent-project_��Ʈ��\logs\schedule.log

chcp 65001 >nul
call D:\sub-agent-project_��Ʈ��\venv\Scripts\activate.bat
cd /d D:\sub-agent-project_��Ʈ��
set PYTHONPATH=D:\sub-agent-project_��Ʈ��

REM -- �ŷ��� ���� (trading_calendar ���) --
python -c "from src.trading_calendar import should_run_bat; exit(0 if should_run_bat('kr') else 1)"
if errorlevel 1 (
    echo [%date% %time%] BAT-A ��ŵ: ��ŷ��� >> logs\schedule.log
    goto :eof
)

REM -- PHASE 1: �̱��� ������ ������Ʈ --

REM 1) US ���� ������ ������Ʈ + Overnight Signal (������ ����)
echo [%date% %time%] [1/7] US Overnight Signal ������Ʈ >> logs\schedule.log
python -u -X utf8 scripts\us_overnight_signal.py --update >> logs\schedule.log 2>&1

REM 2) US-KR ����DB ���� ����
echo [%date% %time%] [2/7] US-KR ����DB ������Ʈ >> logs\schedule.log
python -u -X utf8 scripts\update_us_kr_daily.py >> logs\schedule.log 2>&1

REM 2.5) COT �ְ� ������Ʈ (���� üũ, �ݿ��Ͽ��� ���� ����)
echo [%date% %time%] [2.5/7] COT �ְ� ������Ʈ >> logs\schedule.log
python -u -X utf8 scripts\fetch_cot_weekly.py >> logs\schedule.log 2>&1

REM 2.7) ������ ����Ŭ ������ ������Ʈ (FRED 5�� ��ǥ)
echo [%date% %time%] [2.7/7] ������ ������ ������Ʈ >> logs\schedule.log
python -u -X utf8 scripts\fetch_liquidity_data.py >> logs\schedule.log 2>&1

REM 2.8) ������ �ñ׳� ����
echo [%date% %time%] [2.8/7] ������ �ñ׳� ���� >> logs\schedule.log
python -u -X utf8 scripts\run_liquidity_signal.py >> logs\schedule.log 2>&1

REM 3) ���� ������ ���� (US ������ ������Ʈ + �溸 ����, �ڷ��׷��� ��ħ ���տ� ����)
REM v13: --all �� --update --signal (������ ���� �˸� ����, BAT-B ��ħ �긮�ο��� ����)
echo [%date% %time%] [3/7] ���� ������ ������ ������Ʈ >> logs\schedule.log
python -u -X utf8 scripts\run_relay_engine.py --update --signal >> logs\schedule.log 2>&1

REM 2.9) [ARCHIVED 4/21] fetch_commodity_prices.py, news_scenario_engine.py — orphan 아카이브됨
REM      VPS run_bat.sh에도 미포함, 데이터 미수집 상태

REM -- PHASE 2: ��ħ �罺ĵ (�̱��� �ݿ�) --

REM 4) v3 AI Brain ����� (���� ������ �ݿ� �� ai_v3_picks.json ����)
echo [%date% %time%] [4/7] v3 AI Brain ��ħ �罺ĵ >> logs\schedule.log
python -u -X utf8 scripts\run_v3_brain.py --no-telegram >> logs\schedule.log 2>&1
if errorlevel 1 echo [%date% %time%] v3 Brain ���� (���� picks ����) >> logs\schedule.log

REM 5) ��õ���� �罺ĵ (overnight_signal + v3 picks �ݿ�)
echo [%date% %time%] [5/7] ��õ���� ��ħ �罺ĵ >> logs\schedule.log
python -u -X utf8 scripts\scan_tomorrow_picks.py >> logs\schedule.log 2>&1

REM 6) ��ħ �ڷ��׷� -- BAT-B (07:00) ���� �긮�ο��� 1������ �߼�
REM v13: BAT-A������ �����͸� �غ�, BAT-B���� ���ǻ�+�׸�+������ ���� 1�� �߼�
echo [%date% %time%] [6/7] ��ħ �ڷ��׷��� BAT-B(07:00)���� ���� �߼� >> logs\schedule.log

echo [%date% %time%] BAT-A �Ϸ� (9�ܰ�) >> logs\schedule.log
