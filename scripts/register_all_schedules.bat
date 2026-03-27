@echo off
REM ============================================================
REM  Quantum Master -- ¿¸√º BAT Ω∫ƒ…¡Ÿ ¿œ∞˝ µÓ∑œ
REM  ∞¸∏Æ¿⁄ ±««— « ø‰: øÏ≈¨∏Ø °Ê ∞¸∏Æ¿⁄ ±««—¿∏∑Œ Ω««‡
REM  WakeToRun + StartWhenAvailable »∞º∫»≠
REM ============================================================
echo ========================================
echo  Quantum Master Ω∫ƒ…¡Ÿ ¿œ∞˝ µÓ∑œ
echo ========================================
echo.

REM -- BAT-A: US ¿Â∏∂∞® µ•¿Ã≈Õ π›øµ (06:10) --
schtasks /create /tn "QM_A_USClose" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_A_us_close.bat" /sc daily /st 06:10 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_A_USClose' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [1/18] QM_A_USClose (06:10)

REM -- BAT-B: æ∆ƒß ∏¥◊ ∫Í∏Æ«Œ (07:00) --
schtasks /create /tn "QM_B_Morning" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_B_morning.bat" /sc daily /st 07:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_B_Morning' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [2/18] QM_B_Morning (07:00)

REM -- BAT-K_Safety: æ»¿¸∏∂¡¯ √º≈© (07:30) --
schtasks /create /tn "QM_K_SafetyMargin" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_K_safety_margin.bat" /sc daily /st 07:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_K_SafetyMargin' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [3/18] QM_K_SafetyMargin (07:30)

REM -- BAT-M_NXT: NXT «¡∏Æ∏∂ƒœ (07:55) --
schtasks /create /tn "QM_M_NXTPre" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_M_nxt_pre.bat" /sc daily /st 07:55 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_M_NXTPre' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [4/18] QM_M_NXTPre (07:55)

REM -- BAT-M: ∏¥◊ ∫Í∏Æ«Œ (08:00) --
schtasks /create /tn "QM_M_MorningBriefing" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_M_morning_briefing.bat" /sc daily /st 08:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_M_MorningBriefing' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [5/18] QM_M_MorningBriefing (08:00)

REM -- BAT-N: Ω√±◊≥Œ ∑Œ±◊ (08:20) --
schtasks /create /tn "QM_N_SignalLog" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_N_signal_log.bat" /sc daily /st 08:20 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_N_SignalLog' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [6/18] QM_N_SignalLog (08:20)

REM -- BAT-E: Ω∫∏∂∆Æ ø£∆Æ∏Æ (08:50) --
schtasks /create /tn "QM_E_SmartEntry" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_E_smart_entry.bat" /sc daily /st 08:50 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_E_SmartEntry' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [7/18] QM_E_SmartEntry (08:50)

REM -- BAT-K: ¿Â¡ﬂ Ω«Ω√∞£ ∞®Ω√ (08:55) --
schtasks /create /tn "QM_K_IntradayEye" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_K_intraday_eye.bat" /sc daily /st 08:55 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_K_IntradayEye' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [8/18] QM_K_IntradayEye (08:55)

REM -- BAT-I: VWAP ∏¥œ≈Õ (08:55) --
schtasks /create /tn "QM_I_VWAP" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_I_vwap_monitor.bat" /sc daily /st 08:55 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_I_VWAP' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [9/18] QM_I_VWAP (08:55)

REM -- BAT-P1: µ•¿Ã∆Æ∑π¿Ãµ˘ Ω√±◊≥Œ (09:05) --
schtasks /create /tn "QM_P1_DaySignal" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_P_daytrading.bat log" /sc daily /st 09:05 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_P1_DaySignal' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [10/18] QM_P1_DaySignal (09:05)

REM -- BAT-H: ¿Â¡ﬂ ∫–ºÆ (11:30) --
schtasks /create /tn "QM_H_Midday" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_H_midday_analysis.bat" /sc daily /st 11:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_H_Midday' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [11/18] QM_H_Midday (11:30)

REM -- BAT-G: ±›ø‰¿œ ≥´∆¯ ∏≈ºˆ (14:00) --
schtasks /create /tn "QM_G_FridayDip" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_G_friday_dip.bat" /sc daily /st 14:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_G_FridayDip' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [12/18] QM_G_FridayDip (14:00)

REM -- BAT-P2: ¿Â∏∂∞® Ω√±◊≥Œ (15:20) --
schtasks /create /tn "QM_P2_DayClose" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_P_daytrading.bat close" /sc daily /st 15:20 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_P2_DayClose' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [13/18] QM_P2_DayClose (15:20)

REM -- BAT-L: NXT æ÷«¡≈Õ∏∂ƒœ (15:35) --
schtasks /create /tn "QM_L_NXTAfter" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_L_nxt_after.bat" /sc daily /st 15:35 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_L_NXTAfter' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [14/18] QM_L_NXTAfter (15:35)

REM -- BAT-O: Ω√±◊≥Œ ∆Æ∑°≈∑ (16:10) --
schtasks /create /tn "QM_O_SignalTrack" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_O_signal_track.bat" /sc daily /st 16:10 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_O_SignalTrack' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [15/18] QM_O_SignalTrack (16:10)

REM -- BAT-D: ¿Â∏∂∞® ¿¸√º µ•¿Ã≈Õ ºˆ¡˝ (16:30) °⁄«ŸΩ…°⁄ --
schtasks /create /tn "QM_D_AfterClose" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_D_after_close.bat" /sc daily /st 16:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_D_AfterClose' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [16/18] QM_D_AfterClose (16:30) *** CORE ***

REM -- BAT-J: ∆˜∆Æ∆˙∏Æø¿ æ∆øÙ∑Ë (17:00) --
schtasks /create /tn "QM_J_Outlook" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_J_portfolio_outlook.bat" /sc daily /st 17:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_J_Outlook' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [17/18] QM_J_Outlook (17:00)

REM -- BAT-F: Ω∫≥™¿Ã∆€ øˆƒ° (17:30) --
schtasks /create /tn "QM_F_SniperWatch" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_F_sniper_watch.bat" /sc daily /st 17:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_F_SniperWatch' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [18/18] QM_F_SniperWatch (17:30)

REM -- BAT-SNAP: ¿Â¡ﬂ ºˆ±ﬁ Ω∫≥¿º¶ (09:30, 11:00, 13:30, 15:00) --
schtasks /create /tn "QM_SNAP1" /tr "wscript.exe D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\run_hidden.vbs D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_SNAP_supply.bat 1" /sc daily /st 09:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_SNAP1' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [19/22] QM_SNAP1 (09:30)
schtasks /create /tn "QM_SNAP2" /tr "wscript.exe D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\run_hidden.vbs D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_SNAP_supply.bat 2" /sc daily /st 11:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_SNAP2' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [20/22] QM_SNAP2 (11:00)
schtasks /create /tn "QM_SNAP3" /tr "wscript.exe D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\run_hidden.vbs D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_SNAP_supply.bat 3" /sc daily /st 13:30 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_SNAP3' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [21/22] QM_SNAP3 (13:30)
schtasks /create /tn "QM_SNAP4" /tr "wscript.exe D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\run_hidden.vbs D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_SNAP_supply.bat 4" /sc daily /st 15:00 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_SNAP4' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [22/22] QM_SNAP4 (15:00)

REM -- BAT-D2: ºˆ±ﬁ ∫∏√Ê ºˆ¡˝ (16:10) --
schtasks /create /tn "QM_D2_Supply" /tr "D:\sub-agent-project_ƒˆ∆Æ∫ø\scripts\schedule_D2_supply.bat" /sc daily /st 16:10 /f
powershell -Command "Set-ScheduledTask -TaskName 'QM_D2_Supply' -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true)" 2>nul
echo   [+] QM_D2_Supply (16:10)

echo.
echo ========================================
echo  µÓ∑œ »Æ¿Œ
echo ========================================
powershell -Command "Get-ScheduledTask | Where-Object {$_.TaskName -like 'QM_*'} | Select-Object TaskName, State | Sort-Object TaskName | Format-Table -AutoSize"

echo.
echo √— 23∞≥ Ω∫ƒ…¡Ÿ µÓ∑œ øœ∑·!
echo WakeToRun=True, StartWhenAvailable=True º≥¡§µ 
pause
