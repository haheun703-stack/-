$base = "D:\sub-agent-project_퀀트봇\scripts"
$tasks = @(
  @("QM_A_USClose",        "schedule_A_us_close.bat",          "06:10"),
  @("QM_B_Morning",        "schedule_B_morning.bat",           "07:00"),
  @("QM_K_SafetyMargin",   "schedule_K_safety_margin.bat",     "07:30"),
  @("QM_M_NXTPre",         "schedule_M_nxt_pre.bat",           "07:55"),
  @("QM_M_MorningBriefing","schedule_M_morning_briefing.bat",  "08:00"),
  @("QM_N_SignalLog",      "schedule_N_signal_log.bat",        "08:20"),
  @("QM_E_SmartEntry",     "schedule_E_smart_entry.bat",       "08:50"),
  @("QM_K_IntradayEye",    "schedule_K_intraday_eye.bat",      "08:55"),
  @("QM_I_VWAP",           "schedule_I_vwap_monitor.bat",      "08:55"),
  @("QM_P1_DaySignal",     "schedule_P_daytrading.bat",        "09:05"),
  @("QM_H_Midday",         "schedule_H_midday_analysis.bat",   "11:30"),
  @("QM_G_FridayDip",      "schedule_G_friday_dip.bat",        "14:00"),
  @("QM_P2_DayClose",      "schedule_P_daytrading.bat",        "15:20"),
  @("QM_L_NXTAfter",       "schedule_L_nxt_after.bat",         "15:35"),
  @("QM_O_SignalTrack",    "schedule_O_signal_track.bat",      "16:10"),
  @("QM_D2_Supply",        "schedule_D2_supply.bat",           "16:10"),
  @("QM_D_AfterClose",     "schedule_D_after_close.bat",       "16:30"),
  @("QM_J_Outlook",        "schedule_J_portfolio_outlook.bat",  "17:00"),
  @("QM_F_SniperWatch",    "schedule_F_sniper_watch.bat",      "17:30")
)

# SNAP tasks (cmd /c wrapper for argument passing)
$snaps = @(
  @("QM_SNAP1", "schedule_SNAP_supply.bat", "1", "09:30"),
  @("QM_SNAP2", "schedule_SNAP_supply.bat", "2", "11:00"),
  @("QM_SNAP3", "schedule_SNAP_supply.bat", "3", "13:30"),
  @("QM_SNAP4", "schedule_SNAP_supply.bat", "4", "15:00")
)

$ok = 0; $fail = 0

foreach ($t in $tasks) {
  $name = $t[0]; $bat = $t[1]; $time = $t[2]
  $tr = "$base\$bat"
  $r = & schtasks.exe /create /tn $name /tr $tr /sc daily /st $time /f 2>&1
  if ($LASTEXITCODE -eq 0) {
    $ok++
    Write-Host "  OK  $name ($time)"
  } else {
    $fail++
    Write-Host "  FAIL $name : $r"
  }
  try {
    Set-ScheduledTask -TaskName $name -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true) -ErrorAction SilentlyContinue | Out-Null
  } catch {}
}

foreach ($s in $snaps) {
  $name = $s[0]; $bat = $s[1]; $arg = $s[2]; $time = $s[3]
  $tr = "cmd /c `"$base\$bat`" $arg"
  $r = & schtasks.exe /create /tn $name /tr $tr /sc daily /st $time /f 2>&1
  if ($LASTEXITCODE -eq 0) {
    $ok++
    Write-Host "  OK  $name ($time)"
  } else {
    $fail++
    Write-Host "  FAIL $name : $r"
  }
  try {
    Set-ScheduledTask -TaskName $name -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true) -ErrorAction SilentlyContinue | Out-Null
  } catch {}
}

Write-Host "`nResult: $ok OK, $fail FAIL out of $($ok + $fail) tasks"

# Verify
Write-Host "`nRegistered QM tasks:"
Get-ScheduledTask | Where-Object { $_.TaskName -like "QM_*" } | Select-Object TaskName, State | Sort-Object TaskName | Format-Table -AutoSize
