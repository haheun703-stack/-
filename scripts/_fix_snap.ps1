$base = "D:\sub-agent-project_퀀트봇\scripts"

# 기존 SNAP 삭제
@("QM_SNAP1","QM_SNAP2","QM_SNAP3","QM_SNAP4") | ForEach-Object {
  schtasks.exe /delete /tn $_ /f 2>&1 | Out-Null
}

# cmd /c 로 BAT + 인수 분리하여 등록
$snaps = @(
  @("QM_SNAP1", "schedule_SNAP_supply.bat", "1", "09:30"),
  @("QM_SNAP2", "schedule_SNAP_supply.bat", "2", "11:00"),
  @("QM_SNAP3", "schedule_SNAP_supply.bat", "3", "13:30"),
  @("QM_SNAP4", "schedule_SNAP_supply.bat", "4", "15:00")
)

foreach ($s in $snaps) {
  $name = $s[0]; $bat = $s[1]; $arg = $s[2]; $time = $s[3]
  $tr = "cmd /c `"$base\$bat`" $arg"
  $r = & schtasks.exe /create /tn $name /tr $tr /sc daily /st $time /f 2>&1
  if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK  $name ($time) - cmd /c BAT $arg"
  } else {
    Write-Host "  FAIL $name : $r"
  }
  try {
    Set-ScheduledTask -TaskName $name -Settings (New-ScheduledTaskSettingsSet -WakeToRun $true -StartWhenAvailable $true) -ErrorAction SilentlyContinue | Out-Null
  } catch {}
}

Write-Host "`nSNAP tasks fixed (cmd /c wrapper)"
