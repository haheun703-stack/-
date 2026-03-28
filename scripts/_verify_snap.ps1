@("QM_SNAP1","QM_SNAP2","QM_SNAP3","QM_SNAP4") | ForEach-Object {
  $task = Get-ScheduledTask -TaskName $_
  $action = $task.Actions[0]
  Write-Host "$_ : Execute=$($action.Execute) Args=$($action.Arguments)"
}
