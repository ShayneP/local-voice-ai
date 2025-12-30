param(
  [ValidateSet("cpu", "gpu")]
  [string]$Mode,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ComposeArgs
)

if (-not $Mode) {
  Write-Host "Select target:"
  Write-Host "  1) CPU"
  Write-Host "  2) Nvidia GPU"
  $choice = Read-Host "Enter choice (1/2)"
  switch ($choice) {
    "1" { $Mode = "cpu" }
    "2" { $Mode = "gpu" }
    default {
      Write-Error "Invalid choice. Use 1 for CPU or 2 for GPU."
      exit 1
    }
  }
}

$composeFiles = @("docker-compose.yml")
if ($Mode -eq "gpu") {
  $composeFiles += "docker-compose.gpu.yml"
}

$cmd = @("compose")
foreach ($file in $composeFiles) {
  $cmd += @("-f", $file)
}
$cmd += @("up")
if ($ComposeArgs) {
  $cmd += $ComposeArgs
}

Write-Host ("Running: docker " + ($cmd -join " "))
& docker @cmd
exit $LASTEXITCODE
