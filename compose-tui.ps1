param(
  [ValidateSet("cpu", "gpu")]
  [string]$Mode,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ComposeArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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

$composeCmd = @("compose")
foreach ($file in $composeFiles) {
  $composeCmd += @("-f", $file)
}

$startCmd = $composeCmd + @("up", "--build", "-d")
if ($ComposeArgs) {
  $startCmd += $ComposeArgs
}

Write-Host "Starting services..."
& docker @startCmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cargoArgs = @("run", "--manifest-path", "$ScriptDir\tui\Cargo.toml")
if ($env:LOCAL_VOICE_AI_TUI_RELEASE -eq "1") {
  $cargoArgs += "--release"
}

$appArgs = @()
if ($env:LOCAL_VOICE_AI_TUI_MAX_LINES) {
  $appArgs += @("--max-lines", $env:LOCAL_VOICE_AI_TUI_MAX_LINES)
}
if ($env:LOCAL_VOICE_AI_TUI_TAIL) {
  $appArgs += @("--tail", $env:LOCAL_VOICE_AI_TUI_TAIL)
}
if ($env:LOCAL_VOICE_AI_TUI_INTERVAL_MS) {
  $appArgs += @("--interval-ms", $env:LOCAL_VOICE_AI_TUI_INTERVAL_MS)
}

foreach ($file in $composeFiles) {
  $appArgs += @("-f", $file)
}

Write-Host "Launching TUI..."
& cargo @cargoArgs -- @appArgs
exit $LASTEXITCODE
