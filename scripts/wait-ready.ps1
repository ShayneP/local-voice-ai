# Displays service status and waits until all are ready

$services = @("livekit", "whisper", "llama_cpp", "kokoro", "livekit_agent", "frontend")

function Get-ServiceStatus {
    param([string]$Name)
    
    $json = docker compose ps --format json $Name 2>$null
    if (-not $json) { return "waiting" }
    
    # Check health status first
    if ($json -match '"Health":"([^"]*)"') {
        $health = $matches[1]
        if ($health -eq "healthy") { return "ready" }
    }
    
    # No health check - check if running
    if ($json -match '"State":"([^"]*)"') {
        $state = $matches[1]
        if ($state -eq "running") { return "ready" }
    }
    
    return "waiting"
}

$iteration = 0
$allReady = $false
$lineCount = $services.Count + 2

Write-Host ""
Write-Host "Waiting for services..." -ForegroundColor White
Write-Host ""

foreach ($svc in $services) {
    Write-Host "  " -NoNewline
    Write-Host "$([char]0x25CB)" -ForegroundColor Yellow -NoNewline
    Write-Host " $svc"
}
Write-Host ""

while (-not $allReady) {
    Start-Sleep -Seconds 1
    $iteration++
    
    # Clear lines
    for ($i = 0; $i -lt $lineCount; $i++) {
        Write-Host "`e[A`e[K" -NoNewline
    }
    
    Write-Host "Waiting for services... ($iteration`s)" -ForegroundColor White
    Write-Host ""
    
    $allReady = $true
    foreach ($svc in $services) {
        $status = Get-ServiceStatus -Name $svc
        if ($status -ne "ready") { $allReady = $false }
        
        Write-Host "  " -NoNewline
        if ($status -eq "ready") {
            Write-Host "$([char]0x25CF)" -ForegroundColor Green -NoNewline
        } else {
            Write-Host "$([char]0x25CB)" -ForegroundColor Yellow -NoNewline
        }
        Write-Host " $svc"
    }
    Write-Host ""
}

# Final output
for ($i = 0; $i -lt $lineCount; $i++) {
    Write-Host "`e[A`e[K" -NoNewline
}

Write-Host "All services ready! ($iteration`s)" -ForegroundColor Green
Write-Host ""
foreach ($svc in $services) {
    Write-Host "  " -NoNewline
    Write-Host "$([char]0x25CF)" -ForegroundColor Green -NoNewline
    Write-Host " $svc"
}
Write-Host ""
Write-Host "Open " -NoNewline
Write-Host "http://localhost:3000" -ForegroundColor Cyan -NoNewline
Write-Host " to start chatting."
Write-Host ""
