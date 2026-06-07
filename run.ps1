# Thin wrapper so `.\run.ps1 [gpu|cpu|mac|status|down]` works on Windows.
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }
& $py "$PSScriptRoot\run.py" @args
exit $LASTEXITCODE
