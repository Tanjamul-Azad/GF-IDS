# Waits for run_noniid.ps1's 12-run queue to finish, then trains
# BiPruneFL-Repro (IID, T=45, seed=42 -- the same headline settings as
# every other model in Table V) and re-evaluates everything so it's
# included in results_best.csv.
#
# One GPU, so this polls for the non-IID queue's completion marker in
# logs\noniid.log rather than running alongside it.

$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"

"=== run_bipru.ps1 waiting for non-IID queue $(Get-Date) ===" |
    Out-File -Append "logs\bipru.log"

while ($true) {
    if (Test-Path "logs\noniid.log") {
        $tail = Get-Content "logs\noniid.log" -Tail 5 -ErrorAction SilentlyContinue
        if ($tail -match "run_noniid.ps1 all done") {
            break
        }
    }
    Start-Sleep -Seconds 300
}

$attempt = 0
$succeeded = $false
while (-not $succeeded -and $attempt -lt 3) {
    $attempt++
    "=== BiPruneFL-Repro start (attempt $attempt/3) $(Get-Date) ===" |
        Out-File -Append "logs\bipru.log"
    python code\federated_train.py --model BiPruneFL-Repro --seed 42 `
        --rounds 45 --resume *>> "logs\BiPruneFL-Repro.log"
    if ($LASTEXITCODE -eq 0) {
        $succeeded = $true
        "=== BiPruneFL-Repro done $(Get-Date) ===" | Out-File -Append "logs\bipru.log"
    } else {
        "=== BiPruneFL-Repro FAILED, exit code $LASTEXITCODE, " +
        "attempt $attempt/3 $(Get-Date) ===" | Out-File -Append "logs\bipru.log"
        Start-Sleep -Seconds 30
    }
}
if (-not $succeeded) {
    "=== BiPruneFL-Repro did not complete after 3 attempts -- " +
    "needs a human to look $(Get-Date) ===" | Out-File -Append "logs\bipru.log"
    exit 1
}

python code\evaluate.py --suffix best *>> "logs\evaluate_bipru.log"
"=== evaluate done $(Get-Date) ===" | Out-File -Append "logs\bipru.log"
