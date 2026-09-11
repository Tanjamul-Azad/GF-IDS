# Waits for the in-flight BNN-FULL run to finish, then trains
# BNN-MATCHED and scores every model. One GPU, so the runs are
# strictly sequential: starting a second training alongside the first
# would make both slower and, worse, let two processes write the same
# checkpoint.
param([int]$WaitPid)

$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"

if ($WaitPid -gt 0) {
    "=== waiting for PID $WaitPid (BNN-FULL) at $(Get-Date) ===" |
        Out-File -Append "logs\ablations.log"
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch {}
}

"=== BNN-MATCHED start $(Get-Date) ===" | Out-File -Append "logs\ablations.log"
python code\federated_train.py --model BNN-MATCHED --seed 42 --rounds 45 `
    *>> "logs\BNN-MATCHED.log"
"=== BNN-MATCHED done $(Get-Date) ===" | Out-File -Append "logs\ablations.log"

python code\evaluate.py --suffix best *>> "logs\evaluate_ablations.log"
"=== evaluate done $(Get-Date) ===" | Out-File -Append "logs\ablations.log"
