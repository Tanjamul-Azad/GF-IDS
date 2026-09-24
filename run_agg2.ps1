# Reduced aggregator queue (MLP variants dropped as lower priority):
# FedProx-BNN (resume), SignSGD step 0.05 BNN, then evaluate. Resume-safe.
param([int]$MaxRetries = 3)
$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
$env:PYTHONUNBUFFERED = "1"
function Run-One($m, $agg) {
    $log = "$($m)_$($agg)"; $ok = $false; $try = 0
    while (-not $ok -and $try -lt $MaxRetries) {
        $try++
        "=== $log start (attempt $try) $(Get-Date) ===" | Out-File -Append "logs\agg.log"
        python code\federated_train.py --model $m --seed 42 --aggregator $agg --rounds 45 --resume *>> "logs\$log.log"
        if ($LASTEXITCODE -eq 0) { $ok = $true; "=== $log done $(Get-Date) ===" | Out-File -Append "logs\agg.log" }
        else { "=== $log FAILED $LASTEXITCODE $(Get-Date) ===" | Out-File -Append "logs\agg.log"; Start-Sleep -Seconds 30 }
    }
}
"=== run_agg2.ps1 start $(Get-Date) ===" | Out-File -Append "logs\agg.log"
Run-One "BNN-MATCHED" "fedprox"
Run-One "BNN-MATCHED" "signsgd5"
foreach ($a in @("signsgd","fedprox","signsgd5")) {
    python code\evaluate.py --suffix best --models BNN-MATCHED --seed 42 --aggregator $a *>> "logs\evaluate_agg_$a.log"
}
"=== run_agg.ps1 done (agg2) $(Get-Date) ===" | Out-File -Append "logs\agg.log"
