# Sir's feedback experiments (IID, seed 42, T=45): Qin-style SignSGD majority vote
# (two server step sizes) and FedProx, versus the FedAvg results already in the paper.
# Resume-safe. Then it hands over to run_seeds.ps1 for the extra-seed runs.
param([int]$MaxRetries = 3)
$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
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
"=== run_agg.ps1 start $(Get-Date) ===" | Out-File -Append "logs\agg.log"
Run-One "BNN-MATCHED" "signsgd"
Run-One "BNN-MATCHED" "fedprox"
Run-One "MLP" "fedprox"
Run-One "BNN-MATCHED" "signsgd5"
Run-One "MLP" "signsgd"
foreach ($a in @("signsgd","fedprox","signsgd5")) {
    python code\evaluate.py --suffix best --models BNN-MATCHED MLP --seed 42 --aggregator $a *>> "logs\evaluate_agg_$a.log"
}
"=== run_agg.ps1 done, starting seeds $(Get-Date) ===" | Out-File -Append "logs\agg.log"
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_seeds.ps1
