# Extra seeds (IID only) so the paper can report mean +/- std.
# Group 1: seed 43 for the models that lack it. Group 2: seed 44 for the main models.
# Resume-safe: finished runs are skipped. evaluate.py overwrites results_best.csv,
# so seed-42 is backed up first and restored after every evaluation.
param([int]$MaxRetries = 3)
$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"
if (-not (Test-Path "runs\results_best_seed42_backup.csv")) {
    Copy-Item "runs\results_best.csv" "runs\results_best_seed42_backup.csv"
}
function Run-Model($m, $seed) {
    $log = "$($m)_seed$($seed)_iid"; $ok = $false; $try = 0
    while (-not $ok -and $try -lt $MaxRetries) {
        $try++
        "=== $log start (attempt $try) $(Get-Date) ===" | Out-File -Append "logs\seeds.log"
        python code\federated_train.py --model $m --seed $seed --rounds 45 --resume *>> "logs\$log.log"
        if ($LASTEXITCODE -eq 0) { $ok = $true; "=== $log done $(Get-Date) ===" | Out-File -Append "logs\seeds.log" }
        else { "=== $log FAILED $LASTEXITCODE $(Get-Date) ===" | Out-File -Append "logs\seeds.log"; Start-Sleep -Seconds 30 }
    }
    if (-not $ok) { "=== $log gave up, continuing ===" | Out-File -Append "logs\seeds.log" }
}
function Eval-Group($models, $seed) {
    python code\evaluate.py --suffix best --models $models --seed $seed *>> "logs\evaluate_seeds.log"
    Copy-Item "runs\results_best.csv" "runs\results_best_seed$($seed)_iid_all.csv" -Force
    Copy-Item "runs\results_best_seed42_backup.csv" "runs\results_best.csv" -Force
    "=== evaluated seed $seed $(Get-Date) ===" | Out-File -Append "logs\seeds.log"
}
"=== run_seeds.ps1 start $(Get-Date) ===" | Out-File -Append "logs\seeds.log"
$g1 = @("MLP","CNN","LSTM","MLP-INT8","BiPruneFL-Repro")
foreach ($m in $g1) { Run-Model $m 43 }
Eval-Group $g1 43
$g2 = @("BNN-MATCHED","MLP","LSTM","CNN","MLP-INT8","BNN-INT8IO")
foreach ($m in $g2) { Run-Model $m 44 }
Eval-Group $g2 44
"=== run_seeds.ps1 all done $(Get-Date) ===" | Out-File -Append "logs\seeds.log"
