# Second dataset (NF-ToN-IoT-v2, 10 classes), all six paper models, seeds 42/43/44, IID, T=45.
# Runs are small (44,756 training records) so this finishes in a few hours even beside the main queue.
$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
$env:PYTHONUNBUFFERED = "1"; $env:GFIDS_DATASET = "nfton"
New-Item -ItemType Directory -Force -Path "logs\nfton","runs_nfton" | Out-Null
$models = @("BNN-MATCHED","MLP","LSTM","CNN","MLP-INT8","BNN-INT8IO")
foreach ($seed in 42,43,44) {
  foreach ($m in $models) {
    "=== $m seed $seed $(Get-Date) ===" | Out-File -Append "logs\nfton.log"
    python code\federated_train.py --model $m --seed $seed --rounds 45 --resume *>> "logs\nfton\$($m)_$seed.log"
  }
  python code\evaluate.py --suffix best --models $models --seed $seed *>> "logs\nfton\evaluate_$seed.log"
  Copy-Item "runs_nfton\results_best.csv" "runs_nfton\results_best_seed$seed.csv" -Force
  "=== evaluated seed $seed $(Get-Date) ===" | Out-File -Append "logs\nfton.log"
}
"=== nfton done $(Get-Date) ===" | Out-File -Append "logs\nfton.log"
