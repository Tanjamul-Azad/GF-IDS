# Faithful Qin-style SignSGD majority vote: the server keeps REAL-valued latent
# weights (no re-binarization of the stored model), so hidden weights can change sign.
# Usage: run_agg3.ps1 -Agg signsgdlat   (step 0.01)   or   -Agg signsgdlat5  (step 0.05). Resume-safe.
param([string]$Agg = "signsgdlat")
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
$env:PYTHONUNBUFFERED = "1"
"=== BNN-MATCHED_$Agg start $(Get-Date) ===" | Out-File -Append "logs\agg.log"
python code\federated_train.py --model BNN-MATCHED --seed 42 --aggregator $Agg --rounds 45 --resume *>> "logs\BNN-MATCHED_$Agg.log"
python code\evaluate.py --suffix best --models BNN-MATCHED --seed 42 --aggregator $Agg *>> "logs\evaluate_agg_$Agg.log"
"=== BNN-MATCHED_$Agg done $(Get-Date) ===" | Out-File -Append "logs\agg.log"
