# Two follow-up experiments requested after the non-IID + BiPruneFL
# work:
#
# 1. BNN-INT8IO-MATCHED (seed=42, IID) -- closes the capacity confound
#    in Section V-F, same fix BNN-MATCHED already applied to the
#    Float32-I/O variant.
#
# 2. Sign-flip-rate diagnostics for BNN-MATCHED and BNN-INT8IO at IID,
#    alpha=0.5 and alpha=0.1 -- tests the paper's proposed-but-not-
#    confirmed explanation for BNN-MATCHED's severe-skew collapse
#    (Section V-C): that post-aggregation re-binarization amplifies
#    noise from skewed client updates. Run under seed=43, NOT 42:
#    the paper's own Limitations section documents that even a fixed
#    seed does not guarantee identical results across hardware/library
#    versions, so overwriting the already-cited seed=42 checkpoints
#    would put the exact numbers in main.tex at risk for no benefit.
#    Seed=43 keeps the cited numbers untouched and, as a side effect,
#    is a first data point toward the still-open multi-seed question.
#
# One GPU, so every run is strictly sequential, with the same
# exit-code-checked retry-via-resume pattern as run_noniid.ps1.

param([int]$MaxRetries = 3)

$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"

function Run-Model($modelArgs, $logName) {
    $attempt = 0
    $succeeded = $false
    while (-not $succeeded -and $attempt -lt $MaxRetries) {
        $attempt++
        "=== $logName start (attempt $attempt/$MaxRetries) $(Get-Date) ===" |
            Out-File -Append "logs\investigation.log"
        python code\federated_train.py @modelArgs --rounds 45 --resume `
            *>> "logs\$logName.log"
        if ($LASTEXITCODE -eq 0) {
            $succeeded = $true
            "=== $logName done $(Get-Date) ===" |
                Out-File -Append "logs\investigation.log"
        } else {
            "=== $logName FAILED, exit code $LASTEXITCODE, " +
            "attempt $attempt/$MaxRetries $(Get-Date) ===" |
                Out-File -Append "logs\investigation.log"
            Start-Sleep -Seconds 30
        }
    }
    if (-not $succeeded) {
        "=== $logName did not complete after $MaxRetries attempts " +
        "-- HALTING, needs a human to look $(Get-Date) ===" |
            Out-File -Append "logs\investigation.log"
        exit 1
    }
}

"=== run_investigation.ps1 start $(Get-Date) ===" |
    Out-File -Append "logs\investigation.log"

# 1. Matched-capacity int8IO retrain, seed=42, IID -- closes the
#    Section V-F capacity confound.
Run-Model @("--model", "BNN-INT8IO-MATCHED", "--seed", "42") `
    "BNN-INT8IO-MATCHED"

# 2. Sign-flip diagnostics, seed=43, both models x three severities.
foreach ($m in @("BNN-MATCHED", "BNN-INT8IO")) {
    Run-Model @("--model", $m, "--seed", "43") "$($m)_seed43_iid"
    Run-Model @("--model", $m, "--seed", "43", "--partition", "dirichlet",
               "--alpha", "0.5") "$($m)_seed43_dir0.5"
    Run-Model @("--model", $m, "--seed", "43", "--partition", "dirichlet",
               "--alpha", "0.1") "$($m)_seed43_dir0.1"
}

"=== evaluate start $(Get-Date) ===" | Out-File -Append "logs\investigation.log"
python code\evaluate.py --suffix best --models BNN-INT8IO-MATCHED --seed 42 `
    *>> "logs\evaluate_int8io_matched.log"
python code\evaluate.py --suffix best --models BNN-MATCHED BNN-INT8IO --seed 43 `
    *>> "logs\evaluate_seed43_iid.log"
python code\evaluate.py --suffix best --models BNN-MATCHED BNN-INT8IO --seed 43 `
    --partition dirichlet --alpha 0.5 *>> "logs\evaluate_seed43_dir0.5.log"
python code\evaluate.py --suffix best --models BNN-MATCHED BNN-INT8IO --seed 43 `
    --partition dirichlet --alpha 0.1 *>> "logs\evaluate_seed43_dir0.1.log"
"=== evaluate done $(Get-Date) ===" | Out-File -Append "logs\investigation.log"

"=== run_investigation.ps1 all done $(Get-Date) ===" |
    Out-File -Append "logs\investigation.log"
