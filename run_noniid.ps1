# Trains the six headline models under label-skew non-IID client
# partitioning (Dirichlet, Hsu et al. 2019) at two severities, alpha=0.1
# (severe) and alpha=0.5 (moderate), then evaluates each batch.
#
# One GPU, so every run is strictly sequential -- do not start a second
# training process while this script is running. Each run writes its
# own checkpoint (see run_tag() in federated_train.py: the partition
# and alpha are baked into the filename, e.g.
# BNN-MATCHED_seed42_dir0.1_best.pt), so these never collide with the
# existing IID checkpoints or with each other.
#
# Same T=45/seed=42 as every existing run, so this is a controlled
# comparison against the IID results already in runs/results_best.csv.
#
# Retries a crashed run via --resume, up to $MaxRetries times, and
# HALTS the whole queue (rather than silently moving to the next
# model) if a model still hasn't finished after that many attempts --
# added after BNN-MATCHED dir0.1 crashed mid-run (a Windows DataLoader
# "Couldn't open shared file mapping" error, see federated_train.py's
# get_loader() docstring for the actual fix) and this script's old
# unconditional "=== done ===" log line made it look like the run had
# finished normally when it had actually died at round 40/45. This
# script did not check $LASTEXITCODE before; it does now.

param([int]$MaxRetries = 3)

$ErrorActionPreference = "Continue"
Set-Location "F:\UIU\11th\green\GFIDS_BNN"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"

$models = @("BNN-MATCHED", "MLP", "LSTM", "CNN", "BNN-INT8IO", "MLP-INT8")
$alphas = @(0.1, 0.5)

"=== run_noniid.ps1 start $(Get-Date) ===" | Out-File -Append "logs\noniid.log"

foreach ($alpha in $alphas) {
    foreach ($m in $models) {
        $attempt = 0
        $succeeded = $false
        while (-not $succeeded -and $attempt -lt $MaxRetries) {
            $attempt++
            "=== $m dir$alpha start (attempt $attempt/$MaxRetries) $(Get-Date) ===" |
                Out-File -Append "logs\noniid.log"
            python code\federated_train.py --model $m --seed 42 --rounds 45 `
                --partition dirichlet --alpha $alpha --resume `
                *>> "logs\$($m)_dir$alpha.log"
            if ($LASTEXITCODE -eq 0) {
                $succeeded = $true
                "=== $m dir$alpha done $(Get-Date) ===" |
                    Out-File -Append "logs\noniid.log"
            } else {
                "=== $m dir$alpha FAILED, exit code $LASTEXITCODE, " +
                "attempt $attempt/$MaxRetries $(Get-Date) ===" |
                    Out-File -Append "logs\noniid.log"
                Start-Sleep -Seconds 30
            }
        }
        if (-not $succeeded) {
            "=== $m dir$alpha did not complete after $MaxRetries attempts " +
            "-- HALTING QUEUE, needs a human to look $(Get-Date) ===" |
                Out-File -Append "logs\noniid.log"
            exit 1
        }
    }

    "=== evaluate dir$alpha start $(Get-Date) ===" |
        Out-File -Append "logs\noniid.log"
    python code\evaluate.py --suffix best --models $models `
        --partition dirichlet --alpha $alpha `
        *>> "logs\evaluate_dir$alpha.log"
    "=== evaluate dir$alpha done $(Get-Date) ===" |
        Out-File -Append "logs\noniid.log"
}

"=== run_noniid.ps1 all done $(Get-Date) ===" | Out-File -Append "logs\noniid.log"
