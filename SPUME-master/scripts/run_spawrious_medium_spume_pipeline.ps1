$ErrorActionPreference = "Continue"

$RepoRoot = "D:\SPUME\SPUME-master"
$LogRoot = "D:\SPUME\how_to_run"

Set-Location $RepoRoot

function Run-Step {
    param(
        [string]$Name,
        [string[]]$PythonArgs,
        [string]$LogPath
    )

    $start = Get-Date
    "[$start] START $Name" | Tee-Object -FilePath $LogPath
    & python @PythonArgs >> $LogPath 2>&1
    $exitCode = $LASTEXITCODE
    $end = Get-Date
    "[$end] END $Name exit=$exitCode" | Tee-Object -FilePath $LogPath -Append
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $LogPath"
    }
}

Run-Step `
    -Name "Spawrious medium ViT-GPT2 concept extraction" `
    -PythonArgs @(
        "extract_concepts.py",
        "--dataset", "spawrious_o2o_medium",
        "--concept_model", "vit_gpt2",
        "--caption_batch_size", "8",
        "--caption_num_workers", "0",
        "--log_every", "500",
        "--top_k_report", "50"
    ) `
    -LogPath "$LogRoot\spawrious_medium_vitgpt2_extract.log"

Run-Step `
    -Name "Spawrious medium SPUME-ViT-GPT2 training" `
    -PythonArgs @(
        "train_meta_spurious.py",
        "--config", "config\spawrious_o2o_medium_vitgpt2_spume_full.yaml"
    ) `
    -LogPath "$LogRoot\spawrious_medium_vitgpt2_spume_full_run.log"

Run-Step `
    -Name "Spawrious medium BLIP concept extraction" `
    -PythonArgs @(
        "extract_concepts.py",
        "--dataset", "spawrious_o2o_medium",
        "--concept_model", "blip",
        "--caption_batch_size", "8",
        "--caption_num_workers", "0",
        "--log_every", "500",
        "--top_k_report", "50"
    ) `
    -LogPath "$LogRoot\spawrious_medium_blip_extract.log"

Run-Step `
    -Name "Spawrious medium SPUME-BLIP training" `
    -PythonArgs @(
        "train_meta_spurious.py",
        "--config", "config\spawrious_o2o_medium_blip_spume_full.yaml"
    ) `
    -LogPath "$LogRoot\spawrious_medium_blip_spume_full_run.log"
