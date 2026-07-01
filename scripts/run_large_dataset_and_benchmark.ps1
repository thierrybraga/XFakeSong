param(
    [int]$Epochs = 100,
    [int]$TimeoutMin = 240,
    [string]$Dataset = "app/datasets/benchmark_audio_raw_balanced_15k.npz",
    [string]$Out = "results/large_benchmark_full"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

$env:DOCKER_TRAIN_CPU_LIMIT = if ($env:DOCKER_TRAIN_CPU_LIMIT) { $env:DOCKER_TRAIN_CPU_LIMIT } else { "8" }

Write-Host "== Build/validate GPU image =="
python scripts/docker_build.py benchmark-nvidia build

Write-Host "== Dataset large: downloads, balanceamento e splits =="
docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
    python scripts/build_dataset.py --tier large

Write-Host "== Auditoria do speaker_manifest =="
docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
    python scripts/audit_speaker_manifest.py --scope active --min-identified-ratio 0.40

Write-Host "== Exportando NPZ =="
docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
    python scripts/export_npz_from_splits.py --out $Dataset --sample-rate 16000 --duration-sec 5.0

Write-Host "== Benchmark completo: 14 modelos, GPU, resume =="
docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
    python scripts/run_models_sequential.py `
        --dataset $Dataset `
        --out $Out `
        --epochs $Epochs `
        --batch-size 32 `
        --device-profile gpu `
        --timeout-min $TimeoutMin `
        --latency-runs 30 `
        --snr 30 20 10 `
        --resume
